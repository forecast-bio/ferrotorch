//! Recurrent neural network modules.
//!
//! Implements [`LSTM`], [`GRU`], [`RNN`] (multi-layer modules) and their
//! single-step cell counterparts [`LSTMCell`], [`GRUCell`], [`RNNCell`].
//! Each mirrors the corresponding `torch.nn` module.
//!
//! Because the forward passes are composed entirely from differentiable
//! operations (`mm`, `add`, `mul`, `sigmoid`, `tanh`, `relu` from
//! `ferrotorch_core::grad_fns`), autograd builds the backward graph
//! automatically — no custom backward functions are required.

use ferrotorch_core::grad_fns::activation::{relu, sigmoid, tanh};
use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};
use ferrotorch_core::grad_fns::shape::{cat, reshape};
use ferrotorch_core::ops::linalg::mm;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::init;
use crate::module::Module;
use crate::parameter::Parameter;

/// Output type for LSTM forward: `(output_sequence, (h_n, c_n))`.
type LstmOutput<T> = (Tensor<T>, (Tensor<T>, Tensor<T>));

// ---------------------------------------------------------------------------
// Per-layer parameter set
// ---------------------------------------------------------------------------

/// Parameters for a single LSTM layer.
#[derive(Debug, Clone)]
struct LSTMLayerParams<T: Float> {
    /// Weight matrix for input-to-hidden: shape [4*hidden_size, input_size].
    weight_ih: Parameter<T>,
    /// Weight matrix for hidden-to-hidden: shape [4*hidden_size, hidden_size].
    weight_hh: Parameter<T>,
    /// Bias for input-to-hidden: shape [4*hidden_size].
    bias_ih: Parameter<T>,
    /// Bias for hidden-to-hidden: shape [4*hidden_size].
    bias_hh: Parameter<T>,
}

// ---------------------------------------------------------------------------
// LSTM
// ---------------------------------------------------------------------------

/// A multi-layer Long Short-Term Memory (LSTM) RNN.
///
/// For each element in the input sequence, each layer computes:
///
/// ```text
/// i = sigmoid(W_ii @ x + b_ii + W_hi @ h + b_hi)
/// f = sigmoid(W_if @ x + b_if + W_hf @ h + b_hf)
/// g = tanh(W_ig @ x + b_ig + W_hg @ h + b_hg)
/// o = sigmoid(W_io @ x + b_io + W_ho @ h + b_ho)
/// c' = f * c + i * g
/// h' = o * tanh(c')
/// ```
///
/// The weight matrices for all four gates are concatenated into a single
/// `weight_ih` of shape `[4*hidden_size, input_size]` and `weight_hh` of
/// shape `[4*hidden_size, hidden_size]`.
///
/// # Type parameter
///
/// `T` must implement [`Float`] — currently `f32` or `f64`.
#[derive(Debug)]
pub struct LSTM<T: Float> {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    layers: Vec<LSTMLayerParams<T>>,
    training: bool,
}

impl<T: Float> LSTM<T> {
    /// Create a new LSTM module.
    ///
    /// # Arguments
    ///
    /// * `input_size` — number of expected features in the input `x`.
    /// * `hidden_size` — number of features in the hidden state `h`.
    /// * `num_layers` — number of stacked LSTM layers (must be >= 1).
    ///
    /// # Weight initialization
    ///
    /// All weights are initialized from `U(-k, k)` where `k = 1/sqrt(hidden_size)`.
    /// Biases are initialized to zero. This matches PyTorch's default.
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> FerrotorchResult<Self> {
        if num_layers == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LSTM: num_layers must be >= 1".into(),
            });
        }
        if hidden_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LSTM: hidden_size must be >= 1".into(),
            });
        }
        if input_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LSTM: input_size must be >= 1".into(),
            });
        }

        let k = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 4 * hidden_size;

        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let layer_input_size = if layer_idx == 0 {
                input_size
            } else {
                hidden_size
            };

            let mut weight_ih = Parameter::zeros(&[gate_size, layer_input_size])?;
            let mut weight_hh = Parameter::zeros(&[gate_size, hidden_size])?;
            let mut bias_ih = Parameter::zeros(&[gate_size])?;
            let mut bias_hh = Parameter::zeros(&[gate_size])?;

            init::uniform(&mut weight_ih, -k, k)?;
            init::uniform(&mut weight_hh, -k, k)?;
            init::zeros(&mut bias_ih)?;
            init::zeros(&mut bias_hh)?;

            layers.push(LSTMLayerParams {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
            });
        }

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            layers,
            training: true,
        })
    }

    /// Forward pass with explicit hidden state.
    ///
    /// # Arguments
    ///
    /// * `input` — input tensor of shape `[batch, seq_len, input_size]`.
    /// * `state` — optional `(h_0, c_0)` each of shape `[num_layers, batch, hidden_size]`.
    ///   If `None`, both are initialized to zeros.
    ///
    /// # Returns
    ///
    /// A tuple `(output, (h_n, c_n))` where:
    /// - `output` has shape `[batch, seq_len, hidden_size]` (last layer outputs).
    /// - `h_n`, `c_n` each have shape `[num_layers, batch, hidden_size]`.
    pub fn forward_with_state(
        &self,
        input: &Tensor<T>,
        state: Option<(&Tensor<T>, &Tensor<T>)>,
    ) -> FerrotorchResult<LstmOutput<T>> {
        // Validate input shape: [B, seq_len, input_size]
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LSTM: expected 3-D input [batch, seq_len, input_size], got shape {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let seq_len = input.shape()[1];

        if input.shape()[2] != self.input_size {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LSTM: input_size mismatch: expected {}, got {}",
                    self.input_size,
                    input.shape()[2]
                ),
            });
        }

        // Initialize hidden / cell states.
        let (h_init, c_init) = match state {
            Some((h0, c0)) => {
                // Validate shapes: [num_layers, batch, hidden_size]
                let expected_shape = [self.num_layers, batch, self.hidden_size];
                if h0.shape() != expected_shape {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "LSTM: h_0 shape mismatch: expected {:?}, got {:?}",
                            expected_shape,
                            h0.shape()
                        ),
                    });
                }
                if c0.shape() != expected_shape {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "LSTM: c_0 shape mismatch: expected {:?}, got {:?}",
                            expected_shape,
                            c0.shape()
                        ),
                    });
                }
                (h0.clone(), c0.clone())
            }
            None => {
                let h0 = ferrotorch_core::zeros::<T>(&[self.num_layers, batch, self.hidden_size])?;
                let c0 = ferrotorch_core::zeros::<T>(&[self.num_layers, batch, self.hidden_size])?;
                (h0, c0)
            }
        };

        // Extract per-timestep input slices using differentiable reshape + split.
        // input is [batch, seq_len, input_size].
        // We transpose to get timestep slices as [batch, input_size] tensors.
        let input_data = input.data_vec()?;

        // Build timestep inputs: shape [batch, input_size] each.
        // Use data_vec + from_storage for now. These are leaf-level copies from
        // the input, preserving requires_grad so downstream ops build the graph.
        let mut timestep_inputs: Vec<Tensor<T>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut slice_data = Vec::with_capacity(batch * self.input_size);
            for b in 0..batch {
                let offset = b * seq_len * self.input_size + t * self.input_size;
                slice_data.extend_from_slice(&input_data[offset..offset + self.input_size]);
            }
            timestep_inputs.push(Tensor::from_storage(
                TensorStorage::cpu(slice_data),
                vec![batch, self.input_size],
                input.requires_grad(),
            )?);
        }

        // Extract per-layer initial hidden/cell states.
        let h_init_data = h_init.data_vec()?;
        let c_init_data = c_init.data_vec()?;
        let hs = self.hidden_size;

        let mut layer_h: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);
        let mut layer_c: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);

        for l in 0..self.num_layers {
            let mut h_data = Vec::with_capacity(batch * hs);
            let mut c_data = Vec::with_capacity(batch * hs);
            for b in 0..batch {
                let offset = l * batch * hs + b * hs;
                h_data.extend_from_slice(&h_init_data[offset..offset + hs]);
                c_data.extend_from_slice(&c_init_data[offset..offset + hs]);
            }
            layer_h.push(Tensor::from_storage(
                TensorStorage::cpu(h_data),
                vec![batch, hs],
                false,
            )?);
            layer_c.push(Tensor::from_storage(
                TensorStorage::cpu(c_data),
                vec![batch, hs],
                false,
            )?);
        }

        // Run the LSTM forward pass.
        // For each layer, iterate through all timesteps, then pass the
        // sequence of hidden states as input to the next layer.
        let mut layer_outputs: Vec<Tensor<T>> = timestep_inputs;
        let mut final_h: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);
        let mut final_c: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);

        for (l, params) in self.layers.iter().enumerate() {
            let mut h = layer_h[l].clone();
            let mut c = layer_c[l].clone();
            let mut next_layer_outputs: Vec<Tensor<T>> = Vec::with_capacity(seq_len);

            for x_t in &layer_outputs {
                // gates = x_t @ W_ih^T + bias_ih + h @ W_hh^T + bias_hh
                //
                // W_ih: [4*hs, layer_input_size], need x_t @ W_ih^T => [batch, 4*hs]
                // W_hh: [4*hs, hs], need h @ W_hh^T => [batch, 4*hs]
                let wih_t = transpose_2d(params.weight_ih.tensor())?;
                let whh_t = transpose_2d(params.weight_hh.tensor())?;

                let xw = mm(x_t, &wih_t)?; // [batch, 4*hs]
                let hw = mm(&h, &whh_t)?; // [batch, 4*hs]

                // Broadcast biases: bias_ih and bias_hh are 1-D [4*hs].
                // We need them as [batch, 4*hs] for add.
                let bias_ih_2d = broadcast_bias_to_batch(&params.bias_ih, batch)?;
                let bias_hh_2d = broadcast_bias_to_batch(&params.bias_hh, batch)?;

                let gates = add(&add(&add(&xw, &bias_ih_2d)?, &hw)?, &bias_hh_2d)?;

                // Split gates into i, f, g, o — each [batch, hs].
                // Uses differentiable chunk to preserve the autograd graph.
                let gate_chunks = gates.chunk(4, 1)?;
                let i_pre = gate_chunks[0].clone();
                let f_pre = gate_chunks[1].clone();
                let g_pre = gate_chunks[2].clone();
                let o_pre = gate_chunks[3].clone();

                // Apply activations (differentiable ops — autograd will track).
                let i_gate = sigmoid(&i_pre)?;
                let f_gate = sigmoid(&f_pre)?;
                let g_gate = tanh(&g_pre)?;
                let o_gate = sigmoid(&o_pre)?;

                // c_new = f * c + i * g
                let fc = mul(&f_gate, &c)?;
                let ig = mul(&i_gate, &g_gate)?;
                let c_new = add(&fc, &ig)?;

                // h_new = o * tanh(c_new)
                let tanh_c = tanh(&c_new)?;
                let h_new = mul(&o_gate, &tanh_c)?;

                next_layer_outputs.push(h_new.clone());
                h = h_new;
                c = c_new;
            }

            final_h.push(h);
            final_c.push(c);
            layer_outputs = next_layer_outputs;
        }

        // Assemble output: [batch, seq_len, hidden_size] from the last layer.
        // Each layer_outputs[t] is [batch, hs]. We need to interleave by batch
        // to get [batch, seq_len, hs].
        //
        // Strategy: cat along dim=1 to get [batch, seq_len * hs], then reshape.
        // But layer_outputs[t] is [batch, hs], and we want to stack them along
        // a time dimension. Cat along dim=1 gives [batch, seq_len * hs].
        let output = if seq_len == 1 {
            // Single timestep: just reshape [batch, hs] -> [batch, 1, hs].
            reshape(&layer_outputs[0], &[batch as isize, 1, hs as isize])?
        } else {
            // Cat timestep tensors along dim=1: [batch, seq_len*hs].
            let stacked = cat(&layer_outputs, 1)?;
            // Reshape to [batch, seq_len, hs].
            reshape(&stacked, &[batch as isize, seq_len as isize, hs as isize])?
        };

        // Assemble h_n, c_n: [num_layers, batch, hidden_size].
        // Cat final hidden states along dim=0: each is [batch, hs] -> [num_layers*batch, hs].
        // Then reshape to [num_layers, batch, hs].
        let h_n = if self.num_layers == 1 {
            reshape(&final_h[0], &[1, batch as isize, hs as isize])?
        } else {
            let h_stacked = cat(&final_h, 0)?;
            reshape(
                &h_stacked,
                &[self.num_layers as isize, batch as isize, hs as isize],
            )?
        };
        let c_n = if self.num_layers == 1 {
            reshape(&final_c[0], &[1, batch as isize, hs as isize])?
        } else {
            let c_stacked = cat(&final_c, 0)?;
            reshape(
                &c_stacked,
                &[self.num_layers as isize, batch as isize, hs as isize],
            )?
        };

        Ok((output, (h_n, c_n)))
    }

    /// Number of expected input features.
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Number of features in the hidden state.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of stacked LSTM layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ---------------------------------------------------------------------------
// Module trait implementation
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for LSTM<T> {
    /// Forward pass using the `Module` interface (no explicit hidden state).
    ///
    /// Hidden state defaults to zeros. To pass initial state, use
    /// [`LSTM::forward_with_state`] instead.
    ///
    /// Returns the output tensor of shape `[batch, seq_len, hidden_size]`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let (output, _) = self.forward_with_state(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &self.layers {
            params.push(&layer.weight_ih);
            params.push(&layer.weight_hh);
            params.push(&layer.bias_ih);
            params.push(&layer.bias_hh);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &mut self.layers {
            params.push(&mut layer.weight_ih);
            params.push(&mut layer.weight_hh);
            params.push(&mut layer.bias_ih);
            params.push(&mut layer.bias_hh);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for (i, layer) in self.layers.iter().enumerate() {
            params.push((format!("layers.{i}.weight_ih"), &layer.weight_ih));
            params.push((format!("layers.{i}.weight_hh"), &layer.weight_hh));
            params.push((format!("layers.{i}.bias_ih"), &layer.bias_ih));
            params.push((format!("layers.{i}.bias_hh"), &layer.bias_hh));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Transpose a 2-D tensor (thin wrapper delegating to ops::linalg::transpose).
fn transpose_2d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    ferrotorch_core::ops::linalg::transpose(input)
}

/// Broadcast a 1-D bias of shape `[n]` into shape `[batch, n]` by repeating.
fn broadcast_bias_to_batch<T: Float>(
    bias: &Parameter<T>,
    batch: usize,
) -> FerrotorchResult<Tensor<T>> {
    let bias_data = bias.data()?;
    let n = bias_data.len();
    let mut out = Vec::with_capacity(batch * n);
    for _ in 0..batch {
        out.extend_from_slice(bias_data);
    }
    Tensor::from_storage(
        TensorStorage::cpu(out),
        vec![batch, n],
        bias.requires_grad(),
    )
}

// ---------------------------------------------------------------------------
// Per-layer parameter set (GRU)
// ---------------------------------------------------------------------------

/// Parameters for a single GRU layer.
#[derive(Debug, Clone)]
struct GRULayerParams<T: Float> {
    /// Weight matrix for input-to-hidden: shape [3*hidden_size, input_size].
    weight_ih: Parameter<T>,
    /// Weight matrix for hidden-to-hidden: shape [3*hidden_size, hidden_size].
    weight_hh: Parameter<T>,
    /// Bias for input-to-hidden: shape [3*hidden_size].
    bias_ih: Parameter<T>,
    /// Bias for hidden-to-hidden: shape [3*hidden_size].
    bias_hh: Parameter<T>,
}

// ---------------------------------------------------------------------------
// GRU
// ---------------------------------------------------------------------------

/// A multi-layer Gated Recurrent Unit (GRU) RNN.
///
/// For each element in the input sequence, each layer computes:
///
/// ```text
/// r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)   // reset gate
/// z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)   // update gate
/// n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  // new gate
/// h_t = (1 - z_t) * n_t + z_t * h_{t-1}
/// ```
///
/// The weight matrices for all three gates are concatenated into a single
/// `weight_ih` of shape `[3*hidden_size, input_size]` and `weight_hh` of
/// shape `[3*hidden_size, hidden_size]`.
///
/// # Type parameter
///
/// `T` must implement [`Float`] — currently `f32` or `f64`.
#[derive(Debug)]
pub struct GRU<T: Float> {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    layers: Vec<GRULayerParams<T>>,
    training: bool,
}

impl<T: Float> GRU<T> {
    /// Create a new GRU module.
    ///
    /// # Arguments
    ///
    /// * `input_size` — number of expected features in the input `x`.
    /// * `hidden_size` — number of features in the hidden state `h`.
    ///
    /// Creates a single-layer GRU. Use [`GRU::with_num_layers`] for stacked
    /// layers.
    ///
    /// # Weight initialization
    ///
    /// All weights are initialized from `U(-k, k)` where `k = 1/sqrt(hidden_size)`.
    /// Biases are initialized to zero. This matches PyTorch's default.
    pub fn new(input_size: usize, hidden_size: usize) -> FerrotorchResult<Self> {
        Self::with_num_layers(input_size, hidden_size, 1)
    }

    /// Create a new GRU module with multiple stacked layers.
    ///
    /// # Arguments
    ///
    /// * `input_size` — number of expected features in the input `x`.
    /// * `hidden_size` — number of features in the hidden state `h`.
    /// * `num_layers` — number of stacked GRU layers (must be >= 1).
    pub fn with_num_layers(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> FerrotorchResult<Self> {
        if num_layers == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "GRU: num_layers must be >= 1".into(),
            });
        }
        if hidden_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "GRU: hidden_size must be >= 1".into(),
            });
        }
        if input_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "GRU: input_size must be >= 1".into(),
            });
        }

        let k = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 3 * hidden_size;

        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let layer_input_size = if layer_idx == 0 {
                input_size
            } else {
                hidden_size
            };

            let mut weight_ih = Parameter::zeros(&[gate_size, layer_input_size])?;
            let mut weight_hh = Parameter::zeros(&[gate_size, hidden_size])?;
            let mut bias_ih = Parameter::zeros(&[gate_size])?;
            let mut bias_hh = Parameter::zeros(&[gate_size])?;

            init::uniform(&mut weight_ih, -k, k)?;
            init::uniform(&mut weight_hh, -k, k)?;
            init::zeros(&mut bias_ih)?;
            init::zeros(&mut bias_hh)?;

            layers.push(GRULayerParams {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
            });
        }

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            layers,
            training: true,
        })
    }

    /// Forward pass with explicit hidden state.
    ///
    /// # Arguments
    ///
    /// * `input` — input tensor of shape `[batch, seq_len, input_size]`.
    /// * `h_0` — optional hidden state of shape `[num_layers, batch, hidden_size]`.
    ///   If `None`, initialized to zeros.
    ///
    /// # Returns
    ///
    /// A tuple `(output, h_n)` where:
    /// - `output` has shape `[batch, seq_len, hidden_size]` (last layer outputs).
    /// - `h_n` has shape `[num_layers, batch, hidden_size]`.
    pub fn forward(
        &self,
        input: &Tensor<T>,
        h_0: Option<&Tensor<T>>,
    ) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        // Validate input shape: [B, seq_len, input_size]
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GRU: expected 3-D input [batch, seq_len, input_size], got shape {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let seq_len = input.shape()[1];

        if input.shape()[2] != self.input_size {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GRU: input_size mismatch: expected {}, got {}",
                    self.input_size,
                    input.shape()[2]
                ),
            });
        }

        // Initialize hidden state.
        let h_init = match h_0 {
            Some(h0) => {
                let expected_shape = [self.num_layers, batch, self.hidden_size];
                if h0.shape() != expected_shape {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "GRU: h_0 shape mismatch: expected {:?}, got {:?}",
                            expected_shape,
                            h0.shape()
                        ),
                    });
                }
                h0.clone()
            }
            None => ferrotorch_core::zeros::<T>(&[self.num_layers, batch, self.hidden_size])?,
        };

        // Extract per-timestep input slices.
        let input_data = input.data()?;
        let hs = self.hidden_size;

        let mut timestep_inputs: Vec<Tensor<T>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut slice_data = Vec::with_capacity(batch * self.input_size);
            for b in 0..batch {
                let offset = b * seq_len * self.input_size + t * self.input_size;
                slice_data.extend_from_slice(&input_data[offset..offset + self.input_size]);
            }
            timestep_inputs.push(Tensor::from_storage(
                TensorStorage::cpu(slice_data),
                vec![batch, self.input_size],
                input.requires_grad(),
            )?);
        }

        // Extract per-layer initial hidden states.
        let h_init_data = h_init.data()?;

        let mut layer_h: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let mut h_data = Vec::with_capacity(batch * hs);
            for b in 0..batch {
                let offset = l * batch * hs + b * hs;
                h_data.extend_from_slice(&h_init_data[offset..offset + hs]);
            }
            layer_h.push(Tensor::from_storage(
                TensorStorage::cpu(h_data),
                vec![batch, hs],
                false,
            )?);
        }

        // Run the GRU forward pass.
        let mut layer_outputs: Vec<Tensor<T>> = timestep_inputs;
        let mut final_h: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);

        let is_f32 = std::mem::size_of::<T>() == 4;

        for (l, params) in self.layers.iter().enumerate() {
            let mut h = layer_h[l].clone();
            let mut next_layer_outputs: Vec<Tensor<T>> = Vec::with_capacity(seq_len);

            // Hoist weight transposes outside the timestep loop — these are
            // constant across timesteps.
            let wih_t = ferrotorch_core::grad_fns::shape::transpose_2d(params.weight_ih.tensor())?;
            let whh_t = ferrotorch_core::grad_fns::shape::transpose_2d(params.weight_hh.tensor())?;

            // Check if we can use the fused GPU kernel.
            let use_fused_gpu =
                is_f32 && h.is_cuda() && ferrotorch_core::gpu_dispatch::gpu_backend().is_some();

            for x_t in &layer_outputs {
                // Phase 1: compute gate matrices via cuBLAS GEMMs.
                let xw = mm(x_t, &wih_t)?; // [batch, 3*hs]
                let hw = mm(&h, &whh_t)?; // [batch, 3*hs]

                if use_fused_gpu {
                    // ---- GPU fast path: fused pointwise kernel ----
                    // The kernel takes raw gate matrices (no bias added) + biases
                    // and computes all gate activations + GRU update in one launch.
                    let backend = ferrotorch_core::gpu_dispatch::gpu_backend()
                        .ok_or(FerrotorchError::DeviceUnavailable)?;
                    let (hy_handle, _workspace) = backend.fused_gru_cell_f32(
                        xw.gpu_handle()?,
                        hw.gpu_handle()?,
                        params.bias_ih.tensor().gpu_handle()?,
                        params.bias_hh.tensor().gpu_handle()?,
                        h.gpu_handle()?,
                        hs,
                    )?;
                    let h_new = Tensor::from_storage(
                        TensorStorage::gpu(hy_handle),
                        vec![batch, hs],
                        false,
                    )?;
                    next_layer_outputs.push(h_new.clone());
                    h = h_new;
                } else {
                    // ---- CPU path: scalar gate computation ----
                    let bias_ih_2d = broadcast_bias_to_batch(&params.bias_ih, batch)?;
                    let bias_hh_2d = broadcast_bias_to_batch(&params.bias_hh, batch)?;

                    let xw_b = add(&xw, &bias_ih_2d)?;
                    let hw_b = add(&hw, &bias_hh_2d)?;

                    let xw_data = xw_b.data()?;
                    let hw_data = hw_b.data()?;
                    let gate_size = 3 * hs;

                    let mut rx_data = Vec::with_capacity(batch * hs);
                    let mut zx_data = Vec::with_capacity(batch * hs);
                    let mut nx_data = Vec::with_capacity(batch * hs);
                    let mut rh_data = Vec::with_capacity(batch * hs);
                    let mut zh_data = Vec::with_capacity(batch * hs);
                    let mut nh_data = Vec::with_capacity(batch * hs);

                    for b_idx in 0..batch {
                        let xbase = b_idx * gate_size;
                        rx_data.extend_from_slice(&xw_data[xbase..xbase + hs]);
                        zx_data.extend_from_slice(&xw_data[xbase + hs..xbase + 2 * hs]);
                        nx_data.extend_from_slice(&xw_data[xbase + 2 * hs..xbase + 3 * hs]);

                        let hbase = b_idx * gate_size;
                        rh_data.extend_from_slice(&hw_data[hbase..hbase + hs]);
                        zh_data.extend_from_slice(&hw_data[hbase + hs..hbase + 2 * hs]);
                        nh_data.extend_from_slice(&hw_data[hbase + 2 * hs..hbase + 3 * hs]);
                    }

                    let rg = xw_b.requires_grad() || hw_b.requires_grad();

                    let rx =
                        Tensor::from_storage(TensorStorage::cpu(rx_data), vec![batch, hs], rg)?;
                    let zx =
                        Tensor::from_storage(TensorStorage::cpu(zx_data), vec![batch, hs], rg)?;
                    let nx =
                        Tensor::from_storage(TensorStorage::cpu(nx_data), vec![batch, hs], rg)?;
                    let rh =
                        Tensor::from_storage(TensorStorage::cpu(rh_data), vec![batch, hs], rg)?;
                    let zh =
                        Tensor::from_storage(TensorStorage::cpu(zh_data), vec![batch, hs], rg)?;
                    let nh =
                        Tensor::from_storage(TensorStorage::cpu(nh_data), vec![batch, hs], rg)?;

                    let r_gate = sigmoid(&add(&rx, &rh)?)?;
                    let z_gate = sigmoid(&add(&zx, &zh)?)?;
                    let r_nh = mul(&r_gate, &nh)?;
                    let n_gate = tanh(&add(&nx, &r_nh)?)?;
                    let h_minus_n = sub(&h, &n_gate)?;
                    let z_h_minus_n = mul(&z_gate, &h_minus_n)?;
                    let h_new = add(&n_gate, &z_h_minus_n)?;

                    next_layer_outputs.push(h_new.clone());
                    h = h_new;
                }
            }

            final_h.push(h);
            layer_outputs = next_layer_outputs;
        }

        // Assemble output: [batch, seq_len, hidden_size] from the last layer.
        let mut output_data = Vec::with_capacity(batch * seq_len * hs);
        for b_idx in 0..batch {
            for lo in &layer_outputs {
                let t_data = lo.data()?;
                let offset = b_idx * hs;
                output_data.extend_from_slice(&t_data[offset..offset + hs]);
            }
        }
        let output = Tensor::from_storage(
            TensorStorage::cpu(output_data),
            vec![batch, seq_len, hs],
            false,
        )?;

        // Assemble h_n: [num_layers, batch, hidden_size].
        let mut h_n_data = Vec::with_capacity(self.num_layers * batch * hs);
        for h_l_tensor in &final_h {
            let h_l = h_l_tensor.data()?;
            h_n_data.extend_from_slice(h_l);
        }
        let h_n = Tensor::from_storage(
            TensorStorage::cpu(h_n_data),
            vec![self.num_layers, batch, hs],
            false,
        )?;

        Ok((output, h_n))
    }

    /// Number of expected input features.
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Number of features in the hidden state.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of stacked GRU layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ---------------------------------------------------------------------------
// Module trait implementation (GRU)
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for GRU<T> {
    /// Forward pass using the `Module` interface (no explicit hidden state).
    ///
    /// Hidden state defaults to zeros. To pass initial state, use
    /// [`GRU::forward`] instead.
    ///
    /// Returns the output tensor of shape `[batch, seq_len, hidden_size]`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let (output, _) = GRU::forward(self, input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &self.layers {
            params.push(&layer.weight_ih);
            params.push(&layer.weight_hh);
            params.push(&layer.bias_ih);
            params.push(&layer.bias_hh);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &mut self.layers {
            params.push(&mut layer.weight_ih);
            params.push(&mut layer.weight_hh);
            params.push(&mut layer.bias_ih);
            params.push(&mut layer.bias_hh);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for (i, layer) in self.layers.iter().enumerate() {
            params.push((format!("layers.{i}.weight_ih"), &layer.weight_ih));
            params.push((format!("layers.{i}.weight_hh"), &layer.weight_hh));
            params.push((format!("layers.{i}.bias_ih"), &layer.bias_ih));
            params.push((format!("layers.{i}.bias_hh"), &layer.bias_hh));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// RNNCell
// ===========================================================================

/// Nonlinearity for [`RNNCell`] and [`RNN`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNNonlinearity {
    /// Hyperbolic tangent (default).
    Tanh,
    /// Rectified linear unit.
    ReLU,
}

/// A single-step vanilla RNN cell.
///
/// Computes `h' = nonlinearity(x @ W_ih^T + b_ih + h @ W_hh^T + b_hh)`.
///
/// This is the equivalent of `torch.nn.RNNCell`.
#[derive(Debug)]
pub struct RNNCell<T: Float> {
    input_size: usize,
    hidden_size: usize,
    nonlinearity: RNNNonlinearity,
    weight_ih: Parameter<T>,
    weight_hh: Parameter<T>,
    bias_ih: Parameter<T>,
    bias_hh: Parameter<T>,
    training: bool,
}

impl<T: Float> RNNCell<T> {
    /// Create a new `RNNCell`.
    ///
    /// # Arguments
    ///
    /// * `input_size` — number of expected features in `x`.
    /// * `hidden_size` — number of features in the hidden state `h`.
    ///
    /// Uses tanh nonlinearity by default. Call [`RNNCell::with_nonlinearity`]
    /// for relu.
    pub fn new(input_size: usize, hidden_size: usize) -> FerrotorchResult<Self> {
        Self::with_nonlinearity(input_size, hidden_size, RNNNonlinearity::Tanh)
    }

    /// Create a new `RNNCell` with a specified nonlinearity.
    pub fn with_nonlinearity(
        input_size: usize,
        hidden_size: usize,
        nonlinearity: RNNNonlinearity,
    ) -> FerrotorchResult<Self> {
        if hidden_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RNNCell: hidden_size must be >= 1".into(),
            });
        }
        if input_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RNNCell: input_size must be >= 1".into(),
            });
        }

        let k = 1.0 / (hidden_size as f64).sqrt();

        let mut weight_ih = Parameter::zeros(&[hidden_size, input_size])?;
        let mut weight_hh = Parameter::zeros(&[hidden_size, hidden_size])?;
        let mut bias_ih = Parameter::zeros(&[hidden_size])?;
        let mut bias_hh = Parameter::zeros(&[hidden_size])?;

        init::uniform(&mut weight_ih, -k, k)?;
        init::uniform(&mut weight_hh, -k, k)?;
        init::zeros(&mut bias_ih)?;
        init::zeros(&mut bias_hh)?;

        Ok(Self {
            input_size,
            hidden_size,
            nonlinearity,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            training: true,
        })
    }

    /// Forward pass for the RNN cell.
    ///
    /// # Arguments
    ///
    /// * `input` — input tensor of shape `[batch, input_size]`.
    /// * `h` — hidden state of shape `[batch, hidden_size]`. If `None`,
    ///   initialized to zeros.
    ///
    /// # Returns
    ///
    /// New hidden state `h'` of shape `[batch, hidden_size]`.
    pub fn forward_cell(
        &self,
        input: &Tensor<T>,
        h: Option<&Tensor<T>>,
    ) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RNNCell: expected 2-D input [batch, input_size], got shape {:?}",
                    input.shape()
                ),
            });
        }
        let batch = input.shape()[0];
        if input.shape()[1] != self.input_size {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "RNNCell: input_size mismatch: expected {}, got {}",
                    self.input_size,
                    input.shape()[1]
                ),
            });
        }

        let h_state = match h {
            Some(h0) => {
                if h0.shape() != [batch, self.hidden_size] {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "RNNCell: h shape mismatch: expected {:?}, got {:?}",
                            [batch, self.hidden_size],
                            h0.shape()
                        ),
                    });
                }
                h0.clone()
            }
            None => ferrotorch_core::zeros::<T>(&[batch, self.hidden_size])?,
        };

        let wih_t = transpose_2d(self.weight_ih.tensor())?;
        let whh_t = transpose_2d(self.weight_hh.tensor())?;

        let xw = mm(input, &wih_t)?; // [batch, hidden_size]
        let hw = mm(&h_state, &whh_t)?; // [batch, hidden_size]

        let bias_ih_2d = broadcast_bias_to_batch(&self.bias_ih, batch)?;
        let bias_hh_2d = broadcast_bias_to_batch(&self.bias_hh, batch)?;

        let pre_act = add(&add(&add(&xw, &bias_ih_2d)?, &hw)?, &bias_hh_2d)?;

        match self.nonlinearity {
            RNNNonlinearity::Tanh => tanh(&pre_act),
            RNNNonlinearity::ReLU => relu(&pre_act),
        }
    }

    /// Number of expected input features.
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Number of features in the hidden state.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The nonlinearity used by this cell.
    #[inline]
    pub fn nonlinearity(&self) -> RNNNonlinearity {
        self.nonlinearity
    }
}

impl<T: Float> Module<T> for RNNCell<T> {
    /// Forward with zero initial hidden state.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_cell(input, None)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![
            &mut self.weight_ih,
            &mut self.weight_hh,
            &mut self.bias_ih,
            &mut self.bias_hh,
        ]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![
            ("weight_ih".into(), &self.weight_ih),
            ("weight_hh".into(), &self.weight_hh),
            ("bias_ih".into(), &self.bias_ih),
            ("bias_hh".into(), &self.bias_hh),
        ]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// LSTMCell
// ===========================================================================

/// A single-step LSTM cell.
///
/// Computes:
/// ```text
/// gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
/// i = sigmoid(gates[0:H])
/// f = sigmoid(gates[H:2H])
/// g = tanh(gates[2H:3H])
/// o = sigmoid(gates[3H:4H])
/// c' = f * c + i * g
/// h' = o * tanh(c')
/// ```
///
/// This is the equivalent of `torch.nn.LSTMCell`.
#[derive(Debug)]
pub struct LSTMCell<T: Float> {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Parameter<T>,
    weight_hh: Parameter<T>,
    bias_ih: Parameter<T>,
    bias_hh: Parameter<T>,
    training: bool,
}

impl<T: Float> LSTMCell<T> {
    /// Create a new `LSTMCell`.
    ///
    /// # Arguments
    ///
    /// * `input_size` — number of expected features in `x`.
    /// * `hidden_size` — number of features in the hidden state `h`.
    pub fn new(input_size: usize, hidden_size: usize) -> FerrotorchResult<Self> {
        if hidden_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LSTMCell: hidden_size must be >= 1".into(),
            });
        }
        if input_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LSTMCell: input_size must be >= 1".into(),
            });
        }

        let k = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 4 * hidden_size;

        let mut weight_ih = Parameter::zeros(&[gate_size, input_size])?;
        let mut weight_hh = Parameter::zeros(&[gate_size, hidden_size])?;
        let mut bias_ih = Parameter::zeros(&[gate_size])?;
        let mut bias_hh = Parameter::zeros(&[gate_size])?;

        init::uniform(&mut weight_ih, -k, k)?;
        init::uniform(&mut weight_hh, -k, k)?;
        init::zeros(&mut bias_ih)?;
        init::zeros(&mut bias_hh)?;

        Ok(Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            training: true,
        })
    }

    /// Forward pass for the LSTM cell.
    ///
    /// # Arguments
    ///
    /// * `input` — input tensor of shape `[batch, input_size]`.
    /// * `state` — optional `(h, c)` each of shape `[batch, hidden_size]`.
    ///   If `None`, both are initialized to zeros.
    ///
    /// # Returns
    ///
    /// `(h', c')` each of shape `[batch, hidden_size]`.
    pub fn forward_cell(
        &self,
        input: &Tensor<T>,
        state: Option<(&Tensor<T>, &Tensor<T>)>,
    ) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        if input.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LSTMCell: expected 2-D input [batch, input_size], got shape {:?}",
                    input.shape()
                ),
            });
        }
        let batch = input.shape()[0];
        if input.shape()[1] != self.input_size {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LSTMCell: input_size mismatch: expected {}, got {}",
                    self.input_size,
                    input.shape()[1]
                ),
            });
        }

        let expected_h_shape = [batch, self.hidden_size];

        let (h_state, c_state) = match state {
            Some((h0, c0)) => {
                if h0.shape() != expected_h_shape {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "LSTMCell: h shape mismatch: expected {:?}, got {:?}",
                            expected_h_shape,
                            h0.shape()
                        ),
                    });
                }
                if c0.shape() != expected_h_shape {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "LSTMCell: c shape mismatch: expected {:?}, got {:?}",
                            expected_h_shape,
                            c0.shape()
                        ),
                    });
                }
                (h0.clone(), c0.clone())
            }
            None => {
                let h0 = ferrotorch_core::zeros::<T>(&[batch, self.hidden_size])?;
                let c0 = ferrotorch_core::zeros::<T>(&[batch, self.hidden_size])?;
                (h0, c0)
            }
        };

        let wih_t = transpose_2d(self.weight_ih.tensor())?;
        let whh_t = transpose_2d(self.weight_hh.tensor())?;

        let xw = mm(input, &wih_t)?; // [batch, 4*hs]
        let hw = mm(&h_state, &whh_t)?; // [batch, 4*hs]

        let bias_ih_2d = broadcast_bias_to_batch(&self.bias_ih, batch)?;
        let bias_hh_2d = broadcast_bias_to_batch(&self.bias_hh, batch)?;

        let gates = add(&add(&add(&xw, &bias_ih_2d)?, &hw)?, &bias_hh_2d)?;

        // Split gates into i, f, g, o — each [batch, hidden_size].
        let gate_chunks = gates.chunk(4, 1)?;
        let i_gate = sigmoid(&gate_chunks[0])?;
        let f_gate = sigmoid(&gate_chunks[1])?;
        let g_gate = tanh(&gate_chunks[2])?;
        let o_gate = sigmoid(&gate_chunks[3])?;

        // c' = f * c + i * g
        let c_new = add(&mul(&f_gate, &c_state)?, &mul(&i_gate, &g_gate)?)?;

        // h' = o * tanh(c')
        let h_new = mul(&o_gate, &tanh(&c_new)?)?;

        Ok((h_new, c_new))
    }

    /// Number of expected input features.
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Number of features in the hidden state.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl<T: Float> Module<T> for LSTMCell<T> {
    /// Forward with zero initial state. Returns `h'` only (drops `c'`).
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let (h, _c) = self.forward_cell(input, None)?;
        Ok(h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![
            &mut self.weight_ih,
            &mut self.weight_hh,
            &mut self.bias_ih,
            &mut self.bias_hh,
        ]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![
            ("weight_ih".into(), &self.weight_ih),
            ("weight_hh".into(), &self.weight_hh),
            ("bias_ih".into(), &self.bias_ih),
            ("bias_hh".into(), &self.bias_hh),
        ]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// GRUCell
// ===========================================================================

/// A single-step GRU cell.
///
/// Computes:
/// ```text
/// r = sigmoid(x @ W_ir^T + b_ir + h @ W_hr^T + b_hr)
/// z = sigmoid(x @ W_iz^T + b_iz + h @ W_hz^T + b_hz)
/// n = tanh(x @ W_in^T + b_in + r * (h @ W_hn^T + b_hn))
/// h' = (1 - z) * n + z * h
/// ```
///
/// This is the equivalent of `torch.nn.GRUCell`.
#[derive(Debug)]
pub struct GRUCell<T: Float> {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Parameter<T>,
    weight_hh: Parameter<T>,
    bias_ih: Parameter<T>,
    bias_hh: Parameter<T>,
    training: bool,
}

impl<T: Float> GRUCell<T> {
    /// Create a new `GRUCell`.
    ///
    /// # Arguments
    ///
    /// * `input_size` — number of expected features in `x`.
    /// * `hidden_size` — number of features in the hidden state `h`.
    pub fn new(input_size: usize, hidden_size: usize) -> FerrotorchResult<Self> {
        if hidden_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "GRUCell: hidden_size must be >= 1".into(),
            });
        }
        if input_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "GRUCell: input_size must be >= 1".into(),
            });
        }

        let k = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 3 * hidden_size;

        let mut weight_ih = Parameter::zeros(&[gate_size, input_size])?;
        let mut weight_hh = Parameter::zeros(&[gate_size, hidden_size])?;
        let mut bias_ih = Parameter::zeros(&[gate_size])?;
        let mut bias_hh = Parameter::zeros(&[gate_size])?;

        init::uniform(&mut weight_ih, -k, k)?;
        init::uniform(&mut weight_hh, -k, k)?;
        init::zeros(&mut bias_ih)?;
        init::zeros(&mut bias_hh)?;

        Ok(Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            training: true,
        })
    }

    /// Forward pass for the GRU cell.
    ///
    /// # Arguments
    ///
    /// * `input` — input tensor of shape `[batch, input_size]`.
    /// * `h` — hidden state of shape `[batch, hidden_size]`. If `None`,
    ///   initialized to zeros.
    ///
    /// # Returns
    ///
    /// New hidden state `h'` of shape `[batch, hidden_size]`.
    pub fn forward_cell(
        &self,
        input: &Tensor<T>,
        h: Option<&Tensor<T>>,
    ) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GRUCell: expected 2-D input [batch, input_size], got shape {:?}",
                    input.shape()
                ),
            });
        }
        let batch = input.shape()[0];
        if input.shape()[1] != self.input_size {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GRUCell: input_size mismatch: expected {}, got {}",
                    self.input_size,
                    input.shape()[1]
                ),
            });
        }

        let hs = self.hidden_size;

        let h_state = match h {
            Some(h0) => {
                if h0.shape() != [batch, hs] {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "GRUCell: h shape mismatch: expected {:?}, got {:?}",
                            [batch, hs],
                            h0.shape()
                        ),
                    });
                }
                h0.clone()
            }
            None => ferrotorch_core::zeros::<T>(&[batch, hs])?,
        };

        let wih_t = transpose_2d(self.weight_ih.tensor())?;
        let whh_t = transpose_2d(self.weight_hh.tensor())?;

        let xw = mm(input, &wih_t)?; // [batch, 3*hs]
        let hw = mm(&h_state, &whh_t)?; // [batch, 3*hs]

        let bias_ih_2d = broadcast_bias_to_batch(&self.bias_ih, batch)?;
        let bias_hh_2d = broadcast_bias_to_batch(&self.bias_hh, batch)?;

        let xw_b = add(&xw, &bias_ih_2d)?;
        let hw_b = add(&hw, &bias_hh_2d)?;

        // Split into r, z, n components — each [batch, hs].
        // Use manual slicing like the existing GRU impl.
        let xw_data = xw_b.data()?;
        let hw_data = hw_b.data()?;
        let gate_size = 3 * hs;

        let mut rx_data = Vec::with_capacity(batch * hs);
        let mut zx_data = Vec::with_capacity(batch * hs);
        let mut nx_data = Vec::with_capacity(batch * hs);
        let mut rh_data = Vec::with_capacity(batch * hs);
        let mut zh_data = Vec::with_capacity(batch * hs);
        let mut nh_data = Vec::with_capacity(batch * hs);

        for b_idx in 0..batch {
            let xbase = b_idx * gate_size;
            rx_data.extend_from_slice(&xw_data[xbase..xbase + hs]);
            zx_data.extend_from_slice(&xw_data[xbase + hs..xbase + 2 * hs]);
            nx_data.extend_from_slice(&xw_data[xbase + 2 * hs..xbase + 3 * hs]);

            let hbase = b_idx * gate_size;
            rh_data.extend_from_slice(&hw_data[hbase..hbase + hs]);
            zh_data.extend_from_slice(&hw_data[hbase + hs..hbase + 2 * hs]);
            nh_data.extend_from_slice(&hw_data[hbase + 2 * hs..hbase + 3 * hs]);
        }

        let rg = xw_b.requires_grad() || hw_b.requires_grad();

        let rx = Tensor::from_storage(TensorStorage::cpu(rx_data), vec![batch, hs], rg)?;
        let zx = Tensor::from_storage(TensorStorage::cpu(zx_data), vec![batch, hs], rg)?;
        let nx = Tensor::from_storage(TensorStorage::cpu(nx_data), vec![batch, hs], rg)?;
        let rh = Tensor::from_storage(TensorStorage::cpu(rh_data), vec![batch, hs], rg)?;
        let zh = Tensor::from_storage(TensorStorage::cpu(zh_data), vec![batch, hs], rg)?;
        let nh = Tensor::from_storage(TensorStorage::cpu(nh_data), vec![batch, hs], rg)?;

        // r = sigmoid(rx + rh), z = sigmoid(zx + zh)
        let r_gate = sigmoid(&add(&rx, &rh)?)?;
        let z_gate = sigmoid(&add(&zx, &zh)?)?;

        // n = tanh(nx + r * nh)
        let r_nh = mul(&r_gate, &nh)?;
        let n_gate = tanh(&add(&nx, &r_nh)?)?;

        // h' = (1 - z) * n + z * h
        let h_minus_n = sub(&h_state, &n_gate)?;
        let z_h_minus_n = mul(&z_gate, &h_minus_n)?;
        add(&n_gate, &z_h_minus_n)
    }

    /// Number of expected input features.
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Number of features in the hidden state.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl<T: Float> Module<T> for GRUCell<T> {
    /// Forward with zero initial hidden state.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_cell(input, None)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![
            &mut self.weight_ih,
            &mut self.weight_hh,
            &mut self.bias_ih,
            &mut self.bias_hh,
        ]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![
            ("weight_ih".into(), &self.weight_ih),
            ("weight_hh".into(), &self.weight_hh),
            ("bias_ih".into(), &self.bias_ih),
            ("bias_hh".into(), &self.bias_hh),
        ]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// RNN (multi-layer vanilla RNN)
// ===========================================================================

/// Output type for RNN forward: `(output_sequence, h_n)`.
type RnnOutput<T> = (Tensor<T>, Tensor<T>);

/// Per-layer parameters for the vanilla RNN.
#[derive(Debug, Clone)]
struct RNNLayerParams<T: Float> {
    /// Weight matrix for input-to-hidden: shape [hidden_size, input_size].
    weight_ih: Parameter<T>,
    /// Weight matrix for hidden-to-hidden: shape [hidden_size, hidden_size].
    weight_hh: Parameter<T>,
    /// Bias for input-to-hidden: shape [hidden_size].
    bias_ih: Parameter<T>,
    /// Bias for hidden-to-hidden: shape [hidden_size].
    bias_hh: Parameter<T>,
}

/// A multi-layer vanilla RNN (Elman network).
///
/// For each element in the input sequence, each layer computes:
///
/// ```text
/// h_t = nonlinearity(x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh)
/// ```
///
/// where `nonlinearity` is either `tanh` (default) or `relu`.
///
/// This is the equivalent of `torch.nn.RNN`.
#[derive(Debug)]
pub struct RNN<T: Float> {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    nonlinearity: RNNNonlinearity,
    layers: Vec<RNNLayerParams<T>>,
    training: bool,
}

impl<T: Float> RNN<T> {
    /// Create a new single-layer RNN with tanh nonlinearity.
    pub fn new(input_size: usize, hidden_size: usize) -> FerrotorchResult<Self> {
        Self::with_options(input_size, hidden_size, 1, RNNNonlinearity::Tanh)
    }

    /// Create a new RNN with the specified number of layers and nonlinearity.
    ///
    /// # Arguments
    ///
    /// * `input_size` — number of expected features in the input `x`.
    /// * `hidden_size` — number of features in the hidden state `h`.
    /// * `num_layers` — number of stacked RNN layers (must be >= 1).
    /// * `nonlinearity` — activation function to use.
    pub fn with_options(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        nonlinearity: RNNNonlinearity,
    ) -> FerrotorchResult<Self> {
        if num_layers == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RNN: num_layers must be >= 1".into(),
            });
        }
        if hidden_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RNN: hidden_size must be >= 1".into(),
            });
        }
        if input_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RNN: input_size must be >= 1".into(),
            });
        }

        let k = 1.0 / (hidden_size as f64).sqrt();

        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let layer_input_size = if layer_idx == 0 {
                input_size
            } else {
                hidden_size
            };

            let mut weight_ih = Parameter::zeros(&[hidden_size, layer_input_size])?;
            let mut weight_hh = Parameter::zeros(&[hidden_size, hidden_size])?;
            let mut bias_ih = Parameter::zeros(&[hidden_size])?;
            let mut bias_hh = Parameter::zeros(&[hidden_size])?;

            init::uniform(&mut weight_ih, -k, k)?;
            init::uniform(&mut weight_hh, -k, k)?;
            init::zeros(&mut bias_ih)?;
            init::zeros(&mut bias_hh)?;

            layers.push(RNNLayerParams {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
            });
        }

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            nonlinearity,
            layers,
            training: true,
        })
    }

    /// Forward pass with explicit hidden state.
    ///
    /// # Arguments
    ///
    /// * `input` — input tensor of shape `[batch, seq_len, input_size]`.
    /// * `h_0` — optional hidden state of shape `[num_layers, batch, hidden_size]`.
    ///   If `None`, initialized to zeros.
    ///
    /// # Returns
    ///
    /// A tuple `(output, h_n)` where:
    /// - `output` has shape `[batch, seq_len, hidden_size]` (last layer outputs).
    /// - `h_n` has shape `[num_layers, batch, hidden_size]`.
    pub fn forward_with_state(
        &self,
        input: &Tensor<T>,
        h_0: Option<&Tensor<T>>,
    ) -> FerrotorchResult<RnnOutput<T>> {
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RNN: expected 3-D input [batch, seq_len, input_size], got shape {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let seq_len = input.shape()[1];
        let hs = self.hidden_size;

        if input.shape()[2] != self.input_size {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "RNN: input_size mismatch: expected {}, got {}",
                    self.input_size,
                    input.shape()[2]
                ),
            });
        }

        // Initialize hidden state.
        let h_init = match h_0 {
            Some(h0) => {
                let expected_shape = [self.num_layers, batch, hs];
                if h0.shape() != expected_shape {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "RNN: h_0 shape mismatch: expected {:?}, got {:?}",
                            expected_shape,
                            h0.shape()
                        ),
                    });
                }
                h0.clone()
            }
            None => ferrotorch_core::zeros::<T>(&[self.num_layers, batch, hs])?,
        };

        // Extract per-timestep input slices.
        let input_data = input.data()?;

        let mut timestep_inputs: Vec<Tensor<T>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut slice_data = Vec::with_capacity(batch * self.input_size);
            for b in 0..batch {
                let offset = b * seq_len * self.input_size + t * self.input_size;
                slice_data.extend_from_slice(&input_data[offset..offset + self.input_size]);
            }
            timestep_inputs.push(Tensor::from_storage(
                TensorStorage::cpu(slice_data),
                vec![batch, self.input_size],
                input.requires_grad(),
            )?);
        }

        // Extract per-layer initial hidden states.
        let h_init_data = h_init.data()?;

        let mut layer_h: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let mut h_data = Vec::with_capacity(batch * hs);
            for b in 0..batch {
                let offset = l * batch * hs + b * hs;
                h_data.extend_from_slice(&h_init_data[offset..offset + hs]);
            }
            layer_h.push(Tensor::from_storage(
                TensorStorage::cpu(h_data),
                vec![batch, hs],
                false,
            )?);
        }

        // Run the RNN forward pass.
        let mut layer_outputs: Vec<Tensor<T>> = timestep_inputs;
        let mut final_h: Vec<Tensor<T>> = Vec::with_capacity(self.num_layers);

        for (l, params) in self.layers.iter().enumerate() {
            let mut h = layer_h[l].clone();
            let mut next_layer_outputs: Vec<Tensor<T>> = Vec::with_capacity(seq_len);

            let wih_t = transpose_2d(params.weight_ih.tensor())?;
            let whh_t = transpose_2d(params.weight_hh.tensor())?;

            for x_t in &layer_outputs {
                let xw = mm(x_t, &wih_t)?; // [batch, hs]
                let hw = mm(&h, &whh_t)?; // [batch, hs]

                let bias_ih_2d = broadcast_bias_to_batch(&params.bias_ih, batch)?;
                let bias_hh_2d = broadcast_bias_to_batch(&params.bias_hh, batch)?;

                let pre_act = add(&add(&add(&xw, &bias_ih_2d)?, &hw)?, &bias_hh_2d)?;

                let h_new = match self.nonlinearity {
                    RNNNonlinearity::Tanh => tanh(&pre_act)?,
                    RNNNonlinearity::ReLU => relu(&pre_act)?,
                };

                next_layer_outputs.push(h_new.clone());
                h = h_new;
            }

            final_h.push(h);
            layer_outputs = next_layer_outputs;
        }

        // Assemble output: [batch, seq_len, hidden_size] from the last layer.
        let mut output_data = Vec::with_capacity(batch * seq_len * hs);
        for b_idx in 0..batch {
            for lo in &layer_outputs {
                let t_data = lo.data()?;
                let offset = b_idx * hs;
                output_data.extend_from_slice(&t_data[offset..offset + hs]);
            }
        }
        let output = Tensor::from_storage(
            TensorStorage::cpu(output_data),
            vec![batch, seq_len, hs],
            false,
        )?;

        // Assemble h_n: [num_layers, batch, hidden_size].
        let mut h_n_data = Vec::with_capacity(self.num_layers * batch * hs);
        for h_l_tensor in &final_h {
            let h_l = h_l_tensor.data()?;
            h_n_data.extend_from_slice(h_l);
        }
        let h_n = Tensor::from_storage(
            TensorStorage::cpu(h_n_data),
            vec![self.num_layers, batch, hs],
            false,
        )?;

        Ok((output, h_n))
    }

    /// Number of expected input features.
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Number of features in the hidden state.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of stacked RNN layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// The nonlinearity used by this module.
    #[inline]
    pub fn nonlinearity(&self) -> RNNNonlinearity {
        self.nonlinearity
    }
}

impl<T: Float> Module<T> for RNN<T> {
    /// Forward pass using the `Module` interface (no explicit hidden state).
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let (output, _) = self.forward_with_state(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &self.layers {
            params.push(&layer.weight_ih);
            params.push(&layer.weight_hh);
            params.push(&layer.bias_ih);
            params.push(&layer.bias_hh);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &mut self.layers {
            params.push(&mut layer.weight_ih);
            params.push(&mut layer.weight_hh);
            params.push(&mut layer.bias_ih);
            params.push(&mut layer.bias_hh);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for (i, layer) in self.layers.iter().enumerate() {
            params.push((format!("layers.{i}.weight_ih"), &layer.weight_ih));
            params.push((format!("layers.{i}.weight_hh"), &layer.weight_hh));
            params.push((format!("layers.{i}.bias_ih"), &layer.bias_ih));
            params.push((format!("layers.{i}.bias_hh"), &layer.bias_hh));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_new_basic() {
        let lstm = LSTM::<f32>::new(10, 20, 1).unwrap();
        assert_eq!(lstm.input_size(), 10);
        assert_eq!(lstm.hidden_size(), 20);
        assert_eq!(lstm.num_layers(), 1);
    }

    #[test]
    fn test_lstm_parameter_count() {
        let lstm = LSTM::<f32>::new(10, 20, 2).unwrap();
        // Layer 0: weight_ih [80,10], weight_hh [80,20], bias_ih [80], bias_hh [80]
        // Layer 1: weight_ih [80,20], weight_hh [80,20], bias_ih [80], bias_hh [80]
        let params = lstm.parameters();
        assert_eq!(params.len(), 8); // 4 per layer * 2 layers
    }

    #[test]
    fn test_lstm_parameter_shapes() {
        let lstm = LSTM::<f32>::new(10, 20, 1).unwrap();
        let params = lstm.parameters();
        // weight_ih: [80, 10]
        assert_eq!(params[0].shape(), &[80, 10]);
        // weight_hh: [80, 20]
        assert_eq!(params[1].shape(), &[80, 20]);
        // bias_ih: [80]
        assert_eq!(params[2].shape(), &[80]);
        // bias_hh: [80]
        assert_eq!(params[3].shape(), &[80]);
    }

    #[test]
    fn test_lstm_new_invalid_num_layers() {
        assert!(LSTM::<f32>::new(10, 20, 0).is_err());
    }

    #[test]
    fn test_lstm_new_invalid_hidden_size() {
        assert!(LSTM::<f32>::new(10, 0, 1).is_err());
    }

    #[test]
    fn test_lstm_new_invalid_input_size() {
        assert!(LSTM::<f32>::new(0, 20, 1).is_err());
    }

    // -----------------------------------------------------------------------
    // Weight initialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_weight_init_range() {
        let hs = 100;
        let lstm = LSTM::<f32>::new(50, hs, 1).unwrap();
        let k = 1.0 / (hs as f32).sqrt();
        let params = lstm.parameters();

        // Weights should be in U(-k, k).
        for param in &params[..2] {
            let data = param.data().unwrap();
            for &v in data {
                assert!(
                    v.abs() <= k + 0.01,
                    "weight value {v} exceeds expected range [-{k}, {k}]"
                );
            }
        }

        // Biases should be zeros.
        for param in &params[2..4] {
            let data = param.data().unwrap();
            assert!(
                data.iter().all(|&v| v == 0.0),
                "bias should be initialized to zeros"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Forward pass — output shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_forward_output_shape() {
        let lstm = LSTM::<f32>::new(10, 20, 1).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 10]).unwrap(); // [B=2, T=5, F=10]

        let (output, (h_n, c_n)) = lstm.forward_with_state(&input, None).unwrap();

        assert_eq!(output.shape(), &[2, 5, 20]); // [B, T, hidden]
        assert_eq!(h_n.shape(), &[1, 2, 20]); // [layers, B, hidden]
        assert_eq!(c_n.shape(), &[1, 2, 20]);
    }

    #[test]
    fn test_lstm_forward_multi_layer_shapes() {
        let lstm = LSTM::<f32>::new(8, 16, 3).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[4, 7, 8]).unwrap(); // [B=4, T=7, F=8]

        let (output, (h_n, c_n)) = lstm.forward_with_state(&input, None).unwrap();

        assert_eq!(output.shape(), &[4, 7, 16]); // [B, T, hidden]
        assert_eq!(h_n.shape(), &[3, 4, 16]); // [layers, B, hidden]
        assert_eq!(c_n.shape(), &[3, 4, 16]);
    }

    #[test]
    fn test_lstm_module_forward_shape() {
        let lstm = LSTM::<f32>::new(10, 20, 1).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 10]).unwrap();

        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 20]);
    }

    // -----------------------------------------------------------------------
    // Forward pass — basic sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_forward_does_not_error() {
        let lstm = LSTM::<f32>::new(4, 8, 2).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[3, 10, 4]).unwrap();

        let result = lstm.forward_with_state(&input, None);
        assert!(
            result.is_ok(),
            "forward should not error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_lstm_forward_nonzero_output() {
        // With random weights and random input, the output should not be all zeros.
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[1, 3, 4]).unwrap();

        let (output, _) = lstm.forward_with_state(&input, None).unwrap();
        let data = output.data().unwrap();
        let any_nonzero = data.iter().any(|&v| v.abs() > 1e-10);
        assert!(any_nonzero, "output should have non-zero values");
    }

    #[test]
    fn test_lstm_forward_seq_len_1() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 1, 4]).unwrap();

        let (output, (h_n, c_n)) = lstm.forward_with_state(&input, None).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8]);
        assert_eq!(h_n.shape(), &[1, 1, 8]);
        assert_eq!(c_n.shape(), &[1, 1, 8]);
    }

    // -----------------------------------------------------------------------
    // Forward with explicit state
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_forward_with_initial_state() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();

        let h0 = ferrotorch_core::zeros::<f32>(&[1, 2, 8]).unwrap();
        let c0 = ferrotorch_core::zeros::<f32>(&[1, 2, 8]).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[2, 3, 4]).unwrap();

        let result = lstm.forward_with_state(&input, Some((&h0, &c0)));
        assert!(result.is_ok());
    }

    #[test]
    fn test_lstm_forward_state_shape_mismatch() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();

        // Wrong batch size in h0.
        let h0 = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let c0 = ferrotorch_core::zeros::<f32>(&[1, 2, 8]).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[2, 3, 4]).unwrap();

        assert!(lstm.forward_with_state(&input, Some((&h0, &c0))).is_err());
    }

    #[test]
    fn test_lstm_forward_input_wrong_ndim() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[10, 4]).unwrap(); // 2-D, not 3-D
        assert!(lstm.forward_with_state(&input, None).is_err());
    }

    #[test]
    fn test_lstm_forward_input_size_mismatch() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 5, 7]).unwrap(); // input_size=7 != 4
        assert!(lstm.forward_with_state(&input, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Module trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_named_parameters() {
        let lstm = LSTM::<f32>::new(4, 8, 2).unwrap();
        let named = lstm.named_parameters();
        assert_eq!(named.len(), 8);
        assert_eq!(named[0].0, "layers.0.weight_ih");
        assert_eq!(named[1].0, "layers.0.weight_hh");
        assert_eq!(named[2].0, "layers.0.bias_ih");
        assert_eq!(named[3].0, "layers.0.bias_hh");
        assert_eq!(named[4].0, "layers.1.weight_ih");
        assert_eq!(named[5].0, "layers.1.weight_hh");
        assert_eq!(named[6].0, "layers.1.bias_ih");
        assert_eq!(named[7].0, "layers.1.bias_hh");
    }

    #[test]
    fn test_lstm_train_eval() {
        let mut lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        assert!(lstm.is_training());
        lstm.eval();
        assert!(!lstm.is_training());
        lstm.train();
        assert!(lstm.is_training());
    }

    #[test]
    fn test_lstm_all_parameters_require_grad() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        for param in lstm.parameters() {
            assert!(param.requires_grad());
        }
    }

    #[test]
    fn test_lstm_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LSTM<f32>>();
        assert_send_sync::<LSTM<f64>>();
    }

    // -----------------------------------------------------------------------
    // Multi-layer: second layer input_size equals hidden_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_multi_layer_weight_shapes() {
        let lstm = LSTM::<f32>::new(10, 20, 3).unwrap();
        let params = lstm.parameters();

        // Layer 0: weight_ih [80, 10] (input_size=10)
        assert_eq!(params[0].shape(), &[80, 10]);
        // Layer 0: weight_hh [80, 20]
        assert_eq!(params[1].shape(), &[80, 20]);

        // Layer 1: weight_ih [80, 20] (input_size=hidden_size=20)
        assert_eq!(params[4].shape(), &[80, 20]);
        // Layer 1: weight_hh [80, 20]
        assert_eq!(params[5].shape(), &[80, 20]);

        // Layer 2: weight_ih [80, 20]
        assert_eq!(params[8].shape(), &[80, 20]);
    }

    // -----------------------------------------------------------------------
    // State dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_state_dict_roundtrip() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        let sd = lstm.state_dict();
        assert_eq!(sd.len(), 4);
        assert!(sd.contains_key("layers.0.weight_ih"));
        assert!(sd.contains_key("layers.0.weight_hh"));
        assert!(sd.contains_key("layers.0.bias_ih"));
        assert!(sd.contains_key("layers.0.bias_hh"));

        let mut lstm2 = LSTM::<f32>::new(4, 8, 1).unwrap();
        lstm2.load_state_dict(&sd, true).unwrap();
    }

    // -----------------------------------------------------------------------
    // Consistency: feeding the same input twice gives the same output
    // -----------------------------------------------------------------------

    #[test]
    fn test_lstm_deterministic() {
        let lstm = LSTM::<f32>::new(4, 8, 1).unwrap();
        let input = ferrotorch_core::from_slice::<f32>(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            &[1, 2, 4],
        )
        .unwrap();

        let (out1, _) = lstm.forward_with_state(&input, None).unwrap();
        let (out2, _) = lstm.forward_with_state(&input, None).unwrap();

        let d1 = out1.data().unwrap();
        let d2 = out2.data().unwrap();
        for (i, (&a, &b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "output mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    // =======================================================================
    // GRU tests
    // =======================================================================

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_new_basic() {
        let gru = GRU::<f32>::new(10, 20).unwrap();
        assert_eq!(gru.input_size(), 10);
        assert_eq!(gru.hidden_size(), 20);
        assert_eq!(gru.num_layers(), 1);
    }

    #[test]
    fn test_gru_with_num_layers() {
        let gru = GRU::<f32>::with_num_layers(10, 20, 3).unwrap();
        assert_eq!(gru.num_layers(), 3);
    }

    #[test]
    fn test_gru_parameter_count() {
        let gru = GRU::<f32>::with_num_layers(10, 20, 2).unwrap();
        // Layer 0: weight_ih [60,10], weight_hh [60,20], bias_ih [60], bias_hh [60]
        // Layer 1: weight_ih [60,20], weight_hh [60,20], bias_ih [60], bias_hh [60]
        let params = gru.parameters();
        assert_eq!(params.len(), 8); // 4 per layer * 2 layers
    }

    #[test]
    fn test_gru_parameter_shapes() {
        let gru = GRU::<f32>::new(10, 20).unwrap();
        let params = gru.parameters();
        // weight_ih: [60, 10] (3 * hidden_size = 60)
        assert_eq!(params[0].shape(), &[60, 10]);
        // weight_hh: [60, 20]
        assert_eq!(params[1].shape(), &[60, 20]);
        // bias_ih: [60]
        assert_eq!(params[2].shape(), &[60]);
        // bias_hh: [60]
        assert_eq!(params[3].shape(), &[60]);
    }

    #[test]
    fn test_gru_new_invalid_num_layers() {
        assert!(GRU::<f32>::with_num_layers(10, 20, 0).is_err());
    }

    #[test]
    fn test_gru_new_invalid_hidden_size() {
        assert!(GRU::<f32>::new(10, 0).is_err());
    }

    #[test]
    fn test_gru_new_invalid_input_size() {
        assert!(GRU::<f32>::new(0, 20).is_err());
    }

    // -----------------------------------------------------------------------
    // Weight initialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_weight_init_range() {
        let hs = 100;
        let gru = GRU::<f32>::new(50, hs).unwrap();
        let k = 1.0 / (hs as f32).sqrt();
        let params = gru.parameters();

        // Weights should be in U(-k, k).
        for param in &params[..2] {
            let data = param.data().unwrap();
            for &v in data {
                assert!(
                    v.abs() <= k + 0.01,
                    "weight value {v} exceeds expected range [-{k}, {k}]"
                );
            }
        }

        // Biases should be zeros.
        for param in &params[2..4] {
            let data = param.data().unwrap();
            assert!(
                data.iter().all(|&v| v == 0.0),
                "bias should be initialized to zeros"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Forward pass — output shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_forward_output_shape() {
        let gru = GRU::<f32>::new(10, 20).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 10]).unwrap();

        let (output, h_n) = gru.forward(&input, None).unwrap();

        assert_eq!(output.shape(), &[2, 5, 20]); // [B, T, hidden]
        assert_eq!(h_n.shape(), &[1, 2, 20]); // [layers, B, hidden]
    }

    #[test]
    fn test_gru_forward_multi_layer_shapes() {
        let gru = GRU::<f32>::with_num_layers(8, 16, 3).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[4, 7, 8]).unwrap();

        let (output, h_n) = gru.forward(&input, None).unwrap();

        assert_eq!(output.shape(), &[4, 7, 16]);
        assert_eq!(h_n.shape(), &[3, 4, 16]);
    }

    #[test]
    fn test_gru_module_forward_shape() {
        let gru = GRU::<f32>::new(10, 20).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 10]).unwrap();

        let output = <GRU<f32> as Module<f32>>::forward(&gru, &input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 20]);
    }

    // -----------------------------------------------------------------------
    // Forward pass — basic sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_forward_does_not_error() {
        let gru = GRU::<f32>::with_num_layers(4, 8, 2).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[3, 10, 4]).unwrap();

        let result = gru.forward(&input, None);
        assert!(
            result.is_ok(),
            "forward should not error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gru_forward_nonzero_output() {
        let gru = GRU::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[1, 3, 4]).unwrap();

        let (output, _) = gru.forward(&input, None).unwrap();
        let data = output.data().unwrap();
        let any_nonzero = data.iter().any(|&v| v.abs() > 1e-10);
        assert!(any_nonzero, "output should have non-zero values");
    }

    #[test]
    fn test_gru_forward_seq_len_1() {
        let gru = GRU::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 1, 4]).unwrap();

        let (output, h_n) = gru.forward(&input, None).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8]);
        assert_eq!(h_n.shape(), &[1, 1, 8]);
    }

    // -----------------------------------------------------------------------
    // Forward with explicit state
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_forward_with_initial_state() {
        let gru = GRU::<f32>::new(4, 8).unwrap();

        let h0 = ferrotorch_core::zeros::<f32>(&[1, 2, 8]).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[2, 3, 4]).unwrap();

        let result = gru.forward(&input, Some(&h0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_gru_forward_state_shape_mismatch() {
        let gru = GRU::<f32>::new(4, 8).unwrap();

        // Wrong batch size in h0.
        let h0 = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[2, 3, 4]).unwrap();

        assert!(gru.forward(&input, Some(&h0)).is_err());
    }

    #[test]
    fn test_gru_forward_input_wrong_ndim() {
        let gru = GRU::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[10, 4]).unwrap();
        assert!(gru.forward(&input, None).is_err());
    }

    #[test]
    fn test_gru_forward_input_size_mismatch() {
        let gru = GRU::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 5, 7]).unwrap();
        assert!(gru.forward(&input, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Module trait (GRU)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_named_parameters() {
        let gru = GRU::<f32>::with_num_layers(4, 8, 2).unwrap();
        let named = gru.named_parameters();
        assert_eq!(named.len(), 8);
        assert_eq!(named[0].0, "layers.0.weight_ih");
        assert_eq!(named[1].0, "layers.0.weight_hh");
        assert_eq!(named[2].0, "layers.0.bias_ih");
        assert_eq!(named[3].0, "layers.0.bias_hh");
        assert_eq!(named[4].0, "layers.1.weight_ih");
        assert_eq!(named[5].0, "layers.1.weight_hh");
        assert_eq!(named[6].0, "layers.1.bias_ih");
        assert_eq!(named[7].0, "layers.1.bias_hh");
    }

    #[test]
    fn test_gru_train_eval() {
        let mut gru = GRU::<f32>::new(4, 8).unwrap();
        assert!(gru.is_training());
        gru.eval();
        assert!(!gru.is_training());
        gru.train();
        assert!(gru.is_training());
    }

    #[test]
    fn test_gru_all_parameters_require_grad() {
        let gru = GRU::<f32>::new(4, 8).unwrap();
        for param in gru.parameters() {
            assert!(param.requires_grad());
        }
    }

    #[test]
    fn test_gru_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GRU<f32>>();
        assert_send_sync::<GRU<f64>>();
    }

    // -----------------------------------------------------------------------
    // Multi-layer weight shapes (GRU)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_multi_layer_weight_shapes() {
        let gru = GRU::<f32>::with_num_layers(10, 20, 3).unwrap();
        let params = gru.parameters();

        // Layer 0: weight_ih [60, 10] (input_size=10)
        assert_eq!(params[0].shape(), &[60, 10]);
        // Layer 0: weight_hh [60, 20]
        assert_eq!(params[1].shape(), &[60, 20]);

        // Layer 1: weight_ih [60, 20] (input_size=hidden_size=20)
        assert_eq!(params[4].shape(), &[60, 20]);
        // Layer 1: weight_hh [60, 20]
        assert_eq!(params[5].shape(), &[60, 20]);

        // Layer 2: weight_ih [60, 20]
        assert_eq!(params[8].shape(), &[60, 20]);
    }

    // -----------------------------------------------------------------------
    // State dict roundtrip (GRU)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_state_dict_roundtrip() {
        let gru = GRU::<f32>::new(4, 8).unwrap();
        let sd = gru.state_dict();
        assert_eq!(sd.len(), 4);
        assert!(sd.contains_key("layers.0.weight_ih"));
        assert!(sd.contains_key("layers.0.weight_hh"));
        assert!(sd.contains_key("layers.0.bias_ih"));
        assert!(sd.contains_key("layers.0.bias_hh"));

        let mut gru2 = GRU::<f32>::new(4, 8).unwrap();
        gru2.load_state_dict(&sd, true).unwrap();
    }

    // -----------------------------------------------------------------------
    // Determinism (GRU)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gru_deterministic() {
        let gru = GRU::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::from_slice::<f32>(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            &[1, 2, 4],
        )
        .unwrap();

        let (out1, _) = gru.forward(&input, None).unwrap();
        let (out2, _) = gru.forward(&input, None).unwrap();

        let d1 = out1.data().unwrap();
        let d2 = out2.data().unwrap();
        for (i, (&a, &b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "output mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    // =======================================================================
    // RNNCell tests
    // =======================================================================

    #[test]
    fn test_rnn_cell_new_basic() {
        let cell = RNNCell::<f32>::new(10, 20).unwrap();
        assert_eq!(cell.input_size(), 10);
        assert_eq!(cell.hidden_size(), 20);
        assert_eq!(cell.nonlinearity(), RNNNonlinearity::Tanh);
    }

    #[test]
    fn test_rnn_cell_relu() {
        let cell = RNNCell::<f32>::with_nonlinearity(10, 20, RNNNonlinearity::ReLU).unwrap();
        assert_eq!(cell.nonlinearity(), RNNNonlinearity::ReLU);
    }

    #[test]
    fn test_rnn_cell_invalid_sizes() {
        assert!(RNNCell::<f32>::new(0, 20).is_err());
        assert!(RNNCell::<f32>::new(10, 0).is_err());
    }

    #[test]
    fn test_rnn_cell_parameter_shapes() {
        let cell = RNNCell::<f32>::new(10, 20).unwrap();
        let params = cell.parameters();
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].shape(), &[20, 10]); // weight_ih
        assert_eq!(params[1].shape(), &[20, 20]); // weight_hh
        assert_eq!(params[2].shape(), &[20]); // bias_ih
        assert_eq!(params[3].shape(), &[20]); // bias_hh
    }

    #[test]
    fn test_rnn_cell_forward_output_shape() {
        let cell = RNNCell::<f32>::new(10, 20).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[3, 10]).unwrap();
        let h = cell.forward_cell(&x, None).unwrap();
        assert_eq!(h.shape(), &[3, 20]);
    }

    #[test]
    fn test_rnn_cell_forward_with_hidden() {
        let cell = RNNCell::<f32>::new(10, 20).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[3, 10]).unwrap();
        let h0 = ferrotorch_core::randn::<f32>(&[3, 20]).unwrap();
        let h = cell.forward_cell(&x, Some(&h0)).unwrap();
        assert_eq!(h.shape(), &[3, 20]);
    }

    #[test]
    fn test_rnn_cell_forward_nonzero() {
        let cell = RNNCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[1, 4]).unwrap();
        let h = cell.forward_cell(&x, None).unwrap();
        let data = h.data().unwrap();
        assert!(data.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_rnn_cell_forward_bad_input_ndim() {
        let cell = RNNCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[1, 2, 4]).unwrap();
        assert!(cell.forward_cell(&x, None).is_err());
    }

    #[test]
    fn test_rnn_cell_forward_bad_input_size() {
        let cell = RNNCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[1, 7]).unwrap();
        assert!(cell.forward_cell(&x, None).is_err());
    }

    #[test]
    fn test_rnn_cell_forward_bad_h_shape() {
        let cell = RNNCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
        let h0 = ferrotorch_core::zeros::<f32>(&[3, 8]).unwrap(); // wrong batch
        assert!(cell.forward_cell(&x, Some(&h0)).is_err());
    }

    #[test]
    fn test_rnn_cell_module_forward() {
        let cell = RNNCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[2, 4]).unwrap();
        let h = <RNNCell<f32> as Module<f32>>::forward(&cell, &x).unwrap();
        assert_eq!(h.shape(), &[2, 8]);
    }

    #[test]
    fn test_rnn_cell_named_parameters() {
        let cell = RNNCell::<f32>::new(4, 8).unwrap();
        let named = cell.named_parameters();
        assert_eq!(named.len(), 4);
        assert_eq!(named[0].0, "weight_ih");
        assert_eq!(named[1].0, "weight_hh");
        assert_eq!(named[2].0, "bias_ih");
        assert_eq!(named[3].0, "bias_hh");
    }

    #[test]
    fn test_rnn_cell_train_eval() {
        let mut cell = RNNCell::<f32>::new(4, 8).unwrap();
        assert!(cell.is_training());
        cell.eval();
        assert!(!cell.is_training());
        cell.train();
        assert!(cell.is_training());
    }

    #[test]
    fn test_rnn_cell_deterministic() {
        let cell = RNNCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &[1, 4]).unwrap();
        let h1 = cell.forward_cell(&x, None).unwrap();
        let h2 = cell.forward_cell(&x, None).unwrap();
        let d1 = h1.data().unwrap();
        let d2 = h2.data().unwrap();
        for (i, (&a, &b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_rnn_cell_relu_output_nonneg() {
        // With relu nonlinearity, all outputs should be >= 0.
        let cell = RNNCell::<f32>::with_nonlinearity(4, 8, RNNNonlinearity::ReLU).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[5, 4]).unwrap();
        let h = cell.forward_cell(&x, None).unwrap();
        let data = h.data().unwrap();
        assert!(
            data.iter().all(|&v| v >= 0.0),
            "relu output should be non-negative"
        );
    }

    #[test]
    fn test_rnn_cell_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RNNCell<f32>>();
        assert_send_sync::<RNNCell<f64>>();
    }

    // =======================================================================
    // LSTMCell tests
    // =======================================================================

    #[test]
    fn test_lstm_cell_new_basic() {
        let cell = LSTMCell::<f32>::new(10, 20).unwrap();
        assert_eq!(cell.input_size(), 10);
        assert_eq!(cell.hidden_size(), 20);
    }

    #[test]
    fn test_lstm_cell_invalid_sizes() {
        assert!(LSTMCell::<f32>::new(0, 20).is_err());
        assert!(LSTMCell::<f32>::new(10, 0).is_err());
    }

    #[test]
    fn test_lstm_cell_parameter_shapes() {
        let cell = LSTMCell::<f32>::new(10, 20).unwrap();
        let params = cell.parameters();
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].shape(), &[80, 10]); // weight_ih [4*hs, input]
        assert_eq!(params[1].shape(), &[80, 20]); // weight_hh [4*hs, hs]
        assert_eq!(params[2].shape(), &[80]); // bias_ih
        assert_eq!(params[3].shape(), &[80]); // bias_hh
    }

    #[test]
    fn test_lstm_cell_forward_output_shape() {
        let cell = LSTMCell::<f32>::new(10, 20).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[3, 10]).unwrap();
        let (h, c) = cell.forward_cell(&x, None).unwrap();
        assert_eq!(h.shape(), &[3, 20]);
        assert_eq!(c.shape(), &[3, 20]);
    }

    #[test]
    fn test_lstm_cell_forward_with_state() {
        let cell = LSTMCell::<f32>::new(10, 20).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[3, 10]).unwrap();
        let h0 = ferrotorch_core::randn::<f32>(&[3, 20]).unwrap();
        let c0 = ferrotorch_core::randn::<f32>(&[3, 20]).unwrap();
        let (h, c) = cell.forward_cell(&x, Some((&h0, &c0))).unwrap();
        assert_eq!(h.shape(), &[3, 20]);
        assert_eq!(c.shape(), &[3, 20]);
    }

    #[test]
    fn test_lstm_cell_forward_nonzero() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[1, 4]).unwrap();
        let (h, c) = cell.forward_cell(&x, None).unwrap();
        let hd = h.data().unwrap();
        let cd = c.data().unwrap();
        assert!(hd.iter().any(|&v| v.abs() > 1e-10));
        assert!(cd.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_lstm_cell_forward_bad_input_ndim() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[1, 2, 4]).unwrap();
        assert!(cell.forward_cell(&x, None).is_err());
    }

    #[test]
    fn test_lstm_cell_forward_bad_input_size() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[1, 7]).unwrap();
        assert!(cell.forward_cell(&x, None).is_err());
    }

    #[test]
    fn test_lstm_cell_forward_bad_h_shape() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
        let h0 = ferrotorch_core::zeros::<f32>(&[3, 8]).unwrap(); // wrong batch
        let c0 = ferrotorch_core::zeros::<f32>(&[2, 8]).unwrap();
        assert!(cell.forward_cell(&x, Some((&h0, &c0))).is_err());
    }

    #[test]
    fn test_lstm_cell_forward_bad_c_shape() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
        let h0 = ferrotorch_core::zeros::<f32>(&[2, 8]).unwrap();
        let c0 = ferrotorch_core::zeros::<f32>(&[2, 99]).unwrap(); // wrong hs
        assert!(cell.forward_cell(&x, Some((&h0, &c0))).is_err());
    }

    #[test]
    fn test_lstm_cell_module_forward() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[2, 4]).unwrap();
        // Module::forward returns h only.
        let h = <LSTMCell<f32> as Module<f32>>::forward(&cell, &x).unwrap();
        assert_eq!(h.shape(), &[2, 8]);
    }

    #[test]
    fn test_lstm_cell_named_parameters() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let named = cell.named_parameters();
        assert_eq!(named.len(), 4);
        assert_eq!(named[0].0, "weight_ih");
        assert_eq!(named[1].0, "weight_hh");
        assert_eq!(named[2].0, "bias_ih");
        assert_eq!(named[3].0, "bias_hh");
    }

    #[test]
    fn test_lstm_cell_train_eval() {
        let mut cell = LSTMCell::<f32>::new(4, 8).unwrap();
        assert!(cell.is_training());
        cell.eval();
        assert!(!cell.is_training());
        cell.train();
        assert!(cell.is_training());
    }

    #[test]
    fn test_lstm_cell_deterministic() {
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &[1, 4]).unwrap();
        let (h1, c1) = cell.forward_cell(&x, None).unwrap();
        let (h2, c2) = cell.forward_cell(&x, None).unwrap();
        let hd1 = h1.data().unwrap();
        let hd2 = h2.data().unwrap();
        let cd1 = c1.data().unwrap();
        let cd2 = c2.data().unwrap();
        for (i, (&a, &b)) in hd1.iter().zip(hd2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "h mismatch at {i}: {a} vs {b}");
        }
        for (i, (&a, &b)) in cd1.iter().zip(cd2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "c mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_lstm_cell_h_bounded_by_tanh() {
        // h = o * tanh(c), so |h| <= 1 always.
        let cell = LSTMCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[10, 4]).unwrap();
        let (h, _c) = cell.forward_cell(&x, None).unwrap();
        let data = h.data().unwrap();
        assert!(
            data.iter().all(|&v| v.abs() <= 1.0 + 1e-6),
            "LSTM cell h should be bounded by [-1, 1]"
        );
    }

    #[test]
    fn test_lstm_cell_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LSTMCell<f32>>();
        assert_send_sync::<LSTMCell<f64>>();
    }

    // =======================================================================
    // GRUCell tests
    // =======================================================================

    #[test]
    fn test_gru_cell_new_basic() {
        let cell = GRUCell::<f32>::new(10, 20).unwrap();
        assert_eq!(cell.input_size(), 10);
        assert_eq!(cell.hidden_size(), 20);
    }

    #[test]
    fn test_gru_cell_invalid_sizes() {
        assert!(GRUCell::<f32>::new(0, 20).is_err());
        assert!(GRUCell::<f32>::new(10, 0).is_err());
    }

    #[test]
    fn test_gru_cell_parameter_shapes() {
        let cell = GRUCell::<f32>::new(10, 20).unwrap();
        let params = cell.parameters();
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].shape(), &[60, 10]); // weight_ih [3*hs, input]
        assert_eq!(params[1].shape(), &[60, 20]); // weight_hh [3*hs, hs]
        assert_eq!(params[2].shape(), &[60]); // bias_ih
        assert_eq!(params[3].shape(), &[60]); // bias_hh
    }

    #[test]
    fn test_gru_cell_forward_output_shape() {
        let cell = GRUCell::<f32>::new(10, 20).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[3, 10]).unwrap();
        let h = cell.forward_cell(&x, None).unwrap();
        assert_eq!(h.shape(), &[3, 20]);
    }

    #[test]
    fn test_gru_cell_forward_with_hidden() {
        let cell = GRUCell::<f32>::new(10, 20).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[3, 10]).unwrap();
        let h0 = ferrotorch_core::randn::<f32>(&[3, 20]).unwrap();
        let h = cell.forward_cell(&x, Some(&h0)).unwrap();
        assert_eq!(h.shape(), &[3, 20]);
    }

    #[test]
    fn test_gru_cell_forward_nonzero() {
        let cell = GRUCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[1, 4]).unwrap();
        let h = cell.forward_cell(&x, None).unwrap();
        let data = h.data().unwrap();
        assert!(data.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_gru_cell_forward_bad_input_ndim() {
        let cell = GRUCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[1, 2, 4]).unwrap();
        assert!(cell.forward_cell(&x, None).is_err());
    }

    #[test]
    fn test_gru_cell_forward_bad_input_size() {
        let cell = GRUCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[1, 7]).unwrap();
        assert!(cell.forward_cell(&x, None).is_err());
    }

    #[test]
    fn test_gru_cell_forward_bad_h_shape() {
        let cell = GRUCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
        let h0 = ferrotorch_core::zeros::<f32>(&[3, 8]).unwrap(); // wrong batch
        assert!(cell.forward_cell(&x, Some(&h0)).is_err());
    }

    #[test]
    fn test_gru_cell_module_forward() {
        let cell = GRUCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::randn::<f32>(&[2, 4]).unwrap();
        let h = <GRUCell<f32> as Module<f32>>::forward(&cell, &x).unwrap();
        assert_eq!(h.shape(), &[2, 8]);
    }

    #[test]
    fn test_gru_cell_named_parameters() {
        let cell = GRUCell::<f32>::new(4, 8).unwrap();
        let named = cell.named_parameters();
        assert_eq!(named.len(), 4);
        assert_eq!(named[0].0, "weight_ih");
        assert_eq!(named[1].0, "weight_hh");
        assert_eq!(named[2].0, "bias_ih");
        assert_eq!(named[3].0, "bias_hh");
    }

    #[test]
    fn test_gru_cell_train_eval() {
        let mut cell = GRUCell::<f32>::new(4, 8).unwrap();
        assert!(cell.is_training());
        cell.eval();
        assert!(!cell.is_training());
        cell.train();
        assert!(cell.is_training());
    }

    #[test]
    fn test_gru_cell_deterministic() {
        let cell = GRUCell::<f32>::new(4, 8).unwrap();
        let x = ferrotorch_core::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &[1, 4]).unwrap();
        let h1 = cell.forward_cell(&x, None).unwrap();
        let h2 = cell.forward_cell(&x, None).unwrap();
        let d1 = h1.data().unwrap();
        let d2 = h2.data().unwrap();
        for (i, (&a, &b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_gru_cell_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GRUCell<f32>>();
        assert_send_sync::<GRUCell<f64>>();
    }

    // =======================================================================
    // RNN (multi-layer) tests
    // =======================================================================

    #[test]
    fn test_rnn_new_basic() {
        let rnn = RNN::<f32>::new(10, 20).unwrap();
        assert_eq!(rnn.input_size(), 10);
        assert_eq!(rnn.hidden_size(), 20);
        assert_eq!(rnn.num_layers(), 1);
        assert_eq!(rnn.nonlinearity(), RNNNonlinearity::Tanh);
    }

    #[test]
    fn test_rnn_with_options() {
        let rnn = RNN::<f32>::with_options(10, 20, 3, RNNNonlinearity::ReLU).unwrap();
        assert_eq!(rnn.num_layers(), 3);
        assert_eq!(rnn.nonlinearity(), RNNNonlinearity::ReLU);
    }

    #[test]
    fn test_rnn_invalid_sizes() {
        assert!(RNN::<f32>::with_options(0, 20, 1, RNNNonlinearity::Tanh).is_err());
        assert!(RNN::<f32>::with_options(10, 0, 1, RNNNonlinearity::Tanh).is_err());
        assert!(RNN::<f32>::with_options(10, 20, 0, RNNNonlinearity::Tanh).is_err());
    }

    #[test]
    fn test_rnn_parameter_count() {
        let rnn = RNN::<f32>::with_options(10, 20, 2, RNNNonlinearity::Tanh).unwrap();
        let params = rnn.parameters();
        assert_eq!(params.len(), 8); // 4 per layer * 2 layers
    }

    #[test]
    fn test_rnn_parameter_shapes() {
        let rnn = RNN::<f32>::new(10, 20).unwrap();
        let params = rnn.parameters();
        assert_eq!(params[0].shape(), &[20, 10]); // weight_ih [hs, input]
        assert_eq!(params[1].shape(), &[20, 20]); // weight_hh [hs, hs]
        assert_eq!(params[2].shape(), &[20]); // bias_ih
        assert_eq!(params[3].shape(), &[20]); // bias_hh
    }

    #[test]
    fn test_rnn_multi_layer_weight_shapes() {
        let rnn = RNN::<f32>::with_options(10, 20, 3, RNNNonlinearity::Tanh).unwrap();
        let params = rnn.parameters();

        // Layer 0: weight_ih [20, 10] (input_size=10)
        assert_eq!(params[0].shape(), &[20, 10]);
        // Layer 0: weight_hh [20, 20]
        assert_eq!(params[1].shape(), &[20, 20]);

        // Layer 1: weight_ih [20, 20] (input_size=hidden_size=20)
        assert_eq!(params[4].shape(), &[20, 20]);
        // Layer 1: weight_hh [20, 20]
        assert_eq!(params[5].shape(), &[20, 20]);

        // Layer 2: weight_ih [20, 20]
        assert_eq!(params[8].shape(), &[20, 20]);
    }

    #[test]
    fn test_rnn_forward_output_shape() {
        let rnn = RNN::<f32>::new(10, 20).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 10]).unwrap();

        let (output, h_n) = rnn.forward_with_state(&input, None).unwrap();

        assert_eq!(output.shape(), &[2, 5, 20]); // [B, T, hidden]
        assert_eq!(h_n.shape(), &[1, 2, 20]); // [layers, B, hidden]
    }

    #[test]
    fn test_rnn_forward_multi_layer_shapes() {
        let rnn = RNN::<f32>::with_options(8, 16, 3, RNNNonlinearity::Tanh).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[4, 7, 8]).unwrap();

        let (output, h_n) = rnn.forward_with_state(&input, None).unwrap();

        assert_eq!(output.shape(), &[4, 7, 16]);
        assert_eq!(h_n.shape(), &[3, 4, 16]);
    }

    #[test]
    fn test_rnn_module_forward_shape() {
        let rnn = RNN::<f32>::new(10, 20).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 10]).unwrap();
        let output = <RNN<f32> as Module<f32>>::forward(&rnn, &input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_rnn_forward_does_not_error() {
        let rnn = RNN::<f32>::with_options(4, 8, 2, RNNNonlinearity::Tanh).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[3, 10, 4]).unwrap();
        let result = rnn.forward_with_state(&input, None);
        assert!(
            result.is_ok(),
            "forward should not error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_rnn_forward_nonzero_output() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[1, 3, 4]).unwrap();
        let (output, _) = rnn.forward_with_state(&input, None).unwrap();
        let data = output.data().unwrap();
        assert!(data.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_rnn_forward_seq_len_1() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 1, 4]).unwrap();
        let (output, h_n) = rnn.forward_with_state(&input, None).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8]);
        assert_eq!(h_n.shape(), &[1, 1, 8]);
    }

    #[test]
    fn test_rnn_forward_with_initial_state() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let h0 = ferrotorch_core::zeros::<f32>(&[1, 2, 8]).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[2, 3, 4]).unwrap();
        let result = rnn.forward_with_state(&input, Some(&h0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_rnn_forward_state_shape_mismatch() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let h0 = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap(); // wrong batch
        let input = ferrotorch_core::randn::<f32>(&[2, 3, 4]).unwrap();
        assert!(rnn.forward_with_state(&input, Some(&h0)).is_err());
    }

    #[test]
    fn test_rnn_forward_input_wrong_ndim() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[10, 4]).unwrap();
        assert!(rnn.forward_with_state(&input, None).is_err());
    }

    #[test]
    fn test_rnn_forward_input_size_mismatch() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 5, 7]).unwrap();
        assert!(rnn.forward_with_state(&input, None).is_err());
    }

    #[test]
    fn test_rnn_named_parameters() {
        let rnn = RNN::<f32>::with_options(4, 8, 2, RNNNonlinearity::Tanh).unwrap();
        let named = rnn.named_parameters();
        assert_eq!(named.len(), 8);
        assert_eq!(named[0].0, "layers.0.weight_ih");
        assert_eq!(named[1].0, "layers.0.weight_hh");
        assert_eq!(named[2].0, "layers.0.bias_ih");
        assert_eq!(named[3].0, "layers.0.bias_hh");
        assert_eq!(named[4].0, "layers.1.weight_ih");
        assert_eq!(named[5].0, "layers.1.weight_hh");
        assert_eq!(named[6].0, "layers.1.bias_ih");
        assert_eq!(named[7].0, "layers.1.bias_hh");
    }

    #[test]
    fn test_rnn_train_eval() {
        let mut rnn = RNN::<f32>::new(4, 8).unwrap();
        assert!(rnn.is_training());
        rnn.eval();
        assert!(!rnn.is_training());
        rnn.train();
        assert!(rnn.is_training());
    }

    #[test]
    fn test_rnn_all_parameters_require_grad() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        for param in rnn.parameters() {
            assert!(param.requires_grad());
        }
    }

    #[test]
    fn test_rnn_deterministic() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let input = ferrotorch_core::from_slice::<f32>(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            &[1, 2, 4],
        )
        .unwrap();

        let (out1, _) = rnn.forward_with_state(&input, None).unwrap();
        let (out2, _) = rnn.forward_with_state(&input, None).unwrap();

        let d1 = out1.data().unwrap();
        let d2 = out2.data().unwrap();
        for (i, (&a, &b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "output mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rnn_state_dict_roundtrip() {
        let rnn = RNN::<f32>::new(4, 8).unwrap();
        let sd = rnn.state_dict();
        assert_eq!(sd.len(), 4);
        assert!(sd.contains_key("layers.0.weight_ih"));

        let mut rnn2 = RNN::<f32>::new(4, 8).unwrap();
        rnn2.load_state_dict(&sd, true).unwrap();
    }

    #[test]
    fn test_rnn_relu_forward() {
        let rnn = RNN::<f32>::with_options(4, 8, 1, RNNNonlinearity::ReLU).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[2, 3, 4]).unwrap();
        let (output, h_n) = rnn.forward_with_state(&input, None).unwrap();
        assert_eq!(output.shape(), &[2, 3, 8]);
        assert_eq!(h_n.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_rnn_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RNN<f32>>();
        assert_send_sync::<RNN<f64>>();
    }
}
