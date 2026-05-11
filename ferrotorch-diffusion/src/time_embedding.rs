//! Time-step sinusoidal positional encoding + the MLP that follows it.
//!
//! These match `diffusers.models.embeddings.{Timesteps, TimestepEmbedding}`
//! 1:1 for SD-1.5's settings (`flip_sin_to_cos = true`, `freq_shift = 0`).
//!
//! ```text
//! Timesteps(C):           t -> [B, C]
//!   half = C / 2
//!   exponent = -log(max_period) * arange(half) / half
//!   freqs = exp(exponent)
//!   args  = t.float() * freqs
//!   emb   = cat([cos(args), sin(args)], dim=-1)   (flip_sin_to_cos = true)
//!
//! TimestepEmbedding(C, time_emb_dim):
//!   Linear(C, time_emb_dim) -> SiLU -> Linear(time_emb_dim, time_emb_dim)
//! ```
//!
//! Diffusers' `Timesteps` is parameter-free — it's just an arithmetic
//! recipe. We keep it as a `Module<T>` for ergonomic composition but
//! `parameters()` is empty.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Linear, SiLU};

// ---------------------------------------------------------------------------
// Timesteps (sinusoidal positional encoding)
// ---------------------------------------------------------------------------

/// `Timesteps` — sinusoidal positional encoding of a scalar timestep.
///
/// Parameter-free. Reproduces
/// `diffusers.models.embeddings.Timesteps(num_channels, flip_sin_to_cos,
/// downscale_freq_shift)` for `flip_sin_to_cos=true` and
/// `downscale_freq_shift=0` (the SD-1.5 settings).
#[derive(Debug, Clone)]
pub struct Timesteps {
    /// Output channel count (must be even). For SD UNet: 320.
    pub num_channels: usize,
    /// If true, `cat([cos, sin])` (SD-style). If false, `cat([sin, cos])`.
    pub flip_sin_to_cos: bool,
    /// `downscale_freq_shift` from diffusers (subtracted in the exponent
    /// denominator). Always 0 for SD-1.5.
    pub downscale_freq_shift: f64,
    /// Maximum period of the sinusoid (diffusers default: 10000).
    pub max_period: f64,
}

impl Timesteps {
    /// Build a `Timesteps` module.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when `num_channels`
    /// is not a positive even integer.
    pub fn new(
        num_channels: usize,
        flip_sin_to_cos: bool,
        downscale_freq_shift: f64,
    ) -> FerrotorchResult<Self> {
        if num_channels == 0 || num_channels % 2 != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Timesteps::new: num_channels must be a positive even integer, got {num_channels}"
                ),
            });
        }
        Ok(Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
            max_period: 10_000.0,
        })
    }

    /// Compute the sinusoidal encoding for a batch of timesteps.
    ///
    /// `timesteps` has shape `[B]` (1-D, dtype `T`). The output has
    /// shape `[B, num_channels]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the input is not
    /// rank-1, [`FerrotorchError::InvalidArgument`] if the half-channel
    /// math overflows `T`.
    pub fn forward_t<T: Float>(&self, timesteps: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if timesteps.ndim() != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Timesteps::forward_t: expected 1-D timesteps [B], got {:?}",
                    timesteps.shape()
                ),
            });
        }
        let batch = timesteps.shape()[0];
        let half = self.num_channels / 2;
        // exponent = -ln(max_period) * i / (half - downscale_freq_shift),
        // for i in 0..half. half - downscale_freq_shift defaults to half
        // (downscale_freq_shift = 0 for SD-1.5).
        let denom = (half as f64) - self.downscale_freq_shift;
        if denom <= 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Timesteps::forward_t: invalid denominator {denom} (half={half}, \
                     downscale_freq_shift={})",
                    self.downscale_freq_shift,
                ),
            });
        }
        let log_max = self.max_period.ln();
        let mut freqs = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = -log_max * (i as f64) / denom;
            freqs.push(exponent.exp());
        }
        // Read the timestep values into f64 for the multiply, then cast
        // back to T for the sin/cos call.
        let ts_data = timesteps.data()?;
        let zero_t = T::from(0.0).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "Timesteps::forward_t: failed to cast 0.0 into Float".into(),
        })?;
        let mut out = vec![zero_t; batch * self.num_channels];
        for (b, &t) in ts_data.iter().enumerate() {
            let t_f64: f64 = t
                .to_f64()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "Timesteps::forward_t: failed to cast timestep into f64".into(),
                })?;
            for (i, &freq) in freqs.iter().enumerate() {
                let arg = t_f64 * freq;
                let cos_v = arg.cos();
                let sin_v = arg.sin();
                let (left, right) = if self.flip_sin_to_cos {
                    (cos_v, sin_v)
                } else {
                    (sin_v, cos_v)
                };
                out[b * self.num_channels + i] = T::from(left).ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: "Timesteps: cast left value to T failed".into(),
                    }
                })?;
                out[b * self.num_channels + half + i] = T::from(right).ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: "Timesteps: cast right value to T failed".into(),
                    }
                })?;
            }
        }
        Tensor::from_storage(
            TensorStorage::cpu(out),
            vec![batch, self.num_channels],
            false,
        )
    }
}

// `Module` impl: forward expects a 1-D timestep tensor. Parameter-free.
impl<T: Float> Module<T> for Timesteps {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_t(input)
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        Vec::new()
    }
    fn train(&mut self) {
        // Timesteps is parameter-free and stateless — no training-mode
        // flag to flip. Mirrors diffusers' `Timesteps` (which has no
        // submodules / parameters / buffers; PyTorch nn.Module.train()
        // recurses over children that don't exist here).
    }
    fn eval(&mut self) {
        // Same as `train` — nothing to switch.
    }
    fn is_training(&self) -> bool {
        // Always-inference (deterministic arithmetic, no dropout / norm
        // running stats).
        false
    }
    fn load_state_dict(&mut self, _state: &StateDict<T>, _strict: bool) -> FerrotorchResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TimestepEmbedding (MLP)
// ---------------------------------------------------------------------------

/// `TimestepEmbedding` — `Linear -> SiLU -> Linear` applied to the
/// sinusoidal encoding.
///
/// State-dict key layout (matches diffusers):
///
/// ```text
/// linear_1.{weight,bias}    [time_emb_dim, in_channels]
/// linear_2.{weight,bias}    [time_emb_dim, time_emb_dim]
/// ```
#[derive(Debug)]
pub struct TimestepEmbedding<T: Float> {
    /// First linear `in_channels -> time_emb_dim`.
    pub linear_1: Linear<T>,
    /// Output linear `time_emb_dim -> time_emb_dim`.
    pub linear_2: Linear<T>,
    activation: SiLU,
    training: bool,
}

impl<T: Float> TimestepEmbedding<T> {
    /// Build a randomly-initialized `TimestepEmbedding`.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad dims.
    pub fn new(in_channels: usize, time_emb_dim: usize) -> FerrotorchResult<Self> {
        let linear_1 = Linear::<T>::new(in_channels, time_emb_dim, true)?;
        let linear_2 = Linear::<T>::new(time_emb_dim, time_emb_dim, true)?;
        Ok(Self {
            linear_1,
            linear_2,
            activation: SiLU::new(),
            training: false,
        })
    }
}

impl<T: Float> Module<T> for TimestepEmbedding<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.linear_1.forward(input)?;
        let h = self.activation.forward(&h)?;
        self.linear_2.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.linear_1.parameters());
        o.extend(self.linear_2.parameters());
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.linear_1.parameters_mut());
        o.extend(self.linear_2.parameters_mut());
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (n, p) in self.linear_1.named_parameters() {
            o.push((format!("linear_1.{n}"), p));
        }
        for (n, p) in self.linear_2.named_parameters() {
            o.push((format!("linear_2.{n}"), p));
        }
        o
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

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                if !(k.starts_with("linear_1.") || k.starts_with("linear_2.")) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in TimestepEmbedding state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        self.linear_1
            .load_state_dict(&extract("linear_1"), strict)?;
        self.linear_2
            .load_state_dict(&extract("linear_2"), strict)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timesteps_shape_flip_true() {
        let t = Timesteps::new(8, true, 0.0).unwrap();
        let ts = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32, 50.0, 100.0]),
            vec![3],
            false,
        )
        .unwrap();
        let e = t.forward_t(&ts).unwrap();
        assert_eq!(e.shape(), &[3, 8]);
        // For t=0, cos=1 and sin=0 for all freqs => first half ones, second half zeros.
        let d = e.data().unwrap();
        for i in 0..4 {
            assert!((d[i] - 1.0).abs() < 1e-6);
        }
        for i in 4..8 {
            assert!(d[i].abs() < 1e-6);
        }
    }

    #[test]
    fn timesteps_rejects_odd_channels() {
        assert!(Timesteps::new(7, true, 0.0).is_err());
    }

    #[test]
    fn timestep_embedding_shapes() {
        let mlp = TimestepEmbedding::<f32>::new(8, 16).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.5f32; 8]),
            vec![1, 8],
            false,
        )
        .unwrap();
        let y = mlp.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 16]);
    }

    #[test]
    fn timestep_embedding_named_parameters() {
        let mlp = TimestepEmbedding::<f32>::new(8, 16).unwrap();
        let names: Vec<String> = mlp.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "linear_1.weight",
            "linear_1.bias",
            "linear_2.weight",
            "linear_2.bias",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }
}
