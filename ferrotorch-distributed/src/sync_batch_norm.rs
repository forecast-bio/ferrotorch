//! Synchronized Batch Normalization (SyncBatchNorm).
//!
//! Like [`ferrotorch_nn::BatchNorm2d`] but synchronizes per-channel mean and
//! variance across all ranks of a distributed process group, so the
//! normalization sees the *global* batch statistics instead of just the
//! per-rank local mini-batch.
//!
//! # When to use
//!
//! SyncBatchNorm is the right normalization choice when:
//!
//! - The per-rank batch is small enough that local statistics are noisy
//!   (e.g. detection/segmentation training where per-GPU batch size is
//!   1–4 images).
//! - You want bit-identical normalization to the no-DDP single-GPU case.
//!
//! For large per-rank batches (≥ 32 images on each rank) the synchronization
//! overhead usually outweighs the statistical benefit and a plain
//! `BatchNorm2d` is preferable.
//!
//! # Synchronization
//!
//! Forward pass:
//!   1. Each rank computes its local per-channel `sum` and `sum_sq`.
//!   2. Both vectors are concatenated and `allreduce`d (sum) across ranks.
//!   3. The global mean and variance are computed by dividing by the
//!      *global* element count `N_global = local_count * world_size`.
//!   4. Normalization uses the global statistics.
//!
//! Backward pass:
//!   1. Each rank computes its local per-channel `sum_dl_dx_hat` and
//!      `sum_dl_dx_hat_x_hat`.
//!   2. Both vectors are `allreduce`d (sum) across ranks.
//!   3. `grad_input` uses the global means in the standard BatchNorm
//!      VJP formula, ensuring the gradient is consistent with the
//!      synchronized forward.
//!   4. `grad_weight` and `grad_bias` are accumulated locally per rank;
//!      DDP's gradient hook will sum them across ranks at the next
//!      synchronization point. (We do NOT pre-sum gamma/beta gradients
//!      here, matching PyTorch SyncBatchNorm semantics.)
//!
//! Mirrors `torch.nn.SyncBatchNorm`. CL-392.

use std::sync::{Arc, Mutex};

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};
use ferrotorch_core::{Float, is_grad_enabled};
use ferrotorch_nn::{Module, Parameter};

use crate::backend::Backend;
use crate::collective::{ReduceOp, allreduce};

/// 2-D synchronized batch normalization.
///
/// Same API surface as `BatchNorm2d` plus an `Arc<dyn Backend>` for
/// cross-rank communication. When `world_size == 1` (or no backend is
/// provided), behaves exactly like a plain BatchNorm2d.
pub struct SyncBatchNorm2d<T: Float> {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub affine: bool,
    pub weight: Option<Parameter<T>>,
    pub bias: Option<Parameter<T>>,
    running_mean: Mutex<Vec<f64>>,
    running_var: Mutex<Vec<f64>>,
    num_batches_tracked: Mutex<usize>,
    training: Mutex<bool>,
    /// Optional process group backend used to synchronize statistics.
    /// When `None`, behaves like a non-distributed BatchNorm2d.
    backend: Option<Arc<dyn Backend>>,
}

impl<T: Float> std::fmt::Debug for SyncBatchNorm2d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncBatchNorm2d")
            .field("num_features", &self.num_features)
            .field("eps", &self.eps)
            .field("momentum", &self.momentum)
            .field("affine", &self.affine)
            .field(
                "world_size",
                &self.backend.as_ref().map(|b| b.world_size()).unwrap_or(1),
            )
            .field("training", &self.training)
            .finish()
    }
}

impl<T: Float> SyncBatchNorm2d<T> {
    /// Create a new SyncBatchNorm2d with no backend (acts as a plain
    /// BatchNorm2d). Use [`with_backend`](Self::with_backend) to attach
    /// the process group.
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
    ) -> FerrotorchResult<Self> {
        if num_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "SyncBatchNorm2d: num_features must be positive".into(),
            });
        }
        let weight = if affine {
            Some(Parameter::ones(&[num_features])?)
        } else {
            None
        };
        let bias = if affine {
            Some(Parameter::zeros(&[num_features])?)
        } else {
            None
        };
        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            weight,
            bias,
            running_mean: Mutex::new(vec![0.0; num_features]),
            running_var: Mutex::new(vec![1.0; num_features]),
            num_batches_tracked: Mutex::new(0),
            training: Mutex::new(true),
            backend: None,
        })
    }

    /// Attach a backend so the layer synchronizes its statistics across
    /// the distributed process group.
    pub fn with_backend(mut self, backend: Arc<dyn Backend>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Snapshot of the current per-channel running mean.
    pub fn running_mean(&self) -> Vec<f64> {
        self.running_mean.lock().unwrap().clone()
    }

    /// Snapshot of the current per-channel running variance.
    pub fn running_var(&self) -> Vec<f64> {
        self.running_var.lock().unwrap().clone()
    }

    /// Number of training batches processed so far.
    pub fn num_batches_tracked(&self) -> usize {
        *self.num_batches_tracked.lock().unwrap()
    }
}

impl<T: Float> Module<T> for SyncBatchNorm2d<T> {
    // De-interleaving sum/sum_sq from a packed reduce buffer isn't expressible
    // as a single slice memcpy.
    #[allow(clippy::manual_memcpy)]
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 4 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "SyncBatchNorm2d: expected 4D input [B, C, H, W], got {:?}",
                    shape
                ),
            });
        }
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial = height * width;

        if channels != self.num_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "SyncBatchNorm2d: expected {} channels, got {}",
                    self.num_features, channels
                ),
            });
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "SyncBatchNorm2d::forward",
            });
        }

        let input_data = input.data()?;
        let eps_t = T::from(self.eps).unwrap();
        let weight_data = self.weight.as_ref().map(|w| w.tensor().data().unwrap());
        let bias_data = self.bias.as_ref().map(|b| b.tensor().data().unwrap());
        let is_training = *self.training.lock().unwrap();

        let mut chan_mean = vec![<T as num_traits::Zero>::zero(); channels];
        let mut chan_var = vec![<T as num_traits::Zero>::zero(); channels];

        if is_training {
            // Local per-channel sum and sum of squares.
            let local_count = batch * spatial;
            let mut sum = vec![<T as num_traits::Zero>::zero(); channels];
            let mut sum_sq = vec![<T as num_traits::Zero>::zero(); channels];

            for c in 0..channels {
                for b in 0..batch {
                    let base = b * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        let v = input_data[base + s];
                        sum[c] += v;
                        sum_sq[c] += v * v;
                    }
                }
            }

            // Synchronize across ranks if a backend is attached.
            let global_count = if let Some(ref backend) = self.backend {
                let world_size = backend.world_size();
                if world_size > 1 {
                    // Pack sum and sum_sq into a single tensor for one
                    // allreduce instead of two.
                    let mut packed: Vec<T> = Vec::with_capacity(2 * channels);
                    packed.extend_from_slice(&sum);
                    packed.extend_from_slice(&sum_sq);
                    let packed_t = Tensor::from_storage(
                        TensorStorage::cpu(packed),
                        vec![2 * channels],
                        false,
                    )?;
                    let reduced = allreduce(&packed_t, backend.as_ref(), ReduceOp::Sum)?;
                    let reduced_data = reduced.data()?;
                    for c in 0..channels {
                        sum[c] = reduced_data[c];
                        sum_sq[c] = reduced_data[channels + c];
                    }
                    local_count * world_size
                } else {
                    local_count
                }
            } else {
                local_count
            };

            let global_count_t = T::from(global_count).unwrap();
            for c in 0..channels {
                let m = sum[c] / global_count_t;
                chan_mean[c] = m;
                // E[X^2] - E[X]^2 (biased variance, matches PyTorch)
                chan_var[c] = sum_sq[c] / global_count_t - m * m;
            }

            // Update running statistics with the synchronized batch stats.
            {
                let mut rm = self.running_mean.lock().unwrap();
                let mut rv = self.running_var.lock().unwrap();
                let mut nbt = self.num_batches_tracked.lock().unwrap();
                *nbt += 1;
                let mom = self.momentum;
                let bessel = if global_count > 1 {
                    global_count as f64 / (global_count as f64 - 1.0)
                } else {
                    1.0
                };
                for c in 0..channels {
                    let bm = chan_mean[c].to_f64().unwrap();
                    let bv = chan_var[c].to_f64().unwrap();
                    rm[c] = (1.0 - mom) * rm[c] + mom * bm;
                    rv[c] = (1.0 - mom) * rv[c] + mom * bv * bessel;
                }
            }
        } else {
            // Eval mode: use the running statistics regardless of backend.
            let rm = self.running_mean.lock().unwrap();
            let rv = self.running_var.lock().unwrap();
            for c in 0..channels {
                chan_mean[c] = T::from(rm[c]).unwrap();
                chan_var[c] = T::from(rv[c]).unwrap();
            }
        }

        // Normalize and optionally scale/shift.
        let mut output = vec![<T as num_traits::Zero>::zero(); input.numel()];
        let mut x_hat_data = if is_grad_enabled() && input.requires_grad() {
            Vec::with_capacity(input.numel())
        } else {
            Vec::new()
        };
        let need_x_hat = is_grad_enabled() && input.requires_grad();

        let mut inv_std = vec![<T as num_traits::Zero>::zero(); channels];
        for c in 0..channels {
            inv_std[c] = (chan_var[c] + eps_t).sqrt().recip();
        }

        for b in 0..batch {
            for c in 0..channels {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let normed = (input_data[idx] - chan_mean[c]) * inv_std[c];
                    if need_x_hat {
                        x_hat_data.push(normed);
                    }
                    if self.affine {
                        let w = weight_data.as_ref().unwrap();
                        let bi = bias_data.as_ref().unwrap();
                        output[idx] = normed * w[c] + bi[c];
                    } else {
                        output[idx] = normed;
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(output), shape.clone(), false)?;

        if need_x_hat {
            let weight_tensor = self.weight.as_ref().map(|w| w.tensor().clone());
            let bias_tensor = self.bias.as_ref().map(|b| b.tensor().clone());
            let local_count = batch * spatial;
            let global_count = self
                .backend
                .as_ref()
                .map(|b| local_count * b.world_size())
                .unwrap_or(local_count);
            let grad_fn = Arc::new(SyncBatchNorm2dBackward {
                input: input.clone(),
                x_hat: Tensor::from_storage(TensorStorage::cpu(x_hat_data), shape.clone(), false)?,
                weight: weight_tensor,
                bias: bias_tensor,
                chan_var: chan_var.iter().map(|v| v.to_f64().unwrap()).collect(),
                eps: self.eps,
                affine: self.affine,
                global_count,
                backend: self.backend.clone(),
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        match (&mut self.weight, &mut self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![("weight".to_string(), w), ("bias".to_string(), b)],
            _ => vec![],
        }
    }

    fn train(&mut self) {
        *self.training.lock().unwrap() = true;
    }

    fn eval(&mut self) {
        *self.training.lock().unwrap() = false;
    }

    fn is_training(&self) -> bool {
        *self.training.lock().unwrap()
    }
}

/// Backward node for [`SyncBatchNorm2d`]. Synchronizes the two intermediate
/// sums (`sum_dl_dx_hat` and `sum_dl_dx_hat_x_hat`) across ranks via
/// allreduce so that `grad_input` is consistent with the synchronized
/// forward. `grad_weight` and `grad_bias` are kept local — DDP will reduce
/// them at the parameter sync step.
struct SyncBatchNorm2dBackward<T: Float> {
    input: Tensor<T>,
    x_hat: Tensor<T>,
    weight: Option<Tensor<T>>,
    bias: Option<Tensor<T>>,
    chan_var: Vec<f64>,
    eps: f64,
    affine: bool,
    global_count: usize,
    backend: Option<Arc<dyn Backend>>,
}

impl<T: Float> std::fmt::Debug for SyncBatchNorm2dBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncBatchNorm2dBackward")
            .field("global_count", &self.global_count)
            .finish()
    }
}

impl<T: Float> GradFn<T> for SyncBatchNorm2dBackward<T> {
    #[allow(clippy::manual_memcpy)]
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial = height * width;

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "SyncBatchNorm2dBackward",
            });
        }

        let go_data = grad_output.data()?;
        let x_hat_data = self.x_hat.data()?;
        let weight_data = self.weight.as_ref().map(|w| w.data().unwrap().to_vec());

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); self.input.numel()];
        let mut grad_weight = vec![<T as num_traits::Zero>::zero(); channels];
        let mut grad_bias = vec![<T as num_traits::Zero>::zero(); channels];

        // First pass: per-channel local sums.
        let mut local_dl_dx_hat_sum = vec![<T as num_traits::Zero>::zero(); channels];
        let mut local_dl_dx_hat_x_hat_sum = vec![<T as num_traits::Zero>::zero(); channels];

        for c in 0..channels {
            for b in 0..batch {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];
                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };
                    local_dl_dx_hat_sum[c] += dl_dx_hat;
                    local_dl_dx_hat_x_hat_sum[c] += dl_dx_hat * x_h;
                    if self.affine {
                        grad_weight[c] += go * x_h;
                        grad_bias[c] += go;
                    }
                }
            }
        }

        // Synchronize the two sum vectors across ranks. Pack into a single
        // tensor of shape [2*C] for one allreduce.
        let mut global_dl_dx_hat_sum = local_dl_dx_hat_sum.clone();
        let mut global_dl_dx_hat_x_hat_sum = local_dl_dx_hat_x_hat_sum.clone();

        if let Some(ref backend) = self.backend {
            if backend.world_size() > 1 {
                let mut packed: Vec<T> = Vec::with_capacity(2 * channels);
                packed.extend_from_slice(&local_dl_dx_hat_sum);
                packed.extend_from_slice(&local_dl_dx_hat_x_hat_sum);
                let packed_t =
                    Tensor::from_storage(TensorStorage::cpu(packed), vec![2 * channels], false)?;
                let reduced = allreduce(&packed_t, backend.as_ref(), ReduceOp::Sum)?;
                let reduced_data = reduced.data()?;
                for c in 0..channels {
                    global_dl_dx_hat_sum[c] = reduced_data[c];
                    global_dl_dx_hat_x_hat_sum[c] = reduced_data[channels + c];
                }
            }
        }

        let global_count_t = T::from(self.global_count).unwrap();

        // Second pass: compute grad_input using the synchronized means.
        for c in 0..channels {
            let var_f64 = self.chan_var[c];
            let inv_std = T::from(1.0 / (var_f64 + self.eps).sqrt()).unwrap();

            let dl_dx_hat_mean = global_dl_dx_hat_sum[c] / global_count_t;
            let dl_dx_hat_x_hat_mean = global_dl_dx_hat_x_hat_sum[c] / global_count_t;

            for b in 0..batch {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];
                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };
                    grad_input[idx] =
                        inv_std * (dl_dx_hat - dl_dx_hat_mean - x_h * dl_dx_hat_x_hat_mean);
                }
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        let grad_weight_out = if self.affine {
            self.weight.as_ref().and_then(|w| {
                if w.requires_grad() {
                    Some(
                        Tensor::from_storage(
                            TensorStorage::cpu(grad_weight),
                            vec![channels],
                            false,
                        )
                        .unwrap(),
                    )
                } else {
                    None
                }
            })
        } else {
            None
        };
        let grad_bias_out = if self.affine {
            self.bias.as_ref().and_then(|b| {
                if b.requires_grad() {
                    Some(
                        Tensor::from_storage(TensorStorage::cpu(grad_bias), vec![channels], false)
                            .unwrap(),
                    )
                } else {
                    None
                }
            })
        } else {
            None
        };

        Ok(vec![
            Some(grad_input_tensor),
            grad_weight_out,
            grad_bias_out,
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v: Vec<&Tensor<T>> = vec![&self.input];
        if let Some(ref w) = self.weight {
            v.push(w);
        }
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "SyncBatchNorm2dBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_core::Tensor;
    use ferrotorch_nn::BatchNorm2d;
    use std::thread;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_sync_bn_world_size_1_matches_batch_norm() {
        // With no backend (or world_size=1), SyncBatchNorm2d should produce
        // the exact same output as a plain BatchNorm2d on the same input.
        let input_data: Vec<f32> = (0..24).map(|i| i as f32 / 10.0).collect();
        let input = cpu_tensor(&input_data, &[2, 3, 2, 2]);

        let mut sync = SyncBatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let mut plain = BatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        sync.train();
        plain.train();

        let out_sync = sync.forward(&input).unwrap();
        let out_plain = plain.forward(&input).unwrap();

        let s = out_sync.data().unwrap();
        let p = out_plain.data().unwrap();
        for (i, (a, b)) in s.iter().zip(p.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "out[{i}]: sync={a}, plain={b}");
        }
    }

    #[test]
    fn test_sync_bn_two_ranks_match_full_batch() {
        // Set up a 4-element batch of [4, 3, 2, 2] and split it across
        // two simulated ranks. SyncBatchNorm2d should compute the same
        // mean/var as a single-rank BatchNorm2d on the full batch.
        let full_data: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) / 10.0).collect();
        let full = cpu_tensor(&full_data, &[4, 3, 2, 2]);

        let mut plain = BatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        plain.train();
        let plain_out = plain.forward(&full).unwrap();
        let plain_data = plain_out.data().unwrap().to_vec();
        let plain_running_mean = plain.running_mean();
        let plain_running_var = plain.running_var();

        // Per-rank slices: rank 0 sees first 2 batch elements (indices
        // 0..24), rank 1 sees the last 2 (indices 24..48).
        let r0 = full_data[0..24].to_vec();
        let r1 = full_data[24..48].to_vec();
        let r0_t = cpu_tensor(&r0, &[2, 3, 2, 2]);
        let r1_t = cpu_tensor(&r1, &[2, 3, 2, 2]);

        // Build a 2-rank simulated backend and run forward on each rank
        // in its own thread (allreduce blocks pairs of ranks).
        let group = SimulatedBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let b0 = Arc::new(iter.next().unwrap());
        let b1 = Arc::new(iter.next().unwrap());

        let r0_clone = r0_t.clone();
        let r1_clone = r1_t.clone();
        let b0_clone: Arc<dyn Backend> = b0.clone();
        let b1_clone: Arc<dyn Backend> = b1.clone();

        let h0 = thread::spawn(move || {
            let mut sync = SyncBatchNorm2d::<f32>::new(3, 1e-5, 0.1, true)
                .unwrap()
                .with_backend(b0_clone);
            sync.train();
            let out = sync.forward(&r0_clone).unwrap();
            (
                out.data().unwrap().to_vec(),
                sync.running_mean(),
                sync.running_var(),
            )
        });
        let h1 = thread::spawn(move || {
            let mut sync = SyncBatchNorm2d::<f32>::new(3, 1e-5, 0.1, true)
                .unwrap()
                .with_backend(b1_clone);
            sync.train();
            let out = sync.forward(&r1_clone).unwrap();
            (
                out.data().unwrap().to_vec(),
                sync.running_mean(),
                sync.running_var(),
            )
        });

        let (out0, rm0, rv0) = h0.join().unwrap();
        let (out1, rm1, rv1) = h1.join().unwrap();

        // Concatenate the per-rank outputs in batch order — they should
        // match the single-rank full-batch output element-for-element.
        let mut concat = out0.clone();
        concat.extend_from_slice(&out1);
        for (i, (a, b)) in concat.iter().zip(plain_data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "out[{i}]: sync={a}, plain={b}");
        }

        // Both ranks should have identical running statistics, and they
        // should match the single-rank full-batch running statistics.
        for c in 0..3 {
            assert!(
                (rm0[c] - rm1[c]).abs() < 1e-6,
                "rank0 and rank1 running_mean disagree at c={c}"
            );
            assert!(
                (rm0[c] - plain_running_mean[c]).abs() < 1e-4,
                "running_mean[{c}] sync={} plain={}",
                rm0[c],
                plain_running_mean[c]
            );
            assert!(
                (rv0[c] - rv1[c]).abs() < 1e-6,
                "rank0 and rank1 running_var disagree at c={c}"
            );
            assert!(
                (rv0[c] - plain_running_var[c]).abs() < 1e-4,
                "running_var[{c}] sync={} plain={}",
                rv0[c],
                plain_running_var[c]
            );
        }
    }

    #[test]
    fn test_sync_bn_eval_mode_uses_running_stats() {
        // After warming up running stats in train mode, eval mode should
        // produce deterministic output independent of the input batch
        // distribution.
        let input = cpu_tensor(
            &(0..12).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 3, 2, 2],
        );
        let mut sync = SyncBatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        sync.train();
        // Warm up.
        for _ in 0..3 {
            let _ = sync.forward(&input).unwrap();
        }
        sync.eval();
        // Now run with a different input — output should still be normalized
        // using the running stats from training.
        let other = cpu_tensor(&[100.0_f32; 12], &[1, 3, 2, 2]);
        let out = sync.forward(&other).unwrap();
        // Just verify forward completes and produces finite output.
        for v in out.data().unwrap() {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    #[test]
    fn test_sync_bn_constructor_validates_num_features() {
        assert!(SyncBatchNorm2d::<f32>::new(0, 1e-5, 0.1, true).is_err());
    }

    #[test]
    fn test_sync_bn_rejects_wrong_input_shape() {
        let sync = SyncBatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let bad = cpu_tensor(&[1.0, 2.0, 3.0], &[3]);
        assert!(sync.forward(&bad).is_err());
    }

    #[test]
    fn test_sync_bn_rejects_wrong_channel_count() {
        let sync = SyncBatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let bad = cpu_tensor(&[0.0; 16], &[1, 4, 2, 2]);
        assert!(sync.forward(&bad).is_err());
    }
}
