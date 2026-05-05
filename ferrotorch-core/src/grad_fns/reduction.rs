//! Backward functions for reduction operations: sum, mean, prod.
//!
//! Each reduction collapses an input tensor to a scalar. The VJP
//! (vector-Jacobian product) broadcasts the upstream scalar gradient
//! back to the input shape.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::ops::elementwise;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// SumBackward
// ---------------------------------------------------------------------------

/// Backward node for `sum(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output` for all i (broadcast scalar to input shape).
#[derive(Debug)]
pub struct SumBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Extract the scalar value — works for both CPU and GPU by
        // transferring to CPU if needed (it's just one number).
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let numel = self.input.numel();

        // GPU-native path: skip the `vec![go; numel]` CPU allocation +
        // upload by calling the on-device `fill` primitive. Falls back
        // to the CPU build + `.to(device)` for non-f32/f64 types or if
        // the backend hasn't been initialised.
        if self.input.is_cuda() {
            use crate::device::Device;
            use crate::gpu_dispatch::gpu_backend;
            use std::any::TypeId;
            let ordinal = match self.input.device() {
                Device::Cuda(o) => o,
                _ => 0,
            };
            let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
            let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
            if let Some(backend) = gpu_backend() {
                if is_t_f32 {
                    let scalar_f32: f32 = <T as num_traits::ToPrimitive>::to_f32(&go).unwrap();
                    let handle = backend.fill_f32(numel, scalar_f32, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                } else if is_t_f64 {
                    let scalar_f64: f64 = <T as num_traits::ToPrimitive>::to_f64(&go).unwrap();
                    let handle = backend.fill_f64(numel, scalar_f64, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                }
            }
        }
        // CPU / fallback path — legacy behaviour.
        let data = vec![go; numel];
        let grad_cpu =
            Tensor::from_storage(TensorStorage::cpu(data), self.input.shape().to_vec(), false)?;
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Differentiable sum reduction: returns a scalar that is the sum of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`SumBackward`] node.
pub fn sum<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("sum", "reduction", &[input.shape()], || {
        sum_inner(input)
    })
}

fn sum_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if is_f32 {
            backend.sum_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.sum_f64(input.gpu_handle()?, input.numel())?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = vec![];

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(SumBackward {
                input: input.clone(),
            });
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else if input.is_cuda() {
        Err(FerrotorchError::NotImplementedOnCuda { op: "sum" })
    } else {
        let result = elementwise::sum(input)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(SumBackward {
                input: input.clone(),
            });
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// MeanBackward
// ---------------------------------------------------------------------------

/// Backward node for `mean(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output / numel` for all i.
#[derive(Debug)]
pub struct MeanBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for MeanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let numel = self.input.numel();
        let n = T::from(numel).unwrap();
        let val = go / n;

        // GPU-native path mirrors SumBackward: use on-device fill
        // instead of allocating `vec![val; numel]` on CPU and uploading.
        if self.input.is_cuda() {
            use crate::device::Device;
            use crate::gpu_dispatch::gpu_backend;
            use std::any::TypeId;
            let ordinal = match self.input.device() {
                Device::Cuda(o) => o,
                _ => 0,
            };
            let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
            let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
            if let Some(backend) = gpu_backend() {
                if is_t_f32 {
                    let scalar_f32: f32 = <T as num_traits::ToPrimitive>::to_f32(&val).unwrap();
                    let handle = backend.fill_f32(numel, scalar_f32, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                } else if is_t_f64 {
                    let scalar_f64: f64 = <T as num_traits::ToPrimitive>::to_f64(&val).unwrap();
                    let handle = backend.fill_f64(numel, scalar_f64, ordinal)?;
                    let grad_input = Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        self.input.shape().to_vec(),
                        false,
                    )?;
                    return Ok(vec![Some(grad_input)]);
                }
            }
        }
        let data = vec![val; numel];
        let grad_cpu =
            Tensor::from_storage(TensorStorage::cpu(data), self.input.shape().to_vec(), false)?;
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

/// Differentiable mean reduction: returns a scalar that is the mean of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`MeanBackward`] node.
pub fn mean<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("mean", "reduction", &[input.shape()], || {
        mean_inner(input)
    })
}

fn mean_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // GPU path: use GPU sum kernel + scalar divide (avoids CPU round-trip).
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
    let result = if input.is_cuda() && (is_f32 || is_f64) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let device = input.device();
            let ordinal = match device {
                crate::device::Device::Cuda(o) => o,
                _ => 0,
            };
            let mean_handle = if is_f32 {
                let sum_handle = backend.sum_f32(input.gpu_handle()?, input.numel())?;
                let n = input.numel() as f32;
                let inv_n_data = [1.0f32 / n];
                // SAFETY: `inv_n_data` is a stack array of one f32 (length 1,
                // initialized just above), borrowed for this scope. Reading
                // its 4 bytes (1 * size_of::<f32>()) as &[u8] is sound: f32
                // has no padding, no niches, and the requested length matches
                // the actual byte size of the array.
                let inv_n_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        inv_n_data.as_ptr().cast::<u8>(),
                        inv_n_data.len() * 4,
                    )
                };
                let inv_handle = backend.cpu_to_gpu(inv_n_bytes, 4, ordinal)?;
                backend.mul_f32(&sum_handle, &inv_handle)?
            } else {
                let sum_handle = backend.sum_f64(input.gpu_handle()?, input.numel())?;
                let n = input.numel() as f64;
                let inv_n_data = [1.0f64 / n];
                // SAFETY: `inv_n_data` is a stack array of one f64 (length 1,
                // initialized just above), borrowed for this scope. Reading
                // its 8 bytes (1 * size_of::<f64>()) as &[u8] is sound: f64
                // has no padding, no niches, and the requested length matches
                // the actual byte size of the array.
                let inv_n_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        inv_n_data.as_ptr().cast::<u8>(),
                        inv_n_data.len() * 8,
                    )
                };
                let inv_handle = backend.cpu_to_gpu(inv_n_bytes, 8, ordinal)?;
                backend.mul_f64(&sum_handle, &inv_handle)?
            };
            Tensor::from_storage(TensorStorage::gpu(mean_handle), vec![], false)?
        } else {
            elementwise::mean(input)?
        }
    } else {
        elementwise::mean(input)?
    };

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MeanBackward {
            input: input.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// ProdBackward
// ---------------------------------------------------------------------------

/// Backward node for `prod(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output * prod(input) / input[i]`.
///
/// When any `input[i]` is zero, we recompute the partial product excluding
/// that element to avoid division by zero. This is done via prefix/suffix
/// products.
#[derive(Debug)]
pub struct ProdBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for ProdBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "prod backward",
            });
        }
        let input_data = self.input.data()?;
        let n = input_data.len();

        // Use prefix/suffix products to avoid division by zero.
        // prefix[i] = product of input[0..i]
        // suffix[i] = product of input[i+1..n]
        // grad[i] = go * prefix[i] * suffix[i]
        let mut prefix = vec![<T as num_traits::One>::one(); n];
        for i in 1..n {
            prefix[i] = prefix[i - 1] * input_data[i - 1];
        }

        let mut suffix = vec![<T as num_traits::One>::one(); n];
        if n > 1 {
            for i in (0..n - 1).rev() {
                suffix[i] = suffix[i + 1] * input_data[i + 1];
            }
        }

        let grad_data: Vec<T> = (0..n).map(|i| go * prefix[i] * suffix[i]).collect();

        let grad_cpu = Tensor::from_storage(
            TensorStorage::cpu(grad_data),
            self.input.shape().to_vec(),
            false,
        )?;
        // Place gradient on the same device as the input.
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ProdBackward"
    }
}

/// Differentiable product reduction: returns a scalar that is the product
/// of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`ProdBackward`] node.
pub fn prod<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_all(input)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("prod", "reduction", &[input.shape()], || {
        prod_inner(input)
    })
}

fn prod_inner<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let t_is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let t_is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    // GPU path: native reduce_prod kernel (#524).
    if input.is_cuda() && (t_is_f32 || t_is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if t_is_f32 {
            backend.prod_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.prod_f64(input.gpu_handle()?, input.numel())?
        };
        let storage = TensorStorage::gpu(handle);
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(ProdBackward {
                input: input.clone(),
            });
            return Tensor::from_operation(storage, vec![], grad_fn);
        }
        return Tensor::from_storage(storage, vec![], false);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "prod" });
    }
    let data = input.data()?;
    let total = data
        .iter()
        .copied()
        .fold(<T as num_traits::One>::one(), |a, b| a * b);
    let result = Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(ProdBackward {
            input: input.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// AminBackward / AmaxBackward (#627)
// ---------------------------------------------------------------------------
//
// `amin` / `amax` reduce a tensor to the global min/max scalar. These are
// the closed-form `torch.amin` / `torch.amax` ops. The backward routes
// the gradient to every input position equal to the extremum (subgradient
// at ties), matching torch's behavior.

#[derive(Debug)]
pub struct AminBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AminBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = if grad_output.is_cuda() {
            grad_output.cpu()?.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let input_data = self.input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mn = input_data
            .iter()
            .copied()
            .fold(
                T::from(f64::INFINITY).unwrap(),
                |a, b| if b < a { b } else { a },
            );
        // Distribute `go` evenly across all positions equal to the min
        // (subgradient at ties — matches torch.amin's gradient).
        let count = input_data.iter().filter(|&&v| v == mn).count() as f64;
        let scale = T::from(go.to_f64().unwrap() / count.max(1.0)).unwrap();
        let result: Vec<T> = input_data
            .iter()
            .map(|&v| if v == mn { scale } else { zero })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AminBackward"
    }
}

#[derive(Debug)]
pub struct AmaxBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for AmaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = if grad_output.is_cuda() {
            grad_output.cpu()?.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let input_data = self.input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mx = input_data
            .iter()
            .copied()
            .fold(T::from(f64::NEG_INFINITY).unwrap(), |a, b| {
                if b > a { b } else { a }
            });
        let count = input_data.iter().filter(|&&v| v == mx).count() as f64;
        let scale = T::from(go.to_f64().unwrap() / count.max(1.0)).unwrap();
        let result: Vec<T> = input_data
            .iter()
            .map(|&v| if v == mx { scale } else { zero })
            .collect();
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AmaxBackward"
    }
}

/// Differentiable global minimum reduction. Mirrors `torch.amin(input)`
/// with no `dim` argument: returns a 0-d tensor holding the smallest
/// element. On CUDA f32/f64, dispatches to the native PTX
/// `gpu_reduce_min` kernel; on CPU and other dtypes, walks the buffer.
/// Backward routes the upstream grad to every input element equal to
/// the min (subgradient at ties). (#627)
pub fn amin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if is_f32 {
            backend.min_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.min_f64(input.gpu_handle()?, input.numel())?
        };
        let storage = TensorStorage::gpu(handle);
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(AminBackward {
                input: input.clone(),
            });
            return Tensor::from_operation(storage, vec![], grad_fn);
        }
        return Tensor::from_storage(storage, vec![], false);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "amin" });
    }
    // CPU walk.
    let data = input.data_vec()?;
    let mn = data.iter().copied().fold(
        T::from(f64::INFINITY).unwrap(),
        |a, b| if b < a { b } else { a },
    );
    let storage = TensorStorage::cpu(vec![mn]);
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(AminBackward {
            input: input.clone(),
        });
        Tensor::from_operation(storage, vec![], grad_fn)
    } else {
        Tensor::from_storage(storage, vec![], false)
    }
}

/// Differentiable global maximum reduction. Counterpart of [`amin`].
/// Mirrors `torch.amax(input)`. (#627)
pub fn amax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

    if input.is_cuda() && (is_f32 || is_f64) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if is_f32 {
            backend.max_f32(input.gpu_handle()?, input.numel())?
        } else {
            backend.max_f64(input.gpu_handle()?, input.numel())?
        };
        let storage = TensorStorage::gpu(handle);
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(AmaxBackward {
                input: input.clone(),
            });
            return Tensor::from_operation(storage, vec![], grad_fn);
        }
        return Tensor::from_storage(storage, vec![], false);
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "amax" });
    }
    let data = input.data_vec()?;
    let mx = data
        .iter()
        .copied()
        .fold(T::from(f64::NEG_INFINITY).unwrap(), |a, b| {
            if b > a { b } else { a }
        });
    let storage = TensorStorage::cpu(vec![mx]);
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(AmaxBackward {
            input: input.clone(),
        });
        Tensor::from_operation(storage, vec![], grad_fn)
    } else {
        Tensor::from_storage(storage, vec![], false)
    }
}

// ---------------------------------------------------------------------------
// SumDimBackward
// ---------------------------------------------------------------------------

/// Backward node for `sum_dim(input, dim, keepdim) -> reduced tensor`.
///
/// VJP: expand the gradient back along the reduced dimension to match the
/// input shape. If `keepdim` was false, we first unsqueeze the reduced dim
/// before expanding.
#[derive(Debug)]
pub struct SumDimBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
    keepdim: bool,
}

impl<T: Float> GradFn<T> for SumDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        let outer: usize = input_shape[..self.dim].iter().product::<usize>().max(1);
        let inner: usize = input_shape[(self.dim + 1)..]
            .iter()
            .product::<usize>()
            .max(1);
        let repeat_count = input_shape[self.dim];

        // GPU-native path: expand-along-dim via the dedicated kernel (#524).
        let t_is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let t_is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if grad_output.is_cuda() && (t_is_f32 || t_is_f64) {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let result_h = if t_is_f32 {
                backend.repeat_along_dim_f32(
                    grad_output.gpu_handle()?,
                    outer,
                    repeat_count,
                    inner,
                )?
            } else {
                backend.repeat_along_dim_f64(
                    grad_output.gpu_handle()?,
                    outer,
                    repeat_count,
                    inner,
                )?
            };
            let grad_input =
                Tensor::from_storage(TensorStorage::gpu(result_h), input_shape.to_vec(), false)?;
            return Ok(vec![Some(grad_input)]);
        }
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "sum_dim backward",
            });
        }

        // If keepdim was false, reinsert the reduced dimension as size 1.
        let grad = if self.keepdim {
            grad_output.clone()
        } else {
            let mut unsqueezed_shape = grad_output.shape().to_vec();
            unsqueezed_shape.insert(self.dim, 1);
            let data = grad_output.data()?.to_vec();
            Tensor::from_storage(TensorStorage::cpu(data), unsqueezed_shape, false)?
        };

        // Now expand (repeat) along the reduced dim to match input shape.
        let grad_data = grad.data()?;
        let grad_shape = grad.shape();

        let out_numel: usize = input_shape.iter().product();
        let mut result = Vec::with_capacity(out_numel);

        for flat in 0..out_numel {
            // Decompose flat index into input coords.
            let mut rem = flat;
            let mut coords = vec![0usize; input_shape.len()];
            for d in (0..input_shape.len()).rev() {
                coords[d] = rem % input_shape[d];
                rem /= input_shape[d];
            }
            // Map to grad index: the reduced dim coordinate becomes 0 (size 1 in grad).
            let mut grad_flat = 0usize;
            let mut stride = 1usize;
            for d in (0..grad_shape.len()).rev() {
                let c = if d == self.dim { 0 } else { coords[d] };
                grad_flat += c * stride;
                stride *= grad_shape[d];
            }
            result.push(grad_data[grad_flat]);
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(result), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SumDimBackward"
    }
}

/// Sum along a specific dimension.
///
/// If `keepdim` is true, the output tensor has the reduced dimension with size 1.
/// If `keepdim` is false, the reduced dimension is removed.
///
/// `dim` supports negative indexing: `-1` means the last dimension.
pub fn sum_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_dim(input, dim, keepdim)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("sum_dim", "reduction", &[input.shape()], || {
        sum_dim_inner(input, dim, keepdim)
    })
}

fn sum_dim_inner<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "sum_dim: cannot reduce a scalar (0-D) tensor along a dimension".into(),
        });
    }

    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };

    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "sum_dim: dim {dim} is out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }

    let in_shape = input.shape();

    // Compute output shape.
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }

    // GPU path: use sum_axis kernel (no CPU round-trip).
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
    if input.is_cuda() && (is_f32 || is_f64) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let handle = if is_f32 {
                backend.sum_axis_f32(input.gpu_handle()?, in_shape, norm_dim)?
            } else {
                backend.sum_axis_f64(input.gpu_handle()?, in_shape, norm_dim)?
            };

            let storage = TensorStorage::gpu(handle);
            return if is_grad_enabled() && input.requires_grad() {
                let grad_fn = Arc::new(SumDimBackward {
                    input: input.clone(),
                    dim: norm_dim,
                    keepdim,
                });
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            };
        }
    }

    // Non-f32 CUDA tensors are not supported — user must call .cpu() explicitly.
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "sum_dim" });
    }
    let input_ref = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let in_data = input_ref.data()?;

    // For the accumulation, we work with a "keepdim" view internally.
    let mut accum_shape: Vec<usize> = in_shape.to_vec();
    accum_shape[norm_dim] = 1;
    let accum_numel: usize = accum_shape.iter().product();
    let mut accum = vec![<T as num_traits::Zero>::zero(); accum_numel];

    for (flat, &val) in in_data.iter().enumerate().take(input.numel()) {
        // Decompose flat index into per-axis coordinates.
        let mut rem = flat;
        let mut coords = vec![0usize; in_shape.len()];
        for d in (0..in_shape.len()).rev() {
            coords[d] = rem % in_shape[d];
            rem /= in_shape[d];
        }
        // Map to accumulator index (reduced dim coord -> 0).
        let mut oi = 0usize;
        let mut os = 1usize;
        for d in (0..accum_shape.len()).rev() {
            let c = if d == norm_dim { 0 } else { coords[d] };
            oi += c * os;
            os *= accum_shape[d];
        }
        accum[oi] += val;
    }

    let device = input.device();
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(SumDimBackward {
            input: input.clone(),
            dim: norm_dim,
            keepdim,
        });
        let storage = TensorStorage::on_device(accum, device)?;
        Tensor::from_operation(storage, out_shape, grad_fn)
    } else {
        let storage = TensorStorage::on_device(accum, device)?;
        Tensor::from_storage(storage, out_shape, false)
    }
}

// ---------------------------------------------------------------------------
// MeanDimBackward
// ---------------------------------------------------------------------------

/// Backward node for `mean_dim(input, dim, keepdim) -> reduced tensor`.
///
/// VJP: expand the gradient back along the reduced dimension and divide by
/// the size of that dimension.
#[derive(Debug)]
pub struct MeanDimBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
    keepdim: bool,
}

impl<T: Float> GradFn<T> for MeanDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_shape = self.input.shape();
        let dim_size = input_shape[self.dim];

        // GPU path: native expand-and-scale via fill + broadcast_mul.
        // Conceptually grad_input[..., j, ...] = grad_output[..., 0, ...] / N
        // for every j in the reduced dim. Implement that as:
        //   ones = fill(input_numel, 1/N)         shape: input_shape
        //   grad_input = broadcast_mul(ones, grad_output_keepdim)
        // grad_output_keepdim is grad_output with a size-1 dim re-inserted
        // when keepdim=false (free metadata change, no copy).
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        if grad_output.is_cuda() && is_f32 {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let grad_shape_keepdim: Vec<usize> = if self.keepdim {
                    grad_output.shape().to_vec()
                } else {
                    let mut s = grad_output.shape().to_vec();
                    s.insert(self.dim, 1);
                    s
                };
                let input_numel: usize = input_shape.iter().product();
                let inv_n = 1.0f32 / (dim_size as f32);
                // Use device 0 — current backend doesn't expose handle's
                // ordinal at this layer; the upstream GPU pipeline is
                // single-device for now. (Multi-device support lives in
                // a wider refactor; not blocking this.)
                let ones_handle = backend.fill_f32(input_numel, inv_n, 0)?;
                let grad_handle = grad_output.gpu_handle()?;
                let grad_input_handle = backend.broadcast_mul_f32(
                    &ones_handle,
                    grad_handle,
                    input_shape,
                    &grad_shape_keepdim,
                    input_shape,
                )?;
                let storage = TensorStorage::gpu(grad_input_handle);
                let grad_input = Tensor::from_storage(storage, input_shape.to_vec(), false)?;
                return Ok(vec![Some(grad_input)]);
            }
        }

        // f64 GPU path via the new repeat_along_dim kernel + scale (#524).
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
        if grad_output.is_cuda() && is_f64 {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let outer: usize = input_shape[..self.dim].iter().product::<usize>().max(1);
                let inner: usize = input_shape[(self.dim + 1)..]
                    .iter()
                    .product::<usize>()
                    .max(1);
                let repeat_count = dim_size;
                let expanded = backend.repeat_along_dim_f64(
                    grad_output.gpu_handle()?,
                    outer,
                    repeat_count,
                    inner,
                )?;
                // Scale by 1/repeat_count to get the mean's gradient.
                let scaled = backend.scale_f64(&expanded, 1.0 / repeat_count as f64)?;
                let grad_input =
                    Tensor::from_storage(TensorStorage::gpu(scaled), input_shape.to_vec(), false)?;
                return Ok(vec![Some(grad_input)]);
            }
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "mean_dim backward",
            });
        }

        let n = T::from(dim_size).unwrap();

        // If keepdim was false, reinsert the reduced dimension as size 1.
        let grad = if self.keepdim {
            grad_output.clone()
        } else {
            let mut unsqueezed_shape = grad_output.shape().to_vec();
            unsqueezed_shape.insert(self.dim, 1);
            let data = grad_output.data()?.to_vec();
            Tensor::from_storage(TensorStorage::cpu(data), unsqueezed_shape, false)?
        };

        // Expand along the reduced dim, dividing by dim_size.
        let grad_data = grad.data()?;
        let grad_shape = grad.shape();

        let out_numel: usize = input_shape.iter().product();
        let mut result = Vec::with_capacity(out_numel);

        for flat in 0..out_numel {
            let mut rem = flat;
            let mut coords = vec![0usize; input_shape.len()];
            for d in (0..input_shape.len()).rev() {
                coords[d] = rem % input_shape[d];
                rem /= input_shape[d];
            }
            let mut grad_flat = 0usize;
            let mut stride = 1usize;
            for d in (0..grad_shape.len()).rev() {
                let c = if d == self.dim { 0 } else { coords[d] };
                grad_flat += c * stride;
                stride *= grad_shape[d];
            }
            result.push(grad_data[grad_flat] / n);
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(result), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MeanDimBackward"
    }
}

/// Mean along a specific dimension.
///
/// If `keepdim` is true, the output tensor has the reduced dimension with size 1.
/// If `keepdim` is false, the reduced dimension is removed.
///
/// `dim` supports negative indexing: `-1` means the last dimension.
pub fn mean_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::reduce_dim(input, dim, keepdim)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("mean_dim", "reduction", &[input.shape()], || {
        mean_dim_inner(input, dim, keepdim)
    })
}

fn mean_dim_inner<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "mean_dim: cannot reduce a scalar (0-D) tensor along a dimension".into(),
        });
    }

    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };

    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "mean_dim: dim {dim} is out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }

    let in_shape = input.shape();
    let dim_size = in_shape[norm_dim];
    let n = T::from(dim_size).unwrap();

    // Compute output shape.
    let mut out_shape: Vec<usize> = in_shape.to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }

    // GPU path: native sum-then-scale on the existing backend kernels.
    // Mirrors `sum_dim_inner`'s dispatch but composes a second op
    // (`scale_f32(1/dim_size)`) so the result stays on-device end-to-end.
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
    if input.is_cuda() && (is_f32 || is_f64) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let summed_handle = if is_f32 {
                backend.sum_axis_f32(input.gpu_handle()?, in_shape, norm_dim)?
            } else {
                backend.sum_axis_f64(input.gpu_handle()?, in_shape, norm_dim)?
            };
            // Divide by dim_size on-device. scale_* is only declared
            // for f32; for f64 this path falls through to the error
            // below until a scale_f64 is added.
            if is_f32 {
                let mean_handle = backend.scale_f32(&summed_handle, 1.0 / dim_size as f32)?;
                let storage = TensorStorage::gpu(mean_handle);
                return if is_grad_enabled() && input.requires_grad() {
                    let grad_fn = Arc::new(MeanDimBackward {
                        input: input.clone(),
                        dim: norm_dim,
                        keepdim,
                    });
                    Tensor::from_operation(storage, out_shape, grad_fn)
                } else {
                    Tensor::from_storage(storage, out_shape, false)
                };
            }
        }
    }

    // No GPU handler matched; bail rather than silently round-trip.
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "mean_dim" });
    }

    let in_data = input.data()?;

    // Accumulate sum first, then divide.
    let mut accum_shape: Vec<usize> = in_shape.to_vec();
    accum_shape[norm_dim] = 1;
    let accum_numel: usize = accum_shape.iter().product();
    let mut accum = vec![<T as num_traits::Zero>::zero(); accum_numel];

    for (flat, &val) in in_data.iter().enumerate().take(input.numel()) {
        let mut rem = flat;
        let mut coords = vec![0usize; in_shape.len()];
        for d in (0..in_shape.len()).rev() {
            coords[d] = rem % in_shape[d];
            rem /= in_shape[d];
        }
        let mut oi = 0usize;
        let mut os = 1usize;
        for d in (0..accum_shape.len()).rev() {
            let c = if d == norm_dim { 0 } else { coords[d] };
            oi += c * os;
            os *= accum_shape[d];
        }
        accum[oi] += val;
    }

    // Divide by dim size to get mean.
    for v in &mut accum {
        *v = *v / n;
    }

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MeanDimBackward {
            input: input.clone(),
            dim: norm_dim,
            keepdim,
        });
        let result = Tensor::from_operation(TensorStorage::cpu(accum), out_shape, grad_fn)?;
        result.to(input.device())
    } else {
        let result = Tensor::from_storage(TensorStorage::cpu(accum), out_shape, false)?;
        result.to(input.device())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad::no_grad;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor with given data, shape, and requires_grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    /// Helper: create a leaf scalar.
    fn leaf_scalar(val: f64, requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    // --- Forward tests ---

    #[test]
    fn test_sum_forward_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let s = sum(&x).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_forward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum(&x).unwrap();
        assert!((s.item().unwrap() - 21.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_forward() {
        let x = leaf(&[2.0, 4.0, 6.0, 8.0], &[4], false);
        let m = mean(&x).unwrap();
        assert!((m.item().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward() {
        let x = leaf(&[2.0, 3.0, 4.0], &[3], false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap() - 24.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward_scalar() {
        let x = leaf_scalar(7.0, false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward_with_zero() {
        let x = leaf(&[3.0, 0.0, 5.0], &[3], false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap()).abs() < 1e-12);
    }

    // --- Backward tests ---

    #[test]
    fn test_sum_backward_scalar_input() {
        // sum(x) where x is a scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_backward_1d() {
        // sum([a, b, c]) = a + b + c. d/d(each) = 1.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert_eq!(gd.len(), 3);
        for &v in gd {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sum_backward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_backward_scalar_input() {
        // mean(x) where x is a scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_backward_1d() {
        // mean([a, b, c]) = (a + b + c) / 3. d/d(each) = 1/3.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        let expected = 1.0 / 3.0;
        for &v in gd {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_backward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        let expected = 1.0 / 6.0;
        for &v in g.data().unwrap() {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_prod_backward_scalar_input() {
        // prod(x) where x is scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_backward_1d() {
        // prod([a, b, c]) = a*b*c.
        // d/da = b*c, d/db = a*c, d/dc = a*b.
        let x = leaf(&[2.0, 3.0, 4.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!(
            (gd[0] - 12.0).abs() < 1e-12,
            "d/da = 3*4 = 12, got {}",
            gd[0]
        );
        assert!((gd[1] - 8.0).abs() < 1e-12, "d/db = 2*4 = 8, got {}", gd[1]);
        assert!((gd[2] - 6.0).abs() < 1e-12, "d/dc = 2*3 = 6, got {}", gd[2]);
    }

    #[test]
    fn test_prod_backward_with_zero() {
        // prod([3, 0, 5]) = 0.
        // d/d(x0) = 0*5 = 0, d/d(x1) = 3*5 = 15, d/d(x2) = 3*0 = 0.
        let x = leaf(&[3.0, 0.0, 5.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 0.0).abs() < 1e-12, "got {}", gd[0]);
        assert!((gd[1] - 15.0).abs() < 1e-12, "got {}", gd[1]);
        assert!((gd[2] - 0.0).abs() < 1e-12, "got {}", gd[2]);
    }

    #[test]
    fn test_prod_backward_two_zeros() {
        // prod([0, 0, 5]) = 0.
        // All gradients should be 0 (each product-excluding-one still contains a zero).
        let x = leaf(&[0.0, 0.0, 5.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        for &v in gd {
            assert!((v).abs() < 1e-12, "expected 0, got {v}");
        }
    }

    // --- Gradient tracking / no_grad tests ---

    #[test]
    fn test_sum_no_grad_fn_when_input_not_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let s = sum(&x).unwrap();
        assert!(s.grad_fn().is_none());
        assert!(!s.requires_grad());
    }

    #[test]
    fn test_sum_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum(&x).unwrap();
        assert!(s.grad_fn().is_some());
        assert_eq!(s.grad_fn().unwrap().name(), "SumBackward");
        assert!(s.requires_grad());
    }

    #[test]
    fn test_mean_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean(&x).unwrap();
        assert!(m.grad_fn().is_some());
        assert_eq!(m.grad_fn().unwrap().name(), "MeanBackward");
    }

    #[test]
    fn test_prod_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let p = prod(&x).unwrap();
        assert!(p.grad_fn().is_some());
        assert_eq!(p.grad_fn().unwrap().name(), "ProdBackward");
    }

    #[test]
    fn test_sum_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = no_grad(|| sum(&x)).unwrap();
        assert!(s.grad_fn().is_none());
        assert!(!s.requires_grad());
    }

    #[test]
    fn test_mean_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = no_grad(|| mean(&x)).unwrap();
        assert!(m.grad_fn().is_none());
    }

    #[test]
    fn test_prod_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[2.0, 3.0], &[2], true);
        let p = no_grad(|| prod(&x)).unwrap();
        assert!(p.grad_fn().is_none());
    }

    // --- Numerical gradient checking ---

    /// Finite-difference gradient check for a scalar -> scalar function.
    fn numerical_grad_check(
        f: impl Fn(&Tensor<f64>) -> FerrotorchResult<Tensor<f64>>,
        x_val: f64,
        expected_analytic: f64,
        tol: f64,
    ) {
        let eps = 1e-7;

        let x_plus = leaf_scalar(x_val + eps, false);
        let x_minus = leaf_scalar(x_val - eps, false);

        let f_plus = f(&x_plus).unwrap().item().unwrap();
        let f_minus = f(&x_minus).unwrap().item().unwrap();
        let numerical = (f_plus - f_minus) / (2.0 * eps);

        assert!(
            (numerical - expected_analytic).abs() < tol,
            "numerical gradient {numerical} differs from analytic {expected_analytic} by more than {tol}"
        );
    }

    #[test]
    fn test_sum_numerical_gradient() {
        // sum(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(sum, 3.0, analytic, 1e-5);
    }

    #[test]
    fn test_mean_numerical_gradient() {
        // mean(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(mean, 3.0, analytic, 1e-5);
    }

    #[test]
    fn test_prod_numerical_gradient() {
        // prod(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(prod, 3.0, analytic, 1e-5);
    }

    // --- sum_dim forward tests ---

    #[test]
    fn test_sum_dim_axis0_2d() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 0 -> [5, 7, 9]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum_dim(&x, 0, false).unwrap();
        assert_eq!(s.shape(), &[3]);
        let d = s.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 7.0).abs() < 1e-12);
        assert!((d[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_axis1_2d() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 1 -> [6, 15]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum_dim(&x, 1, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        let d = s.data().unwrap();
        assert!((d[0] - 6.0).abs() < 1e-12);
        assert!((d[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_keepdim_true() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum_dim(&x, 0, true).unwrap();
        assert_eq!(s.shape(), &[1, 3]);
        let d = s.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 7.0).abs() < 1e-12);
        assert!((d[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_negative_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        // dim=-1 means axis 1
        let s = sum_dim(&x, -1, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        let d = s.data().unwrap();
        assert!((d[0] - 6.0).abs() < 1e-12);
        assert!((d[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let s = sum_dim(&x, 0, false).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_1d_keepdim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let s = sum_dim(&x, 0, true).unwrap();
        assert_eq!(s.shape(), &[1]);
        assert!((s.data().unwrap()[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_dim_3d() {
        // shape [2, 2, 3], sum along dim=1
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let x = leaf(&data, &[2, 2, 3], false);
        let s = sum_dim(&x, 1, false).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        let d = s.data().unwrap();
        // [1,2,3] + [4,5,6] = [5,7,9]
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 7.0).abs() < 1e-12);
        assert!((d[2] - 9.0).abs() < 1e-12);
        // [7,8,9] + [10,11,12] = [17,19,21]
        assert!((d[3] - 17.0).abs() < 1e-12);
        assert!((d[4] - 19.0).abs() < 1e-12);
        assert!((d[5] - 21.0).abs() < 1e-12);
    }

    // --- sum_dim backward tests ---

    #[test]
    fn test_sum_dim_backward_axis0_no_keepdim() {
        // x: [2, 3], sum(dim=0) -> [3]
        // grad of sum along axis 0: each row gets the same gradient
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let s = sum_dim(&x, 0, false).unwrap();
        // sum the result to get a scalar for backward
        let loss = sum(&s).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_sum_dim_backward_axis1_keepdim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let s = sum_dim(&x, 1, true).unwrap();
        assert_eq!(s.shape(), &[2, 1]);
        let loss = sum(&s).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_sum_dim_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum_dim(&x, 0, false).unwrap();
        assert!(s.grad_fn().is_some());
        assert_eq!(s.grad_fn().unwrap().name(), "SumDimBackward");
    }

    #[test]
    fn test_sum_dim_no_grad_fn_when_not_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let s = sum_dim(&x, 0, false).unwrap();
        assert!(s.grad_fn().is_none());
    }

    #[test]
    fn test_sum_dim_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = no_grad(|| sum_dim(&x, 0, false)).unwrap();
        assert!(s.grad_fn().is_none());
    }

    // --- mean_dim forward tests ---

    #[test]
    fn test_mean_dim_axis0_2d() {
        // [[1, 2, 3], [4, 5, 6]] mean along axis 0 -> [2.5, 3.5, 4.5]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, 0, false).unwrap();
        assert_eq!(m.shape(), &[3]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.5).abs() < 1e-12);
        assert!((d[1] - 3.5).abs() < 1e-12);
        assert!((d[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_mean_dim_axis1_2d() {
        // [[1, 2, 3], [4, 5, 6]] mean along axis 1 -> [2, 5]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, 1, false).unwrap();
        assert_eq!(m.shape(), &[2]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-12);
        assert!((d[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_dim_keepdim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, 0, true).unwrap();
        assert_eq!(m.shape(), &[1, 3]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.5).abs() < 1e-12);
        assert!((d[1] - 3.5).abs() < 1e-12);
        assert!((d[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_mean_dim_negative_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let m = mean_dim(&x, -1, false).unwrap();
        assert_eq!(m.shape(), &[2]);
        let d = m.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-12);
        assert!((d[1] - 5.0).abs() < 1e-12);
    }

    // --- mean_dim backward tests ---

    #[test]
    fn test_mean_dim_backward_axis0() {
        // x: [2, 3], mean(dim=0) -> [3]
        // grad: each element gets 1/2 (since dim_size=2)
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let m = mean_dim(&x, 0, false).unwrap();
        let loss = sum(&m).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        let expected = 1.0 / 2.0;
        for &v in g.data().unwrap() {
            assert!((v - expected).abs() < 1e-12, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_mean_dim_backward_axis1_keepdim() {
        // x: [2, 3], mean(dim=1, keepdim=true) -> [2, 1]
        // grad: each element gets 1/3 (since dim_size=3)
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let m = mean_dim(&x, 1, true).unwrap();
        assert_eq!(m.shape(), &[2, 1]);
        let loss = sum(&m).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        let expected = 1.0 / 3.0;
        for &v in g.data().unwrap() {
            assert!((v - expected).abs() < 1e-12, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_mean_dim_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean_dim(&x, 0, false).unwrap();
        assert!(m.grad_fn().is_some());
        assert_eq!(m.grad_fn().unwrap().name(), "MeanDimBackward");
    }
}
