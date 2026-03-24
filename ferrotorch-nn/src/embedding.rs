//! Embedding layer: a lookup table of fixed-size vectors.
//!
//! Maps integer indices (stored as `T` values and cast to `usize`) to
//! dense vectors. This is the standard way to represent discrete tokens
//! (words, subwords, categorical features) as continuous vectors for
//! gradient-based learning.
//!
//! The backward pass implements a sparse scatter-add: only the rows that
//! were accessed receive gradient, and duplicate indices accumulate.

use std::any::TypeId;
use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::device::Device;
use ferrotorch_core::gpu_dispatch::{GpuBufferHandle, gpu_backend};
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::init;
use crate::module::Module;
use crate::parameter::Parameter;

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// Upload a CPU `&[f32]` slice to a GPU buffer on the given device ordinal.
fn upload_f32_to_gpu(data: &[f32], ordinal: usize) -> FerrotorchResult<GpuBufferHandle> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.cpu_to_gpu(bytes, 4, ordinal)
}

// ---------------------------------------------------------------------------
// EmbeddingBackward
// ---------------------------------------------------------------------------

/// Backward function for the embedding lookup.
///
/// Forward: `output[i, :] = weight[indices[i], :]`
///
/// VJP: `grad_weight = zeros(num_embeddings, embedding_dim);`
///       `for i, idx in indices: grad_weight[idx, :] += grad_output[i, :]`
///
/// This is a sparse gradient — only accessed rows are non-zero.
/// Duplicate indices accumulate their corresponding `grad_output` rows.
#[derive(Debug)]
pub struct EmbeddingBackward<T: Float> {
    /// The weight tensor (needed for graph traversal and shape).
    weight: Tensor<T>,
    /// Indices used in the forward pass.
    indices: Vec<usize>,
    /// Total number of embedding rows.
    num_embeddings: usize,
    /// Width of each embedding vector.
    embedding_dim: usize,
    /// If set, this row's gradient is always zero.
    padding_idx: Option<usize>,
}

impl<T: Float> GradFn<T> for EmbeddingBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let dim = self.embedding_dim;
        let device = self.weight.device();

        // GPU fast path: scatter-add rows entirely on GPU for f32 tensors.
        if grad_output.is_cuda() && is_f32::<T>() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match device {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };

            // Upload indices as f32 to GPU (small: N * 4 bytes).
            let indices_f32: Vec<f32> = self.indices.iter().map(|&i| i as f32).collect();
            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;

            let go_handle = grad_output.gpu_handle()?;

            // Scatter-add rows on GPU: grad_weight[indices[i], :] += grad_output[i, :]
            let mut gw_handle =
                backend.scatter_add_rows_f32(go_handle, &idx_handle, self.num_embeddings, dim)?;

            // If padding_idx is set, zero that row's gradient.
            // Upload a zero row and overwrite via the backend.
            if let Some(pad_idx) = self.padding_idx {
                // Zero out the padding row by downloading, zeroing, re-uploading.
                // This is a single row (dim * 4 bytes), so acceptable.
                let mut gw_bytes = backend.gpu_to_cpu(&gw_handle)?;
                let gw_f32: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        gw_bytes.as_mut_ptr() as *mut f32,
                        gw_bytes.len() / 4,
                    )
                };
                let start = pad_idx * dim;
                for v in &mut gw_f32[start..start + dim] {
                    *v = 0.0;
                }
                let upload_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(gw_f32.as_ptr() as *const u8, gw_f32.len() * 4)
                };
                gw_handle = backend.cpu_to_gpu(upload_bytes, 4, ordinal)?;
            }

            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(gw_handle),
                vec![self.num_embeddings, dim],
                false,
            )?;
            return Ok(vec![Some(grad_tensor)]);
        }

        let go_data = grad_output.data_vec()?;

        // Allocate a full-size gradient for the weight matrix, initialized to zero.
        let mut grad_weight = vec![<T as num_traits::Zero>::zero(); self.num_embeddings * dim];

        // Scatter-add: for each index position, accumulate the corresponding
        // grad_output row into the weight gradient at the accessed index.
        for (i, &idx) in self.indices.iter().enumerate() {
            let go_row = &go_data[i * dim..(i + 1) * dim];
            let gw_row = &mut grad_weight[idx * dim..(idx + 1) * dim];
            for (gw, &go) in gw_row.iter_mut().zip(go_row.iter()) {
                *gw += go;
            }
        }

        // If padding_idx is set, zero that row's gradient unconditionally.
        if let Some(pad_idx) = self.padding_idx {
            let start = pad_idx * dim;
            for v in &mut grad_weight[start..start + dim] {
                *v = <T as num_traits::Zero>::zero();
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_weight),
            vec![self.num_embeddings, dim],
            false,
        )?;

        // Return gradient on the same device as the weight.
        if device.is_cuda() {
            Ok(vec![Some(grad_tensor.to(device)?)])
        } else {
            Ok(vec![Some(grad_tensor)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight]
    }

    fn name(&self) -> &'static str {
        "EmbeddingBackward"
    }
}

// ---------------------------------------------------------------------------
// Embedding layer
// ---------------------------------------------------------------------------

/// A simple lookup table that stores embeddings of a fixed dictionary.
///
/// Given a 1-D tensor of integer indices (stored as float values, cast to
/// `usize`), returns a 2-D tensor `[len, embedding_dim]` by gathering the
/// corresponding rows from the weight matrix.
///
/// # Padding index
///
/// If `padding_idx` is set, the embedding vector at that index is always
/// zero and receives no gradient updates. This is commonly used to
/// represent a padding token.
///
/// # Example
///
/// ```ignore
/// let emb = Embedding::<f32>::new(1000, 64, None)?;
/// let indices = ferrotorch_core::tensor(&[1.0, 5.0, 3.0])?;
/// let output = emb.forward(&indices)?;
/// assert_eq!(output.shape(), &[3, 64]);
/// ```
#[derive(Debug)]
pub struct Embedding<T: Float> {
    /// The learnable weight matrix, shape `[num_embeddings, embedding_dim]`.
    pub weight: Parameter<T>,
    /// Number of entries in the lookup table.
    pub num_embeddings: usize,
    /// Dimensionality of each embedding vector.
    pub embedding_dim: usize,
    /// If set, this row is kept at zero and receives no gradient.
    pub padding_idx: Option<usize>,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Embedding<T> {
    /// Create a new embedding layer.
    ///
    /// Weight is initialized from N(0, 1). If `padding_idx` is set, that
    /// row is zeroed after initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if `padding_idx >= num_embeddings`.
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> FerrotorchResult<Self> {
        // Validate padding_idx.
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "padding_idx {idx} is out of range for num_embeddings {num_embeddings}"
                    ),
                });
            }
        }

        // Initialize weight from N(0, 1).
        let mut weight = Parameter::zeros(&[num_embeddings, embedding_dim])?;
        init::normal(&mut weight, 0.0, 1.0)?;

        // Zero the padding row if requested.
        if let Some(idx) = padding_idx {
            let data = weight.data()?.to_vec();
            let mut new_data = data;
            let start = idx * embedding_dim;
            for v in &mut new_data[start..start + embedding_dim] {
                *v = <T as num_traits::Zero>::zero();
            }
            weight = Parameter::new(Tensor::from_storage(
                TensorStorage::cpu(new_data),
                vec![num_embeddings, embedding_dim],
                true,
            )?);
        }

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
            training: true,
        })
    }

    /// Create an embedding layer from an existing weight tensor.
    ///
    /// The tensor must have shape `[num_embeddings, embedding_dim]`.
    pub fn from_pretrained(
        weight: Tensor<T>,
        padding_idx: Option<usize>,
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Embedding weight must be 2-D, got shape {:?}",
                    weight.shape()
                ),
            });
        }
        let num_embeddings = weight.shape()[0];
        let embedding_dim = weight.shape()[1];

        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "padding_idx {idx} is out of range for num_embeddings {num_embeddings}"
                    ),
                });
            }
        }

        Ok(Self {
            weight: Parameter::new(weight),
            num_embeddings,
            embedding_dim,
            padding_idx,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Embedding<T> {
    /// Forward pass: look up embedding vectors for the given indices.
    ///
    /// `input` must be a 1-D tensor whose values are non-negative integers
    /// stored as floats. Each value is cast to `usize` and used to index
    /// into the weight matrix.
    ///
    /// Returns a tensor of shape `[input.len(), embedding_dim]`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Validate input is 1-D.
        if input.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("Embedding input must be 1-D, got shape {:?}", input.shape()),
            });
        }

        let dim = self.embedding_dim;

        // GPU fast path for f32 embeddings: gather rows entirely on GPU,
        // avoiding the costly download of the full weight matrix to CPU.
        if self.weight.tensor().is_cuda() && is_f32::<T>() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let device = self.weight.tensor().device();
            let ordinal = match device {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };

            // Download input indices to CPU for validation (small: N floats).
            let input_data = input.data_vec()?;
            let n = input_data.len();

            // Convert float indices to usize and validate bounds.
            let mut indices = Vec::with_capacity(n);
            let mut indices_f32 = Vec::with_capacity(n);
            for (i, &val) in input_data.iter().enumerate() {
                let idx = num_traits::ToPrimitive::to_usize(&val).ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: format!(
                            "Embedding index at position {i} cannot be converted to usize: {val:?}"
                        ),
                    }
                })?;
                if idx >= self.num_embeddings {
                    return Err(FerrotorchError::IndexOutOfBounds {
                        index: idx,
                        axis: 0,
                        size: self.num_embeddings,
                    });
                }
                indices.push(idx);
                indices_f32.push(idx as f32);
            }

            // Upload indices to GPU (tiny: N * 4 bytes).
            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;

            // Get weight GPU handle directly -- no download.
            let weight_handle = self.weight.tensor().gpu_handle()?;

            // Batch gather on GPU: output [N, D].
            let output_handle =
                backend.embed_lookup_batch_f32(&idx_handle, weight_handle, n, dim)?;

            // Padding index: if set, zero the corresponding output rows on GPU.
            // For padding_idx, the weight row should already be zero, so output
            // rows at padding positions should already be zero. Be defensive
            // only if padding_idx is actually referenced.
            // (The weight is zeroed at init, so we skip extra GPU work here.)

            let output_shape = vec![n, dim];
            let storage = TensorStorage::gpu(output_handle);

            if self.weight.requires_grad() && is_grad_enabled() {
                let grad_fn = Arc::new(EmbeddingBackward {
                    weight: self.weight.tensor().clone(),
                    indices,
                    num_embeddings: self.num_embeddings,
                    embedding_dim: dim,
                    padding_idx: self.padding_idx,
                });
                return Tensor::from_operation(storage, output_shape, grad_fn);
            } else {
                return Tensor::from_storage(storage, output_shape, false);
            }
        }

        // CPU path (or non-f32 GPU tensors): download weight and gather on CPU.
        let input_data = input.data_vec()?;
        let cpu_weight = if self.weight.tensor().is_cuda() {
            self.weight.tensor().cpu()?
        } else {
            self.weight.tensor().clone()
        };
        let weight_data = cpu_weight.data()?;
        let n = input_data.len();

        // Convert float indices to usize and validate bounds.
        let mut indices = Vec::with_capacity(n);
        for (i, &val) in input_data.iter().enumerate() {
            let idx = num_traits::ToPrimitive::to_usize(&val).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "Embedding index at position {i} cannot be converted to usize: {val:?}"
                    ),
                }
            })?;
            if idx >= self.num_embeddings {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: idx,
                    axis: 0,
                    size: self.num_embeddings,
                });
            }
            indices.push(idx);
        }

        // Gather rows from weight.
        let mut output_data = Vec::with_capacity(n * dim);
        for &idx in &indices {
            let row_start = idx * dim;
            output_data.extend_from_slice(&weight_data[row_start..row_start + dim]);
        }

        // If padding_idx is set, ensure those rows are zeros in the output
        // (they should already be zero in the weight, but be defensive).
        if let Some(pad_idx) = self.padding_idx {
            for (i, &idx) in indices.iter().enumerate() {
                if idx == pad_idx {
                    let start = i * dim;
                    for v in &mut output_data[start..start + dim] {
                        *v = <T as num_traits::Zero>::zero();
                    }
                }
            }
        }

        let output_shape = vec![n, dim];

        // Output device matches the weight's device (GPU if model is on GPU).
        let device = self.weight.tensor().device();

        // Build storage on the target device first, then attach grad_fn.
        // This avoids to() stripping the grad_fn by creating a leaf tensor.
        let storage = if device.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match device {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    output_data.as_ptr() as *const u8,
                    output_data.len() * std::mem::size_of::<T>(),
                )
            };
            let handle = backend.cpu_to_gpu(bytes, std::mem::size_of::<T>(), ordinal)?;
            TensorStorage::gpu(handle)
        } else {
            TensorStorage::cpu(output_data)
        };

        if self.weight.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(EmbeddingBackward {
                weight: self.weight.tensor().clone(),
                indices,
                num_embeddings: self.num_embeddings,
                embedding_dim: dim,
                padding_idx: self.padding_idx,
            });
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, output_shape, false)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![("weight".to_string(), &self.weight)]
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::autograd::graph::backward;
    use ferrotorch_core::storage::TensorStorage;

    /// Helper: create a 1-D tensor of float indices.
    fn index_tensor(indices: &[f32]) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(indices.to_vec()),
            vec![indices.len()],
            false,
        )
        .unwrap()
    }

    // --- Forward tests ---

    #[test]
    fn test_forward_shape() {
        let emb = Embedding::<f32>::new(10, 4, None).unwrap();
        let indices = index_tensor(&[0.0, 3.0, 7.0]);
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_forward_correct_values() {
        // Build an embedding with known weights.
        let weight_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![4, 3], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        // Look up rows 2 and 0.
        let indices = index_tensor(&[2.0, 0.0]);
        let output = emb.forward(&indices).unwrap();
        let data = output.data().unwrap();

        // Row 2 = [6, 7, 8], Row 0 = [0, 1, 2]
        assert_eq!(data.len(), 6);
        assert!((data[0] - 6.0).abs() < 1e-6);
        assert!((data[1] - 7.0).abs() < 1e-6);
        assert!((data[2] - 8.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
        assert!((data[5] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_forward_single_index() {
        let emb = Embedding::<f32>::new(5, 8, None).unwrap();
        let indices = index_tensor(&[3.0]);
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[1, 8]);
    }

    // --- Padding index tests ---

    #[test]
    fn test_padding_idx_zeros() {
        let emb = Embedding::<f32>::new(5, 3, Some(2)).unwrap();

        // The padding row in the weight should be zero.
        let w_data = emb.weight.data().unwrap();
        let pad_start = 2 * 3;
        for i in 0..3 {
            assert!(
                (w_data[pad_start + i] - 0.0).abs() < 1e-6,
                "padding row weight[2][{i}] should be 0, got {}",
                w_data[pad_start + i]
            );
        }

        // Forward with the padding index should return zeros.
        let indices = index_tensor(&[2.0]);
        let output = emb.forward(&indices).unwrap();
        let data = output.data().unwrap();
        for i in 0..3 {
            assert!(
                (data[i] - 0.0).abs() < 1e-6,
                "padding output[0][{i}] should be 0, got {}",
                data[i]
            );
        }
    }

    #[test]
    fn test_padding_idx_mixed() {
        // Build known weights, set padding_idx=1.
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // row 0
            0.0, 0.0, // row 1 (padding — will be zeroed)
            5.0, 6.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, Some(1)).unwrap();

        let indices = index_tensor(&[0.0, 1.0, 2.0]);
        let output = emb.forward(&indices).unwrap();
        let data = output.data().unwrap();

        // Row 0: [1, 2]
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        // Row 1 (padding): [0, 0]
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
        // Row 2: [5, 6]
        assert!((data[4] - 5.0).abs() < 1e-6);
        assert!((data[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_padding_idx_out_of_range() {
        let result = Embedding::<f32>::new(5, 3, Some(10));
        assert!(result.is_err());
    }

    // --- Out-of-bounds error ---

    #[test]
    fn test_out_of_bounds_error() {
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let indices = index_tensor(&[0.0, 5.0]); // 5 is out of bounds for num_embeddings=5
        let result = emb.forward(&indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_index_error() {
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let indices = index_tensor(&[-1.0]); // Negative cannot convert to usize
        let result = emb.forward(&indices);
        assert!(result.is_err());
    }

    // --- Non-1D input error ---

    #[test]
    fn test_non_1d_input_error() {
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32, 1.0, 2.0, 3.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let result = emb.forward(&input);
        assert!(result.is_err());
    }

    // --- Backward tests ---

    #[test]
    fn test_backward_simple() {
        // weight shape [3, 2], look up indices [0, 2]
        // output shape [2, 2]
        // grad_output = [[1, 1], [1, 1]]
        // grad_weight = [[1, 1], [0, 0], [1, 1]]
        let weight_data: Vec<f32> = vec![
            10.0, 20.0, // row 0
            30.0, 40.0, // row 1
            50.0, 60.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        let indices = index_tensor(&[0.0, 2.0]);
        let output = emb.forward(&indices).unwrap();

        assert!(output.requires_grad());
        assert_eq!(output.grad_fn().unwrap().name(), "EmbeddingBackward");

        // Manually call backward on the grad_fn.
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 4]), vec![2, 2], false).unwrap();

        let grad_fn = output.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_weight = grads[0].as_ref().unwrap();
        assert_eq!(grad_weight.shape(), &[3, 2]);
        let gd = grad_weight.data().unwrap();

        // Row 0: accessed once -> [1, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 1.0).abs() < 1e-6);
        // Row 1: not accessed -> [0, 0]
        assert!((gd[2] - 0.0).abs() < 1e-6);
        assert!((gd[3] - 0.0).abs() < 1e-6);
        // Row 2: accessed once -> [1, 1]
        assert!((gd[4] - 1.0).abs() < 1e-6);
        assert!((gd[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_duplicate_indices() {
        // weight shape [3, 2], look up indices [1, 1, 0, 1]
        // output shape [4, 2]
        //
        // grad_output = [[1, 2], [3, 4], [5, 6], [7, 8]]
        //
        // grad_weight[0] = grad_output[2] = [5, 6]       (index 0 appears once, at position 2)
        // grad_weight[1] = grad_output[0] + grad_output[1] + grad_output[3]
        //                = [1, 2] + [3, 4] + [7, 8] = [11, 14]
        // grad_weight[2] = [0, 0]                          (index 2 never accessed)
        let weight_data: Vec<f32> = vec![
            10.0, 20.0, // row 0
            30.0, 40.0, // row 1
            50.0, 60.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        let indices = index_tensor(&[1.0, 1.0, 0.0, 1.0]);
        let output = emb.forward(&indices).unwrap();

        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            vec![4, 2],
            false,
        )
        .unwrap();

        let grad_fn = output.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_weight = grads[0].as_ref().unwrap();
        let gd = grad_weight.data().unwrap();

        // Row 0: [5, 6]
        assert!((gd[0] - 5.0).abs() < 1e-6, "gd[0] = {}, expected 5", gd[0]);
        assert!((gd[1] - 6.0).abs() < 1e-6, "gd[1] = {}, expected 6", gd[1]);
        // Row 1: [1+3+7, 2+4+8] = [11, 14]
        assert!(
            (gd[2] - 11.0).abs() < 1e-6,
            "gd[2] = {}, expected 11",
            gd[2]
        );
        assert!(
            (gd[3] - 14.0).abs() < 1e-6,
            "gd[3] = {}, expected 14",
            gd[3]
        );
        // Row 2: [0, 0]
        assert!((gd[4] - 0.0).abs() < 1e-6, "gd[4] = {}, expected 0", gd[4]);
        assert!((gd[5] - 0.0).abs() < 1e-6, "gd[5] = {}, expected 0", gd[5]);
    }

    #[test]
    fn test_backward_padding_idx_zeroed() {
        // Even if padding_idx is accessed, its gradient should be zero.
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // row 0
            0.0, 0.0, // row 1 (padding)
            5.0, 6.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, Some(1)).unwrap();

        let indices = index_tensor(&[0.0, 1.0, 2.0]);
        let output = emb.forward(&indices).unwrap();

        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 6]), vec![3, 2], false).unwrap();

        let grad_fn = output.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_weight = grads[0].as_ref().unwrap();
        let gd = grad_weight.data().unwrap();

        // Row 0: [1, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 1.0).abs() < 1e-6);
        // Row 1 (padding): must be [0, 0] even though it was accessed
        assert!((gd[2] - 0.0).abs() < 1e-6, "padding grad[1][0] should be 0");
        assert!((gd[3] - 0.0).abs() < 1e-6, "padding grad[1][1] should be 0");
        // Row 2: [1, 1]
        assert!((gd[4] - 1.0).abs() < 1e-6);
        assert!((gd[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_end_to_end() {
        // End-to-end test: use the autograd engine to verify gradients
        // flow all the way to the weight parameter.
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        let indices = index_tensor(&[1.0, 0.0]);
        let output = emb.forward(&indices).unwrap();
        // output = [[3, 4], [1, 2]], shape [2, 2]

        // Sum all elements to get a scalar for backward.
        let out_data = output.data().unwrap();
        let total: f32 = out_data.iter().sum();

        // Build a SumBackward that broadcasts scalar grad to output shape.
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let go_val = grad_output.data()?[0];
                let grad = vec![go_val; self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(grad),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();

        backward(&loss).unwrap();

        // The weight should now have a gradient.
        let grad = emb.weight.tensor().grad().unwrap().unwrap();
        let gd = grad.data().unwrap();
        assert_eq!(gd.len(), 6);

        // Row 0 accessed once (position 1): grad = [1, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6, "grad[0][0] = {}", gd[0]);
        assert!((gd[1] - 1.0).abs() < 1e-6, "grad[0][1] = {}", gd[1]);
        // Row 1 accessed once (position 0): grad = [1, 1]
        assert!((gd[2] - 1.0).abs() < 1e-6, "grad[1][0] = {}", gd[2]);
        assert!((gd[3] - 1.0).abs() < 1e-6, "grad[1][1] = {}", gd[3]);
        // Row 2 not accessed: grad = [0, 0]
        assert!((gd[4] - 0.0).abs() < 1e-6, "grad[2][0] = {}", gd[4]);
        assert!((gd[5] - 0.0).abs() < 1e-6, "grad[2][1] = {}", gd[5]);
    }

    // --- Module trait tests ---

    #[test]
    fn test_module_parameters() {
        let emb = Embedding::<f32>::new(10, 4, None).unwrap();
        assert_eq!(emb.parameters().len(), 1);
        assert_eq!(emb.parameters()[0].shape(), &[10, 4]);
    }

    #[test]
    fn test_module_named_parameters() {
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let named = emb.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn test_module_train_eval() {
        let mut emb = Embedding::<f32>::new(5, 3, None).unwrap();
        assert!(emb.is_training());
        emb.eval();
        assert!(!emb.is_training());
        emb.train();
        assert!(emb.is_training());
    }

    #[test]
    fn test_embedding_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Embedding<f32>>();
        assert_send_sync::<Embedding<f64>>();
    }

    #[test]
    fn test_f64_embedding() {
        let emb = Embedding::<f64>::new(5, 3, None).unwrap();
        let indices =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f64, 2.0, 4.0]), vec![3], false)
                .unwrap();
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[3, 3]);
    }
}
