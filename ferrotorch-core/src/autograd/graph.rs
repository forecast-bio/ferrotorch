use rustc_hash::FxHashMap as HashMap;
use std::collections::VecDeque;

use crate::autograd::hooks::{run_grad_hooks, run_post_accumulate_hooks};
use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::{Tensor, TensorId};

/// Compute gradients of all leaf tensors that contribute to `root`.
///
/// Implements reverse-mode automatic differentiation:
/// 1. Collect all nodes reachable from `root` that have a `grad_fn`.
/// 2. Topological sort via Kahn's algorithm (iterative, no stack overflow).
/// 3. Walk in reverse topological order, calling each node's `GradFn::backward()`.
/// 4. Accumulate gradients additively on leaf tensors.
///
/// `root` must be a scalar tensor (0-dim or single element). After this call,
/// leaf tensors with `requires_grad = true` will have their `.grad()` populated.
pub fn backward<T: Float>(root: &Tensor<T>) -> FerrotorchResult<()> {
    backward_with_grad(root, None)
}

/// Run backward pass through the computation graph.
///
/// If `gradient` is `None`, the root must be scalar and an implicit seed of 1.0 is used.
/// If `gradient` is `Some`, it is used as the initial gradient for the root tensor,
/// allowing backward on non-scalar tensors (needed for multi-head outputs, Jacobian
/// computation, and custom loss functions).
pub fn backward_with_grad<T: Float>(
    root: &Tensor<T>,
    gradient: Option<&Tensor<T>>,
) -> FerrotorchResult<()> {
    let seed = if let Some(ext_grad) = gradient {
        // Validate that the external gradient shape matches the root shape.
        if ext_grad.shape() != root.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "gradient shape {:?} does not match root shape {:?}",
                    ext_grad.shape(),
                    root.shape(),
                ),
            });
        }
        ext_grad.clone()
    } else {
        // No external gradient: root must be scalar.
        if !root.is_scalar() && root.numel() != 1 {
            return Err(FerrotorchError::BackwardNonScalar {
                shape: root.shape().to_vec(),
            });
        }

        // Seed gradient: d(root)/d(root) = 1, on the same device as root.
        let ones_storage = crate::storage::TensorStorage::cpu(vec![<T as num_traits::One>::one()]);
        let seed_cpu = Tensor::from_storage(ones_storage, vec![], false)?;
        seed_cpu.to(root.device())?
    };

    // Phase 1: Collect all nodes and compute in-degree via BFS.
    //
    // We traverse the graph from `root` backward through `grad_fn().inputs()`.
    // `in_degree[id]` counts how many times a tensor is used as an input to
    // an operation — this is needed for Kahn's algorithm.
    let mut in_degree: HashMap<TensorId, usize> = HashMap::default();
    let mut node_map: HashMap<TensorId, &Tensor<T>> = HashMap::default();
    let mut queue: VecDeque<&Tensor<T>> = VecDeque::new();

    // Start from root.
    queue.push_back(root);
    in_degree.entry(root.id()).or_insert(0);
    node_map.insert(root.id(), root);

    while let Some(node) = queue.pop_front() {
        if let Some(grad_fn) = node.grad_fn() {
            for input in grad_fn.inputs() {
                let input_id = input.id();
                let count = in_degree.entry(input_id).or_insert(0);
                *count += 1;
                if let std::collections::hash_map::Entry::Vacant(e) = node_map.entry(input_id) {
                    e.insert(input);
                    queue.push_back(input);
                }
            }
        }
    }

    // Phase 2: Topological sort (Kahn's algorithm).
    //
    // Start with nodes that have in_degree == 0. The root always has in_degree 0
    // (nothing depends on it in the backward direction). Process nodes in
    // topological order, decrementing in_degree of their inputs.
    let mut topo_order: Vec<TensorId> = Vec::new();
    let mut bfs_queue: VecDeque<TensorId> = VecDeque::new();

    // Find all nodes with in_degree 0 (just the root in a standard graph).
    for (&id, &deg) in &in_degree {
        if deg == 0 {
            bfs_queue.push_back(id);
        }
    }

    while let Some(id) = bfs_queue.pop_front() {
        topo_order.push(id);
        if let Some(node) = node_map.get(&id) {
            if let Some(grad_fn) = node.grad_fn() {
                for input in grad_fn.inputs() {
                    if let Some(deg) = in_degree.get_mut(&input.id()) {
                        *deg -= 1;
                        if *deg == 0 {
                            bfs_queue.push_back(input.id());
                        }
                    }
                }
            }
        }
    }

    // Phase 3: Backward pass in topological order.
    //
    // We maintain a map of accumulated output gradients for each node.
    // For the root, the gradient is the seed (1.0).
    let mut grads: HashMap<TensorId, Tensor<T>> = HashMap::default();
    grads.insert(root.id(), seed);

    for &id in &topo_order {
        let node = match node_map.get(&id) {
            Some(n) => *n,
            None => continue,
        };

        let grad_output = match grads.remove(&id) {
            Some(g) => g,
            None => continue,
        };

        if let Some(grad_fn) = node.grad_fn() {
            // Materialize non-contiguous CPU gradients before backward
            let grad_output = if !grad_output.is_contiguous() && !grad_output.is_cuda() {
                crate::methods::contiguous_t(&grad_output)?
            } else {
                grad_output
            };
            let input_grads = grad_fn.backward(&grad_output)?;
            let inputs = grad_fn.inputs();

            // B3 fix: validate that backward returned the correct number
            // of gradients. Without this, `zip` silently drops trailing
            // gradients when the backward function returns fewer than
            // expected, causing silent incorrect results.
            if input_grads.len() != inputs.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "backward returned {} gradients but expected {}",
                        input_grads.len(),
                        inputs.len(),
                    ),
                });
            }

            for (input, maybe_grad) in inputs.iter().zip(input_grads.into_iter()) {
                if let Some(grad) = maybe_grad {
                    if input.requires_grad() {
                        // Run gradient hooks (if any), which may modify the gradient.
                        let hooks = input.hooks();
                        let has_hooks = {
                            let guard =
                                hooks.lock().map_err(|e| FerrotorchError::LockPoisoned {
                                    message: format!("hook storage mutex: {e}"),
                                })?;
                            (guard.has_grad_hooks(), guard.has_post_accumulate_hooks())
                        };
                        let grad = if has_hooks.0 {
                            run_grad_hooks(hooks, grad)?
                        } else {
                            grad
                        };

                        if input.is_leaf() {
                            // Leaf tensor: accumulate gradient on the tensor itself.
                            input.accumulate_grad(&grad)?;
                            // Run post-accumulate-grad hooks on the leaf (if any).
                            if has_hooks.1 {
                                run_post_accumulate_hooks(hooks, input)?;
                            }
                        } else {
                            // Non-leaf: accumulate into the grads map for the next iteration.
                            accumulate_non_leaf_grad(&mut grads, input, grad)?;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Accumulate a gradient for a non-leaf tensor in the backward grads map.
///
/// This is separated from the main backward loop for clarity and to
/// encapsulate the B1 / B6 fixes:
///
/// - **B1**: In-place accumulation is only attempted when both the outer
///   `Arc<TensorInner>` and the inner `Arc<TensorStorage>` have a strong
///   count of 1, the tensor is contiguous, and it is NOT on GPU. Without
///   the storage refcount check, shared-storage views could be corrupted.
///
/// - **B6**: When both the existing gradient and the incoming gradient are
///   on the same GPU device, we use `backend.add_f32()` / `add_f64()`
///   directly instead of round-tripping through CPU. This eliminates two
///   unnecessary PCIe transfers per accumulation.
fn accumulate_non_leaf_grad<T: Float>(
    grads: &mut HashMap<TensorId, Tensor<T>>,
    input: &Tensor<T>,
    grad: Tensor<T>,
) -> FerrotorchResult<()> {
    let Some(existing) = grads.remove(&input.id()) else {
        grads.insert(input.id(), grad);
        return Ok(());
    };

    // Shape validation.
    if existing.shape() != grad.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "gradient shape mismatch during accumulation: {:?} vs {:?}",
                existing.shape(),
                grad.shape(),
            ),
        });
    }

    // B6 fix: GPU-native accumulation when both tensors are on the same GPU.
    if let (Device::Cuda(_), Device::Cuda(_)) = (existing.device(), grad.device()) {
        if existing.device() == grad.device() {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let a_handle = existing.gpu_handle()?;
                let b_handle = grad.gpu_handle()?;
                // Dispatch by element size to pick add_f32 or add_f64.
                let result_handle = if std::mem::size_of::<T>() == 4 {
                    backend.add_f32(a_handle, b_handle)?
                } else {
                    backend.add_f64(a_handle, b_handle)?
                };
                let storage = crate::storage::TensorStorage::gpu(result_handle);
                let combined = Tensor::from_storage(storage, existing.shape().to_vec(), false)?;
                grads.insert(input.id(), combined);
                return Ok(());
            }
        }
    }

    // B1 fix: in-place accumulation is only safe when we have exclusive
    // ownership of BOTH the TensorInner Arc AND the TensorStorage Arc,
    // the tensor is contiguous, and it is on CPU. Without the storage
    // refcount check, views sharing the same storage would be corrupted.
    if existing.inner_refcount() == 1
        && existing.storage_refcount() == 1
        && existing.is_contiguous()
        && !existing.is_cuda()
    {
        // SAFETY: inner_refcount == 1 && storage_refcount == 1 guarantees
        // exclusive ownership. No other references exist.
        let existing_slice = unsafe { existing.data_mut()? };
        let grad_cpu = if grad.is_cuda() { grad.cpu()? } else { grad };
        let grad_data = grad_cpu.data()?;
        if existing_slice.len() != grad_data.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "gradient length mismatch during accumulation: {} vs {}",
                    existing_slice.len(),
                    grad_data.len(),
                ),
            });
        }
        for (e, &g) in existing_slice.iter_mut().zip(grad_data.iter()) {
            *e += g;
        }
        grads.insert(input.id(), existing);
        return Ok(());
    }

    // Fallback: allocate a new tensor for the sum (CPU path).
    let device = existing.device();
    let existing_cpu = if existing.is_cuda() {
        existing.cpu()?
    } else {
        existing
    };
    let grad_cpu = if grad.is_cuda() { grad.cpu()? } else { grad };
    let mut existing_data = existing_cpu.data()?.to_vec();
    let grad_data = grad_cpu.data()?;
    if existing_data.len() != grad_data.len() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "gradient length mismatch during accumulation: {} vs {}",
                existing_data.len(),
                grad_data.len(),
            ),
        });
    }
    for (e, &g) in existing_data.iter_mut().zip(grad_data.iter()) {
        *e += g;
    }
    let storage = crate::storage::TensorStorage::cpu(existing_data);
    let combined = Tensor::from_storage(storage, existing_cpu.shape().to_vec(), false)?;
    grads.insert(input.id(), combined.to(device)?);
    Ok(())
}

/// Convenience methods on Tensor for calling backward.
impl<T: Float> Tensor<T> {
    /// Compute gradients of all leaf tensors that contribute to this tensor.
    ///
    /// This tensor must be scalar (0-dim or single-element). After this call,
    /// leaf tensors with `requires_grad = true` will have their `.grad()` set.
    pub fn backward(&self) -> FerrotorchResult<()> {
        backward(self)
    }

    /// Run backward with an external gradient.
    ///
    /// This allows backward on non-scalar tensors by providing the initial
    /// gradient explicitly. The gradient shape must match this tensor's shape.
    /// Used for multi-head outputs, Jacobian computation, and custom loss
    /// functions.
    pub fn backward_with_gradient(&self, gradient: &Tensor<T>) -> FerrotorchResult<()> {
        backward_with_grad(self, Some(gradient))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;
    use crate::tensor::GradFn;
    use std::sync::Arc;

    /// A simple grad_fn for testing: output = a + b.
    /// backward: d(a+b)/da = 1, d(a+b)/db = 1.
    #[derive(Debug)]
    struct AddBackward<T: Float> {
        a: Tensor<T>,
        b: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for AddBackward<T> {
        fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            Ok(vec![Some(grad_output.clone()), Some(grad_output.clone())])
        }
        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.a, &self.b]
        }
        fn name(&self) -> &'static str {
            "AddBackward"
        }
    }

    /// A simple grad_fn: output = a * b (elementwise).
    /// backward: d(a*b)/da = b * grad, d(a*b)/db = a * grad.
    #[derive(Debug)]
    struct MulBackward<T: Float> {
        a: Tensor<T>,
        b: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for MulBackward<T> {
        fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let go = grad_output.data()?;
            let a_data = self.a.data()?;
            let b_data = self.b.data()?;

            let grad_a: Vec<T> = go.iter().zip(b_data.iter()).map(|(&g, &b)| g * b).collect();
            let grad_b: Vec<T> = go.iter().zip(a_data.iter()).map(|(&g, &a)| g * a).collect();

            let ta =
                Tensor::from_storage(TensorStorage::cpu(grad_a), self.a.shape().to_vec(), false)?;
            let tb =
                Tensor::from_storage(TensorStorage::cpu(grad_b), self.b.shape().to_vec(), false)?;
            Ok(vec![Some(ta), Some(tb)])
        }
        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.a, &self.b]
        }
        fn name(&self) -> &'static str {
            "MulBackward"
        }
    }

    /// Helper to make a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    #[test]
    fn test_backward_simple_add() {
        // c = a + b, backward from c.
        // dc/da = 1, dc/db = 1.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);

        let sum_val = a.data().unwrap()[0] + b.data().unwrap()[0];
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![sum_val]),
            vec![],
            Arc::new(AddBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        c.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert!((a_grad.item().unwrap() - 1.0).abs() < 1e-6);
        assert!((b_grad.item().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_mul() {
        // c = a * b, backward from c.
        // dc/da = b = 3.0, dc/db = a = 2.0.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);

        let prod_val = a.data().unwrap()[0] * b.data().unwrap()[0];
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![prod_val]),
            vec![],
            Arc::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        c.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert!((a_grad.item().unwrap() - 3.0).abs() < 1e-6);
        assert!((b_grad.item().unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_shared_input() {
        // c = a + a, backward from c.
        // dc/da = 1 + 1 = 2.
        let a = leaf_scalar(5.0, true);

        let sum_val = a.data().unwrap()[0] + a.data().unwrap()[0];
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![sum_val]),
            vec![],
            Arc::new(AddBackward {
                a: a.clone(),
                b: a.clone(),
            }),
        )
        .unwrap();

        c.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        assert!((a_grad.item().unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_chain() {
        // d = (a * b) + b
        // dd/da = b = 3.0
        // dd/db = a + 1 = 2.0 + 1.0 = 3.0
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);

        // c = a * b
        let c_val = 2.0 * 3.0;
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![c_val]),
            vec![],
            Arc::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        // d = c + b
        let d_val = c_val + 3.0;
        let d = Tensor::from_operation(
            TensorStorage::cpu(vec![d_val]),
            vec![],
            Arc::new(AddBackward {
                a: c.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        d.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert!(
            (a_grad.item().unwrap() - 3.0).abs() < 1e-6,
            "expected dd/da = 3.0, got {}",
            a_grad.item().unwrap()
        );
        assert!(
            (b_grad.item().unwrap() - 3.0).abs() < 1e-6,
            "expected dd/db = 3.0, got {}",
            b_grad.item().unwrap()
        );
    }

    #[test]
    fn test_backward_non_scalar_error() {
        let t =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false)
                .unwrap();
        assert!(t.backward().is_err());
    }
}
