use std::collections::{BinaryHeap, VecDeque};
use rustc_hash::FxHashMap as HashMap;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::{Tensor, TensorId};

/// Compute gradients of all leaf tensors that contribute to `root`.
///
/// Implements reverse-mode automatic differentiation:
/// 1. Collect all nodes reachable from `root` that have a `grad_fn`.
/// 2. Process in priority order via Kahn's algorithm with a max-heap on
///    `TensorId`. Higher IDs (later-created ops) execute first, matching
///    PyTorch's `sequence_nr`-based scheduling. This improves memory usage
///    by freeing large intermediates sooner.
/// 3. Call each node's `GradFn::backward()` and accumulate gradients.
/// 4. When the accumulated gradient tensor has no other references and is
///    contiguous CPU, accumulate in-place to avoid allocation overhead.
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
                if !node_map.contains_key(&input_id) {
                    node_map.insert(input_id, input);
                    queue.push_back(input);
                }
            }
        }
    }

    // Check for cycles: if the graph has a cycle, we will never drain all
    // nodes (some will always have in_degree > 0). We detect this after
    // phase 2 by comparing processed count vs total node count.
    let total_nodes = in_degree.len();

    // Phase 2: Priority-queue topological sort (Kahn's algorithm with max-heap).
    //
    // Instead of a FIFO queue, we use a BinaryHeap (max-heap) ordered by TensorId.
    // TensorIds are monotonically increasing, so higher ID = later creation =
    // should execute first in backward. This matches PyTorch's sequence_nr-based
    // priority scheduling and improves memory usage by freeing large intermediates
    // sooner (LIFO-like processing of the computation graph).
    //
    // We process nodes directly from the heap — no separate topo_order Vec needed.
    let mut ready: BinaryHeap<TensorId> = BinaryHeap::new();

    // Find all nodes with in_degree 0 (just the root in a standard graph).
    for (&id, &deg) in &in_degree {
        if deg == 0 {
            ready.push(id);
        }
    }

    // Phase 3: Combined topological traversal + backward pass.
    //
    // We pop the highest-ID ready node, run its backward, accumulate gradients,
    // and push newly-ready nodes onto the heap. This fuses phases 2 and 3 from
    // the old BFS implementation, avoiding the intermediate topo_order Vec.
    let mut grads: HashMap<TensorId, Tensor<T>> = HashMap::default();
    grads.insert(root.id(), seed);

    let mut processed = 0usize;

    while let Some(id) = ready.pop() {
        processed += 1;

        let node = match node_map.get(&id) {
            Some(n) => *n,
            None => continue,
        };

        let grad_output = match grads.remove(&id) {
            Some(g) => g,
            None => {
                // Decrement children even when no gradient flows through this
                // node (e.g. disconnected component reachable from root but
                // with no gradient path). Still need to unblock dependents.
                if let Some(grad_fn) = node.grad_fn() {
                    for input in grad_fn.inputs() {
                        if let Some(deg) = in_degree.get_mut(&input.id()) {
                            *deg -= 1;
                            if *deg == 0 {
                                ready.push(input.id());
                            }
                        }
                    }
                }
                continue;
            }
        };

        if let Some(grad_fn) = node.grad_fn() {
            let inputs = grad_fn.inputs();

            // Decrement in-degree for ALL inputs first, before running backward.
            // This ensures every input gets its in-degree decremented even if
            // backward() returns fewer gradients than inputs.
            for input in &inputs {
                if let Some(deg) = in_degree.get_mut(&input.id()) {
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(input.id());
                    }
                }
            }

            // Materialize non-contiguous CPU gradients before backward
            let grad_output = if !grad_output.is_contiguous() && !grad_output.is_cuda() {
                crate::methods::contiguous_t(&grad_output)?
            } else {
                grad_output
            };
            let input_grads = grad_fn.backward(&grad_output)?;

            for (input, maybe_grad) in inputs.iter().zip(input_grads.into_iter()) {
                if let Some(grad) = maybe_grad {
                    if input.requires_grad() {
                        if input.is_leaf() {
                            // Leaf tensor: accumulate gradient on the tensor itself.
                            input.accumulate_grad(&grad)?;
                        } else {
                            // Non-leaf: accumulate into the grads map for the next iteration.
                            accumulate_non_leaf_grad(&mut grads, input.id(), grad)?;
                        }
                    }
                }
            }
        } else {
            // Leaf node with no grad_fn — nothing to propagate, but we still
            // need to count it as processed (already done above).
        }
    }

    // Cycle detection: if we couldn't process all nodes, the graph has a cycle.
    if processed != total_nodes {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "backward graph contains a cycle: processed {} of {} nodes",
                processed, total_nodes,
            ),
        });
    }

    Ok(())
}

/// Accumulate a gradient for a non-leaf tensor in the `grads` map.
///
/// When the existing gradient has a single Arc reference (inner_refcount == 1),
/// is contiguous, and lives on CPU, we accumulate in-place — avoiding a full
/// allocation + copy cycle. Otherwise, falls back to the safe allocating path.
fn accumulate_non_leaf_grad<T: Float>(
    grads: &mut HashMap<TensorId, Tensor<T>>,
    id: TensorId,
    grad: Tensor<T>,
) -> FerrotorchResult<()> {
    if let Some(existing) = grads.remove(&id) {
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

        // Fast path: in-place accumulation when we hold the only reference,
        // the tensor is contiguous, and it lives on CPU. The inner_refcount
        // check ensures no aliased references exist, making data_mut sound.
        if existing.inner_refcount() == 1 && existing.is_contiguous() && !existing.is_cuda() {
            let grad_cpu = if grad.is_cuda() { grad.cpu()? } else { grad };
            let grad_data = grad_cpu.data()?;
            // SAFETY: inner_refcount == 1 guarantees we hold the only Arc to
            // TensorInner, so no other code can read or write this storage
            // concurrently. The tensor is contiguous and on CPU (checked above).
            let existing_data = unsafe { existing.data_mut()? };
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
                *e = *e + g;
            }
            grads.insert(id, existing);
        } else {
            // Slow path: shared reference or GPU tensor — allocate a new tensor.
            let device = existing.device();
            let existing_cpu = if existing.is_cuda() { existing.cpu()? } else { existing };
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
                *e = *e + g;
            }
            let storage = crate::storage::TensorStorage::cpu(existing_data);
            let combined = Tensor::from_storage(
                storage,
                existing_cpu.shape().to_vec(),
                false,
            )?;
            grads.insert(id, combined.to(device)?);
        }
    } else {
        grads.insert(id, grad);
    }
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

            let ta = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?;
            let tb = Tensor::from_storage(
                TensorStorage::cpu(grad_b),
                self.b.shape().to_vec(),
                false,
            )?;
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
        let t = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0]),
            vec![3],
            false,
        )
        .unwrap();
        assert!(t.backward().is_err());
    }

    #[test]
    fn test_backward_diamond_graph() {
        // Diamond: e = (a+b) + (a*b), shared inputs a and b.
        //
        //       e
        //      / \
        //     c   d
        //      \ / \
        //       a   b
        //
        // c = a + b,  d = a * b,  e = c + d
        // de/da = dc/da + dd/da = 1 + b = 1 + 3 = 4
        // de/db = dc/db + dd/db = 1 + a = 1 + 2 = 3
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);

        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![5.0]),
            vec![],
            Arc::new(AddBackward { a: a.clone(), b: b.clone() }),
        ).unwrap();

        let d = Tensor::from_operation(
            TensorStorage::cpu(vec![6.0]),
            vec![],
            Arc::new(MulBackward { a: a.clone(), b: b.clone() }),
        ).unwrap();

        let e = Tensor::from_operation(
            TensorStorage::cpu(vec![11.0]),
            vec![],
            Arc::new(AddBackward { a: c.clone(), b: d.clone() }),
        ).unwrap();

        e.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert!(
            (a_grad.item().unwrap() - 4.0).abs() < 1e-6,
            "expected de/da = 4.0, got {}",
            a_grad.item().unwrap()
        );
        assert!(
            (b_grad.item().unwrap() - 3.0).abs() < 1e-6,
            "expected de/db = 3.0, got {}",
            b_grad.item().unwrap()
        );
    }

    #[test]
    fn test_backward_single_leaf_no_grad_fn() {
        // A single scalar leaf with requires_grad=true but no grad_fn.
        // backward should succeed (it's the root, seed = 1, no ops to traverse).
        // The leaf's .grad() is NOT set because backward only sets grads via
        // accumulate_grad when the leaf is an *input* to a grad_fn.
        let a = leaf_scalar(42.0, true);

        // backward_with_grad with an explicit seed
        let seed = Tensor::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();
        // There is no grad_fn, so nothing to propagate. This should not error.
        a.backward_with_gradient(&seed).unwrap();
    }

    #[test]
    fn test_backward_vector_accumulation() {
        // Verify non-scalar gradient accumulation works correctly.
        // c = a + a where a is a 3-element vector.
        // dc/da = [1,1,1] + [1,1,1] = [2,2,2]
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
            vec![3],
            true,
        ).unwrap();

        let sum_data: Vec<f32> = a.data().unwrap().iter().map(|&x| x + x).collect();
        let c = Tensor::from_operation(
            TensorStorage::cpu(sum_data),
            vec![3],
            Arc::new(AddBackward { a: a.clone(), b: a.clone() }),
        ).unwrap();

        // backward on non-scalar requires explicit gradient
        let grad_seed = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 1.0, 1.0]),
            vec![3],
            false,
        ).unwrap();
        c.backward_with_gradient(&grad_seed).unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let grad_data = a_grad.data().unwrap();
        assert_eq!(grad_data.len(), 3);
        for &v in grad_data.iter() {
            assert!((v - 2.0).abs() < 1e-6, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn test_backward_deep_chain() {
        // Deep chain: e = ((a * a) * a) * a = a^4
        // de/da = 4 * a^3 = 4 * 8 = 32 (for a = 2)
        let a = leaf_scalar(2.0, true);

        // b = a * a = 4
        let b = Tensor::from_operation(
            TensorStorage::cpu(vec![4.0]),
            vec![],
            Arc::new(MulBackward { a: a.clone(), b: a.clone() }),
        ).unwrap();

        // c = b * a = 8
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![8.0]),
            vec![],
            Arc::new(MulBackward { a: b.clone(), b: a.clone() }),
        ).unwrap();

        // d = c * a = 16
        let d = Tensor::from_operation(
            TensorStorage::cpu(vec![16.0]),
            vec![],
            Arc::new(MulBackward { a: c.clone(), b: a.clone() }),
        ).unwrap();

        d.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        assert!(
            (a_grad.item().unwrap() - 32.0).abs() < 1e-4,
            "expected de/da = 32.0, got {}",
            a_grad.item().unwrap()
        );
    }

    #[test]
    fn test_backward_no_requires_grad_input() {
        // b does not require grad. Only a should get a gradient.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, false); // no grad

        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![5.0]),
            vec![],
            Arc::new(AddBackward { a: a.clone(), b: b.clone() }),
        ).unwrap();

        c.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        assert!((a_grad.item().unwrap() - 1.0).abs() < 1e-6);
        assert!(b.grad().unwrap().is_none(), "b should have no gradient");
    }

    #[test]
    fn test_backward_gradient_shape_mismatch_error() {
        // External gradient with wrong shape should error.
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
            vec![3],
            false,
        ).unwrap();

        let bad_grad = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0]),
            vec![2],
            false,
        ).unwrap();

        let result = a.backward_with_gradient(&bad_grad);
        assert!(result.is_err(), "should error on shape mismatch");
    }

    #[test]
    fn test_tensor_id_ordering() {
        // Verify that TensorIds are monotonically increasing, which is
        // the invariant that makes priority-queue scheduling correct.
        let t1 = leaf_scalar(1.0, false);
        let t2 = leaf_scalar(2.0, false);
        let t3 = leaf_scalar(3.0, false);
        assert!(t1.id() < t2.id());
        assert!(t2.id() < t3.id());
    }
}
