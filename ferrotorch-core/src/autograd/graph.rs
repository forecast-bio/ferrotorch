use std::collections::VecDeque;
use rustc_hash::FxHashMap as HashMap;

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
                if !node_map.contains_key(&input_id) {
                    node_map.insert(input_id, input);
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
            let input_grads = grad_fn.backward(&grad_output)?;
            let inputs = grad_fn.inputs();

            for (input, maybe_grad) in inputs.iter().zip(input_grads.into_iter()) {
                if let Some(grad) = maybe_grad {
                    if input.requires_grad() {
                        if input.is_leaf() {
                            // Leaf tensor: accumulate gradient on the tensor itself.
                            input.accumulate_grad(&grad)?;
                        } else {
                            // Non-leaf: accumulate into the grads map for the next iteration.
                            if let Some(existing) = grads.remove(&input.id()) {
                                // GPU-aware in-place addition.
                                let device = existing.device();
                                let existing_cpu = if existing.is_cuda() { existing.cpu()? } else { existing };
                                let grad_cpu = if grad.is_cuda() { grad.cpu()? } else { grad };
                                let mut existing_data = existing_cpu.data()?.to_vec();
                                let grad_data = grad_cpu.data()?;
                                for (e, &g) in existing_data.iter_mut().zip(grad_data.iter()) {
                                    *e = *e + g;
                                }
                                let storage = crate::storage::TensorStorage::cpu(existing_data);
                                let combined = Tensor::from_storage(
                                    storage,
                                    existing_cpu.shape().to_vec(),
                                    false,
                                )?;
                                grads.insert(input.id(), combined.to(device)?);
                            } else {
                                grads.insert(input.id(), grad);
                            }
                        }
                    }
                }
            }
        }
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
}
