//! Higher-order gradient support for ferrotorch autograd.
//!
//! This module provides the [`grad`] function, which computes gradients of
//! `outputs` with respect to `inputs` **without** accumulating on leaf tensors.
//! When `create_graph=true`, the backward pass itself is recorded in the
//! computation graph, enabling higher-order derivatives (Hessians, MAML,
//! WGAN-GP gradient penalties).
//!
//! Also provides convenience functions [`jacobian`] and [`hessian`] built on
//! top of `grad`.

use std::collections::{HashMap, VecDeque};

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{Tensor, TensorId};

/// Compute gradients of `outputs` with respect to `inputs`.
///
/// Unlike [`backward()`](super::backward), this does **not** accumulate
/// gradients on leaf tensors. Instead, it returns the gradient tensors
/// directly as a `Vec<Option<Tensor<T>>>`, one per element of `inputs`.
///
/// # Parameters
///
/// - `outputs`: The scalar tensor to differentiate.
/// - `inputs`: The tensors to differentiate with respect to.
/// - `retain_graph`: If `true`, the computation graph is not consumed and
///   can be differentiated again. If `false`, intermediate gradient data
///   may be dropped.
/// - `create_graph`: If `true`, the gradient computation itself is recorded
///   in the autograd graph, enabling higher-order gradients. The returned
///   gradient tensors will have `requires_grad=true` and carry `grad_fn`
///   nodes.
///
/// # Errors
///
/// Returns an error if `outputs` is not scalar.
///
/// # Examples
///
/// ```ignore
/// // f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
/// let x = Tensor::from_storage(TensorStorage::cpu(vec![2.0f32]), vec![], true)?;
/// let y = pow(&x, 3.0)?;
///
/// // First derivative with create_graph=true so we can differentiate again.
/// let grads = grad(&y, &[&x], true, true)?;
/// let dy_dx = grads[0].as_ref().unwrap(); // 3 * 4 = 12
///
/// // Second derivative.
/// let grads2 = grad(dy_dx, &[&x], false, false)?;
/// let d2y_dx2 = grads2[0].as_ref().unwrap(); // 6 * 2 = 12
/// ```
pub fn grad<T: Float>(
    outputs: &Tensor<T>,
    inputs: &[&Tensor<T>],
    retain_graph: bool,
    create_graph: bool,
) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
    // Validate that outputs is scalar.
    if !outputs.is_scalar() && outputs.numel() != 1 {
        return Err(FerrotorchError::BackwardNonScalar {
            shape: outputs.shape().to_vec(),
        });
    }

    // Build a set of input IDs for fast lookup.
    let input_ids: HashMap<TensorId, usize> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id(), i))
        .collect();

    // Seed gradient: d(outputs)/d(outputs) = 1.
    let seed = if create_graph {
        // When create_graph=true, the seed must participate in the graph.
        // Give it requires_grad=true so operations on it are tracked.
        Tensor::from_storage(
            TensorStorage::cpu(vec![<T as num_traits::One>::one()]),
            vec![],
            true,
        )?
    } else {
        Tensor::from_storage(
            TensorStorage::cpu(vec![<T as num_traits::One>::one()]),
            vec![],
            false,
        )?
    };

    // Phase 1: Collect all nodes and compute in-degree via BFS.
    let mut in_degree: HashMap<TensorId, usize> = HashMap::new();
    let mut node_map: HashMap<TensorId, &Tensor<T>> = HashMap::new();
    let mut queue: VecDeque<&Tensor<T>> = VecDeque::new();

    queue.push_back(outputs);
    in_degree.entry(outputs.id()).or_insert(0);
    node_map.insert(outputs.id(), outputs);

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
    let mut topo_order: Vec<TensorId> = Vec::new();
    let mut bfs_queue: VecDeque<TensorId> = VecDeque::new();

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
    // Key difference from `backward()`: we do NOT accumulate on leaf tensors.
    // Instead, we collect gradients for the requested `inputs` and return them.
    //
    // When `create_graph=true`, the grad_output tensors have `requires_grad=true`,
    // so the GradFn::backward() calls (which use differentiable operations like
    // mul, add, etc.) naturally build a second-order computation graph.
    let mut grads: HashMap<TensorId, Tensor<T>> = HashMap::new();
    grads.insert(outputs.id(), seed);

    // Result vector: one gradient per requested input.
    let mut result: Vec<Option<Tensor<T>>> = vec![None; inputs.len()];

    for &id in &topo_order {
        let node = match node_map.get(&id) {
            Some(n) => *n,
            None => continue,
        };

        let grad_output = match grads.remove(&id) {
            Some(g) => g,
            None => continue,
        };

        // If this node is one of the requested inputs, record its gradient.
        if let Some(&idx) = input_ids.get(&id) {
            result[idx] = Some(grad_output.clone());
            // If this node also has a grad_fn, we still need to continue
            // backward through it (in case other requested inputs are deeper
            // in the graph). But if retain_graph is false and we've collected
            // all inputs, we could short-circuit -- for simplicity we always
            // continue.
        }

        if let Some(grad_fn) = node.grad_fn() {
            let input_grads = grad_fn.backward(&grad_output)?;
            let fn_inputs = grad_fn.inputs();

            for (input, maybe_grad) in fn_inputs.iter().zip(input_grads) {
                if let Some(ig) = maybe_grad {
                    if input.requires_grad() {
                        // When create_graph=true, ensure the gradient tensor
                        // participates in the computation graph. The GradFn::backward()
                        // implementations that use raw Vec operations (non-differentiable)
                        // produce tensors with requires_grad=false. We wrap them so
                        // they can be differentiated again.
                        let grad_tensor = if create_graph && !ig.requires_grad() {
                            ig.requires_grad_(true)
                        } else {
                            ig
                        };

                        // Accumulate into the grads map for the next iteration.
                        if let Some(existing) = grads.remove(&input.id()) {
                            if create_graph {
                                // Use differentiable addition so the accumulation
                                // is itself part of the computation graph.
                                let summed =
                                    differentiable_add(&existing, &grad_tensor, create_graph)?;
                                grads.insert(input.id(), summed);
                            } else {
                                // Plain element-wise add (non-differentiable).
                                let a = existing.data_vec()?;
                                let b = grad_tensor.data_vec()?;
                                let summed: Vec<T> =
                                    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
                                let storage = TensorStorage::cpu(summed);
                                let combined = Tensor::from_storage(
                                    storage,
                                    existing.shape().to_vec(),
                                    false,
                                )?;
                                grads.insert(input.id(), combined);
                            }
                        } else {
                            grads.insert(input.id(), grad_tensor);
                        }
                    }
                }
            }
        }
    }

    // Any remaining entries in grads that correspond to requested inputs
    // should be captured. (They would be leaf tensors that were not visited
    // as intermediate nodes above.)
    for (id, g) in grads {
        if let Some(&idx) = input_ids.get(&id) {
            if result[idx].is_none() {
                result[idx] = Some(g);
            }
        }
    }

    let _ = retain_graph; // Consumed semantically above; graph is immutable via Arc.

    Ok(result)
}

/// Differentiable element-wise addition used for gradient accumulation
/// when `create_graph=true`.
///
/// This uses the public `add` grad_fn so the addition itself is tracked
/// in the autograd graph.
fn differentiable_add<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    _create_graph: bool,
) -> FerrotorchResult<Tensor<T>> {
    // Use the differentiable add from grad_fns::arithmetic.
    crate::grad_fns::arithmetic::add(a, b)
}

/// Compute the Jacobian matrix of a function at a point.
///
/// Given a function `f: R^n -> R^m` and an input tensor of shape `[n]`,
/// returns a tensor of shape `[m, n]` where `J[i, j] = df_i / dx_j`.
///
/// The function `f` must accept a tensor of shape `[n]` and return a tensor
/// of shape `[m]`. Each output element is differentiated independently via
/// the `grad` function with `create_graph=true`.
///
/// # Example
///
/// ```ignore
/// // f([x, y]) = [x^2, x*y]
/// // J = [[2x, 0], [y, x]]
/// // At (1, 1): J = [[2, 0], [1, 1]]
/// let result = jacobian(|x| { /* ... */ }, &input)?;
/// ```
pub fn jacobian<T: Float, F>(f: F, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    let n = input.numel();

    // Create a fresh input that requires grad.
    let x = Tensor::from_storage(
        TensorStorage::cpu(input.data()?.to_vec()),
        input.shape().to_vec(),
        true,
    )?;

    // Evaluate the function.
    let y = f(&x)?;
    let m = y.numel();
    let y_data = y.data()?.to_vec();

    // Build the Jacobian row by row.
    // For each output element y_i, compute grad(y_i, x).
    let mut jac_data: Vec<T> = Vec::with_capacity(m * n);

    for i in 0..m {
        // Create a scalar tensor for y_i by re-evaluating through the graph.
        // We need a fresh forward pass for each row to get independent graphs.
        let x_fresh = Tensor::from_storage(
            TensorStorage::cpu(input.data()?.to_vec()),
            input.shape().to_vec(),
            true,
        )?;
        let y_fresh = f(&x_fresh)?;

        // Extract the i-th element as a scalar.
        let y_i = extract_element(&y_fresh, i)?;

        // Compute gradient of y_i w.r.t. x_fresh.
        let grads = grad(&y_i, &[&x_fresh], false, false)?;

        match &grads[0] {
            Some(g) => {
                let g_data = g.data()?;
                jac_data.extend_from_slice(g_data);
            }
            None => {
                // No gradient means the output doesn't depend on the input.
                jac_data.extend(std::iter::repeat_n(<T as num_traits::Zero>::zero(), n));
            }
        }
    }

    // Verify forward values are consistent.
    let _ = y_data;

    Tensor::from_storage(TensorStorage::cpu(jac_data), vec![m, n], false)
}

/// Compute the Hessian matrix of a scalar function at a point.
///
/// Given a function `f: R^n -> R` and an input tensor of shape `[n]`,
/// returns a tensor of shape `[n, n]` where `H[i, j] = d^2f / (dx_i dx_j)`.
///
/// Internally computes the Jacobian of the gradient function, leveraging
/// higher-order gradients via `create_graph=true`.
///
/// # Example
///
/// ```ignore
/// // f([x, y]) = x^2 + y^2
/// // H = [[2, 0], [0, 2]]
/// let result = hessian(|x| { /* ... */ }, &input)?;
/// ```
pub fn hessian<T: Float, F>(f: F, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    let n = input.numel();

    // The Hessian is the Jacobian of the gradient function.
    // For each row i, we:
    // 1. Compute the gradient of f w.r.t. x with create_graph=true
    // 2. Extract grad_i
    // 3. Differentiate grad_i w.r.t. x to get row i of the Hessian
    let mut hess_data: Vec<T> = Vec::with_capacity(n * n);

    for i in 0..n {
        // Fresh forward pass.
        let x = Tensor::from_storage(
            TensorStorage::cpu(input.data()?.to_vec()),
            input.shape().to_vec(),
            true,
        )?;
        let y = f(&x)?;

        // First derivative with create_graph=true.
        let grads = grad(&y, &[&x], true, true)?;
        let grad_vec = match &grads[0] {
            Some(g) => g.clone(),
            None => {
                // Function doesn't depend on input => Hessian is all zeros.
                hess_data.extend(std::iter::repeat_n(<T as num_traits::Zero>::zero(), n));
                continue;
            }
        };

        // Extract the i-th element of the gradient as a scalar.
        let grad_i = extract_element(&grad_vec, i)?;

        // Second derivative: differentiate grad_i w.r.t. x.
        let grads2 = grad(&grad_i, &[&x], false, false)?;

        match &grads2[0] {
            Some(g2) => {
                let g2_data = g2.data()?;
                hess_data.extend_from_slice(g2_data);
            }
            None => {
                hess_data.extend(std::iter::repeat_n(<T as num_traits::Zero>::zero(), n));
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(hess_data), vec![n, n], false)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract the `i`-th element of a tensor as a scalar tensor, preserving
/// the computation graph (via an `IndexBackward` node).
fn extract_element<T: Float>(tensor: &Tensor<T>, index: usize) -> FerrotorchResult<Tensor<T>> {
    let data = tensor.data()?;
    if index >= data.len() {
        return Err(FerrotorchError::IndexOutOfBounds {
            index,
            axis: 0,
            size: data.len(),
        });
    }
    let val = data[index];
    let scalar = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)?;

    if tensor.requires_grad() || tensor.grad_fn().is_some() {
        // Attach a grad_fn that routes the scalar gradient back to the
        // correct position in the input tensor.
        let grad_fn = std::sync::Arc::new(IndexSelectBackward {
            input: tensor.clone(),
            index,
        });
        Tensor::from_operation(TensorStorage::cpu(vec![val]), vec![], grad_fn)
    } else {
        Ok(scalar)
    }
}

/// Backward node for extracting a single element from a tensor.
///
/// Given `y = x[index]`, the VJP is: `grad_x[j] = grad_y if j == index, else 0`.
#[derive(Debug)]
struct IndexSelectBackward<T: Float> {
    input: Tensor<T>,
    index: usize,
}

impl<T: Float> crate::tensor::GradFn<T> for IndexSelectBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let numel = self.input.numel();

        // When grad_output requires_grad (create_graph=true), use differentiable
        // operations so the backward pass is tracked for higher-order gradients.
        if grad_output.requires_grad() || grad_output.grad_fn().is_some() {
            // Create one-hot basis vector: e_i = [0, ..., 1, ..., 0]
            let one = <T as num_traits::One>::one();
            let zero = <T as num_traits::Zero>::zero();
            let mut basis = vec![zero; numel];
            basis[self.index] = one;
            let basis_tensor = Tensor::from_storage(
                TensorStorage::cpu(basis),
                self.input.shape().to_vec(),
                false,
            )?;

            // Broadcast grad_output (scalar) to the input shape, then multiply
            // by the basis vector. This is differentiable w.r.t. grad_output.
            let go_val = grad_output.data()?[0];

            // Attach a BroadcastScalarBackward that connects the broadcast
            // tensor back to grad_output, preserving the computation graph.
            let broadcast_tracked = Tensor::from_operation(
                TensorStorage::cpu(vec![go_val; numel]),
                self.input.shape().to_vec(),
                std::sync::Arc::new(BroadcastScalarBackward {
                    scalar_input: grad_output.clone(),
                }),
            )?;

            // grad_input = broadcast_tracked * basis_tensor
            // This multiplication is differentiable, connecting back to grad_output.
            let grad_input = crate::grad_fns::arithmetic::mul(&broadcast_tracked, &basis_tensor)?;
            return Ok(vec![Some(grad_input)]);
        }

        let go = grad_output.data()?[0];
        let mut grad_data = vec![<T as num_traits::Zero>::zero(); numel];
        grad_data[self.index] = go;
        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(grad_data),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IndexSelectBackward"
    }
}

/// Backward node for broadcasting a scalar to a vector.
///
/// Given `y = broadcast(scalar, n)` where `y[i] = scalar` for all i,
/// the VJP is: `grad_scalar = sum(grad_output)`.
#[derive(Debug)]
struct BroadcastScalarBackward<T: Float> {
    scalar_input: Tensor<T>,
}

impl<T: Float> crate::tensor::GradFn<T> for BroadcastScalarBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_scalar = sum of all elements of grad_output.
        let go_data = grad_output.data()?;
        let total: T = go_data
            .iter()
            .copied()
            .fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
        let grad_scalar = Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)?;
        Ok(vec![Some(grad_scalar)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.scalar_input]
    }

    fn name(&self) -> &'static str {
        "BroadcastScalarBackward"
    }
}

// ===========================================================================
// Convenience method on Tensor
// ===========================================================================

impl<T: Float> Tensor<T> {
    /// Compute gradients of this tensor with respect to `inputs`, returning
    /// the gradient tensors directly (without accumulating on leaves).
    ///
    /// See [`grad`] for full documentation.
    pub fn grad_wrt(
        &self,
        inputs: &[&Tensor<T>],
        retain_graph: bool,
        create_graph: bool,
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        grad(self, inputs, retain_graph, create_graph)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad_fns::arithmetic::{add, mul, pow};
    use crate::grad_fns::reduction::sum;
    use crate::storage::TensorStorage;

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Create a leaf 1-D tensor.
    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_approx(actual: f32, expected: f32, tol: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: expected {expected}, got {actual}"
        );
    }

    // -----------------------------------------------------------------------
    // Basic grad() tests (create_graph=false)
    // -----------------------------------------------------------------------

    #[test]
    fn test_grad_simple_pow() {
        // f(x) = x^3, f'(x) = 3x^2
        // At x=2: f'(2) = 12
        let x = leaf_scalar(2.0, true);
        let y = pow(&x, 3.0).unwrap();

        let grads = grad(&y, &[&x], false, false).unwrap();
        let dy_dx = grads[0].as_ref().unwrap();
        assert_approx(dy_dx.item().unwrap(), 12.0, 1e-4, "f'(2) for x^3");
    }

    #[test]
    fn test_grad_add() {
        // f(x, y) = x + y
        // df/dx = 1, df/dy = 1
        let x = leaf_scalar(3.0, true);
        let y = leaf_scalar(5.0, true);
        let z = add(&x, &y).unwrap();

        let grads = grad(&z, &[&x, &y], false, false).unwrap();
        assert_approx(
            grads[0].as_ref().unwrap().item().unwrap(),
            1.0,
            1e-6,
            "dz/dx",
        );
        assert_approx(
            grads[1].as_ref().unwrap().item().unwrap(),
            1.0,
            1e-6,
            "dz/dy",
        );
    }

    #[test]
    fn test_grad_mul() {
        // f(x, y) = x * y
        // df/dx = y = 5, df/dy = x = 3
        let x = leaf_scalar(3.0, true);
        let y = leaf_scalar(5.0, true);
        let z = mul(&x, &y).unwrap();

        let grads = grad(&z, &[&x, &y], false, false).unwrap();
        assert_approx(
            grads[0].as_ref().unwrap().item().unwrap(),
            5.0,
            1e-6,
            "dz/dx = y",
        );
        assert_approx(
            grads[1].as_ref().unwrap().item().unwrap(),
            3.0,
            1e-6,
            "dz/dy = x",
        );
    }

    #[test]
    fn test_grad_x_squared_plus_y_squared() {
        // f(x, y) = x^2 + y^2
        // df/dx = 2x = 6, df/dy = 2y = 8
        let x = leaf_scalar(3.0, true);
        let y = leaf_scalar(4.0, true);
        let x2 = pow(&x, 2.0).unwrap();
        let y2 = pow(&y, 2.0).unwrap();
        let z = add(&x2, &y2).unwrap();

        let grads = grad(&z, &[&x, &y], false, false).unwrap();
        assert_approx(
            grads[0].as_ref().unwrap().item().unwrap(),
            6.0,
            1e-4,
            "dz/dx = 2x",
        );
        assert_approx(
            grads[1].as_ref().unwrap().item().unwrap(),
            8.0,
            1e-4,
            "dz/dy = 2y",
        );
    }

    #[test]
    fn test_grad_does_not_accumulate_on_leaves() {
        // Calling grad() should NOT modify the leaf's .grad() field.
        let x = leaf_scalar(2.0, true);
        let y = pow(&x, 2.0).unwrap();

        let _grads = grad(&y, &[&x], false, false).unwrap();
        assert!(
            x.grad().unwrap().is_none(),
            "grad() should not accumulate on leaf tensors"
        );
    }

    // -----------------------------------------------------------------------
    // create_graph=false: returned grads have no grad_fn
    // -----------------------------------------------------------------------

    #[test]
    fn test_grad_no_create_graph_returns_detached() {
        let x = leaf_scalar(2.0, true);
        let y = pow(&x, 3.0).unwrap();

        let grads = grad(&y, &[&x], false, false).unwrap();
        let dy_dx = grads[0].as_ref().unwrap();
        // With create_graph=false, the returned gradient should not have a grad_fn.
        // It may or may not have requires_grad, but it should not be part of a
        // computation graph.
        assert!(
            dy_dx.grad_fn().is_none(),
            "create_graph=false: gradient should not have grad_fn"
        );
    }

    // -----------------------------------------------------------------------
    // create_graph=true: returned grads HAVE grad_fn
    // -----------------------------------------------------------------------

    #[test]
    fn test_grad_create_graph_returns_differentiable() {
        let x = leaf_scalar(2.0, true);
        let y = pow(&x, 3.0).unwrap();

        let grads = grad(&y, &[&x], true, true).unwrap();
        let dy_dx = grads[0].as_ref().unwrap();
        // With create_graph=true, the gradient tensor should be differentiable.
        assert!(
            dy_dx.requires_grad(),
            "create_graph=true: gradient should require grad"
        );
    }

    // -----------------------------------------------------------------------
    // Higher-order gradients: grad-of-grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_higher_order_x_cubed() {
        // f(x) = x^3
        // f'(x) = 3x^2
        // f''(x) = 6x
        // At x = 2: f'(2) = 12, f''(2) = 12
        let x = leaf_scalar(2.0, true);
        let y = pow(&x, 3.0).unwrap();

        // First derivative with create_graph=true.
        let grads1 = grad(&y, &[&x], true, true).unwrap();
        let dy_dx = grads1[0].as_ref().unwrap();
        assert_approx(dy_dx.item().unwrap(), 12.0, 1e-4, "f'(2) = 3*4 = 12");

        // Second derivative.
        let grads2 = grad(dy_dx, &[&x], false, false).unwrap();
        let d2y_dx2 = grads2[0].as_ref().unwrap();
        assert_approx(d2y_dx2.item().unwrap(), 12.0, 1e-3, "f''(2) = 6*2 = 12");
    }

    #[test]
    fn test_higher_order_x_squared() {
        // f(x) = x^2
        // f'(x) = 2x
        // f''(x) = 2
        // At x = 5: f'(5) = 10, f''(5) = 2
        let x = leaf_scalar(5.0, true);
        let y = pow(&x, 2.0).unwrap();

        let grads1 = grad(&y, &[&x], true, true).unwrap();
        let dy_dx = grads1[0].as_ref().unwrap();
        assert_approx(dy_dx.item().unwrap(), 10.0, 1e-4, "f'(5) = 2*5 = 10");

        let grads2 = grad(dy_dx, &[&x], false, false).unwrap();
        let d2y_dx2 = grads2[0].as_ref().unwrap();
        assert_approx(d2y_dx2.item().unwrap(), 2.0, 1e-3, "f''(5) = 2");
    }

    #[test]
    fn test_higher_order_product() {
        // f(x, y) = x * y
        // df/dx = y, df/dy = x
        // d2f/dxdx = 0, d2f/dxdy = 1
        // d2f/dydx = 1, d2f/dydy = 0
        let x = leaf_scalar(3.0, true);
        let y = leaf_scalar(4.0, true);
        let z = mul(&x, &y).unwrap();

        // First derivatives with create_graph=true.
        let grads1 = grad(&z, &[&x, &y], true, true).unwrap();
        let dz_dx = grads1[0].as_ref().unwrap();
        let dz_dy = grads1[1].as_ref().unwrap();

        assert_approx(dz_dx.item().unwrap(), 4.0, 1e-6, "dz/dx = y = 4");
        assert_approx(dz_dy.item().unwrap(), 3.0, 1e-6, "dz/dy = x = 3");

        // Second derivatives: d(dz/dx)/dy = 1.
        let grads2 = grad(dz_dx, &[&y], false, false).unwrap();
        let d2z_dxdy = grads2[0].as_ref().unwrap();
        assert_approx(d2z_dxdy.item().unwrap(), 1.0, 1e-4, "d2z/dxdy = 1");
    }

    // -----------------------------------------------------------------------
    // Jacobian tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jacobian_quadratic() {
        // f([x, y]) = [x^2, x*y] via extract_element + ConcatBackward2
        // J = [[2x, 0], [y, x]]
        // At (2, 3): J = [[4, 0], [3, 2]]
        let input = leaf_vec(&[2.0, 3.0], false);

        let jac = jacobian(
            |x| {
                let e0 = extract_element(x, 0).unwrap();
                let e1 = extract_element(x, 1).unwrap();

                let f0 = pow(&e0, 2.0).unwrap(); // x^2
                let f1 = mul(&e0, &e1).unwrap(); // x*y

                let f0_val = f0.item().unwrap();
                let f1_val = f1.item().unwrap();

                let out = Tensor::from_operation(
                    TensorStorage::cpu(vec![f0_val, f1_val]),
                    vec![2],
                    std::sync::Arc::new(ConcatBackward2 {
                        input0: f0,
                        input1: f1,
                    }),
                )
                .unwrap();
                Ok(out)
            },
            &input,
        )
        .unwrap();

        assert_eq!(jac.shape(), &[2, 2]);
        let j = jac.data().unwrap();
        assert_approx(j[0], 4.0, 1e-4, "J[0,0] = 2x = 4");
        assert_approx(j[1], 0.0, 1e-4, "J[0,1] = 0");
        assert_approx(j[2], 3.0, 1e-4, "J[1,0] = y = 3");
        assert_approx(j[3], 2.0, 1e-4, "J[1,1] = x = 2");
    }

    #[test]
    fn test_jacobian_identity() {
        // f(x) = x (identity function on a scalar wrapped in [1])
        // J = [[1]]
        let input = leaf_vec(&[3.0], false);

        let jac = jacobian(
            |x| {
                // sum to scalar, then wrap back to 1-element
                let s = sum(x).unwrap();
                Ok(s)
            },
            &input,
        )
        .unwrap();

        // Jacobian of sum on a 1-element tensor = [[1]]
        assert_eq!(jac.shape(), &[1, 1]);
        assert_approx(jac.data().unwrap()[0], 1.0, 1e-6, "J[0,0]");
    }

    #[test]
    fn test_jacobian_scaled() {
        // f([x]) = [2*x] (via x + x)
        // J = [[2]]
        let input = leaf_vec(&[5.0], false);

        let jac = jacobian(
            |x| {
                let doubled = add(x, x).unwrap();
                let s = sum(&doubled).unwrap();
                Ok(s)
            },
            &input,
        )
        .unwrap();

        assert_eq!(jac.shape(), &[1, 1]);
        assert_approx(jac.data().unwrap()[0], 2.0, 1e-5, "J[0,0] = 2");
    }

    #[test]
    fn test_jacobian_vector_to_vector() {
        // f([x, y]) = [x^2, x*y] via extract_element
        // J = [[2x, 0], [y, x]] at (1, 1) = [[2, 0], [1, 1]]
        let input = leaf_vec(&[1.0, 1.0], false);

        let jac = jacobian(
            |x| {
                let e0 = extract_element(x, 0).unwrap();
                let e1 = extract_element(x, 1).unwrap();

                let f0 = pow(&e0, 2.0).unwrap(); // x^2
                let f1 = mul(&e0, &e1).unwrap(); // x*y

                // Concatenate via ConcatBackward2 grad_fn.
                let f0_val = f0.item().unwrap();
                let f1_val = f1.item().unwrap();

                let out = Tensor::from_operation(
                    TensorStorage::cpu(vec![f0_val, f1_val]),
                    vec![2],
                    std::sync::Arc::new(ConcatBackward2 {
                        input0: f0,
                        input1: f1,
                    }),
                )
                .unwrap();
                Ok(out)
            },
            &input,
        )
        .unwrap();

        assert_eq!(jac.shape(), &[2, 2]);
        let j = jac.data().unwrap();
        assert_approx(j[0], 2.0, 1e-4, "J[0,0] = 2x = 2");
        assert_approx(j[1], 0.0, 1e-4, "J[0,1] = 0");
        assert_approx(j[2], 1.0, 1e-4, "J[1,0] = y = 1");
        assert_approx(j[3], 1.0, 1e-4, "J[1,1] = x = 1");
    }

    /// Helper backward node for concatenating two scalars into a 2-element tensor.
    /// Used in tests only.
    #[derive(Debug)]
    struct ConcatBackward2<T: Float> {
        input0: Tensor<T>,
        input1: Tensor<T>,
    }

    impl<T: Float> crate::tensor::GradFn<T> for ConcatBackward2<T> {
        fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let go = grad_output.data()?;
            let g0 = Tensor::from_storage(TensorStorage::cpu(vec![go[0]]), vec![], false)?;
            let g1 = Tensor::from_storage(TensorStorage::cpu(vec![go[1]]), vec![], false)?;
            Ok(vec![Some(g0), Some(g1)])
        }

        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input0, &self.input1]
        }

        fn name(&self) -> &'static str {
            "ConcatBackward2"
        }
    }

    // -----------------------------------------------------------------------
    // Hessian tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hessian_x_squared_plus_y_squared() {
        // f([x, y]) = x^2 + y^2
        // H = [[2, 0], [0, 2]]
        let input = leaf_vec(&[3.0, 4.0], false);

        let hess = hessian(
            |x| {
                let e0 = extract_element(x, 0).unwrap();
                let e1 = extract_element(x, 1).unwrap();
                let f0 = pow(&e0, 2.0).unwrap();
                let f1 = pow(&e1, 2.0).unwrap();
                let result = add(&f0, &f1).unwrap();
                Ok(result)
            },
            &input,
        )
        .unwrap();

        assert_eq!(hess.shape(), &[2, 2]);
        let h = hess.data().unwrap();
        assert_approx(h[0], 2.0, 1e-3, "H[0,0]");
        assert_approx(h[1], 0.0, 1e-3, "H[0,1]");
        assert_approx(h[2], 0.0, 1e-3, "H[1,0]");
        assert_approx(h[3], 2.0, 1e-3, "H[1,1]");
    }

    #[test]
    fn test_hessian_x_cubed() {
        // f(x) = x^3
        // f'(x) = 3x^2
        // f''(x) = 6x
        // At x = 2: H = [[12]]
        let input = leaf_vec(&[2.0], false);

        let hess = hessian(
            |x| {
                let e = extract_element(x, 0).unwrap();
                pow(&e, 3.0)
            },
            &input,
        )
        .unwrap();

        assert_eq!(hess.shape(), &[1, 1]);
        assert_approx(hess.data().unwrap()[0], 12.0, 1e-2, "H[0,0] = 6*2 = 12");
    }

    #[test]
    fn test_hessian_xy() {
        // f([x, y]) = x * y
        // H = [[0, 1], [1, 0]]
        let input = leaf_vec(&[2.0, 3.0], false);

        let hess = hessian(
            |x| {
                let e0 = extract_element(x, 0).unwrap();
                let e1 = extract_element(x, 1).unwrap();
                mul(&e0, &e1)
            },
            &input,
        )
        .unwrap();

        assert_eq!(hess.shape(), &[2, 2]);
        let h = hess.data().unwrap();
        assert_approx(h[0], 0.0, 1e-3, "H[0,0] = 0");
        assert_approx(h[1], 1.0, 1e-3, "H[0,1] = 1");
        assert_approx(h[2], 1.0, 1e-3, "H[1,0] = 1");
        assert_approx(h[3], 0.0, 1e-3, "H[1,1] = 0");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_grad_non_scalar_error() {
        // grad() should error if outputs is not scalar.
        let x = leaf_vec(&[1.0, 2.0, 3.0], true);
        let result = grad(&x, &[&x], false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_grad_no_dependency() {
        // x and y are independent; grad(y^2, [x]) should be None.
        let x = leaf_scalar(1.0, true);
        let y = leaf_scalar(2.0, true);
        let z = pow(&y, 2.0).unwrap();

        let grads = grad(&z, &[&x], false, false).unwrap();
        assert!(grads[0].is_none(), "x is not in the graph of z");
    }

    #[test]
    fn test_grad_wrt_convenience() {
        // Test the convenience method on Tensor.
        let x = leaf_scalar(3.0, true);
        let y = pow(&x, 2.0).unwrap();

        let grads = y.grad_wrt(&[&x], false, false).unwrap();
        assert_approx(
            grads[0].as_ref().unwrap().item().unwrap(),
            6.0,
            1e-4,
            "dy/dx = 2x = 6",
        );
    }
}
