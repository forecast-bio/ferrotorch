//! Symbolic shapes with guards for dynamic batch sizes.
//!
//! The core JIT trace captures concrete input shapes: if you trace a
//! model with `batch=4` and then call it with `batch=8`, any operation
//! that hard-coded the batch dimension (like a `Reshape` to `[4, 10]`)
//! will fail. [`SymbolicTracedModule`] makes the trace polymorphic over
//! a declared set of symbolic dimensions — typically the batch axis —
//! and emits a runtime guard that verifies the concrete input shapes
//! match the signature before executing the graph.
//!
//! # High-level API
//!
//! ```ignore
//! use ferrotorch_jit::{compile_symbolic, ShapeSignature};
//!
//! // Trace with batch=4, declare dim 0 of input 0 as symbolic.
//! let example = ferrotorch_core::from_vec(vec![0.0f32; 4 * 10], &[4, 10])?;
//! let sig = ShapeSignature::new().symbolic_dim(0, 0);  // input 0, dim 0
//! let compiled = compile_symbolic(
//!     |inputs| some_forward(&inputs[0]),
//!     &[example],
//!     sig,
//! )?;
//!
//! // Now call with any batch size; the guard validates the shape.
//! let big = ferrotorch_core::from_vec(vec![0.0f32; 16 * 10], &[16, 10])?;
//! let out = compiled.forward_symbolic(&[big])?;
//! ```
//!
//! # Guards
//!
//! On every forward call, the input shapes are checked before
//! interpretation:
//!
//! 1. The number of inputs matches.
//! 2. Each input's rank matches the trace-time rank.
//! 3. Each **concrete** (non-symbolic) dim matches the trace-time value.
//! 4. Symbolic dims are free to vary, subject to an optional
//!    `[min, max]` range carried on each [`SymbolicDim`].
//!
//! A guard failure returns a descriptive `InvalidArgument` error.
//!
//! # Reshape patching
//!
//! When the traced graph contains a `Reshape` op whose target shape
//! literally equals the trace-time symbolic dim value, that would
//! hard-code `batch=4` into the reshape. To make the reshape
//! polymorphic over the symbolic dim, [`compile_symbolic`] rewrites
//! such occurrences to `-1` (the "infer this dim" sentinel that
//! `ferrotorch_core::grad_fns::shape::reshape` already understands).
//! This only fires when there is exactly one symbolic position in the
//! reshape target — multiple ambiguous symbolic positions are left as
//! the user wrote them, and a guard failure at runtime will surface
//! the shape conflict.
//!
//! CL-367.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use crate::graph::{IrGraph, IrOpKind};
use crate::interpreter::interpret;
use crate::module::TracedModule;
use crate::optimize::{OptimizationConfig, optimize};
use crate::trace::trace;

// ---------------------------------------------------------------------------
// ShapeSignature
// ---------------------------------------------------------------------------

/// Describes which dimensions of each graph input are symbolic
/// (allowed to vary at runtime).
///
/// Constructed incrementally with [`symbolic_dim`](Self::symbolic_dim)
/// and [`symbolic_dim_with_range`](Self::symbolic_dim_with_range).
/// All dims not declared symbolic are treated as concrete and must
/// match the trace-time value exactly.
#[derive(Debug, Clone, Default)]
pub struct ShapeSignature {
    /// `dims[i]` is the list of symbolic dim specs for input `i`.
    /// Missing inputs are all-concrete.
    entries: Vec<Vec<SymbolicDim>>,
}

/// A single symbolic dimension: its position and an optional
/// `[min, max]` inclusive range constraint enforced by the guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicDim {
    /// Dimension index within the input's shape (0 = outermost).
    pub dim_index: usize,
    /// Optional inclusive minimum. `None` = no lower bound (>= 1 still
    /// enforced because a zero-size dim is almost always a bug).
    pub min: Option<usize>,
    /// Optional inclusive maximum.
    pub max: Option<usize>,
}

impl ShapeSignature {
    /// Create an empty signature (no symbolic dims).
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Mark dimension `dim` of input `input_index` as symbolic with no
    /// range constraint.
    pub fn symbolic_dim(mut self, input_index: usize, dim: usize) -> Self {
        self.ensure_capacity(input_index);
        self.entries[input_index].push(SymbolicDim {
            dim_index: dim,
            min: None,
            max: None,
        });
        self
    }

    /// Mark dimension `dim` of input `input_index` as symbolic and
    /// constrain it to `[min, max]` (inclusive).
    pub fn symbolic_dim_with_range(
        mut self,
        input_index: usize,
        dim: usize,
        min: usize,
        max: usize,
    ) -> Self {
        self.ensure_capacity(input_index);
        self.entries[input_index].push(SymbolicDim {
            dim_index: dim,
            min: Some(min),
            max: Some(max),
        });
        self
    }

    /// Return the symbolic dim specs for input `input_index`, or an
    /// empty slice if the input has no symbolic dims.
    pub fn symbolic_dims_for(&self, input_index: usize) -> &[SymbolicDim] {
        self.entries
            .get(input_index)
            .map_or(&[], std::vec::Vec::as_slice)
    }

    /// Returns `true` if dim `dim` of input `input_index` is declared
    /// symbolic.
    pub fn is_symbolic(&self, input_index: usize, dim: usize) -> bool {
        self.symbolic_dims_for(input_index)
            .iter()
            .any(|sd| sd.dim_index == dim)
    }

    fn ensure_capacity(&mut self, input_index: usize) {
        while self.entries.len() <= input_index {
            self.entries.push(Vec::new());
        }
    }
}

// ---------------------------------------------------------------------------
// Guard
// ---------------------------------------------------------------------------

/// Guard that validates runtime input shapes against a signature.
///
/// Built by [`SymbolicTracedModule::new`] from the trace-time shapes
/// and the user-provided [`ShapeSignature`]. Each call to
/// [`SymbolicTracedModule::forward_symbolic`] invokes
/// [`Guard::check`] before executing the traced graph.
#[derive(Debug, Clone)]
pub struct Guard {
    /// Trace-time shape for each input, used as the baseline for
    /// concrete-dim validation.
    trace_shapes: Vec<Vec<usize>>,
    signature: ShapeSignature,
}

impl Guard {
    /// Create a new guard from trace-time shapes and a signature.
    pub fn new(trace_shapes: Vec<Vec<usize>>, signature: ShapeSignature) -> Self {
        Self {
            trace_shapes,
            signature,
        }
    }

    /// Validate that the given runtime input shapes satisfy the
    /// signature. Returns `Ok(())` on success, or an
    /// `InvalidArgument` error describing the first violation.
    pub fn check(&self, inputs: &[&[usize]]) -> FerrotorchResult<()> {
        if inputs.len() != self.trace_shapes.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "SymbolicTracedModule guard: expected {} inputs, got {}",
                    self.trace_shapes.len(),
                    inputs.len()
                ),
            });
        }

        for (i, (got, expected)) in inputs.iter().zip(self.trace_shapes.iter()).enumerate() {
            if got.len() != expected.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "SymbolicTracedModule guard: input {i} rank mismatch: \
                         expected {}, got {} (shape {:?} vs signature {:?})",
                        expected.len(),
                        got.len(),
                        got,
                        expected,
                    ),
                });
            }

            for (dim, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                let sym = self
                    .signature
                    .symbolic_dims_for(i)
                    .iter()
                    .find(|sd| sd.dim_index == dim);
                match sym {
                    Some(sd) => {
                        // Symbolic dim: check range.
                        if g == 0 {
                            return Err(FerrotorchError::InvalidArgument {
                                message: format!(
                                    "SymbolicTracedModule guard: input {i} dim {dim} \
                                     is symbolic but runtime value is 0"
                                ),
                            });
                        }
                        if let Some(min) = sd.min {
                            if g < min {
                                return Err(FerrotorchError::InvalidArgument {
                                    message: format!(
                                        "SymbolicTracedModule guard: input {i} dim {dim} = {g} \
                                         is below min {min}"
                                    ),
                                });
                            }
                        }
                        if let Some(max) = sd.max {
                            if g > max {
                                return Err(FerrotorchError::InvalidArgument {
                                    message: format!(
                                        "SymbolicTracedModule guard: input {i} dim {dim} = {g} \
                                         is above max {max}"
                                    ),
                                });
                            }
                        }
                    }
                    None => {
                        // Concrete dim: must match exactly.
                        if g != e {
                            return Err(FerrotorchError::InvalidArgument {
                                message: format!(
                                    "SymbolicTracedModule guard: input {i} dim {dim} \
                                     is concrete, expected {e}, got {g} (shape {got:?})"
                                ),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SymbolicTracedModule
// ---------------------------------------------------------------------------

/// A traced module whose forward pass is polymorphic over declared
/// symbolic dimensions, with runtime guards.
///
/// Construct via [`compile_symbolic`] or manually by wrapping an
/// existing [`TracedModule`] with [`SymbolicTracedModule::new`].
#[derive(Debug, Clone)]
pub struct SymbolicTracedModule<T: Float> {
    inner: TracedModule<T>,
    guard: Guard,
}

impl<T: Float> SymbolicTracedModule<T> {
    /// Wrap an existing [`TracedModule`] with a guard built from the
    /// given trace-time input shapes and signature.
    pub fn new(
        inner: TracedModule<T>,
        trace_shapes: Vec<Vec<usize>>,
        signature: ShapeSignature,
    ) -> Self {
        Self {
            inner,
            guard: Guard::new(trace_shapes, signature),
        }
    }

    /// Access the underlying traced module.
    pub fn inner(&self) -> &TracedModule<T> {
        &self.inner
    }

    /// Access the runtime guard for inspection or manual validation.
    pub fn guard(&self) -> &Guard {
        &self.guard
    }

    /// Run the underlying graph with the given inputs, after passing
    /// them through the guard.
    pub fn forward_symbolic(&self, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        let shape_refs: Vec<&[usize]> = inputs.iter().map(ferrotorch_core::Tensor::shape).collect();
        self.guard.check(&shape_refs)?;
        interpret(self.inner.graph(), inputs)
    }
}

// ---------------------------------------------------------------------------
// Reshape patching
// ---------------------------------------------------------------------------

/// Rewrite `Reshape` ops in the graph so any target dim that literally
/// equals a trace-time symbolic dim value is replaced with `-1` (the
/// "infer from numel" sentinel). Only applies the rewrite when the
/// reshape target has **exactly one** symbolic match; with zero or
/// multiple matches the op is left alone and the runtime guard will
/// handle the conflict.
pub(crate) fn patch_reshape_for_symbolic_dims(
    graph: &mut IrGraph,
    trace_shapes: &[Vec<usize>],
    signature: &ShapeSignature,
) {
    // Collect the symbolic dim values across all inputs, as a
    // deduplicated set. If only one symbolic value exists (the common
    // case for a single batch axis), we can unambiguously substitute
    // it with -1 in any reshape target that contains exactly one
    // occurrence of that value.
    let mut symbolic_values: Vec<usize> = Vec::new();
    for (i, shape) in trace_shapes.iter().enumerate() {
        for sd in signature.symbolic_dims_for(i) {
            if let Some(&v) = shape.get(sd.dim_index) {
                if !symbolic_values.contains(&v) {
                    symbolic_values.push(v);
                }
            }
        }
    }

    // Nothing to do if no symbolic dims are declared.
    if symbolic_values.is_empty() {
        return;
    }

    for node in &mut graph.nodes {
        if let IrOpKind::Reshape { shape } = &mut node.op {
            // Count how many positions in the reshape target match any
            // symbolic value.
            let match_positions: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| {
                    if d >= 0 && symbolic_values.contains(&(d as usize)) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            // Only patch when exactly one position matches — avoids
            // ambiguity when the user happens to reshape to a shape
            // that coincides with the symbolic value in multiple slots.
            if match_positions.len() == 1 {
                let pos = match_positions[0];
                // If there's already a -1 elsewhere we shouldn't add a
                // second one; reshape only supports a single -1.
                if !shape.contains(&-1) {
                    shape[pos] = -1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// compile_symbolic
// ---------------------------------------------------------------------------

/// Trace a function, build a symbolic traced module with the given
/// shape signature, and return it.
///
/// Equivalent to [`crate::compile`] followed by wrapping in a
/// [`SymbolicTracedModule`] and patching the graph's reshape ops for
/// the declared symbolic dims.
///
/// # Arguments
///
/// * `f` — The function to trace.
/// * `example_inputs` — Concrete example tensors used for one forward
///   pass. The shapes of these tensors become the trace-time shapes
///   enforced by the guard for non-symbolic dims.
/// * `signature` — Declares which input dims are symbolic and their
///   optional ranges.
pub fn compile_symbolic<T, F>(
    f: F,
    example_inputs: &[Tensor<T>],
    signature: ShapeSignature,
) -> FerrotorchResult<SymbolicTracedModule<T>>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    let trace_shapes: Vec<Vec<usize>> = example_inputs.iter().map(|t| t.shape().to_vec()).collect();

    let mut graph = trace(f, example_inputs)?;
    // Patch Reshape ops BEFORE running optimization passes so the
    // optimizer sees the polymorphic form and doesn't fold away the
    // symbolic values.
    patch_reshape_for_symbolic_dims(&mut graph, &trace_shapes, &signature);
    let _memory_plan = optimize(&mut graph, &OptimizationConfig::default());

    let traced = TracedModule::new(graph);
    Ok(SymbolicTracedModule::new(traced, trace_shapes, signature))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::from_vec;
    use ferrotorch_core::grad_fns::activation;

    #[test]
    fn test_shape_signature_symbolic_dim_flags() {
        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        assert!(sig.is_symbolic(0, 0));
        assert!(!sig.is_symbolic(0, 1));
        assert!(!sig.is_symbolic(1, 0));
        let dims = sig.symbolic_dims_for(0);
        assert_eq!(dims.len(), 1);
        assert_eq!(dims[0].dim_index, 0);
        assert_eq!(dims[0].min, None);
        assert_eq!(dims[0].max, None);
    }

    #[test]
    fn test_shape_signature_with_range() {
        let sig = ShapeSignature::new().symbolic_dim_with_range(0, 0, 1, 64);
        let dims = sig.symbolic_dims_for(0);
        assert_eq!(dims[0].min, Some(1));
        assert_eq!(dims[0].max, Some(64));
    }

    #[test]
    fn test_guard_passes_symbolic_batch() {
        let trace_shapes = vec![vec![4, 10]];
        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        let guard = Guard::new(trace_shapes, sig);
        // Different batch: fine.
        guard.check(&[&[16, 10]]).unwrap();
        guard.check(&[&[1, 10]]).unwrap();
    }

    #[test]
    fn test_guard_rejects_concrete_mismatch() {
        let trace_shapes = vec![vec![4, 10]];
        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        let guard = Guard::new(trace_shapes, sig);
        // dim 1 is concrete; mismatch should fail.
        let err = guard.check(&[&[16, 7]]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("dim 1"), "msg = {msg}");
        assert!(msg.contains("expected 10, got 7"), "msg = {msg}");
    }

    #[test]
    fn test_guard_rejects_rank_mismatch() {
        let trace_shapes = vec![vec![4, 10]];
        let sig = ShapeSignature::new();
        let guard = Guard::new(trace_shapes, sig);
        let err = guard.check(&[&[4, 10, 2]]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("rank mismatch"), "msg = {msg}");
    }

    #[test]
    fn test_guard_rejects_input_count_mismatch() {
        let trace_shapes = vec![vec![4, 10]];
        let sig = ShapeSignature::new();
        let guard = Guard::new(trace_shapes, sig);
        let err = guard.check(&[&[4, 10], &[4, 10]]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected 1 inputs, got 2"), "msg = {msg}");
    }

    #[test]
    fn test_guard_enforces_range() {
        let sig = ShapeSignature::new().symbolic_dim_with_range(0, 0, 2, 32);
        let guard = Guard::new(vec![vec![4, 10]], sig);
        // In range: fine.
        guard.check(&[&[8, 10]]).unwrap();
        guard.check(&[&[2, 10]]).unwrap();
        guard.check(&[&[32, 10]]).unwrap();
        // Below min.
        let err = guard.check(&[&[1, 10]]).unwrap_err();
        assert!(format!("{err}").contains("below min"));
        // Above max.
        let err = guard.check(&[&[33, 10]]).unwrap_err();
        assert!(format!("{err}").contains("above max"));
    }

    #[test]
    fn test_guard_rejects_zero_symbolic_dim() {
        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        let guard = Guard::new(vec![vec![4, 10]], sig);
        let err = guard.check(&[&[0, 10]]).unwrap_err();
        assert!(format!("{err}").contains("runtime value is 0"));
    }

    #[test]
    fn test_symbolic_module_relu_polymorphic_batch() {
        // Trace a single-input relu(x) with batch=4. Then run it
        // with batch=16 and verify the output matches a fresh eager
        // relu on the new input.
        let example = from_vec(
            (0..40).map(|i| (i as f32 - 20.0) * 0.1).collect::<Vec<_>>(),
            &[4, 10],
        )
        .unwrap()
        .requires_grad_(true);

        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        let compiled =
            compile_symbolic::<f32, _>(|inputs| activation::relu(&inputs[0]), &[example], sig)
                .unwrap();

        // Run with a bigger batch.
        let big: Tensor<f32> = from_vec(
            (0..160)
                .map(|i| (i as f32 - 80.0) * 0.05)
                .collect::<Vec<_>>(),
            &[16, 10],
        )
        .unwrap();
        let out = compiled
            .forward_symbolic(std::slice::from_ref(&big))
            .unwrap();
        assert_eq!(out.shape(), &[16, 10]);

        // Verify values match eager relu(x) on the big input.
        let data = out.data_vec().unwrap();
        let big_data = big.data_vec().unwrap();
        for (i, (&o, &x)) in data.iter().zip(big_data.iter()).enumerate() {
            let expected = x.max(0.0);
            assert!(
                (o - expected).abs() < 1e-6,
                "mismatch at {i}: got {o}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_symbolic_module_reshape_repatches_batch_end_to_end() {
        // Trace a graph that reshapes its input to its own concrete
        // shape (batch=4, features=10), then relu. Rerun with batch=9
        // and verify the reshape-patching makes it work.
        use ferrotorch_core::grad_fns::shape::reshape;

        let example = from_vec(
            (0..40).map(|i| (i as f32 - 20.0) * 0.1).collect::<Vec<_>>(),
            &[4, 10],
        )
        .unwrap()
        .requires_grad_(true);

        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        let compiled = compile_symbolic::<f32, _>(
            |inputs| {
                // Use the input's current concrete shape [4, 10] — the
                // patcher should rewrite this to [-1, 10] once it's
                // traced as a concrete Reshape op.
                let shape_isize: Vec<isize> =
                    inputs[0].shape().iter().map(|&d| d as isize).collect();
                let r = reshape(&inputs[0], &shape_isize)?;
                activation::relu(&r)
            },
            &[example],
            sig,
        )
        .unwrap();

        // A batch-9 input — impossible if the traced reshape was
        // [4, 10], possible if the patcher rewrote it to [-1, 10].
        let big: Tensor<f32> = from_vec(
            (0..90).map(|i| (i as f32 - 45.0) * 0.1).collect::<Vec<_>>(),
            &[9, 10],
        )
        .unwrap();
        let out = compiled
            .forward_symbolic(std::slice::from_ref(&big))
            .unwrap();
        assert_eq!(out.shape(), &[9, 10]);

        // Sanity on values: the reshape is a no-op so output == relu(input).
        let d = out.data_vec().unwrap();
        let bd = big.data_vec().unwrap();
        for (i, (&o, &x)) in d.iter().zip(bd.iter()).enumerate() {
            assert!(
                (o - x.max(0.0)).abs() < 1e-6,
                "mismatch at {i}: got {o}, expected {}",
                x.max(0.0),
            );
        }
    }

    #[test]
    fn test_symbolic_module_guard_rejects_wrong_non_batch_dim() {
        let example = from_vec(vec![1.0f32; 4 * 10], &[4, 10])
            .unwrap()
            .requires_grad_(true);
        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        let compiled =
            compile_symbolic::<f32, _>(|inputs| activation::relu(&inputs[0]), &[example], sig)
                .unwrap();

        // Wrong second dim should be rejected by the guard.
        let bad = from_vec(vec![1.0f32; 16 * 7], &[16, 7]).unwrap();
        let err = compiled.forward_symbolic(&[bad]).unwrap_err();
        assert!(format!("{err}").contains("dim 1"));
    }

    #[test]
    fn test_patch_reshape_substitutes_single_symbolic_value() {
        // Build a graph: input [4, 10] -> reshape to [4, 10] -> output.
        // After patching with symbolic dim 0, the reshape target
        // should become [-1, 10].
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4, 10]);
        let (_, outs) = g.add_node(
            IrOpKind::Reshape { shape: vec![4, 10] },
            vec![x],
            vec![vec![4, 10]],
        );
        g.set_outputs(vec![outs[0]]);

        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        patch_reshape_for_symbolic_dims(&mut g, &[vec![4, 10]], &sig);

        let reshape_node = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Reshape { .. }))
            .unwrap();
        if let IrOpKind::Reshape { shape } = &reshape_node.op {
            assert_eq!(shape, &vec![-1, 10]);
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_patch_reshape_leaves_ambiguous_shapes_alone() {
        // If the reshape target has TWO matching positions, the
        // patcher should not touch it.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4, 4]);
        let (_, outs) = g.add_node(
            IrOpKind::Reshape { shape: vec![4, 4] },
            vec![x],
            vec![vec![4, 4]],
        );
        g.set_outputs(vec![outs[0]]);

        let sig = ShapeSignature::new().symbolic_dim(0, 0);
        patch_reshape_for_symbolic_dims(&mut g, &[vec![4, 4]], &sig);

        let reshape_node = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Reshape { .. }))
            .unwrap();
        if let IrOpKind::Reshape { shape } = &reshape_node.op {
            // Still both 4s — patcher bailed out due to ambiguity.
            assert_eq!(shape, &vec![4, 4]);
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_patch_reshape_leaves_graphs_without_symbolic_dims_alone() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4, 10]);
        let (_, outs) = g.add_node(
            IrOpKind::Reshape { shape: vec![4, 10] },
            vec![x],
            vec![vec![4, 10]],
        );
        g.set_outputs(vec![outs[0]]);

        patch_reshape_for_symbolic_dims(&mut g, &[vec![4, 10]], &ShapeSignature::new());

        let reshape_node = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Reshape { .. }))
            .unwrap();
        if let IrOpKind::Reshape { shape } = &reshape_node.op {
            assert_eq!(shape, &vec![4, 10]);
        } else {
            unreachable!();
        }
    }
}
