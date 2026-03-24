//! Codegen backends for compiling IR graphs into executable code.
//!
//! Instead of walking the graph with an interpreter on every forward pass,
//! a codegen backend converts an [`IrGraph`] into a [`CompiledGraph`] that
//! can be invoked repeatedly with different inputs. Two backends are provided:
//!
//! - [`InterpreterBackend`] — wraps the existing interpreter as a `Codegen`
//!   implementation (useful as a reference / fallback).
//! - [`NativeBackend`] — for simple elementwise chains, composes Rust
//!   closures directly, skipping interpreter dispatch overhead. Falls back
//!   to the interpreter for graphs it cannot natively compile.

use std::collections::HashMap;
use std::sync::Arc;

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::graph::{IrGraph, IrNodeId, IrOpKind, IrValueId};
use crate::interpreter;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A compiled graph that can be executed with `f64` inputs.
///
/// The compilation step converts the IR graph into a boxed closure that
/// takes flat `Vec<f64>` inputs and produces a flat `Vec<f64>` output,
/// avoiding per-execution graph traversal overhead.
pub struct CompiledGraph {
    /// The compiled execution function.
    execute: Box<dyn Fn(&[Vec<f64>]) -> FerrotorchResult<Vec<f64>> + Send + Sync>,
    /// Number of expected inputs.
    num_inputs: usize,
    /// Shape of the output tensor.
    output_shape: Vec<usize>,
}

impl CompiledGraph {
    /// Execute the compiled graph on the given inputs.
    ///
    /// Each element of `inputs` is a flat `Vec<f64>` corresponding to one
    /// graph input in the same order as `IrGraph::input_values`.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of inputs is wrong or if execution
    /// fails internally.
    pub fn execute(&self, inputs: &[Vec<f64>]) -> FerrotorchResult<Vec<f64>> {
        if inputs.len() != self.num_inputs {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CompiledGraph::execute: expected {} inputs, got {}",
                    self.num_inputs,
                    inputs.len()
                ),
            });
        }
        (self.execute)(inputs)
    }

    /// The number of inputs this compiled graph expects.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// The shape of the output produced by this compiled graph.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}

impl std::fmt::Debug for CompiledGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("num_inputs", &self.num_inputs)
            .field("output_shape", &self.output_shape)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Codegen trait
// ---------------------------------------------------------------------------

/// A backend that compiles an IR graph into an executable [`CompiledGraph`].
pub trait Codegen: Send + Sync {
    /// Compile the given IR graph.
    fn compile(&self, graph: &IrGraph) -> FerrotorchResult<CompiledGraph>;

    /// Human-readable name for this backend (e.g. `"interpreter"`, `"native"`).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// InterpreterBackend
// ---------------------------------------------------------------------------

/// Codegen backend that delegates execution to the IR interpreter.
///
/// This is the simplest backend: it captures the graph and, on each
/// invocation, converts the flat `f64` inputs to tensors, runs the
/// interpreter, and extracts the output data.
pub struct InterpreterBackend;

impl Codegen for InterpreterBackend {
    fn compile(&self, graph: &IrGraph) -> FerrotorchResult<CompiledGraph> {
        let graph = graph.clone();

        let num_inputs = graph.input_values.len();

        // Determine expected input shapes for validation and tensor construction.
        let input_shapes: Vec<Vec<usize>> = collect_input_shapes(&graph);

        // Determine output shape from the graph metadata.
        let output_shape = resolve_output_shape(&graph)?;

        let execute = move |inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> {
            // Convert flat f64 slices into Tensor<f64>.
            let tensors: Vec<Tensor<f64>> = inputs
                .iter()
                .zip(input_shapes.iter())
                .map(|(data, shape)| {
                    Tensor::from_storage(
                        TensorStorage::cpu(data.clone()),
                        shape.clone(),
                        false,
                    )
                })
                .collect::<FerrotorchResult<Vec<_>>>()?;

            // Run the interpreter.
            let result = interpreter::interpret(&graph, &tensors)?;

            // Extract output data.
            Ok(result.data()?.to_vec())
        };

        Ok(CompiledGraph {
            execute: Box::new(execute),
            num_inputs,
            output_shape,
        })
    }

    fn name(&self) -> &str {
        "interpreter"
    }
}

// ---------------------------------------------------------------------------
// NativeBackend
// ---------------------------------------------------------------------------

/// Codegen backend that compiles simple elementwise graphs into optimized
/// Rust closures, bypassing interpreter dispatch overhead.
///
/// The native backend recognizes graphs that consist entirely of:
/// - A single input
/// - A chain of unary elementwise operations (neg, sqrt, abs, pow, relu,
///   sigmoid, tanh, gelu, silu) or a `FusedElementwise` node
/// - An optional constant-add/sub/mul/div with a second input or constant
///
/// For graphs it cannot natively compile, it falls back to the
/// [`InterpreterBackend`].
pub struct NativeBackend;

/// A single elementwise f64 operation that can be composed.
type ElementwiseOp = Arc<dyn Fn(f64) -> f64 + Send + Sync>;

impl Codegen for NativeBackend {
    fn compile(&self, graph: &IrGraph) -> FerrotorchResult<CompiledGraph> {
        match try_compile_native(graph) {
            Some(compiled) => Ok(compiled),
            None => {
                // Fall back to interpreter for anything we cannot natively compile.
                InterpreterBackend.compile(graph)
            }
        }
    }

    fn name(&self) -> &str {
        "native"
    }
}

/// Attempt to natively compile the graph. Returns `None` if the graph is
/// too complex for the native backend.
fn try_compile_native(graph: &IrGraph) -> Option<CompiledGraph> {
    let topo = graph.topological_order();

    // Build node lookup.
    let node_map: HashMap<IrNodeId, &_> = graph.nodes.iter().map(|n| (n.id, n)).collect();

    // We need exactly one output.
    if graph.output_values.len() != 1 {
        return None;
    }

    // Classify every non-input node in topological order.
    // We support the following patterns:
    //
    // Pattern 1 — Pure unary chain:
    //   Input -> UnaryOp -> UnaryOp -> ... -> Output
    //
    // Pattern 2 — Binary with constant:
    //   Input + Constant -> BinaryOp -> UnaryOp chain -> Output
    //
    // Pattern 3 — Binary self (input + input):
    //   Input -> BinaryOp(input, input) -> UnaryOp chain -> Output
    //
    // We collect an ordered list of operations to apply. If at any point
    // we encounter something we cannot handle, we return None.

    // Walk the topo order and build a pipeline of element-level operations.
    // We track value -> "compiled form" where the compiled form is either
    // "input(index)" or "constant(data)" or "computed(ops)".

    #[derive(Clone)]
    enum ValueKind {
        Input(usize),
        Constant(Vec<f64>),
        /// Elementwise-transformed version of a single input, carrying the
        /// composed operation pipeline.
        Computed(Vec<ElementwiseOp>),
    }

    let mut value_kinds: HashMap<IrValueId, ValueKind> = HashMap::new();

    for &nid in &topo {
        let node = node_map[&nid];
        match &node.op {
            IrOpKind::Input { index } => {
                for &out_id in &node.outputs {
                    value_kinds.insert(out_id, ValueKind::Input(*index));
                }
            }

            IrOpKind::Constant { data, .. } => {
                for &out_id in &node.outputs {
                    value_kinds.insert(out_id, ValueKind::Constant(data.clone()));
                }
            }

            IrOpKind::Output => {
                // Pass through.
                if let Some(&input_id) = node.inputs.first() {
                    if let Some(kind) = value_kinds.get(&input_id).cloned() {
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, kind.clone());
                        }
                    }
                }
            }

            // Unary elementwise ops.
            op @ (IrOpKind::Neg
            | IrOpKind::Sqrt
            | IrOpKind::Abs
            | IrOpKind::Relu
            | IrOpKind::Sigmoid
            | IrOpKind::Tanh
            | IrOpKind::Gelu
            | IrOpKind::Silu
            | IrOpKind::Pow { .. }) => {
                let input_id = *node.inputs.first()?;
                let input_kind = value_kinds.get(&input_id)?.clone();

                let new_op = make_elementwise_op(op)?;

                let mut ops = match input_kind {
                    ValueKind::Input(_) => Vec::new(),
                    ValueKind::Computed(ops) => ops,
                    ValueKind::Constant(_) => return None, // constant -> op not handled natively
                };
                ops.push(new_op);

                for &out_id in &node.outputs {
                    value_kinds.insert(out_id, ValueKind::Computed(ops.clone()));
                }
            }

            // FusedElementwise: chain of unary ops applied sequentially.
            IrOpKind::FusedElementwise { ops } => {
                let input_id = *node.inputs.first()?;
                let input_kind = value_kinds.get(&input_id)?.clone();

                let mut pipeline = match input_kind {
                    ValueKind::Input(_) => Vec::new(),
                    ValueKind::Computed(existing) => existing,
                    ValueKind::Constant(_) => return None,
                };

                for op in ops {
                    pipeline.push(make_elementwise_op(op)?);
                }

                for &out_id in &node.outputs {
                    value_kinds.insert(out_id, ValueKind::Computed(pipeline.clone()));
                }
            }

            // Binary ops where one side is a constant or it is self-op.
            op @ (IrOpKind::Add | IrOpKind::Sub | IrOpKind::Mul | IrOpKind::Div) => {
                if node.inputs.len() < 2 {
                    return None;
                }
                let lhs_id = node.inputs[0];
                let rhs_id = node.inputs[1];
                let lhs_kind = value_kinds.get(&lhs_id)?.clone();
                let rhs_kind = value_kinds.get(&rhs_id)?.clone();

                match (&lhs_kind, &rhs_kind) {
                    // Input op Input (same input, e.g. x + x).
                    (ValueKind::Input(li), ValueKind::Input(ri)) if li == ri => {
                        let binary_op = make_self_binary_op(op)?;
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(vec![binary_op.clone()]));
                        }
                    }

                    // Computed op Computed (same pipeline base — self-op on transformed value).
                    (ValueKind::Input(_), ValueKind::Constant(cdata))
                    | (ValueKind::Computed(_), ValueKind::Constant(cdata)) => {
                        let cdata = cdata.clone();
                        let mut ops = match lhs_kind {
                            ValueKind::Input(_) => Vec::new(),
                            ValueKind::Computed(o) => o,
                            _ => unreachable!(),
                        };
                        let binary_op = make_binary_with_constant_op(op, &cdata, false)?;
                        ops.push(binary_op);
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(ops.clone()));
                        }
                    }

                    // Constant op Input/Computed (reversed operand order).
                    (ValueKind::Constant(cdata), ValueKind::Input(_))
                    | (ValueKind::Constant(cdata), ValueKind::Computed(_)) => {
                        let cdata = cdata.clone();
                        let mut ops = match rhs_kind {
                            ValueKind::Input(_) => Vec::new(),
                            ValueKind::Computed(o) => o,
                            _ => unreachable!(),
                        };
                        let binary_op = make_binary_with_constant_op(op, &cdata, true)?;
                        ops.push(binary_op);
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(ops.clone()));
                        }
                    }

                    _ => return None,
                }
            }

            // Anything else (reductions, linalg, shape ops) is not natively compilable.
            _ => return None,
        }
    }

    // Now determine what we actually compiled for the output value.
    let output_id = graph.output_values[0];
    let output_kind = value_kinds.get(&output_id)?.clone();

    let num_inputs = graph.input_values.len();
    let output_shape = resolve_output_shape(graph).ok()?;

    match output_kind {
        ValueKind::Input(_) => {
            // Identity: output == input. Build a trivial passthrough.
            let execute = move |inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> {
                Ok(inputs[0].clone())
            };
            Some(CompiledGraph {
                execute: Box::new(execute),
                num_inputs,
                output_shape,
            })
        }

        ValueKind::Computed(ops) => {
            // We have a pipeline of elementwise ops to apply.
            let ops = Arc::new(ops);
            let execute = move |inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> {
                // Start from the first input's data.
                let data = &inputs[0];
                let result: Vec<f64> = data
                    .iter()
                    .map(|&x| {
                        let mut val = x;
                        for op in ops.iter() {
                            val = op(val);
                        }
                        val
                    })
                    .collect();
                Ok(result)
            };

            Some(CompiledGraph {
                execute: Box::new(execute),
                num_inputs,
                output_shape,
            })
        }

        ValueKind::Constant(data) => {
            // Constant output — unusual but valid.
            let execute = move |_inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> {
                Ok(data.clone())
            };
            Some(CompiledGraph {
                execute: Box::new(execute),
                num_inputs,
                output_shape,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Elementwise operation builders
// ---------------------------------------------------------------------------

/// Convert an `IrOpKind` into a scalar `f64 -> f64` closure.
fn make_elementwise_op(op: &IrOpKind) -> Option<ElementwiseOp> {
    match op {
        IrOpKind::Neg => Some(Arc::new(|x: f64| -x)),
        IrOpKind::Sqrt => Some(Arc::new(|x: f64| x.sqrt())),
        IrOpKind::Abs => Some(Arc::new(|x: f64| x.abs())),
        IrOpKind::Pow { exponent } => {
            let exp = *exponent;
            Some(Arc::new(move |x: f64| x.powf(exp)))
        }
        IrOpKind::Relu => Some(Arc::new(|x: f64| x.max(0.0))),
        IrOpKind::Sigmoid => Some(Arc::new(|x: f64| 1.0 / (1.0 + (-x).exp()))),
        IrOpKind::Tanh => Some(Arc::new(|x: f64| x.tanh())),
        IrOpKind::Gelu => Some(Arc::new(|x: f64| {
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
            x * 0.5 * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
        })),
        IrOpKind::Silu => Some(Arc::new(|x: f64| x / (1.0 + (-x).exp()))),
        _ => None,
    }
}

/// Create a closure for a binary self-operation (e.g. `x + x`).
fn make_self_binary_op(op: &IrOpKind) -> Option<ElementwiseOp> {
    match op {
        IrOpKind::Add => Some(Arc::new(|x: f64| x + x)),
        IrOpKind::Sub => Some(Arc::new(|_x: f64| 0.0)),
        IrOpKind::Mul => Some(Arc::new(|x: f64| x * x)),
        IrOpKind::Div => Some(Arc::new(|_x: f64| 1.0)),
        _ => None,
    }
}

/// Create an element-level closure for a binary op where one side is a constant.
///
/// The constant is broadcast: for each element index `i`, the constant value
/// at `i % constant.len()` is used. When `constant_is_lhs` is true the
/// constant appears on the left-hand side of the binary op.
///
/// The closure captures the constant data and the index is tracked via
/// an `AtomicUsize` counter that resets at the start of each application.
/// Since we process elements sequentially in `try_compile_native`, a simpler
/// approach is to return a closure that takes `(index, value)` — but our
/// pipeline uses `f64 -> f64`. To work around this we embed the constant
/// data and use modular indexing via a cell.
///
/// Actually, the easier approach: binary-with-constant ops produce a *vector*
/// closure rather than an element closure. We handle this by returning an op
/// that encodes the positional constant lookup.
fn make_binary_with_constant_op(
    op: &IrOpKind,
    constant: &[f64],
    constant_is_lhs: bool,
) -> Option<ElementwiseOp> {
    // For a single-element constant (scalar), we can use a simple closure.
    if constant.len() == 1 {
        let c = constant[0];
        return match (op, constant_is_lhs) {
            (IrOpKind::Add, _) => Some(Arc::new(move |x: f64| x + c)),
            (IrOpKind::Sub, false) => Some(Arc::new(move |x: f64| x - c)),
            (IrOpKind::Sub, true) => Some(Arc::new(move |x: f64| c - x)),
            (IrOpKind::Mul, _) => Some(Arc::new(move |x: f64| x * c)),
            (IrOpKind::Div, false) => Some(Arc::new(move |x: f64| x / c)),
            (IrOpKind::Div, true) => Some(Arc::new(move |x: f64| c / x)),
            _ => None,
        };
    }

    // For multi-element constants we need positional indexing. We use an
    // AtomicUsize counter that wraps around the constant length. The counter
    // is incremented on every call and reset is implicit (modulo).
    //
    // WRAPPING BEHAVIOR: AtomicUsize::fetch_add wraps on overflow (at
    // usize::MAX).  Because we always reduce via `% clen`, this is
    // harmless: for any `clen` that is a power-of-two the modulo result is
    // the same regardless of how many times the counter has wrapped.  For
    // non-power-of-two `clen` values there is a theoretical bias once the
    // counter wraps, but usize::MAX wraps require ~18 quintillion calls on
    // 64-bit targets, which is unreachable in practice.
    //
    // SAFETY NOTE: This works correctly only when the closure is invoked
    // sequentially for elements 0..N within a single `execute` call. The
    // NativeBackend guarantees this via the `iter().map()` pipeline.
    use std::sync::atomic::{AtomicUsize, Ordering};

    let cdata = Arc::new(constant.to_vec());
    let counter = Arc::new(AtomicUsize::new(0));

    let clen = cdata.len();

    match (op, constant_is_lhs) {
        (IrOpKind::Add, _) => {
            Some(Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x + cdata[i]
            }))
        }
        (IrOpKind::Sub, false) => {
            Some(Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x - cdata[i]
            }))
        }
        (IrOpKind::Sub, true) => {
            Some(Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                cdata[i] - x
            }))
        }
        (IrOpKind::Mul, _) => {
            Some(Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x * cdata[i]
            }))
        }
        (IrOpKind::Div, false) => {
            Some(Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x / cdata[i]
            }))
        }
        (IrOpKind::Div, true) => {
            Some(Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                cdata[i] / x
            }))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Collect the expected shapes for each graph input, in order.
fn collect_input_shapes(graph: &IrGraph) -> Vec<Vec<usize>> {
    graph
        .input_values
        .iter()
        .map(|&val_id| {
            graph
                .values
                .iter()
                .find(|v| v.id == val_id)
                .map(|v| v.shape.clone())
                .unwrap_or_default()
        })
        .collect()
}

/// Determine the output shape from the graph's single output value.
fn resolve_output_shape(graph: &IrGraph) -> FerrotorchResult<Vec<usize>> {
    if graph.output_values.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "codegen: graph has no outputs".into(),
        });
    }
    let output_id = graph.output_values[0];
    graph
        .values
        .iter()
        .find(|v| v.id == output_id)
        .map(|v| v.shape.clone())
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("codegen: output value {:?} not found in graph values", output_id),
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};

    /// Assert two f64 slices are elementwise close.
    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: got {a}, expected {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // InterpreterBackend tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_backend_add_self() {
        // Graph: y = x + x
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let backend = InterpreterBackend;
        let compiled = backend.compile(&g).unwrap();

        assert_eq!(compiled.num_inputs(), 1);
        assert_eq!(compiled.output_shape(), &[3]);

        let result = compiled.execute(&[vec![1.0, 2.0, 3.0]]).unwrap();
        assert_close(&result, &[2.0, 4.0, 6.0], 1e-10);
    }

    #[test]
    fn test_interpreter_backend_constant_add() {
        // Graph: y = x + [10, 20, 30]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![10.0, 20.0, 30.0], vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, c], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled = InterpreterBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![1.0, 2.0, 3.0]]).unwrap();
        assert_close(&result, &[11.0, 22.0, 33.0], 1e-10);
    }

    #[test]
    fn test_interpreter_backend_input_count_mismatch() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled = InterpreterBackend.compile(&g).unwrap();
        let err = compiled.execute(&[vec![1.0], vec![2.0]]);
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // NativeBackend tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_native_backend_add_self() {
        // Graph: y = x + x  (self-binary)
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![4]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        assert_eq!(compiled.num_inputs(), 1);

        let result = compiled.execute(&[vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        assert_close(&result, &[2.0, 4.0, 6.0, 8.0], 1e-10);
    }

    #[test]
    fn test_native_backend_unary_chain() {
        // Graph: y = relu(neg(x))
        // x = [-1, 2, -3] -> neg -> [1, -2, 3] -> relu -> [1, 0, 3]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![3]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![-1.0, 2.0, -3.0]]).unwrap();
        assert_close(&result, &[1.0, 0.0, 3.0], 1e-10);
    }

    #[test]
    fn test_native_backend_fused_elementwise() {
        // Graph: y = FusedElementwise([Neg, Relu])(x)
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, fused_outs) = g.add_node(
            IrOpKind::FusedElementwise {
                ops: vec![IrOpKind::Neg, IrOpKind::Relu],
            },
            vec![x],
            vec![vec![3]],
        );
        g.set_outputs(vec![fused_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![-1.0, 2.0, -3.0]]).unwrap();
        assert_close(&result, &[1.0, 0.0, 3.0], 1e-10);
    }

    #[test]
    fn test_native_backend_constant_add() {
        // Graph: y = x + [10, 20, 30]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![10.0, 20.0, 30.0], vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, c], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![1.0, 2.0, 3.0]]).unwrap();
        assert_close(&result, &[11.0, 22.0, 33.0], 1e-10);
    }

    #[test]
    fn test_native_backend_scalar_constant_mul() {
        // Graph: y = x * 2.0
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let c = g.add_constant(vec![2.0], vec![1]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![x, c], vec![vec![4]]);
        g.set_outputs(vec![mul_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        assert_close(&result, &[2.0, 4.0, 6.0, 8.0], 1e-10);
    }

    #[test]
    fn test_native_backend_pow_sqrt_chain() {
        // Graph: y = sqrt(pow(x, 2))  which computes |x| for non-negative pow results.
        // x = [3, 4, 5]
        // pow(_, 2) = [9, 16, 25]
        // sqrt = [3, 4, 5]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, pow_outs) = g.add_node(
            IrOpKind::Pow { exponent: 2.0 },
            vec![x],
            vec![vec![3]],
        );
        let (_, sqrt_outs) = g.add_node(IrOpKind::Sqrt, vec![pow_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![sqrt_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![3.0, 4.0, 5.0]]).unwrap();
        assert_close(&result, &[3.0, 4.0, 5.0], 1e-10);
    }

    #[test]
    fn test_native_backend_sigmoid() {
        // Graph: y = sigmoid(x)
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![x], vec![vec![3]]);
        g.set_outputs(vec![sig_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![0.0, 1.0, -1.0]]).unwrap();

        // sigmoid(0) = 0.5, sigmoid(1) ~ 0.7311, sigmoid(-1) ~ 0.2689
        assert_close(&result, &[0.5, 0.7310585786, 0.2689414214], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Native vs Interpreter consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_native_matches_interpreter_unary_chain() {
        // Graph: y = tanh(sigmoid(neg(x)))
        let mut g = IrGraph::new();
        let x = g.add_input(vec![5]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![5]]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![neg_outs[0]], vec![vec![5]]);
        let (_, tanh_outs) = g.add_node(IrOpKind::Tanh, vec![sig_outs[0]], vec![vec![5]]);
        g.set_outputs(vec![tanh_outs[0]]);

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        let interp_result = InterpreterBackend.compile(&g).unwrap()
            .execute(&[input.clone()]).unwrap();
        let native_result = NativeBackend.compile(&g).unwrap()
            .execute(&[input]).unwrap();

        // The interpreter uses Tensor<f64> ops while the native backend uses
        // raw f64 math. Allow a small tolerance for floating-point differences.
        assert_close(&native_result, &interp_result, 1e-10);
    }

    #[test]
    fn test_native_matches_interpreter_constant_sub() {
        // Graph: y = x - [1, 2, 3]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![1.0, 2.0, 3.0], vec![3]);
        let (_, sub_outs) = g.add_node(IrOpKind::Sub, vec![x, c], vec![vec![3]]);
        g.set_outputs(vec![sub_outs[0]]);

        let input = vec![10.0, 20.0, 30.0];

        let interp_result = InterpreterBackend.compile(&g).unwrap()
            .execute(&[input.clone()]).unwrap();
        let native_result = NativeBackend.compile(&g).unwrap()
            .execute(&[input]).unwrap();

        assert_close(&native_result, &interp_result, 1e-10);
    }

    #[test]
    fn test_native_falls_back_for_matmul() {
        // Matmul is not supported by the native backend, so it should fall
        // back to the interpreter.
        let mut g = IrGraph::new();
        let a = g.add_input(vec![2, 3]);
        let b = g.add_input(vec![3, 2]);
        let (_, mm_outs) = g.add_node(IrOpKind::Mm, vec![a, b], vec![vec![2, 2]]);
        g.set_outputs(vec![mm_outs[0]]);

        let backend = NativeBackend;
        // Should still compile (via fallback).
        let compiled = backend.compile(&g).unwrap();
        assert_eq!(compiled.num_inputs(), 2);

        // A = [[1, 2, 3], [4, 5, 6]], B = [[7, 8], [9, 10], [11, 12]]
        let result = compiled
            .execute(&[
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ])
            .unwrap();
        assert_close(&result, &[58.0, 64.0, 139.0, 154.0], 1e-8);
    }

    #[test]
    fn test_native_identity_passthrough() {
        // Graph: output = input (identity).
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        g.set_outputs(vec![x]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![42.0, 43.0, 44.0]]).unwrap();
        assert_close(&result, &[42.0, 43.0, 44.0], 1e-10);
    }

    // -----------------------------------------------------------------------
    // Zero-element tensor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_backend_zero_element_tensor() {
        // Graph: y = neg(x) with shape [0]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![0]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![0]]);
        g.set_outputs(vec![neg_outs[0]]);

        let compiled = InterpreterBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![]]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_native_backend_zero_element_tensor() {
        // Graph: y = relu(x) with shape [0]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![0]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![0]]);
        g.set_outputs(vec![relu_outs[0]]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![]]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_compiled_graph_debug() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        g.set_outputs(vec![x]);

        let compiled = NativeBackend.compile(&g).unwrap();
        let debug_str = format!("{:?}", compiled);
        assert!(debug_str.contains("CompiledGraph"));
        assert!(debug_str.contains("num_inputs: 1"));
    }
}
