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
/// Type alias for a compiled execution closure.
type CompiledExecFn = Box<dyn Fn(&[Vec<f64>]) -> FerrotorchResult<Vec<f64>> + Send + Sync>;

pub struct CompiledGraph {
    /// The compiled execution function.
    execute: CompiledExecFn,
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
                    Tensor::from_storage(TensorStorage::cpu(data.clone()), shape.clone(), false)
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

/// An `AtomicUsize` counter used by multi-element constant binary ops.
/// Collected so that counters can be reset to 0 at the start of each
/// `execute` call, preventing drift across invocations.
type CounterRef = Arc<std::sync::atomic::AtomicUsize>;

/// A binary operation on two f64 values.
type BinaryOp = Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>;

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
        /// Binary operation on two different inputs, optionally followed
        /// by a unary chain. Stores: (lhs_input_idx, rhs_input_idx, binary_op, unary_chain).
        BinaryComputed {
            lhs_input: usize,
            rhs_input: usize,
            binary: BinaryOp,
            unary_chain: Vec<ElementwiseOp>,
        },
    }

    let mut value_kinds: HashMap<IrValueId, ValueKind> = HashMap::new();

    // Registry of AtomicUsize counters used by multi-element constant ops.
    // These must be reset to 0 at the start of each `execute` call.
    let mut counters: Vec<CounterRef> = Vec::new();

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
            | IrOpKind::Exp
            | IrOpKind::Log
            | IrOpKind::Pow { .. }) => {
                let input_id = *node.inputs.first()?;
                let input_kind = value_kinds.get(&input_id)?.clone();

                let new_op = make_elementwise_op(op)?;

                match input_kind {
                    ValueKind::Input(_) => {
                        let ops = vec![new_op];
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(ops.clone()));
                        }
                    }
                    ValueKind::Computed(mut ops) => {
                        ops.push(new_op);
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(ops.clone()));
                        }
                    }
                    ValueKind::BinaryComputed {
                        lhs_input,
                        rhs_input,
                        binary,
                        mut unary_chain,
                    } => {
                        unary_chain.push(new_op);
                        for &out_id in &node.outputs {
                            value_kinds.insert(
                                out_id,
                                ValueKind::BinaryComputed {
                                    lhs_input,
                                    rhs_input,
                                    binary: binary.clone(),
                                    unary_chain: unary_chain.clone(),
                                },
                            );
                        }
                    }
                    ValueKind::Constant(_) => return None, // constant -> op not handled natively
                }
            }

            // FusedElementwise: chain of unary ops applied sequentially.
            IrOpKind::FusedElementwise { ops } => {
                let input_id = *node.inputs.first()?;
                let input_kind = value_kinds.get(&input_id)?.clone();

                let mut new_ops: Vec<ElementwiseOp> = Vec::new();
                for op in ops {
                    new_ops.push(make_elementwise_op(op)?);
                }

                match input_kind {
                    ValueKind::BinaryComputed {
                        lhs_input,
                        rhs_input,
                        binary,
                        mut unary_chain,
                    } => {
                        unary_chain.extend(new_ops);
                        for &out_id in &node.outputs {
                            value_kinds.insert(
                                out_id,
                                ValueKind::BinaryComputed {
                                    lhs_input,
                                    rhs_input,
                                    binary: binary.clone(),
                                    unary_chain: unary_chain.clone(),
                                },
                            );
                        }
                    }
                    ValueKind::Input(_) => {
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(new_ops.clone()));
                        }
                    }
                    ValueKind::Computed(mut existing) => {
                        existing.extend(new_ops);
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(existing.clone()));
                        }
                    }
                    ValueKind::Constant(_) => return None,
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
                            value_kinds
                                .insert(out_id, ValueKind::Computed(vec![binary_op.clone()]));
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
                        let (binary_op, counter) = make_binary_with_constant_op(op, &cdata, false)?;
                        if let Some(c) = counter {
                            counters.push(c);
                        }
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
                        let (binary_op, counter) = make_binary_with_constant_op(op, &cdata, true)?;
                        if let Some(c) = counter {
                            counters.push(c);
                        }
                        ops.push(binary_op);
                        for &out_id in &node.outputs {
                            value_kinds.insert(out_id, ValueKind::Computed(ops.clone()));
                        }
                    }

                    // Two different inputs: Input(i) op Input(j) where i != j.
                    (ValueKind::Input(li), ValueKind::Input(ri)) => {
                        let binary = make_two_input_binary_op(op)?;
                        for &out_id in &node.outputs {
                            value_kinds.insert(
                                out_id,
                                ValueKind::BinaryComputed {
                                    lhs_input: *li,
                                    rhs_input: *ri,
                                    binary: binary.clone(),
                                    unary_chain: Vec::new(),
                                },
                            );
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
            let execute =
                move |inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> { Ok(inputs[0].clone()) };
            Some(CompiledGraph {
                execute: Box::new(execute),
                num_inputs,
                output_shape,
            })
        }

        ValueKind::Computed(ops) => {
            // We have a pipeline of elementwise ops to apply.
            let ops = Arc::new(ops);
            let counters = Arc::new(counters);
            let execute = move |inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> {
                // Reset all positional counters to 0 so that multi-element
                // constant binary ops index correctly on each call.
                for c in counters.iter() {
                    c.store(0, std::sync::atomic::Ordering::Relaxed);
                }
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

        ValueKind::BinaryComputed {
            lhs_input,
            rhs_input,
            binary,
            unary_chain,
        } => {
            // Binary op on two inputs, optionally followed by unary chain.
            let unary_chain = Arc::new(unary_chain);
            let execute = move |inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> {
                let lhs = &inputs[lhs_input];
                let rhs = &inputs[rhs_input];
                let len = lhs.len().min(rhs.len());
                let result: Vec<f64> = (0..len)
                    .map(|i| {
                        let mut val = binary(lhs[i], rhs[i]);
                        for op in unary_chain.iter() {
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
            let execute =
                move |_inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> { Ok(data.clone()) };
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
/// Create a `(f64, f64) -> f64` closure for a binary op on two inputs.
fn make_two_input_binary_op(op: &IrOpKind) -> Option<BinaryOp> {
    match op {
        IrOpKind::Add => Some(Arc::new(|a: f64, b: f64| a + b)),
        IrOpKind::Sub => Some(Arc::new(|a: f64, b: f64| a - b)),
        IrOpKind::Mul => Some(Arc::new(|a: f64, b: f64| a * b)),
        IrOpKind::Div => Some(Arc::new(|a: f64, b: f64| a / b)),
        _ => None,
    }
}

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
        IrOpKind::Exp => Some(Arc::new(|x: f64| x.exp())),
        IrOpKind::Log => Some(Arc::new(|x: f64| x.ln())),
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
/// Returns `(op, optional_counter)`. The counter must be reset to 0 before
/// each batch of element-level invocations to prevent drift across
/// successive `execute` calls.
fn make_binary_with_constant_op(
    op: &IrOpKind,
    constant: &[f64],
    constant_is_lhs: bool,
) -> Option<(ElementwiseOp, Option<CounterRef>)> {
    // For a single-element constant (scalar), we can use a simple closure.
    // No counter needed — the value is the same for every element.
    if constant.len() == 1 {
        let c = constant[0];
        return match (op, constant_is_lhs) {
            (IrOpKind::Add, _) => Some((Arc::new(move |x: f64| x + c), None)),
            (IrOpKind::Sub, false) => Some((Arc::new(move |x: f64| x - c), None)),
            (IrOpKind::Sub, true) => Some((Arc::new(move |x: f64| c - x), None)),
            (IrOpKind::Mul, _) => Some((Arc::new(move |x: f64| x * c), None)),
            (IrOpKind::Div, false) => Some((Arc::new(move |x: f64| x / c), None)),
            (IrOpKind::Div, true) => Some((Arc::new(move |x: f64| c / x), None)),
            _ => None,
        };
    }

    // For multi-element constants we need positional indexing. We use an
    // AtomicUsize counter that wraps around the constant length. The counter
    // is reset to 0 at the start of each `execute` call (via the counter
    // registry) so that element indexing is correct across invocations.
    //
    // SAFETY NOTE: This works correctly only when the closure is invoked
    // sequentially for elements 0..N within a single `execute` call. The
    // NativeBackend guarantees this via the `iter().map()` pipeline.
    use std::sync::atomic::{AtomicUsize, Ordering};

    let cdata = Arc::new(constant.to_vec());
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);

    let clen = cdata.len();

    match (op, constant_is_lhs) {
        (IrOpKind::Add, _) => Some((
            Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x + cdata[i]
            }),
            Some(counter_clone),
        )),
        (IrOpKind::Sub, false) => Some((
            Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x - cdata[i]
            }),
            Some(counter_clone),
        )),
        (IrOpKind::Sub, true) => Some((
            Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                cdata[i] - x
            }),
            Some(counter_clone),
        )),
        (IrOpKind::Mul, _) => Some((
            Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x * cdata[i]
            }),
            Some(counter_clone),
        )),
        (IrOpKind::Div, false) => Some((
            Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                x / cdata[i]
            }),
            Some(counter_clone),
        )),
        (IrOpKind::Div, true) => Some((
            Arc::new(move |x: f64| {
                let i = counter.fetch_add(1, Ordering::Relaxed) % clen;
                cdata[i] / x
            }),
            Some(counter_clone),
        )),
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
            message: format!(
                "codegen: output value {:?} not found in graph values",
                output_id
            ),
        })
}

// ---------------------------------------------------------------------------
// InductorBackend
// ---------------------------------------------------------------------------

/// The target for the inductor-style code generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InductorTarget {
    /// Emit Rust source code with SIMD-friendly patterns.
    CpuRust,
    /// Emit C source code with OpenMP SIMD/parallel pragmas.
    CpuC,
    /// Emit PTX assembly targeting sm_52.
    GpuPtx,
    /// Emit CUDA C source code.
    GpuCuda,
}

/// Inductor-style codegen backend that performs DAG-level fusion and
/// emits optimized native code.
///
/// Unlike the [`NativeBackend`] which composes Rust closures for simple
/// elementwise chains, the inductor backend:
///
/// 1. Discovers fusion groups across the entire DAG (via [`dag_fusion`]).
/// 2. Lowers each group to loop-level IR ([`codegen_ir::LoopIR`]).
/// 3. Emits target-specific source code (Rust, C, CUDA, or PTX).
///
/// For `CpuRust` targets, the generated source is returned as a string
/// inside a `CompiledGraph` whose `execute` falls back to the interpreter
/// (the generated source serves as an optimization artifact). For all
/// other targets, `execute` also falls back to the interpreter while the
/// generated source is available via the [`InductorBackend::generate`]
/// method.
///
/// [`dag_fusion`]: crate::dag_fusion
/// [`codegen_ir::LoopIR`]: crate::codegen_ir::LoopIR
pub struct InductorBackend {
    /// The target to generate code for.
    pub target: InductorTarget,
    /// Thread block size for GPU targets (ignored for CPU).
    pub block_size: usize,
}

impl InductorBackend {
    /// Create a new inductor backend for the given target.
    pub fn new(target: InductorTarget) -> Self {
        Self {
            target,
            block_size: 256,
        }
    }

    /// Create a new inductor backend with a custom GPU block size.
    pub fn with_block_size(target: InductorTarget, block_size: usize) -> Self {
        Self { target, block_size }
    }

    /// Generate source code for the given IR graph without compiling it.
    ///
    /// Returns a vector of generated source strings, one per fusion group.
    pub fn generate(&self, graph: &IrGraph) -> FerrotorchResult<Vec<String>> {
        let groups = crate::dag_fusion::find_fusion_groups(graph);
        let loops_per_group = crate::dag_fusion::fuse_dag(&groups, graph);

        let num_graph_inputs = graph.input_values.len();

        let mut sources = Vec::with_capacity(groups.len());

        for (i, (group, loops)) in groups.iter().zip(loops_per_group.iter()).enumerate() {
            let fn_name = format!("kernel_{i}");
            let num_inputs = group.external_inputs.len().max(1);

            let source = match self.target {
                InductorTarget::CpuRust => {
                    crate::codegen_cpu::CpuCodegen::generate_rust_source(loops, &fn_name)
                }
                InductorTarget::CpuC => {
                    crate::codegen_cpu::CpuCodegen::generate_c_source(loops, &fn_name, num_inputs)
                }
                InductorTarget::GpuCuda => crate::codegen_gpu::GpuCodegen::generate_cuda_source(
                    loops, &fn_name, num_inputs,
                ),
                InductorTarget::GpuPtx => crate::codegen_gpu::GpuCodegen::generate_ptx_source(
                    loops,
                    &fn_name,
                    self.block_size,
                    num_inputs,
                ),
            };

            sources.push(source);
        }

        // If no groups were found (e.g., identity graph), produce an
        // empty-body kernel for completeness.
        if sources.is_empty() {
            let source = match self.target {
                InductorTarget::CpuRust => {
                    crate::codegen_cpu::CpuCodegen::generate_rust_source(&[], "kernel_identity")
                }
                InductorTarget::CpuC => crate::codegen_cpu::CpuCodegen::generate_c_source(
                    &[],
                    "kernel_identity",
                    num_graph_inputs.max(1),
                ),
                InductorTarget::GpuCuda => crate::codegen_gpu::GpuCodegen::generate_cuda_source(
                    &[],
                    "kernel_identity",
                    num_graph_inputs.max(1),
                ),
                InductorTarget::GpuPtx => crate::codegen_gpu::GpuCodegen::generate_ptx_source(
                    &[],
                    "kernel_identity",
                    self.block_size,
                    num_graph_inputs.max(1),
                ),
            };
            sources.push(source);
        }

        Ok(sources)
    }
}

impl Codegen for InductorBackend {
    fn compile(&self, graph: &IrGraph) -> FerrotorchResult<CompiledGraph> {
        // For the CpuC target, attempt the real JIT path: generate C,
        // shell out to the system C compiler, load the resulting shared
        // library, and dispatch to it on every `execute` call.  Graphs
        // that are a single elementwise fusion group are fully JIT'd;
        // anything else falls back to the interpreter.
        if self.target == InductorTarget::CpuC {
            if let Some(compiled) = try_jit_compile_cpu_c(graph)? {
                return Ok(compiled);
            }
        }

        // Fallback: generate source for inspection, then interpret.
        let _sources = self.generate(graph)?;
        InterpreterBackend.compile(graph)
    }

    fn name(&self) -> &str {
        match self.target {
            InductorTarget::CpuRust => "inductor-cpu-rust",
            InductorTarget::CpuC => "inductor-cpu-c",
            InductorTarget::GpuPtx => "inductor-gpu-ptx",
            InductorTarget::GpuCuda => "inductor-gpu-cuda",
        }
    }
}

// ---------------------------------------------------------------------------
// JIT compile path for CpuC
// ---------------------------------------------------------------------------

/// Resolve how an `external_input` of a fusion group is supplied at
/// runtime. Graph inputs become indices into the `execute` call's input
/// list; constants are baked into the closure.
#[derive(Debug, Clone)]
enum InputSource {
    /// `inputs[index]` from the execute-time input slice.
    GraphInput(usize),
    /// A constant buffer captured at compile time.
    Constant(Vec<f64>),
}

/// Attempt to JIT-compile the graph to a native shared library via the
/// CpuC target. Returns `Ok(Some(..))` on a successful JIT compile,
/// `Ok(None)` if the graph shape is unsupported, or `Err(..)` if the
/// compile pipeline itself failed.
///
/// The supported shape is:
///
/// - Exactly one fusion group, of kind `Elementwise`
/// - Exactly one graph output, which is the group's sole external output
/// - All external inputs are either graph inputs or graph constants
fn try_jit_compile_cpu_c(graph: &IrGraph) -> FerrotorchResult<Option<CompiledGraph>> {
    use crate::dag_fusion::{FusionGroupKind, find_fusion_groups, fuse_dag};
    use crate::graph::IrOpKind;

    let groups = find_fusion_groups(graph);
    if groups.len() != 1 {
        return Ok(None);
    }
    let group = &groups[0];
    if group.kind != FusionGroupKind::Elementwise {
        return Ok(None);
    }
    if graph.output_values.len() != 1 {
        return Ok(None);
    }
    if group.external_outputs.len() != 1 {
        return Ok(None);
    }
    if group.external_outputs[0] != graph.output_values[0] {
        return Ok(None);
    }
    if group.external_inputs.is_empty() {
        // A group with no external inputs is pathological; bail.
        return Ok(None);
    }

    // Map each external_input to its runtime source.
    let mut sources: Vec<InputSource> = Vec::with_capacity(group.external_inputs.len());
    for &val_id in &group.external_inputs {
        let value = graph
            .values
            .iter()
            .find(|v| v.id == val_id)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("inductor-jit: external input {val_id:?} has no value entry"),
            })?;
        let producer_id = match value.producer {
            Some(p) => p,
            None => return Ok(None),
        };
        let producer = graph
            .nodes
            .iter()
            .find(|n| n.id == producer_id)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("inductor-jit: producer {producer_id:?} not found"),
            })?;
        match &producer.op {
            IrOpKind::Input { index } => {
                sources.push(InputSource::GraphInput(*index));
            }
            IrOpKind::Constant { data, .. } => {
                sources.push(InputSource::Constant(data.clone()));
            }
            _ => {
                // The external input was produced by a non-input, non-constant
                // node. That shouldn't happen for a single-group graph
                // (the producer would have been in the same group), but
                // if it does, bail to the interpreter.
                return Ok(None);
            }
        }
    }

    // Determine output length from the graph's output shape.
    let output_shape = resolve_output_shape(graph)?;
    let output_len: usize = output_shape.iter().product();
    if output_len == 0 {
        return Ok(None);
    }

    // Lower the group to LoopIR and emit C source.
    let loops_per_group = fuse_dag(&groups, graph);
    let kernel_loops = &loops_per_group[0];
    let fn_name = "kernel_0";
    let num_inputs = group.external_inputs.len();
    let kernel_source =
        crate::codegen_cpu::CpuCodegen::generate_c_source(kernel_loops, fn_name, num_inputs);

    // Compile through the JIT pipeline (with cache).
    let kernel =
        crate::codegen_jit::compile_c_kernel(&kernel_source, fn_name, num_inputs, output_len)?;

    // Capture everything we need for the execute closure.
    let num_graph_inputs = graph.input_values.len();
    let input_shapes = collect_input_shapes(graph);

    let execute = move |inputs: &[Vec<f64>]| -> FerrotorchResult<Vec<f64>> {
        if inputs.len() != num_graph_inputs {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "inductor-jit: expected {} graph inputs, got {}",
                    num_graph_inputs,
                    inputs.len()
                ),
            });
        }
        // Optional: validate input lengths against declared shapes.
        for (i, (data, shape)) in inputs.iter().zip(input_shapes.iter()).enumerate() {
            let expected: usize = shape.iter().product();
            if expected != 0 && data.len() != expected {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "inductor-jit: input {i} expected {expected} elements \
                         (shape {shape:?}), got {}",
                        data.len()
                    ),
                });
            }
        }

        // Build the kernel's per-external-input buffer list.
        let mut kernel_inputs: Vec<&[f64]> = Vec::with_capacity(sources.len());
        for src in &sources {
            match src {
                InputSource::GraphInput(idx) => {
                    let buf = inputs
                        .get(*idx)
                        .ok_or_else(|| FerrotorchError::InvalidArgument {
                            message: format!("inductor-jit: graph input {idx} out of range"),
                        })?;
                    kernel_inputs.push(buf.as_slice());
                }
                InputSource::Constant(data) => {
                    kernel_inputs.push(data.as_slice());
                }
            }
        }

        let mut output = vec![0.0f64; output_len];
        kernel.execute(&kernel_inputs, &mut output)?;
        Ok(output)
    };

    Ok(Some(CompiledGraph {
        execute: Box::new(execute),
        num_inputs: num_graph_inputs,
        output_shape,
    }))
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
        let (_, pow_outs) = g.add_node(IrOpKind::Pow { exponent: 2.0 }, vec![x], vec![vec![3]]);
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

        let interp_result = InterpreterBackend
            .compile(&g)
            .unwrap()
            .execute(std::slice::from_ref(&input))
            .unwrap();
        let native_result = NativeBackend
            .compile(&g)
            .unwrap()
            .execute(&[input])
            .unwrap();

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

        let interp_result = InterpreterBackend
            .compile(&g)
            .unwrap()
            .execute(std::slice::from_ref(&input))
            .unwrap();
        let native_result = NativeBackend
            .compile(&g)
            .unwrap()
            .execute(&[input])
            .unwrap();

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

    // -----------------------------------------------------------------------
    // InductorBackend tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inductor_backend_name() {
        assert_eq!(
            InductorBackend::new(InductorTarget::CpuRust).name(),
            "inductor-cpu-rust"
        );
        assert_eq!(
            InductorBackend::new(InductorTarget::CpuC).name(),
            "inductor-cpu-c"
        );
        assert_eq!(
            InductorBackend::new(InductorTarget::GpuPtx).name(),
            "inductor-gpu-ptx"
        );
        assert_eq!(
            InductorBackend::new(InductorTarget::GpuCuda).name(),
            "inductor-gpu-cuda"
        );
    }

    #[test]
    fn test_inductor_compile_runs_interpreter_fallback() {
        // Graph: y = x + x
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let backend = InductorBackend::new(InductorTarget::CpuRust);
        let compiled = backend.compile(&g).unwrap();
        let result = compiled.execute(&[vec![1.0, 2.0, 3.0]]).unwrap();
        assert_close(&result, &[2.0, 4.0, 6.0], 1e-10);
    }

    #[test]
    fn test_inductor_generate_rust() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let backend = InductorBackend::new(InductorTarget::CpuRust);
        let sources = backend.generate(&g).unwrap();

        assert!(!sources.is_empty());
        let src = &sources[0];
        assert!(src.contains("pub unsafe fn"));
        assert!(src.contains("for "));
    }

    #[test]
    fn test_inductor_generate_c() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let backend = InductorBackend::new(InductorTarget::CpuC);
        let sources = backend.generate(&g).unwrap();

        assert!(!sources.is_empty());
        let src = &sources[0];
        assert!(src.contains("#include <math.h>"));
        assert!(src.contains("void kernel_0"));
        assert!(src.contains("restrict"));
    }

    #[test]
    fn test_inductor_generate_cuda() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![1024]);
        let (_, sigmoid_outs) = g.add_node(IrOpKind::Sigmoid, vec![x], vec![vec![1024]]);
        g.set_outputs(vec![sigmoid_outs[0]]);

        let backend = InductorBackend::new(InductorTarget::GpuCuda);
        let sources = backend.generate(&g).unwrap();

        assert!(!sources.is_empty());
        let src = &sources[0];
        assert!(src.contains("__global__"));
        assert!(src.contains("blockIdx"));
        assert!(src.contains("threadIdx"));
    }

    #[test]
    fn test_inductor_generate_ptx() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![1024]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![1024]]);
        g.set_outputs(vec![neg_outs[0]]);

        let backend = InductorBackend::with_block_size(InductorTarget::GpuPtx, 512);
        let sources = backend.generate(&g).unwrap();

        assert!(!sources.is_empty());
        let src = &sources[0];
        assert!(src.contains(".version 7.0"));
        assert!(src.contains(".target sm_52"));
        assert!(src.contains("neg.f32"));
        assert!(src.contains("recommended block size: 512"));
    }

    #[test]
    fn test_inductor_multiple_groups() {
        // Graph: x -> relu -> sum -> output
        // This should produce two fusion groups: elementwise (relu) and reduction (sum)
        let mut g = IrGraph::new();
        let x = g.add_input(vec![8]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![8]]);
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![relu_outs[0]], vec![vec![1]]);
        g.set_outputs(vec![sum_outs[0]]);

        let backend = InductorBackend::new(InductorTarget::CpuRust);
        let sources = backend.generate(&g).unwrap();

        assert!(
            sources.len() >= 2,
            "expected at least 2 fusion groups, got {}",
            sources.len()
        );
    }

    #[test]
    fn test_inductor_matches_interpreter() {
        // Graph: y = relu(neg(x))
        let mut g = IrGraph::new();
        let x = g.add_input(vec![5]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![5]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![5]]);
        g.set_outputs(vec![relu_outs[0]]);

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        let interp_result = InterpreterBackend
            .compile(&g)
            .unwrap()
            .execute(std::slice::from_ref(&input))
            .unwrap();
        let inductor_result = InductorBackend::new(InductorTarget::CpuRust)
            .compile(&g)
            .unwrap()
            .execute(&[input])
            .unwrap();

        assert_close(&inductor_result, &interp_result, 1e-10);
    }

    #[test]
    fn test_inductor_identity_graph() {
        // Identity graph should produce at least one source
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        g.set_outputs(vec![x]);

        let backend = InductorBackend::new(InductorTarget::CpuRust);
        let sources = backend.generate(&g).unwrap();
        assert!(!sources.is_empty());
    }

    #[test]
    fn test_inductor_fused_chain_codegen() {
        // x -> neg -> relu -> sigmoid -> output
        // All should fuse into one kernel
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![sig_outs[0]]);

        // Test all four targets
        for target in [
            InductorTarget::CpuRust,
            InductorTarget::CpuC,
            InductorTarget::GpuCuda,
            InductorTarget::GpuPtx,
        ] {
            let backend = InductorBackend::new(target);
            let sources = backend.generate(&g).unwrap();
            assert!(!sources.is_empty(), "no sources generated for {:?}", target);
            assert!(!sources[0].is_empty(), "empty source for {:?}", target);
        }
    }

    // -----------------------------------------------------------------------
    // Inductor CpuC JIT tests
    //
    // These exercise the real end-to-end path: C source is generated,
    // handed to the system C compiler, loaded as a shared library, and
    // invoked via FFI on every `execute` call. They skip gracefully when
    // no C compiler is available on the host.
    // -----------------------------------------------------------------------

    fn have_c_compiler() -> bool {
        // Private to codegen_jit, but reflected here via an env probe on
        // the standard names the module checks.
        if std::env::var("CC")
            .ok()
            .and_then(|cc| {
                std::env::split_paths(&std::env::var_os("PATH")?).find(|p| p.join(&cc).is_file())
            })
            .is_some()
        {
            return true;
        }
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                for cand in ["cc", "gcc", "clang"] {
                    if dir.join(cand).is_file() {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[test]
    fn test_inductor_cpuc_jit_unary_chain() {
        if !have_c_compiler() {
            eprintln!("skipping JIT test: no C compiler");
            return;
        }
        // y = relu(neg(x))
        let mut g = IrGraph::new();
        let x = g.add_input(vec![5]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![5]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![5]]);
        g.set_outputs(vec![relu_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();
        assert_eq!(compiled.num_inputs(), 1);
        assert_eq!(compiled.output_shape(), &[5]);

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let out = compiled.execute(&[input]).unwrap();
        // relu(neg([-2,-1,0,1,2])) = relu([2,1,0,-1,-2]) = [2,1,0,0,0]
        assert_close(&out, &[2.0, 1.0, 0.0, 0.0, 0.0], 1e-10);
    }

    #[test]
    fn test_inductor_cpuc_jit_two_input_add() {
        if !have_c_compiler() {
            return;
        }
        // y = a + b
        let mut g = IrGraph::new();
        let a = g.add_input(vec![4]);
        let b = g.add_input(vec![4]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![4]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();
        assert_eq!(compiled.num_inputs(), 2);
        let out = compiled
            .execute(&[vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]])
            .unwrap();
        assert_close(&out, &[11.0, 22.0, 33.0, 44.0], 1e-10);
    }

    #[test]
    fn test_inductor_cpuc_jit_matches_interpreter() {
        if !have_c_compiler() {
            return;
        }
        // y = sigmoid(tanh(neg(x)))
        let mut g = IrGraph::new();
        let x = g.add_input(vec![6]);
        let (_, n_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![6]]);
        let (_, t_outs) = g.add_node(IrOpKind::Tanh, vec![n_outs[0]], vec![vec![6]]);
        let (_, s_outs) = g.add_node(IrOpKind::Sigmoid, vec![t_outs[0]], vec![vec![6]]);
        g.set_outputs(vec![s_outs[0]]);

        let input = vec![-3.0, -1.5, -0.5, 0.5, 1.5, 3.0];

        let interp = InterpreterBackend
            .compile(&g)
            .unwrap()
            .execute(std::slice::from_ref(&input))
            .unwrap();
        let jit = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap()
            .execute(&[input])
            .unwrap();

        assert_close(&jit, &interp, 1e-9);
    }

    #[test]
    fn test_inductor_cpuc_jit_constant_input() {
        if !have_c_compiler() {
            return;
        }
        // y = x + [10, 20, 30]  — one graph input + one constant
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![10.0, 20.0, 30.0], vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, c], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();
        assert_eq!(compiled.num_inputs(), 1);
        let out = compiled.execute(&[vec![1.0, 2.0, 3.0]]).unwrap();
        assert_close(&out, &[11.0, 22.0, 33.0], 1e-10);
    }

    #[test]
    fn test_inductor_cpuc_jit_rejects_wrong_input_count() {
        if !have_c_compiler() {
            return;
        }
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![3]]);
        g.set_outputs(vec![neg_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();
        assert!(compiled.execute(&[]).is_err());
        assert!(
            compiled
                .execute(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
                .is_err()
        );
    }

    #[test]
    fn test_inductor_cpuc_jit_rejects_wrong_input_shape() {
        if !have_c_compiler() {
            return;
        }
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
        g.set_outputs(vec![neg_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();
        // Graph declares 4-element input; supplying 3 is an error.
        assert!(compiled.execute(&[vec![1.0, 2.0, 3.0]]).is_err());
    }

    #[test]
    fn test_inductor_cpuc_jit_reduction_falls_back_to_interpreter() {
        if !have_c_compiler() {
            return;
        }
        // x -> sum (single reduction, not elementwise) → JIT path not taken
        // → InterpreterBackend fallback produces the correct scalar.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![x], vec![vec![1]]);
        g.set_outputs(vec![sum_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();
        let out = compiled.execute(&[vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        assert_close(&out, &[10.0], 1e-10);
    }

    #[test]
    fn test_inductor_cpuc_jit_matmul_falls_back_to_interpreter() {
        if !have_c_compiler() {
            return;
        }
        // Matmul is not elementwise → falls back to the interpreter.
        let mut g = IrGraph::new();
        let a = g.add_input(vec![2, 3]);
        let b = g.add_input(vec![3, 2]);
        let (_, mm_outs) = g.add_node(IrOpKind::Matmul, vec![a, b], vec![vec![2, 2]]);
        g.set_outputs(vec![mm_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();
        // Simple matmul: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
        //   = [[1+6+15, 2+8+18], [4+15+30, 8+20+36]]
        //   = [[22, 28], [49, 64]]
        let out = compiled
            .execute(&[
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ])
            .unwrap();
        assert_close(&out, &[22.0, 28.0, 49.0, 64.0], 1e-10);
    }

    #[test]
    fn test_inductor_cpuc_jit_repeated_execution() {
        if !have_c_compiler() {
            return;
        }
        // Compile once, run many times — each call should produce the
        // correct output without re-invoking the compiler (cache-hit).
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![3]]);
        g.set_outputs(vec![neg_outs[0]]);

        let compiled = InductorBackend::new(InductorTarget::CpuC)
            .compile(&g)
            .unwrap();

        for _ in 0..5 {
            let out = compiled.execute(&[vec![1.0, -2.0, 3.5]]).unwrap();
            assert_close(&out, &[-1.0, 2.0, -3.5], 1e-10);
        }
    }
}
