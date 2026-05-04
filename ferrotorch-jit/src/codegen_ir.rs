//! Low-level loop-based IR for code generation.
//!
//! [`LoopIR`] sits below the high-level [`IrOpKind`](crate::graph::IrOpKind)
//! graph representation and above the textual code emitters.  Lowering from
//! the graph IR to `LoopIR` makes the iteration structure explicit (loop nests,
//! index arithmetic, memory accesses) so that the CPU and GPU code generators
//! can emit target-specific source text without re-deriving this information.
//!
//! The lowering path is:
//!
//! ```text
//! IrGraph  -->  FusionGroup[]  -->  LoopIR[]  -->  {Rust, C, CUDA, PTX} source
//! ```

use std::fmt;

use crate::graph::IrOpKind;

// ---------------------------------------------------------------------------
// Expression AST
// ---------------------------------------------------------------------------

/// A scalar expression in the loop body.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A named variable (loop index, accumulator, temp).
    Var(String),
    /// A floating-point literal.
    Const(f64),
    /// An integer literal (for indices, sizes).
    IntConst(i64),
    /// Binary operation.
    BinOp {
        /// The binary operator.
        op: BinOpKind,
        /// Left-hand operand expression.
        lhs: Box<Expr>,
        /// Right-hand operand expression.
        rhs: Box<Expr>,
    },
    /// Unary operation.
    UnaryOp {
        /// The unary operator.
        op: UnaryOpKind,
        /// The single operand expression.
        operand: Box<Expr>,
    },
    /// Named function call (e.g. `expf`, `logf`).
    FnCall {
        /// Function identifier as it appears in the emitted source.
        name: String,
        /// Positional arguments.
        args: Vec<Expr>,
    },
    /// Indexed load: `buffer[index]`.
    Index {
        /// Name of the buffer being indexed.
        buffer: String,
        /// Linear-index expression.
        index: Box<Expr>,
    },
    /// Cast expression (used for index <-> float conversions).
    Cast {
        /// Target type name as a backend-source string (e.g. `"f32"`, `"i64"`).
        target_type: String,
        /// Expression being cast.
        operand: Box<Expr>,
    },
}

/// Binary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    /// `lhs + rhs`.
    Add,
    /// `lhs - rhs`.
    Sub,
    /// `lhs * rhs`.
    Mul,
    /// `lhs / rhs`.
    Div,
    /// `lhs % rhs`.
    Mod,
}

/// Unary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOpKind {
    /// Arithmetic negation.
    Neg,
    /// Natural exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Rectified linear unit.
    Relu,
    /// Gaussian error linear unit.
    Gelu,
    /// Sigmoid-weighted linear unit.
    Silu,
}

// ---------------------------------------------------------------------------
// Loop IR statements
// ---------------------------------------------------------------------------

/// Low-level loop-based IR for code generation.
///
/// Each variant represents a statement in the generated code.  A complete
/// kernel is expressed as `Vec<LoopIR>`.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopIR {
    /// A loop: `for var in start..end { body }`.
    Loop {
        /// Name of the induction variable.
        var: String,
        /// Inclusive lower bound expression.
        start: Expr,
        /// Exclusive upper bound expression.
        end: Expr,
        /// Statements executed for each iteration.
        body: Vec<LoopIR>,
    },
    /// Store a value to an output buffer: `buffer[index] = value`.
    Store {
        /// Name of the destination buffer.
        buffer: String,
        /// Linear index into the buffer.
        index: Expr,
        /// Value expression to write.
        value: Expr,
    },
    /// Declare and initialise a local variable: `let var = value`.
    Let {
        /// Name of the local being declared.
        var: String,
        /// Initialiser expression.
        value: Expr,
    },
    /// Assign to an existing local: `var = value`.
    Assign {
        /// Name of the local being reassigned.
        var: String,
        /// New value expression.
        value: Expr,
    },
    /// Accumulate: `var += value`.
    Accumulate {
        /// Name of the accumulator local.
        var: String,
        /// Value expression added to the accumulator.
        value: Expr,
    },
    /// Conditional: `if condition { then_body } else { else_body }`.
    If {
        /// Boolean predicate expression.
        condition: Expr,
        /// Statements executed when `condition` is true.
        then_body: Vec<LoopIR>,
        /// Statements executed when `condition` is false (may be empty).
        else_body: Vec<LoopIR>,
    },
    /// A comment in the generated code (useful for debugging).
    Comment(String),
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl fmt::Display for BinOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOpKind::Add => write!(f, "+"),
            BinOpKind::Sub => write!(f, "-"),
            BinOpKind::Mul => write!(f, "*"),
            BinOpKind::Div => write!(f, "/"),
            BinOpKind::Mod => write!(f, "%"),
        }
    }
}

impl fmt::Display for UnaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOpKind::Neg => write!(f, "neg"),
            UnaryOpKind::Exp => write!(f, "exp"),
            UnaryOpKind::Log => write!(f, "log"),
            UnaryOpKind::Sqrt => write!(f, "sqrt"),
            UnaryOpKind::Abs => write!(f, "abs"),
            UnaryOpKind::Sigmoid => write!(f, "sigmoid"),
            UnaryOpKind::Tanh => write!(f, "tanh"),
            UnaryOpKind::Relu => write!(f, "relu"),
            UnaryOpKind::Gelu => write!(f, "gelu"),
            UnaryOpKind::Silu => write!(f, "silu"),
        }
    }
}

// ---------------------------------------------------------------------------
// Expr builder helpers
// ---------------------------------------------------------------------------

impl Expr {
    /// Shorthand for `Expr::Var(name.into())`.
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    /// Shorthand for `Expr::Const(v)`.
    pub fn constant(v: f64) -> Self {
        Expr::Const(v)
    }

    /// Shorthand for `Expr::IntConst(v)`.
    pub fn int(v: i64) -> Self {
        Expr::IntConst(v)
    }

    /// Create a binary operation expression.
    pub fn bin(op: BinOpKind, lhs: Expr, rhs: Expr) -> Self {
        Expr::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Create a unary operation expression.
    pub fn unary(op: UnaryOpKind, operand: Expr) -> Self {
        Expr::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }

    /// Create an index expression: `buffer[index]`.
    pub fn index(buffer: impl Into<String>, idx: Expr) -> Self {
        Expr::Index {
            buffer: buffer.into(),
            index: Box::new(idx),
        }
    }

    /// Create a function call expression.
    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::FnCall {
            name: name.into(),
            args,
        }
    }

    /// Create `lhs + rhs`.
    pub fn sum(lhs: Expr, rhs: Expr) -> Self {
        Expr::bin(BinOpKind::Add, lhs, rhs)
    }

    /// Create `lhs * rhs`.
    pub fn prod(lhs: Expr, rhs: Expr) -> Self {
        Expr::bin(BinOpKind::Mul, lhs, rhs)
    }
}

// ---------------------------------------------------------------------------
// IrOpKind -> UnaryOpKind conversion
// ---------------------------------------------------------------------------

/// Try to convert a high-level `IrOpKind` into a `UnaryOpKind`.
///
/// Returns `None` for ops that are not unary elementwise (binary ops,
/// reductions, linalg, etc.).
pub fn ir_op_to_unary(op: &IrOpKind) -> Option<UnaryOpKind> {
    match op {
        IrOpKind::Neg => Some(UnaryOpKind::Neg),
        IrOpKind::Exp => Some(UnaryOpKind::Exp),
        IrOpKind::Log => Some(UnaryOpKind::Log),
        IrOpKind::Sqrt => Some(UnaryOpKind::Sqrt),
        IrOpKind::Abs => Some(UnaryOpKind::Abs),
        IrOpKind::Sigmoid => Some(UnaryOpKind::Sigmoid),
        IrOpKind::Tanh => Some(UnaryOpKind::Tanh),
        IrOpKind::Relu => Some(UnaryOpKind::Relu),
        IrOpKind::Gelu => Some(UnaryOpKind::Gelu),
        IrOpKind::Silu => Some(UnaryOpKind::Silu),
        _ => None,
    }
}

/// Try to convert a high-level `IrOpKind` into a `BinOpKind`.
pub fn ir_op_to_binary(op: &IrOpKind) -> Option<BinOpKind> {
    match op {
        IrOpKind::Add => Some(BinOpKind::Add),
        IrOpKind::Sub => Some(BinOpKind::Sub),
        IrOpKind::Mul => Some(BinOpKind::Mul),
        IrOpKind::Div => Some(BinOpKind::Div),
        _ => None,
    }
}

/// Returns `true` if the op is a unary elementwise operation.
pub fn is_unary_elementwise(op: &IrOpKind) -> bool {
    ir_op_to_unary(op).is_some() || matches!(op, IrOpKind::Pow { .. })
}

/// Returns `true` if the op is a binary elementwise operation.
pub fn is_binary_elementwise(op: &IrOpKind) -> bool {
    ir_op_to_binary(op).is_some()
}

/// Returns `true` if the op is any kind of elementwise operation
/// (unary or binary).
pub fn is_elementwise(op: &IrOpKind) -> bool {
    is_unary_elementwise(op) || is_binary_elementwise(op)
}

/// Returns `true` if the op is a reduction (Sum, Mean, Prod).
pub fn is_reduction(op: &IrOpKind) -> bool {
    matches!(op, IrOpKind::Sum | IrOpKind::Mean | IrOpKind::Prod)
}

// ---------------------------------------------------------------------------
// Lowering: apply a single op to an expression
// ---------------------------------------------------------------------------

/// Given an expression representing the current value, apply a unary or
/// parameterized elementwise operation and return the resulting expression.
pub fn apply_op_expr(val: Expr, op: &IrOpKind) -> Option<Expr> {
    match op {
        IrOpKind::Neg => Some(Expr::unary(UnaryOpKind::Neg, val)),
        IrOpKind::Exp => Some(Expr::unary(UnaryOpKind::Exp, val)),
        IrOpKind::Log => Some(Expr::unary(UnaryOpKind::Log, val)),
        IrOpKind::Sqrt => Some(Expr::unary(UnaryOpKind::Sqrt, val)),
        IrOpKind::Abs => Some(Expr::unary(UnaryOpKind::Abs, val)),
        IrOpKind::Sigmoid => Some(Expr::unary(UnaryOpKind::Sigmoid, val)),
        IrOpKind::Tanh => Some(Expr::unary(UnaryOpKind::Tanh, val)),
        IrOpKind::Relu => Some(Expr::unary(UnaryOpKind::Relu, val)),
        IrOpKind::Gelu => Some(Expr::unary(UnaryOpKind::Gelu, val)),
        IrOpKind::Silu => Some(Expr::unary(UnaryOpKind::Silu, val)),
        IrOpKind::Pow { exponent } => {
            Some(Expr::call("powf", vec![val, Expr::constant(*exponent)]))
        }
        _ => None,
    }
}

/// Apply a binary op between `lhs` and `rhs`.
pub fn apply_binary_op_expr(lhs: Expr, rhs: Expr, op: &IrOpKind) -> Option<Expr> {
    ir_op_to_binary(op).map(|bin_op| Expr::bin(bin_op, lhs, rhs))
}

// ---------------------------------------------------------------------------
// Lowering: IrOpKind -> LoopIR
// ---------------------------------------------------------------------------

/// Lower a sequence of high-level IR operations into loop-based IR.
///
/// This is the main entry point for converting from the graph level to the
/// loop level.  It handles:
///
/// - Elementwise ops (unary and binary): single flat loop
/// - Reductions (sum, mean, prod): accumulator pattern
/// - Matmul/Mm: triple-nested loop
/// - Fused elementwise: single loop with chained ops
///
/// # Arguments
///
/// * `ops` - The IR operations, in the order they should be applied.
/// * `input_names` - Buffer names for each input (e.g. `["in0", "in1"]`).
/// * `output_name` - Buffer name for the output.
/// * `numel` - Total number of elements in the iteration domain.
pub fn lower_to_loops(
    ops: &[IrOpKind],
    input_names: &[&str],
    output_name: &str,
    numel: usize,
) -> Vec<LoopIR> {
    if ops.is_empty() {
        return Vec::new();
    }

    // Special case: single op
    if ops.len() == 1 {
        return lower_single_op(&ops[0], input_names, output_name, numel);
    }

    // Fuse all elementwise ops (unary and binary) into a single loop.
    let all_elementwise = ops.iter().all(is_elementwise);
    if all_elementwise {
        return lower_fused_elementwise(ops, input_names, output_name, numel);
    }

    // Otherwise, lower each op individually and concatenate.
    let mut result = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let in_names: Vec<&str> = if i == 0 {
            input_names.to_vec()
        } else {
            // Intermediate: previous output becomes input
            vec![output_name]
        };
        let mut lowered = lower_single_op(op, &in_names, output_name, numel);
        result.append(&mut lowered);
    }
    result
}

/// Lower a single IR operation into loop-based IR.
fn lower_single_op(
    op: &IrOpKind,
    input_names: &[&str],
    output_name: &str,
    numel: usize,
) -> Vec<LoopIR> {
    match op {
        // Unary elementwise
        op if is_unary_elementwise(op) => {
            let in_name = input_names.first().copied().unwrap_or("in0");
            lower_unary_elementwise(op, in_name, output_name, numel)
        }

        // Binary elementwise
        op if is_binary_elementwise(op) => {
            let in0 = input_names.first().copied().unwrap_or("in0");
            let in1 = input_names.get(1).copied().unwrap_or("in1");
            lower_binary_elementwise(op, in0, in1, output_name, numel)
        }

        // Reductions
        IrOpKind::Sum => {
            let in_name = input_names.first().copied().unwrap_or("in0");
            lower_sum_reduction(in_name, output_name, numel)
        }
        IrOpKind::Mean => {
            let in_name = input_names.first().copied().unwrap_or("in0");
            lower_mean_reduction(in_name, output_name, numel)
        }
        IrOpKind::Prod => {
            let in_name = input_names.first().copied().unwrap_or("in0");
            lower_prod_reduction(in_name, output_name, numel)
        }

        // Fused elementwise
        IrOpKind::FusedElementwise { ops } => {
            lower_fused_elementwise(ops, input_names, output_name, numel)
        }

        // Anything else: emit a comment noting it cannot be lowered
        _ => {
            vec![LoopIR::Comment(format!(
                "unsupported op for loop lowering: {op:?}"
            ))]
        }
    }
}

/// Lower a unary elementwise op to a single flat loop.
fn lower_unary_elementwise(
    op: &IrOpKind,
    in_name: &str,
    out_name: &str,
    numel: usize,
) -> Vec<LoopIR> {
    let idx = Expr::var("i");
    let load = Expr::index(in_name, idx.clone());
    let applied = apply_op_expr(load, op);

    match applied {
        Some(expr) => vec![LoopIR::Loop {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(numel as i64),
            body: vec![LoopIR::Store {
                buffer: out_name.into(),
                index: Expr::var("i"),
                value: expr,
            }],
        }],
        None => vec![LoopIR::Comment(format!("failed to lower unary op: {op:?}"))],
    }
}

/// Lower a binary elementwise op to a single flat loop.
fn lower_binary_elementwise(
    op: &IrOpKind,
    in0: &str,
    in1: &str,
    out_name: &str,
    numel: usize,
) -> Vec<LoopIR> {
    let bin_op = match ir_op_to_binary(op) {
        Some(b) => b,
        None => {
            return vec![LoopIR::Comment(format!(
                "failed to lower binary op: {op:?}"
            ))];
        }
    };

    let idx = Expr::var("i");
    let load_a = Expr::index(in0, idx.clone());
    let load_b = Expr::index(in1, idx.clone());
    let expr = Expr::bin(bin_op, load_a, load_b);

    vec![LoopIR::Loop {
        var: "i".into(),
        start: Expr::int(0),
        end: Expr::int(numel as i64),
        body: vec![LoopIR::Store {
            buffer: out_name.into(),
            index: Expr::var("i"),
            value: expr,
        }],
    }]
}

/// Lower a sum reduction: `acc = 0; for i in 0..n { acc += in[i]; } out[0] = acc;`
fn lower_sum_reduction(in_name: &str, out_name: &str, numel: usize) -> Vec<LoopIR> {
    vec![
        LoopIR::Let {
            var: "acc".into(),
            value: Expr::constant(0.0),
        },
        LoopIR::Loop {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(numel as i64),
            body: vec![LoopIR::Accumulate {
                var: "acc".into(),
                value: Expr::index(in_name, Expr::var("i")),
            }],
        },
        LoopIR::Store {
            buffer: out_name.into(),
            index: Expr::int(0),
            value: Expr::var("acc"),
        },
    ]
}

/// Lower a mean reduction: sum then divide by count.
fn lower_mean_reduction(in_name: &str, out_name: &str, numel: usize) -> Vec<LoopIR> {
    vec![
        LoopIR::Let {
            var: "acc".into(),
            value: Expr::constant(0.0),
        },
        LoopIR::Loop {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(numel as i64),
            body: vec![LoopIR::Accumulate {
                var: "acc".into(),
                value: Expr::index(in_name, Expr::var("i")),
            }],
        },
        LoopIR::Store {
            buffer: out_name.into(),
            index: Expr::int(0),
            value: Expr::bin(
                BinOpKind::Div,
                Expr::var("acc"),
                Expr::constant(numel as f64),
            ),
        },
    ]
}

/// Lower a prod reduction: `acc = 1; for i in 0..n { acc *= in[i]; } out[0] = acc;`
fn lower_prod_reduction(in_name: &str, out_name: &str, numel: usize) -> Vec<LoopIR> {
    vec![
        LoopIR::Let {
            var: "acc".into(),
            value: Expr::constant(1.0),
        },
        LoopIR::Loop {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(numel as i64),
            body: vec![LoopIR::Assign {
                var: "acc".into(),
                value: Expr::bin(
                    BinOpKind::Mul,
                    Expr::var("acc"),
                    Expr::index(in_name, Expr::var("i")),
                ),
            }],
        },
        LoopIR::Store {
            buffer: out_name.into(),
            index: Expr::int(0),
            value: Expr::var("acc"),
        },
    ]
}

/// Lower a chain of fused elementwise operations into a single loop.
///
/// Each operation is applied in sequence to the running value, and the
/// final result is stored.  This eliminates intermediate memory traffic.
fn lower_fused_elementwise(
    ops: &[IrOpKind],
    input_names: &[&str],
    out_name: &str,
    numel: usize,
) -> Vec<LoopIR> {
    let mut body = Vec::new();

    // Load all inputs at the start of the loop body.
    for (i, &name) in input_names.iter().enumerate() {
        body.push(LoopIR::Let {
            var: format!("in{i}_val"),
            value: Expr::index(name, Expr::var("i")),
        });
    }

    // Initialize the accumulator with the first input.
    body.push(LoopIR::Let {
        var: "val".into(),
        value: Expr::var("in0_val"),
    });

    // Apply each op. Binary ops consume the next available input.
    let mut next_input = 1usize; // in0 is the initial accumulator
    for op in ops {
        if is_binary_elementwise(op) {
            // Binary op: accumulator <op> next_input
            let rhs_var = format!("in{next_input}_val");
            next_input += 1;
            match apply_binary_op_expr(Expr::var("val"), Expr::var(&rhs_var), op) {
                Some(expr) => {
                    body.push(LoopIR::Assign {
                        var: "val".into(),
                        value: expr,
                    });
                }
                None => {
                    body.push(LoopIR::Comment(format!(
                        "skipped unsupported binary op: {op:?}"
                    )));
                }
            }
        } else {
            // Unary op: applied to the accumulator
            match apply_op_expr(Expr::var("val"), op) {
                Some(expr) => {
                    body.push(LoopIR::Assign {
                        var: "val".into(),
                        value: expr,
                    });
                }
                None => {
                    body.push(LoopIR::Comment(format!(
                        "skipped unsupported unary op: {op:?}"
                    )));
                }
            }
        }
    }

    // Store the result
    body.push(LoopIR::Store {
        buffer: out_name.into(),
        index: Expr::var("i"),
        value: Expr::var("val"),
    });

    vec![LoopIR::Loop {
        var: "i".into(),
        start: Expr::int(0),
        end: Expr::int(numel as i64),
        body,
    }]
}

/// Lower a matrix multiplication to a triple-nested loop.
///
/// Computes `out[m, n] = sum_k(a[m, k] * b[k, n])` for
/// `a` of shape `[M, K]` and `b` of shape `[K, N]`.
pub fn lower_matmul(
    in_a: &str,
    in_b: &str,
    out_name: &str,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<LoopIR> {
    // for i in 0..M:
    //   for j in 0..N:
    //     let acc = 0.0
    //     for p in 0..K:
    //       acc += a[i * K + p] * b[p * N + j]
    //     out[i * N + j] = acc
    vec![LoopIR::Loop {
        var: "i".into(),
        start: Expr::int(0),
        end: Expr::int(m as i64),
        body: vec![LoopIR::Loop {
            var: "j".into(),
            start: Expr::int(0),
            end: Expr::int(n as i64),
            body: vec![
                LoopIR::Let {
                    var: "acc".into(),
                    value: Expr::constant(0.0),
                },
                LoopIR::Loop {
                    var: "p".into(),
                    start: Expr::int(0),
                    end: Expr::int(k as i64),
                    body: vec![LoopIR::Accumulate {
                        var: "acc".into(),
                        value: Expr::bin(
                            BinOpKind::Mul,
                            Expr::index(
                                in_a,
                                Expr::sum(
                                    Expr::prod(Expr::var("i"), Expr::int(k as i64)),
                                    Expr::var("p"),
                                ),
                            ),
                            Expr::index(
                                in_b,
                                Expr::sum(
                                    Expr::prod(Expr::var("p"), Expr::int(n as i64)),
                                    Expr::var("j"),
                                ),
                            ),
                        ),
                    }],
                },
                LoopIR::Store {
                    buffer: out_name.into(),
                    index: Expr::sum(
                        Expr::prod(Expr::var("i"), Expr::int(n as i64)),
                        Expr::var("j"),
                    ),
                    value: Expr::var("acc"),
                },
            ],
        }],
    }]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary Expr::Const test value, not π.
    fn test_expr_builders() {
        let v = Expr::var("x");
        assert_eq!(v, Expr::Var("x".into()));

        let c = Expr::constant(3.14);
        assert_eq!(c, Expr::Const(3.14));

        let i = Expr::int(42);
        assert_eq!(i, Expr::IntConst(42));

        let sum = Expr::sum(Expr::var("a"), Expr::var("b"));
        assert_eq!(
            sum,
            Expr::BinOp {
                op: BinOpKind::Add,
                lhs: Box::new(Expr::Var("a".into())),
                rhs: Box::new(Expr::Var("b".into())),
            }
        );
    }

    #[test]
    fn test_ir_op_to_unary_conversion() {
        assert_eq!(ir_op_to_unary(&IrOpKind::Neg), Some(UnaryOpKind::Neg));
        assert_eq!(ir_op_to_unary(&IrOpKind::Relu), Some(UnaryOpKind::Relu));
        assert_eq!(
            ir_op_to_unary(&IrOpKind::Sigmoid),
            Some(UnaryOpKind::Sigmoid)
        );
        assert_eq!(ir_op_to_unary(&IrOpKind::Add), None);
        assert_eq!(ir_op_to_unary(&IrOpKind::Sum), None);
    }

    #[test]
    fn test_ir_op_to_binary_conversion() {
        assert_eq!(ir_op_to_binary(&IrOpKind::Add), Some(BinOpKind::Add));
        assert_eq!(ir_op_to_binary(&IrOpKind::Sub), Some(BinOpKind::Sub));
        assert_eq!(ir_op_to_binary(&IrOpKind::Mul), Some(BinOpKind::Mul));
        assert_eq!(ir_op_to_binary(&IrOpKind::Div), Some(BinOpKind::Div));
        assert_eq!(ir_op_to_binary(&IrOpKind::Neg), None);
    }

    #[test]
    fn test_is_elementwise_classification() {
        assert!(is_unary_elementwise(&IrOpKind::Neg));
        assert!(is_unary_elementwise(&IrOpKind::Relu));
        assert!(is_unary_elementwise(&IrOpKind::Pow { exponent: 2.0 }));
        assert!(!is_unary_elementwise(&IrOpKind::Add));
        assert!(!is_unary_elementwise(&IrOpKind::Sum));

        assert!(is_binary_elementwise(&IrOpKind::Add));
        assert!(!is_binary_elementwise(&IrOpKind::Neg));

        assert!(is_elementwise(&IrOpKind::Add));
        assert!(is_elementwise(&IrOpKind::Neg));
        assert!(!is_elementwise(&IrOpKind::Sum));
        assert!(!is_elementwise(&IrOpKind::Matmul));
    }

    #[test]
    fn test_is_reduction() {
        assert!(is_reduction(&IrOpKind::Sum));
        assert!(is_reduction(&IrOpKind::Mean));
        assert!(is_reduction(&IrOpKind::Prod));
        assert!(!is_reduction(&IrOpKind::Add));
        assert!(!is_reduction(&IrOpKind::Relu));
    }

    #[test]
    fn test_lower_unary_neg() {
        let loops = lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 8);
        assert_eq!(loops.len(), 1);
        match &loops[0] {
            LoopIR::Loop {
                var,
                start,
                end,
                body,
            } => {
                assert_eq!(var, "i");
                assert_eq!(*start, Expr::int(0));
                assert_eq!(*end, Expr::int(8));
                assert_eq!(body.len(), 1);
                match &body[0] {
                    LoopIR::Store { buffer, .. } => {
                        assert_eq!(buffer, "out");
                    }
                    _ => panic!("expected Store in loop body"),
                }
            }
            _ => panic!("expected Loop"),
        }
    }

    #[test]
    fn test_lower_binary_add() {
        let loops = lower_to_loops(&[IrOpKind::Add], &["a", "b"], "out", 4);
        assert_eq!(loops.len(), 1);
        match &loops[0] {
            LoopIR::Loop { var, end, body, .. } => {
                assert_eq!(var, "i");
                assert_eq!(*end, Expr::int(4));
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected Loop"),
        }
    }

    #[test]
    fn test_lower_sum_reduction() {
        let loops = lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 10);
        // Let + Loop + Store = 3 statements
        assert_eq!(loops.len(), 3);
        match &loops[0] {
            LoopIR::Let { var, value } => {
                assert_eq!(var, "acc");
                assert_eq!(*value, Expr::constant(0.0));
            }
            _ => panic!("expected Let"),
        }
        match &loops[1] {
            LoopIR::Loop { var, end, body, .. } => {
                assert_eq!(var, "i");
                assert_eq!(*end, Expr::int(10));
                assert_eq!(body.len(), 1);
                match &body[0] {
                    LoopIR::Accumulate { var, .. } => assert_eq!(var, "acc"),
                    _ => panic!("expected Accumulate"),
                }
            }
            _ => panic!("expected Loop"),
        }
        match &loops[2] {
            LoopIR::Store { buffer, index, .. } => {
                assert_eq!(buffer, "out");
                assert_eq!(*index, Expr::int(0));
            }
            _ => panic!("expected Store"),
        }
    }

    #[test]
    fn test_lower_mean_reduction() {
        let loops = lower_to_loops(&[IrOpKind::Mean], &["in0"], "out", 5);
        assert_eq!(loops.len(), 3);
        // The final store should divide by 5
        match &loops[2] {
            LoopIR::Store { value, .. } => match value {
                Expr::BinOp { op, rhs, .. } => {
                    assert_eq!(*op, BinOpKind::Div);
                    assert_eq!(**rhs, Expr::constant(5.0));
                }
                _ => panic!("expected BinOp Div for mean"),
            },
            _ => panic!("expected Store"),
        }
    }

    #[test]
    fn test_lower_prod_reduction() {
        let loops = lower_to_loops(&[IrOpKind::Prod], &["in0"], "out", 3);
        assert_eq!(loops.len(), 3);
        // Initial value should be 1.0 for product
        match &loops[0] {
            LoopIR::Let { value, .. } => {
                assert_eq!(*value, Expr::constant(1.0));
            }
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_lower_fused_elementwise() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Relu, IrOpKind::Sigmoid];
        let loops = lower_to_loops(&ops, &["in0"], "out", 4);

        // Should produce a single fused loop
        assert_eq!(loops.len(), 1);
        match &loops[0] {
            LoopIR::Loop { body, .. } => {
                // body: Let(in0_val), Let(val=in0_val), Assign(neg),
                //       Assign(relu), Assign(sigmoid), Store
                assert_eq!(body.len(), 6);
                match &body[0] {
                    LoopIR::Let { var, .. } => assert_eq!(var, "in0_val"),
                    _ => panic!("expected Let(in0_val)"),
                }
                match &body[1] {
                    LoopIR::Let { var, .. } => assert_eq!(var, "val"),
                    _ => panic!("expected Let(val)"),
                }
                match &body[5] {
                    LoopIR::Store { buffer, .. } => assert_eq!(buffer, "out"),
                    _ => panic!("expected Store"),
                }
            }
            _ => panic!("expected Loop"),
        }
    }

    #[test]
    fn test_lower_fused_multi_input() {
        // x + y → relu: binary add fused with unary relu
        let ops = vec![IrOpKind::Add, IrOpKind::Relu];
        let loops = lower_to_loops(&ops, &["a", "b"], "out", 4);

        assert_eq!(loops.len(), 1);
        match &loops[0] {
            LoopIR::Loop { body, .. } => {
                // body: Let(in0_val), Let(in1_val), Let(val=in0_val),
                //       Assign(add), Assign(relu), Store
                assert_eq!(body.len(), 6);
                // The add should reference both inputs
                match &body[3] {
                    LoopIR::Assign { var, value } => {
                        assert_eq!(var, "val");
                        // Should be a BinOp(Add, val, in1_val)
                        matches!(value, Expr::BinOp { .. });
                    }
                    _ => panic!("expected Assign for add"),
                }
            }
            _ => panic!("expected Loop"),
        }
    }

    #[test]
    fn test_lower_matmul() {
        let loops = lower_matmul("a", "b", "out", 2, 3, 4);
        assert_eq!(loops.len(), 1);
        match &loops[0] {
            LoopIR::Loop { var, end, body, .. } => {
                assert_eq!(var, "i");
                assert_eq!(*end, Expr::int(2));
                assert_eq!(body.len(), 1);
                match &body[0] {
                    LoopIR::Loop { var, end, body, .. } => {
                        assert_eq!(var, "j");
                        assert_eq!(*end, Expr::int(4));
                        // Let(acc), Loop(k), Store
                        assert_eq!(body.len(), 3);
                    }
                    _ => panic!("expected inner Loop"),
                }
            }
            _ => panic!("expected outer Loop"),
        }
    }

    #[test]
    fn test_lower_empty_ops() {
        let loops = lower_to_loops(&[], &["in0"], "out", 4);
        assert!(loops.is_empty());
    }

    #[test]
    fn test_apply_op_expr_pow() {
        let val = Expr::var("x");
        let result = apply_op_expr(val, &IrOpKind::Pow { exponent: 3.0 });
        assert!(result.is_some());
        let expr = result.unwrap();
        match expr {
            Expr::FnCall { name, args } => {
                assert_eq!(name, "powf");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Expr::var("x"));
                assert_eq!(args[1], Expr::constant(3.0));
            }
            _ => panic!("expected FnCall for pow"),
        }
    }

    #[test]
    fn test_binop_display() {
        assert_eq!(format!("{}", BinOpKind::Add), "+");
        assert_eq!(format!("{}", BinOpKind::Sub), "-");
        assert_eq!(format!("{}", BinOpKind::Mul), "*");
        assert_eq!(format!("{}", BinOpKind::Div), "/");
        assert_eq!(format!("{}", BinOpKind::Mod), "%");
    }

    #[test]
    fn test_unaryop_display() {
        assert_eq!(format!("{}", UnaryOpKind::Neg), "neg");
        assert_eq!(format!("{}", UnaryOpKind::Exp), "exp");
        assert_eq!(format!("{}", UnaryOpKind::Relu), "relu");
    }
}
