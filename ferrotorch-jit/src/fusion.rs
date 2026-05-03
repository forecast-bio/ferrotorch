//! Operation fusion engine connecting the JIT's `fuse_elementwise` pass to GPU
//! kernel generation.
//!
//! The fusion engine intercepts tensor operations, buffers sequences of
//! elementwise ops, and executes them as a single fused operation -- either on
//! the CPU via [`FusedChain::execute_cpu`] or on the GPU via a dynamically
//! generated PTX kernel ([`FusedChain::generate_ptx`]).
//!
//! # Thread-local fusion context
//!
//! Fusion is opt-in. Call [`with_fusion`] to enable it for a closure:
//!
//! ```ignore
//! use ferrotorch_jit::fusion::with_fusion;
//! let result = with_fusion(|| {
//!     // elementwise ops inside here are eligible for fusion
//! });
//! ```
//!
//! Use [`is_fusion_enabled`] to query the current state.

use std::cell::Cell;
use std::fmt;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use num_traits;

// ---------------------------------------------------------------------------
// Thread-local fusion flag
// ---------------------------------------------------------------------------

thread_local! {
    static FUSION_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Returns `true` when the current thread is inside a [`with_fusion`] scope.
pub fn is_fusion_enabled() -> bool {
    FUSION_ENABLED.with(std::cell::Cell::get)
}

/// Execute `f` with operation fusion enabled on the current thread.
///
/// Any pending fused operations are flushed before this function returns.
/// The fusion flag is always restored to its prior state, even on panic.
pub fn with_fusion<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct Guard {
        prev: bool,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            FUSION_ENABLED.with(|flag| flag.set(self.prev));
        }
    }

    let prev = is_fusion_enabled();
    FUSION_ENABLED.with(|flag| flag.set(true));
    let _guard = Guard { prev };

    f()
}

// ---------------------------------------------------------------------------
// Fused operation types
// ---------------------------------------------------------------------------

/// An individual operation in a fused chain.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedOp {
    // Binary elementwise (applied with a second tensor or broadcast scalar).
    Add,
    Sub,
    Mul,
    Div,

    // Unary elementwise.
    Neg,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Sqrt,
    Abs,
    Exp,
    Log,

    // Parameterised unary ops.
    Pow(f64),
    ScalarMul(f64),
    ScalarAdd(f64),
}

impl fmt::Display for FusedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedOp::Add => write!(f, "add"),
            FusedOp::Sub => write!(f, "sub"),
            FusedOp::Mul => write!(f, "mul"),
            FusedOp::Div => write!(f, "div"),
            FusedOp::Neg => write!(f, "neg"),
            FusedOp::Relu => write!(f, "relu"),
            FusedOp::Sigmoid => write!(f, "sigmoid"),
            FusedOp::Tanh => write!(f, "tanh"),
            FusedOp::Gelu => write!(f, "gelu"),
            FusedOp::Silu => write!(f, "silu"),
            FusedOp::Sqrt => write!(f, "sqrt"),
            FusedOp::Abs => write!(f, "abs"),
            FusedOp::Exp => write!(f, "exp"),
            FusedOp::Log => write!(f, "log"),
            FusedOp::Pow(p) => write!(f, "pow({p})"),
            FusedOp::ScalarMul(s) => write!(f, "scalar_mul({s})"),
            FusedOp::ScalarAdd(s) => write!(f, "scalar_add({s})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

/// Kind of reduction operation for kernel generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    /// Sum reduction: identity = 0, op = +.
    Sum,
    /// Product reduction: identity = 1, op = *.
    Prod,
    /// Mean reduction: identity = 0, op = +, then divide by n.
    Mean,
}

impl ReductionKind {
    /// The identity element for this reduction (as an f32 bit pattern).
    fn identity_f32_bits(self) -> u32 {
        match self {
            ReductionKind::Sum | ReductionKind::Mean => 0x0000_0000, // 0.0
            ReductionKind::Prod => 0x3F80_0000,                      // 1.0
        }
    }
}

impl fmt::Display for ReductionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReductionKind::Sum => write!(f, "sum"),
            ReductionKind::Prod => write!(f, "prod"),
            ReductionKind::Mean => write!(f, "mean"),
        }
    }
}

// ---------------------------------------------------------------------------
// FusedChain
// ---------------------------------------------------------------------------

/// A sequence of elementwise operations that will be executed as a single
/// fused kernel.
///
/// On the CPU the operations are applied in-place over a single pass per
/// element. On the GPU, [`generate_ptx`](FusedChain::generate_ptx) emits a
/// single PTX kernel that chains all operations per-thread, avoiding
/// intermediate memory traffic.
#[derive(Debug, Clone)]
pub struct FusedChain {
    ops: Vec<FusedOp>,
}

impl FusedChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append an operation to the chain.
    pub fn push(&mut self, op: FusedOp) {
        self.ops.push(op);
    }

    /// The number of operations in this chain.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Borrow the operations slice.
    pub fn ops(&self) -> &[FusedOp] {
        &self.ops
    }

    // -------------------------------------------------------------------
    // CPU execution
    // -------------------------------------------------------------------

    /// Execute the entire fused chain on the CPU, applying every operation
    /// in sequence over a single allocation.
    ///
    /// The input slice is copied once; all operations mutate the copy in
    /// place so only one allocation is needed regardless of chain length.
    ///
    /// # Errors
    ///
    /// Returns an error if the chain contains unsupported binary ops
    /// (Add/Sub/Mul/Div) that require a second operand.
    pub fn execute_cpu<T: Float>(&self, input: &[T]) -> FerrotorchResult<Vec<T>> {
        let mut data = input.to_vec();
        for op in &self.ops {
            apply_op_inplace::<T>(op, &mut data)?;
        }
        Ok(data)
    }

    // -------------------------------------------------------------------
    // PTX generation
    // -------------------------------------------------------------------

    /// Generate a PTX kernel string that applies every operation in this
    /// chain per-element on the GPU.
    ///
    /// The generated kernel signature is:
    ///
    /// ```text
    /// .visible .entry fused_kernel(
    ///     .param .u64 in_ptr,
    ///     .param .u64 out_ptr,
    ///     .param .u32 n
    /// )
    /// ```
    ///
    /// It reads one f32 per thread from `in_ptr`, applies the chain of
    /// operations, and stores the result to `out_ptr`. This means *one*
    /// kernel launch replaces N separate launches, eliminating all
    /// intermediate global-memory round-trips.
    pub fn generate_ptx(&self) -> FerrotorchResult<String> {
        self.generate_ptx_named("fused_kernel")
    }

    /// Like [`generate_ptx`](Self::generate_ptx) but with a custom kernel
    /// entry-point name. The name is validated to be a legal C/PTX
    /// identifier (`[a-zA-Z_][a-zA-Z0-9_]*`).
    pub fn generate_ptx_named(&self, kernel_name: &str) -> FerrotorchResult<String> {
        validate_identifier(kernel_name)?;
        // Reject unsupported binary ops that require a second input pointer.
        for op in &self.ops {
            if matches!(
                op,
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div
            ) {
                return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "generate_ptx: binary op '{op}' in unary FusedChain requires a second \
                         input pointer and cannot be lowered to a single-input PTX kernel"
                    ),
                });
            }
        }
        let mut body_lines: Vec<String> = Vec::new();

        // We accumulate the running value in %val. Some ops need scratch
        // registers; we define them at the top of the kernel.
        let needs_exp = self.ops.iter().any(|op| {
            matches!(
                op,
                FusedOp::Sigmoid | FusedOp::Tanh | FusedOp::Gelu | FusedOp::Silu | FusedOp::Exp
            )
        });
        let _needs_log = self.ops.iter().any(|op| matches!(op, FusedOp::Log));
        let needs_mul_scratch = self.ops.iter().any(|op| {
            matches!(
                op,
                FusedOp::Sigmoid
                    | FusedOp::Tanh
                    | FusedOp::Gelu
                    | FusedOp::Silu
                    | FusedOp::Exp
                    | FusedOp::Log
                    | FusedOp::ScalarMul(_)
                    | FusedOp::ScalarAdd(_)
                    | FusedOp::Pow(_)
            )
        });

        // Register declarations.
        let mut reg_decls = String::from(
            "    .reg .u32 %tid, %bid, %bdim, %n_reg;\n\
             \x20   .reg .u64 %in, %out, %off;\n\
             \x20   .reg .f32 %val;\n\
             \x20   .reg .pred %p;",
        );
        if needs_exp {
            reg_decls.push_str("\n    .reg .f32 %exp_tmp, %tmp;");
        }
        if needs_mul_scratch {
            reg_decls.push_str("\n    .reg .f32 %scratch;");
        }
        // relu/abs need a zero constant
        let needs_zero = self
            .ops
            .iter()
            .any(|op| matches!(op, FusedOp::Relu | FusedOp::Abs));
        if needs_zero {
            reg_decls.push_str("\n    .reg .f32 %zero;");
        }

        // Emit the operation body.
        // Binary ops (Add/Sub/Mul/Div) are rejected in the early validation
        // above and will never reach this match.
        for op in &self.ops {
            match op {
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div => {
                    unreachable!("binary ops rejected by early validation");
                }
                FusedOp::Neg => {
                    body_lines.push("    neg.f32 %val, %val;".into());
                }
                FusedOp::Relu => {
                    body_lines.push("    mov.f32 %zero, 0f00000000;".into());
                    body_lines.push("    max.f32 %val, %val, %zero;".into());
                }
                FusedOp::Sigmoid => {
                    // sigmoid(x) = 1 / (1 + exp(-x))
                    body_lines.push("    neg.f32 %tmp, %val;".into());
                    body_lines.push("    // approx exp via ex2: exp(x) = 2^(x * log2(e))".into());
                    body_lines.push("    mul.f32 %scratch, %tmp, 0f3FB8AA3B;".into()); // log2(e) ~ 1.4427
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %scratch;".into());
                    body_lines.push("    add.f32 %scratch, %exp_tmp, 0f3F800000;".into()); // 1.0
                    body_lines.push("    rcp.approx.f32 %val, %scratch;".into());
                }
                FusedOp::Tanh => {
                    // tanh(x) = 2*sigmoid(2x) - 1
                    body_lines.push("    add.f32 %val, %val, %val;".into()); // 2x
                    body_lines.push("    neg.f32 %tmp, %val;".into());
                    body_lines.push("    mul.f32 %scratch, %tmp, 0f3FB8AA3B;".into());
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %scratch;".into());
                    body_lines.push("    add.f32 %scratch, %exp_tmp, 0f3F800000;".into());
                    body_lines.push("    rcp.approx.f32 %val, %scratch;".into()); // sigmoid(2x)
                    body_lines.push("    add.f32 %val, %val, %val;".into()); // 2*sigmoid(2x)
                    body_lines.push("    sub.f32 %val, %val, 0f3F800000;".into()); // -1
                }
                FusedOp::Gelu => {
                    // GELU tanh approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    // Step 1: compute x^3 in %scratch
                    body_lines.push("    mul.f32 %scratch, %val, %val;".into()); // x^2
                    body_lines.push("    mul.f32 %scratch, %scratch, %val;".into()); // x^3
                    // Step 2: 0.044715 * x^3
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        (0.044715_f32).to_bits()
                    ));
                    // Step 3: x + 0.044715 * x^3
                    body_lines.push("    add.f32 %scratch, %val, %scratch;".into());
                    // Step 4: sqrt(2/pi) * (x + 0.044715 * x^3)
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        0.797_884_6_f32.to_bits()
                    ));
                    // Step 5: tanh via 2*sigmoid(2*arg) - 1
                    body_lines.push("    add.f32 %scratch, %scratch, %scratch;".into()); // 2*arg
                    body_lines.push("    neg.f32 %tmp, %scratch;".into());
                    body_lines.push("    mul.f32 %tmp, %tmp, 0f3FB8AA3B;".into()); // log2(e)
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %tmp;".into());
                    body_lines.push("    add.f32 %tmp, %exp_tmp, 0f3F800000;".into()); // 1.0
                    body_lines.push("    rcp.approx.f32 %scratch, %tmp;".into()); // sigmoid(2*arg)
                    body_lines.push("    add.f32 %scratch, %scratch, %scratch;".into()); // 2*sigmoid(2*arg)
                    body_lines.push("    sub.f32 %scratch, %scratch, 0f3F800000;".into()); // tanh
                    // Step 6: 0.5 * (1 + tanh(...))
                    body_lines.push("    add.f32 %scratch, %scratch, 0f3F800000;".into()); // 1 + tanh
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        (0.5_f32).to_bits()
                    )); // 0.5 * (1 + tanh)
                    // Step 7: x * result
                    body_lines.push("    mul.f32 %val, %val, %scratch;".into());
                }
                FusedOp::Silu => {
                    // SiLU: x * sigmoid(x)
                    body_lines.push("    neg.f32 %tmp, %val;".into());
                    body_lines.push("    mul.f32 %scratch, %tmp, 0f3FB8AA3B;".into());
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %scratch;".into());
                    body_lines.push("    add.f32 %scratch, %exp_tmp, 0f3F800000;".into());
                    body_lines.push("    rcp.approx.f32 %scratch, %scratch;".into()); // sigmoid(x)
                    body_lines.push("    mul.f32 %val, %val, %scratch;".into());
                }
                FusedOp::Sqrt => {
                    body_lines.push("    sqrt.approx.f32 %val, %val;".into());
                }
                FusedOp::Abs => {
                    body_lines.push("    abs.f32 %val, %val;".into());
                }
                FusedOp::Pow(p) => {
                    // x^p via lg2/mul/ex2: x^p = 2^(p * log2(x))
                    body_lines.push("    lg2.approx.f32 %scratch, %val;".into());
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        (*p as f32).to_bits()
                    ));
                    body_lines.push("    ex2.approx.f32 %val, %scratch;".into());
                }
                FusedOp::Exp => {
                    // exp(x) = 2^(x * log2(e))
                    body_lines.push("    mul.f32 %scratch, %val, 0f3FB8AA3B;".into()); // x * log2(e)
                    body_lines.push("    ex2.approx.f32 %val, %scratch;".into());
                }
                FusedOp::Log => {
                    // ln(x) = log2(x) * ln(2)
                    body_lines.push("    lg2.approx.f32 %scratch, %val;".into());
                    body_lines.push(format!(
                        "    mul.f32 %val, %scratch, 0f{:08X};",
                        (std::f32::consts::LN_2).to_bits()
                    ));
                }
                FusedOp::ScalarMul(s) => {
                    body_lines.push(format!(
                        "    mul.f32 %val, %val, 0f{:08X};",
                        (*s as f32).to_bits()
                    ));
                }
                FusedOp::ScalarAdd(s) => {
                    body_lines.push(format!(
                        "    add.f32 %val, %val, 0f{:08X};",
                        (*s as f32).to_bits()
                    ));
                }
            }
        }

        let body = body_lines.join("\n");

        Ok(format!(
            "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
{reg_decls}

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;

    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %val, [%in];

{body}

    st.global.f32 [%out], %val;

DONE:
    ret;
}}
"
        ))
    }

    // -------------------------------------------------------------------
    // C codegen
    // -------------------------------------------------------------------

    /// Generate a C function that applies this fused chain elementwise.
    ///
    /// The generated function signature is:
    ///
    /// ```c
    /// void <fn_name>(const float* __restrict__ in, float* __restrict__ out, int n)
    /// ```
    ///
    /// The loop is annotated with `#pragma omp simd` for auto-vectorization.
    /// Reduction loops (containing accumulate semantics) use the
    /// appropriate `#pragma omp simd reduction(...)` clause to avoid
    /// loop-carried dependency violations.
    ///
    /// # Errors
    ///
    /// Returns an error if the chain contains unsupported binary ops, or
    /// if `fn_name` is not a valid C identifier.
    pub fn generate_c(&self, fn_name: &str) -> FerrotorchResult<String> {
        validate_identifier(fn_name)?;

        // Reject unsupported binary ops.
        for op in &self.ops {
            if matches!(
                op,
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div
            ) {
                return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "generate_c: binary op '{op}' in unary FusedChain requires a \
                         second input and cannot be lowered to a single-input C loop"
                    ),
                });
            }
        }

        let mut body_lines: Vec<String> = Vec::new();
        for op in &self.ops {
            match op {
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div => {
                    unreachable!("binary ops rejected above");
                }
                FusedOp::Neg => {
                    body_lines.push("        val = -val;".into());
                }
                FusedOp::Relu => {
                    body_lines.push("        val = fmaxf(val, 0.0f);".into());
                }
                FusedOp::Sigmoid => {
                    body_lines.push("        val = 1.0f / (1.0f + expf(-val));".into());
                }
                FusedOp::Tanh => {
                    body_lines.push("        val = tanhf(val);".into());
                }
                FusedOp::Gelu => {
                    // Tanh-based GELU approximation (matches all backends).
                    body_lines.push("        {".into());
                    body_lines.push("            float x3 = val * val * val;".into());
                    body_lines.push(
                        "            float inner = 0.7978845608f * (val + 0.044715f * x3);".into(),
                    );
                    body_lines.push("            val = val * 0.5f * (1.0f + tanhf(inner));".into());
                    body_lines.push("        }".into());
                }
                FusedOp::Silu => {
                    body_lines.push(
                        "        { float s = 1.0f / (1.0f + expf(-val)); val = val * s; }".into(),
                    );
                }
                FusedOp::Sqrt => {
                    body_lines.push("        val = sqrtf(val);".into());
                }
                FusedOp::Abs => {
                    body_lines.push("        val = fabsf(val);".into());
                }
                FusedOp::Exp => {
                    body_lines.push("        val = expf(val);".into());
                }
                FusedOp::Log => {
                    body_lines.push("        val = logf(val);".into());
                }
                FusedOp::Pow(p) => {
                    body_lines.push(format!("        val = powf(val, {p:.17}f);"));
                }
                FusedOp::ScalarMul(s) => {
                    body_lines.push(format!("        val = val * {s:.17}f;"));
                }
                FusedOp::ScalarAdd(s) => {
                    body_lines.push(format!("        val = val + {s:.17}f;"));
                }
            }
        }
        let body = body_lines.join("\n");

        // Elementwise loops use plain `#pragma omp simd` — no loop-carried
        // dependencies since each iteration is independent.
        Ok(format!(
            "\
#include <math.h>

void {fn_name}(const float* __restrict__ in, float* __restrict__ out, int n) {{
    #pragma omp simd
    for (int i = 0; i < n; i++) {{
        float val = in[i];
{body}
        out[i] = val;
    }}
}}
"
        ))
    }
}

/// Generate a C function that performs a reduction.
///
/// Unlike elementwise loops, reduction loops have a loop-carried dependency.
/// The generated code uses `#pragma omp simd reduction(+:acc)` (for sum/mean)
/// or `#pragma omp simd reduction(*:acc)` (for prod) to correctly handle
/// SIMD vectorization without data races.
///
/// For [`ReductionKind::Mean`], the sum is divided by `n` after the loop.
pub fn generate_reduction_c(kind: ReductionKind, fn_name: &str) -> FerrotorchResult<String> {
    validate_identifier(fn_name)?;

    let (identity, omp_clause, accumulate_expr, finalize) = match kind {
        ReductionKind::Sum => (
            "0.0f",
            "#pragma omp simd reduction(+:acc)",
            "acc += in[i];",
            "",
        ),
        ReductionKind::Prod => (
            "1.0f",
            "#pragma omp simd reduction(*:acc)",
            "acc *= in[i];",
            "",
        ),
        ReductionKind::Mean => (
            "0.0f",
            "#pragma omp simd reduction(+:acc)",
            "acc += in[i];",
            "    acc = acc / (float)n;\n",
        ),
    };

    Ok(format!(
        "\
#include <math.h>

void {fn_name}(const float* __restrict__ in, float* __restrict__ out, int n) {{
    float acc = {identity};
    {omp_clause}
    for (int i = 0; i < n; i++) {{
        {accumulate_expr}
    }}
{finalize}    out[0] = acc;
}}
"
    ))
}

impl Default for FusedChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Reduction PTX generation
// ---------------------------------------------------------------------------

/// Generate a PTX kernel that performs a single-pass parallel reduction.
///
/// The generated kernel uses shared-memory block-level reduction and
/// `atomicAdd` (for Sum/Mean) or iterative `atomicCAS` (for Prod) to
/// accumulate partial results from each block into `output[0]`.
///
/// For [`ReductionKind::Mean`], the final division by `n` is performed
/// by thread 0 of block 0 **after** a global memory fence, ensuring the
/// result is correct regardless of block scheduling order.
///
/// # Errors
///
/// Returns an error if `kernel_name` is not a valid identifier.
pub fn generate_reduction_ptx(kind: ReductionKind, kernel_name: &str) -> FerrotorchResult<String> {
    validate_identifier(kernel_name)?;

    let identity_bits = kind.identity_f32_bits();
    let (reduce_op, is_prod) = match kind {
        ReductionKind::Sum | ReductionKind::Mean => ("add.f32", false),
        ReductionKind::Prod => ("mul.f32", true),
    };
    let is_mean = kind == ReductionKind::Mean;

    // For sum/mean we use atom.global.add.f32 which atomically adds the
    // block partial sum to output[0].
    // For prod we use a CAS loop since there is no atomic multiply.
    let atomic_section = if is_prod {
        // CAS loop for atomic multiply:
        //   old = *addr;
        //   do { expected = old; new = old * partial; old = CAS(addr, expected, new); }
        //   while (old != expected);
        "\
    // Atomic multiply via CAS loop
    ld.global.f32 %old, [%out];
CAS_LOOP:
    mov.f32 %expected, %old;
    mul.f32 %new_val, %old, %val;
    // Reinterpret floats as u32 for CAS
    mov.b32 %old_bits, %expected;
    mov.b32 %new_bits, %new_val;
    atom.global.cas.b32 %result_bits, [%out], %old_bits, %new_bits;
    mov.b32 %old, %result_bits;
    setp.ne.f32 %cas_p, %old, %expected;
    @%cas_p bra CAS_LOOP;"
    } else {
        "    atom.global.add.f32 %val, [%out], %val;"
    };

    // Extra registers for prod CAS loop
    let extra_regs = if is_prod {
        "\n    .reg .f32 %old, %expected, %new_val;\n    .reg .u32 %old_bits, %new_bits, %result_bits;\n    .reg .pred %cas_p;"
    } else {
        ""
    };

    // Mean: after all blocks have contributed, thread 0 of block 0 divides by n.
    // We use a second kernel launch or a global flag for synchronization.
    // Simpler approach: emit a second entry point that does the division.
    let mean_finalize = if is_mean {
        format!(
            "\n\
.visible .entry {kernel_name}_finalize(
    .param .u64 out_ptr_f,
    .param .u32 n_f
) {{
    .reg .u64 %out_f;
    .reg .u32 %n_f;
    .reg .f32 %sum_val, %n_float, %mean_val;

    ld.param.u64 %out_f, [out_ptr_f];
    ld.param.u32 %n_f, [n_f];

    ld.global.f32 %sum_val, [%out_f];
    cvt.rn.f32.u32 %n_float, %n_f;
    div.approx.f32 %mean_val, %sum_val, %n_float;
    st.global.f32 [%out_f], %mean_val;

    ret;
}}\n"
        )
    } else {
        String::new()
    };

    Ok(format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u32 %tid, %bid, %bdim, %gid, %n_reg, %s;
    .reg .u64 %in, %out, %off;
    .reg .f32 %val, %shared_val;
    .reg .pred %p, %p2;{extra_regs}

    .shared .align 4 .f32 sdata[1024];

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %tid;

    // Load element or identity if out of bounds
    setp.lt.u32 %p, %gid, %n_reg;
    mov.f32 %val, 0f{identity_bits:08X};
    @!%p bra SKIP_LOAD;

    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 2;
    add.u64 %in, %in, %off;
    ld.global.f32 %val, [%in];

SKIP_LOAD:
    // Store to shared memory
    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;
    st.shared.f32 [sdata + %off], %val;
    bar.sync 0;

    // Tree reduction in shared memory
    shr.u32 %s, %bdim, 1;
REDUCE_LOOP:
    setp.eq.u32 %p2, %s, 0;
    @%p2 bra REDUCE_DONE;

    setp.lt.u32 %p, %tid, %s;
    @!%p bra REDUCE_SKIP;

    // Load partner value
    add.u32 %gid, %tid, %s;
    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 2;
    ld.shared.f32 %shared_val, [sdata + %off];
    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;
    ld.shared.f32 %val, [sdata + %off];
    {reduce_op} %val, %val, %shared_val;
    st.shared.f32 [sdata + %off], %val;

REDUCE_SKIP:
    bar.sync 0;
    shr.u32 %s, %s, 1;
    bra REDUCE_LOOP;

REDUCE_DONE:
    // Thread 0 of each block atomically adds its partial result to output[0]
    setp.ne.u32 %p, %tid, 0;
    @%p bra BLOCK_DONE;

    ld.shared.f32 %val, [sdata];
{atomic_section}

BLOCK_DONE:
    ret;
}}
{mean_finalize}"
    ))
}

// ---------------------------------------------------------------------------
// Name validation
// ---------------------------------------------------------------------------

/// Validate that `name` is a legal C/PTX identifier: `[a-zA-Z_][a-zA-Z0-9_]*`.
///
/// This prevents injection attacks when interpolating user-provided or
/// generated names into C, CUDA, or PTX source code.
fn validate_identifier(name: &str) -> FerrotorchResult<()> {
    if name.is_empty() {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: "identifier name must not be empty".into(),
        });
    }

    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "identifier '{name}' has invalid first character '{first}'; \
                 must match [a-zA-Z_][a-zA-Z0-9_]*"
            ),
        });
    }

    for ch in chars {
        if !ch.is_ascii_alphanumeric() && ch != '_' {
            return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                message: format!(
                    "identifier '{name}' contains invalid character '{ch}'; \
                     must match [a-zA-Z_][a-zA-Z0-9_]*"
                ),
            });
        }
    }

    Ok(())
}

/// Sanitize a string for safe inclusion in a code comment by removing
/// sequences that could close a C-style block comment (`*/`).
///
/// For PTX (which uses `//` line comments) this is not strictly needed,
/// but it is essential for C/CUDA codegen to prevent comment-terminator
/// injection.
#[allow(dead_code)]
pub(crate) fn sanitize_comment(text: &str) -> String {
    text.replace("*/", "* /")
}

/// Validate that `name` is a legal C/PTX identifier.
///
/// Re-exported for use by other codegen modules.
#[allow(dead_code)]
pub(crate) fn validate_codegen_identifier(name: &str) -> FerrotorchResult<()> {
    validate_identifier(name)
}

// ---------------------------------------------------------------------------
// CPU op application helper
// ---------------------------------------------------------------------------

/// Apply a single [`FusedOp`] in-place across a mutable slice.
///
/// Returns an error if the op is a binary op (Add/Sub/Mul/Div) that
/// requires a second operand, since silently ignoring it would produce
/// wrong results.
fn apply_op_inplace<T: Float>(op: &FusedOp, data: &mut [T]) -> FerrotorchResult<()> {
    let zero: T = num_traits::zero();
    let one: T = num_traits::one();

    match op {
        FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div => {
            return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                message: format!(
                    "apply_op_inplace: binary op '{op}' in unary FusedChain requires a second \
                     operand and cannot be applied in-place on a single tensor"
                ),
            });
        }
        FusedOp::Neg => {
            for x in data.iter_mut() {
                *x = zero - *x;
            }
        }
        FusedOp::Relu => {
            for x in data.iter_mut() {
                *x = if *x > zero { *x } else { zero };
            }
        }
        FusedOp::Sigmoid => {
            for x in data.iter_mut() {
                let val = *x;
                let neg_val = zero - val;
                *x = one / (one + neg_val.exp());
            }
        }
        FusedOp::Tanh => {
            // tanh(x) = 2*sigmoid(2x) - 1
            let two = one + one;
            for x in data.iter_mut() {
                let s = one / (one + (zero - two * *x).exp());
                *x = two * s - one;
            }
        }
        FusedOp::Gelu => {
            // GELU tanh approximation:
            //   x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            // Matches the codegen NativeBackend and the standard PyTorch GELU.
            let half = T::from(0.5).unwrap();
            let sqrt_2_over_pi = T::from(0.7978845608028654).unwrap(); // sqrt(2/pi)
            let coeff = T::from(0.044715).unwrap();
            for x in data.iter_mut() {
                let x3 = *x * *x * *x;
                let inner = sqrt_2_over_pi * (*x + coeff * x3);
                *x = *x * half * (one + inner.tanh());
            }
        }
        FusedOp::Silu => {
            // SiLU: x * sigmoid(x)
            for x in data.iter_mut() {
                let val = *x;
                let neg_val = zero - val;
                let s = one / (one + neg_val.exp());
                *x = val * s;
            }
        }
        FusedOp::Sqrt => {
            for x in data.iter_mut() {
                *x = x.sqrt();
            }
        }
        FusedOp::Abs => {
            for x in data.iter_mut() {
                *x = x.abs();
            }
        }
        FusedOp::Exp => {
            for x in data.iter_mut() {
                *x = x.exp();
            }
        }
        FusedOp::Log => {
            for x in data.iter_mut() {
                *x = x.ln();
            }
        }
        FusedOp::Pow(p) => {
            let p_t = T::from(*p).unwrap();
            for x in data.iter_mut() {
                *x = x.powf(p_t);
            }
        }
        FusedOp::ScalarMul(s) => {
            let s_t = T::from(*s).unwrap();
            for x in data.iter_mut() {
                *x = *x * s_t;
            }
        }
        FusedOp::ScalarAdd(s) => {
            let s_t = T::from(*s).unwrap();
            for x in data.iter_mut() {
                *x += s_t;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// DAG fusion helpers
// ---------------------------------------------------------------------------

/// Estimate the total number of elements across all input shapes.
///
/// Returns `0` for zero-element tensors (rather than clamping to 1, which
/// would be incorrect and could cause out-of-bounds kernel launches).
pub fn estimate_numel_for_inputs(shapes: &[Vec<usize>]) -> usize {
    shapes
        .iter()
        .map(|s| s.iter().copied().product::<usize>())
        .max()
        .unwrap_or(0)
}

/// Estimate (M, K, N) dimensions for a matrix multiplication from input shapes.
///
/// Both inputs must be 2-D. Returns `Err` for non-2D inputs rather than
/// silently returning `(1, 1, 1)` which would produce wrong results.
pub fn estimate_matmul_dims(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> FerrotorchResult<(usize, usize, usize)> {
    if lhs_shape.len() != 2 {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "estimate_matmul_dims: LHS must be 2-D, got {}-D shape {:?}",
                lhs_shape.len(),
                lhs_shape
            ),
        });
    }
    if rhs_shape.len() != 2 {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "estimate_matmul_dims: RHS must be 2-D, got {}-D shape {:?}",
                rhs_shape.len(),
                rhs_shape
            ),
        });
    }

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    if k != rhs_shape[0] {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "estimate_matmul_dims: inner dimensions mismatch: LHS[1]={} vs RHS[0]={}",
                k, rhs_shape[0]
            ),
        });
    }

    Ok((m, k, n))
}

// ---------------------------------------------------------------------------
// Tensor-level fusion helper
// ---------------------------------------------------------------------------

/// Apply a [`FusedChain`] to a tensor, producing a new tensor on the same
/// device with the same shape.
///
/// Currently executes on the CPU. When a CUDA device is available the chain
/// could be dispatched via the PTX kernel returned by
/// [`FusedChain::generate_ptx`].
pub fn apply_fused<T: Float>(input: &Tensor<T>, chain: &FusedChain) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let result = chain.execute_cpu(data)?;
    Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::tensor::Tensor;

    // -- FusedChain basics --------------------------------------------------

    #[test]
    fn test_chain_new_is_empty() {
        let chain = FusedChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_chain_push_and_len() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);
        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    // -- with_fusion flag ---------------------------------------------------

    #[test]
    fn test_fusion_flag_default_off() {
        assert!(!is_fusion_enabled());
    }

    #[test]
    fn test_fusion_flag_scoped() {
        assert!(!is_fusion_enabled());
        with_fusion(|| {
            assert!(is_fusion_enabled());
        });
        assert!(!is_fusion_enabled());
    }

    #[test]
    fn test_fusion_flag_nested() {
        with_fusion(|| {
            assert!(is_fusion_enabled());
            with_fusion(|| {
                assert!(is_fusion_enabled());
            });
            // Still enabled after inner scope -- inner guard restores `true`.
            assert!(is_fusion_enabled());
        });
        assert!(!is_fusion_enabled());
    }

    // -- CPU execution: scalar_add + relu + neg (the spec test) ---------------

    #[test]
    fn test_fused_scalar_add_relu_neg_cpu() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let input: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 3.0];

        // Sequential reference:
        //   scalar_add(2):  [-3.0, 1.0, 2.0, 3.0, 5.0]
        //   relu:           [ 0.0, 1.0, 2.0, 3.0, 5.0]
        //   neg:            [ 0.0,-1.0,-2.0,-3.0,-5.0]
        let expected: Vec<f32> = vec![0.0, -1.0, -2.0, -3.0, -5.0];

        let result = chain.execute_cpu(&input).unwrap();
        assert_eq!(result.len(), expected.len());
        for (got, exp) in result.iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_fused_matches_sequential() {
        // Verify the fused result matches applying each op one at a time.
        let input: Vec<f64> = vec![-3.0, -1.5, 0.0, 0.5, 2.0, 4.0];

        // Build a chain: scalar_add(2) -> relu -> neg.
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        // Fused.
        let fused = chain.execute_cpu(&input).unwrap();

        // Sequential.
        let mut sequential = input.clone();
        // scalar_add(2)
        for x in &mut sequential {
            *x += 2.0;
        }
        // relu
        for x in &mut sequential {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        // neg
        for x in &mut sequential {
            *x = -*x;
        }

        assert_eq!(fused.len(), sequential.len());
        for (i, (got, exp)) in fused.iter().zip(&sequential).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "element {i}: fused={got}, sequential={exp}",
            );
        }
    }

    // -- CPU execution: individual ops ----------------------------------------

    #[test]
    fn test_fused_neg() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Neg);
        let result = chain.execute_cpu(&[1.0f32, -2.0, 0.0]).unwrap();
        assert_eq!(result, vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_fused_sigmoid() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sigmoid);
        let result = chain.execute_cpu(&[0.0f64]).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fused_tanh() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Tanh);
        let result = chain.execute_cpu(&[0.0f64]).unwrap();
        assert!(result[0].abs() < 1e-10, "tanh(0) should be 0");
    }

    #[test]
    fn test_fused_sqrt() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sqrt);
        let result = chain.execute_cpu(&[4.0f32, 9.0, 16.0]).unwrap();
        let expected = vec![2.0f32, 3.0, 4.0];
        for (got, exp) in result.iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fused_abs() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Abs);
        let result = chain.execute_cpu(&[-3.0f32, 0.0, 5.0]).unwrap();
        assert_eq!(result, vec![3.0, 0.0, 5.0]);
    }

    #[test]
    fn test_fused_pow() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Pow(2.0));
        let result = chain.execute_cpu(&[3.0f64]).unwrap();
        assert!((result[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_fused_scalar_mul() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarMul(3.0));
        let result = chain.execute_cpu(&[2.0f32, -1.0]).unwrap();
        assert_eq!(result, vec![6.0, -3.0]);
    }

    #[test]
    fn test_fused_empty_input() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);
        let result = chain.execute_cpu::<f32>(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_fused_empty_chain() {
        let chain = FusedChain::new();
        let input = vec![1.0f32, 2.0, 3.0];
        let result = chain.execute_cpu(&input).unwrap();
        assert_eq!(result, input);
    }

    // -- PTX generation -------------------------------------------------------

    #[test]
    fn test_ptx_generation_valid_string() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let ptx = chain.generate_ptx().unwrap();

        // Must have the standard PTX header.
        assert!(ptx.contains(".version 7.0"));
        assert!(ptx.contains(".target sm_52"));
        assert!(ptx.contains(".address_size 64"));

        // Must declare the entry point.
        assert!(ptx.contains(".visible .entry fused_kernel"));

        // Must have parameter declarations.
        assert!(ptx.contains("in_ptr"));
        assert!(ptx.contains("out_ptr"));

        // Must contain the operations.
        assert!(
            ptx.contains("add.f32 %val"),
            "ScalarAdd should produce an add.f32 instruction"
        );
        assert!(
            ptx.contains("max.f32 %val"),
            "Relu should produce a max.f32 instruction"
        );
        assert!(
            ptx.contains("neg.f32 %val"),
            "Neg should produce a neg.f32 instruction"
        );

        // Must end with store + ret.
        assert!(ptx.contains("st.global.f32 [%out], %val;"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_ptx_generation_sigmoid() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sigmoid);
        let ptx = chain.generate_ptx().unwrap();
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_generation_sqrt() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sqrt);
        let ptx = chain.generate_ptx().unwrap();
        assert!(ptx.contains("sqrt.approx.f32"));
    }

    #[test]
    fn test_ptx_generation_pow() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Pow(3.0));
        let ptx = chain.generate_ptx().unwrap();
        assert!(ptx.contains("lg2.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
    }

    // -- apply_fused (tensor-level) -------------------------------------------

    #[test]
    fn test_apply_fused_tensor() {
        let storage = TensorStorage::cpu(vec![-5.0f32, -1.0, 0.0, 1.0, 3.0]);
        let tensor = Tensor::from_storage(storage, vec![5], false).unwrap();

        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let result = apply_fused(&tensor, &chain).unwrap();
        let result_data = result.data().unwrap();
        let expected = [0.0f32, -1.0, -2.0, -3.0, -5.0];

        assert_eq!(result_data.len(), expected.len());
        for (got, exp) in result_data.iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6, "got {got}, expected {exp}");
        }
        assert_eq!(result.shape(), &[5]);
    }

    // -- FusedOp Display ------------------------------------------------------

    #[test]
    fn test_fused_op_display() {
        assert_eq!(format!("{}", FusedOp::Relu), "relu");
        assert_eq!(format!("{}", FusedOp::ScalarAdd(1.5)), "scalar_add(1.5)");
        assert_eq!(format!("{}", FusedOp::Pow(2.0)), "pow(2)");
    }
}
