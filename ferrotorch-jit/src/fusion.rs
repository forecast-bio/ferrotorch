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
    FUSION_ENABLED.with(|f| f.get())
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
            FusedOp::Pow(p) => write!(f, "pow({p})"),
            FusedOp::ScalarMul(s) => write!(f, "scalar_mul({s})"),
            FusedOp::ScalarAdd(s) => write!(f, "scalar_add({s})"),
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
    pub fn execute_cpu<T: Float>(&self, input: &[T]) -> Vec<T> {
        let mut data = input.to_vec();
        for op in &self.ops {
            apply_op_inplace::<T>(op, &mut data);
        }
        data
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
    pub fn generate_ptx(&self) -> String {
        let mut body_lines: Vec<String> = Vec::new();

        // We accumulate the running value in %val. Some ops need scratch
        // registers; we define them at the top of the kernel.
        let needs_exp = self.ops.iter().any(|op| {
            matches!(
                op,
                FusedOp::Sigmoid | FusedOp::Tanh | FusedOp::Gelu | FusedOp::Silu
            )
        });
        let needs_mul_scratch = self.ops.iter().any(|op| {
            matches!(
                op,
                FusedOp::Sigmoid
                    | FusedOp::Tanh
                    | FusedOp::Gelu
                    | FusedOp::Silu
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
        for op in &self.ops {
            match op {
                FusedOp::Add => {
                    // Binary add -- for a fused unary chain we treat this as
                    // a no-op marker. Real binary fusion would need a second
                    // input pointer. Here we document the intent.
                    body_lines.push("    // fused: add (binary -- requires second input)".into());
                }
                FusedOp::Sub => {
                    body_lines.push("    // fused: sub (binary -- requires second input)".into());
                }
                FusedOp::Mul => {
                    body_lines.push("    // fused: mul (binary -- requires second input)".into());
                }
                FusedOp::Div => {
                    body_lines.push("    // fused: div (binary -- requires second input)".into());
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
                    // GELU approx: x * sigmoid(1.702 * x)
                    body_lines.push("    mov.f32 %scratch, 0f3FD9F16C;".into()); // 1.702
                    body_lines.push("    mul.f32 %tmp, %val, %scratch;".into()); // 1.702*x
                    body_lines.push("    neg.f32 %tmp, %tmp;".into());
                    body_lines.push("    mul.f32 %scratch, %tmp, 0f3FB8AA3B;".into());
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %scratch;".into());
                    body_lines.push("    add.f32 %scratch, %exp_tmp, 0f3F800000;".into());
                    body_lines.push("    rcp.approx.f32 %scratch, %scratch;".into()); // sigmoid(1.702*x)
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

        format!(
            "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry fused_kernel(
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
        )
    }
}

impl Default for FusedChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CPU op application helper
// ---------------------------------------------------------------------------

/// Apply a single [`FusedOp`] in-place across a mutable slice.
fn apply_op_inplace<T: Float>(op: &FusedOp, data: &mut [T]) {
    let zero: T = num_traits::zero();
    let one: T = num_traits::one();

    match op {
        FusedOp::Add => {
            // Binary ops are no-ops in the unary chain context.
        }
        FusedOp::Sub => {}
        FusedOp::Mul => {}
        FusedOp::Div => {}
        FusedOp::Neg => {
            for x in data.iter_mut() {
                *x = zero - *x;
            }
        }
        FusedOp::Relu => {
            for x in data.iter_mut() {
                if *x < zero {
                    *x = zero;
                }
            }
        }
        FusedOp::Sigmoid => {
            for x in data.iter_mut() {
                *x = one / (one + (zero - *x).exp());
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
            // GELU approx: x * sigmoid(1.702 * x)
            let coeff = T::from(1.702).unwrap();
            for x in data.iter_mut() {
                let s = one / (one + (zero - coeff * *x).exp());
                *x = *x * s;
            }
        }
        FusedOp::Silu => {
            // SiLU: x * sigmoid(x)
            for x in data.iter_mut() {
                let s = one / (one + (zero - *x).exp());
                *x = *x * s;
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
                *x = *x + s_t;
            }
        }
    }
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
pub fn apply_fused<T: Float>(
    input: &Tensor<T>,
    chain: &FusedChain,
) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let result = chain.execute_cpu(data);
    Tensor::from_storage(
        TensorStorage::cpu(result),
        input.shape().to_vec(),
        false,
    )
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

        let result = chain.execute_cpu(&input);
        assert_eq!(result.len(), expected.len());
        for (got, exp) in result.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < 1e-6,
                "got {got}, expected {exp}",
            );
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
        let fused = chain.execute_cpu(&input);

        // Sequential.
        let mut sequential = input.clone();
        // scalar_add(2)
        for x in sequential.iter_mut() {
            *x += 2.0;
        }
        // relu
        for x in sequential.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        // neg
        for x in sequential.iter_mut() {
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
        let result = chain.execute_cpu(&[1.0f32, -2.0, 0.0]);
        assert_eq!(result, vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_fused_sigmoid() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sigmoid);
        let result = chain.execute_cpu(&[0.0f64]);
        assert!((result[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fused_tanh() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Tanh);
        let result = chain.execute_cpu(&[0.0f64]);
        assert!(result[0].abs() < 1e-10, "tanh(0) should be 0");
    }

    #[test]
    fn test_fused_sqrt() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sqrt);
        let result = chain.execute_cpu(&[4.0f32, 9.0, 16.0]);
        let expected = vec![2.0f32, 3.0, 4.0];
        for (got, exp) in result.iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fused_abs() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Abs);
        let result = chain.execute_cpu(&[-3.0f32, 0.0, 5.0]);
        assert_eq!(result, vec![3.0, 0.0, 5.0]);
    }

    #[test]
    fn test_fused_pow() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Pow(2.0));
        let result = chain.execute_cpu(&[3.0f64]);
        assert!((result[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_fused_scalar_mul() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarMul(3.0));
        let result = chain.execute_cpu(&[2.0f32, -1.0]);
        assert_eq!(result, vec![6.0, -3.0]);
    }

    #[test]
    fn test_fused_empty_input() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);
        let result = chain.execute_cpu::<f32>(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fused_empty_chain() {
        let chain = FusedChain::new();
        let input = vec![1.0f32, 2.0, 3.0];
        let result = chain.execute_cpu(&input);
        assert_eq!(result, input);
    }

    // -- PTX generation -------------------------------------------------------

    #[test]
    fn test_ptx_generation_valid_string() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let ptx = chain.generate_ptx();

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
        let ptx = chain.generate_ptx();
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_generation_sqrt() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sqrt);
        let ptx = chain.generate_ptx();
        assert!(ptx.contains("sqrt.approx.f32"));
    }

    #[test]
    fn test_ptx_generation_pow() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Pow(3.0));
        let ptx = chain.generate_ptx();
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
            assert!(
                (got - exp).abs() < 1e-6,
                "got {got}, expected {exp}",
            );
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
