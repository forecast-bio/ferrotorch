//! GPU code generators: emit CUDA C and PTX source from [`LoopIR`].
//!
//! Two code generators are provided:
//!
//! - [`GpuCodegen::generate_cuda_source`] — emits CUDA C with `__global__`
//!   kernels, `blockIdx`/`threadIdx` mapping, and shared memory for reductions.
//! - [`GpuCodegen::generate_ptx_source`] — emits PTX assembly targeting
//!   `sm_52` with hand-scheduled register allocation and approximate
//!   transcendental instructions (f32 only — see below).
//!
//! Both generators convert the outermost loop of a `LoopIR` program into
//! thread-parallel GPU execution while keeping inner loops as thread-local
//! serial computation.
//!
//! # Dtype dispatch (#729)
//!
//! Both generators dispatch on a [`Dtype`] parameter (currently `F32` or
//! `F64`) at every site where the emission differs between scalar widths:
//! CUDA `float`/`double` declarations, PTX `.f32`/`.f64` suffixes, register
//! declarations, load/store widths, and constant literal encoding (`0f...`
//! 8 hex digits for f32; `0d...` 16 hex digits for f64).
//!
//! Transcendentals (`exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `relu`, `abs`,
//! `gelu`, `silu`, `pow`) on the PTX path use direct hardware approximation
//! instructions (`ex2.approx.f32`, `lg2.approx.f32`, `rcp.approx.f32`,
//! `sqrt.approx.f32`) for f32. PTX has no `*.approx.f64` instructions, so
//! emitting them on f64 is invalid; demote-promote silently loses precision.
//!
//! For f64 transcendentals the codegen routes through NVRTC, which links
//! libdevice (`__nv_exp`, `__nv_log`, `__nv_sqrt`, `__nv_tanh`, `__nv_pow`,
//! ...) and returns a PTX module with libdevice's polynomial expansions
//! inlined. The CUDA C frontend already supports f64 transcendentals via
//! the host math headers (`exp(double)`, `tanh(double)`, ...), so the f64
//! transcendental PTX path is implemented as `generate_cuda_source` →
//! NVRTC-compile → PTX — gated behind the `cuda` feature flag (default
//! off). Without the feature, f64 transcendentals still return
//! `Err(JitError::Unsupported { op, dtype })` per `rust-gpu-discipline` §3
//! (`PyTorch` parity: structured `NotImplementedError`-like error).
//!
//! See `nvrtc_libdevice` below for the integration; closes #748 follow-up
//! to #729.

use crate::codegen_ir::{BinOpKind, Expr, LoopIR, UnaryOpKind};
use crate::error::JitError;
use crate::graph::Dtype;

/// A GPU code generator targeting CUDA C and PTX output.
#[derive(Debug)]
pub struct GpuCodegen;

// ===========================================================================
// CUDA C code generation
// ===========================================================================

impl GpuCodegen {
    /// Generate a CUDA C kernel from a `LoopIR` program.
    ///
    /// The outermost loop is parallelized across GPU threads. Inner loops
    /// remain serial per-thread.  Reductions use shared memory with a
    /// tree-reduction pattern.
    ///
    /// The emitted code includes:
    /// - `__global__` kernel with `blockIdx.x * blockDim.x + threadIdx.x` indexing
    /// - Bounds checking against `n`
    /// - Shared memory declarations for reductions
    /// - Coalesced memory access patterns (sequential thread -> sequential address)
    ///
    /// # Errors
    ///
    /// Returns `Err(JitError::Unsupported { op, dtype })` is reserved for
    /// future cases; the CUDA C path lowers transcendentals via the host
    /// math headers (`expf`/`exp`, `logf`/`log`, etc.), which work for both
    /// f32 and f64. The result is wrapped in `Result` for shape symmetry
    /// with [`GpuCodegen::generate_ptx_source`].
    pub fn generate_cuda_source(
        loops: &[LoopIR],
        fn_name: &str,
        num_inputs: usize,
        dtype: Dtype,
    ) -> Result<String, JitError> {
        let mut out = String::new();
        let scalar = cuda_scalar_name(dtype);

        out.push_str("#include <math.h>\n\n");

        // Detect if this is a reduction (has accumulate statements)
        let has_reduction = loops_contain_accumulate(loops);

        // Build function signature
        out.push_str(&format!("__global__ void {fn_name}(\n"));
        for i in 0..num_inputs {
            out.push_str(&format!("    const {scalar}* __restrict__ in{i},\n"));
        }
        out.push_str(&format!("    {scalar}* __restrict__ output,\n"));
        out.push_str("    int n\n");
        out.push_str(") {\n");

        // Thread index computation
        out.push_str("    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");

        if has_reduction {
            emit_cuda_reduction(&mut out, loops, num_inputs, dtype);
        } else {
            // Elementwise / matmul: each thread handles one element of the
            // outermost loop
            emit_cuda_elementwise(&mut out, loops, dtype);
        }

        out.push_str("}\n");
        Ok(out)
    }
}

/// Emit CUDA code for elementwise operations where the outer loop maps to threads.
fn emit_cuda_elementwise(out: &mut String, loops: &[LoopIR], dtype: Dtype) {
    out.push_str("    if (tid >= n) return;\n\n");

    for stmt in loops {
        match stmt {
            LoopIR::Loop { var, body, .. } => {
                // The outermost loop variable becomes `tid`
                out.push_str(&format!("    // outer loop var '{var}' -> tid\n"));
                for s in body {
                    emit_cuda_stmt_with_var_replace(out, s, var, "tid", 1, dtype);
                }
            }
            other => {
                emit_cuda_stmt(out, other, 1, dtype);
            }
        }
    }
}

/// Emit CUDA code for reduction operations using shared memory.
fn emit_cuda_reduction(out: &mut String, loops: &[LoopIR], _num_inputs: usize, dtype: Dtype) {
    let scalar = cuda_scalar_name(dtype);
    let zero_lit = cuda_zero_literal(dtype);

    out.push_str(&format!("    extern __shared__ {scalar} sdata[];\n"));
    out.push_str("    int local_tid = threadIdx.x;\n\n");

    // Each thread computes a partial result, then we tree-reduce in shared memory
    out.push_str("    // Load phase: each thread accumulates elements stride-apart\n");
    out.push_str(&format!("    {scalar} thread_acc = {zero_lit};\n"));
    out.push_str("    for (int idx = tid; idx < n; idx += blockDim.x * gridDim.x) {\n");

    // Find the accumulation expression
    let acc_expr = find_accumulate_expr(loops);
    match acc_expr {
        Some(expr) => {
            let val = emit_cuda_expr_replace(&expr, "i", "idx", dtype);
            out.push_str(&format!("        thread_acc += {val};\n"));
        }
        None => {
            out.push_str("        thread_acc += in0[idx];\n");
        }
    }

    out.push_str("    }\n\n");

    // Store to shared memory
    out.push_str("    sdata[local_tid] = thread_acc;\n");
    out.push_str("    __syncthreads();\n\n");

    // Tree reduction in shared memory
    out.push_str("    // Tree reduction in shared memory\n");
    out.push_str("    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n");
    out.push_str("        if (local_tid < s) {\n");
    out.push_str("            sdata[local_tid] += sdata[local_tid + s];\n");
    out.push_str("        }\n");
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n\n");

    // Write final result (thread 0 of each block writes to output)
    out.push_str("    if (local_tid == 0) {\n");

    // Check for mean: divide by n
    let is_mean = find_mean_divisor(loops);
    match is_mean {
        Some(_) => {
            out.push_str(&format!(
                "        atomicAdd(&output[0], sdata[0] / ({scalar})n);\n"
            ));
        }
        None => {
            out.push_str("        atomicAdd(&output[0], sdata[0]);\n");
        }
    }

    out.push_str("    }\n");
}

/// Emit a CUDA C statement at the given indentation.
fn emit_cuda_stmt(out: &mut String, stmt: &LoopIR, indent: usize, dtype: Dtype) {
    let pad = "    ".repeat(indent);
    let scalar = cuda_scalar_name(dtype);

    match stmt {
        LoopIR::Loop {
            var,
            start,
            end,
            body,
        } => {
            let start_s = emit_cuda_expr(start, dtype);
            let end_s = emit_cuda_expr(end, dtype);
            out.push_str(&format!(
                "{pad}for (int {var} = {start_s}; {var} < {end_s}; {var}++) {{\n"
            ));
            for s in body {
                emit_cuda_stmt(out, s, indent + 1, dtype);
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Store {
            buffer,
            index,
            value,
        } => {
            let idx = emit_cuda_expr(index, dtype);
            let val = emit_cuda_expr(value, dtype);
            let buf = cuda_buffer_name(buffer);
            out.push_str(&format!("{pad}{buf}[{idx}] = {val};\n"));
        }

        LoopIR::Let { var, value } => {
            let val = emit_cuda_expr(value, dtype);
            out.push_str(&format!("{pad}{scalar} {var} = {val};\n"));
        }

        LoopIR::Assign { var, value } => {
            let val = emit_cuda_expr(value, dtype);
            out.push_str(&format!("{pad}{var} = {val};\n"));
        }

        LoopIR::Accumulate { var, value } => {
            let val = emit_cuda_expr(value, dtype);
            out.push_str(&format!("{pad}{var} += {val};\n"));
        }

        LoopIR::If {
            condition,
            then_body,
            else_body,
        } => {
            let cond = emit_cuda_expr(condition, dtype);
            out.push_str(&format!("{pad}if ({cond}) {{\n"));
            for s in then_body {
                emit_cuda_stmt(out, s, indent + 1, dtype);
            }
            if !else_body.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_cuda_stmt(out, s, indent + 1, dtype);
                }
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Comment(text) => {
            out.push_str(&format!("{pad}/* {text} */\n"));
        }
    }
}

/// Emit a CUDA statement, replacing occurrences of `old_var` with `new_var`
/// in all expressions.
fn emit_cuda_stmt_with_var_replace(
    out: &mut String,
    stmt: &LoopIR,
    old_var: &str,
    new_var: &str,
    indent: usize,
    dtype: Dtype,
) {
    let pad = "    ".repeat(indent);
    let scalar = cuda_scalar_name(dtype);

    match stmt {
        LoopIR::Loop {
            var,
            start,
            end,
            body,
        } => {
            let start_s = emit_cuda_expr_replace(start, old_var, new_var, dtype);
            let end_s = emit_cuda_expr_replace(end, old_var, new_var, dtype);
            out.push_str(&format!(
                "{pad}for (int {var} = {start_s}; {var} < {end_s}; {var}++) {{\n"
            ));
            for s in body {
                emit_cuda_stmt_with_var_replace(out, s, old_var, new_var, indent + 1, dtype);
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Store {
            buffer,
            index,
            value,
        } => {
            let idx = emit_cuda_expr_replace(index, old_var, new_var, dtype);
            let val = emit_cuda_expr_replace(value, old_var, new_var, dtype);
            let buf = cuda_buffer_name(buffer);
            out.push_str(&format!("{pad}{buf}[{idx}] = {val};\n"));
        }

        LoopIR::Let { var, value } => {
            let val = emit_cuda_expr_replace(value, old_var, new_var, dtype);
            out.push_str(&format!("{pad}{scalar} {var} = {val};\n"));
        }

        LoopIR::Assign { var, value } => {
            let val = emit_cuda_expr_replace(value, old_var, new_var, dtype);
            let actual_var = if var == old_var {
                new_var
            } else {
                var.as_str()
            };
            out.push_str(&format!("{pad}{actual_var} = {val};\n"));
        }

        LoopIR::Accumulate { var, value } => {
            let val = emit_cuda_expr_replace(value, old_var, new_var, dtype);
            let actual_var = if var == old_var {
                new_var
            } else {
                var.as_str()
            };
            out.push_str(&format!("{pad}{actual_var} += {val};\n"));
        }

        LoopIR::If {
            condition,
            then_body,
            else_body,
        } => {
            let cond = emit_cuda_expr_replace(condition, old_var, new_var, dtype);
            out.push_str(&format!("{pad}if ({cond}) {{\n"));
            for s in then_body {
                emit_cuda_stmt_with_var_replace(out, s, old_var, new_var, indent + 1, dtype);
            }
            if !else_body.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_cuda_stmt_with_var_replace(out, s, old_var, new_var, indent + 1, dtype);
                }
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Comment(text) => {
            out.push_str(&format!("{pad}/* {text} */\n"));
        }
    }
}

/// Emit a CUDA expression.
fn emit_cuda_expr(expr: &Expr, dtype: Dtype) -> String {
    emit_cuda_expr_replace(expr, "", "", dtype)
}

/// Emit a CUDA expression, replacing variable references.
fn emit_cuda_expr_replace(expr: &Expr, old_var: &str, new_var: &str, dtype: Dtype) -> String {
    match expr {
        Expr::Var(name) => {
            if name == old_var && !old_var.is_empty() {
                new_var.to_string()
            } else {
                name.clone()
            }
        }
        Expr::Const(v) => format_cuda_const(*v, dtype),
        Expr::IntConst(v) => format!("{v}"),
        Expr::BinOp { op, lhs, rhs } => {
            let l = emit_cuda_expr_replace(lhs, old_var, new_var, dtype);
            let r = emit_cuda_expr_replace(rhs, old_var, new_var, dtype);
            format!("({l} {op} {r})")
        }
        Expr::UnaryOp { op, operand } => {
            let inner = emit_cuda_expr_replace(operand, old_var, new_var, dtype);
            // CUDA C math header offers both single-precision (suffix `f`,
            // e.g. `expf`) and double-precision (no suffix, e.g. `exp`)
            // overloads. Pick the right family per dtype.
            let (exp_fn, log_fn, sqrt_fn, abs_fn, tanh_fn, fmax_fn) = match dtype {
                Dtype::F32 => ("expf", "logf", "sqrtf", "fabsf", "tanhf", "fmaxf"),
                Dtype::F64 => ("exp", "log", "sqrt", "fabs", "tanh", "fmax"),
            };
            // Suffixes on numeric literals: `1.0f` for f32, `1.0` for f64.
            let one = cuda_one_literal(dtype);
            let zero = cuda_zero_literal(dtype);
            let half = cuda_const_literal(0.5, dtype);
            let gelu_a = cuda_const_literal(0.797_884_560_8, dtype);
            let gelu_b = cuda_const_literal(0.044_715, dtype);
            match op {
                UnaryOpKind::Neg => format!("(-{inner})"),
                UnaryOpKind::Exp => format!("{exp_fn}({inner})"),
                UnaryOpKind::Log => format!("{log_fn}({inner})"),
                UnaryOpKind::Sqrt => format!("{sqrt_fn}({inner})"),
                UnaryOpKind::Abs => format!("{abs_fn}({inner})"),
                UnaryOpKind::Tanh => format!("{tanh_fn}({inner})"),
                UnaryOpKind::Sigmoid => {
                    format!("({one} / ({one} + {exp_fn}(-{inner})))")
                }
                UnaryOpKind::Relu => {
                    format!("{fmax_fn}({inner}, {zero})")
                }
                UnaryOpKind::Gelu => {
                    format!(
                        "({inner} * {half} * ({one} + {tanh_fn}({gelu_a} * ({inner} + {gelu_b} * {inner} * {inner} * {inner}))))"
                    )
                }
                UnaryOpKind::Silu => {
                    format!("({inner} / ({one} + {exp_fn}(-{inner})))")
                }
            }
        }
        Expr::FnCall { name, args } => {
            let args_s: Vec<String> = args
                .iter()
                .map(|a| emit_cuda_expr_replace(a, old_var, new_var, dtype))
                .collect();
            // CUDA `powf` is f32-only; the f64 form is `pow`.
            let pow_name = match dtype {
                Dtype::F32 => "powf",
                Dtype::F64 => "pow",
            };
            match name.as_str() {
                "powf" => format!("{pow_name}({})", args_s.join(", ")),
                _ => format!("{}({})", name, args_s.join(", ")),
            }
        }
        Expr::Index { buffer, index } => {
            let idx = emit_cuda_expr_replace(index, old_var, new_var, dtype);
            let buf = cuda_buffer_name(buffer);
            format!("{buf}[{idx}]")
        }
        Expr::Cast {
            target_type,
            operand,
        } => {
            let inner = emit_cuda_expr_replace(operand, old_var, new_var, dtype);
            format!("(({target_type}){inner})")
        }
    }
}

/// Map a [`Dtype`] to its CUDA C scalar type name.
fn cuda_scalar_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "float",
        Dtype::F64 => "double",
    }
}

/// Canonical CUDA C literal for `0.0` at the given precision.
fn cuda_zero_literal(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "0.0f",
        Dtype::F64 => "0.0",
    }
}

/// Canonical CUDA C literal for `1.0` at the given precision.
fn cuda_one_literal(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "1.0f",
        Dtype::F64 => "1.0",
    }
}

/// Format an arbitrary scalar `v` as a CUDA C literal of the given dtype
/// (no special-case handling — used for inline math constants where the
/// numeric value is fixed at codegen time).
fn cuda_const_literal(v: f64, dtype: Dtype) -> String {
    match dtype {
        Dtype::F32 => format!("{v}f"),
        Dtype::F64 => format!("{v}"),
    }
}

/// Format a scalar value as a CUDA literal of the given dtype. Picks
/// canonical short forms for `0.0`, `1.0`, `±INFINITY`, and `NAN`; emits a
/// plain decimal literal with a suffix matching `dtype` otherwise.
//
// Exact float comparison is intentional: only bit-identical `0.0` / `1.0`
// emit the canonical short literal.
#[allow(clippy::float_cmp)]
fn format_cuda_const(v: f64, dtype: Dtype) -> String {
    let (zero_lit, one_lit, suffix) = match dtype {
        Dtype::F32 => ("0.0f", "1.0f", "f"),
        Dtype::F64 => ("0.0", "1.0", ""),
    };
    if v == 0.0 {
        zero_lit.into()
    } else if v == 1.0 {
        one_lit.into()
    } else if v.is_infinite() && v > 0.0 {
        "INFINITY".into()
    } else if v.is_infinite() {
        "(-INFINITY)".into()
    } else if v.is_nan() {
        "NAN".into()
    } else {
        format!("{v}{suffix}")
    }
}

/// Map a buffer name to a CUDA identifier.
fn cuda_buffer_name(name: &str) -> String {
    if name == "out" || name == "output" {
        return "output".into();
    }
    name.into()
}

// ===========================================================================
// PTX code generation
// ===========================================================================

impl GpuCodegen {
    /// Generate PTX assembly from a `LoopIR` program.
    ///
    /// The generated kernel maps the outermost loop to GPU threads using
    /// `ctaid.x * ntid.x + tid.x` indexing. Inner loops become serial
    /// per-thread computation.
    ///
    /// The kernel operates on values of `dtype` (currently `f32` or `f64`);
    /// arithmetic, load/store, register declarations, and constant emission
    /// all dispatch on `dtype`. Transcendentals
    /// (`exp`/`log`/`sqrt`/`tanh`/`sigmoid`/`relu`/`abs`/`gelu`/`silu`)
    /// use approximate instructions (`ex2.approx.f32`, `lg2.approx.f32`,
    /// etc.) which exist only for f32 in PTX — see the module-level docs
    /// for the policy.
    ///
    /// # Arguments
    ///
    /// * `loops` - The loop IR to convert.
    /// * `fn_name` - The kernel entry point name.
    /// * `block_size` - The intended thread block size (used in documentation
    ///   comments; actual block size is set at launch time).
    /// * `num_inputs` - Number of input buffers.
    /// * `dtype` - Element dtype for all loads, stores, registers, and
    ///   constants emitted in the kernel.
    ///
    /// # Errors
    ///
    /// When `dtype` is `F64` and the program contains a transcendental op:
    /// - With the `cuda` feature enabled, the f64 path delegates to NVRTC
    ///   which links libdevice (`__nv_exp`/`__nv_log`/...) and returns a
    ///   PTX module with libdevice's polynomial expansions inlined.
    /// - Without the `cuda` feature, returns
    ///   `Err(JitError::Unsupported { op, dtype })` with a message naming
    ///   the missing feature. This preserves the `rust-gpu-discipline` §3
    ///   structured-rejection contract on builds without a CUDA toolkit.
    ///
    /// PTX itself has no `*.approx.f64` instructions, so the hand-written
    /// f64 path always routes through NVRTC + libdevice when transcendentals
    /// are present.
    pub fn generate_ptx_source(
        loops: &[LoopIR],
        fn_name: &str,
        block_size: usize,
        num_inputs: usize,
        dtype: Dtype,
    ) -> Result<String, JitError> {
        // f64 transcendentals: PTX has no `*.approx.f64` hardware
        // instructions, so the only correct path is libdevice (`__nv_exp`,
        // `__nv_log`, ...). NVRTC handles libdevice resolution
        // automatically when compiling CUDA C source. When the `cuda`
        // feature is on we delegate; otherwise we return a structured
        // error per rust-gpu-discipline §3.
        if dtype == Dtype::F64 {
            if let Some(op) = find_transcendental_op(loops) {
                return f64_transcendental_ptx(loops, fn_name, num_inputs, op);
            }
        }

        let mut out = String::new();
        let dn = dtype.name(); // "f32" or "f64"
        // PTX register width and address-shift width. f32 = 4-byte stride;
        // f64 = 8-byte stride. The byte-offset for a tid is `tid << shl`.
        let shl = match dtype {
            Dtype::F32 => 2,
            Dtype::F64 => 3,
        };

        // PTX header
        out.push_str(".version 7.0\n");
        out.push_str(".target sm_52\n");
        out.push_str(".address_size 64\n\n");

        // Kernel entry point with parameters
        out.push_str(&format!(".visible .entry {fn_name}(\n"));
        for i in 0..num_inputs {
            out.push_str(&format!("    .param .u64 in{i}_ptr,\n"));
        }
        out.push_str("    .param .u64 out_ptr,\n");
        out.push_str("    .param .u32 n\n");
        out.push_str(") {\n");

        // Analyze what registers and operations we need
        let needs = analyze_ptx_needs(loops);

        // Register declarations
        out.push_str("    .reg .u32 %tid, %bid, %bdim, %n_reg;\n");
        out.push_str("    .reg .u64 %off;\n");
        for i in 0..num_inputs {
            out.push_str(&format!("    .reg .u64 %in{i};\n"));
        }
        out.push_str("    .reg .u64 %out;\n");
        out.push_str(&format!("    .reg .{dn} %val;\n"));
        out.push_str("    .reg .pred %p;\n");

        if needs.extra_scratch_regs > 0 {
            for r in 0..needs.extra_scratch_regs {
                out.push_str(&format!("    .reg .{dn} %t{r};\n"));
            }
        }
        if needs.needs_loop_regs {
            out.push_str("    .reg .u32 %loop_i, %loop_end;\n");
            out.push_str("    .reg .u64 %loop_off;\n");
            out.push_str(&format!("    .reg .{dn} %acc;\n"));
        }
        if needs.needs_zero {
            out.push_str(&format!("    .reg .{dn} %zero;\n"));
        }

        out.push('\n');

        // Load parameters
        for i in 0..num_inputs {
            out.push_str(&format!("    ld.param.u64 %in{i}, [in{i}_ptr];\n"));
        }
        out.push_str("    ld.param.u64 %out, [out_ptr];\n");
        out.push_str("    ld.param.u32 %n_reg, [n];\n\n");

        // Thread index: tid = ctaid.x * ntid.x + tid.x
        out.push_str("    mov.u32 %bid, %ctaid.x;\n");
        out.push_str("    mov.u32 %bdim, %ntid.x;\n");
        out.push_str("    mov.u32 %tid, %tid.x;\n");
        out.push_str("    mad.lo.u32 %tid, %bid, %bdim, %tid;\n\n");

        // Bounds check
        out.push_str("    setp.ge.u32 %p, %tid, %n_reg;\n");
        out.push_str("    @%p bra DONE;\n\n");

        // Compute byte offset (shift by log2(sizeof(scalar)))
        out.push_str("    cvt.u64.u32 %off, %tid;\n");
        out.push_str(&format!("    shl.b64 %off, %off, {shl};\n\n"));

        // Add offset to base pointers
        for i in 0..num_inputs {
            out.push_str(&format!("    add.u64 %in{i}, %in{i}, %off;\n"));
        }
        out.push_str("    add.u64 %out, %out, %off;\n\n");

        // Load input value(s)
        if num_inputs >= 1 {
            out.push_str(&format!("    ld.global.{dn} %val, [%in0];\n"));
        }

        // Zero register if needed
        if needs.needs_zero {
            // Bit-pattern `0.0` is all zeros in both IEEE 754 binary32 and
            // binary64. Width of the literal differs, however: 8 hex digits
            // (`0f...`) for f32, 16 hex digits (`0d...`) for f64.
            let zero_lit = ptx_const_literal(0.0, dtype);
            out.push_str(&format!("    mov.{dn} %zero, {zero_lit};\n"));
        }

        out.push('\n');

        // Block size hint comment
        out.push_str(&format!("    // recommended block size: {block_size}\n\n"));

        // Emit the kernel body
        emit_ptx_body(&mut out, loops, dtype);

        // Store result
        out.push_str(&format!("\n    st.global.{dn} [%out], %val;\n\n"));

        out.push_str("DONE:\n");
        out.push_str("    ret;\n");
        out.push_str("}\n");

        Ok(out)
    }
}

/// Analysis result for PTX register and instruction needs.
struct PtxNeeds {
    extra_scratch_regs: usize,
    needs_loop_regs: bool,
    needs_zero: bool,
}

/// Analyze the loop IR to determine what PTX registers are needed.
fn analyze_ptx_needs(loops: &[LoopIR]) -> PtxNeeds {
    let mut extra = 0usize;
    let mut needs_loop = false;
    let mut needs_zero = false;

    analyze_ptx_needs_recursive(loops, &mut extra, &mut needs_loop, &mut needs_zero);

    PtxNeeds {
        extra_scratch_regs: extra,
        needs_loop_regs: needs_loop,
        needs_zero,
    }
}

fn analyze_ptx_needs_recursive(
    stmts: &[LoopIR],
    extra: &mut usize,
    needs_loop: &mut bool,
    needs_zero: &mut bool,
) {
    for stmt in stmts {
        match stmt {
            LoopIR::Loop { body, .. } => {
                // Inner loops in PTX need loop registers
                *needs_loop = true;
                analyze_ptx_needs_recursive(body, extra, needs_loop, needs_zero);
            }
            LoopIR::Store { value, .. }
            | LoopIR::Assign { value, .. }
            | LoopIR::Let { value, .. }
            | LoopIR::Accumulate { value, .. } => {
                count_expr_regs(value, extra, needs_zero);
            }
            LoopIR::If {
                condition,
                then_body,
                else_body,
                ..
            } => {
                count_expr_regs(condition, extra, needs_zero);
                analyze_ptx_needs_recursive(then_body, extra, needs_loop, needs_zero);
                analyze_ptx_needs_recursive(else_body, extra, needs_loop, needs_zero);
            }
            LoopIR::Comment(_) => {}
        }
    }
}

fn count_expr_regs(expr: &Expr, extra: &mut usize, needs_zero: &mut bool) {
    match expr {
        Expr::UnaryOp { op, operand } => {
            match op {
                UnaryOpKind::Sigmoid
                | UnaryOpKind::Tanh
                | UnaryOpKind::Gelu
                | UnaryOpKind::Silu => {
                    // These ops need scratch registers
                    *extra = (*extra).max(3);
                }
                UnaryOpKind::Relu | UnaryOpKind::Abs => {
                    *needs_zero = true;
                }
                _ => {}
            }
            count_expr_regs(operand, extra, needs_zero);
        }
        Expr::BinOp { lhs, rhs, .. } => {
            *extra = (*extra).max(1);
            count_expr_regs(lhs, extra, needs_zero);
            count_expr_regs(rhs, extra, needs_zero);
        }
        Expr::FnCall { args, .. } => {
            *extra = (*extra).max(1);
            for a in args {
                count_expr_regs(a, extra, needs_zero);
            }
        }
        Expr::Index { index, .. } => {
            count_expr_regs(index, extra, needs_zero);
        }
        _ => {}
    }
}

/// Emit PTX instructions for the loop body.
///
/// The outermost loop is already handled by the thread mapping.
/// This function emits PTX for the operations inside that loop.
fn emit_ptx_body(out: &mut String, stmts: &[LoopIR], dtype: Dtype) {
    let dn = dtype.name();
    for stmt in stmts {
        match stmt {
            LoopIR::Loop { body, .. } => {
                // Outermost loop: thread-mapped, process the body directly
                emit_ptx_body(out, body, dtype);
            }
            LoopIR::Let { var, value } => {
                if var == "val" {
                    // Initial load already done above
                    match value {
                        Expr::Index { buffer, .. } => {
                            // If loading from a non-primary input, emit the load
                            if buffer != "in0" {
                                if let Some(idx) = buffer.strip_prefix("in") {
                                    if idx.parse::<usize>().is_ok() {
                                        out.push_str(&format!(
                                            "    ld.global.{dn} %val, [%{buffer}];\n"
                                        ));
                                    }
                                }
                            }
                            // else: primary input already loaded
                        }
                        _ => {
                            emit_ptx_expr_to_reg(out, value, "%val", dtype);
                        }
                    }
                } else if var == "acc" {
                    // Accumulator initialization
                    match value {
                        Expr::Const(v) => {
                            let lit = ptx_const_literal(*v, dtype);
                            out.push_str(&format!("    mov.{dn} %acc, {lit};\n"));
                        }
                        _ => {
                            emit_ptx_expr_to_reg(out, value, "%acc", dtype);
                        }
                    }
                }
            }
            LoopIR::Assign { var, value } => {
                if var == "val" {
                    emit_ptx_op(out, value, dtype);
                } else if var == "acc" {
                    emit_ptx_expr_to_reg(out, value, "%acc", dtype);
                }
            }
            LoopIR::Accumulate { var, value } if var == "acc" => {
                // Load the value into a temp, then add to acc
                emit_ptx_expr_to_reg(out, value, "%t0", dtype);
                out.push_str(&format!("    add.{dn} %acc, %acc, %t0;\n"));
            }
            LoopIR::Store { value, .. } => {
                // Store already handled by the caller (st.global.<dn>)
                // But if the value is not %val, we need to move it there
                match value {
                    Expr::Var(v) if v == "acc" => {
                        out.push_str(&format!("    mov.{dn} %val, %acc;\n"));
                    }
                    Expr::Var(v) if v == "val" => {
                        // Already in %val
                    }
                    _ => {
                        emit_ptx_expr_to_reg(out, value, "%val", dtype);
                    }
                }
            }
            LoopIR::Comment(text) => {
                out.push_str(&format!("    // {text}\n"));
            }
            _ => {}
        }
    }
}

/// Emit PTX instructions to compute an expression and put the result in the
/// specified register.
fn emit_ptx_expr_to_reg(out: &mut String, expr: &Expr, dest: &str, dtype: Dtype) {
    let dn = dtype.name();
    match expr {
        Expr::Const(v) => {
            let lit = ptx_const_literal(*v, dtype);
            out.push_str(&format!("    mov.{dn} {dest}, {lit};\n"));
        }
        Expr::Var(name) => {
            let reg = ptx_var_to_reg(name);
            if reg != dest {
                out.push_str(&format!("    mov.{dn} {dest}, {reg};\n"));
            }
        }
        Expr::Index { buffer, .. } => {
            out.push_str(&format!("    ld.global.{dn} {dest}, [%{buffer}];\n"));
        }
        Expr::BinOp { op, lhs, rhs } => {
            emit_ptx_expr_to_reg(out, lhs, dest, dtype);
            emit_ptx_expr_to_reg(out, rhs, "%t0", dtype);
            // PTX has hardware `add`/`sub`/`mul` for both .f32 and .f64.
            // `div.approx.f64` does NOT exist; use the IEEE-rounded
            // `div.rn.f64` form instead.
            let div_op = match dtype {
                Dtype::F32 => "div.approx.f32",
                Dtype::F64 => "div.rn.f64",
            };
            let cvt_rzi = match dtype {
                Dtype::F32 => "cvt.rzi.f32.f32",
                Dtype::F64 => "cvt.rzi.f64.f64",
            };
            let ptx_op = match op {
                BinOpKind::Add => format!("add.{dn}"),
                BinOpKind::Sub => format!("sub.{dn}"),
                BinOpKind::Mul => format!("mul.{dn}"),
                BinOpKind::Div => div_op.to_string(),
                BinOpKind::Mod => {
                    // PTX doesn't have a direct fmod; approximate with
                    // a - floor(a/b) * b
                    out.push_str(&format!("    {div_op} %t1, {dest}, %t0;\n"));
                    out.push_str(&format!("    {cvt_rzi} %t1, %t1;\n"));
                    out.push_str(&format!("    mul.{dn} %t1, %t1, %t0;\n"));
                    out.push_str(&format!("    sub.{dn} {dest}, {dest}, %t1;\n"));
                    return;
                }
            };
            out.push_str(&format!("    {ptx_op} {dest}, {dest}, %t0;\n"));
        }
        Expr::UnaryOp { op, operand } => {
            emit_ptx_expr_to_reg(out, operand, dest, dtype);
            emit_ptx_unary_op(out, *op, dest, dtype);
        }
        Expr::FnCall { name, args } => {
            if name == "powf" && args.len() == 2 {
                // x^p = 2^(p * log2(x)) — uses ex2/lg2 .approx instructions
                // which exist only for .f32. f64 powf would have been
                // rejected up front by `find_transcendental_op`.
                debug_assert_eq!(
                    dtype,
                    Dtype::F32,
                    "powf reached emission with non-f32 dtype despite f64 reject guard",
                );
                emit_ptx_expr_to_reg(out, &args[0], dest, dtype);
                emit_ptx_expr_to_reg(out, &args[1], "%t0", dtype);
                out.push_str(&format!("    lg2.approx.f32 %t1, {dest};\n"));
                out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                out.push_str(&format!("    ex2.approx.f32 {dest}, %t1;\n"));
            } else {
                // Generic: just put the first arg in dest
                if let Some(arg) = args.first() {
                    emit_ptx_expr_to_reg(out, arg, dest, dtype);
                }
            }
        }
        _ => {}
    }
}

/// Emit PTX for a unary operation on a register.
///
/// All transcendentals here use `*.approx.f32` instructions. F64 graphs
/// containing transcendentals are rejected up-front by
/// [`find_transcendental_op`] before this function is reached, so the f32
/// codepath is the only one that needs to exist. (Hardware-arithmetic
/// unaries — `neg`, `abs` — work for both dtypes and dispatch on `dtype`.)
fn emit_ptx_unary_op(out: &mut String, op: UnaryOpKind, reg: &str, dtype: Dtype) {
    let dn = dtype.name();
    match op {
        UnaryOpKind::Neg => {
            // Hardware `neg` exists for both .f32 and .f64.
            out.push_str(&format!("    neg.{dn} {reg}, {reg};\n"));
        }
        UnaryOpKind::Abs => {
            // Hardware `abs` exists for both .f32 and .f64.
            out.push_str(&format!("    abs.{dn} {reg}, {reg};\n"));
        }
        // Everything below this line uses `*.approx.f32` instructions and
        // is unreachable for f64 (caller filters with `find_transcendental_op`).
        UnaryOpKind::Sqrt => {
            debug_assert_eq!(dtype, Dtype::F32, "sqrt: f64 should have been rejected");
            out.push_str(&format!("    sqrt.approx.f32 {reg}, {reg};\n"));
        }
        UnaryOpKind::Exp => {
            debug_assert_eq!(dtype, Dtype::F32, "exp: f64 should have been rejected");
            // exp(x) = 2^(x * log2(e))
            out.push_str(&format!("    mul.f32 {reg}, {reg}, 0f3FB8AA3B;\n")); // log2(e)
            out.push_str(&format!("    ex2.approx.f32 {reg}, {reg};\n"));
        }
        UnaryOpKind::Log => {
            debug_assert_eq!(dtype, Dtype::F32, "log: f64 should have been rejected");
            // log(x) = log2(x) / log2(e) = log2(x) * ln(2)
            out.push_str(&format!("    lg2.approx.f32 {reg}, {reg};\n"));
            out.push_str(&format!("    mul.f32 {reg}, {reg}, 0f3F317218;\n")); // ln(2)
        }
        UnaryOpKind::Relu => {
            // `max.f32` / `max.f64` both exist as hardware ops.
            out.push_str(&format!("    max.{dn} {reg}, {reg}, %zero;\n"));
        }
        UnaryOpKind::Sigmoid => {
            debug_assert_eq!(dtype, Dtype::F32, "sigmoid: f64 should have been rejected");
            // sigmoid(x) = 1 / (1 + exp(-x))
            out.push_str(&format!("    neg.f32 %t0, {reg};\n"));
            out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n"); // * log2(e)
            out.push_str("    ex2.approx.f32 %t0, %t0;\n");
            out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n"); // + 1.0
            out.push_str(&format!("    rcp.approx.f32 {reg}, %t0;\n"));
        }
        UnaryOpKind::Tanh => {
            debug_assert_eq!(dtype, Dtype::F32, "tanh: f64 should have been rejected");
            // tanh(x) = 2*sigmoid(2x) - 1
            out.push_str(&format!("    add.f32 {reg}, {reg}, {reg};\n")); // 2x
            out.push_str(&format!("    neg.f32 %t0, {reg};\n"));
            out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n");
            out.push_str("    ex2.approx.f32 %t0, %t0;\n");
            out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n");
            out.push_str(&format!("    rcp.approx.f32 {reg}, %t0;\n"));
            out.push_str(&format!("    add.f32 {reg}, {reg}, {reg};\n")); // 2*sigmoid(2x)
            out.push_str(&format!("    sub.f32 {reg}, {reg}, 0f3F800000;\n")); // -1
        }
        UnaryOpKind::Gelu => {
            debug_assert_eq!(dtype, Dtype::F32, "gelu: f64 should have been rejected");
            // GELU approx: x * sigmoid(1.702 * x)
            out.push_str(&format!("    mov.f32 %t2, {reg};\n")); // save x
            out.push_str("    mul.f32 %t0, %t2, 0f3FD9F16C;\n"); // 1.702 * x
            out.push_str("    neg.f32 %t0, %t0;\n");
            out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n");
            out.push_str("    ex2.approx.f32 %t0, %t0;\n");
            out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n");
            out.push_str("    rcp.approx.f32 %t0, %t0;\n"); // sigmoid(1.702*x)
            out.push_str(&format!("    mul.f32 {reg}, %t2, %t0;\n")); // x * sigmoid(1.702*x)
        }
        UnaryOpKind::Silu => {
            debug_assert_eq!(dtype, Dtype::F32, "silu: f64 should have been rejected");
            // SiLU: x * sigmoid(x)
            out.push_str(&format!("    mov.f32 %t2, {reg};\n")); // save x
            out.push_str(&format!("    neg.f32 %t0, {reg};\n"));
            out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n");
            out.push_str("    ex2.approx.f32 %t0, %t0;\n");
            out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n");
            out.push_str("    rcp.approx.f32 %t0, %t0;\n"); // sigmoid(x)
            out.push_str(&format!("    mul.f32 {reg}, %t2, %t0;\n")); // x * sigmoid(x)
        }
    }
}

/// Emit PTX for a composite operation stored in an Assign to %val.
fn emit_ptx_op(out: &mut String, expr: &Expr, dtype: Dtype) {
    let dn = dtype.name();
    match expr {
        Expr::UnaryOp { op, .. } => {
            emit_ptx_unary_op(out, *op, "%val", dtype);
        }
        Expr::BinOp { op, lhs, rhs } => {
            // Binary op where lhs is typically %val
            let _ = lhs; // lhs is already in %val
            emit_ptx_expr_to_reg(out, rhs, "%t0", dtype);
            let div_op = match dtype {
                Dtype::F32 => "div.approx.f32",
                Dtype::F64 => "div.rn.f64",
            };
            let cvt_rzi = match dtype {
                Dtype::F32 => "cvt.rzi.f32.f32",
                Dtype::F64 => "cvt.rzi.f64.f64",
            };
            let ptx_op = match op {
                BinOpKind::Add => format!("add.{dn}"),
                BinOpKind::Sub => format!("sub.{dn}"),
                BinOpKind::Mul => format!("mul.{dn}"),
                BinOpKind::Div => div_op.to_string(),
                BinOpKind::Mod => {
                    out.push_str(&format!("    {div_op} %t1, %val, %t0;\n"));
                    out.push_str(&format!("    {cvt_rzi} %t1, %t1;\n"));
                    out.push_str(&format!("    mul.{dn} %t1, %t1, %t0;\n"));
                    out.push_str(&format!("    sub.{dn} %val, %val, %t1;\n"));
                    return;
                }
            };
            out.push_str(&format!("    {ptx_op} %val, %val, %t0;\n"));
        }
        Expr::FnCall { name, args } => {
            if name == "powf" && args.len() == 2 {
                debug_assert_eq!(
                    dtype,
                    Dtype::F32,
                    "powf reached emission with non-f32 dtype despite f64 reject guard",
                );
                // x^p = 2^(p * log2(x)) — f32-only path.
                emit_ptx_expr_to_reg(out, &args[1], "%t0", dtype);
                out.push_str("    lg2.approx.f32 %t1, %val;\n");
                out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                out.push_str("    ex2.approx.f32 %val, %t1;\n");
            }
        }
        _ => {
            emit_ptx_expr_to_reg(out, expr, "%val", dtype);
        }
    }
}

/// Map a variable name to a PTX register. Defaults to `%val` for any
/// name other than the reserved `"acc"` accumulator register.
fn ptx_var_to_reg(name: &str) -> &str {
    match name {
        "acc" => "%acc",
        _ => "%val",
    }
}

/// Format a scalar `v` as a PTX hex literal of the given dtype.
///
/// PTX literal forms:
/// - `f32`: `0f` prefix + 8 hex digits = 32-bit IEEE 754 binary32 bit pattern.
/// - `f64`: `0d` prefix + 16 hex digits = 64-bit IEEE 754 binary64 bit pattern.
fn ptx_const_literal(v: f64, dtype: Dtype) -> String {
    match dtype {
        // The original f32 path narrowed via `(*v as f32).to_bits()` and
        // formatted as `{:08X}`. Preserved verbatim for byte-for-byte
        // compatibility with the existing PTX output.
        Dtype::F32 => format!("0f{:08X}", (v as f32).to_bits()),
        Dtype::F64 => format!("0d{:016X}", v.to_bits()),
    }
}

/// Human-readable name of a unary op for diagnostics.
///
/// Used by the `JitError::Unsupported` rejection path for f64
/// transcendentals when the `cuda` feature is not enabled, and by the
/// non-cuda-feature reject test. Tagged `cfg_attr(..., allow(dead_code))`
/// because with the `cuda` feature on the rejection path is unreachable
/// — NVRTC handles every transcendental — but the tests in this module
/// still reference it under one cfg branch or the other.
#[cfg_attr(feature = "cuda", allow(dead_code))]
fn ptx_unary_op_name(op: UnaryOpKind) -> &'static str {
    match op {
        UnaryOpKind::Neg => "neg",
        UnaryOpKind::Exp => "exp",
        UnaryOpKind::Log => "log",
        UnaryOpKind::Sqrt => "sqrt",
        UnaryOpKind::Abs => "abs",
        UnaryOpKind::Sigmoid => "sigmoid",
        UnaryOpKind::Tanh => "tanh",
        UnaryOpKind::Relu => "relu",
        UnaryOpKind::Gelu => "gelu",
        UnaryOpKind::Silu => "silu",
    }
}

/// Walk `stmts` looking for the first transcendental op. Returns `Some(op)`
/// if found, `None` otherwise.
///
/// "Transcendental" here means any unary that the PTX backend lowers via
/// `*.approx.f32` instructions (no f64 hardware equivalent), plus `powf`
/// which is implemented via `lg2/ex2` on f32. Pure arithmetic unaries
/// (`neg`, `abs`) and hardware `max` (`relu`) are not included because
/// they have direct .f64 hardware support.
fn find_transcendental_op(stmts: &[LoopIR]) -> Option<UnaryOpKind> {
    fn walk_expr(expr: &Expr) -> Option<UnaryOpKind> {
        match expr {
            Expr::UnaryOp { op, operand } => {
                if matches!(
                    op,
                    UnaryOpKind::Exp
                        | UnaryOpKind::Log
                        | UnaryOpKind::Sqrt
                        | UnaryOpKind::Sigmoid
                        | UnaryOpKind::Tanh
                        | UnaryOpKind::Gelu
                        | UnaryOpKind::Silu
                ) {
                    return Some(*op);
                }
                walk_expr(operand)
            }
            Expr::BinOp { lhs, rhs, .. } => walk_expr(lhs).or_else(|| walk_expr(rhs)),
            Expr::FnCall { name, args } => {
                if name == "powf" {
                    // Bucket `powf` under `Exp` since it uses ex2/lg2 internally.
                    return Some(UnaryOpKind::Exp);
                }
                args.iter().find_map(walk_expr)
            }
            Expr::Index { index, .. } => walk_expr(index),
            Expr::Cast { operand, .. } => walk_expr(operand),
            _ => None,
        }
    }

    for stmt in stmts {
        let hit = match stmt {
            LoopIR::Loop { body, .. } => find_transcendental_op(body),
            LoopIR::Store { value, index, .. } => walk_expr(value).or_else(|| walk_expr(index)),
            LoopIR::Let { value, .. }
            | LoopIR::Assign { value, .. }
            | LoopIR::Accumulate { value, .. } => walk_expr(value),
            LoopIR::If {
                condition,
                then_body,
                else_body,
            } => walk_expr(condition)
                .or_else(|| find_transcendental_op(then_body))
                .or_else(|| find_transcendental_op(else_body)),
            LoopIR::Comment(_) => None,
        };
        if hit.is_some() {
            return hit;
        }
    }
    None
}

// ---------------------------------------------------------------------------
// F64 transcendental PTX via NVRTC + libdevice
// ---------------------------------------------------------------------------

/// Lower an f64 program containing a transcendental op to PTX by routing
/// through NVRTC.
///
/// PTX has no `*.approx.f64` hardware instructions for `exp`/`log`/`sqrt`/
/// `tanh`/etc. The correct lowering is libdevice (`__nv_exp(double)`,
/// `__nv_log(double)`, ...), which NVRTC links automatically when
/// compiling CUDA C source. So the f64 transcendental path is implemented
/// as: emit CUDA C source via [`emit_cuda_source`] (which already supports
/// f64 transcendentals via the host math headers), invoke NVRTC, and
/// return the resulting PTX text.
///
/// Behind the `cuda` feature flag because NVRTC requires linking against
/// the CUDA toolkit. Without the feature we return a structured error
/// per `rust-gpu-discipline` §3 (PyTorch-parity `NotImplementedError` shape).
///
/// # Arguments
///
/// * `loops` — the IR program (already verified to contain an f64
///   transcendental op by the caller).
/// * `fn_name` — kernel entry point name.
/// * `num_inputs` — number of input buffers.
/// * `failing_op` — the op that triggered the f64-transcendental dispatch;
///   used in the error variant when the `cuda` feature is not enabled.
fn f64_transcendental_ptx(
    loops: &[LoopIR],
    fn_name: &str,
    num_inputs: usize,
    #[cfg_attr(feature = "cuda", allow(unused_variables))] failing_op: UnaryOpKind,
) -> Result<String, JitError> {
    // Generate CUDA C source — the existing path supports f64
    // transcendentals via host math headers (`exp`, `log`, `tanh`,
    // `sqrt`, `pow` for double overloads). NVRTC inlines libdevice's
    // implementations of those when it lowers to PTX.
    let cuda_source = GpuCodegen::generate_cuda_source(loops, fn_name, num_inputs, Dtype::F64)?;

    #[cfg(feature = "cuda")]
    {
        compile_cuda_source_to_ptx(&cuda_source, fn_name)
    }

    #[cfg(not(feature = "cuda"))]
    {
        // The cuda feature is the only thing standing between us and a
        // working f64 transcendental kernel; surface that explicitly so
        // callers know to enable it. The CUDA C source we generated would
        // compile cleanly under NVRTC.
        let _ = cuda_source; // silence unused-binding warning on this cfg
        Err(JitError::Unsupported {
            op: ptx_unary_op_name(failing_op).to_string(),
            dtype: Dtype::F64.name().to_string(),
        })
    }
}

/// NVRTC-compile a CUDA C source string to a PTX module string.
///
/// NVRTC links libdevice automatically when the source uses f64 math
/// intrinsics (`exp`, `log`, `tanh`, `pow`, ...), so the resulting PTX
/// has no unresolved external symbols — every `__nv_*` call is replaced
/// with libdevice's polynomial expansion inlined into the kernel.
///
/// `kernel_name` is documented in the error message for traceability;
/// NVRTC does not need it for compilation (the kernel's `__global__`
/// declaration in `cuda_source` is what NVRTC keys on).
#[cfg(feature = "cuda")]
fn compile_cuda_source_to_ptx(cuda_source: &str, kernel_name: &str) -> Result<String, JitError> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    // NVRTC ships its own math intrinsics (`exp`, `log`, `sqrt`, `tanh`,
    // `pow`, ...) for both f32 and f64 — these are auto-available without
    // any `#include`. The `#include <math.h>` line that
    // `generate_cuda_source` prepends targets host nvcc compilation,
    // where libc's <math.h> declares the host overloads. NVRTC has no
    // host headers in its include path and rejects the line. Strip it
    // before compilation; the device-math symbols are still resolved.
    //
    // We also rewrite `__global__ void <name>(...)` to
    // `extern "C" __global__ void <name>(...)`. Without `extern "C"`,
    // NVRTC C++-mangles the symbol (e.g.
    // `_Z9k_f64_expPKdPdi`); cudarc's `cuModuleGetFunction` keys on the
    // unmangled name so the load would fail. The CUDA C codegen's
    // `__global__` declarations are flat C-style signatures, so
    // `extern "C"` is safe — there's no overloading to disambiguate.
    let nvrtc_source = cuda_source
        .lines()
        .filter(|l| !l.trim().starts_with("#include <math.h>"))
        .map(|l| {
            if l.starts_with("__global__ void ") {
                format!("extern \"C\" {l}")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    // sm_75 is the floor for non-deprecated NVRTC targets in CUDA 13.x
    // (Volta sm_70 emits a deprecation warning) and supports f64
    // hardware ops on every Turing-and-newer GPU. libdevice's
    // polynomial expansions for f64 transcendentals depend on f64 FMA,
    // which is universally available at this baseline.
    let opts = CompileOptions {
        arch: Some("compute_75"),
        // `--use_fast_math` would enable approximate intrinsics that
        // sacrifice f64 precision; we want libdevice's IEEE-correct
        // polynomial expansions instead.
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(&nvrtc_source, opts).map_err(|e| JitError::CodegenError {
        message: format!(
            "NVRTC compile of CUDA C source for f64 transcendental kernel '{kernel_name}' failed: {e}"
        ),
    })?;

    // `ptx.to_src()` returns the PTX text. NVRTC stores the compiled
    // image as a CString-bytes payload internally; `to_src` decodes it
    // back to a regular Rust String for downstream `Ptx::from_src`
    // consumption.
    Ok(ptx.to_src())
}

// ---------------------------------------------------------------------------
// Helpers for reduction detection
// ---------------------------------------------------------------------------

/// Check if any statements contain an Accumulate operation.
fn loops_contain_accumulate(stmts: &[LoopIR]) -> bool {
    for stmt in stmts {
        match stmt {
            LoopIR::Accumulate { .. } => return true,
            LoopIR::Loop { body, .. } if loops_contain_accumulate(body) => {
                return true;
            }
            LoopIR::If {
                then_body,
                else_body,
                ..
            } if (loops_contain_accumulate(then_body) || loops_contain_accumulate(else_body)) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Find the expression being accumulated in a reduction.
fn find_accumulate_expr(stmts: &[LoopIR]) -> Option<Expr> {
    for stmt in stmts {
        match stmt {
            LoopIR::Accumulate { value, .. } => return Some(value.clone()),
            LoopIR::Loop { body, .. } => {
                if let Some(expr) = find_accumulate_expr(body) {
                    return Some(expr);
                }
            }
            _ => {}
        }
    }
    None
}

/// Check if the loops represent a mean reduction (divide by count).
fn find_mean_divisor(stmts: &[LoopIR]) -> Option<f64> {
    for stmt in stmts {
        if let LoopIR::Store {
            value:
                Expr::BinOp {
                    op: BinOpKind::Div,
                    rhs,
                    ..
                },
            ..
        } = stmt
        {
            if let Expr::Const(divisor) = rhs.as_ref() {
                return Some(*divisor);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_ir;
    use crate::graph::IrOpKind;

    // -----------------------------------------------------------------------
    // CUDA codegen tests (F32)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cuda_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_neg", 1, Dtype::F32).unwrap();

        assert!(src.contains("__global__ void kernel_neg"));
        assert!(src.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(src.contains("if (tid >= n) return"));
        assert!(src.contains("const float* __restrict__ in0"));
        assert!(src.contains("float* __restrict__ output"));
    }

    #[test]
    fn test_cuda_binary_add() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_add", 2, Dtype::F32).unwrap();

        assert!(src.contains("const float* __restrict__ in0"));
        assert!(src.contains("const float* __restrict__ in1"));
        assert!(src.contains('+'));
    }

    #[test]
    fn test_cuda_sigmoid() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_cuda_source(&loops, "kernel_sigmoid", 1, Dtype::F32).unwrap();

        assert!(src.contains("expf("));
        assert!(src.contains("1.0f"));
    }

    #[test]
    fn test_cuda_relu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Relu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_relu", 1, Dtype::F32).unwrap();

        assert!(src.contains("fmaxf("));
    }

    #[test]
    fn test_cuda_reduction_sum() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_sum", 1, Dtype::F32).unwrap();

        assert!(src.contains("__shared__"));
        assert!(src.contains("__syncthreads"));
        assert!(src.contains("atomicAdd"));
    }

    #[test]
    fn test_cuda_reduction_mean() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Mean], &["in0"], "out", 100);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_mean", 1, Dtype::F32).unwrap();

        assert!(src.contains("__shared__"));
        assert!(src.contains("(float)n"));
    }

    #[test]
    fn test_cuda_fused_chain() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Relu];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_fused", 1, Dtype::F32).unwrap();

        assert!(src.contains("__global__"));
        assert!(src.contains("tid"));
    }

    #[test]
    fn test_cuda_gelu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Gelu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_gelu", 1, Dtype::F32).unwrap();

        assert!(src.contains("tanhf("));
        assert!(src.contains("0.044715"));
    }

    #[test]
    fn test_cuda_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_silu", 1, Dtype::F32).unwrap();

        assert!(src.contains("expf("));
    }

    // -----------------------------------------------------------------------
    // PTX codegen tests (F32)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ptx_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_neg", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains(".version 7.0"));
        assert!(src.contains(".target sm_52"));
        assert!(src.contains(".visible .entry kernel_neg"));
        assert!(src.contains("neg.f32 %val, %val"));
        assert!(src.contains("st.global.f32 [%out], %val"));
        assert!(src.contains("ret;"));
    }

    #[test]
    fn test_ptx_relu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Relu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_relu", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("max.f32 %val, %val, %zero"));
    }

    #[test]
    fn test_ptx_sigmoid() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_sigmoid", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_sqrt() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sqrt], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_sqrt", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("sqrt.approx.f32"));
    }

    #[test]
    fn test_ptx_exp() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Exp], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_exp", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("3FB8AA3B")); // log2(e) float bits
    }

    #[test]
    fn test_ptx_log() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Log], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_log", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("lg2.approx.f32"));
        assert!(src.contains("3F317218")); // ln(2) float bits
    }

    #[test]
    fn test_ptx_tanh() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Tanh], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_tanh", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("rcp.approx.f32"));
        assert!(src.contains("sub.f32")); // -1 step
    }

    #[test]
    fn test_ptx_fused_chain() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Relu, IrOpKind::Sigmoid];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_fused", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("neg.f32"));
        assert!(src.contains("max.f32"));
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_block_size_comment() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel", 512, 1, Dtype::F32).unwrap();
        assert!(src.contains("recommended block size: 512"));
    }

    #[test]
    fn test_ptx_multiple_inputs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 4);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_add", 256, 2, Dtype::F32).unwrap();

        assert!(src.contains("in0_ptr"));
        assert!(src.contains("in1_ptr"));
        assert!(src.contains("%in0"));
        assert!(src.contains("%in1"));
    }

    #[test]
    fn test_ptx_gelu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Gelu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_gelu", 256, 1, Dtype::F32).unwrap();

        // Should use GELU approximation: x * sigmoid(1.702 * x)
        assert!(src.contains("3FD9F16C")); // 1.702 float bits
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_silu", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_abs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Abs], &["in0"], "out", 4);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_abs", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("abs.f32"));
    }

    #[test]
    fn test_format_cuda_const_f32_values() {
        assert_eq!(format_cuda_const(0.0, Dtype::F32), "0.0f");
        assert_eq!(format_cuda_const(1.0, Dtype::F32), "1.0f");
        assert_eq!(format_cuda_const(f64::INFINITY, Dtype::F32), "INFINITY");
        assert_eq!(
            format_cuda_const(f64::NEG_INFINITY, Dtype::F32),
            "(-INFINITY)"
        );
        assert_eq!(format_cuda_const(f64::NAN, Dtype::F32), "NAN");
    }

    // -----------------------------------------------------------------------
    // F64 dispatch tests (#729)
    // -----------------------------------------------------------------------

    /// CUDA C: F64 elementwise neg emits `double` declarations and a `-x`
    /// expression, with no `f`-suffix on numeric literals.
    #[test]
    fn test_cuda_f64_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_neg_f64", 1, Dtype::F64)
            .expect("F64 neg should generate CUDA C");

        assert!(
            src.contains("const double* __restrict__ in0"),
            "expected double input declaration; got:\n{src}"
        );
        assert!(
            src.contains("double* __restrict__ output"),
            "expected double output declaration; got:\n{src}"
        );
        // No f-suffixed `float` declarations should sneak in.
        assert!(
            !src.contains("const float*"),
            "F64 path leaked an f32 input decl:\n{src}"
        );
        assert!(
            !src.contains("float* __restrict__ output"),
            "F64 path leaked an f32 output decl:\n{src}"
        );
    }

    /// PTX: F64 add round-trip — load.f64, add.f64, store.f64, and `.reg .f64`.
    #[test]
    fn test_ptx_f64_binary_add() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_add_f64", 256, 2, Dtype::F64)
            .expect("F64 add should generate PTX");

        assert!(
            src.contains("ld.global.f64"),
            "expected f64 load; got:\n{src}"
        );
        assert!(src.contains("add.f64"), "expected add.f64; got:\n{src}");
        assert!(
            src.contains("st.global.f64 [%out], %val"),
            "expected f64 store; got:\n{src}"
        );
        assert!(
            src.contains(".reg .f64 %val"),
            "expected f64 %val reg; got:\n{src}"
        );
        // shl by 3 (sizeof double = 8 bytes) — not 2.
        assert!(
            src.contains("shl.b64 %off, %off, 3;"),
            "expected shl by 3 for 8-byte stride; got:\n{src}"
        );
    }

    /// PTX: F64 elementwise mul/sub/div all dispatch correctly.
    #[test]
    fn test_ptx_f64_arith_dispatch() {
        // mul
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Mul], &["in0", "in1"], "out", 8);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_mul", 256, 2, Dtype::F64).unwrap();
        assert!(src.contains("mul.f64"), "missing mul.f64 in:\n{src}");

        // sub
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sub], &["in0", "in1"], "out", 8);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_sub", 256, 2, Dtype::F64).unwrap();
        assert!(src.contains("sub.f64"), "missing sub.f64 in:\n{src}");

        // div — uses div.rn.f64 (no .approx for f64)
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Div], &["in0", "in1"], "out", 8);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_div", 256, 2, Dtype::F64).unwrap();
        assert!(src.contains("div.rn.f64"), "missing div.rn.f64 in:\n{src}");
        assert!(
            !src.contains("div.approx.f64"),
            "div.approx.f64 is invalid PTX, must not be emitted:\n{src}"
        );
    }

    /// PTX: F64 const emission uses `0d` prefix and 16 hex digits.
    #[test]
    fn test_ptx_f64_const_format() {
        // Direct unit test of the literal helper — bit-for-bit canonical
        // representation of 1.0 and 0.5 in IEEE 754 binary64.
        assert_eq!(
            ptx_const_literal(1.0_f64, Dtype::F64),
            "0d3FF0000000000000",
            "1.0 should encode as 3FF0000000000000",
        );
        assert_eq!(
            ptx_const_literal(0.5_f64, Dtype::F64),
            "0d3FE0000000000000",
            "0.5 should encode as 3FE0000000000000",
        );
        // f32 path unchanged: 1.0 encodes as 3F800000.
        assert_eq!(ptx_const_literal(1.0_f64, Dtype::F32), "0f3F800000");
        assert_eq!(ptx_const_literal(0.5_f64, Dtype::F32), "0f3F000000");
    }

    /// PTX: F64 abs uses hardware `abs.f64`, not `abs.f32` — and the
    /// `%zero` register (when present) is .f64.
    #[test]
    fn test_ptx_f64_abs_hardware() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Abs], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_abs", 256, 1, Dtype::F64).unwrap();
        assert!(src.contains("abs.f64"), "expected abs.f64; got:\n{src}");
        assert!(!src.contains("abs.f32"), "F64 path leaked abs.f32:\n{src}");
    }

    /// PTX: F64 graphs containing transcendentals are rejected with
    /// `JitError::Unsupported` when the `cuda` feature is *not* enabled.
    /// Covers exp/log/sqrt/tanh/sigmoid/gelu/silu.
    ///
    /// With the `cuda` feature ON the f64 path delegates to NVRTC + libdevice
    /// and returns valid PTX — see [`test_ptx_f64_transcendental_succeeds`].
    /// This test only exercises the off-by-default fallback and is therefore
    /// only meaningful in builds without `--features cuda`.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_ptx_f64_transcendental_rejected_without_cuda_feature() {
        for (op_kind, expected_name) in [
            (IrOpKind::Exp, "exp"),
            (IrOpKind::Log, "log"),
            (IrOpKind::Sqrt, "sqrt"),
            (IrOpKind::Tanh, "tanh"),
            (IrOpKind::Sigmoid, "sigmoid"),
            (IrOpKind::Gelu, "gelu"),
            (IrOpKind::Silu, "silu"),
        ] {
            let loops =
                codegen_ir::lower_to_loops(std::slice::from_ref(&op_kind), &["in0"], "out", 4);
            let err = GpuCodegen::generate_ptx_source(&loops, "k", 256, 1, Dtype::F64).expect_err(
                &format!("f64 {expected_name} must reject without `cuda` feature"),
            );
            match err {
                JitError::Unsupported { op, dtype } => {
                    assert_eq!(op, expected_name, "wrong op name in error");
                    assert_eq!(dtype, "f64", "wrong dtype name in error");
                }
                other => panic!("expected JitError::Unsupported, got {other:?}"),
            }
        }
    }

    /// PTX: F64 graphs containing transcendentals succeed when the
    /// `cuda` feature is enabled, by routing through NVRTC + libdevice
    /// (#748). Verifies the resulting PTX is non-empty and contains the
    /// expected `.entry` for the kernel — meaning NVRTC successfully
    /// compiled the CUDA C source we generated, libdevice's f64 math
    /// expansions are inlined, and the result is a valid module the
    /// driver can load.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_ptx_f64_transcendental_succeeds() {
        for (op_kind, expected_name) in [
            (IrOpKind::Exp, "exp"),
            (IrOpKind::Log, "log"),
            (IrOpKind::Sqrt, "sqrt"),
            (IrOpKind::Tanh, "tanh"),
            (IrOpKind::Sigmoid, "sigmoid"),
            (IrOpKind::Gelu, "gelu"),
            (IrOpKind::Silu, "silu"),
        ] {
            let loops =
                codegen_ir::lower_to_loops(std::slice::from_ref(&op_kind), &["in0"], "out", 4);
            let kernel_name = format!("k_f64_{expected_name}");
            let ptx = GpuCodegen::generate_ptx_source(&loops, &kernel_name, 256, 1, Dtype::F64)
                .unwrap_or_else(|e| panic!("f64 {expected_name} via NVRTC must succeed: {e:?}"));

            assert!(
                ptx.contains(".version"),
                "f64 {expected_name} PTX missing `.version` header:\n{ptx}",
            );
            assert!(
                ptx.contains(".target sm_70")
                    || ptx.contains(".target sm_7")
                    || ptx.contains(".target sm_8")
                    || ptx.contains(".target sm_9"),
                "f64 {expected_name} PTX missing expected target arch:\n{ptx}",
            );
            assert!(
                ptx.contains(&format!(".entry {kernel_name}")),
                "f64 {expected_name} PTX missing entry point '{kernel_name}':\n{ptx}",
            );
            // NVRTC lowers each f64 transcendental to a libdevice
            // expansion (polynomial fma chains for exp/log/tanh/sigmoid/
            // gelu/silu, direct hardware for sqrt). At least ONE f64
            // hardware op must appear — `fma.rn.f64`, `mul.f64`,
            // `add.f64`, `sqrt.rn.f64`, `div.rn.f64`, or `sub.f64`.
            assert!(
                ptx.contains("fma.rn.f64")
                    || ptx.contains("mul.f64")
                    || ptx.contains("add.f64")
                    || ptx.contains("sqrt.rn.f64")
                    || ptx.contains("sqrt.approx.f64")
                    || ptx.contains("div.rn.f64")
                    || ptx.contains("sub.f64"),
                "f64 {expected_name} PTX missing f64 hardware ops:\n{ptx}",
            );
        }
    }

    /// PTX: F64 powf via NVRTC + libdevice. Separate test from the unary
    /// transcendentals because powf takes two operands and goes through
    /// the `FnCall` path rather than `UnaryOp`.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_ptx_f64_powf_succeeds() {
        let loops =
            codegen_ir::lower_to_loops(&[IrOpKind::Pow { exponent: 2.5 }], &["in0"], "out", 4);
        let ptx = GpuCodegen::generate_ptx_source(&loops, "k_f64_pow", 256, 1, Dtype::F64)
            .expect("f64 pow via NVRTC must succeed");
        assert!(
            ptx.contains(".entry k_f64_pow"),
            "f64 pow PTX missing entry point:\n{ptx}",
        );
        assert!(
            ptx.contains("fma.rn.f64") || ptx.contains("mul.f64"),
            "f64 pow PTX missing f64 hardware ops:\n{ptx}",
        );
    }

    /// PTX: F64 graphs with only hardware-supported ops (neg, abs, relu, add,
    /// mul, sub, div) must succeed. Belt-and-braces against the rejection
    /// guard being too aggressive.
    #[test]
    fn test_ptx_f64_hardware_ops_succeed() {
        // neg
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "k", 256, 1, Dtype::F64).unwrap();
        assert!(src.contains("neg.f64"));

        // relu (hardware max.f64)
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Relu], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "k", 256, 1, Dtype::F64).unwrap();
        assert!(src.contains("max.f64"));
    }

    /// PTX: F64 fused arithmetic chain (neg then add) — both ops dispatch
    /// to f64 in a single kernel.
    #[test]
    fn test_ptx_f64_fused_arith() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Abs];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_fused", 256, 1, Dtype::F64).unwrap();

        assert!(src.contains("neg.f64"));
        assert!(src.contains("abs.f64"));
        // No leakage of f32 paths.
        assert!(
            !src.contains(".f32"),
            "F64 fused chain leaked .f32 instruction:\n{src}"
        );
    }

    /// CUDA C: F64 transcendentals lower to host-math doubles (`exp`, `tanh`)
    /// and don't use the `f` suffix. CUDA C path supports both dtypes.
    #[test]
    fn test_cuda_f64_transcendental_ok() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 4);
        let src = GpuCodegen::generate_cuda_source(&loops, "k", 1, Dtype::F64)
            .expect("CUDA C f64 sigmoid should generate");
        assert!(src.contains("exp("), "expected double exp; got:\n{src}");
        assert!(!src.contains("expf("), "F64 path leaked expf:\n{src}");
    }
}
