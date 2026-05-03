//! GPU code generators: emit CUDA C and PTX source from [`LoopIR`].
//!
//! Two code generators are provided:
//!
//! - [`GpuCodegen::generate_cuda_source`] — emits CUDA C with `__global__`
//!   kernels, `blockIdx`/`threadIdx` mapping, and shared memory for reductions.
//! - [`GpuCodegen::generate_ptx_source`] — emits PTX assembly targeting
//!   `sm_52` with hand-scheduled register allocation and approximate
//!   transcendental instructions.
//!
//! Both generators convert the outermost loop of a `LoopIR` program into
//! thread-parallel GPU execution while keeping inner loops as thread-local
//! serial computation.

use crate::codegen_ir::{BinOpKind, Expr, LoopIR, UnaryOpKind};

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
    pub fn generate_cuda_source(loops: &[LoopIR], fn_name: &str, num_inputs: usize) -> String {
        let mut out = String::new();

        out.push_str("#include <math.h>\n\n");

        // Detect if this is a reduction (has accumulate statements)
        let has_reduction = loops_contain_accumulate(loops);

        // Build function signature
        out.push_str(&format!("__global__ void {fn_name}(\n"));
        for i in 0..num_inputs {
            out.push_str(&format!("    const float* __restrict__ in{i},\n"));
        }
        out.push_str("    float* __restrict__ output,\n");
        out.push_str("    int n\n");
        out.push_str(") {\n");

        // Thread index computation
        out.push_str("    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");

        if has_reduction {
            emit_cuda_reduction(&mut out, loops, num_inputs);
        } else {
            // Elementwise / matmul: each thread handles one element of the
            // outermost loop
            emit_cuda_elementwise(&mut out, loops);
        }

        out.push_str("}\n");
        out
    }
}

/// Emit CUDA code for elementwise operations where the outer loop maps to threads.
fn emit_cuda_elementwise(out: &mut String, loops: &[LoopIR]) {
    out.push_str("    if (tid >= n) return;\n\n");

    for stmt in loops {
        match stmt {
            LoopIR::Loop { var, body, .. } => {
                // The outermost loop variable becomes `tid`
                out.push_str(&format!("    // outer loop var '{var}' -> tid\n"));
                for s in body {
                    emit_cuda_stmt_with_var_replace(out, s, var, "tid", 1);
                }
            }
            other => {
                emit_cuda_stmt(out, other, 1);
            }
        }
    }
}

/// Emit CUDA code for reduction operations using shared memory.
fn emit_cuda_reduction(out: &mut String, loops: &[LoopIR], _num_inputs: usize) {
    out.push_str("    extern __shared__ float sdata[];\n");
    out.push_str("    int local_tid = threadIdx.x;\n\n");

    // Each thread computes a partial result, then we tree-reduce in shared memory
    out.push_str("    // Load phase: each thread accumulates elements stride-apart\n");
    out.push_str("    float thread_acc = 0.0f;\n");
    out.push_str("    for (int idx = tid; idx < n; idx += blockDim.x * gridDim.x) {\n");

    // Find the accumulation expression
    let acc_expr = find_accumulate_expr(loops);
    match acc_expr {
        Some(expr) => {
            let val = emit_cuda_expr_replace(&expr, "i", "idx");
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
            out.push_str("        atomicAdd(&output[0], sdata[0] / (float)n);\n");
        }
        None => {
            out.push_str("        atomicAdd(&output[0], sdata[0]);\n");
        }
    }

    out.push_str("    }\n");
}

/// Emit a CUDA C statement at the given indentation.
fn emit_cuda_stmt(out: &mut String, stmt: &LoopIR, indent: usize) {
    let pad = "    ".repeat(indent);

    match stmt {
        LoopIR::Loop {
            var,
            start,
            end,
            body,
        } => {
            let start_s = emit_cuda_expr(start);
            let end_s = emit_cuda_expr(end);
            out.push_str(&format!(
                "{pad}for (int {var} = {start_s}; {var} < {end_s}; {var}++) {{\n"
            ));
            for s in body {
                emit_cuda_stmt(out, s, indent + 1);
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Store {
            buffer,
            index,
            value,
        } => {
            let idx = emit_cuda_expr(index);
            let val = emit_cuda_expr(value);
            let buf = cuda_buffer_name(buffer);
            out.push_str(&format!("{pad}{buf}[{idx}] = {val};\n"));
        }

        LoopIR::Let { var, value } => {
            let val = emit_cuda_expr(value);
            out.push_str(&format!("{pad}float {var} = {val};\n"));
        }

        LoopIR::Assign { var, value } => {
            let val = emit_cuda_expr(value);
            out.push_str(&format!("{pad}{var} = {val};\n"));
        }

        LoopIR::Accumulate { var, value } => {
            let val = emit_cuda_expr(value);
            out.push_str(&format!("{pad}{var} += {val};\n"));
        }

        LoopIR::If {
            condition,
            then_body,
            else_body,
        } => {
            let cond = emit_cuda_expr(condition);
            out.push_str(&format!("{pad}if ({cond}) {{\n"));
            for s in then_body {
                emit_cuda_stmt(out, s, indent + 1);
            }
            if !else_body.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_cuda_stmt(out, s, indent + 1);
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
) {
    let pad = "    ".repeat(indent);

    match stmt {
        LoopIR::Loop {
            var,
            start,
            end,
            body,
        } => {
            let start_s = emit_cuda_expr_replace(start, old_var, new_var);
            let end_s = emit_cuda_expr_replace(end, old_var, new_var);
            out.push_str(&format!(
                "{pad}for (int {var} = {start_s}; {var} < {end_s}; {var}++) {{\n"
            ));
            for s in body {
                emit_cuda_stmt_with_var_replace(out, s, old_var, new_var, indent + 1);
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Store {
            buffer,
            index,
            value,
        } => {
            let idx = emit_cuda_expr_replace(index, old_var, new_var);
            let val = emit_cuda_expr_replace(value, old_var, new_var);
            let buf = cuda_buffer_name(buffer);
            out.push_str(&format!("{pad}{buf}[{idx}] = {val};\n"));
        }

        LoopIR::Let { var, value } => {
            let val = emit_cuda_expr_replace(value, old_var, new_var);
            out.push_str(&format!("{pad}float {var} = {val};\n"));
        }

        LoopIR::Assign { var, value } => {
            let val = emit_cuda_expr_replace(value, old_var, new_var);
            let actual_var = if var == old_var {
                new_var
            } else {
                var.as_str()
            };
            out.push_str(&format!("{pad}{actual_var} = {val};\n"));
        }

        LoopIR::Accumulate { var, value } => {
            let val = emit_cuda_expr_replace(value, old_var, new_var);
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
            let cond = emit_cuda_expr_replace(condition, old_var, new_var);
            out.push_str(&format!("{pad}if ({cond}) {{\n"));
            for s in then_body {
                emit_cuda_stmt_with_var_replace(out, s, old_var, new_var, indent + 1);
            }
            if !else_body.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_cuda_stmt_with_var_replace(out, s, old_var, new_var, indent + 1);
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
fn emit_cuda_expr(expr: &Expr) -> String {
    emit_cuda_expr_replace(expr, "", "")
}

/// Emit a CUDA expression, replacing variable references.
fn emit_cuda_expr_replace(expr: &Expr, old_var: &str, new_var: &str) -> String {
    match expr {
        Expr::Var(name) => {
            if name == old_var && !old_var.is_empty() {
                new_var.to_string()
            } else {
                name.clone()
            }
        }
        Expr::Const(v) => format_f32_cuda(*v),
        Expr::IntConst(v) => format!("{v}"),
        Expr::BinOp { op, lhs, rhs } => {
            let l = emit_cuda_expr_replace(lhs, old_var, new_var);
            let r = emit_cuda_expr_replace(rhs, old_var, new_var);
            format!("({l} {op} {r})")
        }
        Expr::UnaryOp { op, operand } => {
            let inner = emit_cuda_expr_replace(operand, old_var, new_var);
            match op {
                UnaryOpKind::Neg => format!("(-{inner})"),
                UnaryOpKind::Exp => format!("expf({inner})"),
                UnaryOpKind::Log => format!("logf({inner})"),
                UnaryOpKind::Sqrt => format!("sqrtf({inner})"),
                UnaryOpKind::Abs => format!("fabsf({inner})"),
                UnaryOpKind::Tanh => format!("tanhf({inner})"),
                UnaryOpKind::Sigmoid => {
                    format!("(1.0f / (1.0f + expf(-{inner})))")
                }
                UnaryOpKind::Relu => {
                    format!("fmaxf({inner}, 0.0f)")
                }
                UnaryOpKind::Gelu => {
                    format!(
                        "({inner} * 0.5f * (1.0f + tanhf(0.7978845608f * ({inner} + 0.044715f * {inner} * {inner} * {inner}))))"
                    )
                }
                UnaryOpKind::Silu => {
                    format!("({inner} / (1.0f + expf(-{inner})))")
                }
            }
        }
        Expr::FnCall { name, args } => {
            let args_s: Vec<String> = args
                .iter()
                .map(|a| emit_cuda_expr_replace(a, old_var, new_var))
                .collect();
            match name.as_str() {
                "powf" => format!("powf({})", args_s.join(", ")),
                _ => format!("{}({})", name, args_s.join(", ")),
            }
        }
        Expr::Index { buffer, index } => {
            let idx = emit_cuda_expr_replace(index, old_var, new_var);
            let buf = cuda_buffer_name(buffer);
            format!("{buf}[{idx}]")
        }
        Expr::Cast {
            target_type,
            operand,
        } => {
            let inner = emit_cuda_expr_replace(operand, old_var, new_var);
            format!("(({target_type}){inner})")
        }
    }
}

/// Format an f64 value as a CUDA float literal.
//
// Exact float comparison is intentional: only bit-identical `0.0` / `1.0`
// emit the canonical short literal.
#[allow(clippy::float_cmp)]
fn format_f32_cuda(v: f64) -> String {
    if v == 0.0 {
        "0.0f".into()
    } else if v == 1.0 {
        "1.0f".into()
    } else if v.is_infinite() && v > 0.0 {
        "INFINITY".into()
    } else if v.is_infinite() {
        "(-INFINITY)".into()
    } else if v.is_nan() {
        "NAN".into()
    } else {
        format!("{v}f")
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
    /// `ctaid.x * ntid.x + tid.x` indexing.  Inner loops become serial
    /// per-thread computation.
    ///
    /// The kernel operates on `f32` values and uses approximate transcendental
    /// instructions (`ex2.approx.f32`, `lg2.approx.f32`, etc.) for performance.
    ///
    /// # Arguments
    ///
    /// * `loops` - The loop IR to convert.
    /// * `fn_name` - The kernel entry point name.
    /// * `block_size` - The intended thread block size (used in documentation
    ///   comments; actual block size is set at launch time).
    /// * `num_inputs` - Number of input buffers.
    pub fn generate_ptx_source(
        loops: &[LoopIR],
        fn_name: &str,
        block_size: usize,
        num_inputs: usize,
    ) -> String {
        let mut out = String::new();

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
        out.push_str("    .reg .f32 %val;\n");
        out.push_str("    .reg .pred %p;\n");

        if needs.extra_f32_regs > 0 {
            for r in 0..needs.extra_f32_regs {
                out.push_str(&format!("    .reg .f32 %t{r};\n"));
            }
        }
        if needs.needs_loop_regs {
            out.push_str("    .reg .u32 %loop_i, %loop_end;\n");
            out.push_str("    .reg .u64 %loop_off;\n");
            out.push_str("    .reg .f32 %acc;\n");
        }
        if needs.needs_zero {
            out.push_str("    .reg .f32 %zero;\n");
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

        // Compute byte offset
        out.push_str("    cvt.u64.u32 %off, %tid;\n");
        out.push_str("    shl.b64 %off, %off, 2;\n\n");

        // Add offset to base pointers
        for i in 0..num_inputs {
            out.push_str(&format!("    add.u64 %in{i}, %in{i}, %off;\n"));
        }
        out.push_str("    add.u64 %out, %out, %off;\n\n");

        // Load input value(s)
        if num_inputs >= 1 {
            out.push_str("    ld.global.f32 %val, [%in0];\n");
        }

        // Zero register if needed
        if needs.needs_zero {
            out.push_str("    mov.f32 %zero, 0f00000000;\n");
        }

        out.push('\n');

        // Block size hint comment
        out.push_str(&format!("    // recommended block size: {block_size}\n\n"));

        // Emit the kernel body
        emit_ptx_body(&mut out, loops);

        // Store result
        out.push_str("\n    st.global.f32 [%out], %val;\n\n");

        out.push_str("DONE:\n");
        out.push_str("    ret;\n");
        out.push_str("}\n");

        out
    }
}

/// Analysis result for PTX register and instruction needs.
struct PtxNeeds {
    extra_f32_regs: usize,
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
        extra_f32_regs: extra,
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
fn emit_ptx_body(out: &mut String, stmts: &[LoopIR]) {
    for stmt in stmts {
        match stmt {
            LoopIR::Loop { body, .. } => {
                // Outermost loop: thread-mapped, process the body directly
                emit_ptx_body(out, body);
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
                                            "    ld.global.f32 %val, [%{buffer}];\n"
                                        ));
                                    }
                                }
                            }
                            // else: primary input already loaded
                        }
                        _ => {
                            emit_ptx_expr_to_reg(out, value, "%val");
                        }
                    }
                } else if var == "acc" {
                    // Accumulator initialization
                    match value {
                        Expr::Const(v) => {
                            out.push_str(&format!(
                                "    mov.f32 %acc, 0f{:08X};\n",
                                (*v as f32).to_bits()
                            ));
                        }
                        _ => {
                            emit_ptx_expr_to_reg(out, value, "%acc");
                        }
                    }
                }
            }
            LoopIR::Assign { var, value } => {
                if var == "val" {
                    emit_ptx_op(out, value);
                } else if var == "acc" {
                    emit_ptx_expr_to_reg(out, value, "%acc");
                }
            }
            LoopIR::Accumulate { var, value } if var == "acc" => {
                // Load the value into a temp, then add to acc
                emit_ptx_expr_to_reg(out, value, "%t0");
                out.push_str("    add.f32 %acc, %acc, %t0;\n");
            }
            LoopIR::Store { value, .. } => {
                // Store already handled by the caller (st.global.f32)
                // But if the value is not %val, we need to move it there
                match value {
                    Expr::Var(v) if v == "acc" => {
                        out.push_str("    mov.f32 %val, %acc;\n");
                    }
                    Expr::Var(v) if v == "val" => {
                        // Already in %val
                    }
                    _ => {
                        emit_ptx_expr_to_reg(out, value, "%val");
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
fn emit_ptx_expr_to_reg(out: &mut String, expr: &Expr, dest: &str) {
    match expr {
        Expr::Const(v) => {
            out.push_str(&format!(
                "    mov.f32 {dest}, 0f{:08X};\n",
                (*v as f32).to_bits()
            ));
        }
        Expr::Var(name) => {
            let reg = ptx_var_to_reg(name);
            if reg != dest {
                out.push_str(&format!("    mov.f32 {dest}, {reg};\n"));
            }
        }
        Expr::Index { buffer, .. } => {
            out.push_str(&format!("    ld.global.f32 {dest}, [%{buffer}];\n"));
        }
        Expr::BinOp { op, lhs, rhs } => {
            emit_ptx_expr_to_reg(out, lhs, dest);
            emit_ptx_expr_to_reg(out, rhs, "%t0");
            let ptx_op = match op {
                BinOpKind::Add => "add.f32",
                BinOpKind::Sub => "sub.f32",
                BinOpKind::Mul => "mul.f32",
                BinOpKind::Div => "div.approx.f32",
                BinOpKind::Mod => {
                    // PTX doesn't have a direct fmod; approximate with
                    // a - floor(a/b) * b
                    out.push_str(&format!("    div.approx.f32 %t1, {dest}, %t0;\n"));
                    out.push_str("    cvt.rzi.f32.f32 %t1, %t1;\n");
                    out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                    out.push_str(&format!("    sub.f32 {dest}, {dest}, %t1;\n"));
                    return;
                }
            };
            out.push_str(&format!("    {ptx_op} {dest}, {dest}, %t0;\n"));
        }
        Expr::UnaryOp { op, operand } => {
            emit_ptx_expr_to_reg(out, operand, dest);
            emit_ptx_unary_op(out, *op, dest);
        }
        Expr::FnCall { name, args } => {
            if name == "powf" && args.len() == 2 {
                // x^p = 2^(p * log2(x))
                emit_ptx_expr_to_reg(out, &args[0], dest);
                emit_ptx_expr_to_reg(out, &args[1], "%t0");
                out.push_str(&format!("    lg2.approx.f32 %t1, {dest};\n"));
                out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                out.push_str(&format!("    ex2.approx.f32 {dest}, %t1;\n"));
            } else {
                // Generic: just put the first arg in dest
                if let Some(arg) = args.first() {
                    emit_ptx_expr_to_reg(out, arg, dest);
                }
            }
        }
        _ => {}
    }
}

/// Emit PTX for a unary operation on a register.
fn emit_ptx_unary_op(out: &mut String, op: UnaryOpKind, reg: &str) {
    match op {
        UnaryOpKind::Neg => {
            out.push_str(&format!("    neg.f32 {reg}, {reg};\n"));
        }
        UnaryOpKind::Abs => {
            out.push_str(&format!("    abs.f32 {reg}, {reg};\n"));
        }
        UnaryOpKind::Sqrt => {
            out.push_str(&format!("    sqrt.approx.f32 {reg}, {reg};\n"));
        }
        UnaryOpKind::Exp => {
            // exp(x) = 2^(x * log2(e))
            out.push_str(&format!("    mul.f32 {reg}, {reg}, 0f3FB8AA3B;\n")); // log2(e)
            out.push_str(&format!("    ex2.approx.f32 {reg}, {reg};\n"));
        }
        UnaryOpKind::Log => {
            // log(x) = log2(x) / log2(e) = log2(x) * ln(2)
            out.push_str(&format!("    lg2.approx.f32 {reg}, {reg};\n"));
            out.push_str(&format!("    mul.f32 {reg}, {reg}, 0f3F317218;\n")); // ln(2)
        }
        UnaryOpKind::Relu => {
            out.push_str(&format!("    max.f32 {reg}, {reg}, %zero;\n"));
        }
        UnaryOpKind::Sigmoid => {
            // sigmoid(x) = 1 / (1 + exp(-x))
            out.push_str(&format!("    neg.f32 %t0, {reg};\n"));
            out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n"); // * log2(e)
            out.push_str("    ex2.approx.f32 %t0, %t0;\n");
            out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n"); // + 1.0
            out.push_str(&format!("    rcp.approx.f32 {reg}, %t0;\n"));
        }
        UnaryOpKind::Tanh => {
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
fn emit_ptx_op(out: &mut String, expr: &Expr) {
    match expr {
        Expr::UnaryOp { op, .. } => {
            emit_ptx_unary_op(out, *op, "%val");
        }
        Expr::BinOp { op, lhs, rhs } => {
            // Binary op where lhs is typically %val
            let _ = lhs; // lhs is already in %val
            emit_ptx_expr_to_reg(out, rhs, "%t0");
            let ptx_op = match op {
                BinOpKind::Add => "add.f32",
                BinOpKind::Sub => "sub.f32",
                BinOpKind::Mul => "mul.f32",
                BinOpKind::Div => "div.approx.f32",
                BinOpKind::Mod => {
                    out.push_str("    div.approx.f32 %t1, %val, %t0;\n");
                    out.push_str("    cvt.rzi.f32.f32 %t1, %t1;\n");
                    out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                    out.push_str("    sub.f32 %val, %val, %t1;\n");
                    return;
                }
            };
            out.push_str(&format!("    {ptx_op} %val, %val, %t0;\n"));
        }
        Expr::FnCall { name, args } => {
            if name == "powf" && args.len() == 2 {
                // x^p = 2^(p * log2(x))
                emit_ptx_expr_to_reg(out, &args[1], "%t0");
                out.push_str("    lg2.approx.f32 %t1, %val;\n");
                out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                out.push_str("    ex2.approx.f32 %val, %t1;\n");
            }
        }
        _ => {
            emit_ptx_expr_to_reg(out, expr, "%val");
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
    // CUDA codegen tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cuda_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_neg", 1);

        assert!(src.contains("__global__ void kernel_neg"));
        assert!(src.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(src.contains("if (tid >= n) return"));
        assert!(src.contains("const float* __restrict__ in0"));
        assert!(src.contains("float* __restrict__ output"));
    }

    #[test]
    fn test_cuda_binary_add() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_add", 2);

        assert!(src.contains("const float* __restrict__ in0"));
        assert!(src.contains("const float* __restrict__ in1"));
        assert!(src.contains('+'));
    }

    #[test]
    fn test_cuda_sigmoid() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_sigmoid", 1);

        assert!(src.contains("expf("));
        assert!(src.contains("1.0f"));
    }

    #[test]
    fn test_cuda_relu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Relu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_relu", 1);

        assert!(src.contains("fmaxf("));
    }

    #[test]
    fn test_cuda_reduction_sum() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_sum", 1);

        assert!(src.contains("__shared__"));
        assert!(src.contains("__syncthreads"));
        assert!(src.contains("atomicAdd"));
    }

    #[test]
    fn test_cuda_reduction_mean() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Mean], &["in0"], "out", 100);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_mean", 1);

        assert!(src.contains("__shared__"));
        assert!(src.contains("(float)n"));
    }

    #[test]
    fn test_cuda_fused_chain() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Relu];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_fused", 1);

        assert!(src.contains("__global__"));
        assert!(src.contains("tid"));
    }

    #[test]
    fn test_cuda_gelu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Gelu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_gelu", 1);

        assert!(src.contains("tanhf("));
        assert!(src.contains("0.044715"));
    }

    #[test]
    fn test_cuda_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_cuda_source(&loops, "kernel_silu", 1);

        assert!(src.contains("expf("));
    }

    // -----------------------------------------------------------------------
    // PTX codegen tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ptx_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_neg", 256, 1);

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
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_relu", 256, 1);

        assert!(src.contains("max.f32 %val, %val, %zero"));
    }

    #[test]
    fn test_ptx_sigmoid() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_sigmoid", 256, 1);

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_sqrt() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sqrt], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_sqrt", 256, 1);

        assert!(src.contains("sqrt.approx.f32"));
    }

    #[test]
    fn test_ptx_exp() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Exp], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_exp", 256, 1);

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("3FB8AA3B")); // log2(e) float bits
    }

    #[test]
    fn test_ptx_log() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Log], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_log", 256, 1);

        assert!(src.contains("lg2.approx.f32"));
        assert!(src.contains("3F317218")); // ln(2) float bits
    }

    #[test]
    fn test_ptx_tanh() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Tanh], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_tanh", 256, 1);

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("rcp.approx.f32"));
        assert!(src.contains("sub.f32")); // -1 step
    }

    #[test]
    fn test_ptx_fused_chain() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Relu, IrOpKind::Sigmoid];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_fused", 256, 1);

        assert!(src.contains("neg.f32"));
        assert!(src.contains("max.f32"));
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_block_size_comment() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel", 512, 1);
        assert!(src.contains("recommended block size: 512"));
    }

    #[test]
    fn test_ptx_multiple_inputs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_add", 256, 2);

        assert!(src.contains("in0_ptr"));
        assert!(src.contains("in1_ptr"));
        assert!(src.contains("%in0"));
        assert!(src.contains("%in1"));
    }

    #[test]
    fn test_ptx_gelu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Gelu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_gelu", 256, 1);

        // Should use GELU approximation: x * sigmoid(1.702 * x)
        assert!(src.contains("3FD9F16C")); // 1.702 float bits
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_silu", 256, 1);

        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_abs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Abs], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_abs", 256, 1);

        assert!(src.contains("abs.f32"));
    }

    #[test]
    fn test_format_f32_cuda_values() {
        assert_eq!(format_f32_cuda(0.0), "0.0f");
        assert_eq!(format_f32_cuda(1.0), "1.0f");
        assert_eq!(format_f32_cuda(f64::INFINITY), "INFINITY");
        assert_eq!(format_f32_cuda(f64::NEG_INFINITY), "(-INFINITY)");
        assert_eq!(format_f32_cuda(f64::NAN), "NAN");
    }
}
