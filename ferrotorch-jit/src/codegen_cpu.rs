//! CPU code generators: emit Rust or C source code from [`LoopIR`].
//!
//! Two code generators are provided:
//!
//! - [`generate_rust_source`] — emits Rust source code with `#[inline(always)]`
//!   hints and SIMD-friendly sequential access patterns.
//! - [`generate_c_source`] — emits C source code with `#pragma omp simd` and
//!   `#pragma omp parallel for` annotations, `restrict` pointers, and
//!   `math.h` intrinsics.
//!
//! Both generators produce self-contained functions that take raw pointer
//! arguments and a length parameter.

use crate::codegen_ir::{Expr, LoopIR, UnaryOpKind};

/// Size threshold above which outer loops get parallel annotations.
const PARALLEL_THRESHOLD: usize = 1024;

// ===========================================================================
// Rust code generation
// ===========================================================================

/// A CPU code generator targeting Rust source output.
pub struct CpuCodegen;

impl CpuCodegen {
    /// Generate Rust source code from a `LoopIR` program.
    ///
    /// The emitted function has the signature:
    /// ```text
    /// #[inline(always)]
    /// pub unsafe fn <fn_name>(inputs: &[&[f64]], output: &mut [f64])
    /// ```
    ///
    /// Buffer references in the `LoopIR` are mapped to `inputs[0]`, `inputs[1]`,
    /// etc., and the output buffer is `output`.
    pub fn generate_rust_source(loops: &[LoopIR], fn_name: &str) -> String {
        let mut out = String::new();

        out.push_str("#[inline(always)]\n");
        out.push_str(&format!(
            "pub unsafe fn {fn_name}(inputs: &[&[f64]], output: &mut [f64]) {{\n"
        ));

        for stmt in loops {
            emit_rust_stmt(&mut out, stmt, 1);
        }

        out.push_str("}\n");
        out
    }
}

/// Emit a single LoopIR statement as Rust code at the given indentation level.
fn emit_rust_stmt(out: &mut String, stmt: &LoopIR, indent: usize) {
    let pad = "    ".repeat(indent);

    match stmt {
        LoopIR::Loop {
            var,
            start,
            end,
            body,
        } => {
            let start_s = emit_rust_expr(start);
            let end_s = emit_rust_expr(end);

            // Add SIMD-friendly comment for inner loops
            if body.iter().all(|s| !matches!(s, LoopIR::Loop { .. })) {
                out.push_str(&format!(
                    "{pad}// SIMD: sequential access, no dependencies\n"
                ));
            }

            // Add parallelism hint for large outer loops
            if let Expr::IntConst(n) = end {
                if *n as usize >= PARALLEL_THRESHOLD
                    && body.iter().any(|s| matches!(s, LoopIR::Loop { .. }))
                {
                    out.push_str(&format!(
                        "{pad}// NOTE: candidate for rayon par_iter (n={n})\n"
                    ));
                }
            }

            out.push_str(&format!(
                "{pad}for {var} in ({start_s} as usize)..({end_s} as usize) {{\n"
            ));
            for s in body {
                emit_rust_stmt(out, s, indent + 1);
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Store {
            buffer,
            index,
            value,
        } => {
            let idx = emit_rust_expr(index);
            let val = emit_rust_expr(value);
            let buf = rust_buffer_access(buffer);
            out.push_str(&format!("{pad}{buf}[{idx} as usize] = {val};\n"));
        }

        LoopIR::Let { var, value } => {
            let val = emit_rust_expr(value);
            out.push_str(&format!("{pad}let mut {var} = {val};\n"));
        }

        LoopIR::Assign { var, value } => {
            let val = emit_rust_expr(value);
            out.push_str(&format!("{pad}{var} = {val};\n"));
        }

        LoopIR::Accumulate { var, value } => {
            let val = emit_rust_expr(value);
            out.push_str(&format!("{pad}{var} += {val};\n"));
        }

        LoopIR::If {
            condition,
            then_body,
            else_body,
        } => {
            let cond = emit_rust_expr(condition);
            out.push_str(&format!("{pad}if {cond} {{\n"));
            for s in then_body {
                emit_rust_stmt(out, s, indent + 1);
            }
            if !else_body.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_rust_stmt(out, s, indent + 1);
                }
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Comment(text) => {
            out.push_str(&format!("{pad}// {text}\n"));
        }
    }
}

/// Emit a Rust expression from an `Expr`.
fn emit_rust_expr(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => name.clone(),
        Expr::Const(v) => format_f64_rust(*v),
        Expr::IntConst(v) => format!("{v}"),
        Expr::BinOp { op, lhs, rhs } => {
            let l = emit_rust_expr(lhs);
            let r = emit_rust_expr(rhs);
            format!("({l} {op} {r})")
        }
        Expr::UnaryOp { op, operand } => {
            let inner = emit_rust_expr(operand);
            match op {
                UnaryOpKind::Neg => format!("(-{inner})"),
                UnaryOpKind::Exp => format!("{inner}.exp()"),
                UnaryOpKind::Log => format!("{inner}.ln()"),
                UnaryOpKind::Sqrt => format!("{inner}.sqrt()"),
                UnaryOpKind::Abs => format!("{inner}.abs()"),
                UnaryOpKind::Tanh => format!("{inner}.tanh()"),
                UnaryOpKind::Sigmoid => {
                    format!("(1.0_f64 / (1.0_f64 + (-{inner}).exp()))")
                }
                UnaryOpKind::Relu => {
                    format!("(if {inner} > 0.0_f64 {{ {inner} }} else {{ 0.0_f64 }})")
                }
                UnaryOpKind::Gelu => {
                    // GELU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    format!(
                        "{{ let _gx = {inner}; _gx * 0.5_f64 * (1.0_f64 + (0.7978845608_f64 * (_gx + 0.044715_f64 * _gx * _gx * _gx)).tanh()) }}"
                    )
                }
                UnaryOpKind::Silu => {
                    format!("({inner} / (1.0_f64 + (-{inner}).exp()))")
                }
            }
        }
        Expr::FnCall { name, args } => {
            let args_s: Vec<String> = args.iter().map(emit_rust_expr).collect();
            match name.as_str() {
                "powf" if args_s.len() == 2 => {
                    format!("{}.powf({})", args_s[0], args_s[1])
                }
                _ => {
                    format!("{}({})", name, args_s.join(", "))
                }
            }
        }
        Expr::Index { buffer, index } => {
            let idx = emit_rust_expr(index);
            let buf = rust_buffer_access(buffer);
            format!("{buf}[{idx} as usize]")
        }
        Expr::Cast {
            target_type,
            operand,
        } => {
            let inner = emit_rust_expr(operand);
            format!("({inner} as {target_type})")
        }
    }
}

/// Map a buffer name to a Rust expression.
///
/// Buffers named `in0`, `in1`, ... map to `inputs[0]`, `inputs[1]`, etc.
/// A buffer named `out` or `output` maps to `output`.
/// Anything else is used verbatim.
fn rust_buffer_access(name: &str) -> String {
    if let Some(suffix) = name.strip_prefix("in") {
        if let Ok(idx) = suffix.parse::<usize>() {
            return format!("inputs[{idx}]");
        }
    }
    if name == "out" || name == "output" {
        return "output".into();
    }
    name.into()
}

/// Format an f64 as a Rust literal.
fn format_f64_rust(v: f64) -> String {
    if v == f64::INFINITY {
        "f64::INFINITY".into()
    } else if v == f64::NEG_INFINITY {
        "f64::NEG_INFINITY".into()
    } else if v.is_nan() {
        "f64::NAN".into()
    } else if v == 0.0 {
        "0.0_f64".into()
    } else if v == 1.0 {
        "1.0_f64".into()
    } else {
        format!("{v}_f64")
    }
}

// ===========================================================================
// C code generation
// ===========================================================================

impl CpuCodegen {
    /// Generate C source code from a `LoopIR` program.
    ///
    /// The emitted function has the signature:
    /// ```text
    /// void <fn_name>(
    ///     const double* restrict in0,
    ///     const double* restrict in1,
    ///     ...,
    ///     double* restrict output,
    ///     int n
    /// )
    /// ```
    ///
    /// The function uses `restrict` pointers to enable alias analysis,
    /// `#pragma omp simd` on inner loops, and `#pragma omp parallel for`
    /// on outer loops above a size threshold.
    pub fn generate_c_source(loops: &[LoopIR], fn_name: &str, num_inputs: usize) -> String {
        let mut out = String::new();

        out.push_str("#include <math.h>\n");
        out.push_str("#include <stddef.h>\n\n");

        // Function signature
        out.push_str(&format!("void {fn_name}(\n"));
        for i in 0..num_inputs {
            out.push_str(&format!("    const double* restrict in{i},\n"));
        }
        out.push_str("    double* restrict output,\n");
        out.push_str("    int n\n");
        out.push_str(") {\n");

        for stmt in loops {
            emit_c_stmt(&mut out, stmt, 1, true);
        }

        out.push_str("}\n");
        out
    }
}

/// Emit a single LoopIR statement as C code.
///
/// `is_outer` indicates whether this is a top-level (outer) loop, used
/// to decide whether to emit `#pragma omp parallel for`.
fn emit_c_stmt(out: &mut String, stmt: &LoopIR, indent: usize, is_outer: bool) {
    let pad = "    ".repeat(indent);

    match stmt {
        LoopIR::Loop {
            var,
            start,
            end,
            body,
        } => {
            let start_s = emit_c_expr(start);
            let end_s = emit_c_expr(end);

            let is_inner = body.iter().all(|s| !matches!(s, LoopIR::Loop { .. }));

            if is_outer && !is_inner {
                // Outer loop with inner loops: candidate for thread parallelism
                if let Expr::IntConst(n) = end {
                    if *n as usize >= PARALLEL_THRESHOLD {
                        out.push_str(&format!("{pad}#pragma omp parallel for\n"));
                    }
                }
            }

            if is_inner {
                // Inner loop: candidate for SIMD
                out.push_str(&format!("{pad}#pragma omp simd\n"));
            }

            out.push_str(&format!(
                "{pad}for (int {var} = {start_s}; {var} < {end_s}; {var}++) {{\n"
            ));
            for s in body {
                emit_c_stmt(out, s, indent + 1, false);
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Store {
            buffer,
            index,
            value,
        } => {
            let idx = emit_c_expr(index);
            let val = emit_c_expr(value);
            let buf = c_buffer_name(buffer);
            out.push_str(&format!("{pad}{buf}[{idx}] = {val};\n"));
        }

        LoopIR::Let { var, value } => {
            let val = emit_c_expr(value);
            out.push_str(&format!("{pad}double {var} = {val};\n"));
        }

        LoopIR::Assign { var, value } => {
            let val = emit_c_expr(value);
            out.push_str(&format!("{pad}{var} = {val};\n"));
        }

        LoopIR::Accumulate { var, value } => {
            let val = emit_c_expr(value);
            out.push_str(&format!("{pad}{var} += {val};\n"));
        }

        LoopIR::If {
            condition,
            then_body,
            else_body,
        } => {
            let cond = emit_c_expr(condition);
            out.push_str(&format!("{pad}if ({cond}) {{\n"));
            for s in then_body {
                emit_c_stmt(out, s, indent + 1, false);
            }
            if !else_body.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_c_stmt(out, s, indent + 1, false);
                }
            }
            out.push_str(&format!("{pad}}}\n"));
        }

        LoopIR::Comment(text) => {
            out.push_str(&format!("{pad}/* {text} */\n"));
        }
    }
}

/// Emit a C expression from an `Expr`.
fn emit_c_expr(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => name.clone(),
        Expr::Const(v) => format_f64_c(*v),
        Expr::IntConst(v) => format!("{v}"),
        Expr::BinOp { op, lhs, rhs } => {
            let l = emit_c_expr(lhs);
            let r = emit_c_expr(rhs);
            format!("({l} {op} {r})")
        }
        Expr::UnaryOp { op, operand } => {
            let inner = emit_c_expr(operand);
            match op {
                UnaryOpKind::Neg => format!("(-{inner})"),
                UnaryOpKind::Exp => format!("exp({inner})"),
                UnaryOpKind::Log => format!("log({inner})"),
                UnaryOpKind::Sqrt => format!("sqrt({inner})"),
                UnaryOpKind::Abs => format!("fabs({inner})"),
                UnaryOpKind::Tanh => format!("tanh({inner})"),
                UnaryOpKind::Sigmoid => {
                    format!("(1.0 / (1.0 + exp(-{inner})))")
                }
                UnaryOpKind::Relu => {
                    format!("(({inner}) > 0.0 ? ({inner}) : 0.0)")
                }
                UnaryOpKind::Gelu => {
                    // GELU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    format!(
                        "({inner} * 0.5 * (1.0 + tanh(0.7978845608 * ({inner} + 0.044715 * {inner} * {inner} * {inner}))))"
                    )
                }
                UnaryOpKind::Silu => {
                    format!("({inner} / (1.0 + exp(-{inner})))")
                }
            }
        }
        Expr::FnCall { name, args } => {
            let args_s: Vec<String> = args.iter().map(emit_c_expr).collect();
            match name.as_str() {
                "powf" => format!("pow({})", args_s.join(", ")),
                _ => format!("{}({})", name, args_s.join(", ")),
            }
        }
        Expr::Index { buffer, index } => {
            let idx = emit_c_expr(index);
            let buf = c_buffer_name(buffer);
            format!("{buf}[{idx}]")
        }
        Expr::Cast {
            target_type,
            operand,
        } => {
            let inner = emit_c_expr(operand);
            format!("(({target_type}){inner})")
        }
    }
}

/// Map a buffer name to a C identifier.
fn c_buffer_name(name: &str) -> String {
    if name == "out" || name == "output" {
        return "output".into();
    }
    name.into()
}

/// Format an f64 as a C literal.
fn format_f64_c(v: f64) -> String {
    if v == f64::INFINITY {
        "INFINITY".into()
    } else if v == f64::NEG_INFINITY {
        "(-INFINITY)".into()
    } else if v.is_nan() {
        "NAN".into()
    } else if v == 0.0 {
        "0.0".into()
    } else if v == 1.0 {
        "1.0".into()
    } else {
        // Ensure there is a decimal point
        let s = format!("{v}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_ir::{self, BinOpKind, Expr, LoopIR};
    use crate::graph::IrOpKind;

    // -----------------------------------------------------------------------
    // Rust codegen tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_neg");

        assert!(src.contains("#[inline(always)]"));
        assert!(src.contains("pub unsafe fn kernel_neg"));
        assert!(src.contains("inputs[0]"));
        assert!(src.contains("output["));
        assert!(src.contains("for i in"));
    }

    #[test]
    fn test_rust_binary_add() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 8);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_add");

        assert!(src.contains("inputs[0]"));
        assert!(src.contains("inputs[1]"));
        assert!(src.contains("output["));
        assert!(src.contains("+"));
    }

    #[test]
    fn test_rust_fused_chain() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Relu];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_fused");

        // Should have both neg and relu in the same loop
        assert!(src.contains("let mut val"));
        assert!(src.contains("val ="));
        assert!(src.contains("output["));
    }

    #[test]
    fn test_rust_sum_reduction() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 10);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_sum");

        assert!(src.contains("let mut acc"));
        assert!(src.contains("acc +="));
        assert!(src.contains("output[0"));
    }

    #[test]
    fn test_rust_matmul() {
        let loops = codegen_ir::lower_matmul("in0", "in1", "out", 2, 3, 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_matmul");

        assert!(src.contains("for i in"));
        assert!(src.contains("for j in"));
        assert!(src.contains("for p in"));
        assert!(src.contains("let mut acc"));
        assert!(src.contains("acc +="));
    }

    #[test]
    fn test_rust_sigmoid() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_sigmoid");

        assert!(src.contains("1.0_f64"));
        assert!(src.contains(".exp()"));
    }

    #[test]
    fn test_rust_pow() {
        let loops =
            codegen_ir::lower_to_loops(&[IrOpKind::Pow { exponent: 2.0 }], &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_pow");

        assert!(src.contains(".powf("));
    }

    #[test]
    fn test_rust_comment() {
        let loops = vec![LoopIR::Comment("test comment".into())];
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_test");
        assert!(src.contains("// test comment"));
    }

    #[test]
    fn test_rust_if_statement() {
        let loops = vec![LoopIR::If {
            condition: Expr::bin(BinOpKind::Mod, Expr::var("i"), Expr::int(2)),
            then_body: vec![LoopIR::Store {
                buffer: "out".into(),
                index: Expr::var("i"),
                value: Expr::constant(1.0),
            }],
            else_body: vec![LoopIR::Store {
                buffer: "out".into(),
                index: Expr::var("i"),
                value: Expr::constant(0.0),
            }],
        }];
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_if");
        assert!(src.contains("if "));
        assert!(src.contains("} else {"));
    }

    // -----------------------------------------------------------------------
    // C codegen tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_c_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_neg", 1);

        assert!(src.contains("#include <math.h>"));
        assert!(src.contains("void kernel_neg("));
        assert!(src.contains("const double* restrict in0"));
        assert!(src.contains("double* restrict output"));
        assert!(src.contains("for (int i ="));
        assert!(src.contains("#pragma omp simd"));
    }

    #[test]
    fn test_c_binary_add() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 8);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_add", 2);

        assert!(src.contains("const double* restrict in0"));
        assert!(src.contains("const double* restrict in1"));
        assert!(src.contains("+"));
    }

    #[test]
    fn test_c_matmul() {
        let loops = codegen_ir::lower_matmul("in0", "in1", "out", 2, 3, 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_matmul", 2);

        assert!(src.contains("for (int i ="));
        assert!(src.contains("for (int j ="));
        assert!(src.contains("for (int p ="));
        assert!(src.contains("double acc"));
        assert!(src.contains("acc +="));
    }

    #[test]
    fn test_c_sigmoid() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_sigmoid", 1);

        assert!(src.contains("exp("));
    }

    #[test]
    fn test_c_abs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Abs], &["in0"], "out", 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_abs", 1);
        assert!(src.contains("fabs("));
    }

    #[test]
    fn test_c_sum_reduction() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 10);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_sum", 1);

        assert!(src.contains("double acc = 0.0;"));
        assert!(src.contains("acc +="));
        assert!(src.contains("output[0]"));
    }

    #[test]
    fn test_c_gelu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Gelu], &["in0"], "out", 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_gelu", 1);

        assert!(src.contains("tanh("));
        assert!(src.contains("0.044715"));
    }

    #[test]
    fn test_c_pow() {
        let loops =
            codegen_ir::lower_to_loops(&[IrOpKind::Pow { exponent: 3.0 }], &["in0"], "out", 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_pow", 1);

        assert!(src.contains("pow("));
    }

    #[test]
    fn test_c_large_loop_parallel() {
        // A matmul with M >= PARALLEL_THRESHOLD should get #pragma omp parallel for
        let loops = codegen_ir::lower_matmul("in0", "in1", "out", 2048, 64, 64);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_big_mm", 2);

        assert!(src.contains("#pragma omp parallel for"));
    }

    #[test]
    fn test_c_comment() {
        let loops = vec![LoopIR::Comment("test comment".into())];
        let src = CpuCodegen::generate_c_source(&loops, "kernel_test", 0);
        assert!(src.contains("/* test comment */"));
    }

    // -----------------------------------------------------------------------
    // Expression rendering edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_special_float_values() {
        assert_eq!(format_f64_rust(0.0), "0.0_f64");
        assert_eq!(format_f64_rust(1.0), "1.0_f64");
        assert_eq!(format_f64_rust(f64::INFINITY), "f64::INFINITY");
        assert_eq!(format_f64_rust(f64::NEG_INFINITY), "f64::NEG_INFINITY");
        assert_eq!(format_f64_rust(f64::NAN), "f64::NAN");
    }

    #[test]
    fn test_c_special_float_values() {
        assert_eq!(format_f64_c(0.0), "0.0");
        assert_eq!(format_f64_c(1.0), "1.0");
        assert_eq!(format_f64_c(f64::INFINITY), "INFINITY");
        assert_eq!(format_f64_c(f64::NEG_INFINITY), "(-INFINITY)");
        assert_eq!(format_f64_c(f64::NAN), "NAN");
    }

    #[test]
    fn test_rust_buffer_mapping() {
        assert_eq!(rust_buffer_access("in0"), "inputs[0]");
        assert_eq!(rust_buffer_access("in1"), "inputs[1]");
        assert_eq!(rust_buffer_access("in42"), "inputs[42]");
        assert_eq!(rust_buffer_access("out"), "output");
        assert_eq!(rust_buffer_access("output"), "output");
        assert_eq!(rust_buffer_access("custom"), "custom");
    }

    #[test]
    fn test_c_buffer_mapping() {
        assert_eq!(c_buffer_name("in0"), "in0");
        assert_eq!(c_buffer_name("out"), "output");
        assert_eq!(c_buffer_name("output"), "output");
    }

    #[test]
    fn test_rust_cast_expr() {
        let cast = Expr::Cast {
            target_type: "f64".into(),
            operand: Box::new(Expr::var("i")),
        };
        let result = emit_rust_expr(&cast);
        assert_eq!(result, "(i as f64)");
    }

    #[test]
    fn test_c_cast_expr() {
        let cast = Expr::Cast {
            target_type: "double".into(),
            operand: Box::new(Expr::var("i")),
        };
        let result = emit_c_expr(&cast);
        assert_eq!(result, "((double)i)");
    }

    #[test]
    fn test_rust_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_silu");
        assert!(src.contains(".exp()"));
    }

    #[test]
    fn test_c_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_silu", 1);
        assert!(src.contains("exp("));
    }

    #[test]
    fn test_rust_log() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Log], &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_log");
        assert!(src.contains(".ln()"));
    }

    #[test]
    fn test_c_log() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Log], &["in0"], "out", 4);
        let src = CpuCodegen::generate_c_source(&loops, "kernel_log", 1);
        assert!(src.contains("log("));
    }
}
