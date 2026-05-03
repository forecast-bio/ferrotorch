//! CPU code generator: emits Rust source code from [`LoopIR`].
//!
//! [`CpuCodegen::generate_rust_source`] emits Rust functions with
//! `#[inline(always)]` hints and SIMD-friendly sequential access patterns,
//! suitable for compilation by `rustc` (see [`crate::codegen_jit`] for the
//! end-to-end JIT pipeline).
//!
//! Generated functions take `&[&[f64]]` inputs and a `&mut [f64]` output
//! buffer. The trampoline that bridges this signature to the FFI-friendly
//! `extern "C" fn(*const *const f64, *mut f64, i32)` lives in `codegen_jit`.

use crate::codegen_ir::{Expr, LoopIR, UnaryOpKind};

/// Size threshold above which outer loops get parallel annotations.
const PARALLEL_THRESHOLD: usize = 1024;

// ===========================================================================
// Rust code generation
// ===========================================================================

/// A CPU code generator targeting Rust source output.
#[derive(Debug)]
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

/// Emit a single `LoopIR` statement as Rust code at the given indentation level.
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
//
// Exact float comparison is intentional: we only emit the canonical short
// literal when `v` is bit-identically `0.0` / `1.0`. Approximate matches
// must format with the full numeric value.
#[allow(clippy::float_cmp)]
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_ir::{self, BinOpKind, Expr, LoopIR};
    use crate::graph::IrOpKind;

    // -----------------------------------------------------------------------
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
        assert!(src.contains('+'));
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
    fn test_rust_buffer_mapping() {
        assert_eq!(rust_buffer_access("in0"), "inputs[0]");
        assert_eq!(rust_buffer_access("in1"), "inputs[1]");
        assert_eq!(rust_buffer_access("in42"), "inputs[42]");
        assert_eq!(rust_buffer_access("out"), "output");
        assert_eq!(rust_buffer_access("output"), "output");
        assert_eq!(rust_buffer_access("custom"), "custom");
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
    fn test_rust_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_silu");
        assert!(src.contains(".exp()"));
    }

    #[test]
    fn test_rust_log() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Log], &["in0"], "out", 4);
        let src = CpuCodegen::generate_rust_source(&loops, "kernel_log");
        assert!(src.contains(".ln()"));
    }
}
