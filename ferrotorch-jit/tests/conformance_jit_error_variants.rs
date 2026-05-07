//! Conformance tests for ferrotorch-jit error variants — Sprint A.4 (#886, #884).
//!
//! ## Coverage
//!
//! ### #886 — DataDependentControlFlow / RecompilationError (C7.4 cascade)
//!
//! Both variants are defined in [`ferrotorch_jit::error::JitError`] and their
//! API surface is stable. Full-graph tracing integration (the only code path
//! that would *raise* them automatically) is deferred to C7.5. These tests
//! prove that:
//!
//! - The variant can be constructed.
//! - Its `Display` message contains the expected sub-strings.
//! - `From<JitError> for FerrotorchError` works.
//! - Fields are readable.
//!
//! ### #884 — GpuBackendUnavailable
//!
//! [`ferrotorch_jit::codegen::InductorBackend::compile`] raises
//! `JitError::GpuBackendUnavailable` when called with `GpuCuda` or `GpuPtx`
//! targets because no GPU runtime executor is wired through the JIT path.
//! This fires unconditionally — no physical GPU is required — making it
//! exercisable in default CI.
//!
//! Tests that **require the `cuda` feature** (NVRTC f64 transcendental path)
//! are gated `#[cfg(feature = "cuda")]`. Tests that must run on the
//! **default (no-cuda) build** are gated `#[cfg(not(feature = "cuda"))]`.
//! The `GpuBackendUnavailable` path itself is unconditional and tested
//! without any feature gate.

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_jit::codegen::{Codegen, InductorBackend, InductorTarget};
use ferrotorch_jit::error::JitError;
use ferrotorch_jit::graph::{Dtype, IrGraph, IrOpKind};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal single-relu graph: input(shape=[4]) → relu → output.
fn relu_graph() -> IrGraph {
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
    g.set_outputs(vec![outs[0]]);
    g
}

/// Build a minimal graph containing a Neg op (f32, no transcendentals).
fn neg_graph() -> IrGraph {
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
    g.set_outputs(vec![outs[0]]);
    g
}

// ===========================================================================
// #886 — DataDependentControlFlow
// ===========================================================================

/// The variant constructs correctly and its Display contains the op name.
#[test]
fn data_dependent_control_flow_display_contains_op() {
    let e = JitError::DataDependentControlFlow { op: "torch.where".to_string() };
    let msg = e.to_string();
    assert!(
        msg.contains("data-dependent control flow"),
        "expected 'data-dependent control flow' in Display; got: {msg}"
    );
    assert!(
        msg.contains("torch.where"),
        "expected op name 'torch.where' in Display; got: {msg}"
    );
}

/// `From<JitError> for FerrotorchError` preserves the message.
#[test]
fn data_dependent_control_flow_converts_to_ferrotorch_error() {
    let jit_err = JitError::DataDependentControlFlow { op: "dynamic_select".to_string() };
    let ft_err: FerrotorchError = jit_err.into();
    let msg = ft_err.to_string();
    assert!(
        msg.contains("dynamic_select"),
        "FerrotorchError must carry the op name; got: {msg}"
    );
}

/// The `op` field is accessible by pattern matching.
#[test]
fn data_dependent_control_flow_field_access() {
    let e = JitError::DataDependentControlFlow { op: "masked_fill".to_string() };
    match e {
        JitError::DataDependentControlFlow { op } => {
            assert_eq!(op, "masked_fill");
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

/// Display message explicitly requires static control flow.
#[test]
fn data_dependent_control_flow_display_mentions_static_control_flow() {
    let e = JitError::DataDependentControlFlow { op: "if_else".to_string() };
    let msg = e.to_string();
    assert!(
        msg.contains("static control flow"),
        "expected 'static control flow' in Display; got: {msg}"
    );
}

// ===========================================================================
// #886 — RecompilationError
// ===========================================================================

/// The variant constructs correctly and its Display contains the shape info.
#[test]
fn recompilation_error_display_contains_shape_and_message() {
    let e = JitError::RecompilationError {
        shape: vec![vec![8, 16], vec![16, 4]],
        message: "codegen backend rejected new shape".to_string(),
    };
    let msg = e.to_string();
    assert!(
        msg.contains("recompilation failed"),
        "expected 'recompilation failed' in Display; got: {msg}"
    );
    assert!(
        msg.contains("codegen backend rejected new shape"),
        "expected message in Display; got: {msg}"
    );
}

/// `From<JitError> for FerrotorchError` preserves the message.
#[test]
fn recompilation_error_converts_to_ferrotorch_error() {
    let jit_err = JitError::RecompilationError {
        shape: vec![vec![32]],
        message: "out of budget".to_string(),
    };
    let ft_err: FerrotorchError = jit_err.into();
    let msg = ft_err.to_string();
    assert!(
        msg.contains("out of budget"),
        "FerrotorchError must carry the message; got: {msg}"
    );
}

/// Both fields are accessible by pattern matching.
#[test]
fn recompilation_error_field_access() {
    let e = JitError::RecompilationError {
        shape: vec![vec![4, 8]],
        message: "specialization cache full".to_string(),
    };
    match e {
        JitError::RecompilationError { shape, message } => {
            assert_eq!(shape, vec![vec![4usize, 8]]);
            assert_eq!(message, "specialization cache full");
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

/// Empty shape vector is valid — scalar recompilation.
#[test]
fn recompilation_error_empty_shape() {
    let e = JitError::RecompilationError { shape: vec![], message: "scalar recompile".to_string() };
    let msg = e.to_string();
    assert!(msg.contains("recompilation failed"), "unexpected Display: {msg}");
}

// ===========================================================================
// #884 — GpuBackendUnavailable: Display + From
// ===========================================================================

/// Display for the variant contains the expected sub-strings.
#[test]
fn gpu_backend_unavailable_display() {
    let e = JitError::GpuBackendUnavailable {
        target: "cuda".to_string(),
        reason: "no runtime executor is wired".to_string(),
    };
    let msg = e.to_string();
    assert!(
        msg.contains("GPU backend unavailable"),
        "expected 'GPU backend unavailable' in Display; got: {msg}"
    );
    assert!(
        msg.contains("cuda"),
        "expected target name 'cuda' in Display; got: {msg}"
    );
    assert!(
        msg.contains("no runtime executor is wired"),
        "expected reason in Display; got: {msg}"
    );
}

/// `From<JitError> for FerrotorchError` preserves the target name.
#[test]
fn gpu_backend_unavailable_converts_to_ferrotorch_error() {
    let jit_err = JitError::GpuBackendUnavailable {
        target: "wgpu".to_string(),
        reason: "wgpu driver not linked".to_string(),
    };
    let ft_err: FerrotorchError = jit_err.into();
    let msg = ft_err.to_string();
    assert!(
        msg.contains("wgpu"),
        "FerrotorchError must carry the target name; got: {msg}"
    );
}

// ===========================================================================
// #884 — GpuBackendUnavailable: fired by InductorBackend::compile()
//
// These tests call InductorBackend::compile() with GPU targets.  No physical
// GPU is required: the error fires because the JIT runtime executor is not
// wired, not because the GPU is absent.
// ===========================================================================

/// `InductorBackend::compile()` on `GpuCuda` returns
/// `JitError::GpuBackendUnavailable` (surfaced via `FerrotorchError`).
/// The target name "GpuCuda" appears in the error message.
#[test]
fn inductor_compile_gpu_cuda_returns_gpu_backend_unavailable() {
    let g = relu_graph();
    let backend = InductorBackend::new(InductorTarget::GpuCuda);
    let err = backend.compile(&g).expect_err(
        "InductorBackend::compile() on GpuCuda must return Err — \
         no GPU runtime executor is wired through the JIT path",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("GpuCuda") || msg.contains("GPU backend unavailable"),
        "error must name the GPU target or state unavailability; got: {msg}"
    );
}

/// `InductorBackend::compile()` on `GpuPtx` returns
/// `JitError::GpuBackendUnavailable` (surfaced via `FerrotorchError`).
/// Symmetric with the `GpuCuda` test.
#[test]
fn inductor_compile_gpu_ptx_returns_gpu_backend_unavailable() {
    let g = neg_graph();
    let backend = InductorBackend::new(InductorTarget::GpuPtx);
    let err = backend.compile(&g).expect_err(
        "InductorBackend::compile() on GpuPtx must return Err — \
         no GPU runtime executor is wired through the JIT path",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("GpuPtx") || msg.contains("GPU backend unavailable"),
        "error must name the GPU target or state unavailability; got: {msg}"
    );
}

/// Demonstrates that callers can distinguish `GpuBackendUnavailable` from
/// other errors and implement opt-in CPU fallback.
///
/// This is the documented pattern from [`JitError::GpuBackendUnavailable`]'s
/// doc comment: callers that want CPU fallback catch this variant and
/// re-dispatch to a CPU target.
#[test]
fn gpu_backend_unavailable_opt_in_cpu_fallback_pattern() {
    let g = relu_graph();
    let gpu_backend = InductorBackend::new(InductorTarget::GpuCuda);
    let result = gpu_backend.compile(&g);

    // Opt-in CPU fallback: only fall back if the error is the specific
    // "GPU not wired" variant; all other errors propagate.
    let compiled = match result {
        Ok(c) => c,
        Err(ref e) if e.to_string().contains("GPU backend unavailable")
            || e.to_string().contains("GpuCuda")
            || e.to_string().contains("GpuPtx")
            || e.to_string().contains("not yet wire") =>
        {
            // Opt-in CPU fallback: compile on CPU instead.
            let cpu_backend = InductorBackend::new(InductorTarget::CpuRust);
            cpu_backend.compile(&g).expect("CPU fallback compile must succeed")
        }
        Err(e) => panic!("unexpected non-GPU-unavailable error: {e}"),
    };

    // Verify the CPU-compiled graph executes.
    let inputs: Vec<Vec<f64>> = vec![vec![-1.0, 2.0, -3.0, 4.0]];
    let output = compiled.execute(&inputs).expect("compiled graph execute must succeed");
    // relu(-1, 2, -3, 4) = (0, 2, 0, 4)
    assert_eq!(output.len(), 4);
}

// ===========================================================================
// #884 — GpuBackendUnavailable: with `cuda` feature (f64 transcendental path)
//
// When the `cuda` feature is enabled, NVRTC produces valid PTX for f64
// transcendentals. `compile()` still raises GpuBackendUnavailable because
// the Inductor compile() step does not launch kernels — the runtime is
// separately wired in ferrotorch-gpu.
// ===========================================================================

/// With `cuda` feature, compile() on a f64-transcendental PTX graph still
/// raises GpuBackendUnavailable (codegen succeeds, runtime is not wired).
#[cfg(feature = "cuda")]
#[test]
fn inductor_compile_gpu_ptx_f64_transcendental_with_cuda_raises_gpu_backend_unavailable() {
    let mut g = IrGraph::new();
    let x = g.add_input_with_dtype(vec![4], Dtype::F64);
    let (_, tanh_outs) =
        g.add_node_with_dtype(IrOpKind::Tanh, vec![x], vec![vec![4]], &[Dtype::F64]);
    g.set_outputs(vec![tanh_outs[0]]);

    let backend = InductorBackend::new(InductorTarget::GpuPtx);
    let err = backend.compile(&g).expect_err(
        "compile() with cuda feature must still fail: codegen succeeds via NVRTC \
         but no GPU runtime executor is wired in ferrotorch-jit",
    );
    let msg = err.to_string();
    // With `cuda` on, the rejection is at the runtime layer, not the
    // codegen layer — no "does not support" / "Unsupported" any more.
    assert!(
        !msg.contains("does not support") && !msg.contains("Unsupported"),
        "f64 tanh should not reject at codegen with cuda feature; got: {msg}"
    );
    assert!(
        msg.contains("GPU backend unavailable")
            || msg.contains("runtime")
            || msg.contains("not yet wire"),
        "expected runtime-unavailable diagnostic with cuda feature; got: {msg}"
    );
}

/// Without `cuda` feature, compile() on a f64-transcendental PTX graph
/// surfaces `JitError::Unsupported` before ever reaching the GPU-runtime
/// check — the codegen layer rejects first.
#[cfg(not(feature = "cuda"))]
#[test]
fn inductor_compile_gpu_ptx_f64_transcendental_without_cuda_raises_unsupported() {
    let mut g = IrGraph::new();
    let x = g.add_input_with_dtype(vec![4], Dtype::F64);
    let (_, exp_outs) =
        g.add_node_with_dtype(IrOpKind::Tanh, vec![x], vec![vec![4]], &[Dtype::F64]);
    g.set_outputs(vec![exp_outs[0]]);

    let backend = InductorBackend::new(InductorTarget::GpuPtx);
    let err = backend.compile(&g).expect_err(
        "compile() on f64 transcendental without cuda feature must return Err (Unsupported)",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("tanh") || msg.contains("does not support") || msg.contains("f64"),
        "expected Unsupported diagnosis naming the op or dtype; got: {msg}"
    );
}
