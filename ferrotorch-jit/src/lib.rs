//! Tracing JIT compiler and graph optimizer for ferrotorch.
//!
//! Provides graph capture (`trace`), an IR (`graph`), fusion + DAG planning
//! (`fusion`, `dag_fusion`), backend codegen (`codegen`, `codegen_cpu`,
//! `codegen_gpu`, `codegen_ir`, `codegen_jit`), `AoT` autograd
//! (`aot_autograd`), shape-symbolic specialisation (`symbolic`), `AoT` export
//! (`export`), and the eager interpreter fallback (`interpreter`).

#![warn(clippy::all, clippy::pedantic)]
#![deny(unsafe_code, rust_2018_idioms, missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Add to this list only with a one-line justification.
#![allow(
    // The IR is laid out so helper structs inherit their parents' naming;
    // unifying would break call-site ergonomics.
    clippy::module_name_repetitions,
    // # Errors / # Panics sections will be added during the rustdoc pass
    // tracked as a follow-up issue, not gated on this lint baseline.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Codegen `match` blocks mirror the IR taxonomy 1:1; splitting reduces
    // legibility.
    clippy::too_many_lines,
    // Trivial casts are pervasive in codegen (offsets, indices) and the
    // explicit cast is more readable than alternatives.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    // Codegen builds large strings with `push_str(&format!(...))`. Rewriting
    // to `write!(s, ...).unwrap()` adds a fallible path that genuinely
    // can't fail and obscures the build pattern.
    clippy::format_push_string,
    // `#[must_use]` on every getter is churn for marginal value; callers in
    // this codebase already use the returned values.
    clippy::must_use_candidate,
    // `let ... else { return }` rewrites of `match { Some(x) => x, None => return }`
    // are often less readable when the match arm is the natural pattern.
    clippy::manual_let_else,
    // Test/helper modules define small private fns after `let`-bindings; the
    // hoisting requirement is style-only.
    clippy::items_after_statements,
    // Hex-encoded constants in codegen and hashing don't gain readability
    // from the underscore separators clippy prefers.
    clippy::unreadable_literal,
    // `HashMap<K, ()>` is used in places where insertion order doesn't
    // matter and `HashSet` would be a worse fit for the surrounding API.
    clippy::zero_sized_map_values,
    // Builder-style methods on configs that return `Self` already document
    // their consume-and-return pattern; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
)]
// `missing_docs` is held at warn while the crate-wide rustdoc pass is
// tracked in a follow-up issue; flip to deny once that pass lands.
#![allow(missing_docs)]

pub mod aot_autograd;
pub mod autotune;
pub mod codegen;
pub mod codegen_cpu;
pub mod codegen_gpu;
pub mod codegen_ir;
pub mod codegen_jit;
pub mod dag_fusion;
pub mod error;
pub mod export;
pub mod fusion;
pub mod graph;
pub mod graph_break;
pub mod interpreter;
pub mod memory_plan;
pub mod module;
pub mod optimize;
pub mod serialize;
pub mod symbolic;
pub mod trace;

pub use aot_autograd::{AotGraphPair, compile_aot, decompose_forward_backward};
pub use autotune::{AutotuneCandidate, AutotuneKey, AutotuneResult, Autotuner};
pub use codegen::{
    Codegen, CompiledGraph, InductorBackend, InductorTarget, InterpreterBackend, NativeBackend,
};
pub use codegen_cpu::CpuCodegen;
pub use codegen_gpu::GpuCodegen;
pub use codegen_ir::{BinOpKind, Expr, LoopIR, UnaryOpKind};
pub use codegen_jit::{JitCompiledKernel, compile_loop_ir_kernel};
pub use dag_fusion::{FusionGroup, FusionGroupKind};
pub use error::JitError;
pub use export::{
    DimSpec, ExportedProgram, ExportedProgramMetadata, InputSpec, export,
    export_with_dynamic_shapes,
};
pub use fusion::{
    FusedChain, FusedOp, ReductionKind, apply_fused, estimate_matmul_dims,
    estimate_numel_for_inputs, generate_reduction_c, generate_reduction_ptx, is_fusion_enabled,
    with_fusion,
};
pub use graph_break::{GraphSegment, SegmentedModule, TraceResult, trace_with_breaks};
pub use interpreter::{interpret, interpret_multi};
pub use memory_plan::{MemoryPlan, plan_memory};
pub use module::{AotCompiledModule, CompileConfig, TracedModule, compile, compile_with_config};
pub use optimize::{OptimizationConfig, optimize};
pub use symbolic::{Guard, ShapeSignature, SymbolicDim, SymbolicTracedModule, compile_symbolic};
pub use trace::trace;
