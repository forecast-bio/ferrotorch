pub mod codegen;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod graph_break;
pub mod interpreter;
pub mod memory_plan;
pub mod module;
pub mod optimize;
pub mod serialize;
pub mod trace;

pub use codegen::{Codegen, CompiledGraph, InterpreterBackend, NativeBackend};
pub use error::JitError;
pub use fusion::{
    apply_fused, estimate_matmul_dims, estimate_numel_for_inputs,
    generate_reduction_c, generate_reduction_ptx,
    is_fusion_enabled, with_fusion, FusedChain, FusedOp, ReductionKind,
};
pub use graph_break::{GraphSegment, SegmentedModule, TraceResult, trace_with_breaks};
pub use interpreter::interpret;
pub use memory_plan::{plan_memory, MemoryPlan};
pub use module::{compile, compile_with_config, CompileConfig, TracedModule};
pub use optimize::{optimize, OptimizationConfig};
pub use trace::trace;
