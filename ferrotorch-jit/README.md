# ferrotorch-jit

Tracing JIT compiler and graph optimizer for ferrotorch.

## What it provides

- **Tracing** -- `trace` captures a computation graph from a forward pass
- **`TracedModule`** -- compiled module via `compile` / `compile_with_config`
- **Graph IR** -- intermediate representation with `graph` module
- **Optimization** -- constant folding, dead code elimination, and more via `optimize` / `OptimizationConfig`
- **Operator fusion** -- `FusedOp`, `FusedChain`, `apply_fused`, `with_fusion` for fusing element-wise ops
- **Graph breaks** -- `trace_with_breaks`, `SegmentedModule`, `GraphSegment` for handling dynamic control flow
- **Code generation** -- `Codegen`, `CompiledGraph` with `InterpreterBackend` and `NativeBackend`
- **Memory planning** -- `plan_memory`, `MemoryPlan` for optimizing tensor allocation
- **Interpreter** -- `interpret` for executing graph IR directly

## Quick start

```rust
use ferrotorch_jit::{compile, TracedModule};
use ferrotorch_core::tensor;

let example_input = tensor(&[0.0_f32; 784]).reshape(&[1, 784]);
let traced = compile(&model, &example_input)?;
let output = traced.forward(&new_input)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
