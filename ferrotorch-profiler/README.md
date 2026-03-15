# ferrotorch-profiler

Operation profiling and Chrome trace export for ferrotorch.

## What it provides

- **`Profiler`** -- records operation timings, memory events, and input shapes during forward/backward passes
- **`with_profiler`** -- scoped profiling that captures a `ProfileReport` from a closure
- **`ProfileReport`** -- summary with `table()` for human-readable output and Chrome trace JSON export (`chrome://tracing`)
- **`ProfileEvent`** -- individual timing event with operation name, category, and input shapes
- **`OpSummary`** -- aggregated statistics per operation type (count, total time, mean, min, max)
- **`ProfileConfig`** -- configure which events to record

## Quick start

```rust
use ferrotorch_profiler::{with_profiler, ProfileConfig};

let config = ProfileConfig::default();
let (result, report) = with_profiler(config, |profiler| {
    profiler.record("matmul", "tensor_op", &[&[32, 784], &[784, 256]]);
    profiler.record("relu", "tensor_op", &[&[32, 256]]);
    42
});

println!("{}", report.table(10));  // top 10 ops by total time
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
