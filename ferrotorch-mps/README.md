# ferrotorch-mps

Apple Silicon Metal Performance Shaders (MPS) backend skeleton for
ferrotorch.

## What's here

This crate ships the platform-detection and `Device::Mps(_)` plumbing
that any caller can use unconditionally. The real Metal kernel layer
lands behind the `metal-backend` feature flag, intentionally **off by
default** so Linux / Windows builds compile cleanly and surface the
"unavailable" path at runtime.

```rust
use ferrotorch_mps::{is_mps_available, MpsDevice};

if is_mps_available() {
    let dev = MpsDevice::new(0).unwrap();
    // dispatch tensor ops via Device::Mps(0) — kernels gated on `metal-backend`
}
```

## Status

| Piece | State |
|---|---|
| `is_mps_available()` runtime probe | shipping (returns `false` off-Apple) |
| `MpsDevice` ordinal handle | shipping |
| `Device::Mps(_)` core enum integration | shipping |
| MSL kernel layer (`metal-backend`) | 10 kernels scaffolded (matmul, bmm, softmax, sum_axis, add/sub/mul/div, relu, sigmoid); ~80 `GpuBackend` trait methods pending macOS CI |

The 10 scaffolded MSL kernels cover the critical path for single-layer inference. Full `GpuBackend` parity (~80 methods) is deferred pending Apple hardware access in CI. The public API contract is stable so downstream code can wire `Device::Mps(0)` paths today and pick up kernel coverage without source changes.

## Why split out

- Linux is the workspace's primary CI target. Adding `metal` /
  `objc2-metal` dependencies to a non-optional crate would break that.
- macOS users who want a native Metal path (independent of WGPU) get a
  dedicated home for the kernel layer.

## License

Dual-licensed under MIT or Apache-2.0 at your option.
