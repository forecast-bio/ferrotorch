# ferrotorch-cubecl

Portable GPU backend for ferrotorch via CubeCL (CUDA, ROCm, WGPU).

## What it provides

- **`CubeRuntime`** -- auto-detects the best available GPU backend
- **`CubeDevice`** -- device handle for CubeCL-backed computation
- **Portable GPU ops** -- same operations as `ferrotorch-gpu` but compiled to CUDA PTX, AMD HIP, or WGPU (Vulkan/Metal/DX12)

CubeCL compiles a single kernel definition to multiple backends, so your code runs on NVIDIA, AMD, Intel, and Apple GPUs without changes.

## Feature flags

| Feature | Backend              | GPU vendors            |
|---------|----------------------|------------------------|
| `cuda`  | NVIDIA CUDA via PTX  | NVIDIA                 |
| `wgpu`  | WGPU (Vulkan/Metal)  | AMD, Intel, Apple, ... |
| `rocm`  | AMD HIP (native)     | AMD                    |

Enable at least one backend feature for GPU acceleration. Without any backend feature the crate still compiles (useful for CI) but `CubeRuntime::auto` returns `None`.

## Quick start

```rust
use ferrotorch_cubecl::{CubeDevice, CubeRuntime};

if let Some(rt) = CubeRuntime::auto() {
    println!("Using device: {:?}", rt.device());
}
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
