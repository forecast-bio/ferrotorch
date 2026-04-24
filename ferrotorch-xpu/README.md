# ferrotorch-xpu

Intel XPU (Arc / Data Center GPU Max) backend for ferrotorch via CubeCL wgpu.

## What it provides

- **`XpuDevice`** -- device handle wrapping a CubeCL wgpu runtime targeting Intel GPUs through Vulkan
- **Element-wise ops** -- `xpu_add`, `xpu_sub`, `xpu_mul`, `xpu_div`, `xpu_neg`, `xpu_abs`, `xpu_relu`
- **Transcendentals** -- `xpu_exp`, `xpu_ln`, `xpu_sqrt`, `xpu_sin`, `xpu_cos`, `xpu_tanh`, `xpu_sigmoid`
- **Matrix multiply** -- `xpu_matmul` (2-D)

Switching a model from CUDA to XPU is a single line:

```rust
use ferrotorch_core::Device;
use ferrotorch_xpu::XpuDevice;

let xpu = XpuDevice::new(0)?;
let a = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0])?.to(Device::Xpu(0))?;
let b = ferrotorch_core::tensor(&[10.0_f32, 20.0, 30.0])?.to(Device::Xpu(0))?;
let c = ferrotorch_xpu::xpu_add(&a, &b, &xpu)?;
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wgpu`  | **yes** | Enables the CubeCL wgpu runtime. Without it, all ops return `DeviceUnavailable`. |

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
