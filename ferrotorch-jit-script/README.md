# ferrotorch-jit-script

Source-based (`#[script]`) JIT frontend for ferrotorch.

`ferrotorch-jit` provides a *capture-based* tracer (`trace(...)` runs the
function once and snapshots the autograd tape). This crate ships the
complementary *source-based* path: annotate a Rust function with
`#[ferrotorch_jit_script::script]` and the proc macro rewrites the body
to build a `TracedModule` ahead of time.

```rust
use ferrotorch_jit_script::script;
use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;

#[script]
fn weighted_sum(a: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32> {
    let prod = mul(&a, &w)?;
    sum(&prod)
}
```

Calling `weighted_sum(a, w)` now returns
`FerrotorchResult<ferrotorch_jit::TracedModule<f32>>` — the IR captured
by tracing the body once with `requires_grad=true` leaves.

## What's in scope

The macro recognizes:

- `let x = …;` bindings.
- Calls into `ferrotorch_core::grad_fns::{arithmetic, reduction,
  activation, linalg}` on `&Tensor` arguments.
- A trailing expression as the function's return value.

Anything outside that set is left untouched and captured at the autograd
level by `trace`. Op coverage stays in lockstep with the existing tracer
— there is no second source of truth for op-name → IR mapping.

## Mirrors `torch.jit.script`

The *script* (Rust-source-to-IR) variant from PyTorch's JIT, paired with
the *trace* variant in `ferrotorch-jit`.

## License

Dual-licensed under MIT or Apache-2.0 at your option.
