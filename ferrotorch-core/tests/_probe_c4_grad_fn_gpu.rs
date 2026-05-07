//! Permanent regression sentinel for Sprint C.4 (#524 partial): GPU backward
//! `grad_fn` paths wired for the transformer / training-loop hot paths.
//!
//! Pre-fix observable failure: calling `.backward()` on a CUDA tensor through
//! any of the ops below returned `Err(NotImplementedOnCuda { op: … })`.
//!
//! Post-fix (this probe):
//! - Each site now has a GPU-native backward that keeps gradients on-device.
//! - `grad.is_cuda()` asserts confirm no host round-trip.
//! - Gradient values match the CPU reference within F32_GRAD = 1e-5.
//!
//! PyTorch parity (rust-gpu-discipline §3): PyTorch's autograd backward runs on
//! the same device as the forward. Returning `Err(NotImplementedOnCuda)` was a
//! parity violation — these sites now return structured `Err(…)` only for dtypes
//! without a GPU kernel, mirroring PyTorch's `NotImplementedError` for those
//! combinations.
//!
//! Covered sites (top-10 from the sprint):
//! 1.  `MvBackward` (f32/f64)
//! 2.  `DotBackward` (f32/f64)
//! 3.  `MatmulBackward(vm: 1D@2D)` (f32/f64)
//! 4.  `MatmulBackward(broadcast Nd)` (f32/f64)
//! 5.  `GatherBackward` (f32) — grad_fn tested directly
//! 6.  `ScatterBackward` (f32) — grad_fn tested directly
//! 7.  `ScatterAddBackward` (f32) — grad_fn tested directly
//! 8.  `WhereCondBackward` (f32) — grad_fn tested directly

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::indexing::{
    GatherBackward, ScatterAddBackward, ScatterBackward, WhereCondBackward,
};
use ferrotorch_core::grad_fns::linalg::{
    dot_differentiable, matmul_differentiable, mv_differentiable,
};
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{Device, Tensor};

/// F32 gradient tolerance (1 ULP at f32 near 1.0 is ~1.2e-7; 1e-5 gives margin
/// for the extra PTX rounding vs the reference CPU path).
const F32_GRAD: f32 = 1e-5;
const F64_GRAD: f64 = 1e-5;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the C4 probe suite");
    });
}

/// Build a CPU f32 tensor with requires_grad, then move to CUDA.
fn gpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    ensure_cuda_backend();
    from_vec(data, shape)
        .expect("from_vec")
        .requires_grad_(true)
        .to(Device::Cuda(0))
        .expect("to cuda")
}

/// Build a CPU f64 tensor with requires_grad, then move to CUDA.
fn gpu_f64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    ensure_cuda_backend();
    from_vec(data, shape)
        .expect("from_vec f64")
        .requires_grad_(true)
        .to(Device::Cuda(0))
        .expect("to cuda")
}

/// Build a CPU f32 tensor (no grad, not on CUDA) — used for seed/reference.
fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    from_vec(data, shape).expect("from_vec cpu f32")
}

/// Build a CPU f64 tensor.
fn cpu_f64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    from_vec(data, shape).expect("from_vec cpu f64")
}

/// Build a CPU f32 tensor with requires_grad (for reference CPU backward).
fn cpu_f32_grad(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    from_vec(data, shape).expect("from_vec").requires_grad_(true)
}

/// Build a CPU f64 tensor with requires_grad.
fn cpu_f64_grad(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    from_vec(data, shape).expect("from_vec").requires_grad_(true)
}

fn read_f32(t: &Tensor<f32>) -> Vec<f32> {
    let cpu = if t.is_cuda() { t.cpu().expect("d2h") } else { t.clone() };
    cpu.data().expect("read").to_vec()
}

fn read_f64(t: &Tensor<f64>) -> Vec<f64> {
    let cpu = if t.is_cuda() { t.cpu().expect("d2h") } else { t.clone() };
    cpu.data().expect("read").to_vec()
}

// ---------------------------------------------------------------------------
// 1. MvBackward — y = A @ x
// ---------------------------------------------------------------------------

#[test]
fn mv_backward_f32_gpu_stays_on_device() {
    // A: (3,4), x: (4,)
    let a_data: Vec<f32> = (1..=12).map(|v| v as f32 * 0.1).collect();
    let x_data: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0];

    let a_gpu = gpu_f32(a_data.clone(), &[3, 4]);
    let x_gpu = gpu_f32(x_data.clone(), &[4]);

    let y_gpu = mv_differentiable(&a_gpu, &x_gpu).expect("mv forward gpu");
    assert!(y_gpu.is_cuda(), "mv output must be on CUDA");

    // seed on GPU
    let seed_gpu = gpu_f32(vec![1.0; 3], &[3]);
    let grad_fn = y_gpu.grad_fn().expect("mv must have grad_fn");
    let grads = grad_fn.backward(&seed_gpu).expect("mv backward gpu");

    let grad_a = grads[0].as_ref().expect("grad_a");
    let grad_x = grads[1].as_ref().expect("grad_x");
    assert!(grad_a.is_cuda(), "grad_A must stay on CUDA — §3 parity");
    assert!(grad_x.is_cuda(), "grad_x must stay on CUDA — §3 parity");

    // CPU reference
    let a_cpu = cpu_f32_grad(a_data.clone(), &[3, 4]);
    let x_cpu = cpu_f32_grad(x_data.clone(), &[4]);
    let y_cpu = mv_differentiable(&a_cpu, &x_cpu).expect("mv cpu");
    let seed_cpu = cpu_f32(vec![1.0; 3], &[3]);
    let grads_cpu = y_cpu.grad_fn().unwrap().backward(&seed_cpu).expect("mv cpu backward");

    for (g, c) in read_f32(grad_a).iter().zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "grad_A mismatch: {g} vs {c}");
    }
    for (g, c) in read_f32(grad_x).iter().zip(read_f32(grads_cpu[1].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "grad_x mismatch: {g} vs {c}");
    }
}

#[test]
fn mv_backward_f64_gpu_stays_on_device() {
    let a_data: Vec<f64> = (1..=6).map(|v| v as f64 * 0.1).collect();
    let x_data: Vec<f64> = vec![1.0, -1.0, 0.5];

    let a_gpu = gpu_f64(a_data.clone(), &[2, 3]);
    let x_gpu = gpu_f64(x_data.clone(), &[3]);

    let y_gpu = mv_differentiable(&a_gpu, &x_gpu).expect("mv f64 forward gpu");
    let seed_gpu = gpu_f64(vec![1.0; 2], &[2]);
    let grads = y_gpu.grad_fn().unwrap().backward(&seed_gpu).expect("mv f64 backward gpu");

    assert!(grads[0].as_ref().unwrap().is_cuda(), "grad_A f64 must be on CUDA — §3");
    assert!(grads[1].as_ref().unwrap().is_cuda(), "grad_x f64 must be on CUDA — §3");

    let a_cpu = cpu_f64_grad(a_data.clone(), &[2, 3]);
    let x_cpu = cpu_f64_grad(x_data.clone(), &[3]);
    let y_cpu = mv_differentiable(&a_cpu, &x_cpu).expect("mv f64 cpu");
    let seed_cpu = cpu_f64(vec![1.0; 2], &[2]);
    let grads_cpu = y_cpu.grad_fn().unwrap().backward(&seed_cpu).expect("mv f64 cpu backward");

    for (g, c) in read_f64(grads[0].as_ref().unwrap())
        .iter()
        .zip(read_f64(grads_cpu[0].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F64_GRAD, "grad_A f64 mismatch: {g} vs {c}");
    }
}

// ---------------------------------------------------------------------------
// 2. DotBackward — s = dot(a, b)
// ---------------------------------------------------------------------------

#[test]
fn dot_backward_f32_gpu_stays_on_device() {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![0.5f32, -0.5, 1.0, -1.0];

    let a_gpu = gpu_f32(a_data.clone(), &[4]);
    let b_gpu = gpu_f32(b_data.clone(), &[4]);

    let s_gpu = dot_differentiable(&a_gpu, &b_gpu).expect("dot forward gpu");
    // scalar seed on GPU
    let seed_gpu = gpu_f32(vec![1.0], &[]);
    let grads = s_gpu.grad_fn().unwrap().backward(&seed_gpu).expect("dot backward gpu");

    let grad_a = grads[0].as_ref().unwrap();
    let grad_b = grads[1].as_ref().unwrap();
    assert!(grad_a.is_cuda(), "dot grad_a must be on CUDA — §3");
    assert!(grad_b.is_cuda(), "dot grad_b must be on CUDA — §3");

    let a_cpu = cpu_f32_grad(a_data.clone(), &[4]);
    let b_cpu = cpu_f32_grad(b_data.clone(), &[4]);
    let s_cpu = dot_differentiable(&a_cpu, &b_cpu).expect("cpu dot");
    let seed_cpu = cpu_f32(vec![1.0], &[]);
    let grads_cpu = s_cpu.grad_fn().unwrap().backward(&seed_cpu).expect("cpu dot backward");

    for (g, c) in read_f32(grad_a).iter().zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "dot grad_a mismatch: {g} vs {c}");
    }
    for (g, c) in read_f32(grad_b).iter().zip(read_f32(grads_cpu[1].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "dot grad_b mismatch: {g} vs {c}");
    }
}

#[test]
fn dot_backward_f64_gpu_stays_on_device() {
    let a_data = vec![1.0f64, -2.0, 0.5];
    let b_data = vec![2.0f64, 1.0, -1.0];

    let a_gpu = gpu_f64(a_data.clone(), &[3]);
    let b_gpu = gpu_f64(b_data.clone(), &[3]);

    let s_gpu = dot_differentiable(&a_gpu, &b_gpu).expect("dot f64 gpu");
    let seed_gpu = gpu_f64(vec![1.0], &[]);
    let grads = s_gpu.grad_fn().unwrap().backward(&seed_gpu).expect("dot f64 backward gpu");

    assert!(grads[0].as_ref().unwrap().is_cuda(), "dot f64 grad_a must be on CUDA — §3");
    assert!(grads[1].as_ref().unwrap().is_cuda(), "dot f64 grad_b must be on CUDA — §3");

    let a_cpu = cpu_f64_grad(a_data.clone(), &[3]);
    let b_cpu = cpu_f64_grad(b_data.clone(), &[3]);
    let s_cpu = dot_differentiable(&a_cpu, &b_cpu).expect("cpu dot f64");
    let seed_cpu = cpu_f64(vec![1.0], &[]);
    let grads_cpu = s_cpu.grad_fn().unwrap().backward(&seed_cpu).unwrap();

    for (g, c) in read_f64(grads[0].as_ref().unwrap())
        .iter()
        .zip(read_f64(grads_cpu[0].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F64_GRAD, "dot f64 grad_a mismatch: {g} vs {c}");
    }
}

// ---------------------------------------------------------------------------
// 3. MatmulBackward(vm: 1D@2D)
// ---------------------------------------------------------------------------

#[test]
fn matmul_vm_backward_f32_gpu_stays_on_device() {
    // a: (3,), B: (3,4) → y: (4,)
    let a_data = vec![1.0f32, 2.0, 3.0];
    let b_data: Vec<f32> = (1..=12).map(|v| v as f32 * 0.1).collect();

    let a_gpu = gpu_f32(a_data.clone(), &[3]);
    let b_gpu = gpu_f32(b_data.clone(), &[3, 4]);

    let y_gpu = matmul_differentiable(&a_gpu, &b_gpu).expect("vm forward gpu");
    assert!(y_gpu.is_cuda());

    let seed_gpu = gpu_f32(vec![1.0; 4], &[4]);
    let grads = y_gpu.grad_fn().unwrap().backward(&seed_gpu).expect("vm backward gpu");

    let grad_a = grads[0].as_ref().unwrap();
    let grad_b = grads[1].as_ref().unwrap();
    assert!(grad_a.is_cuda(), "vm grad_a must be on CUDA — §3");
    assert!(grad_b.is_cuda(), "vm grad_b must be on CUDA — §3");
    assert_eq!(grad_a.shape(), &[3]);
    assert_eq!(grad_b.shape(), &[3, 4]);

    let a_cpu = cpu_f32_grad(a_data.clone(), &[3]);
    let b_cpu = cpu_f32_grad(b_data.clone(), &[3, 4]);
    let y_cpu = matmul_differentiable(&a_cpu, &b_cpu).expect("cpu vm");
    let seed_cpu = cpu_f32(vec![1.0; 4], &[4]);
    let grads_cpu = y_cpu.grad_fn().unwrap().backward(&seed_cpu).expect("cpu vm backward");

    for (g, c) in read_f32(grad_a).iter().zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "vm grad_a mismatch: {g} vs {c}");
    }
    for (g, c) in read_f32(grad_b).iter().zip(read_f32(grads_cpu[1].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "vm grad_b mismatch: {g} vs {c}");
    }
}

// ---------------------------------------------------------------------------
// 4. MatmulBackward(broadcast Nd)
// ---------------------------------------------------------------------------

#[test]
fn matmul_broadcast_backward_f32_gpu_stays_on_device() {
    // A: (2,3,4), B: (4,5) → C: (2,3,5)
    let a_data: Vec<f32> = (1..=24).map(|v| v as f32 * 0.05).collect();
    let b_data: Vec<f32> = (1..=20).map(|v| v as f32 * 0.05).collect();

    let a_gpu = gpu_f32(a_data.clone(), &[2, 3, 4]);
    let b_gpu = gpu_f32(b_data.clone(), &[4, 5]);

    let c_gpu = matmul_differentiable(&a_gpu, &b_gpu).expect("broadcast matmul gpu");
    assert!(c_gpu.is_cuda());

    let seed_gpu = gpu_f32(vec![1.0; 30], &[2, 3, 5]);
    let grads = c_gpu.grad_fn().unwrap().backward(&seed_gpu).expect("broadcast backward gpu");

    let grad_a = grads[0].as_ref().unwrap();
    let grad_b = grads[1].as_ref().unwrap();
    assert!(grad_a.is_cuda(), "broadcast grad_A must be on CUDA — §3");
    assert!(grad_b.is_cuda(), "broadcast grad_B must be on CUDA — §3");
    assert_eq!(grad_a.shape(), &[2, 3, 4]);
    assert_eq!(grad_b.shape(), &[4, 5]);

    let a_cpu = cpu_f32_grad(a_data.clone(), &[2, 3, 4]);
    let b_cpu = cpu_f32_grad(b_data.clone(), &[4, 5]);
    let c_cpu = matmul_differentiable(&a_cpu, &b_cpu).expect("cpu broadcast");
    let seed_cpu = cpu_f32(vec![1.0; 30], &[2, 3, 5]);
    let grads_cpu = c_cpu.grad_fn().unwrap().backward(&seed_cpu).expect("cpu broadcast backward");

    for (g, c) in read_f32(grad_a).iter().zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "broadcast grad_A mismatch: {g} vs {c}");
    }
    for (g, c) in read_f32(grad_b).iter().zip(read_f32(grads_cpu[1].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "broadcast grad_B mismatch: {g} vs {c}");
    }
}

// ---------------------------------------------------------------------------
// 5. GatherBackward — test grad_fn directly with GPU tensors
// ---------------------------------------------------------------------------

#[test]
fn gather_backward_f32_gpu_stays_on_device() {
    // input: (3,4), gather along dim=1 with indices [[0,2],[1,3],[0,1]]
    let inp_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let idx_data = vec![0usize, 2, 1, 3, 0, 1];
    let idx_shape = vec![3, 2];
    let input_shape = vec![3, 4];

    // Construct a GPU input tensor and a GPU grad_output.
    let inp_gpu = gpu_f32(inp_data.clone(), &input_shape);
    // The grad_output has shape = idx_shape (output of gather).
    let go_gpu = gpu_f32(vec![1.0; 6], &[3, 2]);

    // Construct the GatherBackward node directly (pre-fix this returned NotImplementedOnCuda).
    let grad_fn = GatherBackward {
        input: inp_gpu.clone(),
        dim: 1,
        index: idx_data.clone(),
        index_shape: idx_shape.clone(),
    };
    let grads = grad_fn.backward(&go_gpu).expect("gather backward gpu");
    let grad_inp = grads[0].as_ref().unwrap();
    assert!(grad_inp.is_cuda(), "gather grad_input must be on CUDA — §3");
    assert_eq!(grad_inp.shape(), &[3, 4]);

    // CPU reference: same grad_fn with CPU tensors and CPU seed.
    let inp_cpu = cpu_f32_grad(inp_data.clone(), &input_shape);
    let go_cpu = cpu_f32(vec![1.0; 6], &[3, 2]);
    let grad_fn_cpu = GatherBackward {
        input: inp_cpu.clone(),
        dim: 1,
        index: idx_data.clone(),
        index_shape: idx_shape.clone(),
    };
    let grads_cpu = grad_fn_cpu.backward(&go_cpu).expect("gather backward cpu");

    for (g, c) in read_f32(grad_inp).iter().zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter()) {
        assert!((g - c).abs() < F32_GRAD, "gather grad mismatch: {g} vs {c}");
    }
}

// ---------------------------------------------------------------------------
// 6. ScatterBackward — grad_fn tested directly
// ---------------------------------------------------------------------------

#[test]
fn scatter_backward_f32_gpu_stays_on_device() {
    let inp_data: Vec<f32> = vec![0.0; 12];
    let src_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let idx_data = vec![0usize, 2, 1, 3, 0, 1];
    let idx_shape = vec![3, 2];
    let input_shape = vec![3, 4];

    let inp_gpu = gpu_f32(inp_data.clone(), &input_shape);
    let src_gpu = gpu_f32(src_data.clone(), &[3, 2]);
    // grad_output has same shape as the scatter output (= input shape).
    let go_gpu = gpu_f32((1..=12).map(|v| v as f32).collect(), &input_shape);

    let grad_fn = ScatterBackward {
        input: inp_gpu.clone(),
        src: src_gpu.clone(),
        dim: 1,
        index: idx_data.clone(),
        index_shape: idx_shape.clone(),
    };
    let grads = grad_fn.backward(&go_gpu).expect("scatter backward gpu");
    let grad_inp = grads[0].as_ref().unwrap();
    let grad_src = grads[1].as_ref().unwrap();
    assert!(grad_inp.is_cuda(), "scatter grad_input must be on CUDA — §3");
    assert!(grad_src.is_cuda(), "scatter grad_src must be on CUDA — §3");

    // CPU reference
    let inp_cpu = cpu_f32_grad(inp_data.clone(), &input_shape);
    let src_cpu = cpu_f32_grad(src_data.clone(), &[3, 2]);
    let go_cpu = cpu_f32((1..=12).map(|v| v as f32).collect(), &input_shape);
    let grad_fn_cpu = ScatterBackward {
        input: inp_cpu.clone(),
        src: src_cpu.clone(),
        dim: 1,
        index: idx_data.clone(),
        index_shape: idx_shape.clone(),
    };
    let grads_cpu = grad_fn_cpu.backward(&go_cpu).expect("scatter backward cpu");

    for (g, c) in read_f32(grad_inp)
        .iter()
        .zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F32_GRAD, "scatter grad_input mismatch: {g} vs {c}");
    }
    for (g, c) in read_f32(grad_src)
        .iter()
        .zip(read_f32(grads_cpu[1].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F32_GRAD, "scatter grad_src mismatch: {g} vs {c}");
    }
}

// ---------------------------------------------------------------------------
// 7. ScatterAddBackward — grad_fn tested directly
// ---------------------------------------------------------------------------

#[test]
fn scatter_add_backward_f32_gpu_stays_on_device() {
    let inp_data: Vec<f32> = vec![1.0; 12];
    let src_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let idx_data = vec![0usize, 2, 1, 3, 0, 1];
    let idx_shape = vec![3, 2];
    let input_shape = vec![3, 4];

    let inp_gpu = gpu_f32(inp_data.clone(), &input_shape);
    let src_gpu = gpu_f32(src_data.clone(), &[3, 2]);
    let go_gpu = gpu_f32((1..=12).map(|v| v as f32).collect(), &input_shape);

    let grad_fn = ScatterAddBackward {
        input: inp_gpu.clone(),
        src: src_gpu.clone(),
        dim: 1,
        index: idx_data.clone(),
        index_shape: idx_shape.clone(),
    };
    let grads = grad_fn.backward(&go_gpu).expect("scatter_add backward gpu");
    let grad_inp = grads[0].as_ref().unwrap();
    let grad_src = grads[1].as_ref().unwrap();
    assert!(grad_inp.is_cuda(), "scatter_add grad_input must be on CUDA — §3");
    assert!(grad_src.is_cuda(), "scatter_add grad_src must be on CUDA — §3");

    // CPU reference
    let inp_cpu = cpu_f32_grad(inp_data.clone(), &input_shape);
    let src_cpu = cpu_f32_grad(src_data.clone(), &[3, 2]);
    let go_cpu = cpu_f32((1..=12).map(|v| v as f32).collect(), &input_shape);
    let grad_fn_cpu = ScatterAddBackward {
        input: inp_cpu.clone(),
        src: src_cpu.clone(),
        dim: 1,
        index: idx_data.clone(),
        index_shape: idx_shape.clone(),
    };
    let grads_cpu = grad_fn_cpu.backward(&go_cpu).expect("scatter_add backward cpu");

    for (g, c) in read_f32(grad_inp)
        .iter()
        .zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F32_GRAD, "scatter_add grad_input mismatch: {g} vs {c}");
    }
    for (g, c) in read_f32(grad_src)
        .iter()
        .zip(read_f32(grads_cpu[1].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F32_GRAD, "scatter_add grad_src mismatch: {g} vs {c}");
    }
}

// ---------------------------------------------------------------------------
// 8. WhereCondBackward — grad_fn tested directly
// ---------------------------------------------------------------------------

#[test]
fn where_cond_backward_f32_gpu_stays_on_device() {
    let condition = vec![true, false, true, false, true, false];
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let shape = [6];

    let x_gpu = gpu_f32(x_data.clone(), &shape);
    let y_gpu = gpu_f32(y_data.clone(), &shape);
    let go_gpu = gpu_f32(vec![1.0; 6], &shape);

    let grad_fn = WhereCondBackward {
        x: x_gpu.clone(),
        y: y_gpu.clone(),
        condition: condition.clone(),
    };
    let grads = grad_fn.backward(&go_gpu).expect("where_cond backward gpu");
    let grad_x = grads[0].as_ref().unwrap();
    let grad_y = grads[1].as_ref().unwrap();
    assert!(grad_x.is_cuda(), "where_cond grad_x must be on CUDA — §3");
    assert!(grad_y.is_cuda(), "where_cond grad_y must be on CUDA — §3");

    // CPU reference
    let x_cpu = cpu_f32_grad(x_data.clone(), &shape);
    let y_cpu = cpu_f32_grad(y_data.clone(), &shape);
    let go_cpu = cpu_f32(vec![1.0; 6], &shape);
    let grad_fn_cpu = WhereCondBackward {
        x: x_cpu.clone(),
        y: y_cpu.clone(),
        condition: condition.clone(),
    };
    let grads_cpu = grad_fn_cpu.backward(&go_cpu).expect("where_cond backward cpu");

    for (g, c) in read_f32(grad_x)
        .iter()
        .zip(read_f32(grads_cpu[0].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F32_GRAD, "where_cond grad_x mismatch: {g} vs {c}");
    }
    for (g, c) in read_f32(grad_y)
        .iter()
        .zip(read_f32(grads_cpu[1].as_ref().unwrap()).iter())
    {
        assert!((g - c).abs() < F32_GRAD, "where_cond grad_y mismatch: {g} vs {c}");
    }
}
