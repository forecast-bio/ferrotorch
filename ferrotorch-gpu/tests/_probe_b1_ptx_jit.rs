//! Probe: direct PTX JIT test for gelu_tanh and avgpool2d on sm_86.
//! Run: cargo test -p ferrotorch-gpu --features cuda --test _probe_b1_ptx_jit -- --nocapture

#![cfg(feature = "cuda")]

use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::transfer::cpu_to_gpu;

fn device() -> GpuDevice {
    GpuDevice::new(0).expect("GpuDevice::new(0)")
}

#[test]
fn probe_gelu_tanh_ptx_jit_error() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA init");
    let dev = device();
    let a_h: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let a = cpu_to_gpu(&a_h, &dev).unwrap();
    let result = ferrotorch_gpu::kernels::gpu_gelu_tanh(&a, &dev);
    println!("gelu_tanh result: {:?}", result);
    match &result {
        Ok(_) => println!("PROBE: gelu_tanh JIT SUCCEEDED on this sm"),
        Err(e) => println!("PROBE BEFORE: gelu_tanh JIT FAILED: {e}"),
    }
}

#[test]
fn probe_avgpool2d_ptx_jit_error() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA init");
    let dev = device();
    let inp_h: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let inp = cpu_to_gpu(&inp_h, &dev).unwrap();
    let result = ferrotorch_gpu::kernels::gpu_avgpool2d(&inp, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, &dev);
    println!("avgpool2d result: {:?}", result.as_ref().map(|_| "Ok(...)"));
    match &result {
        Ok(_) => println!("PROBE: avgpool2d JIT SUCCEEDED on this sm"),
        Err(e) => println!("PROBE BEFORE: avgpool2d JIT FAILED: {e}"),
    }
}

#[test]
fn probe_batchnorm_is_stub() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA init");
    let dev = device();
    let inp_h: Vec<f32> = vec![1.0f32; 4 * 3 * 8 * 8];
    let weight_h = vec![1.0f32; 3];
    let bias_h = vec![0.0f32; 3];
    let rmean_h = vec![0.0f32; 3];
    let rvar_h = vec![1.0f32; 3];
    let inp = cpu_to_gpu(&inp_h, &dev).unwrap();
    let weight = cpu_to_gpu(&weight_h, &dev).unwrap();
    let bias = cpu_to_gpu(&bias_h, &dev).unwrap();
    let mut rmean = cpu_to_gpu(&rmean_h, &dev).unwrap();
    let mut rvar = cpu_to_gpu(&rvar_h, &dev).unwrap();
    let result = ferrotorch_gpu::kernels::gpu_batchnorm_forward(
        &inp, &weight, &bias, &mut rmean, &mut rvar,
        3, 64, 1e-5, 0.1, true, &dev,
    );
    println!("batchnorm_forward result: {:?}", result.as_ref().map(|_| "Ok(...)"));
    match &result {
        Ok(_) => println!("PROBE: batchnorm_forward returned Ok (stub removed)"),
        Err(e) => println!("PROBE BEFORE: batchnorm_forward is a stub, returned Err: {e}"),
    }
}
