//! Probe B1 -- Issue #894: avgpool2d_forward_kernel PTX JIT fail on sm_86 (RTX 3090).
//!
//! Root cause: AVGPOOL2D_PTX contained a non-ASCII em-dash (U+2014, "--") inside an
//! inline PTX comment. The CUDA JIT driver rejects PTX strings with non-ASCII bytes,
//! returning CUDA_ERROR_INVALID_PTX. Same root cause as #893 (gelu_tanh).
//!
//! BEFORE fix: gpu_avgpool2d returned Err(PtxCompileFailed { kernel:
//!   "avgpool2d_forward_kernel", source: DriverError(CUDA_ERROR_INVALID_PTX) })
//!
//! AFTER fix: em-dash replaced with ASCII "--". gpu_avgpool2d succeeds and output
//!   matches torch.nn.functional.avg_pool2d reference.
//!
//! Shape: [1, 1, 4, 4] input, 2x2 kernel, stride 2, no padding.
//! Expected output: [1, 1, 2, 2] with each element = mean of the 2x2 input tile.

#![cfg(feature = "gpu")]

use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for probe B1 avgpool2d");
    });
}

/// Probe: AFTER fix avgpool2d_forward_kernel JIT-compiles and produces correct output.
#[test]
fn probe_b1_avgpool2d_ptx_after() {
    ensure_cuda();
    use ferrotorch_gpu::device::GpuDevice;
    use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

    let dev = GpuDevice::new(0).expect("GpuDevice::new(0)");

    // Input: [1, 1, 4, 4] row-major
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    // 12 13 14 15
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();

    // 2x2 kernel, stride 2, no padding -> [1, 1, 2, 2] output
    // tile (0,0): mean(0,1,4,5)  = 2.5
    // tile (0,1): mean(2,3,6,7)  = 4.5
    // tile (1,0): mean(8,9,12,13)= 10.5
    // tile (1,1): mean(10,11,14,15)=12.5
    let expected = [2.5f32, 4.5, 10.5, 12.5];

    let inp = cpu_to_gpu(&input, &dev).unwrap();

    // BEFORE fix: Err(PtxCompileFailed { kernel: "avgpool2d_forward_kernel", ... INVALID_PTX })
    // AFTER  fix: Ok((output, [1, 1, 2, 2]))
    let (out, out_shape) = ferrotorch_gpu::kernels::gpu_avgpool2d(
        &inp,
        1, // batch
        1, // channels
        4, // h_in
        4, // w_in
        2, // kh
        2, // kw
        2, // sh
        2, // sw
        0, // ph
        0, // pw
        &dev,
    )
    .expect("AFTER fix #894: avgpool2d_forward_kernel must JIT-compile and run on sm_86");

    assert_eq!(out_shape, [1, 1, 2, 2], "probe B1 #894: output shape mismatch");

    let actual = gpu_to_cpu(&out, &dev).unwrap();
    assert_eq!(actual.len(), 4, "probe B1 #894: output length");

    const TOL: f32 = 1e-5;
    for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (act - exp).abs() <= TOL,
            "probe B1 #894 AFTER: index {i} actual={act:.4} expected={exp:.4}"
        );
    }
}
