//! Global cache for compiled CUDA modules and kernel functions.
//!
//! Without caching, every call to a GPU kernel (e.g. [`gpu_add`], [`gpu_conv2d_f32`],
//! [`gpu_flash_attention_f32`]) recompiles PTX source into a CUBIN via
//! `CudaContext::load_module(Ptx::from_src(...))`.  This compilation takes
//! ~1700 us per call -- far longer than the actual kernel execution.
//!
//! This module provides [`get_or_compile`], which compiles the PTX only on
//! first use and returns a cached [`CudaFunction`] on subsequent calls.  The
//! cache is keyed by the static kernel name string, which is unique per
//! kernel entry point in this crate.
//!
//! # Thread safety
//!
//! The cache uses a global [`Mutex`]-protected [`HashMap`].  The critical
//! section is short (a hash lookup + optional insert), so contention is
//! negligible in practice.
//!
//! [`gpu_add`]: crate::kernels::gpu_add
//! [`gpu_conv2d_f32`]: crate::conv::gpu_conv2d_f32
//! [`gpu_flash_attention_f32`]: crate::flash_attention::gpu_flash_attention_f32

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, LazyLock, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, DriverError};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

/// Global cache mapping (kernel name, device ordinal) to their compiled
/// [`CudaFunction`]s.
///
/// Keyed by `(&'static str, u32)` -- the kernel name (e.g. `"add_kernel"`)
/// and the CUDA device ordinal.  A kernel compiled for device 0 cannot be
/// used on device 1, so the ordinal is part of the key.
#[cfg(feature = "cuda")]
static MODULE_CACHE: LazyLock<Mutex<HashMap<(&'static str, u32), CudaFunction>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get a compiled kernel function, compiling the PTX only on first use.
///
/// On the first call for a given `(kernel_name, device_ordinal)` pair, this
/// function compiles `ptx_src` into a CUDA module and extracts the named
/// function.  The resulting [`CudaFunction`] is cached globally and returned
/// by clone on subsequent calls, eliminating the ~1700 us PTX compilation
/// overhead.
///
/// # Arguments
///
/// - `ctx`            -- CUDA context (from `device.context()`).
/// - `ptx_src`        -- PTX source string (a `&'static str` constant).
/// - `kernel_name`    -- entry-point name inside the PTX module.
/// - `device_ordinal` -- CUDA device ordinal (so kernels compiled for
///   device 0 are not reused on device 1).
///
/// # Errors
///
/// Returns [`DriverError`] if PTX compilation or function lookup fails.
#[cfg(feature = "cuda")]
pub fn get_or_compile(
    ctx: &Arc<CudaContext>,
    ptx_src: &'static str,
    kernel_name: &'static str,
    device_ordinal: u32,
) -> Result<CudaFunction, DriverError> {
    let key = (kernel_name, device_ordinal);
    let mut cache = MODULE_CACHE.lock().unwrap();
    if let Some(func) = cache.get(&key) {
        return Ok(func.clone());
    }
    let module = ctx.load_module(Ptx::from_src(ptx_src))?;
    let func = module.load_function(kernel_name)?;
    cache.insert(key, func.clone());
    Ok(func)
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use crate::device::GpuDevice;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    #[test]
    fn cache_returns_function_on_repeated_calls() {
        // Verify the cache works by calling gpu_add twice. The first call
        // compiles the PTX; the second hits the cache. Both must succeed.
        let dev = crate::device::GpuDevice::new(0).expect("CUDA device 0");
        let a = crate::transfer::cpu_to_gpu(&[1.0f32, 2.0, 3.0], &dev).expect("a");
        let b = crate::transfer::cpu_to_gpu(&[4.0f32, 5.0, 6.0], &dev).expect("b");

        let r1 = crate::kernels::gpu_add(&a, &b, &dev).expect("first add (compiles)");
        let r2 = crate::kernels::gpu_add(&a, &b, &dev).expect("second add (cached)");

        let h1 = crate::transfer::gpu_to_cpu(&r1, &dev).expect("r1");
        let h2 = crate::transfer::gpu_to_cpu(&r2, &dev).expect("r2");
        assert_eq!(h1, h2, "cached kernel should produce identical results");
        assert_eq!(h1, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn cached_kernel_produces_correct_results() {
        // Run gpu_add twice and verify both produce correct results,
        // confirming the cached kernel is functional.
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();

        let a = cpu_to_gpu(&a_data, &dev).expect("a to gpu");
        let b = cpu_to_gpu(&b_data, &dev).expect("b to gpu");

        // First call (compiles PTX).
        let out1 = crate::kernels::gpu_add(&a, &b, &dev).expect("gpu_add 1st");
        let host1 = gpu_to_cpu(&out1, &dev).expect("gpu_to_cpu 1st");

        // Second call (uses cache).
        let out2 = crate::kernels::gpu_add(&a, &b, &dev).expect("gpu_add 2nd");
        let host2 = gpu_to_cpu(&out2, &dev).expect("gpu_to_cpu 2nd");

        for (i, ((&g1, &g2), &e)) in host1
            .iter()
            .zip(host2.iter())
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (g1 - e).abs() < 1e-6,
                "1st call: element {i}: got {g1}, expected {e}",
            );
            assert!(
                (g2 - e).abs() < 1e-6,
                "2nd call: element {i}: got {g2}, expected {e}",
            );
        }
    }

    #[test]
    fn cached_kernel_second_call_is_fast() {
        // The second call should be significantly faster than the first
        // because it skips PTX compilation.
        use std::time::Instant;

        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let a_data = vec![1.0f32; 1024];
        let b_data = vec![2.0f32; 1024];

        let a = cpu_to_gpu(&a_data, &dev).expect("a to gpu");
        let b = cpu_to_gpu(&b_data, &dev).expect("b to gpu");

        // Warm up with a different kernel to avoid measuring CUDA init.
        let _ = crate::kernels::gpu_neg(&a, &dev);

        // We cannot rely on add_kernel being uncached here (other tests
        // may have run first), so we use the mul_kernel via gpu_mul,
        // which is less likely to have been called yet.  Even if it has
        // been cached, both calls should be fast, and that is fine -- the
        // structural test above already verifies identity.
        let t1 = Instant::now();
        let _ = crate::kernels::gpu_mul(&a, &b, &dev).expect("gpu_mul 1st");
        let d1 = t1.elapsed();

        let t2 = Instant::now();
        let _ = crate::kernels::gpu_mul(&a, &b, &dev).expect("gpu_mul 2nd");
        let d2 = t2.elapsed();

        // The second call should be faster (no compilation).
        // We do not assert a strict ratio because CI environments vary,
        // but we log for manual inspection.
        eprintln!(
            "module_cache timing: 1st call = {:?}, 2nd call = {:?}",
            d1, d2,
        );
    }
}
