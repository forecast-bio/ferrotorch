//! Host-to-device and device-to-host memory transfers.
//!
//! These functions copy data between CPU (`&[T]` / `Vec<T>`) and GPU
//! ([`CudaBuffer`]) memory via the device's default CUDA stream.

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};

/// Copy a host slice to device memory, returning a new [`CudaBuffer`].
///
/// The transfer uses the device's default CUDA stream and blocks until
/// the copy is complete.
///
/// # Errors
///
/// Returns [`GpuError::Driver`] if the CUDA memcpy fails.
#[cfg(feature = "cuda")]
pub fn cpu_to_gpu<T>(data: &[T], device: &GpuDevice) -> GpuResult<CudaBuffer<T>>
where
    T: cudarc::driver::DeviceRepr,
{
    let slice = device.stream().clone_htod(data)?;
    Ok(CudaBuffer {
        data: Some(slice),
        len: data.len(),
        alloc_len: data.len(),
        device_ordinal: device.ordinal(),
        pool_fn: None,
    })
}

/// Copy device memory back to a host `Vec<T>`.
///
/// # Errors
///
/// Returns [`GpuError::DeviceMismatch`] if the buffer's device ordinal does
/// not match the provided device, or [`GpuError::Driver`] on CUDA errors.
#[cfg(feature = "cuda")]
pub fn gpu_to_cpu<T>(buffer: &CudaBuffer<T>, device: &GpuDevice) -> GpuResult<Vec<T>>
where
    T: cudarc::driver::DeviceRepr,
{
    if buffer.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: buffer.device_ordinal(),
            got: device.ordinal(),
        });
    }
    let mut vec = device.stream().clone_dtoh(buffer.inner())?;
    // When the allocation is rounded up (pooled buffers), the CudaSlice
    // contains more elements than the logical `len`. Truncate to the
    // logical length so callers only see the meaningful data.
    vec.truncate(buffer.len());
    Ok(vec)
}

/// Allocate a zero-initialized [`CudaBuffer<f32>`] on the given device.
///
/// Checks the global buffer pool first. On a pool hit, the existing
/// `CudaSlice` (with its CUDA events) is reused and only `cuMemsetD8Async`
/// is called. On a miss, a fresh allocation is made via cudarc with the
/// rounded length so the buffer is findable in the pool on subsequent lookups.
///
/// `memset_zeros` is called on the full `alloc_len` (rounded) allocation,
/// not just the logical `len`. This is intentional: it ensures no stale
/// data from previous uses leaks into the padding region.
#[cfg(feature = "cuda")]
pub fn alloc_zeros_f32(len: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::CudaSlice;

    let rounded = crate::pool::round_len(len);

    // Pool hit: reuse a cached CudaSlice — no cuMemAllocAsync, no cuEventCreate.
    if let Some(mut slice) = crate::pool::pool_take::<CudaSlice<f32>>(device.ordinal(), rounded, 4)
    {
        // Zero the full allocation to ensure no stale data (P10: intentional).
        device.stream().memset_zeros(&mut slice)?;
        return Ok(CudaBuffer::<f32>::new_pooled(
            slice,
            len,
            rounded,
            device.ordinal(),
        ));
    }

    // Pool miss: fresh allocation from CUDA driver with rounded length
    // so the pool key matches on return. Allocating `rounded` elements
    // (not `len`) ensures the CudaSlice size matches what pool_take
    // will look for later (B12 fix).
    let slice = device.stream().alloc_zeros::<f32>(rounded)?;
    Ok(CudaBuffer::<f32>::new_pooled(
        slice,
        len,
        rounded,
        device.ordinal(),
    ))
}

/// Allocate a zero-initialized [`CudaBuffer<f64>`] on the given device.
///
/// Pool-aware variant for f64 buffers. See [`alloc_zeros_f32`] for details.
#[cfg(feature = "cuda")]
pub fn alloc_zeros_f64(len: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::CudaSlice;

    let rounded = crate::pool::round_len(len);

    if let Some(mut slice) = crate::pool::pool_take::<CudaSlice<f64>>(device.ordinal(), rounded, 8)
    {
        device.stream().memset_zeros(&mut slice)?;
        return Ok(CudaBuffer::<f64>::new_pooled(
            slice,
            len,
            rounded,
            device.ordinal(),
        ));
    }

    let slice = device.stream().alloc_zeros::<f64>(rounded)?;
    Ok(CudaBuffer::<f64>::new_pooled(
        slice,
        len,
        rounded,
        device.ordinal(),
    ))
}

/// Generic alloc_zeros — kept for backward compatibility and non-f32/f64 types.
/// Does NOT use the pool (no pool support for arbitrary T).
#[cfg(feature = "cuda")]
pub fn alloc_zeros<T>(len: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<T>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    let slice = device.stream().alloc_zeros::<T>(len)?;
    Ok(CudaBuffer {
        data: Some(slice),
        len,
        alloc_len: len,
        device_ordinal: device.ordinal(),
        pool_fn: None,
    })
}

/// Copy a host slice to device memory via pinned (page-locked) host memory.
///
/// Allocates a temporary `PinnedHostSlice`, copies `data` into it, then
/// transfers to the GPU using DMA. The pinned allocation is freed after
/// the transfer completes. For large tensors, this is ~2x faster than
/// [`cpu_to_gpu`] which uses pageable memory.
///
/// # When to use
///
/// Use this in DataLoader's prefetch pipeline when `pin_memory=True`.
/// For small tensors (< 64KB), the overhead of pinned allocation may
/// outweigh the DMA benefit — prefer [`cpu_to_gpu`] instead.
#[cfg(feature = "cuda")]
pub fn cpu_to_gpu_pinned<T>(data: &[T], device: &GpuDevice) -> GpuResult<CudaBuffer<T>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy,
{
    let ctx = device.context();
    let stream = device.stream();

    // Allocate pinned host memory and copy data into it.
    // SAFETY:
    // - `CudaContext::alloc_pinned` (cudarc 0.19.4 src/driver/safe/core.rs:1346)
    //   is `unsafe` because the returned `PinnedHostSlice<T>` contains
    //   uninitialized memory after `cuMemAllocHost`/`cuMemHostAlloc`
    //   (CUDA driver API). The caller's obligation is to fully initialize
    //   the slice before any read.
    // - We initialize every element on the very next line via
    //   `pinned.as_mut_slice()?.copy_from_slice(data)`. The `copy_from_slice`
    //   contract requires source and destination to have equal lengths;
    //   `pinned` was allocated with `data.len()` elements (line 167), so the
    //   length precondition holds and every element of the pinned region
    //   is overwritten before the subsequent `clone_htod` read on line 171.
    // - `T: DeviceRepr + ValidAsZeroBits + Copy` (function bound on line
    //   161-162) guarantees the bit layout is suitable for both pinned host
    //   memory and DMA transfer to device.
    // - `ctx` is a valid `Arc<CudaContext>` obtained from `device.context()`
    //   on line 163; cudarc upholds the bind-to-thread invariant inside
    //   `alloc_pinned` (line 1350 of upstream).
    // - Lifetime: `pinned` is owned by this stack frame and explicitly
    //   `drop`-ped on line 174 after `clone_htod` consumes it as `&pinned`,
    //   so the pinned allocation outlives every read.
    let mut pinned = unsafe { ctx.alloc_pinned::<T>(data.len())? };
    pinned.as_mut_slice()?.copy_from_slice(data);

    // Transfer from pinned host to device (uses DMA, ~2x faster than pageable).
    let slice = stream.clone_htod(&pinned)?;

    // pinned is dropped here, freeing the host memory.
    drop(pinned);

    Ok(CudaBuffer {
        data: Some(slice),
        len: data.len(),
        alloc_len: data.len(),
        device_ordinal: device.ordinal(),
        pool_fn: None,
    })
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn cpu_to_gpu_pinned<T>(_data: &[T], _device: &GpuDevice) -> GpuResult<CudaBuffer<T>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn cpu_to_gpu<T>(_data: &[T], _device: &GpuDevice) -> GpuResult<CudaBuffer<T>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_to_cpu<T>(_buffer: &CudaBuffer<T>, _device: &GpuDevice) -> GpuResult<Vec<T>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn alloc_zeros<T>(_len: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<T>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn alloc_zeros_f32(_len: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn alloc_zeros_f64(_len: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests — require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn round_trip_f32() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), 5);
        assert_eq!(gpu_buf.device_ordinal(), 0);

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert_eq!(back, host);
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary round-trip value, not π.
    fn round_trip_f64() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f64> = vec![1.0, -2.5, 3.14, 0.0, f64::MAX];

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), 5);

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert_eq!(back, host);
    }

    #[test]
    fn alloc_zeros_f32_basic() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let buf = alloc_zeros_f32(1024, &device).expect("alloc_zeros_f32");
        assert_eq!(buf.len(), 1024);
        assert!(buf.pool_fn.is_some());

        let host = gpu_to_cpu(&buf, &device).expect("gpu_to_cpu");
        assert!(host.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pool_reuse_f32() {
        let device = GpuDevice::new(0).expect("CUDA device 0");

        // Allocate and drop — should go to pool.
        let buf = alloc_zeros_f32(512, &device).expect("alloc 1");
        assert!(buf.pool_fn.is_some());
        drop(buf);

        assert!(crate::pool::cached_bytes(0) > 0);

        // Second allocation of same size — should hit pool.
        let buf2 = alloc_zeros_f32(512, &device).expect("alloc 2");
        assert!(buf2.pool_fn.is_some());

        let host = gpu_to_cpu(&buf2, &device).expect("gpu_to_cpu");
        assert!(
            host.iter().all(|&x| x == 0.0),
            "pooled buffer must be zeroed"
        );
    }

    #[test]
    fn empty_cache_clears_pool() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let buf = alloc_zeros_f32(256, &device).expect("alloc");
        drop(buf);
        assert!(crate::pool::cached_bytes(0) > 0);

        crate::pool::empty_cache(0);
        assert_eq!(crate::pool::cached_bytes(0), 0);
    }

    #[test]
    fn alloc_zeros_generic() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let buf = alloc_zeros::<f32>(1024, &device).expect("alloc_zeros");
        assert_eq!(buf.len(), 1024);

        let host = gpu_to_cpu(&buf, &device).expect("gpu_to_cpu");
        assert!(host.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn empty_transfer() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f32> = vec![];

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), 0);
        assert!(gpu_buf.is_empty());

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert!(back.is_empty());
    }

    #[test]
    fn large_transfer() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let n = 1_000_000;
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), n);

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert_eq!(back, host);
    }

    #[test]
    fn device_mismatch_rejected() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f32> = vec![1.0];
        let mut buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        buf.device_ordinal = 99;

        let err = gpu_to_cpu(&buf, &device).unwrap_err();
        match err {
            GpuError::DeviceMismatch {
                expected: 99,
                got: 0,
            } => {}
            other => panic!("unexpected error: {other}"),
        }
    }
}
