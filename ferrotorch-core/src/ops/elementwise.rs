//! Elementwise tensor operations.
//!
//! Uses ferray-ufunc SIMD kernels for f32/f64 fast paths and falls back
//! to scalar loops for generic/broadcasting operations.
//!
//! For tensors above `PARALLEL_THRESHOLD` elements, work is split across
//! rayon worker threads so each chunk is still processed by the SIMD kernel.

use crate::cpu_pool::{pool_alloc_cpu_uninit_f32, pool_alloc_cpu_uninit_f64};
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::broadcast_shapes;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// Minimum number of elements before switching to rayon parallelism.
/// 1M f32s = 4 MiB — below this the per-element SIMD kernel is fast enough
/// that rayon's work-stealing overhead dominates. At 1M+ elements the memory
/// bandwidth saturates a single core and parallelism helps.
/// Minimum number of elements before splitting work across rayon threads.
///
/// PyTorch uses 32K as the grain size (at::internal::GRAIN_SIZE). The
/// previous value of 2M left tensors with 32K–2M elements single-threaded,
/// which is a significant missed parallelism window for typical NN layer
/// sizes (embedding dims, hidden states, etc.).
const PARALLEL_THRESHOLD: usize = 32_768;

// --- SIMD-accelerated specializations for f32 ---

/// SIMD-accelerated add for same-shape f32 tensors.
pub fn simd_add_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut output = vec![0.0f32; a_data.len()];
    ferray_ufunc::kernels::simd_f32::add_f32(a_data, b_data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), a.shape().to_vec(), false)
}

/// SIMD-accelerated mul for same-shape f32 tensors.
pub fn simd_mul_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut output = vec![0.0f32; a_data.len()];
    ferray_ufunc::kernels::simd_f32::mul_f32(a_data, b_data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), a.shape().to_vec(), false)
}

/// SIMD-accelerated exp for f32.
pub fn simd_exp_f32(input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let data = input.data()?;
    let mut output = vec![0.0f32; data.len()];
    ferray_ufunc::kernels::simd_f32::exp_f32(data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), input.shape().to_vec(), false)
}

/// SIMD-accelerated log for f32.
pub fn simd_log_f32(input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let data = input.data()?;
    let mut output = vec![0.0f32; data.len()];
    ferray_ufunc::kernels::simd_f32::log_f32(data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), input.shape().to_vec(), false)
}

/// SIMD-accelerated sqrt for f32.
pub fn simd_sqrt_f32(input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let data = input.data()?;
    let mut output = vec![0.0f32; data.len()];
    ferray_ufunc::kernels::simd_f32::sqrt_f32(data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), input.shape().to_vec(), false)
}

// --- SIMD-accelerated specializations for f64 ---

/// SIMD-accelerated add for same-shape f64 tensors.
pub fn simd_add_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> FerrotorchResult<Tensor<f64>> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut output = vec![0.0f64; a_data.len()];
    ferray_ufunc::kernels::simd_f64::add_f64(a_data, b_data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), a.shape().to_vec(), false)
}

/// SIMD-accelerated mul for same-shape f64 tensors.
pub fn simd_mul_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> FerrotorchResult<Tensor<f64>> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut output = vec![0.0f64; a_data.len()];
    ferray_ufunc::kernels::simd_f64::mul_f64(a_data, b_data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), a.shape().to_vec(), false)
}

/// SIMD-accelerated exp for f64.
pub fn simd_exp_f64(input: &Tensor<f64>) -> FerrotorchResult<Tensor<f64>> {
    let data = input.data()?;
    let mut output = vec![0.0f64; data.len()];
    ferray_ufunc::kernels::simd_f64::exp_f64(data, &mut output);
    Tensor::from_storage(TensorStorage::cpu(output), input.shape().to_vec(), false)
}

// --- SIMD-dispatching generic wrappers ---

/// Transmute a Vec<f32> to Vec<T> (zero-cost when T is f32).
///
/// SAFETY: Only call when size_of::<T>() == size_of::<f32>() (i.e., T is f32).
#[inline]
unsafe fn transmute_vec_f32_to_t<T: Float>(v: Vec<f32>) -> Vec<T> {
    let mut v = std::mem::ManuallyDrop::new(v);
    // SAFETY: caller's contract guarantees T == f32 (size_of::<T>() == 4),
    // so f32 and T have identical layout (size, align, niches). The Vec was
    // placed in ManuallyDrop on the line above, so the pointer/len/capacity
    // we pass here are no longer owned by any other Vec — reconstructing a
    // Vec<T> from them transfers ownership to the new Vec without double-free.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast::<T>(), v.len(), v.capacity()) }
}

/// Transmute a Vec<f64> to Vec<T> (zero-cost when T is f64).
#[inline]
unsafe fn transmute_vec_f64_to_t<T: Float>(v: Vec<f64>) -> Vec<T> {
    let mut v = std::mem::ManuallyDrop::new(v);
    // SAFETY: caller's contract guarantees T == f64 (size_of::<T>() == 8),
    // so f64 and T have identical layout (size, align, niches). The Vec was
    // placed in ManuallyDrop on the line above, so the pointer/len/capacity
    // we pass here are no longer owned by any other Vec — reconstructing a
    // Vec<T> from them transfers ownership to the new Vec without double-free.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast::<T>(), v.len(), v.capacity()) }
}

/// SIMD-accelerated add: dispatches to f32/f64 SIMD for same-shape tensors,
/// falls back to generic binary_map with broadcasting.
///
/// For tensors >= `PARALLEL_THRESHOLD` elements the work is split across
/// rayon threads, each chunk processed by the SIMD kernel.
pub fn fast_add<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.shape() == b.shape() {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let n = a_data.len();
        if std::mem::size_of::<T>() == 4 {
            // SAFETY: size_of::<T>() == 4 holds in this branch. `Float` is
            // bounded to f32/f64/bf16/f16 in this crate; only f32 has size 4,
            // so T == f32 and the cast preserves alignment, length, and
            // initialization. `a_data` is borrowed for the duration of `n`
            // elements, so the resulting slice does not outlive its source.
            let a_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f32>(), n) };
            // SAFETY: same invariant as a_f32 above (T == f32 by size guard).
            let b_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f32>(), n) };
            let mut out = pool_alloc_cpu_uninit_f32(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        ferray_ufunc::kernels::simd_f32::add_f32(
                            &a_f32[offset..offset + len],
                            &b_f32[offset..offset + len],
                            chunk,
                        );
                    });
            } else {
                ferray_ufunc::kernels::simd_f32::add_f32(a_f32, b_f32, &mut out);
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            let result = unsafe { transmute_vec_f32_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        } else if std::mem::size_of::<T>() == 8 {
            // SAFETY: size_of::<T>() == 8 holds in this branch. Among the
            // Float-bounded types in this crate (f32/f64/bf16/f16), only f64
            // has size 8, so T == f64 and the cast preserves layout. `a_data`
            // is borrowed for `n` elements; the slice does not outlive it.
            let a_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f64>(), n) };
            // SAFETY: same invariant as a_f64 above (T == f64 by size guard).
            let b_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f64>(), n) };
            let mut out = pool_alloc_cpu_uninit_f64(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        ferray_ufunc::kernels::simd_f64::add_f64(
                            &a_f64[offset..offset + len],
                            &b_f64[offset..offset + len],
                            chunk,
                        );
                    });
            } else {
                ferray_ufunc::kernels::simd_f64::add_f64(a_f64, b_f64, &mut out);
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 8, so
            // T == f64 (only Float impl with that size); satisfies
            // transmute_vec_f64_to_t's documented contract.
            let result = unsafe { transmute_vec_f64_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        }
    }
    binary_map(a, b, |x, y| x + y)
}

/// SIMD-accelerated mul with rayon parallelism for large tensors.
pub fn fast_mul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.shape() == b.shape() {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let n = a_data.len();
        if std::mem::size_of::<T>() == 4 {
            // SAFETY: enclosing branch asserts size_of::<T>() == 4. The
            // crate-internal `Float` trait is implemented for f32/f64/bf16/f16;
            // only f32 has size 4, so T == f32 and the cast preserves layout
            // and alignment. `a_data` outlives this slice (it borrows a_data's
            // buffer for n elements, all initialized as part of a Tensor<T>).
            let a_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f32>(), n) };
            // SAFETY: identical reasoning to a_f32 above (T == f32 by size_of guard).
            let b_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f32>(), n) };
            let mut out = pool_alloc_cpu_uninit_f32(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        ferray_ufunc::kernels::simd_f32::mul_f32(
                            &a_f32[offset..offset + len],
                            &b_f32[offset..offset + len],
                            chunk,
                        );
                    });
            } else {
                ferray_ufunc::kernels::simd_f32::mul_f32(a_f32, b_f32, &mut out);
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            let result = unsafe { transmute_vec_f32_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        } else if std::mem::size_of::<T>() == 8 {
            // SAFETY: enclosing branch asserts size_of::<T>() == 8. The
            // crate-internal `Float` trait is implemented for f32/f64/bf16/f16;
            // only f64 has size 8, so T == f64 and the cast preserves layout
            // and alignment. `a_data` outlives this slice (n initialized
            // f64-sized elements borrowed from a_data's buffer).
            let a_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f64>(), n) };
            // SAFETY: identical reasoning to a_f64 above (T == f64 by size_of guard).
            let b_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f64>(), n) };
            let mut out = pool_alloc_cpu_uninit_f64(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        ferray_ufunc::kernels::simd_f64::mul_f64(
                            &a_f64[offset..offset + len],
                            &b_f64[offset..offset + len],
                            chunk,
                        );
                    });
            } else {
                ferray_ufunc::kernels::simd_f64::mul_f64(a_f64, b_f64, &mut out);
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 8, so
            // T == f64 (only Float impl with that size); satisfies
            // transmute_vec_f64_to_t's documented contract.
            let result = unsafe { transmute_vec_f64_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        }
    }
    binary_map(a, b, |x, y| x * y)
}

/// SIMD-accelerated sub with rayon parallelism for large tensors.
///
/// Same-shape fast path uses a simple vectorizable loop that LLVM
/// auto-vectorizes to SIMD. Falls back to `binary_map` for broadcasting.
pub fn fast_sub<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.shape() == b.shape() {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let n = a_data.len();
        if std::mem::size_of::<T>() == 4 {
            // SAFETY: enclosing branch asserts size_of::<T>() == 4. The
            // crate-internal `Float` trait is implemented for f32/f64/bf16/f16;
            // only f32 has size 4, so T == f32 and the cast preserves layout
            // and alignment. `a_data` outlives this slice (it borrows a_data's
            // buffer for n elements, all initialized as part of a Tensor<T>).
            let a_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f32>(), n) };
            // SAFETY: identical reasoning to a_f32 above (T == f32 by size_of guard).
            let b_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f32>(), n) };
            let mut out = pool_alloc_cpu_uninit_f32(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        let a_s = &a_f32[offset..offset + len];
                        let b_s = &b_f32[offset..offset + len];
                        for i in 0..len {
                            chunk[i] = a_s[i] - b_s[i];
                        }
                    });
            } else {
                for i in 0..n {
                    out[i] = a_f32[i] - b_f32[i];
                }
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            let result = unsafe { transmute_vec_f32_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        } else if std::mem::size_of::<T>() == 8 {
            // SAFETY: enclosing branch asserts size_of::<T>() == 8. The
            // crate-internal `Float` trait is implemented for f32/f64/bf16/f16;
            // only f64 has size 8, so T == f64 and the cast preserves layout
            // and alignment. `a_data` outlives this slice (n initialized
            // f64-sized elements borrowed from a_data's buffer).
            let a_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f64>(), n) };
            // SAFETY: identical reasoning to a_f64 above (T == f64 by size_of guard).
            let b_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f64>(), n) };
            let mut out = pool_alloc_cpu_uninit_f64(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        let a_s = &a_f64[offset..offset + len];
                        let b_s = &b_f64[offset..offset + len];
                        for i in 0..len {
                            chunk[i] = a_s[i] - b_s[i];
                        }
                    });
            } else {
                for i in 0..n {
                    out[i] = a_f64[i] - b_f64[i];
                }
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 8, so
            // T == f64 (only Float impl with that size); satisfies
            // transmute_vec_f64_to_t's documented contract.
            let result = unsafe { transmute_vec_f64_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        }
    }
    binary_map(a, b, |x, y| x - y)
}

/// SIMD-accelerated div with rayon parallelism for large tensors.
pub fn fast_div<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.shape() == b.shape() {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let n = a_data.len();
        if std::mem::size_of::<T>() == 4 {
            // SAFETY: enclosing branch asserts size_of::<T>() == 4. The
            // crate-internal `Float` trait is implemented for f32/f64/bf16/f16;
            // only f32 has size 4, so T == f32 and the cast preserves layout
            // and alignment. `a_data` outlives this slice (it borrows a_data's
            // buffer for n elements, all initialized as part of a Tensor<T>).
            let a_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f32>(), n) };
            // SAFETY: identical reasoning to a_f32 above (T == f32 by size_of guard).
            let b_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f32>(), n) };
            let mut out = pool_alloc_cpu_uninit_f32(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        let a_s = &a_f32[offset..offset + len];
                        let b_s = &b_f32[offset..offset + len];
                        for i in 0..len {
                            chunk[i] = a_s[i] / b_s[i];
                        }
                    });
            } else {
                for i in 0..n {
                    out[i] = a_f32[i] / b_f32[i];
                }
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
            // T == f32 (only Float impl with that size); satisfies
            // transmute_vec_f32_to_t's documented contract.
            let result = unsafe { transmute_vec_f32_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        } else if std::mem::size_of::<T>() == 8 {
            // SAFETY: enclosing branch asserts size_of::<T>() == 8. The
            // crate-internal `Float` trait is implemented for f32/f64/bf16/f16;
            // only f64 has size 8, so T == f64 and the cast preserves layout
            // and alignment. `a_data` outlives this slice (n initialized
            // f64-sized elements borrowed from a_data's buffer).
            let a_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f64>(), n) };
            // SAFETY: identical reasoning to a_f64 above (T == f64 by size_of guard).
            let b_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f64>(), n) };
            let mut out = pool_alloc_cpu_uninit_f64(n);
            if n >= PARALLEL_THRESHOLD {
                let chunk_size = (n / rayon::current_num_threads()).max(4096);
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * chunk_size;
                        let len = chunk.len();
                        let a_s = &a_f64[offset..offset + len];
                        let b_s = &b_f64[offset..offset + len];
                        for i in 0..len {
                            chunk[i] = a_s[i] / b_s[i];
                        }
                    });
            } else {
                for i in 0..n {
                    out[i] = a_f64[i] / b_f64[i];
                }
            }
            // SAFETY: enclosing branch guards on size_of::<T>() == 8, so
            // T == f64 (only Float impl with that size); satisfies
            // transmute_vec_f64_to_t's documented contract.
            let result = unsafe { transmute_vec_f64_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        }
    }
    binary_map(a, b, |x, y| x / y)
}

/// Vectorizable f32 exp kernel — polynomial range reduction that LLVM
/// auto-vectorizes to vexpps (AVX2) or equivalent SIMD.
///
/// Algorithm: exp(x) = 2^n * exp(r) where n = round(x / ln2), r = x - n*ln2.
/// The reduced exp(r) is evaluated via a degree-5 minimax polynomial.
#[inline(always)]
fn vexp_f32(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    // ln(2) Cody-Waite decomposition: LN2_HI + LN2_LO ≈ ln(2) with extra
    // precision needed by the range reduction `x - n*ln2`.
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;

    // Clamp to avoid overflow/underflow in the integer exponent.
    let x = x.clamp(-87.33654, 88.72284);

    // Range reduction: n = round(x * log2e), r = x - n * ln2.
    let n = (x * LOG2E).round();
    let r = x - n * LN2_HI - n * LN2_LO;

    // Degree-5 minimax polynomial for exp(r) on [-ln2/2, ln2/2].
    let p =
        1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * 0.008_333_334))));

    // Scale by 2^n via bit manipulation (branchless).
    let n_i32 = n as i32;
    let scale = f32::from_bits(((127 + n_i32) as u32) << 23);
    p * scale
}

/// Vectorizable f32 ln kernel — polynomial approximation.
///
/// Algorithm: ln(x) = (e-127)*ln2 + ln(m) where x = m * 2^e.
/// Normalizes mantissa to [sqrt(2)/2, sqrt(2)) for better polynomial
/// conditioning, then evaluates a degree-7 minimax polynomial for
/// ln((1+s)/(1-s)) where s = (m-1)/(m+1).
///
/// The polynomial body is only valid for finite positive inputs. For NaN, ±∞,
/// zero, or negatives, delegate to [`fast_log_f32`] which returns the IEEE-754
/// correct value via `f32::ln()`. Without this guard, +∞ → ~88.7 (the
/// polynomial's exponent path treats it as a finite number with biased
/// exponent 255), and NaN → arbitrary finite garbage (the bit-level mantissa
/// extraction produces a normal float that the polynomial happily evaluates).
#[inline(always)]
fn vlog_f32(x: f32) -> f32 {
    const LN2: f32 = std::f32::consts::LN_2;
    if !(x > 0.0 && x.is_finite()) {
        return fast_log_f32(x);
    }

    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xFF) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);

    // Normalize mantissa to [sqrt(2)/2, sqrt(2)) for better conditioning.
    if m > std::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }

    // Use the identity: ln(m) = 2*atanh(s) where s = (m-1)/(m+1).
    // atanh(s) = s + s^3/3 + s^5/5 + s^7/7 + ...
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    // Horner evaluation of 2*(s + s^3/3 + s^5/5 + s^7/7 + s^9/9)
    let p = s * (2.0 + s2 * (0.666_666_7 + s2 * (0.4 + s2 * (0.285_714_3 + s2 * 0.222_222_22))));

    (e as f32) * LN2 + p
}

/// SIMD-accelerated exp with rayon parallelism for large tensors.
///
/// Uses a vectorizable polynomial kernel (vexp_f32) that LLVM auto-vectorizes
/// to AVX2/SSE SIMD instructions, instead of scalar libm expf.
pub fn fast_exp<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let data = input.data()?;
    let n = data.len();
    if std::mem::size_of::<T>() == 4 {
        // SAFETY: enclosing branch asserts size_of::<T>() == 4. Among the
        // crate-internal `Float` impls (f32/f64/bf16/f16) only f32 has size 4,
        // so T == f32. `data` (a `Tensor<T>::data()` borrow) outlives this slice
        // and contains exactly `n` initialized T-sized elements.
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let len = chunk.len();
                    for (o, &x) in chunk.iter_mut().zip(inp[offset..offset + len].iter()) {
                        *o = vexp_f32(x);
                    }
                });
        } else {
            for (o, &x) in out.iter_mut().zip(inp.iter()) {
                *o = vexp_f32(x);
            }
        }
        // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
        // T == f32 (only Float impl with that size); satisfies
        // transmute_vec_f32_to_t's documented contract.
        let result = unsafe { transmute_vec_f32_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    }
    unary_map(&input, |x| x.exp())
}

/// SIMD-accelerated log with rayon parallelism for large tensors.
///
/// Uses a vectorizable polynomial kernel (vlog_f32) that LLVM auto-vectorizes.
pub fn fast_log<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let data = input.data()?;
    let n = data.len();
    if std::mem::size_of::<T>() == 4 {
        // SAFETY: enclosing branch asserts size_of::<T>() == 4. Among the
        // crate-internal `Float` impls (f32/f64/bf16/f16) only f32 has size 4,
        // so T == f32. `data` (a `Tensor<T>::data()` borrow) outlives this slice
        // and contains exactly `n` initialized T-sized elements.
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let len = chunk.len();
                    for (o, &x) in chunk.iter_mut().zip(inp[offset..offset + len].iter()) {
                        *o = vlog_f32(x);
                    }
                });
        } else {
            for (o, &x) in out.iter_mut().zip(inp.iter()) {
                *o = vlog_f32(x);
            }
        }
        // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
        // T == f32 (only Float impl with that size); satisfies
        // transmute_vec_f32_to_t's documented contract.
        let result = unsafe { transmute_vec_f32_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    }
    unary_map(&input, |x| x.ln())
}

/// Fast f32 exp — single-element, fused, auto-vectorization friendly.
///
/// Guards special values (NaN, +inf, -inf) before delegating to `f32::exp()`,
/// which LLVM vectorizes to vexpps (AVX2) or equivalent when compiled with
/// `target-cpu=native`. The explicit checks use bitwise NaN detection
/// (`x != x`) that auto-vectorizes cleanly, and tightened clamp bounds
/// keep the internal exponent n in [-126, 127] to avoid bit-manipulation
/// overflow that would produce garbage at n=128.
#[allow(clippy::inline_always)] // reason: hot inner-loop scalar; must inline into SIMD-vectorized unary_map
#[inline(always)]
fn fast_exp_f32(x: f32) -> f32 {
    // Guard special values BEFORE clamping (auto-vectorizes via is_nan)
    if x.is_nan() {
        return f32::NAN;
    } // NaN passthrough
    if x > 88.72284 {
        return f32::INFINITY;
    } // overflow → +inf
    if x < -87.33654 {
        return 0.0;
    } // underflow → 0
    // Clamp to keep internal integer exponent n <= 127 (avoids n=128 UB)
    let x_clamped = x.min(88.0);
    x_clamped.exp()
}

/// Edge-case-correct scalar f32 log.
///
/// Used by [`vlog_f32`] as the slow-path fallback for non-finite or
/// non-positive inputs. The polynomial in `vlog_f32` produces wrong values
/// for NaN and `+∞`; this function delegates to `f32::ln()` (libm-backed)
/// after explicit guards so each special case maps to its IEEE-754 result:
/// `0.0 → −∞`, negatives → `NaN`, `NaN → NaN`, `+∞ → +∞`.
#[allow(clippy::inline_always)] // reason: hot inner-loop scalar; must inline into SIMD-vectorized unary_map
#[inline(always)]
fn fast_log_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return if x == 0.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }
    if x.is_nan() {
        return f32::NAN;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    x.ln()
}

/// Fused single-pass sigmoid: `1 / (1 + exp(-x))`.
///
/// No intermediate allocations — each element is computed in registers.
/// With `target-cpu=native`, the inner loop auto-vectorizes to AVX2 (8-wide).
/// For large tensors (>= PARALLEL_THRESHOLD), work is split across rayon threads.
pub fn fast_sigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let n = data.len();
    if std::mem::size_of::<T>() == 4 {
        // SAFETY: enclosing branch asserts size_of::<T>() == 4. Among the
        // crate-internal `Float` impls (f32/f64/bf16/f16) only f32 has size 4,
        // so T == f32. `data` (a `Tensor<T>::data()` borrow) outlives this slice
        // and contains exactly `n` initialized T-sized elements.
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let slice = &inp[offset..offset + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = 1.0 / (1.0 + fast_exp_f32(-slice[i]));
                    }
                });
        } else {
            for i in 0..n {
                out[i] = 1.0 / (1.0 + fast_exp_f32(-inp[i]));
            }
        }
        // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
        // T == f32 (only Float impl with that size); satisfies
        // transmute_vec_f32_to_t's documented contract.
        let result = unsafe { transmute_vec_f32_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    }
    let one = <T as num_traits::One>::one();
    unary_map(input, move |x| one / (one + (-x).exp()))
}

/// Fused single-pass tanh: `(exp(2x) - 1) / (exp(2x) + 1)`.
///
/// No intermediate allocations — each element computed in registers.
pub fn fast_tanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let n = data.len();
    if std::mem::size_of::<T>() == 4 {
        // SAFETY: enclosing branch asserts size_of::<T>() == 4. Among the
        // crate-internal `Float` impls (f32/f64/bf16/f16) only f32 has size 4,
        // so T == f32. `data` (a `Tensor<T>::data()` borrow) outlives this slice
        // and contains exactly `n` initialized T-sized elements.
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let slice = &inp[offset..offset + chunk.len()];
                    for i in 0..chunk.len() {
                        let x_clamped = slice[i].clamp(-9.0, 9.0); // tanh(9) ≈ 1 - 1.6e-8
                        let e2x = fast_exp_f32(2.0 * x_clamped);
                        chunk[i] = (e2x - 1.0) / (e2x + 1.0);
                    }
                });
        } else {
            for i in 0..n {
                let x_clamped = inp[i].clamp(-9.0, 9.0); // tanh(9) ≈ 1 - 1.6e-8
                let e2x = fast_exp_f32(2.0 * x_clamped);
                out[i] = (e2x - 1.0) / (e2x + 1.0);
            }
        }
        // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
        // T == f32 (only Float impl with that size); satisfies
        // transmute_vec_f32_to_t's documented contract.
        let result = unsafe { transmute_vec_f32_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    }
    unary_map(input, |x| x.tanh())
}

/// Fused single-pass sin for f32.
///
/// Delegates to `f32::sin()` (libm / LLVM intrinsic), which uses Cody-Waite
/// range reduction for moderate inputs and Payne-Hanek for |x| > ~10^5.
/// The Cody-Waite reduction is accurate to ~1 ULP for |x| < ~10^5; beyond
/// that range the slower Payne-Hanek path kicks in automatically via libm.
/// LLVM may auto-vectorize the inner loop via SLEEF-style lowering when
/// compiled with `target-cpu=native`.
pub fn fast_sin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let n = data.len();
    if std::mem::size_of::<T>() == 4 {
        // SAFETY: enclosing branch asserts size_of::<T>() == 4. Among the
        // crate-internal `Float` impls (f32/f64/bf16/f16) only f32 has size 4,
        // so T == f32. `data` (a `Tensor<T>::data()` borrow) outlives this slice
        // and contains exactly `n` initialized T-sized elements.
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let slice = &inp[offset..offset + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = slice[i].sin();
                    }
                });
        } else {
            for i in 0..n {
                out[i] = inp[i].sin();
            }
        }
        // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
        // T == f32 (only Float impl with that size); satisfies
        // transmute_vec_f32_to_t's documented contract.
        let result = unsafe { transmute_vec_f32_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    }
    unary_map(input, |x| x.sin())
}

/// Fused single-pass cos for f32.
///
/// Delegates to `f32::cos()` (libm / LLVM intrinsic), which uses Cody-Waite
/// range reduction for moderate inputs and Payne-Hanek for |x| > ~10^5.
/// See [`fast_sin`] for details on the reduction algorithm and accuracy.
pub fn fast_cos<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let n = data.len();
    if std::mem::size_of::<T>() == 4 {
        // SAFETY: enclosing branch asserts size_of::<T>() == 4. Among the
        // crate-internal `Float` impls (f32/f64/bf16/f16) only f32 has size 4,
        // so T == f32. `data` (a `Tensor<T>::data()` borrow) outlives this slice
        // and contains exactly `n` initialized T-sized elements.
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let slice = &inp[offset..offset + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = slice[i].cos();
                    }
                });
        } else {
            for i in 0..n {
                out[i] = inp[i].cos();
            }
        }
        // SAFETY: enclosing branch guards on size_of::<T>() == 4, so
        // T == f32 (only Float impl with that size); satisfies
        // transmute_vec_f32_to_t's documented contract.
        let result = unsafe { transmute_vec_f32_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    }
    unary_map(input, |x| x.cos())
}

// --- Generic fallback operations ---

/// Apply a unary function elementwise, producing a new tensor.
pub fn unary_map<T: Float>(input: &Tensor<T>, f: impl Fn(T) -> T) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    if device.is_cuda() {
        // GPU path: transfer to CPU, compute, transfer back.
        let data = input.data_vec()?;
        let result: Vec<T> = data.iter().map(|&x| f(x)).collect();
        let out = Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)?;
        out.to(device)
    } else {
        // CPU path: borrow directly — zero copy.
        let data = input.data()?;
        let result: Vec<T> = data.iter().map(|&x| f(x)).collect();
        Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)
    }
}

/// Apply a binary function elementwise on two tensors with broadcasting.
///
/// Uses stride-based iteration with an N-D counter (odometer pattern) to
/// avoid per-element modulo/division. The innermost dimension is iterated
/// as a tight loop when both inputs are contiguous there.
pub fn binary_map<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    f: impl Fn(T, T) -> T,
) -> FerrotorchResult<Tensor<T>> {
    // Materialize non-contiguous views for data() access.
    let a = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()?
    };
    let b = if b.is_contiguous() {
        b.clone()
    } else {
        b.contiguous()?
    };

    // Same-shape fast path.
    if a.shape() == b.shape() {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let result: Vec<T> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
    }

    // Broadcasting path with stride-based N-D counter (odometer pattern).
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let out_numel: usize = out_shape.iter().product();
    // strides: outermost-first, (a_stride, b_stride, out_dim)
    let strides = precompute_broadcast_strides(a.shape(), b.shape(), &out_shape);
    let ndim = strides.len();

    let a_data = a.data()?;
    let b_data = b.data()?;

    // Separate the innermost dimension for a tight inner loop.
    let inner_dim = strides[ndim - 1].2;
    let inner_a_stride = strides[ndim - 1].0;
    let inner_b_stride = strides[ndim - 1].1;
    let outer_count = out_numel / inner_dim.max(1);

    let mut result = Vec::with_capacity(out_numel);

    // N-D counter for the outer dimensions (indices 0..ndim-1).
    let mut coords = vec![0usize; ndim];
    let mut a_base = 0usize;
    let mut b_base = 0usize;

    for _ in 0..outer_count {
        // Inner loop: iterate over the innermost dimension.
        let mut a_idx = a_base;
        let mut b_idx = b_base;
        for _ in 0..inner_dim {
            result.push(f(a_data[a_idx], b_data[b_idx]));
            a_idx += inner_a_stride;
            b_idx += inner_b_stride;
        }

        // Increment the N-D counter (odometer carry, innermost outer dim first).
        // This is O(1) amortized — carry propagates rarely.
        for d in (0..ndim - 1).rev() {
            coords[d] += 1;
            let (a_s, b_s, dim_size) = strides[d];
            if coords[d] < dim_size {
                a_base += a_s;
                b_base += b_s;
                break;
            }
            // Carry: reset this dimension, adjust base indices.
            coords[d] = 0;
            a_base -= a_s * (dim_size - 1);
            b_base -= b_s * (dim_size - 1);
        }
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

/// Apply a binary function between a tensor and a scalar.
pub fn scalar_map<T: Float>(
    input: &Tensor<T>,
    scalar: T,
    f: impl Fn(T, T) -> T,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "scalar_map" });
    }
    let data = input.data()?;
    let result: Vec<T> = data.iter().map(|&x| f(x, scalar)).collect();
    Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)
}

/// Precompute per-dimension `(a_stride, b_stride, out_dim)` triples for a
/// broadcast between shapes `a` and `b` into `out_shape`.
///
/// For each dimension, the stride is 0 when the input has size 1 (broadcast),
/// otherwise it is the product of all trailing dimensions of that input.
/// The returned vector is in outermost-first order so the caller can iterate
/// from the innermost end with `.iter().rev()`.
fn precompute_broadcast_strides(
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<(usize, usize, usize)> {
    let ndim = out_shape.len();
    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    let mut strides = Vec::with_capacity(ndim);
    let mut a_stride: usize = 1;
    let mut b_stride: usize = 1;

    // Build from innermost to outermost, then reverse.
    for i in 0..ndim {
        let out_dim = out_shape[ndim - 1 - i];

        let a_dim = if i < a_ndim {
            a_shape[a_ndim - 1 - i]
        } else {
            1
        };
        let b_dim = if i < b_ndim {
            b_shape[b_ndim - 1 - i]
        } else {
            1
        };

        let a_s = if a_dim == 1 { 0 } else { a_stride };
        let b_s = if b_dim == 1 { 0 } else { b_stride };

        strides.push((a_s, b_s, out_dim));

        a_stride *= a_dim;
        b_stride *= b_dim;
    }

    strides.reverse();
    strides
}

// --- Reduction operations ---

/// Sum all elements of a tensor, returning a scalar tensor.
pub fn sum<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let total = data
        .iter()
        .copied()
        .fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
    Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)
}

/// Sum along a given axis, reducing that dimension.
pub fn sum_axis<T: Float>(input: &Tensor<T>, axis: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if axis >= shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "axis {} out of bounds for tensor with {} dims",
                axis,
                shape.len()
            ),
        });
    }

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);

    let data = input.data()?;

    let out_numel: usize = out_shape.iter().product();
    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel.max(1)];

    for (i, &val) in data.iter().enumerate() {
        // Decompose flat index into per-axis coordinates.
        let mut coords = vec![0usize; shape.len()];
        let mut rem = i;
        for d in (0..shape.len()).rev() {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        // Compute output flat index by skipping the reduced axis.
        let mut oi = 0;
        let mut os = 1;
        for d in (0..shape.len()).rev() {
            if d != axis {
                oi += coords[d] * os;
                os *= shape[d];
            }
        }
        result[oi] += val;
    }

    if out_shape.is_empty() {
        // Reduced to scalar.
        Tensor::from_storage(TensorStorage::cpu(result), vec![], false)
    } else {
        Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
    }
}

/// Mean of all elements, returning a scalar.
pub fn mean<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "mean" });
    }
    let data = input.data()?;
    let n = T::from(data.len()).unwrap();
    let total = data
        .iter()
        .copied()
        .fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
    Tensor::from_storage(TensorStorage::cpu(vec![total / n]), vec![], false)
}

/// Sum of all non-NaN elements.
///
/// NaN values are treated as zero. Returns a scalar tensor.
/// Matches PyTorch's `torch.nansum`.
pub fn nansum<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "nansum" });
    }
    let data = input.data()?;
    let total = data
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
    Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)
}

/// Mean of all non-NaN elements.
///
/// NaN values are excluded from both the sum and the count.
/// Returns a scalar tensor. Returns NaN if all elements are NaN.
/// Matches PyTorch's `torch.nanmean`.
pub fn nanmean<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "nanmean" });
    }
    let data = input.data()?;
    let mut total = <T as num_traits::Zero>::zero();
    let mut count = 0usize;
    for &v in data {
        if !v.is_nan() {
            total += v;
            count += 1;
        }
    }
    let result = if count == 0 {
        T::nan()
    } else {
        total / T::from(count).unwrap()
    };
    Tensor::from_storage(TensorStorage::cpu(vec![result]), vec![], false)
}

/// Log-sum-exp: `log(sum(exp(input)))`.
///
/// Numerically stable: subtracts the max before exponentiating.
/// Returns a scalar tensor.
/// Matches PyTorch's `torch.logsumexp`.
pub fn logsumexp<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "logsumexp" });
    }
    let data = input.data()?;
    if data.is_empty() {
        return Tensor::from_storage(TensorStorage::cpu(vec![T::neg_infinity()]), vec![], false);
    }

    // Find max for numerical stability.
    let max_val = data
        .iter()
        .copied()
        .fold(T::neg_infinity(), |a, b| if b > a { b } else { a });

    if max_val.is_infinite() && max_val < <T as num_traits::Zero>::zero() {
        // All -inf → result is -inf
        return Tensor::from_storage(TensorStorage::cpu(vec![T::neg_infinity()]), vec![], false);
    }

    let sum_exp = data
        .iter()
        .copied()
        .fold(<T as num_traits::Zero>::zero(), |acc, v| {
            acc + (v - max_val).exp()
        });

    let result = max_val + sum_exp.ln();
    Tensor::from_storage(TensorStorage::cpu(vec![result]), vec![], false)
}

/// Log-sum-exp along a dimension.
///
/// `logsumexp_dim(input, dim, keepdim)` computes `log(sum(exp(input), dim))`.
/// Numerically stable.
/// Matches PyTorch's `torch.logsumexp(input, dim)`.
pub fn logsumexp_dim<T: Float>(
    input: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
            op: "logsumexp_dim",
        });
    }
    let shape = input.shape();
    if dim >= shape.len() {
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: format!("logsumexp_dim: dim {dim} out of range for shape {shape:?}"),
        });
    }

    let data = input.data()?;
    let dim_size = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let out_numel = outer * inner;

    let mut result = Vec::with_capacity(out_numel);

    for o in 0..outer {
        for i in 0..inner {
            // Find max along this dim slice.
            let mut max_val = T::neg_infinity();
            for d in 0..dim_size {
                let idx = o * dim_size * inner + d * inner + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }

            // Sum exp(x - max).
            let mut sum_exp = <T as num_traits::Zero>::zero();
            for d in 0..dim_size {
                let idx = o * dim_size * inner + d * inner + i;
                sum_exp += (data[idx] - max_val).exp();
            }

            result.push(max_val + sum_exp.ln());
        }
    }

    let mut out_shape = shape.to_vec();
    if keepdim {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_unary_map() {
        let a = t(&[1.0, 4.0, 9.0], &[3]);
        let b = unary_map(&a, |x| x.sqrt()).unwrap();
        let d = b.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!((d[1] - 2.0).abs() < 1e-6);
        assert!((d[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_map_same_shape() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_binary_map_broadcast() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[10.0], &[1]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.data().unwrap(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_binary_map_broadcast_2d() {
        // [2,3] + [1,3] -> [2,3]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[10.0, 20.0, 30.0], &[1, 3]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data().unwrap(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_scalar_map() {
        let a = t(&[2.0, 4.0, 6.0], &[3]);
        let b = scalar_map(&a, 2.0, |x, s| x * s).unwrap();
        assert_eq!(b.data().unwrap(), &[4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_sum() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = sum(&a).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_axis() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 0 -> [5, 7, 9]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = sum_axis(&a, 0).unwrap();
        assert_eq!(s.shape(), &[3]);
        let d = s.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-6);
        assert!((d[1] - 7.0).abs() < 1e-6);
        assert!((d[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_axis_1() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 1 -> [6, 15]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = sum_axis(&a, 1).unwrap();
        assert_eq!(s.shape(), &[2]);
        let d = s.data().unwrap();
        assert!((d[0] - 6.0).abs() < 1e-6);
        assert!((d[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean() {
        let a = t(&[2.0, 4.0, 6.0, 8.0], &[4]);
        let m = mean(&a).unwrap();
        assert!((m.item().unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_fast_sigmoid_small() {
        let a = t(&[0.0, 1.0, -1.0, 5.0, -5.0], &[5]);
        let s = fast_sigmoid(&a).unwrap();
        let d = s.data().unwrap();
        // sigmoid(0) = 0.5
        assert!((d[0] - 0.5).abs() < 1e-5, "sigmoid(0) = {}", d[0]);
        // sigmoid(1) ≈ 0.7311
        assert!((d[1] - 0.7310586).abs() < 1e-5, "sigmoid(1) = {}", d[1]);
        // sigmoid(-1) ≈ 0.2689
        assert!((d[2] - 0.26894143).abs() < 1e-5, "sigmoid(-1) = {}", d[2]);
        // sigmoid(5) ≈ 0.9933
        assert!((d[3] - 0.9933072).abs() < 1e-5, "sigmoid(5) = {}", d[3]);
        // sigmoid(-5) ≈ 0.0067
        assert!((d[4] - 0.006692851).abs() < 1e-5, "sigmoid(-5) = {}", d[4]);
    }

    #[test]
    fn test_fast_sigmoid_large() {
        // Above PARALLEL_THRESHOLD to exercise the rayon path.
        let n = PARALLEL_THRESHOLD + 1024;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 10.0 - 5.0).collect();
        let a = t(&data, &[n]);
        let s = fast_sigmoid(&a).unwrap();
        let d = s.data().unwrap();
        for (i, &x) in data.iter().enumerate() {
            let expected = 1.0 / (1.0 + (-x).exp());
            assert!(
                (d[i] - expected).abs() < 1e-4,
                "sigmoid({}) = {}, expected {}",
                x,
                d[i],
                expected,
            );
        }
    }

    #[test]
    fn test_fast_tanh_small() {
        let a = t(&[0.0, 1.0, -1.0, 3.0, -3.0], &[5]);
        let s = fast_tanh(&a).unwrap();
        let d = s.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-5, "tanh(0) = {}", d[0]);
        assert!((d[1] - 1.0f32.tanh()).abs() < 1e-5, "tanh(1) = {}", d[1]);
        assert!(
            (d[2] - (-1.0f32).tanh()).abs() < 1e-5,
            "tanh(-1) = {}",
            d[2]
        );
        assert!((d[3] - 3.0f32.tanh()).abs() < 1e-5, "tanh(3) = {}", d[3]);
        assert!(
            (d[4] - (-3.0f32).tanh()).abs() < 1e-5,
            "tanh(-3) = {}",
            d[4]
        );
    }

    #[test]
    fn test_fast_tanh_large() {
        let n = PARALLEL_THRESHOLD + 1024;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 6.0 - 3.0).collect();
        let a = t(&data, &[n]);
        let s = fast_tanh(&a).unwrap();
        let d = s.data().unwrap();
        for (i, &x) in data.iter().enumerate() {
            let expected = x.tanh();
            assert!(
                (d[i] - expected).abs() < 1e-4,
                "tanh({}) = {}, expected {}",
                x,
                d[i],
                expected,
            );
        }
    }

    #[test]
    // reason: parallel-vs-scalar bit-equality — fast_add must produce the
    // same bit pattern as the scalar `a + b` it shadows. Both sides perform
    // a single non-fused float add of integer-valued operands, which is
    // bit-exact on every IEEE-754 target; epsilon would mask real drift.
    #[allow(clippy::float_cmp)]
    fn test_fast_add_parallel() {
        let n = PARALLEL_THRESHOLD + 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let a = t(&a_data, &[n]);
        let b = t(&b_data, &[n]);
        let c = fast_add(&a, &b).unwrap();
        let d = c.data().unwrap();
        for i in 0..n {
            assert_eq!(d[i], a_data[i] + b_data[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn test_fast_mul_parallel() {
        let n = PARALLEL_THRESHOLD + 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.01).collect();
        let a = t(&a_data, &[n]);
        let b = t(&b_data, &[n]);
        let c = fast_mul(&a, &b).unwrap();
        let d = c.data().unwrap();
        for i in 0..n {
            assert!(
                (d[i] - a_data[i] * b_data[i]).abs() < 1e-4,
                "mismatch at index {i}",
            );
        }
    }

    #[test]
    fn test_fast_exp_parallel() {
        let n = PARALLEL_THRESHOLD + 1024;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 10.0 - 5.0).collect();
        let a = t(&data, &[n]);
        let c = fast_exp(&a).unwrap();
        let d = c.data().unwrap();
        for i in 0..n {
            let expected = data[i].exp();
            assert!(
                (d[i] - expected).abs() / expected.max(1e-10) < 1e-4,
                "exp({}) = {}, expected {}",
                data[i],
                d[i],
                expected,
            );
        }
    }

    #[test]
    fn test_broadcast_strides_2d() {
        // [2,3] + [1,3] -> [2,3]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[10.0, 20.0, 30.0], &[1, 3]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data().unwrap(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_strides_3d() {
        // [2,1,3] + [1,2,1] -> [2,2,3]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
        let b = t(&[10.0, 100.0], &[1, 2, 1]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[2, 2, 3]);
        let d = c.data().unwrap();
        // row 0 + 10: [11, 12, 13], row 0 + 100: [101, 102, 103]
        // row 1 + 10: [14, 15, 16], row 1 + 100: [104, 105, 106]
        assert_eq!(
            d,
            &[
                11.0, 12.0, 13.0, 101.0, 102.0, 103.0, 14.0, 15.0, 16.0, 104.0, 105.0, 106.0
            ],
        );
    }

    // --- Edge-case tests for fast_exp_f32 ---

    #[test]
    fn test_fast_exp_f32_nan() {
        let result = fast_exp_f32(f32::NAN);
        assert!(
            result.is_nan(),
            "fast_exp_f32(NaN) should be NaN, got {result}"
        );
    }

    #[test]
    fn test_fast_exp_f32_pos_inf() {
        let result = fast_exp_f32(f32::INFINITY);
        assert!(
            result.is_infinite() && result > 0.0,
            "fast_exp_f32(+inf) should be +inf, got {result}"
        );
    }

    #[test]
    // reason: IEEE-754 sentinel — exp(-inf) is exactly 0.0, not "close to 0".
    // The fast-path must hit the IEEE limit, so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_fast_exp_f32_neg_inf() {
        let result = fast_exp_f32(f32::NEG_INFINITY);
        assert_eq!(
            result, 0.0,
            "fast_exp_f32(-inf) should be 0.0, got {result}"
        );
    }

    #[test]
    fn test_fast_exp_f32_zero() {
        let result = fast_exp_f32(0.0);
        assert!(
            (result - 1.0).abs() < 1e-7,
            "fast_exp_f32(0.0) should be 1.0, got {result}"
        );
    }

    // --- Edge-case tests for fast_log_f32 ---

    #[test]
    fn test_fast_log_f32_zero() {
        let result = fast_log_f32(0.0);
        assert!(
            result == f32::NEG_INFINITY,
            "fast_log_f32(0.0) should be -inf, got {result}"
        );
    }

    #[test]
    fn test_fast_log_f32_negative() {
        let result = fast_log_f32(-1.0);
        assert!(
            result.is_nan(),
            "fast_log_f32(-1.0) should be NaN, got {result}"
        );
    }

    #[test]
    fn test_fast_log_f32_nan() {
        let result = fast_log_f32(f32::NAN);
        assert!(
            result.is_nan(),
            "fast_log_f32(NaN) should be NaN, got {result}"
        );
    }

    #[test]
    fn test_fast_log_f32_pos_inf() {
        let result = fast_log_f32(f32::INFINITY);
        assert!(
            result.is_infinite() && result > 0.0,
            "fast_log_f32(+inf) should be +inf, got {result}"
        );
    }

    // --- Regression tests for vlog_f32 edge cases via fast_log_f32 wiring ---
    //
    // The polynomial body in vlog_f32 produces incorrect values for non-finite
    // or non-positive inputs (e.g. +inf → ~88.7, NaN → arbitrary finite garbage).
    // These tests pin the IEEE-754-correct values that the fast_log_f32 fallback
    // is responsible for delivering — a regression in the guard would fail here.

    #[test]
    fn test_vlog_f32_handles_pos_inf() {
        let result = vlog_f32(f32::INFINITY);
        assert!(
            result.is_infinite() && result > 0.0,
            "vlog_f32(+inf) must be +inf, got {result}"
        );
    }

    #[test]
    fn test_vlog_f32_handles_nan() {
        let result = vlog_f32(f32::NAN);
        assert!(result.is_nan(), "vlog_f32(NaN) must be NaN, got {result}");
    }

    #[test]
    // reason: IEEE-754 sentinel — log(0) is exactly -inf, not "close to -inf".
    // The fallback guard must produce the IEEE limit, so equality is correct.
    #[allow(clippy::float_cmp)]
    fn test_vlog_f32_handles_zero_and_negative() {
        assert_eq!(vlog_f32(0.0), f32::NEG_INFINITY);
        assert!(vlog_f32(-1.0).is_nan());
    }

    // --- Edge-case tests for fast_sigmoid ---

    #[test]
    fn test_fast_sigmoid_extreme_negative() {
        let a = t(&[-100.0], &[1]);
        let s = fast_sigmoid(&a).unwrap();
        let d = s.data().unwrap();
        assert!(
            (d[0] - 0.0).abs() < 1e-6,
            "sigmoid(-100) should be ~0.0, got {}",
            d[0]
        );
    }

    #[test]
    fn test_fast_sigmoid_extreme_positive() {
        let a = t(&[100.0], &[1]);
        let s = fast_sigmoid(&a).unwrap();
        let d = s.data().unwrap();
        assert!(
            (d[0] - 1.0).abs() < 1e-6,
            "sigmoid(100) should be ~1.0, got {}",
            d[0]
        );
    }

    // --- Edge-case tests for fast_tanh ---

    #[test]
    fn test_fast_tanh_extreme_negative() {
        let a = t(&[-50.0], &[1]);
        let s = fast_tanh(&a).unwrap();
        let d = s.data().unwrap();
        assert!(
            (d[0] - (-1.0)).abs() < 1e-6,
            "tanh(-50) should be ~-1.0, got {}",
            d[0]
        );
    }

    #[test]
    fn test_fast_tanh_extreme_positive() {
        let a = t(&[50.0], &[1]);
        let s = fast_tanh(&a).unwrap();
        let d = s.data().unwrap();
        assert!(
            (d[0] - 1.0).abs() < 1e-6,
            "tanh(50) should be ~1.0, got {}",
            d[0]
        );
    }

    #[test]
    fn test_nansum_skips_nan() {
        let a = t(&[1.0, f32::NAN, 3.0, f32::NAN, 5.0], &[5]);
        let s = nansum(&a).unwrap();
        let d = s.data().unwrap();
        assert!((d[0] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_nansum_no_nan() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let s = nansum(&a).unwrap();
        assert!((s.data().unwrap()[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_nanmean_skips_nan() {
        let a = t(&[1.0, f32::NAN, 3.0], &[3]);
        let m = nanmean(&a).unwrap();
        assert!((m.data().unwrap()[0] - 2.0).abs() < 1e-6); // (1+3)/2
    }

    #[test]
    fn test_nanmean_all_nan() {
        let a = t(&[f32::NAN, f32::NAN], &[2]);
        let m = nanmean(&a).unwrap();
        assert!(m.data().unwrap()[0].is_nan());
    }

    #[test]
    fn test_logsumexp_basic() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let r = logsumexp(&a).unwrap();
        let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
        assert!((r.data().unwrap()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_logsumexp_large_values() {
        // Numerical stability: large values shouldn't overflow.
        let a = t(&[1000.0, 1001.0, 1002.0], &[3]);
        let r = logsumexp(&a).unwrap();
        let d = r.data().unwrap()[0];
        assert!(
            !d.is_nan() && !d.is_infinite(),
            "logsumexp should be finite, got {d}"
        );
        // Should be approximately 1002 + ln(exp(-2)+exp(-1)+1) ≈ 1002.41
        assert!((d - 1002.408).abs() < 0.01);
    }

    #[test]
    fn test_logsumexp_dim() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = logsumexp_dim(&a, 1, false).unwrap();
        assert_eq!(r.shape(), &[2]);
        let d = r.data().unwrap();
        let expected0 = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
        let expected1 = (4.0f32.exp() + 5.0f32.exp() + 6.0f32.exp()).ln();
        assert!((d[0] - expected0).abs() < 1e-5);
        assert!((d[1] - expected1).abs() < 1e-5);
    }

    #[test]
    fn test_logsumexp_dim_keepdim() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let r = logsumexp_dim(&a, 0, true).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
    }
}
