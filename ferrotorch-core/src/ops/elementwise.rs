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
const PARALLEL_THRESHOLD: usize = 2_000_000;

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
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity()) }
}

/// Transmute a Vec<f64> to Vec<T> (zero-cost when T is f64).
#[inline]
unsafe fn transmute_vec_f64_to_t<T: Float>(v: Vec<f64>) -> Vec<T> {
    let mut v = std::mem::ManuallyDrop::new(v);
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity()) }
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
            let a_f32: &[f32] = unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const f32, n) };
            let b_f32: &[f32] = unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const f32, n) };
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
            let result = unsafe { transmute_vec_f32_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        } else if std::mem::size_of::<T>() == 8 {
            let a_f64: &[f64] = unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const f64, n) };
            let b_f64: &[f64] = unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const f64, n) };
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
            let a_f32: &[f32] = unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const f32, n) };
            let b_f32: &[f32] = unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const f32, n) };
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
            let result = unsafe { transmute_vec_f32_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        } else if std::mem::size_of::<T>() == 8 {
            let a_f64: &[f64] = unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const f64, n) };
            let b_f64: &[f64] = unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const f64, n) };
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
            let result = unsafe { transmute_vec_f64_to_t(out) };
            return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
        }
    }
    binary_map(a, b, |x, y| x * y)
}

/// SIMD-accelerated exp with rayon parallelism for large tensors.
pub fn fast_exp<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let n = data.len();
    if std::mem::size_of::<T>() == 4 {
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let len = chunk.len();
                    ferray_ufunc::kernels::simd_f32::exp_f32(
                        &inp[offset..offset + len],
                        chunk,
                    );
                });
        } else {
            ferray_ufunc::kernels::simd_f32::exp_f32(inp, &mut out);
        }
        let result = unsafe { transmute_vec_f32_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    } else if std::mem::size_of::<T>() == 8 {
        let inp: &[f64] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, n) };
        let mut out = pool_alloc_cpu_uninit_f64(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let len = chunk.len();
                    ferray_ufunc::kernels::simd_f64::exp_f64(
                        &inp[offset..offset + len],
                        chunk,
                    );
                });
        } else {
            ferray_ufunc::kernels::simd_f64::exp_f64(inp, &mut out);
        }
        let result = unsafe { transmute_vec_f64_to_t(out) };
        return Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false);
    }
    unary_map(input, |x| x.exp())
}

/// Fast f32 exp — single-element, fused, auto-vectorization friendly.
///
/// Guards special values (NaN, +inf, -inf) before delegating to `f32::exp()`,
/// which LLVM vectorizes to vexpps (AVX2) or equivalent when compiled with
/// `target-cpu=native`. The explicit checks use bitwise NaN detection
/// (`x != x`) that auto-vectorizes cleanly, and tightened clamp bounds
/// keep the internal exponent n in [-126, 127] to avoid bit-manipulation
/// overflow that would produce garbage at n=128.
#[inline(always)]
fn fast_exp_f32(x: f32) -> f32 {
    // Guard special values BEFORE clamping (auto-vectorizes via != self)
    if x != x { return f32::NAN; }         // NaN passthrough
    if x > 88.72284 { return f32::INFINITY; } // overflow → +inf
    if x < -87.33654 { return 0.0; }        // underflow → 0
    // Clamp to keep internal integer exponent n <= 127 (avoids n=128 UB)
    let x_clamped = x.min(88.0);
    x_clamped.exp()
}

/// Fast f32 log — single-element with special-case guards.
///
/// Guards negative, zero, NaN, and +inf inputs before delegating to
/// `f32::ln()`. The explicit checks prevent undefined polynomial results
/// for out-of-domain inputs.
#[inline(always)]
#[allow(dead_code)] // used by fused ops (log-softmax) and edge-case tests
fn fast_log_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return if x == 0.0 { f32::NEG_INFINITY } else { f32::NAN };
    }
    if x != x { return f32::NAN; }           // NaN passthrough
    if x == f32::INFINITY { return f32::INFINITY; } // +inf passthrough
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
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
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
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        let mut out = pool_alloc_cpu_uninit_f32(n);
        if n >= PARALLEL_THRESHOLD {
            let chunk_size = (n / rayon::current_num_threads()).max(4096);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let offset = ci * chunk_size;
                    let slice = &inp[offset..offset + chunk.len()];
                    for i in 0..chunk.len() {
                        let x_clamped = slice[i].max(-9.0).min(9.0); // tanh(9) ≈ 1 - 1.6e-8
                        let e2x = fast_exp_f32(2.0 * x_clamped);
                        chunk[i] = (e2x - 1.0) / (e2x + 1.0);
                    }
                });
        } else {
            for i in 0..n {
                let x_clamped = inp[i].max(-9.0).min(9.0); // tanh(9) ≈ 1 - 1.6e-8
                let e2x = fast_exp_f32(2.0 * x_clamped);
                out[i] = (e2x - 1.0) / (e2x + 1.0);
            }
        }
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
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
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
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
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
pub fn binary_map<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    f: impl Fn(T, T) -> T,
) -> FerrotorchResult<Tensor<T>> {
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

    // Broadcasting path with precomputed strides.
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let out_numel: usize = out_shape.iter().product();
    let strides = precompute_broadcast_strides(a.shape(), b.shape(), &out_shape);
    let mut result = Vec::with_capacity(out_numel);

    let a_data = a.data()?;
    let b_data = b.data()?;

    for i in 0..out_numel {
        let mut a_idx = 0usize;
        let mut b_idx = 0usize;
        let mut rem = i;
        // Walk dimensions from innermost to outermost (strides stored reversed).
        for &(a_stride, b_stride, out_dim) in strides.iter().rev() {
            let coord = rem % out_dim;
            rem /= out_dim;
            a_idx += coord * a_stride;
            b_idx += coord * b_stride;
        }
        result.push(f(a_data[a_idx], b_data[b_idx]));
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

/// Apply a binary function between a tensor and a scalar.
pub fn scalar_map<T: Float>(
    input: &Tensor<T>,
    scalar: T,
    f: impl Fn(T, T) -> T,
) -> FerrotorchResult<Tensor<T>> {
    // GPU fallback: transfer to CPU, compute, transfer back.
    let (cpu_input, device) = if input.is_cuda() {
        (input.cpu()?, input.device())
    } else {
        (input.clone(), input.device())
    };
    let data = cpu_input.data()?;
    let result: Vec<T> = data.iter().map(|&x| f(x, scalar)).collect();
    let out = Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)?;
    if device.is_cuda() { out.to(device) } else { Ok(out) }
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

        let a_dim = if i < a_ndim { a_shape[a_ndim - 1 - i] } else { 1 };
        let b_dim = if i < b_ndim { b_shape[b_ndim - 1 - i] } else { 1 };

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
    let total = data.iter().copied().fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
    Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)
}

/// Sum along a given axis, reducing that dimension.
pub fn sum_axis<T: Float>(input: &Tensor<T>, axis: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if axis >= shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("axis {} out of bounds for tensor with {} dims", axis, shape.len()),
        });
    }

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);

    let data = input.data()?;

    let out_numel: usize = out_shape.iter().product();
    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel.max(1)];

    for i in 0..input.numel() {
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
        result[oi] = result[oi] + data[i];
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
    // GPU fallback: transfer to CPU, compute, transfer back.
    let (cpu_input, device) = if input.is_cuda() {
        (input.cpu()?, input.device())
    } else {
        (input.clone(), input.device())
    };
    let data = cpu_input.data()?;
    let n = T::from(data.len()).unwrap();
    let total = data.iter().copied().fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
    let out = Tensor::from_storage(TensorStorage::cpu(vec![total / n]), vec![], false)?;
    if device.is_cuda() { out.to(device) } else { Ok(out) }
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
                x, d[i], expected,
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
        assert!((d[2] - (-1.0f32).tanh()).abs() < 1e-5, "tanh(-1) = {}", d[2]);
        assert!((d[3] - 3.0f32.tanh()).abs() < 1e-5, "tanh(3) = {}", d[3]);
        assert!((d[4] - (-3.0f32).tanh()).abs() < 1e-5, "tanh(-3) = {}", d[4]);
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
                x, d[i], expected,
            );
        }
    }

    #[test]
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
                data[i], d[i], expected,
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
            &[11.0, 12.0, 13.0, 101.0, 102.0, 103.0,
              14.0, 15.0, 16.0, 104.0, 105.0, 106.0],
        );
    }

    // --- Edge-case tests for fast_exp_f32 ---

    #[test]
    fn test_fast_exp_f32_nan() {
        let result = fast_exp_f32(f32::NAN);
        assert!(result.is_nan(), "fast_exp_f32(NaN) should be NaN, got {result}");
    }

    #[test]
    fn test_fast_exp_f32_pos_inf() {
        let result = fast_exp_f32(f32::INFINITY);
        assert!(result.is_infinite() && result > 0.0,
            "fast_exp_f32(+inf) should be +inf, got {result}");
    }

    #[test]
    fn test_fast_exp_f32_neg_inf() {
        let result = fast_exp_f32(f32::NEG_INFINITY);
        assert_eq!(result, 0.0, "fast_exp_f32(-inf) should be 0.0, got {result}");
    }

    #[test]
    fn test_fast_exp_f32_zero() {
        let result = fast_exp_f32(0.0);
        assert!((result - 1.0).abs() < 1e-7,
            "fast_exp_f32(0.0) should be 1.0, got {result}");
    }

    // --- Edge-case tests for fast_log_f32 ---

    #[test]
    fn test_fast_log_f32_zero() {
        let result = fast_log_f32(0.0);
        assert!(result == f32::NEG_INFINITY,
            "fast_log_f32(0.0) should be -inf, got {result}");
    }

    #[test]
    fn test_fast_log_f32_negative() {
        let result = fast_log_f32(-1.0);
        assert!(result.is_nan(), "fast_log_f32(-1.0) should be NaN, got {result}");
    }

    #[test]
    fn test_fast_log_f32_nan() {
        let result = fast_log_f32(f32::NAN);
        assert!(result.is_nan(), "fast_log_f32(NaN) should be NaN, got {result}");
    }

    #[test]
    fn test_fast_log_f32_pos_inf() {
        let result = fast_log_f32(f32::INFINITY);
        assert!(result.is_infinite() && result > 0.0,
            "fast_log_f32(+inf) should be +inf, got {result}");
    }

    // --- Edge-case tests for fast_sigmoid ---

    #[test]
    fn test_fast_sigmoid_extreme_negative() {
        let a = t(&[-100.0], &[1]);
        let s = fast_sigmoid(&a).unwrap();
        let d = s.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6,
            "sigmoid(-100) should be ~0.0, got {}", d[0]);
    }

    #[test]
    fn test_fast_sigmoid_extreme_positive() {
        let a = t(&[100.0], &[1]);
        let s = fast_sigmoid(&a).unwrap();
        let d = s.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6,
            "sigmoid(100) should be ~1.0, got {}", d[0]);
    }

    // --- Edge-case tests for fast_tanh ---

    #[test]
    fn test_fast_tanh_extreme_negative() {
        let a = t(&[-50.0], &[1]);
        let s = fast_tanh(&a).unwrap();
        let d = s.data().unwrap();
        assert!((d[0] - (-1.0)).abs() < 1e-6,
            "tanh(-50) should be ~-1.0, got {}", d[0]);
    }

    #[test]
    fn test_fast_tanh_extreme_positive() {
        let a = t(&[50.0], &[1]);
        let s = fast_tanh(&a).unwrap();
        let d = s.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6,
            "tanh(50) should be ~1.0, got {}", d[0]);
    }
}
