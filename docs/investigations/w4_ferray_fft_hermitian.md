# W4 — ferray-fft strict-Hermitian rejection (irfftn / hfft / ihfft)

**Status:** investigation only. No source modified under `/home/doll/ferray/` or `/home/doll/ferrotorch/src/`. Fix proposal at the bottom is staged for **W5**.

**Repro:** ferray-fft `0.3.7` (workspace). `irfft`/`hfft` panics or errors with the message
`Imaginary parts of both first and last values were non-zero.` (or one of the single-bin variants) when the input spectrum is not strictly Hermitian on the c2r axis. `torch.fft.irfftn` and `scipy.fft.irfftn` accept the same input silently, projecting it onto the Hermitian subspace.

This document mirrors the structure of `w1_taylor_divergence.md`.

---

## 1. scipy reference: silent Hermitian projection

scipy 1.17.1, numpy 2.4.4 (probe: `/tmp/w4_scipy_probe.py`).

### 1-D irfft

Input deliberately non-Hermitian — both DC (`spec[0]`) and Nyquist (`spec[-1]`) carry non-zero imaginary parts:

```python
spec = np.array([1+1j, 2+3j, 4-1j, -1+0.5j, 7+9j])  # n/2+1 = 5  →  output n=8
out = scipy.fft.irfft(spec, n=8)          # NO error, NO warning
```

Output:

```
[ 2.25       -0.58838835 -0.625      -2.14904852  1.75       -0.41161165
  0.625       0.14904852]
```

Compare against the same input with the imaginary parts of bins 0 and N/2 manually zeroed:

```python
spec_proj = spec.copy(); spec_proj[0] = spec_proj[0].real; spec_proj[-1] = spec_proj[-1].real
out_proj = scipy.fft.irfft(spec_proj, n=8)
np.max(np.abs(out - out_proj))  # → 0.0  (bit-identical)
```

scipy emits **no warning** even with `warnings.simplefilter("always")`.

### 2-D irfftn (last axis is c2r; inner axes are c2c inverse passes)

```python
spec2d = (rng.standard_normal((3,5)) + 1j*rng.standard_normal((3,5)))  # arbitrary complex
out2d = scipy.fft.irfftn(spec2d, s=(3, 8))   # NO error
# shape (3, 8), dtype float64
```

### Where in scipy this happens

`scipy.fft._pocketfft.basic.c2r` (file: `scipy/fft/_pocketfft/basic.py:70-95`):

```python
def c2r(forward, x, n=None, axis=-1, norm=None, ...):
    tmp = _asfarray(x)
    ...
    if n is None:
        n = (tmp.shape[axis] - 1) * 2
    else:
        tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)
    return pfft.c2r(tmp, (axis,), n, forward, norm, None, workers)
```

The validity contract on bins 0 and N/2 is **never explicitly enforced** in scipy's Python wrappers. The C extension `pypocketfft.c2r` (pocketfft, the underlying C library) implements c2r via a half-size complex transform whose algorithm reads only the *real* part of bin 0 and the *real* part of bin N/2 — the imaginary parts of those two bins are arithmetically discarded by construction. So scipy's silent projection is *implicit* in pocketfft's c2r kernel: there is no extra zeroing pass, the kernel just doesn't touch those imaginary parts. There is **no fallback, no warning, no error path**: the projection is the algorithm.

Conclusion: scipy and PyTorch both **silently project**; ferray-fft is the outlier.

---

## 2. ferray-fft source dump

### 2.1 The irfft entry point

**File:** `/home/doll/ferray/ferray-fft/src/real.rs`, lines 125–151.

```rust
pub fn irfft<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;

    let half_len = shape[ax];
    let output_len = n.unwrap_or(2 * (half_len - 1));
    if output_len == 0 {
        return Err(FerrayError::invalid_value("irfft output length must be > 0"));
    }
    let complex_data: Vec<Complex<T>> = a.iter().copied().collect();
    let (out_shape, out_data) = irfft_along_axis::<T>(&complex_data, &shape, ax, output_len, norm)?;
    Array::from_vec(IxDyn::new(&out_shape), out_data)
}
```

`irfft` simply delegates to `irfft_along_axis` (in `nd.rs`).

### 2.2 The lane helper that actually calls realfft

**File:** `/home/doll/ferray/ferray-fft/src/nd.rs`, lines 383–503.

The 1-D fast path (lines 432–449) runs `plan.process_with_scratch(...)` and propagates the error as `FerrayError::invalid_value(format!("inverse real FFT process failed: {e}"))`:

```rust
// nd.rs lines 433-448
if ndim == 1 {
    let mut input_buf: Vec<Complex<T>> = vec![Complex::zero(); half_len];
    input_buf[..copy_len].copy_from_slice(&data[..copy_len]);
    let mut output_buf: Vec<T> = vec![t_zero; output_len];
    let mut scratch = plan.make_scratch_vec();
    plan.process_with_scratch(&mut input_buf, &mut output_buf, &mut scratch)
        .map_err(|e| {
            FerrayError::invalid_value(format!("inverse real FFT process failed: {e}"))
        })?;
    ...
    return Ok((new_shape, output_buf));
}
```

The multi-lane parallel path (lines 457–483) **panics** via `.expect(...)`:

```rust
// nd.rs lines 467-482
|(input_buf, scratch), (out_chunk, &start_offset)| {
    for (i, slot) in input_buf.iter_mut().take(copy_len).enumerate() {
        *slot = data[start_offset + i * stride];
    }
    for slot in input_buf.iter_mut().skip(copy_len) {
        *slot = Complex::zero();
    }
    plan.process_with_scratch(input_buf, out_chunk, scratch)
        .expect("inverse real FFT process failed");      // <-- LINE 476
    if scale != one {
        for v in out_chunk.iter_mut() {
            *v = *v * scale;
        }
    }
}
```

### 2.3 The realfft contract

**File:** `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/realfft-3.5.0/src/lib.rs`.

`ComplexToRealEven::process_with_scratch` (lines 649–756):

```rust
// realfft 3.5.0 lib.rs lines 671-682
let first_invalid = if input[0].im != T::from_f64(0.0).unwrap() {
    input[0].im = T::from_f64(0.0).unwrap();              // zeros it in place
    true
} else {
    false
};
let last_invalid = if input[input.len() - 1].im != T::from_f64(0.0).unwrap() {
    input[input.len() - 1].im = T::from_f64(0.0).unwrap();  // zeros it in place
    true
} else {
    false
};
...
self.fft.process_outofplace_with_scratch(&mut input[..buf_out.len()], buf_out, scratch);
if first_invalid || last_invalid {
    return Err(FftError::InputValues(first_invalid, last_invalid));   // <-- LINE 753
}
Ok(())
```

(`ComplexToRealOdd` does the same for the DC bin only — odd N has no Nyquist; lines 567–593.)

`FftError::InputValues(first_invalid, last_invalid)` formats as
`"Imaginary parts of both first and last values were non-zero."` for `(true, true)`,
`"Imaginary part of first value was non-zero."` for `(true, false)`, and
`"Imaginary part of last value was non-zero."` for `(false, true)` (lib.rs lines 59–62).

**Critical observation:** realfft 3.5.0 already performs the projection internally — it zeros the imaginary parts in place **before** running the FFT, gets a numerically correct result, then *also* returns an error to flag the contract violation. So the projection is "free" in realfft; ferray-fft is just propagating the error as a panic. The fix is essentially "swallow the `InputValues` error variant when projection is the desired behavior".

### 2.4 hfft / ihfft delegate to irfft / rfft

**File:** `/home/doll/ferray/ferray-fft/src/hermitian.rs`.

`hfft` (lines 54–90) — conjugates the input, then calls `irfft` with swapped norm:

```rust
let conj_data: Vec<Complex<T>> = a.iter().map(num_complex::Complex::conj).collect();
let conj_arr = Array::<Complex<T>, IxDyn>::from_vec(IxDyn::new(&shape), conj_data)?;
crate::real::irfft::<T, IxDyn>(&conj_arr, Some(output_len), Some(ax as isize), swap_norm(norm))
```

So `hfft` inherits the same panic from `irfft` (the inputs DC/Nyquist remain DC/Nyquist after conjugation; conjugation only flips the imaginary sign, doesn't make it zero).

`hfftn` (lines 150–163) — same: conjugate, then `crate::real::irfftn`. Inherits panic from `irfftn`.

`ihfft` and `ihfftn` go through `rfft`/`rfftn` on **real** inputs — the input has no imaginary parts to violate, so they cannot trigger this panic. (Confirmed by case D in the probe.)

### 2.5 irfftn invokes the c2r path on the LAST transformed axis

**File:** `/home/doll/ferray/ferray-fft/src/real.rs`, lines 390–456 (`irfftn_impl`).

```rust
// real.rs lines 432-453
// NumPy's irfftn runs complex inverse FFTs on axes[0..last] first, then
// the complex-to-real inverse transform on axes[last]. We match that order...
let mut current_data: Vec<Complex<T>> = a.iter().copied().collect();
let mut current_shape = input_shape;

for i in 0..last_idx {
    let ax = axes[i];
    let n = Some(sizes[i]);
    let (new_shape, new_data) =
        fft_along_axis::<T>(&current_data, &current_shape, ax, n, true, norm)?;
    current_shape = new_shape;
    current_data = new_data;
}

// Final step: complex-to-real inverse transform on the last axis...
let last_ax = axes[last_idx];
let output_len = sizes[last_idx];
let (final_shape, final_data) =
    irfft_along_axis::<T>(&current_data, &current_shape, last_ax, output_len, norm)?;
```

**Important for the multi-axis fix:** the inner inverse complex FFTs run **before** the c2r step. They mix bins along non-c2r axes, so the Hermitian-validity check can only meaningfully be applied to `current_data` *after* the inner ifft loop and *before* the `irfft_along_axis` call.

---

## 3. ferrotorch's existing mitigation (reference implementation for the upstream fix)

**File:** `/home/doll/ferrotorch/ferrotorch-core/src/fft.rs`, lines 762–834, 938–1004.

### 3.1 The projection routine

```rust
// ferrotorch-core/src/fft.rs lines 777-834
fn project_hermitian_in_place(
    arr: &mut FerrayArray<Complex<f64>, FerrayIxDyn>,
    axis: usize,
    output_n: usize,
) {
    let shape = arr.shape().to_vec();
    let axis_len = shape[axis];
    if axis_len == 0 { return; }

    // Identify the bins along `axis` to zero the imaginary part on:
    //   * DC (index 0) — always.
    //   * Nyquist (index output_n / 2) — only when `output_n` is even and
    //     that index actually exists in the input array.
    let mut nyq_idx: Option<usize> = None;
    if output_n % 2 == 0 {
        let nyq = output_n / 2;
        if nyq < axis_len { nyq_idx = Some(nyq); }
    }

    let Some(buf) = arr.as_slice_mut() else { return; };
    let ndim = shape.len();
    let mut idx = vec![0usize; ndim];
    loop {
        let on_dc = idx[axis] == 0;
        let on_nyq = nyq_idx == Some(idx[axis]);
        if on_dc || on_nyq {
            let lin = compute_linear_index(&shape, &idx);
            let v = buf[lin];
            buf[lin] = Complex::new(v.re, 0.0);
        }
        // ... row-major increment ...
    }
}
```

### 3.2 The two-staged multi-axis mitigation in `irfftn`

```rust
// ferrotorch-core/src/fft.rs lines 933-1004 (abridged)
pub fn irfftn<T: Float>(...) -> FerrotorchResult<Tensor<T>> {
    let mut arr = tensor_to_complex_array(input, "irfftn")?;
    let resolved = resolve_irfftn_axes(arr.shape(), s, axes);
    let Some((resolved_axes, sizes)) = resolved else { ...defer to ferray-fft... };
    let last_idx = resolved_axes.len() - 1;
    let c2r_axis = resolved_axes[last_idx];
    let output_n = sizes[last_idx];

    if last_idx == 0 {
        // Single-axis: project the input directly.
        project_hermitian_in_place(&mut arr, c2r_axis, output_n);
        let result = ferray_fft::irfftn(&arr, s, axes, FftNorm::Backward)?;
        return real_array_to_tensor(&result);
    }

    // Multi-axis: inverse complex FFTs on axes[..last], project, then c2r.
    let inner_axes: Vec<isize> = resolved_axes[..last_idx].iter().map(|&a| a as isize).collect();
    let inner_sizes: Vec<usize> = sizes[..last_idx].to_vec();
    let intermediate =
        ferray_fft::ifftn(&arr, Some(&inner_sizes), Some(&inner_axes), FftNorm::Backward)?;
    let mut intermediate = intermediate;
    project_hermitian_in_place(&mut intermediate, c2r_axis, output_n);
    let result = ferray_fft::irfftn(
        &intermediate,
        Some(&[output_n]),
        Some(&[c2r_axis as isize]),
        FftNorm::Backward,
    )?;
    real_array_to_tensor(&result)
}
```

### 3.3 The hfft mitigation

```rust
// ferrotorch-core/src/fft.rs lines 1016-1038
pub fn hfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    let mut arr = tensor_to_complex_array(input, "hfft")?;
    let shape = arr.shape().to_vec();
    if let Some(&half_n) = shape.last() {
        if half_n > 0 {
            let last_ax = shape.len() - 1;
            let output_n = n.unwrap_or(2 * (half_n - 1));
            project_hermitian_in_place(&mut arr, last_ax, output_n);
        }
    }
    let result = ferray_fft::hfft(&arr, n, None, FftNorm::Backward)?;
    real_array_to_tensor(&result)
}
```

(`ihfft` does not need a mitigation — its input is real.)

The key insight from §3.2 — the two-staged "inner-ifftn → project → c2r" structure — is exactly how the upstream fix should integrate inside `irfftn_impl`, since `irfftn_impl` already runs the inner ifft loop before the c2r step (§2.5).

---

## 4. Step-by-step trace of `irfftn(non_hermitian)` panic

Probe: `/tmp/w4_ferray_fft_probe/src/main.rs`, case (B) — 2-D, axis 1, n=8, 3 lanes, DC and Nyquist im non-zero.

1. `ferray_fft::irfftn(arr, Some(&[8]), Some(&[1]), Backward)` → `real.rs:323` `irfftn`.
2. `resolve_axes(2, Some(&[1]))` → `vec![1]`. Calls `irfftn_impl::<f64, Ix2>(arr, Some(&[8]), &[1], Backward)` (`real.rs:324`).
3. `irfftn_impl`: `axes = [1]`, `last_idx = 0`, so the inner ifft loop (`real.rs:439-446`) is skipped. Goes straight to:
   ```rust
   let (final_shape, final_data) =
       irfft_along_axis::<T>(&current_data, &current_shape, last_ax=1, output_len=8, norm)?;
   ```
   (`real.rs:452`).
4. `irfft_along_axis` (`nd.rs:383`): ndim=2, so the 1-D fast path branch (line 433) is **skipped**. Falls into the multi-lane path:
   ```rust
   let plan = T::cached_real_inverse(output_len);   // realfft ComplexToRealEven for n=8
   ...
   lane_outputs
       .par_chunks_mut(output_len)
       .zip(lane_starts.par_iter())
       .for_each_init(...,
           |(input_buf, scratch), (out_chunk, &start_offset)| {
               for (i, slot) in input_buf.iter_mut().take(copy_len).enumerate() {
                   *slot = data[start_offset + i * stride];
               }
               ...
               plan.process_with_scratch(input_buf, out_chunk, scratch)
                   .expect("inverse real FFT process failed");        // ← LINE 476
               ...
           },
       );
   ```
5. For each lane (rayon worker), `realfft::ComplexToRealEven::process_with_scratch` (realfft `lib.rs:649`):
   - sees `input[0].im != 0`, sets `first_invalid = true`, zeros it in place.
   - sees `input[input.len()-1].im != 0`, sets `last_invalid = true`, zeros it in place.
   - runs the inner half-size complex FFT (correct now that DC/Nyquist are real).
   - returns `Err(FftError::InputValues(true, true))` (lib.rs:753).
6. `.expect("inverse real FFT process failed")` panics with the formatted error message:
   `inverse real FFT process failed: Imaginary parts of both first and last values were non-zero.`

This panic is observed **per-lane**, so multi-lane c2r prints multiple panic messages (3 in the probe) before the rayon worker pool unwinds back to the caller. Reproduced exactly in the probe (3 panic lines from rayon worker threads).

**Quoted line:** `nd.rs:476`:

```rust
plan.process_with_scratch(input_buf, out_chunk, scratch)
    .expect("inverse real FFT process failed");
```

Symmetric panic site on the forward side at `nd.rs:345` (`.expect("real FFT process failed")`) — but that one **cannot fire** because realfft's r2c half never returns `InputValues` (the input is real, no imaginary parts to invalidate). It would only matter if a future realfft version added new error variants, but for the strict-Hermitian rejection at hand, only line 476 is reachable.

---

## 5. scipy's algorithm: projection is the only difference

Confirmed via probe (§1) and source dump:

- **No warning.** `warnings.simplefilter("always")` captures nothing.
- **No fallback.** scipy doesn't try the strict path first and then retry; it just calls `pfft.c2r`.
- **No error path.** The pocketfft C kernel ignores the imaginary parts of bins 0 and N/2 by construction — there's literally no validation step.
- **PyTorch parity.** `torch.fft.irfftn` and `torch.fft.hfft` likewise project silently inside `aten::_fft_c2r`. (Documented in ferrotorch-core/src/fft.rs:769 — "matches the pre-pass that PyTorch's `aten::_fft_c2r` runs before the underlying FFTW/cuFFT call".)

Therefore the projection is the **complete** behavioral difference. Inserting a Hermitian-projection pre-pass into ferray-fft's c2r path will bring it to parity with both reference implementations.

---

## 6. Root cause + proposed fix

### 6.1 Root cause

`irfft_along_axis` (in `nd.rs`) calls realfft's `ComplexToReal::process_with_scratch` and treats `FftError::InputValues` — a **soft** error that realfft already recovered from internally — as a hard panic (multi-lane path) or hard error (1-D path). The contract violation is *cosmetic* from realfft's perspective (it already zeroed the bins and produced a correct result); ferray-fft is the only layer treating it as fatal.

### 6.2 The fix in three lines, conceptually

In `nd.rs`, where `irfft_along_axis` calls realfft's `process_with_scratch`, swallow the `FftError::InputValues` variant — the output buffer already contains the correct projected result.

But that creates a fragile coupling to realfft's error enum (and would also silently swallow legitimate input-shape errors from a future realfft version that reused the variant). The cleaner approach mirrors PyTorch and ferrotorch's mitigation: **do the projection ourselves before calling realfft**, so the `InputValues` error variant cannot fire.

### 6.3 Proposed pseudocode / annotated diff

#### 6.3.1 Helper to add to `nd.rs` (or `real.rs`) — module-private

```rust
/// Zero the imaginary parts of the DC and (when N is even) Nyquist bins
/// of every lane along `axis`, in place, where `axis` is the c2r axis.
/// `output_len` is the real-axis output length (i.e. N), which determines
/// whether a Nyquist bin exists. The input axis length is `output_len/2 + 1`
/// (or the value the caller passed; we tolerate truncation/padding).
///
/// Matches PyTorch's `aten::_fft_c2r` pre-pass and scipy/pocketfft's
/// implicit projection. For inputs already on the Hermitian subspace
/// this is a no-op; for inputs off the subspace it projects them onto it,
/// which is the documented numpy/scipy/torch semantics.
fn project_hermitian_along_axis<T: FftFloat>(
    data: &mut [Complex<T>],
    shape: &[usize],
    axis: usize,
    output_len: usize,
) {
    let axis_len = shape[axis];
    if axis_len == 0 { return; }

    // Bins to project: DC always; Nyquist (output_len/2) only when output_len is
    // even AND that index exists in the input.
    let nyq_idx: Option<usize> = if output_len % 2 == 0 {
        let nyq = output_len / 2;
        (nyq < axis_len).then_some(nyq)
    } else {
        None
    };

    let strides = compute_strides(shape);
    let stride = strides[axis] as usize;

    // For each lane along `axis`, zero data[lane_start + bin*stride].im
    // for bin == 0 and bin == nyq_idx (if any).
    let total: usize = shape.iter().product();
    let num_lanes = total / axis_len;
    let lane_starts = compute_lane_starts(shape, &strides, axis, num_lanes);

    let zero = T::zero();
    for &start in &lane_starts {
        let v = data[start];
        data[start] = Complex::new(v.re, zero);   // DC bin
        if let Some(k) = nyq_idx {
            let v = data[start + k * stride];
            data[start + k * stride] = Complex::new(v.re, zero);
        }
    }
}
```

(For ndim=1 the loop runs over a single lane at offset 0, which collapses correctly.)

#### 6.3.2 `irfft_along_axis` — call the helper before realfft

```rust
// nd.rs irfft_along_axis (around the entry, after `let half_len = output_len / 2 + 1;`)
//
// PROPOSED ADDITION:
// We OWN `complex_data` (the caller passed `data: &[Complex<T>]`, but the
// 1-D path copies into `input_buf` and the multi-lane path also copies into
// `input_buf`; the lane copy is where projection should happen).

// In the 1-D fast path, after copying into `input_buf`:
input_buf[..copy_len].copy_from_slice(&data[..copy_len]);
// NEW: project in place.
input_buf[0].im = T::zero();
if output_len % 2 == 0 && half_len > output_len / 2 {
    let nyq = output_len / 2;
    input_buf[nyq].im = T::zero();
}
// existing call:
plan.process_with_scratch(&mut input_buf, &mut output_buf, &mut scratch)
    .map_err(|e| FerrayError::invalid_value(format!("inverse real FFT process failed: {e}")))?;
```

For the multi-lane path, project the per-lane `input_buf` after the strided copy (around line 471), before the realfft call:

```rust
|(input_buf, scratch), (out_chunk, &start_offset)| {
    for (i, slot) in input_buf.iter_mut().take(copy_len).enumerate() {
        *slot = data[start_offset + i * stride];
    }
    for slot in input_buf.iter_mut().skip(copy_len) {
        *slot = Complex::zero();
    }
    // NEW: Hermitian projection on this lane's DC and (if even N) Nyquist bins.
    input_buf[0].im = T::zero();
    if output_len % 2 == 0 {
        let nyq = output_len / 2;
        if nyq < input_buf.len() {
            input_buf[nyq].im = T::zero();
        }
    }
    plan.process_with_scratch(input_buf, out_chunk, scratch)
        .expect("inverse real FFT process failed");   // now unreachable for InputValues
    ...
}
```

This places the projection inside the lane copy, costs O(num_lanes) (negligible vs the O(num_lanes · half_len · log) FFT), and is naturally rayon-parallel.

The `.expect(...)` on line 476 is no longer reachable for `InputValues` because we've zeroed the offending bins before calling realfft. It would still be reachable for `InputBuffer`/`OutputBuffer`/`ScratchBuffer` size mismatches — those are genuine programming errors and *should* panic. The wording could be clarified in a follow-up, but that's out of scope for W5.

#### 6.3.3 Multi-axis irfftn (`real.rs:irfftn_impl`)

`irfftn_impl` already runs inner ifft passes before the c2r step (§2.5). The c2r step is `irfft_along_axis`, which the §6.3.2 patch now projects internally. **No change to `irfftn_impl` is needed** — once `irfft_along_axis` projects on the way in, the multi-axis case is automatically correct, because the inner inverse complex FFTs run first and `irfft_along_axis` projects on their output before the c2r kernel sees it. This is a strict simplification compared to ferrotorch's `irfftn` mitigation (which had to drive the two-stage flow manually because it sat *outside* ferray-fft).

#### 6.3.4 hfft / hfftn

`hfft` (`hermitian.rs:54`) conjugates and calls `irfft`. `hfftn` (`hermitian.rs:150`) conjugates and calls `irfftn`. Both inherit the §6.3.2 fix via the `irfft_along_axis` call inside `irfft`/`irfftn_impl`. **No change to hermitian.rs is needed.**

#### 6.3.5 ihfft / ihfftn / rfft / rfftn — **no fix needed**

Real-input transforms cannot violate Hermitian symmetry on the input side. Forward `rfft` (and therefore `ihfft = conj(rfft)`) takes a real array, so the `InputValues` variant cannot fire on the forward side (confirmed by §2.3: realfft's `RealToComplex::process` doesn't validate imaginary parts because there are none).

---

## 7. Validation plan

After the §6 fix lands upstream:

1. **ferrotorch retirement (W6).** `project_hermitian_in_place` and the multi-stage path inside `ferrotorch-core/src/fft.rs::irfftn` and `::hfft` become redundant — they are the same projection, applied at the same point (post-inner-ifftn, pre-c2r), with the same semantics. W6 should:
   - Delete `project_hermitian_in_place` and `compute_linear_index`.
   - Replace the multi-stage `irfftn` body (§3.2) with a single direct call to `ferray_fft::irfftn(&arr, s, axes, FftNorm::Backward)`.
   - Replace the projection-then-call body of `hfft` with a single `ferray_fft::hfft(&arr, n, None, FftNorm::Backward)` call.
   - Bump the `ferray-fft` workspace dep version to whatever is published with the W5 fix.
2. **Regression suite.** Add ferray-fft tests that mirror `/tmp/w4_ferray_fft_probe/src/main.rs`:
   - `irfft` 1-D, non-Hermitian → succeeds, output matches manually-projected input.
   - `irfft` 2-D multi-lane, non-Hermitian → succeeds (no panic), output matches.
   - `hfft` 1-D, non-Hermitian → succeeds.
   - `irfftn` 2-D and 3-D, non-Hermitian → succeeds, parity with `scipy.fft.irfftn`.
   - `irfft` already-Hermitian → byte-identical to current behavior (no-op projection).
   - Odd-N (no Nyquist) → only DC is projected.
3. **scipy parity probe.** The §1 probe values become the reference oracle for ferray-fft tests:
   ```
   irfft([1+1j, 2+3j, 4-1j, -1+0.5j, 7+9j], n=8) == [2.25, -0.588388..., -0.625, ...]
   ```
4. **PyTorch parity test in ferrotorch.** Run `crates/ferrotorch-tests` against the same non-Hermitian input on both `ferrotorch::ops::fft::irfftn` and `torch.fft.irfftn`; expect ≤ 1e-10 max abs diff.

---

## 8. Edge cases

| Input form | Expected behavior under §6 fix |
|---|---|
| **Already Hermitian** (DC.im=0, Nyq.im=0 if even N) | No-op projection (zeroing zero is zero). Output bit-identical to current behavior. **Strict no-op: no precision change for valid inputs.** |
| **Real input** wrapped as Complex with .im=0 | Same as above — no-op. |
| **N=1** (output length 1) | `half_len = 1/2 + 1 = 1`. Only the DC bin exists; no Nyquist. Projection zeros DC.im. realfft `ComplexToRealOdd` (N=1 is odd) only validates DC. Already correct. |
| **Odd N** (e.g. N=5) | No Nyquist exists; only DC is projected. realfft's `ComplexToRealOdd` only validates DC (lib.rs:567). Matches. |
| **Even N**, axis_len < N/2+1 (truncated input via the `irfft` `n` parameter) | If `nyq = N/2 >= axis_len`, the Nyquist bin index isn't in the input slice; we skip it. `(nyq < axis_len).then_some(nyq)` guards this. realfft never sees the missing bin (the input buffer is zero-padded by the lane copy to the correct half_len). |
| **Multi-axis irfftn with non-c2r axes also non-real** | The inner c2c inverse FFTs are unconstrained — any complex input is valid for `fft_along_axis`. The Hermitian constraint applies only at the *output* of the inner pass, on the c2r axis. The §6.3.2 patch projects exactly there. |
| **Broadcast / non-contiguous input** | `irfftn_impl` materializes via `a.iter().copied().collect::<Vec<_>>()` (real.rs:436), giving a contiguous owned buffer. The lane-copy inside `irfft_along_axis` then reads via strided indexing into a contiguous per-lane `input_buf`, so the projection sees a contiguous lane slice — safe to mutate. |
| **f32 vs f64** | `T: FftFloat` is generic; `T::zero()` is precision-correct. realfft's `T::from_f64(0.0).unwrap()` is `0_f32` or `0_f64` depending on the plan precision. Bit-identical zero either way. |
| **Empty input (total == 0)** | `irfft_along_axis` short-circuits at line 417 (`if total == 0 { return Ok(...zeros...); }`) before reaching the projection or the FFT. Safe. |

---

## 9. Compatibility

| Caller-side input | Pre-fix behavior | Post-fix behavior | Compatible? |
|---|---|---|---|
| Valid Hermitian | Works correctly | Works correctly (projection is no-op) | **Yes — bit-identical** |
| Invalid (off-Hermitian) — 1-D | `Err(FerrayError::invalid_value("inverse real FFT process failed: ..."))` | `Ok(projected_output)` matching scipy | Behavior change: `Err` → `Ok`. **Strict improvement; matches scipy/torch.** |
| Invalid (off-Hermitian) — multi-lane | **Panic** in rayon worker | `Ok(projected_output)` matching scipy | Behavior change: panic → `Ok`. **Strict improvement.** |

No existing valid call site changes its observable output. The only behavior change is that previously-rejected invalid inputs now succeed with the correct projected result — no test in ferray-fft can be relying on the panic (panics in rayon workers are not catchable at API call sites without `catch_unwind`, and ferray-fft has no such tests).

This is the same reasoning ferrotorch's batch 7 mitigation used (§3) — *that* shipped without controversy, and the upstream version is just the same projection one layer deeper.

**Recommendation:** non-breaking patch release (e.g. ferray-fft 0.3.8). No deprecation or feature flag required.

---

## 10. Proposed fix scope

Exactly two code paths in ferray-fft need the projection pre-pass; both inside `nd.rs::irfft_along_axis`:

### 10.1 1-D fast path

```rust
// Current (real.rs imports nd::irfft_along_axis; the 1-D path is at nd.rs:432-449)
pub fn irfft_along_axis<T: FftFloat>(
    data: &[Complex<T>],
    shape: &[usize],
    axis: usize,
    output_len: usize,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<T>)>
where
    Complex<T>: ferray_core::Element,
{ ... }
```

### 10.2 Multi-lane path (same function, lane-copy site at nd.rs:467-482)

Same function signature; the patch is inside the `for_each_init` closure body. Since both paths share the same outer function, **only `irfft_along_axis` needs editing.**

### 10.3 Functions that get the fix transitively (no edit)

| Function | File | Signature | How it picks up the fix |
|---|---|---|---|
| `irfft<T, D>(a, n, axis, norm) -> FerrayResult<Array<T, IxDyn>>` | `real.rs:125` | `pub fn irfft<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>, n: Option<usize>, axis: Option<isize>, norm: FftNorm) -> FerrayResult<Array<T, IxDyn>>` | Calls `irfft_along_axis` directly |
| `irfft_into<T, D>(...)` | `real.rs:161` | `pub fn irfft_into<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>, n: Option<usize>, axis: Option<isize>, norm: FftNorm, out: &mut Array<T, IxDyn>) -> FerrayResult<()>` | Calls `irfft_along_axis` directly |
| `irfftn<T, D>(a, s, axes, norm)` | `real.rs:314` | `pub fn irfftn<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>, s: Option<&[usize]>, axes: Option<&[isize]>, norm: FftNorm) -> FerrayResult<Array<T, IxDyn>>` | Goes through `irfftn_impl` → `irfft_along_axis` on the last axis |
| `irfft2<T, D>(a, s, axes, norm)` | `real.rs:249` | `pub fn irfft2<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>, s: Option<&[usize]>, axes: Option<&[isize]>, norm: FftNorm) -> FerrayResult<Array<T, IxDyn>>` | Same as `irfftn` |
| `hfft<T, D>(a, n, axis, norm)` | `hermitian.rs:54` | `pub fn hfft<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>, n: Option<usize>, axis: Option<isize>, norm: FftNorm) -> FerrayResult<Array<T, IxDyn>>` | Conjugates input, calls `irfft` |
| `hfft2<T, D>(...)` | `hermitian.rs:190` | `pub fn hfft2<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>, s: Option<&[usize]>, axes: Option<&[isize]>, norm: FftNorm) -> FerrayResult<Array<T, IxDyn>>` | Calls `hfftn` → `irfftn` |
| `hfftn<T, D>(...)` | `hermitian.rs:150` | `pub fn hfftn<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>, s: Option<&[usize]>, axes: Option<&[isize]>, norm: FftNorm) -> FerrayResult<Array<T, IxDyn>>` | Conjugates input, calls `irfftn` |

`ihfft` / `ihfft2` / `ihfftn` / `rfft` / `rfft2` / `rfftn` / `rfft_into` operate on real input and **cannot** trigger the strict-Hermitian rejection. No change.

---

## 11. Summary

- Panic site: `nd.rs:476`, `.expect("inverse real FFT process failed")`. (1-D mode propagates the same condition as `Err` rather than panic; behavior in both cases is wrong relative to scipy/torch.)
- Root cause: ferray-fft surfaces realfft's `FftError::InputValues` as a hard failure, even though realfft has already projected the input to the Hermitian subspace and produced a numerically correct result.
- Reference implementations: scipy's pocketfft kernel (`pfft.c2r`) ignores DC/Nyquist imaginary parts by construction; PyTorch's `aten::_fft_c2r` projects explicitly.
- Fix: insert a `project_hermitian_along_axis` pre-pass inside `nd.rs::irfft_along_axis` (one helper, two call sites — the 1-D fast path and the multi-lane lane-copy closure). Costs O(num_lanes), negligible. Strict no-op on valid Hermitian inputs.
- Surface area: **only `irfft_along_axis`** in ferray-fft needs editing. All public entry points (`irfft`, `irfft_into`, `irfft2`, `irfftn`, `hfft`, `hfft2`, `hfftn`) pick the fix up transitively.
- Compatibility: non-breaking; no deprecation needed. Bumps a 0.3.x patch (e.g. 0.3.8).
- Downstream cleanup (W6): retire `project_hermitian_in_place` and the two-staged `irfftn` body in `ferrotorch-core/src/fft.rs`, replacing both with single direct calls to `ferray_fft::irfftn` / `ferray_fft::hfft`.

**DONE: W4 investigation complete; fix proposal at docs/investigations/w4_ferray_fft_hermitian.md; ready for W5**
