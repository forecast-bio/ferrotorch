//! Window functions for signal processing.
//!
//! Mirrors `torch.signal.windows.*` (and `numpy`/`scipy` window APIs). These
//! are tiny coefficient generators that build a 1-D `Tensor<f64>` of length
//! `m`. They never allocate GPU memory — the cost of a GPU upload would
//! dwarf the cost of generating the window — so the returned tensor lives
//! on [`Device::Cpu`](crate::Device::Cpu).
//!
//! # GPU discipline
//!
//! All functions in this module return CPU tensors. To use a window on a
//! CUDA / XPU model, generate it once and move it explicitly:
//!
//! ```ignore
//! use ferrotorch_core::{Device, signal::windows};
//! let w = windows::hann(1024)?.to(Device::Cuda(0))?;
//! ```
//!
//! There is no silent device dispatch here: a fake GPU path would replace
//! a 1-microsecond CPU compute with a CPU compute + cudaMemcpy, which is
//! strictly slower. Per `/rust-gpu-discipline`, when the GPU path would
//! be a regression we don't add it.

use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Bartlett (triangular) window of length `m`.
///
/// ```text
/// w[n] = 1 - |2n / (M - 1) - 1|,   n = 0..M-1
/// ```
///
/// Mirrors `torch.signal.windows.bartlett` / `numpy.bartlett`.
pub fn bartlett(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::bartlett(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Blackman window of length `m`. Mirrors `torch.signal.windows.blackman`.
pub fn blackman(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::blackman(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Hamming window of length `m`. Mirrors `torch.signal.windows.hamming`.
pub fn hamming(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::hamming(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Hann window of length `m` (NumPy spells this `hanning`; both are
/// provided for ergonomic compatibility).
///
/// Mirrors `torch.signal.windows.hann` / `numpy.hanning`.
pub fn hann(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::hanning(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Alias of [`hann`] matching `numpy.hanning`'s spelling.
#[inline]
pub fn hanning(m: usize) -> FerrotorchResult<Tensor<f64>> {
    hann(m)
}

/// Kaiser window of length `m` with shape parameter `beta`.
///
/// `beta = 0` reduces to a rectangular window; common engineering values
/// are `beta = 8.6` (≈Blackman main-lobe width with -65 dB sidelobes) and
/// `beta = 14` (very low sidelobes).
///
/// Mirrors `torch.signal.windows.kaiser` / `numpy.kaiser`.
pub fn kaiser(m: usize, beta: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::kaiser(m, beta).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

// ---------------------------------------------------------------------------
// SciPy-extended windows (added in ferray-window 0.3.1)
//
// These complete the `torch.signal.windows` surface beyond the 5 NumPy-core
// windows above. All return CPU-resident `Tensor<f64>`; user moves to device
// explicitly with `.to(Device::Cuda(_))` per the same /rust-gpu-discipline
// rationale.
// ---------------------------------------------------------------------------

/// Half-cycle cosine ("sine") window of length `m`.
///
/// `w(n) = sin(pi (n + 0.5) / m)`. Mirrors `torch.signal.windows.cosine`.
pub fn cosine(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::cosine(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Exponentially-decaying window centred on `center` with time-constant `tau`.
///
/// `w(n) = exp(-|n - center| / tau)`. If `center` is `None`, defaults to
/// the geometric centre `(m - 1) / 2`. `tau` must be positive and finite.
///
/// Mirrors `torch.signal.windows.exponential` /
/// `scipy.signal.windows.exponential`.
pub fn exponential(m: usize, center: Option<f64>, tau: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::exponential(m, center, tau).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Gaussian window of length `m` with standard deviation `std`.
///
/// `w(n) = exp(-((n - (m-1)/2) / std)^2 / 2)`. `std` must be positive and
/// finite. Mirrors `torch.signal.windows.gaussian`.
pub fn gaussian(m: usize, std: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::gaussian(m, std).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Generalised cosine-sum window:
/// `w(n) = Σ_k (-1)^k a[k] cos(2π k n / (m - 1))`.
///
/// Setting `coeffs = [0.5, 0.5]` recovers the Hann window;
/// `[0.42, 0.5, 0.08]` recovers classical Blackman. Mirrors
/// `torch.signal.windows.general_cosine`.
pub fn general_cosine(m: usize, coeffs: &[f64]) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::general_cosine(m, coeffs).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Generalised Hamming window: `w(n) = α - (1 - α) cos(2π n / (m - 1))`.
///
/// `alpha = 0.5` recovers Hann; `alpha = 0.54` recovers NumPy-style
/// Hamming. Mirrors `torch.signal.windows.general_hamming`.
pub fn general_hamming(m: usize, alpha: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::general_hamming(m, alpha).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Nuttall window: 4-term minimum-derivative Blackman-Harris.
///
/// Coefficients `[0.3635819, 0.4891775, 0.1365995, 0.0106411]` (Nuttall
/// 1981, "minimum 4-term Blackman-Harris with continuous first
/// derivative"). Mirrors `torch.signal.windows.nuttall`.
pub fn nuttall(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::nuttall(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Parzen (de la Vallée Poussin) window: a piecewise-cubic B-spline.
///
/// Mirrors `torch.signal.windows.parzen`.
pub fn parzen(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::parzen(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Taylor window with `nbar` near-side-lobes design and sidelobe-level
/// (`sll`, in **positive** dB).
///
/// Common defaults: `nbar = 4`, `sll = 30`, `norm = true`. When `norm`
/// is true the window is rescaled so the centre sample is `1`. Mirrors
/// `torch.signal.windows.taylor`.
pub fn taylor(m: usize, nbar: usize, sll: f64, norm: bool) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::taylor(m, nbar, sll, norm).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

/// Tukey (cosine-tapered) window with taper ratio `alpha` ∈ [0, 1].
///
/// `alpha = 0` is rectangular; `alpha = 1` is Hann. Mirrors
/// `torch.signal.windows.tukey`.
pub fn tukey(m: usize, alpha: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::tukey(m, alpha).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}

// ---------------------------------------------------------------------------
// Internal: ferray Array<f64, Ix1> -> ferrotorch Tensor<f64>
// ---------------------------------------------------------------------------

fn array_to_tensor(
    arr: ferray_core::Array<f64, ferray_core::Ix1>,
    m: usize,
) -> FerrotorchResult<Tensor<f64>> {
    // Consume `arr` to silence clippy::needless_pass_by_value while still
    // funneling through the standard contiguous-vec path. `into_iter()` on
    // an owned `Array` yields owned `f64` so no copy is required.
    let data: Vec<f64> = arr.into_iter().collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![m], false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ----- length / shape sanity -------------------------------------------

    #[test]
    fn bartlett_length() {
        let w = bartlett(8).unwrap();
        assert_eq!(w.shape(), &[8]);
        assert_eq!(w.device(), crate::Device::Cpu);
    }

    #[test]
    fn blackman_length() {
        let w = blackman(16).unwrap();
        assert_eq!(w.shape(), &[16]);
    }

    #[test]
    fn hann_length() {
        let w = hann(32).unwrap();
        assert_eq!(w.shape(), &[32]);
    }

    #[test]
    fn hamming_length() {
        let w = hamming(64).unwrap();
        assert_eq!(w.shape(), &[64]);
    }

    #[test]
    fn kaiser_length() {
        let w = kaiser(128, 14.0).unwrap();
        assert_eq!(w.shape(), &[128]);
    }

    // ----- mathematical properties -----------------------------------------

    #[test]
    fn bartlett_endpoints_are_zero() {
        // Bartlett is a triangular window with zero at both endpoints.
        let w = bartlett(8).unwrap();
        let d = w.data().unwrap();
        assert!(close(d[0], 0.0, 1e-12));
        assert!(close(d[d.len() - 1], 0.0, 1e-12));
    }

    #[test]
    fn bartlett_is_symmetric() {
        let w = bartlett(11).unwrap();
        let d = w.data().unwrap();
        let n = d.len();
        for i in 0..n {
            assert!(
                close(d[i], d[n - 1 - i], 1e-12),
                "bartlett not symmetric at {i}: {} vs {}",
                d[i],
                d[n - 1 - i]
            );
        }
    }

    #[test]
    fn hann_endpoints_are_zero() {
        // Hann (a.k.a. hanning) has zero endpoints by construction.
        let w = hann(16).unwrap();
        let d = w.data().unwrap();
        assert!(close(d[0], 0.0, 1e-12));
        assert!(close(d[d.len() - 1], 0.0, 1e-12));
    }

    #[test]
    fn hann_peak_is_one() {
        // The Hann window peaks at 1 in the middle (for odd m exactly,
        // for even m at the two centre samples nearly).
        let w = hann(11).unwrap();
        let d = w.data().unwrap();
        assert!(close(d[d.len() / 2], 1.0, 1e-12));
    }

    #[test]
    fn hamming_endpoints_match_alpha_minus_beta() {
        // Hamming endpoints are 0.08 (= 0.54 - 0.46), not zero.
        let w = hamming(8).unwrap();
        let d = w.data().unwrap();
        // Slack in case ferray uses 0.54/0.46 vs 25/46/21/46 conventions.
        assert!((d[0] - 0.08).abs() < 0.02);
        assert!((d[d.len() - 1] - 0.08).abs() < 0.02);
    }

    #[test]
    fn blackman_is_symmetric() {
        let w = blackman(15).unwrap();
        let d = w.data().unwrap();
        let n = d.len();
        for i in 0..n {
            assert!(close(d[i], d[n - 1 - i], 1e-12));
        }
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        // Kaiser with beta=0 collapses to a rectangular window: all 1.0.
        let w = kaiser(16, 0.0).unwrap();
        let d = w.data().unwrap();
        for &v in d {
            assert!(close(v, 1.0, 1e-12), "expected 1.0, got {v}");
        }
    }

    #[test]
    fn kaiser_peak_centre() {
        // Kaiser peaks at the centre.
        let w = kaiser(11, 8.6).unwrap();
        let d = w.data().unwrap();
        let mid = d[d.len() / 2];
        for (i, &v) in d.iter().enumerate() {
            assert!(
                v <= mid + 1e-12,
                "kaiser sample {i}={v} exceeds centre {mid}",
            );
        }
    }

    #[test]
    // reason: hanning() is a deliberate alias of hann() and the test pins
    // that they emit byte-identical buffers; any drift means the alias broke
    // and equality (not an epsilon) is exactly the right check.
    #[allow(clippy::float_cmp)]
    fn hanning_is_alias_for_hann() {
        let a = hann(13).unwrap();
        let b = hanning(13).unwrap();
        let ad = a.data().unwrap();
        let bd = b.data().unwrap();
        assert_eq!(ad.len(), bd.len());
        for i in 0..ad.len() {
            // hann and hanning compute the exact same values; use bit-exact
            // equality. (Strict `<` against 0.0 in `close` would reject
            // even bit-equal values.)
            assert_eq!(ad[i], bd[i]);
        }
    }

    #[test]
    fn output_lives_on_cpu() {
        // GPU discipline: every window function returns CPU storage.
        // The user is responsible for moving to device with .to(Device::Cuda).
        for w in [
            bartlett(4).unwrap(),
            blackman(4).unwrap(),
            hamming(4).unwrap(),
            hann(4).unwrap(),
            hanning(4).unwrap(),
            kaiser(4, 5.0).unwrap(),
            cosine(4).unwrap(),
            exponential(4, None, 1.0).unwrap(),
            gaussian(4, 1.0).unwrap(),
            general_cosine(4, &[0.5, 0.5]).unwrap(),
            general_hamming(4, 0.54).unwrap(),
            nuttall(4).unwrap(),
            parzen(4).unwrap(),
            taylor(8, 4, 30.0, true).unwrap(),
            tukey(4, 0.5).unwrap(),
        ] {
            assert_eq!(w.device(), crate::Device::Cpu);
        }
    }

    // ----- SciPy-extended windows (cosine/exponential/gaussian/...) -------

    #[test]
    fn cosine_length_and_symmetry() {
        let w = cosine(8).unwrap();
        assert_eq!(w.shape(), &[8]);
        let d = w.data().unwrap();
        for i in 0..4 {
            assert!(close(d[i], d[7 - i], 1e-14));
        }
    }

    #[test]
    fn exponential_default_centre_is_symmetric() {
        let w = exponential(8, None, 1.0).unwrap();
        let d = w.data().unwrap();
        for i in 0..4 {
            assert!(close(d[i], d[7 - i], 1e-14));
        }
    }

    #[test]
    fn exponential_rejects_invalid_tau() {
        // Should propagate the underlying ferray-window InvalidValue error.
        assert!(exponential(8, None, 0.0).is_err());
        assert!(exponential(8, None, -1.0).is_err());
    }

    #[test]
    fn gaussian_centre_is_one_for_odd_m() {
        let w = gaussian(11, 2.0).unwrap();
        let d = w.data().unwrap();
        assert!(close(d[5], 1.0, 1e-14));
    }

    #[test]
    fn gaussian_known_value() {
        // gaussian(7, 1) at n=4: z = 1, exp(-0.5) = 0.6065...
        let w = gaussian(7, 1.0).unwrap();
        assert!(close(w.data().unwrap()[4], (-0.5_f64).exp(), 1e-14));
    }

    #[test]
    fn gaussian_rejects_nonpositive_std() {
        assert!(gaussian(8, 0.0).is_err());
        assert!(gaussian(8, -1.0).is_err());
    }

    #[test]
    fn general_cosine_with_hann_coeffs_matches_hann() {
        let m = 9;
        let gc = general_cosine(m, &[0.5, 0.5]).unwrap();
        let hn = hann(m).unwrap();
        for (a, b) in gc.data().unwrap().iter().zip(hn.data().unwrap().iter()) {
            assert!(close(*a, *b, 1e-14));
        }
    }

    #[test]
    fn general_cosine_with_blackman_coeffs_matches_blackman() {
        let m = 9;
        let gc = general_cosine(m, &[0.42, 0.5, 0.08]).unwrap();
        let bk = blackman(m).unwrap();
        for (a, b) in gc.data().unwrap().iter().zip(bk.data().unwrap().iter()) {
            assert!(close(*a, *b, 1e-12));
        }
    }

    #[test]
    fn general_cosine_rejects_empty_coeffs() {
        assert!(general_cosine(8, &[]).is_err());
    }

    #[test]
    fn general_hamming_alpha_half_matches_hann() {
        let m = 9;
        let gh = general_hamming(m, 0.5).unwrap();
        let hn = hann(m).unwrap();
        for (a, b) in gh.data().unwrap().iter().zip(hn.data().unwrap().iter()) {
            assert!(close(*a, *b, 1e-14));
        }
    }

    #[test]
    fn general_hamming_alpha_054_matches_hamming() {
        let m = 9;
        let gh = general_hamming(m, 0.54).unwrap();
        let hm = hamming(m).unwrap();
        for (a, b) in gh.data().unwrap().iter().zip(hm.data().unwrap().iter()) {
            assert!(close(*a, *b, 1e-14));
        }
    }

    #[test]
    fn nuttall_length_and_symmetry() {
        let m = 33;
        let w = nuttall(m).unwrap();
        let d = w.data().unwrap();
        for i in 0..m / 2 {
            assert!(close(d[i], d[m - 1 - i], 1e-14));
        }
    }

    #[test]
    fn nuttall_endpoints_are_small() {
        let w = nuttall(64).unwrap();
        let d = w.data().unwrap();
        assert!(d[0].abs() < 1e-2);
        assert!(d[d.len() - 1].abs() < 1e-2);
    }

    #[test]
    fn parzen_centre_is_one() {
        let w = parzen(13).unwrap();
        assert!(close(w.data().unwrap()[6], 1.0, 1e-14));
    }

    #[test]
    fn parzen_is_symmetric() {
        let m = 21;
        let w = parzen(m).unwrap();
        let d = w.data().unwrap();
        for i in 0..m / 2 {
            assert!(close(d[i], d[m - 1 - i], 1e-14));
        }
    }

    #[test]
    fn taylor_normalised_centre_is_one() {
        let w = taylor(33, 4, 30.0, true).unwrap();
        assert!(close(w.data().unwrap()[16], 1.0, 1e-12));
    }

    #[test]
    fn taylor_is_symmetric() {
        let m = 33;
        let w = taylor(m, 4, 30.0, true).unwrap();
        let d = w.data().unwrap();
        for i in 0..m / 2 {
            assert!(close(d[i], d[m - 1 - i], 1e-12));
        }
    }

    #[test]
    fn taylor_rejects_invalid_args() {
        assert!(taylor(8, 0, 30.0, true).is_err());
        assert!(taylor(8, 4, f64::NAN, true).is_err());
    }

    #[test]
    fn tukey_alpha_zero_is_rectangular() {
        let w = tukey(8, 0.0).unwrap();
        for &v in w.data().unwrap() {
            assert!(close(v, 1.0, 1e-14));
        }
    }

    #[test]
    fn tukey_alpha_one_matches_hann() {
        let m = 9;
        let tk = tukey(m, 1.0).unwrap();
        let hn = hann(m).unwrap();
        for (a, b) in tk.data().unwrap().iter().zip(hn.data().unwrap().iter()) {
            assert!(close(*a, *b, 1e-12));
        }
    }

    #[test]
    fn tukey_centre_is_one() {
        let m = 21;
        let w = tukey(m, 0.5).unwrap();
        assert!(close(w.data().unwrap()[m / 2], 1.0, 1e-14));
    }

    #[test]
    fn tukey_rejects_invalid_alpha() {
        assert!(tukey(8, -0.1).is_err());
        assert!(tukey(8, 1.1).is_err());
        assert!(tukey(8, f64::NAN).is_err());
    }
}
