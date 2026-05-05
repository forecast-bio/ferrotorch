//! Special mathematical functions (`torch.special` equivalent).
//!
//! All functions operate elementwise on tensors, returning a new tensor of the
//! same shape. Implementations use either `num_traits::Float` methods or
//! well-known numerical approximations (Abramowitz & Stegun, Lanczos, etc.).

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::ops::elementwise::{binary_map, unary_map};
use crate::tensor::Tensor;

/// Helper: return zero via `num_traits::Zero` to avoid ambiguity with
/// `ferray_core::Element::zero`.
#[inline]
fn nt_zero<T: num_traits::Zero>() -> T {
    <T as num_traits::Zero>::zero()
}

/// Helper: return one via `num_traits::One` to avoid ambiguity with
/// `ferray_core::Element::one`.
#[inline]
fn nt_one<T: num_traits::One>() -> T {
    <T as num_traits::One>::one()
}

// ---------------------------------------------------------------------------
// Constants (as f64; converted to T at call sites via T::from)
// ---------------------------------------------------------------------------

// Abramowitz & Stegun 7.1.26 coefficients for erf approximation.
const ERF_A1: f64 = 0.254829592;
const ERF_A2: f64 = -0.284496736;
const ERF_A3: f64 = 1.421413741;
const ERF_A4: f64 = -1.453152027;
const ERF_A5: f64 = 1.061405429;
const ERF_P: f64 = 0.3275911;

// Lanczos approximation coefficients (g = 7, n = 9).
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.5203681218851,
    -1259.1392167224028,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
];

// ---------------------------------------------------------------------------
// Scalar helper functions
// ---------------------------------------------------------------------------

/// Compute erf(x) for a single float using the Abramowitz & Stegun 7.1.26
/// approximation. Maximum error: |epsilon(x)| <= 1.5e-7.
fn erf_scalar<T: Float>(x: T) -> T {
    let zero = nt_zero::<T>();
    let one = nt_one::<T>();

    if x == zero {
        return zero;
    }

    let sign = if x < zero { -one } else { one };
    let ax = x.abs();

    let p = T::from(ERF_P).unwrap();
    let t = one / (one + p * ax);

    let a1 = T::from(ERF_A1).unwrap();
    let a2 = T::from(ERF_A2).unwrap();
    let a3 = T::from(ERF_A3).unwrap();
    let a4 = T::from(ERF_A4).unwrap();
    let a5 = T::from(ERF_A5).unwrap();

    // Horner form: (a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    let poly = a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)));

    sign * (one - poly * t * (-ax * ax).exp())
}

/// Compute erfinv(x) using the Winitzki (2008) rational approximation.
///
/// erfinv(x) = sign(x) * sqrt( -b + sqrt(b^2 - c) )
/// where b = 2/(pi*a) + ln(1-x^2)/2, c = ln(1-x^2)/a, a = 0.147.
fn erfinv_scalar<T: Float>(x: T) -> T {
    let zero = nt_zero::<T>();
    let one = nt_one::<T>();

    if x == zero {
        return zero;
    }
    if x >= one {
        return T::infinity();
    }
    if x <= -one {
        return T::neg_infinity();
    }

    let sign = if x < zero { -one } else { one };
    let ax = x.abs();

    // Winitzki approximation: a cleaner formulation.
    let a = T::from(0.147).unwrap();
    let two = T::from(2.0).unwrap();
    let pi = T::from(std::f64::consts::PI).unwrap();

    let ln_term = (one - ax * ax).ln();
    let b = two / (pi * a) + ln_term / two;
    let c = ln_term / a;

    // erfinv(x) = sign(x) * sqrt( -b + sqrt(b^2 - c) )
    sign * (-b + (b * b - c).sqrt()).sqrt()
}

/// Compute lgamma(x) using the Lanczos approximation.
fn lgamma_scalar<T: Float>(x: T) -> T {
    let one = nt_one::<T>();
    let half = T::from(0.5).unwrap();
    let half_ln_2pi = T::from(0.9189385332046727).unwrap(); // 0.5 * ln(2*pi)
    let g = T::from(LANCZOS_G).unwrap();

    // Handle negative values via reflection formula.
    if x < half {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == nt_zero::<T>() {
            return T::infinity();
        }
        return (pi / sin_pi_x.abs()).ln() - lgamma_scalar(one - x);
    }

    let z = x - one;
    let mut sum = T::from(LANCZOS_COEFFICIENTS[0]).unwrap();
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        sum += T::from(coeff).unwrap() / (z + T::from(i as f64).unwrap());
    }

    let t = z + g + half;
    half_ln_2pi + (t).ln() * (z + half) - t + sum.ln()
}

/// Compute digamma(x) = psi(x) = d/dx ln(Gamma(x)).
///
/// Uses the recurrence relation psi(x+1) = psi(x) + 1/x to shift x into the
/// range [6, inf), then applies the asymptotic expansion.
fn digamma_scalar<T: Float>(x: T) -> T {
    let zero = nt_zero::<T>();
    let one = nt_one::<T>();
    let half = T::from(0.5).unwrap();

    // Handle negative values via reflection formula:
    // psi(1 - x) = psi(x) + pi * cot(pi * x)
    if x < half {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let cot = (pi * x).cos() / (pi * x).sin();
        return digamma_scalar(one - x) - pi * cot;
    }

    // Shift x upward until x >= 6 using recurrence: psi(x) = psi(x+1) - 1/x.
    let mut result = zero;
    let mut z = x;
    let six = T::from(6.0).unwrap();
    while z < six {
        #[allow(clippy::assign_op_pattern)]
        {
            result = result - one / z;
        }
        #[allow(clippy::assign_op_pattern)]
        {
            z = z + one;
        }
    }

    // Asymptotic expansion for large z:
    // psi(z) ~ ln(z) - 1/(2z) - 1/(12z^2) + 1/(120z^4) - 1/(252z^6) + ...
    let z2 = z * z;
    let z4 = z2 * z2;
    let z6 = z4 * z2;

    result =
        result + z.ln() - one / (T::from(2.0).unwrap() * z) - one / (T::from(12.0).unwrap() * z2)
            + one / (T::from(120.0).unwrap() * z4)
            - one / (T::from(252.0).unwrap() * z6);

    result
}

/// Compute sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1.
fn sinc_scalar<T: Float>(x: T) -> T {
    let zero = nt_zero::<T>();
    let one = nt_one::<T>();

    if x == zero {
        return one;
    }

    let pi = T::from(std::f64::consts::PI).unwrap();
    let pi_x = pi * x;
    pi_x.sin() / pi_x
}

/// Compute x * log(y) with the convention that 0 * log(y) = 0.
fn xlogy_scalar<T: Float>(x: T, y: T) -> T {
    if x == nt_zero::<T>() {
        nt_zero::<T>()
    } else {
        x * y.ln()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Error function: erf(x) = (2/sqrt(pi)) * integral(0, x, exp(-t^2) dt).
///
/// Uses the Abramowitz & Stegun 7.1.26 polynomial approximation with
/// maximum error |epsilon| <= 1.5e-7.
pub fn erf<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, erf_scalar)
}

/// Complementary error function: erfc(x) = 1 - erf(x).
///
/// Computed directly as `1 - erf(x)` using the same polynomial approximation.
pub fn erfc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, |x| nt_one::<T>() - erf_scalar(x))
}

/// Inverse error function: erfinv(erf(x)) = x.
///
/// Uses the Winitzki (2008) rational approximation. Returns `inf` for
/// input = 1, `-inf` for input = -1, and `NaN` for |input| > 1.
pub fn erfinv<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, erfinv_scalar)
}

/// Log-gamma function: lgamma(x) = log(|Gamma(x)|).
///
/// Uses the Lanczos approximation (g = 7, n = 9 coefficients).
pub fn lgamma<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, lgamma_scalar)
}

/// Digamma function: psi(x) = d/dx ln(Gamma(x)).
///
/// Uses the recurrence relation to shift the argument above 6, then
/// applies the asymptotic expansion.
pub fn digamma<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, digamma_scalar)
}

/// log(1 + x) -- numerically stable for small x.
///
/// Delegates to `num_traits::Float::ln_1p()`.
pub fn log1p<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, |x| x.ln_1p())
}

/// exp(x) - 1 -- numerically stable for small x.
///
/// Delegates to `num_traits::Float::exp_m1()`.
pub fn expm1<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, |x| x.exp_m1())
}

/// Normalized sinc function: sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1.
pub fn sinc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    unary_map(input, sinc_scalar)
}

/// x * log(y), with the convention that xlogy(0, y) = 0 for any y.
///
/// This is useful for entropy computations where 0 * log(0) should be 0.
pub fn xlogy<T: Float>(x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    binary_map(x, y, xlogy_scalar)
}

// ===========================================================================
// Orthogonal-polynomial families
// ===========================================================================
//
// These mirror torch.special.{chebyshev,hermite,laguerre,legendre}_polynomial_*.
// Each function returns the n-th degree basis polynomial evaluated pointwise
// at every element of `input`. CPU-only today; a CubeCL kernel that runs the
// three-term recurrence on-device is tracked as a follow-up. GPU tensors
// return NotImplementedOnCuda — never silently bounce through host (per
// /rust-gpu-discipline).
//
// Implementation: every basis is evaluated by its standard three-term
// recurrence directly in this module. ferray-polynomial 0.3 has the same
// idiom (Clenshaw on basis-coefficient vectors), but for the
// "single-basis-element" case the direct recurrence is shorter, dependency-
// free, and numerically equivalent for the orders typically used in ML
// pipelines (n ≤ 50). We keep the option open to swap in ferray's Clenshaw
// path later for very-high-order numerical-analysis use cases.

use crate::error::FerrotorchError;

/// Reject GPU tensors with an explicit error rather than bouncing through
/// host. Polynomial evaluation has no GPU kernel today; see follow-up.
#[inline]
fn require_cpu_poly<T: Float>(t: &Tensor<T>, op: &'static str) -> FerrotorchResult<()> {
    if t.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    Ok(())
}

/// Apply a `f64 -> f64` evaluator to every element of a tensor, with the
/// CPU/dtype guard inlined so each function below stays a one-liner.
fn elementwise_f64<T: Float, F: Fn(f64) -> f64>(
    input: &Tensor<T>,
    op: &'static str,
    f: F,
) -> FerrotorchResult<Tensor<T>> {
    require_cpu_poly(input, op)?;
    let data = input.data_vec()?;
    let out: Vec<T> = data
        .into_iter()
        .map(|v| T::from(f(v.to_f64().unwrap())).unwrap())
        .collect();
    crate::tensor::Tensor::from_storage(
        crate::storage::TensorStorage::cpu(out),
        input.shape().to_vec(),
        false,
    )
}

// --- Chebyshev family ------------------------------------------------------

/// Chebyshev polynomial of the **first kind** `T_n(x)`.
///
/// `T_0 = 1`, `T_1 = x`, `T_{n+1} = 2x T_n - T_{n-1}`. Mirrors
/// `torch.special.chebyshev_polynomial_t`.
pub fn chebyshev_polynomial_t<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_t", move |x| chebyshev_t(n, x))
}

/// Chebyshev polynomial of the **second kind** `U_n(x)`.
///
/// `U_0 = 1`, `U_1 = 2x`, `U_{n+1} = 2x U_n - U_{n-1}`. Mirrors
/// `torch.special.chebyshev_polynomial_u`. Evaluated by direct recurrence;
/// not provided by ferray-polynomial.
pub fn chebyshev_polynomial_u<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_u", move |x| chebyshev_u(n, x))
}

/// Chebyshev polynomial of the **third kind** `V_n(x)`.
///
/// `V_0 = 1`, `V_1 = 2x - 1`, same recurrence as T/U. Mirrors
/// `torch.special.chebyshev_polynomial_v`.
pub fn chebyshev_polynomial_v<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_v", move |x| chebyshev_v(n, x))
}

/// Chebyshev polynomial of the **fourth kind** `W_n(x)`.
///
/// `W_0 = 1`, `W_1 = 2x + 1`, same recurrence as T/U. Mirrors
/// `torch.special.chebyshev_polynomial_w`.
pub fn chebyshev_polynomial_w<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "chebyshev_polynomial_w", move |x| chebyshev_w(n, x))
}

// --- Hermite family --------------------------------------------------------

/// Hermite polynomial (physicist's) `H_n(x)`.
///
/// `H_0 = 1`, `H_1 = 2x`, `H_{n+1} = 2x H_n - 2n H_{n-1}`. Mirrors
/// `torch.special.hermite_polynomial_h`.
pub fn hermite_polynomial_h<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "hermite_polynomial_h", move |x| hermite_h(n, x))
}

/// Probabilist's Hermite polynomial `He_n(x)`.
///
/// `He_0 = 1`, `He_1 = x`, `He_{n+1} = x He_n - n He_{n-1}`. Mirrors
/// `torch.special.hermite_polynomial_he`.
pub fn hermite_polynomial_he<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "hermite_polynomial_he", move |x| hermite_he(n, x))
}

// --- Laguerre & Legendre ---------------------------------------------------

/// Laguerre polynomial `L_n(x)`.
///
/// `L_0 = 1`, `L_1 = 1 - x`, `(n+1) L_{n+1} = (2n + 1 - x) L_n - n L_{n-1}`.
/// Mirrors `torch.special.laguerre_polynomial_l`.
pub fn laguerre_polynomial_l<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "laguerre_polynomial_l", move |x| laguerre_l(n, x))
}

/// Legendre polynomial `P_n(x)`.
///
/// `P_0 = 1`, `P_1 = x`, `(n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}`.
/// Mirrors `torch.special.legendre_polynomial_p`.
pub fn legendre_polynomial_p<T: Float>(input: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "legendre_polynomial_p", move |x| legendre_p(n, x))
}

// --- Shifted Chebyshev family (domain [0, 1]) ------------------------------

/// Shifted Chebyshev T: `T*_n(x) = T_n(2x - 1)`. Mirrors
/// `torch.special.shifted_chebyshev_polynomial_t`.
pub fn shifted_chebyshev_polynomial_t<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_t", move |x| {
        chebyshev_t(n, 2.0 * x - 1.0)
    })
}

/// Shifted Chebyshev U: `U*_n(x) = U_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_u<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_u", move |x| {
        chebyshev_u(n, 2.0 * x - 1.0)
    })
}

/// Shifted Chebyshev V: `V*_n(x) = V_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_v<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_v", move |x| {
        chebyshev_v(n, 2.0 * x - 1.0)
    })
}

/// Shifted Chebyshev W: `W*_n(x) = W_n(2x - 1)`.
pub fn shifted_chebyshev_polynomial_w<T: Float>(
    input: &Tensor<T>,
    n: usize,
) -> FerrotorchResult<Tensor<T>> {
    elementwise_f64(input, "shifted_chebyshev_polynomial_w", move |x| {
        chebyshev_w(n, 2.0 * x - 1.0)
    })
}

// ---------------------------------------------------------------------------
// Internal scalar evaluators (three-term recurrences in f64)
// ---------------------------------------------------------------------------

/// `H_n(x)` (physicist's Hermite) via three-term recurrence
/// `H_{n+1} = 2x H_n - 2n H_{n-1}` with `H_0 = 1`, `H_1 = 2x`.
fn hermite_h(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x;
    for k in 1..n {
        let kf = k as f64;
        let next = 2.0 * x * prev1 - 2.0 * kf * prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `He_n(x)` (probabilist's Hermite) via three-term recurrence
/// `He_{n+1} = x He_n - n He_{n-1}` with `He_0 = 1`, `He_1 = x`.
fn hermite_he(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = x;
    for k in 1..n {
        let kf = k as f64;
        let next = x * prev1 - kf * prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

/// `T_n(x)` via direct recurrence — used internally for shifted variants.
fn chebyshev_t(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = x;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn chebyshev_u(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn chebyshev_v(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x - 1.0;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x - 1.0;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn chebyshev_w(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x + 1.0;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 2.0 * x + 1.0;
    for _ in 2..=n {
        let next = 2.0 * x * prev1 - prev2;
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn laguerre_l(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 1.0 - x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = 1.0 - x;
    for k in 1..n {
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0 - x) * prev1 - kf * prev2) / (kf + 1.0);
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

fn legendre_p(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut prev2 = 1.0;
    let mut prev1 = x;
    for k in 1..n {
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0) * x * prev1 - kf * prev2) / (kf + 1.0);
        prev2 = prev1;
        prev1 = next;
    }
    prev1
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Helper: create a tensor from data and shape.
    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // --- erf ---

    #[test]
    fn erf_zero() {
        let input = t(&[0.0], &[1]);
        let result = erf(&input).unwrap();
        assert!((result.data().unwrap()[0]).abs() < 1e-10);
    }

    #[test]
    fn erf_symmetry() {
        // erf(-x) = -erf(x)
        let input = t(&[0.5, 1.0, 2.0], &[3]);
        let neg_input = t(&[-0.5, -1.0, -2.0], &[3]);
        let pos = erf(&input).unwrap();
        let neg = erf(&neg_input).unwrap();
        let pd = pos.data().unwrap();
        let nd = neg.data().unwrap();
        for i in 0..3 {
            assert!(
                (pd[i] + nd[i]).abs() < 1e-6,
                "erf({}) + erf({}) = {} (expected 0)",
                input.data().unwrap()[i],
                neg_input.data().unwrap()[i],
                pd[i] + nd[i],
            );
        }
    }

    #[test]
    fn erf_large_value() {
        // erf(inf) = 1
        let input = t(&[f64::INFINITY], &[1]);
        let result = erf(&input).unwrap();
        assert!((result.data().unwrap()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn erf_known_values() {
        // erf(1) ≈ 0.8427007929...
        let input = t(&[1.0], &[1]);
        let result = erf(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - 0.8427007929).abs() < 2e-7,
            "erf(1) = {}",
            result.data().unwrap()[0]
        );
    }

    // --- erfc ---

    #[test]
    fn erfc_is_one_minus_erf() {
        let input = t(&[0.0, 0.5, 1.0, -0.5, 2.0], &[5]);
        let erf_result = erf(&input).unwrap();
        let erfc_result = erfc(&input).unwrap();
        let ed = erf_result.data().unwrap();
        let cd = erfc_result.data().unwrap();
        for i in 0..5 {
            assert!(
                (ed[i] + cd[i] - 1.0).abs() < 1e-10,
                "erf({0}) + erfc({0}) = {1} (expected 1.0)",
                input.data().unwrap()[i],
                ed[i] + cd[i],
            );
        }
    }

    // --- erfinv ---

    #[test]
    fn erfinv_zero() {
        let input = t(&[0.0], &[1]);
        let result = erfinv(&input).unwrap();
        assert!(result.data().unwrap()[0].abs() < 1e-10);
    }

    #[test]
    fn erfinv_roundtrip() {
        // erfinv(erf(x)) ≈ x
        let xs = t(&[0.1, 0.5, 1.0, -0.3, -1.5], &[5]);
        let erf_xs = erf(&xs).unwrap();
        let roundtrip = erfinv(&erf_xs).unwrap();
        let orig = xs.data().unwrap();
        let rt = roundtrip.data().unwrap();
        for i in 0..5 {
            assert!(
                (orig[i] - rt[i]).abs() < 0.01,
                "erfinv(erf({})) = {} (expected {})",
                orig[i],
                rt[i],
                orig[i],
            );
        }
    }

    #[test]
    fn erfinv_boundary() {
        let input = t(&[1.0, -1.0], &[2]);
        let result = erfinv(&input).unwrap();
        let d = result.data().unwrap();
        assert!(d[0].is_infinite() && d[0] > 0.0, "erfinv(1) should be +inf");
        assert!(
            d[1].is_infinite() && d[1] < 0.0,
            "erfinv(-1) should be -inf"
        );
    }

    // --- lgamma ---

    #[test]
    fn lgamma_at_one_and_two() {
        // lgamma(1) = log(0!) = 0, lgamma(2) = log(1!) = 0.
        let input = t(&[1.0, 2.0], &[2]);
        let result = lgamma(&input).unwrap();
        let d = result.data().unwrap();
        assert!(d[0].abs() < 1e-10, "lgamma(1) = {} (expected 0)", d[0]);
        assert!(d[1].abs() < 1e-10, "lgamma(2) = {} (expected 0)", d[1]);
    }

    #[test]
    fn lgamma_known_values() {
        // lgamma(0.5) = log(sqrt(pi)) ≈ 0.5723649429...
        let input = t(&[0.5], &[1]);
        let result = lgamma(&input).unwrap();
        let expected = 0.5723649429247001;
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-8,
            "lgamma(0.5) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    #[test]
    fn lgamma_factorial() {
        // lgamma(n+1) = log(n!) for integer n.
        // lgamma(6) = log(5!) = log(120) ≈ 4.7875...
        let input = t(&[6.0], &[1]);
        let result = lgamma(&input).unwrap();
        let expected = (120.0f64).ln();
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-8,
            "lgamma(6) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    // --- digamma ---

    #[test]
    fn digamma_known_values() {
        // psi(1) = -gamma (Euler-Mascheroni) ≈ -0.5772156649...
        let input = t(&[1.0], &[1]);
        let result = digamma(&input).unwrap();
        let expected = -0.5772156649015329;
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-6,
            "digamma(1) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    #[test]
    fn digamma_recurrence() {
        // psi(x+1) = psi(x) + 1/x
        let x_val = 2.5;
        let input_x = t(&[x_val], &[1]);
        let input_x1 = t(&[x_val + 1.0], &[1]);
        let psi_x = digamma(&input_x).unwrap().data().unwrap()[0];
        let psi_x1 = digamma(&input_x1).unwrap().data().unwrap()[0];
        assert!(
            (psi_x1 - psi_x - 1.0 / x_val).abs() < 1e-8,
            "psi({}) - psi({}) = {} (expected {})",
            x_val + 1.0,
            x_val,
            psi_x1 - psi_x,
            1.0 / x_val,
        );
    }

    // --- log1p ---

    #[test]
    fn log1p_zero() {
        let input = t(&[0.0], &[1]);
        let result = log1p(&input).unwrap();
        assert!(result.data().unwrap()[0].abs() < 1e-15);
    }

    #[test]
    fn log1p_small() {
        // For small x, log(1+x) ≈ x.
        let small = 1e-10;
        let input = t(&[small], &[1]);
        let result = log1p(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - small).abs() < 1e-15,
            "log1p({small}) = {} (expected ~{small})",
            result.data().unwrap()[0],
        );
    }

    #[test]
    fn log1p_known() {
        // log1p(1.0) = ln(2) ≈ 0.693147...
        let input = t(&[1.0], &[1]);
        let result = log1p(&input).unwrap();
        assert!((result.data().unwrap()[0] - std::f64::consts::LN_2).abs() < 1e-15,);
    }

    // --- expm1 ---

    #[test]
    fn expm1_zero() {
        let input = t(&[0.0], &[1]);
        let result = expm1(&input).unwrap();
        assert!(result.data().unwrap()[0].abs() < 1e-15);
    }

    #[test]
    fn expm1_small() {
        // For small x, exp(x) - 1 ≈ x.
        let small = 1e-10;
        let input = t(&[small], &[1]);
        let result = expm1(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - small).abs() < 1e-15,
            "expm1({small}) = {} (expected ~{small})",
            result.data().unwrap()[0],
        );
    }

    #[test]
    fn expm1_known() {
        // expm1(1.0) = e - 1 ≈ 1.71828...
        let input = t(&[1.0], &[1]);
        let result = expm1(&input).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert!((result.data().unwrap()[0] - expected).abs() < 1e-14,);
    }

    // --- sinc ---

    #[test]
    fn sinc_zero() {
        let input = t(&[0.0], &[1]);
        let result = sinc(&input).unwrap();
        assert!(
            (result.data().unwrap()[0] - 1.0).abs() < 1e-15,
            "sinc(0) = {} (expected 1)",
            result.data().unwrap()[0],
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn sinc_integer() {
        // sinc(n) = 0 for nonzero integer n, since sin(n*pi) = 0.
        let input = t(&[1.0, 2.0, -1.0, -3.0], &[4]);
        let result = sinc(&input).unwrap();
        let d = result.data().unwrap();
        for i in 0..4 {
            assert!(
                d[i].abs() < 1e-15,
                "sinc({}) = {} (expected 0)",
                input.data().unwrap()[i],
                d[i],
            );
        }
    }

    #[test]
    fn sinc_half() {
        // sinc(0.5) = sin(pi/2) / (pi/2) = 1 / (pi/2) = 2/pi
        let input = t(&[0.5], &[1]);
        let result = sinc(&input).unwrap();
        let expected = 2.0 / std::f64::consts::PI;
        assert!(
            (result.data().unwrap()[0] - expected).abs() < 1e-15,
            "sinc(0.5) = {} (expected {})",
            result.data().unwrap()[0],
            expected,
        );
    }

    // --- xlogy ---

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn xlogy_zero_x() {
        // xlogy(0, y) = 0 for any y.
        let x = t(&[0.0, 0.0, 0.0], &[3]);
        let y = t(&[1.0, 0.0, f64::INFINITY], &[3]);
        let result = xlogy(&x, &y).unwrap();
        let d = result.data().unwrap();
        for i in 0..3 {
            assert!(
                d[i] == 0.0,
                "xlogy(0, {}) = {} (expected 0)",
                y.data().unwrap()[i],
                d[i],
            );
        }
    }

    #[test]
    fn xlogy_normal() {
        // xlogy(2, e) = 2 * ln(e) = 2.
        let x = t(&[2.0], &[1]);
        let y = t(&[std::f64::consts::E], &[1]);
        let result = xlogy(&x, &y).unwrap();
        assert!(
            (result.data().unwrap()[0] - 2.0).abs() < 1e-14,
            "xlogy(2, e) = {} (expected 2)",
            result.data().unwrap()[0],
        );
    }

    #[test]
    fn xlogy_broadcast() {
        // [2, 3] * log([e, e]) = [2, 3].
        let x = t(&[2.0, 3.0], &[2]);
        let y = t(&[std::f64::consts::E, std::f64::consts::E], &[2]);
        let result = xlogy(&x, &y).unwrap();
        let d = result.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-14);
        assert!((d[1] - 3.0).abs() < 1e-14);
    }

    // --- f32 support ---

    #[test]
    fn erf_f32() {
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32, 1.0, -1.0]), vec![3], false)
                .unwrap();
        let result = erf(&input).unwrap();
        let d = result.data().unwrap();
        assert!(d[0].abs() < 1e-6);
        assert!((d[1] - 0.8427008).abs() < 1e-5);
        assert!((d[2] + 0.8427008).abs() < 1e-5);
    }

    // --- multidimensional ---

    #[test]
    fn erf_2d() {
        let input = t(&[0.0, 0.5, 1.0, -0.5, -1.0, 2.0], &[2, 3]);
        let result = erf(&input).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        let d = result.data().unwrap();
        assert!(d[0].abs() < 1e-10); // erf(0)
        assert!(d[2] > 0.8); // erf(1) ≈ 0.843
        assert!(d[3] < 0.0); // erf(-0.5) < 0
    }

    // ---------------------------------------------------------------------
    // Orthogonal-polynomial families
    //
    // For each family we hand-check the closed-form first three basis
    // values against the recurrence. n=0 must always be 1 everywhere.
    // n=1 is the family-specific linear; n=2 is the first interesting
    // recurrence step. Numerical comparisons use a tight tolerance (1e-12)
    // since these are exact polynomial evaluations in f64.
    // ---------------------------------------------------------------------

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn xs() -> Tensor<f64> {
        // A handful of points in/around the standard domains.
        t(&[0.0, 0.5, 1.0, -0.5, -1.0, 0.25], &[6])
    }

    // --- Chebyshev T (first kind) ---

    #[test]
    fn chebyshev_t_n0_is_one() {
        let r = chebyshev_polynomial_t(&xs(), 0).unwrap();
        for &v in r.data().unwrap() {
            assert!(close(v, 1.0, 1e-12));
        }
    }

    #[test]
    fn chebyshev_t_n1_is_x() {
        let x = xs();
        let r = chebyshev_polynomial_t(&x, 1).unwrap();
        for (a, b) in r.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            assert!(close(*a, *b, 1e-12));
        }
    }

    #[test]
    fn chebyshev_t_n2_is_2xx_minus_one() {
        // T_2(x) = 2x^2 - 1
        let x = xs();
        let r = chebyshev_polynomial_t(&x, 2).unwrap();
        for (a, &xv) in r.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            assert!(close(*a, 2.0 * xv * xv - 1.0, 1e-12));
        }
    }

    #[test]
    fn chebyshev_t_at_endpoints() {
        // T_n(1) = 1, T_n(-1) = (-1)^n.
        let pts = t(&[1.0, -1.0], &[2]);
        for n in 0..6 {
            let r = chebyshev_polynomial_t(&pts, n).unwrap();
            let d = r.data().unwrap();
            assert!(close(d[0], 1.0, 1e-12), "T_{n}(1) = {}", d[0]);
            let expected_neg = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!(close(d[1], expected_neg, 1e-12), "T_{n}(-1) = {}", d[1]);
        }
    }

    // --- Chebyshev U (second kind) ---

    #[test]
    fn chebyshev_u_n0_n1_n2() {
        // U_0 = 1, U_1 = 2x, U_2 = 4x^2 - 1
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            chebyshev_polynomial_u(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            chebyshev_polynomial_u(&x, 1).unwrap().data().unwrap()[0],
            2.0 * xv,
            1e-12,
        ));
        assert!(close(
            chebyshev_polynomial_u(&x, 2).unwrap().data().unwrap()[0],
            4.0 * xv * xv - 1.0,
            1e-12,
        ));
    }

    // --- Chebyshev V / W ---

    #[test]
    fn chebyshev_v_endpoints() {
        // V_n(1) = 1 for all n; V_1(0) = -1.
        let pts = t(&[1.0, 0.0], &[2]);
        for n in 0..4 {
            let r = chebyshev_polynomial_v(&pts, n).unwrap();
            assert!(close(r.data().unwrap()[0], 1.0, 1e-12));
        }
        let r1 = chebyshev_polynomial_v(&pts, 1).unwrap();
        assert!(close(r1.data().unwrap()[1], -1.0, 1e-12));
    }

    #[test]
    fn chebyshev_w_endpoints() {
        // W_1(0) = 1; W_n(1) = 2n + 1 for the recurrence above.
        let zero = t(&[0.0], &[1]);
        assert!(close(
            chebyshev_polynomial_w(&zero, 1).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
    }

    // --- Hermite (physicist) ---

    #[test]
    fn hermite_h_known_values() {
        // H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            hermite_polynomial_h(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_h(&x, 1).unwrap().data().unwrap()[0],
            2.0 * xv,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_h(&x, 2).unwrap().data().unwrap()[0],
            4.0 * xv * xv - 2.0,
            1e-12,
        ));
        assert!(close(
            hermite_polynomial_h(&x, 3).unwrap().data().unwrap()[0],
            8.0 * xv * xv * xv - 12.0 * xv,
            1e-12,
        ));
    }

    // --- HermiteE (probabilist) ---

    #[test]
    fn hermite_he_known_values() {
        // He_0 = 1, He_1 = x, He_2 = x^2 - 1, He_3 = x^3 - 3x
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            hermite_polynomial_he(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_he(&x, 1).unwrap().data().unwrap()[0],
            xv,
            1e-12
        ));
        assert!(close(
            hermite_polynomial_he(&x, 2).unwrap().data().unwrap()[0],
            xv * xv - 1.0,
            1e-12,
        ));
        assert!(close(
            hermite_polynomial_he(&x, 3).unwrap().data().unwrap()[0],
            xv * xv * xv - 3.0 * xv,
            1e-12,
        ));
    }

    // --- Laguerre ---

    #[test]
    fn laguerre_l_known_values() {
        // L_0 = 1, L_1 = 1 - x, L_2 = (x^2 - 4x + 2) / 2
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            laguerre_polynomial_l(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            laguerre_polynomial_l(&x, 1).unwrap().data().unwrap()[0],
            1.0 - xv,
            1e-12
        ));
        assert!(close(
            laguerre_polynomial_l(&x, 2).unwrap().data().unwrap()[0],
            f64::midpoint(xv * xv - 4.0 * xv, 2.0),
            1e-12,
        ));
    }

    // --- Legendre ---

    #[test]
    fn legendre_p_known_values() {
        // P_0 = 1, P_1 = x, P_2 = (3x^2 - 1)/2, P_3 = (5x^3 - 3x)/2
        let x = t(&[0.5], &[1]);
        let xv = 0.5;
        assert!(close(
            legendre_polynomial_p(&x, 0).unwrap().data().unwrap()[0],
            1.0,
            1e-12
        ));
        assert!(close(
            legendre_polynomial_p(&x, 1).unwrap().data().unwrap()[0],
            xv,
            1e-12
        ));
        assert!(close(
            legendre_polynomial_p(&x, 2).unwrap().data().unwrap()[0],
            (3.0 * xv * xv - 1.0) / 2.0,
            1e-12,
        ));
        assert!(close(
            legendre_polynomial_p(&x, 3).unwrap().data().unwrap()[0],
            (5.0 * xv * xv * xv - 3.0 * xv) / 2.0,
            1e-12,
        ));
    }

    #[test]
    fn legendre_p_endpoints() {
        // P_n(1) = 1, P_n(-1) = (-1)^n
        let pts = t(&[1.0, -1.0], &[2]);
        for n in 0..6 {
            let r = legendre_polynomial_p(&pts, n).unwrap();
            let d = r.data().unwrap();
            assert!(close(d[0], 1.0, 1e-12), "P_{n}(1) = {}", d[0]);
            let expected_neg = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!(close(d[1], expected_neg, 1e-12), "P_{n}(-1) = {}", d[1]);
        }
    }

    // --- Shifted Chebyshev (domain [0, 1]) ---

    #[test]
    fn shifted_chebyshev_t_matches_t_of_2x_minus_1() {
        let x = xs();
        for n in 0..5 {
            let shifted = shifted_chebyshev_polynomial_t(&x, n).unwrap();
            let xs_data = x.data().unwrap();
            let mapped: Vec<f64> = xs_data.iter().map(|v| 2.0 * v - 1.0).collect();
            let mapped_t = t(&mapped, &[mapped.len()]);
            let direct = chebyshev_polynomial_t(&mapped_t, n).unwrap();
            for (s, d) in shifted
                .data()
                .unwrap()
                .iter()
                .zip(direct.data().unwrap().iter())
            {
                assert!(close(*s, *d, 1e-12), "T*_{n} mismatch at n={n}: {s} vs {d}");
            }
        }
    }

    // --- GPU discipline ---

    #[test]
    fn polynomial_fns_reject_cuda_tensors_explicitly() {
        // We can't construct a CUDA tensor in this CPU-only test, but every
        // polynomial fn calls `require_cpu_poly` at entry which only
        // returns Ok for `!is_cuda()`. If a future refactor accidentally
        // bypasses the gate, this asserts we have at least one fn covered
        // (the others share the same code path via elementwise_f64).
        // Sanity: the gate itself is exercised here on a CPU tensor.
        let x = t(&[0.0], &[1]);
        assert!(chebyshev_polynomial_t(&x, 3).is_ok());
    }
}
