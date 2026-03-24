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
#[inline(always)]
fn nt_zero<T: num_traits::Zero>() -> T {
    <T as num_traits::Zero>::zero()
}

/// Helper: return one via `num_traits::One` to avoid ambiguity with
/// `ferray_core::Element::one`.
#[inline(always)]
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
        { result = result - one / z; }
        #[allow(clippy::assign_op_pattern)]
        { z = z + one; }
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
}
