//! Scalar special functions used internally by distribution implementations.
//!
//! These mirror the implementations in `ferrotorch_core::special` but operate
//! on scalars rather than tensors, which is what the distribution code needs
//! for per-element map operations.
//!
//! [CL-329]

use ferrotorch_core::dtype::Float;

// ---------------------------------------------------------------------------
// Lanczos approximation for lgamma
// ---------------------------------------------------------------------------

const LANCZOS_G: f64 = 7.0;

#[rustfmt::skip]
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
   -1_259.139_216_722_402_8,
    771.323_428_777_653_08,
   -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Compute lgamma(x) = log(|Gamma(x)|) using the Lanczos approximation.
pub(crate) fn lgamma_scalar<T: Float>(x: T) -> T {
    let one = <T as num_traits::One>::one();
    let half = T::from(0.5).unwrap();
    let half_ln_2pi = T::from(0.918_938_533_204_672_7).unwrap();
    let g = T::from(LANCZOS_G).unwrap();

    // Handle negative values via reflection formula.
    if x < half {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == <T as num_traits::Zero>::zero() {
            return T::infinity();
        }
        return (pi / sin_pi_x.abs()).ln() - lgamma_scalar(one - x);
    }

    let z = x - one;
    let mut sum = T::from(LANCZOS_COEFFICIENTS[0]).unwrap();
    for i in 1..LANCZOS_COEFFICIENTS.len() {
        sum = sum + T::from(LANCZOS_COEFFICIENTS[i]).unwrap() / (z + T::from(i as f64).unwrap());
    }

    let t = z + g + half;
    half_ln_2pi + t.ln() * (z + half) - t + sum.ln()
}

/// Compute digamma(x) = psi(x) = d/dx ln(Gamma(x)).
///
/// Uses the recurrence relation psi(x+1) = psi(x) + 1/x to shift x into the
/// range [6, inf), then applies the asymptotic expansion.
pub(crate) fn digamma_scalar<T: Float>(x: T) -> T {
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let half = T::from(0.5).unwrap();

    if x != x {
        return x; // NaN
    }

    // For negative x, use the reflection formula:
    // psi(1 - x) - pi * cot(pi * x) = psi(x)
    // => psi(x) = psi(1 - x) - pi * cos(pi*x) / sin(pi*x)
    if x < zero {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let pi_x = pi * x;
        let cot = pi_x.cos() / pi_x.sin();
        return digamma_scalar(one - x) - pi * cot;
    }

    // Shift x upward until x >= 6 using psi(x) = psi(x+1) - 1/x.
    let mut result = zero;
    let mut y = x;
    let six = T::from(6.0).unwrap();
    while y < six {
        result = result - one / y;
        y = y + one;
    }

    // Asymptotic expansion (Abramowitz & Stegun 6.3.18).
    let y2 = one / (y * y);
    result = result + y.ln() - half / y
        - y2 * (T::from(1.0 / 12.0).unwrap()
            - y2 * (T::from(1.0 / 120.0).unwrap()
                - y2 * (T::from(1.0 / 252.0).unwrap()
                    - y2 * (T::from(1.0 / 240.0).unwrap()
                        - y2 * T::from(1.0 / 132.0).unwrap()))));

    result
}
