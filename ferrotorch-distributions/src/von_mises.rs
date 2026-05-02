//! Von Mises distribution (circular normal).
//!
//! `VonMises(loc, concentration)` — distribution on the circle [-pi, pi].

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Von Mises distribution parameterized by `loc` (mean direction) and
/// `concentration` (kappa, analogous to inverse variance).
///
/// PDF: `f(x) = exp(kappa * cos(x - loc)) / (2 * pi * I_0(kappa))`
/// where `I_0` is the modified Bessel function of the first kind, order 0.
///
/// Values are on [-pi, pi].
pub struct VonMises<T: Float> {
    loc: Tensor<T>,
    concentration: Tensor<T>,
}

impl<T: Float> VonMises<T> {
    pub fn new(loc: Tensor<T>, concentration: Tensor<T>) -> FerrotorchResult<Self> {
        if loc.shape() != concentration.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "VonMises: loc shape {:?} != concentration shape {:?}",
                    loc.shape(),
                    concentration.shape()
                ),
            });
        }
        Ok(Self { loc, concentration })
    }

    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }
    pub fn concentration(&self) -> &Tensor<T> {
        &self.concentration
    }
}

/// Approximate log of modified Bessel function I_0(x).
/// Uses the polynomial approximation from Abramowitz & Stegun.
fn log_bessel_i0<T: Float>(x: T) -> T {
    let xf = num_traits::ToPrimitive::to_f64(&x).unwrap();
    let result = if xf < 3.75 {
        // Small argument: I_0(x) ≈ polynomial
        let t = (xf / 3.75).powi(2);
        let i0 = 1.0
            + t * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        i0.ln()
    } else {
        // Large argument: asymptotic expansion
        let t = 3.75 / xf;
        let factor = 0.39894228
            + t * (0.01328592
                + t * (0.00225319
                    + t * (-0.00157565
                        + t * (0.00916281
                            + t * (-0.02057706
                                + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
        xf - 0.5 * xf.ln() + factor.ln()
    };
    T::from(result).unwrap()
}

impl<T: Float> Distribution<T> for VonMises<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Best's algorithm for Von Mises sampling.
        let l_data = self.loc.data()?;
        let k_data = self.concentration.data()?;
        let numel: usize = shape.iter().product();

        // Use uniform samples and rejection sampling.
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two = T::from(2.0).unwrap();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(numel);
        let mut rng_state = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut h);
            std::thread::current().id().hash(&mut h);
            h.finish()
        };

        let mut next_u = || -> T {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            T::from((rng_state as f64) / (u64::MAX as f64)).unwrap()
        };

        for i in 0..numel {
            let li = if l_data.len() == 1 {
                0
            } else {
                i % l_data.len()
            };
            let ki = if k_data.len() == 1 {
                0
            } else {
                i % k_data.len()
            };
            let kappa = k_data[ki];

            // Best's algorithm
            let tau = one + (one + T::from(4.0).unwrap() * kappa * kappa).sqrt();
            let rho = (tau - (two * tau).sqrt()) / (two * kappa);
            let r = (one + rho * rho) / (two * rho);

            let sample = loop {
                let u1 = next_u();
                let z = (pi * u1).cos();
                let w = (one + r * z) / (r + z);
                let u2 = next_u();
                let c = kappa * (r - w);

                if c * (two - c) > u2 || c.ln() >= u2.ln() + one - c {
                    let u3 = next_u();
                    let sign = if u3 > T::from(0.5).unwrap() {
                        one
                    } else {
                        zero - one
                    };
                    break sign * w.acos() + l_data[li];
                }
            };

            // Wrap to [-pi, pi]
            let wrapped = ((sample + pi) % (two * pi) + two * pi) % (two * pi) - pi;
            out.push(wrapped);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "VonMises: rsample not supported (discrete rejection sampling)".into(),
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let v = value.data()?;
        let l = self.loc.data()?;
        let k = self.concentration.data()?;
        let numel = v.len();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let li = if l.len() == 1 { 0 } else { i % l.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            // log_prob = kappa * cos(x - loc) - log(2*pi*I_0(kappa))
            let lp = k[ki] * (v[i] - l[li]).cos() - two_pi.ln() - log_bessel_i0(k[ki]);
            out.push(lp);
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    #[allow(clippy::needless_range_loop)]
    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // H = log(2*pi*I_0(kappa)) - kappa * I_1(kappa)/I_0(kappa)
        // Approximate I_1/I_0 ≈ 1 - 1/(2*kappa) for large kappa.
        let k = self.concentration.data()?;
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();

        let mut out = Vec::with_capacity(k.len());
        for i in 0..k.len() {
            let ratio = if k[i] > T::from(0.01).unwrap() {
                one - one / (two * k[i]) // asymptotic approximation of I_1/I_0
            } else {
                k[i] / two // small kappa approximation
            };
            let h = two_pi.ln() + log_bessel_i0(k[i]) - k[i] * ratio;
            out.push(h);
        }

        Tensor::from_storage(
            TensorStorage::cpu(out),
            self.concentration.shape().to_vec(),
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_von_mises_sample_range() {
        let d = VonMises::new(scalar(0.0), scalar(2.0)).unwrap();
        let s = d.sample(&[500]).unwrap();
        let pi = std::f64::consts::PI;
        for &v in s.data().unwrap() {
            assert!(
                v >= -pi && v <= pi,
                "VonMises sample should be in [-pi,pi], got {v}"
            );
        }
    }

    #[test]
    fn test_von_mises_log_prob_at_mode() {
        let d = VonMises::new(scalar(0.0), scalar(5.0)).unwrap();
        let at_mode = Tensor::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], false).unwrap();
        let away = Tensor::from_storage(
            TensorStorage::cpu(vec![std::f64::consts::PI]),
            vec![1],
            false,
        )
        .unwrap();
        let lp_mode = d.log_prob(&at_mode).unwrap().data().unwrap()[0];
        let lp_away = d.log_prob(&away).unwrap().data().unwrap()[0];
        assert!(lp_mode > lp_away, "log_prob should be highest at mode");
    }

    #[test]
    fn test_von_mises_entropy_positive() {
        let d = VonMises::new(scalar(0.0), scalar(1.0)).unwrap();
        let h = d.entropy().unwrap();
        assert!(h.data().unwrap()[0] > 0.0, "entropy should be positive");
    }
}
