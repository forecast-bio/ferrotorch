//! Kumaraswamy distribution.
//!
//! `Kumaraswamy(a, b)` — a two-parameter distribution on [0, 1], similar to
//! Beta but with a closed-form CDF and simpler sampling.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Kumaraswamy distribution parameterized by concentration parameters `a` > 0
/// and `b` > 0.
///
/// PDF: `f(x) = a * b * x^(a-1) * (1 - x^a)^(b-1)` for `x in [0, 1]`.
///
/// CDF: `F(x) = 1 - (1 - x^a)^b`.
///
/// Sampling via inverse CDF: `x = (1 - (1-u)^(1/b))^(1/a)`.
pub struct Kumaraswamy<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> Kumaraswamy<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> FerrotorchResult<Self> {
        if a.shape() != b.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Kumaraswamy: a shape {:?} != b shape {:?}",
                    a.shape(),
                    b.shape()
                ),
            });
        }
        Ok(Self { a, b })
    }

    pub fn a(&self) -> &Tensor<T> {
        &self.a
    }
    pub fn b(&self) -> &Tensor<T> {
        &self.b
    }
}

impl<T: Float> Distribution<T> for Kumaraswamy<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let a_data = self.a.data()?;
        let b_data = self.b.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            let bi = if b_data.len() == 1 {
                0
            } else {
                i % b_data.len()
            };
            // x = (1 - (1-u)^(1/b))^(1/a)
            let inner = (one - u_data[i]).powf(one / b_data[bi]);
            let val = (one - inner)
                .max(T::from(1e-30).unwrap())
                .powf(one / a_data[ai]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Kumaraswamy: rsample not yet implemented".into(),
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let v = value.data()?;
        let a = self.a.data()?;
        let b = self.b.data()?;
        let numel = v.len();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let bi = if b.len() == 1 { 0 } else { i % b.len() };
            if v[i] <= zero || v[i] >= one {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(a) + log(b) + (a-1)*log(x) + (b-1)*log(1 - x^a)
                let lp = a[ai].ln()
                    + b[bi].ln()
                    + (a[ai] - one) * v[i].ln()
                    + (b[bi] - one) * (one - v[i].powf(a[ai])).ln();
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // H = (1 - 1/b) + (1 - 1/a) * H_b - log(a) - log(b)
        // where H_b = digamma(b+1) + euler_gamma (harmonic number approximation)
        // Simplified: H ≈ (1-1/a)*(digamma(b+1)+euler) + (1-1/b) - ln(a*b)
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let euler = T::from(0.5772156649015329).unwrap();

        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            // Approximate digamma(b+1) ≈ ln(b) + 1/(2*b) for large b
            let bf = num_traits::ToPrimitive::to_f64(&b[i]).unwrap();
            let digamma_b1 = T::from(if bf > 5.0 {
                bf.ln() + 0.5 / bf
            } else {
                // For small b, use the recurrence: digamma(x+1) = digamma(x) + 1/x
                // digamma(1) = -euler_gamma
                let mut val = -0.5772156649015329;
                let mut x = 1.0;
                while x < bf + 1.0 {
                    val += 1.0 / x;
                    x += 1.0;
                }
                val
            })
            .unwrap();

            let h = (one - one / a[i]) * (digamma_b1 + euler) + (one - one / b[i])
                - a[i].ln()
                - b[i].ln();
            out.push(h);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // F(x) = 1 - (1 - x^a)^b for x in [0, 1].
        let v = value.data()?;
        let a = self.a.data()?;
        let b = self.b.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(v.len());
        for (i, &x) in v.iter().enumerate() {
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let bi = if b.len() == 1 { 0 } else { i % b.len() };
            if x <= zero {
                out.push(zero);
            } else if x >= one {
                out.push(one);
            } else {
                out.push(one - (one - x.powf(a[ai])).powf(b[bi]));
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // F^{-1}(p) = (1 - (1 - p)^(1/b))^(1/a)
        let p = q.data()?;
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(p.len());
        for (i, &pi) in p.iter().enumerate() {
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let bi = if b.len() == 1 { 0 } else { i % b.len() };
            let inner = (one - pi).powf(one / b[bi]);
            out.push((one - inner).max(T::from(1e-30).unwrap()).powf(one / a[ai]));
        }
        Tensor::from_storage(TensorStorage::cpu(out), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // E[X] = b * Beta(1 + 1/a, b)
        // = b * Gamma(1+1/a) * Gamma(b) / Gamma(1+1/a+b)
        // = exp( ln(b) + lgamma(1+1/a) + lgamma(b) - lgamma(1+1/a+b) )
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            let alpha = one + one / a[i];
            let lg = lgamma_scalar(alpha) + lgamma_scalar(b[i]) - lgamma_scalar(alpha + b[i]);
            out.push(b[i] * lg.exp());
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // For a > 1 and b >= 1: mode = ((a-1) / (a*b - 1))^(1/a)
        // For other parameter combinations the mode is at 0 or 1; we return
        // 0 as a defensive default to match the torch convention of returning
        // a representative mode-point rather than NaN.
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            if a[i] > one && b[i] >= one {
                let denom = a[i] * b[i] - one;
                if denom > zero {
                    out.push(((a[i] - one) / denom).powf(one / a[i]));
                } else {
                    out.push(zero);
                }
            } else {
                out.push(zero);
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // m1 = b * Beta(1 + 1/a, b)
        // m2 = b * Beta(1 + 2/a, b)
        // Var = m2 - m1^2
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            let m1_log = lgamma_scalar(one + one / a[i]) + lgamma_scalar(b[i])
                - lgamma_scalar(one + one / a[i] + b[i]);
            let m1 = b[i] * m1_log.exp();
            let m2_log = lgamma_scalar(one + two / a[i]) + lgamma_scalar(b[i])
                - lgamma_scalar(one + two / a[i] + b[i]);
            let m2 = b[i] * m2_log.exp();
            out.push(m2 - m1 * m1);
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }
}

/// Scalar lgamma via the existing tensor-shaped `special::lgamma` op.
/// Avoids reimplementing the Lanczos series for a single scalar evaluation.
fn lgamma_scalar<T: Float>(x: T) -> T {
    let t = Tensor::from_storage(TensorStorage::cpu(vec![x]), vec![1], false)
        .expect("lgamma_scalar: scalar tensor build");
    let r = ferrotorch_core::special::lgamma(&t).expect("lgamma_scalar: lgamma op");
    r.data().expect("lgamma_scalar: lgamma data")[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_kumaraswamy_sample_range() {
        let d = Kumaraswamy::new(scalar(2.0), scalar(5.0)).unwrap();
        let s = d.sample(&[500]).unwrap();
        for &v in s.data().unwrap() {
            assert!(
                v > 0.0 && v < 1.0,
                "Kumaraswamy sample should be in (0,1), got {v}"
            );
        }
    }

    #[test]
    fn test_kumaraswamy_log_prob_boundary() {
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let at_zero = Tensor::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], false).unwrap();
        let at_one = Tensor::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], false).unwrap();
        assert!(d.log_prob(&at_zero).unwrap().data().unwrap()[0].is_infinite());
        assert!(d.log_prob(&at_one).unwrap().data().unwrap()[0].is_infinite());
    }

    #[test]
    fn test_kumaraswamy_uniform_case() {
        // a=1, b=1 should be uniform — log_prob should be 0 everywhere in (0,1)
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![0.5]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap().data().unwrap()[0];
        assert!(
            (lp - 0.0).abs() < 1e-6,
            "a=1,b=1 should be uniform, log_prob={lp}"
        );
    }

    // ---- properties (#608) ----

    #[test]
    fn test_kumaraswamy_cdf_unit_case_is_identity() {
        // a=1, b=1 -> Uniform(0,1), so F(x) = x.
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let v = Tensor::from_storage(TensorStorage::cpu(vec![x]), vec![1], false).unwrap();
            let c = d.cdf(&v).unwrap().data().unwrap()[0];
            assert!((c - x).abs() < 1e-9, "x={x}, cdf={c}");
        }
    }

    #[test]
    fn test_kumaraswamy_cdf_icdf_roundtrip() {
        let d = Kumaraswamy::new(scalar(2.0), scalar(3.0)).unwrap();
        for p in [0.1, 0.3, 0.7, 0.9] {
            let q = Tensor::from_storage(TensorStorage::cpu(vec![p]), vec![1], false).unwrap();
            let x = d.icdf(&q).unwrap();
            let p2 = d.cdf(&x).unwrap().data().unwrap()[0];
            assert!((p2 - p).abs() < 1e-6, "p={p}, recovered={p2}");
        }
    }

    #[test]
    fn test_kumaraswamy_mean_uniform_case_is_half() {
        // For a=1, b=1, mean should be 0.5.
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let m = d.mean().unwrap().data().unwrap()[0];
        assert!((m - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_kumaraswamy_variance_uniform_case_is_one_twelfth() {
        // For a=1, b=1, variance should be 1/12 ≈ 0.0833.
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let v = d.variance().unwrap().data().unwrap()[0];
        assert!((v - 1.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_kumaraswamy_mode_well_defined_when_a_gt_one() {
        // a=2, b=2: mode = ((2-1)/(4-1))^(1/2) = sqrt(1/3) ≈ 0.5774
        let d = Kumaraswamy::new(scalar(2.0), scalar(2.0)).unwrap();
        let m = d.mode().unwrap().data().unwrap()[0];
        assert!((m - (1.0_f64 / 3.0).sqrt()).abs() < 1e-9);
    }
}
