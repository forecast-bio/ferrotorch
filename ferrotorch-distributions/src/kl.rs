//! KL divergence between probability distributions.
//!
//! Provides closed-form analytical KL divergence formulas for same-family
//! and select cross-family distribution pairs.
//!
//! This mirrors PyTorch's `torch.distributions.kl` module.
//!
//! CL-330

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::{Bernoulli, Categorical, Distribution, Normal, Uniform};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Kullback-Leibler divergence `KL(p || q)` between two
/// distributions.
///
/// The KL divergence is defined as:
///
/// ```text
/// KL(p || q) = integral p(x) * log(p(x) / q(x)) dx
/// ```
///
/// Returns a tensor whose shape matches the batch shape of the distributions.
///
/// # Supported pairs (f32 and f64)
///
/// | P | Q |
/// |---|---|
/// | Normal | Normal |
/// | Bernoulli | Bernoulli |
/// | Uniform | Uniform |
/// | Categorical | Categorical |
/// | Normal | Uniform |
/// | Uniform | Normal |
///
/// # Errors
///
/// Returns an error if no KL formula is registered for the `(P, Q)` pair.
///
/// # Examples
///
/// ```ignore
/// use ferrotorch_distributions::{Normal, kl::kl_divergence};
/// let p = Normal::new(scalar(0.0f32)?, scalar(1.0)?)?;
/// let q = Normal::new(scalar(1.0f32)?, scalar(2.0)?)?;
/// let kl = kl_divergence(&p, &q)?;
/// ```
pub fn kl_divergence<T: Float, P, Q>(p: &P, q: &Q) -> FerrotorchResult<Tensor<T>>
where
    P: Distribution<T> + 'static,
    Q: Distribution<T> + 'static,
{
    kl_dispatch::<T>(p, q)
}

fn kl_dispatch<T: Float>(
    p: &dyn std::any::Any,
    q: &dyn std::any::Any,
) -> FerrotorchResult<Tensor<T>> {
    // Normal-Normal
    if let (Some(pn), Some(qn)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Normal<T>>()) {
        return kl_normal_normal(pn, qn);
    }
    // Bernoulli-Bernoulli
    if let (Some(pb), Some(qb)) = (p.downcast_ref::<Bernoulli<T>>(), q.downcast_ref::<Bernoulli<T>>()) {
        return kl_bernoulli_bernoulli(pb, qb);
    }
    // Uniform-Uniform
    if let (Some(pu), Some(qu)) = (p.downcast_ref::<Uniform<T>>(), q.downcast_ref::<Uniform<T>>()) {
        return kl_uniform_uniform(pu, qu);
    }
    // Categorical-Categorical
    if let (Some(pc), Some(qc)) = (p.downcast_ref::<Categorical<T>>(), q.downcast_ref::<Categorical<T>>()) {
        return kl_categorical_categorical(pc, qc);
    }
    // Normal-Uniform
    if let (Some(pn), Some(qu)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Uniform<T>>()) {
        return kl_normal_uniform(pn, qu);
    }
    // Uniform-Normal
    if let (Some(pu), Some(qn)) = (p.downcast_ref::<Uniform<T>>(), q.downcast_ref::<Normal<T>>()) {
        return kl_uniform_normal(pu, qn);
    }

    Err(FerrotorchError::InvalidArgument {
        message: "No KL divergence formula registered for this distribution pair. \
                  Supported same-family pairs: Normal-Normal, Bernoulli-Bernoulli, \
                  Uniform-Uniform, Categorical-Categorical. \
                  Cross-family: Normal-Uniform, Uniform-Normal."
            .into(),
    })
}

// ---------------------------------------------------------------------------
// KL divergence formulas (generic over T: Float)
// ---------------------------------------------------------------------------

/// KL(Normal(loc1, scale1) || Normal(loc2, scale2))
///
/// = 0.5 * (var_ratio + (loc1-loc2)^2/var2 - 1 - ln(var_ratio))
///
/// where var_ratio = (scale1/scale2)^2
fn kl_normal_normal<T: Float>(p: &Normal<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;

    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let var_ratio = (ps / qs) * (ps / qs);
            let mean_diff_sq = ((pl - ql) / qs) * ((pl - ql) / qs);
            half * (var_ratio + mean_diff_sq - one - var_ratio.ln())
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.loc().shape().to_vec(),
        false,
    )
}

/// KL(Bernoulli(p) || Bernoulli(q))
///
/// = p * log(p/q) + (1-p) * log((1-p)/(1-q))
fn kl_bernoulli_bernoulli<T: Float>(p: &Bernoulli<T>, q: &Bernoulli<T>) -> FerrotorchResult<Tensor<T>> {
    let p_probs = p.probs().data_vec()?;
    let q_probs = q.probs().data_vec()?;

    let one = T::from(1.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    let result: Vec<T> = p_probs
        .iter()
        .zip(q_probs.iter().cycle())
        .map(|(&pp, &qp)| {
            let pp = pp.max(eps).min(one - eps);
            let qp = qp.max(eps).min(one - eps);
            pp * (pp / qp).ln() + (one - pp) * ((one - pp) / (one - qp)).ln()
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.probs().shape().to_vec(),
        false,
    )
}

/// KL(Uniform(a1, b1) || Uniform(a2, b2))
///
/// = log((b2-a2) / (b1-a1)) if [a1,b1] subset of [a2,b2], else infinity
fn kl_uniform_uniform<T: Float>(p: &Uniform<T>, q: &Uniform<T>) -> FerrotorchResult<Tensor<T>> {
    let p_low = p.low().data_vec()?;
    let p_high = p.high().data_vec()?;
    let q_low = q.low().data_vec()?;
    let q_high = q.high().data_vec()?;

    let result: Vec<T> = p_low
        .iter()
        .zip(p_high.iter())
        .zip(q_low.iter().cycle())
        .zip(q_high.iter().cycle())
        .map(|(((&pl, &ph), &ql), &qh)| {
            if ql > pl || qh < ph {
                T::infinity()
            } else {
                ((qh - ql) / (ph - pl)).ln()
            }
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.low().shape().to_vec(),
        false,
    )
}

/// KL(Categorical(p) || Categorical(q))
///
/// = sum_k p_k * log(p_k / q_k)
fn kl_categorical_categorical<T: Float>(
    p: &Categorical<T>,
    q: &Categorical<T>,
) -> FerrotorchResult<Tensor<T>> {
    let p_probs = p.probs().data_vec()?;
    let q_probs = q.probs().data_vec()?;

    let zero = T::from(0.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    // Normalize both
    let p_total: T = p_probs.iter().copied().fold(zero, |a, b| a + b);
    let q_total: T = q_probs.iter().copied().fold(zero, |a, b| a + b);

    let kl: T = p_probs
        .iter()
        .zip(q_probs.iter())
        .fold(zero, |acc, (&pp, &qp)| {
            let pp_norm = pp / p_total;
            let qp_norm = (qp / q_total).max(eps);
            if pp_norm <= eps {
                acc
            } else if qp_norm <= eps {
                T::infinity()
            } else {
                acc + pp_norm * (pp_norm / qp_norm).ln()
            }
        });

    // Categorical KL is a scalar
    Tensor::from_storage(TensorStorage::cpu(vec![kl]), vec![], false)
}

/// KL(Normal(loc, scale) || Uniform(a, b))
///
/// = -entropy(Normal) - log(1/(b-a))  if the normal is "contained"
///
/// More precisely:
/// KL(N || U) = -H(N) + log(b-a)
///
/// where H(N) = 0.5 * ln(2*pi*e*scale^2).
///
/// Note: this is only finite when the Uniform support covers the Normal
/// effectively. We compute the analytical formula unconditionally (as
/// PyTorch does for some cross-family pairs).
fn kl_normal_uniform<T: Float>(p: &Normal<T>, q: &Uniform<T>) -> FerrotorchResult<Tensor<T>> {
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_low = q.low().data_vec()?;
    let q_high = q.high().data_vec()?;

    let half = T::from(0.5).unwrap();
    let two_pi_e = T::from(2.0 * std::f64::consts::PI * std::f64::consts::E).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_low.iter().cycle())
        .zip(q_high.iter().cycle())
        .map(|(((&_pl, &ps), &ql), &qh)| {
            let entropy = half * (two_pi_e * ps * ps).ln();
            let log_range = (qh - ql).ln();
            -entropy + log_range
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.loc().shape().to_vec(),
        false,
    )
}

/// KL(Uniform(a, b) || Normal(loc, scale))
///
/// = -H(Uniform) + 0.5 * log(2*pi*scale^2) + (1/(2*scale^2)) * ((b-a)^2/12 + ((a+b)/2 - loc)^2)
///
/// where H(Uniform(a,b)) = log(b-a).
fn kl_uniform_normal<T: Float>(p: &Uniform<T>, q: &Normal<T>) -> FerrotorchResult<Tensor<T>> {
    let p_low = p.low().data_vec()?;
    let p_high = p.high().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;

    let half = T::from(0.5).unwrap();
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
    let twelve = T::from(12.0).unwrap();
    let two = T::from(2.0).unwrap();

    let result: Vec<T> = p_low
        .iter()
        .zip(p_high.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ph), &ql), &qs)| {
            let range = ph - pl;
            let entropy_uniform = range.ln();
            let log_normal_term = half * (two_pi * qs * qs).ln();
            let mean_p = (pl + ph) / two;
            let var_p = range * range / twelve;
            let mse = (mean_p - ql) * (mean_p - ql);
            let second_moment = var_p + mse;
            -entropy_uniform + log_normal_term + second_moment / (two * qs * qs)
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.low().shape().to_vec(),
        false,
    )
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{scalar, tensor};

    // -- Normal-Normal -------------------------------------------------------

    #[test]
    fn test_kl_normal_normal_same() {
        // KL(N(0,1) || N(0,1)) = 0
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-6,
            "KL(same, same) should be 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_different_mean() {
        // KL(N(0,1) || N(1,1)) = 0.5 * (1 + 1 - 1 - 0) = 0.5
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            (kl.item().unwrap() - 0.5).abs() < 1e-5,
            "expected 0.5, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_different_scale() {
        // KL(N(0,1) || N(0,2)) = 0.5 * (0.25 + 0 - 1 - ln(0.25))
        //                       = 0.5 * (0.25 - 1 + ln(4))
        //                       = 0.5 * (-0.75 + 1.3863) = 0.5 * 0.6363 = 0.3181
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.5 * (0.25 + 0.0 - 1.0 - 0.25f32.ln());
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_nonnegative() {
        // KL divergence is always >= 0
        let p = Normal::new(scalar(2.0f32).unwrap(), scalar(0.5f32).unwrap()).unwrap();
        let q = Normal::new(scalar(-1.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap() >= 0.0,
            "KL should be non-negative, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_normal_normal_asymmetric() {
        // KL(p||q) != KL(q||p) in general
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl_pq = kl_divergence(&p, &q).unwrap().item().unwrap();
        let kl_qp = kl_divergence(&q, &p).unwrap().item().unwrap();
        assert!(
            (kl_pq - kl_qp).abs() > 1e-3,
            "KL should be asymmetric: KL(p||q)={kl_pq}, KL(q||p)={kl_qp}"
        );
    }

    // -- Bernoulli-Bernoulli -------------------------------------------------

    #[test]
    fn test_kl_bernoulli_same() {
        let p = Bernoulli::new(scalar(0.3f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.3f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-5,
            "KL(same, same) = 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_bernoulli_different() {
        // KL(Bern(0.4) || Bern(0.6)) = 0.4*ln(0.4/0.6) + 0.6*ln(0.6/0.4)
        let p = Bernoulli::new(scalar(0.4f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.6f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.4f32 * (0.4f32 / 0.6).ln() + 0.6 * (0.6f32 / 0.4).ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_bernoulli_nonnegative() {
        let p = Bernoulli::new(scalar(0.1f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.9f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap() >= 0.0);
    }

    // -- Uniform-Uniform -----------------------------------------------------

    #[test]
    fn test_kl_uniform_same() {
        let p = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-6,
            "KL(same, same) = 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_uniform_contained() {
        // KL(U(0,1) || U(-1,2)) = ln(3/1) = ln(3)
        let p = Uniform::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(-1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 3.0f32.ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_uniform_not_contained() {
        // If q doesn't cover p, KL = infinity
        let p = Uniform::new(scalar(0.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().is_infinite(),
            "expected infinity, got {}",
            kl.item().unwrap()
        );
    }

    // -- Categorical-Categorical ---------------------------------------------

    #[test]
    fn test_kl_categorical_same() {
        let p = Categorical::new(tensor(&[0.2f32, 0.3, 0.5]).unwrap()).unwrap();
        let q = Categorical::new(tensor(&[0.2f32, 0.3, 0.5]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-5,
            "KL(same, same) = 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_categorical_different() {
        let p = Categorical::new(tensor(&[0.5f32, 0.5]).unwrap()).unwrap();
        let q = Categorical::new(tensor(&[0.25f32, 0.75]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // KL = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75) = 0.5*ln(2) + 0.5*ln(2/3)
        let expected = 0.5f32 * 2.0f32.ln() + 0.5 * (2.0f32 / 3.0).ln();
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_categorical_nonnegative() {
        let p = Categorical::new(tensor(&[0.1f32, 0.2, 0.7]).unwrap()).unwrap();
        let q = Categorical::new(tensor(&[0.3f32, 0.3, 0.4]).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap() >= -1e-6);
    }

    // -- Normal-Uniform (cross-family) ---------------------------------------

    #[test]
    fn test_kl_normal_uniform() {
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Uniform::new(scalar(-10.0f32).unwrap(), scalar(10.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // Should be finite and non-negative-ish
        assert!(kl.item().unwrap().is_finite());
    }

    // -- Uniform-Normal (cross-family) ---------------------------------------

    #[test]
    fn test_kl_uniform_normal() {
        let p = Uniform::new(scalar(-1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().is_finite());
        assert!(kl.item().unwrap() >= -1e-6);
    }

    // -- f64 -----------------------------------------------------------------

    #[test]
    fn test_kl_normal_normal_f64() {
        let p = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().abs() < 1e-12);
    }

    #[test]
    fn test_kl_bernoulli_f64() {
        let p = Bernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.7f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 0.3f64 * (0.3 / 0.7f64).ln() + 0.7 * (0.7 / 0.3f64).ln();
        assert!((kl.item().unwrap() - expected).abs() < 1e-8);
    }

    #[test]
    fn test_kl_uniform_f64() {
        let p = Uniform::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let q = Uniform::new(scalar(0.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!((kl.item().unwrap() - 2.0f64.ln()).abs() < 1e-10);
    }

    // -- Error case ----------------------------------------------------------

    #[test]
    fn test_kl_unsupported_pair() {
        // Normal-Bernoulli should fail (not registered)
        let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Bernoulli::new(scalar(0.5f32).unwrap()).unwrap();
        assert!(kl_divergence(&p, &q).is_err());
    }
}
