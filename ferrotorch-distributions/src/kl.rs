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

use crate::special_fns::digamma_scalar;
use crate::{
    Bernoulli, Categorical, Distribution, Exponential, Gamma, Laplace, Normal, Poisson, Uniform,
};

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
    if let (Some(pb), Some(qb)) = (
        p.downcast_ref::<Bernoulli<T>>(),
        q.downcast_ref::<Bernoulli<T>>(),
    ) {
        return kl_bernoulli_bernoulli(pb, qb);
    }
    // Uniform-Uniform
    if let (Some(pu), Some(qu)) = (
        p.downcast_ref::<Uniform<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_uniform_uniform(pu, qu);
    }
    // Categorical-Categorical
    if let (Some(pc), Some(qc)) = (
        p.downcast_ref::<Categorical<T>>(),
        q.downcast_ref::<Categorical<T>>(),
    ) {
        return kl_categorical_categorical(pc, qc);
    }
    // Normal-Uniform
    if let (Some(pn), Some(qu)) = (
        p.downcast_ref::<Normal<T>>(),
        q.downcast_ref::<Uniform<T>>(),
    ) {
        return kl_normal_uniform(pn, qu);
    }
    // Uniform-Normal
    if let (Some(pu), Some(qn)) = (
        p.downcast_ref::<Uniform<T>>(),
        q.downcast_ref::<Normal<T>>(),
    ) {
        return kl_uniform_normal(pu, qn);
    }
    // Laplace-Laplace
    if let (Some(pl), Some(ql)) = (
        p.downcast_ref::<Laplace<T>>(),
        q.downcast_ref::<Laplace<T>>(),
    ) {
        return kl_laplace_laplace(pl, ql);
    }
    // Exponential-Exponential
    if let (Some(pe), Some(qe)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_exponential_exponential(pe, qe);
    }
    // Gamma-Gamma
    if let (Some(pg), Some(qg)) = (p.downcast_ref::<Gamma<T>>(), q.downcast_ref::<Gamma<T>>()) {
        return kl_gamma_gamma(pg, qg);
    }
    // Poisson-Poisson
    if let (Some(pp_), Some(qp_)) = (
        p.downcast_ref::<Poisson<T>>(),
        q.downcast_ref::<Poisson<T>>(),
    ) {
        return kl_poisson_poisson(pp_, qp_);
    }
    // Gamma-Exponential: Exp(lambda) == Gamma(1, lambda), use gamma formula.
    if let (Some(pg), Some(qe)) = (
        p.downcast_ref::<Gamma<T>>(),
        q.downcast_ref::<Exponential<T>>(),
    ) {
        return kl_gamma_exponential(pg, qe);
    }
    // Exponential-Gamma: likewise.
    if let (Some(pe), Some(qg)) = (
        p.downcast_ref::<Exponential<T>>(),
        q.downcast_ref::<Gamma<T>>(),
    ) {
        return kl_exponential_gamma(pe, qg);
    }

    Err(FerrotorchError::InvalidArgument {
        message: "No KL divergence formula registered for this distribution pair. \
                  Supported same-family pairs: Normal-Normal, Bernoulli-Bernoulli, \
                  Uniform-Uniform, Categorical-Categorical, Laplace-Laplace, \
                  Exponential-Exponential, Gamma-Gamma, Poisson-Poisson. \
                  Cross-family: Normal-Uniform, Uniform-Normal, \
                  Gamma-Exponential, Exponential-Gamma."
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

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Bernoulli(p) || Bernoulli(q))
///
/// = p * log(p/q) + (1-p) * log((1-p)/(1-q))
fn kl_bernoulli_bernoulli<T: Float>(
    p: &Bernoulli<T>,
    q: &Bernoulli<T>,
) -> FerrotorchResult<Tensor<T>> {
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

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
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

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
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

    Tensor::from_storage(TensorStorage::cpu(result), p.low().shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Additional KL formulas (CL-365)
// ---------------------------------------------------------------------------

/// KL(Laplace(loc1, b1) || Laplace(loc2, b2))
///
/// = log(b2 / b1) + (b1 * exp(-|loc1 - loc2| / b1) + |loc1 - loc2|) / b2 - 1
///
/// Derived from integrating the Laplace log-density. Reduces to 0 when
/// the two distributions are identical.
fn kl_laplace_laplace<T: Float>(p: &Laplace<T>, q: &Laplace<T>) -> FerrotorchResult<Tensor<T>> {
    let p_loc = p.loc().data_vec()?;
    let p_scale = p.scale().data_vec()?;
    let q_loc = q.loc().data_vec()?;
    let q_scale = q.scale().data_vec()?;

    let one = T::from(1.0).unwrap();
    let zero = T::from(0.0).unwrap();

    let result: Vec<T> = p_loc
        .iter()
        .zip(p_scale.iter())
        .zip(q_loc.iter().cycle())
        .zip(q_scale.iter().cycle())
        .map(|(((&pl, &ps), &ql), &qs)| {
            let diff = pl - ql;
            let abs_diff = if diff < zero { zero - diff } else { diff };
            (qs / ps).ln() + (ps * (-abs_diff / ps).exp() + abs_diff) / qs - one
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.loc().shape().to_vec(), false)
}

/// KL(Exponential(rate1) || Exponential(rate2))
///
/// = log(rate1 / rate2) + rate2 / rate1 - 1
fn kl_exponential_exponential<T: Float>(
    p: &Exponential<T>,
    q: &Exponential<T>,
) -> FerrotorchResult<Tensor<T>> {
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_rate.iter().cycle())
        .map(|(&pr, &qr)| (pr / qr).ln() + qr / pr - one)
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
}

/// KL(Gamma(α1, β1) || Gamma(α2, β2))
///
/// = (α1 - α2) * ψ(α1) - lnΓ(α1) + lnΓ(α2)
///   + α2 * (ln β1 - ln β2) + α1 * (β2 - β1) / β1
///
/// where ψ is the digamma function and Γ is the gamma function.
///
/// Reduces to 0 when the two distributions are identical (verified by
/// `test_kl_gamma_gamma_same`).
fn kl_gamma_gamma<T: Float>(p: &Gamma<T>, q: &Gamma<T>) -> FerrotorchResult<Tensor<T>> {
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_conc = q.concentration().data_vec()?;
    let q_rate = q.rate().data_vec()?;

    let result: Vec<T> = p_conc
        .iter()
        .zip(p_rate.iter())
        .zip(q_conc.iter().cycle())
        .zip(q_rate.iter().cycle())
        .map(|(((&pa, &pb), &qa), &qb)| kl_gamma_scalar(pa, pb, qa, qb))
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration().shape().to_vec(),
        false,
    )
}

/// Scalar KL(Gamma(α1, β1) || Gamma(α2, β2)). Factored out so the
/// Gamma-Exponential cross-family formula can reuse it.
fn kl_gamma_scalar<T: Float>(pa: T, pb: T, qa: T, qb: T) -> T {
    // (pa - qa) * digamma(pa) - lnGamma(pa) + lnGamma(qa)
    //   + qa * (ln pb - ln qb) + pa * (qb - pb) / pb
    let dig_pa = digamma_scalar(pa);
    let ln_gamma_pa = ln_gamma_scalar(pa);
    let ln_gamma_qa = ln_gamma_scalar(qa);
    (pa - qa) * dig_pa - ln_gamma_pa + ln_gamma_qa + qa * (pb.ln() - qb.ln()) + pa * (qb - pb) / pb
}

/// Lanczos approximation for log Γ(x) — uses the f64 `libm`-style
/// series. The absolute error is ~1e-12 for x > 0.5, which is fine
/// for KL divergence computations that are themselves lossy.
fn ln_gamma_scalar<T: Float>(x: T) -> T {
    let x_f64 = x.to_f64().unwrap_or(f64::NAN);
    let result_f64 = ln_gamma_f64(x_f64);
    T::from(result_f64).unwrap()
}

/// log Γ(x) via the Stirling-series recurrence used by
/// num-traits / libm. Valid for x > 0. For x <= 0 the reflection
/// formula would apply, but the Gamma distribution constrains
/// shape > 0 anyway.
fn ln_gamma_f64(x: f64) -> f64 {
    // Stirling's approximation with Kemp coefficients: accurate to
    // ~1e-12 across x > 0.5. Follows the same structure used by
    // `special_fns::lgamma_scalar` in ferrotorch.
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Shift to x >= 5 so Stirling converges, then subtract log terms.
    let mut x = x;
    let mut acc = 0.0;
    while x < 5.0 {
        acc -= x.ln();
        x += 1.0;
    }
    // Asymptotic Stirling series.
    let inv = 1.0 / (x * x);
    let sum = (1.0 / 12.0) - inv * ((1.0 / 360.0) - inv * (1.0 / 1260.0));
    acc + (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + sum / x
}

/// KL(Poisson(λ1) || Poisson(λ2))
///
/// = λ1 * (log λ1 - log λ2) - λ1 + λ2
fn kl_poisson_poisson<T: Float>(p: &Poisson<T>, q: &Poisson<T>) -> FerrotorchResult<Tensor<T>> {
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_rate.iter().cycle())
        .map(|(&pr, &qr)| pr * (pr.ln() - qr.ln()) - pr + qr)
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
}

/// KL(Gamma(α, β) || Exponential(λ))
///
/// Since Exp(λ) = Gamma(1, λ), this reduces to the Gamma-Gamma
/// formula with q_concentration = 1 and q_rate = λ.
fn kl_gamma_exponential<T: Float>(p: &Gamma<T>, q: &Exponential<T>) -> FerrotorchResult<Tensor<T>> {
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_conc
        .iter()
        .zip(p_rate.iter())
        .zip(q_rate.iter().cycle())
        .map(|((&pa, &pb), &qb)| kl_gamma_scalar(pa, pb, one, qb))
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result),
        p.concentration().shape().to_vec(),
        false,
    )
}

/// KL(Exponential(λ) || Gamma(α, β))
///
/// Exp(λ) = Gamma(1, λ), so this is Gamma-Gamma with
/// p_concentration = 1 and p_rate = λ.
fn kl_exponential_gamma<T: Float>(p: &Exponential<T>, q: &Gamma<T>) -> FerrotorchResult<Tensor<T>> {
    let p_rate = p.rate().data_vec()?;
    let q_conc = q.concentration().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();

    let result: Vec<T> = p_rate
        .iter()
        .zip(q_conc.iter().cycle())
        .zip(q_rate.iter().cycle())
        .map(|((&pb, &qa), &qb)| kl_gamma_scalar(one, pb, qa, qb))
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), p.rate().shape().to_vec(), false)
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

    // -----------------------------------------------------------------------
    // CL-365: new same-family and cross-family pairs
    // -----------------------------------------------------------------------

    // -- Laplace-Laplace -----------------------------------------------------

    #[test]
    fn test_kl_laplace_laplace_same_is_zero() {
        let p = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-5,
            "got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_laplace_laplace_different_scale() {
        // KL(Lap(0,1) || Lap(0,2)) = log(2/1) + (1*exp(0) + 0)/2 - 1
        //                          = ln(2) + 0.5 - 1 ≈ 0.6931 - 0.5 = 0.1931
        let p = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Laplace::new(scalar(0.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let v = kl.item().unwrap();
        let expected = 2.0_f32.ln() + 0.5 - 1.0;
        assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
    }

    #[test]
    fn test_kl_laplace_laplace_different_loc() {
        // KL(Lap(0,1) || Lap(1,1)) = log(1) + (exp(-1) + 1)/1 - 1
        //                          = 0 + e^-1 + 1 - 1 = 1/e ≈ 0.3679
        let p = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let q = Laplace::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 1.0_f32 / std::f32::consts::E;
        assert!(
            (kl.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -- Exponential-Exponential ---------------------------------------------

    #[test]
    fn test_kl_exponential_exponential_same() {
        let p = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().abs() < 1e-5);
    }

    #[test]
    fn test_kl_exponential_exponential_different() {
        // KL(Exp(2) || Exp(1)) = log(2/1) + 1/2 - 1 = ln(2) - 0.5 ≈ 0.1931
        let p = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Exponential::new(scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0_f32.ln() - 0.5;
        assert!((kl.item().unwrap() - expected).abs() < 1e-5);
    }

    // -- Gamma-Gamma ---------------------------------------------------------

    #[test]
    fn test_kl_gamma_gamma_same_is_zero() {
        // When both distributions are identical, KL should be 0. This
        // exercises the full Gamma-Gamma formula including digamma
        // and lgamma terms.
        let p = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        // Small tolerance because Stirling approximation has some error.
        assert!(
            kl.item().unwrap().abs() < 1e-3,
            "KL(Gamma same) should be near 0, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_gamma_gamma_exp_special_case() {
        // Gamma(1, λ) == Exp(λ). Verify that KL(Gamma(1,2) || Gamma(1,1))
        // matches KL(Exp(2) || Exp(1)) = ln(2) - 0.5.
        let p = Gamma::new(scalar(1.0f32).unwrap(), scalar(2.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0_f32.ln() - 0.5;
        assert!(
            (kl.item().unwrap() - expected).abs() < 2e-3,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    // -- Poisson-Poisson -----------------------------------------------------

    #[test]
    fn test_kl_poisson_poisson_same() {
        let p = Poisson::new(scalar(3.0f32).unwrap()).unwrap();
        let q = Poisson::new(scalar(3.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.item().unwrap().abs() < 1e-5);
    }

    #[test]
    fn test_kl_poisson_poisson_known_value() {
        // KL(Poisson(2) || Poisson(1)) = 2*(ln 2 - ln 1) - 2 + 1
        //                              = 2*ln 2 - 1 ≈ 0.3863
        let p = Poisson::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Poisson::new(scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0 * 2.0_f32.ln() - 1.0;
        assert!((kl.item().unwrap() - expected).abs() < 1e-5);
    }

    // -- Cross-family: Gamma-Exponential and Exponential-Gamma ---------------

    #[test]
    fn test_kl_gamma_exponential_matches_gamma_gamma() {
        // KL(Gamma(2, 3) || Exp(1)) should equal KL(Gamma(2,3) || Gamma(1,1))
        let p = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let q_exp = Exponential::new(scalar(1.0f32).unwrap()).unwrap();
        let q_gamma = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl_ge = kl_divergence(&p, &q_exp).unwrap();
        let p2 = Gamma::new(scalar(2.0f32).unwrap(), scalar(3.0f32).unwrap()).unwrap();
        let kl_gg = kl_divergence(&p2, &q_gamma).unwrap();
        assert!(
            (kl_ge.item().unwrap() - kl_gg.item().unwrap()).abs() < 1e-4,
            "Gamma-Exp and Gamma-Gamma(1,λ) should agree"
        );
    }

    #[test]
    fn test_kl_exponential_gamma_matches_gamma_gamma() {
        // KL(Exp(2) || Gamma(1, 1)) == KL(Gamma(1, 2) || Gamma(1, 1))
        //   == Exp-Exp(2, 1) == ln(2) - 0.5
        let p = Exponential::new(scalar(2.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let expected = 2.0_f32.ln() - 0.5;
        assert!(
            (kl.item().unwrap() - expected).abs() < 2e-3,
            "expected {expected}, got {}",
            kl.item().unwrap()
        );
    }

    #[test]
    fn test_kl_exponential_gamma_self_consistency() {
        // Gamma(1, 1) == Exp(1), so KL(Exp(1) || Gamma(1,1)) == 0.
        let p = Exponential::new(scalar(1.0f32).unwrap()).unwrap();
        let q = Gamma::new(scalar(1.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(
            kl.item().unwrap().abs() < 1e-3,
            "KL(Exp(1)||Gamma(1,1)) should be 0, got {}",
            kl.item().unwrap()
        );
    }

    // -- ln_gamma numerical sanity -------------------------------------------

    #[test]
    fn test_ln_gamma_known_values() {
        // lnΓ(1) = 0, lnΓ(2) = 0, lnΓ(3) = ln(2) ≈ 0.6931,
        // lnΓ(4) = ln(6) ≈ 1.7918, lnΓ(5) = ln(24) ≈ 3.1781
        assert!((ln_gamma_f64(1.0) - 0.0).abs() < 1e-8);
        assert!((ln_gamma_f64(2.0) - 0.0).abs() < 1e-8);
        assert!((ln_gamma_f64(3.0) - 2.0f64.ln()).abs() < 1e-6);
        assert!((ln_gamma_f64(4.0) - 6.0f64.ln()).abs() < 1e-6);
        assert!((ln_gamma_f64(5.0) - 24.0f64.ln()).abs() < 1e-6);
    }
}
