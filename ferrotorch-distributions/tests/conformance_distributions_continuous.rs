//! Conformance Layer 3 — continuous distribution family tests.
//!
//! Tracking issue: #859 — ferrotorch-distributions conformance suite.
//! Reference: torch == 2.11.0 (torch.distributions)
//!
//! Covers: Normal, Beta, Gamma, Exponential, Laplace, Cauchy, Gumbel,
//! HalfNormal, LogNormal, StudentT, Uniform, MultivariateNormal,
//! LowRankMultivariateNormal, Dirichlet, Kumaraswamy, Pareto, VonMises,
//! Weibull, TransformedDistribution (Normal+Exp=LogNormal), and the
//! `transforms` module (ExpTransform, AffineTransform, SigmoidTransform,
//! TanhTransform, SoftplusTransform, ComposeTransform).
//!
//! Also covers the Distribution trait methods (log_prob, entropy, mean,
//! variance, stddev, cdf, icdf, sample shape contract, rsample).
//!
//! Exact sample values are NOT tested (RNG algorithm divergence);
//! analytic outputs are tested against torch fixtures.

use std::path::PathBuf;

use ferrotorch_core::creation::{from_slice, scalar};
use ferrotorch_distributions::{
    Beta, Cauchy, Dirichlet, Distribution, Exponential, Gamma, Gumbel, HalfNormal, Independent,
    Kumaraswamy, Laplace, LogNormal, LowRankMultivariateNormal, MixtureSameFamily,
    MultivariateNormal, Normal, Pareto, StudentT, Uniform, VonMises, Weibull,
    transforms::{
        AffineTransform, ComposeTransform, ExpTransform, SigmoidTransform, SoftplusTransform,
        TanhTransform, Transform,
    },
};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Fixture loading helpers
// ---------------------------------------------------------------------------

fn fixtures() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures.json");
    let body = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read fixtures.json: {e}"));
    serde_json::from_str(&body).expect("parse fixtures.json")
}

fn f(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) if s == "Inf" || s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-Inf" || s == "-inf" => f64::NEG_INFINITY,
        Value::String(s) if s == "NaN" || s == "nan" => f64::NAN,
        other => panic!("expected number, got {other:?}"),
    }
}

fn fvec(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

/// Emit a skip notice rather than panic — used for exact-sample tests.
#[allow(dead_code)]
fn cascade_skip(reason: &str) {
    eprintln!("cascade_skip: {reason}");
}

const TOL: f64 = 1e-5;
const TOL_LOOSE: f64 = 5e-3;

fn assert_close(actual: f64, expected: f64, tol: f64, ctx: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{ctx}: expected NaN, got {actual}");
        return;
    }
    if expected.is_infinite() {
        assert!(
            actual.is_infinite() && actual.signum() == expected.signum(),
            "{ctx}: expected {expected}, got {actual}"
        );
        return;
    }
    assert!(
        (actual - expected).abs() <= tol,
        "{ctx}: expected {expected}, got {actual} (diff={})",
        (actual - expected).abs()
    );
}

fn assert_close_vec(actual: &[f64], expected: &[f64], tol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, tol, &format!("{ctx}[{i}]"));
    }
}

// ---------------------------------------------------------------------------
// Distribution trait methods — tested via Normal as a representative
// ---------------------------------------------------------------------------

#[test]
fn distribution_trait_sample_returns_correct_shape() {
    let d = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let s = d.sample(&[5, 3]).unwrap();
    assert_eq!(s.shape(), &[5, 3]);
    assert!(!s.requires_grad());
}

#[test]
fn distribution_trait_rsample_shape_and_grad() {
    let loc = scalar(0.0f64).unwrap().requires_grad_(true);
    let scale = scalar(1.0f64).unwrap().requires_grad_(true);
    let d = Normal::new(loc, scale).unwrap();
    let s = d.rsample(&[10]).unwrap();
    assert_eq!(s.shape(), &[10]);
    assert!(s.requires_grad());
}

#[test]
fn distribution_trait_stddev_default_impl() {
    // stddev() default: sqrt(variance). Normal overrides it; Exponential uses default.
    let d = Exponential::new(scalar(2.0f64).unwrap()).unwrap();
    let var = d.variance().unwrap().item().unwrap();
    let std = d.stddev().unwrap().item().unwrap();
    assert_close(std, var.sqrt(), TOL, "stddev default via Exponential");
}

#[test]
fn distribution_trait_cdf_default_returns_error() {
    // Pareto does not implement icdf — the default trait impl returns InvalidArgument.
    // (Kumaraswamy has an actual icdf; use Pareto as the representative no-icdf dist.)
    let d = Pareto::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    assert!(
        d.icdf(&scalar(0.5f64).unwrap()).is_err(),
        "icdf should error"
    );
}

#[test]
fn distribution_trait_mean_mode_variance_default_error() {
    // VonMises doesn't implement variance; calls through to the default error.
    let d = VonMises::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert!(d.variance().is_err());
}

// ---------------------------------------------------------------------------
// Normal
// ---------------------------------------------------------------------------

#[test]
fn normal_new_accessors() {
    let loc = scalar(2.0f64).unwrap();
    let scale = scalar(0.5f64).unwrap();
    let d = Normal::new(loc.clone(), scale.clone()).unwrap();
    assert_close(d.loc().item().unwrap(), 2.0, TOL, "Normal::loc");
    assert_close(d.scale().item().unwrap(), 0.5, TOL, "Normal::scale");
}

#[test]
fn normal_new_shape_mismatch_errors() {
    let loc = scalar(0.0f64).unwrap();
    let scale = from_slice(&[1.0f64, 2.0], &[2]).unwrap();
    assert!(Normal::new(loc, scale).is_err());
}

#[test]
fn normal_fixtures_log_prob_mean_variance_entropy_cdf_icdf() {
    let fix = fixtures();
    for case in fix["normal"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc_v = &case["loc"];
        let scale_v = &case["scale"];

        // Scalar vs batch
        if loc_v.is_array() {
            let locs = fvec(loc_v);
            let scales = fvec(scale_v);
            let loc_t = from_slice::<f64>(&locs, &[locs.len()]).unwrap();
            let scale_t = from_slice::<f64>(&scales, &[scales.len()]).unwrap();
            let d = Normal::new(loc_t, scale_t).unwrap();

            let x_pts = fvec(&case["x_points"]);
            let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
            let lp = d.log_prob(&x_t).unwrap();
            let lp_data = lp.data_vec().unwrap();
            let exp_lp = fvec(&case["log_prob"]);
            assert_close_vec(&lp_data, &exp_lp, TOL, &format!("Normal[{label}] log_prob"));

            let m = d.mean().unwrap().data_vec().unwrap();
            let exp_m = fvec(&case["mean"]);
            assert_close_vec(&m, &exp_m, TOL, &format!("Normal[{label}] mean"));

            let v = d.variance().unwrap().data_vec().unwrap();
            let exp_v = fvec(&case["variance"]);
            assert_close_vec(&v, &exp_v, TOL, &format!("Normal[{label}] variance"));

            let h = d.entropy().unwrap().data_vec().unwrap();
            let exp_h = fvec(&case["entropy"]);
            assert_close_vec(&h, &exp_h, TOL, &format!("Normal[{label}] entropy"));
        } else {
            let loc_s = f(loc_v);
            let scale_s = f(scale_v);
            let d = Normal::new(scalar(loc_s).unwrap(), scalar(scale_s).unwrap()).unwrap();

            let x_pts = fvec(&case["x_points"]);
            let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
            let lp = d.log_prob(&x_t).unwrap();
            let lp_data = lp.data_vec().unwrap();
            let exp_lp = fvec(&case["log_prob"]);
            assert_close_vec(&lp_data, &exp_lp, TOL, &format!("Normal[{label}] log_prob"));

            assert_close(
                d.mean().unwrap().item().unwrap(),
                f(&case["mean"]),
                TOL,
                &format!("Normal[{label}] mean"),
            );
            assert_close(
                d.variance().unwrap().item().unwrap(),
                f(&case["variance"]),
                TOL,
                &format!("Normal[{label}] variance"),
            );
            assert_close(
                d.entropy().unwrap().item().unwrap(),
                f(&case["entropy"]),
                TOL,
                &format!("Normal[{label}] entropy"),
            );

            // CDF
            let cdf_x = fvec(&case["cdf_x"]);
            let cdf_x_t = from_slice::<f64>(&cdf_x, &[cdf_x.len()]).unwrap();
            let cdf_out = d.cdf(&cdf_x_t).unwrap().data_vec().unwrap();
            let exp_cdf = fvec(&case["cdf"]);
            assert_close_vec(&cdf_out, &exp_cdf, TOL, &format!("Normal[{label}] cdf"));

            // ICDF
            let icdf_q = fvec(&case["icdf_q"]);
            let icdf_q_t = from_slice::<f64>(&icdf_q, &[icdf_q.len()]).unwrap();
            let icdf_out = d.icdf(&icdf_q_t).unwrap().data_vec().unwrap();
            let exp_icdf = fvec(&case["icdf"]);
            assert_close_vec(&icdf_out, &exp_icdf, TOL, &format!("Normal[{label}] icdf"));
        }
    }
}

#[test]
fn normal_sample_shape_contract() {
    cascade_skip("RNG algorithm divergence; analytic moments verified via parameterless tests");
    let d = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let s = d.sample(&[50]).unwrap();
    assert_eq!(s.shape(), &[50]);
}

// ---------------------------------------------------------------------------
// Beta
// ---------------------------------------------------------------------------

#[test]
fn beta_new_accessors() {
    let c1 = scalar(2.0f64).unwrap();
    let c0 = scalar(3.0f64).unwrap();
    let d = Beta::new(c1, c0).unwrap();
    assert_close(
        d.concentration1().item().unwrap(),
        2.0,
        TOL,
        "Beta::concentration1",
    );
    assert_close(
        d.concentration0().item().unwrap(),
        3.0,
        TOL,
        "Beta::concentration0",
    );
}

#[test]
fn beta_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["beta"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let c1 = f(&case["concentration1"]);
        let c0 = f(&case["concentration0"]);
        let d = Beta::new(scalar(c1).unwrap(), scalar(c0).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        let exp_lp = fvec(&case["log_prob"]);
        assert_close_vec(&lp, &exp_lp, TOL, &format!("Beta[{label}] log_prob"));

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Beta[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Beta[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Beta[{label}] entropy"),
        );
    }
}

#[test]
fn beta_sample_shape_contract() {
    cascade_skip("RNG algorithm divergence; analytic moments verified via parameterless tests");
    let d = Beta::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    let s = d.sample(&[20]).unwrap();
    assert_eq!(s.shape(), &[20]);
}

// ---------------------------------------------------------------------------
// Gamma
// ---------------------------------------------------------------------------

#[test]
fn gamma_new_accessors() {
    let conc = scalar(2.0f64).unwrap();
    let rate = scalar(1.0f64).unwrap();
    let d = Gamma::new(conc, rate).unwrap();
    assert_close(
        d.concentration().item().unwrap(),
        2.0,
        TOL,
        "Gamma::concentration",
    );
    assert_close(d.rate().item().unwrap(), 1.0, TOL, "Gamma::rate");
}

#[test]
fn gamma_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["gamma"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let conc = f(&case["concentration"]);
        let rate = f(&case["rate"]);
        let d = Gamma::new(scalar(conc).unwrap(), scalar(rate).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        let exp_lp = fvec(&case["log_prob"]);
        assert_close_vec(&lp, &exp_lp, TOL, &format!("Gamma[{label}] log_prob"));

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Gamma[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Gamma[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Gamma[{label}] entropy"),
        );
    }
}

#[test]
fn gamma_sample_shape_contract() {
    cascade_skip("RNG algorithm divergence; analytic moments verified via parameterless tests");
    let d = Gamma::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let s = d.sample(&[30]).unwrap();
    assert_eq!(s.shape(), &[30]);
}

// ---------------------------------------------------------------------------
// Exponential
// ---------------------------------------------------------------------------

#[test]
fn exponential_new_accessor() {
    let d = Exponential::new(scalar(2.0f64).unwrap()).unwrap();
    assert_close(d.rate().item().unwrap(), 2.0, TOL, "Exponential::rate");
}

#[test]
fn exponential_fixtures_log_prob_mean_variance_entropy_cdf_icdf() {
    let fix = fixtures();
    for case in fix["exponential"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let rate = f(&case["rate"]);
        let d = Exponential::new(scalar(rate).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        let exp_lp = fvec(&case["log_prob"]);
        assert_close_vec(&lp, &exp_lp, TOL, &format!("Exponential[{label}] log_prob"));

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Exponential[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Exponential[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Exponential[{label}] entropy"),
        );

        let cdf_out = d.cdf(&x_t).unwrap().data_vec().unwrap();
        let exp_cdf = fvec(&case["cdf"]);
        assert_close_vec(
            &cdf_out,
            &exp_cdf,
            TOL,
            &format!("Exponential[{label}] cdf"),
        );

        let icdf_q = fvec(&case["icdf_q"]);
        let icdf_q_t = from_slice::<f64>(&icdf_q, &[icdf_q.len()]).unwrap();
        let icdf_out = d.icdf(&icdf_q_t).unwrap().data_vec().unwrap();
        let exp_icdf = fvec(&case["icdf"]);
        assert_close_vec(
            &icdf_out,
            &exp_icdf,
            TOL,
            &format!("Exponential[{label}] icdf"),
        );
    }
}

#[test]
fn exponential_sample_shape_contract() {
    cascade_skip("RNG algorithm divergence; analytic moments verified via parameterless tests");
    let d = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[40]).unwrap().shape(), &[40]);
}

// ---------------------------------------------------------------------------
// Laplace
// ---------------------------------------------------------------------------

#[test]
fn laplace_new() {
    let d = Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn laplace_fixtures_log_prob_mean_variance_entropy_cdf_icdf() {
    let fix = fixtures();
    for case in fix["laplace"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc = f(&case["loc"]);
        let scale = f(&case["scale"]);
        let d = Laplace::new(scalar(loc).unwrap(), scalar(scale).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Laplace[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Laplace[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Laplace[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Laplace[{label}] entropy"),
        );

        let cdf_out = d.cdf(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &cdf_out,
            &fvec(&case["cdf"]),
            TOL,
            &format!("Laplace[{label}] cdf"),
        );

        let icdf_q = fvec(&case["icdf_q"]);
        let icdf_q_t = from_slice::<f64>(&icdf_q, &[icdf_q.len()]).unwrap();
        let icdf_out = d.icdf(&icdf_q_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &icdf_out,
            &fvec(&case["icdf"]),
            TOL,
            &format!("Laplace[{label}] icdf"),
        );
    }
}

// ---------------------------------------------------------------------------
// Cauchy
// ---------------------------------------------------------------------------

#[test]
fn cauchy_new() {
    let d = Cauchy::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn cauchy_fixtures_log_prob_entropy_cdf_icdf() {
    let fix = fixtures();
    for case in fix["cauchy"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc = f(&case["loc"]);
        let scale = f(&case["scale"]);
        let d = Cauchy::new(scalar(loc).unwrap(), scalar(scale).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Cauchy[{label}] log_prob"),
        );

        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Cauchy[{label}] entropy"),
        );

        let cdf_out = d.cdf(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &cdf_out,
            &fvec(&case["cdf"]),
            TOL,
            &format!("Cauchy[{label}] cdf"),
        );

        let icdf_q = fvec(&case["icdf_q"]);
        let icdf_q_t = from_slice::<f64>(&icdf_q, &[icdf_q.len()]).unwrap();
        let icdf_out = d.icdf(&icdf_q_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &icdf_out,
            &fvec(&case["icdf"]),
            TOL,
            &format!("Cauchy[{label}] icdf"),
        );
    }
}

// ---------------------------------------------------------------------------
// Gumbel
// ---------------------------------------------------------------------------

#[test]
fn gumbel_new() {
    let d = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn gumbel_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["gumbel"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc = f(&case["loc"]);
        let scale = f(&case["scale"]);
        let d = Gumbel::new(scalar(loc).unwrap(), scalar(scale).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Gumbel[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Gumbel[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Gumbel[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Gumbel[{label}] entropy"),
        );
    }
}

// ---------------------------------------------------------------------------
// HalfNormal
// ---------------------------------------------------------------------------

#[test]
fn half_normal_new() {
    let d = HalfNormal::new(scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn half_normal_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["half_normal"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let scale = f(&case["scale"]);
        let d = HalfNormal::new(scalar(scale).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("HalfNormal[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("HalfNormal[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("HalfNormal[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("HalfNormal[{label}] entropy"),
        );
    }
}

// ---------------------------------------------------------------------------
// LogNormal
// ---------------------------------------------------------------------------

#[test]
fn log_normal_new() {
    let d = LogNormal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn log_normal_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["log_normal"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc = f(&case["loc"]);
        let scale = f(&case["scale"]);
        let d = LogNormal::new(scalar(loc).unwrap(), scalar(scale).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("LogNormal[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("LogNormal[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("LogNormal[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("LogNormal[{label}] entropy"),
        );
    }
}

// ---------------------------------------------------------------------------
// StudentT
// ---------------------------------------------------------------------------

#[test]
fn student_t_new() {
    let d = StudentT::new(
        scalar(3.0f64).unwrap(),
        scalar(0.0f64).unwrap(),
        scalar(1.0f64).unwrap(),
    )
    .unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn student_t_fixtures_log_prob_entropy_mean_variance() {
    let fix = fixtures();
    for case in fix["student_t"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let df = f(&case["df"]);
        let loc = f(&case["loc"]);
        let scale = f(&case["scale"]);
        let d = StudentT::new(
            scalar(df).unwrap(),
            scalar(loc).unwrap(),
            scalar(scale).unwrap(),
        )
        .unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("StudentT[{label}] log_prob"),
        );

        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("StudentT[{label}] entropy"),
        );

        // mean and variance are null for df==1 (Cauchy case); use inherent methods
        if !case["mean"].is_null() {
            let mv = d.mean_value().unwrap();
            assert_close(
                mv[0],
                f(&case["mean"]),
                TOL,
                &format!("StudentT[{label}] mean"),
            );
            let vv = d.variance_value().unwrap();
            assert_close(
                vv[0],
                f(&case["variance"]),
                TOL,
                &format!("StudentT[{label}] variance"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Uniform
// ---------------------------------------------------------------------------

#[test]
fn uniform_new() {
    let d = Uniform::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn uniform_fixtures_log_prob_mean_variance_entropy_cdf_icdf() {
    let fix = fixtures();
    for case in fix["uniform"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let low = f(&case["low"]);
        let high = f(&case["high"]);
        let d = Uniform::new(scalar(low).unwrap(), scalar(high).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Uniform[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Uniform[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Uniform[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Uniform[{label}] entropy"),
        );

        let cdf_out = d.cdf(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &cdf_out,
            &fvec(&case["cdf"]),
            TOL,
            &format!("Uniform[{label}] cdf"),
        );

        let icdf_q = fvec(&case["icdf_q"]);
        let icdf_q_t = from_slice::<f64>(&icdf_q, &[icdf_q.len()]).unwrap();
        let icdf_out = d.icdf(&icdf_q_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &icdf_out,
            &fvec(&case["icdf"]),
            TOL,
            &format!("Uniform[{label}] icdf"),
        );
    }
}

// ---------------------------------------------------------------------------
// MultivariateNormal
// ---------------------------------------------------------------------------

#[test]
fn multivariate_normal_new() {
    let loc = from_slice::<f64>(&[0.0, 0.0], &[2]).unwrap();
    let scale_tril = from_slice::<f64>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let d = MultivariateNormal::from_scale_tril(loc, scale_tril).unwrap();
    assert_eq!(d.sample(&[3]).unwrap().shape(), &[3, 2]);
}

#[test]
fn multivariate_normal_fixtures_log_prob_entropy_mean() {
    let fix = fixtures();
    for case in fix["multivariate_normal"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc_v = fvec(&case["loc"]);
        let n = loc_v.len();
        let loc_t = from_slice::<f64>(&loc_v, &[n]).unwrap();

        // #1087 closed: every MVN fixture entry now records `mean`
        // explicitly (see `scripts/regenerate_distributions_fixtures.py`).
        // Read it directly — a missing field is now a fixture-regen bug
        // rather than a known gap, and we want the test to fail loudly
        // in that case rather than silently falling back to `loc_v`.
        let mean_expected: Vec<f64> = fvec(&case["mean"]);

        // Internal-consistency guard: for a MultivariateNormal, the
        // analytic mean equals `loc`; a fixture where they disagree is
        // malformed regardless of which variant produced it.
        assert_close_vec(
            &loc_v,
            &mean_expected,
            TOL,
            &format!("MultivariateNormal[{label}] fixture loc-mean consistency"),
        );

        let tril_rows: Vec<f64> = case["scale_tril"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(fvec)
            .collect();
        let tril_t = from_slice::<f64>(&tril_rows, &[n, n]).unwrap();
        let d = MultivariateNormal::from_scale_tril(loc_t, tril_t).unwrap();

        let x_pts: Vec<Vec<f64>> = case["x_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(fvec)
            .collect();
        let m = x_pts.len();
        let x_flat: Vec<f64> = x_pts.iter().flat_map(|v| v.iter().copied()).collect();
        let x_t = from_slice::<f64>(&x_flat, &[m, n]).unwrap();

        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("MultivariateNormal[{label}] log_prob"),
        );

        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("MultivariateNormal[{label}] entropy"),
        );

        let mean = d.mean().unwrap().data_vec().unwrap();
        // Compare the distribution's computed mean against the fixture's
        // `mean` field (resolved into `mean_expected` above) — NOT the
        // `loc_v` we used to construct the distribution. Asserting against
        // `loc_v` would be self-referential and pass even if `d.mean()`
        // returned its constructor input as a stub.
        assert_close_vec(
            &mean,
            &mean_expected,
            TOL,
            &format!("MultivariateNormal[{label}] mean"),
        );
    }
}

// ---------------------------------------------------------------------------
// LowRankMultivariateNormal
// ---------------------------------------------------------------------------

#[test]
fn low_rank_multivariate_normal_new() {
    let loc = from_slice::<f64>(&[0.0, 0.0], &[2]).unwrap();
    let cov_factor = from_slice::<f64>(&[1.0, 0.5], &[2, 1]).unwrap();
    let cov_diag = from_slice::<f64>(&[0.5, 0.5], &[2]).unwrap();
    let d = LowRankMultivariateNormal::new(loc, cov_factor, cov_diag).unwrap();
    assert_eq!(d.sample(&[3]).unwrap().shape(), &[3, 2]);
}

#[test]
fn low_rank_multivariate_normal_fixtures_log_prob_entropy_mean() {
    let fix = fixtures();
    for case in fix["low_rank_multivariate_normal"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc_v = fvec(&case["loc"]);
        let n = loc_v.len();
        let loc_t = from_slice::<f64>(&loc_v, &[n]).unwrap();

        let fac_rows: Vec<f64> = case["cov_factor"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(fvec)
            .collect();
        let rank = case["cov_factor"].as_array().unwrap()[0]
            .as_array()
            .unwrap()
            .len();
        let fac_t = from_slice::<f64>(&fac_rows, &[n, rank]).unwrap();

        let diag_v = fvec(&case["cov_diag"]);
        let diag_t = from_slice::<f64>(&diag_v, &[n]).unwrap();

        let d = LowRankMultivariateNormal::new(loc_t, fac_t, diag_t).unwrap();

        let x_pts: Vec<Vec<f64>> = case["x_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(fvec)
            .collect();
        let m = x_pts.len();
        let x_flat: Vec<f64> = x_pts.iter().flat_map(|v| v.iter().copied()).collect();
        let x_t = from_slice::<f64>(&x_flat, &[m, n]).unwrap();

        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("LRMVN[{label}] log_prob"),
        );

        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("LRMVN[{label}] entropy"),
        );

        let mean = d.mean().unwrap().data_vec().unwrap();
        assert_close_vec(&mean, &loc_v, TOL, &format!("LRMVN[{label}] mean"));
    }
}

// ---------------------------------------------------------------------------
// Dirichlet
// ---------------------------------------------------------------------------

#[test]
fn dirichlet_new() {
    let conc = from_slice::<f64>(&[1.0, 1.0, 1.0], &[3]).unwrap();
    let d = Dirichlet::new(conc).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5, 3]);
}

#[test]
fn dirichlet_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["dirichlet"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let conc_v = fvec(&case["concentration"]);
        let k = conc_v.len();
        let conc_t = from_slice::<f64>(&conc_v, &[k]).unwrap();
        let d = Dirichlet::new(conc_t).unwrap();

        let x_pts: Vec<Vec<f64>> = case["x_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(fvec)
            .collect();
        let m = x_pts.len();
        let x_flat: Vec<f64> = x_pts.iter().flat_map(|v| v.iter().copied()).collect();
        let x_t = from_slice::<f64>(&x_flat, &[m, k]).unwrap();

        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Dirichlet[{label}] log_prob"),
        );

        let mean = d.mean().unwrap().data_vec().unwrap();
        assert_close_vec(
            &mean,
            &fvec(&case["mean"]),
            TOL,
            &format!("Dirichlet[{label}] mean"),
        );

        let var = d.variance().unwrap().data_vec().unwrap();
        assert_close_vec(
            &var,
            &fvec(&case["variance"]),
            TOL,
            &format!("Dirichlet[{label}] variance"),
        );

        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Dirichlet[{label}] entropy"),
        );
    }
}

// ---------------------------------------------------------------------------
// Kumaraswamy
// ---------------------------------------------------------------------------

#[test]
fn kumaraswamy_new() {
    let d = Kumaraswamy::new(scalar(2.0f64).unwrap(), scalar(5.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn kumaraswamy_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["kumaraswamy"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let c1 = f(&case["concentration1"]);
        let c0 = f(&case["concentration0"]);
        let d = Kumaraswamy::new(scalar(c1).unwrap(), scalar(c0).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Kumaraswamy[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Kumaraswamy[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Kumaraswamy[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Kumaraswamy[{label}] entropy"),
        );
    }
}

#[test]
fn kumaraswamy_icdf_not_implemented() {
    // Kumaraswamy has a closed-form icdf; the trait default error path is tested
    // via distribution_trait_cdf_default_returns_error (Pareto).  Here we just
    // verify Kumaraswamy::icdf succeeds on a valid quantile as a smoke-test.
    let d = Kumaraswamy::new(scalar(2.0f64).unwrap(), scalar(3.0f64).unwrap()).unwrap();
    assert!(
        d.icdf(&scalar(0.5f64).unwrap()).is_ok(),
        "Kumaraswamy::icdf should succeed"
    );
}

// ---------------------------------------------------------------------------
// Pareto
// ---------------------------------------------------------------------------

#[test]
fn pareto_new() {
    let d = Pareto::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn pareto_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["pareto"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let scale = f(&case["scale"]);
        let alpha = f(&case["alpha"]);
        let d = Pareto::new(scalar(scale).unwrap(), scalar(alpha).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Pareto[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Pareto[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Pareto[{label}] variance"),
        );

        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Pareto[{label}] entropy"),
        );
    }
}

// ---------------------------------------------------------------------------
// VonMises
// ---------------------------------------------------------------------------

#[test]
fn von_mises_new() {
    let d = VonMises::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn von_mises_fixtures_log_prob_mean() {
    let fix = fixtures();
    for case in fix["von_mises"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc = f(&case["loc"]);
        let conc = f(&case["concentration"]);
        let d = VonMises::new(scalar(loc).unwrap(), scalar(conc).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("VonMises[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("VonMises[{label}] mean"),
        );
    }
}

// ---------------------------------------------------------------------------
// Weibull
// ---------------------------------------------------------------------------

#[test]
fn weibull_new() {
    let d = Weibull::new(scalar(1.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    assert_eq!(d.sample(&[5]).unwrap().shape(), &[5]);
}

#[test]
fn weibull_fixtures_log_prob_mean_variance_entropy() {
    let fix = fixtures();
    for case in fix["weibull"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let scale = f(&case["scale"]);
        let conc = f(&case["concentration"]);
        let d = Weibull::new(scalar(scale).unwrap(), scalar(conc).unwrap()).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        assert_close_vec(
            &lp,
            &fvec(&case["log_prob"]),
            TOL,
            &format!("Weibull[{label}] log_prob"),
        );

        assert_close(
            d.mean().unwrap().item().unwrap(),
            f(&case["mean"]),
            TOL,
            &format!("Weibull[{label}] mean"),
        );
        assert_close(
            d.variance().unwrap().item().unwrap(),
            f(&case["variance"]),
            TOL,
            &format!("Weibull[{label}] variance"),
        );
        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Weibull[{label}] entropy"),
        );
    }
}

// ---------------------------------------------------------------------------
// Independent (wraps continuous base)
// ---------------------------------------------------------------------------

#[test]
fn independent_new_sample_shape() {
    let loc = from_slice::<f64>(&[0.0, 1.0, 2.0], &[3]).unwrap();
    let scale = from_slice::<f64>(&[1.0, 1.0, 1.0], &[3]).unwrap();
    let base = Normal::new(loc, scale).unwrap();
    let d = Independent::new(base, 1).unwrap();
    let s = d.sample(&[4]).unwrap();
    // sample([4]) over batch-size-3 Normal → shape [4, 3] per PyTorch semantics
    assert_eq!(s.shape(), &[4, 3]);
    assert_eq!(s.shape()[s.shape().len() - 1], 3);
}

#[test]
fn independent_fixtures_log_prob_entropy() {
    let fix = fixtures();
    for case in fix["independent"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let loc_v = fvec(&case["loc"]);
        let scale_v = fvec(&case["scale"]);
        let n = loc_v.len();
        let loc_t = from_slice::<f64>(&loc_v, &[n]).unwrap();
        let scale_t = from_slice::<f64>(&scale_v, &[n]).unwrap();
        let base = Normal::new(loc_t, scale_t).unwrap();
        let ndims = case["reinterpreted_batch_ndims"].as_u64().unwrap() as usize;
        let d = Independent::new(base, ndims).unwrap();

        let xp_v = fvec(&case["x_point"]);
        let x_t = from_slice::<f64>(&xp_v, &[n]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().item().unwrap();
        assert_close(
            lp,
            f(&case["log_prob"]),
            TOL,
            &format!("Independent[{label}] log_prob"),
        );

        assert_close(
            d.entropy().unwrap().item().unwrap(),
            f(&case["entropy"]),
            TOL,
            &format!("Independent[{label}] entropy"),
        );
    }
}

// ---------------------------------------------------------------------------
// MixtureSameFamily
// ---------------------------------------------------------------------------

#[test]
fn mixture_same_family_new_sample_shape() {
    use ferrotorch_distributions::Categorical;
    let mix_probs = from_slice::<f64>(&[0.4, 0.6], &[2]).unwrap();
    let mix = Categorical::new(mix_probs).unwrap();
    let comp_loc = from_slice::<f64>(&[-2.0, 2.0], &[2]).unwrap();
    let comp_scale = from_slice::<f64>(&[0.5, 0.5], &[2]).unwrap();
    let comp = Normal::new(comp_loc, comp_scale).unwrap();
    let d = MixtureSameFamily::new(mix, comp).unwrap();
    let s = d.sample(&[10]).unwrap();
    assert_eq!(s.shape(), &[10]);
}

#[test]
fn mixture_same_family_fixtures_log_prob() {
    use ferrotorch_distributions::Categorical;
    let fix = fixtures();
    for case in fix["mixture_same_family"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let mix_p = fvec(&case["mix_probs"]);
        let k = mix_p.len();
        let mix_t = from_slice::<f64>(&mix_p, &[k]).unwrap();
        let mix = Categorical::new(mix_t).unwrap();

        let comp_loc = fvec(&case["comp_loc"]);
        let comp_scale = fvec(&case["comp_scale"]);
        let loc_t = from_slice::<f64>(&comp_loc, &[k]).unwrap();
        let scale_t = from_slice::<f64>(&comp_scale, &[k]).unwrap();
        let comp = Normal::new(loc_t, scale_t).unwrap();

        let d = MixtureSameFamily::new(mix, comp).unwrap();

        let x_pts = fvec(&case["x_points"]);
        let exp_lp = fvec(&case["log_prob"]);
        for (x_val, exp_val) in x_pts.iter().zip(exp_lp.iter()) {
            let x_t = scalar(*x_val).unwrap();
            let lp = d.log_prob(&x_t).unwrap().item().unwrap();
            assert_close(
                lp,
                *exp_val,
                TOL,
                &format!("MixtureSameFamily[{label}] log_prob at x={x_val}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// TransformedDistribution: Normal + ExpTransform == LogNormal
// ---------------------------------------------------------------------------

#[test]
fn transformed_distribution_new_sample_shape() {
    use ferrotorch_distributions::TransformedDistribution;
    let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
    let d = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
    let s = d.sample(&[10]).unwrap();
    assert_eq!(s.shape(), &[10]);
}

#[test]
fn transformed_distribution_fixtures_log_prob_equals_lognormal() {
    use ferrotorch_distributions::TransformedDistribution;
    let fix = fixtures();
    for case in fix["transformed_distribution"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let base_loc = f(&case["base_loc"]);
        let base_scale = f(&case["base_scale"]);
        let base = Normal::new(scalar(base_loc).unwrap(), scalar(base_scale).unwrap()).unwrap();
        let d = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);

        let x_pts = fvec(&case["x_points"]);
        let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
        let lp = d.log_prob(&x_t).unwrap().data_vec().unwrap();
        let exp_lp = fvec(&case["log_prob_transformed"]);
        assert_close_vec(
            &lp,
            &exp_lp,
            TOL,
            &format!("TransformedDistribution[{label}] log_prob"),
        );
    }
}

// ---------------------------------------------------------------------------
// Transforms module
// ---------------------------------------------------------------------------

#[test]
fn exp_transform_forward_inverse_log_det_jacobian() {
    let fix = fixtures();
    let case = fix["transforms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "exp_transform")
        .unwrap();

    let x_pts = fvec(&case["x_points"]);
    let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
    let t = ExpTransform;

    let fwd = t.forward(&x_t).unwrap().data_vec().unwrap();
    assert_close_vec(&fwd, &fvec(&case["forward"]), TOL, "ExpTransform::forward");

    let y_t = from_slice::<f64>(&fwd, &[fwd.len()]).unwrap();
    let inv = t.inverse(&y_t).unwrap().data_vec().unwrap();
    // inverse(forward(x)) ≈ x
    assert_close_vec(&inv, &x_pts, TOL, "ExpTransform::inverse");

    let ladj = t
        .log_abs_det_jacobian(&x_t, &y_t)
        .unwrap()
        .data_vec()
        .unwrap();
    assert_close_vec(
        &ladj,
        &fvec(&case["log_abs_det_jacobian"]),
        TOL,
        "ExpTransform::log_abs_det_jacobian",
    );
}

#[test]
fn affine_transform_new_forward_inverse_log_det_jacobian() {
    let fix = fixtures();
    let case = fix["transforms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "affine_transform_loc2_scale3")
        .unwrap();

    let loc = f(&case["loc"]);
    let scale = f(&case["scale"]);
    let t = AffineTransform::new(loc, scale);

    let x_pts = fvec(&case["x_points"]);
    let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();

    let fwd = t.forward(&x_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &fwd,
        &fvec(&case["forward"]),
        TOL,
        "AffineTransform::forward",
    );

    let y_t = from_slice::<f64>(&fwd, &[fwd.len()]).unwrap();
    let inv = t.inverse(&y_t).unwrap().data_vec().unwrap();
    // inverse(forward(x)) should ≈ x (forward maps x→loc+scale*x so inverse gives back x)
    assert_close_vec(
        &inv,
        &fvec(&case["inverse"]),
        TOL,
        "AffineTransform::inverse",
    );

    let ladj = t
        .log_abs_det_jacobian(&x_t, &y_t)
        .unwrap()
        .data_vec()
        .unwrap();
    assert_close_vec(
        &ladj,
        &fvec(&case["log_abs_det_jacobian"]),
        TOL,
        "AffineTransform::log_abs_det_jacobian",
    );
}

#[test]
fn sigmoid_transform_forward_inverse_log_det_jacobian() {
    let fix = fixtures();
    let case = fix["transforms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "sigmoid_transform")
        .unwrap();

    let x_pts = fvec(&case["x_points"]);
    let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
    let t = SigmoidTransform;

    let fwd = t.forward(&x_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &fwd,
        &fvec(&case["forward"]),
        TOL,
        "SigmoidTransform::forward",
    );

    let y_t = from_slice::<f64>(&fwd, &[fwd.len()]).unwrap();
    let inv = t.inverse(&y_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &inv,
        &fvec(&case["inverse"]),
        TOL,
        "SigmoidTransform::inverse",
    );

    let ladj = t
        .log_abs_det_jacobian(&x_t, &y_t)
        .unwrap()
        .data_vec()
        .unwrap();
    assert_close_vec(
        &ladj,
        &fvec(&case["log_abs_det_jacobian"]),
        TOL,
        "SigmoidTransform::log_abs_det_jacobian",
    );
}

#[test]
fn tanh_transform_forward_inverse_log_det_jacobian() {
    let fix = fixtures();
    let case = fix["transforms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "tanh_transform")
        .unwrap();

    let x_pts = fvec(&case["x_points"]);
    let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
    let t = TanhTransform;

    let fwd = t.forward(&x_t).unwrap().data_vec().unwrap();
    assert_close_vec(&fwd, &fvec(&case["forward"]), TOL, "TanhTransform::forward");

    let y_t = from_slice::<f64>(&fwd, &[fwd.len()]).unwrap();
    let inv = t.inverse(&y_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &inv,
        &fvec(&case["inverse"]),
        TOL_LOOSE,
        "TanhTransform::inverse",
    );

    let ladj = t
        .log_abs_det_jacobian(&x_t, &y_t)
        .unwrap()
        .data_vec()
        .unwrap();
    assert_close_vec(
        &ladj,
        &fvec(&case["log_abs_det_jacobian"]),
        TOL,
        "TanhTransform::log_abs_det_jacobian",
    );
}

#[test]
fn softplus_transform_forward_inverse_log_det_jacobian() {
    let fix = fixtures();
    let case = fix["transforms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "softplus_transform")
        .unwrap();

    let x_pts = fvec(&case["x_points"]);
    let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();
    let t = SoftplusTransform;

    let fwd = t.forward(&x_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &fwd,
        &fvec(&case["forward"]),
        TOL,
        "SoftplusTransform::forward",
    );

    let y_t = from_slice::<f64>(&fwd, &[fwd.len()]).unwrap();
    let inv = t.inverse(&y_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &inv,
        &fvec(&case["inverse"]),
        TOL,
        "SoftplusTransform::inverse",
    );

    let ladj = t
        .log_abs_det_jacobian(&x_t, &y_t)
        .unwrap()
        .data_vec()
        .unwrap();
    assert_close_vec(
        &ladj,
        &fvec(&case["log_abs_det_jacobian"]),
        TOL,
        "SoftplusTransform::log_abs_det_jacobian",
    );
}

#[test]
fn compose_transform_new_len_is_empty_forward_inverse_log_det_jacobian() {
    // ComposeTransform: AffineTransform(loc=0, scale=2) then ExpTransform
    // forward(x) = exp(2*x)
    // fixture uses compose_affine_then_exp which is actually AffineTransform(2,2) then ExpTransform
    let fix = fixtures();
    let case = fix["transforms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "compose_affine_then_exp")
        .unwrap();

    // Recreate: affine(loc=2, scale=3) then exp — matches the fixture generator
    let aff: Box<dyn Transform<f64>> = Box::new(AffineTransform::new(2.0f64, 3.0f64));
    let exp: Box<dyn Transform<f64>> = Box::new(ExpTransform);
    let compose = ComposeTransform::new(vec![aff, exp]);
    assert_eq!(compose.len(), 2);
    assert!(!compose.is_empty());

    let x_pts = fvec(&case["x_points"]);
    let x_t = from_slice::<f64>(&x_pts, &[x_pts.len()]).unwrap();

    let fwd = compose.forward(&x_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &fwd,
        &fvec(&case["forward"]),
        TOL,
        "ComposeTransform::forward",
    );

    let y_t = from_slice::<f64>(&fwd, &[fwd.len()]).unwrap();
    let inv = compose.inverse(&y_t).unwrap().data_vec().unwrap();
    assert_close_vec(
        &inv,
        &fvec(&case["inverse"]),
        TOL,
        "ComposeTransform::inverse",
    );

    let ladj = compose
        .log_abs_det_jacobian(&x_t, &y_t)
        .unwrap()
        .data_vec()
        .unwrap();
    assert_close_vec(
        &ladj,
        &fvec(&case["log_abs_det_jacobian"]),
        TOL,
        "ComposeTransform::log_abs_det_jacobian",
    );
}

#[test]
fn compose_transform_empty_is_identity() {
    let empty_compose: ComposeTransform<f64> = ComposeTransform::new(vec![]);
    assert!(empty_compose.is_empty());
    assert_eq!(empty_compose.len(), 0);
}
