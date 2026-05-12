//! Distributions parity dump binary for the ferrotorch-distributions
//! real-artifact harness (Phase G.1, #1167).
//!
//! Companion to `scripts/verify_distributions_inference.py` and
//! `scripts/pin_pretrained_distributions_fixtures.py`. Given a config
//! name (e.g. `normal_standard`), a path to the per-config fixture
//! folder, and an output dir, this example:
//!
//!   1. Builds the corresponding ferrotorch distribution with the
//!      hardcoded params (same params torch used at pin time — the
//!      config matrix is the single source of truth).
//!   2. Samples `N=10_000` points (moments only — see verifier docs).
//!   3. Reads the reference `test_points.bin` from the fixture folder.
//!   4. Calls `log_prob(test_points)` and `entropy()`.
//!   5. For `kl_*` configs: computes `kl_divergence(p, q)` between two
//!      hard-coded distributions matching the pin script's KL spec.
//!   6. Writes one `.bin` per metric in the same single-tensor f32 LE
//!      format the pin script uses:
//!        * `<config>_mean.bin`
//!        * `<config>_var.bin`
//!        * `<config>_log_prob.bin`
//!        * `<config>_entropy.bin`
//!
//!      or for KL configs:
//!        * `<config>_kl.bin`
//!
//! The dispatch contract is: **moments only** for sample comparison.
//! `ferrotorch_core::creation::rand` uses a time-seeded xorshift PRNG;
//! aligning byte-for-byte with `torch.Generator`'s Philox is not
//! possible. The harness compares mean / variance with an MC budget
//! (0.05 / 0.1 at N=10000) and analytic outputs at tight 1e-4.
//!
//! Single-tensor binary format (little-endian):
//!
//! ```text
//! [u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]
//! ```
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-distributions --release \
//!   --example distributions_dump -- \
//!     --config normal_standard \
//!     --fixture-dir /tmp/.../normal_standard \
//!     --output-dir  /tmp/.../rust_out
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::creation::{from_slice, scalar};
use ferrotorch_core::tensor::Tensor;
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::transforms::{AffineTransform, Transform, TransformedDistribution};
use ferrotorch_distributions::{
    Bernoulli, Beta, Categorical, Cauchy, Dirichlet, Distribution, Exponential, Gamma, HalfNormal,
    Laplace, LogNormal, Multinomial, MultivariateNormal, Normal, Poisson, StudentT, Uniform,
};

const N_SAMPLES: usize = 10_000;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    config: String,
    fixture_dir: PathBuf,
    output_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut config: Option<String> = None;
    let mut fixture_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--config" => {
                config = Some(argv.get(i + 1).ok_or("--config needs a value")?.clone());
                i += 2;
            }
            "--fixture-dir" => {
                fixture_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--fixture-dir needs a value")?,
                ));
                i += 2;
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output-dir needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        config: config.ok_or("--config is required")?,
        fixture_dir: fixture_dir.ok_or("--fixture-dir is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
    })
}

// ---------------------------------------------------------------------------
// Single-tensor binary format (mirrors the Python pin script).
// ---------------------------------------------------------------------------

/// Read a single-tensor `[u32 ndim][u32 * ndim shape][f32 * prod(shape)]`
/// little-endian file into `(shape, flat row-major data)`.
fn read_single_tensor_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let mut f =
        File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)
        .map_err(|e| format!("read ndim from {}: {e}", path.display()))?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for di in 0..ndim {
        f.read_exact(&mut buf4)
            .map_err(|e| format!("read shape[{di}] from {}: {e}", path.display()))?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }
    let numel: usize = shape.iter().product();
    let mut data_bytes = vec![0u8; numel * 4];
    f.read_exact(&mut data_bytes)
        .map_err(|e| format!("read data from {}: {e}", path.display()))?;
    let mut data = Vec::with_capacity(numel);
    for chunk in data_bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, data))
}

fn write_single_tensor_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())?;
    }
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Moment helpers
// ---------------------------------------------------------------------------

/// Compute per-event-axis mean over a `[N, *event_shape]`-flat buffer.
///
/// Returns a `(event_shape, mean_per_event)` tuple. For scalar event
/// shapes the returned shape is `[1]` and `data.len() == 1`. For
/// event_shape `[K]` the returned shape is `[K]` and `data.len() == K`.
fn moments_mean_var(
    sample_shape: &[usize],
    sample_data: &[f32],
    event_ndim: usize,
) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    // sample_shape = [N, *event_shape]; event dims are the trailing
    // `event_ndim`. event_ndim==0 for scalar samples (sample_shape=[N]).
    let n = sample_shape[0];
    let event_shape: Vec<usize> = if event_ndim == 0 {
        vec![]
    } else {
        sample_shape[sample_shape.len() - event_ndim..].to_vec()
    };
    let event_numel: usize = event_shape.iter().product::<usize>().max(1);
    let mut mean = vec![0.0f64; event_numel];
    let mut m2 = vec![0.0f64; event_numel];
    for row in 0..n {
        let base = row * event_numel;
        for k in 0..event_numel {
            let x = sample_data[base + k] as f64;
            mean[k] += x;
            m2[k] += x * x;
        }
    }
    let n_f = n as f64;
    for k in 0..event_numel {
        mean[k] /= n_f;
        m2[k] = m2[k] / n_f - mean[k] * mean[k];
    }
    let mean_f32: Vec<f32> = mean.into_iter().map(|x| x as f32).collect();
    let var_f32: Vec<f32> = m2.into_iter().map(|x| x as f32).collect();
    let out_shape = if event_shape.is_empty() {
        vec![1]
    } else {
        event_shape
    };
    (out_shape, mean_f32, var_f32)
}

// ---------------------------------------------------------------------------
// Distribution + KL drivers — each takes a config name and returns
// `(mean, var, log_prob, entropy_or_kl)` and the event_ndim used for moments.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct DumpResult {
    mean_shape: Vec<usize>,
    mean_data: Vec<f32>,
    var_data: Vec<f32>,
    log_prob_shape: Vec<usize>,
    log_prob_data: Vec<f32>,
    /// Either entropy data (`is_kl=false`) or KL data (`is_kl=true`).
    aux_shape: Vec<usize>,
    aux_data: Vec<f32>,
    is_kl: bool,
    /// `None` if the distribution does not implement entropy (e.g. Multinomial).
    has_aux: bool,
}

/// Build the ferrotorch distribution + run sample/log_prob/entropy for
/// the given config. Test points come from the fixture's
/// `test_points.bin`. Returns moments + analytic outputs.
fn dump_distribution(
    config: &str,
    test_points_shape: &[usize],
    test_points_data: &[f32],
) -> Result<DumpResult, String> {
    let value = || -> Result<Tensor<f32>, String> {
        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(test_points_data.to_vec()),
            test_points_shape.to_vec(),
            false,
        )
        .map_err(|e| format!("build value tensor: {e:?}"))
    };

    match config {
        // --------------------------- Normal ---------------------------
        "normal_standard" | "normal_shifted" => {
            let (loc, scale) = if config == "normal_standard" {
                (0.0f32, 1.0f32)
            } else {
                (2.0f32, 0.5f32)
            };
            let dist = Normal::new(scalar(loc).unwrap(), scalar(scale).unwrap())
                .map_err(|e| format!("Normal::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // ---------------------------- Beta ----------------------------
        "beta_25" => {
            let dist = Beta::new(scalar(2.0f32).unwrap(), scalar(5.0f32).unwrap())
                .map_err(|e| format!("Beta::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // ---------------------------- Gamma ---------------------------
        "gamma_21" => {
            let dist = Gamma::new(scalar(2.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("Gamma::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // --------------------------- Cauchy ---------------------------
        "cauchy_standard" => {
            let dist = Cauchy::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("Cauchy::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // ------------------------ Exponential -------------------------
        "exponential_1p5" => {
            let dist = Exponential::new(scalar(1.5f32).unwrap())
                .map_err(|e| format!("Exponential::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // -------------------------- Uniform ---------------------------
        "uniform_neg2_3" => {
            let dist = Uniform::new(scalar(-2.0f32).unwrap(), scalar(3.0f32).unwrap())
                .map_err(|e| format!("Uniform::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // -------------------------- LogNormal -------------------------
        "lognormal_0_p5" => {
            let dist = LogNormal::new(scalar(0.0f32).unwrap(), scalar(0.5f32).unwrap())
                .map_err(|e| format!("LogNormal::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // -------------------------- Laplace ---------------------------
        "laplace_0_1" => {
            let dist = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("Laplace::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // ------------------------- HalfNormal -------------------------
        "halfnormal_1" => {
            let dist = HalfNormal::new(scalar(1.0f32).unwrap())
                .map_err(|e| format!("HalfNormal::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // -------------------------- StudentT --------------------------
        "studentt_df5" => {
            let dist = StudentT::new(
                scalar(5.0f32).unwrap(),
                scalar(0.0f32).unwrap(),
                scalar(1.0f32).unwrap(),
            )
            .map_err(|e| format!("StudentT::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // ------------------------- Bernoulli --------------------------
        "bernoulli_p3" => {
            let dist = Bernoulli::new(scalar(0.3f32).unwrap())
                .map_err(|e| format!("Bernoulli::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // -------------------------- Poisson ---------------------------
        "poisson_3" => {
            let dist = Poisson::new(scalar(3.0f32).unwrap())
                .map_err(|e| format!("Poisson::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // ------------------------ Categorical -------------------------
        "categorical_k4" => {
            let probs = from_slice(&[0.1f32, 0.3, 0.4, 0.2], &[4])
                .map_err(|e| format!("probs: {e:?}"))?;
            let dist =
                Categorical::new(probs).map_err(|e| format!("Categorical::new: {e:?}"))?;
            run_univariate(&dist, &value()?)
        }

        // ------------------------- Dirichlet --------------------------
        "dirichlet_k4" => {
            let conc = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])
                .map_err(|e| format!("conc: {e:?}"))?;
            let dist = Dirichlet::new(conc).map_err(|e| format!("Dirichlet::new: {e:?}"))?;
            run_multivariate(&dist, &value()?, 1)
        }

        // --------------------- MultivariateNormal ---------------------
        "mvn_3d" => {
            let loc = from_slice(&[0.0f32, 0.0, 0.0], &[3])
                .map_err(|e| format!("loc: {e:?}"))?;
            let cov = from_slice(
                &[1.0f32, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0],
                &[3, 3],
            )
            .map_err(|e| format!("cov: {e:?}"))?;
            let dist = MultivariateNormal::from_covariance(loc, cov)
                .map_err(|e| format!("MVN::from_covariance: {e:?}"))?;
            run_multivariate(&dist, &value()?, 1)
        }

        // ---------------- TransformedDistribution (#1109) -------------
        // TransformedDistribution(Normal(0, 1), [AffineTransform(2, 3)]) —
        // the entropy dispatch must surface the closed-form
        // `base.entropy() + log|3|` (equal to entropy(Normal(2, 3))).
        "transformed_normal_affine" => {
            let base = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("Normal::new: {e:?}"))?;
            let affine = AffineTransform::new(2.0f32, 3.0f32);
            let transforms: Vec<Box<dyn Transform<f32>>> = vec![Box::new(affine)];
            let dist = TransformedDistribution::new(Box::new(base), transforms);
            run_univariate(&dist, &value()?)
        }

        // ------------------------ Multinomial -------------------------
        "multinomial_k3_n20" => {
            let probs = from_slice(&[0.2f32, 0.3, 0.5], &[3])
                .map_err(|e| format!("probs: {e:?}"))?;
            let dist = Multinomial::new(20, probs)
                .map_err(|e| format!("Multinomial::new: {e:?}"))?;
            run_multivariate(&dist, &value()?, 1)
        }

        other => Err(format!("unknown distribution config {other:?}")),
    }
}

/// Sample, log_prob, entropy for a scalar-event univariate distribution.
fn run_univariate(
    dist: &dyn Distribution<f32>,
    value: &Tensor<f32>,
) -> Result<DumpResult, String> {
    let samples = dist
        .sample(&[N_SAMPLES])
        .map_err(|e| format!("sample: {e:?}"))?;
    let sample_data = samples.data_vec().map_err(|e| format!("sample data: {e:?}"))?;
    let (m_shape, mean, var) = moments_mean_var(samples.shape(), &sample_data, 0);

    let lp = dist
        .log_prob(value)
        .map_err(|e| format!("log_prob: {e:?}"))?;
    let lp_data = lp.data_vec().map_err(|e| format!("log_prob data: {e:?}"))?;
    let lp_shape = lp.shape().to_vec();

    let (aux_shape, aux_data, has_aux) = match dist.entropy() {
        Ok(t) => {
            let d = t.data_vec().map_err(|e| format!("entropy data: {e:?}"))?;
            let s = t.shape().to_vec();
            let s = if s.is_empty() { vec![1] } else { s };
            (s, d, true)
        }
        Err(_) => (vec![1], vec![0.0f32], false),
    };

    Ok(DumpResult {
        mean_shape: m_shape,
        mean_data: mean,
        var_data: var,
        log_prob_shape: lp_shape,
        log_prob_data: lp_data,
        aux_shape,
        aux_data,
        is_kl: false,
        has_aux,
    })
}

/// Sample, log_prob, entropy for a multivariate distribution whose
/// event has `event_ndim` trailing dims (e.g. Dirichlet/MVN: 1).
fn run_multivariate(
    dist: &dyn Distribution<f32>,
    value: &Tensor<f32>,
    event_ndim: usize,
) -> Result<DumpResult, String> {
    let samples = dist
        .sample(&[N_SAMPLES])
        .map_err(|e| format!("sample: {e:?}"))?;
    let sample_data = samples.data_vec().map_err(|e| format!("sample data: {e:?}"))?;
    let (m_shape, mean, var) =
        moments_mean_var(samples.shape(), &sample_data, event_ndim);

    let lp = dist
        .log_prob(value)
        .map_err(|e| format!("log_prob: {e:?}"))?;
    let lp_data = lp.data_vec().map_err(|e| format!("log_prob data: {e:?}"))?;
    let lp_shape = lp.shape().to_vec();

    let (aux_shape, aux_data, has_aux) = match dist.entropy() {
        Ok(t) => {
            let d = t.data_vec().map_err(|e| format!("entropy data: {e:?}"))?;
            let s = t.shape().to_vec();
            let s = if s.is_empty() { vec![1] } else { s };
            (s, d, true)
        }
        Err(_) => (vec![1], vec![0.0f32], false),
    };

    Ok(DumpResult {
        mean_shape: m_shape,
        mean_data: mean,
        var_data: var,
        log_prob_shape: lp_shape,
        log_prob_data: lp_data,
        aux_shape,
        aux_data,
        is_kl: false,
        has_aux,
    })
}

/// Build (P, Q) for a KL config and emit only the KL tensor.
fn dump_kl(config: &str) -> Result<DumpResult, String> {
    let one_scalar = || vec![1usize];
    let kl_tensor: Tensor<f32> = match config {
        "kl_normal_normal" => {
            let p = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("p: {e:?}"))?;
            let q = Normal::new(scalar(1.0f32).unwrap(), scalar(2.0f32).unwrap())
                .map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        "kl_bernoulli_bernoulli" => {
            let p = Bernoulli::new(scalar(0.3f32).unwrap())
                .map_err(|e| format!("p: {e:?}"))?;
            let q = Bernoulli::new(scalar(0.5f32).unwrap())
                .map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        "kl_uniform_uniform" => {
            let p = Uniform::new(scalar(-1.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("p: {e:?}"))?;
            let q = Uniform::new(scalar(-2.0f32).unwrap(), scalar(2.0f32).unwrap())
                .map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        "kl_categorical_categorical" => {
            let p_probs = from_slice(&[0.1f32, 0.3, 0.4, 0.2], &[4])
                .map_err(|e| format!("p_probs: {e:?}"))?;
            let q_probs = from_slice(&[0.25f32, 0.25, 0.25, 0.25], &[4])
                .map_err(|e| format!("q_probs: {e:?}"))?;
            let p =
                Categorical::new(p_probs).map_err(|e| format!("p: {e:?}"))?;
            let q =
                Categorical::new(q_probs).map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        "kl_laplace_laplace" => {
            let p = Laplace::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("p: {e:?}"))?;
            let q = Laplace::new(scalar(0.5f32).unwrap(), scalar(2.0f32).unwrap())
                .map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        "kl_exponential_exponential" => {
            let p = Exponential::new(scalar(1.5f32).unwrap())
                .map_err(|e| format!("p: {e:?}"))?;
            let q = Exponential::new(scalar(1.0f32).unwrap())
                .map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        "kl_gamma_gamma" => {
            let p = Gamma::new(scalar(2.0f32).unwrap(), scalar(1.0f32).unwrap())
                .map_err(|e| format!("p: {e:?}"))?;
            let q = Gamma::new(scalar(3.0f32).unwrap(), scalar(2.0f32).unwrap())
                .map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        "kl_poisson_poisson" => {
            let p = Poisson::new(scalar(3.0f32).unwrap())
                .map_err(|e| format!("p: {e:?}"))?;
            let q = Poisson::new(scalar(5.0f32).unwrap())
                .map_err(|e| format!("q: {e:?}"))?;
            kl_divergence::<f32, _, _>(&p, &q).map_err(|e| format!("kl: {e:?}"))?
        }
        other => return Err(format!("unknown KL config {other:?}")),
    };

    let kl_data = kl_tensor.data_vec().map_err(|e| format!("kl data: {e:?}"))?;
    let kl_shape = kl_tensor.shape().to_vec();
    let kl_shape = if kl_shape.is_empty() { vec![1] } else { kl_shape };

    Ok(DumpResult {
        mean_shape: one_scalar(),
        mean_data: vec![0.0],
        var_data: vec![0.0],
        log_prob_shape: one_scalar(),
        log_prob_data: vec![0.0],
        aux_shape: kl_shape,
        aux_data: kl_data,
        is_kl: true,
        has_aux: true,
    })
}

// ---------------------------------------------------------------------------
// Main flow.
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args = parse_args()?;
    eprintln!(
        "[distributions_dump] config={} fixture_dir={} output_dir={}",
        args.config,
        args.fixture_dir.display(),
        args.output_dir.display(),
    );
    std::fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("mkdir output_dir: {e}"))?;

    let result = if args.config.starts_with("kl_") {
        dump_kl(&args.config)?
    } else {
        // Read test_points from the fixture.
        let tp_path = args.fixture_dir.join("test_points.bin");
        let (tp_shape, tp_data) = read_single_tensor_f32(&tp_path)?;
        dump_distribution(&args.config, &tp_shape, &tp_data)?
    };

    // Write outputs.
    if !result.is_kl {
        write_single_tensor_f32(
            &args.output_dir.join(format!("{}_mean.bin", args.config)),
            &result.mean_shape,
            &result.mean_data,
        )
        .map_err(|e| format!("write mean: {e}"))?;
        write_single_tensor_f32(
            &args.output_dir.join(format!("{}_var.bin", args.config)),
            &result.mean_shape,
            &result.var_data,
        )
        .map_err(|e| format!("write var: {e}"))?;
        write_single_tensor_f32(
            &args.output_dir.join(format!("{}_log_prob.bin", args.config)),
            &result.log_prob_shape,
            &result.log_prob_data,
        )
        .map_err(|e| format!("write log_prob: {e}"))?;
        if result.has_aux {
            write_single_tensor_f32(
                &args.output_dir.join(format!("{}_entropy.bin", args.config)),
                &result.aux_shape,
                &result.aux_data,
            )
            .map_err(|e| format!("write entropy: {e}"))?;
        }
    } else {
        write_single_tensor_f32(
            &args.output_dir.join(format!("{}_kl.bin", args.config)),
            &result.aux_shape,
            &result.aux_data,
        )
        .map_err(|e| format!("write kl: {e}"))?;
    }

    // JSON verdict line.
    println!(
        "{{\"config\":\"{}\",\"is_kl\":{},\"has_entropy\":{}}}",
        args.config, result.is_kl, result.has_aux,
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[distributions_dump] error: {e}");
        std::process::exit(1);
    }
}
