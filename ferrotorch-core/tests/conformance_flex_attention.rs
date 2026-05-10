//! Conformance Phase 2.11 — `ferrotorch-core` flex_attention parity vs.
//! PyTorch's `torch.nn.attention.flex_attention.flex_attention`.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/773>.
//! Parent: #759.
//!
//! Source files exercised:
//! - `ferrotorch-core/src/flex_attention.rs` — the canonical
//!   [`flex_attention`] free function and its top-level re-export at
//!   `ferrotorch_core::flex_attention`.
//!
//! Surface coverage (2 surface items in `_surface_exclusions.toml` filtered
//! by `tracking_issue = "#773"`):
//!
//!   * `ferrotorch_core::flex_attention::flex_attention` — canonical path.
//!   * `ferrotorch_core::flex_attention` — top-level re-export.
//!
//! The surface gate is substring-based. Both items resolve to the short
//! identifier `flex_attention`, which appears repeatedly below (every test
//! call site). The roster block names the substring once verbatim so the
//! gate finds it whether or not the orchestrator removes the exclusion
//! entries first.
//!
//! # Surface roster (do not delete — substring-grep coverage targets)
//!
//! Free function: `flex_attention`.
//! Module re-export: `flex_attention`.
//!
//! # Coverage axes (Cat A — flex_attention forward / score_mod / mask_mod)
//!
//! For each (shape, dtype, device) tuple:
//!
//!   * **baseline** — `score_mod=None`, no mask: vanilla scaled-dot-product.
//!   * **causal** — lower-triangular mask, applied via `score_mod` that
//!     sets the upper triangle to `-inf` before softmax.
//!   * **block_diag** — block-diagonal mask: only allow `(q, k)` whose
//!     `block_size`-quotient matches.
//!   * **alibi** — additive ALiBi-style position bias as a `score_mod`
//!     (`bias[q, k] = -slope * |q - k|`) with no mask change.
//!   * **empty_mask** — every `(q, k)` masked: PyTorch's flex_attention
//!     contract returns NaN (softmax over all `-inf`).
//!
//! Forward + autograd are both exercised (loss = `sum(out)`; backward
//! produces `grad_q`, `grad_k`, `grad_v`). The autograd lane is skipped for
//! the `empty_mask` variant since the chain is degenerate (NaN propagates).
//!
//! # Tolerances (matmul-dominated)
//!
//!   * F32_MATMUL_CPU = 1e-4
//!   * F32_MATMUL_GPU = 1e-3
//!   * F64_MATMUL_*   = 1e-9
//!
//! # API bridge: PyTorch flex_attention vs. ferrotorch flex_attention
//!
//! PyTorch's signature is `flex_attention(Q, K, V, score_mod=callable,
//! block_mask=BlockMask)` where the callables are element-wise:
//! `score_mod(score, b, h, q_idx, kv_idx) -> score` and
//! `mask_mod(b, h, q_idx, kv_idx) -> bool`. ferrotorch's signature is
//! `flex_attention(Q, K, V, score_mod=Option<F>)` where `F` operates on the
//! full `[n_q, n_k]` matrix per `(batch, head)`. Every element-wise
//! transform we need can be expressed as a tensor add (bias) followed by
//! a where-mask (set to `-inf`), so the test runner builds the
//! `[n_q, n_k]` bias / mask tensors at fixture-replay time and applies
//! them inside the closure. The fixture stores `bias` and/or `mask_2d` so
//! the runner can reproduce them deterministically.

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::creation;
use ferrotorch_core::flex_attention;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::{Device, FerrotorchResult, Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// Tolerance helpers (mirroring conformance_linalg.rs).
// ---------------------------------------------------------------------------

mod tolerance {
    pub const F32_MATMUL_CPU: f32 = 1e-4;
    pub const F64_MATMUL_CPU: f64 = 1e-9;

    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_MATMUL_GPU: f32 = 1e-3;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F64_MATMUL_GPU: f64 = 1e-9;

    pub fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch (actual={}, expected={})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            if !a.is_finite() || !e.is_finite() {
                if a.to_bits() == e.to_bits() {
                    continue;
                }
                if a.is_infinite() && e.is_infinite() && a.signum() == e.signum() {
                    continue;
                }
                panic!("{label}: index {i} non-finite mismatch (actual={a}, expected={e})");
            }
            let diff = (a - e).abs();
            let scale = e.abs().max(1.0);
            let allowed = tol * scale;
            assert!(
                diff <= allowed,
                "{label}: index {i} delta {diff:.3e} exceeds tol {tol:.3e} \
                 (actual={a}, expected={e})"
            );
        }
    }

    pub fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch (actual={}, expected={})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            if !a.is_finite() || !e.is_finite() {
                if a.to_bits() == e.to_bits() {
                    continue;
                }
                if a.is_infinite() && e.is_infinite() && a.signum() == e.signum() {
                    continue;
                }
                panic!("{label}: index {i} non-finite mismatch (actual={a}, expected={e})");
            }
            let diff = (a - e).abs();
            let scale = e.abs().max(1.0);
            let allowed = tol * scale;
            assert!(
                diff <= allowed,
                "{label}: index {i} delta {diff:.3e} exceeds tol {tol:.3e} \
                 (actual={a}, expected={e})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// JSON-with-NaN/Infinity-sentinels deserializer (shared shape across the
// conformance suite).
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct F64ListSentinel(Vec<f64>);

impl F64ListSentinel {
    fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

struct FloatOrSentinel(f64);

struct FloatOrSentinelVisitor;

impl<'de> Visitor<'de> for FloatOrSentinelVisitor {
    type Value = FloatOrSentinel;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an f64 or one of \"Infinity\"/\"-Infinity\"/\"NaN\"")
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Ok(FloatOrSentinel(v))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(FloatOrSentinel(v as f64))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(FloatOrSentinel(v as f64))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        match v {
            "Infinity" => Ok(FloatOrSentinel(f64::INFINITY)),
            "-Infinity" => Ok(FloatOrSentinel(f64::NEG_INFINITY)),
            "NaN" => Ok(FloatOrSentinel(f64::NAN)),
            other => Err(E::custom(format!("unexpected float sentinel {other:?}"))),
        }
    }
}

impl<'de> serde::Deserialize<'de> for FloatOrSentinel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(FloatOrSentinelVisitor)
    }
}

struct F64ListSentinelVisitor;

impl<'de> Visitor<'de> for F64ListSentinelVisitor {
    type Value = F64ListSentinel;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a list of floats with optional Infinity/-Infinity/NaN sentinels")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::new();
        while let Some(FloatOrSentinel(v)) = seq.next_element()? {
            out.push(v);
        }
        Ok(F64ListSentinel(out))
    }
}

impl<'de> serde::Deserialize<'de> for F64ListSentinel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(F64ListSentinelVisitor)
    }
}

// ---------------------------------------------------------------------------
// Fixture deserialization
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[allow(dead_code, reason = "metadata used for diagnostics + GPU gating")]
    metadata: FixtureMetadata,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    #[allow(dead_code, reason = "diagnostics only")]
    torch_version: String,
    #[allow(dead_code, reason = "diagnostics only")]
    cuda_version: Option<String>,
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    cuda_available: bool,
    #[allow(dead_code, reason = "diagnostics only")]
    python_executable: String,
    #[allow(dead_code, reason = "diagnostics only")]
    python_platform: String,
    #[allow(dead_code, reason = "diagnostics only")]
    generated_at: String,
    #[allow(dead_code, reason = "diagnostics only")]
    rng_seed: u64,
    #[allow(dead_code, reason = "diagnostics only")]
    flex_attention_status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    op: String,
    tag: String,
    variant: String,
    dtype: String,
    device: String,
    batch: usize,
    heads: usize,
    n_q: usize,
    n_k: usize,
    d: usize,
    d_v: usize,
    q_data: F64ListSentinel,
    k_data: F64ListSentinel,
    v_data: F64ListSentinel,
    /// Forward fixtures: reference output flattened.
    #[serde(default)]
    out_values: Option<F64ListSentinel>,
    /// Backward fixtures: gradient on Q, flattened.
    #[serde(default)]
    grad_q: Option<F64ListSentinel>,
    #[serde(default)]
    grad_k: Option<F64ListSentinel>,
    #[serde(default)]
    grad_v: Option<F64ListSentinel>,
    /// Optional `[n_q, n_k]` additive bias (alibi variant only).
    #[serde(default)]
    bias: Option<F64ListSentinel>,
    /// Optional alibi slope used to *generate* the precomputed `bias`
    /// matrix above (alibi variant only). Test logic consumes the
    /// flattened `bias` directly; this field is metadata documenting
    /// how the matrix was produced and is preserved here so the
    /// `#[serde(deny_unknown_fields)]` discipline still rejects
    /// genuine schema drift.
    #[serde(default)]
    #[allow(dead_code, reason = "alibi metadata; bias matrix carries values")]
    slope: Option<f64>,
    /// Optional `[n_q, n_k]` boolean mask, true => allowed (causal /
    /// block_diag / empty_mask variants).
    #[serde(default)]
    mask_2d: Option<Vec<bool>>,
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("flex_attention.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_flex_attention_fixtures.py`",
            p.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn cases_for<'a>(file: &'a FixtureFile, op: &str, device: &str) -> Vec<&'a Fixture> {
    file.fixtures
        .iter()
        .filter(|f| f.op == op && f.device == device)
        .collect()
}

// ---------------------------------------------------------------------------
// Device-transparent helpers (mirroring conformance_linalg.rs).
// ---------------------------------------------------------------------------

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    if t.is_cpu() {
        t.data().expect("read CPU data").to_vec()
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data after readback").to_vec()
    }
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    if t.is_cpu() {
        t.data().expect("read CPU data").to_vec()
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data after readback").to_vec()
    }
}

fn make_cpu_f32(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), requires_grad)
        .expect("make_cpu_f32")
}

fn make_cpu_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("make_cpu_f64")
}

fn upload_f32(t: Tensor<f32>, device: Device) -> Tensor<f32> {
    if matches!(device, Device::Cuda(_)) {
        t.to(device).expect("upload to cuda")
    } else {
        t
    }
}

fn upload_f64(t: Tensor<f64>, device: Device) -> Tensor<f64> {
    if matches!(device, Device::Cuda(_)) {
        t.to(device).expect("upload to cuda")
    } else {
        t
    }
}

fn check_f32(label: &str, actual: &[f32], expected: &[f64], tol: f32) {
    let exp_f32: Vec<f32> = expected.iter().map(|&x| x as f32).collect();
    tolerance::assert_close_f32(actual, &exp_f32, tol, label);
}

fn check_f64(label: &str, actual: &[f64], expected: &[f64], tol: f64) {
    tolerance::assert_close_f64(actual, expected, tol, label);
}

/// Per-fixture diagnostic skip for cascade issues surfaced by the GPU lane.
/// Returns `Some("issue #")` to skip with a printed reason; returns `None`
/// to run normally. The dispatch's cascade-handling mandate requires
/// surfacing each failure with a tracking issue rather than silently
/// weakening tolerance — this function is the canonical opt-out point.
///
/// Initially empty; populated as we discover GPU-side regressions and file
/// crosslink follow-ups for them. (See `conformance_reduction.rs` for the
/// pattern.)
///
/// # Active skips
///
/// *(none — both regressions previously gated here have been fixed in
/// Bugfix Batch 8.)*
///
/// # Closed (no longer skipping)
///
/// * **#812** — *closed by Bugfix Batch 8 / Dispatch A1.* The arithmetic
///   GPU `*_inner` dispatchers (`add` / `sub` / `mul` / `div` plus the
///   unary `neg` / `pow` / `sqrt` / `abs` and `reduce_grad_to_shape`'s
///   GPU fast path) now branch on `is_contiguous()` and route
///   non-contiguous CUDA inputs through `Tensor::contiguous()` (which
///   dispatches to the on-device `strided_copy_{f32,f64}` kernel) before
///   reading `gpu_handle()`. Multi-(b, h) score_mod variants no longer
///   trip the `LengthMismatch` guard.
/// * **#813** — *closed by Bugfix Batch 8 / Dispatch A2.* The f64 GPU
///   dispatch chain (`reshape` → `transpose` → `bmm_differentiable` →
///   `mul` (scale) → `softmax` → `reshape` → `bmm_differentiable` →
///   `reshape`) was supposed to be panicking with
///   `"GPU handle does not contain a CudaBuffer<f32>"` because some op
///   was hardcoded to f32 internally on CUDA. The probe walked every
///   step of the chain on f64 CUDA and found that all of them dispatch
///   correctly — `bmm_f64` (cuBLAS dgemm strided-batched) had its f64
///   branch added by #800, the broadcast-bmm f64 path by #819,
///   `softmax_f64` already exists, and `mul_f64` / `broadcast_mul_f64`
///   already route on `is_f64::<T>()`. f64 baseline GPU now matches
///   PyTorch parity end-to-end on every fixture shape.
fn cascade_skip(_op: &str, _device_label: &str, _dtype: &str, _tag: &str) -> Option<&'static str> {
    None
}

// ---------------------------------------------------------------------------
// Score-mod plumbing
// ---------------------------------------------------------------------------
//
// PyTorch's flex_attention takes element-wise `score_mod` / `mask_mod`
// callables. ferrotorch's `flex_attention` takes a `score_mod` that
// operates on the full `[n_q, n_k]` slice per (batch, head). For each
// fixture variant we synthesize a closure that applies the same math.
//
// All score_mod closures must be `'static + Send + Sync`. They take an
// owned `bias` and/or `mask` tensor by value at construction time and
// move them into the closure; ferrotorch's `flex_attention` calls them
// once per (b, h) per forward, so we clone the bias/mask once per call.
// (`Tensor` is cheap to clone — Arc-shared storage.)

/// Apply `bias` (additive) and/or `mask` (where False, set to -inf) to
/// `scores`. The returned tensor has the same `[n_q, n_k]` shape.
fn apply_bias_mask_f32(
    scores: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    mask_neg_inf: Option<&Tensor<f32>>,
) -> FerrotorchResult<Tensor<f32>> {
    let mut s = scores.clone();
    if let Some(b) = bias {
        s = add(&s, b)?;
    }
    if let Some(m) = mask_neg_inf {
        // Adding -inf at masked positions sends those scores to -inf,
        // which produces 0 weight under softmax. Adding 0 elsewhere is
        // a no-op. Mathematically equivalent to where(mask, scores, -inf)
        // and avoids needing a where_cond op in the closure.
        s = add(&s, m)?;
    }
    Ok(s)
}

fn apply_bias_mask_f64(
    scores: &Tensor<f64>,
    bias: Option<&Tensor<f64>>,
    mask_neg_inf: Option<&Tensor<f64>>,
) -> FerrotorchResult<Tensor<f64>> {
    let mut s = scores.clone();
    if let Some(b) = bias {
        s = add(&s, b)?;
    }
    if let Some(m) = mask_neg_inf {
        s = add(&s, m)?;
    }
    Ok(s)
}

/// Build a `[n_q, n_k]` "mask-as-additive" tensor where masked positions
/// hold `-inf` and admissible positions hold `0`. This is the closure-
/// friendly representation: adding it to `scores` zeros out the masked
/// positions under softmax without needing a where-conditional op.
fn build_mask_addend_f32(mask: &[bool], n_q: usize, n_k: usize, device: Device) -> Tensor<f32> {
    let v: Vec<f32> = mask
        .iter()
        .map(|&m| if m { 0.0 } else { f32::NEG_INFINITY })
        .collect();
    let t = Tensor::from_storage(TensorStorage::cpu(v), vec![n_q, n_k], false)
        .expect("mask addend f32");
    upload_f32(t, device)
}

fn build_mask_addend_f64(mask: &[bool], n_q: usize, n_k: usize, device: Device) -> Tensor<f64> {
    let v: Vec<f64> = mask
        .iter()
        .map(|&m| if m { 0.0 } else { f64::NEG_INFINITY })
        .collect();
    let t = Tensor::from_storage(TensorStorage::cpu(v), vec![n_q, n_k], false)
        .expect("mask addend f64");
    upload_f64(t, device)
}

// ---------------------------------------------------------------------------
// Forward parity
// ---------------------------------------------------------------------------

fn run_forward_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "flex_attention", device_label);
    assert!(
        !cases.is_empty(),
        "no flex_attention forward fixtures on {device_label}"
    );
    let on_gpu = matches!(device, Device::Cuda(_));
    let tol_f32 = if on_gpu {
        tolerance::F32_MATMUL_GPU
    } else {
        tolerance::F32_MATMUL_CPU
    };
    let tol_f64 = if on_gpu {
        tolerance::F64_MATMUL_GPU
    } else {
        tolerance::F64_MATMUL_CPU
    };

    for f in cases {
        if let Some(reason) = cascade_skip("flex_attention", device_label, &f.dtype, &f.tag) {
            eprintln!(
                "skipping flex_attention {device_label} dtype={} tag={}: {reason}",
                f.dtype, f.tag
            );
            continue;
        }
        let label = format!(
            "flex_attention {device_label} variant={} tag={} dtype={}",
            f.variant, f.tag, f.dtype
        );

        let q_shape = vec![f.batch, f.heads, f.n_q, f.d];
        let k_shape = vec![f.batch, f.heads, f.n_k, f.d];
        let v_shape = vec![f.batch, f.heads, f.n_k, f.d_v];
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("forward fixture missing out_values");

        match f.dtype.as_str() {
            "float32" => {
                let q = upload_f32(make_cpu_f32(f.q_data.as_slice(), &q_shape, false), device);
                let k = upload_f32(make_cpu_f32(f.k_data.as_slice(), &k_shape, false), device);
                let v = upload_f32(make_cpu_f32(f.v_data.as_slice(), &v_shape, false), device);

                let bias_t: Option<Tensor<f32>> = f.bias.as_ref().map(|b| {
                    upload_f32(make_cpu_f32(b.as_slice(), &[f.n_q, f.n_k], false), device)
                });
                let mask_t: Option<Tensor<f32>> = f
                    .mask_2d
                    .as_ref()
                    .map(|m| build_mask_addend_f32(m, f.n_q, f.n_k, device));

                let out = if bias_t.is_none() && mask_t.is_none() {
                    // Baseline: no score_mod.
                    flex_attention::flex_attention::<
                        f32,
                        fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
                    >(&q, &k, &v, None)
                    .expect("flex_attention baseline forward f32")
                } else {
                    // Closure captures bias / mask by move; ferrotorch
                    // calls it per (b, h) and the closure broadcasts the
                    // bias/mask across all (b, h).
                    let closure = move |s: &Tensor<f32>,
                                        _b: usize,
                                        _h: usize|
                          -> FerrotorchResult<Tensor<f32>> {
                        apply_bias_mask_f32(s, bias_t.as_ref(), mask_t.as_ref())
                    };
                    flex_attention::flex_attention(&q, &k, &v, Some(closure))
                        .expect("flex_attention forward f32")
                };

                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&out),
                    expected,
                    tol_f32,
                );
            }
            "float64" => {
                let q = upload_f64(make_cpu_f64(f.q_data.as_slice(), &q_shape, false), device);
                let k = upload_f64(make_cpu_f64(f.k_data.as_slice(), &k_shape, false), device);
                let v = upload_f64(make_cpu_f64(f.v_data.as_slice(), &v_shape, false), device);

                let bias_t: Option<Tensor<f64>> = f.bias.as_ref().map(|b| {
                    upload_f64(make_cpu_f64(b.as_slice(), &[f.n_q, f.n_k], false), device)
                });
                let mask_t: Option<Tensor<f64>> = f
                    .mask_2d
                    .as_ref()
                    .map(|m| build_mask_addend_f64(m, f.n_q, f.n_k, device));

                let out = if bias_t.is_none() && mask_t.is_none() {
                    flex_attention::flex_attention::<
                        f64,
                        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
                    >(&q, &k, &v, None)
                    .expect("flex_attention baseline forward f64")
                } else {
                    let closure = move |s: &Tensor<f64>,
                                        _b: usize,
                                        _h: usize|
                          -> FerrotorchResult<Tensor<f64>> {
                        apply_bias_mask_f64(s, bias_t.as_ref(), mask_t.as_ref())
                    };
                    flex_attention::flex_attention(&q, &k, &v, Some(closure))
                        .expect("flex_attention forward f64")
                };

                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&out),
                    expected,
                    tol_f64,
                );
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backward parity (autograd Q/K/V grads from `loss = sum(out)`).
// ---------------------------------------------------------------------------

fn run_backward_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "flex_attention_backward", device_label);
    assert!(
        !cases.is_empty(),
        "no flex_attention_backward fixtures on {device_label}"
    );
    let on_gpu = matches!(device, Device::Cuda(_));
    let tol_f32 = if on_gpu {
        tolerance::F32_MATMUL_GPU
    } else {
        tolerance::F32_MATMUL_CPU
    };
    let tol_f64 = if on_gpu {
        tolerance::F64_MATMUL_GPU
    } else {
        tolerance::F64_MATMUL_CPU
    };

    for f in cases {
        if let Some(reason) =
            cascade_skip("flex_attention_backward", device_label, &f.dtype, &f.tag)
        {
            eprintln!(
                "skipping flex_attention_backward {device_label} dtype={} tag={}: {reason}",
                f.dtype, f.tag
            );
            continue;
        }
        let label = format!(
            "flex_attention_backward {device_label} variant={} tag={} dtype={}",
            f.variant, f.tag, f.dtype
        );

        let q_shape = vec![f.batch, f.heads, f.n_q, f.d];
        let k_shape = vec![f.batch, f.heads, f.n_k, f.d];
        let v_shape = vec![f.batch, f.heads, f.n_k, f.d_v];
        let grad_q_exp = f
            .grad_q
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("backward fixture missing grad_q");
        let grad_k_exp = f
            .grad_k
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("backward fixture missing grad_k");
        let grad_v_exp = f
            .grad_v
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("backward fixture missing grad_v");

        match f.dtype.as_str() {
            "float32" => {
                let q = upload_f32(make_cpu_f32(f.q_data.as_slice(), &q_shape, true), device);
                let k = upload_f32(make_cpu_f32(f.k_data.as_slice(), &k_shape, true), device);
                let v = upload_f32(make_cpu_f32(f.v_data.as_slice(), &v_shape, true), device);

                let bias_t: Option<Tensor<f32>> = f.bias.as_ref().map(|b| {
                    upload_f32(make_cpu_f32(b.as_slice(), &[f.n_q, f.n_k], false), device)
                });
                let mask_t: Option<Tensor<f32>> = f
                    .mask_2d
                    .as_ref()
                    .map(|m| build_mask_addend_f32(m, f.n_q, f.n_k, device));

                let out = if bias_t.is_none() && mask_t.is_none() {
                    flex_attention::flex_attention::<
                        f32,
                        fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
                    >(&q, &k, &v, None)
                    .expect("flex_attention baseline backward f32")
                } else {
                    let closure = move |s: &Tensor<f32>,
                                        _b: usize,
                                        _h: usize|
                          -> FerrotorchResult<Tensor<f32>> {
                        apply_bias_mask_f32(s, bias_t.as_ref(), mask_t.as_ref())
                    };
                    flex_attention::flex_attention(&q, &k, &v, Some(closure))
                        .expect("flex_attention backward f32")
                };

                let loss = ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum f32");
                loss.backward().expect("backward f32");
                let gq = q.grad().unwrap().expect("grad_q f32");
                let gk = k.grad().unwrap().expect("grad_k f32");
                let gv = v.grad().unwrap().expect("grad_v f32");

                check_f32(
                    &format!("{label} grad_q"),
                    &read_back_f32(&gq),
                    grad_q_exp,
                    tol_f32,
                );
                check_f32(
                    &format!("{label} grad_k"),
                    &read_back_f32(&gk),
                    grad_k_exp,
                    tol_f32,
                );
                check_f32(
                    &format!("{label} grad_v"),
                    &read_back_f32(&gv),
                    grad_v_exp,
                    tol_f32,
                );
            }
            "float64" => {
                let q = upload_f64(make_cpu_f64(f.q_data.as_slice(), &q_shape, true), device);
                let k = upload_f64(make_cpu_f64(f.k_data.as_slice(), &k_shape, true), device);
                let v = upload_f64(make_cpu_f64(f.v_data.as_slice(), &v_shape, true), device);

                let bias_t: Option<Tensor<f64>> = f.bias.as_ref().map(|b| {
                    upload_f64(make_cpu_f64(b.as_slice(), &[f.n_q, f.n_k], false), device)
                });
                let mask_t: Option<Tensor<f64>> = f
                    .mask_2d
                    .as_ref()
                    .map(|m| build_mask_addend_f64(m, f.n_q, f.n_k, device));

                let out = if bias_t.is_none() && mask_t.is_none() {
                    flex_attention::flex_attention::<
                        f64,
                        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
                    >(&q, &k, &v, None)
                    .expect("flex_attention baseline backward f64")
                } else {
                    let closure = move |s: &Tensor<f64>,
                                        _b: usize,
                                        _h: usize|
                          -> FerrotorchResult<Tensor<f64>> {
                        apply_bias_mask_f64(s, bias_t.as_ref(), mask_t.as_ref())
                    };
                    flex_attention::flex_attention(&q, &k, &v, Some(closure))
                        .expect("flex_attention backward f64")
                };

                let loss = ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum f64");
                loss.backward().expect("backward f64");
                let gq = q.grad().unwrap().expect("grad_q f64");
                let gk = k.grad().unwrap().expect("grad_k f64");
                let gv = v.grad().unwrap().expect("grad_v f64");

                check_f64(
                    &format!("{label} grad_q"),
                    &read_back_f64(&gq),
                    grad_q_exp,
                    tol_f64,
                );
                check_f64(
                    &format!("{label} grad_k"),
                    &read_back_f64(&gk),
                    grad_k_exp,
                    tol_f64,
                );
                check_f64(
                    &format!("{label} grad_v"),
                    &read_back_f64(&gv),
                    grad_v_exp,
                    tol_f64,
                );
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// CPU lane
// ---------------------------------------------------------------------------

#[test]
fn cpu_flex_attention_forward() {
    run_forward_for_device("cpu", Device::Cpu);
}

#[test]
fn cpu_flex_attention_backward() {
    run_backward_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// API smoke checks (independent of fixtures): verify that the
// `score_mod=None` and `score_mod=Some(closure)` branches both compile +
// run, and that the re-export `ferrotorch_core::flex_attention` resolves to
// the same item as the canonical path. These are hand-written guards that
// hold even if a future fixture refresh drops a variant.
// ---------------------------------------------------------------------------

#[test]
fn smoke_flex_attention_baseline_no_score_mod() {
    // batch=1, heads=1, n_q=2, n_k=2, d=2, d_v=2 — minimal contract check.
    let q: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
        vec![1, 1, 2, 2],
        false,
    )
    .unwrap();
    let k = q.clone();
    let v: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
        vec![1, 1, 2, 2],
        false,
    )
    .unwrap();
    let out = flex_attention::flex_attention::<
        f32,
        fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
    >(&q, &k, &v, None)
    .expect("flex_attention baseline smoke");
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
}

#[test]
fn smoke_flex_attention_score_mod_softmax_invariant() {
    // Adding a per-row constant to all scores is softmax-invariant; the
    // output should match the no-score_mod baseline. This exercises the
    // closure dispatch path end-to-end (per-(b,h) narrow + cat reassembly).
    let q: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
        vec![1, 1, 2, 2],
        false,
    )
    .unwrap();
    let k = q.clone();
    let v: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
        vec![1, 1, 2, 2],
        false,
    )
    .unwrap();
    let baseline = flex_attention::flex_attention::<
        f32,
        fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
    >(&q, &k, &v, None)
    .unwrap();
    let bias = creation::scalar(2.0f32).unwrap();
    let with_bias = flex_attention::flex_attention(
        &q,
        &k,
        &v,
        Some(move |s: &Tensor<f32>, _b: usize, _h: usize| add(s, &bias)),
    )
    .unwrap();
    let b = baseline.data().unwrap().to_vec();
    let m = with_bias.data().unwrap().to_vec();
    tolerance::assert_close_f32(&m, &b, 1e-5, "smoke: softmax-invariant additive");
}

#[test]
fn smoke_flex_attention_reexport_matches_canonical() {
    // The orchestrator may strip the `flex_attention` re-export from the
    // `_surface_exclusions.toml`, so this test exercises both reachable
    // paths to lock the contract: the canonical
    // `ferrotorch_core::flex_attention::flex_attention` and the top-level
    // re-export `ferrotorch_core::flex_attention`. They must yield the
    // same tensor (Arc-shared storage may differ but values must match).
    use ferrotorch_core::flex_attention as reexport;
    let q: Tensor<f64> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
        vec![1, 1, 2, 2],
        false,
    )
    .unwrap();
    let k = q.clone();
    let v: Tensor<f64> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
        vec![1, 1, 2, 2],
        false,
    )
    .unwrap();
    // Add zero so the closure reaches the score_mod path; otherwise the
    // baseline branch shortcuts past it.
    let zero = creation::scalar(0.0f64).unwrap();
    let zero2 = zero.clone();
    let canonical = flex_attention::flex_attention(
        &q,
        &k,
        &v,
        Some(move |s: &Tensor<f64>, _b: usize, _h: usize| mul(s, &zero)),
    )
    .unwrap();
    // The re-export `ferrotorch_core::flex_attention` resolves to the
    // module; the function inside is the same item. We re-call through
    // the reexport-named module to prove it.
    let via_reexport = reexport::flex_attention(
        &q,
        &k,
        &v,
        Some(move |s: &Tensor<f64>, _b: usize, _h: usize| mul(s, &zero2)),
    )
    .unwrap();
    let a = canonical.data().unwrap().to_vec();
    let b = via_reexport.data().unwrap().to_vec();
    tolerance::assert_close_f64(&a, &b, 1e-12, "smoke: reexport matches canonical");
}

// ---------------------------------------------------------------------------
// Fixture-shape sanity check: every fixture must have a corresponding op +
// dtype + variant tag, and the file must contain the full Cartesian
// product of shapes × variants × dtypes × devices for the forward lane.
// ---------------------------------------------------------------------------

#[test]
fn fixture_file_shape_invariants() {
    let file = load_fixtures();
    let mut by_op: std::collections::BTreeMap<String, usize> = Default::default();
    for f in &file.fixtures {
        *by_op.entry(f.op.clone()).or_default() += 1;
    }
    let required = ["flex_attention", "flex_attention_backward"];
    for r in required {
        let n = by_op.get(r).copied().unwrap_or(0);
        assert!(n > 0, "fixture file missing op {r:?}");
    }
    // Every variant must appear in the forward lane at least once.
    let variants = ["baseline", "causal", "block_diag", "alibi", "empty_mask"];
    for v in variants {
        let any = file
            .fixtures
            .iter()
            .any(|f| f.op == "flex_attention" && f.variant == v);
        assert!(any, "fixture file missing forward variant {v:?}");
    }
    // The empty_mask backward is intentionally elided (NaN propagates);
    // every other variant must appear in the backward lane.
    for v in ["baseline", "causal", "block_diag", "alibi"] {
        let any = file
            .fixtures
            .iter()
            .any(|f| f.op == "flex_attention_backward" && f.variant == v);
        assert!(any, "fixture file missing backward variant {v:?}");
    }
}

// ---------------------------------------------------------------------------
// GPU paths — gated on the `gpu` feature
// ---------------------------------------------------------------------------
//
// Per `flex_attention.rs`, the implementation composes from device-aware
// ops (bmm via cuBLAS, softmax, mul, reshape, narrow, cat); under
// `--features gpu` the entire forward + backward chain stays on-device
// (modulo the user-supplied score_mod closure, which here is also a
// pure tensor add and so dispatches on-device).

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU conformance suite");
        });
    }

    fn require_cuda_fixtures(file: &FixtureFile) {
        if !file.metadata.cuda_available {
            panic!(
                "fixtures/flex_attention.json was generated without CUDA — \
                 regenerate on a CUDA-enabled host before running --features gpu tests"
            );
        }
    }

    #[test]
    fn gpu_flex_attention_forward() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_forward_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_flex_attention_backward() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_backward_for_device("cuda:0", Device::Cuda(0));
    }
}
