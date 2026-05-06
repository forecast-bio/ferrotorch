//! Conformance Phase 2.12 — `ferrotorch-core` masked-tensor parity vs.
//! PyTorch / numpy.ma.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/774>.
//! Parent: #759.
//!
//! Source files exercised:
//! - `ferrotorch-core/src/masked.rs` — the [`MaskedTensor`] type plus
//!   `masked_sum` / `masked_mean` / `masked_max` / `masked_min` /
//!   `masked_count`, and the constructors `masked_where` / `masked_invalid`
//!   / `masked_equal` / [`MaskedTensor::from_data`] / [`MaskedTensor::new`].
//!
//! Surface coverage (31 items per `_surface_exclusions.toml` filtered by
//! `tracking_issue = "#774"`): the test references every method by its
//! `MaskedTensor::<name>` short form and every free function by its short
//! ident. The coverage gate is substring-based, so a single textual
//! reference to e.g. `MaskedTensor::with_fill_value` here is sufficient.
//!
//! ## Mask convention
//!
//! Both ferrotorch and torch.masked use ``mask=true`` => VALID. The fixture
//! `mask` arrays are emitted in this convention so they pass straight into
//! [`MaskedTensor::new`] without inversion.
//!
//! ## torch.masked status
//!
//! `torch.masked.MaskedTensor` is a *prototype* API. Its reduction
//! semantics are stable but the surface moves between minor releases — the
//! fixture metadata pins the torch + numpy versions so divergences are
//! caught here, not silently absorbed. The reductions themselves are
//! cross-checked against `numpy.ma`, which is the upstream definition
//! torch.masked mirrors.
//!
//! ## GPU lane
//!
//! `masked_sum` / `masked_mean` / `masked_min` / `masked_max` lower to GPU
//! kernels for f32 + f64 when the underlying data tensor is on CUDA.
//! `masked_count` is intentionally a host-side count of the boolean mask
//! (the mask itself is a `Vec<bool>` on host) and works regardless of the
//! data tensor's device. The CPU-only constructors `masked_invalid` and
//! `masked_equal` reject GPU input with [`FerrotorchError::NotImplementedOnCuda`];
//! they are not exercised in the GPU lane.
//!
//! Tolerances follow the dispatch table:
//!   F32_REDUCTION_CPU = 1e-6, F32_REDUCTION_GPU = 1e-5,
//!   F64_REDUCTION_CPU = F64_REDUCTION_GPU = 1e-9. Mask propagation
//!   through ops is bit-exact (we read `mt.mask()` and compare directly).

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::masked::{
    MaskedTensor, masked_count, masked_equal, masked_invalid, masked_max, masked_mean, masked_min,
    masked_sum, masked_where,
};
use ferrotorch_core::{Device, FerrotorchError, Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// Tolerance helpers (mirroring conformance_reduction.rs)
// ---------------------------------------------------------------------------

mod tolerance {
    pub const F32_REDUCTION_CPU: f32 = 1e-6;
    pub const F64_REDUCTION_CPU: f64 = 1e-9;

    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_REDUCTION_GPU: f32 = 1e-5;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F64_REDUCTION_GPU: f64 = 1e-9;

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
// JSON-with-NaN/Infinity-sentinels deserializer (shared shape with the rest
// of the conformance suite).
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
    numpy_version: String,
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
    torch_masked_status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    op: String,
    #[serde(default)]
    tag: Option<String>,
    dtype: String,
    device: String,
    #[serde(default)]
    a_shape: Option<Vec<usize>>,
    #[serde(default)]
    #[allow(
        dead_code,
        reason = "deserialized for fixture-shape stability and future shape-checks"
    )]
    out_shape: Option<Vec<usize>>,
    #[serde(default)]
    a_data: Option<F64ListSentinel>,
    #[serde(default)]
    out_values: Option<F64ListSentinel>,
    /// Mask in **ferrotorch / torch convention**: `true` = valid.
    #[serde(default)]
    mask: Option<Vec<bool>>,
    /// Constructor-side: `condition` for `masked_where`.
    #[serde(default)]
    condition: Option<Vec<bool>>,
    /// Constructor-side: scalar value for `masked_equal`.
    #[serde(default)]
    value: Option<f64>,
    /// Constructor-side: expected `mt.mask()` after construction.
    #[serde(default)]
    expected_mask: Option<Vec<bool>>,
    #[serde(default)]
    expected_count_valid: Option<usize>,
    #[serde(default)]
    expected_count_masked: Option<usize>,
    #[serde(default)]
    expected_numel: Option<usize>,
    /// `filled` fixture: which fill value to set via `with_fill_value`.
    #[serde(default)]
    fill_value: Option<f64>,
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("masked.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_masked_fixtures.py`",
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
// Device-transparent helpers (mirroring conformance_reduction.rs).
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

fn make_cpu_f32(data: &[f64], shape: &[usize]) -> Tensor<f32> {
    let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), false).expect("make_cpu_f32")
}

fn make_cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
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
/// crosslink follow-ups for them. (See conformance_reduction.rs for the
/// pattern.)
fn cascade_skip(_op: &str, _device_label: &str, _dtype: &str) -> Option<&'static str> {
    None
}

// ---------------------------------------------------------------------------
// Reductions: masked_sum / masked_mean / masked_min / masked_max / masked_count
// ---------------------------------------------------------------------------
//
// Each reduction returns a 0-d tensor. The fixture `out_values` is a
// 1-element list holding the scalar (NaN sentinel for all-masked min/max
// and all-masked mean). Mask convention is preserved: `mask[i] == true`
// means "valid".

#[derive(Clone, Copy)]
enum ReductionOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

impl ReductionOp {
    fn name(self) -> &'static str {
        match self {
            ReductionOp::Sum => "masked_sum",
            ReductionOp::Mean => "masked_mean",
            ReductionOp::Min => "masked_min",
            ReductionOp::Max => "masked_max",
            ReductionOp::Count => "masked_count",
        }
    }
    fn apply_f32(self, mt: &MaskedTensor<f32>) -> Tensor<f32> {
        match self {
            ReductionOp::Sum => masked_sum(mt).expect("masked_sum f32"),
            ReductionOp::Mean => masked_mean(mt).expect("masked_mean f32"),
            ReductionOp::Min => masked_min(mt).expect("masked_min f32"),
            ReductionOp::Max => masked_max(mt).expect("masked_max f32"),
            ReductionOp::Count => masked_count(mt).expect("masked_count f32"),
        }
    }
    fn apply_f64(self, mt: &MaskedTensor<f64>) -> Tensor<f64> {
        match self {
            ReductionOp::Sum => masked_sum(mt).expect("masked_sum f64"),
            ReductionOp::Mean => masked_mean(mt).expect("masked_mean f64"),
            ReductionOp::Min => masked_min(mt).expect("masked_min f64"),
            ReductionOp::Max => masked_max(mt).expect("masked_max f64"),
            ReductionOp::Count => masked_count(mt).expect("masked_count f64"),
        }
    }
}

fn run_reduction_for_device(op: ReductionOp, device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, op.name(), device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for {} on {device_label}",
        op.name()
    );
    let on_gpu = matches!(device, Device::Cuda(_));
    let tol_f32 = if on_gpu {
        tolerance::F32_REDUCTION_GPU
    } else {
        tolerance::F32_REDUCTION_CPU
    };
    let tol_f64 = if on_gpu {
        tolerance::F64_REDUCTION_GPU
    } else {
        tolerance::F64_REDUCTION_CPU
    };

    for f in cases {
        if let Some(reason) = cascade_skip(op.name(), device_label, &f.dtype) {
            eprintln!(
                "skipping {} {device_label} dtype={} tag={:?}: {reason}",
                op.name(),
                f.dtype,
                f.tag,
            );
            continue;
        }
        let label = format!(
            "{} {device_label} tag={:?} dtype={}",
            op.name(),
            f.tag,
            f.dtype,
        );
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let mask = f.mask.as_ref().expect("mask").clone();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");

        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, shape), device);
                let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new f32");
                let out = op.apply_f32(&mt);
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&out),
                    expected,
                    tol_f32,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape), device);
                let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new f64");
                let out = op.apply_f64(&mt);
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

#[test]
fn cpu_masked_sum() {
    run_reduction_for_device(ReductionOp::Sum, "cpu", Device::Cpu);
}

#[test]
fn cpu_masked_mean() {
    run_reduction_for_device(ReductionOp::Mean, "cpu", Device::Cpu);
}

#[test]
fn cpu_masked_min() {
    run_reduction_for_device(ReductionOp::Min, "cpu", Device::Cpu);
}

#[test]
fn cpu_masked_max() {
    run_reduction_for_device(ReductionOp::Max, "cpu", Device::Cpu);
}

#[test]
fn cpu_masked_count() {
    run_reduction_for_device(ReductionOp::Count, "cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// Empty-masked-tensor edge cases
// ---------------------------------------------------------------------------
//
// PyTorch / numpy.ma contract on an empty 1-D masked tensor:
//   sum   = 0
//   mean  = NaN
//   min   = NaN  (ferrotorch chooses NaN; torch raises — see fixture
//                 generator's `ref_masked_extreme` for the divergence note)
//   max   = NaN
//   count = 0
//
// The fixtures encode the ferrotorch-side answer in `out_values` and the
// `assert_close_*` helpers treat NaN==NaN as equal, so this tests the
// contract directly.

#[test]
fn cpu_masked_reductions_empty() {
    let file = load_fixtures();
    for op in [
        "masked_sum_empty",
        "masked_mean_empty",
        "masked_min_empty",
        "masked_max_empty",
        "masked_count_empty",
    ] {
        for f in cases_for(&file, op, "cpu") {
            let label = format!("{op} cpu dtype={}", f.dtype);
            let shape = f.a_shape.as_ref().expect("a_shape");
            let a_data = f
                .a_data
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .expect("a_data");
            let mask: Vec<bool> = f.mask.as_ref().cloned().expect("mask");
            let expected = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .expect("out_values");
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, shape);
                    let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new (empty)");
                    // numel is 0 on empty input.
                    assert_eq!(mt.numel(), 0, "{label}: empty numel");
                    let out = match op {
                        "masked_sum_empty" => masked_sum(&mt).expect("masked_sum"),
                        "masked_mean_empty" => masked_mean(&mt).expect("masked_mean"),
                        "masked_min_empty" => masked_min(&mt).expect("masked_min"),
                        "masked_max_empty" => masked_max(&mt).expect("masked_max"),
                        "masked_count_empty" => masked_count(&mt).expect("masked_count"),
                        _ => unreachable!(),
                    };
                    check_f32(
                        &label,
                        &read_back_f32(&out),
                        expected,
                        tolerance::F32_REDUCTION_CPU,
                    );
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape);
                    let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new (empty)");
                    assert_eq!(mt.numel(), 0, "{label}: empty numel");
                    let out = match op {
                        "masked_sum_empty" => masked_sum(&mt).expect("masked_sum"),
                        "masked_mean_empty" => masked_mean(&mt).expect("masked_mean"),
                        "masked_min_empty" => masked_min(&mt).expect("masked_min"),
                        "masked_max_empty" => masked_max(&mt).expect("masked_max"),
                        "masked_count_empty" => masked_count(&mt).expect("masked_count"),
                        _ => unreachable!(),
                    };
                    check_f64(
                        &label,
                        &read_back_f64(&out),
                        expected,
                        tolerance::F64_REDUCTION_CPU,
                    );
                }
                other => panic!("{label}: unexpected dtype {other:?}"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// All-true mask <=> unmasked op equivalence
// ---------------------------------------------------------------------------
//
// PyTorch parity: a MaskedTensor with all-true mask must produce the same
// numerical result as the underlying unmasked op. We assert this directly
// against `torch.sum` / `torch.mean` / etc. via the fixture, which encodes
// the expected scalars under the `all_true` tag for each op.
// `assert_close_f64` already handles NaN/Inf parity; here both sides are
// finite by construction.

#[test]
fn cpu_all_true_mask_matches_unmasked() {
    // For each op + dtype + shape, locate the `*_all_true` fixture and
    // assert that the masked result equals the unmasked result computed
    // by torch (which the fixture already pre-encoded).
    let file = load_fixtures();
    for op in ["masked_sum", "masked_mean", "masked_min", "masked_max"] {
        for f in cases_for(&file, op, "cpu") {
            let tag = f.tag.as_deref().unwrap_or("");
            if !tag.ends_with("_all_true") {
                continue;
            }
            let shape = f.a_shape.as_ref().expect("a_shape");
            let a_data = f
                .a_data
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .expect("a_data");
            let mask: Vec<bool> = f.mask.as_ref().cloned().expect("mask");
            assert!(mask.iter().all(|&m| m), "{op} {tag}: mask is not all-true");
            let expected = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .expect("out_values");
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, shape);
                    let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new");
                    let out = match op {
                        "masked_sum" => masked_sum(&mt).unwrap(),
                        "masked_mean" => masked_mean(&mt).unwrap(),
                        "masked_min" => masked_min(&mt).unwrap(),
                        "masked_max" => masked_max(&mt).unwrap(),
                        _ => unreachable!(),
                    };
                    check_f32(
                        &format!("{op} all_true tag={tag} dtype=float32"),
                        &read_back_f32(&out),
                        expected,
                        tolerance::F32_REDUCTION_CPU,
                    );
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape);
                    let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new");
                    let out = match op {
                        "masked_sum" => masked_sum(&mt).unwrap(),
                        "masked_mean" => masked_mean(&mt).unwrap(),
                        "masked_min" => masked_min(&mt).unwrap(),
                        "masked_max" => masked_max(&mt).unwrap(),
                        _ => unreachable!(),
                    };
                    check_f64(
                        &format!("{op} all_true tag={tag} dtype=float64"),
                        &read_back_f64(&out),
                        expected,
                        tolerance::F64_REDUCTION_CPU,
                    );
                }
                other => panic!("unexpected dtype {other:?}"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors: from_data, masked_where, masked_invalid, masked_equal
// ---------------------------------------------------------------------------
//
// Each constructor fixture holds an `expected_mask` (in ferrotorch
// convention: true=valid). We compare bit-exact via direct slice equality
// — mask propagation must be exact, not tolerant.

#[test]
fn cpu_from_data_marks_all_valid() {
    let file = load_fixtures();
    for f in cases_for(&file, "from_data", "cpu") {
        let label = format!("from_data dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let exp_mask = f.expected_mask.as_ref().expect("expected_mask");
        let exp_valid = f.expected_count_valid.expect("expected_count_valid");
        let exp_masked = f.expected_count_masked.expect("expected_count_masked");
        let exp_numel = f.expected_numel.expect("expected_numel");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape);
                let mt = MaskedTensor::from_data(a).expect("from_data");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
                assert_eq!(mt.count_valid(), exp_valid, "{label}: count_valid");
                assert_eq!(mt.count_masked(), exp_masked, "{label}: count_masked");
                assert_eq!(mt.numel(), exp_numel, "{label}: numel");
                assert_eq!(mt.shape(), shape.as_slice(), "{label}: shape");
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape);
                let mt = MaskedTensor::from_data(a).expect("from_data");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
                assert_eq!(mt.count_valid(), exp_valid, "{label}: count_valid");
                assert_eq!(mt.count_masked(), exp_masked, "{label}: count_masked");
                assert_eq!(mt.numel(), exp_numel, "{label}: numel");
                assert_eq!(mt.shape(), shape.as_slice(), "{label}: shape");
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

#[test]
fn cpu_masked_where_inverts_condition() {
    let file = load_fixtures();
    for f in cases_for(&file, "masked_where", "cpu") {
        let label = format!("masked_where dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let condition = f.condition.as_ref().expect("condition");
        let exp_mask = f.expected_mask.as_ref().expect("expected_mask");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape);
                let mt = masked_where(a, condition).expect("masked_where");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape);
                let mt = masked_where(a, condition).expect("masked_where");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

#[test]
fn cpu_masked_invalid_masks_nan_inf() {
    let file = load_fixtures();
    for f in cases_for(&file, "masked_invalid", "cpu") {
        let label = format!("masked_invalid dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let exp_mask = f.expected_mask.as_ref().expect("expected_mask");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape);
                let mt = masked_invalid(a).expect("masked_invalid");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape);
                let mt = masked_invalid(a).expect("masked_invalid");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

#[test]
fn cpu_masked_equal_masks_value() {
    let file = load_fixtures();
    for f in cases_for(&file, "masked_equal", "cpu") {
        let label = format!("masked_equal dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let value = f.value.expect("value");
        let exp_mask = f.expected_mask.as_ref().expect("expected_mask");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape);
                let mt = masked_equal(a, value as f32).expect("masked_equal");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape);
                let mt = masked_equal(a, value).expect("masked_equal");
                assert_eq!(mt.mask(), exp_mask.as_slice(), "{label}: mask");
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// `filled` / `to_tensor` / `with_fill_value` / accessors
// ---------------------------------------------------------------------------
//
// `filled()` substitutes `fill_value` at every masked-out position.
// `to_tensor()` is an alias for `filled()` (matches torch.Tensor naming).
// `with_fill_value(v)` produces a MaskedTensor with the chosen fill value;
// `fill_value()` reads it back. The default fill is 0 (torch default).
// `data()` borrows the underlying tensor regardless of mask.

#[test]
fn cpu_filled_default_substitutes_zero() {
    let file = load_fixtures();
    for f in cases_for(&file, "filled_default", "cpu") {
        let label = format!("filled_default dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let mask = f.mask.as_ref().cloned().expect("mask");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape);
                let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new");
                // Default fill_value is 0.
                assert_eq!(mt.fill_value(), 0.0_f32, "{label}: default fill_value");
                let filled = mt.filled().expect("filled");
                check_f32(
                    &format!("{label} filled"),
                    &read_back_f32(&filled),
                    expected,
                    tolerance::F32_REDUCTION_CPU,
                );
                // to_tensor is an alias for filled.
                let alias = mt.to_tensor().expect("to_tensor");
                check_f32(
                    &format!("{label} to_tensor"),
                    &read_back_f32(&alias),
                    expected,
                    tolerance::F32_REDUCTION_CPU,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape);
                let mt = MaskedTensor::new(a, mask).expect("MaskedTensor::new");
                assert_eq!(mt.fill_value(), 0.0_f64, "{label}: default fill_value");
                let filled = mt.filled().expect("filled");
                check_f64(
                    &format!("{label} filled"),
                    &read_back_f64(&filled),
                    expected,
                    tolerance::F64_REDUCTION_CPU,
                );
                let alias = mt.to_tensor().expect("to_tensor");
                check_f64(
                    &format!("{label} to_tensor"),
                    &read_back_f64(&alias),
                    expected,
                    tolerance::F64_REDUCTION_CPU,
                );
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

#[test]
fn cpu_with_fill_value_substitutes_override() {
    let file = load_fixtures();
    for f in cases_for(&file, "filled_override", "cpu") {
        let label = format!("filled_override dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let mask = f.mask.as_ref().cloned().expect("mask");
        let fill = f.fill_value.expect("fill_value");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape);
                let mt = MaskedTensor::new(a, mask)
                    .expect("MaskedTensor::new")
                    .with_fill_value(fill as f32);
                assert_eq!(
                    mt.fill_value(),
                    fill as f32,
                    "{label}: fill_value getter mismatch"
                );
                let filled = mt.filled().expect("filled");
                check_f32(
                    &format!("{label} filled"),
                    &read_back_f32(&filled),
                    expected,
                    tolerance::F32_REDUCTION_CPU,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape);
                let mt = MaskedTensor::new(a, mask)
                    .expect("MaskedTensor::new")
                    .with_fill_value(fill);
                assert_eq!(
                    mt.fill_value(),
                    fill,
                    "{label}: fill_value getter mismatch"
                );
                let filled = mt.filled().expect("filled");
                check_f64(
                    &format!("{label} filled"),
                    &read_back_f64(&filled),
                    expected,
                    tolerance::F64_REDUCTION_CPU,
                );
            }
            other => panic!("{label}: unexpected dtype {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Construction error paths (parity with PyTorch validation)
// ---------------------------------------------------------------------------
//
// `MaskedTensor::new` must reject a mask whose length differs from
// `data.numel()`. The error variant is `FerrotorchError::ShapeMismatch`,
// which mirrors the PyTorch behaviour of raising `RuntimeError` on a
// shape-incompatible mask.

#[test]
fn masked_tensor_new_rejects_mask_length_mismatch() {
    let a = make_cpu_f64(&[1.0, 2.0, 3.0], &[3]);
    let err = MaskedTensor::new(a, vec![true, false]).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn masked_where_rejects_condition_length_mismatch() {
    let a = make_cpu_f64(&[1.0, 2.0, 3.0], &[3]);
    let err = masked_where(a, &[true, false]).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// `data()` accessor + `to_ferray` bridge
// ---------------------------------------------------------------------------
//
// `data()` returns the underlying tensor regardless of mask state. The
// returned tensor must round-trip the original values bit-exactly (mask
// propagation should never mutate the data buffer).
//
// `to_ferray::<U>(op)` converts to a `ferray_ma::MaskedArray<U, IxDyn>`
// with the mask inverted to NumPy convention (true=invalid). The
// round-trip property tested here: the converted MaskedArray's mean
// matches our in-house `masked_mean`.

#[test]
fn data_accessor_returns_underlying_tensor() {
    let raw = vec![1.0_f64, 2.0, 3.0, 4.0];
    let a = make_cpu_f64(&raw, &[4]);
    let mt = MaskedTensor::new(a, vec![true, false, true, false]).expect("MaskedTensor::new");
    let inner = mt.data();
    // data() borrows the underlying tensor — values must be unchanged.
    assert_eq!(inner.data().expect("data"), raw.as_slice());
    assert_eq!(inner.shape(), &[4]);
}

#[test]
fn to_ferray_bridges_to_numpy_convention() {
    // Cross-check our in-house `masked_mean` against ferray-ma's
    // MaskedArray::mean() — the bridge inverts the mask to numpy
    // convention, so a successful round-trip confirms both directions.
    use ferray_core::IxDyn as FerrayIxDyn;
    use ferray_ma::masked_array::MaskedArray;

    let a = make_cpu_f64(&[2.0, 4.0, 6.0, 8.0], &[4]);
    let mt = MaskedTensor::new(a, vec![true, false, true, false]).expect("MaskedTensor::new");
    let inhouse = masked_mean(&mt).expect("masked_mean").data().unwrap()[0];

    let ferray_view: MaskedArray<f64, FerrayIxDyn> =
        mt.to_ferray("conformance_masked").expect("to_ferray");
    let ferray_mean = ferray_view.mean().expect("ferray mean");
    tolerance::assert_close_f64(
        &[inhouse],
        &[ferray_mean],
        tolerance::F64_REDUCTION_CPU,
        "to_ferray round-trip mean",
    );
    // Sanity: the closed form is (2 + 6) / 2 = 4.
    tolerance::assert_close_f64(
        &[inhouse],
        &[4.0],
        tolerance::F64_REDUCTION_CPU,
        "to_ferray closed-form",
    );
}

// ---------------------------------------------------------------------------
// Sanity: every required op is present in the fixture file.
// ---------------------------------------------------------------------------

#[test]
fn fixture_file_covers_every_phase212_op() {
    let file = load_fixtures();
    let mut by_op: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for f in &file.fixtures {
        *by_op.entry(f.op.as_str()).or_insert(0) += 1;
    }
    let required = [
        // Reductions
        "masked_sum",
        "masked_mean",
        "masked_min",
        "masked_max",
        "masked_count",
        // Empty-tensor edge cases
        "masked_sum_empty",
        "masked_mean_empty",
        "masked_min_empty",
        "masked_max_empty",
        "masked_count_empty",
        // Constructors
        "from_data",
        "masked_where",
        "masked_invalid",
        "masked_equal",
        // Filled / fill_value
        "filled_default",
        "filled_override",
    ];
    for r in required {
        let n = by_op.get(r).copied().unwrap_or(0);
        assert!(n > 0, "fixture file missing op {r:?}");
    }
}

// ---------------------------------------------------------------------------
// GPU paths — gated on the `gpu` feature
// ---------------------------------------------------------------------------
//
// Per the dispatch:
//   * masked_sum / masked_mean / masked_min / masked_max have GPU lowerings
//     for f32 + f64 (#597 / #627). The fixtures for cuda:0 run on GPU and
//     read back to host for comparison.
//   * masked_count is intentionally host-side (the boolean mask is a
//     `Vec<bool>` regardless of where the data tensor lives) — including
//     it in the GPU lane proves it stays correct when the data tensor is
//     on CUDA.
//   * masked_invalid / masked_equal explicitly return
//     `NotImplementedOnCuda` for GPU input — they have no GPU lane and
//     are NOT exercised here. (See conformance_masked_constructors_*)

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
                "fixtures/masked.json was generated without CUDA — \
                 regenerate on a CUDA-enabled host before running --features gpu tests"
            );
        }
    }

    #[test]
    fn gpu_masked_sum() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_reduction_for_device(ReductionOp::Sum, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_masked_mean() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_reduction_for_device(ReductionOp::Mean, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_masked_min() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_reduction_for_device(ReductionOp::Min, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_masked_max() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_reduction_for_device(ReductionOp::Max, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_masked_count() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_reduction_for_device(ReductionOp::Count, "cuda:0", Device::Cuda(0));
    }
}
