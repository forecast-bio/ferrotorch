//! Conformance Phase 2.8 — `ferrotorch-core` nested + sparse parity against
//! PyTorch.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/770>.
//! Parent: #759.
//!
//! Source files exercised:
//! - `ferrotorch-core/src/nested.rs` — [`NestedTensor`], [`PackedNestedTensor`],
//!   [`nested_scaled_dot_product_attention`].
//! - `ferrotorch-core/src/sparse.rs` — [`SparseTensor`], [`CooTensor`],
//!   [`CsrTensor`], [`CscTensor`], [`SemiStructuredSparseTensor`],
//!   [`SparseGrad`], [`sparse_matmul_24`].
//!
//! ## Scope per dispatch
//!
//! * **Cat A — NestedTensor**: from_padded, to_padded, sizes_per_layer
//!   (= `ragged_lengths` / `consistent_shape` / `ndim` / `num_components` /
//!   `tensors` accessors), construction (`new`), SDPA forward.
//! * **Cat A — PackedNestedTensor**: `from_sequences`, `from_nested`,
//!   `to_nested`, `from_padded`, `to_padded`, `add`/`sub`/`mul`/`div`/`map`,
//!   `sum_per_component`/`mean_per_component`, accessors (`offsets`,
//!   `tail_shape`, `data`, `length`, `total_numel`, `component_slice`,
//!   `num_components`).
//! * **Cat A — SparseTensor (rank-N COO)**: `new`, `from_dense`, `to_dense`,
//!   `coalesce`, `add`, `mul_scalar`, `t`, `spmm`, accessors (`indices`,
//!   `values`, `nnz`, `shape`, `ndim`).
//! * **Cat A — CooTensor / CsrTensor / CscTensor**: `new`, all
//!   inter-format conversions (`from_csr`, `from_coo`, `to_csr`),
//!   `to_dense`, `coalesce` (Coo only), accessors.
//! * **Cat A — SemiStructuredSparseTensor**: `compress`, `decompress`,
//!   accessors (`shape`, `values`, `mask`, `num_groups`,
//!   `compression_ratio`, `group_mask`).
//! * **Cat A — sparse_matmul_24**: 2:4 reference matmul.
//! * **Cat A — SparseGrad**: `new`, `coalesce`, `apply_sgd`, accessors
//!   (`indices`, `values`, `nnz`, `slab_shape`, `slab_size`).
//!
//! ## Edge cases (per dispatch)
//!
//! * Empty nested tensor (no-component construction is rejected — the closest
//!   stable form per #770 since PyTorch's `torch.nested.nested_tensor` rejects
//!   the empty list as well).
//! * Ragged dim 0 vs ragged dim 1.
//! * Sparse to_dense round-trip equality.
//! * Sparse matmul on small (5x5) matrices.
//! * Sparsity sweep: 1% / 50% / 99% non-zeros.
//! * COO with duplicate indices — coalesce parity.
//!
//! ## GPU policy
//!
//! `nested.rs` and `sparse.rs` in ferrotorch-core are CPU-only modules
//! (every storage construction uses `TensorStorage::cpu`). PyTorch supports
//! GPU sparse via cuSPARSE; ferrotorch does not yet. Per
//! `rust-gpu-discipline` §3 the honest response is to surface this as a
//! tracked cascade issue and `cascade_skip()` the GPU lane against that
//! issue rather than silently passing CPU work as GPU coverage. The skip
//! is loud (`eprintln!`) and references the cascade issue so it shows up
//! in audit trails.
//!
//! ## Tolerances
//!
//! Matches the dispatch table for non-arithmetic structural assertions
//! (bit-exact for shape / index / offset / nnz) and per-op tolerances for
//! arithmetic results. Reductions and matmuls use the "matmul-like" /
//! "reduction-like" tolerances from earlier phases.

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::nested::{
    NestedTensor, PackedNestedTensor, nested_scaled_dot_product_attention,
};
use ferrotorch_core::sparse::{
    CooTensor, CscTensor, CsrTensor, SemiStructuredSparseTensor, SparseGrad, SparseTensor,
    sparse_matmul_24,
};
use ferrotorch_core::{Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// Tolerance helpers — mirror the structure used in earlier phases.
// ---------------------------------------------------------------------------

mod tolerance {
    pub const F32_STRUCTURAL: f32 = 0.0; // bit-exact for shape / index / offset
    pub const F32_ELEMENTWISE: f32 = 1e-6;
    pub const F32_MATMUL: f32 = 1e-5;
    pub const F32_SDPA: f32 = 1e-5;

    pub const F64_ELEMENTWISE: f64 = 1e-12;
    pub const F64_MATMUL: f64 = 1e-9;
    pub const F64_SDPA: f64 = 1e-9;

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
// JSON sentinel deserializer — matches the elementwise/reduction phases.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
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
// Fixture file shape. `serde(deny_unknown_fields)` is intentionally NOT used
// here because the fixture is heterogeneous across many ops (each op has its
// own field set). Per-op deserialization picks the fields it needs.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[allow(dead_code, reason = "metadata used for diagnostics")]
    metadata: FixtureMetadata,
    fixtures: Vec<serde_json::Value>,
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
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("nested_sparse.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_nested_sparse_fixtures.py`",
            p.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn pick<'a>(
    file: &'a FixtureFile,
    op: &str,
    dtype: &str,
    tag: Option<&str>,
) -> &'a serde_json::Value {
    file.fixtures
        .iter()
        .find(|f| {
            f.get("op").and_then(|v| v.as_str()) == Some(op)
                && f.get("dtype").and_then(|v| v.as_str()) == Some(dtype)
                && tag.is_none_or(|t| f.get("tag").and_then(|v| v.as_str()) == Some(t))
        })
        .unwrap_or_else(|| panic!("fixture not found: op={op} dtype={dtype} tag={tag:?}"))
}

fn pick_all<'a>(file: &'a FixtureFile, op: &str, dtype: &str) -> Vec<&'a serde_json::Value> {
    file.fixtures
        .iter()
        .filter(|f| {
            f.get("op").and_then(|v| v.as_str()) == Some(op)
                && f.get("dtype").and_then(|v| v.as_str()) == Some(dtype)
        })
        .collect()
}

fn as_f64_list(v: &serde_json::Value) -> Vec<f64> {
    let s: F64ListSentinel = serde_json::from_value(v.clone()).expect("f64 list with sentinels");
    s.as_slice().to_vec()
}

fn as_usize_list(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("u64") as usize)
        .collect()
}

fn as_usize_2d(v: &serde_json::Value) -> Vec<Vec<usize>> {
    v.as_array()
        .expect("array of arrays")
        .iter()
        .map(as_usize_list)
        .collect()
}

fn make_tensor_f32(data: &[f64], shape: &[usize]) -> Tensor<f32> {
    let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), false).expect("make_tensor_f32")
}

fn make_tensor_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("make_tensor_f64")
}

/// Cascade-skip table. Returns `Some(reason)` to skip the GPU lane with a
/// printed cascade issue reference. Per `rust-gpu-discipline` §3, the
/// nested + sparse modules are CPU-only so every GPU test routes through
/// this skip.
///
/// The cascade issue covering all of nested+sparse GPU support is filed
/// once for the phase rather than per-op (every op has the same finding:
/// no GPU implementation exists yet) — see #806.
#[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
fn cascade_skip_gpu() -> Option<&'static str> {
    Some(
        "ferrotorch-core nested+sparse modules are CPU-only — GPU coverage \
         cascade-skipped pending #806 (parent #770)",
    )
}

// ===========================================================================
// CPU tests — NestedTensor / PackedNestedTensor / Sparse — these are the
// real conformance tests; they reference every public name in `Type::method`
// form so the substring-grep coverage gate finds them after the exclusion
// table is cleaned up by the orchestrator.
// ===========================================================================

// ---------------------------------------------------------------------------
// NestedTensor — construction, accessors, to_padded / from_padded, SDPA.
// ---------------------------------------------------------------------------

fn check_nested_to_padded_f32(file: &FixtureFile, tag: &str) {
    let f = pick(file, "nested_to_padded", "float32", Some(tag));
    let comps_data: Vec<Vec<f64>> = f["components"]
        .as_array()
        .expect("components array")
        .iter()
        .map(as_f64_list)
        .collect();
    let comp_shapes: Vec<Vec<usize>> = as_usize_2d(&f["component_shapes"]);
    let ragged_dim = f["ragged_dim"].as_u64().expect("ragged_dim") as usize;
    let pad_value = f["pad_value"].as_f64().expect("pad_value") as f32;
    let expected_padded = as_f64_list(&f["padded_data"]);
    let expected_shape = as_usize_list(&f["padded_shape"]);

    // Build a NestedTensor from the components and exercise every accessor.
    let tensors: Vec<Tensor<f32>> = comps_data
        .iter()
        .zip(comp_shapes.iter())
        .map(|(d, s)| make_tensor_f32(d, s))
        .collect();
    let nt = NestedTensor::new(tensors, ragged_dim).expect("NestedTensor::new");

    assert_eq!(nt.num_components(), comps_data.len(), "num_components");
    assert_eq!(nt.ragged_dim(), ragged_dim, "ragged_dim");
    assert_eq!(nt.ndim(), comp_shapes[0].len(), "ndim");
    let cs = nt.consistent_shape();
    assert_eq!(cs, comp_shapes[0], "consistent_shape");
    let lens = nt.ragged_lengths();
    let expected_lens: Vec<usize> = comp_shapes.iter().map(|s| s[ragged_dim]).collect();
    assert_eq!(lens, expected_lens, "ragged_lengths");
    assert_eq!(nt.tensors().len(), comps_data.len(), "tensors");

    // to_padded.
    let padded = nt.to_padded(pad_value).expect("to_padded");
    assert_eq!(padded.shape(), expected_shape.as_slice(), "padded shape");
    let padded_data = padded.data().expect("padded data").to_vec();
    let expected_padded_f32: Vec<f32> = expected_padded.iter().map(|&x| x as f32).collect();
    tolerance::assert_close_f32(
        &padded_data,
        &expected_padded_f32,
        tolerance::F32_STRUCTURAL,
        &format!("nested_to_padded {tag} f32"),
    );

    // from_padded round-trip.
    let reconstructed =
        NestedTensor::from_padded(&padded, &lens, ragged_dim).expect("NestedTensor::from_padded");
    assert_eq!(
        reconstructed.num_components(),
        nt.num_components(),
        "round-trip num_components"
    );
    for (i, t) in reconstructed.tensors().iter().enumerate() {
        let actual = t.data().expect("comp data").to_vec();
        let expected: Vec<f32> = comps_data[i].iter().map(|&x| x as f32).collect();
        tolerance::assert_close_f32(
            &actual,
            &expected,
            tolerance::F32_STRUCTURAL,
            &format!("nested_from_padded {tag} comp{i} f32"),
        );
    }
}

fn check_nested_to_padded_f64(file: &FixtureFile, tag: &str) {
    let f = pick(file, "nested_to_padded", "float64", Some(tag));
    let comps_data: Vec<Vec<f64>> = f["components"]
        .as_array()
        .expect("components array")
        .iter()
        .map(as_f64_list)
        .collect();
    let comp_shapes: Vec<Vec<usize>> = as_usize_2d(&f["component_shapes"]);
    let ragged_dim = f["ragged_dim"].as_u64().expect("ragged_dim") as usize;
    let pad_value = f["pad_value"].as_f64().expect("pad_value");
    let expected_padded = as_f64_list(&f["padded_data"]);
    let expected_shape = as_usize_list(&f["padded_shape"]);

    let tensors: Vec<Tensor<f64>> = comps_data
        .iter()
        .zip(comp_shapes.iter())
        .map(|(d, s)| make_tensor_f64(d, s))
        .collect();
    let nt = NestedTensor::new(tensors, ragged_dim).expect("NestedTensor::new f64");

    let padded = nt.to_padded(pad_value).expect("to_padded f64");
    assert_eq!(padded.shape(), expected_shape.as_slice(), "padded shape");
    let padded_data = padded.data().expect("padded data").to_vec();
    tolerance::assert_close_f64(
        &padded_data,
        &expected_padded,
        tolerance::F64_ELEMENTWISE,
        &format!("nested_to_padded {tag} f64"),
    );

    let lens = nt.ragged_lengths();
    let reconstructed = NestedTensor::from_padded(&padded, &lens, ragged_dim)
        .expect("NestedTensor::from_padded f64");
    for (i, t) in reconstructed.tensors().iter().enumerate() {
        let actual = t.data().expect("comp data f64").to_vec();
        tolerance::assert_close_f64(
            &actual,
            &comps_data[i],
            tolerance::F64_ELEMENTWISE,
            &format!("nested_from_padded {tag} comp{i} f64"),
        );
    }
}

#[test]
fn cpu_nested_to_padded_ragged_dim0() {
    let file = load_fixtures();
    check_nested_to_padded_f32(&file, "ragged_dim0_2d");
    check_nested_to_padded_f64(&file, "ragged_dim0_2d");
}

#[test]
fn cpu_nested_to_padded_ragged_dim1() {
    let file = load_fixtures();
    check_nested_to_padded_f32(&file, "ragged_dim1_2d");
    check_nested_to_padded_f64(&file, "ragged_dim1_2d");
}

#[test]
fn cpu_nested_single_component_1d() {
    // Exercises the degenerate batch=1 edge path through to_padded /
    // from_padded — no actual padding occurs because max_len equals
    // the only component's length.
    let file = load_fixtures();
    check_nested_to_padded_f32(&file, "single_component_1d");
}

#[test]
fn cpu_nested_construction_errors() {
    // NestedTensor::new with empty list -> Err.
    let res = NestedTensor::<f32>::new(vec![], 0);
    assert!(
        res.is_err(),
        "NestedTensor::new should reject an empty component list"
    );

    // NestedTensor::new with mismatched non-ragged dims -> Err.
    let t1 = make_tensor_f32(&[1.0; 6], &[3, 2]);
    let t2 = make_tensor_f32(&[1.0; 6], &[2, 3]);
    let res = NestedTensor::new(vec![t1, t2], 0);
    assert!(
        res.is_err(),
        "NestedTensor::new should reject mismatched non-ragged dims"
    );

    // ragged_dim out of range -> Err.
    let t1 = make_tensor_f32(&[1.0; 6], &[3, 2]);
    let t2 = make_tensor_f32(&[1.0; 4], &[2, 2]);
    let res = NestedTensor::new(vec![t1, t2], 5);
    assert!(res.is_err(), "ragged_dim out of range should error");
}

// ---------------------------------------------------------------------------
// nested_scaled_dot_product_attention — and its top-level re-export.
// ---------------------------------------------------------------------------

#[test]
fn cpu_nested_scaled_dot_product_attention_f32() {
    let file = load_fixtures();
    let f = pick(&file, "nested_sdpa", "float32", Some("two_components"));
    let qs = f["queries"].as_array().expect("queries");
    let ks = f["keys"].as_array().expect("keys");
    let vs = f["values"].as_array().expect("values");
    let q_shapes = as_usize_2d(&f["query_shapes"]);
    let k_shapes = as_usize_2d(&f["key_shapes"]);
    let v_shapes = as_usize_2d(&f["value_shapes"]);
    let outs = f["outputs"].as_array().expect("outputs");
    let out_shapes = as_usize_2d(&f["output_shapes"]);

    let q_t: Vec<Tensor<f32>> = qs
        .iter()
        .zip(q_shapes.iter())
        .map(|(d, s)| make_tensor_f32(&as_f64_list(d), s))
        .collect();
    let k_t: Vec<Tensor<f32>> = ks
        .iter()
        .zip(k_shapes.iter())
        .map(|(d, s)| make_tensor_f32(&as_f64_list(d), s))
        .collect();
    let v_t: Vec<Tensor<f32>> = vs
        .iter()
        .zip(v_shapes.iter())
        .map(|(d, s)| make_tensor_f32(&as_f64_list(d), s))
        .collect();

    let q_nt = NestedTensor::new(q_t, 0).expect("query NestedTensor");
    let k_nt = NestedTensor::new(k_t, 0).expect("key NestedTensor");
    let v_nt = NestedTensor::new(v_t, 0).expect("value NestedTensor");

    let result = nested_scaled_dot_product_attention(&q_nt, &k_nt, &v_nt)
        .expect("nested_scaled_dot_product_attention");
    assert_eq!(result.num_components(), 2);
    for (i, t) in result.tensors().iter().enumerate() {
        assert_eq!(t.shape(), out_shapes[i].as_slice());
        let actual = t.data().expect("output data").to_vec();
        let expected: Vec<f32> = as_f64_list(&outs[i]).iter().map(|&x| x as f32).collect();
        tolerance::assert_close_f32(
            &actual,
            &expected,
            tolerance::F32_SDPA,
            &format!("nested_sdpa comp{i} f32"),
        );
    }
}

#[test]
fn cpu_nested_scaled_dot_product_attention_f64() {
    let file = load_fixtures();
    let f = pick(&file, "nested_sdpa", "float64", Some("two_components"));
    let qs = f["queries"].as_array().expect("queries");
    let ks = f["keys"].as_array().expect("keys");
    let vs = f["values"].as_array().expect("values");
    let q_shapes = as_usize_2d(&f["query_shapes"]);
    let k_shapes = as_usize_2d(&f["key_shapes"]);
    let v_shapes = as_usize_2d(&f["value_shapes"]);
    let outs = f["outputs"].as_array().expect("outputs");

    let q_t: Vec<Tensor<f64>> = qs
        .iter()
        .zip(q_shapes.iter())
        .map(|(d, s)| make_tensor_f64(&as_f64_list(d), s))
        .collect();
    let k_t: Vec<Tensor<f64>> = ks
        .iter()
        .zip(k_shapes.iter())
        .map(|(d, s)| make_tensor_f64(&as_f64_list(d), s))
        .collect();
    let v_t: Vec<Tensor<f64>> = vs
        .iter()
        .zip(v_shapes.iter())
        .map(|(d, s)| make_tensor_f64(&as_f64_list(d), s))
        .collect();

    let q_nt = NestedTensor::new(q_t, 0).expect("query NestedTensor f64");
    let k_nt = NestedTensor::new(k_t, 0).expect("key NestedTensor f64");
    let v_nt = NestedTensor::new(v_t, 0).expect("value NestedTensor f64");

    let result = nested_scaled_dot_product_attention(&q_nt, &k_nt, &v_nt)
        .expect("nested_scaled_dot_product_attention f64");
    for (i, t) in result.tensors().iter().enumerate() {
        let actual = t.data().expect("output data f64").to_vec();
        let expected = as_f64_list(&outs[i]);
        tolerance::assert_close_f64(
            &actual,
            &expected,
            tolerance::F64_SDPA,
            &format!("nested_sdpa comp{i} f64"),
        );
    }
}

// ---------------------------------------------------------------------------
// PackedNestedTensor — every public method.
// ---------------------------------------------------------------------------

#[test]
fn cpu_packed_from_sequences_1d() {
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        let f = pick(&file, "packed_from_sequences", dtype, Some("1d"));
        let seqs_raw = f["sequences"].as_array().expect("sequences");
        let lengths = as_usize_list(&f["lengths"]);
        let tail_shape = as_usize_list(&f["tail_shape"]);
        let expected_offsets = as_usize_list(&f["expected_offsets"]);
        let expected_data = as_f64_list(&f["expected_data"]);
        let expected_num_components = f["expected_num_components"].as_u64().expect("nc") as usize;
        let expected_total_numel = f["expected_total_numel"].as_u64().expect("total") as usize;

        match dtype {
            "float32" => {
                let seqs: Vec<Vec<f32>> = seqs_raw
                    .iter()
                    .map(|s| as_f64_list(s).iter().map(|&x| x as f32).collect())
                    .collect();
                let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail_shape)
                    .expect("PackedNestedTensor::from_sequences");
                assert_eq!(pnt.num_components(), expected_num_components);
                assert_eq!(pnt.offsets(), expected_offsets.as_slice());
                assert_eq!(pnt.tail_shape(), tail_shape.as_slice());
                assert_eq!(pnt.total_numel(), expected_total_numel);
                let data: Vec<f32> = pnt.data().to_vec();
                let exp_f32: Vec<f32> = expected_data.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    &data,
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "packed_from_sequences 1d f32 data",
                );
                // length(i) and component_slice(i) accessors.
                for i in 0..expected_num_components {
                    assert_eq!(pnt.length(i), lengths[i]);
                    let slice: Vec<f32> = pnt.component_slice(i).to_vec();
                    let start = expected_offsets[i];
                    let end = expected_offsets[i + 1];
                    let exp_slice = &exp_f32[start..end];
                    tolerance::assert_close_f32(
                        &slice,
                        exp_slice,
                        tolerance::F32_STRUCTURAL,
                        &format!("component_slice({i}) f32"),
                    );
                }
            }
            "float64" => {
                let seqs: Vec<Vec<f64>> = seqs_raw.iter().map(as_f64_list).collect();
                let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail_shape)
                    .expect("PackedNestedTensor::from_sequences f64");
                assert_eq!(pnt.num_components(), expected_num_components);
                assert_eq!(pnt.offsets(), expected_offsets.as_slice());
                assert_eq!(pnt.total_numel(), expected_total_numel);
                tolerance::assert_close_f64(
                    pnt.data(),
                    &expected_data,
                    tolerance::F64_ELEMENTWISE,
                    "packed_from_sequences 1d f64 data",
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_packed_from_sequences_2d_tail() {
    let file = load_fixtures();
    let f = pick(&file, "packed_from_sequences", "float32", Some("2d_tail4"));
    let seqs_raw = f["sequences"].as_array().expect("sequences");
    let lengths = as_usize_list(&f["lengths"]);
    let tail_shape = as_usize_list(&f["tail_shape"]);
    let expected_offsets = as_usize_list(&f["expected_offsets"]);
    let expected_data = as_f64_list(&f["expected_data"]);

    let seqs: Vec<Vec<f32>> = seqs_raw
        .iter()
        .map(|s| as_f64_list(s).iter().map(|&x| x as f32).collect())
        .collect();
    let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail_shape)
        .expect("PackedNestedTensor::from_sequences 2d");
    assert_eq!(pnt.tail_shape(), tail_shape.as_slice());
    assert_eq!(pnt.offsets(), expected_offsets.as_slice());
    let exp_f32: Vec<f32> = expected_data.iter().map(|&x| x as f32).collect();
    tolerance::assert_close_f32(
        pnt.data(),
        &exp_f32,
        tolerance::F32_STRUCTURAL,
        "packed_from_sequences 2d_tail data",
    );
}

#[test]
fn cpu_packed_elementwise() {
    let file = load_fixtures();
    let f = pick(
        &file,
        "packed_elementwise",
        "float32",
        Some("add_sub_mul_div"),
    );
    let a_seqs_raw = f["a_sequences"].as_array().expect("a_sequences");
    let b_seqs_raw = f["b_sequences"].as_array().expect("b_sequences");
    let lengths = as_usize_list(&f["lengths"]);
    let tail_shape = as_usize_list(&f["tail_shape"]);

    let a_seqs: Vec<Vec<f32>> = a_seqs_raw
        .iter()
        .map(|s| as_f64_list(s).iter().map(|&x| x as f32).collect())
        .collect();
    let b_seqs: Vec<Vec<f32>> = b_seqs_raw
        .iter()
        .map(|s| as_f64_list(s).iter().map(|&x| x as f32).collect())
        .collect();
    let a = PackedNestedTensor::from_sequences(a_seqs, &lengths, &tail_shape)
        .expect("a PackedNestedTensor");
    let b = PackedNestedTensor::from_sequences(b_seqs, &lengths, &tail_shape)
        .expect("b PackedNestedTensor");

    let exp_add: Vec<f32> = as_f64_list(&f["expected_add"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let exp_sub: Vec<f32> = as_f64_list(&f["expected_sub"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let exp_mul: Vec<f32> = as_f64_list(&f["expected_mul"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let exp_div: Vec<f32> = as_f64_list(&f["expected_div"])
        .iter()
        .map(|&x| x as f32)
        .collect();

    let added = a.add(&b).expect("add");
    let subbed = a.sub(&b).expect("sub");
    let mulled = a.mul(&b).expect("mul");
    let divved = a.div(&b).expect("div");

    tolerance::assert_close_f32(
        added.data(),
        &exp_add,
        tolerance::F32_ELEMENTWISE,
        "packed add",
    );
    tolerance::assert_close_f32(
        subbed.data(),
        &exp_sub,
        tolerance::F32_ELEMENTWISE,
        "packed sub",
    );
    tolerance::assert_close_f32(
        mulled.data(),
        &exp_mul,
        tolerance::F32_ELEMENTWISE,
        "packed mul",
    );
    tolerance::assert_close_f32(
        divved.data(),
        &exp_div,
        tolerance::F32_ELEMENTWISE,
        "packed div",
    );

    // Exercise PackedNestedTensor::map with a closure (covers the
    // public `map` method).
    let doubled = a.map(|x| x * 2.0);
    let exp_double: Vec<f32> = a.data().iter().map(|&x| x * 2.0).collect();
    tolerance::assert_close_f32(
        doubled.data(),
        &exp_double,
        tolerance::F32_ELEMENTWISE,
        "packed map x2",
    );
}

#[test]
fn cpu_packed_reductions() {
    let file = load_fixtures();
    let f = pick(
        &file,
        "packed_reductions",
        "float64",
        Some("sum_mean_per_component"),
    );
    let seqs_raw = f["sequences"].as_array().expect("sequences");
    let lengths = as_usize_list(&f["lengths"]);
    let tail_shape = as_usize_list(&f["tail_shape"]);
    let expected_sums = as_f64_list(&f["expected_sums"]);
    let expected_means = as_f64_list(&f["expected_means"]);

    let seqs: Vec<Vec<f64>> = seqs_raw.iter().map(as_f64_list).collect();
    let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail_shape)
        .expect("PackedNestedTensor::from_sequences");
    let sums = pnt.sum_per_component();
    let means = pnt.mean_per_component();
    tolerance::assert_close_f64(
        &sums,
        &expected_sums,
        tolerance::F64_ELEMENTWISE,
        "packed sum_per_component",
    );
    tolerance::assert_close_f64(
        &means,
        &expected_means,
        tolerance::F64_ELEMENTWISE,
        "packed mean_per_component",
    );
}

#[test]
fn cpu_packed_to_padded_and_from_padded_round_trip() {
    let file = load_fixtures();
    let f = pick(&file, "packed_to_padded", "float32", Some("1d_padval_neg1"));
    let seqs_raw = f["sequences"].as_array().expect("sequences");
    let lengths = as_usize_list(&f["lengths"]);
    let tail_shape = as_usize_list(&f["tail_shape"]);
    let pad_value = f["pad_value"].as_f64().expect("pad_value") as f32;
    let expected_shape = as_usize_list(&f["expected_padded_shape"]);
    let expected_padded = as_f64_list(&f["expected_padded"]);

    let seqs: Vec<Vec<f32>> = seqs_raw
        .iter()
        .map(|s| as_f64_list(s).iter().map(|&x| x as f32).collect())
        .collect();
    let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail_shape)
        .expect("PackedNestedTensor::from_sequences");
    let padded = pnt.to_padded(pad_value).expect("to_padded");
    assert_eq!(padded.shape(), expected_shape.as_slice());
    let padded_data = padded.data().expect("padded data").to_vec();
    let exp_f32: Vec<f32> = expected_padded.iter().map(|&x| x as f32).collect();
    tolerance::assert_close_f32(
        &padded_data,
        &exp_f32,
        tolerance::F32_STRUCTURAL,
        "packed to_padded shape+values",
    );

    // Round-trip via PackedNestedTensor::from_padded.
    let recovered = PackedNestedTensor::from_padded(&padded, &lengths)
        .expect("PackedNestedTensor::from_padded");
    assert_eq!(recovered.offsets(), pnt.offsets());
    tolerance::assert_close_f32(
        recovered.data(),
        pnt.data(),
        tolerance::F32_STRUCTURAL,
        "packed from_padded round-trip",
    );
}

#[test]
fn cpu_packed_from_nested_and_to_nested() {
    // Exercise PackedNestedTensor::from_nested and PackedNestedTensor::to_nested
    // with a small NestedTensor (ragged_dim=0 required by from_nested).
    let t1 = make_tensor_f32(&[1.0, 2.0, 3.0], &[3]);
    let t2 = make_tensor_f32(&[4.0, 5.0], &[2]);
    let nt = NestedTensor::new(vec![t1, t2], 0).expect("NestedTensor");
    let pnt = PackedNestedTensor::from_nested(&nt).expect("from_nested");
    assert_eq!(pnt.num_components(), 2);
    assert_eq!(pnt.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let round = pnt.to_nested().expect("to_nested");
    assert_eq!(round.num_components(), 2);
    assert_eq!(round.tensors()[0].data().expect("c0"), &[1.0, 2.0, 3.0]);
    assert_eq!(round.tensors()[1].data().expect("c1"), &[4.0, 5.0]);
}

// ---------------------------------------------------------------------------
// SparseTensor (rank-N COO).
// ---------------------------------------------------------------------------

#[test]
fn cpu_sparse_tensor_to_dense_round_trip() {
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        let f = pick(&file, "sparse_to_dense", dtype, Some("3x3_basic"));
        let indices = as_usize_2d(&f["indices"]);
        let values = as_f64_list(&f["values"]);
        let shape = as_usize_list(&f["shape"]);
        let expected_dense = as_f64_list(&f["expected_dense"]);
        let expected_nnz = f["expected_nnz"].as_u64().expect("nnz") as usize;
        let expected_ndim = f["expected_ndim"].as_u64().expect("ndim") as usize;

        match dtype {
            "float32" => {
                let vals: Vec<f32> = values.iter().map(|&x| x as f32).collect();
                let sp = SparseTensor::new(indices.clone(), vals, shape.clone())
                    .expect("SparseTensor::new");
                assert_eq!(sp.nnz(), expected_nnz);
                assert_eq!(sp.shape(), shape.as_slice());
                assert_eq!(sp.ndim(), expected_ndim);
                assert_eq!(sp.indices().len(), indices.len());
                assert_eq!(sp.values().len(), expected_nnz);

                let dense = sp.to_dense().expect("to_dense f32");
                let dense_data = dense.data().expect("dense data").to_vec();
                let exp_f32: Vec<f32> = expected_dense.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    &dense_data,
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "sparse to_dense f32",
                );

                // round-trip via from_dense — values stored are exactly the
                // ones above the threshold (0.0 means "any non-zero"). Note
                // that PyTorch's order may differ from ferrotorch's flat-index
                // scan, so compare via to_dense parity rather than index order.
                let sp_round = SparseTensor::from_dense(&dense, 0.0).expect("from_dense");
                let round_dense = sp_round.to_dense().expect("round to_dense");
                tolerance::assert_close_f32(
                    round_dense.data().expect("round dense data"),
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "sparse from_dense round-trip f32",
                );
            }
            "float64" => {
                let sp = SparseTensor::new(indices.clone(), values.clone(), shape.clone())
                    .expect("SparseTensor::new f64");
                assert_eq!(sp.nnz(), expected_nnz);
                let dense = sp.to_dense().expect("to_dense f64");
                tolerance::assert_close_f64(
                    dense.data().expect("dense data f64"),
                    &expected_dense,
                    tolerance::F64_ELEMENTWISE,
                    "sparse to_dense f64",
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_sparse_tensor_from_dense_with_threshold() {
    let file = load_fixtures();
    let f = pick(&file, "sparse_from_dense", "float32", Some("threshold_1"));
    let dense_data = as_f64_list(&f["dense_data"]);
    let shape = as_usize_list(&f["shape"]);
    let threshold = f["threshold"].as_f64().expect("threshold") as f32;
    let expected_indices = as_usize_2d(&f["expected_indices"]);
    let expected_values: Vec<f32> = as_f64_list(&f["expected_values"])
        .iter()
        .map(|&x| x as f32)
        .collect();

    let dense = make_tensor_f32(&dense_data, &shape);
    let sp = SparseTensor::from_dense(&dense, threshold).expect("from_dense threshold");
    // The order of stored entries matches the row-major scan in `from_dense`.
    assert_eq!(sp.indices().len(), expected_indices.len());
    for (i, idx) in sp.indices().iter().enumerate() {
        assert_eq!(idx, &expected_indices[i], "from_dense index {i}");
    }
    tolerance::assert_close_f32(
        sp.values(),
        &expected_values,
        tolerance::F32_STRUCTURAL,
        "from_dense values",
    );
}

#[test]
fn cpu_sparse_tensor_coalesce_duplicates() {
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        let f = pick(&file, "sparse_coalesce", dtype, Some("duplicates_summed"));
        let indices = as_usize_2d(&f["indices"]);
        let values = as_f64_list(&f["values"]);
        let shape = as_usize_list(&f["shape"]);
        let expected_indices = as_usize_2d(&f["expected_coalesced_indices"]);
        let expected_values = as_f64_list(&f["expected_coalesced_values"]);
        let expected_nnz = f["expected_coalesced_nnz"].as_u64().expect("nnz") as usize;

        match dtype {
            "float32" => {
                let vals: Vec<f32> = values.iter().map(|&x| x as f32).collect();
                let sp = SparseTensor::new(indices, vals, shape).expect("SparseTensor::new");
                let coal = sp.coalesce();
                assert_eq!(coal.nnz(), expected_nnz);
                assert_eq!(coal.indices().len(), expected_indices.len());
                for (i, idx) in coal.indices().iter().enumerate() {
                    assert_eq!(idx, &expected_indices[i], "coalesce index {i}");
                }
                let exp_f32: Vec<f32> = expected_values.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    coal.values(),
                    &exp_f32,
                    tolerance::F32_ELEMENTWISE,
                    "coalesce values f32",
                );
            }
            "float64" => {
                let sp = SparseTensor::new(indices, values, shape).expect("SparseTensor::new f64");
                let coal = sp.coalesce();
                assert_eq!(coal.nnz(), expected_nnz);
                tolerance::assert_close_f64(
                    coal.values(),
                    &expected_values,
                    tolerance::F64_ELEMENTWISE,
                    "coalesce values f64",
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_sparse_tensor_coalesce_zero_sum_cancellation() {
    let file = load_fixtures();
    let f = pick(&file, "sparse_coalesce", "float32", Some("zero_sum_cancel"));
    let indices = as_usize_2d(&f["indices"]);
    let values: Vec<f32> = as_f64_list(&f["values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let shape = as_usize_list(&f["shape"]);
    let sp = SparseTensor::new(indices, values, shape).expect("SparseTensor::new");
    let coal = sp.coalesce();
    assert_eq!(coal.nnz(), 0, "5 + (-5) should cancel to nnz=0");
}

#[test]
fn cpu_sparse_tensor_mul_scalar() {
    let file = load_fixtures();
    let f = pick(&file, "sparse_mul_scalar", "float32", Some("scalar_3"));
    let indices = as_usize_2d(&f["indices"]);
    let values: Vec<f32> = as_f64_list(&f["values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let shape = as_usize_list(&f["shape"]);
    let scalar = f["scalar"].as_f64().expect("scalar") as f32;
    let expected: Vec<f32> = as_f64_list(&f["expected_values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let sp = SparseTensor::new(indices, values, shape).expect("SparseTensor::new");
    let scaled = sp.mul_scalar(scalar);
    tolerance::assert_close_f32(
        scaled.values(),
        &expected,
        tolerance::F32_ELEMENTWISE,
        "mul_scalar values",
    );
}

#[test]
fn cpu_sparse_tensor_add() {
    let file = load_fixtures();
    let f = pick(&file, "sparse_add", "float32", Some("two_2x2"));
    let a_indices = as_usize_2d(&f["a_indices"]);
    let a_values: Vec<f32> = as_f64_list(&f["a_values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let b_indices = as_usize_2d(&f["b_indices"]);
    let b_values: Vec<f32> = as_f64_list(&f["b_values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let shape = as_usize_list(&f["shape"]);
    let expected_indices = as_usize_2d(&f["expected_coalesced_indices"]);
    let expected_values: Vec<f32> = as_f64_list(&f["expected_coalesced_values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let expected_uncoal = f["expected_uncoalesced_nnz"].as_u64().expect("nnz") as usize;

    let a = SparseTensor::new(a_indices, a_values, shape.clone()).expect("a");
    let b = SparseTensor::new(b_indices, b_values, shape).expect("b");
    let sum = a.add(&b).expect("SparseTensor::add");
    assert_eq!(sum.nnz(), expected_uncoal, "uncoalesced add nnz");

    let coal = sum.coalesce();
    assert_eq!(coal.nnz(), expected_indices.len());
    for (i, idx) in coal.indices().iter().enumerate() {
        assert_eq!(idx, &expected_indices[i], "add coalesced index {i}");
    }
    tolerance::assert_close_f32(
        coal.values(),
        &expected_values,
        tolerance::F32_ELEMENTWISE,
        "add coalesced values",
    );
}

#[test]
fn cpu_sparse_tensor_transpose() {
    let file = load_fixtures();
    let f = pick(&file, "sparse_transpose", "float32", Some("3x4"));
    let indices = as_usize_2d(&f["indices"]);
    let values: Vec<f32> = as_f64_list(&f["values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let shape = as_usize_list(&f["shape"]);
    let expected_indices = as_usize_2d(&f["expected_indices"]);
    let expected_values: Vec<f32> = as_f64_list(&f["expected_values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let expected_shape = as_usize_list(&f["expected_shape"]);

    let sp = SparseTensor::new(indices, values, shape).expect("SparseTensor::new");
    let t = sp.t().expect("transpose");
    assert_eq!(t.shape(), expected_shape.as_slice(), "transposed shape");
    for (i, idx) in t.indices().iter().enumerate() {
        assert_eq!(idx, &expected_indices[i], "transposed index {i}");
    }
    tolerance::assert_close_f32(
        t.values(),
        &expected_values,
        tolerance::F32_STRUCTURAL,
        "transposed values",
    );
}

#[test]
fn cpu_sparse_tensor_spmm_sweep() {
    // Sparse matmul on small matrices (10x10) at 1%, 50%, 99% non-zero
    // density — covers the dispatch's "sparse matmul on small matrices" and
    // "various sparsity levels" edge cases.
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        for fixture in pick_all(&file, "sparse_spmm", dtype) {
            let tag = fixture["tag"].as_str().unwrap_or("?");
            let indices = as_usize_2d(&fixture["indices"]);
            let values = as_f64_list(&fixture["values"]);
            let shape = as_usize_list(&fixture["shape"]);
            let rhs_data = as_f64_list(&fixture["rhs_data"]);
            let rhs_shape = as_usize_list(&fixture["rhs_shape"]);
            let expected_spmm = as_f64_list(&fixture["expected_spmm"]);
            let expected_spmm_shape = as_usize_list(&fixture["expected_spmm_shape"]);

            match dtype {
                "float32" => {
                    let vals: Vec<f32> = values.iter().map(|&x| x as f32).collect();
                    let sp = SparseTensor::new(indices, vals, shape).expect("sparse");
                    let rhs = make_tensor_f32(&rhs_data, &rhs_shape);
                    let out = sp.spmm(&rhs).expect("spmm");
                    assert_eq!(out.shape(), expected_spmm_shape.as_slice());
                    let out_data = out.data().expect("spmm data").to_vec();
                    let exp_f32: Vec<f32> = expected_spmm.iter().map(|&x| x as f32).collect();
                    tolerance::assert_close_f32(
                        &out_data,
                        &exp_f32,
                        tolerance::F32_MATMUL,
                        &format!("spmm {tag} f32"),
                    );
                }
                "float64" => {
                    let sp = SparseTensor::new(indices, values, shape).expect("sparse");
                    let rhs = make_tensor_f64(&rhs_data, &rhs_shape);
                    let out = sp.spmm(&rhs).expect("spmm");
                    tolerance::assert_close_f64(
                        out.data().expect("spmm data f64"),
                        &expected_spmm,
                        tolerance::F64_MATMUL,
                        &format!("spmm {tag} f64"),
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CooTensor / CsrTensor / CscTensor — 5x5 conversion matrix.
// ---------------------------------------------------------------------------

#[test]
fn cpu_coo_csr_csc_format_round_trip_5x5() {
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        let f = pick(&file, "coo_csr_csc_5x5", dtype, Some("format_round_trip"));
        let row_indices = as_usize_list(&f["row_indices"]);
        let col_indices = as_usize_list(&f["col_indices"]);
        let values = as_f64_list(&f["values"]);
        let nrows = f["nrows"].as_u64().expect("nrows") as usize;
        let ncols = f["ncols"].as_u64().expect("ncols") as usize;
        let expected_dense = as_f64_list(&f["expected_dense"]);
        let csr_row_ptrs = as_usize_list(&f["expected_csr_row_ptrs"]);
        let csr_col_indices = as_usize_list(&f["expected_csr_col_indices"]);
        let csr_values = as_f64_list(&f["expected_csr_values"]);
        let csc_col_ptrs = as_usize_list(&f["expected_csc_col_ptrs"]);
        let csc_row_indices = as_usize_list(&f["expected_csc_row_indices"]);
        let csc_values = as_f64_list(&f["expected_csc_values"]);
        let expected_nnz = f["expected_nnz"].as_u64().expect("nnz") as usize;

        match dtype {
            "float32" => {
                let vals: Vec<f32> = values.iter().map(|&x| x as f32).collect();

                // CooTensor::new + accessors.
                let coo = CooTensor::new(
                    row_indices.clone(),
                    col_indices.clone(),
                    vals.clone(),
                    nrows,
                    ncols,
                )
                .expect("CooTensor::new");
                assert_eq!(coo.nnz(), expected_nnz);
                assert_eq!(coo.nrows(), nrows);
                assert_eq!(coo.ncols(), ncols);
                assert_eq!(coo.row_indices(), row_indices.as_slice());
                assert_eq!(coo.col_indices(), col_indices.as_slice());
                assert_eq!(coo.values(), vals.as_slice());
                assert!(
                    !coo.is_coalesced(),
                    "freshly built CooTensor is uncoalesced"
                );
                let coo_dense = coo.to_dense().expect("CooTensor::to_dense");
                let exp_f32: Vec<f32> = expected_dense.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    coo_dense.data().expect("coo dense data"),
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "CooTensor::to_dense f32",
                );

                // CooTensor::coalesce — no duplicates here, but exercises the
                // method and sets `is_coalesced = true`.
                let coal = coo.coalesce();
                assert!(coal.is_coalesced());
                assert_eq!(coal.nnz(), expected_nnz);

                // CsrTensor::from_coo.
                let csr = CsrTensor::from_coo(&coo).expect("CsrTensor::from_coo");
                assert_eq!(csr.nnz(), expected_nnz);
                assert_eq!(csr.row_ptrs(), csr_row_ptrs.as_slice());
                assert_eq!(csr.col_indices(), csr_col_indices.as_slice());
                let csr_vals_f32: Vec<f32> = csr_values.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    csr.values(),
                    &csr_vals_f32,
                    tolerance::F32_STRUCTURAL,
                    "CsrTensor::values",
                );
                let csr_dense = csr.to_dense().expect("CsrTensor::to_dense");
                tolerance::assert_close_f32(
                    csr_dense.data().expect("csr dense data"),
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "CsrTensor::to_dense f32",
                );

                // CsrTensor::new direct construction (round-trip).
                let csr_direct = CsrTensor::new(
                    csr_row_ptrs.clone(),
                    csr_col_indices.clone(),
                    csr_vals_f32.clone(),
                    nrows,
                    ncols,
                )
                .expect("CsrTensor::new");
                assert_eq!(csr_direct.nnz(), expected_nnz);

                // CooTensor::from_csr.
                let coo_back = CooTensor::from_csr(&csr);
                let coo_back_dense = coo_back.to_dense().expect("from_csr to_dense");
                tolerance::assert_close_f32(
                    coo_back_dense.data().expect("coo_back dense"),
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "CooTensor::from_csr round-trip",
                );

                // CscTensor::from_csr.
                let csc = CscTensor::from_csr(&csr);
                assert_eq!(csc.nnz(), expected_nnz);
                assert_eq!(csc.nrows(), nrows);
                assert_eq!(csc.ncols(), ncols);
                assert_eq!(csc.col_ptrs(), csc_col_ptrs.as_slice());
                assert_eq!(csc.row_indices(), csc_row_indices.as_slice());
                let csc_vals_f32: Vec<f32> = csc_values.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    csc.values(),
                    &csc_vals_f32,
                    tolerance::F32_STRUCTURAL,
                    "CscTensor::values",
                );

                // CscTensor::to_dense.
                let csc_dense = csc.to_dense().expect("CscTensor::to_dense");
                tolerance::assert_close_f32(
                    csc_dense.data().expect("csc dense"),
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "CscTensor::to_dense f32",
                );

                // CscTensor::to_csr round-trip.
                let csr_back = csc.to_csr();
                let csr_back_dense = csr_back.to_dense().expect("csr_back dense");
                tolerance::assert_close_f32(
                    csr_back_dense.data().expect("csr_back dense data"),
                    &exp_f32,
                    tolerance::F32_STRUCTURAL,
                    "CscTensor::to_csr round-trip",
                );

                // CscTensor::new direct construction.
                let csc_direct = CscTensor::new(
                    csc_col_ptrs.clone(),
                    csc_row_indices.clone(),
                    csc_vals_f32,
                    nrows,
                    ncols,
                )
                .expect("CscTensor::new");
                assert_eq!(csc_direct.nnz(), expected_nnz);
            }
            "float64" => {
                let coo = CooTensor::new(row_indices, col_indices, values.clone(), nrows, ncols)
                    .expect("CooTensor::new f64");
                let coo_dense = coo.to_dense().expect("coo dense f64");
                tolerance::assert_close_f64(
                    coo_dense.data().expect("coo dense f64"),
                    &expected_dense,
                    tolerance::F64_ELEMENTWISE,
                    "CooTensor::to_dense f64",
                );
                let csr = CsrTensor::from_coo(&coo).expect("from_coo f64");
                tolerance::assert_close_f64(
                    csr.values(),
                    &csr_values,
                    tolerance::F64_ELEMENTWISE,
                    "CsrTensor::values f64",
                );
                let csc = CscTensor::from_csr(&csr);
                tolerance::assert_close_f64(
                    csc.values(),
                    &csc_values,
                    tolerance::F64_ELEMENTWISE,
                    "CscTensor::values f64",
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_coo_coalesce_with_duplicates() {
    let file = load_fixtures();
    let f = pick(&file, "coo_coalesce", "float32", Some("duplicates"));
    let row_indices = as_usize_list(&f["row_indices"]);
    let col_indices = as_usize_list(&f["col_indices"]);
    let values: Vec<f32> = as_f64_list(&f["values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let nrows = f["nrows"].as_u64().expect("nrows") as usize;
    let ncols = f["ncols"].as_u64().expect("ncols") as usize;
    let expected_rows = as_usize_list(&f["expected_coalesced_rows"]);
    let expected_cols = as_usize_list(&f["expected_coalesced_cols"]);
    let expected_vals: Vec<f32> = as_f64_list(&f["expected_coalesced_values"])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let expected_nnz = f["expected_coalesced_nnz"].as_u64().expect("nnz") as usize;

    let coo = CooTensor::new(row_indices, col_indices, values, nrows, ncols)
        .expect("CooTensor::new with duplicates");
    let coal = coo.coalesce();
    assert_eq!(coal.nnz(), expected_nnz);
    assert!(coal.is_coalesced());
    assert_eq!(coal.row_indices(), expected_rows.as_slice());
    assert_eq!(coal.col_indices(), expected_cols.as_slice());
    tolerance::assert_close_f32(
        coal.values(),
        &expected_vals,
        tolerance::F32_ELEMENTWISE,
        "CooTensor::coalesce values",
    );
}

// ---------------------------------------------------------------------------
// SemiStructuredSparseTensor + sparse_matmul_24.
// ---------------------------------------------------------------------------

#[test]
fn cpu_semi_structured_compress_decompress() {
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        let f = pick(&file, "semi_structured_24", dtype, Some("1d_8elem"));
        let dense_data = as_f64_list(&f["dense_data"]);
        let shape = as_usize_list(&f["shape"]);
        let expected_kept = as_f64_list(&f["expected_kept_values"]);
        let expected_nibbles: Vec<u8> = f["expected_nibbles"]
            .as_array()
            .expect("nibbles")
            .iter()
            .map(|x| x.as_u64().expect("nibble") as u8)
            .collect();
        let expected_decompressed = as_f64_list(&f["expected_decompressed"]);
        let expected_num_groups = f["expected_num_groups"].as_u64().expect("num_groups") as usize;

        match dtype {
            "float32" => {
                let dense = make_tensor_f32(&dense_data, &shape);
                let sst = SemiStructuredSparseTensor::compress(&dense)
                    .expect("SemiStructuredSparseTensor::compress");
                assert_eq!(sst.shape(), shape.as_slice());
                assert_eq!(sst.num_groups(), expected_num_groups);
                let exp_kept_f32: Vec<f32> = expected_kept.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    sst.values(),
                    &exp_kept_f32,
                    tolerance::F32_STRUCTURAL,
                    "kept values f32",
                );
                // Each group nibble matches.
                for (g, &expected_nibble) in expected_nibbles
                    .iter()
                    .enumerate()
                    .take(expected_num_groups)
                {
                    assert_eq!(sst.group_mask(g), expected_nibble, "group_mask({g})");
                }
                // Mask byte-packing: 2 groups per byte.
                assert_eq!(sst.mask().len(), expected_num_groups.div_ceil(2));

                // compression_ratio is bounded for non-empty inputs.
                let ratio = sst.compression_ratio();
                assert!(ratio > 0.0 && ratio <= 1.0);

                // decompress.
                let recovered = sst.decompress().expect("decompress");
                let exp_decomp_f32: Vec<f32> =
                    expected_decompressed.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    recovered.data().expect("decompressed"),
                    &exp_decomp_f32,
                    tolerance::F32_STRUCTURAL,
                    "decompressed values f32",
                );
            }
            "float64" => {
                let dense = make_tensor_f64(&dense_data, &shape);
                let sst = SemiStructuredSparseTensor::compress(&dense).expect("compress f64");
                tolerance::assert_close_f64(
                    sst.values(),
                    &expected_kept,
                    tolerance::F64_ELEMENTWISE,
                    "kept values f64",
                );
                let recovered = sst.decompress().expect("decompress f64");
                tolerance::assert_close_f64(
                    recovered.data().expect("decompressed f64"),
                    &expected_decompressed,
                    tolerance::F64_ELEMENTWISE,
                    "decompressed values f64",
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_sparse_matmul_24_3x8_8x4() {
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        let f = pick(&file, "sparse_matmul_24", dtype, Some("3x8_8x4"));
        let a_data = as_f64_list(&f["a_data"]);
        let a_shape = as_usize_list(&f["a_shape"]);
        let b_data = as_f64_list(&f["b_dense_data"]);
        let b_shape = as_usize_list(&f["b_shape"]);
        let expected = as_f64_list(&f["expected_matmul"]);
        let expected_shape = as_usize_list(&f["expected_matmul_shape"]);

        match dtype {
            "float32" => {
                let a = make_tensor_f32(&a_data, &a_shape);
                let b_dense = make_tensor_f32(&b_data, &b_shape);
                let b = SemiStructuredSparseTensor::compress(&b_dense).expect("compress B");
                let out = sparse_matmul_24(&a, &b).expect("sparse_matmul_24");
                assert_eq!(out.shape(), expected_shape.as_slice());
                let exp_f32: Vec<f32> = expected.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    out.data().expect("matmul data"),
                    &exp_f32,
                    tolerance::F32_MATMUL,
                    "sparse_matmul_24 f32",
                );
            }
            "float64" => {
                let a = make_tensor_f64(&a_data, &a_shape);
                let b_dense = make_tensor_f64(&b_data, &b_shape);
                let b = SemiStructuredSparseTensor::compress(&b_dense).expect("compress B f64");
                let out = sparse_matmul_24(&a, &b).expect("sparse_matmul_24 f64");
                tolerance::assert_close_f64(
                    out.data().expect("matmul data f64"),
                    &expected,
                    tolerance::F64_MATMUL,
                    "sparse_matmul_24 f64",
                );
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// SparseGrad — coalesce + apply_sgd.
// ---------------------------------------------------------------------------

#[test]
fn cpu_sparse_grad_coalesce_and_apply_sgd() {
    let file = load_fixtures();
    for dtype in ["float32", "float64"] {
        let f = pick(&file, "sparse_grad", dtype, Some("embedding_4x3_dup_row0"));
        let indices = as_usize_list(&f["indices"]);
        let values = as_f64_list(&f["values"]);
        let slab_shape = as_usize_list(&f["slab_shape"]);
        let coalesced_indices = as_usize_list(&f["expected_coalesced_indices"]);
        let coalesced_values = as_f64_list(&f["expected_coalesced_values"]);
        let expected_nnz = f["expected_nnz"].as_u64().expect("nnz") as usize;
        let expected_coalesced_nnz = f["expected_coalesced_nnz"].as_u64().expect("nnz") as usize;
        let param_shape = as_usize_list(&f["param_shape"]);
        let param_data = as_f64_list(&f["param_data"]);
        let lr = f["lr"].as_f64().expect("lr");
        let expected_uncoal = as_f64_list(&f["expected_uncoalesced_param"]);
        let expected_coal = as_f64_list(&f["expected_coalesced_param"]);

        match dtype {
            "float32" => {
                let vals: Vec<f32> = values.iter().map(|&x| x as f32).collect();
                let sg = SparseGrad::new(indices.clone(), vals, slab_shape.clone())
                    .expect("SparseGrad::new");
                assert_eq!(sg.nnz(), expected_nnz);
                assert_eq!(sg.indices(), indices.as_slice());
                assert_eq!(sg.slab_shape(), slab_shape.as_slice());
                assert_eq!(sg.slab_size(), slab_shape.iter().product::<usize>().max(1));

                // values() length == nnz * slab_size.
                let expected_values_len = expected_nnz * sg.slab_size();
                assert_eq!(sg.values().len(), expected_values_len);

                // coalesce.
                let sgc = sg.coalesce();
                assert_eq!(sgc.nnz(), expected_coalesced_nnz);
                assert_eq!(sgc.indices(), coalesced_indices.as_slice());
                let exp_coal_f32: Vec<f32> = coalesced_values.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    sgc.values(),
                    &exp_coal_f32,
                    tolerance::F32_ELEMENTWISE,
                    "SparseGrad coalesce values",
                );

                // apply_sgd on the uncoalesced grad — visits all 3 slabs in order.
                let mut param = make_tensor_f32(&param_data, &param_shape);
                sg.apply_sgd(&mut param, lr as f32)
                    .expect("apply_sgd uncoalesced");
                let exp_uncoal_f32: Vec<f32> = expected_uncoal.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    param.data().expect("param data"),
                    &exp_uncoal_f32,
                    tolerance::F32_ELEMENTWISE,
                    "apply_sgd uncoalesced",
                );

                // apply_sgd on the coalesced grad.
                let mut param2 = make_tensor_f32(&param_data, &param_shape);
                sgc.apply_sgd(&mut param2, lr as f32)
                    .expect("apply_sgd coalesced");
                let exp_coal_param_f32: Vec<f32> =
                    expected_coal.iter().map(|&x| x as f32).collect();
                tolerance::assert_close_f32(
                    param2.data().expect("param2 data"),
                    &exp_coal_param_f32,
                    tolerance::F32_ELEMENTWISE,
                    "apply_sgd coalesced",
                );
            }
            "float64" => {
                let sg = SparseGrad::new(indices.clone(), values, slab_shape.clone())
                    .expect("SparseGrad::new f64");
                let sgc = sg.coalesce();
                assert_eq!(sgc.nnz(), expected_coalesced_nnz);
                tolerance::assert_close_f64(
                    sgc.values(),
                    &coalesced_values,
                    tolerance::F64_ELEMENTWISE,
                    "SparseGrad coalesce values f64",
                );

                let mut param = make_tensor_f64(&param_data, &param_shape);
                sg.apply_sgd(&mut param, lr).expect("apply_sgd f64");
                tolerance::assert_close_f64(
                    param.data().expect("param data f64"),
                    &expected_uncoal,
                    tolerance::F64_ELEMENTWISE,
                    "apply_sgd uncoalesced f64",
                );
            }
            _ => unreachable!(),
        }
    }
}

// ===========================================================================
// GPU lane — every test cascade-skips against the tracked issue.
//
// Per `rust-gpu-discipline` §3 (PyTorch parity, hard requirement), the
// honest response when a module has no GPU implementation is to surface
// it as a tracked cascade issue rather than silently pass CPU work as
// GPU coverage. ferrotorch's nested + sparse modules build all storage
// via `TensorStorage::cpu` — there is no GPU code path to exercise.
//
// The skip is loud (`eprintln!`) and references the cascade issue so it
// shows up in audit trails. When a GPU implementation lands, this lane
// is replaced with real GPU dispatch (mirroring the pattern in
// `conformance_reduction.rs` Phase 2.2).
// ===========================================================================

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

    fn note_cascade_skip(test_name: &str) {
        // Loud skip — the cascade-handling mandate requires surfacing each
        // GPU-not-implemented case rather than silently passing.
        if let Some(reason) = cascade_skip_gpu() {
            eprintln!("skipping {test_name} on GPU: {reason}");
        }
    }

    /// Live GPU `NestedTensor::to_padded` / `from_padded` round-trip,
    /// ragged_dim = 0 (P4 of #806). Pre-P4 both methods called
    /// `tensor.data()?` which returns `GpuTensorNotAccessible` for any
    /// CUDA-resident component; post-P4 the path composes from the
    /// existing `fill_f{32,64}` + `strided_scatter_f{32,64}` +
    /// `strided_copy_f{32,64}` GPU primitives. PyTorch parity:
    /// `torch.nested.to_padded_tensor` on a CUDA nested tensor stays on
    /// device.
    #[test]
    fn gpu_nested_to_padded_ragged_dim0() {
        ensure_cuda_backend();

        // 3 components of shapes [(2,4), (3,4), (1,4)] on CUDA. Padded
        // output shape: [3, 3, 4].
        let cpu_t1 = make_tensor_f32(&(1..=8).map(|i| i as f64).collect::<Vec<_>>(), &[2, 4]);
        let cpu_t2 = make_tensor_f32(
            &(1..=12).map(|i| (i as f64) + 10.0).collect::<Vec<_>>(),
            &[3, 4],
        );
        let cpu_t3 = make_tensor_f32(
            &(1..=4).map(|i| (i as f64) + 100.0).collect::<Vec<_>>(),
            &[1, 4],
        );

        let t1 = cpu_t1.to(ferrotorch_core::Device::Cuda(0)).expect("t1->gpu");
        let t2 = cpu_t2.to(ferrotorch_core::Device::Cuda(0)).expect("t2->gpu");
        let t3 = cpu_t3.to(ferrotorch_core::Device::Cuda(0)).expect("t3->gpu");

        let nt = NestedTensor::new(vec![t1, t2, t3], 0).expect("nested");
        let padded = nt.to_padded(0.0_f32).expect("gpu to_padded");
        assert!(
            padded.is_cuda(),
            "to_padded output must remain on CUDA when components are CUDA"
        );
        assert_eq!(padded.shape(), &[3, 3, 4]);

        // Round-trip back to a NestedTensor on CUDA.
        let lengths = nt.ragged_lengths();
        let recovered =
            NestedTensor::<f32>::from_padded(&padded, &lengths, 0).expect("gpu from_padded");
        for comp in recovered.tensors() {
            assert!(
                comp.is_cuda(),
                "from_padded components must stay on CUDA when input is CUDA"
            );
        }
        assert_eq!(recovered.tensors()[0].shape(), &[2, 4]);
        assert_eq!(recovered.tensors()[1].shape(), &[3, 4]);
        assert_eq!(recovered.tensors()[2].shape(), &[1, 4]);

        // Compare the materialised values against the CPU oracle.
        let cpu_oracle = NestedTensor::new(
            vec![
                make_tensor_f32(&(1..=8).map(|i| i as f64).collect::<Vec<_>>(), &[2, 4]),
                make_tensor_f32(
                    &(1..=12).map(|i| (i as f64) + 10.0).collect::<Vec<_>>(),
                    &[3, 4],
                ),
                make_tensor_f32(
                    &(1..=4).map(|i| (i as f64) + 100.0).collect::<Vec<_>>(),
                    &[1, 4],
                ),
            ],
            0,
        )
        .expect("cpu oracle nested");
        let cpu_padded = cpu_oracle.to_padded(0.0_f32).expect("cpu to_padded");

        let gpu_padded_host = padded.cpu().expect("padded gpu->cpu");
        assert_eq!(
            gpu_padded_host.data().expect("padded data"),
            cpu_padded.data().expect("cpu padded data"),
            "gpu to_padded vs cpu oracle parity (ragged_dim=0)"
        );

        for (i, comp) in recovered.tensors().iter().enumerate() {
            let comp_host = comp.cpu().expect("comp gpu->cpu");
            assert_eq!(
                comp_host.data().expect("comp data"),
                cpu_oracle.tensors()[i].data().expect("oracle data"),
                "gpu from_padded vs cpu oracle parity, component {i}"
            );
        }
    }

    /// Live GPU `to_padded` / `from_padded` with ragged_dim != 0 (P4 of
    /// #806). Components are shape [d0, L_i] with the inner dim ragged;
    /// padded output is `[batch, d0, max_L]`. Exercises the
    /// non-contiguous narrow path inside `from_padded` (narrowing the
    /// inner dim produces a non-contiguous CUDA view, which materialises
    /// via `strided_copy_f{32,64}`).
    #[test]
    fn gpu_nested_to_padded_ragged_dim1() {
        ensure_cuda_backend();

        let cpu_t1 = make_tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let cpu_t2 = make_tensor_f32(&[7.0, 8.0, 9.0, 10.0], &[2, 2]);
        let t1 = cpu_t1.to(ferrotorch_core::Device::Cuda(0)).expect("t1->gpu");
        let t2 = cpu_t2.to(ferrotorch_core::Device::Cuda(0)).expect("t2->gpu");

        let nt = NestedTensor::new(vec![t1, t2], 1).expect("nested ragged_dim=1");
        let padded = nt.to_padded(0.0_f32).expect("gpu to_padded");
        assert!(padded.is_cuda());
        assert_eq!(padded.shape(), &[2, 2, 3]);

        // CPU oracle for parity.
        let cpu_oracle = NestedTensor::new(
            vec![
                make_tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
                make_tensor_f32(&[7.0, 8.0, 9.0, 10.0], &[2, 2]),
            ],
            1,
        )
        .expect("cpu oracle");
        let cpu_padded = cpu_oracle.to_padded(0.0_f32).expect("cpu to_padded");
        let gpu_padded_host = padded.cpu().expect("gpu->cpu");
        assert_eq!(
            gpu_padded_host.data().expect("data"),
            cpu_padded.data().expect("oracle data"),
            "gpu to_padded vs cpu oracle parity (ragged_dim=1)"
        );

        let lengths = nt.ragged_lengths();
        let recovered =
            NestedTensor::<f32>::from_padded(&padded, &lengths, 1).expect("gpu from_padded");
        for (i, comp) in recovered.tensors().iter().enumerate() {
            assert!(comp.is_cuda(), "component {i} must remain CUDA");
            let comp_host = comp.cpu().expect("gpu->cpu");
            assert_eq!(
                comp_host.data().expect("data"),
                cpu_oracle.tensors()[i].data().expect("oracle data"),
                "gpu from_padded vs cpu oracle parity (ragged_dim=1), component {i}"
            );
        }
    }

    #[test]
    fn gpu_nested_scaled_dot_product_attention() {
        ensure_cuda_backend();
        note_cascade_skip("gpu_nested_scaled_dot_product_attention");
    }

    #[test]
    fn gpu_packed_nested_tensor_ops() {
        ensure_cuda_backend();
        note_cascade_skip("gpu_packed_nested_tensor_ops");
    }

    /// Live GPU `SparseTensor::to_dense_on(Device::Cuda(0))` test (P3).
    /// PyTorch parity: `torch.sparse_coo_tensor(...).to_dense()` on a CUDA
    /// sparse tensor materialises via `cusparseSparseToDense`. Pre-P3 the
    /// only path was `to_dense()` (CPU) followed by `.to(Device::Cuda)`,
    /// which is a host detour; this test asserts the on-device materialisation.
    #[test]
    fn gpu_sparse_tensor_to_dense() {
        ensure_cuda_backend();
        let sp = SparseTensor::<f32>::new(
            vec![
                vec![0, 1],
                vec![0, 3],
                vec![1, 2],
                vec![2, 0],
                vec![2, 3],
                vec![3, 1],
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![4, 4],
        )
        .expect("sparse fixture");

        let cpu_dense = sp.to_dense().expect("cpu to_dense");
        let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

        let gpu_dense = sp
            .to_dense_on(ferrotorch_core::Device::Cuda(0))
            .expect("gpu to_dense_on");
        assert!(
            gpu_dense.is_cuda(),
            "to_dense_on(Cuda) output must remain on CUDA"
        );
        assert_eq!(gpu_dense.shape(), &[4, 4]);

        let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
        let gpu_data = gpu_back.data().expect("gpu->cpu data");
        for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "to_dense_on f32 elem {i}: gpu={a} cpu={b}"
            );
        }
    }

    /// Live GPU `SparseTensor::from_dense(&cuda_tensor, 0.0)` test (P3).
    /// PyTorch parity: `tensor.to_sparse()` on a CUDA dense tensor
    /// dispatches to `cusparseDenseToSparse_*`. Pre-P3 this errored with
    /// `GpuTensorNotAccessible` because `from_dense` called `tensor.data()?`
    /// which fails on CUDA.
    #[test]
    fn gpu_sparse_tensor_from_dense() {
        ensure_cuda_backend();

        let dense_data: Vec<f64> = vec![
            0.0, 1.0, 0.0, 2.0,
            0.0, 0.0, 3.0, 0.0,
            4.0, 0.0, 0.0, 5.0,
            0.0, 6.0, 0.0, 0.0,
        ];
        let dense_cpu = make_tensor_f32(&dense_data, &[4, 4]);
        let dense_gpu = dense_cpu
            .to(ferrotorch_core::Device::Cuda(0))
            .expect("dense->gpu");

        // Pre-fix: SparseTensor::from_dense calls tensor.data() which
        // returns GpuTensorNotAccessible for a CUDA tensor.
        let sp = SparseTensor::<f32>::from_dense(&dense_gpu, 0.0)
            .expect("gpu from_dense");
        assert_eq!(sp.shape(), &[4, 4]);
        assert_eq!(sp.nnz(), 6, "expected 6 non-zero entries");

        // Round-trip via CPU to_dense.
        let re_dense = sp.to_dense().expect("re-densify");
        let re_data = re_dense.data().expect("re data");
        for (i, (&got, &exp)) in re_data.iter().zip(dense_data.iter()).enumerate() {
            assert!(
                ((got as f64) - exp).abs() < 1e-5,
                "from_dense round-trip f32 elem {i}: got {got}, exp {exp}"
            );
        }
    }

    /// Live GPU SpMM test (P2): `SparseTensor::spmm` dispatches to
    /// `cusparseSpMM` when the dense operand is on CUDA. PyTorch parity:
    /// `torch.sparse.mm` runs on cuSPARSE in the same configuration.
    #[test]
    fn gpu_sparse_tensor_spmm() {
        ensure_cuda_backend();
        // 4x4 sparse, 4x3 dense, f32; dense lives on CUDA so spmm must
        // route through the cuSPARSE path (pre-P2 this errored with
        // GpuTensorNotAccessible from `dense.data()?`).
        let sp = SparseTensor::<f32>::new(
            vec![
                vec![0, 1],
                vec![0, 3],
                vec![1, 2],
                vec![2, 0],
                vec![2, 3],
                vec![3, 1],
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![4, 4],
        )
        .expect("sparse fixture");
        let dense_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let dense_cpu = make_tensor_f32(&dense_data, &[4, 3]);
        let dense_gpu = dense_cpu
            .to(ferrotorch_core::Device::Cuda(0))
            .expect("dense->gpu");

        let cpu_ref = sp.spmm(&dense_cpu).expect("cpu spmm reference");
        let cpu_ref_data = cpu_ref.data().expect("cpu spmm data").to_vec();

        let out = sp.spmm(&dense_gpu).expect("cusparse spmm");
        assert!(
            out.is_cuda(),
            "spmm output must remain on CUDA when input was CUDA"
        );
        assert_eq!(out.shape(), &[4, 3]);

        let out_cpu = out.cpu().expect("out gpu->cpu");
        let out_data = out_cpu.data().expect("out data");
        for (i, (&a, &b)) in out_data.iter().zip(cpu_ref_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "spmm GPU vs CPU mismatch at {i}: gpu={a} cpu={b}"
            );
        }
    }

    #[test]
    fn gpu_coo_csr_csc_conversions() {
        ensure_cuda_backend();
        note_cascade_skip("gpu_coo_csr_csc_conversions");
    }

    #[test]
    fn gpu_semi_structured_24() {
        ensure_cuda_backend();
        note_cascade_skip("gpu_semi_structured_24");
    }

    #[test]
    fn gpu_sparse_matmul_24() {
        ensure_cuda_backend();
        note_cascade_skip("gpu_sparse_matmul_24");
    }

    #[test]
    fn gpu_sparse_grad_apply_sgd() {
        ensure_cuda_backend();
        note_cascade_skip("gpu_sparse_grad_apply_sgd");
    }
}
