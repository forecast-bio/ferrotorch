//! Conformance suite for `ferrotorch-vision` — Layer 3, io module.
//!
//! Tracking issue: #870 (ferrotorch-vision conformance suite).
//!
//! Reference libraries:
//!   - `torch == 2.11.0`
//!   - `torchvision == 0.21.0`
//!
//! Fixtures live in `tests/conformance/fixtures.json`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
)]

use std::path::PathBuf;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_vision::{raw_image_to_tensor, read_image, read_image_rgba,
    read_image_as_tensor, tensor_to_raw_image};
use serde::Deserialize;

// cascade_skip is defined but only invoked conditionally; allow unused in this
// test binary since not all divergences manifest here.
#[allow(unused_macros)]
macro_rules! cascade_skip {
    ($reason:literal) => {{
        eprintln!("  [cascade_skip] {} — {}", module_path!(), $reason);
        return;
    }};
}

// ---------------------------------------------------------------------------
// Fixture loading helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[allow(dead_code)]
    metadata: serde_json::Value,
    fixtures: Vec<serde_json::Value>,
}

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures.json")
}

fn load_fixtures() -> FixtureFile {
    let path = fixtures_path();
    assert!(
        path.exists(),
        "fixtures.json not found. Run: python3 scripts/regenerate_vision_fixtures.py"
    );
    let body = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&body).unwrap()
}

fn get_fixture<'a>(fixtures: &'a [serde_json::Value], id: &str) -> Option<&'a serde_json::Value> {
    fixtures.iter().find(|f| f["id"] == id)
}

fn flatten_nested_f64(v: &serde_json::Value) -> Vec<f64> {
    match v {
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap()],
        serde_json::Value::Array(arr) => arr.iter().flat_map(flatten_nested_f64).collect(),
        _ => vec![],
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Layer 3: raw_image_to_tensor
// ---------------------------------------------------------------------------

#[test]
fn raw_image_to_tensor_4x4_rgb_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "raw_image_to_tensor_4x4_rgb")
        .expect("fixture raw_image_to_tensor_4x4_rgb not found");

    let expected_data = flatten_nested_f64(&fix["expected"]);
    let expected_shape: Vec<usize> = fix["expected_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    // Build a RawImage via tensor_to_raw_image (RawImage is #[non_exhaustive]).
    // We construct a CHW tensor whose pixel values match the fixture's u8 input,
    // convert to RawImage via tensor_to_raw_image, then call raw_image_to_tensor.
    let flat_u8: Vec<u8> = flatten_nested_f64(&fix["input_u8"])
        .iter()
        .map(|&v| v as u8)
        .collect();
    // Reinterpret u8 bytes as a float tensor in HWC order (u8/255), then transpose CHW.
    // Easier: write to a tempfile PNG and read back to get a proper RawImage.
    let tmp = tempfile::Builder::new().suffix(".png").tempfile().expect("tempfile");
    let path = tmp.path().to_path_buf();
    {
        use ferrotorch_vision::write_image;
        // Write via a CHW tensor built from the u8 fixture data.
        let chw_data: Vec<f64> = {
            let h = 4usize;
            let w = 4usize;
            let c = 3usize;
            let mut v = vec![0.0f64; c * h * w];
            for row in 0..h {
                for col in 0..w {
                    for ch in 0..c {
                        let src = row * w * c + col * c + ch;
                        let dst = ch * h * w + row * w + col;
                        v[dst] = flat_u8[src] as f64 / 255.0;
                    }
                }
            }
            v
        };
        let t = Tensor::from_storage(
            TensorStorage::cpu(chw_data),
            vec![3, 4, 4],
            false,
        ).unwrap();
        write_image(&path, &tensor_to_raw_image(&t).unwrap()).expect("write_image");
    }
    let raw = read_image(&path).expect("read_image");

    let out: Tensor<f64> = raw_image_to_tensor(&raw).unwrap();

    assert_eq!(out.shape(), &expected_shape[..]);
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-6, "raw_image_to_tensor");
}

// ---------------------------------------------------------------------------
// Layer 3: RawImage struct construction
// ---------------------------------------------------------------------------

#[test]
fn raw_image_struct_fields() {
    // Exercises the RawImage public fields (surface gate ref).
    // RawImage is #[non_exhaustive] — construct via tensor_to_raw_image.
    // A [3,1,3] CHW tensor with all-1 values → u8 255 → raw.data len == 9.
    let data = vec![1.0_f64; 9]; // [3, 1, 3] CHW
    let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 1, 3], false).unwrap();
    let raw = tensor_to_raw_image(&t).unwrap();
    assert_eq!(raw.width, 3);
    assert_eq!(raw.height, 1);
    assert_eq!(raw.channels, 3);
    assert_eq!(raw.data.len(), 9);
}

// ---------------------------------------------------------------------------
// Layer 3: tensor_to_raw_image (inverse of raw_image_to_tensor)
// ---------------------------------------------------------------------------

#[test]
fn tensor_to_raw_image_roundtrip() {
    // Build a CHW tensor, convert to RawImage, convert back, compare.
    let data: Vec<f64> = (0..48).map(|i| i as f64 / 48.0).collect(); // [3, 4, 4]
    let t = Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 4, 4], false).unwrap();

    let raw = tensor_to_raw_image(&t).unwrap();
    assert_eq!(raw.width, 4);
    assert_eq!(raw.height, 4);
    assert_eq!(raw.channels, 3);
    assert_eq!(raw.data.len(), 48);

    // Round-trip back to tensor.
    let t2: Tensor<f64> = raw_image_to_tensor(&raw).unwrap();
    assert_eq!(t2.shape(), &[3, 4, 4]);
    // Values go through u8 quantization (float->u8->float), so 1/255 tolerance.
    for (&orig, &rt) in data.iter().zip(t2.data().unwrap().iter()) {
        let diff = (orig - rt).abs();
        assert!(
            diff < 1.0 / 255.0 + 1e-6,
            "roundtrip diff {diff} too large: orig={orig} rt={rt}"
        );
    }
}

// ---------------------------------------------------------------------------
// Layer 3: read_image (file-backed — cascade_skip if no test fixture on disk)
// ---------------------------------------------------------------------------

#[test]
fn read_image_returns_err_on_missing_file() {
    // read_image must return Err for a non-existent path, not panic.
    let result = read_image("/nonexistent/path/no_file.png");
    assert!(result.is_err(), "read_image must return Err for missing file");
}

#[test]
fn read_image_rgba_returns_err_on_missing_file() {
    let result = read_image_rgba("/nonexistent/path/no_file.png");
    assert!(result.is_err());
}

#[test]
fn read_image_as_tensor_returns_err_on_missing_file() {
    let result = read_image_as_tensor::<f32>("/nonexistent/path/no_file.png");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Layer 3: write_image / write_tensor_as_image (tempfile round-trip)
// Note: excluded from surface gate fixture-level; tested here structurally.
// ---------------------------------------------------------------------------

#[test]
fn write_image_and_read_back_roundtrip() {
    use ferrotorch_vision::write_image;

    let tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("tempfile");
    let path = tmp.path().to_path_buf();

    // Build RawImage via tensor_to_raw_image (RawImage is #[non_exhaustive]).
    // CHW layout: channel 0 = red (1,0,0,0), channel 1 = green (0,1,0,0),
    // channel 2 = blue (0,0,1,0) → 2×2 image.
    // pixel (0,0): R=1.0 G=0.0 B=0.0 (red)
    // pixel (0,1): R=0.0 G=1.0 B=0.0 (green)
    // pixel (1,0): R=0.0 G=0.0 B=1.0 (blue)
    // pixel (1,1): R=0.5 G=0.5 B=0.5 (~gray)
    #[rustfmt::skip]
    let chw: Vec<f64> = vec![
        1.0, 0.0,  0.0, 0.5,   // R channel [2,2]
        0.0, 1.0,  0.0, 0.5,   // G channel
        0.0, 0.0,  1.0, 0.5,   // B channel
    ];
    let t = Tensor::from_storage(TensorStorage::cpu(chw), vec![3, 2, 2], false).unwrap();
    let raw = tensor_to_raw_image(&t).unwrap();
    assert_eq!(raw.width, 2);
    assert_eq!(raw.height, 2);

    write_image(&path, &raw).expect("write_image");

    let loaded = read_image(&path).expect("read_image after write");
    assert_eq!(loaded.width, 2);
    assert_eq!(loaded.height, 2);
    assert_eq!(loaded.channels, 3);

    // PNG is lossless: pixel values must be bit-exact after write/read.
    for (expected, actual) in raw.data.iter().zip(loaded.data.iter()) {
        assert_eq!(expected, actual, "write/read roundtrip pixel mismatch");
    }
}

#[test]
fn write_tensor_as_image_roundtrip() {
    use ferrotorch_vision::{write_tensor_as_image};

    let tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("tempfile");
    let path = tmp.path().to_path_buf();

    // [3, 2, 2] CHW float tensor with values in [0, 1].
    let data: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 2, 2], false).unwrap();
    write_tensor_as_image(&path, &t).expect("write_tensor_as_image");

    let loaded = read_image(&path).expect("read after write_tensor");
    assert_eq!(loaded.width, 2);
    assert_eq!(loaded.height, 2);
    assert_eq!(loaded.channels, 3);
}
