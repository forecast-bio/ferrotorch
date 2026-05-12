//! ML-op dump binary for the ferrotorch-ml real-artifact parity harness
//! (Phase D.3, #1159).
//!
//! Companion to `scripts/verify_ml_inference.py` and the pin script
//! `scripts/pin_pretrained_ml_fixtures.py`. For each of the 5 configs in
//! the matrix, this example reads the pinned inputs from the
//! `ferrotorch/ml-sklearn-parity-v1` HF mirror (paths passed by the
//! Python harness via CLI flags so this binary stays network-free),
//! invokes the matching ferrotorch-ml op, and dumps the output in the
//! same `[u32 num_tensors]` + per-tensor `[u32 ndim][u32 × ndim shape]
//! [f32 data]` multi-tensor binary format the pin script produces.
//!
//! ## Equality semantics (mirror of pin script `meta.json`)
//!
//! * `pca_n4` — output is `[100, 4]` f32 projected coords. Harness checks
//!   cosine_sim ≥ 0.9999 PER principal component (handles per-PC sign
//!   flip). Rust dumps the f32 output.
//! * `standard_scaler` — output is `[100, 10]` f32 normalised features.
//!   Harness checks max_abs ≤ 1e-6.
//! * `one_hot_encoder` — output is `[5, 3]` f32 one-hot. Exact integer
//!   equality.
//! * `kfold_5` — output is a 5-line JSON manifest on stdout (no .bin);
//!   each line is a (train_idx, test_idx) pair. The harness applies
//!   SET-equality.
//! * `train_test_split_80_20` — output is split index lists (which rows
//!   ended up in train vs test) in JSON, plus the resulting label
//!   vectors so the harness can check label consistency. No .bin output
//!   (the split is identity-on-row-indices, not a numeric transform).
//!
//! ## Usage
//!
//! ```text
//! cargo run -p ferrotorch-ml --release --example ml_op_dump -- \
//!   --config pca_n4 \
//!   --input-X /path/to/input_X.bin \
//!   --output /path/to/rust_output.bin
//! ```
//!
//! The `--config` flag selects one of the 5 configs hard-coded below.
//! Input flags vary per config — the Python harness wires the matching
//! pinned fixture paths from `ferrotorch/ml-sklearn-parity-v1`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2};

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_decomp::PCA;
use ferrolearn_model_sel::{train_test_split, KFold};
use ferrolearn_preprocess::{OneHotEncoder, StandardScaler};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Args {
    config: String,
    inputs: BTreeMap<String, PathBuf>,
    output: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        let arg = &argv[i];
        let val = || {
            argv.get(i + 1)
                .ok_or_else(|| format!("{arg} needs a value"))
                .cloned()
        };
        match arg.as_str() {
            "--config" => {
                args.config = val()?;
                i += 2;
            }
            "--output" => {
                args.output = Some(PathBuf::from(val()?));
                i += 2;
            }
            other if other.starts_with("--input-") => {
                let key = other.trim_start_matches("--input-").to_string();
                args.inputs.insert(key, PathBuf::from(val()?));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if args.config.is_empty() {
        return Err("--config is required".to_string());
    }
    Ok(args)
}

// ---------------------------------------------------------------------------
// Multi-tensor binary format — mirrors the Python pin script + dataloader
// example.
// ---------------------------------------------------------------------------

/// One tensor read out of a `.bin` file: `(shape, row-major f32 data)`.
type TensorBuf = (Vec<usize>, Vec<f32>);

fn read_multi_tensor_f32(path: &Path) -> Result<Vec<TensorBuf>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw)
        .map_err(|e| format!("read {}: {e}", path.display()))?;

    if raw.len() < 4 {
        return Err(format!("{}: file too short", path.display()));
    }
    let mut off = 0usize;
    let n = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as usize;
    off += 4;

    let mut out = Vec::with_capacity(n);
    for ti in 0..n {
        if raw.len() < off + 4 {
            return Err(format!("{}: truncated reading ndim[{ti}]", path.display()));
        }
        let ndim = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if raw.len() < off + 4 * ndim {
            return Err(format!("{}: truncated reading shape[{ti}]", path.display()));
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as usize);
            off += 4;
        }
        let numel: usize = shape.iter().product();
        if raw.len() < off + 4 * numel {
            return Err(format!("{}: truncated reading data[{ti}]", path.display()));
        }
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            data.push(f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.push((shape, data));
    }
    if off != raw.len() {
        return Err(format!(
            "{}: {} trailing bytes after {n} tensors",
            path.display(),
            raw.len() - off
        ));
    }
    Ok(out)
}

fn write_multi_tensor_f32(path: &Path, tensors: &[TensorBuf]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(
        &u32::try_from(tensors.len())
            .expect("num_tensors fits u32")
            .to_le_bytes(),
    )?;
    for (shape, data) in tensors {
        let expect: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expect,
            "tensor data {} disagrees with shape product {}",
            data.len(),
            expect
        );
        f.write_all(
            &u32::try_from(shape.len())
                .expect("ndim fits u32")
                .to_le_bytes(),
        )?;
        for &d in shape {
            f.write_all(&u32::try_from(d).expect("dim fits u32").to_le_bytes())?;
        }
        let mut buf = Vec::with_capacity(data.len() * 4);
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    Ok(())
}

fn require_input(args: &Args, key: &str) -> Result<PathBuf, String> {
    args.inputs
        .get(key)
        .cloned()
        .ok_or_else(|| format!("config {:?} requires --input-{key}", args.config))
}

fn require_output(args: &Args) -> Result<PathBuf, String> {
    args.output
        .clone()
        .ok_or_else(|| format!("config {:?} requires --output", args.config))
}

// ---------------------------------------------------------------------------
// Per-config ferrotorch-ml invocations.
// ---------------------------------------------------------------------------

fn load_array2_f32(path: &Path, expect_shape: &[usize]) -> Result<Array2<f32>, String> {
    let tensors = read_multi_tensor_f32(path)?;
    if tensors.len() != 1 {
        return Err(format!(
            "{}: expected 1 tensor, got {}",
            path.display(),
            tensors.len()
        ));
    }
    let (shape, data) = tensors.into_iter().next().unwrap();
    if shape != expect_shape {
        return Err(format!(
            "{}: shape {:?} != expected {:?}",
            path.display(),
            shape,
            expect_shape
        ));
    }
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| format!("{}: from_shape_vec: {e}", path.display()))
}

fn load_array1_f32(path: &Path, expect_len: usize) -> Result<Array1<f32>, String> {
    let tensors = read_multi_tensor_f32(path)?;
    if tensors.len() != 1 {
        return Err(format!(
            "{}: expected 1 tensor, got {}",
            path.display(),
            tensors.len()
        ));
    }
    let (shape, data) = tensors.into_iter().next().unwrap();
    if shape != [expect_len] {
        return Err(format!(
            "{}: shape {:?} != expected [{}]",
            path.display(),
            shape,
            expect_len
        ));
    }
    Ok(Array1::from(data))
}

fn run_pca_n4(args: &Args) -> Result<String, String> {
    // sklearn PCA computes with f64 internally; we mirror that here so
    // the comparison reflects the *algorithmic* parity, not f32 noise.
    let x_path = require_input(args, "X")?;
    let out_path = require_output(args)?;

    let x_f32 = load_array2_f32(&x_path, &[100, 10])?;
    let x_f64 = x_f32.mapv(f64::from);

    let pca = PCA::<f64>::new(4);
    let fitted = pca
        .fit(&x_f64, &())
        .map_err(|e| format!("PCA::fit failed: {e}"))?;
    let y_f64 = fitted
        .transform(&x_f64)
        .map_err(|e| format!("PCA::transform failed: {e}"))?;

    if y_f64.shape() != [100, 4] {
        return Err(format!("PCA output shape {:?} != [100, 4]", y_f64.shape()));
    }
    let y_f32: Vec<f32> = y_f64.iter().map(|&v| v as f32).collect();

    write_multi_tensor_f32(&out_path, &[(vec![100, 4], y_f32.clone())])
        .map_err(|e| format!("write {}: {e}", out_path.display()))?;

    let l2 = y_f32.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
    Ok(format!(
        "{{\"config\":\"pca_n4\",\"output_shape\":[100,4],\"l2_norm\":{l2:.6}}}"
    ))
}

fn run_standard_scaler(args: &Args) -> Result<String, String> {
    let x_path = require_input(args, "X")?;
    let out_path = require_output(args)?;

    let x_f32 = load_array2_f32(&x_path, &[100, 10])?;
    // Use f64 internally to mirror sklearn's compute precision; the
    // tolerance is 1e-6 (≈ exact f32 arithmetic) so the cast back to f32
    // for the dump is consistent with the pin script.
    let x_f64 = x_f32.mapv(f64::from);

    let sc = StandardScaler::<f64>::new();
    let y_f64 = sc
        .fit_transform(&x_f64)
        .map_err(|e| format!("StandardScaler::fit_transform failed: {e}"))?;

    if y_f64.shape() != [100, 10] {
        return Err(format!(
            "StandardScaler output shape {:?} != [100, 10]",
            y_f64.shape()
        ));
    }
    let y_f32: Vec<f32> = y_f64.iter().map(|&v| v as f32).collect();
    write_multi_tensor_f32(&out_path, &[(vec![100, 10], y_f32.clone())])
        .map_err(|e| format!("write {}: {e}", out_path.display()))?;

    Ok("{\"config\":\"standard_scaler\",\"output_shape\":[100,10]}".to_string())
}

fn run_one_hot_encoder(args: &Args) -> Result<String, String> {
    let x_path = require_input(args, "X-indices")?;
    let out_path = require_output(args)?;

    // The pin script ships category indices as f32 in input_X_indices.bin
    // (Array2<usize> over {a:0, b:1, c:2}). Decode to usize for
    // ferrolearn's OneHotEncoder<F=f64>, which keys on Array2<usize>.
    let x_f32 = load_array2_f32(&x_path, &[5, 1])?;
    let mut x_idx = Array2::<usize>::zeros((5, 1));
    for ((i, j), &v) in x_f32.indexed_iter() {
        if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
            return Err(format!(
                "one_hot_encoder: input_X_indices[{i},{j}] = {v} is not a non-negative integer"
            ));
        }
        x_idx[[i, j]] = v as usize;
    }

    let enc = OneHotEncoder::<f64>::new();
    let y_f64 = enc
        .fit_transform(&x_idx)
        .map_err(|e| format!("OneHotEncoder::fit_transform failed: {e}"))?;
    if y_f64.shape() != [5, 3] {
        return Err(format!(
            "OneHotEncoder output shape {:?} != [5, 3]",
            y_f64.shape()
        ));
    }
    let y_f32: Vec<f32> = y_f64.iter().map(|&v| v as f32).collect();
    write_multi_tensor_f32(&out_path, &[(vec![5, 3], y_f32.clone())])
        .map_err(|e| format!("write {}: {e}", out_path.display()))?;

    Ok("{\"config\":\"one_hot_encoder\",\"output_shape\":[5,3]}".to_string())
}

fn run_kfold_5(_args: &Args) -> Result<String, String> {
    // KFold has no input file — it just takes n_samples and produces the
    // (train, test) index pairs. The harness compares SET-equality
    // against the sklearn manifest in fold_indices.json.
    let n_samples = 50usize;
    let n_splits = 5usize;
    let kf = KFold::new(n_splits).shuffle(true).random_state(42);
    let folds = kf.split(n_samples);
    if folds.len() != n_splits {
        return Err(format!("KFold returned {} folds, expected {n_splits}", folds.len()));
    }

    // Emit a JSON manifest as the verdict line so the Python harness can
    // parse fold contents directly from stdout (avoiding a separate .json
    // file on disk for a CLI ergonomics win).
    let mut s = String::new();
    s.push('{');
    s.push_str("\"config\":\"kfold_5\",");
    s.push_str(&format!("\"n_samples\":{n_samples},"));
    s.push_str(&format!("\"n_splits\":{n_splits},"));
    s.push_str("\"folds\":[");
    for (i, (train, test)) in folds.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"train\":[");
        for (j, t) in train.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&t.to_string());
        }
        s.push_str("],\"test\":[");
        for (j, t) in test.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&t.to_string());
        }
        s.push_str("]}");
    }
    s.push_str("]}");
    Ok(s)
}

fn run_train_test_split_80_20(args: &Args) -> Result<String, String> {
    let x_path = require_input(args, "X")?;
    let y_path = require_input(args, "y")?;

    let x = load_array2_f32(&x_path, &[100, 10])?;
    let y = load_array1_f32(&y_path, 100)?;

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42))
        .map_err(|e| format!("train_test_split failed: {e}"))?;

    if x_train.shape() != [80, 10] || x_test.shape() != [20, 10] {
        return Err(format!(
            "train_test_split: shapes wrong: x_train={:?} x_test={:?}",
            x_train.shape(),
            x_test.shape()
        ));
    }
    if y_train.len() != 80 || y_test.len() != 20 {
        return Err(format!(
            "train_test_split: y_train.len={} y_test.len={}",
            y_train.len(),
            y_test.len()
        ));
    }

    // Reverse-look-up each split row's index in the original X (every
    // row of X is unique because the pin script materialised it from
    // randn(100, 10) — checked at pin time). The harness uses these
    // indices to verify the SET-equality invariant
    // (train ∪ test == [0, 100), disjoint, sizes 80/20).
    let find_row = |row: &ndarray::ArrayView1<'_, f32>| -> Option<usize> {
        for i in 0..100 {
            let orig = x.row(i);
            if orig.iter().zip(row.iter()).all(|(a, b)| a == b) {
                return Some(i);
            }
        }
        None
    };

    let mut train_indices: Vec<usize> = Vec::with_capacity(80);
    for r in x_train.rows() {
        let idx = find_row(&r)
            .ok_or_else(|| "train row not present in original X".to_string())?;
        train_indices.push(idx);
    }
    let mut test_indices: Vec<usize> = Vec::with_capacity(20);
    for r in x_test.rows() {
        let idx = find_row(&r)
            .ok_or_else(|| "test row not present in original X".to_string())?;
        test_indices.push(idx);
    }

    // Label consistency check (per #1159 spec): the y_test[k] returned
    // by train_test_split must match the y of the original test_indices[k]
    // row. Mismatch here would mean ferrolearn's train_test_split decoupled
    // x and y indexing — a real bug, not a PRNG-order quirk.
    for (k, &i) in test_indices.iter().enumerate() {
        if y_test[k] != y[i] {
            return Err(format!(
                "train_test_split: label mismatch at test[{k}]: y_test={} != y[{i}]={}",
                y_test[k], y[i]
            ));
        }
    }
    for (k, &i) in train_indices.iter().enumerate() {
        if y_train[k] != y[i] {
            return Err(format!(
                "train_test_split: label mismatch at train[{k}]: y_train={} != y[{i}]={}",
                y_train[k], y[i]
            ));
        }
    }

    let mut s = String::new();
    s.push('{');
    s.push_str("\"config\":\"train_test_split_80_20\",");
    s.push_str(&format!("\"n_train\":{},", train_indices.len()));
    s.push_str(&format!("\"n_test\":{},", test_indices.len()));
    s.push_str("\"train_indices\":[");
    for (i, t) in train_indices.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&t.to_string());
    }
    s.push_str("],\"test_indices\":[");
    for (i, t) in test_indices.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&t.to_string());
    }
    s.push_str("]}");
    Ok(s)
}

// ---------------------------------------------------------------------------
// Dispatch.
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args = parse_args()?;
    eprintln!(
        "[ml_op_dump] config={} inputs={:?} output={:?}",
        args.config, args.inputs, args.output
    );

    let verdict = match args.config.as_str() {
        "pca_n4" => run_pca_n4(&args)?,
        "standard_scaler" => run_standard_scaler(&args)?,
        "one_hot_encoder" => run_one_hot_encoder(&args)?,
        "kfold_5" => run_kfold_5(&args)?,
        "train_test_split_80_20" => run_train_test_split_80_20(&args)?,
        other => return Err(format!("unknown config {other:?}")),
    };
    println!("{verdict}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[ml_op_dump] error: {e}");
        std::process::exit(1);
    }
}
