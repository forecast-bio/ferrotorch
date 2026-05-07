//! Conformance Layer 2/3 — dataset module.
//!
//! Exercises `VecDataset`, `TensorDataset`, `ConcatDataset`, `ChainDataset`,
//! and `WorkerInfo` against PyTorch-generated fixtures in
//! `tests/conformance/fixtures/data.json`.
//!
//! Tracking issue: #838.

use std::path::PathBuf;
use std::sync::Arc;

use ferrotorch_data::{
    ChainDataset, ConcatDataset, Dataset, IterableDataset, TensorDataset, VecDataset, WorkerInfo,
};
use ferrotorch_core::{Tensor, TensorStorage};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("data.json")
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixtures: Vec<serde_json::Value>,
}

fn load_fixtures() -> Vec<serde_json::Value> {
    let body = std::fs::read_to_string(fixtures_path())
        .expect("read fixtures/data.json — run scripts/regenerate_data_fixtures.py first");
    let f: FixtureFile = serde_json::from_str(&body).expect("parse fixtures/data.json");
    f.fixtures
}

fn fixtures_of_kind<'a>(
    fixtures: &'a [serde_json::Value],
    kind: &str,
) -> Vec<&'a serde_json::Value> {
    fixtures
        .iter()
        .filter(|f| f["kind"].as_str() == Some(kind))
        .collect()
}

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    let storage = TensorStorage::cpu(data.to_vec());
    Tensor::from_storage(storage, shape.to_vec(), false).unwrap()
}

// ---------------------------------------------------------------------------
// VecDataset
// ---------------------------------------------------------------------------

/// `VecDataset::len` matches fixture `expected_len`.
#[test]
fn vec_dataset_len() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "vec_dataset") {
        if fx["subtest"].as_str() != Some("len") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let expected_len = fx["expected_len"].as_u64().unwrap() as usize;

        let ds = VecDataset::new((0..n).collect::<Vec<usize>>());
        assert_eq!(
            ds.len(),
            expected_len,
            "VecDataset(n={n}).len() should be {expected_len}"
        );
    }
}

/// `VecDataset::get` returns the value at the given index.
#[test]
fn vec_dataset_get() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "vec_dataset") {
        if fx["subtest"].as_str() != Some("get") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let index = fx["index"].as_u64().unwrap() as usize;
        let expected_value = fx["expected_value"].as_u64().unwrap() as usize;

        let ds = VecDataset::new((0..n).collect::<Vec<usize>>());
        let got = ds.get(index).expect("VecDataset::get should succeed");
        assert_eq!(
            got, expected_value,
            "VecDataset(n={n}).get({index}) expected {expected_value}"
        );
    }
}

/// `VecDataset::get` with an out-of-bounds index returns `Err(IndexOutOfBounds)`.
#[test]
fn vec_dataset_get_oob() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "vec_dataset") {
        if fx["subtest"].as_str() != Some("get_oob") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let index = fx["index"].as_u64().unwrap() as usize;

        let ds = VecDataset::new((0..n).collect::<Vec<usize>>());
        let result = ds.get(index);
        assert!(
            result.is_err(),
            "VecDataset(n={n}).get({index}) should return Err for OOB index"
        );
    }
}

// ---------------------------------------------------------------------------
// TensorDataset
// ---------------------------------------------------------------------------

/// `TensorDataset::is_empty` returns `false` for a non-empty dataset and
/// `true` only when `len() == 0`.
#[test]
fn tensor_dataset_is_empty() {
    // Non-empty dataset.
    let x = t32(&[1.0, 2.0], &[2, 1]);
    let ds = TensorDataset::new(vec![x]).unwrap();
    assert!(!ds.is_empty(), "TensorDataset::is_empty should be false when len > 0");
    assert_eq!(ds.len(), 2);
}

/// `TensorDataset::len` equals the size of dimension 0.
#[test]
fn tensor_dataset_len() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "tensor_dataset") {
        if fx["subtest"].as_str() != Some("len") {
            continue;
        }
        let x_shape: Vec<usize> = fx["x_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let y_shape: Vec<usize> = fx["y_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_len = fx["expected_len"].as_u64().unwrap() as usize;

        let x_numel: usize = x_shape.iter().product();
        let y_numel: usize = y_shape.iter().product();
        let x = t32(
            &(0..x_numel).map(|i| i as f32).collect::<Vec<_>>(),
            &x_shape,
        );
        let y = t32(
            &(0..y_numel).map(|i| i as f32).collect::<Vec<_>>(),
            &y_shape,
        );
        let ds = TensorDataset::new(vec![x, y]).unwrap();
        assert_eq!(
            ds.len(),
            expected_len,
            "TensorDataset len should be {expected_len}"
        );
        assert!(!ds.is_empty());
    }
}

/// `TensorDataset::get(i)` returns the i-th row of each stored tensor.
#[test]
fn tensor_dataset_get() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "tensor_dataset") {
        if fx["subtest"].as_str() != Some("get") {
            continue;
        }

        let index = fx["index"].as_u64().unwrap() as usize;
        let x_shape: Vec<usize> = fx["x_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let y_shape: Vec<usize> = fx["y_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        // Reconstruct the tensors from flattened fixture data.
        let x_data: Vec<f32> = fx["x_data"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
            })
            .collect();
        let y_data: Vec<f32> = fx["y_data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let expected_x: Vec<f32> = fx["expected_x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected_y = fx["expected_y"].as_f64().unwrap() as f32;

        let x = t32(&x_data, &x_shape);
        let y = t32(&y_data, &y_shape);
        let ds = TensorDataset::new(vec![x, y]).unwrap();

        let sample = ds.get(index).unwrap();
        assert_eq!(sample.len(), 2);

        let got_x = sample[0].data_vec().unwrap();
        assert_eq!(
            got_x, expected_x,
            "TensorDataset.get({index}) x row mismatch"
        );

        let got_y = sample[1].data_vec().unwrap();
        assert_eq!(
            got_y[0], expected_y,
            "TensorDataset.get({index}) y scalar mismatch"
        );
    }
}

// ---------------------------------------------------------------------------
// WorkerInfo
// ---------------------------------------------------------------------------

/// `WorkerInfo::new` stores worker_id and num_workers fields correctly.
#[test]
fn worker_info_new() {
    let info = WorkerInfo::new(2, 4);
    assert_eq!(info.worker_id, 2);
    assert_eq!(info.num_workers, 4);
}

/// `WorkerInfo` is Clone and the clone equals the original.
#[test]
fn worker_info_clone() {
    let a = WorkerInfo::new(0, 1);
    let b = a.clone();
    assert_eq!(a.worker_id, b.worker_id);
    assert_eq!(a.num_workers, b.num_workers);
}

// ---------------------------------------------------------------------------
// ConcatDataset
// ---------------------------------------------------------------------------

/// `ConcatDataset::len` equals the sum of sub-dataset lengths.
#[test]
fn concat_dataset_len() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "concat_dataset") {
        if fx["subtest"].as_str() != Some("len") {
            continue;
        }
        let sizes: Vec<usize> = fx["sizes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_len = fx["expected_len"].as_u64().unwrap() as usize;

        // ConcatDataset fixture uses a global sequence multiplied by 10:
        // d0=[10,20,30], d1=[40,50], so the counter runs across all datasets.
        let datasets: Vec<VecDataset<i32>> = {
            let mut counter = 1i32;
            sizes
                .iter()
                .map(|&sz| {
                    let data: Vec<i32> = (0..sz as i32).map(|_| { let v = counter * 10; counter += 1; v }).collect();
                    VecDataset::new(data)
                })
                .collect()
        };
        let ds = ConcatDataset::new(datasets).unwrap();
        assert_eq!(
            ds.len(),
            expected_len,
            "ConcatDataset(sizes={sizes:?}).len() should be {expected_len}"
        );
    }
}

/// `ConcatDataset::get(i)` maps to the correct sub-dataset and local index.
#[test]
fn concat_dataset_get() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "concat_dataset") {
        if fx["subtest"].as_str() != Some("get") {
            continue;
        }
        let sizes: Vec<usize> = fx["sizes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let index = fx["index"].as_u64().unwrap() as usize;
        let expected_value = fx["expected_value"].as_i64().unwrap() as i32;

        // Reconstruct datasets: d0=[10,20,30], d1=[40,50] as in the fixture.
        // The counter is global across sub-datasets (10,20,30,40,50 = 1..5 × 10).
        let datasets: Vec<VecDataset<i32>> = {
            let mut counter = 1i32;
            sizes
                .iter()
                .map(|&sz| {
                    let data: Vec<i32> = (0..sz as i32)
                        .map(|_| { let v = counter * 10; counter += 1; v })
                        .collect();
                    VecDataset::new(data)
                })
                .collect()
        };
        let ds = ConcatDataset::new(datasets).unwrap();
        let got = ds.get(index).unwrap();
        assert_eq!(
            got, expected_value,
            "ConcatDataset.get({index}) expected {expected_value}"
        );
    }
}

/// `ConcatDataset::new` returns `Err` when given an empty list.
#[test]
fn concat_dataset_empty_err() {
    let result = ConcatDataset::<VecDataset<i32>>::new(vec![]);
    assert!(result.is_err(), "ConcatDataset::new([]) must return Err");
}

// ---------------------------------------------------------------------------
// ChainDataset
// ---------------------------------------------------------------------------

/// `ChainDataset` as `IterableDataset` yields items in fixture order.
#[test]
fn chain_dataset_iter_order() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "chain_dataset") {
        if fx["subtest"].as_str() != Some("iter_order") {
            continue;
        }
        let a_values: Vec<i32> = fx["a_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let b_values: Vec<i32> = fx["b_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let expected: Vec<i32> = fx["expected_items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();

        let ds = ChainDataset::new(vec![
            VecDataset::new(a_values),
            VecDataset::new(b_values),
        ])
        .unwrap();

        let got: Vec<i32> = IterableDataset::iter(&ds, None)
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(got, expected, "ChainDataset iter_order mismatch");
    }
}

/// `ChainDataset::new` returns `Err` when given an empty list.
#[test]
fn chain_dataset_empty_err() {
    let result = ChainDataset::<VecDataset<i32>>::new(vec![]);
    assert!(result.is_err(), "ChainDataset::new([]) must return Err");
}

/// `ChainDataset` also implements `Dataset` (map-style access).
#[test]
fn chain_dataset_as_dataset() {
    let a = VecDataset::new(vec![10i32, 20]);
    let b = VecDataset::new(vec![30i32, 40]);
    let ds = ChainDataset::new(vec![a, b]).unwrap();
    assert_eq!(ds.len(), 4);
    assert_eq!(Dataset::get(&ds, 0).unwrap(), 10);
    assert_eq!(Dataset::get(&ds, 2).unwrap(), 30);
}

/// `Dataset` is object-safe: `VecDataset` can be used as `dyn Dataset`.
#[test]
fn dataset_trait_object_safe() {
    let ds = VecDataset::new(vec![1u32, 2, 3]);
    let dyn_ds: &dyn Dataset<Sample = u32> = &ds;
    assert_eq!(dyn_ds.len(), 3);
    assert_eq!(dyn_ds.get(1).unwrap(), 2);
}

/// `Dataset` implementations are `Send + Sync` so they can be shared across threads.
#[test]
fn dataset_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VecDataset<i32>>();
    assert_send_sync::<ConcatDataset<VecDataset<i32>>>();
    assert_send_sync::<ChainDataset<VecDataset<i32>>>();
}

/// `Arc<VecDataset>` implements `Dataset` (required by `DataLoader::new`).
#[test]
fn vec_dataset_arc_dataset() {
    let ds = Arc::new(VecDataset::new(vec![100i32, 200, 300]));
    assert_eq!(ds.len(), 3);
    assert_eq!(ds.get(2).unwrap(), 300);
}
