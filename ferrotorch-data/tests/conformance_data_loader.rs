//! Conformance Layer 2/3 — dataloader, collate, and transforms modules.
//!
//! Exercises `DataLoader`, `WorkerMode`, `default_collate`,
//! `default_collate_pair`, `Normalize`, `ToTensor`, `Compose`,
//! `RandomHorizontalFlip`, `RandomCrop`, and `manual_seed` against
//! PyTorch-generated fixtures in `tests/conformance/fixtures/data.json`.
//!
//! Tracking issue: #838.

use std::path::PathBuf;
use std::sync::Arc;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_data::{
    Compose, DataLoader, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor, Transform,
    VecDataset, WorkerMode, default_collate, default_collate_pair, manual_seed,
};
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

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    let storage = TensorStorage::cpu(data.to_vec());
    Tensor::from_storage(storage, shape.to_vec(), false).unwrap()
}

// ---------------------------------------------------------------------------
// DataLoader — sequential iteration
// ---------------------------------------------------------------------------

/// Sequential `DataLoader::iter` yields batches matching `expected_batches`.
#[test]
fn dataloader_sequential() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "dataloader") {
        if fx["subtest"].as_str() != Some("sequential") {
            continue;
        }
        matched += 1;
        let n = fx["n"].as_u64().unwrap() as usize;
        let batch_size = fx["batch_size"].as_u64().unwrap() as usize;
        let shuffle = fx["shuffle"].as_bool().unwrap();
        let drop_last = fx["drop_last"].as_bool().unwrap();
        let expected: Vec<Vec<f64>> = fx["expected_batches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect();

        let data: Vec<i32> = (0..n as i32).collect();
        let ds = Arc::new(VecDataset::new(data));
        let loader = DataLoader::new(ds, batch_size)
            .unwrap()
            .shuffle(shuffle)
            .drop_last(drop_last)
            // Disable prefetch for deterministic single-thread iteration.
            .prefetch_factor(0);

        let batches: Vec<Vec<f64>> = loader
            .iter(0)
            .map(|r| r.unwrap().into_iter().map(|v| v as f64).collect())
            .collect();

        assert_eq!(
            batches, expected,
            "DataLoader sequential(n={n}, batch={batch_size}) mismatch"
        );
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=dataloader subtest=sequential — \
         did the fixture file get regenerated?"
    );
}

/// `DataLoader` with `drop_last=true` drops the final partial batch.
#[test]
fn dataloader_drop_last() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "dataloader") {
        if fx["subtest"].as_str() != Some("drop_last") {
            continue;
        }
        matched += 1;
        let n = fx["n"].as_u64().unwrap() as usize;
        let batch_size = fx["batch_size"].as_u64().unwrap() as usize;
        let drop_last = fx["drop_last"].as_bool().unwrap();
        let expected_num_batches = fx["expected_num_batches"].as_u64().unwrap() as usize;
        let expected: Vec<Vec<f64>> = fx["expected_batches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect();

        let ds = Arc::new(VecDataset::new((0..n as i32).collect::<Vec<_>>()));
        let loader = DataLoader::new(ds, batch_size)
            .unwrap()
            .drop_last(drop_last)
            .prefetch_factor(0);

        let batches: Vec<Vec<f64>> = loader
            .iter(0)
            .map(|r| r.unwrap().into_iter().map(|v| v as f64).collect())
            .collect();

        assert_eq!(
            batches, expected,
            "DataLoader drop_last batch content mismatch"
        );
        assert_eq!(
            batches.len(),
            expected_num_batches,
            "DataLoader drop_last num_batches mismatch"
        );
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=dataloader subtest=drop_last — \
         did the fixture file get regenerated?"
    );
}

/// Shuffled `DataLoader` covers all n samples exactly once per epoch
/// (permutation completeness). The fixture notes that ferrotorch uses a
/// different PRNG, so exact order is not asserted.
/// Tests `DataLoader::shuffle` builder.
#[test]
fn dataloader_shuffle_coverage() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "dataloader") {
        if fx["subtest"].as_str() != Some("shuffle_coverage") {
            continue;
        }
        matched += 1;
        let n = fx["n"].as_u64().unwrap() as usize;
        let batch_size = fx["batch_size"].as_u64().unwrap() as usize;
        let expected_sorted: Vec<i32> = fx["expected_sorted_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();

        let ds = Arc::new(VecDataset::new((0..n as i32).collect::<Vec<_>>()));
        let loader = DataLoader::new(ds, batch_size)
            .unwrap()
            .shuffle(true)
            .seed(42)
            .prefetch_factor(0);

        let mut all: Vec<i32> = loader.iter(0).flat_map(|r| r.unwrap()).collect();
        all.sort_unstable();

        assert_eq!(
            all, expected_sorted,
            "DataLoader shuffle_coverage: sorted output should equal 0..n"
        );
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=dataloader subtest=shuffle_coverage — \
         did the fixture file get regenerated?"
    );
}

/// `DataLoader::len` counts the number of batches it will produce.
#[test]
fn dataloader_num_batches() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "dataloader") {
        if fx["subtest"].as_str() != Some("num_batches") {
            continue;
        }
        matched += 1;
        let n = fx["n"].as_u64().unwrap() as usize;
        let batch_size = fx["batch_size"].as_u64().unwrap() as usize;
        let drop_last = fx["drop_last"].as_bool().unwrap_or(false);
        let expected_num_batches = fx["expected_num_batches"].as_u64().unwrap() as usize;

        let ds = Arc::new(VecDataset::new((0..n as i32).collect::<Vec<_>>()));
        let loader = DataLoader::new(ds, batch_size)
            .unwrap()
            .drop_last(drop_last);

        assert_eq!(
            loader.len(),
            expected_num_batches,
            "DataLoader(n={n}, batch={batch_size}, drop_last={drop_last}).len() \
             should be {expected_num_batches}"
        );
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=dataloader subtest=num_batches — \
         did the fixture file get regenerated?"
    );
}

// ---------------------------------------------------------------------------
// DataLoader — WorkerMode
// ---------------------------------------------------------------------------

/// `WorkerMode::IntraBatch` is the default; `worker_mode()` and
/// `current_worker_mode()` round-trip correctly.
#[test]
fn worker_mode_default_and_roundtrip() {
    let ds = Arc::new(VecDataset::new(vec![1i32, 2, 3]));
    let loader = DataLoader::new(ds.clone(), 2).unwrap();
    assert_eq!(
        loader.current_worker_mode(),
        WorkerMode::IntraBatch,
        "default WorkerMode should be IntraBatch"
    );

    let loader = DataLoader::new(ds, 2)
        .unwrap()
        .worker_mode(WorkerMode::CrossBatch);
    assert_eq!(
        loader.current_worker_mode(),
        WorkerMode::CrossBatch,
        "worker_mode(CrossBatch) should set CrossBatch"
    );
}

/// `DataLoader::drop_last` builder correctly influences `len()`.
#[test]
fn dataloader_drop_last_builder() {
    let ds = Arc::new(VecDataset::new((0..10i32).collect::<Vec<_>>()));
    let with_drop = DataLoader::new(ds.clone(), 3).unwrap().drop_last(true);
    let without_drop = DataLoader::new(ds, 3).unwrap().drop_last(false);
    assert_eq!(with_drop.len(), 3);
    assert_eq!(without_drop.len(), 4);
}

/// `DataLoader::seed` builder influences shuffled output (different seed →
/// different epoch-0 order, same seed → same order).
#[test]
fn dataloader_seed_builder() {
    let n = 20;
    let ds = Arc::new(VecDataset::new((0..n as i32).collect::<Vec<_>>()));

    let a: Vec<i32> = DataLoader::new(ds.clone(), n)
        .unwrap()
        .shuffle(true)
        .seed(11)
        .prefetch_factor(0)
        .iter(0)
        .flat_map(|r| r.unwrap())
        .collect();

    let b: Vec<i32> = DataLoader::new(ds.clone(), n)
        .unwrap()
        .shuffle(true)
        .seed(11)
        .prefetch_factor(0)
        .iter(0)
        .flat_map(|r| r.unwrap())
        .collect();

    let c: Vec<i32> = DataLoader::new(ds, n)
        .unwrap()
        .shuffle(true)
        .seed(99)
        .prefetch_factor(0)
        .iter(0)
        .flat_map(|r| r.unwrap())
        .collect();

    assert_eq!(a, b, "same seed must produce same order");
    assert_ne!(a, c, "different seeds should produce different order");
}

/// `DataLoader::iter_collated` collates each batch via the supplied function.
#[test]
fn dataloader_iter_collated() {
    let data: Vec<Vec<i32>> = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let ds = Arc::new(VecDataset::new(data));

    // Custom collate: sum all elements across the batch.
    let loader = DataLoader::new(ds, 2)
        .unwrap()
        .prefetch_factor(0)
        .with_collate(|batch: Vec<Vec<i32>>| {
            let sum: i32 = batch.into_iter().flatten().sum();
            Ok(vec![sum])
        });

    let results: Vec<Vec<i32>> = loader
        .iter_collated(0)
        .expect("iter_collated should succeed when collate_fn is set")
        .map(|r| r.unwrap())
        .collect();

    // batch 0: [1,2] + [3,4] => sum=10; batch 1: [5,6] => sum=11
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], vec![10]);
    assert_eq!(results[1], vec![11]);
}

/// `DataLoader::iter_collated` without a collate function returns `Err`.
#[test]
fn dataloader_iter_collated_no_fn_returns_err() {
    let ds = Arc::new(VecDataset::new(vec![1i32, 2, 3]));
    let loader = DataLoader::new(ds, 2).unwrap();
    assert!(
        loader.iter_collated(0).is_err(),
        "iter_collated without with_collate should return Err"
    );
}

// ---------------------------------------------------------------------------
// default_collate
// ---------------------------------------------------------------------------

/// `default_collate` stacks 1-D tensors into a 2-D batch.
#[test]
fn collate_stack_1d() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "collate") {
        if fx["subtest"].as_str() != Some("stack_1d") {
            continue;
        }
        matched += 1;
        let rows: Vec<Vec<f32>> = fx["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let expected_shape: Vec<usize> = fx["expected_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_data: Vec<f32> = fx["expected_data"]
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

        let input_shape: Vec<usize> = fx["input_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let tensors: Vec<Tensor<f32>> = rows.iter().map(|row| t32(row, &input_shape)).collect();

        let batch = default_collate(tensors).unwrap();
        assert_eq!(
            batch.shape(),
            expected_shape.as_slice(),
            "stack_1d shape mismatch"
        );
        assert_eq!(
            batch.data_vec().unwrap(),
            expected_data,
            "stack_1d data mismatch"
        );
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=collate subtest=stack_1d — \
         did the fixture file get regenerated?"
    );
}

/// `default_collate` stacks 2-D tensors into a 3-D batch.
#[test]
fn collate_stack_2d() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "collate") {
        if fx["subtest"].as_str() != Some("stack_2d") {
            continue;
        }
        matched += 1;
        let num_samples = fx["num_samples"].as_u64().unwrap() as usize;
        let input_shape: Vec<usize> = fx["input_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_shape: Vec<usize> = fx["expected_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        // Reconstruct the sample tensors from expected_data rows.
        let expected_data_flat: Vec<f32> = fx["expected_data"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|mat| {
                mat.as_array().unwrap().iter().flat_map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                })
            })
            .collect();

        let numel_per_sample: usize = input_shape.iter().product();
        let tensors: Vec<Tensor<f32>> = (0..num_samples)
            .map(|i| {
                let start = i * numel_per_sample;
                t32(
                    &expected_data_flat[start..start + numel_per_sample],
                    &input_shape,
                )
            })
            .collect();

        let batch = default_collate(tensors).unwrap();
        assert_eq!(
            batch.shape(),
            expected_shape.as_slice(),
            "stack_2d shape mismatch"
        );
        assert_eq!(
            batch.data_vec().unwrap(),
            expected_data_flat,
            "stack_2d data mismatch"
        );
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=collate subtest=stack_2d — \
         did the fixture file get regenerated?"
    );
}

/// `default_collate` returns `Err` for an empty input.
#[test]
fn collate_empty_returns_err() {
    let result = default_collate::<f32>(vec![]);
    assert!(result.is_err(), "default_collate([]) must return Err");
}

/// `default_collate_pair` stacks (input, target) tensor pairs.
#[test]
fn collate_pair_basic() {
    let x1 = t32(&[1.0, 2.0], &[2]);
    let y1 = t32(&[0.0], &[1]);
    let x2 = t32(&[3.0, 4.0], &[2]);
    let y2 = t32(&[1.0], &[1]);

    let (bx, by) = default_collate_pair(vec![(x1, y1), (x2, y2)]).unwrap();
    assert_eq!(bx.shape(), &[2, 2]);
    assert_eq!(by.shape(), &[2, 1]);
    assert_eq!(bx.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(by.data_vec().unwrap(), vec![0.0, 1.0]);
}

/// `default_collate_pair` returns `Err` for an empty input.
#[test]
fn collate_pair_empty_returns_err() {
    let result = default_collate_pair::<f32>(vec![]);
    assert!(result.is_err(), "default_collate_pair([]) must return Err");
}

// ---------------------------------------------------------------------------
// Normalize
// ---------------------------------------------------------------------------

/// `Normalize` two-channel fixture: `(x - mean) / std` per channel.
#[test]
fn normalize_two_channel() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "normalize") {
        if fx["subtest"].as_str() != Some("two_channel") {
            continue;
        }
        matched += 1;
        let input_flat: Vec<f64> = fx["input"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let input_shape: Vec<usize> = fx["input_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mean: Vec<f64> = fx["mean"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let std: Vec<f64> = fx["std"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_flat: Vec<f64> = fx["expected"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();

        let tensor = t64(&input_flat, &input_shape);
        let norm = Normalize::<f64>::new(mean, std).unwrap();
        let out = norm.apply(tensor).unwrap();
        let got = out.data_vec().unwrap();

        assert_eq!(
            got.len(),
            expected_flat.len(),
            "Normalize output length mismatch"
        );
        for (i, (&g, &e)) in got.iter().zip(expected_flat.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-9,
                "Normalize two_channel: index {i}: got {g}, expected {e}"
            );
        }
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=normalize subtest=two_channel — \
         did the fixture file get regenerated?"
    );
}

/// `Normalize` with mean=0 and std=1 is the identity.
#[test]
fn normalize_identity() {
    let fixtures = load_fixtures();
    let mut matched: usize = 0;
    for fx in fixtures_of_kind(&fixtures, "normalize") {
        if fx["subtest"].as_str() != Some("identity") {
            continue;
        }
        matched += 1;
        let input_flat: Vec<f64> = fx["input"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let mean: Vec<f64> = fx["mean"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let std: Vec<f64> = fx["std"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_flat: Vec<f64> = fx["expected"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();

        // Shape: input is [[1, 2, 3]] => [1, 3]
        let tensor = t64(&input_flat, &[1, input_flat.len()]);
        let norm = Normalize::<f64>::new(mean, std).unwrap();
        let out = norm.apply(tensor).unwrap();
        let got = out.data_vec().unwrap();

        for (i, (&g, &e)) in got.iter().zip(expected_flat.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-9,
                "Normalize identity: index {i}: got {g}, expected {e}"
            );
        }
    }
    assert!(
        matched > 0,
        "no fixtures matched kind=normalize subtest=identity — \
         did the fixture file get regenerated?"
    );
}

/// `Normalize::new` returns `Err` when mean and std have different lengths.
#[test]
fn normalize_length_mismatch_err() {
    let result = Normalize::<f32>::new(vec![0.0, 1.0], vec![1.0]);
    assert!(
        result.is_err(),
        "Normalize::new should return Err when mean.len() != std.len()"
    );
}

// ---------------------------------------------------------------------------
// ToTensor
// ---------------------------------------------------------------------------

/// `ToTensor` is the identity transform under the `Transform<T>` trait
/// (the inherent `ToTensor::apply` is the image→tensor path; pre-#1113
/// `Compose<T>` pipelines see the identity path).
#[test]
fn to_tensor_identity() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let t = t32(&data, &[3]);
    let out = <ToTensor as Transform<f32>>::apply(&ToTensor, t).unwrap();
    assert_eq!(out.data_vec().unwrap(), data, "ToTensor should be identity");
}

// ---------------------------------------------------------------------------
// Compose
// ---------------------------------------------------------------------------

/// `Compose` applies transforms in order.
#[test]
fn compose_chains_in_order() {
    // Two Normalize transforms applied sequentially.
    // Channel 0: input=[5.0], mean=2.0, std=1.0 => 3.0; then mean=1.0, std=1.0 => 2.0
    let t = t64(&[5.0, 10.0, 15.0], &[1, 3]);
    let pipeline = Compose::new(vec![
        Box::new(Normalize::<f64>::new(vec![0.0], vec![5.0]).unwrap()),
        Box::new(Normalize::<f64>::new(vec![0.0], vec![1.0]).unwrap()),
    ]);
    let out = pipeline.apply(t).unwrap();
    let got = out.data_vec().unwrap();
    // [5/5, 10/5, 15/5] = [1.0, 2.0, 3.0] then /1.0 => same
    for (i, (&g, &e)) in got.iter().zip([1.0_f64, 2.0, 3.0].iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-9,
            "Compose index {i}: got {g}, expected {e}"
        );
    }
}

/// `Compose` with an empty list is identity.
#[test]
fn compose_empty_is_identity() {
    let data = vec![7.0_f32, 8.0, 9.0];
    let t = t32(&data, &[3]);
    let pipeline = Compose::<f32>::new(vec![]);
    let out = pipeline.apply(t).unwrap();
    assert_eq!(
        out.data_vec().unwrap(),
        data,
        "Compose([]) should be identity"
    );
}

// ---------------------------------------------------------------------------
// RandomHorizontalFlip
// ---------------------------------------------------------------------------

/// `RandomHorizontalFlip` with `p=1.0` always flips along the last dimension.
#[test]
fn random_horizontal_flip_always() {
    // Use manual_seed for determinism in this test.
    manual_seed(0);
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = t64(&data, &[2, 3]);
    let flip = RandomHorizontalFlip::<f64>::new(1.0).unwrap();
    let out = flip.apply(t).unwrap();
    let got = out.data_vec().unwrap();
    // Row 0: [1,2,3] -> [3,2,1]; Row 1: [4,5,6] -> [6,5,4]
    assert_eq!(got, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

/// `RandomHorizontalFlip` with `p=0.0` never flips.
#[test]
fn random_horizontal_flip_never() {
    let data = vec![1.0_f64, 2.0, 3.0];
    let t = t64(&data, &[1, 3]);
    let flip = RandomHorizontalFlip::<f64>::new(0.0).unwrap();
    let out = flip.apply(t).unwrap();
    assert_eq!(
        out.data_vec().unwrap(),
        data,
        "RandomHorizontalFlip(p=0) must be identity"
    );
}

/// `RandomHorizontalFlip::new` rejects `p` outside `[0, 1]`.
#[test]
fn random_horizontal_flip_invalid_p() {
    assert!(RandomHorizontalFlip::<f32>::new(1.5).is_err());
    assert!(RandomHorizontalFlip::<f32>::new(-0.1).is_err());
}

// ---------------------------------------------------------------------------
// RandomCrop
// ---------------------------------------------------------------------------

/// `RandomCrop` output shape equals `[C, crop_h, crop_w]`.
#[test]
fn random_crop_output_shape() {
    manual_seed(1);
    // [3, 8, 8] input, crop to [3, 5, 5]
    let numel = 3 * 8 * 8;
    let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let t = t32(&data, &[3, 8, 8]);
    // RandomCrop::new takes (height, width).
    let crop = RandomCrop::<f32>::new(5, 5);
    let out = crop.apply(t).unwrap();
    assert_eq!(out.shape(), &[3, 5, 5]);
}

/// `RandomCrop` with crop equal to input size is identity.
#[test]
fn random_crop_exact_size_is_identity() {
    let numel = 2 * 4 * 4;
    let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let t = t32(&data, &[2, 4, 4]);
    let expected = data.clone();
    let crop = RandomCrop::<f32>::new(4, 4);
    let out = crop.apply(t).unwrap();
    assert_eq!(out.shape(), &[2, 4, 4]);
    assert_eq!(out.data_vec().unwrap(), expected);
}

/// `RandomCrop` with crop larger than input returns `Err`.
#[test]
fn random_crop_too_large_returns_err() {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let t = t32(&data, &[1, 4, 4]);
    let crop = RandomCrop::<f32>::new(5, 4);
    assert!(
        crop.apply(t).is_err(),
        "RandomCrop larger than input must return Err"
    );
}

// ---------------------------------------------------------------------------
// manual_seed
// ---------------------------------------------------------------------------

/// `manual_seed` resets the global RNG so that subsequent RandomHorizontalFlip
/// calls produce reproducible output.
#[test]
fn manual_seed_reproducibility() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let flip = RandomHorizontalFlip::<f32>::new(0.5).unwrap();

    manual_seed(42);
    let results_a: Vec<Vec<f32>> = (0..20)
        .map(|_| {
            let t = t32(&data, &[1, 3]);
            flip.apply(t).unwrap().data_vec().unwrap()
        })
        .collect();

    manual_seed(42);
    let results_b: Vec<Vec<f32>> = (0..20)
        .map(|_| {
            let t = t32(&data, &[1, 3]);
            flip.apply(t).unwrap().data_vec().unwrap()
        })
        .collect();

    assert_eq!(
        results_a, results_b,
        "manual_seed should make RandomHorizontalFlip sequence reproducible"
    );
}

// ---------------------------------------------------------------------------
// Transform trait object safety
// ---------------------------------------------------------------------------

/// `Transform<T>` is object-safe via `Box<dyn Transform<T>>`.
#[test]
fn transform_trait_object_safe() {
    let t: Box<dyn Transform<f32>> = Box::new(ToTensor);
    let data = vec![1.0_f32, 2.0];
    let input = t32(&data, &[2]);
    let out = t.apply(input).unwrap();
    assert_eq!(out.data_vec().unwrap(), data);
}
