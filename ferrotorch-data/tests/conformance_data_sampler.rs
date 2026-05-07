//! Conformance Layer 2/3 — sampler module.
//!
//! Exercises `SequentialSampler`, `RandomSampler`, `BatchSampler`,
//! `WeightedRandomSampler`, `DistributedSampler`, and `shuffle_with_seed`
//! against PyTorch-generated fixtures in
//! `tests/conformance/fixtures/data.json`.
//!
//! Tracking issue: #838.

use std::collections::HashSet;
use std::path::PathBuf;

use ferrotorch_data::{
    BatchSampler, DistributedSampler, RandomSampler, Sampler, SequentialSampler,
    WeightedRandomSampler, shuffle_with_seed,
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

// ---------------------------------------------------------------------------
// SequentialSampler
// ---------------------------------------------------------------------------

/// `SequentialSampler::indices` matches the fixture's `expected_indices`.
#[test]
fn sequential_sampler_indices() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "sequential_sampler") {
        if fx["subtest"].as_str() != Some("indices") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let expected: Vec<usize> = fx["expected_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let s = SequentialSampler::new(n);
        let got = s.indices(0);
        assert_eq!(
            got, expected,
            "SequentialSampler(n={n}) indices mismatch"
        );
    }
}

/// `SequentialSampler` returns the same order every epoch.
#[test]
fn sequential_sampler_epoch_invariant() {
    let s = SequentialSampler::new(6);
    let e0 = s.indices(0);
    let e5 = s.indices(5);
    assert_eq!(e0, e5, "SequentialSampler should produce same order every epoch");
}

/// `SequentialSampler::len` matches the construction size.
#[test]
fn sequential_sampler_len() {
    let s = SequentialSampler::new(7);
    assert_eq!(s.len(), 7);
    assert!(!s.is_empty());
    assert!(SequentialSampler::new(0).is_empty());
}

// ---------------------------------------------------------------------------
// RandomSampler
// ---------------------------------------------------------------------------

/// `RandomSampler::indices` produces a permutation of 0..n (permutation
/// completeness). The fixture notes that ferrotorch uses a different PRNG, so
/// only the permutation property is asserted — not the exact order.
#[test]
fn random_sampler_is_permutation() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "random_sampler") {
        if fx["subtest"].as_str() != Some("is_permutation") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let seed = fx["seed"].as_u64().unwrap();

        let s = RandomSampler::new(n, seed);
        let got = s.indices(0);
        assert_eq!(got.len(), n, "RandomSampler should return {n} indices");

        let mut sorted = got.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(
            sorted, expected,
            "RandomSampler(n={n}) result must be a permutation of 0..n"
        );
    }
}

/// Calling `indices` twice with the same epoch yields the same ordering.
#[test]
fn random_sampler_reproducible() {
    let s = RandomSampler::new(20, 42);
    let a = s.indices(0);
    let b = s.indices(0);
    assert_eq!(a, b, "RandomSampler same seed+epoch must be reproducible");
}

/// Different epochs produce different orderings (with overwhelming probability
/// for n=20, seed=42).
#[test]
fn random_sampler_epoch_varies() {
    let s = RandomSampler::new(20, 42);
    let e0 = s.indices(0);
    let e1 = s.indices(1);
    assert_ne!(e0, e1, "RandomSampler different epochs should produce different order");
}

// ---------------------------------------------------------------------------
// shuffle_with_seed
// ---------------------------------------------------------------------------

/// `shuffle_with_seed` is deterministic: same seed → same permutation.
#[test]
fn shuffle_with_seed_deterministic() {
    let mut a: Vec<usize> = (0..50).collect();
    let mut b: Vec<usize> = (0..50).collect();
    shuffle_with_seed(&mut a, 999);
    shuffle_with_seed(&mut b, 999);
    assert_eq!(a, b, "shuffle_with_seed must be deterministic");
}

/// Different seeds produce different permutations (with overwhelming probability
/// for n=100).
#[test]
fn shuffle_with_seed_different_seeds() {
    let mut a: Vec<usize> = (0..100).collect();
    let mut b: Vec<usize> = (0..100).collect();
    shuffle_with_seed(&mut a, 1);
    shuffle_with_seed(&mut b, 2);
    assert_ne!(a, b, "different seeds should produce different permutations");
}

/// `shuffle_with_seed` is a permutation: the sorted output equals the sorted input.
#[test]
fn shuffle_with_seed_is_permutation() {
    let mut v: Vec<usize> = (0..30).collect();
    shuffle_with_seed(&mut v, 77);
    let mut sorted = v.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..30).collect::<Vec<_>>(),
        "shuffle_with_seed output must be a permutation of the input"
    );
}

// ---------------------------------------------------------------------------
// BatchSampler
// ---------------------------------------------------------------------------

/// `BatchSampler::batches` with `drop_last=false` matches `expected_batches`.
#[test]
fn batch_sampler_sequential_no_drop() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "batch_sampler") {
        if fx["subtest"].as_str() != Some("sequential_no_drop") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let batch_size = fx["batch_size"].as_u64().unwrap() as usize;
        let drop_last = fx["drop_last"].as_bool().unwrap();
        let expected: Vec<Vec<usize>> = fx["expected_batches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();

        let inner = SequentialSampler::new(n);
        let bs = BatchSampler::new(inner, batch_size, drop_last);
        let got = bs.batches(0);
        assert_eq!(got, expected, "BatchSampler sequential_no_drop mismatch");
    }
}

/// `BatchSampler::batches` with `drop_last=true` drops the final incomplete batch.
#[test]
fn batch_sampler_sequential_drop_last() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "batch_sampler") {
        if fx["subtest"].as_str() != Some("sequential_drop_last") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let batch_size = fx["batch_size"].as_u64().unwrap() as usize;
        let drop_last = fx["drop_last"].as_bool().unwrap();
        let expected: Vec<Vec<usize>> = fx["expected_batches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();

        let inner = SequentialSampler::new(n);
        let bs = BatchSampler::new(inner, batch_size, drop_last);
        let got = bs.batches(0);
        assert_eq!(got, expected, "BatchSampler sequential_drop_last mismatch");
    }
}

/// `BatchSampler::batches` with exact divisor produces no remainder batch.
#[test]
fn batch_sampler_exact_division() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "batch_sampler") {
        if fx["subtest"].as_str() != Some("exact_division") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let batch_size = fx["batch_size"].as_u64().unwrap() as usize;
        let drop_last = fx["drop_last"].as_bool().unwrap();
        let expected: Vec<Vec<usize>> = fx["expected_batches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();

        let inner = SequentialSampler::new(n);
        let bs = BatchSampler::new(inner, batch_size, drop_last);
        let got = bs.batches(0);
        assert_eq!(got, expected, "BatchSampler exact_division mismatch");
    }
}

/// `BatchSampler::num_batches` reports the correct count for several
/// (n, batch_size, drop_last) combinations.
#[test]
fn batch_sampler_num_batches() {
    // n=10, batch=3, drop=false => 4
    let s = BatchSampler::new(SequentialSampler::new(10), 3, false);
    assert_eq!(s.num_batches(), 4);
    // n=10, batch=3, drop=true  => 3
    let s = BatchSampler::new(SequentialSampler::new(10), 3, true);
    assert_eq!(s.num_batches(), 3);
    // n=9, batch=3, drop=false  => 3
    let s = BatchSampler::new(SequentialSampler::new(9), 3, false);
    assert_eq!(s.num_batches(), 3);
    // n=9, batch=3, drop=true   => 3 (exact division)
    let s = BatchSampler::new(SequentialSampler::new(9), 3, true);
    assert_eq!(s.num_batches(), 3);
}

// ---------------------------------------------------------------------------
// WeightedRandomSampler
// ---------------------------------------------------------------------------

/// `WeightedRandomSampler` with a heavily biased weight draws the dominant
/// class more than 70 % of the time (fixture: weight 100× vs 1× others).
#[test]
fn weighted_random_sampler_heavy_bias() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "weighted_random_sampler") {
        if fx["subtest"].as_str() != Some("heavy_bias") {
            continue;
        }
        let weights: Vec<f64> = fx["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let n_samples = fx["n_samples"].as_u64().unwrap() as usize;
        let heavy_index = fx["heavy_index"].as_u64().unwrap() as usize;

        let s = WeightedRandomSampler::new(weights, n_samples, true, 42);
        let idx = s.indices(0);
        assert_eq!(idx.len(), n_samples);

        let heavy_count = idx.iter().filter(|&&i| i == heavy_index).count();
        let threshold = (n_samples as f64 * 0.7) as usize;
        assert!(
            heavy_count > threshold,
            "WeightedRandomSampler heavy_bias: heavy_index count {heavy_count} \
             should exceed {threshold} (70% of {n_samples})"
        );
    }
}

/// `WeightedRandomSampler` produces indices only within `0..weights.len()`.
#[test]
fn weighted_random_sampler_indices_in_range() {
    let weights = vec![1.0, 2.0, 3.0, 4.0];
    let s = WeightedRandomSampler::new(weights.clone(), 50, true, 7);
    let idx = s.indices(0);
    assert!(
        idx.iter().all(|&i| i < weights.len()),
        "WeightedRandomSampler indices must be in 0..weights.len()"
    );
}

/// `WeightedRandomSampler::len` equals `num_samples`.
#[test]
fn weighted_random_sampler_len() {
    let s = WeightedRandomSampler::new(vec![1.0, 1.0, 1.0], 15, true, 0);
    assert_eq!(s.len(), 15);
}

// ---------------------------------------------------------------------------
// DistributedSampler
// ---------------------------------------------------------------------------

/// `DistributedSampler` with `shuffle=false` partitions indices exactly
/// as the fixture specifies.
#[test]
fn distributed_sampler_sequential_partition() {
    let fixtures = load_fixtures();
    for fx in fixtures_of_kind(&fixtures, "distributed_sampler") {
        if fx["subtest"].as_str() != Some("sequential_partition") {
            continue;
        }
        let n = fx["n"].as_u64().unwrap() as usize;
        let num_replicas = fx["num_replicas"].as_u64().unwrap() as usize;
        let rank = fx["rank"].as_u64().unwrap() as usize;
        let expected: Vec<usize> = fx["expected_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_per_rank_len = fx["expected_per_rank_len"].as_u64().unwrap() as usize;

        let s = DistributedSampler::new(n, num_replicas, rank).shuffle(false);
        let got = s.indices(0);

        assert_eq!(
            got, expected,
            "DistributedSampler(n={n}, replicas={num_replicas}, rank={rank}) \
             sequential partition mismatch"
        );
        assert_eq!(
            got.len(),
            expected_per_rank_len,
            "DistributedSampler per-rank length mismatch"
        );
    }
}

/// With shuffle enabled and a fixed seed, `indices` is reproducible.
#[test]
fn distributed_sampler_shuffle_reproducible() {
    let s = DistributedSampler::new(20, 4, 0).seed(42);
    let a = s.indices(0);
    let b = s.indices(0);
    assert_eq!(
        a, b,
        "DistributedSampler shuffle must be reproducible with the same epoch"
    );
}

/// All ranks together cover all `0..n` indices (padding may cause some to
/// repeat, but none should be absent).
#[test]
fn distributed_sampler_all_ranks_cover_population() {
    let n = 10;
    let num_replicas = 3;
    let mut all: HashSet<usize> = HashSet::new();
    for rank in 0..num_replicas {
        let s = DistributedSampler::new(n, num_replicas, rank).shuffle(false);
        all.extend(s.indices(0));
    }
    let missing: Vec<usize> = (0..n).filter(|i| !all.contains(i)).collect();
    assert!(
        missing.is_empty(),
        "DistributedSampler: indices {missing:?} not covered by any rank"
    );
}

/// `DistributedSampler::len` equals `ceil(num_samples / num_replicas)`.
#[test]
fn distributed_sampler_len() {
    // 10 samples, 3 replicas => ceil(10/3) = 4
    let s = DistributedSampler::new(10, 3, 0);
    assert_eq!(s.len(), 4);
    // Exact: 12 samples, 4 replicas => 3
    let s = DistributedSampler::new(12, 4, 2);
    assert_eq!(s.len(), 3);
}

/// `Sampler` is object-safe via `Box<dyn Sampler>`.
#[test]
fn sampler_trait_object_safe() {
    let s: Box<dyn Sampler> = Box::new(SequentialSampler::new(5));
    assert_eq!(s.len(), 5);
    let idx = s.indices(0);
    assert_eq!(idx, vec![0, 1, 2, 3, 4]);
}
