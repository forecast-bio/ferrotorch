// CL-332: Vision Transforms & Augmentation — shared RNG utilities
//
// This module provides a seedable PRNG for vision augmentation transforms.
// It uses the same splitmix64 algorithm as ferrotorch-data's RNG to ensure
// consistent statistical properties. The state is separate so that vision
// transforms have independent reproducibility control.

use std::sync::atomic::{AtomicU64, Ordering};

static VISION_SEED: AtomicU64 = AtomicU64::new(42);
static VISION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Set the random seed for vision augmentation transforms.
///
/// Resets the internal counter so that subsequent random operations produce
/// the same sequence as a fresh start with this seed.
pub fn vision_manual_seed(seed: u64) {
    VISION_SEED.store(seed, Ordering::SeqCst);
    VISION_COUNTER.store(0, Ordering::SeqCst);
}

/// Generate a random `f64` in [0, 1) using a seedable splitmix64 PRNG.
///
/// Each call atomically increments a global counter, ensuring unique outputs
/// across threads. Use [`vision_manual_seed`] to reset the sequence for
/// reproducibility.
pub(crate) fn random_f64() -> f64 {
    let seed = VISION_SEED.load(Ordering::Relaxed);
    let counter = VISION_COUNTER.fetch_add(1, Ordering::Relaxed);
    // splitmix64 — good statistical properties for a counter-based PRNG.
    let mut state = seed.wrapping_add(counter.wrapping_mul(0x9E3779B97F4A7C15));
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D049BB133111EB);
    state = state ^ (state >> 31);
    (state as f64) / (u64::MAX as f64)
}
