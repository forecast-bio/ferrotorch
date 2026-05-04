//! CUDA RNG state management with Philox 4x32-10 counter-based generator.
//!
//! Provides deterministic, parallelizable random number generation for GPU
//! operations. The Philox algorithm is the same one used by CUDA's cuRAND
//! library: it maps a (counter, key) pair through 10 rounds of bijective
//! mixing to produce 4 uniform `u32` values per invocation.
//!
//! # Key types
//!
//! - [`PhiloxGenerator`] — stateful generator that tracks counter/offset
//! - [`PhiloxState`] — serializable snapshot for checkpoint save/restore
//! - [`CudaRngManager`] — per-device generator registry (one per GPU)
//! - [`cuda_rng_manager`] — global singleton accessor
//!
//! # GPU kernels
//!
//! Two PTX kernels generate random numbers directly on device without
//! CPU-to-GPU transfer:
//!
//! - `philox_uniform_kernel` — fills a buffer with uniform f32 in [0, 1)
//! - `philox_normal_kernel` — fills with standard normal f32 (Box-Muller)
//!
//! # Fork/join for data parallelism
//!
//! [`fork_rng`] and [`join_rng`] snapshot and restore RNG states across
//! multiple devices, ensuring each DDP rank gets independent RNG streams.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;

#[cfg(feature = "cuda")]
use crate::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::alloc_zeros_f32;

// ---------------------------------------------------------------------------
// Philox 4x32-10 constants
// ---------------------------------------------------------------------------

/// Philox multiplier constants (from the original Salmon et al. paper).
const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;

/// Philox Weyl sequence constants for key advancement.
const PHILOX_W0: u32 = 0x9E3779B9; // golden ratio
const PHILOX_W1: u32 = 0xBB67AE85; // sqrt(3) - 1

// ---------------------------------------------------------------------------
// PhiloxState — serializable snapshot
// ---------------------------------------------------------------------------

/// Serializable snapshot of a [`PhiloxGenerator`]'s state.
///
/// Used for checkpoint save/restore and fork/join in data parallelism.
///
/// # Construction
///
/// Use [`PhiloxState::new`] when starting from a fresh `(counter, seed)`
/// pair (offset starts at zero). Use [`PhiloxState::from_parts`] when
/// reconstructing from a checkpoint that captured a non-zero offset; that
/// constructor validates the offset is in the legal range `0..4`.
///
/// `counter` and `seed` remain public fields because they are legitimate
/// snapshot values that callers commonly read (and may compare). `offset`
/// is `pub(crate)` because external code can put it in an invalid state:
/// values `>= 4` produce a generator state the algorithm cannot represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PhiloxState {
    /// Counter value — incremented for each 4-tuple generated.
    pub counter: u64,
    /// Key/seed — set by the user via `manual_seed`.
    pub seed: u64,
    /// Offset into the current 4-tuple (0..4). Tracks how many of the
    /// 4 values from the last Philox round have been consumed.
    pub(crate) offset: u64,
}

impl PhiloxState {
    /// Create a new snapshot starting at counter `counter` with seed `seed`
    /// and a zero offset (no values consumed from the current 4-tuple).
    #[must_use]
    pub fn new(counter: u64, seed: u64) -> Self {
        Self {
            counter,
            seed,
            offset: 0,
        }
    }

    /// Reconstruct a snapshot from raw parts, validating the offset.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidState`] if `offset >= 4`. The Philox
    /// 4x32-10 algorithm produces 4 `u32` values per counter step, so the
    /// offset cursor must be in `0..4`.
    pub fn from_parts(counter: u64, seed: u64, offset: u64) -> GpuResult<Self> {
        if offset >= 4 {
            return Err(GpuError::InvalidState {
                message: format!("invalid Philox offset {offset}; must be < 4"),
            });
        }
        Ok(Self {
            counter,
            seed,
            offset,
        })
    }

    /// Read the offset cursor (`0..4`).
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

// ---------------------------------------------------------------------------
// PhiloxGenerator
// ---------------------------------------------------------------------------

/// Philox 4x32-10 counter-based random number generator.
///
/// This is a CBRNG (counter-based RNG): given a counter and key, it
/// deterministically produces 4 uniform `u32` values. The counter is
/// incremented after each group of 4 values is consumed, and the offset
/// tracks which of the 4 values in the current group has been consumed.
///
/// The algorithm is:
/// 1. Split the 64-bit counter into two 32-bit halves (counter_lo, counter_hi)
/// 2. Split the 64-bit seed/key into two 32-bit halves (key_lo, key_hi)
/// 3. Run 10 rounds of Philox mixing (multiply + xor + key advance)
/// 4. Output the 4 mixed 32-bit values
pub struct PhiloxGenerator {
    /// Counter — incremented for each group of 4 random numbers generated.
    counter: u64,
    /// Key/seed — set by the user.
    seed: u64,
    /// Offset into the current 4-tuple (0..4).
    offset: u64,
    /// Cached output from the last Philox round. When offset is 0, this is
    /// invalid and needs to be regenerated.
    cached: [u32; 4],
}

impl PhiloxGenerator {
    /// Create a new Philox generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            counter: 0,
            seed,
            offset: 0,
            cached: [0; 4],
        }
    }

    /// Set the seed, resetting the counter and offset.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.counter = 0;
        self.offset = 0;
        self.cached = [0; 4];
    }

    /// Snapshot the generator state for checkpoint save.
    pub fn get_state(&self) -> PhiloxState {
        PhiloxState {
            counter: self.counter,
            seed: self.seed,
            offset: self.offset,
        }
    }

    /// Restore the generator from a previously saved state.
    pub fn set_state(&mut self, state: PhiloxState) {
        self.seed = state.seed;
        self.counter = state.counter;
        self.offset = state.offset;
        self.cached = [0; 4];
        // If offset is non-zero, we need to regenerate the cached tuple
        // so that subsequent next_u32() calls return the correct values.
        if self.offset > 0 {
            self.cached = philox_4x32_10(self.counter, self.seed);
        }
    }

    /// Advance the generator by `n_counters` counter steps, resetting the
    /// offset and cached values. Used when a GPU kernel consumes random
    /// numbers directly and we need to keep the CPU-side state in sync.
    pub fn advance(&mut self, n_counters: u64) {
        self.counter += n_counters;
        self.offset = 0;
        self.cached = [0; 4];
    }

    /// Generate the next uniform `u32` value.
    pub fn next_u32(&mut self) -> u32 {
        if self.offset == 0 {
            self.cached = philox_4x32_10(self.counter, self.seed);
        }
        let val = self.cached[self.offset as usize];
        self.offset += 1;
        if self.offset >= 4 {
            self.offset = 0;
            self.counter += 1;
        }
        val
    }

    /// Generate a uniform f32 value in [0, 1).
    ///
    /// Uses the standard conversion: `(u32 >> 8) * 2^-24`, which produces
    /// all representable floats in [0, 1) with uniform probability.
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32();
        // Use the upper 24 bits for the mantissa (f32 has 23-bit mantissa + 1 implicit).
        // This gives 2^24 equally spaced values in [0, 1).
        (bits >> 8) as f32 * (1.0 / 16777216.0) // 2^-24
    }

    /// Generate `n` uniform f32 values in [0, 1).
    pub fn generate_uniform(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.next_f32());
        }
        out
    }

    /// Generate `n` standard normal f32 values using the Box-Muller transform.
    ///
    /// Generates pairs of normal values from pairs of uniform values:
    ///   z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
    ///   z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
    ///
    /// If `n` is odd, the last unpaired value is still generated correctly
    /// (we just discard the second value of the final pair).
    pub fn generate_normal(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let two_pi = 2.0 * std::f32::consts::PI;

        while out.len() < n {
            // Generate u1 in (0, 1] to avoid ln(0).
            let mut u1 = self.next_f32();
            while u1 == 0.0 {
                u1 = self.next_f32();
            }
            let u2 = self.next_f32();

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = two_pi * u2;

            out.push(r * theta.cos());
            if out.len() < n {
                out.push(r * theta.sin());
            }
        }

        out
    }
}

impl std::fmt::Debug for PhiloxGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhiloxGenerator")
            .field("counter", &self.counter)
            .field("seed", &self.seed)
            .field("offset", &self.offset)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Philox 4x32-10 core algorithm
// ---------------------------------------------------------------------------

/// Single Philox round: multiply-xor-swap.
///
/// Takes 4 state values and 2 key values, produces 4 mixed state values.
#[inline]
fn philox_round(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> (u32, u32, u32, u32) {
    // hi/lo decomposition of 32x32 -> 64 multiply
    let prod0 = (PHILOX_M0 as u64) * (c0 as u64);
    let hi0 = (prod0 >> 32) as u32;
    let lo0 = prod0 as u32;

    let prod1 = (PHILOX_M1 as u64) * (c2 as u64);
    let hi1 = (prod1 >> 32) as u32;
    let lo1 = prod1 as u32;

    // Feistel-like swap and xor with key
    let new_c0 = hi1 ^ c1 ^ k0;
    let new_c1 = lo1;
    let new_c2 = hi0 ^ c3 ^ k1;
    let new_c3 = lo0;

    (new_c0, new_c1, new_c2, new_c3)
}

/// Philox 4x32-10: 10 rounds of mixing a (counter, key) pair.
///
/// Takes a 64-bit counter and 64-bit key, returns 4 uniform u32 values.
fn philox_4x32_10(counter: u64, key: u64) -> [u32; 4] {
    // Split counter and key into 32-bit halves
    let mut c0 = counter as u32;
    let mut c1 = (counter >> 32) as u32;
    let mut c2 = 0u32; // Second counter half (we use a single 64-bit counter)
    let mut c3 = 0u32;

    let mut k0 = key as u32;
    let mut k1 = (key >> 32) as u32;

    // 10 rounds of mixing
    // Round 1
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 2
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 3
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 4
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 5
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 6
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 7
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 8
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 9
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 10 (final — no key advance needed)
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);

    [c0, c1, c2, c3]
}

// ---------------------------------------------------------------------------
// CudaRngManager — per-device generator registry
// ---------------------------------------------------------------------------

/// Per-device RNG state manager.
///
/// Maintains one [`PhiloxGenerator`] per GPU device, initialized lazily with
/// a default seed. The manager is accessed through the global singleton
/// [`cuda_rng_manager`].
pub struct CudaRngManager {
    /// One generator per GPU device, keyed by device ordinal.
    generators: HashMap<usize, PhiloxGenerator>,
    /// Default seed used when a device's generator is first accessed.
    default_seed: u64,
}

impl CudaRngManager {
    /// Create a new manager with the given default seed.
    fn new(default_seed: u64) -> Self {
        Self {
            generators: HashMap::new(),
            default_seed,
        }
    }

    /// Set the seed for a specific device, resetting its counter and offset.
    pub fn manual_seed(&mut self, device: usize, seed: u64) {
        let rng_gen = self
            .generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(seed));
        rng_gen.set_seed(seed);
    }

    /// Set the seed for all currently-initialized devices.
    ///
    /// Also updates the default seed so that any future devices will use it.
    pub fn manual_seed_all(&mut self, seed: u64) {
        self.default_seed = seed;
        for rng_gen in self.generators.values_mut() {
            rng_gen.set_seed(seed);
        }
    }

    /// Get the RNG state for a specific device.
    ///
    /// Initializes the generator with the default seed if not already present.
    pub fn get_rng_state(&mut self, device: usize) -> PhiloxState {
        let default_seed = self.default_seed;
        self.generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(default_seed))
            .get_state()
    }

    /// Set the RNG state for a specific device from a snapshot.
    pub fn set_rng_state(&mut self, device: usize, state: PhiloxState) {
        let rng_gen = self
            .generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(state.seed));
        rng_gen.set_state(state);
    }

    /// Get a mutable reference to the generator for a specific device.
    ///
    /// Initializes the generator with the default seed if not already present.
    pub fn generator(&mut self, device: usize) -> &mut PhiloxGenerator {
        let default_seed = self.default_seed;
        self.generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(default_seed))
    }
}

impl std::fmt::Debug for CudaRngManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRngManager")
            .field("num_devices", &self.generators.len())
            .field("default_seed", &self.default_seed)
            .finish()
    }
}

/// Global singleton for the CUDA RNG manager.
///
/// The default seed is 0 (matching PyTorch's default). Use `manual_seed`
/// or `manual_seed_all` to set deterministic seeds before training.
static CUDA_RNG_MANAGER: LazyLock<Mutex<CudaRngManager>> =
    LazyLock::new(|| Mutex::new(CudaRngManager::new(0)));

/// Access the global CUDA RNG manager.
///
/// # Example
///
/// ```rust,no_run
/// use ferrotorch_gpu::rng::cuda_rng_manager;
///
/// let mut mgr = cuda_rng_manager().lock().unwrap();
/// mgr.manual_seed(0, 42);
/// let val = mgr.generator(0).next_f32();
/// ```
pub fn cuda_rng_manager() -> &'static Mutex<CudaRngManager> {
    &CUDA_RNG_MANAGER
}

// ---------------------------------------------------------------------------
// Fork/join for data parallelism
// ---------------------------------------------------------------------------

/// Snapshot the RNG state of multiple devices.
///
/// Used by DDP (distributed data parallel) to save each rank's RNG state
/// before a training step, ensuring reproducibility when resuming.
///
/// # Arguments
///
/// * `devices` — slice of device ordinals to snapshot
///
/// # Returns
///
/// A vector of `PhiloxState` in the same order as `devices`.
pub fn fork_rng(devices: &[usize]) -> Vec<PhiloxState> {
    let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
    devices.iter().map(|&d| mgr.get_rng_state(d)).collect()
}

/// Restore RNG states for multiple devices from a previous [`fork_rng`] call.
///
/// # Arguments
///
/// * `devices` — slice of device ordinals (must match the `fork_rng` call)
/// * `states` — vector of `PhiloxState` to restore, in device order
///
/// # Panics
///
/// Panics if `devices.len() != states.len()`.
pub fn join_rng(devices: &[usize], states: Vec<PhiloxState>) {
    assert_eq!(
        devices.len(),
        states.len(),
        "join_rng: devices.len() ({}) != states.len() ({})",
        devices.len(),
        states.len()
    );
    let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
    for (&device, state) in devices.iter().zip(states) {
        mgr.set_rng_state(device, state);
    }
}

// ---------------------------------------------------------------------------
// PTX kernels for Philox RNG on GPU
// ---------------------------------------------------------------------------

/// PTX source for `philox_uniform_kernel`: fills buffer with uniform f32 in [0, 1).
///
/// Parameters:
///   out_ptr    — pointer to output f32 buffer
///   n          — number of elements
///   seed_lo    — lower 32 bits of the seed
///   seed_hi    — upper 32 bits of the seed
///   counter_lo — lower 32 bits of the starting counter
///   counter_hi — upper 32 bits of the starting counter
///
/// Each thread generates one f32 value. The counter for each thread is
/// `base_counter + thread_idx / 4`, and the sub-index within the 4-tuple
/// is `thread_idx % 4`.
#[cfg(feature = "cuda")]
pub(crate) const PHILOX_UNIFORM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry philox_uniform_kernel(
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 seed_lo,
    .param .u32 seed_hi,
    .param .u32 counter_lo,
    .param .u32 counter_hi
) {
    .reg .u32 %tid, %bid, %bdim, %gid, %n_reg;
    .reg .u32 %slo, %shi, %clo, %chi;
    .reg .u32 %group, %sub, %rem;
    .reg .u32 %c0, %c1, %c2, %c3, %k0, %k1;
    .reg .u32 %hi_val, %lo_val, %t0, %t1, %t2, %t3;
    .reg .u64 %prod, %out, %off;
    .reg .u32 %result, %shifted;
    .reg .f32 %fval, %scale;
    .reg .pred %p, %p_sub;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %slo, [seed_lo];
    ld.param.u32 %shi, [seed_hi];
    ld.param.u32 %clo, [counter_lo];
    ld.param.u32 %chi, [counter_hi];

    // Global thread index
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %gid, %n_reg;
    @%p bra DONE;

    // group = gid / 4, sub = gid % 4
    shr.u32 %group, %gid, 2;
    and.b32 %sub, %gid, 3;

    // counter = base_counter + group (64-bit add via carry)
    add.cc.u32 %c0, %clo, %group;
    addc.u32 %c1, %chi, 0;
    mov.u32 %c2, 0;
    mov.u32 %c3, 0;
    mov.u32 %k0, %slo;
    mov.u32 %k1, %shi;

    // === 10 rounds of Philox mixing ===
    // Round 1
    // prod0 = M0 * c0 -> hi0, lo0
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    // prod1 = M1 * c2 -> hi1, lo1
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    // new values
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    // key advance
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 2
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 3
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 4
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 5
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 6
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 7
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 8
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 9
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 10 (final)
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;

    // Select output based on sub-index (gid % 4)
    // sub == 0 -> c0, sub == 1 -> c1, sub == 2 -> c2, sub == 3 -> c3
    mov.u32 %result, %c0;
    setp.eq.u32 %p_sub, %sub, 1;
    @%p_sub mov.u32 %result, %c1;
    setp.eq.u32 %p_sub, %sub, 2;
    @%p_sub mov.u32 %result, %c2;
    setp.eq.u32 %p_sub, %sub, 3;
    @%p_sub mov.u32 %result, %c3;

    // Convert to f32 in [0, 1): (result >> 8) * 2^-24
    shr.u32 %shifted, %result, 8;
    cvt.rn.f32.u32 %fval, %shifted;
    mov.f32 %scale, 0f33800000;
    mul.f32 %fval, %fval, %scale;

    // Store
    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %fval;

DONE:
    ret;
}
";

/// PTX source for `philox_normal_kernel`: fills buffer with standard normal f32
/// values using Box-Muller transform on Philox-generated uniforms.
///
/// Each thread generates 2 normal values (or 1 if at the end). Threads are
/// dispatched for n/2 pairs. Thread `i` produces output[2*i] and output[2*i+1].
///
/// Parameters:
///   out_ptr    — pointer to output f32 buffer
///   n          — number of elements
///   seed_lo    — lower 32 bits of the seed
///   seed_hi    — upper 32 bits of the seed
///   counter_lo — lower 32 bits of the starting counter
///   counter_hi — upper 32 bits of the starting counter
#[cfg(feature = "cuda")]
pub(crate) const PHILOX_NORMAL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry philox_normal_kernel(
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 seed_lo,
    .param .u32 seed_hi,
    .param .u32 counter_lo,
    .param .u32 counter_hi
) {
    .reg .u32 %tid, %bid, %bdim, %gid, %n_reg;
    .reg .u32 %slo, %shi, %clo, %chi;
    .reg .u32 %c0, %c1, %c2, %c3, %k0, %k1;
    .reg .u32 %hi_val, %lo_val, %t0, %t1, %t2, %t3;
    .reg .u64 %prod, %out, %off;
    .reg .u32 %idx0, %idx1, %shifted0, %shifted1;
    .reg .f32 %u1, %u2, %r, %theta, %z0, %z1;
    .reg .f32 %scale, %two_pi, %neg2, %ln_u1;
    .reg .pred %p, %p2;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %slo, [seed_lo];
    ld.param.u32 %shi, [seed_hi];
    ld.param.u32 %clo, [counter_lo];
    ld.param.u32 %chi, [counter_hi];

    // Global thread index (each thread handles 2 output elements)
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %tid;

    // Each thread produces elements at idx0 = 2*gid and idx1 = 2*gid+1
    shl.b32 %idx0, %gid, 1;
    setp.ge.u32 %p, %idx0, %n_reg;
    @%p bra DONE;
    add.u32 %idx1, %idx0, 1;

    // Counter for this thread = base_counter + gid
    // We use c0, c1 from the Philox state; each thread gets a unique counter
    add.cc.u32 %c0, %clo, %gid;
    addc.u32 %c1, %chi, 0;
    mov.u32 %c2, 0;
    mov.u32 %c3, 0;
    mov.u32 %k0, %slo;
    mov.u32 %k1, %shi;

    // === 10 rounds of Philox mixing (same as uniform kernel) ===
    // Round 1
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 2
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 3
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 4
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 5
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 6
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 7
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 8
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 9
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;

    // Round 10 (final)
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;

    // Use c0/c1 as the two uniform u32 values for Box-Muller
    // u1 = (c0 >> 8) * 2^-24, ensure u1 > 0 by OR-ing in 1 if zero
    // u2 = (c1 >> 8) * 2^-24
    shr.u32 %shifted0, %c0, 8;
    // Ensure shifted0 > 0 to avoid log(0)
    setp.eq.u32 %p2, %shifted0, 0;
    @%p2 mov.u32 %shifted0, 1;
    cvt.rn.f32.u32 %u1, %shifted0;
    mov.f32 %scale, 0f33800000;
    mul.f32 %u1, %u1, %scale;

    shr.u32 %shifted1, %c1, 8;
    cvt.rn.f32.u32 %u2, %shifted1;
    mul.f32 %u2, %u2, %scale;

    // Box-Muller: z0 = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    //             z1 = sqrt(-2*ln(u1)) * sin(2*pi*u2)
    // ln(u1)
    lg2.approx.f32 %ln_u1, %u1;
    // Convert log2 to ln: ln(x) = log2(x) * ln(2) = log2(x) * 0.693147
    mul.f32 %ln_u1, %ln_u1, 0f3F317218;
    // -2 * ln(u1)
    mov.f32 %neg2, 0fC0000000;
    mul.f32 %r, %neg2, %ln_u1;
    // sqrt
    sqrt.approx.f32 %r, %r;
    // 2*pi*u2
    mov.f32 %two_pi, 0f40C90FDB;
    mul.f32 %theta, %two_pi, %u2;
    // cos and sin
    cos.approx.f32 %z0, %theta;
    mul.f32 %z0, %r, %z0;
    sin.approx.f32 %z1, %theta;
    mul.f32 %z1, %r, %z1;

    // Store z0 at output[idx0]
    cvt.u64.u32 %off, %idx0;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %z0;

    // Store z1 at output[idx1] (if idx1 < n)
    setp.ge.u32 %p2, %idx1, %n_reg;
    @%p2 bra DONE;
    cvt.u64.u32 %off, %idx1;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %z1;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// GPU kernel launch functions
// ---------------------------------------------------------------------------

/// Standard 1-D launch config for `n` elements.
#[cfg(feature = "cuda")]
fn rng_launch_cfg(n: usize) -> GpuResult<LaunchConfig> {
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "rng_kernel_launch",
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    const BLOCK: u32 = 256;
    let grid = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Fill a GPU buffer with uniform random f32 values in [0, 1) using the
/// Philox 4x32-10 algorithm.
///
/// The values are generated entirely on device — no CPU-to-GPU transfer.
/// The global RNG state for the device is advanced by `ceil(n/4)` counters.
///
/// # Arguments
///
/// * `n` — number of f32 values to generate
/// * `device` — the GPU device
///
/// # CPU fallback
///
/// If PTX compilation fails (architecture mismatch), falls back to generating
/// values on CPU and transferring to GPU.
#[cfg(feature = "cuda")]
pub fn gpu_philox_uniform(n: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if n == 0 {
        return alloc_zeros_f32(0, device);
    }

    // Get the current RNG state and advance it.
    let state = {
        let mut mgr = CUDA_RNG_MANAGER
            .lock()
            .map_err(|e| GpuError::InvalidState {
                message: format!("CUDA RNG manager mutex poisoned: {e}"),
            })?;
        let rng_gen = mgr.generator(device.ordinal());
        let state = rng_gen.get_state();
        // Advance the generator by ceil(n/4) counters (each counter produces 4 values)
        let counters_needed = n.div_ceil(4);
        rng_gen.advance(counters_needed as u64);
        state
    };

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        PHILOX_UNIFORM_PTX,
        "philox_uniform_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback: generate on CPU with the same state, then transfer.
            let mut rng_gen = PhiloxGenerator::new(state.seed);
            rng_gen.set_state(state);
            let vals = rng_gen.generate_uniform(n);
            return crate::transfer::cpu_to_gpu(&vals, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = rng_launch_cfg(n)?;
    let n_u32 = n as u32;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Fill a GPU buffer with standard normal f32 values using the Philox 4x32-10
/// algorithm and Box-Muller transform.
///
/// # Arguments
///
/// * `n` — number of f32 values to generate
/// * `device` — the GPU device
#[cfg(feature = "cuda")]
pub fn gpu_philox_normal(n: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if n == 0 {
        return alloc_zeros_f32(0, device);
    }

    // Get the current RNG state and advance it.
    // Each thread consumes one Philox 4-tuple (using 2 of 4 values for Box-Muller),
    // and each thread produces 2 output values, so we need ceil(n/2) counters.
    let state = {
        let mut mgr = CUDA_RNG_MANAGER
            .lock()
            .map_err(|e| GpuError::InvalidState {
                message: format!("CUDA RNG manager mutex poisoned: {e}"),
            })?;
        let rng_gen = mgr.generator(device.ordinal());
        let state = rng_gen.get_state();
        let counters_needed = n.div_ceil(2);
        rng_gen.advance(counters_needed as u64);
        state
    };

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        PHILOX_NORMAL_PTX,
        "philox_normal_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let mut rng_gen = PhiloxGenerator::new(state.seed);
            rng_gen.set_state(state);
            let vals = rng_gen.generate_normal(n);
            return crate::transfer::cpu_to_gpu(&vals, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    // Each thread handles 2 elements, so we need ceil(n/2) threads.
    let num_threads = n.div_ceil(2);
    let cfg = rng_launch_cfg(num_threads)?;
    let n_u32 = n as u32;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_philox_uniform(_n: usize) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_philox_normal(_n: usize) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Philox core algorithm tests
    // -----------------------------------------------------------------------

    #[test]
    fn philox_deterministic() {
        // Same counter + key must produce the same output.
        let a = philox_4x32_10(0, 0);
        let b = philox_4x32_10(0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn philox_different_counters_differ() {
        let a = philox_4x32_10(0, 42);
        let b = philox_4x32_10(1, 42);
        assert_ne!(a, b);
    }

    #[test]
    fn philox_different_keys_differ() {
        let a = philox_4x32_10(0, 0);
        let b = philox_4x32_10(0, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn philox_outputs_nonzero() {
        // With high probability, at least some outputs are non-zero.
        let out = philox_4x32_10(1, 1);
        assert!(
            out.iter().any(|&x| x != 0),
            "all Philox outputs are zero — very unlikely"
        );
    }

    #[test]
    fn philox_avalanche_effect() {
        // Changing a single bit in the counter should change many output bits.
        let a = philox_4x32_10(0, 42);
        let b = philox_4x32_10(1, 42); // counter differs by 1
        let mut total_differing_bits = 0u32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            total_differing_bits += (x ^ y).count_ones();
        }
        // With 128 output bits, roughly half should differ (64 +/- some).
        // We accept anything in [20, 108] as a sanity check.
        assert!(
            total_differing_bits > 20 && total_differing_bits < 108,
            "poor avalanche: {total_differing_bits} bits differ out of 128"
        );
    }

    // -----------------------------------------------------------------------
    // PhiloxGenerator tests
    // -----------------------------------------------------------------------

    #[test]
    fn generator_deterministic_with_same_seed() {
        let mut g1 = PhiloxGenerator::new(42);
        let mut g2 = PhiloxGenerator::new(42);

        let vals1: Vec<u32> = (0..100).map(|_| g1.next_u32()).collect();
        let vals2: Vec<u32> = (0..100).map(|_| g2.next_u32()).collect();
        assert_eq!(vals1, vals2);
    }

    #[test]
    fn generator_different_seeds_differ() {
        let mut g1 = PhiloxGenerator::new(42);
        let mut g2 = PhiloxGenerator::new(43);

        let vals1: Vec<u32> = (0..10).map(|_| g1.next_u32()).collect();
        let vals2: Vec<u32> = (0..10).map(|_| g2.next_u32()).collect();
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn generator_produces_4_values_per_counter() {
        let mut rng_gen = PhiloxGenerator::new(12345);

        // First 4 values should come from counter 0
        let _ = rng_gen.next_u32();
        assert_eq!(rng_gen.counter, 0);
        assert_eq!(rng_gen.offset, 1);

        let _ = rng_gen.next_u32();
        let _ = rng_gen.next_u32();
        let _ = rng_gen.next_u32();
        // After 4 values, counter should advance to 1
        assert_eq!(rng_gen.counter, 1);
        assert_eq!(rng_gen.offset, 0);
    }

    #[test]
    fn generator_set_seed_resets_state() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let first_val = rng_gen.next_u32();

        // Advance a bunch
        for _ in 0..100 {
            rng_gen.next_u32();
        }

        // Reset
        rng_gen.set_seed(42);
        let after_reset = rng_gen.next_u32();
        assert_eq!(first_val, after_reset);
    }

    #[test]
    fn generator_state_save_restore() {
        let mut rng_gen = PhiloxGenerator::new(42);

        // Advance partway
        for _ in 0..7 {
            rng_gen.next_u32();
        }

        let state = rng_gen.get_state();

        // Generate 20 more values
        let vals1: Vec<u32> = (0..20).map(|_| rng_gen.next_u32()).collect();

        // Restore and generate the same 20
        rng_gen.set_state(state);
        let vals2: Vec<u32> = (0..20).map(|_| rng_gen.next_u32()).collect();

        assert_eq!(vals1, vals2);
    }

    #[test]
    fn generator_state_save_restore_at_offset() {
        // Save state when offset is non-zero (mid-tuple)
        let mut rng_gen = PhiloxGenerator::new(99);

        // Consume 2 of 4 values from counter 0
        rng_gen.next_u32();
        rng_gen.next_u32();
        assert_eq!(rng_gen.offset, 2);

        let state = rng_gen.get_state();

        let vals1: Vec<u32> = (0..10).map(|_| rng_gen.next_u32()).collect();

        rng_gen.set_state(state);
        let vals2: Vec<u32> = (0..10).map(|_| rng_gen.next_u32()).collect();

        assert_eq!(vals1, vals2);
    }

    // -----------------------------------------------------------------------
    // next_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn f32_in_unit_interval() {
        let mut rng_gen = PhiloxGenerator::new(42);
        for _ in 0..10000 {
            let v = rng_gen.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 value {v} outside [0, 1)");
        }
    }

    #[test]
    fn f32_not_all_same() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals: Vec<f32> = (0..100).map(|_| rng_gen.next_f32()).collect();
        let first = vals[0];
        assert!(
            vals.iter().any(|&v| v != first),
            "all f32 values are the same: {first}"
        );
    }

    // -----------------------------------------------------------------------
    // generate_uniform tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_uniform_correct_length() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_uniform(1000);
        assert_eq!(vals.len(), 1000);
    }

    #[test]
    fn generate_uniform_values_in_range() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_uniform(10000);
        for &v in &vals {
            assert!((0.0..1.0).contains(&v), "uniform value {v} outside [0, 1)");
        }
    }

    #[test]
    fn generate_uniform_reasonable_mean() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_uniform(100_000);
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "uniform mean = {mean}, expected ~0.5"
        );
    }

    // -----------------------------------------------------------------------
    // generate_normal tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_normal_correct_length() {
        let mut rng_gen = PhiloxGenerator::new(42);
        assert_eq!(rng_gen.generate_normal(1000).len(), 1000);
        // Odd count
        assert_eq!(rng_gen.generate_normal(999).len(), 999);
    }

    #[test]
    fn generate_normal_reasonable_mean_and_std() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_normal(100_000);

        let n = vals.len() as f64;
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt();

        assert!(mean.abs() < 0.02, "normal mean = {mean}, expected ~0.0");
        assert!(
            (std - 1.0).abs() < 0.02,
            "normal std = {std}, expected ~1.0"
        );
    }

    #[test]
    fn generate_normal_no_nan_or_inf() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_normal(100_000);
        for &v in &vals {
            assert!(v.is_finite(), "normal value is not finite: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // CudaRngManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn manager_initializes_device_on_access() {
        let mut mgr = CudaRngManager::new(42);
        let state = mgr.get_rng_state(0);
        assert_eq!(state.seed, 42);
        assert_eq!(state.counter, 0);
        assert_eq!(state.offset, 0);
    }

    #[test]
    fn manager_manual_seed() {
        let mut mgr = CudaRngManager::new(0);
        mgr.manual_seed(0, 12345);

        let rng_gen = mgr.generator(0);
        assert_eq!(rng_gen.seed, 12345);
        assert_eq!(rng_gen.counter, 0);
    }

    #[test]
    fn manager_manual_seed_all() {
        let mut mgr = CudaRngManager::new(0);
        // Initialize a few devices
        mgr.manual_seed(0, 100);
        mgr.manual_seed(1, 200);
        mgr.manual_seed(2, 300);

        // Now set all to the same seed
        mgr.manual_seed_all(42);

        assert_eq!(mgr.get_rng_state(0).seed, 42);
        assert_eq!(mgr.get_rng_state(1).seed, 42);
        assert_eq!(mgr.get_rng_state(2).seed, 42);

        // Newly-created device should also get the new default
        assert_eq!(mgr.get_rng_state(3).seed, 42);
    }

    #[test]
    fn manager_set_rng_state() {
        let mut mgr = CudaRngManager::new(0);
        let custom_state = PhiloxState::from_parts(100, 999, 2).expect("offset 2 is in 0..4");
        mgr.set_rng_state(0, custom_state);

        let state = mgr.get_rng_state(0);
        assert_eq!(state, custom_state);
    }

    #[test]
    fn manager_independent_devices() {
        let mut mgr = CudaRngManager::new(0);
        mgr.manual_seed(0, 42);
        mgr.manual_seed(1, 43);

        let v0 = mgr.generator(0).next_u32();
        let v1 = mgr.generator(1).next_u32();
        // Different seeds should produce different values
        assert_ne!(v0, v1);
    }

    // -----------------------------------------------------------------------
    // Fork/join tests
    // -----------------------------------------------------------------------

    #[test]
    fn fork_join_roundtrip() {
        // Set up known state via the global manager
        {
            let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
            mgr.manual_seed(10, 42);
            mgr.manual_seed(11, 43);
        }

        let devices = &[10, 11];
        let states = fork_rng(devices);

        // Advance the generators
        {
            let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
            for _ in 0..100 {
                mgr.generator(10).next_u32();
                mgr.generator(11).next_u32();
            }
        }

        // Restore
        join_rng(devices, states);

        // Verify restoration
        {
            let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
            assert_eq!(mgr.get_rng_state(10).counter, 0);
            assert_eq!(mgr.get_rng_state(10).seed, 42);
            assert_eq!(mgr.get_rng_state(11).counter, 0);
            assert_eq!(mgr.get_rng_state(11).seed, 43);
        }
    }

    #[test]
    #[should_panic(expected = "devices.len()")]
    fn fork_join_length_mismatch_panics() {
        let states = vec![PhiloxState::new(0, 0)];
        join_rng(&[0, 1], states);
    }

    // -----------------------------------------------------------------------
    // Global singleton test
    // -----------------------------------------------------------------------

    #[test]
    fn global_singleton_accessible() {
        let mgr = cuda_rng_manager();
        let mut guard = mgr.lock().unwrap();
        guard.manual_seed(99, 12345);
        let state = guard.get_rng_state(99);
        assert_eq!(state.seed, 12345);
    }
}
