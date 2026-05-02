//! Multi-dispatch key system for composable tensor backends. CL-397.
//!
//! Mirrors PyTorch's `DispatchKey` / `DispatchKeySet` / `Dispatcher`
//! architecture: every tensor carries a set of active dispatch keys
//! (e.g. `Autograd`, `Quantized`, `Sparse`, `CPU`, `CUDA`), and when
//! an op is invoked the dispatcher picks the kernel registered for
//! the **highest-priority** active key.
//!
//! This enables layered semantics without hard-coding each
//! combination in every op:
//!
//! - `Autograd` kernels record a backward node and forward to the
//!   next layer.
//! - `Quantized` kernels dequantize, forward, and re-quantize.
//! - `Sparse` kernels call the sparse backend when the tensor is a
//!   sparse view.
//! - `CPU` / `CUDA` are the terminal "backend" keys that actually
//!   run the math.
//!
//! The dispatcher walks the set from highest to lowest priority,
//! picks the first registered kernel, and runs it. The kernel can
//! mask its own key off and call the dispatcher again to delegate
//! to the next layer ("redispatch" in PyTorch terminology).
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_core::dispatch::{DispatchKey, DispatchKeySet, Dispatcher};
//!
//! let mut dispatcher = Dispatcher::<f32>::new();
//!
//! // Register a CPU kernel for the "add" op.
//! dispatcher.register("add", DispatchKey::Cpu, |inputs, _keyset, _disp| {
//!     // Actually do the addition...
//!     Ok(inputs[0].clone())
//! });
//!
//! // Layer an autograd kernel on top that records a backward node
//! // and redispatches with Autograd masked off.
//! dispatcher.register("add", DispatchKey::Autograd, |inputs, keyset, disp| {
//!     // ... record backward ...
//!     let remaining = keyset.remove(DispatchKey::Autograd);
//!     disp.call("add", inputs, remaining)
//! });
//!
//! // Call the op with a keyset that has both Autograd and CPU set.
//! // The dispatcher picks Autograd first (higher priority), which
//! // then redispatches to Cpu.
//! let keyset = DispatchKeySet::from([DispatchKey::Autograd, DispatchKey::Cpu]);
//! let result = dispatcher.call("add", &[tensor], keyset).unwrap();
//! ```

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

use std::collections::HashMap;

/// One of the 16 possible dispatch keys, ordered from lowest to
/// highest priority. The `u8` repr matches the bit position in
/// [`DispatchKeySet`]'s internal `u16` bitmask, so the priority
/// ordering is both the enum declaration order and the numeric
/// order of the discriminants.
///
/// Keys are resolved highest-priority-first: the dispatcher walks
/// from the largest discriminant down and picks the first key that
/// has a registered kernel for the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DispatchKey {
    /// Backend: CPU — terminal key for CPU kernels.
    Cpu = 0,
    /// Backend: CUDA — terminal key for CUDA kernels.
    Cuda = 1,
    /// Backend: Meta device — shape-only dry runs, no data.
    Meta = 2,
    /// Tensor contains sparse data. Sparse kernels intercept ops
    /// and either call a sparse-specific backend or densify and
    /// redispatch.
    Sparse = 3,
    /// Tensor contains quantized values. Quantized kernels
    /// dequantize, redispatch, and requantize (for ops without
    /// native quantized kernels).
    Quantized = 4,
    /// Tensor is a nested/jagged tensor. Nested kernels iterate
    /// per-component and redispatch to the backend.
    Nested = 5,
    /// Auto-mixed-precision: cast inputs to the autocast dtype
    /// before redispatching. Higher priority than Quantized so
    /// AMP happens before quantization layering.
    Autocast = 6,
    /// Autograd: record a backward node and redispatch with
    /// Autograd masked off. Highest-priority non-profiling key so
    /// the backward graph sees the post-dispatch view of each op.
    Autograd = 7,
    /// Vmap (batched tensor): intercept ops and apply them over
    /// the batch dimension. Stacks above Autograd so batched
    /// forwards still see autograd semantics.
    Vmap = 8,
    /// Profiler: record an entry in the active profiler before
    /// redispatching. Sits above Vmap so the profiler sees the
    /// outer call exactly once regardless of batching.
    Profiler = 9,
    /// Tracer: emit an IR node into the active JIT trace.
    /// Highest priority so tracing happens before any other
    /// layering transforms the op.
    Tracer = 10,
}

impl DispatchKey {
    /// The numeric priority of this key. Larger = higher priority.
    #[inline]
    pub fn priority(self) -> u8 {
        self as u8
    }

    /// All 11 defined keys, in priority order (lowest to highest).
    /// Useful for iterating the full set.
    pub const ALL: [DispatchKey; 11] = [
        DispatchKey::Cpu,
        DispatchKey::Cuda,
        DispatchKey::Meta,
        DispatchKey::Sparse,
        DispatchKey::Quantized,
        DispatchKey::Nested,
        DispatchKey::Autocast,
        DispatchKey::Autograd,
        DispatchKey::Vmap,
        DispatchKey::Profiler,
        DispatchKey::Tracer,
    ];
}

/// A set of active [`DispatchKey`]s, stored as a `u16` bitmask for
/// constant-time membership testing and iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DispatchKeySet {
    bits: u16,
}

impl DispatchKeySet {
    /// An empty set.
    #[inline]
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    /// A set containing every defined key.
    pub fn all() -> Self {
        let mut set = Self::empty();
        for &k in &DispatchKey::ALL {
            set = set.insert(k);
        }
        set
    }

    /// Construct a set from an iterable of keys. Convenience wrapper over
    /// the [`FromIterator`] impl below for callers that don't want to chain
    /// `.into_iter().collect()`.
    pub fn from_keys<I: IntoIterator<Item = DispatchKey>>(keys: I) -> Self {
        keys.into_iter().collect()
    }

    /// Returns true if `key` is in this set.
    #[inline]
    pub fn contains(self, key: DispatchKey) -> bool {
        (self.bits >> key.priority()) & 1 != 0
    }

    /// Returns a new set with `key` added.
    #[inline]
    #[must_use]
    pub fn insert(self, key: DispatchKey) -> Self {
        Self {
            bits: self.bits | (1 << key.priority()),
        }
    }

    /// Returns a new set with `key` removed.
    #[inline]
    #[must_use]
    pub fn remove(self, key: DispatchKey) -> Self {
        Self {
            bits: self.bits & !(1 << key.priority()),
        }
    }

    /// Union of two sets.
    #[inline]
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Intersection of two sets.
    #[inline]
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    /// Returns true if this set has no keys.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Number of keys in this set.
    #[inline]
    pub fn len(self) -> usize {
        self.bits.count_ones() as usize
    }

    /// Highest-priority key in this set, or `None` if empty. This
    /// is the "next" key the dispatcher will resolve.
    pub fn highest(self) -> Option<DispatchKey> {
        if self.bits == 0 {
            return None;
        }
        // Walk keys from highest to lowest discriminant and return
        // the first one present.
        DispatchKey::ALL
            .iter()
            .rev()
            .find(|&&k| self.contains(k))
            .copied()
    }

    /// Returns an iterator over all keys in the set, in
    /// **descending** priority order (highest first).
    pub fn iter_desc(self) -> impl Iterator<Item = DispatchKey> {
        let mut bits = self.bits;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            // Find the highest set bit.
            let top = 15 - bits.leading_zeros() as u8;
            bits &= !(1 << top);
            // Map bit position back to a DispatchKey if valid.
            DispatchKey::ALL
                .iter()
                .find(|k| k.priority() == top)
                .copied()
        })
    }
}

impl Default for DispatchKeySet {
    fn default() -> Self {
        Self::empty()
    }
}

impl FromIterator<DispatchKey> for DispatchKeySet {
    fn from_iter<I: IntoIterator<Item = DispatchKey>>(keys: I) -> Self {
        let mut set = Self::empty();
        for k in keys {
            set = set.insert(k);
        }
        set
    }
}

impl<const N: usize> From<[DispatchKey; N]> for DispatchKeySet {
    fn from(arr: [DispatchKey; N]) -> Self {
        Self::from_keys(arr)
    }
}

// ---------------------------------------------------------------------------
// Kernel type and Dispatcher
// ---------------------------------------------------------------------------

/// A dispatched kernel: takes the op's input tensors, the
/// currently-active keyset (after all higher-priority keys have
/// been resolved), and a reference to the dispatcher so the kernel
/// can redispatch to a lower-priority key.
///
/// Kernels return a single output tensor. Ops with multiple
/// outputs are not yet supported by this dispatcher — they'd need
/// a separate `KernelMulti` variant.
pub type Kernel<T> = Box<
    dyn Fn(&[Tensor<T>], DispatchKeySet, &Dispatcher<T>) -> FerrotorchResult<Tensor<T>>
        + Send
        + Sync,
>;

/// A kernel registration table keyed by `(op_name, dispatch_key)`.
/// Looking up a kernel is a single HashMap probe.
///
/// `T` is the scalar dtype the dispatcher operates on (f32 / f64).
/// Different dispatchers are typically held per-dtype.
pub struct Dispatcher<T: Float> {
    kernels: HashMap<(String, DispatchKey), Kernel<T>>,
}

impl<T: Float> Dispatcher<T> {
    /// Create an empty dispatcher with no registered kernels.
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Register a kernel for `(op_name, key)`. Overwrites any
    /// existing registration for the same pair.
    pub fn register<F>(&mut self, op_name: impl Into<String>, key: DispatchKey, kernel: F)
    where
        F: Fn(&[Tensor<T>], DispatchKeySet, &Dispatcher<T>) -> FerrotorchResult<Tensor<T>>
            + Send
            + Sync
            + 'static,
    {
        self.kernels.insert((op_name.into(), key), Box::new(kernel));
    }

    /// Returns true if a kernel is registered for `(op_name, key)`.
    pub fn has_kernel(&self, op_name: &str, key: DispatchKey) -> bool {
        self.kernels.contains_key(&(op_name.to_string(), key))
    }

    /// Number of registered kernels.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Call `op_name` with `inputs` and the given active keyset.
    /// Walks the keyset in descending priority order, picks the
    /// first key that has a kernel registered for the op, and runs
    /// it. The kernel receives the full `keyset` (not just its
    /// own key) so it can decide which keys to mask off before
    /// redispatching.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if no kernel
    /// is registered for any active key in the set, or if the set
    /// is empty.
    pub fn call(
        &self,
        op_name: &str,
        inputs: &[Tensor<T>],
        keyset: DispatchKeySet,
    ) -> FerrotorchResult<Tensor<T>> {
        if keyset.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dispatcher::call({op_name}): empty keyset — no backend to run on"
                ),
            });
        }
        for key in keyset.iter_desc() {
            if let Some(kernel) = self.kernels.get(&(op_name.to_string(), key)) {
                return kernel(inputs, keyset, self);
            }
        }
        Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Dispatcher::call({op_name}): no kernel registered for any key in {keyset:?}"
            ),
        })
    }

    /// Call `op_name` with the kernel for a specific `key`,
    /// bypassing priority resolution. Returns an error if no
    /// kernel is registered for that key.
    ///
    /// Primarily useful for testing and for kernels that want to
    /// forward directly to a specific lower-priority layer.
    pub fn call_direct(
        &self,
        op_name: &str,
        inputs: &[Tensor<T>],
        keyset: DispatchKeySet,
        key: DispatchKey,
    ) -> FerrotorchResult<Tensor<T>> {
        match self.kernels.get(&(op_name.to_string(), key)) {
            Some(kernel) => kernel(inputs, keyset, self),
            None => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dispatcher::call_direct({op_name}, {key:?}): no kernel registered"
                ),
            }),
        }
    }
}

impl<T: Float> Default for Dispatcher<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> std::fmt::Debug for Dispatcher<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dispatcher")
            .field("kernel_count", &self.kernels.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    // ── DispatchKey priority ────────────────────────────────────────

    #[test]
    fn dispatch_key_priority_ordering() {
        assert!(DispatchKey::Tracer.priority() > DispatchKey::Autograd.priority());
        assert!(DispatchKey::Autograd.priority() > DispatchKey::Autocast.priority());
        assert!(DispatchKey::Autocast.priority() > DispatchKey::Cpu.priority());
        assert!(DispatchKey::Cuda.priority() > DispatchKey::Cpu.priority());
    }

    #[test]
    fn dispatch_key_all_contains_every_key() {
        assert_eq!(DispatchKey::ALL.len(), 11);
        // Each key appears exactly once.
        for k in &DispatchKey::ALL {
            let count = DispatchKey::ALL.iter().filter(|&other| other == k).count();
            assert_eq!(count, 1, "duplicate key {k:?}");
        }
    }

    // ── DispatchKeySet membership ───────────────────────────────────

    #[test]
    fn dispatch_key_set_empty() {
        let set = DispatchKeySet::empty();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.highest(), None);
        assert!(!set.contains(DispatchKey::Cpu));
    }

    #[test]
    fn dispatch_key_set_insert_and_contains() {
        let set = DispatchKeySet::empty()
            .insert(DispatchKey::Cpu)
            .insert(DispatchKey::Autograd);
        assert_eq!(set.len(), 2);
        assert!(set.contains(DispatchKey::Cpu));
        assert!(set.contains(DispatchKey::Autograd));
        assert!(!set.contains(DispatchKey::Cuda));
    }

    #[test]
    fn dispatch_key_set_remove() {
        let set = DispatchKeySet::from([DispatchKey::Cpu, DispatchKey::Autograd]);
        let without_autograd = set.remove(DispatchKey::Autograd);
        assert_eq!(without_autograd.len(), 1);
        assert!(without_autograd.contains(DispatchKey::Cpu));
        assert!(!without_autograd.contains(DispatchKey::Autograd));
    }

    #[test]
    fn dispatch_key_set_highest() {
        let set = DispatchKeySet::from([
            DispatchKey::Cpu,
            DispatchKey::Autograd,
            DispatchKey::Profiler,
        ]);
        assert_eq!(set.highest(), Some(DispatchKey::Profiler));
    }

    #[test]
    fn dispatch_key_set_iter_desc_gives_priority_order() {
        let set = DispatchKeySet::from([
            DispatchKey::Cpu,
            DispatchKey::Tracer,
            DispatchKey::Autograd,
            DispatchKey::Cuda,
        ]);
        let order: Vec<_> = set.iter_desc().collect();
        assert_eq!(
            order,
            vec![
                DispatchKey::Tracer,
                DispatchKey::Autograd,
                DispatchKey::Cuda,
                DispatchKey::Cpu,
            ]
        );
    }

    #[test]
    fn dispatch_key_set_union_and_intersection() {
        let a = DispatchKeySet::from([DispatchKey::Cpu, DispatchKey::Autograd]);
        let b = DispatchKeySet::from([DispatchKey::Autograd, DispatchKey::Quantized]);
        let u = a.union(b);
        assert_eq!(u.len(), 3);
        assert!(u.contains(DispatchKey::Cpu));
        assert!(u.contains(DispatchKey::Autograd));
        assert!(u.contains(DispatchKey::Quantized));

        let i = a.intersection(b);
        assert_eq!(i.len(), 1);
        assert!(i.contains(DispatchKey::Autograd));
    }

    #[test]
    fn dispatch_key_set_all_contains_every_key() {
        let set = DispatchKeySet::all();
        assert_eq!(set.len(), 11);
        for &k in &DispatchKey::ALL {
            assert!(set.contains(k));
        }
    }

    #[test]
    fn dispatch_key_set_from_array_literal() {
        let set = DispatchKeySet::from([DispatchKey::Cpu, DispatchKey::Cuda]);
        assert_eq!(set.len(), 2);
    }

    // ── Dispatcher registration and lookup ──────────────────────────

    #[test]
    fn dispatcher_register_and_has_kernel() {
        let mut d = Dispatcher::<f32>::new();
        assert_eq!(d.kernel_count(), 0);
        assert!(!d.has_kernel("add", DispatchKey::Cpu));

        d.register(
            "add",
            DispatchKey::Cpu,
            |inputs, _, _| Ok(inputs[0].clone()),
        );
        assert_eq!(d.kernel_count(), 1);
        assert!(d.has_kernel("add", DispatchKey::Cpu));
        assert!(!d.has_kernel("add", DispatchKey::Cuda));
        assert!(!d.has_kernel("sub", DispatchKey::Cpu));
    }

    #[test]
    fn dispatcher_call_empty_keyset_errors() {
        let d = Dispatcher::<f32>::new();
        let t = make_tensor(vec![1.0], vec![1]);
        let result = d.call("add", &[t], DispatchKeySet::empty());
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("empty keyset"));
    }

    #[test]
    fn dispatcher_call_no_kernel_errors() {
        let d = Dispatcher::<f32>::new();
        let t = make_tensor(vec![1.0], vec![1]);
        let keyset = DispatchKeySet::from([DispatchKey::Cpu]);
        let result = d.call("add", &[t], keyset);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("no kernel registered"));
    }

    #[test]
    fn dispatcher_call_picks_highest_priority_key() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Track which kernel was called by name.
        let cpu_count = Arc::new(AtomicUsize::new(0));
        let autograd_count = Arc::new(AtomicUsize::new(0));

        let mut d = Dispatcher::<f32>::new();
        let cpu_c = Arc::clone(&cpu_count);
        d.register("add", DispatchKey::Cpu, move |inputs, _, _| {
            cpu_c.fetch_add(1, Ordering::Relaxed);
            Ok(inputs[0].clone())
        });
        let ag_c = Arc::clone(&autograd_count);
        d.register("add", DispatchKey::Autograd, move |inputs, _, _| {
            ag_c.fetch_add(1, Ordering::Relaxed);
            Ok(inputs[0].clone())
        });

        let t = make_tensor(vec![1.0], vec![1]);
        let keyset = DispatchKeySet::from([DispatchKey::Cpu, DispatchKey::Autograd]);
        d.call("add", &[t], keyset).unwrap();

        // Autograd is higher priority, so it should be called.
        assert_eq!(autograd_count.load(Ordering::Relaxed), 1);
        assert_eq!(cpu_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn dispatcher_redispatch_chains_through_keys() {
        // Autograd kernel masks itself off and calls down to Cpu.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let cpu_count = Arc::new(AtomicUsize::new(0));
        let autograd_count = Arc::new(AtomicUsize::new(0));

        let mut d = Dispatcher::<f32>::new();
        let cpu_c = Arc::clone(&cpu_count);
        d.register("add", DispatchKey::Cpu, move |inputs, _, _| {
            cpu_c.fetch_add(1, Ordering::Relaxed);
            Ok(inputs[0].clone())
        });
        let ag_c = Arc::clone(&autograd_count);
        d.register("add", DispatchKey::Autograd, move |inputs, keyset, disp| {
            ag_c.fetch_add(1, Ordering::Relaxed);
            // Mask off autograd and redispatch.
            let rest = keyset.remove(DispatchKey::Autograd);
            disp.call("add", inputs, rest)
        });

        let t = make_tensor(vec![1.0], vec![1]);
        let keyset = DispatchKeySet::from([DispatchKey::Cpu, DispatchKey::Autograd]);
        d.call("add", &[t], keyset).unwrap();

        assert_eq!(autograd_count.load(Ordering::Relaxed), 1);
        assert_eq!(cpu_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dispatcher_skips_keys_without_kernel() {
        // Register Cpu only. A keyset that includes Autograd + Cpu
        // should still resolve because Autograd has no kernel but
        // Cpu does.
        let mut d = Dispatcher::<f32>::new();
        d.register(
            "add",
            DispatchKey::Cpu,
            |inputs, _, _| Ok(inputs[0].clone()),
        );

        let t = make_tensor(vec![1.0, 2.0], vec![2]);
        let keyset = DispatchKeySet::from([DispatchKey::Autograd, DispatchKey::Cpu]);
        let result = d.call("add", &[t], keyset).unwrap();
        assert_eq!(result.shape(), &[2]);
    }

    #[test]
    fn dispatcher_call_direct_bypasses_priority() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let cpu_count = Arc::new(AtomicUsize::new(0));
        let cuda_count = Arc::new(AtomicUsize::new(0));

        let mut d = Dispatcher::<f32>::new();
        let cpu_c = Arc::clone(&cpu_count);
        d.register("add", DispatchKey::Cpu, move |inputs, _, _| {
            cpu_c.fetch_add(1, Ordering::Relaxed);
            Ok(inputs[0].clone())
        });
        let cuda_c = Arc::clone(&cuda_count);
        d.register("add", DispatchKey::Cuda, move |inputs, _, _| {
            cuda_c.fetch_add(1, Ordering::Relaxed);
            Ok(inputs[0].clone())
        });

        // call() with both keys → Cuda (higher priority).
        let t = make_tensor(vec![1.0], vec![1]);
        let keyset = DispatchKeySet::from([DispatchKey::Cpu, DispatchKey::Cuda]);
        d.call("add", std::slice::from_ref(&t), keyset).unwrap();
        assert_eq!(cuda_count.load(Ordering::Relaxed), 1);
        assert_eq!(cpu_count.load(Ordering::Relaxed), 0);

        // call_direct(Cpu) → forces Cpu kernel.
        d.call_direct("add", &[t], keyset, DispatchKey::Cpu)
            .unwrap();
        assert_eq!(cpu_count.load(Ordering::Relaxed), 1);
        assert_eq!(cuda_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dispatcher_call_direct_missing_kernel_errors() {
        let d = Dispatcher::<f32>::new();
        let t = make_tensor(vec![1.0], vec![1]);
        let keyset = DispatchKeySet::from([DispatchKey::Cpu]);
        let result = d.call_direct("add", &[t], keyset, DispatchKey::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn dispatcher_full_three_layer_stack() {
        // Realistic chain: Tracer → Autograd → Cpu.
        // Tracer emits an IR node marker and redispatches.
        // Autograd records a backward marker and redispatches.
        // Cpu does the actual math.
        use std::sync::Arc;
        use std::sync::Mutex;

        let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        let mut d = Dispatcher::<f32>::new();

        let log_c = Arc::clone(&log);
        d.register("add", DispatchKey::Cpu, move |inputs, _, _| {
            log_c.lock().unwrap().push("cpu");
            Ok(inputs[0].clone())
        });

        let log_a = Arc::clone(&log);
        d.register("add", DispatchKey::Autograd, move |inputs, keyset, disp| {
            log_a.lock().unwrap().push("autograd");
            let rest = keyset.remove(DispatchKey::Autograd);
            disp.call("add", inputs, rest)
        });

        let log_t = Arc::clone(&log);
        d.register("add", DispatchKey::Tracer, move |inputs, keyset, disp| {
            log_t.lock().unwrap().push("tracer");
            let rest = keyset.remove(DispatchKey::Tracer);
            disp.call("add", inputs, rest)
        });

        let t = make_tensor(vec![1.0, 2.0], vec![2]);
        let keyset =
            DispatchKeySet::from([DispatchKey::Tracer, DispatchKey::Autograd, DispatchKey::Cpu]);
        d.call("add", &[t], keyset).unwrap();

        let final_log = log.lock().unwrap();
        assert_eq!(*final_log, vec!["tracer", "autograd", "cpu"]);
    }
}
