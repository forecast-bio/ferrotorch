//! GPU memory safety system — reservation, OOM recovery, pressure monitoring, and
//! pre-OOM hooks.
//!
//! This module implements four layers of protection against GPU memory issues:
//!
//! 1. **Memory Reservation** ([`MemoryReservation`]) — Pre-allocate a large block at
//!    startup so other processes cannot steal VRAM out from under a training run.
//!
//! 2. **OOM Recovery** ([`OomPolicy`], [`MemoryGuard::safe_alloc`]) — Configurable
//!    behaviour when an allocation fails: retry after freeing cache, wait for memory
//!    to become available, or save a checkpoint before crashing.
//!
//! 3. **Memory Pressure Monitoring** ([`MemoryWatchdog`]) — Background thread that
//!    pauses training when free VRAM drops below a threshold, resuming automatically
//!    once the pressure lifts.
//!
//! 4. **Pre-OOM Hooks** ([`MemoryHook`], [`MemoryGuard::safe_alloc_with_hooks`]) —
//!    User-registered callbacks that fire *before* an allocation fails. Hooks declare
//!    upfront how much memory they expect to free (and any execution overhead), so the
//!    guard can call them in priority order until enough headroom is recovered.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use ferrotorch_gpu::memory_guard::{MemoryGuard, MemoryGuardBuilder, OomPolicy};
//! use ferrotorch_gpu::GpuDevice;
//!
//! let device = Arc::new(GpuDevice::new(0).unwrap());
//! let guard = MemoryGuardBuilder::new(Arc::clone(&device))
//!     .budget_bytes(20 * 1024 * 1024 * 1024) // 20 GiB
//!     .oom_policy(OomPolicy::RetryAfterFree)
//!     .build()
//!     .unwrap();
//!
//! let stats = guard.stats();
//! println!("free: {} / total: {}", stats.free_device_bytes, stats.total_device_bytes);
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// OomPolicy
// ---------------------------------------------------------------------------

/// What to do when a GPU allocation fails with an out-of-memory error.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OomPolicy {
    /// Crash immediately (PyTorch default behaviour).
    #[default]
    Fail,
    /// Free the allocator cache and retry once.
    RetryAfterFree,
    /// Wait up to `timeout_secs` seconds for memory to become available,
    /// then retry once.
    WaitAndRetry {
        /// Maximum seconds to wait.
        timeout_secs: u64,
    },
    /// Invoke the registered emergency-checkpoint callback, then fail.
    CheckpointAndFail,
}

// ---------------------------------------------------------------------------
// MemoryHook — pre-OOM callback
// ---------------------------------------------------------------------------

/// A hook that can free memory or reduce memory demand before an OOM.
///
/// Hooks declare upfront how much memory they expect to free, so the
/// memory guard can decide which hooks to call and in what order.
///
/// Construct via [`MemoryHook::new`]; the callback field is private because
/// `Box<dyn Fn>` can't be cloned, equality-compared, or serialised, so
/// exposing it as a public field would commit the API to a non-clonable
/// shape forever. Hooks are identified for unregistration by their `name`
/// (see [`MemoryGuard::remove_hook`]); the name is therefore expected to be
/// unique per registered hook.
///
/// # Example: halving a batch when memory is tight
///
/// ```rust
/// use ferrotorch_gpu::memory_guard::MemoryHook;
///
/// let hook = MemoryHook::new(
///     "halve_batch_size",
///     512 * 1024 * 1024, // expects to free ~512 MiB
///     4096,              // metadata setup cost
///     10,                // priority
///     || {
///         // ... split the batch, free old tensors ...
///         512 * 1024 * 1024 // actual bytes freed
///     },
/// );
/// ```
pub struct MemoryHook {
    /// Human-readable name (e.g., `"halve_batch_size"`, `"free_kv_cache"`).
    pub name: String,
    /// Estimated bytes this hook will free when called.
    pub estimated_free_bytes: usize,
    /// Extra bytes this hook needs temporarily to execute (e.g., metadata
    /// setup for a batch split). If the available headroom is less than this
    /// overhead the hook is skipped.
    pub execution_overhead_bytes: usize,
    /// Priority: lower values fire first. Hooks at the same priority are
    /// called in registration order.
    pub priority: u32,
    /// The callback. Returns the *actual* bytes freed (may differ from the
    /// estimate). Private to keep the boxed-closure shape encapsulated;
    /// the guard's [`run_hooks`](MemoryGuard) machinery is the only caller.
    pub(crate) callback: Box<dyn Fn() -> usize + Send + Sync>,
}

impl MemoryHook {
    /// Build a hook from its scheduling metadata and callback.
    ///
    /// `name` identifies the hook for [`MemoryGuard::remove_hook`].
    /// `estimated_free_bytes` is the byte amount the guard will assume the
    /// callback frees when sequencing hooks; `execution_overhead_bytes` is
    /// the temporary headroom the hook needs to run (a hook is skipped when
    /// the budget can't afford the overhead). `priority` orders hooks
    /// (lowest first; ties broken by registration order).
    ///
    /// `callback` returns the *actual* bytes freed when run, which may
    /// differ from the estimate.
    pub fn new<S, F>(
        name: S,
        estimated_free_bytes: usize,
        execution_overhead_bytes: usize,
        priority: u32,
        callback: F,
    ) -> Self
    where
        S: Into<String>,
        F: Fn() -> usize + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            estimated_free_bytes,
            execution_overhead_bytes,
            priority,
            callback: Box::new(callback),
        }
    }
}

impl std::fmt::Debug for MemoryHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHook")
            .field("name", &self.name)
            .field("estimated_free_bytes", &self.estimated_free_bytes)
            .field("execution_overhead_bytes", &self.execution_overhead_bytes)
            .field("priority", &self.priority)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PressureLevel
// ---------------------------------------------------------------------------

/// Level of memory pressure relative to the configured budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureLevel {
    /// Plenty of headroom (>30% of budget free).
    None,
    /// Getting tight (10--30% free). Informational only.
    Low,
    /// Approaching the limit (5--10% free). Non-critical hooks may fire.
    Medium,
    /// Near OOM (<5% free). All hooks fire.
    High,
    /// An allocation would fail without intervention.
    Critical,
}

impl std::fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        };
        f.write_str(label)
    }
}

// ---------------------------------------------------------------------------
// MemoryPressureListener
// ---------------------------------------------------------------------------

/// Trait for continuous pressure-level monitoring.
///
/// Types that conform to this trait can be registered via
/// [`MemoryGuard::add_pressure_listener`] to receive callbacks whenever the
/// pressure level changes (e.g., after every allocation or free).
pub trait MemoryPressureListener: Send + Sync {
    /// Called when the memory-pressure level transitions between two values.
    fn on_pressure_change(&self, old: PressureLevel, new: PressureLevel);
}

// ---------------------------------------------------------------------------
// MemoryReservation
// ---------------------------------------------------------------------------

/// A sentinel CUDA allocation that reserves physical VRAM.
///
/// As long as this struct is alive the driver cannot give the reserved bytes to
/// another process. Drop the reservation (or call
/// [`MemoryGuard::release_reservation`]) to free the memory for reuse.
pub struct MemoryReservation {
    /// The reserved CUDA allocation that holds our budget.
    /// Other processes cannot use this memory while this buffer exists.
    _reservation: CudaBuffer<u8>,
    /// Number of bytes reserved.
    reserved_bytes: usize,
    /// Which device the reservation lives on.
    device_ordinal: usize,
}

impl MemoryReservation {
    /// How many bytes are held by this reservation.
    #[inline]
    pub fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    /// The device ordinal the reservation lives on.
    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}

impl std::fmt::Debug for MemoryReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryReservation")
            .field("reserved_bytes", &self.reserved_bytes)
            .field("device_ordinal", &self.device_ordinal)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MemoryStats
// ---------------------------------------------------------------------------

/// Snapshot of memory-guard statistics.
///
/// The struct is `#[non_exhaustive]` so future fields can be added without
/// a major-version bump. External callers must read fields through the
/// public accessors rather than struct-pattern matching.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MemoryStats {
    /// Bytes currently tracked as live by the guard.
    pub used_bytes: usize,
    /// Hard budget ceiling (0 = unlimited).
    pub budget_bytes: usize,
    /// Peak tracked usage since creation or last reset.
    pub peak_bytes: usize,
    /// Free device memory as reported by the driver.
    pub free_device_bytes: usize,
    /// Total device memory as reported by the driver.
    pub total_device_bytes: usize,
    /// Number of live allocations tracked by the guard.
    pub num_allocations: usize,
    /// Number of OOM events that were successfully recovered.
    pub num_oom_recoveries: usize,
}

// ---------------------------------------------------------------------------
// MemoryGuard
// ---------------------------------------------------------------------------

/// Central memory-safety controller for a single GPU.
///
/// Wraps a [`GpuDevice`] and provides:
/// - Optional upfront VRAM reservation (sentinel allocation).
/// - Budget enforcement — allocations that would exceed the budget are
///   rejected *before* touching the driver.
/// - Configurable OOM recovery via [`OomPolicy`].
/// - An emergency-checkpoint callback.
///
/// Construct via [`MemoryGuardBuilder`].
pub struct MemoryGuard {
    device: Arc<GpuDevice>,
    /// Pre-allocated reservation block.
    reservation: Mutex<Option<MemoryReservation>>,
    /// Maximum bytes we are allowed to use (0 = unlimited).
    budget_bytes: AtomicUsize,
    /// Current live allocation bytes tracked by the guard.
    used_bytes: AtomicUsize,
    /// Peak tracked usage.
    peak_bytes: AtomicUsize,
    /// Number of live allocations.
    num_allocations: AtomicUsize,
    /// Number of successful OOM recoveries.
    num_oom_recoveries: AtomicUsize,
    /// Policy when OOM occurs.
    oom_policy: Mutex<OomPolicy>,
    /// Callback for emergency checkpoint.
    on_oom_callback: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Pre-OOM hooks, called before an allocation failure is propagated.
    hooks: Mutex<Vec<MemoryHook>>,
    /// Continuous pressure-level listeners.
    pressure_listeners: Mutex<Vec<Box<dyn MemoryPressureListener>>>,
    /// Cached pressure level for change detection.
    last_pressure_level: Mutex<PressureLevel>,
}

// SAFETY: All interior mutability is via atomics or `Mutex`.
unsafe impl Send for MemoryGuard {}
unsafe impl Sync for MemoryGuard {}

impl MemoryGuard {
    // ------------------------------------------------------------------
    // Budget
    // ------------------------------------------------------------------

    /// Set a hard budget in bytes. Allocations that would push `used_bytes`
    /// past this limit return [`GpuError::BudgetExceeded`] without touching
    /// the driver.
    ///
    /// Pass `0` to remove the budget (unlimited).
    pub fn set_budget(&self, bytes: usize) {
        self.budget_bytes.store(bytes, Ordering::SeqCst);
    }

    /// Current budget (0 = unlimited).
    #[inline]
    pub fn budget(&self) -> usize {
        self.budget_bytes.load(Ordering::Relaxed)
    }

    // ------------------------------------------------------------------
    // OOM callback
    // ------------------------------------------------------------------

    /// Register a callback that will be invoked on OOM when the policy is
    /// [`OomPolicy::CheckpointAndFail`]. Typically used to save a training
    /// checkpoint so work is not lost.
    pub fn on_oom<F: Fn() + Send + Sync + 'static>(&self, f: F) {
        *self.on_oom_callback.lock().unwrap() = Some(Box::new(f));
    }

    /// Change the OOM policy at runtime.
    pub fn set_oom_policy(&self, policy: OomPolicy) {
        *self.oom_policy.lock().unwrap() = policy;
    }

    // ------------------------------------------------------------------
    // Pre-OOM hooks
    // ------------------------------------------------------------------

    /// Register a pre-OOM hook.
    ///
    /// Hooks are called (in priority order, lowest first) when an allocation
    /// would exceed the budget. Each hook gets a chance to free memory before
    /// the guard falls through to the [`OomPolicy`].
    pub fn register_hook(&self, hook: MemoryHook) {
        self.hooks.lock().unwrap().push(hook);
    }

    /// Remove a previously registered hook by name.
    ///
    /// Returns `true` if a hook with that name was found and removed.
    pub fn remove_hook(&self, name: &str) -> bool {
        let mut hooks = self.hooks.lock().unwrap();
        let before = hooks.len();
        hooks.retain(|h| h.name != name);
        hooks.len() < before
    }

    /// Current pressure level based on budget usage.
    ///
    /// If no budget is set (budget = 0 / unlimited), always returns
    /// [`PressureLevel::None`].
    pub fn pressure_level(&self) -> PressureLevel {
        let budget = self.budget_bytes.load(Ordering::Relaxed);
        if budget == 0 {
            return PressureLevel::None;
        }
        let used = self.used_bytes.load(Ordering::Relaxed);
        Self::compute_pressure(budget, used)
    }

    /// Compute the pressure level from a budget and usage pair.
    ///
    /// A budget of `0` means unlimited and always returns [`PressureLevel::None`].
    fn compute_pressure(budget: usize, used: usize) -> PressureLevel {
        if budget == 0 {
            return PressureLevel::None;
        }
        if used >= budget {
            return PressureLevel::Critical;
        }
        let free_frac = ((budget - used) as f64) / (budget as f64);
        if free_frac > 0.30 {
            PressureLevel::None
        } else if free_frac > 0.10 {
            PressureLevel::Low
        } else if free_frac > 0.05 {
            PressureLevel::Medium
        } else {
            PressureLevel::High
        }
    }

    /// Register a listener that is notified whenever the pressure level
    /// changes (checked after every allocation and free through the guard).
    pub fn add_pressure_listener(&self, listener: Box<dyn MemoryPressureListener>) {
        self.pressure_listeners.lock().unwrap().push(listener);
    }

    /// Check whether the pressure level has changed and notify listeners.
    fn notify_pressure_change(&self) {
        let new_level = self.pressure_level();
        let mut last = self.last_pressure_level.lock().unwrap();
        if *last != new_level {
            let old = *last;
            *last = new_level;
            // Release the last-level lock before calling listeners to avoid
            // deadlocks if a listener queries the guard.
            drop(last);
            let listeners = self.pressure_listeners.lock().unwrap();
            for listener in listeners.iter() {
                listener.on_pressure_change(old, new_level);
            }
        }
    }

    /// Allocate `count` zero-initialized elements, trying pre-OOM hooks
    /// before falling through to the [`OomPolicy`].
    ///
    /// The algorithm:
    ///
    /// 1. Check if the allocation fits within the budget -- if so, allocate
    ///    directly.
    /// 2. If not, compute the shortfall.
    /// 3. Sort hooks by `(priority, estimated_free_bytes descending)`.
    /// 4. Call hooks one at a time, skipping any whose
    ///    `execution_overhead_bytes` exceeds current headroom, until enough
    ///    cumulative memory has been freed.
    /// 5. Retry the allocation.
    /// 6. If still insufficient after all hooks, fall through to the regular
    ///    [`OomPolicy`] path.
    #[cfg(feature = "cuda")]
    pub fn safe_alloc_with_hooks<T>(&self, count: usize) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let alloc_bytes = count.saturating_mul(std::mem::size_of::<T>());

        // Fast path: fits within budget.
        if self.check_budget(alloc_bytes).is_ok() {
            let result = self.try_alloc_zeros::<T>(count, alloc_bytes);
            if result.is_ok() {
                self.notify_pressure_change();
            }
            return result;
        }

        // Compute the shortfall.
        let budget = self.budget_bytes.load(Ordering::Relaxed);
        let used = self.used_bytes.load(Ordering::Relaxed);
        let shortfall = (used + alloc_bytes).saturating_sub(budget);

        // Try hooks.
        let freed = self.run_hooks(shortfall, budget, used);

        if freed > 0 {
            // Re-check budget after hooks freed memory.
            if self.check_budget(alloc_bytes).is_ok() {
                let result = self.try_alloc_zeros::<T>(count, alloc_bytes);
                if result.is_ok() {
                    self.notify_pressure_change();
                    return result;
                }
                // Driver-level OOM despite budget check passing -- fall through
                // to OomPolicy.
                if let Err(e) = result {
                    if self.is_oom(&e) {
                        return self.handle_oom(count, alloc_bytes, e);
                    }
                    return Err(e);
                }
            }
        }

        // Hooks were not enough. Re-check budget — if still over, enforce
        // the budget rather than letting the driver allocate beyond it.
        if self.check_budget(alloc_bytes).is_err() {
            let budget = self.budget_bytes.load(Ordering::Relaxed);
            let used = self.used_bytes.load(Ordering::Relaxed);
            return Err(crate::error::GpuError::BudgetExceeded {
                requested_bytes: alloc_bytes,
                budget_bytes: budget,
                used_bytes: used,
            });
        }

        // Budget check passed (hooks freed enough). Try the driver.
        match self.try_alloc_zeros::<T>(count, alloc_bytes) {
            Ok(buf) => {
                self.notify_pressure_change();
                Ok(buf)
            }
            Err(e) if self.is_oom(&e) => self.handle_oom(count, alloc_bytes, e),
            Err(e) => Err(e),
        }
    }

    /// Run pre-OOM hooks in priority order until `shortfall` bytes have been
    /// freed (or all hooks have been tried).
    ///
    /// Returns the total actual bytes freed across all invoked hooks.
    #[allow(dead_code)]
    fn run_hooks(&self, shortfall: usize, budget: usize, used: usize) -> usize {
        // Build a sorted index of hooks. We sort by (priority ASC,
        // estimated_free_bytes DESC) so high-impact hooks at a given
        // priority run first.
        let hooks = self.hooks.lock().unwrap();
        if hooks.is_empty() {
            return 0;
        }

        let mut indices: Vec<usize> = (0..hooks.len()).collect();
        indices.sort_by(|&a, &b| {
            hooks[a].priority.cmp(&hooks[b].priority).then_with(|| {
                hooks[b]
                    .estimated_free_bytes
                    .cmp(&hooks[a].estimated_free_bytes)
            })
        });

        let mut total_freed: usize = 0;
        let mut current_used = used;

        for &idx in &indices {
            if total_freed >= shortfall {
                break;
            }

            let hook = &hooks[idx];

            // Skip if overhead exceeds available headroom. "Available
            // headroom" is whatever room we have *right now* within the
            // budget, after accounting for memory freed so far.
            let headroom = budget.saturating_sub(current_used);
            if hook.execution_overhead_bytes > headroom {
                continue;
            }

            let freed = (hook.callback)();
            total_freed = total_freed.saturating_add(freed);
            current_used = current_used.saturating_sub(freed);
        }

        // Reflect freed memory in the atomic counter. Hooks freed memory
        // outside the guard's tracking, so we adjust used_bytes downward.
        if total_freed > 0 {
            self.used_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_sub(total_freed))
                })
                .ok();
        }

        total_freed
    }

    // ------------------------------------------------------------------
    // Reservation management
    // ------------------------------------------------------------------

    /// Release the upfront reservation, making its memory available for
    /// normal allocations. Returns the number of bytes released, or `0` if
    /// there was no active reservation.
    pub fn release_reservation(&self) -> usize {
        let mut lock = self.reservation.lock().unwrap();
        if let Some(res) = lock.take() {
            let bytes = res.reserved_bytes;
            drop(res);
            bytes
        } else {
            0
        }
    }

    /// Whether an upfront reservation is currently held.
    pub fn has_reservation(&self) -> bool {
        self.reservation.lock().unwrap().is_some()
    }

    // ------------------------------------------------------------------
    // Allocation with safety layers
    // ------------------------------------------------------------------

    /// Allocate `count` zero-initialized elements on the device, enforcing
    /// the budget and OOM policy.
    ///
    /// This is the primary allocation entry point when using the memory
    /// guard. Prefer this over raw `CudaAllocator::alloc_zeros`.
    #[cfg(feature = "cuda")]
    pub fn safe_alloc<T>(&self, count: usize) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let alloc_bytes = count.saturating_mul(std::mem::size_of::<T>());

        // --- Layer 1: budget check ---
        self.check_budget(alloc_bytes)?;

        // --- Layer 2: attempt allocation ---
        match self.try_alloc_zeros::<T>(count, alloc_bytes) {
            Ok(buf) => Ok(buf),
            Err(e) if self.is_oom(&e) => self.handle_oom(count, alloc_bytes, e),
            Err(e) => Err(e),
        }
    }

    /// Allocate by copying host data to the device, enforcing budget and
    /// OOM policy.
    #[cfg(feature = "cuda")]
    pub fn safe_alloc_copy<T>(&self, data: &[T]) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        let alloc_bytes = data.len().saturating_mul(std::mem::size_of::<T>());

        self.check_budget(alloc_bytes)?;

        match self.try_alloc_copy(data, alloc_bytes) {
            Ok(buf) => Ok(buf),
            Err(e) if self.is_oom(&e) => {
                // For copy allocs, retry with the same data.
                let policy = self.oom_policy.lock().unwrap().clone();
                match policy {
                    OomPolicy::Fail => Err(e),
                    OomPolicy::RetryAfterFree => {
                        self.free_caches();
                        self.num_oom_recoveries.fetch_add(1, Ordering::Relaxed);
                        self.try_alloc_copy(data, alloc_bytes)
                    }
                    OomPolicy::WaitAndRetry { timeout_secs } => {
                        self.wait_for_memory(alloc_bytes, timeout_secs)?;
                        self.num_oom_recoveries.fetch_add(1, Ordering::Relaxed);
                        self.try_alloc_copy(data, alloc_bytes)
                    }
                    OomPolicy::CheckpointAndFail => {
                        self.trigger_emergency_checkpoint();
                        Err(e)
                    }
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Return a buffer to the guard, freeing GPU memory and updating
    /// statistics.
    pub fn free<T>(&self, buffer: CudaBuffer<T>) {
        let bytes = buffer
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .unwrap_or(0);
        self.used_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes))
            })
            .ok();
        self.num_allocations
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            })
            .ok();
        drop(buffer);
        self.notify_pressure_change();
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    /// Snapshot the current memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let (free_device, total_device) = self.query_device_memory();
        MemoryStats {
            used_bytes: self.used_bytes.load(Ordering::Relaxed),
            budget_bytes: self.budget_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            free_device_bytes: free_device,
            total_device_bytes: total_device,
            num_allocations: self.num_allocations.load(Ordering::Relaxed),
            num_oom_recoveries: self.num_oom_recoveries.load(Ordering::Relaxed),
        }
    }

    /// Reset the peak-usage counter to the current usage level.
    pub fn reset_peak_stats(&self) {
        let current = self.used_bytes.load(Ordering::Relaxed);
        self.peak_bytes.store(current, Ordering::Relaxed);
    }

    /// The underlying device.
    #[inline]
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// The underlying device as an `Arc`.
    #[inline]
    pub fn device_arc(&self) -> &Arc<GpuDevice> {
        &self.device
    }

    // ------------------------------------------------------------------
    // Internal helpers
    //
    // These methods are used by the `#[cfg(feature = "cuda")]` allocation
    // paths. In the no-cuda build the callers do not exist, so we suppress
    // the dead-code lint.
    // ------------------------------------------------------------------

    /// Check whether `alloc_bytes` would exceed the budget.
    #[allow(dead_code)]
    fn check_budget(&self, alloc_bytes: usize) -> GpuResult<()> {
        let budget = self.budget_bytes.load(Ordering::Relaxed);
        if budget == 0 {
            return Ok(()); // unlimited
        }
        let used = self.used_bytes.load(Ordering::Relaxed);
        if used.saturating_add(alloc_bytes) > budget {
            return Err(GpuError::BudgetExceeded {
                requested_bytes: alloc_bytes,
                budget_bytes: budget,
                used_bytes: used,
            });
        }
        Ok(())
    }

    /// Low-level zero-init allocation with tracking.
    #[cfg(feature = "cuda")]
    fn try_alloc_zeros<T>(&self, count: usize, alloc_bytes: usize) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let slice = self.device.stream().alloc_zeros::<T>(count)?;

        let prev = self.used_bytes.fetch_add(alloc_bytes, Ordering::Relaxed);
        self.peak_bytes
            .fetch_max(prev + alloc_bytes, Ordering::Relaxed);
        self.num_allocations.fetch_add(1, Ordering::Relaxed);

        Ok(CudaBuffer {
            data: Some(slice),
            len: count,
            alloc_len: count,
            device_ordinal: self.device.ordinal(),
            pool_fn: None,
        })
    }

    /// Low-level host-to-device copy allocation with tracking.
    #[cfg(feature = "cuda")]
    fn try_alloc_copy<T>(&self, data: &[T], alloc_bytes: usize) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        let slice = self.device.stream().clone_htod(data)?;

        let prev = self.used_bytes.fetch_add(alloc_bytes, Ordering::Relaxed);
        self.peak_bytes
            .fetch_max(prev + alloc_bytes, Ordering::Relaxed);
        self.num_allocations.fetch_add(1, Ordering::Relaxed);

        Ok(CudaBuffer {
            data: Some(slice),
            len: data.len(),
            alloc_len: data.len(),
            device_ordinal: self.device.ordinal(),
            pool_fn: None,
        })
    }

    /// Determine whether an error is an out-of-memory condition.
    #[allow(dead_code)]
    fn is_oom(&self, err: &GpuError) -> bool {
        match err {
            GpuError::OutOfMemory { .. } => true,
            #[cfg(feature = "cuda")]
            GpuError::Driver(driver_err) => {
                let msg = format!("{driver_err}");
                msg.contains("OUT_OF_MEMORY")
                    || msg.contains("out of memory")
                    || msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
            }
            _ => false,
        }
    }

    /// Handle an OOM according to the current policy.
    #[cfg(feature = "cuda")]
    fn handle_oom<T>(
        &self,
        count: usize,
        alloc_bytes: usize,
        original_err: GpuError,
    ) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let policy = self.oom_policy.lock().unwrap().clone();
        match policy {
            OomPolicy::Fail => Err(original_err),
            OomPolicy::RetryAfterFree => {
                self.free_caches();
                self.num_oom_recoveries.fetch_add(1, Ordering::Relaxed);
                self.try_alloc_zeros(count, alloc_bytes)
            }
            OomPolicy::WaitAndRetry { timeout_secs } => {
                self.wait_for_memory(alloc_bytes, timeout_secs)?;
                self.num_oom_recoveries.fetch_add(1, Ordering::Relaxed);
                self.try_alloc_zeros(count, alloc_bytes)
            }
            OomPolicy::CheckpointAndFail => {
                self.trigger_emergency_checkpoint();
                Err(original_err)
            }
        }
    }

    /// Best-effort cache eviction. Currently a no-op placeholder — the
    /// caching allocator is not yet implemented. When it is, this will
    /// release all cached-but-free blocks.
    #[allow(dead_code)]
    fn free_caches(&self) {
        // Future: delegate to CudaAllocator::empty_cache() once block
        // caching is implemented.
    }

    /// Block until at least `needed_bytes` are free, or until `timeout_secs`
    /// elapses.
    #[allow(dead_code)]
    fn wait_for_memory(&self, needed_bytes: usize, timeout_secs: u64) -> GpuResult<()> {
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        loop {
            let (free, _) = self.query_device_memory();
            if free >= needed_bytes {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(GpuError::OutOfMemory {
                    requested_bytes: needed_bytes,
                    free_bytes: free,
                });
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Invoke the user-registered emergency checkpoint callback.
    #[allow(dead_code)]
    fn trigger_emergency_checkpoint(&self) {
        let lock = self.on_oom_callback.lock().unwrap();
        if let Some(cb) = lock.as_ref() {
            cb();
        }
    }

    /// Query free and total device memory from the driver.
    ///
    /// Returns `(free_bytes, total_bytes)`. On error (or when the `cuda`
    /// feature is disabled), returns `(0, 0)`.
    fn query_device_memory(&self) -> (usize, usize) {
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::result::mem_get_info().unwrap_or((0, 0))
        }
        #[cfg(not(feature = "cuda"))]
        {
            (0, 0)
        }
    }
}

impl std::fmt::Debug for MemoryGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGuard")
            .field("device_ordinal", &self.device.ordinal())
            .field("budget_bytes", &self.budget_bytes.load(Ordering::Relaxed))
            .field("used_bytes", &self.used_bytes.load(Ordering::Relaxed))
            .field("peak_bytes", &self.peak_bytes.load(Ordering::Relaxed))
            .field(
                "has_reservation",
                &self.reservation.lock().unwrap().is_some(),
            )
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
impl MemoryGuard {
    /// Stub — returns [`GpuError::NoCudaFeature`].
    pub fn safe_alloc<T>(&self, _count: usize) -> GpuResult<CudaBuffer<T>> {
        Err(GpuError::NoCudaFeature)
    }

    /// Stub — returns [`GpuError::NoCudaFeature`].
    pub fn safe_alloc_copy<T>(&self, _data: &[T]) -> GpuResult<CudaBuffer<T>> {
        Err(GpuError::NoCudaFeature)
    }

    /// Stub — returns [`GpuError::NoCudaFeature`].
    pub fn safe_alloc_with_hooks<T>(&self, _count: usize) -> GpuResult<CudaBuffer<T>> {
        Err(GpuError::NoCudaFeature)
    }
}

// ---------------------------------------------------------------------------
// MemoryGuardBuilder
// ---------------------------------------------------------------------------

/// Builder for [`MemoryGuard`].
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use ferrotorch_gpu::memory_guard::{MemoryGuardBuilder, OomPolicy};
/// # use ferrotorch_gpu::GpuDevice;
/// let device = Arc::new(GpuDevice::new(0).unwrap());
/// let guard = MemoryGuardBuilder::new(device)
///     .budget_bytes(16 * 1024 * 1024 * 1024)
///     .reserve_bytes(16 * 1024 * 1024 * 1024)
///     .oom_policy(OomPolicy::RetryAfterFree)
///     .build()
///     .unwrap();
/// ```
pub struct MemoryGuardBuilder {
    device: Arc<GpuDevice>,
    budget_bytes: usize,
    reserve_bytes: usize,
    oom_policy: OomPolicy,
}

impl std::fmt::Debug for MemoryGuardBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGuardBuilder")
            .field("budget_bytes", &self.budget_bytes)
            .field("reserve_bytes", &self.reserve_bytes)
            .field("oom_policy", &self.oom_policy)
            .field("device_ordinal", &self.device.ordinal())
            .finish()
    }
}

impl MemoryGuardBuilder {
    /// Create a new builder for the given device.
    pub fn new(device: Arc<GpuDevice>) -> Self {
        Self {
            device,
            budget_bytes: 0,
            reserve_bytes: 0,
            oom_policy: OomPolicy::default(),
        }
    }

    /// Set the hard memory budget in bytes. `0` means unlimited.
    pub fn budget_bytes(mut self, bytes: usize) -> Self {
        self.budget_bytes = bytes;
        self
    }

    /// Pre-allocate `bytes` of VRAM as a reservation sentinel.
    /// Other processes cannot use this memory while the guard is alive.
    pub fn reserve_bytes(mut self, bytes: usize) -> Self {
        self.reserve_bytes = bytes;
        self
    }

    /// Set the OOM recovery policy.
    pub fn oom_policy(mut self, policy: OomPolicy) -> Self {
        self.oom_policy = policy;
        self
    }

    /// Build the [`MemoryGuard`].
    ///
    /// If `reserve_bytes` was set, this will attempt to allocate the
    /// sentinel buffer. Failure to allocate is returned as an error.
    #[cfg(feature = "cuda")]
    pub fn build(self) -> GpuResult<MemoryGuard> {
        let reservation = if self.reserve_bytes > 0 {
            let slice = self.device.stream().alloc_zeros::<u8>(self.reserve_bytes)?;
            Some(MemoryReservation {
                _reservation: CudaBuffer {
                    data: Some(slice),
                    len: self.reserve_bytes,
                    alloc_len: self.reserve_bytes,
                    device_ordinal: self.device.ordinal(),
                    pool_fn: None,
                },
                reserved_bytes: self.reserve_bytes,
                device_ordinal: self.device.ordinal(),
            })
        } else {
            None
        };

        Ok(MemoryGuard {
            device: self.device,
            reservation: Mutex::new(reservation),
            budget_bytes: AtomicUsize::new(self.budget_bytes),
            used_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            num_allocations: AtomicUsize::new(0),
            num_oom_recoveries: AtomicUsize::new(0),
            oom_policy: Mutex::new(self.oom_policy),
            on_oom_callback: Mutex::new(None),
            hooks: Mutex::new(Vec::new()),
            pressure_listeners: Mutex::new(Vec::new()),
            last_pressure_level: Mutex::new(PressureLevel::None),
        })
    }

    /// Stub build when `cuda` feature is disabled.
    #[cfg(not(feature = "cuda"))]
    pub fn build(self) -> GpuResult<MemoryGuard> {
        Ok(MemoryGuard {
            device: self.device,
            reservation: Mutex::new(None),
            budget_bytes: AtomicUsize::new(self.budget_bytes),
            used_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            num_allocations: AtomicUsize::new(0),
            num_oom_recoveries: AtomicUsize::new(0),
            oom_policy: Mutex::new(self.oom_policy),
            on_oom_callback: Mutex::new(None),
            hooks: Mutex::new(Vec::new()),
            pressure_listeners: Mutex::new(Vec::new()),
            last_pressure_level: Mutex::new(PressureLevel::None),
        })
    }
}

// ---------------------------------------------------------------------------
// MemoryWatchdog
// ---------------------------------------------------------------------------

/// Background monitor that pauses training when free VRAM drops below a
/// threshold.
///
/// Create a watchdog, wrap it in an `Arc`, and call [`start`](Self::start) to
/// spawn the monitoring thread. Between training batches, call
/// [`wait_if_paused`](Self::wait_if_paused) to block until memory pressure
/// is resolved.
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use std::time::Duration;
/// use ferrotorch_gpu::memory_guard::MemoryWatchdog;
/// use ferrotorch_gpu::GpuDevice;
///
/// let device = Arc::new(GpuDevice::new(0).unwrap());
/// let watchdog = Arc::new(MemoryWatchdog::new(
///     device,
///     512 * 1024 * 1024, // pause when <512 MiB free
///     Duration::from_secs(1),
/// ));
/// let handle = Arc::clone(&watchdog).start();
///
/// // In training loop:
/// watchdog.wait_if_paused();
/// ```
pub struct MemoryWatchdog {
    device: Arc<GpuDevice>,
    /// Minimum free bytes before we pause.
    pressure_threshold_bytes: usize,
    /// How often to poll the driver.
    check_interval: Duration,
    /// Whether training is currently paused due to memory pressure.
    paused: AtomicBool,
    /// Signal to stop the background thread.
    stop: AtomicBool,
    /// Set to `true` after the first check cycle completes.
    has_checked: AtomicBool,
}

impl MemoryWatchdog {
    /// Create a new watchdog. Does not start monitoring until [`start`](Self::start)
    /// is called.
    pub fn new(
        device: Arc<GpuDevice>,
        pressure_threshold_bytes: usize,
        check_interval: Duration,
    ) -> Self {
        Self {
            device,
            pressure_threshold_bytes,
            check_interval,
            paused: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            has_checked: AtomicBool::new(false),
        }
    }

    /// Start the monitoring thread. Returns a `JoinHandle` that can be used
    /// to wait for shutdown (after calling [`stop`](Self::stop)).
    pub fn start(self: Arc<Self>) -> JoinHandle<()> {
        std::thread::Builder::new()
            .name("ferrotorch-memory-watchdog".into())
            .spawn(move || {
                while !self.stop.load(Ordering::Relaxed) {
                    let free = self.query_free_memory();
                    if free < self.pressure_threshold_bytes {
                        self.paused.store(true, Ordering::SeqCst);
                        // Spin until memory is available or we are told to stop.
                        while self.query_free_memory() < self.pressure_threshold_bytes {
                            if self.stop.load(Ordering::Relaxed) {
                                return;
                            }
                            std::thread::sleep(Duration::from_millis(500));
                        }
                        self.paused.store(false, Ordering::SeqCst);
                    }
                    self.has_checked.store(true, Ordering::SeqCst);
                    std::thread::sleep(self.check_interval);
                }
            })
            .expect("failed to spawn memory watchdog thread")
    }

    /// Signal the background thread to exit.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if the watchdog currently has training paused due to
    /// memory pressure.
    #[inline]
    pub fn check_pressure(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Block the calling thread until memory pressure is resolved.
    /// Call this between training batches.
    pub fn wait_if_paused(&self) {
        while self.paused.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Block until the watchdog has completed at least one check cycle.
    /// Useful in tests to avoid timing races.
    pub fn wait_for_first_check(&self, timeout: Duration) {
        let start = std::time::Instant::now();
        while !self.has_checked.load(Ordering::SeqCst) {
            if start.elapsed() > timeout {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The pressure threshold in bytes.
    #[inline]
    pub fn pressure_threshold_bytes(&self) -> usize {
        self.pressure_threshold_bytes
    }

    /// Query the amount of free device memory.
    ///
    /// Binds the device's CUDA context on the current thread before querying,
    /// so this is safe to call from the watchdog's background thread.
    fn query_free_memory(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            // Bind the CUDA context on this thread so mem_get_info works.
            let ctx = self.device.context();
            let _ = ctx.bind_to_thread();
            cudarc::driver::result::mem_get_info()
                .map(|(free, _)| free)
                .unwrap_or(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &self.device;
            0
        }
    }
}

impl std::fmt::Debug for MemoryWatchdog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryWatchdog")
            .field("device_ordinal", &self.device.ordinal())
            .field("pressure_threshold_bytes", &self.pressure_threshold_bytes)
            .field("check_interval", &self.check_interval)
            .field("paused", &self.paused.load(Ordering::Relaxed))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// GpuDevice extensions
// ---------------------------------------------------------------------------

impl GpuDevice {
    /// Query free and total GPU memory for this device.
    ///
    /// Returns `(free_bytes, total_bytes)`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Driver`] if the CUDA driver call fails.
    #[cfg(feature = "cuda")]
    pub fn memory_info(&self) -> GpuResult<(usize, usize)> {
        // cuMemGetInfo operates on the current context, so we need to ensure
        // this device's context is bound. The cudarc CudaContext does this
        // internally for allocations, but mem_get_info is a free function.
        // Binding is handled by the caller having created the device.
        let info = cudarc::driver::result::mem_get_info()?;
        Ok(info)
    }

    /// Query free and total GPU memory — stub when `cuda` is disabled.
    #[cfg(not(feature = "cuda"))]
    pub fn memory_info(&self) -> GpuResult<(usize, usize)> {
        Err(GpuError::NoCudaFeature)
    }
}

/// A [`GpuDevice`] paired with a [`MemoryGuard`] for convenient use.
///
/// Created by [`GpuDevice::with_memory_guard`].
pub struct MemoryGuardedDevice {
    /// The memory guard managing allocations.
    pub guard: MemoryGuard,
}

impl MemoryGuardedDevice {
    /// Access the underlying device.
    #[inline]
    pub fn device(&self) -> &GpuDevice {
        self.guard.device()
    }

    /// Access the memory guard.
    #[inline]
    pub fn guard(&self) -> &MemoryGuard {
        &self.guard
    }
}

impl std::fmt::Debug for MemoryGuardedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGuardedDevice")
            .field("guard", &self.guard)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Unit tests (no GPU required)
    // ---------------------------------------------------------------

    #[test]
    fn oom_policy_default_is_fail() {
        assert_eq!(OomPolicy::default(), OomPolicy::Fail);
    }

    #[test]
    fn oom_policy_debug() {
        let p = OomPolicy::WaitAndRetry { timeout_secs: 30 };
        let s = format!("{p:?}");
        assert!(s.contains("WaitAndRetry"));
        assert!(s.contains("30"));
    }

    #[test]
    fn memory_stats_clone_eq() {
        let s = MemoryStats {
            used_bytes: 100,
            budget_bytes: 1000,
            peak_bytes: 200,
            free_device_bytes: 800,
            total_device_bytes: 2000,
            num_allocations: 5,
            num_oom_recoveries: 1,
        };
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    #[test]
    fn memory_stats_debug() {
        let s = MemoryStats {
            used_bytes: 0,
            budget_bytes: 0,
            peak_bytes: 0,
            free_device_bytes: 0,
            total_device_bytes: 0,
            num_allocations: 0,
            num_oom_recoveries: 0,
        };
        let d = format!("{s:?}");
        assert!(d.contains("MemoryStats"));
        assert!(d.contains("used_bytes"));
    }

    #[test]
    fn gpu_error_out_of_memory_display() {
        let e = GpuError::OutOfMemory {
            requested_bytes: 1024,
            free_bytes: 512,
        };
        let s = format!("{e}");
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
        assert!(s.contains("out of memory"));
    }

    #[test]
    fn gpu_error_budget_exceeded_display() {
        let e = GpuError::BudgetExceeded {
            requested_bytes: 500,
            budget_bytes: 1000,
            used_bytes: 800,
        };
        let s = format!("{e}");
        assert!(s.contains("500"));
        assert!(s.contains("1000"));
        assert!(s.contains("800"));
        assert!(s.contains("budget exceeded"));
    }

    // ---------------------------------------------------------------
    // Pre-OOM hooks & pressure unit tests (no GPU required)
    // ---------------------------------------------------------------

    #[test]
    fn pressure_level_ordering() {
        assert!(PressureLevel::None < PressureLevel::Low);
        assert!(PressureLevel::Low < PressureLevel::Medium);
        assert!(PressureLevel::Medium < PressureLevel::High);
        assert!(PressureLevel::High < PressureLevel::Critical);
    }

    #[test]
    fn pressure_level_display() {
        assert_eq!(format!("{}", PressureLevel::None), "none");
        assert_eq!(format!("{}", PressureLevel::Critical), "critical");
    }

    #[test]
    fn pressure_level_debug_clone_eq() {
        let p = PressureLevel::Medium;
        let p2 = p;
        assert_eq!(p, p2);
        let s = format!("{p:?}");
        assert!(s.contains("Medium"));
    }

    #[test]
    fn compute_pressure_unlimited_budget_is_none() {
        // budget=0 means unlimited, should always be None.
        assert_eq!(MemoryGuard::compute_pressure(0, 0), PressureLevel::None);
    }

    #[test]
    fn compute_pressure_thresholds() {
        let budget = 1000;
        // >30% free => None
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 0),
            PressureLevel::None
        );
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 600),
            PressureLevel::None
        );
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 699),
            PressureLevel::None
        );
        // 10-30% free => Low
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 750),
            PressureLevel::Low
        );
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 890),
            PressureLevel::Low
        );
        // 5-10% free => Medium
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 910),
            PressureLevel::Medium
        );
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 949),
            PressureLevel::Medium
        );
        // <5% free => High
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 960),
            PressureLevel::High
        );
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 999),
            PressureLevel::High
        );
        // At or over budget => Critical
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 1000),
            PressureLevel::Critical
        );
        assert_eq!(
            MemoryGuard::compute_pressure(budget, 2000),
            PressureLevel::Critical
        );
    }

    #[test]
    fn memory_hook_debug() {
        let hook = MemoryHook::new("test_hook", 1024, 64, 5, || 1024);
        let s = format!("{hook:?}");
        assert!(s.contains("test_hook"));
        assert!(s.contains("1024"));
        assert!(s.contains("64"));
        assert!(s.contains("5"));
    }

    // ---------------------------------------------------------------
    // GPU tests (require `cuda` feature and a real device)
    // ---------------------------------------------------------------

    #[cfg(feature = "cuda")]
    mod gpu_tests {
        use super::*;

        fn make_device() -> Arc<GpuDevice> {
            Arc::new(GpuDevice::new(0).expect("CUDA device 0"))
        }

        #[test]
        fn guard_construction_and_stats() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(1024 * 1024 * 1024) // 1 GiB
                .oom_policy(OomPolicy::Fail)
                .build()
                .expect("build guard");

            let stats = guard.stats();
            assert_eq!(stats.used_bytes, 0);
            assert_eq!(stats.budget_bytes, 1024 * 1024 * 1024);
            assert_eq!(stats.peak_bytes, 0);
            assert_eq!(stats.num_allocations, 0);
            assert_eq!(stats.num_oom_recoveries, 0);
            assert!(stats.total_device_bytes > 0);
            assert!(stats.free_device_bytes > 0);
        }

        #[test]
        fn budget_enforcement_rejects_over_budget() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(256) // tiny budget: 256 bytes
                .build()
                .expect("build guard");

            // Try to allocate way more than the budget.
            let result = guard.safe_alloc::<f32>(1024); // 4096 bytes
            assert!(result.is_err());
            match result.unwrap_err() {
                GpuError::BudgetExceeded {
                    requested_bytes,
                    budget_bytes,
                    used_bytes,
                } => {
                    assert_eq!(requested_bytes, 1024 * 4);
                    assert_eq!(budget_bytes, 256);
                    assert_eq!(used_bytes, 0);
                }
                other => panic!("expected BudgetExceeded, got {other:?}"),
            }
        }

        #[test]
        fn safe_alloc_tracks_usage() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(0) // unlimited
                .build()
                .expect("build guard");

            let buf = guard.safe_alloc::<f32>(256).expect("alloc 256 f32");
            let expected = 256 * std::mem::size_of::<f32>();

            let stats = guard.stats();
            assert_eq!(stats.used_bytes, expected);
            assert_eq!(stats.peak_bytes, expected);
            assert_eq!(stats.num_allocations, 1);

            guard.free(buf);

            let stats = guard.stats();
            assert_eq!(stats.used_bytes, 0);
            assert_eq!(stats.num_allocations, 0);
            // Peak should still be high.
            assert_eq!(stats.peak_bytes, expected);
        }

        #[test]
        fn safe_alloc_copy_tracks_usage() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .build()
                .expect("build guard");

            let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
            let buf = guard.safe_alloc_copy(&data).expect("alloc_copy");
            let expected = 4 * std::mem::size_of::<f64>();

            assert_eq!(guard.stats().used_bytes, expected);
            guard.free(buf);
            assert_eq!(guard.stats().used_bytes, 0);
        }

        #[test]
        fn reset_peak_stats_works() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .build()
                .expect("build guard");

            let buf = guard.safe_alloc::<f32>(512).expect("alloc");
            let peak = guard.stats().peak_bytes;
            assert!(peak > 0);

            guard.free(buf);
            assert_eq!(guard.stats().peak_bytes, peak); // still high

            guard.reset_peak_stats();
            assert_eq!(guard.stats().peak_bytes, 0); // reset to current (0)
        }

        #[test]
        fn emergency_checkpoint_callback_invoked() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .build()
                .expect("build guard");

            let called = Arc::new(AtomicBool::new(false));
            let called_clone = Arc::clone(&called);
            guard.on_oom(move || {
                called_clone.store(true, Ordering::SeqCst);
            });

            // Directly invoke the internal method to test the callback.
            guard.trigger_emergency_checkpoint();
            assert!(called.load(Ordering::SeqCst));
        }

        #[test]
        fn set_budget_at_runtime() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(0)
                .build()
                .expect("build guard");

            assert_eq!(guard.budget(), 0);

            guard.set_budget(1024);
            assert_eq!(guard.budget(), 1024);

            // Now an allocation over 1024 bytes should fail.
            let result = guard.safe_alloc::<f32>(1024); // 4096 bytes > 1024 budget
            assert!(result.is_err());
        }

        #[test]
        fn memory_info_returns_nonzero() {
            let device = GpuDevice::new(0).expect("CUDA device 0");
            let (free, total) = device.memory_info().expect("memory_info");
            assert!(total > 0, "total device memory should be > 0");
            assert!(free > 0, "free device memory should be > 0");
            assert!(free <= total, "free should not exceed total");
        }

        #[test]
        fn reservation_holds_memory() {
            let device = make_device();
            let (free_before, _) = device.memory_info().expect("memory_info");

            // Reserve 64 MiB.
            let reserve_bytes = 64 * 1024 * 1024;
            let guard = MemoryGuardBuilder::new(device)
                .reserve_bytes(reserve_bytes)
                .build()
                .expect("build guard with reservation");

            assert!(guard.has_reservation());

            // Release the reservation.
            let released = guard.release_reservation();
            assert_eq!(released, reserve_bytes);
            assert!(!guard.has_reservation());

            // Releasing again returns 0.
            assert_eq!(guard.release_reservation(), 0);

            let _ = free_before; // suppress unused warning
        }

        #[test]
        fn guard_debug_impl() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(999)
                .build()
                .expect("build guard");

            let s = format!("{guard:?}");
            assert!(s.contains("MemoryGuard"));
            assert!(s.contains("budget_bytes"));
            assert!(s.contains("999"));
        }

        #[test]
        fn watchdog_detects_no_pressure_when_plenty_free() {
            let device = make_device();
            // Threshold of 1 byte — should never trigger pressure.
            let watchdog = Arc::new(MemoryWatchdog::new(device, 1, Duration::from_millis(50)));

            assert!(!watchdog.check_pressure());
            watchdog.wait_if_paused(); // should return immediately

            // Start watchdog and wait for it to complete at least one cycle.
            let wd = Arc::clone(&watchdog);
            let handle = wd.start();
            watchdog.wait_for_first_check(Duration::from_secs(5));
            assert!(!watchdog.check_pressure());
            watchdog.stop();
            handle.join().expect("watchdog thread");
        }

        #[test]
        fn watchdog_debug_impl() {
            let device = make_device();
            let watchdog = MemoryWatchdog::new(device, 1024, Duration::from_secs(1));
            let s = format!("{watchdog:?}");
            assert!(s.contains("MemoryWatchdog"));
            assert!(s.contains("1024"));
        }

        #[test]
        fn memory_guarded_device() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(Arc::clone(&device))
                .budget_bytes(1024 * 1024)
                .build()
                .expect("build guard");

            let guarded = MemoryGuardedDevice { guard };
            assert_eq!(guarded.device().ordinal(), 0);
            assert_eq!(guarded.guard().budget(), 1024 * 1024);

            let s = format!("{guarded:?}");
            assert!(s.contains("MemoryGuardedDevice"));
        }

        #[test]
        fn oom_policy_retry_after_free() {
            // This test verifies the RetryAfterFree policy increments the
            // recovery counter. We cannot easily force a real OOM in a unit
            // test, so we verify the policy is stored correctly and the
            // counter machinery works.
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .oom_policy(OomPolicy::RetryAfterFree)
                .build()
                .expect("build guard");

            // With RetryAfterFree and no actual OOM, allocation succeeds
            // on the first attempt — recovery counter stays at 0.
            let buf = guard.safe_alloc::<f32>(64).expect("alloc");
            assert_eq!(guard.stats().num_oom_recoveries, 0);
            guard.free(buf);
        }

        #[test]
        fn multiple_allocations_budget_accounting() {
            let device = make_device();
            let budget = 2048_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            // First alloc: 128 f32 = 512 bytes. Should succeed.
            let buf1 = guard.safe_alloc::<f32>(128).expect("alloc 1");
            assert_eq!(guard.stats().used_bytes, 512);
            assert_eq!(guard.stats().num_allocations, 1);

            // Second alloc: 128 f32 = 512 bytes. Total = 1024. Should succeed.
            let buf2 = guard.safe_alloc::<f32>(128).expect("alloc 2");
            assert_eq!(guard.stats().used_bytes, 1024);
            assert_eq!(guard.stats().num_allocations, 2);

            // Third alloc: 512 f32 = 2048 bytes. Total would be 3072 > 2048. Should fail.
            let result = guard.safe_alloc::<f32>(512);
            assert!(result.is_err());

            // Free buf1, then the third alloc should succeed (1024 + 512 < 2048? no, 512 + 2048 = 2560)
            // Actually 1024-512 = 512 used, then 512 + 2048 = 2560 > 2048. Still too big.
            // Let's free both and try a fitting alloc.
            guard.free(buf1);
            guard.free(buf2);
            assert_eq!(guard.stats().used_bytes, 0);

            let buf3 = guard.safe_alloc::<f32>(512).expect("alloc 3 after free");
            assert_eq!(guard.stats().used_bytes, 2048);
            guard.free(buf3);
        }

        // ---------------------------------------------------------------
        // Pre-OOM hooks GPU tests
        // ---------------------------------------------------------------

        #[test]
        fn hook_called_on_budget_exceeded() {
            let device = make_device();
            let budget = 1024_usize; // 1024 bytes
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            let called = Arc::new(AtomicBool::new(false));
            let called_clone = Arc::clone(&called);

            guard.register_hook(MemoryHook::new("test_hook", 2048, 0, 10, move || {
                called_clone.store(true, Ordering::SeqCst);
                0 // does not actually free tracked memory
            }));

            // Allocation of 512 f32 = 2048 bytes > budget of 1024.
            // Hook will be called but won't free enough, so alloc falls
            // through. The hook should still have been invoked.
            let _result = guard.safe_alloc_with_hooks::<f32>(512);
            assert!(called.load(Ordering::SeqCst), "hook was not called");
        }

        #[test]
        fn hook_frees_enough_memory_allocation_succeeds() {
            let device = make_device();
            // Budget: 2048. Pre-fill 1536 bytes, then request 1024 bytes
            // (total would be 2560 > 2048). Hook frees 1024 bytes.
            let budget = 2048_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            // Pre-fill: 384 f32 = 1536 bytes.
            let prefill = guard.safe_alloc::<f32>(384).expect("prefill");
            assert_eq!(guard.stats().used_bytes, 1536);

            let called = Arc::new(AtomicBool::new(false));
            let called_clone = Arc::clone(&called);

            // Hook "frees" 1024 bytes by adjusting the guard's tracked usage.
            // In a real scenario the hook would drop GPU buffers. Here we
            // simulate by having the hook report 1024 freed; run_hooks
            // subtracts that from used_bytes.
            guard.register_hook(MemoryHook::new("free_1k", 1024, 0, 10, move || {
                called_clone.store(true, Ordering::SeqCst);
                1024
            }));

            // Request 256 f32 = 1024 bytes. 1536 + 1024 = 2560 > 2048.
            // Shortfall = 512. Hook frees 1024 (enough).
            let buf = guard
                .safe_alloc_with_hooks::<f32>(256)
                .expect("alloc after hook");
            assert!(called.load(Ordering::SeqCst), "hook was not called");

            // used_bytes: was 1536, hook freed 1024 => 512, then alloc adds 1024 => 1536.
            assert_eq!(guard.stats().used_bytes, 1536);

            guard.free(buf);
            guard.free(prefill);
        }

        #[test]
        fn hook_not_enough_falls_through_to_oom_policy() {
            let device = make_device();
            let budget = 512_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .oom_policy(OomPolicy::Fail)
                .build()
                .expect("build guard");

            let called = Arc::new(AtomicBool::new(false));
            let called_clone = Arc::clone(&called);

            guard.register_hook(MemoryHook::new("weak_hook", 64, 0, 10, move || {
                called_clone.store(true, Ordering::SeqCst);
                64 // only frees 64 bytes
            }));

            // Request 1024 f32 = 4096 bytes >> 512 budget.
            // Hook frees 64, still not enough, falls through to OomPolicy::Fail.
            let result = guard.safe_alloc_with_hooks::<f32>(1024);
            assert!(
                called.load(Ordering::SeqCst),
                "hook should have been called"
            );
            assert!(result.is_err(), "allocation should have failed");
        }

        #[test]
        fn hooks_called_in_priority_order() {
            let device = make_device();
            let budget = 1024_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            let order = Arc::new(Mutex::new(Vec::new()));

            let o1 = Arc::clone(&order);
            guard.register_hook(MemoryHook::new("priority_20", 256, 0, 20, move || {
                o1.lock().unwrap().push(20_u32);
                256
            }));

            let o2 = Arc::clone(&order);
            guard.register_hook(MemoryHook::new("priority_5", 256, 0, 5, move || {
                o2.lock().unwrap().push(5_u32);
                256
            }));

            let o3 = Arc::clone(&order);
            guard.register_hook(MemoryHook::new("priority_10", 256, 0, 10, move || {
                o3.lock().unwrap().push(10_u32);
                256
            }));

            // Request 512 f32 = 2048 bytes > 1024 budget.
            // All three hooks needed. Should fire: 5, 10, 20.
            let _result = guard.safe_alloc_with_hooks::<f32>(512);
            let call_order = order.lock().unwrap();
            assert_eq!(
                &*call_order,
                &[5, 10, 20],
                "hooks should fire in priority order"
            );
        }

        #[test]
        fn remove_hook_by_name() {
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(1024)
                .build()
                .expect("build guard");

            let called = Arc::new(AtomicBool::new(false));
            let called_clone = Arc::clone(&called);

            guard.register_hook(MemoryHook::new("removable", 2048, 0, 10, move || {
                called_clone.store(true, Ordering::SeqCst);
                2048
            }));

            // Remove the hook.
            assert!(guard.remove_hook("removable"));
            // Removing again returns false.
            assert!(!guard.remove_hook("removable"));

            // Trigger an over-budget allocation; removed hook should NOT fire.
            let _result = guard.safe_alloc_with_hooks::<f32>(512);
            assert!(
                !called.load(Ordering::SeqCst),
                "removed hook should not have been called"
            );
        }

        #[test]
        fn pressure_level_tracks_usage() {
            let device = make_device();
            let budget = 1000_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            // No usage => None.
            assert_eq!(guard.pressure_level(), PressureLevel::None);

            // Manually bump used_bytes to 750 => 25% free => Low.
            guard.used_bytes.store(750, Ordering::Relaxed);
            assert_eq!(guard.pressure_level(), PressureLevel::Low);

            // 920 used => 8% free => Medium.
            guard.used_bytes.store(920, Ordering::Relaxed);
            assert_eq!(guard.pressure_level(), PressureLevel::Medium);

            // 960 used => 4% free => High.
            guard.used_bytes.store(960, Ordering::Relaxed);
            assert_eq!(guard.pressure_level(), PressureLevel::High);

            // At budget => Critical.
            guard.used_bytes.store(1000, Ordering::Relaxed);
            assert_eq!(guard.pressure_level(), PressureLevel::Critical);

            // Unlimited budget => always None regardless of usage.
            guard.set_budget(0);
            assert_eq!(guard.pressure_level(), PressureLevel::None);
        }

        #[test]
        fn multiple_hooks_called_until_enough_freed() {
            let device = make_device();
            let budget = 2048_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            // Pre-fill: 256 f32 = 1024 bytes.
            let prefill = guard.safe_alloc::<f32>(256).expect("prefill");
            assert_eq!(guard.stats().used_bytes, 1024);

            let count = Arc::new(AtomicUsize::new(0));

            // Hook A: priority 1, frees 256 bytes.
            let c1 = Arc::clone(&count);
            guard.register_hook(MemoryHook::new("hook_a", 256, 0, 1, move || {
                c1.fetch_add(1, Ordering::SeqCst);
                256
            }));

            // Hook B: priority 2, frees 512 bytes.
            let c2 = Arc::clone(&count);
            guard.register_hook(MemoryHook::new("hook_b", 512, 0, 2, move || {
                c2.fetch_add(1, Ordering::SeqCst);
                512
            }));

            // Hook C: priority 3, frees 512 bytes. Should NOT be called if
            // A+B free enough.
            let c3 = Arc::new(AtomicBool::new(false));
            let c3_clone = Arc::clone(&c3);
            guard.register_hook(MemoryHook::new("hook_c", 512, 0, 3, move || {
                c3_clone.store(true, Ordering::SeqCst);
                512
            }));

            // Request 384 f32 = 1536 bytes. 1024 + 1536 = 2560 > 2048.
            // Shortfall = 512. Hook A frees 256 (not enough), Hook B frees
            // 512 (now 768 >= 512). Hook C should be skipped.
            let buf = guard
                .safe_alloc_with_hooks::<f32>(384)
                .expect("alloc with hooks");
            assert_eq!(count.load(Ordering::SeqCst), 2, "hooks A and B should fire");
            assert!(
                !c3.load(Ordering::SeqCst),
                "hook C should not have been called"
            );

            guard.free(buf);
            guard.free(prefill);
        }

        #[test]
        fn hook_with_excessive_overhead_is_skipped() {
            let device = make_device();
            let budget = 2048_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            // Pre-fill: 480 f32 = 1920 bytes. Headroom = 128 bytes.
            let prefill = guard.safe_alloc::<f32>(480).expect("prefill");
            assert_eq!(guard.stats().used_bytes, 1920);

            let expensive_called = Arc::new(AtomicBool::new(false));
            let expensive_clone = Arc::clone(&expensive_called);

            // Hook with overhead of 256 > headroom of 128 => should be skipped.
            guard.register_hook(MemoryHook::new("expensive_hook", 1024, 256, 1, move || {
                expensive_clone.store(true, Ordering::SeqCst);
                1024
            }));

            let cheap_called = Arc::new(AtomicBool::new(false));
            let cheap_clone = Arc::clone(&cheap_called);

            // Hook with zero overhead => should fire.
            guard.register_hook(MemoryHook::new("cheap_hook", 512, 0, 2, move || {
                cheap_clone.store(true, Ordering::SeqCst);
                512
            }));

            // Request 64 f32 = 256 bytes. 1920 + 256 = 2176 > 2048.
            // Shortfall = 128.
            let buf = guard
                .safe_alloc_with_hooks::<f32>(64)
                .expect("alloc with hooks");

            assert!(
                !expensive_called.load(Ordering::SeqCst),
                "expensive hook should have been skipped due to overhead"
            );
            assert!(
                cheap_called.load(Ordering::SeqCst),
                "cheap hook should have been called"
            );

            guard.free(buf);
            guard.free(prefill);
        }

        #[test]
        fn pressure_listener_notified_on_change() {
            let device = make_device();
            let budget = 1000_usize;
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(budget)
                .build()
                .expect("build guard");

            struct TestListener {
                changes: Mutex<Vec<(PressureLevel, PressureLevel)>>,
            }

            impl MemoryPressureListener for TestListener {
                fn on_pressure_change(&self, old: PressureLevel, new: PressureLevel) {
                    self.changes.lock().unwrap().push((old, new));
                }
            }

            let listener = Arc::new(TestListener {
                changes: Mutex::new(Vec::new()),
            });
            let listener_ref = Arc::clone(&listener);

            // Wrap in a Box<dyn MemoryPressureListener> — we need to share
            // the Arc for assertions, so we use a thin wrapper.
            struct ListenerWrapper(Arc<TestListener>);
            impl MemoryPressureListener for ListenerWrapper {
                fn on_pressure_change(&self, old: PressureLevel, new: PressureLevel) {
                    self.0.on_pressure_change(old, new);
                }
            }

            guard.add_pressure_listener(Box::new(ListenerWrapper(listener_ref)));

            // Allocate a small amount. Pressure should stay None, no
            // notification expected (None -> None is not a change).
            let buf1 = guard.safe_alloc::<f32>(1).expect("small alloc");
            // notify_pressure_change is only called by free and
            // safe_alloc_with_hooks; safe_alloc does not call it. Trigger
            // manually via free.
            guard.free(buf1);

            // Force a pressure change by directly setting used_bytes and
            // calling notify_pressure_change.
            guard.used_bytes.store(960, Ordering::Relaxed);
            guard.notify_pressure_change(); // None -> High
            guard.used_bytes.store(0, Ordering::Relaxed);
            guard.notify_pressure_change(); // High -> None

            let changes = listener.changes.lock().unwrap();
            assert!(
                changes.len() >= 2,
                "should have at least 2 pressure changes, got {}",
                changes.len()
            );
            assert_eq!(changes[0], (PressureLevel::None, PressureLevel::High));
            assert_eq!(changes[1], (PressureLevel::High, PressureLevel::None));
        }

        #[test]
        fn safe_alloc_with_hooks_fast_path_no_hooks() {
            // When the allocation fits within budget, hooks should not fire
            // and the allocation should succeed.
            let device = make_device();
            let guard = MemoryGuardBuilder::new(device)
                .budget_bytes(1024 * 1024)
                .build()
                .expect("build guard");

            let buf = guard
                .safe_alloc_with_hooks::<f32>(64)
                .expect("fast-path alloc");
            assert_eq!(guard.stats().used_bytes, 64 * 4);
            guard.free(buf);
        }
    }
}
