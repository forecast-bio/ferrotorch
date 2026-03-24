// [CL-311] Autograd hooks: gradient hooks and post-accumulate hooks for tensors.
//
// Provides `register_hook` (called when gradient is computed for a tensor)
// and `register_post_accumulate_grad_hook` (called after gradient accumulation
// on leaf tensors). Each registration returns a `HookHandle` that can be used
// to remove the hook later.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

/// Type alias for a gradient hook function: receives a gradient tensor and
/// optionally returns a replacement.
pub(crate) type GradHookFn<T> = Box<dyn Fn(&Tensor<T>) -> Option<Tensor<T>> + Send + Sync>;

/// Type alias for a post-accumulate-grad hook function.
pub(crate) type PostAccumulateHookFn<T> = Box<dyn Fn(&Tensor<T>) + Send + Sync>;

/// Monotonically increasing handle counter for hook identification.
static NEXT_HOOK_ID: AtomicU64 = AtomicU64::new(0);

/// An opaque handle returned by `register_hook` / `register_post_accumulate_grad_hook`.
///
/// Used to remove the hook via `Tensor::remove_hook`. Handles are unique across
/// all tensors — two hooks on different tensors will never share a handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookHandle(u64);

impl HookHandle {
    fn next() -> Self {
        Self(NEXT_HOOK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A gradient hook: receives the computed gradient and returns either `None`
/// (keep original gradient) or `Some(new_grad)` (replace the gradient).
///
/// This matches PyTorch's `register_hook` semantics: the hook can inspect
/// and optionally modify the gradient flowing through the backward pass.
pub(crate) struct GradHook<T: Float> {
    pub handle: HookHandle,
    pub func: GradHookFn<T>,
}

/// A post-accumulate-grad hook: called after gradient has been accumulated
/// onto a leaf tensor. Receives a reference to the tensor itself (so the
/// hook can read `.grad()`). Cannot modify the gradient — use `register_hook`
/// for that.
pub(crate) struct PostAccumulateGradHook<T: Float> {
    pub handle: HookHandle,
    pub func: PostAccumulateHookFn<T>,
}

/// Storage for hooks attached to a tensor.
///
/// Wrapped in a `Mutex` inside `TensorInner` for thread safety.
/// The `Vec`s are expected to be short (typically 0-2 hooks per tensor),
/// so linear scan for removal is fine.
pub(crate) struct HookStorage<T: Float> {
    pub grad_hooks: Vec<GradHook<T>>,
    pub post_accumulate_hooks: Vec<PostAccumulateGradHook<T>>,
}

impl<T: Float> HookStorage<T> {
    pub fn new() -> Self {
        Self {
            grad_hooks: Vec::new(),
            post_accumulate_hooks: Vec::new(),
        }
    }

    /// Register a gradient hook, returning its handle.
    pub fn add_grad_hook<F>(&mut self, func: F) -> HookHandle
    where
        F: Fn(&Tensor<T>) -> Option<Tensor<T>> + Send + Sync + 'static,
    {
        let handle = HookHandle::next();
        self.grad_hooks.push(GradHook {
            handle,
            func: Box::new(func),
        });
        handle
    }

    /// Register a post-accumulate-grad hook, returning its handle.
    pub fn add_post_accumulate_hook<F>(&mut self, func: F) -> HookHandle
    where
        F: Fn(&Tensor<T>) + Send + Sync + 'static,
    {
        let handle = HookHandle::next();
        self.post_accumulate_hooks.push(PostAccumulateGradHook {
            handle,
            func: Box::new(func),
        });
        handle
    }

    /// Remove a hook (either kind) by handle. Returns `true` if found.
    pub fn remove(&mut self, handle: HookHandle) -> bool {
        let before = self.grad_hooks.len() + self.post_accumulate_hooks.len();
        self.grad_hooks.retain(|h| h.handle != handle);
        self.post_accumulate_hooks.retain(|h| h.handle != handle);
        let after = self.grad_hooks.len() + self.post_accumulate_hooks.len();
        after < before
    }

    /// Returns `true` if there are any gradient hooks.
    pub fn has_grad_hooks(&self) -> bool {
        !self.grad_hooks.is_empty()
    }

    /// Returns `true` if there are any post-accumulate hooks.
    pub fn has_post_accumulate_hooks(&self) -> bool {
        !self.post_accumulate_hooks.is_empty()
    }
}

/// Run all gradient hooks on a tensor, returning the (possibly modified) gradient.
///
/// Each hook receives the current gradient and may return `Some(new_grad)` to
/// replace it. Hooks run in registration order; each sees the output of the
/// previous hook.
pub(crate) fn run_grad_hooks<T: Float>(
    hooks: &Mutex<HookStorage<T>>,
    grad: Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let guard = hooks.lock().map_err(|e| FerrotorchError::LockPoisoned {
        message: format!("hook storage mutex: {e}"),
    })?;
    let mut current = grad;
    for hook in &guard.grad_hooks {
        if let Some(replacement) = (hook.func)(&current) {
            current = replacement;
        }
    }
    Ok(current)
}

/// Run all post-accumulate-grad hooks on a tensor.
///
/// Called after gradient accumulation completes on a leaf tensor.
pub(crate) fn run_post_accumulate_hooks<T: Float>(
    hooks: &Mutex<HookStorage<T>>,
    tensor: &Tensor<T>,
) -> FerrotorchResult<()> {
    let guard = hooks.lock().map_err(|e| FerrotorchError::LockPoisoned {
        message: format!("hook storage mutex: {e}"),
    })?;
    for hook in &guard.post_accumulate_hooks {
        (hook.func)(tensor);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn scalar(val: f32) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false).unwrap()
    }

    #[test]
    fn test_hook_handle_uniqueness() {
        let h1 = HookHandle::next();
        let h2 = HookHandle::next();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hook_storage_add_remove() {
        let storage: Mutex<HookStorage<f32>> = Mutex::new(HookStorage::new());
        let handle = {
            let mut guard = storage.lock().unwrap();
            guard.add_grad_hook(|_g| None)
        };
        assert!(storage.lock().unwrap().has_grad_hooks());
        assert!(storage.lock().unwrap().remove(handle));
        assert!(!storage.lock().unwrap().has_grad_hooks());
    }

    #[test]
    fn test_run_grad_hooks_passthrough() {
        let storage: Mutex<HookStorage<f32>> = Mutex::new(HookStorage::new());
        let grad = scalar(3.0);
        let result = run_grad_hooks(&storage, grad).unwrap();
        assert!((result.item().unwrap() - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_run_grad_hooks_replace() {
        let storage: Mutex<HookStorage<f32>> = Mutex::new(HookStorage::new());
        {
            let mut guard = storage.lock().unwrap();
            guard.add_grad_hook(|_g| {
                Some(Tensor::from_storage(TensorStorage::cpu(vec![99.0]), vec![], false).unwrap())
            });
        }
        let grad = scalar(3.0);
        let result = run_grad_hooks(&storage, grad).unwrap();
        assert!((result.item().unwrap() - 99.0).abs() < 1e-7);
    }

    #[test]
    fn test_run_grad_hooks_chain() {
        // Two hooks: first doubles, second adds 1.
        let storage: Mutex<HookStorage<f32>> = Mutex::new(HookStorage::new());
        {
            let mut guard = storage.lock().unwrap();
            guard.add_grad_hook(|g| {
                let v = g.item().unwrap() * 2.0;
                Some(Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![], false).unwrap())
            });
            guard.add_grad_hook(|g| {
                let v = g.item().unwrap() + 1.0;
                Some(Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![], false).unwrap())
            });
        }
        let grad = scalar(5.0);
        let result = run_grad_hooks(&storage, grad).unwrap();
        // 5 * 2 = 10, 10 + 1 = 11
        assert!((result.item().unwrap() - 11.0).abs() < 1e-7);
    }

    #[test]
    fn test_post_accumulate_hook_fires() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        let storage: Mutex<HookStorage<f32>> = Mutex::new(HookStorage::new());
        {
            let mut guard = storage.lock().unwrap();
            guard.add_post_accumulate_hook(move |_t| {
                fired_clone.store(true, Ordering::Relaxed);
            });
        }
        let t = scalar(1.0);
        run_post_accumulate_hooks(&storage, &t).unwrap();
        assert!(fired.load(Ordering::Relaxed));
    }

    #[test]
    fn test_remove_nonexistent_handle() {
        let storage: Mutex<HookStorage<f32>> = Mutex::new(HookStorage::new());
        let fake = HookHandle(999_999);
        assert!(!storage.lock().unwrap().remove(fake));
    }
}
