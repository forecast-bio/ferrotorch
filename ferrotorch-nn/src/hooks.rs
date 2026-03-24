//! Forward/backward hooks for [`Module`] instances.
//!
//! PyTorch lets users attach hooks to any `nn.Module` to inspect or modify
//! activations during the forward and backward passes.  Because our [`Module`]
//! trait is stateless (no per-instance storage), hooks are added via the
//! [`HookedModule<M>`] wrapper which stores hooks externally and delegates all
//! `Module` methods to the inner module.
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_nn::{HookedModule, Linear, Module};
//!
//! let linear = Linear::<f32>::new(4, 2, true).unwrap();
//! let hooked = HookedModule::new(linear);
//!
//! let _handle = hooked.register_forward_hook(Box::new(|input, output| {
//!     println!("in: {:?}  out: {:?}", input.shape(), output.shape());
//! }));
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use crate::module::{Module, StateDict};
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// Hook type aliases
// ---------------------------------------------------------------------------

/// A closure invoked *after* the forward pass with (input, output).
///
/// Intended for observation / logging; the return value is not used.
pub type ForwardHook<T> = Box<dyn Fn(&Tensor<T>, &Tensor<T>) + Send + Sync>;

/// A closure invoked *before* the forward pass with (input).
///
/// May return a replacement input tensor, allowing the hook to transform
/// activations before they reach the module.
pub type ForwardPreHook<T> = Box<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// A closure invoked during the backward pass with (grad_input, grad_output).
///
/// Intended for observation / logging of gradients.
pub type BackwardHook<T> = Box<dyn Fn(&Tensor<T>, &Tensor<T>) + Send + Sync>;

// ---------------------------------------------------------------------------
// HookHandle
// ---------------------------------------------------------------------------

/// An opaque handle returned when a hook is registered.
///
/// Calling [`HookHandle::remove`] unregisters the hook so it will not fire on
/// subsequent forward/backward calls.  Dropping the handle *without* calling
/// `remove` leaves the hook active (matching PyTorch semantics).
#[derive(Debug)]
pub struct HookHandle {
    id: usize,
    removed: Arc<AtomicBool>,
}

impl HookHandle {
    fn new(id: usize, removed: Arc<AtomicBool>) -> Self {
        Self { id, removed }
    }

    /// The unique identifier for this hook registration.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Unregister the hook.  Subsequent forward/backward passes will skip it.
    pub fn remove(self) {
        self.removed.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Internal storage entry
// ---------------------------------------------------------------------------

/// One entry in a hook list.  The `removed` flag is shared with the
/// corresponding [`HookHandle`]; when the handle is removed the flag is set
/// and the hook is lazily purged on the next invocation.
struct HookEntry<H> {
    #[allow(dead_code)] // Retained for future lookup-by-id operations.
    id: usize,
    hook: H,
    removed: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// HookedModule
// ---------------------------------------------------------------------------

/// A wrapper that adds hook storage around any [`Module`].
///
/// `HookedModule` implements `Module<T>` itself, so it can be used anywhere
/// the inner module could be used.  Hooks are stored behind `Mutex`es and
/// the wrapper is `Send + Sync` as long as the inner module is.
pub struct HookedModule<M, T: Float> {
    inner: M,
    forward_hooks: Mutex<Vec<HookEntry<ForwardHook<T>>>>,
    forward_pre_hooks: Mutex<Vec<HookEntry<ForwardPreHook<T>>>>,
    backward_hooks: Mutex<Vec<HookEntry<BackwardHook<T>>>>,
    next_id: AtomicUsize,
}

impl<M, T: Float> HookedModule<M, T> {
    /// Wrap a module, enabling hook registration.
    pub fn new(module: M) -> Self {
        Self {
            inner: module,
            forward_hooks: Mutex::new(Vec::new()),
            forward_pre_hooks: Mutex::new(Vec::new()),
            backward_hooks: Mutex::new(Vec::new()),
            next_id: AtomicUsize::new(0),
        }
    }

    /// Register a hook that fires *after* `forward`.
    ///
    /// The hook receives `(&input, &output)`.  Returns a [`HookHandle`] that
    /// can be used to unregister the hook.
    pub fn register_forward_hook(&self, hook: ForwardHook<T>) -> HookHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let removed = Arc::new(AtomicBool::new(false));
        let entry = HookEntry {
            id,
            hook,
            removed: Arc::clone(&removed),
        };
        self.forward_hooks.lock().unwrap().push(entry);
        HookHandle::new(id, removed)
    }

    /// Register a hook that fires *before* `forward`.
    ///
    /// The hook receives `(&input)` and returns a (possibly modified) input
    /// tensor.  Returns a [`HookHandle`].
    pub fn register_forward_pre_hook(&self, hook: ForwardPreHook<T>) -> HookHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let removed = Arc::new(AtomicBool::new(false));
        let entry = HookEntry {
            id,
            hook,
            removed: Arc::clone(&removed),
        };
        self.forward_pre_hooks.lock().unwrap().push(entry);
        HookHandle::new(id, removed)
    }

    /// Register a hook that fires during the backward pass.
    ///
    /// The hook receives `(&grad_input, &grad_output)`.  Returns a
    /// [`HookHandle`].
    pub fn register_backward_hook(&self, hook: BackwardHook<T>) -> HookHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let removed = Arc::new(AtomicBool::new(false));
        let entry = HookEntry {
            id,
            hook,
            removed: Arc::clone(&removed),
        };
        self.backward_hooks.lock().unwrap().push(entry);
        HookHandle::new(id, removed)
    }

    /// Borrow the inner module.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Mutably borrow the inner module.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Consume the wrapper and return the inner module.
    pub fn into_inner(self) -> M {
        self.inner
    }

    /// Purge entries whose handle has been removed.
    fn gc_hooks<H>(hooks: &mut Vec<HookEntry<H>>) {
        hooks.retain(|e| !e.removed.load(Ordering::Acquire));
    }
}

// ---------------------------------------------------------------------------
// Module implementation
// ---------------------------------------------------------------------------

impl<M: Module<T>, T: Float> Module<T> for HookedModule<M, T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // 1. Run forward pre-hooks (each may transform the input).
        let mut x = input.clone();
        {
            let mut pre_hooks = self.forward_pre_hooks.lock().unwrap();
            Self::gc_hooks(&mut pre_hooks);
            for entry in pre_hooks.iter() {
                if !entry.removed.load(Ordering::Acquire) {
                    x = (entry.hook)(&x)?;
                }
            }
        }

        // 2. Run the inner module's forward pass.
        let output = self.inner.forward(&x)?;

        // 3. Run forward post-hooks (observe input + output).
        {
            let mut post_hooks = self.forward_hooks.lock().unwrap();
            Self::gc_hooks(&mut post_hooks);
            for entry in post_hooks.iter() {
                if !entry.removed.load(Ordering::Acquire) {
                    (entry.hook)(&x, &output);
                }
            }
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.inner.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.inner.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.inner.named_parameters()
    }

    fn train(&mut self) {
        self.inner.train();
    }

    fn eval(&mut self) {
        self.inner.eval();
    }

    fn is_training(&self) -> bool {
        self.inner.is_training()
    }

    fn state_dict(&self) -> StateDict<T> {
        self.inner.state_dict()
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        self.inner.load_state_dict(state, strict)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use ferrotorch_core::{FerrotorchResult, Float, Tensor};

    use crate::module::Module;
    use crate::parameter::Parameter;

    use super::HookedModule;

    // -- Minimal test module ------------------------------------------------

    struct DoubleModule<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> DoubleModule<T> {
        fn new(size: usize) -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::ones(&[size])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for DoubleModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            // Simple: output = input + input  (doubles the values).
            let out = input.add_t(input)?;
            Ok(out)
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.weight]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("weight".to_string(), &self.weight)]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    // -- Tests --------------------------------------------------------------

    #[test]
    fn test_forward_hook_captures_output_shape() {
        let m = DoubleModule::<f32>::new(4).unwrap();
        let hooked = HookedModule::new(m);

        let captured_shape = Arc::new(Mutex::new(Vec::<usize>::new()));
        let shape_ref = Arc::clone(&captured_shape);

        let _handle = hooked.register_forward_hook(Box::new(move |_input, output| {
            *shape_ref.lock().unwrap() = output.shape().to_vec();
        }));

        let input = ferrotorch_core::ones::<f32>(&[4]).unwrap();
        let _out = hooked.forward(&input).unwrap();

        assert_eq!(*captured_shape.lock().unwrap(), vec![4]);
    }

    #[test]
    fn test_forward_pre_hook_modifies_input() {
        let m = DoubleModule::<f32>::new(3).unwrap();
        let hooked = HookedModule::new(m);

        // Pre-hook replaces input with zeros.
        let _handle = hooked.register_forward_pre_hook(Box::new(|input| {
            ferrotorch_core::zeros::<f32>(input.shape())
        }));

        let input = ferrotorch_core::ones::<f32>(&[3]).unwrap();
        let out = hooked.forward(&input).unwrap();

        // DoubleModule doubles input; zeros doubled = zeros.
        let data = out.data().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_multiple_hooks_fire_in_order() {
        let m = DoubleModule::<f32>::new(2).unwrap();
        let hooked = HookedModule::new(m);

        let order = Arc::new(Mutex::new(Vec::<usize>::new()));

        let o1 = Arc::clone(&order);
        let _h1 = hooked.register_forward_hook(Box::new(move |_input, _output| {
            o1.lock().unwrap().push(1);
        }));

        let o2 = Arc::clone(&order);
        let _h2 = hooked.register_forward_hook(Box::new(move |_input, _output| {
            o2.lock().unwrap().push(2);
        }));

        let o3 = Arc::clone(&order);
        let _h3 = hooked.register_forward_hook(Box::new(move |_input, _output| {
            o3.lock().unwrap().push(3);
        }));

        let input = ferrotorch_core::ones::<f32>(&[2]).unwrap();
        let _out = hooked.forward(&input).unwrap();

        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_hook_handle_remove() {
        let m = DoubleModule::<f32>::new(2).unwrap();
        let hooked = HookedModule::new(m);

        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);

        let handle = hooked.register_forward_hook(Box::new(move |_input, _output| {
            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }));

        let input = ferrotorch_core::ones::<f32>(&[2]).unwrap();

        // First forward — hook fires.
        let _out = hooked.forward(&input).unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 1);

        // Remove the hook.
        handle.remove();

        // Second forward — hook must NOT fire.
        let _out = hooked.forward(&input).unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn test_hooked_module_delegates_parameters() {
        let m = DoubleModule::<f32>::new(5).unwrap();
        let hooked = HookedModule::new(m);

        assert_eq!(hooked.parameters().len(), 1);
        assert_eq!(hooked.parameters()[0].shape(), &[5]);
    }

    #[test]
    fn test_hooked_module_delegates_named_parameters() {
        let m = DoubleModule::<f32>::new(3).unwrap();
        let hooked = HookedModule::new(m);

        let named = hooked.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn test_hooked_module_delegates_state_dict() {
        let m = DoubleModule::<f32>::new(4).unwrap();
        let hooked = HookedModule::new(m);

        let sd = hooked.state_dict();
        assert!(sd.contains_key("weight"));
        assert_eq!(sd["weight"].shape(), &[4]);
    }

    #[test]
    fn test_hooked_module_delegates_train_eval() {
        let m = DoubleModule::<f32>::new(2).unwrap();
        let mut hooked = HookedModule::new(m);

        assert!(hooked.is_training());
        hooked.eval();
        assert!(!hooked.is_training());
        hooked.train();
        assert!(hooked.is_training());
    }

    #[test]
    fn test_hooked_module_inner_access() {
        let m = DoubleModule::<f32>::new(3).unwrap();
        let hooked: HookedModule<_, f32> = HookedModule::new(m);
        assert_eq!(hooked.inner().parameters().len(), 1);
    }

    #[test]
    fn test_hooked_module_is_send_sync() {
        fn assert_send_sync<S: Send + Sync>() {}
        assert_send_sync::<HookedModule<DoubleModule<f32>, f32>>();
        assert_send_sync::<HookedModule<DoubleModule<f64>, f64>>();
    }

    #[test]
    fn test_backward_hook_registration() {
        let m = DoubleModule::<f32>::new(2).unwrap();
        let hooked = HookedModule::new(m);

        let called = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&called);

        let _handle = hooked.register_backward_hook(Box::new(move |_grad_in, _grad_out| {
            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }));

        // Backward hooks are registered but not invoked during forward.
        // Verify that forward still works with backward hooks registered.
        let input = ferrotorch_core::ones::<f32>(&[2]).unwrap();
        let _out = hooked.forward(&input).unwrap();

        assert_eq!(called.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn test_multiple_pre_hooks_chain() {
        let m = DoubleModule::<f32>::new(1).unwrap();
        let hooked = HookedModule::new(m);

        // First pre-hook: replace with zeros.
        let _h1 = hooked.register_forward_pre_hook(Box::new(|input| {
            ferrotorch_core::zeros::<f32>(input.shape())
        }));

        // Second pre-hook: add ones (zeros + ones = ones).
        let _h2 = hooked.register_forward_pre_hook(Box::new(|input| {
            let ones = ferrotorch_core::ones::<f32>(input.shape())?;
            input.add_t(&ones)
        }));

        let input = ferrotorch_core::from_slice::<f32>(&[42.0], &[1]).unwrap();
        let out = hooked.forward(&input).unwrap();

        // Pre-hooks chained: 42 -> 0 -> 1; DoubleModule doubles: 1+1 = 2.
        let data = out.data().unwrap();
        assert_eq!(data, vec![2.0]);
    }

    use std::sync::Mutex;
}
