use std::collections::HashMap;

use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::buffer::Buffer;
use crate::hooks::{BackwardHook, ForwardHook, ForwardPreHook, HookHandle, HookedModule};
use crate::parameter::Parameter;

/// A map from parameter names to tensors, used for serialization.
pub type StateDict<T> = HashMap<String, Tensor<T>>;

/// Reduction mode for loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Return the mean of all losses.
    Mean,
    /// Return the sum of all losses.
    Sum,
    /// Return the unreduced loss tensor.
    None,
}

/// The trait that all neural network layers implement.
///
/// Requires `Send + Sync` to match `Tensor<T>`'s thread-safety guarantees.
pub trait Module<T: Float>: Send + Sync {
    /// Forward pass. Takes input tensor, returns output tensor.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Iterate over all learnable parameters.
    fn parameters(&self) -> Vec<&Parameter<T>>;

    /// Iterate over all learnable parameters mutably.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>>;

    /// Named parameters for state dict serialization.
    ///
    /// Keys use dot-separated paths for nested modules
    /// (e.g., `"layer1.weight"`, `"layer1.bias"`).
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)>;

    /// Set training mode. Affects dropout, batchnorm, etc.
    fn train(&mut self);

    /// Set evaluation mode.
    fn eval(&mut self);

    /// Whether the module is in training mode.
    fn is_training(&self) -> bool;

    /// Move all parameters and buffers to a device.
    ///
    /// Default implementation iterates `parameters_mut()` and `buffers_mut()`
    /// and transfers each.
    fn to_device(&mut self, device: Device) -> FerrotorchResult<()> {
        for param in self.parameters_mut() {
            *param = param.to(device)?;
        }
        for buffer in self.buffers_mut() {
            *buffer = buffer.to(device)?;
        }
        Ok(())
    }

    /// Export parameters and buffers as a state dict (torch parity).
    ///
    /// Buffers are included alongside parameters since both are persistent
    /// module state. Keys are dot-separated paths.
    fn state_dict(&self) -> StateDict<T> {
        let mut out: StateDict<T> = self
            .named_parameters()
            .into_iter()
            .map(|(name, param)| (name, param.tensor().clone()))
            .collect();
        for (name, buffer) in self.named_buffers() {
            out.insert(name, buffer.tensor().clone());
        }
        out
    }

    // -----------------------------------------------------------------
    // Buffers — non-trainable persistent state. (#583)
    // -----------------------------------------------------------------

    /// Iterate over all non-trainable buffers (e.g. running mean / variance
    /// in BatchNorm). Default returns empty — concrete modules with buffers
    /// override.
    fn buffers(&self) -> Vec<&Buffer<T>> {
        Vec::new()
    }

    /// Mutable iteration over all buffers. Default returns empty.
    fn buffers_mut(&mut self) -> Vec<&mut Buffer<T>> {
        Vec::new()
    }

    /// Named buffers (dot-separated paths for nested modules). Default
    /// returns empty.
    fn named_buffers(&self) -> Vec<(String, &Buffer<T>)> {
        Vec::new()
    }

    // -----------------------------------------------------------------
    // Submodule iteration. (#583)
    // -----------------------------------------------------------------

    /// Direct child modules. Default returns empty (leaf module).
    fn children(&self) -> Vec<&dyn Module<T>> {
        Vec::new()
    }

    /// Direct child modules with their attribute names. Default returns
    /// empty.
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        Vec::new()
    }

    /// All modules in this subtree, depth-first (self first, then each
    /// child's descendants in order).
    ///
    /// Requires `Self: Sized` so we can coerce `self` to `&dyn Module<T>`.
    /// Trait-object callers can use [`Module::descendants_dyn`] (which yields
    /// descendants only) and prepend their own reference.
    fn modules(&self) -> Vec<&dyn Module<T>>
    where
        Self: Sized,
    {
        let mut out: Vec<&dyn Module<T>> = vec![self];
        out.extend(self.descendants_dyn());
        out
    }

    /// All strict descendants of `self` in depth-first order. Object-safe.
    fn descendants_dyn(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = Vec::new();
        for child in self.children() {
            out.push(child);
            out.extend(child.descendants_dyn());
        }
        out
    }

    /// All modules in this subtree with dot-separated path names. The root
    /// is named `""`; children paths are joined with `.`.
    fn named_modules(&self) -> Vec<(String, &dyn Module<T>)>
    where
        Self: Sized,
    {
        let mut out: Vec<(String, &dyn Module<T>)> = vec![(String::new(), self)];
        out.extend(self.named_descendants_dyn());
        out
    }

    /// Strict descendants with dot-paths. Object-safe.
    fn named_descendants_dyn(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> = Vec::new();
        for (name, child) in self.named_children() {
            out.push((name.clone(), child));
            for (sub_name, sub_module) in child.named_descendants_dyn() {
                let full = if sub_name.is_empty() {
                    name.clone()
                } else {
                    format!("{name}.{sub_name}")
                };
                out.push((full, sub_module));
            }
        }
        out
    }

    // -----------------------------------------------------------------
    // Helpers. (#583)
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // Hooks (#606)
    //
    // These consume `self` and return a [`HookedModule<Self, T>`] with the
    // requested hook already registered. Mirrors `torch.nn.Module
    // .register_*_hook(...)` ergonomically — callers no longer need to
    // wrap manually with `HookedModule::new(..)` first. Gated on
    // `Self: Sized` so the trait stays dyn-compatible.
    //
    // Named with the `with_*` prefix (rather than `register_*` directly) to
    // avoid clashing with `HookedModule`'s own inherent `register_*` methods,
    // which take `&self` and append a hook to an already-wrapped instance.
    // The two surfaces compose: `Linear::new(..)?.with_forward_hook(h1).0`
    // is a `HookedModule` that can `.register_forward_hook(h2)` again.
    // -----------------------------------------------------------------

    /// Wrap this module in a [`HookedModule`] and register a forward hook.
    /// Returns the wrapper paired with a [`HookHandle`] that can be used to
    /// remove the hook later. The wrapper implements `Module<T>` itself, so
    /// it slots into any place the original module did. Mirrors
    /// `torch.nn.Module.register_forward_hook`.
    fn with_forward_hook(self, hook: ForwardHook<T>) -> (HookedModule<Self, T>, HookHandle)
    where
        Self: Sized,
    {
        let wrapped = HookedModule::new(self);
        let handle = wrapped.register_forward_hook(hook);
        (wrapped, handle)
    }

    /// Wrap this module in a [`HookedModule`] and register a forward
    /// pre-hook. See [`Self::with_forward_hook`]. Mirrors
    /// `torch.nn.Module.register_forward_pre_hook`.
    fn with_forward_pre_hook(self, hook: ForwardPreHook<T>) -> (HookedModule<Self, T>, HookHandle)
    where
        Self: Sized,
    {
        let wrapped = HookedModule::new(self);
        let handle = wrapped.register_forward_pre_hook(hook);
        (wrapped, handle)
    }

    /// Wrap this module in a [`HookedModule`] and register a backward hook.
    /// See [`Self::with_forward_hook`]. Mirrors
    /// `torch.nn.Module.register_backward_hook`.
    fn with_backward_hook(self, hook: BackwardHook<T>) -> (HookedModule<Self, T>, HookHandle)
    where
        Self: Sized,
    {
        let wrapped = HookedModule::new(self);
        let handle = wrapped.register_backward_hook(hook);
        (wrapped, handle)
    }

    /// Set the gradient of every parameter to `None`.
    ///
    /// Equivalent to calling `tensor.zero_grad()` on each parameter's
    /// underlying tensor. Mirrors `torch.nn.Module.zero_grad`.
    fn zero_grad(&self) -> FerrotorchResult<()> {
        for param in self.parameters() {
            param.tensor().zero_grad()?;
        }
        Ok(())
    }

    /// Toggle `requires_grad` on every parameter (freeze / unfreeze the
    /// module). Mirrors `torch.nn.Module.requires_grad_`.
    fn requires_grad_(&mut self, requires_grad: bool) {
        for param in self.parameters_mut() {
            param.set_requires_grad(requires_grad);
        }
    }

    /// Apply a function to every parameter in this module. Mirrors
    /// `torch.nn.Module.apply` for the parameter case (true `apply` recurses
    /// over all submodules; the recursive form requires `&mut dyn Module`
    /// which conflicts with this trait's `&mut self` borrow).
    ///
    /// Takes `&mut dyn FnMut(...)` (rather than a generic closure) so the
    /// trait stays dyn-compatible — `Box<dyn Module<T>>` is a common usage
    /// pattern.
    fn apply_to_parameters(&mut self, f: &mut dyn FnMut(&mut Parameter<T>)) {
        for param in self.parameters_mut() {
            f(param);
        }
    }

    /// Load parameters from a state dict.
    ///
    /// When `strict` is `true` (default), unexpected keys are an error.
    /// When `false`, unexpected keys are silently ignored and missing
    /// keys leave existing parameter values unchanged.
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        // Known keys: union of parameter and buffer paths.
        let mut known_keys: std::collections::HashSet<String> = self
            .named_parameters()
            .iter()
            .map(|(k, _)| k.clone())
            .collect();
        for (k, _) in self.named_buffers() {
            known_keys.insert(k);
        }

        if strict {
            for key in state.keys() {
                if !known_keys.contains(key) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in state_dict: \"{key}\""),
                    });
                }
            }
        }

        // We need mutable access to parameters. Use named_parameters to get
        // the mapping, then parameters_mut to actually update.
        // This two-pass approach avoids borrowing issues.
        let param_names: Vec<String> = self
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();

        let params_mut = self.parameters_mut();

        for (name, param) in param_names.iter().zip(params_mut) {
            if let Some(tensor) = state.get(name) {
                if param.shape() != tensor.shape() {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "state_dict shape mismatch for \"{name}\": expected {:?}, got {:?}",
                            param.shape(),
                            tensor.shape()
                        ),
                    });
                }
                // Replace the parameter data with the loaded tensor.
                *param = Parameter::new(tensor.clone());
            } else if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("missing key in state_dict: \"{name}\""),
                });
            }
        }

        // Same dance for buffers.
        let buffer_names: Vec<String> = self
            .named_buffers()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        let buffers_mut = self.buffers_mut();
        for (name, buf) in buffer_names.iter().zip(buffers_mut) {
            if let Some(tensor) = state.get(name) {
                if buf.shape() != tensor.shape() {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "state_dict shape mismatch for buffer \"{name}\": expected {:?}, got {:?}",
                            buf.shape(),
                            tensor.shape()
                        ),
                    });
                }
                *buf = Buffer::new(tensor.clone());
            } else if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("missing buffer key in state_dict: \"{name}\""),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A minimal test module with one parameter.
    struct SimpleModule<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> SimpleModule<T> {
        fn new(size: usize) -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::zeros(&[size])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for SimpleModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            // Just return input for testing.
            Ok(input.clone())
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

    #[test]
    fn test_module_parameters() {
        let m = SimpleModule::<f32>::new(5).unwrap();
        assert_eq!(m.parameters().len(), 1);
        assert_eq!(m.parameters()[0].shape(), &[5]);
    }

    #[test]
    fn test_module_named_parameters() {
        let m = SimpleModule::<f32>::new(3).unwrap();
        let named = m.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn test_module_train_eval() {
        let mut m = SimpleModule::<f32>::new(2).unwrap();
        assert!(m.is_training());
        m.eval();
        assert!(!m.is_training());
        m.train();
        assert!(m.is_training());
    }

    #[test]
    fn test_module_state_dict_roundtrip() {
        let m = SimpleModule::<f32>::new(4).unwrap();
        let sd = m.state_dict();
        assert!(sd.contains_key("weight"));
        assert_eq!(sd["weight"].shape(), &[4]);

        let mut m2 = SimpleModule::<f32>::new(4).unwrap();
        m2.load_state_dict(&sd, true).unwrap();
    }

    #[test]
    fn test_module_state_dict_strict_extra_key() {
        let mut m = SimpleModule::<f32>::new(3).unwrap();
        let mut sd = HashMap::new();
        sd.insert(
            "weight".to_string(),
            ferrotorch_core::zeros::<f32>(&[3]).unwrap(),
        );
        sd.insert(
            "extra".to_string(),
            ferrotorch_core::zeros::<f32>(&[1]).unwrap(),
        );

        assert!(m.load_state_dict(&sd, true).is_err());
        assert!(m.load_state_dict(&sd, false).is_ok());
    }

    #[test]
    fn test_module_state_dict_shape_mismatch() {
        let mut m = SimpleModule::<f32>::new(3).unwrap();
        let mut sd = HashMap::new();
        sd.insert(
            "weight".to_string(),
            ferrotorch_core::zeros::<f32>(&[5]).unwrap(),
        );

        assert!(m.load_state_dict(&sd, true).is_err());
    }

    #[test]
    fn test_module_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SimpleModule<f32>>();
    }

    #[test]
    fn test_reduction_enum() {
        assert_eq!(Reduction::Mean, Reduction::Mean);
        assert_ne!(Reduction::Mean, Reduction::Sum);
    }

    #[test]
    fn test_to_device_cpu_preserves_weights() {
        let mut m = SimpleModule::<f32>::new(4).unwrap();
        m.to_device(ferrotorch_core::Device::Cpu).unwrap();
        assert_eq!(m.parameters().len(), 1);
        assert_eq!(m.parameters()[0].shape(), &[4]);
    }

    #[test]
    fn test_to_device_cuda_without_backend() {
        let mut m = SimpleModule::<f32>::new(3).unwrap();
        let result = m.to_device(ferrotorch_core::Device::Cuda(0));
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Module trait additions: buffers / children / zero_grad / requires_grad_ /
    // apply_to_parameters / modules iteration. (#583)
    // -----------------------------------------------------------------------

    /// A module with one parameter, one buffer, and a child.
    struct ParentModule<T: Float> {
        weight: Parameter<T>,
        running_mean: Buffer<T>,
        child: SimpleModule<T>,
    }

    impl<T: Float> ParentModule<T> {
        fn new() -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::ones(&[2, 2])?,
                running_mean: Buffer::zeros(&[2])?,
                child: SimpleModule::new(3)?,
            })
        }
    }

    impl<T: Float> Module<T> for ParentModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            self.child.forward(input)
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            // self.weight + child.parameters()
            let mut out: Vec<&Parameter<T>> = vec![&self.weight];
            out.extend(self.child.parameters());
            out
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            let mut out: Vec<&mut Parameter<T>> = vec![&mut self.weight];
            out.extend(self.child.parameters_mut());
            out
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            let mut out: Vec<(String, &Parameter<T>)> = vec![("weight".to_string(), &self.weight)];
            for (n, p) in self.child.named_parameters() {
                out.push((format!("child.{n}"), p));
            }
            out
        }

        fn buffers(&self) -> Vec<&Buffer<T>> {
            vec![&self.running_mean]
        }

        fn buffers_mut(&mut self) -> Vec<&mut Buffer<T>> {
            vec![&mut self.running_mean]
        }

        fn named_buffers(&self) -> Vec<(String, &Buffer<T>)> {
            vec![("running_mean".to_string(), &self.running_mean)]
        }

        fn children(&self) -> Vec<&dyn Module<T>> {
            vec![&self.child]
        }

        fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
            vec![("child".to_string(), &self.child)]
        }

        fn train(&mut self) {
            self.child.train();
        }

        fn eval(&mut self) {
            self.child.eval();
        }

        fn is_training(&self) -> bool {
            self.child.is_training()
        }
    }

    #[test]
    fn module_buffers_default_is_empty() {
        // SimpleModule doesn't override buffers() — default impl returns empty.
        let m = SimpleModule::<f32>::new(3).unwrap();
        assert!(m.buffers().is_empty());
        assert!(m.named_buffers().is_empty());
    }

    #[test]
    fn module_buffers_listed_for_overriding_module() {
        let m = ParentModule::<f32>::new().unwrap();
        assert_eq!(m.buffers().len(), 1);
        assert_eq!(m.buffers()[0].shape(), &[2]);
        let nb = m.named_buffers();
        assert_eq!(nb.len(), 1);
        assert_eq!(nb[0].0, "running_mean");
    }

    #[test]
    fn module_children_listed_for_parent() {
        let m = ParentModule::<f32>::new().unwrap();
        assert_eq!(m.children().len(), 1);
        assert_eq!(m.named_children().len(), 1);
        assert_eq!(m.named_children()[0].0, "child");
    }

    #[test]
    fn module_named_modules_includes_self_and_descendants() {
        let m = ParentModule::<f32>::new().unwrap();
        let nm = m.named_modules();
        // Root + 1 child = 2 entries.
        assert_eq!(nm.len(), 2);
        assert_eq!(nm[0].0, "");
        assert_eq!(nm[1].0, "child");
    }

    #[test]
    fn module_modules_includes_self_and_descendants() {
        let m = ParentModule::<f32>::new().unwrap();
        let mods = m.modules();
        assert_eq!(mods.len(), 2);
    }

    #[test]
    fn module_zero_grad_succeeds() {
        // No grads yet on a fresh module — zero_grad should still succeed.
        let m = SimpleModule::<f32>::new(3).unwrap();
        m.zero_grad().unwrap();
    }

    #[test]
    fn module_requires_grad_toggles_all_parameters() {
        let mut m = ParentModule::<f32>::new().unwrap();
        for p in m.parameters() {
            assert!(p.requires_grad());
        }
        m.requires_grad_(false);
        for p in m.parameters() {
            assert!(!p.requires_grad());
        }
        m.requires_grad_(true);
        for p in m.parameters() {
            assert!(p.requires_grad());
        }
    }

    #[test]
    fn module_apply_to_parameters_visits_all() {
        let mut m = ParentModule::<f32>::new().unwrap();
        let n_params = m.parameters().len();
        let mut count = 0;
        m.apply_to_parameters(&mut |_p| count += 1);
        assert_eq!(count, n_params);
    }

    #[test]
    fn module_state_dict_includes_buffers() {
        let m = ParentModule::<f32>::new().unwrap();
        let sd = m.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("running_mean"));
        assert!(sd.contains_key("child.weight"));
        assert_eq!(sd.len(), 3);
    }

    #[test]
    fn module_load_state_dict_with_buffer() {
        let mut m = ParentModule::<f32>::new().unwrap();
        let mut sd: StateDict<f32> = HashMap::new();
        sd.insert(
            "weight".into(),
            ferrotorch_core::ones::<f32>(&[2, 2]).unwrap(),
        );
        sd.insert(
            "running_mean".into(),
            ferrotorch_core::from_slice::<f32>(&[7.0, 9.0], &[2]).unwrap(),
        );
        sd.insert(
            "child.weight".into(),
            ferrotorch_core::zeros::<f32>(&[3]).unwrap(),
        );
        m.load_state_dict(&sd, true).unwrap();
        assert_eq!(m.buffers()[0].data().unwrap(), &[7.0, 9.0]);
    }

    #[test]
    fn module_descendants_dyn_excludes_self() {
        let m = ParentModule::<f32>::new().unwrap();
        let d = m.descendants_dyn();
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn module_named_descendants_dyn_paths() {
        let m = ParentModule::<f32>::new().unwrap();
        let nd = m.named_descendants_dyn();
        assert_eq!(nd.len(), 1);
        assert_eq!(nd[0].0, "child");
    }

    // -------------------------------------------------------------------
    // Hook-registration trait methods (#606)
    // -------------------------------------------------------------------

    #[test]
    fn with_forward_hook_wraps_and_fires() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let m = SimpleModule::<f32>::new(2).unwrap();
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_for_hook = std::sync::Arc::clone(&counter);

        let (wrapped, _handle) = m.with_forward_hook(Box::new(move |_input, _output| {
            counter_for_hook.fetch_add(1, Ordering::SeqCst);
        }));

        let input = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![1.0_f32, 2.0]),
            vec![2],
            false,
        )
        .unwrap();
        let _ = wrapped.forward(&input).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn with_forward_pre_hook_wraps_and_fires() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let m = SimpleModule::<f32>::new(2).unwrap();
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_for_hook = std::sync::Arc::clone(&counter);

        let (wrapped, _handle) = m.with_forward_pre_hook(Box::new(move |input| {
            counter_for_hook.fetch_add(1, Ordering::SeqCst);
            Ok(input.clone())
        }));

        let input = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![1.0_f32, 2.0]),
            vec![2],
            false,
        )
        .unwrap();
        let _ = wrapped.forward(&input).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn with_backward_hook_returns_handle() {
        // backward hook fires only on the backward pass; just verify the
        // wrapping API resolves and returns a usable HookedModule + handle.
        let m = SimpleModule::<f32>::new(2).unwrap();
        let (wrapped, handle) = m.with_backward_hook(Box::new(|_gi, _go| {}));
        // Wrapper still implements Module<T> trait — slot it into a forward
        // call to confirm it round-trips.
        let input = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![3.0_f32]),
            vec![1],
            false,
        )
        .unwrap();
        let _ = wrapped.forward(&input).unwrap();
        // Handle is droppable; explicit remove is also fine.
        handle.remove();
    }
}
