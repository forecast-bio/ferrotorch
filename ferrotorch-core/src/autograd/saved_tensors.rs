//! Saved-tensors hooks for memory offloading in autograd.
//!
//! When a `GradFn` saves tensors for the backward pass (e.g., inputs needed
//! to compute gradients), those tensors consume GPU memory for the entire
//! training iteration. **Saved-tensors hooks** let users intercept the
//! save/restore cycle to offload tensors to CPU, compress them, or apply
//! any custom transformation.
//!
//! # Usage
//!
//! ```ignore
//! use ferrotorch_core::autograd::saved_tensors::saved_tensors_hooks;
//!
//! // Offload saved tensors to CPU during forward, reload during backward:
//! saved_tensors_hooks(
//!     |t| t.cpu(),                  // pack: move to CPU
//!     |t| t.to(Device::Cuda(0)),    // unpack: move back to GPU
//!     || {
//!         let y = model.forward(&x)?;
//!         y.backward()
//!     },
//! )?;
//! ```
//!
//! Hooks are thread-local and nestable (inner scopes override outer ones).

use std::cell::RefCell;
use std::sync::Arc;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::tensor::Tensor;

/// A pack hook transforms a tensor when it is saved for backward.
pub type PackHook<T> = Arc<dyn Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// An unpack hook transforms a tensor when it is retrieved during backward.
pub type UnpackHook<T> = Arc<dyn Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

// Thread-local saved-tensors hook state for f32.
thread_local! {
    static HOOKS_F32: RefCell<Option<(PackHook<f32>, UnpackHook<f32>)>> =
        const { RefCell::new(None) };
}

// Thread-local saved-tensors hook state for f64.
thread_local! {
    static HOOKS_F64: RefCell<Option<(PackHook<f64>, UnpackHook<f64>)>> =
        const { RefCell::new(None) };
}

/// Run a closure with saved-tensors hooks active on the current thread.
///
/// The `pack` hook is called on every tensor saved for backward during `f()`.
/// The `unpack` hook is called when those tensors are accessed in the backward
/// pass. Hooks are restored (or cleared) when this function returns.
///
/// Hooks are nestable — inner calls override outer hooks for their scope.
pub fn saved_tensors_hooks<T, F, R>(
    pack: impl Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
    unpack: impl Fn(Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
    f: F,
) -> FerrotorchResult<R>
where
    T: Float,
    F: FnOnce() -> FerrotorchResult<R>,
{
    let pack = Arc::new(pack) as PackHook<T>;
    let unpack = Arc::new(unpack) as UnpackHook<T>;

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // SAFETY: T is f32, transmute the Arc types.
        let pack_f32: PackHook<f32> = unsafe { std::mem::transmute(pack) };
        let unpack_f32: UnpackHook<f32> = unsafe { std::mem::transmute(unpack) };

        let prev = HOOKS_F32.with(|h| h.borrow_mut().replace((pack_f32, unpack_f32)));
        let result = f();
        HOOKS_F32.with(|h| *h.borrow_mut() = prev);
        result
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let pack_f64: PackHook<f64> = unsafe { std::mem::transmute(pack) };
        let unpack_f64: UnpackHook<f64> = unsafe { std::mem::transmute(unpack) };

        let prev = HOOKS_F64.with(|h| h.borrow_mut().replace((pack_f64, unpack_f64)));
        let result = f();
        HOOKS_F64.with(|h| *h.borrow_mut() = prev);
        result
    } else {
        // No hooks for other types — just run the closure.
        f()
    }
}

/// Apply the current pack hook to a tensor (if one is active).
///
/// Called by `GradFn` constructors when saving tensors for backward.
/// Returns the tensor unchanged if no hooks are active.
pub fn pack_saved_tensor<T: Float>(tensor: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        HOOKS_F32.with(|h| {
            let guard = h.borrow();
            if let Some((ref pack, _)) = *guard {
                // SAFETY: T is f32.
                let t_f32: Tensor<f32> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f32>>(tensor) };
                let result = pack(t_f32)?;
                Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        HOOKS_F64.with(|h| {
            let guard = h.borrow();
            if let Some((ref pack, _)) = *guard {
                let t_f64: Tensor<f64> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f64>>(tensor) };
                let result = pack(t_f64)?;
                Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else {
        Ok(tensor)
    }
}

/// Apply the current unpack hook to a tensor (if one is active).
///
/// Called during backward when a saved tensor is accessed.
/// Returns the tensor unchanged if no hooks are active.
pub fn unpack_saved_tensor<T: Float>(tensor: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        HOOKS_F32.with(|h| {
            let guard = h.borrow();
            if let Some((_, ref unpack)) = *guard {
                let t_f32: Tensor<f32> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f32>>(tensor) };
                let result = unpack(t_f32)?;
                Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        HOOKS_F64.with(|h| {
            let guard = h.borrow();
            if let Some((_, ref unpack)) = *guard {
                let t_f64: Tensor<f64> =
                    unsafe { std::mem::transmute::<Tensor<T>, Tensor<f64>>(tensor) };
                let result = unpack(t_f64)?;
                Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
            } else {
                Ok(tensor)
            }
        })
    } else {
        Ok(tensor)
    }
}

/// Returns `true` if saved-tensors hooks are currently active on this thread.
pub fn has_saved_tensor_hooks() -> bool {
    HOOKS_F32.with(|h| h.borrow().is_some()) || HOOKS_F64.with(|h| h.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    #[test]
    fn test_pack_unpack_identity() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], false)
            .unwrap();

        // No hooks active — pack/unpack are identity.
        let packed = pack_saved_tensor(t.clone()).unwrap();
        assert_eq!(packed.data_vec().unwrap(), vec![1.0, 2.0, 3.0]);

        let unpacked = unpack_saved_tensor(packed).unwrap();
        assert_eq!(unpacked.data_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_saved_tensors_hooks_transform() {
        let result = saved_tensors_hooks(
            |t: Tensor<f32>| {
                // Pack: multiply by 2
                let data: Vec<f32> = t.data_vec()?.iter().map(|&x| x * 2.0).collect();
                Tensor::from_storage(TensorStorage::cpu(data), t.shape().to_vec(), false)
            },
            |t: Tensor<f32>| {
                // Unpack: divide by 2
                let data: Vec<f32> = t.data_vec()?.iter().map(|&x| x / 2.0).collect();
                Tensor::from_storage(TensorStorage::cpu(data), t.shape().to_vec(), false)
            },
            || {
                let t = Tensor::from_storage(
                    TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
                    vec![3],
                    false,
                )?;
                let packed = pack_saved_tensor(t)?;
                // Packed values should be doubled.
                assert_eq!(packed.data_vec()?, vec![2.0, 4.0, 6.0]);

                let unpacked = unpack_saved_tensor(packed)?;
                // Unpacked values should be back to original.
                assert_eq!(unpacked.data_vec()?, vec![1.0, 2.0, 3.0]);

                Ok(())
            },
        );
        result.unwrap();
    }

    #[test]
    fn test_hooks_cleared_after_scope() {
        saved_tensors_hooks(
            |t: Tensor<f32>| Ok(t),
            |t: Tensor<f32>| Ok(t),
            || {
                assert!(has_saved_tensor_hooks());
                Ok(())
            },
        )
        .unwrap();

        // Hooks should be cleared after scope.
        assert!(!has_saved_tensor_hooks());
    }
}
