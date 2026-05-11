//! BatchNorm running-stat (buffer) loader for pretrained state dicts.
//!
//! ## Why this module exists
//!
//! `Module::load_state_dict(state, strict=false)` (defined in
//! `ferrotorch-nn`) walks `named_parameters()` and `named_buffers()`,
//! copying tensors out of `state` into the matching slots. That path is
//! correct for modules whose state lives in `Parameter<T>` or
//! `Buffer<T>` — but `BatchNorm{1,2,3}d`'s running statistics live in
//! `Mutex<Vec<f64>>` (for accumulator stability) and
//! `num_batches_tracked` lives in `Mutex<usize>`. Neither participates
//! in `named_buffers()`. With `strict=false`, the loader silently
//! drops those keys, leaving every BN layer at its construction
//! defaults (`running_mean=0`, `running_var=1`, `num_batches_tracked=0`).
//!
//! For pretrained inference this is catastrophic: BN's eval-mode
//! forward applies `(x - running_mean) / sqrt(running_var + eps)`, so
//! defaults turn pretrained features into garbage. Empirically (#1141
//! diagnosis) this manifested as ResNet-50 layer2 activations 26x
//! smaller than torchvision's, FPN features uniformly small, and
//! Faster R-CNN softmax near-uniform — all detections under the
//! `score_thresh` floor, N_det == 0.
//!
//! ## What this module does
//!
//! [`apply_bn_buffers_from_state_dict`] walks the model with
//! `Module::named_descendants_dyn()`, scans every key in `state_dict`
//! whose suffix is one of `running_mean` / `running_var` /
//! `num_batches_tracked`, finds the BN module at the parent path via
//! [`Module::as_any`] downcast, and calls the typed setter
//! (`BatchNorm{1,2,3}d::set_running_mean` / `set_running_var` /
//! `set_num_batches_tracked`).
//!
//! Lifted from the value-parity test harness in
//! `ferrotorch-vision/tests/conformance_vision_models.rs` (Phase 2 of
//! the #984 / #995 pipeline) so production `maybe_load_pretrained`
//! can call it after `model.load_state_dict(state, strict=false)`.
//! The test-side helper now delegates here.
//!
//! ## Silent-fallback contract
//!
//! Matching the test-side behaviour exactly, two skip paths are
//! accepted and do NOT error (preserves #984 → #995 Phase 1A
//! semantics for vision models that have not yet closed the
//! `named_children` gap):
//!
//!   - **Unreachable path** — the BN parent path is absent from
//!     `named_descendants_dyn()` (model didn't override
//!     `named_children`). The buffer is left at construction default.
//!   - **No `as_any` opt-in** — the module exists in the tree but
//!     `Module::as_any` returns `None`. Same fallback.
//!
//! A non-BN module opting into `as_any` and matching at a BN buffer
//! path remains a hard error (Phase 2 invariant: only BN modules opt
//! in to the downcast hook).

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::Module;
use ferrotorch_nn::norm::{BatchNorm1d, BatchNorm2d, BatchNorm3d};
use ferrotorch_nn::StateDict;

/// The three BN buffer-key suffixes torchvision emits. A key qualifies
/// as a BN buffer key iff `key.ends_with('.' + suffix)` for one of
/// these three.
const BN_BUFFER_SUFFIXES: &[&str] = &["running_mean", "running_var", "num_batches_tracked"];

/// Tolerance for clamping tiny-negative `running_var` rounding noise to
/// zero.
///
/// Some torchvision pretrained checkpoints (observed first on
/// `fcn_resnet50`, also in deeplabv3 / faster-rcnn / mask-rcnn) carry
/// `running_var` entries on the order of `-1e-12` — pure f32 rounding
/// noise around zero from BN's running accumulator. torchvision's
/// `BatchNorm` forward applies `(x - mean) / sqrt(var + eps)` with
/// `eps ~ 1e-5`, which trivially absorbs sub-microscopic noise; our
/// `BatchNorm{1,2,3}d::set_running_var` setter however rejects any
/// negative input as a corrupt-fixture sentinel.
///
/// `1e-6` is two orders of magnitude above f32 epsilon (~1.19e-7) and
/// six orders below typical BN `eps`, so it captures rounding-noise
/// negatives (`|v| < 1e-6` clamped to zero) while letting genuinely
/// negative variance (e.g. `-0.1`, `-1e-3`) surface as a setter error.
const RUNNING_VAR_CLAMP_TOL_F64: f64 = 1e-6;

/// Clamp tiny-negative `running_var` rounding noise to zero.
///
/// For each element `v` in `slice`:
///
///   - If `v < 0` AND `|v| < RUNNING_VAR_CLAMP_TOL_F64` (cast to `T`):
///     replace with `T::zero()`. This handles torchvision's
///     `-1e-12`-scale rounding noise that its forward absorbs via
///     `sqrt(var + eps)`, which our `set_running_var` setter rejects.
///   - Otherwise: keep `v` verbatim. Genuinely negative variance
///     (anything `<= -RUNNING_VAR_CLAMP_TOL_F64`) is preserved so the
///     setter still rejects it as a corrupt-fixture sentinel.
///
/// Always returns an owned `Vec<T>` to give the caller a stable
/// lifetime; the (rare) all-positive case still pays one `to_vec`
/// allocation, which is negligible compared to a full state-dict
/// load.
fn clamp_running_var_noise<T: Float>(slice: &[T]) -> Vec<T> {
    // Cast the f64 constant down to T at the call site. For T=f32 this
    // is exact (1e-6 is representable); for T=f64 it's identity. If a
    // future T cannot represent 1e-6 (`from` returns None) we fall
    // back to NOT clamping anything — the setter then surfaces the
    // negative as an error, preserving the loud-failure contract.
    let tol: T = T::from(RUNNING_VAR_CLAMP_TOL_F64)
        .unwrap_or_else(<T as num_traits::Zero>::zero);
    let zero = <T as num_traits::Zero>::zero();
    slice
        .iter()
        .map(|&v| {
            if v < zero && (zero - v) < tol {
                zero
            } else {
                v
            }
        })
        .collect()
}

/// Apply every BN running-stat / batch-counter key in `state_dict` to
/// the matching `BatchNorm{1,2,3}d` module in `model`.
///
/// `state_dict` is scanned for keys matching the pattern
/// `<bn-path>.{running_mean,running_var,num_batches_tracked}`. For
/// each, the module at `<bn-path>` is located via
/// [`Module::named_descendants_dyn`], downcast via [`Module::as_any`]
/// to a concrete BN type, and updated via the typed setter.
///
/// See the module-level doc for the silent-fallback contract.
///
/// # Errors
///
/// - The buffer tensor cannot be read as a CPU slice
///   ([`FerrotorchError::InvalidArgument`] wrapping the underlying
///   `Tensor::data` error).
/// - A BN setter rejects the slice (length mismatch, non-finite
///   value, negative variance). Propagates the setter's
///   [`FerrotorchError::ShapeMismatch`] or
///   [`FerrotorchError::InvalidArgument`].
/// - A `num_batches_tracked` tensor has length != 1 or a non-integer
///   value.
/// - A module at a BN buffer path opted into `as_any` but the
///   downcast matched none of `BatchNorm{1,2,3}d<T>` (Phase 2
///   invariant violation).
pub fn apply_bn_buffers_from_state_dict<T: Float + 'static>(
    model: &dyn Module<T>,
    state_dict: &StateDict<T>,
) -> FerrotorchResult<()> {
    // Build path → module index once. Cost: O(num_modules); reused
    // across all buffer keys.
    let mut path_to_module: std::collections::HashMap<String, &dyn Module<T>> =
        std::collections::HashMap::new();
    path_to_module.insert(String::new(), model);
    for (name, child) in model.named_descendants_dyn() {
        path_to_module.insert(name, child);
    }

    for full_key in state_dict.keys() {
        // Filter: only act on keys ending in a known BN suffix.
        let Some((bn_path, suffix)) = split_bn_buffer_key(full_key) else {
            continue;
        };

        // Phase 1A silent fallback: parent path not reachable via
        // `named_descendants_dyn`. Leave the BN module at its
        // construction default. (Tracked under #995 for vision models
        // that haven't yet closed the `named_children` gap.)
        let Some(bn_module) = path_to_module.get(bn_path).copied() else {
            continue;
        };

        // Phase 1A silent fallback: module is in the tree but did not
        // opt into the `as_any` downcast hook.
        let Some(any) = bn_module.as_any() else {
            continue;
        };

        let value = state_dict.get(full_key).ok_or_else(|| {
            // Should never trigger — we iterated `state_dict.keys()`
            // ourselves — but the explicit error keeps the contract
            // honest if `state_dict` is mutated under our feet.
            FerrotorchError::InvalidArgument {
                message: format!(
                    "bn_buffer_loader: BN buffer key \"{full_key}\" disappeared \
                     mid-iteration"
                ),
            }
        })?;

        let value_data_raw = value.data().map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "bn_buffer_loader: failed to read buffer \"{full_key}\" as CPU \
                 slice: {e}"
            ),
        })?;

        // Some torchvision pretrained checkpoints carry tiny negative
        // values in `running_var` (~ -1e-12) — pure f32 rounding noise
        // around zero from BN's accumulator. torchvision tolerates
        // these because its forward uses `sqrt(var + eps)` (and eps
        // ~ 1e-5 swamps the noise), but our `set_running_var` setter
        // rejects any negative input. For the `running_var` slot we
        // therefore clamp tiny negatives (|v| < CLAMP_TOL) to zero
        // before handing off; anything substantially negative still
        // surfaces as a setter error so genuinely corrupt fixtures
        // fail loudly. `running_mean` and `num_batches_tracked` keep
        // the raw slice.
        let clamped: Vec<T>;
        let value_data: &[T] = if suffix == "running_var" {
            clamped = clamp_running_var_noise(value_data_raw);
            &clamped
        } else {
            value_data_raw
        };

        // Try each concrete BN type. Exactly one will match for a
        // legitimate BN module; a non-BN type opting into `as_any` is
        // a Phase 2 invariant violation and surfaces as a hard error.
        if let Some(bn) = any.downcast_ref::<BatchNorm2d<T>>() {
            apply_bn_suffix_2d(full_key, suffix, bn, value_data)?;
        } else if let Some(bn) = any.downcast_ref::<BatchNorm1d<T>>() {
            apply_bn_suffix_1d(full_key, suffix, bn, value_data)?;
        } else if let Some(bn) = any.downcast_ref::<BatchNorm3d<T>>() {
            apply_bn_suffix_3d(full_key, suffix, bn, value_data)?;
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "bn_buffer_loader: BN buffer key \"{full_key}\" routed to \
                     module path \"{bn_path}\" whose as_any downcast matched \
                     none of BatchNorm{{1,2,3}}d<T> (Phase 2 invariant: only \
                     BN modules opt into the as_any hook)"
                ),
            });
        }
    }

    Ok(())
}

/// Split a state-dict key into `(bn_path, suffix)` iff the suffix is
/// one of the recognised BN-buffer suffixes. Returns `None` if the
/// key does not end in `.running_mean` / `.running_var` /
/// `.num_batches_tracked` — those keys are not BN buffers and the
/// loader skips them (they belong to parameters or other buffers
/// already handled by `Module::load_state_dict`).
fn split_bn_buffer_key(full_key: &str) -> Option<(&str, &str)> {
    let (path, suffix) = full_key.rsplit_once('.')?;
    if BN_BUFFER_SUFFIXES.contains(&suffix) {
        Some((path, suffix))
    } else {
        None
    }
}

fn apply_bn_suffix_2d<T: Float>(
    full_key: &str,
    suffix: &str,
    bn: &BatchNorm2d<T>,
    value: &[T],
) -> FerrotorchResult<()> {
    match suffix {
        "running_mean" => bn.set_running_mean(value),
        "running_var" => bn.set_running_var(value),
        "num_batches_tracked" => {
            let nbt = bn_nbt_from_slice(full_key, value)?;
            bn.set_num_batches_tracked(nbt)
        }
        // `split_bn_buffer_key` filters to the three recognised
        // suffixes, so this arm is unreachable. We leave it as an
        // explicit Err rather than `unreachable!()` to keep the
        // failure mode debuggable if the suffix list ever drifts.
        other => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "bn_buffer_loader: BN2d buffer key \"{full_key}\" has \
                 unrecognised suffix \"{other}\""
            ),
        }),
    }
}

fn apply_bn_suffix_1d<T: Float>(
    full_key: &str,
    suffix: &str,
    bn: &BatchNorm1d<T>,
    value: &[T],
) -> FerrotorchResult<()> {
    match suffix {
        "running_mean" => bn.set_running_mean(value),
        "running_var" => bn.set_running_var(value),
        "num_batches_tracked" => {
            let nbt = bn_nbt_from_slice(full_key, value)?;
            bn.set_num_batches_tracked(nbt)
        }
        other => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "bn_buffer_loader: BN1d buffer key \"{full_key}\" has \
                 unrecognised suffix \"{other}\""
            ),
        }),
    }
}

fn apply_bn_suffix_3d<T: Float>(
    full_key: &str,
    suffix: &str,
    bn: &BatchNorm3d<T>,
    value: &[T],
) -> FerrotorchResult<()> {
    match suffix {
        "running_mean" => bn.set_running_mean(value),
        "running_var" => bn.set_running_var(value),
        "num_batches_tracked" => {
            let nbt = bn_nbt_from_slice(full_key, value)?;
            bn.set_num_batches_tracked(nbt)
        }
        other => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "bn_buffer_loader: BN3d buffer key \"{full_key}\" has \
                 unrecognised suffix \"{other}\""
            ),
        }),
    }
}

/// Decode a single-element tensor as a non-negative integer batch
/// counter. PyTorch state dicts store `num_batches_tracked` as int64,
/// but the ferrotorch safetensors loader represents every leaf
/// tensor as `Tensor<T>` (typed by `T`). The 1-element payload is
/// converted via `T -> f64 -> usize` with explicit non-negativity,
/// finiteness, and integrality checks so a malformed file fails
/// loudly instead of writing a nonsense counter.
fn bn_nbt_from_slice<T: Float>(full_key: &str, value: &[T]) -> FerrotorchResult<usize> {
    if value.len() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "bn_buffer_loader: num_batches_tracked buffer \"{full_key}\" \
                 must be a 1-element tensor, got length {}",
                value.len()
            ),
        });
    }
    let v = value[0].to_f64().ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!(
            "bn_buffer_loader: num_batches_tracked buffer \"{full_key}\" \
             element could not be widened to f64"
        ),
    })?;
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "bn_buffer_loader: num_batches_tracked buffer \"{full_key}\" \
                 must be a non-negative integer, got {v}"
            ),
        });
    }
    Ok(v as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};
    use ferrotorch_nn::norm::BatchNorm2d;
    use ferrotorch_nn::parameter::Parameter;

    /// Build a 1-D `Tensor<f32>` from a vec.
    fn t1(v: Vec<f32>) -> Tensor<f32> {
        let len = v.len();
        Tensor::from_storage(TensorStorage::cpu(v), vec![len], false).unwrap()
    }

    /// Wrap a single BatchNorm2d as a Module subtree at child path "bn".
    /// We need a parent Module so the BN appears at a non-empty path and
    /// the loader exercises the path-keyed walk (not just root).
    struct WithBn {
        bn: BatchNorm2d<f32>,
    }

    impl WithBn {
        fn new(num_features: usize) -> Self {
            Self {
                bn: BatchNorm2d::new(num_features, 1e-5, 0.1, true).unwrap(),
            }
        }
    }

    impl Module<f32> for WithBn {
        fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            self.bn.forward(input)
        }
        fn parameters(&self) -> Vec<&Parameter<f32>> {
            self.bn.parameters()
        }
        fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
            self.bn.parameters_mut()
        }
        fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
            self.bn
                .named_parameters()
                .into_iter()
                .map(|(n, p)| (format!("bn.{n}"), p))
                .collect()
        }
        fn train(&mut self) {
            self.bn.train();
        }
        fn eval(&mut self) {
            self.bn.eval();
        }
        fn is_training(&self) -> bool {
            self.bn.is_training()
        }
        fn children(&self) -> Vec<&dyn Module<f32>> {
            vec![&self.bn]
        }
        fn named_children(&self) -> Vec<(String, &dyn Module<f32>)> {
            vec![("bn".to_string(), &self.bn)]
        }
    }

    #[test]
    fn split_recognises_three_suffixes() {
        assert_eq!(
            split_bn_buffer_key("layer1.0.bn1.running_mean"),
            Some(("layer1.0.bn1", "running_mean"))
        );
        assert_eq!(
            split_bn_buffer_key("layer1.0.bn1.running_var"),
            Some(("layer1.0.bn1", "running_var"))
        );
        assert_eq!(
            split_bn_buffer_key("layer1.0.bn1.num_batches_tracked"),
            Some(("layer1.0.bn1", "num_batches_tracked"))
        );
    }

    #[test]
    fn split_rejects_non_buffer_keys() {
        assert_eq!(split_bn_buffer_key("layer1.0.bn1.weight"), None);
        assert_eq!(split_bn_buffer_key("layer1.0.bn1.bias"), None);
        assert_eq!(split_bn_buffer_key("conv1.weight"), None);
        assert_eq!(split_bn_buffer_key("no_dot_at_all"), None);
    }

    #[test]
    fn loader_applies_running_mean_and_var() {
        let model = WithBn::new(4);
        let mut state: StateDict<f32> = StateDict::new();
        state.insert(
            "bn.running_mean".to_string(),
            t1(vec![1.0, 2.0, 3.0, 4.0]),
        );
        state.insert(
            "bn.running_var".to_string(),
            t1(vec![0.5, 0.6, 0.7, 0.8]),
        );

        apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state).unwrap();

        let rm = model.bn.running_mean();
        let rv = model.bn.running_var();
        // running_mean values are exact integers in f32 → exact in f64.
        assert_eq!(rm, vec![1.0_f64, 2.0, 3.0, 4.0]);
        // running_var values aren't exactly representable in f32 (0.6,
        // 0.7, 0.8 are repeating binary), so the widening from f32 →
        // f64 surfaces the original f32 quantisation error (~1e-7).
        // Tolerance must be at the f32-precision floor.
        for (a, b) in rv.iter().zip([0.5_f64, 0.6, 0.7, 0.8].iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "running_var element mismatch: got {a}, expected {b} (rv={rv:?})"
            );
        }
    }

    #[test]
    fn loader_applies_num_batches_tracked() {
        let model = WithBn::new(2);
        let mut state: StateDict<f32> = StateDict::new();
        state.insert("bn.num_batches_tracked".to_string(), t1(vec![17.0]));
        apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state).unwrap();
        assert_eq!(model.bn.num_batches_tracked(), 17);
    }

    #[test]
    fn loader_silently_skips_unreachable_paths() {
        // No `named_children` override → BN module not reachable.
        // The inner `bn` carries its own training flag (interior
        // mutability), so the outer `train`/`eval` forward to it
        // rather than being empty stubs.
        struct Opaque {
            bn: BatchNorm2d<f32>,
        }
        impl Module<f32> for Opaque {
            fn forward(&self, x: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
                self.bn.forward(x)
            }
            fn parameters(&self) -> Vec<&Parameter<f32>> {
                self.bn.parameters()
            }
            fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
                self.bn.parameters_mut()
            }
            fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
                self.bn.named_parameters()
            }
            fn train(&mut self) {
                self.bn.train();
            }
            fn eval(&mut self) {
                self.bn.eval();
            }
            fn is_training(&self) -> bool {
                self.bn.is_training()
            }
        }
        let model = Opaque {
            bn: BatchNorm2d::new(2, 1e-5, 0.1, true).unwrap(),
        };
        let mut state: StateDict<f32> = StateDict::new();
        state.insert("bn.running_mean".to_string(), t1(vec![9.9, 9.9]));
        // Loader returns Ok, BN running_mean stays at construction default (zeros).
        apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state).unwrap();
        let rm = model.bn.running_mean();
        assert_eq!(rm, vec![0.0_f64, 0.0]);
    }

    #[test]
    fn loader_skips_non_bn_keys() {
        // Parameters & non-BN buffers in the state dict must be silently
        // ignored by this loader (they are handled upstream by
        // `Module::load_state_dict`). Only the three BN suffixes apply.
        let model = WithBn::new(2);
        let mut state: StateDict<f32> = StateDict::new();
        state.insert("bn.weight".to_string(), t1(vec![1.0, 1.0]));
        state.insert("bn.bias".to_string(), t1(vec![0.0, 0.0]));
        state.insert("bn.running_mean".to_string(), t1(vec![3.0, 4.0]));
        apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state).unwrap();
        let rm = model.bn.running_mean();
        assert_eq!(rm, vec![3.0_f64, 4.0]);
    }

    #[test]
    fn loader_rejects_length_mismatch_for_nbt() {
        let model = WithBn::new(2);
        let mut state: StateDict<f32> = StateDict::new();
        state.insert(
            "bn.num_batches_tracked".to_string(),
            t1(vec![1.0, 2.0, 3.0]),
        );
        let err = apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state)
            .expect_err("multi-element nbt must error");
        let msg = format!("{err}");
        assert!(msg.contains("num_batches_tracked"));
        assert!(msg.contains("1-element"));
    }

    #[test]
    fn loader_rejects_negative_nbt() {
        let model = WithBn::new(2);
        let mut state: StateDict<f32> = StateDict::new();
        state.insert("bn.num_batches_tracked".to_string(), t1(vec![-1.0]));
        let err = apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state)
            .expect_err("negative nbt must error");
        let msg = format!("{err}");
        assert!(msg.contains("non-negative integer"));
    }

    #[test]
    fn loader_propagates_running_var_negativity() {
        // BatchNorm2d::set_running_var rejects negative entries; the
        // loader must forward that error verbatim. The `-0.1` here is
        // far below `-RUNNING_VAR_CLAMP_TOL_F64`, so it bypasses the
        // tiny-negative clamp and reaches the setter.
        let model = WithBn::new(2);
        let mut state: StateDict<f32> = StateDict::new();
        state.insert("bn.running_var".to_string(), t1(vec![1.0, -0.1]));
        let err = apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state)
            .expect_err("negative running_var must error");
        let msg = format!("{err}");
        assert!(msg.contains("running_var"));
        assert!(msg.contains("negative"));
    }
}
