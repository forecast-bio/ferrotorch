//! `NamedTensor<T>` — dim-name annotations on top of `Tensor<T>`. (#621)
//!
//! Mirrors `torch.Tensor.refine_names` / `align_to` / `rename` semantics.
//! A `NamedTensor` pairs an underlying `Tensor<T>` with one optional name
//! per dimension. Names are advisory: ops like `align_to` permute dims by
//! name to match a target ordering, but the underlying tensor math is
//! unchanged.
//!
//! # Why advisory
//!
//! PyTorch's named-tensor experimental feature aimed to surface name
//! mismatches as errors at op boundaries. We don't (yet) intercept every
//! op; instead this module gives users a way to *annotate* tensors and
//! reorder them by name before passing into op surface that's still
//! position-based. That covers the most common use case (avoiding
//! "did I get the batch dim right?" bugs in attention / einsum prep).

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

/// A `Tensor<T>` paired with one optional dim name per axis.
///
/// `names.len()` always equals `inner.ndim()`. Each entry is either
/// `Some(String)` (the axis has a name) or `None` (anonymous).
#[derive(Clone, Debug)]
pub struct NamedTensor<T: Float> {
    inner: Tensor<T>,
    names: Vec<Option<String>>,
}

impl<T: Float> NamedTensor<T> {
    /// Wrap a tensor with explicit per-dim names.
    ///
    /// Errors if `names.len() != inner.ndim()` or two names are equal
    /// (and not both `None`).
    pub fn new(inner: Tensor<T>, names: Vec<Option<String>>) -> FerrotorchResult<Self> {
        if names.len() != inner.ndim() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "NamedTensor::new: names.len()={} != ndim={}",
                    names.len(),
                    inner.ndim()
                ),
            });
        }
        // Reject duplicate non-None names.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for n in names.iter().flatten() {
            if !seen.insert(n.as_str()) {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("NamedTensor::new: duplicate dim name '{n}'"),
                });
            }
        }
        Ok(Self { inner, names })
    }

    /// Wrap with a flat `&[&str]` of names; convenient for the common
    /// fully-named case. Use `""` to leave a dim anonymous.
    pub fn refined(inner: Tensor<T>, names: &[&str]) -> FerrotorchResult<Self> {
        let owned: Vec<Option<String>> = names
            .iter()
            .map(|s| {
                if s.is_empty() {
                    None
                } else {
                    Some((*s).to_string())
                }
            })
            .collect();
        Self::new(inner, owned)
    }

    /// Borrow the underlying float tensor. Use this at op boundaries
    /// where the named annotation isn't propagated.
    pub fn tensor(&self) -> &Tensor<T> {
        &self.inner
    }

    /// Consume and return the inner tensor (drops names).
    pub fn into_tensor(self) -> Tensor<T> {
        self.inner
    }

    /// Per-dim names (length = ndim).
    pub fn names(&self) -> &[Option<String>] {
        &self.names
    }

    /// Logical shape (alias for `tensor().shape()`).
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Total elements.
    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Index of the dim with the given name, if any.
    pub fn dim_index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n.as_deref() == Some(name))
    }

    /// Size of the dim with the given name, or `None` if not found.
    pub fn size_of(&self, name: &str) -> Option<usize> {
        self.dim_index(name).map(|i| self.shape()[i])
    }

    /// Replace some dim names according to `mapping: [(old, new), ...]`.
    /// Names not in the mapping are kept as-is. `None` names are unchanged.
    pub fn rename(&self, mapping: &[(&str, &str)]) -> FerrotorchResult<Self> {
        let map: std::collections::HashMap<&str, &str> = mapping.iter().copied().collect();
        let new_names: Vec<Option<String>> = self
            .names
            .iter()
            .map(|n| {
                n.as_ref().map(|s| {
                    map.get(s.as_str())
                        .map(|n| (*n).to_string())
                        .unwrap_or_else(|| s.clone())
                })
            })
            .collect();
        Self::new(self.inner.clone(), new_names)
    }

    /// Return a new NamedTensor whose dims are permuted to match
    /// `target_names`. `target_names` must contain every named dim of
    /// `self` (and only those); anonymous dims are not allowed in
    /// `target_names` since the permutation is name-driven.
    pub fn align_to(&self, target_names: &[&str]) -> FerrotorchResult<Self> {
        if target_names.len() != self.ndim() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "NamedTensor::align_to: target len={} != ndim={}",
                    target_names.len(),
                    self.ndim()
                ),
            });
        }

        let mut perm: Vec<usize> = Vec::with_capacity(self.ndim());
        for &t in target_names {
            let idx = self.dim_index(t).ok_or(FerrotorchError::InvalidArgument {
                message: format!(
                    "NamedTensor::align_to: target name '{t}' not present in {:?}",
                    self.names
                ),
            })?;
            perm.push(idx);
        }

        // Apply permutation to the tensor + names.
        let permuted = crate::methods::permute_t(&self.inner, &perm)?;
        let new_names: Vec<Option<String>> = perm.iter().map(|&i| self.names[i].clone()).collect();
        Self::new(permuted, new_names)
    }

    /// Drop names and return an unnamed `NamedTensor` (i.e. all `None`).
    /// Useful before passing into ops that don't preserve names.
    pub fn detached(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            names: vec![None; self.ndim()],
        }
    }
}

impl<T: Float> std::fmt::Display for NamedTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self
            .names
            .iter()
            .map(|n| n.as_deref().unwrap_or("_"))
            .collect();
        write!(
            f,
            "NamedTensor(shape={:?}, names={:?})",
            self.shape(),
            names
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn t_f32(shape: &[usize]) -> Tensor<f32> {
        let n: usize = shape.iter().product::<usize>().max(1);
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn named_tensor_basic_construction() {
        let nt = NamedTensor::refined(t_f32(&[2, 3, 4]), &["batch", "seq", "feat"]).unwrap();
        assert_eq!(nt.ndim(), 3);
        assert_eq!(nt.size_of("batch"), Some(2));
        assert_eq!(nt.size_of("seq"), Some(3));
        assert_eq!(nt.size_of("feat"), Some(4));
        assert_eq!(nt.size_of("missing"), None);
    }

    #[test]
    fn named_tensor_rejects_length_mismatch() {
        let err = NamedTensor::refined(t_f32(&[2, 3]), &["only", "two", "names"]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn named_tensor_rejects_duplicate_names() {
        let err = NamedTensor::refined(t_f32(&[2, 3]), &["x", "x"]).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn named_tensor_anonymous_dim_via_empty_string() {
        let nt = NamedTensor::refined(t_f32(&[2, 3]), &["batch", ""]).unwrap();
        assert_eq!(nt.names()[0].as_deref(), Some("batch"));
        assert_eq!(nt.names()[1], None);
    }

    #[test]
    fn named_tensor_align_permutes_dims() {
        // [batch=2, seq=3, feat=4] aligned to [feat, batch, seq] should
        // yield shape [4, 2, 3].
        let nt = NamedTensor::refined(t_f32(&[2, 3, 4]), &["batch", "seq", "feat"]).unwrap();
        let aligned = nt.align_to(&["feat", "batch", "seq"]).unwrap();
        assert_eq!(aligned.shape(), &[4, 2, 3]);
        assert_eq!(aligned.names()[0].as_deref(), Some("feat"));
        assert_eq!(aligned.names()[1].as_deref(), Some("batch"));
        assert_eq!(aligned.names()[2].as_deref(), Some("seq"));
    }

    #[test]
    fn named_tensor_align_identity_is_clone() {
        let nt = NamedTensor::refined(t_f32(&[2, 3]), &["a", "b"]).unwrap();
        let aligned = nt.align_to(&["a", "b"]).unwrap();
        assert_eq!(aligned.shape(), nt.shape());
    }

    #[test]
    fn named_tensor_align_rejects_unknown_name() {
        let nt = NamedTensor::refined(t_f32(&[2, 3]), &["a", "b"]).unwrap();
        let err = nt.align_to(&["a", "z"]).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn named_tensor_align_rejects_length_mismatch() {
        let nt = NamedTensor::refined(t_f32(&[2, 3]), &["a", "b"]).unwrap();
        let err = nt.align_to(&["a"]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn named_tensor_rename_replaces_specified_names() {
        let nt = NamedTensor::refined(t_f32(&[2, 3, 4]), &["b", "s", "f"]).unwrap();
        let renamed = nt.rename(&[("b", "batch"), ("s", "seq")]).unwrap();
        assert_eq!(renamed.names()[0].as_deref(), Some("batch"));
        assert_eq!(renamed.names()[1].as_deref(), Some("seq"));
        assert_eq!(renamed.names()[2].as_deref(), Some("f"));
    }

    #[test]
    fn named_tensor_detached_drops_names() {
        let nt = NamedTensor::refined(t_f32(&[2, 3]), &["a", "b"]).unwrap();
        let d = nt.detached();
        for n in d.names() {
            assert!(n.is_none());
        }
    }

    #[test]
    fn named_tensor_into_tensor_recovers_inner() {
        let nt = NamedTensor::refined(t_f32(&[2, 3]), &["a", "b"]).unwrap();
        let t = nt.into_tensor();
        assert_eq!(t.shape(), &[2, 3]);
    }

    #[test]
    fn named_tensor_dim_index_lookup() {
        let nt = NamedTensor::refined(t_f32(&[2, 3, 4]), &["batch", "seq", "feat"]).unwrap();
        assert_eq!(nt.dim_index("batch"), Some(0));
        assert_eq!(nt.dim_index("seq"), Some(1));
        assert_eq!(nt.dim_index("feat"), Some(2));
        assert_eq!(nt.dim_index("missing"), None);
    }
}
