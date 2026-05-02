//! Boolean tensors for masks and logical operations. (#596)
//!
//! `BoolTensor` is a CPU-resident, contiguous tensor of `bool`s, used
//! for `masked_fill`, `where`, and any predicate-driven indexing where
//! a float-valued mask would lose semantic clarity.
//!
//! Construction:
//! - [`BoolTensor::zeros`] / [`ones`] — uniform fill
//! - [`BoolTensor::from_predicate`] — build from a `Tensor<T>` + closure
//! - [`BoolTensor::from_vec`] / [`from_slice`] — explicit data + shape
//!
//! Logical ops: `not`, `and`, `or`, `xor` are pointwise. `count_true`
//! and `any` / `all` are reductions.

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

/// CPU-resident tensor of booleans. Shape is metadata; storage is a flat
/// `Arc<Vec<bool>>` for cheap clones.
#[derive(Debug, Clone)]
pub struct BoolTensor {
    data: Arc<Vec<bool>>,
    shape: Vec<usize>,
}

impl BoolTensor {
    /// Build from a Vec + shape. Errors on numel mismatch.
    pub fn from_vec(data: Vec<bool>, shape: Vec<usize>) -> FerrotorchResult<Self> {
        let expected: usize = shape.iter().product::<usize>().max(1);
        if data.len() != expected && !(shape.is_empty() && data.len() == 1) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::from_vec: data.len()={} != prod(shape)={} for shape {:?}",
                    data.len(),
                    expected,
                    shape
                ),
            });
        }
        Ok(Self {
            data: Arc::new(data),
            shape,
        })
    }

    /// Build from a slice + shape (clones into a fresh `Vec`).
    pub fn from_slice(data: &[bool], shape: &[usize]) -> FerrotorchResult<Self> {
        Self::from_vec(data.to_vec(), shape.to_vec())
    }

    /// All-false tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product::<usize>().max(1);
        Self {
            data: Arc::new(vec![false; total]),
            shape: shape.to_vec(),
        }
    }

    /// All-true tensor of the given shape.
    pub fn ones(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product::<usize>().max(1);
        Self {
            data: Arc::new(vec![true; total]),
            shape: shape.to_vec(),
        }
    }

    /// Build a mask by applying `pred` to every element of `t`.
    /// Useful for `Tensor < 0`, `Tensor.is_finite()`, etc.
    pub fn from_predicate<T: Float>(
        t: &Tensor<T>,
        pred: impl Fn(T) -> bool,
    ) -> FerrotorchResult<Self> {
        let data = t.data_vec()?;
        let mask: Vec<bool> = data.iter().map(|&v| pred(v)).collect();
        Self::from_vec(mask, t.shape().to_vec())
    }

    /// Logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Borrow the contiguous buffer.
    pub fn data(&self) -> &[bool] {
        &self.data
    }

    /// Pointwise NOT.
    pub fn not(&self) -> Self {
        let out: Vec<bool> = self.data.iter().map(|&b| !b).collect();
        Self {
            data: Arc::new(out),
            shape: self.shape.clone(),
        }
    }

    /// Pointwise AND. Errors on shape mismatch.
    pub fn and(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |a, b| a && b, "and")
    }

    /// Pointwise OR.
    pub fn or(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |a, b| a || b, "or")
    }

    /// Pointwise XOR.
    pub fn xor(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |a, b| a ^ b, "xor")
    }

    fn binary_op(
        &self,
        other: &Self,
        f: impl Fn(bool, bool) -> bool,
        op_name: &str,
    ) -> FerrotorchResult<Self> {
        if self.shape != other.shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::{op_name}: shapes {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let out: Vec<bool> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Self {
            data: Arc::new(out),
            shape: self.shape.clone(),
        })
    }

    /// Reshape (must preserve numel; no data copy).
    pub fn reshape(&self, shape: &[usize]) -> FerrotorchResult<Self> {
        let new_total: usize = shape.iter().product::<usize>().max(1);
        if new_total != self.data.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::reshape: new shape {:?} (numel {}) != current numel {}",
                    shape,
                    new_total,
                    self.data.len()
                ),
            });
        }
        Ok(Self {
            data: Arc::clone(&self.data),
            shape: shape.to_vec(),
        })
    }

    /// Number of true elements.
    pub fn count_true(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }

    /// True if any element is true.
    pub fn any(&self) -> bool {
        self.data.iter().any(|&b| b)
    }

    /// True if all elements are true.
    pub fn all(&self) -> bool {
        self.data.iter().all(|&b| b)
    }

    /// Pointwise `>` comparing two float tensors of the same shape;
    /// produces a `BoolTensor` of matching shape. Mirrors
    /// `torch.gt(a, b)` returning a bool tensor. (#615)
    pub fn gt<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare(a, b, |x, y| x > y, "gt")
    }

    /// Pointwise `<`. (#615)
    pub fn lt<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare(a, b, |x, y| x < y, "lt")
    }

    /// Pointwise `>=`. (#615)
    pub fn ge<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare(a, b, |x, y| x >= y, "ge")
    }

    /// Pointwise `<=`. (#615)
    pub fn le<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare(a, b, |x, y| x <= y, "le")
    }

    /// Pointwise `==`. (#615)
    pub fn eq_t<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare(a, b, |x, y| x == y, "eq")
    }

    /// Pointwise `!=`. (#615)
    pub fn ne<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare(a, b, |x, y| x != y, "ne")
    }

    fn compare<T: Float>(
        a: &Tensor<T>,
        b: &Tensor<T>,
        f: impl Fn(T, T) -> bool,
        op: &str,
    ) -> FerrotorchResult<Self> {
        if a.shape() != b.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::{op}: shapes {:?} vs {:?}",
                    a.shape(),
                    b.shape()
                ),
            });
        }
        let a_data = a.data_vec()?;
        let b_data = b.data_vec()?;
        let result: Vec<bool> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        Self::from_vec(result, a.shape().to_vec())
    }

    /// Convert to a float tensor: true → 1.0, false → 0.0.
    pub fn to_float<T: Float>(&self) -> FerrotorchResult<Tensor<T>> {
        let one = T::from(1.0).unwrap();
        let zero = T::from(0.0).unwrap();
        let data: Vec<T> = self
            .data
            .iter()
            .map(|&b| if b { one } else { zero })
            .collect();
        Tensor::from_storage(
            crate::storage::TensorStorage::cpu(data),
            self.shape.clone(),
            false,
        )
    }
}

impl std::fmt::Display for BoolTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BoolTensor(shape={:?}, len={}, true={})",
            self.shape,
            self.data.len(),
            self.count_true()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_ones() {
        let z = BoolTensor::zeros(&[2, 3]);
        let o = BoolTensor::ones(&[2, 3]);
        assert_eq!(z.numel(), 6);
        assert_eq!(o.numel(), 6);
        assert!(z.data().iter().all(|&b| !b));
        assert!(o.data().iter().all(|&b| b));
    }

    #[test]
    fn from_vec_shape_mismatch_errors() {
        let err = BoolTensor::from_vec(vec![true, false], vec![3]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn from_predicate_builds_mask() {
        let t = crate::creation::from_slice::<f32>(&[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let mask = BoolTensor::from_predicate(&t, |x| x > 0.0).unwrap();
        assert_eq!(mask.data(), &[false, false, true, true]);
    }

    #[test]
    fn pointwise_not() {
        let m = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        let n = m.not();
        assert_eq!(n.data(), &[false, true, false]);
    }

    #[test]
    fn pointwise_and_or_xor() {
        let a = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
        let b = BoolTensor::from_vec(vec![true, true, false, false], vec![4]).unwrap();
        assert_eq!(a.and(&b).unwrap().data(), &[true, false, false, false]);
        assert_eq!(a.or(&b).unwrap().data(), &[true, true, true, false]);
        assert_eq!(a.xor(&b).unwrap().data(), &[false, true, true, false]);
    }

    #[test]
    fn binary_op_shape_mismatch() {
        let a = BoolTensor::ones(&[3]);
        let b = BoolTensor::ones(&[2]);
        assert!(matches!(
            a.and(&b).unwrap_err(),
            FerrotorchError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn count_true_any_all() {
        let m = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        assert_eq!(m.count_true(), 2);
        assert!(m.any());
        assert!(!m.all());

        let z = BoolTensor::zeros(&[3]);
        assert!(!z.any());
        assert_eq!(z.count_true(), 0);

        let o = BoolTensor::ones(&[3]);
        assert!(o.all());
        assert_eq!(o.count_true(), 3);
    }

    #[test]
    fn reshape_preserves_data() {
        let m = BoolTensor::from_vec(vec![true, false, true, false, true, false], vec![6]).unwrap();
        let r = m.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.data(), m.data());
    }

    #[test]
    fn to_float_emits_zeros_and_ones() {
        let m = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        let f = m.to_float::<f32>().unwrap();
        assert_eq!(f.data().unwrap(), &[1.0_f32, 0.0, 1.0]);
    }

    #[test]
    fn clone_shares_arc_storage() {
        let m = BoolTensor::ones(&[5]);
        let m2 = m.clone();
        assert!(Arc::ptr_eq(&m.data, &m2.data));
    }

    // -----------------------------------------------------------------------
    // Comparison ops returning BoolTensor (#615)
    // -----------------------------------------------------------------------

    #[test]
    fn compare_gt_basic() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[0.0, 3.0, 3.0, 5.0], &[4]).unwrap();
        let m = BoolTensor::gt(&a, &b).unwrap();
        assert_eq!(m.data(), &[true, false, false, false]);
    }

    #[test]
    fn compare_lt_basic() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[2.0, 2.0, 4.0], &[3]).unwrap();
        let m = BoolTensor::lt(&a, &b).unwrap();
        assert_eq!(m.data(), &[true, false, true]);
    }

    #[test]
    fn compare_ge_le() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[1.0, 3.0, 2.0], &[3]).unwrap();
        assert_eq!(BoolTensor::ge(&a, &b).unwrap().data(), &[true, false, true]);
        assert_eq!(BoolTensor::le(&a, &b).unwrap().data(), &[true, true, false]);
    }

    #[test]
    fn compare_eq_ne() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[1.0, 5.0, 3.0], &[3]).unwrap();
        assert_eq!(
            BoolTensor::eq_t(&a, &b).unwrap().data(),
            &[true, false, true]
        );
        assert_eq!(
            BoolTensor::ne(&a, &b).unwrap().data(),
            &[false, true, false]
        );
    }

    #[test]
    fn compare_rejects_shape_mismatch() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let err = BoolTensor::gt(&a, &b).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }
}
