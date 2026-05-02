//! Integer-typed tensors for indexing, embedding lookups, and any other
//! workload that needs first-class non-float storage. (#596)
//!
//! `IntTensor<I>` is a CPU-resident, contiguous tensor of integers
//! (`i32` or `i64`). It is intentionally **not** generic over `Float`
//! — the existing `Tensor<T: Float>` is the right type for differentiable
//! float math. `IntTensor` is for indices and counts where autograd is
//! a category error.
//!
//! # Conversions
//!
//! - [`Tensor::to_int`] — round-then-cast a float tensor to ints
//! - [`IntTensor::to_float`] — widen back into a float tensor
//!
//! Both copy data; there is no shared-storage path because the element
//! sizes differ.

use std::sync::Arc;

use crate::error::{FerrotorchError, FerrotorchResult};

/// Element types supported by [`IntTensor`].
pub trait IntElement: Copy + Send + Sync + 'static + std::fmt::Debug + std::fmt::Display {
    /// Bit-width of one element, used for dtype tagging.
    const BITS: u32;
    /// Returns this element type's printable name (e.g. `"i32"`).
    fn dtype_name() -> &'static str;
    /// Convert from i64. Returns `None` on out-of-range.
    fn try_from_i64(v: i64) -> Option<Self>;
    /// Widen to i64.
    fn to_i64(self) -> i64;
}

impl IntElement for i32 {
    const BITS: u32 = 32;
    fn dtype_name() -> &'static str {
        "i32"
    }
    fn try_from_i64(v: i64) -> Option<Self> {
        if (i32::MIN as i64..=i32::MAX as i64).contains(&v) {
            Some(v as i32)
        } else {
            None
        }
    }
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl IntElement for i64 {
    const BITS: u32 = 64;
    fn dtype_name() -> &'static str {
        "i64"
    }
    fn try_from_i64(v: i64) -> Option<Self> {
        Some(v)
    }
    fn to_i64(self) -> i64 {
        self
    }
}

/// CPU-resident, contiguous tensor of integers. `Arc<Vec<I>>` storage so
/// clones are cheap and shape views are trivial.
#[derive(Debug, Clone)]
pub struct IntTensor<I: IntElement> {
    data: Arc<Vec<I>>,
    shape: Vec<usize>,
}

impl<I: IntElement> IntTensor<I> {
    /// Build from a Vec + shape. Returns an error if `data.len()` does
    /// not match the shape's total numel.
    pub fn from_vec(data: Vec<I>, shape: Vec<usize>) -> FerrotorchResult<Self> {
        let expected: usize = shape.iter().product::<usize>().max(1);
        if data.len() != expected && !(shape.is_empty() && data.len() == 1) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "IntTensor::from_vec: data.len()={} != prod(shape)={} for shape {:?}",
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
    pub fn from_slice(data: &[I], shape: &[usize]) -> FerrotorchResult<Self> {
        Self::from_vec(data.to_vec(), shape.to_vec())
    }

    /// Zeros of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product::<usize>().max(1);
        let zero = I::try_from_i64(0).expect("0 fits in any IntElement");
        Self {
            data: Arc::new(vec![zero; total]),
            shape: shape.to_vec(),
        }
    }

    /// 1-D `arange`-style `[0, 1, ..., n-1]`.
    pub fn arange(n: usize) -> FerrotorchResult<Self> {
        let mut data: Vec<I> = Vec::with_capacity(n);
        for i in 0..n {
            data.push(
                I::try_from_i64(i as i64).ok_or(FerrotorchError::InvalidArgument {
                    message: format!(
                        "IntTensor::arange: {i} out of range for {}",
                        I::dtype_name()
                    ),
                })?,
            );
        }
        Self::from_vec(data, vec![n])
    }

    /// 0-d scalar.
    pub fn scalar(v: I) -> Self {
        Self {
            data: Arc::new(vec![v]),
            shape: Vec::new(),
        }
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
    pub fn data(&self) -> &[I] {
        &self.data
    }

    /// Element type name (`"i32"` / `"i64"`).
    pub fn dtype_name(&self) -> &'static str {
        I::dtype_name()
    }

    /// Cast this `IntTensor<I>` to `IntTensor<J>`. Returns an error on
    /// out-of-range elements.
    pub fn cast<J: IntElement>(&self) -> FerrotorchResult<IntTensor<J>> {
        let mut out: Vec<J> = Vec::with_capacity(self.data.len());
        for (i, &v) in self.data.iter().enumerate() {
            let widened = v.to_i64();
            out.push(
                J::try_from_i64(widened).ok_or(FerrotorchError::InvalidArgument {
                    message: format!(
                        "IntTensor::cast: element {i} = {v} out of range for {}",
                        J::dtype_name()
                    ),
                })?,
            );
        }
        IntTensor::<J>::from_vec(out, self.shape.clone())
    }

    /// Reshape (must preserve numel; no data copy).
    pub fn reshape(&self, shape: &[usize]) -> FerrotorchResult<Self> {
        let new_total: usize = shape.iter().product::<usize>().max(1);
        if new_total != self.data.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "IntTensor::reshape: new shape {:?} (numel {}) != current numel {}",
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
}

impl<I: IntElement> std::fmt::Display for IntTensor<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IntTensor<{}>(shape={:?}, len={})",
            I::dtype_name(),
            self.shape,
            self.data.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_basic() {
        let t = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn from_vec_shape_mismatch_errors() {
        let err = IntTensor::<i32>::from_vec(vec![1, 2, 3], vec![2, 2]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn zeros_correct_size() {
        let t = IntTensor::<i64>::zeros(&[3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.data().iter().all(|&x| x == 0));
    }

    #[test]
    fn arange_sequence() {
        let t = IntTensor::<i32>::arange(5).unwrap();
        assert_eq!(t.data(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn arange_oob_for_i32() {
        // Synthetic OOB check: i32::arange beyond i32::MAX would fail.
        // The conversion is via i64; we can't easily trigger that with
        // a usize, so pick the first IntElement-fail path another way:
        // i32 from i64 OOB.
        assert!(i32::try_from_i64(i64::MAX).is_none());
    }

    #[test]
    fn cast_i64_to_i32_in_range() {
        let t = IntTensor::<i64>::from_vec(vec![1, -1, 100], vec![3]).unwrap();
        let c = t.cast::<i32>().unwrap();
        assert_eq!(c.data(), &[1, -1, 100]);
        assert_eq!(c.dtype_name(), "i32");
    }

    #[test]
    fn cast_i64_to_i32_out_of_range_errors() {
        let t = IntTensor::<i64>::from_vec(vec![i64::MAX], vec![1]).unwrap();
        let err = t.cast::<i32>().unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn reshape_preserves_data() {
        let t = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4, 5, 6], vec![6]).unwrap();
        let r = t.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.data(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn reshape_size_mismatch_errors() {
        let t = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4], vec![4]).unwrap();
        let err = t.reshape(&[3, 2]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn scalar_constructor() {
        let t = IntTensor::<i64>::scalar(42);
        assert_eq!(t.shape(), &[] as &[usize]);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data()[0], 42);
    }

    #[test]
    fn dtype_name_reports_i32_or_i64() {
        let t32 = IntTensor::<i32>::scalar(0);
        let t64 = IntTensor::<i64>::scalar(0);
        assert_eq!(t32.dtype_name(), "i32");
        assert_eq!(t64.dtype_name(), "i64");
    }

    #[test]
    fn clone_shares_data_arc() {
        let t = IntTensor::<i32>::arange(4).unwrap();
        let t2 = t.clone();
        // Same Arc → same buffer ptr.
        assert!(Arc::ptr_eq(&t.data, &t2.data));
    }
}
