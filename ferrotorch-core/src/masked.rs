//! Masked tensors — `torch.masked.MaskedTensor` analog.
//!
//! A [`MaskedTensor`] pairs a data tensor with a boolean mask, where mask
//! entries indicate which positions are "valid". Reductions, arithmetic,
//! and `to_tensor` / `filled` all honour the mask.
//!
//! # Mask convention
//!
//! Matches `torch.masked.MaskedTensor`: `mask[i] == true` means the value
//! is valid (use it); `mask[i] == false` means the value is masked out
//! (ignored by reductions, replaced by `fill_value` when materialised).
//! This is the **opposite** of NumPy's `numpy.ma`, which uses
//! `mask=True` to mean "invalid". Helpers below translate at the
//! boundary when delegating to [`ferray_ma`].
//!
//! # GPU discipline
//!
//! CPU-only today. Constructors and reductions reject GPU input tensors
//! with [`FerrotorchError::NotImplementedOnCuda`] — they never silently
//! download data through host. A future GPU `MaskedTensor` would lower
//! reductions to existing kernels (`gpu_softmax_into` + `where_cond`,
//! `masked_select`, etc.) and stays a separate concern (#569 follow-up).

use ferray_core::{Array as FerrayArray, IxDyn as FerrayIxDyn};
use ferray_ma::masked_array::MaskedArray;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// MaskedTensor
// ---------------------------------------------------------------------------

/// A tensor paired with a boolean mask.
///
/// `mask[i] == true` means the entry is **valid**; `false` means it is
/// **masked out**. This matches `torch.masked.MaskedTensor`.
///
/// `fill_value` is substituted for masked entries when [`to_tensor`] /
/// [`filled`](Self::filled) is called. Defaults to zero.
#[derive(Clone, Debug)]
pub struct MaskedTensor<T: Float> {
    data: Tensor<T>,
    /// Length equals `data.numel()`. Stored flat in C-order to match the
    /// underlying tensor layout.
    mask: Vec<bool>,
    fill_value: T,
}

impl<T: Float> MaskedTensor<T> {
    /// Build a masked tensor from a data tensor + boolean mask.
    ///
    /// `mask` must have exactly `data.numel()` elements. Accepts both CPU
    /// and CUDA tensors — GPU paths in [`masked_sum`] / [`masked_mean`]
    /// lower to `mul + reduce_sum`. (#597)
    ///
    /// Mask convention: `mask[i] == true` means VALID (torch convention).
    pub fn new(data: Tensor<T>, mask: Vec<bool>) -> FerrotorchResult<Self> {
        if mask.len() != data.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MaskedTensor::new: mask length {} != data numel {}",
                    mask.len(),
                    data.numel()
                ),
            });
        }
        Ok(Self {
            data,
            mask,
            fill_value: <T as num_traits::Zero>::zero(),
        })
    }

    /// Build a masked tensor from data only, with all entries marked valid.
    pub fn from_data(data: Tensor<T>) -> FerrotorchResult<Self> {
        let n = data.numel();
        Self::new(data, vec![true; n])
    }

    /// Override the fill value used by [`Self::filled`] / [`Self::to_tensor`].
    pub fn with_fill_value(mut self, fill_value: T) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// The underlying data tensor (regardless of mask).
    #[inline]
    pub fn data(&self) -> &Tensor<T> {
        &self.data
    }

    /// Borrow the boolean mask. `true` = valid, `false` = masked out.
    #[inline]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// The fill value used when materialising masked entries.
    #[inline]
    pub fn fill_value(&self) -> T {
        self.fill_value
    }

    /// Logical shape (same as `data().shape()`).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Total number of entries, masked or not.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.numel()
    }

    /// Number of entries currently marked valid.
    pub fn count_valid(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// Number of entries currently masked out.
    pub fn count_masked(&self) -> usize {
        self.mask.iter().filter(|&&v| !v).count()
    }

    /// Materialise into a plain `Tensor<T>` by substituting `fill_value`
    /// at every masked-out position.
    pub fn filled(&self) -> FerrotorchResult<Tensor<T>> {
        let data_vec = self.data.data_vec()?;
        let out: Vec<T> = data_vec
            .iter()
            .zip(self.mask.iter())
            .map(|(&v, &valid)| if valid { v } else { self.fill_value })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(out), self.data.shape().to_vec(), false)
    }

    /// Alias of [`Self::filled`] mirroring `torch.Tensor`'s naming.
    #[inline]
    pub fn to_tensor(&self) -> FerrotorchResult<Tensor<T>> {
        self.filled()
    }
}

// ---------------------------------------------------------------------------
// ferray-ma bridge
//
// ferray-ma's MaskedArray uses NumPy semantics (mask=true means INVALID).
// We invert at the boundary so internal callers see the torch convention
// (mask=true means VALID).
// ---------------------------------------------------------------------------

impl<T: Float> MaskedTensor<T> {
    /// Convert to a `ferray_ma::MaskedArray<U, IxDyn>` for delegating to
    /// ferray-ma's wider op surface (var/std, masked sort, ufunc support,
    /// etc.). Element type is generic over `U: Float + Element` because
    /// ferray-ma's bound is more restrictive than ferrotorch's `Float`
    /// trait — typical choices are `f32` or `f64`.
    ///
    /// Inverts the mask to match NumPy semantics (`true` = invalid)
    /// since ferrotorch uses the torch convention (`true` = valid).
    pub fn to_ferray<U>(&self, op: &'static str) -> FerrotorchResult<MaskedArray<U, FerrayIxDyn>>
    where
        U: ferray_core::Element + Copy + num_traits::Float + 'static,
    {
        let data_vec = self.data.data_vec()?;
        let data_u: Vec<U> = data_vec
            .into_iter()
            .map(|v| U::from(v.to_f64().unwrap()).unwrap())
            .collect();
        let arr =
            FerrayArray::<U, FerrayIxDyn>::from_vec(FerrayIxDyn::new(self.data.shape()), data_u)
                .map_err(FerrotorchError::Ferray)?;
        // Invert mask: ferrotorch true=valid → numpy true=invalid.
        let inv: Vec<bool> = self.mask.iter().map(|&v| !v).collect();
        let mask_arr =
            FerrayArray::<bool, FerrayIxDyn>::from_vec(FerrayIxDyn::new(self.data.shape()), inv)
                .map_err(FerrotorchError::Ferray)?;
        MaskedArray::new(arr, mask_arr).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("{op}: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Reductions (sum / mean / count)
// ---------------------------------------------------------------------------

/// Sum of valid entries; returns a 0-d tensor.
///
/// Mirrors `torch.masked.MaskedTensor.sum()` (torch.masked uses the same
/// "ignore masked, sum the rest" semantics as numpy.ma).
///
/// On GPU, lowers to `data * mask_as_float → reduce_sum` (#597). The mask
/// is uploaded once and reused for `masked_mean`'s denominator if both
/// are computed.
pub fn masked_sum<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    if mt.data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        return masked_sum_gpu(mt);
    }
    if mt.data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "masked_sum" });
    }
    // Walk the data + mask in one pass.
    let data = mt.data.data_vec()?;
    let mut acc = <T as num_traits::Zero>::zero();
    for (&v, &valid) in data.iter().zip(mt.mask.iter()) {
        if valid {
            acc += v;
        }
    }
    Tensor::from_storage(TensorStorage::cpu(vec![acc]), vec![], false)
}

/// GPU lowering: build a float-valued mask tensor, multiply, reduce-sum.
fn masked_sum_gpu<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let device = mt.data.device();
    let mask_t: Tensor<T> = mask_as_float_tensor(&mt.mask, mt.data.shape(), device)?;
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let numel = mt.data.numel();
    let prod_h = if is_f32::<T>() {
        backend.mul_f32(mt.data.gpu_handle()?, mask_t.gpu_handle()?)?
    } else {
        backend.mul_f64(mt.data.gpu_handle()?, mask_t.gpu_handle()?)?
    };
    let sum_h = if is_f32::<T>() {
        backend.sum_f32(&prod_h, numel)?
    } else {
        backend.sum_f64(&prod_h, numel)?
    };
    Tensor::from_storage(TensorStorage::gpu(sum_h), vec![], false)
}

/// Build a float Tensor<T> on `device` from a bool mask, with shape
/// matching the masked-tensor data. true → 1, false → 0.
fn mask_as_float_tensor<T: Float>(
    mask: &[bool],
    shape: &[usize],
    device: crate::device::Device,
) -> FerrotorchResult<Tensor<T>> {
    let one = T::from(1.0).unwrap();
    let zero = <T as num_traits::Zero>::zero();
    let data: Vec<T> = mask.iter().map(|&b| if b { one } else { zero }).collect();
    let cpu = Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)?;
    if device.is_cuda() {
        cpu.to(device)
    } else {
        Ok(cpu)
    }
}

/// Helper: are we operating on `f32`?
#[inline]
fn is_f32<T: Float>() -> bool {
    std::mem::size_of::<T>() == 4
}

/// Helper: are we operating on `f64`?
#[inline]
fn is_f64<T: Float>() -> bool {
    std::mem::size_of::<T>() == 8
}

/// Mean of valid entries; returns a 0-d tensor.
///
/// If every entry is masked, returns `NaN` (matches torch.masked).
///
/// GPU path computes `sum(data * mask_f) / count_valid` using the same
/// `mul + reduce_sum` lowering as [`masked_sum`]. Empty-mask case is
/// detected on host (the count is constant in `mask` so no GPU round-trip
/// is needed for it). (#597)
pub fn masked_mean<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    if mt.data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        return masked_mean_gpu(mt);
    }
    if mt.data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "masked_mean" });
    }
    let data = mt.data.data_vec()?;
    let mut acc = <T as num_traits::Zero>::zero();
    let mut count: usize = 0;
    for (&v, &valid) in data.iter().zip(mt.mask.iter()) {
        if valid {
            acc += v;
            count += 1;
        }
    }
    let val = if count == 0 {
        T::from(f64::NAN).unwrap()
    } else {
        acc / T::from(count as f64).unwrap()
    };
    Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
}

fn masked_mean_gpu<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let count = mt.count_valid();
    if count == 0 {
        // All-masked → NaN. Skip GPU work entirely.
        let nan = T::from(f64::NAN).unwrap();
        return Tensor::from_storage(TensorStorage::cpu(vec![nan]), vec![], false);
    }
    let sum = masked_sum_gpu(mt)?;
    // sum is a 0-d tensor on GPU. Divide by count on host (single element).
    // Pull just the one element back, divide, return a 0-d CPU tensor.
    let sum_val = sum.cpu()?.data()?[0];
    let mean = sum_val / T::from(count as f64).unwrap();
    Tensor::from_storage(TensorStorage::cpu(vec![mean]), vec![], false)
}

/// Min of valid entries; returns a 0-d tensor (NaN if all masked).
///
/// GPU path: uses the fused `masked_reduce_min` PTX kernel (#627). Single
/// launch reads `(data, mask_f)` directly and combines `mask_f != 0 ?
/// data : +inf` into the running min accumulator — no intermediate
/// buffers, no CPU-side sentinel construction. Same f32/f64-only gate as
/// `masked_sum` / `masked_mean`; other dtypes (bf16/f16) take the CPU
/// walk, matching the existing masked surface.
pub fn masked_min<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    if mt.data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        return masked_extremum_gpu(mt, true);
    }
    masked_extremum_cpu(mt, true)
}

/// Max of valid entries; returns a 0-d tensor (NaN if all masked).
pub fn masked_max<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    if mt.data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        return masked_extremum_gpu(mt, false);
    }
    masked_extremum_cpu(mt, false)
}

/// CPU implementation: walk data + mask in one pass.
fn masked_extremum_cpu<T: Float>(
    mt: &MaskedTensor<T>,
    pick_min: bool,
) -> FerrotorchResult<Tensor<T>> {
    let device = mt.data.device();
    let data = mt.data.data_vec()?;
    let mut best: Option<T> = None;
    for (&v, &valid) in data.iter().zip(mt.mask.iter()) {
        if !valid {
            continue;
        }
        best = Some(match best {
            None => v,
            Some(b) if pick_min => {
                if v < b {
                    v
                } else {
                    b
                }
            }
            Some(b) => {
                if v > b {
                    v
                } else {
                    b
                }
            }
        });
    }
    let val = best.unwrap_or_else(|| T::from(f64::NAN).unwrap());
    let cpu = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)?;
    if device.is_cuda() {
        cpu.to(device)
    } else {
        Ok(cpu)
    }
}

/// GPU lowering via the **fused** masked-reduce kernel (#627).
///
/// Single PTX launch that combines `mask_f[i] != 0 ? data[i] : ±inf`
/// directly into the running min/max accumulator. No intermediate
/// `prod` / `filled` buffers, no CPU-side sentinel construction — the
/// only data uploaded is the float mask itself, which we already need
/// for the indicator role.
fn masked_extremum_gpu<T: Float>(
    mt: &MaskedTensor<T>,
    pick_min: bool,
) -> FerrotorchResult<Tensor<T>> {
    // All-masked → NaN, short-circuit before allocating GPU buffers.
    if mt.count_valid() == 0 {
        let nan = T::from(f64::NAN).unwrap();
        return Tensor::from_storage(TensorStorage::cpu(vec![nan]), vec![], false);
    }

    let device = mt.data.device();
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let numel = mt.data.numel();

    // Build the [0/1] float mask on device. This is the only host upload —
    // the mask is fundamentally a boolean Vec on the host side, so it has
    // to land on the device once per call regardless. The fused kernel
    // reads it directly and folds the sentinel-fill into the reduce.
    let mask_t: Tensor<T> = mask_as_float_tensor(&mt.mask, mt.data.shape(), device)?;

    let result_h = if pick_min {
        if is_f32::<T>() {
            backend.masked_min_f32(mt.data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
        } else {
            backend.masked_min_f64(mt.data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
        }
    } else if is_f32::<T>() {
        backend.masked_max_f32(mt.data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
    } else {
        backend.masked_max_f64(mt.data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
    };

    Tensor::from_storage(TensorStorage::gpu(result_h), vec![], false)
}

/// Number of valid (unmasked) entries; returns a 0-d tensor in `T`.
pub fn masked_count<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let n = mt.count_valid();
    Tensor::from_storage(
        TensorStorage::cpu(vec![T::from(n as f64).unwrap()]),
        vec![],
        false,
    )
}

// ---------------------------------------------------------------------------
// Constructors mirroring numpy.ma / torch.masked
// ---------------------------------------------------------------------------

/// Wrap `data` with `condition` interpreted as "where condition is true,
/// mask the value out". Matches `numpy.ma.masked_where`. The resulting
/// [`MaskedTensor`] has `mask = !condition` under the torch convention.
pub fn masked_where<T: Float>(
    data: Tensor<T>,
    condition: &[bool],
) -> FerrotorchResult<MaskedTensor<T>> {
    if condition.len() != data.numel() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_where: condition length {} != data numel {}",
                condition.len(),
                data.numel()
            ),
        });
    }
    let mask: Vec<bool> = condition.iter().map(|&c| !c).collect();
    MaskedTensor::new(data, mask)
}

/// Mask out non-finite entries (NaN, ±∞). Matches `numpy.ma.masked_invalid`.
pub fn masked_invalid<T: Float>(data: Tensor<T>) -> FerrotorchResult<MaskedTensor<T>> {
    if data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "masked_invalid",
        });
    }
    let data_vec = data.data_vec()?;
    // mask=true means VALID, so finite -> true.
    let mask: Vec<bool> = data_vec
        .iter()
        .map(|v| {
            let f = v.to_f64().unwrap();
            f.is_finite()
        })
        .collect();
    MaskedTensor::new(data, mask)
}

/// Mask out entries equal to `value`. Matches `numpy.ma.masked_equal`.
pub fn masked_equal<T: Float + PartialEq>(
    data: Tensor<T>,
    value: T,
) -> FerrotorchResult<MaskedTensor<T>> {
    if data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "masked_equal" });
    }
    let data_vec = data.data_vec()?;
    let mask: Vec<bool> = data_vec.iter().map(|&v| v != value).collect();
    MaskedTensor::new(data, mask)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::tensor;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ----- Construction --------------------------------------------------

    #[test]
    fn new_with_matching_mask() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let m = MaskedTensor::new(d, vec![true, false, true]).unwrap();
        assert_eq!(m.shape(), &[3]);
        assert_eq!(m.numel(), 3);
        assert_eq!(m.count_valid(), 2);
        assert_eq!(m.count_masked(), 1);
    }

    #[test]
    fn new_rejects_mask_length_mismatch() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let err = MaskedTensor::new(d, vec![true, false]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn from_data_marks_all_valid() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let m = MaskedTensor::from_data(d).unwrap();
        assert_eq!(m.count_valid(), 3);
        assert_eq!(m.count_masked(), 0);
    }

    // ----- masked_where (numpy-style) ------------------------------------

    #[test]
    fn masked_where_inverts_condition() {
        // condition=[F, T, F, T] → mask=[T, F, T, F] (i.e. positions 1 and 3
        // are masked OUT in torch convention).
        let d = t(&[10.0, 20.0, 30.0, 40.0], &[4]);
        let mt = masked_where(d, &[false, true, false, true]).unwrap();
        assert_eq!(mt.mask(), &[true, false, true, false]);
        assert_eq!(mt.count_valid(), 2);
    }

    // ----- masked_invalid ------------------------------------------------

    #[test]
    fn masked_invalid_masks_nan() {
        let d = t(&[1.0, f64::NAN, 3.0, f64::INFINITY], &[4]);
        let mt = masked_invalid(d).unwrap();
        // 1.0 finite → valid; NaN → invalid; 3.0 finite → valid; inf → invalid
        assert_eq!(mt.mask(), &[true, false, true, false]);
    }

    // ----- masked_equal --------------------------------------------------

    #[test]
    fn masked_equal_masks_matching() {
        let d = t(&[1.0, 5.0, 5.0, 2.0], &[4]);
        let mt = masked_equal(d, 5.0).unwrap();
        // 5.0 → masked OUT; others → valid.
        assert_eq!(mt.mask(), &[true, false, false, true]);
    }

    // ----- Reductions ----------------------------------------------------

    #[test]
    fn masked_sum_skips_masked_entries() {
        let d = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        // Mask out 2 and 4: valid = 1, 3, 5 → sum 9.
        let mt = MaskedTensor::new(d, vec![true, false, true, false, true]).unwrap();
        let s = masked_sum(&mt).unwrap();
        assert!(close(s.data().unwrap()[0], 9.0, 1e-12));
    }

    #[test]
    fn masked_mean_divides_by_valid_count() {
        let d = t(&[10.0, 0.0, 30.0, 0.0, 50.0], &[5]);
        // valid: 10, 30, 50 → mean 30
        let mt = MaskedTensor::new(d, vec![true, false, true, false, true]).unwrap();
        let r = masked_mean(&mt).unwrap();
        assert!(close(r.data().unwrap()[0], 30.0, 1e-12));
    }

    #[test]
    fn masked_mean_all_masked_returns_nan() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![false, false, false]).unwrap();
        let r = masked_mean(&mt).unwrap();
        assert!(r.data().unwrap()[0].is_nan());
    }

    #[test]
    fn masked_min_max_skip_masked() {
        let d = t(&[5.0, 1.0, 9.0, 2.0], &[4]);
        // Mask out the 9.0 (max) and 1.0 (min) → among valids: 5.0, 2.0
        // min=2.0, max=5.0
        let mt = MaskedTensor::new(d, vec![true, false, false, true]).unwrap();
        assert!(close(
            masked_min(&mt).unwrap().data().unwrap()[0],
            2.0,
            1e-12
        ));
        assert!(close(
            masked_max(&mt).unwrap().data().unwrap()[0],
            5.0,
            1e-12
        ));
    }

    #[test]
    fn masked_count_returns_valid_count() {
        let d = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let mt = MaskedTensor::new(d, vec![true, false, true, true]).unwrap();
        let c = masked_count(&mt).unwrap();
        assert_eq!(c.data().unwrap()[0], 3.0);
    }

    // ----- filled / to_tensor --------------------------------------------

    #[test]
    fn filled_substitutes_default_zero() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![true, false, true]).unwrap();
        let f = mt.filled().unwrap();
        assert_eq!(f.data().unwrap(), &[1.0, 0.0, 3.0]);
    }

    #[test]
    fn filled_uses_fill_value() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![true, false, true])
            .unwrap()
            .with_fill_value(-99.0);
        let f = mt.filled().unwrap();
        assert_eq!(f.data().unwrap(), &[1.0, -99.0, 3.0]);
    }

    #[test]
    fn to_tensor_is_alias_for_filled() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![true, false, true]).unwrap();
        let a = mt.filled().unwrap();
        let b = mt.to_tensor().unwrap();
        assert_eq!(a.data().unwrap(), b.data().unwrap());
    }

    // ----- ferray-ma bridge ----------------------------------------------

    #[test]
    fn to_ferray_round_trip_mean_matches_inhouse() {
        // Cross-check our in-house masked_mean against ferray-ma's
        // MaskedArray::mean() to confirm the mask-inversion bridge is
        // semantically correct.
        let d = t(&[2.0, 4.0, 6.0, 8.0], &[4]);
        let mt = MaskedTensor::new(d, vec![true, false, true, false]).unwrap();
        let inhouse = masked_mean(&mt).unwrap().data().unwrap()[0];
        // Build ferray-ma view via our internal bridge.
        let ferray_ma_view: MaskedArray<f64, FerrayIxDyn> = mt.to_ferray("test").unwrap();
        let ferray_mean = ferray_ma_view.mean().unwrap();
        assert!(close(inhouse, ferray_mean, 1e-12));
        // Sanity: in-house value matches the closed form (2 + 6) / 2 = 4.
        assert!(close(inhouse, 4.0, 1e-12));
    }

    // ----- GPU discipline -------------------------------------------------

    #[test]
    fn constructors_accept_cpu_tensors() {
        // Sanity: every constructor path is reachable for a CPU input.
        let d = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        assert!(MaskedTensor::from_data(d.clone()).is_ok());
        assert!(masked_where(d.clone(), &[false, true, false]).is_ok());
        assert!(masked_invalid(d.clone()).is_ok());
        assert!(masked_equal(d, 2.0).is_ok());
    }

    // -------------------------------------------------------------------
    // #616: masked_min/max no longer error on GPU — they fall back to a
    // host-bounce reduce. CPU branch is exercised here; the GPU branch
    // shares the same data_vec() entry point so the same code drives both.
    // -------------------------------------------------------------------

    #[test]
    fn masked_min_max_match_cpu_definition() {
        let d = tensor(&[1.0_f64, -3.0, 5.0, 7.0]).unwrap();
        // mask: [valid, masked, valid, masked] -> visible = {1.0, 5.0}
        let mt = MaskedTensor::new(d, vec![true, false, true, false]).unwrap();
        assert_eq!(masked_min(&mt).unwrap().data().unwrap(), &[1.0]);
        assert_eq!(masked_max(&mt).unwrap().data().unwrap(), &[5.0]);
    }

    #[test]
    fn masked_min_max_all_masked_returns_nan() {
        let d = tensor(&[1.0_f64, 2.0]).unwrap();
        let mt = MaskedTensor::new(d, vec![false, false]).unwrap();
        assert!(masked_min(&mt).unwrap().data().unwrap()[0].is_nan());
        assert!(masked_max(&mt).unwrap().data().unwrap()[0].is_nan());
    }
}
