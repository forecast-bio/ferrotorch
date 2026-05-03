//! Post-training quantization (PTQ) for ferrotorch tensors.
//!
//! Provides symmetric and asymmetric quantization to INT8, INT4, and UINT8,
//! with per-tensor or per-channel granularity. Designed for inference-time
//! model compression — quantize once after training, then run forward passes
//! with reduced memory and (on supported hardware) faster matmul.

use std::collections::HashMap;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Granularity of quantization parameters (scale / zero_point).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// One scale and zero_point for the entire tensor.
    PerTensor,
    /// One scale and zero_point per slice along the given axis.
    PerChannel(usize),
}

/// Target integer dtype for quantized storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDtype {
    /// Signed 8-bit: [-128, 127].
    Int8,
    /// Signed 4-bit: [-8, 7].  Stored packed in `i8` values.
    Int4,
    /// Unsigned 8-bit: [0, 255].
    Uint8,
}

impl QuantDtype {
    /// Minimum representable value.
    #[inline]
    fn qmin(self) -> i32 {
        match self {
            QuantDtype::Int8 => -128,
            QuantDtype::Int4 => -8,
            QuantDtype::Uint8 => 0,
        }
    }

    /// Maximum representable value.
    #[inline]
    fn qmax(self) -> i32 {
        match self {
            QuantDtype::Int8 => 127,
            QuantDtype::Int4 => 7,
            QuantDtype::Uint8 => 255,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedTensor
// ---------------------------------------------------------------------------

/// A tensor stored in quantized (integer) representation.
///
/// The real value is recovered by `x = (q - zero_point) * scale`.
///
/// `scale` and `zero_point` are vectors whose length equals:
/// * 1 for `PerTensor`
/// * `shape[axis]` for `PerChannel(axis)`
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized values stored as `i8` regardless of logical dtype.
    /// For `Uint8`, the stored `i8` is reinterpreted as `u8` via
    /// wrapping cast; for `Int4` only the low 4 bits are significant.
    data: Vec<i8>,
    /// Per-tensor or per-channel scales.
    scale: Vec<f32>,
    /// Per-tensor or per-channel zero points (in quantized domain).
    zero_point: Vec<i32>,
    /// Original tensor shape.
    shape: Vec<usize>,
    /// Quantization granularity.
    scheme: QuantScheme,
    /// Target quantized dtype.
    dtype: QuantDtype,
}

impl QuantizedTensor {
    /// Number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Borrow the shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the quantized data.
    #[inline]
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Borrow the scale vector.
    #[inline]
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// Borrow the zero-point vector.
    #[inline]
    pub fn zero_point(&self) -> &[i32] {
        &self.zero_point
    }

    /// The quantization scheme used.
    #[inline]
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// The quantized dtype.
    #[inline]
    pub fn qdtype(&self) -> QuantDtype {
        self.dtype
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute scale and zero_point for a given (min, max) range and target dtype.
///
/// Uses the standard asymmetric affine quantization formula:
///   scale = (max - min) / (qmax - qmin)
///   zero_point = round(qmin - min / scale)
///
/// The range is always expanded to include zero so that `0.0` maps exactly
/// to an integer quantized value (important for zero-padding and ReLU outputs).
/// When min == max the range would collapse to zero, so this expansion also
/// prevents division-by-zero.
fn compute_scale_zp(min_val: f32, max_val: f32, dtype: QuantDtype) -> (f32, i32) {
    let qmin = dtype.qmin();
    let qmax = dtype.qmax();

    // Ensure the range includes zero (standard PyTorch behaviour).
    let min_val = min_val.min(0.0);
    let max_val = max_val.max(0.0);

    // After including zero the range is at least max(|min|, |max|) > 0,
    // but guard against the degenerate all-zeros case.
    let range = (max_val - min_val).max(f32::EPSILON);
    let scale = range / (qmax - qmin) as f32;

    // zero_point is intentionally NOT clamped to [qmin, qmax]. It is stored
    // as i32 and may lie outside the quantized integer range. This is correct
    // for asymmetric affine quantization — clamping the zero_point distorts
    // the mapping when the float range doesn't straddle zero.
    let zp = (qmin as f32 - min_val / scale).round() as i32;

    (scale, zp)
}

/// Clamp and round a float to the quantized integer range.
///
/// Returns the result as `i8`. For `Uint8` the caller passes `qmin=0`,
/// `qmax=255`; the clamped i32 is cast to `u8` first then transmuted to `i8`
/// so that values 128..=255 are preserved through the bit pattern.
#[inline]
fn quantize_val(x: f32, scale: f32, zp: i32, qmin: i32, qmax: i32, is_unsigned: bool) -> i8 {
    let q = (x / scale + zp as f32).round() as i32;
    let clamped = q.clamp(qmin, qmax);
    if is_unsigned {
        (clamped as u8) as i8
    } else {
        clamped as i8
    }
}

/// Recover the i32 quantized value from the stored `i8`, accounting for
/// unsigned dtypes where the bit pattern represents a `u8`.
#[inline]
fn stored_to_i32(val: i8, is_unsigned: bool) -> i32 {
    if is_unsigned {
        (val as u8) as i32
    } else {
        val as i32
    }
}

/// Map a linear flat index to per-channel parameters.
///
/// For a tensor of shape `[d0, d1, ..., dn]` with channel axis `axis`,
/// returns the channel index for the element at `flat_index`.
#[inline]
fn channel_index(flat_index: usize, shape: &[usize], axis: usize) -> usize {
    // stride of the channel axis = product of dims after axis.
    let stride: usize = shape[axis + 1..].iter().product();
    (flat_index / stride) % shape[axis]
}

// ---------------------------------------------------------------------------
// Quantize
// ---------------------------------------------------------------------------

/// Quantize a floating-point tensor.
///
/// # Per-tensor
///
/// Computes a single (scale, zero_point) pair from the global min/max.
///
/// # Per-channel
///
/// Computes one (scale, zero_point) per slice along the given axis. This is
/// common for weight tensors where each output channel has its own range.
pub fn quantize<T: Float>(
    tensor: &Tensor<T>,
    scheme: QuantScheme,
    dtype: QuantDtype,
) -> FerrotorchResult<QuantizedTensor> {
    let data = tensor.data()?;
    let shape = tensor.shape().to_vec();
    let numel = tensor.numel();
    let qmin = dtype.qmin();
    let qmax = dtype.qmax();

    let is_unsigned = dtype == QuantDtype::Uint8;

    match scheme {
        QuantScheme::PerTensor => {
            // Global min/max.
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for &v in data {
                let f = v.to_f32().unwrap();
                if f < min_val {
                    min_val = f;
                }
                if f > max_val {
                    max_val = f;
                }
            }

            let (scale, zp) = compute_scale_zp(min_val, max_val, dtype);

            let qdata: Vec<i8> = data
                .iter()
                .map(|&v| quantize_val(v.to_f32().unwrap(), scale, zp, qmin, qmax, is_unsigned))
                .collect();

            Ok(QuantizedTensor {
                data: qdata,
                scale: vec![scale],
                zero_point: vec![zp],
                shape,
                scheme,
                dtype,
            })
        }

        QuantScheme::PerChannel(axis) => {
            if axis >= shape.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "PerChannel axis {axis} out of range for {}-d tensor",
                        shape.len()
                    ),
                });
            }

            let num_channels = shape[axis];
            let mut mins = vec![f32::INFINITY; num_channels];
            let mut maxs = vec![f32::NEG_INFINITY; num_channels];

            for (i, &v) in data.iter().enumerate() {
                let ch = channel_index(i, &shape, axis);
                let f = v.to_f32().unwrap();
                if f < mins[ch] {
                    mins[ch] = f;
                }
                if f > maxs[ch] {
                    maxs[ch] = f;
                }
            }

            let params: Vec<(f32, i32)> = mins
                .iter()
                .zip(maxs.iter())
                .map(|(&mn, &mx)| compute_scale_zp(mn, mx, dtype))
                .collect();

            let scales: Vec<f32> = params.iter().map(|&(s, _)| s).collect();
            let zps: Vec<i32> = params.iter().map(|&(_, z)| z).collect();

            let mut qdata = Vec::with_capacity(numel);
            for (i, &v) in data.iter().enumerate() {
                let ch = channel_index(i, &shape, axis);
                qdata.push(quantize_val(
                    v.to_f32().unwrap(),
                    scales[ch],
                    zps[ch],
                    qmin,
                    qmax,
                    is_unsigned,
                ));
            }

            Ok(QuantizedTensor {
                data: qdata,
                scale: scales,
                zero_point: zps,
                shape,
                scheme,
                dtype,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Dequantize
// ---------------------------------------------------------------------------

/// Dequantize back to a floating-point tensor.
///
/// Applies the inverse mapping: `x = (q - zero_point) * scale`.
pub fn dequantize<T: Float>(qtensor: &QuantizedTensor) -> FerrotorchResult<Tensor<T>> {
    let numel = qtensor.numel();
    let mut result = Vec::with_capacity(numel);
    let is_unsigned = qtensor.dtype == QuantDtype::Uint8;

    match qtensor.scheme {
        QuantScheme::PerTensor => {
            let scale = qtensor.scale[0];
            let zp = qtensor.zero_point[0];
            for &q in &qtensor.data {
                let val = (stored_to_i32(q, is_unsigned) - zp) as f32 * scale;
                result.push(T::from(val).unwrap());
            }
        }
        QuantScheme::PerChannel(axis) => {
            for (i, &q) in qtensor.data.iter().enumerate() {
                let ch = channel_index(i, &qtensor.shape, axis);
                let val = (stored_to_i32(q, is_unsigned) - qtensor.zero_point[ch]) as f32
                    * qtensor.scale[ch];
                result.push(T::from(val).unwrap());
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(result), qtensor.shape.clone(), false)
}

// ---------------------------------------------------------------------------
// Quantized matmul
// ---------------------------------------------------------------------------

/// Multiply two quantized 2-D matrices and return a quantized result.
///
/// Strategy: accumulate in `i32` to avoid overflow, then rescale to the output
/// quantized domain. This avoids a full dequantize-matmul-requantize round-trip
/// while remaining numerically correct for INT8.
///
/// Both inputs must be 2-D, with compatible inner dimensions (standard matmul
/// rules: `[M, K] x [K, N] -> [M, N]`).
pub fn quantized_matmul(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
) -> FerrotorchResult<QuantizedTensor> {
    // Validate shapes.
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "quantized_matmul requires 2-D tensors, got shapes {:?} and {:?}",
                a.shape, b.shape
            ),
        });
    }

    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];

    if k != k2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "quantized_matmul inner dimensions mismatch: [{m}, {k}] x [{k2}, {n}]"
            ),
        });
    }

    // Both inputs must be PerTensor for the fast path.
    if a.scale.len() != 1 || b.scale.len() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: "quantized_matmul currently requires PerTensor-quantized inputs".into(),
        });
    }

    let a_scale = a.scale[0];
    let a_zp = a.zero_point[0];
    let b_scale = b.scale[0];
    let b_zp = b.zero_point[0];

    let a_unsigned = a.dtype == QuantDtype::Uint8;
    let b_unsigned = b.dtype == QuantDtype::Uint8;

    // Accumulate in i32.
    let mut acc = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for p in 0..k {
                let qa = stored_to_i32(a.data[i * k + p], a_unsigned) - a_zp;
                let qb = stored_to_i32(b.data[p * n + j], b_unsigned) - b_zp;
                sum += qa * qb;
            }
            acc[i * n + j] = sum;
        }
    }

    // The real-valued result element is: acc[i,j] * a_scale * b_scale.
    // Requantize: pick INT8 output with its own scale/zp.
    let combined_scale = a_scale * b_scale;

    // Find the real-valued min/max of the output.
    let mut out_min = f32::INFINITY;
    let mut out_max = f32::NEG_INFINITY;
    for &a_val in &acc {
        let real = a_val as f32 * combined_scale;
        if real < out_min {
            out_min = real;
        }
        if real > out_max {
            out_max = real;
        }
    }

    let out_dtype = QuantDtype::Int8;
    let (out_scale, out_zp) = compute_scale_zp(out_min, out_max, out_dtype);
    let qmin = out_dtype.qmin();
    let qmax = out_dtype.qmax();

    let qdata: Vec<i8> = acc
        .iter()
        .map(|&a_val| {
            let real = a_val as f32 * combined_scale;
            quantize_val(real, out_scale, out_zp, qmin, qmax, false)
        })
        .collect();

    Ok(QuantizedTensor {
        data: qdata,
        scale: vec![out_scale],
        zero_point: vec![out_zp],
        shape: vec![m, n],
        scheme: QuantScheme::PerTensor,
        dtype: out_dtype,
    })
}

// ---------------------------------------------------------------------------
// Module-level quantization utility
// ---------------------------------------------------------------------------

/// Quantize every weight tensor in a module, returning a name -> QuantizedTensor
/// map suitable for serialization or quantized inference.
///
/// This accepts any type implementing the `Module` trait from `ferrotorch-nn`.
/// Because `ferrotorch-core` does not depend on `ferrotorch-nn`, we accept a
/// generic iterator of named tensors instead.
pub fn quantize_named_tensors<T: Float>(
    named_tensors: impl IntoIterator<Item = (String, Tensor<T>)>,
    scheme: QuantScheme,
    dtype: QuantDtype,
) -> FerrotorchResult<HashMap<String, QuantizedTensor>> {
    let mut result = HashMap::new();
    for (name, tensor) in named_tensors {
        let qtensor = quantize(&tensor, scheme, dtype)?;
        result.insert(name, qtensor);
    }
    Ok(result)
}

// ===========================================================================
// QParams — quantization parameters
// ===========================================================================

/// Computed quantization parameters (scale and zero_point).
#[derive(Debug, Clone)]
pub struct QParams {
    /// Per-tensor or per-channel scales.
    pub scale: Vec<f32>,
    /// Per-tensor or per-channel zero points.
    pub zero_point: Vec<i32>,
}

impl QParams {
    /// Compute symmetric quantization parameters.
    ///
    /// For symmetric quantization the range is `[-max_abs, max_abs]` and:
    /// - INT8: `zero_point = 0`, `scale = max_abs / 127`
    /// - INT4: `zero_point = 0`, `scale = max_abs / 7`
    /// - UINT8: `zero_point = 128`, `scale = max_abs / 128`
    pub fn symmetric(max_abs: f32, dtype: QuantDtype) -> Self {
        let max_abs = max_abs.max(f32::EPSILON);
        match dtype {
            QuantDtype::Int8 => QParams {
                scale: vec![max_abs / 127.0],
                zero_point: vec![0],
            },
            QuantDtype::Int4 => QParams {
                scale: vec![max_abs / 7.0],
                zero_point: vec![0],
            },
            QuantDtype::Uint8 => QParams {
                scale: vec![max_abs / 128.0],
                zero_point: vec![128],
            },
        }
    }

    /// Compute asymmetric quantization parameters from observed min/max.
    pub fn asymmetric(min_val: f32, max_val: f32, dtype: QuantDtype) -> Self {
        let (scale, zp) = compute_scale_zp(min_val, max_val, dtype);
        QParams {
            scale: vec![scale],
            zero_point: vec![zp],
        }
    }
}

// ===========================================================================
// Observers — collect statistics for quantization calibration
// ===========================================================================

/// Trait for quantization observers that collect data statistics.
pub trait Observer {
    /// Update the observer with a batch of floating-point values.
    fn observe(&mut self, data: &[f32]);
    /// Calculate quantization parameters from collected statistics.
    fn calculate_qparams(&self, dtype: QuantDtype) -> QParams;
    /// Reset the observer state.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// MinMaxObserver
// ---------------------------------------------------------------------------

/// Tracks the running min/max of observed values.
///
/// Filters out NaN and Inf values before updating min/max.
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    min_val: f32,
    max_val: f32,
}

impl MinMaxObserver {
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
        }
    }
}

impl Default for MinMaxObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for MinMaxObserver {
    fn observe(&mut self, data: &[f32]) {
        for &x in data {
            if !x.is_finite() {
                continue;
            }
            if x < self.min_val {
                self.min_val = x;
            }
            if x > self.max_val {
                self.max_val = x;
            }
        }
    }

    fn calculate_qparams(&self, dtype: QuantDtype) -> QParams {
        QParams::asymmetric(self.min_val, self.max_val, dtype)
    }

    fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
    }
}

// ---------------------------------------------------------------------------
// PerChannelMinMaxObserver
// ---------------------------------------------------------------------------

/// Tracks per-channel running min/max of observed values.
///
/// Filters out NaN and Inf values before updating min/max.
/// Logs a warning and returns an error when the channel count of incoming
/// data doesn't match the configured number of channels.
#[derive(Debug, Clone)]
pub struct PerChannelMinMaxObserver {
    num_channels: usize,
    axis: usize,
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
}

impl PerChannelMinMaxObserver {
    /// Create a new per-channel observer.
    ///
    /// * `num_channels` — expected number of channels.
    /// * `axis` — the axis along which channels are sliced.
    pub fn new(num_channels: usize, axis: usize) -> Self {
        Self {
            num_channels,
            axis,
            min_vals: vec![f32::INFINITY; num_channels],
            max_vals: vec![f32::NEG_INFINITY; num_channels],
        }
    }

    /// Observe a tensor's data with the given shape.
    ///
    /// Returns `Err` if the channel count along `self.axis` doesn't match.
    pub fn observe_with_shape(&mut self, data: &[f32], shape: &[usize]) -> FerrotorchResult<()> {
        if self.axis >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PerChannelMinMaxObserver axis {} out of range for {}-d tensor",
                    self.axis,
                    shape.len()
                ),
            });
        }
        let actual_channels = shape[self.axis];
        if actual_channels != self.num_channels {
            // The `Err` below carries the same information as the previous
            // `eprintln!` (channel count, axis, observed value); duplicating
            // it on stderr is `print_stderr` noise per `rust-quality` §4.
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PerChannelMinMaxObserver expected {} channels on axis {}, got {}",
                    self.num_channels, self.axis, actual_channels
                ),
            });
        }

        for (i, &x) in data.iter().enumerate() {
            if !x.is_finite() {
                continue;
            }
            let ch = channel_index(i, shape, self.axis);
            if x < self.min_vals[ch] {
                self.min_vals[ch] = x;
            }
            if x > self.max_vals[ch] {
                self.max_vals[ch] = x;
            }
        }
        Ok(())
    }
}

impl Observer for PerChannelMinMaxObserver {
    fn observe(&mut self, data: &[f32]) {
        // Without shape info, we treat data as [num_channels, N] where N = len / num_channels.
        // If `data` isn't divisible by `num_channels`, skip silently — the
        // caller can use the shape-aware `observe_with_shape` if they need a
        // reportable error. The previous `eprintln!` was unactionable noise
        // and is forbidden by `rust-quality` §4 (`print_stderr` lint).
        if data.len() % self.num_channels != 0 {
            return;
        }
        let per_channel = data.len() / self.num_channels;
        for (i, &x) in data.iter().enumerate() {
            if !x.is_finite() {
                continue;
            }
            let ch = i / per_channel;
            if ch >= self.num_channels {
                continue;
            }
            if x < self.min_vals[ch] {
                self.min_vals[ch] = x;
            }
            if x > self.max_vals[ch] {
                self.max_vals[ch] = x;
            }
        }
    }

    fn calculate_qparams(&self, dtype: QuantDtype) -> QParams {
        let params: Vec<(f32, i32)> = self
            .min_vals
            .iter()
            .zip(self.max_vals.iter())
            .map(|(&mn, &mx)| compute_scale_zp(mn, mx, dtype))
            .collect();
        QParams {
            scale: params.iter().map(|&(s, _)| s).collect(),
            zero_point: params.iter().map(|&(_, z)| z).collect(),
        }
    }

    fn reset(&mut self) {
        self.min_vals.fill(f32::INFINITY);
        self.max_vals.fill(f32::NEG_INFINITY);
    }
}

// ---------------------------------------------------------------------------
// HistogramObserver
// ---------------------------------------------------------------------------

/// Histogram-based observer that collects a distribution of values.
///
/// When the observed range expands, existing bin counts are redistributed
/// into the new bin layout via linear interpolation rather than being zeroed.
#[derive(Debug, Clone)]
pub struct HistogramObserver {
    num_bins: usize,
    bins: Vec<u64>,
    min_val: f32,
    max_val: f32,
    /// Whether we've seen any data yet.
    initialized: bool,
}

impl HistogramObserver {
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            bins: vec![0u64; num_bins],
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            initialized: false,
        }
    }

    /// Redistribute old bins into a new range via linear interpolation.
    fn redistribute(&mut self, new_min: f32, new_max: f32) {
        if !self.initialized || self.bins.iter().all(|&c| c == 0) {
            self.min_val = new_min;
            self.max_val = new_max;
            return;
        }

        let old_min = self.min_val;
        let old_max = self.max_val;
        let old_range = old_max - old_min;
        let new_range = new_max - new_min;

        if old_range <= 0.0 || new_range <= 0.0 {
            self.min_val = new_min;
            self.max_val = new_max;
            return;
        }

        let n = self.num_bins;
        let old_bins = self.bins.clone();
        self.bins.fill(0);

        let old_bin_width = old_range / n as f32;
        let new_bin_width = new_range / n as f32;

        for (old_idx, &old_count) in old_bins.iter().enumerate().take(n) {
            if old_count == 0 {
                continue;
            }
            // Center of the old bin in value space.
            let old_center = old_min + (old_idx as f32 + 0.5) * old_bin_width;
            // Map to new bin index.
            let new_frac = (old_center - new_min) / new_bin_width;
            let new_idx = (new_frac as usize).min(n - 1);
            self.bins[new_idx] += old_count;
        }

        self.min_val = new_min;
        self.max_val = new_max;
    }
}

impl Observer for HistogramObserver {
    fn observe(&mut self, data: &[f32]) {
        // First pass: find min/max of new data, filtering NaN/Inf.
        let mut batch_min = f32::INFINITY;
        let mut batch_max = f32::NEG_INFINITY;
        for &x in data {
            if !x.is_finite() {
                continue;
            }
            if x < batch_min {
                batch_min = x;
            }
            if x > batch_max {
                batch_max = x;
            }
        }

        if batch_min > batch_max {
            // No finite values in this batch.
            return;
        }

        // Check if range needs expanding.
        let new_min = if self.initialized {
            self.min_val.min(batch_min)
        } else {
            batch_min
        };
        let new_max = if self.initialized {
            self.max_val.max(batch_max)
        } else {
            batch_max
        };

        if self.initialized && (new_min < self.min_val || new_max > self.max_val) {
            // Range expanded — redistribute existing counts into new layout.
            self.redistribute(new_min, new_max);
        } else if !self.initialized {
            self.min_val = new_min;
            self.max_val = new_max;
            self.initialized = true;
        }

        // Insert new data into bins.
        let range = (self.max_val - self.min_val).max(f32::EPSILON);
        let n = self.num_bins;
        for &x in data {
            if !x.is_finite() {
                continue;
            }
            let frac = (x - self.min_val) / range;
            let idx = ((frac * n as f32) as usize).min(n - 1);
            self.bins[idx] += 1;
        }
    }

    fn calculate_qparams(&self, dtype: QuantDtype) -> QParams {
        QParams::asymmetric(self.min_val, self.max_val, dtype)
    }

    fn reset(&mut self) {
        self.bins.fill(0);
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
        self.initialized = false;
    }
}

// ===========================================================================
// FakeQuantize — differentiable quantize/dequantize for QAT
// ===========================================================================

/// Simulates quantization during training by quantizing and immediately
/// dequantizing values, while allowing gradients to flow through via the
/// straight-through estimator (STE).
///
/// Implements clipped STE: gradients are passed through unchanged for
/// values within the quantization range `[dequantize(qmin), dequantize(qmax)]`,
/// and zeroed for out-of-range values.
#[derive(Debug, Clone)]
pub struct FakeQuantize {
    /// Target quantized dtype.
    pub dtype: QuantDtype,
    /// Cached quantization parameters.
    pub qparams: Option<QParams>,
    /// Whether the observer is enabled (collects statistics).
    pub observer_enabled: bool,
    /// Whether fake quantization is enabled.
    pub fake_quant_enabled: bool,
    /// The observer used to compute qparams.
    observer: MinMaxObserver,
}

impl FakeQuantize {
    /// Create a new FakeQuantize module.
    pub fn new(dtype: QuantDtype) -> Self {
        Self {
            dtype,
            qparams: None,
            observer_enabled: true,
            fake_quant_enabled: true,
            observer: MinMaxObserver::new(),
        }
    }

    /// Forward pass: observe data, fake-quantize, and return the result.
    ///
    /// Returns the fake-quantized data and a gradient mask for clipped STE.
    /// The mask is 1.0 for in-range values and 0.0 for out-of-range values.
    pub fn forward(&mut self, data: &[f32]) -> (Vec<f32>, Vec<f32>) {
        if !self.fake_quant_enabled {
            let ones = vec![1.0f32; data.len()];
            return (data.to_vec(), ones);
        }

        // Observe if enabled.
        if self.observer_enabled {
            self.observer.observe(data);
        }

        // Calculate or use cached qparams.
        // When observer is disabled and we have cached params, skip recalculation.
        let qparams = if let Some(cached) = self.qparams.as_ref().filter(|_| !self.observer_enabled)
        {
            cached.clone()
        } else {
            let qp = self.observer.calculate_qparams(self.dtype);
            self.qparams = Some(qp.clone());
            qp
        };

        let scale = qparams.scale[0];
        let zp = qparams.zero_point[0];
        let qmin = self.dtype.qmin();
        let qmax = self.dtype.qmax();

        // Compute the dequantized range boundaries for clipped STE.
        let range_min = (qmin as f32 - zp as f32) * scale;
        let range_max = (qmax as f32 - zp as f32) * scale;

        let mut output = Vec::with_capacity(data.len());
        let mut grad_mask = Vec::with_capacity(data.len());

        for &x in data {
            // Fake quantize: quantize then dequantize.
            let q = (x / scale + zp as f32)
                .round()
                .clamp(qmin as f32, qmax as f32);
            let dq = (q - zp as f32) * scale;
            output.push(dq);

            // Clipped STE: zero gradient for out-of-range inputs.
            if x >= range_min && x <= range_max {
                grad_mask.push(1.0);
            } else {
                grad_mask.push(0.0);
            }
        }

        (output, grad_mask)
    }
}

// ===========================================================================
// QatModel — quantization-aware training wrapper
// ===========================================================================

/// A layer with associated FakeQuantize modules for QAT.
#[derive(Debug, Clone)]
pub struct QatLayer {
    /// FakeQuantize for this layer's weights.
    pub weight_fq: FakeQuantize,
    /// FakeQuantize for this layer's activations (applied after forward).
    pub activation_fq: FakeQuantize,
}

/// Wraps a collection of named weight tensors for quantization-aware training.
///
/// Applies `FakeQuantize` to weights before forward and to activations after
/// each layer's forward pass. Original weights are saved before fake-quantization
/// and restored after forward to preserve full-precision values for gradient
/// updates.
#[derive(Debug)]
pub struct QatModel {
    /// Per-layer FakeQuantize state, keyed by layer name.
    pub layers: HashMap<String, QatLayer>,
    /// Target quantized dtype.
    pub dtype: QuantDtype,
}

impl QatModel {
    /// Create a new QAT model wrapper.
    pub fn new(dtype: QuantDtype) -> Self {
        Self {
            layers: HashMap::new(),
            dtype,
        }
    }

    /// Register a layer for QAT.
    pub fn register_layer(&mut self, name: &str) {
        self.layers.insert(
            name.to_string(),
            QatLayer {
                weight_fq: FakeQuantize::new(self.dtype),
                activation_fq: FakeQuantize::new(self.dtype),
            },
        );
    }

    /// Fake-quantize weights for a named layer.
    ///
    /// Returns `(fake_quantized_weights, original_weights)` so the caller
    /// can restore originals after the forward pass.
    pub fn fake_quantize_weights(
        &mut self,
        layer_name: &str,
        weights: &[f32],
    ) -> FerrotorchResult<(Vec<f32>, Vec<f32>)> {
        let layer =
            self.layers
                .get_mut(layer_name)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("layer '{layer_name}' not registered for QAT"),
                })?;

        // Save original weights.
        let originals = weights.to_vec();

        // Fake-quantize (gradient mask is used during backward, not here).
        let (fq_weights, _mask) = layer.weight_fq.forward(weights);

        Ok((fq_weights, originals))
    }

    /// Fake-quantize activations for a named layer.
    ///
    /// Applied after each layer's forward output, not just the last layer.
    pub fn fake_quantize_activations(
        &mut self,
        layer_name: &str,
        activations: &[f32],
    ) -> FerrotorchResult<(Vec<f32>, Vec<f32>)> {
        let layer =
            self.layers
                .get_mut(layer_name)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("layer '{layer_name}' not registered for QAT"),
                })?;

        let (fq_activations, grad_mask) = layer.activation_fq.forward(activations);
        Ok((fq_activations, grad_mask))
    }
}

/// Prepare a set of named parameters for quantization-aware training.
///
/// Creates a `QatModel` and registers layers. Only parameters whose name
/// contains "weight" get weight FakeQuantize; bias parameters are skipped.
pub fn prepare_qat(param_names: &[&str], dtype: QuantDtype) -> QatModel {
    let mut model = QatModel::new(dtype);

    for &name in param_names {
        // Extract the layer name (everything before the last `.weight` or `.bias`).
        let layer_name = if let Some(prefix) = name.strip_suffix(".weight") {
            prefix
        } else if let Some(prefix) = name.strip_suffix(".bias") {
            // Only register the layer if not already registered — don't apply
            // weight FakeQuantize to bias parameters.
            if !model.layers.contains_key(prefix) {
                model.register_layer(prefix);
            }
            continue;
        } else {
            name
        };

        model.register_layer(layer_name);
    }

    model
}

// ===========================================================================
// CUDA RNG — fork/join for reproducible GPU random state
// ===========================================================================

/// Thread-safe GPU RNG state for fork/join semantics.
///
/// Uses `Mutex` with graceful poisoning recovery to avoid panics
/// when a thread panics while holding the lock.
pub mod cuda_rng {
    use std::sync::Mutex;

    /// Global RNG state — a simple seed counter.
    static RNG_STATE: Mutex<u64> = Mutex::new(0xdeadbeef_cafebabe);

    /// Saved RNG states for fork/join.
    static RNG_STACK: Mutex<Vec<u64>> = Mutex::new(Vec::new());

    /// Get the current RNG state, recovering gracefully from mutex poisoning.
    pub fn get_state() -> u64 {
        let guard = RNG_STATE.lock().unwrap_or_else(|e| e.into_inner());
        *guard
    }

    /// Set the RNG state.
    pub fn set_state(state: u64) {
        let mut guard = RNG_STATE.lock().unwrap_or_else(|e| e.into_inner());
        *guard = state;
    }

    /// Save the current RNG state to a stack and set a new state.
    ///
    /// Uses `unwrap_or_else(|e| e.into_inner())` to handle poisoned mutexes
    /// gracefully instead of panicking.
    pub fn fork_rng(new_seed: u64) {
        let current = {
            let guard = RNG_STATE.lock().unwrap_or_else(|e| e.into_inner());
            *guard
        };

        {
            let mut stack = RNG_STACK.lock().unwrap_or_else(|e| e.into_inner());
            stack.push(current);
        }

        set_state(new_seed);
    }

    /// Restore the previously saved RNG state from the stack.
    ///
    /// Uses `unwrap_or_else(|e| e.into_inner())` to handle poisoned mutexes
    /// gracefully instead of panicking.
    pub fn join_rng() {
        let saved = {
            let mut stack = RNG_STACK.lock().unwrap_or_else(|e| e.into_inner());
            stack.pop()
        };

        if let Some(state) = saved {
            set_state(state);
        }
    }

    /// Advance the RNG state and return the new value.
    pub fn next_seed() -> u64 {
        let mut guard = RNG_STATE.lock().unwrap_or_else(|e| e.into_inner());
        // Simple splitmix64 step.
        *guard = guard.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *guard;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a tensor from f32 data.
    fn make_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        crate::from_slice(data, shape).unwrap()
    }

    // ----- Round-trip quantize/dequantize -----

    #[test]
    fn test_per_tensor_int8_roundtrip() {
        let data: Vec<f32> = (-10..=10).map(|x| x as f32 * 0.5).collect();
        let t = make_tensor(&data, &[data.len()]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        assert_eq!(rt.shape(), t.shape());
        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // INT8 over [-5, 5]: step ≈ 10/255 ≈ 0.04, max error ≈ half step ≈ 0.02
            assert!(
                err < 0.05,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_per_tensor_uint8_roundtrip() {
        let data: Vec<f32> = (0..=20).map(|x| x as f32 * 0.1).collect();
        let t = make_tensor(&data, &[data.len()]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Uint8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // UINT8 over [0, 2]: step ≈ 2/255 ≈ 0.008
            assert!(
                err < 0.02,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_per_tensor_int4_roundtrip() {
        // INT4 has only 16 levels, so larger quantization error is expected.
        let data: Vec<f32> = (-8..=7).map(|x| x as f32).collect();
        let t = make_tensor(&data, &[data.len()]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int4).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // INT4 over [-8, 7]: step = 15/15 = 1.0, max error ≈ 0.5
            assert!(
                err < 1.01,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    // ----- Per-channel -----

    #[test]
    fn test_per_channel_int8_roundtrip() {
        // Shape [3, 4]: 3 channels along axis 0, each with different ranges.
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            // channel 0: range [0, 3]
            0.0, 1.0, 2.0, 3.0,
            // channel 1: range [-10, 10]
            -10.0, -5.0, 5.0, 10.0,
            // channel 2: range [100, 200]
            100.0, 130.0, 170.0, 200.0,
        ];
        let t = make_tensor(&data, &[3, 4]);
        let qt = quantize(&t, QuantScheme::PerChannel(0), QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        assert_eq!(qt.scale.len(), 3);
        assert_eq!(qt.zero_point.len(), 3);

        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // Each channel has its own scale, so error is relative to the
            // channel's range. Worst case channel 2: 100/255 ≈ 0.39.
            assert!(
                err < 0.5,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_per_channel_axis_out_of_bounds() {
        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        let result = quantize(&t, QuantScheme::PerChannel(5), QuantDtype::Int8);
        assert!(result.is_err());
    }

    // ----- Quantized matmul -----

    #[test]
    fn test_quantized_matmul_identity() {
        // A * I should ≈ A after quantize -> matmul -> dequantize.
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = make_tensor(&a_data, &[2, 2]);
        let eye = make_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qi = quantize(&eye, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qc = quantized_matmul(&qa, &qi).unwrap();
        let c: Tensor<f32> = dequantize(&qc).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let c_data = c.data().unwrap();
        for (i, (&expected, &got)) in a_data.iter().zip(c_data.iter()).enumerate() {
            let err = (expected - got).abs();
            assert!(
                err < 0.5,
                "element {i}: expected={expected}, got={got}, error={err}"
            );
        }
    }

    #[test]
    fn test_quantized_matmul_correctness() {
        // [2,3] x [3,2] -> [2,2]
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        // B = [[7,  8],
        //      [9, 10],
        //      [11, 12]]
        // A @ B = [[ 58,  64],
        //          [139, 154]]
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qc = quantized_matmul(&qa, &qb).unwrap();
        let c: Tensor<f32> = dequantize(&qc).unwrap();

        let expected = [58.0f32, 64.0, 139.0, 154.0];
        let c_data = c.data().unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        for (i, (&e, &g)) in expected.iter().zip(c_data.iter()).enumerate() {
            let err = (e - g).abs();
            // Quantization introduces some error; for small integers in INT8
            // the error should be small relative to the values.
            assert!(err < 3.0, "element {i}: expected={e}, got={g}, error={err}");
        }
    }

    #[test]
    fn test_quantized_matmul_shape_mismatch() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let result = quantized_matmul(&qa, &qb);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_matmul_non_2d() {
        let a = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        let b = make_tensor(&[4.0, 5.0, 6.0], &[3]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let result = quantized_matmul(&qa, &qb);
        assert!(result.is_err());
    }

    // ----- Module quantization utility -----

    #[test]
    fn test_quantize_named_tensors() {
        let w1 = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w2 = make_tensor(&[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[3, 2]);

        let named = vec![
            ("layer.weight".to_string(), w1),
            ("layer2.weight".to_string(), w2),
        ];

        let qmap = quantize_named_tensors(named, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();

        assert_eq!(qmap.len(), 2);
        assert!(qmap.contains_key("layer.weight"));
        assert!(qmap.contains_key("layer2.weight"));
        assert_eq!(qmap["layer.weight"].shape(), &[2, 2]);
        assert_eq!(qmap["layer2.weight"].shape(), &[3, 2]);
    }

    // ----- Constant values / edge cases -----

    #[test]
    fn test_quantize_constant_tensor() {
        // All values identical — scale should not be zero.
        let t = make_tensor(&[5.0, 5.0, 5.0, 5.0], &[4]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        let recovered = rt.data().unwrap();
        for &r in recovered {
            assert!(
                (r - 5.0).abs() < 0.1,
                "constant tensor dequantized to {r}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_quantize_single_element() {
        let t = make_tensor(&[42.0], &[1]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();
        assert!((rt.data().unwrap()[0] - 42.0).abs() < 0.5);
    }

    #[test]
    fn test_per_channel_int4() {
        // 2 channels, 3 elements each.
        let data = vec![0.0, 1.0, 2.0, -4.0, 0.0, 4.0];
        let t = make_tensor(&data, &[2, 3]);
        let qt = quantize(&t, QuantScheme::PerChannel(0), QuantDtype::Int4).unwrap();

        assert_eq!(qt.scale.len(), 2);
        assert_eq!(qt.zero_point.len(), 2);

        let rt: Tensor<f32> = dequantize(&qt).unwrap();
        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // INT4 has coarse resolution, but channel-level ranges are small.
            assert!(
                err < 1.0,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_dequantize_f64() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = crate::from_slice(&data, &[4]).unwrap();
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f64> = dequantize(&qt).unwrap();

        assert_eq!(rt.shape(), &[4]);
        let recovered = rt.data().unwrap();
        for (i, &r) in recovered.iter().enumerate() {
            let expected = data[i] as f64;
            let err = (expected - r).abs();
            assert!(
                err < 0.05,
                "element {i}: expected={expected}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_quantized_tensor_accessors() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();

        assert_eq!(qt.numel(), 6);
        assert_eq!(qt.shape(), &[2, 3]);
        assert_eq!(qt.data().len(), 6);
        assert_eq!(qt.scale().len(), 1);
        assert_eq!(qt.zero_point().len(), 1);
        assert_eq!(qt.scheme(), QuantScheme::PerTensor);
        assert_eq!(qt.qdtype(), QuantDtype::Int8);
    }

    // ----- QParams -----

    #[test]
    fn test_qparams_symmetric_int8() {
        let qp = QParams::symmetric(5.0, QuantDtype::Int8);
        assert_eq!(qp.zero_point, vec![0]);
        assert!((qp.scale[0] - 5.0 / 127.0).abs() < 1e-7);
    }

    #[test]
    fn test_qparams_symmetric_uint8() {
        let qp = QParams::symmetric(5.0, QuantDtype::Uint8);
        assert_eq!(qp.zero_point, vec![128]);
        assert!((qp.scale[0] - 5.0 / 128.0).abs() < 1e-7);
    }

    #[test]
    fn test_qparams_symmetric_int4() {
        let qp = QParams::symmetric(7.0, QuantDtype::Int4);
        assert_eq!(qp.zero_point, vec![0]);
        assert!((qp.scale[0] - 1.0).abs() < 1e-7);
    }

    // ----- MinMaxObserver -----

    #[test]
    fn test_minmax_observer() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&[1.0, 2.0, 3.0]);
        obs.observe(&[-1.0, 5.0]);
        let qp = obs.calculate_qparams(QuantDtype::Int8);
        // Range includes zero: min=-1, max=5.
        assert_eq!(qp.scale.len(), 1);
        assert_eq!(qp.zero_point.len(), 1);
    }

    #[test]
    fn test_minmax_observer_filters_nan_inf() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&[1.0, f32::NAN, 2.0, f32::INFINITY, -1.0, f32::NEG_INFINITY]);
        let qp = obs.calculate_qparams(QuantDtype::Int8);
        // Should only see range [-1, 2], NaN/Inf filtered.
        let expected_range = 2.0 - (-1.0); // = 3.0
        let expected_scale = expected_range / 255.0;
        assert!((qp.scale[0] - expected_scale).abs() < 1e-5);
    }

    // ----- PerChannelMinMaxObserver -----

    #[test]
    fn test_per_channel_observer_with_shape() {
        let mut obs = PerChannelMinMaxObserver::new(2, 0);
        // Shape [2, 3]: channel 0 = [0, 1, 2], channel 1 = [10, 20, 30]
        obs.observe_with_shape(&[0.0, 1.0, 2.0, 10.0, 20.0, 30.0], &[2, 3])
            .unwrap();
        let qp = obs.calculate_qparams(QuantDtype::Int8);
        assert_eq!(qp.scale.len(), 2);
        assert_eq!(qp.zero_point.len(), 2);
    }

    #[test]
    fn test_per_channel_observer_shape_mismatch() {
        let mut obs = PerChannelMinMaxObserver::new(3, 0);
        // Shape [2, 3] has 2 channels on axis 0, but observer expects 3.
        let result = obs.observe_with_shape(&[1.0; 6], &[2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_per_channel_observer_axis() {
        let mut obs = PerChannelMinMaxObserver::new(3, 1);
        // Shape [2, 3]: axis 1 has 3 channels.
        obs.observe_with_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();
        let qp = obs.calculate_qparams(QuantDtype::Int8);
        assert_eq!(qp.scale.len(), 3);
    }

    #[test]
    fn test_per_channel_observer_filters_nan_inf() {
        let mut obs = PerChannelMinMaxObserver::new(2, 0);
        obs.observe_with_shape(&[f32::NAN, 1.0, 2.0, 10.0, f32::INFINITY, 30.0], &[2, 3])
            .unwrap();
        // Channel 0 should only see [1, 2], channel 1 should only see [10, 30].
        let qp = obs.calculate_qparams(QuantDtype::Int8);
        assert_eq!(qp.scale.len(), 2);
    }

    // ----- HistogramObserver -----

    #[test]
    fn test_histogram_observer_basic() {
        let mut obs = HistogramObserver::new(100);
        obs.observe(&[0.0, 0.5, 1.0]);
        let qp = obs.calculate_qparams(QuantDtype::Int8);
        assert_eq!(qp.scale.len(), 1);
    }

    #[test]
    fn test_histogram_observer_range_expansion() {
        let mut obs = HistogramObserver::new(100);
        obs.observe(&[0.0, 1.0]);
        // Initial range is [0, 1].
        let bins_after_first = obs.bins.clone();
        let total_first: u64 = bins_after_first.iter().sum();
        assert_eq!(total_first, 2);

        obs.observe(&[-1.0, 2.0]);
        // Range expanded to [-1, 2]. Old counts should be redistributed, not zeroed.
        let total_second: u64 = obs.bins.iter().sum();
        // Should have 4 total counts (2 original redistributed + 2 new).
        assert_eq!(total_second, 4);
    }

    #[test]
    fn test_histogram_observer_filters_nan_inf() {
        let mut obs = HistogramObserver::new(50);
        obs.observe(&[f32::NAN, 1.0, f32::INFINITY, 2.0]);
        let total: u64 = obs.bins.iter().sum();
        // Only 2 finite values should be counted.
        assert_eq!(total, 2);
    }

    // ----- FakeQuantize -----

    #[test]
    fn test_fake_quantize_roundtrip() {
        let mut fq = FakeQuantize::new(QuantDtype::Int8);
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let (output, mask) = fq.forward(&data);
        assert_eq!(output.len(), 5);
        assert_eq!(mask.len(), 5);

        // Output should be close to input (quantize then dequantize).
        for (i, (&o, &d)) in output.iter().zip(data.iter()).enumerate() {
            assert!((o - d).abs() < 0.1, "element {i}: output={o}, data={d}");
        }
    }

    #[test]
    fn test_fake_quantize_ste_clipping() {
        let mut fq = FakeQuantize::new(QuantDtype::Int8);
        // First, observe a range [0, 2].
        let (_, _) = fq.forward(&[0.0, 1.0, 2.0]);

        // Disable observer so range stays locked at [0, 2].
        fq.observer_enabled = false;

        // Now forward with values outside the observed range.
        let (_, mask) = fq.forward(&[0.5, 1.0, 100.0, -100.0]);
        // In-range values should have mask = 1.0.
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[1], 1.0);
        // Out-of-range values should have mask = 0.0.
        assert_eq!(mask[2], 0.0);
        assert_eq!(mask[3], 0.0);
    }

    #[test]
    fn test_fake_quantize_observer_disabled_uses_cached() {
        let mut fq = FakeQuantize::new(QuantDtype::Int8);
        // Observe initial range.
        let (_, _) = fq.forward(&[0.0, 10.0]);
        let cached_scale = fq.qparams.as_ref().unwrap().scale[0];

        // Disable observer.
        fq.observer_enabled = false;

        // Forward with a much larger range — should NOT update qparams.
        let (_, _) = fq.forward(&[0.0, 1000.0]);
        let scale_after = fq.qparams.as_ref().unwrap().scale[0];
        assert!(
            (scale_after - cached_scale).abs() < 1e-10,
            "scale should not change when observer is disabled"
        );
    }

    #[test]
    fn test_fake_quantize_disabled_is_identity() {
        let mut fq = FakeQuantize::new(QuantDtype::Int8);
        fq.fake_quant_enabled = false;
        let data = vec![1.234, 5.678, -9.012];
        let (output, mask) = fq.forward(&data);
        assert_eq!(output, data);
        assert!(mask.iter().all(|&m| m == 1.0));
    }

    // ----- QatModel -----

    #[test]
    fn test_qat_model_register_and_fq_weights() {
        let mut model = QatModel::new(QuantDtype::Int8);
        model.register_layer("fc1");

        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let (fq_weights, originals) = model.fake_quantize_weights("fc1", &weights).unwrap();

        // Originals should be exact copies.
        assert_eq!(originals, weights);
        // Fake-quantized weights should be close to originals.
        for (i, (&fq, &orig)) in fq_weights.iter().zip(weights.iter()).enumerate() {
            assert!((fq - orig).abs() < 0.1, "weight {i}: fq={fq}, orig={orig}");
        }
    }

    #[test]
    fn test_qat_model_activation_fq_per_layer() {
        let mut model = QatModel::new(QuantDtype::Int8);
        model.register_layer("layer1");
        model.register_layer("layer2");

        // Both layers should have independent activation FakeQuantize.
        let (act1, _) = model
            .fake_quantize_activations("layer1", &[1.0, 2.0])
            .unwrap();
        let (act2, _) = model
            .fake_quantize_activations("layer2", &[10.0, 20.0])
            .unwrap();
        assert_eq!(act1.len(), 2);
        assert_eq!(act2.len(), 2);
    }

    #[test]
    fn test_qat_model_unregistered_layer_errors() {
        let mut model = QatModel::new(QuantDtype::Int8);
        let result = model.fake_quantize_weights("nonexistent", &[1.0]);
        assert!(result.is_err());
    }

    // ----- prepare_qat -----

    #[test]
    fn test_prepare_qat_skips_bias() {
        let names = &["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"];
        let model = prepare_qat(names, QuantDtype::Int8);

        assert!(model.layers.contains_key("fc1"));
        assert!(model.layers.contains_key("fc2"));
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_prepare_qat_bias_only_still_registers() {
        let names = &["fc1.bias"];
        let model = prepare_qat(names, QuantDtype::Int8);
        // Even bias-only parameters should get a layer registered.
        assert!(model.layers.contains_key("fc1"));
    }

    // ----- cuda_rng -----
    //
    // The cuda_rng module exposes a process-global `Mutex<u64>` state plus
    // a fork/join stack. Both tests below mutate that state and read it
    // back; under cargo's default parallel test runner they race with each
    // other. Serialise via a local static mutex (same pattern as the
    // capture-lock used in #602 for the GPU-graph tests).
    fn cuda_rng_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn test_cuda_rng_fork_join() {
        let _g = cuda_rng_test_lock();
        let initial = cuda_rng::get_state();
        cuda_rng::fork_rng(0x12345678);
        assert_eq!(cuda_rng::get_state(), 0x12345678);
        cuda_rng::join_rng();
        assert_eq!(cuda_rng::get_state(), initial);
    }

    #[test]
    fn test_cuda_rng_next_seed() {
        let _g = cuda_rng_test_lock();
        let s1 = cuda_rng::next_seed();
        let s2 = cuda_rng::next_seed();
        assert_ne!(s1, s2, "consecutive seeds should differ");
    }
}
