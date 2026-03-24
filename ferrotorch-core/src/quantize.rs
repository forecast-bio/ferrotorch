//! Post-training quantization (PTQ) for ferrotorch tensors.
//!
//! Provides symmetric and asymmetric quantization to INT8, INT4, and UINT8,
//! with per-tensor or per-channel granularity. Designed for inference-time
//! model compression — quantize once after training, then run forward passes
//! with reduced memory and (on supported hardware) faster matmul.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

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
                .map(|&v| {
                    quantize_val(v.to_f32().unwrap(), scale, zp, qmin, qmax, is_unsigned)
                })
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

// ---------------------------------------------------------------------------
// QParams — quantization parameters bundle
// ---------------------------------------------------------------------------

/// Quantization parameters: everything needed to convert between float and
/// integer domains.
#[derive(Debug, Clone)]
pub struct QParams {
    /// Scale factor: `real_value = (quantized_value - zero_point) * scale`.
    pub scale: f32,
    /// Zero point in quantized domain.
    pub zero_point: i32,
    /// Minimum representable quantized value.
    pub qmin: i32,
    /// Maximum representable quantized value.
    pub qmax: i32,
    /// Target quantized dtype.
    pub dtype: QuantDtype,
}

impl QParams {
    /// Create new quantization parameters from observed min/max range.
    pub fn from_min_max(min_val: f32, max_val: f32, dtype: QuantDtype) -> Self {
        let (scale, zero_point) = compute_scale_zp(min_val, max_val, dtype);
        Self {
            scale,
            zero_point,
            qmin: dtype.qmin(),
            qmax: dtype.qmax(),
            dtype,
        }
    }

    /// Create symmetric quantization parameters (zero_point = 0 for signed types).
    pub fn symmetric(max_abs: f32, dtype: QuantDtype) -> Self {
        let qmin = dtype.qmin();
        let qmax = dtype.qmax();
        let max_abs = max_abs.max(f32::EPSILON);
        let scale = max_abs / qmax as f32;
        Self {
            scale,
            zero_point: 0,
            qmin,
            qmax,
            dtype,
        }
    }
}

// ---------------------------------------------------------------------------
// Observer trait — tracks tensor statistics for quantization calibration
// ---------------------------------------------------------------------------

/// Observes tensor distributions to compute optimal quantization parameters.
///
/// Observers are the core calibration mechanism for both post-training
/// quantization (PTQ) and quantization-aware training (QAT). They track
/// running statistics across forward passes and derive `QParams` from them.
pub trait Observer: Send + Sync + std::fmt::Debug {
    /// Record a tensor's distribution statistics.
    fn observe(&mut self, tensor: &Tensor<f32>);

    /// Compute quantization parameters from accumulated statistics.
    fn calculate_qparams(&self) -> QParams;

    /// Reset all accumulated statistics.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// MinMaxObserver — global min/max tracking
// ---------------------------------------------------------------------------

/// Tracks the global min and max of all observed tensors.
///
/// Supports both symmetric (scale around zero, zero_point = 0 for signed)
/// and affine (asymmetric, arbitrary zero_point) quantization.
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    min_val: f32,
    max_val: f32,
    dtype: QuantDtype,
    symmetric: bool,
    has_data: bool,
}

impl MinMaxObserver {
    /// Create a new min/max observer.
    pub fn new(dtype: QuantDtype, symmetric: bool) -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            dtype,
            symmetric,
            has_data: false,
        }
    }
}

impl Observer for MinMaxObserver {
    fn observe(&mut self, tensor: &Tensor<f32>) {
        if let Ok(data) = tensor.data() {
            for &v in data {
                if v < self.min_val {
                    self.min_val = v;
                }
                if v > self.max_val {
                    self.max_val = v;
                }
            }
            self.has_data = true;
        }
    }

    fn calculate_qparams(&self) -> QParams {
        if !self.has_data {
            return QParams::from_min_max(0.0, 0.0, self.dtype);
        }
        if self.symmetric {
            let max_abs = self.min_val.abs().max(self.max_val.abs());
            QParams::symmetric(max_abs, self.dtype)
        } else {
            QParams::from_min_max(self.min_val, self.max_val, self.dtype)
        }
    }

    fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
        self.has_data = false;
    }
}

// ---------------------------------------------------------------------------
// PerChannelMinMaxObserver — per-output-channel quantization
// ---------------------------------------------------------------------------

/// Tracks min/max per output channel (axis 0 by default).
///
/// Used for weight quantization where each output channel may have a
/// significantly different range. The resulting `QParams` uses the global
/// range across all channels for simplicity in the STE backward path;
/// the per-channel scales are available via `channel_qparams()`.
#[derive(Debug, Clone)]
pub struct PerChannelMinMaxObserver {
    channel_mins: Vec<f32>,
    channel_maxs: Vec<f32>,
    num_channels: usize,
    dtype: QuantDtype,
    symmetric: bool,
    has_data: bool,
}

impl PerChannelMinMaxObserver {
    /// Create a per-channel observer for the given number of output channels.
    pub fn new(num_channels: usize, dtype: QuantDtype, symmetric: bool) -> Self {
        Self {
            channel_mins: vec![f32::INFINITY; num_channels],
            channel_maxs: vec![f32::NEG_INFINITY; num_channels],
            num_channels,
            dtype,
            symmetric,
            has_data: false,
        }
    }

    /// Per-channel quantization parameters.
    pub fn channel_qparams(&self) -> Vec<QParams> {
        (0..self.num_channels)
            .map(|ch| {
                if !self.has_data {
                    return QParams::from_min_max(0.0, 0.0, self.dtype);
                }
                if self.symmetric {
                    let max_abs = self.channel_mins[ch]
                        .abs()
                        .max(self.channel_maxs[ch].abs());
                    QParams::symmetric(max_abs, self.dtype)
                } else {
                    QParams::from_min_max(self.channel_mins[ch], self.channel_maxs[ch], self.dtype)
                }
            })
            .collect()
    }
}

impl Observer for PerChannelMinMaxObserver {
    fn observe(&mut self, tensor: &Tensor<f32>) {
        if let Ok(data) = tensor.data() {
            let shape = tensor.shape();
            if shape.is_empty() {
                return;
            }
            // Axis 0 is the channel dimension.
            let actual_channels = shape[0];
            if actual_channels != self.num_channels {
                return;
            }
            let elements_per_channel: usize = shape[1..].iter().product();
            for ch in 0..self.num_channels {
                let start = ch * elements_per_channel;
                let end = start + elements_per_channel;
                for &v in &data[start..end] {
                    if v < self.channel_mins[ch] {
                        self.channel_mins[ch] = v;
                    }
                    if v > self.channel_maxs[ch] {
                        self.channel_maxs[ch] = v;
                    }
                }
            }
            self.has_data = true;
        }
    }

    fn calculate_qparams(&self) -> QParams {
        // For FakeQuantize, return the global range across all channels.
        // Per-channel details are available via channel_qparams().
        if !self.has_data {
            return QParams::from_min_max(0.0, 0.0, self.dtype);
        }
        let global_min = self
            .channel_mins
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let global_max = self
            .channel_maxs
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if self.symmetric {
            QParams::symmetric(global_min.abs().max(global_max.abs()), self.dtype)
        } else {
            QParams::from_min_max(global_min, global_max, self.dtype)
        }
    }

    fn reset(&mut self) {
        self.channel_mins = vec![f32::INFINITY; self.num_channels];
        self.channel_maxs = vec![f32::NEG_INFINITY; self.num_channels];
        self.has_data = false;
    }
}

// ---------------------------------------------------------------------------
// HistogramObserver — entropy/percentile range finding
// ---------------------------------------------------------------------------

/// Builds a histogram of observed values and uses either entropy minimization
/// or percentile clipping to find the optimal quantization range.
///
/// This produces tighter ranges than min/max when distributions have long
/// tails (common in activations after ReLU-variants).
#[derive(Debug, Clone)]
pub struct HistogramObserver {
    /// Bin edges: `num_bins + 1` values.
    bins: Vec<f32>,
    /// Counts for each bin.
    counts: Vec<u64>,
    /// Number of histogram bins.
    num_bins: usize,
    /// Current tracked min.
    min_val: f32,
    /// Current tracked max.
    max_val: f32,
    /// Target quantized dtype.
    dtype: QuantDtype,
    /// Percentile for range clipping (e.g., 99.99).
    percentile: f32,
    /// Whether any data has been observed.
    has_data: bool,
}

impl HistogramObserver {
    /// Create a histogram observer.
    ///
    /// - `num_bins`: number of histogram bins (default: 2048).
    /// - `percentile`: the percentile of the distribution to use as the range
    ///   boundary (e.g., 99.99 clips the top/bottom 0.01%).
    pub fn new(num_bins: usize, percentile: f32, dtype: QuantDtype) -> Self {
        Self {
            bins: Vec::new(),
            counts: vec![0; num_bins],
            num_bins,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            dtype,
            percentile: percentile.clamp(90.0, 100.0),
            has_data: false,
        }
    }

    /// Default histogram observer: 2048 bins, 99.99th percentile.
    pub fn default_with_dtype(dtype: QuantDtype) -> Self {
        Self::new(2048, 99.99, dtype)
    }

    /// Rebuild bin edges from the current min/max.
    fn rebuild_bins(&mut self) {
        let range = (self.max_val - self.min_val).max(f32::EPSILON);
        let step = range / self.num_bins as f32;
        self.bins = (0..=self.num_bins)
            .map(|i| self.min_val + step * i as f32)
            .collect();
    }

    /// Find which bin a value falls into.
    #[inline]
    fn bin_index(&self, val: f32) -> usize {
        if self.bins.len() < 2 {
            return 0;
        }
        let range = self.bins[self.bins.len() - 1] - self.bins[0];
        if range <= 0.0 {
            return 0;
        }
        let normalized = (val - self.bins[0]) / range;
        let idx = (normalized * self.num_bins as f32) as usize;
        idx.min(self.num_bins - 1)
    }

    /// Compute the range from the percentile.
    fn percentile_range(&self) -> (f32, f32) {
        let total: u64 = self.counts.iter().sum();
        if total == 0 {
            return (0.0, 0.0);
        }

        let low_target = (total as f64 * (1.0 - self.percentile as f64 / 100.0)) as u64;
        let high_target = (total as f64 * (self.percentile as f64 / 100.0)) as u64;

        let mut cumsum: u64 = 0;
        let mut low_bin = 0;
        for (i, &c) in self.counts.iter().enumerate() {
            cumsum += c;
            if cumsum > low_target {
                low_bin = i;
                break;
            }
        }

        cumsum = 0;
        let mut high_bin = self.num_bins - 1;
        for (i, &c) in self.counts.iter().enumerate() {
            cumsum += c;
            if cumsum >= high_target {
                high_bin = i;
                break;
            }
        }

        let step = if self.bins.len() > 1 {
            (self.bins[self.bins.len() - 1] - self.bins[0]) / self.num_bins as f32
        } else {
            0.0
        };

        let low_val = self.min_val + low_bin as f32 * step;
        let high_val = self.min_val + (high_bin + 1) as f32 * step;
        (low_val, high_val)
    }
}

impl Observer for HistogramObserver {
    fn observe(&mut self, tensor: &Tensor<f32>) {
        if let Ok(data) = tensor.data() {
            // Update global min/max.
            let mut new_min = self.min_val;
            let mut new_max = self.max_val;
            for &v in data {
                if v < new_min {
                    new_min = v;
                }
                if v > new_max {
                    new_max = v;
                }
            }

            let range_changed = new_min < self.min_val || new_max > self.max_val;
            self.min_val = new_min;
            self.max_val = new_max;

            if range_changed || !self.has_data {
                // Rebuild bins with the new range and re-count everything.
                // For online use, this is a simplification — a production
                // implementation would merge histograms.
                self.counts = vec![0; self.num_bins];
                self.rebuild_bins();
            }

            // Insert data into bins.
            for &v in data {
                let idx = self.bin_index(v);
                self.counts[idx] += 1;
            }
            self.has_data = true;
        }
    }

    fn calculate_qparams(&self) -> QParams {
        if !self.has_data {
            return QParams::from_min_max(0.0, 0.0, self.dtype);
        }
        let (pmin, pmax) = self.percentile_range();
        QParams::from_min_max(pmin, pmax, self.dtype)
    }

    fn reset(&mut self) {
        self.counts = vec![0; self.num_bins];
        self.bins.clear();
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
        self.has_data = false;
    }
}

// ---------------------------------------------------------------------------
// MovingAverageMinMaxObserver — EMA for online calibration
// ---------------------------------------------------------------------------

/// Tracks exponential moving averages of min and max values, suitable for
/// online calibration during QAT training where the distribution shifts
/// gradually across training steps.
#[derive(Debug, Clone)]
pub struct MovingAverageMinMaxObserver {
    min_val: f32,
    max_val: f32,
    /// Smoothing factor for EMA: `new = averaging_constant * batch + (1 - averaging_constant) * old`.
    averaging_constant: f32,
    dtype: QuantDtype,
    symmetric: bool,
    has_data: bool,
}

impl MovingAverageMinMaxObserver {
    /// Create a moving-average observer.
    ///
    /// - `averaging_constant`: EMA weight for the new observation (typically 0.01).
    pub fn new(averaging_constant: f32, dtype: QuantDtype, symmetric: bool) -> Self {
        Self {
            min_val: 0.0,
            max_val: 0.0,
            averaging_constant: averaging_constant.clamp(0.0, 1.0),
            dtype,
            symmetric,
            has_data: false,
        }
    }
}

impl Observer for MovingAverageMinMaxObserver {
    fn observe(&mut self, tensor: &Tensor<f32>) {
        if let Ok(data) = tensor.data() {
            let mut batch_min = f32::INFINITY;
            let mut batch_max = f32::NEG_INFINITY;
            for &v in data {
                if v < batch_min {
                    batch_min = v;
                }
                if v > batch_max {
                    batch_max = v;
                }
            }

            if !self.has_data {
                self.min_val = batch_min;
                self.max_val = batch_max;
                self.has_data = true;
            } else {
                let c = self.averaging_constant;
                self.min_val = c * batch_min + (1.0 - c) * self.min_val;
                self.max_val = c * batch_max + (1.0 - c) * self.max_val;
            }
        }
    }

    fn calculate_qparams(&self) -> QParams {
        if !self.has_data {
            return QParams::from_min_max(0.0, 0.0, self.dtype);
        }
        if self.symmetric {
            let max_abs = self.min_val.abs().max(self.max_val.abs());
            QParams::symmetric(max_abs, self.dtype)
        } else {
            QParams::from_min_max(self.min_val, self.max_val, self.dtype)
        }
    }

    fn reset(&mut self) {
        self.min_val = 0.0;
        self.max_val = 0.0;
        self.has_data = false;
    }
}

// ---------------------------------------------------------------------------
// FakeQuantize — simulated quantization for training (STE)
// ---------------------------------------------------------------------------

/// Straight-Through Estimator backward: gradient passes through unchanged.
///
/// During the forward pass of QAT, `FakeQuantize` quantizes and immediately
/// dequantizes the tensor (`clamp(round(x/s) + zp, qmin, qmax) * s - zp*s`).
/// The backward pass uses the STE: the gradient of the quantize+dequantize
/// operation is treated as the identity, so upstream gradients pass through
/// to downstream without modification.
#[derive(Debug)]
struct FakeQuantizeBackward {
    input: Tensor<f32>,
}

impl GradFn<f32> for FakeQuantizeBackward {
    fn backward(&self, grad_output: &Tensor<f32>) -> FerrotorchResult<Vec<Option<Tensor<f32>>>> {
        // STE: gradient passes through unchanged.
        if self.input.requires_grad() {
            Ok(vec![Some(grad_output.clone())])
        } else {
            Ok(vec![None])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<f32>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FakeQuantizeBackward"
    }
}

/// Simulated quantization module for quantization-aware training (QAT).
///
/// In training mode, `FakeQuantize` applies quantize-then-dequantize in the
/// forward pass to simulate quantization error, while using the Straight-
/// Through Estimator (STE) for the backward pass so gradients flow through
/// unmodified.
///
/// In eval mode (or when disabled), it either applies the frozen
/// quantize+dequantize with the last computed parameters, or passes the
/// input through unchanged.
#[derive(Debug)]
pub struct FakeQuantize {
    /// Observer that tracks tensor statistics to compute scale/zero_point.
    observer: Box<dyn Observer>,
    /// Whether fake-quantization is active.
    enabled: bool,
    /// Cached quantization parameters from the observer.
    qparams: Option<QParams>,
    /// Whether to update the observer on each forward call.
    observer_enabled: bool,
}

impl FakeQuantize {
    /// Create a new `FakeQuantize` with the given observer.
    pub fn new(observer: Box<dyn Observer>) -> Self {
        Self {
            observer,
            enabled: true,
            qparams: None,
            observer_enabled: true,
        }
    }

    /// Create a `FakeQuantize` with a default `MinMaxObserver` for symmetric INT8.
    pub fn default_symmetric_int8() -> Self {
        Self::new(Box::new(MinMaxObserver::new(QuantDtype::Int8, true)))
    }

    /// Create a `FakeQuantize` with a default `MinMaxObserver` for affine INT8.
    pub fn default_affine_int8() -> Self {
        Self::new(Box::new(MinMaxObserver::new(QuantDtype::Int8, false)))
    }

    /// Enable fake-quantization.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable fake-quantization (passthrough).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Whether fake-quantization is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable observer updates on forward.
    pub fn enable_observer(&mut self) {
        self.observer_enabled = true;
    }

    /// Disable observer updates (freeze calibration).
    pub fn disable_observer(&mut self) {
        self.observer_enabled = false;
    }

    /// Get the current quantization parameters (if computed).
    pub fn qparams(&self) -> Option<&QParams> {
        self.qparams.as_ref()
    }

    /// Force-set quantization parameters (for testing or manual calibration).
    pub fn set_qparams(&mut self, qparams: QParams) {
        self.qparams = Some(qparams);
    }

    /// Forward pass: apply fake quantization with STE backward.
    ///
    /// 1. Observe the tensor (update running statistics).
    /// 2. Compute quantization parameters from the observer.
    /// 3. Quantize then dequantize: `clamp(round(x/s) + zp, qmin, qmax) * s - zp*s`.
    /// 4. Attach STE backward for autograd.
    pub fn forward(&mut self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        if !self.enabled {
            return Ok(input.clone());
        }

        // Step 1: observe.
        if self.observer_enabled {
            self.observer.observe(input);
        }

        // Step 2: compute qparams.
        let qp = self.observer.calculate_qparams();
        self.qparams = Some(qp.clone());

        // Step 3: fake-quantize (quantize + dequantize in float domain).
        let scale = qp.scale;
        let zp = qp.zero_point;
        let qmin = qp.qmin;
        let qmax = qp.qmax;

        let inv_scale = 1.0_f32 / scale;
        let zp_f = zp as f32;
        let qmin_f = qmin as f32;
        let qmax_f = qmax as f32;

        let data = input.data()?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| {
                // Quantize: q = clamp(round(x / scale) + zero_point, qmin, qmax)
                let q = (x * inv_scale + zp_f).round().clamp(qmin_f, qmax_f);
                // Dequantize: x' = (q - zero_point) * scale
                (q - zp_f) * scale
            })
            .collect();

        let output = Tensor::from_storage(
            TensorStorage::cpu(result),
            input.shape().to_vec(),
            false,
        )?;

        // Step 4: attach STE backward if needed.
        let needs_grad = crate::autograd::no_grad::is_grad_enabled() && input.requires_grad();
        if needs_grad {
            let (storage, shape) = output.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(FakeQuantizeBackward {
                    input: input.clone(),
                }),
            )
        } else {
            Ok(output)
        }
    }
}

/// Quantize a tensor per-element using explicit parameters, returning a
/// `QuantizedTensor` (integer storage).
///
/// This is the "hard" quantization used after QAT training is complete.
pub fn quantize_per_tensor(
    tensor: &Tensor<f32>,
    scale: f32,
    zero_point: i32,
    dtype: QuantDtype,
) -> FerrotorchResult<QuantizedTensor> {
    let data = tensor.data()?;
    let shape = tensor.shape().to_vec();
    let qmin = dtype.qmin();
    let qmax = dtype.qmax();
    let is_unsigned = dtype == QuantDtype::Uint8;

    let qdata: Vec<i8> = data
        .iter()
        .map(|&v| quantize_val(v, scale, zero_point, qmin, qmax, is_unsigned))
        .collect();

    Ok(QuantizedTensor {
        data: qdata,
        scale: vec![scale],
        zero_point: vec![zero_point],
        shape,
        scheme: QuantScheme::PerTensor,
        dtype,
    })
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
            assert!(
                err < 3.0,
                "element {i}: expected={e}, got={g}, error={err}"
            );
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

        let qmap =
            quantize_named_tensors(named, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();

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
    fn test_qparams_from_min_max() {
        let qp = QParams::from_min_max(-1.0, 1.0, QuantDtype::Int8);
        assert!(qp.scale > 0.0);
        assert_eq!(qp.qmin, -128);
        assert_eq!(qp.qmax, 127);
    }

    #[test]
    fn test_qparams_symmetric() {
        let qp = QParams::symmetric(5.0, QuantDtype::Int8);
        assert_eq!(qp.zero_point, 0);
        assert!((qp.scale - 5.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_qparams_symmetric_uint8() {
        // Symmetric with Uint8 still uses the max range.
        let qp = QParams::symmetric(2.0, QuantDtype::Uint8);
        assert!(qp.scale > 0.0);
    }

    // ----- MinMaxObserver -----

    #[test]
    fn test_min_max_observer_affine() {
        let mut obs = MinMaxObserver::new(QuantDtype::Int8, false);
        let t1 = make_tensor(&[1.0, 3.0, 5.0], &[3]);
        let t2 = make_tensor(&[-2.0, 0.0, 4.0], &[3]);
        obs.observe(&t1);
        obs.observe(&t2);

        let qp = obs.calculate_qparams();
        // Range should be [-2, 5].
        assert!(qp.scale > 0.0);
        assert_eq!(qp.dtype, QuantDtype::Int8);
    }

    #[test]
    fn test_min_max_observer_symmetric() {
        let mut obs = MinMaxObserver::new(QuantDtype::Int8, true);
        let t = make_tensor(&[-3.0, 0.0, 2.0], &[3]);
        obs.observe(&t);

        let qp = obs.calculate_qparams();
        assert_eq!(qp.zero_point, 0);
        // max_abs = 3.0.
        assert!((qp.scale - 3.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_max_observer_reset() {
        let mut obs = MinMaxObserver::new(QuantDtype::Int8, false);
        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        obs.observe(&t);
        obs.reset();

        // After reset, no data — should produce a safe default.
        let qp = obs.calculate_qparams();
        assert!(qp.scale > 0.0);
    }

    // ----- PerChannelMinMaxObserver -----

    #[test]
    fn test_per_channel_observer() {
        let mut obs = PerChannelMinMaxObserver::new(2, QuantDtype::Int8, false);
        // Shape [2, 3]: 2 channels, 3 elements each.
        let t = make_tensor(&[1.0, 2.0, 3.0, -5.0, 0.0, 10.0], &[2, 3]);
        obs.observe(&t);

        let channel_qps = obs.channel_qparams();
        assert_eq!(channel_qps.len(), 2);
        // Channel 0: range [0, 3] (includes zero).
        // Channel 1: range [-5, 10].
        assert!(channel_qps[1].scale > channel_qps[0].scale);
    }

    #[test]
    fn test_per_channel_observer_global_qparams() {
        let mut obs = PerChannelMinMaxObserver::new(2, QuantDtype::Int8, true);
        let t = make_tensor(&[1.0, 2.0, 3.0, -5.0, 0.0, 10.0], &[2, 3]);
        obs.observe(&t);

        let qp = obs.calculate_qparams();
        // Global max_abs = 10.0, symmetric.
        assert_eq!(qp.zero_point, 0);
        assert!((qp.scale - 10.0 / 127.0).abs() < 1e-5);
    }

    // ----- HistogramObserver -----

    #[test]
    fn test_histogram_observer_basic() {
        let mut obs = HistogramObserver::new(256, 99.99, QuantDtype::Int8);
        let data: Vec<f32> = (-100..=100).map(|x| x as f32 * 0.1).collect();
        let t = make_tensor(&data, &[data.len()]);
        obs.observe(&t);

        let qp = obs.calculate_qparams();
        assert!(qp.scale > 0.0);
        // The range should be approximately [-10, 10] at 99.99 percentile.
    }

    #[test]
    fn test_histogram_observer_with_outliers() {
        let mut obs = HistogramObserver::new(256, 99.0, QuantDtype::Int8);
        let mut data: Vec<f32> = (0..1000).map(|x| x as f32 * 0.01).collect();
        // Add outliers.
        data.push(1000.0);
        data.push(-1000.0);
        let t = make_tensor(&data, &[data.len()]);
        obs.observe(&t);

        let qp = obs.calculate_qparams();
        // At 99th percentile, the range should exclude the extreme outliers.
        assert!(qp.scale > 0.0);
    }

    #[test]
    fn test_histogram_observer_reset() {
        let mut obs = HistogramObserver::default_with_dtype(QuantDtype::Int8);
        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        obs.observe(&t);
        obs.reset();
        let qp = obs.calculate_qparams();
        assert!(qp.scale > 0.0);
    }

    // ----- MovingAverageMinMaxObserver -----

    #[test]
    fn test_moving_average_observer() {
        let mut obs = MovingAverageMinMaxObserver::new(0.1, QuantDtype::Int8, false);

        // First observation sets initial min/max.
        let t1 = make_tensor(&[-1.0, 0.0, 1.0], &[3]);
        obs.observe(&t1);
        let qp1 = obs.calculate_qparams();

        // Second observation with larger range: EMA shifts slowly.
        let t2 = make_tensor(&[-10.0, 0.0, 10.0], &[3]);
        obs.observe(&t2);
        let qp2 = obs.calculate_qparams();

        // The scale should increase but not jump all the way to the new range.
        assert!(qp2.scale > qp1.scale);
    }

    #[test]
    fn test_moving_average_observer_symmetric() {
        let mut obs = MovingAverageMinMaxObserver::new(0.5, QuantDtype::Int8, true);
        let t = make_tensor(&[-2.0, 0.0, 3.0], &[3]);
        obs.observe(&t);
        let qp = obs.calculate_qparams();
        assert_eq!(qp.zero_point, 0);
    }

    #[test]
    fn test_moving_average_observer_reset() {
        let mut obs = MovingAverageMinMaxObserver::new(0.1, QuantDtype::Int8, false);
        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        obs.observe(&t);
        obs.reset();
        assert!(!obs.has_data);
    }

    // ----- FakeQuantize -----

    #[test]
    fn test_fake_quantize_basic() {
        let mut fq = FakeQuantize::default_symmetric_int8();
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let out = fq.forward(&t).unwrap();

        assert_eq!(out.shape(), &[4]);
        // Output should be close to input (quantization error within one step).
        let orig = t.data().unwrap();
        let faked = out.data().unwrap();
        for (i, (&o, &f)) in orig.iter().zip(faked.iter()).enumerate() {
            let err = (o - f).abs();
            assert!(
                err < 0.1,
                "element {i}: original={o}, fake_quantized={f}, error={err}"
            );
        }
    }

    #[test]
    fn test_fake_quantize_disabled() {
        let mut fq = FakeQuantize::default_symmetric_int8();
        fq.disable();
        let t = make_tensor(&[1.5, 2.7, 3.3], &[3]);
        let out = fq.forward(&t).unwrap();
        // When disabled, output should be identical to input.
        let orig = t.data().unwrap();
        let faked = out.data().unwrap();
        for (&o, &f) in orig.iter().zip(faked.iter()) {
            assert_eq!(o, f);
        }
    }

    #[test]
    fn test_fake_quantize_ste_gradient() {
        // FakeQuantize with STE: gradient should pass through unchanged.
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![4],
            true,
        )
        .unwrap();

        let mut fq = FakeQuantize::default_symmetric_int8();
        let output = fq.forward(&input).unwrap();

        // Sum and backward.
        let loss = crate::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // STE: grad_input = grad_output = ones.
        let grad = input.grad().unwrap().expect("input should have grad");
        assert_eq!(grad.shape(), &[4]);
        let grad_data = grad.data().unwrap();
        for &g in grad_data {
            assert!(
                (g - 1.0).abs() < 1e-6,
                "STE gradient should be 1.0, got {g}"
            );
        }
    }

    #[test]
    fn test_fake_quantize_enable_disable() {
        let mut fq = FakeQuantize::default_affine_int8();
        assert!(fq.is_enabled());
        fq.disable();
        assert!(!fq.is_enabled());
        fq.enable();
        assert!(fq.is_enabled());
    }

    #[test]
    fn test_fake_quantize_qparams_computed() {
        let mut fq = FakeQuantize::default_symmetric_int8();
        assert!(fq.qparams().is_none());

        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        fq.forward(&t).unwrap();

        assert!(fq.qparams().is_some());
        let qp = fq.qparams().unwrap();
        assert!(qp.scale > 0.0);
    }

    #[test]
    fn test_fake_quantize_observer_disable() {
        let mut fq = FakeQuantize::default_symmetric_int8();

        // First pass to calibrate.
        let t1 = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        fq.forward(&t1).unwrap();
        let qp1_scale = fq.qparams().unwrap().scale;

        // Freeze observer.
        fq.disable_observer();

        // New data with much larger range should NOT change qparams.
        let t2 = make_tensor(&[100.0, 200.0, 300.0], &[3]);
        fq.forward(&t2).unwrap();
        let qp2_scale = fq.qparams().unwrap().scale;

        assert!((qp1_scale - qp2_scale).abs() < 1e-7);
    }

    // ----- quantize_per_tensor -----

    #[test]
    fn test_quantize_per_tensor_explicit() {
        let t = make_tensor(&[0.0, 1.0, 2.0, 3.0], &[4]);
        let qt = quantize_per_tensor(&t, 3.0 / 127.0, 0, QuantDtype::Int8).unwrap();
        assert_eq!(qt.shape(), &[4]);
        assert_eq!(qt.scale().len(), 1);
        assert_eq!(qt.scheme(), QuantScheme::PerTensor);

        // Dequantize and check round-trip.
        let rt: Tensor<f32> = dequantize(&qt).unwrap();
        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            assert!(
                err < 0.05,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_quantize_per_tensor_uint8() {
        let t = make_tensor(&[0.0, 0.5, 1.0, 1.5, 2.0], &[5]);
        let qt = quantize_per_tensor(&t, 2.0 / 255.0, 0, QuantDtype::Uint8).unwrap();
        assert_eq!(qt.qdtype(), QuantDtype::Uint8);

        let rt: Tensor<f32> = dequantize(&qt).unwrap();
        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            assert!(
                err < 0.02,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_fake_quantize_set_qparams() {
        let mut fq = FakeQuantize::default_symmetric_int8();
        let qp = QParams {
            scale: 0.1,
            zero_point: 0,
            qmin: -128,
            qmax: 127,
            dtype: QuantDtype::Int8,
        };
        fq.set_qparams(qp);
        assert!(fq.qparams().is_some());
        assert!((fq.qparams().unwrap().scale - 0.1).abs() < 1e-7);
    }

    #[test]
    fn test_fake_quantize_roundtrip_accuracy() {
        // FakeQuantize should introduce quantization noise but stay within
        // one quantization step of the original.
        let mut fq = FakeQuantize::new(Box::new(MinMaxObserver::new(QuantDtype::Int8, false)));
        let data: Vec<f32> = (-50..=50).map(|x| x as f32 * 0.1).collect();
        let t = make_tensor(&data, &[data.len()]);
        let out = fq.forward(&t).unwrap();

        let qp = fq.qparams().unwrap();
        let step = qp.scale;

        let orig = t.data().unwrap();
        let faked = out.data().unwrap();
        for (i, (&o, &f)) in orig.iter().zip(faked.iter()).enumerate() {
            let err = (o - f).abs();
            // Error should be at most half a quantization step.
            assert!(
                err <= step * 0.5 + 1e-5,
                "element {i}: original={o}, fake_quantized={f}, error={err}, step={step}"
            );
        }
    }
}
