//! Weight initialization functions.
//!
//! All functions operate on `Parameter<T>` in-place, matching PyTorch's
//! `nn.init` module. Each layer's constructor applies the appropriate
//! default initialization.

use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

use crate::parameter::Parameter;

/// Non-linearity type for computing the correct gain in Kaiming init.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonLinearity {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f64),
}

impl NonLinearity {
    /// Recommended gain for this non-linearity.
    pub fn gain(&self) -> f64 {
        match self {
            NonLinearity::Linear | NonLinearity::Sigmoid => 1.0,
            NonLinearity::Tanh => 5.0 / 3.0,
            NonLinearity::ReLU => (2.0f64).sqrt(),
            NonLinearity::LeakyReLU(neg_slope) => (2.0 / (1.0 + neg_slope * neg_slope)).sqrt(),
        }
    }
}

/// Compute fan_in and fan_out for a parameter tensor.
///
/// - 1D: fan_in = fan_out = shape[0]
/// - 2D: fan_in = shape[1], fan_out = shape[0]
/// - 3D+: fan_in = shape[1] * product(shape[2..]), fan_out = shape[0] * product(shape[2..])
fn compute_fans(shape: &[usize]) -> FerrotorchResult<(usize, usize)> {
    match shape.len() {
        0 => Err(FerrotorchError::InvalidArgument {
            message: "cannot compute fan for scalar tensor".into(),
        }),
        1 => Ok((shape[0], shape[0])),
        2 => Ok((shape[1], shape[0])),
        _ => {
            let receptive_field: usize = shape[2..].iter().product();
            Ok((shape[1] * receptive_field, shape[0] * receptive_field))
        }
    }
}

/// Fill parameter with a constant value.
pub fn constant<T: Float>(param: &mut Parameter<T>, value: T) -> FerrotorchResult<()> {
    let data = vec![value; param.numel()];
    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        param.shape().to_vec(),
        true,
    )?);
    Ok(())
}

/// Fill parameter with zeros.
pub fn zeros<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    constant(param, <T as num_traits::Zero>::zero())
}

/// Fill parameter with ones.
pub fn ones<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    constant(param, <T as num_traits::One>::one())
}

/// Fill parameter with values from U(low, high).
pub fn uniform<T: Float>(param: &mut Parameter<T>, low: f64, high: f64) -> FerrotorchResult<()> {
    let numel = param.numel();
    let data: Vec<T> = simple_uniform(numel, low, high);
    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        param.shape().to_vec(),
        true,
    )?);
    Ok(())
}

/// Fill parameter with values from N(mean, std).
pub fn normal<T: Float>(param: &mut Parameter<T>, mean: f64, std: f64) -> FerrotorchResult<()> {
    let numel = param.numel();
    let data: Vec<T> = simple_normal(numel, mean, std);
    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        param.shape().to_vec(),
        true,
    )?);
    Ok(())
}

/// Xavier uniform initialization (Glorot).
///
/// Fills with values from U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out)).
pub fn xavier_uniform<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    let (fan_in, fan_out) = compute_fans(param.shape())?;
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    uniform(param, -limit, limit)
}

/// Xavier normal initialization (Glorot).
///
/// Fills with values from N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
pub fn xavier_normal<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    let (fan_in, fan_out) = compute_fans(param.shape())?;
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    normal(param, 0.0, std)
}

/// Kaiming uniform initialization (He).
///
/// Fills with values from U(-limit, limit) where limit = gain * sqrt(3 / fan_in).
pub fn kaiming_uniform<T: Float>(
    param: &mut Parameter<T>,
    nonlinearity: NonLinearity,
) -> FerrotorchResult<()> {
    let (fan_in, _) = compute_fans(param.shape())?;
    let gain = nonlinearity.gain();
    let std = gain / (fan_in as f64).sqrt();
    let limit = (3.0f64).sqrt() * std;
    uniform(param, -limit, limit)
}

/// Kaiming normal initialization (He).
///
/// Fills with values from N(0, std) where std = gain / sqrt(fan_in).
pub fn kaiming_normal<T: Float>(
    param: &mut Parameter<T>,
    nonlinearity: NonLinearity,
) -> FerrotorchResult<()> {
    let (fan_in, _) = compute_fans(param.shape())?;
    let gain = nonlinearity.gain();
    let std = gain / (fan_in as f64).sqrt();
    normal(param, 0.0, std)
}

// CL-318: trunc_normal, orthogonal, sparse, dirac, eye init functions

/// Fill parameter with values from a truncated normal distribution.
///
/// Samples from N(mean, std) clipped to the interval `[a, b]`. Values
/// outside the bounds are resampled (rejection sampling). This is
/// the standard initialization for Vision Transformer position embeddings
/// and similar.
///
/// # Arguments
///
/// * `param` -- Parameter to initialize in-place.
/// * `mean` -- Mean of the normal distribution.
/// * `std` -- Standard deviation of the normal distribution.
/// * `a` -- Lower truncation bound.
/// * `b` -- Upper truncation bound.
pub fn trunc_normal_<T: Float>(
    param: &mut Parameter<T>,
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
) -> FerrotorchResult<()> {
    if a >= b {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("trunc_normal_: a ({a}) must be less than b ({b})"),
        });
    }
    if std <= 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("trunc_normal_: std ({std}) must be positive"),
        });
    }

    let numel = param.numel();
    // Over-sample to minimize rejection iterations. We need `numel` valid
    // samples; each draw has acceptance probability at least ~68% for a
    // +-1 sigma bound, so 2x is a safe initial batch.
    let mut data: Vec<T> = Vec::with_capacity(numel);
    let mut remaining = numel;
    while remaining > 0 {
        let batch_size = remaining * 2 + 64;
        let candidates: Vec<T> = simple_normal(batch_size, mean, std);
        for v in candidates {
            let f = v.to_f64().unwrap();
            if f >= a && f <= b {
                data.push(v);
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
    }
    data.truncate(numel);

    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        param.shape().to_vec(),
        true,
    )?);
    Ok(())
}

/// Fill parameter with an orthogonal matrix, scaled by `gain`.
///
/// For a 2D parameter `[rows, cols]`, generates a random matrix, computes
/// its QR decomposition via modified Gram-Schmidt, and uses
/// `Q * diag(sign(diag(R))) * gain`. For higher-dimensional parameters
/// the weight is reshaped to 2D first, initialized, then reshaped back.
///
/// Matches `torch.nn.init.orthogonal_`.
pub fn orthogonal_<T: Float>(param: &mut Parameter<T>, gain: f64) -> FerrotorchResult<()> {
    let shape = param.shape().to_vec();
    if shape.len() < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "orthogonal_ requires at least a 2D tensor".into(),
        });
    }

    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();
    let (n, m) = (rows, cols);

    // Generate random matrix N(0,1).
    let flat: Vec<f64> = simple_normal::<f64>(n * m, 0.0, 1.0);

    // We want to orthogonalize either rows or columns depending on shape.
    // PyTorch orthogonalizes along the larger dimension. If n < m, we
    // transpose, orthogonalize, transpose back.
    let transpose = n < m;

    let (rows_eff, cols_eff) = if transpose { (m, n) } else { (n, m) };
    let k_eff = rows_eff.min(cols_eff);

    // Build the working matrix (always rows_eff x cols_eff, row-major).
    let mut q: Vec<f64> = if transpose {
        // Transpose: from [n x m] row-major to [m x n] row-major.
        let mut t = vec![0.0; m * n];
        for i in 0..n {
            for j in 0..m {
                t[j * n + i] = flat[i * m + j];
            }
        }
        t
    } else {
        flat
    };
    let ce = cols_eff;

    // r_diag stores diagonal of R for sign correction.
    let mut r_diag = vec![0.0f64; k_eff];

    // Modified Gram-Schmidt on columns of q [rows_eff x cols_eff].
    for j in 0..k_eff {
        // Compute norm of column j.
        let mut norm: f64 = 0.0;
        for i in 0..rows_eff {
            let v = q[i * ce + j];
            norm += v * v;
        }
        norm = norm.sqrt();
        if norm < 1e-15 {
            // Degenerate column, leave as zero.
            r_diag[j] = 1.0;
            continue;
        }
        r_diag[j] = norm;

        // Normalize column j.
        for i in 0..rows_eff {
            q[i * ce + j] /= norm;
        }

        // Subtract projection from all subsequent columns.
        for jj in (j + 1)..cols_eff {
            let mut dot = 0.0;
            for i in 0..rows_eff {
                dot += q[i * ce + j] * q[i * ce + jj];
            }
            for i in 0..rows_eff {
                q[i * ce + jj] -= dot * q[i * ce + j];
            }
        }
    }

    // Sign correction: Q_corrected[:, j] = Q[:, j] * sign(R[j, j])
    for j in 0..k_eff {
        let sign = if r_diag[j] >= 0.0 { 1.0 } else { -1.0 };
        for i in 0..rows_eff {
            q[i * ce + j] *= sign * gain;
        }
    }

    // Extract the result: we want [n x m] output, but only the first
    // k_eff columns of q are orthonormalized (the rest may be garbage).
    let mut result = vec![T::from(0.0).unwrap(); n * m];
    if transpose {
        // q is [m x n], we need [n x m]. Transpose back, taking first
        // k_eff columns of q as the first k_eff rows of result.
        for i in 0..n.min(k_eff) {
            for j in 0..m {
                result[i * m + j] = T::from(q[j * ce + i]).unwrap();
            }
        }
        // If n > k_eff, remaining rows stay zero.
    } else {
        // q is [n x m], take first k_eff columns.
        for i in 0..n {
            for j in 0..m.min(k_eff) {
                result[i * m + j] = T::from(q[i * ce + j]).unwrap();
            }
        }
    }

    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(result),
        shape,
        true,
    )?);
    Ok(())
}

/// Sparse initialization: fill parameter as a sparse matrix.
///
/// For each column, randomly zeroes out a fraction of the rows equal to
/// `sparsity` (e.g. 0.9 means 90% of elements per column are zero).
/// Non-zero entries are drawn from N(0, `std`).
///
/// The parameter must be 2D.
pub fn sparse_<T: Float>(
    param: &mut Parameter<T>,
    sparsity: f64,
    std: f64,
) -> FerrotorchResult<()> {
    let shape = param.shape().to_vec();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "sparse_ requires a 2D tensor".into(),
        });
    }
    if !(0.0..1.0).contains(&sparsity) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("sparse_: sparsity ({sparsity}) must be in [0, 1)"),
        });
    }

    let (rows, cols) = (shape[0], shape[1]);
    let num_zeros_per_col = ((rows as f64) * sparsity).ceil() as usize;

    // Generate the dense normal values.
    let values: Vec<f64> = simple_normal::<f64>(rows * cols, 0.0, std);

    let mut data = vec![T::from(0.0).unwrap(); rows * cols];

    // For each column, pick which rows to zero out using a Fisher-Yates
    // partial shuffle to select `num_zeros_per_col` indices.
    let mut rng_state = xorshift_seed();

    for j in 0..cols {
        // Create index array for this column's rows.
        let mut indices: Vec<usize> = (0..rows).collect();

        // Partial Fisher-Yates to select `num_zeros_per_col` indices to zero.
        let num_to_pick = num_zeros_per_col.min(rows);
        for k in 0..num_to_pick {
            rng_state = xorshift_step(rng_state);
            let swap_idx = k + (rng_state as usize) % (rows - k);
            indices.swap(k, swap_idx);
        }

        // The first `num_to_pick` indices in `indices` are zeroed out.
        // Remaining indices keep their normal values.
        let zero_set: std::collections::HashSet<usize> =
            indices[..num_to_pick].iter().copied().collect();

        for i in 0..rows {
            if zero_set.contains(&i) {
                data[i * cols + j] = T::from(0.0).unwrap();
            } else {
                data[i * cols + j] = T::from(values[i * cols + j]).unwrap();
            }
        }
    }

    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        shape,
        true,
    )?);
    Ok(())
}

/// Dirac delta initialization for convolutional layers.
///
/// Fills the parameter with the Dirac delta function. For 3D weights
/// `[out_channels, in_channels/groups, kernel_size]` or 4D weights
/// `[out_channels, in_channels/groups, kH, kW]`, sets the center of each
/// filter to form an identity mapping (preserving input channels through
/// output channels).
///
/// `groups` must evenly divide `out_channels`.
pub fn dirac_<T: Float>(param: &mut Parameter<T>, groups: usize) -> FerrotorchResult<()> {
    let shape = param.shape().to_vec();
    if shape.len() < 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: "dirac_ requires at least a 3D tensor (out_ch, in_ch/groups, *kernel_size)"
                .into(),
        });
    }
    if groups == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "dirac_: groups must be > 0".into(),
        });
    }

    let out_channels = shape[0];
    let in_channels_per_group = shape[1];

    if out_channels % groups != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "dirac_: out_channels ({out_channels}) must be divisible by groups ({groups})"
            ),
        });
    }

    let min_dim = (out_channels / groups).min(in_channels_per_group);
    let kernel_size: usize = shape[2..].iter().product();
    let center = kernel_size / 2;
    let numel = param.numel();

    let mut data = vec![T::from(0.0).unwrap(); numel];
    let one = T::from(1.0).unwrap();

    // Stride computation for the flattened buffer.
    // shape = [out_ch, in_ch_per_group, *kernel_dims]
    // flat index = out * (in_ch_per_group * kernel_size) + in_ * kernel_size + k
    let in_stride = kernel_size;
    let out_stride = in_channels_per_group * kernel_size;

    for g in 0..groups {
        let out_offset = g * (out_channels / groups);
        for d in 0..min_dim {
            let out_idx = out_offset + d;
            let in_idx = d;
            data[out_idx * out_stride + in_idx * in_stride + center] = one;
        }
    }

    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        shape,
        true,
    )?);
    Ok(())
}

/// Fill parameter with an identity matrix.
///
/// For 2D parameters, fills with the identity matrix (1s on the diagonal,
/// 0s elsewhere). For non-square matrices, the identity is placed in the
/// top-left corner.
pub fn eye_<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    let shape = param.shape().to_vec();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "eye_ requires a 2D tensor".into(),
        });
    }

    let (rows, cols) = (shape[0], shape[1]);
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    let mut data = vec![zero; rows * cols];
    for i in 0..rows.min(cols) {
        data[i * cols + i] = one;
    }

    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        shape,
        true,
    )?);
    Ok(())
}

// --- Internal PRNG helpers ---

/// Seed a xorshift64 PRNG from system time + thread id.
fn xorshift_seed() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let state = hasher.finish();
    if state == 0 { 0xdeadbeefcafe } else { state }
}

/// Single step of xorshift64.
fn xorshift_step(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

pub(crate) fn simple_uniform<T: Float>(n: usize, low: f64, high: f64) -> Vec<T> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 0xdeadbeefcafe;
    }

    let range = high - low;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state as f64) / (u64::MAX as f64);
            T::from(low + u * range).unwrap()
        })
        .collect()
}

pub(crate) fn simple_normal<T: Float>(n: usize, mean: f64, std: f64) -> Vec<T> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 0xdeadbeefcafe;
    }

    let mut next_uniform = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64) / (u64::MAX as f64)).max(1e-300)
    };

    let mut data = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1 = next_uniform();
        let u2 = next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push(T::from(mean + std * r * theta.cos()).unwrap());
        if i + 1 < n {
            data.push(T::from(mean + std * r * theta.sin()).unwrap());
        }
        i += 2;
    }
    data.truncate(n);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_init() {
        let mut p = Parameter::<f32>::ones(&[3, 4]).unwrap();
        zeros(&mut p).unwrap();
        assert!(p.data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_init() {
        let mut p = Parameter::<f32>::zeros(&[2, 3]).unwrap();
        ones(&mut p).unwrap();
        assert!(p.data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_constant_init() {
        let mut p = Parameter::<f32>::zeros(&[5]).unwrap();
        constant(&mut p, 3.14).unwrap();
        assert!(p.data().unwrap().iter().all(|&x| (x - 3.14).abs() < 1e-5));
    }

    #[test]
    fn test_uniform_init_bounds() {
        let mut p = Parameter::<f32>::zeros(&[10000]).unwrap();
        uniform(&mut p, -1.0, 1.0).unwrap();
        let data = p.data().unwrap();
        assert!(data.iter().all(|&x| x >= -1.0 && x <= 1.0));
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_normal_init_stats() {
        let mut p = Parameter::<f32>::zeros(&[10000]).unwrap();
        normal(&mut p, 0.0, 1.0).unwrap();
        let data = p.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.1, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.2, "var = {var}");
    }

    #[test]
    fn test_xavier_uniform_stats() {
        let mut p = Parameter::<f32>::zeros(&[256, 128]).unwrap();
        xavier_uniform(&mut p).unwrap();
        let data = p.data().unwrap();
        let limit = (6.0 / (128.0 + 256.0) as f64).sqrt() as f32;
        assert!(data.iter().all(|&x| x.abs() <= limit + 0.01));
    }

    #[test]
    fn test_xavier_normal_stats() {
        let mut p = Parameter::<f32>::zeros(&[256, 128]).unwrap();
        xavier_normal(&mut p).unwrap();
        let data = p.data().unwrap();
        let expected_std = (2.0 / (128.0 + 256.0) as f64).sqrt() as f32;
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.05, "mean = {mean}");
        assert!(
            (var.sqrt() - expected_std).abs() < expected_std * 0.15,
            "std = {}, expected = {expected_std}",
            var.sqrt()
        );
    }

    #[test]
    fn test_kaiming_uniform_relu() {
        let mut p = Parameter::<f32>::zeros(&[64, 128]).unwrap();
        kaiming_uniform(&mut p, NonLinearity::ReLU).unwrap();
        let data = p.data().unwrap();
        let gain = (2.0f64).sqrt();
        let std = gain / (128.0f64).sqrt();
        let limit = (3.0f64).sqrt() * std;
        assert!(data.iter().all(|&x| (x as f64).abs() <= limit + 0.01));
    }

    #[test]
    fn test_kaiming_normal_relu() {
        let mut p = Parameter::<f32>::zeros(&[64, 128]).unwrap();
        kaiming_normal(&mut p, NonLinearity::ReLU).unwrap();
        let data = p.data().unwrap();
        let expected_std = (2.0f64).sqrt() / (128.0f64).sqrt();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.1, "mean = {mean}");
        assert!(
            ((var.sqrt() as f64) - expected_std).abs() < expected_std * 0.2,
            "std = {}, expected = {expected_std}",
            var.sqrt()
        );
    }

    #[test]
    fn test_compute_fans_2d() {
        let (fi, fo) = compute_fans(&[64, 128]).unwrap();
        assert_eq!(fi, 128);
        assert_eq!(fo, 64);
    }

    #[test]
    fn test_compute_fans_4d() {
        let (fi, fo) = compute_fans(&[32, 16, 3, 3]).unwrap();
        assert_eq!(fi, 16 * 9);
        assert_eq!(fo, 32 * 9);
    }

    #[test]
    fn test_nonlinearity_gain() {
        assert!((NonLinearity::ReLU.gain() - (2.0f64).sqrt()).abs() < 1e-10);
        assert!((NonLinearity::Linear.gain() - 1.0).abs() < 1e-10);
        assert!((NonLinearity::Tanh.gain() - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_init_preserves_requires_grad() {
        let mut p = Parameter::<f32>::zeros(&[5]).unwrap();
        xavier_uniform(&mut p).unwrap();
        assert!(p.requires_grad());
    }

    // --- trunc_normal_ tests ---

    #[test]
    fn test_trunc_normal_bounds() {
        let mut p = Parameter::<f32>::zeros(&[10000]).unwrap();
        trunc_normal_(&mut p, 0.0, 1.0, -2.0, 2.0).unwrap();
        let data = p.data().unwrap();
        assert!(
            data.iter().all(|&x| x >= -2.0 && x <= 2.0),
            "all values must be within [-2, 2]"
        );
    }

    #[test]
    fn test_trunc_normal_stats() {
        let mut p = Parameter::<f32>::zeros(&[50000]).unwrap();
        trunc_normal_(&mut p, 0.0, 1.0, -2.0, 2.0).unwrap();
        let data = p.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        // Truncated N(0,1) on [-2,2] has mean 0.
        assert!(mean.abs() < 0.05, "mean = {mean}");
    }

    #[test]
    fn test_trunc_normal_rejects_bad_bounds() {
        let mut p = Parameter::<f32>::zeros(&[10]).unwrap();
        assert!(trunc_normal_(&mut p, 0.0, 1.0, 2.0, -2.0).is_err());
    }

    #[test]
    fn test_trunc_normal_rejects_zero_std() {
        let mut p = Parameter::<f32>::zeros(&[10]).unwrap();
        assert!(trunc_normal_(&mut p, 0.0, 0.0, -1.0, 1.0).is_err());
    }

    // --- orthogonal_ tests ---

    #[test]
    fn test_orthogonal_columns_orthonormal() {
        // For a square matrix, Q^T Q should be close to I.
        let mut p = Parameter::<f64>::zeros(&[32, 32]).unwrap();
        orthogonal_(&mut p, 1.0).unwrap();
        let data = p.data().unwrap();
        let n = 32;

        // Check Q^T Q ~= I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += data[k * n + i] as f64 * data[k * n + j] as f64;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "Q^T Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_gain() {
        let mut p = Parameter::<f64>::zeros(&[16, 16]).unwrap();
        orthogonal_(&mut p, 2.0).unwrap();
        let data = p.data().unwrap();
        let n = 16;

        // With gain=2, Q^T Q should be 4*I.
        for i in 0..n {
            let mut col_norm_sq = 0.0;
            for k in 0..n {
                let v = data[k * n + i] as f64;
                col_norm_sq += v * v;
            }
            assert!(
                (col_norm_sq - 4.0).abs() < 1e-5,
                "column {i} norm^2 = {col_norm_sq}, expected 4.0"
            );
        }
    }

    #[test]
    fn test_orthogonal_tall_matrix() {
        // More rows than cols: [64, 16]. The 16 columns should be orthonormal.
        let mut p = Parameter::<f64>::zeros(&[64, 16]).unwrap();
        orthogonal_(&mut p, 1.0).unwrap();
        let data = p.data().unwrap();
        let (n, m) = (64, 16);

        for i in 0..m {
            for j in 0..m {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += data[k * m + i] as f64 * data[k * m + j] as f64;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "tall Q^T Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_wide_matrix() {
        // More cols than rows: [16, 64]. The 16 rows should be orthonormal.
        let mut p = Parameter::<f64>::zeros(&[16, 64]).unwrap();
        orthogonal_(&mut p, 1.0).unwrap();
        let data = p.data().unwrap();
        let (n, m) = (16, 64);

        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += data[i * m + k] as f64 * data[j * m + k] as f64;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "wide Q Q^T [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_rejects_1d() {
        let mut p = Parameter::<f32>::zeros(&[10]).unwrap();
        assert!(orthogonal_(&mut p, 1.0).is_err());
    }

    // --- sparse_ tests ---

    #[test]
    fn test_sparse_sparsity_ratio() {
        let mut p = Parameter::<f32>::zeros(&[100, 50]).unwrap();
        sparse_(&mut p, 0.9, 0.01).unwrap();
        let data = p.data().unwrap();
        let num_zeros = data.iter().filter(|&&x| x == 0.0).count();
        let total = data.len();
        let actual_sparsity = num_zeros as f64 / total as f64;
        // Should be approximately 0.9 (+/- small tolerance for rounding).
        assert!(
            (actual_sparsity - 0.9).abs() < 0.05,
            "sparsity = {actual_sparsity}, expected ~0.9"
        );
    }

    #[test]
    fn test_sparse_nonzero_drawn_from_normal() {
        let mut p = Parameter::<f32>::zeros(&[200, 100]).unwrap();
        sparse_(&mut p, 0.5, 1.0).unwrap();
        let data = p.data().unwrap();
        let nonzero: Vec<f64> = data
            .iter()
            .filter(|&&x| x != 0.0)
            .map(|&x| x as f64)
            .collect();
        assert!(!nonzero.is_empty());
        let mean: f64 = nonzero.iter().sum::<f64>() / nonzero.len() as f64;
        assert!(mean.abs() < 0.15, "nonzero mean = {mean}");
    }

    #[test]
    fn test_sparse_rejects_non_2d() {
        let mut p = Parameter::<f32>::zeros(&[10]).unwrap();
        assert!(sparse_(&mut p, 0.5, 1.0).is_err());
    }

    #[test]
    fn test_sparse_rejects_bad_sparsity() {
        let mut p = Parameter::<f32>::zeros(&[10, 10]).unwrap();
        assert!(sparse_(&mut p, 1.0, 1.0).is_err());
        assert!(sparse_(&mut p, -0.1, 1.0).is_err());
    }

    // --- dirac_ tests ---

    #[test]
    fn test_dirac_3d_identity() {
        // [4, 4, 3] conv1d: center element should be 1 on diagonal.
        let mut p = Parameter::<f32>::zeros(&[4, 4, 3]).unwrap();
        dirac_(&mut p, 1).unwrap();
        let data = p.data().unwrap();
        let center = 1; // kernel_size / 2

        for out_ch in 0..4 {
            for in_ch in 0..4 {
                let val = data[out_ch * 4 * 3 + in_ch * 3 + center];
                if out_ch == in_ch {
                    assert!(
                        (val - 1.0).abs() < 1e-6,
                        "diag [{out_ch},{in_ch}] = {val}"
                    );
                } else {
                    assert!(val.abs() < 1e-6, "off-diag [{out_ch},{in_ch}] = {val}");
                }
            }
        }
    }

    #[test]
    fn test_dirac_4d_identity() {
        // [2, 2, 3, 3] conv2d: center element should be 1 on diagonal.
        let mut p = Parameter::<f32>::zeros(&[2, 2, 3, 3]).unwrap();
        dirac_(&mut p, 1).unwrap();
        let data = p.data().unwrap();
        let kernel_size = 9;
        let center = 4; // 9 / 2

        for out_ch in 0..2 {
            for in_ch in 0..2 {
                let val = data[out_ch * 2 * 9 + in_ch * 9 + center];
                if out_ch == in_ch {
                    assert!((val - 1.0).abs() < 1e-6);
                } else {
                    assert!(val.abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_dirac_groups() {
        // [4, 2, 3] with groups=2: channels 0-1 map to inputs 0-1,
        // channels 2-3 map to inputs 0-1 in the second group.
        let mut p = Parameter::<f32>::zeros(&[4, 2, 3]).unwrap();
        dirac_(&mut p, 2).unwrap();
        let data = p.data().unwrap();
        let center = 1;

        // Group 0: out_ch 0,1 map to in_ch 0,1
        assert!((data[0 * 2 * 3 + 0 * 3 + center] - 1.0).abs() < 1e-6);
        assert!((data[1 * 2 * 3 + 1 * 3 + center] - 1.0).abs() < 1e-6);
        // Group 1: out_ch 2,3 map to in_ch 0,1
        assert!((data[2 * 2 * 3 + 0 * 3 + center] - 1.0).abs() < 1e-6);
        assert!((data[3 * 2 * 3 + 1 * 3 + center] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dirac_rejects_2d() {
        let mut p = Parameter::<f32>::zeros(&[4, 4]).unwrap();
        assert!(dirac_(&mut p, 1).is_err());
    }

    // --- eye_ tests ---

    #[test]
    fn test_eye_square() {
        let mut p = Parameter::<f32>::zeros(&[4, 4]).unwrap();
        eye_(&mut p).unwrap();
        let data = p.data().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (data[i * 4 + j] - expected).abs() < 1e-6,
                    "eye[{i},{j}] = {}",
                    data[i * 4 + j]
                );
            }
        }
    }

    #[test]
    fn test_eye_tall() {
        let mut p = Parameter::<f32>::zeros(&[6, 3]).unwrap();
        eye_(&mut p).unwrap();
        let data = p.data().unwrap();
        for i in 0..6 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((data[i * 3 + j] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_eye_wide() {
        let mut p = Parameter::<f32>::zeros(&[3, 6]).unwrap();
        eye_(&mut p).unwrap();
        let data = p.data().unwrap();
        for i in 0..3 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((data[i * 6 + j] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_eye_rejects_non_2d() {
        let mut p = Parameter::<f32>::zeros(&[4]).unwrap();
        assert!(eye_(&mut p).is_err());
    }

    #[test]
    fn test_eye_preserves_requires_grad() {
        let mut p = Parameter::<f32>::zeros(&[3, 3]).unwrap();
        eye_(&mut p).unwrap();
        assert!(p.requires_grad());
    }
}
