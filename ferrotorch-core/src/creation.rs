use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Create a tensor filled with zeros.
pub fn zeros<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let data = vec![<T as num_traits::Zero>::zero(); numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor filled with ones.
pub fn ones<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let data = vec![<T as num_traits::One>::one(); numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor filled with a given value.
pub fn full<T: Float>(shape: &[usize], value: T) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let data = vec![value; numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor from a slice, copying the data.
pub fn from_slice<T: Float>(data: &[T], shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
}

/// Create a tensor from a `Vec<T>`, taking ownership.
pub fn from_vec<T: Float>(data: Vec<T>, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a 1-D tensor from a slice (shape inferred).
pub fn tensor<T: Float>(data: &[T]) -> FerrotorchResult<Tensor<T>> {
    let shape = vec![data.len()];
    from_slice(data, &shape)
}

/// Create a scalar (0-D) tensor.
pub fn scalar<T: Float>(value: T) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(TensorStorage::cpu(vec![value]), vec![], false)
}

/// Create an identity matrix of size `n x n`.
pub fn eye<T: Float>(n: usize) -> FerrotorchResult<Tensor<T>> {
    let mut data = vec![<T as num_traits::Zero>::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = <T as num_traits::One>::one();
    }
    Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
}

/// Create a 1-D tensor with values from `start` to `end` (exclusive) with step `step`.
pub fn arange<T: Float>(start: T, end: T, step: T) -> FerrotorchResult<Tensor<T>> {
    let mut data = Vec::new();
    let mut val = start;
    if step > <T as num_traits::Zero>::zero() {
        while val < end {
            data.push(val);
            val += step;
        }
    } else if step < <T as num_traits::Zero>::zero() {
        while val > end {
            data.push(val);
            val += step;
        }
    } else {
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: "arange: step cannot be zero".into(),
        });
    }
    let len = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![len], false)
}

/// Create a 1-D tensor of `num` evenly spaced values from `start` to `end` (inclusive).
pub fn linspace<T: Float>(start: T, end: T, num: usize) -> FerrotorchResult<Tensor<T>> {
    if num == 0 {
        return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0], false);
    }
    if num == 1 {
        return Tensor::from_storage(TensorStorage::cpu(vec![start]), vec![1], false);
    }
    let n = T::from(num - 1).unwrap();
    let step = (end - start) / n;
    let data: Vec<T> = (0..num)
        .map(|i| start + step * T::from(i).unwrap())
        .collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![num], false)
}

/// Create a tensor with random values uniformly distributed in [0, 1).
///
/// Uses a simple xorshift-based PRNG seeded from system time and thread id.
/// For reproducible results, use ferray-random directly and pass the data
/// to `from_vec`.
///
/// # Thread-local RNG and rayon
///
/// **Warning:** The RNG state is seeded per-call from `SystemTime` and the
/// current thread's id. There is no persistent thread-local RNG state, so
/// there is nothing to save/restore for gradient checkpointing. However,
/// this also means rayon worker threads will get independent seeds — if you
/// call `rand` from a rayon thread, the results are not correlated with the
/// main thread's sequence. When proper seeded RNG state management is added
/// (e.g., `manual_seed` + thread-local generator), checkpoint will need to
/// save and restore that state to ensure reproducible recomputation.
pub fn rand<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

    // Seed from system time + thread id for basic uniqueness.
    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 0xdeadbeefcafe;
    }

    for _ in 0..numel {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let val = (state as f64) / (u64::MAX as f64);
        data.push(T::from(val).unwrap());
    }

    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor with random values from a standard normal distribution.
///
/// Uses Box-Muller transform over a xorshift PRNG. See [`rand`] for notes
/// on thread-local RNG limitations and gradient checkpointing.
pub fn randn<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

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
        // Ensure non-zero for log.
        ((state as f64) / (u64::MAX as f64)).max(1e-300)
    };

    let mut i = 0;
    while i < numel {
        let u1 = next_uniform();
        let u2 = next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push(T::from(r * theta.cos()).unwrap());
        if i + 1 < numel {
            data.push(T::from(r * theta.sin()).unwrap());
        }
        i += 2;
    }

    data.truncate(numel);
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor of zeros with the same shape as `other`.
pub fn zeros_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    zeros(other.shape())
}

/// Create a tensor of ones with the same shape as `other`.
pub fn ones_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    ones(other.shape())
}

/// Create a tensor filled with `value` with the same shape as `other`.
pub fn full_like<T: Float>(other: &Tensor<T>, value: T) -> FerrotorchResult<Tensor<T>> {
    full(other.shape(), value)
}

/// Create a random tensor [0,1) with the same shape as `other`.
pub fn rand_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    rand(other.shape())
}

/// Create a random normal tensor with the same shape as `other`.
pub fn randn_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    randn(other.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t: Tensor<f32> = zeros(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t: Tensor<f64> = ones(&[4]).unwrap();
        assert_eq!(t.shape(), &[4]);
        assert!(t.data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let t: Tensor<f32> = full(&[2, 2], 3.14).unwrap();
        assert!(t.data().unwrap().iter().all(|&x| (x - 3.14).abs() < 1e-6));
    }

    #[test]
    fn test_from_slice() {
        let t: Tensor<f32> = from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_1d() {
        let t = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_scalar() {
        let t = scalar(42.0f64).unwrap();
        assert!(t.is_scalar());
        assert_eq!(t.item().unwrap(), 42.0);
    }

    #[test]
    fn test_eye() {
        let t: Tensor<f32> = eye(3).unwrap();
        assert_eq!(t.shape(), &[3, 3]);
        let d = t.data().unwrap();
        assert_eq!(d[0], 1.0); // [0,0]
        assert_eq!(d[1], 0.0); // [0,1]
        assert_eq!(d[4], 1.0); // [1,1]
        assert_eq!(d[8], 1.0); // [2,2]
    }

    #[test]
    fn test_arange() {
        let t: Tensor<f32> = arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(t.shape(), &[5]);
        let d = t.data().unwrap();
        assert_eq!(d, &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_step() {
        let t: Tensor<f64> = arange(0.0, 1.0, 0.25).unwrap();
        assert_eq!(t.shape(), &[4]);
    }

    #[test]
    fn test_arange_zero_step() {
        let result: Result<Tensor<f32>, _> = arange(0.0, 5.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linspace() {
        let t: Tensor<f32> = linspace(0.0, 1.0, 5).unwrap();
        assert_eq!(t.shape(), &[5]);
        let d = t.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6);
        assert!((d[2] - 0.5).abs() < 1e-6);
        assert!((d[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_single() {
        let t: Tensor<f32> = linspace(3.0, 3.0, 1).unwrap();
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.item().unwrap(), 3.0);
    }

    #[test]
    fn test_linspace_empty() {
        let t: Tensor<f32> = linspace(0.0, 1.0, 0).unwrap();
        assert_eq!(t.shape(), &[0]);
    }

    #[test]
    fn test_rand_shape() {
        let t: Tensor<f32> = rand(&[10, 20]).unwrap();
        assert_eq!(t.shape(), &[10, 20]);
        // Values should be in [0, 1).
        assert!(t.data().unwrap().iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_randn_shape() {
        let t: Tensor<f32> = randn(&[100]).unwrap();
        assert_eq!(t.shape(), &[100]);
        // Mean should be roughly 0 for 100 samples.
        let mean: f32 = t.data().unwrap().iter().sum::<f32>() / 100.0;
        assert!(mean.abs() < 1.0); // Very loose check.
    }

    #[test]
    fn test_zeros_empty() {
        let t: Tensor<f32> = zeros(&[0, 3]).unwrap();
        assert_eq!(t.shape(), &[0, 3]);
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn test_zeros_like() {
        let t: Tensor<f32> = rand(&[3, 4]).unwrap();
        let z = zeros_like(&t).unwrap();
        assert_eq!(z.shape(), &[3, 4]);
        assert!(z.data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_like() {
        let t: Tensor<f64> = zeros(&[2, 5]).unwrap();
        let o = ones_like(&t).unwrap();
        assert_eq!(o.shape(), &[2, 5]);
        assert!(o.data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full_like() {
        let t: Tensor<f32> = zeros(&[4, 3]).unwrap();
        let f = full_like(&t, 7.0).unwrap();
        assert_eq!(f.shape(), &[4, 3]);
        assert!(f.data().unwrap().iter().all(|&x| (x - 7.0).abs() < 1e-6));
    }

    #[test]
    fn test_rand_like() {
        let t: Tensor<f32> = zeros(&[5, 6]).unwrap();
        let r = rand_like(&t).unwrap();
        assert_eq!(r.shape(), &[5, 6]);
        assert!(r.data().unwrap().iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_randn_like() {
        let t: Tensor<f32> = zeros(&[50]).unwrap();
        let r = randn_like(&t).unwrap();
        assert_eq!(r.shape(), &[50]);
    }
}
