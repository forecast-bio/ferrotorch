//! `ComplexTensor<T>` — first-class complex-valued tensors. (#618)
//!
//! Stores real and imaginary parts as two parallel buffers of `T`. This
//! is **structure-of-arrays**, opposite to the **array-of-structures**
//! `[..., 2]`-trailing layout used by `fft::*` (which packs `[re, im]`
//! pairs interleaved). Both representations are valid; the SoA form
//! makes elementwise complex math cheap to write without per-element
//! struct fiddling, while the AoS form is what cuFFT and `safetensors`
//! expect on the wire.
//!
//! Conversions to/from the AoS / interleaved representation are O(N).
//!
//! # Why not `Tensor<num_complex::Complex<f32>>`?
//!
//! That would require generalizing `Tensor<T: Float>` over complex
//! types — touches every op surface and the GPU dispatch trait.
//! `ComplexTensor` is a focused standalone type, same shape as the
//! `IntTensor` / `BoolTensor` additions in #596: callers who need
//! complex math get the right surface; existing float ops keep working.

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// CPU-resident, contiguous, structure-of-arrays complex tensor.
///
/// Invariant: `re.len() == im.len()`. `Arc`-shared so clones are cheap.
#[derive(Debug, Clone)]
pub struct ComplexTensor<T: Float> {
    re: Arc<Vec<T>>,
    im: Arc<Vec<T>>,
    shape: Vec<usize>,
}

impl<T: Float> ComplexTensor<T> {
    /// Build from separate real and imaginary buffers + shape.
    pub fn from_re_im(re: Vec<T>, im: Vec<T>, shape: Vec<usize>) -> FerrotorchResult<Self> {
        let expected: usize = shape.iter().product::<usize>().max(1);
        if re.len() != expected || im.len() != expected {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ComplexTensor::from_re_im: re.len()={}, im.len()={}, expected {} for shape {:?}",
                    re.len(),
                    im.len(),
                    expected,
                    shape
                ),
            });
        }
        if re.len() != im.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: "ComplexTensor::from_re_im: re/im length mismatch".into(),
            });
        }
        Ok(Self {
            re: Arc::new(re),
            im: Arc::new(im),
            shape,
        })
    }

    /// Build a complex tensor from a real-valued one (zero imaginary).
    pub fn from_real(real: &Tensor<T>) -> FerrotorchResult<Self> {
        let re = real.data_vec()?;
        let im = vec![<T as num_traits::Zero>::zero(); re.len()];
        Self::from_re_im(re, im, real.shape().to_vec())
    }

    /// Zero-filled complex tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product::<usize>().max(1);
        let zero = <T as num_traits::Zero>::zero();
        Self {
            re: Arc::new(vec![zero; total]),
            im: Arc::new(vec![zero; total]),
            shape: shape.to_vec(),
        }
    }

    /// 0-d scalar `re + im·i`.
    pub fn scalar(re: T, im: T) -> Self {
        Self {
            re: Arc::new(vec![re]),
            im: Arc::new(vec![im]),
            shape: Vec::new(),
        }
    }

    /// Build from the interleaved AoS form used by `fft::*`: a real
    /// tensor of shape `[..., 2]` where the trailing dim packs `[re, im]`
    /// pairs. Returns a `ComplexTensor` of shape `[...]` (one less dim).
    pub fn from_interleaved(t: &Tensor<T>) -> FerrotorchResult<Self> {
        let shape = t.shape();
        if shape.is_empty() || *shape.last().unwrap() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ComplexTensor::from_interleaved: input must have trailing dim 2, got {:?}",
                    shape
                ),
            });
        }
        let n = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let data = t.data_vec()?;
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        for i in 0..n {
            re.push(data[2 * i]);
            im.push(data[2 * i + 1]);
        }
        let new_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        Self::from_re_im(re, im, new_shape)
    }

    /// Inverse of [`from_interleaved`]: emit a real tensor with trailing
    /// dim 2 containing `[re, im]` pairs. Useful before passing into
    /// the existing `fft::fft` etc.
    pub fn to_interleaved(&self) -> FerrotorchResult<Tensor<T>> {
        let n = self.re.len();
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.push(self.re[i]);
            data.push(self.im[i]);
        }
        let mut new_shape = self.shape.clone();
        new_shape.push(2);
        Tensor::from_storage(TensorStorage::cpu(data), new_shape, false)
    }

    /// Real-part view as a `Tensor<T>` (clones into a fresh tensor).
    pub fn real(&self) -> FerrotorchResult<Tensor<T>> {
        Tensor::from_storage(
            TensorStorage::cpu(self.re.as_ref().clone()),
            self.shape.clone(),
            false,
        )
    }

    /// Imaginary-part view as a `Tensor<T>`.
    pub fn imag(&self) -> FerrotorchResult<Tensor<T>> {
        Tensor::from_storage(
            TensorStorage::cpu(self.im.as_ref().clone()),
            self.shape.clone(),
            false,
        )
    }

    /// Logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total elements.
    pub fn numel(&self) -> usize {
        self.re.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Borrow real part buffer.
    pub fn re(&self) -> &[T] {
        &self.re
    }

    /// Borrow imaginary part buffer.
    pub fn im(&self) -> &[T] {
        &self.im
    }

    /// Pointwise complex add: `(a+bi) + (c+di) = (a+c) + (b+d)i`.
    pub fn add(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary(
            other,
            |a_re, a_im, b_re, b_im| (a_re + b_re, a_im + b_im),
            "add",
        )
    }

    /// Pointwise complex subtract.
    pub fn sub(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary(
            other,
            |a_re, a_im, b_re, b_im| (a_re - b_re, a_im - b_im),
            "sub",
        )
    }

    /// Pointwise complex multiply: `(a+bi)(c+di) = (ac - bd) + (ad + bc)i`.
    pub fn mul(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary(
            other,
            |a_re, a_im, b_re, b_im| (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re),
            "mul",
        )
    }

    /// Complex conjugate: negate the imaginary part.
    pub fn conj(&self) -> Self {
        let im: Vec<T> = self.im.iter().map(|&v| -v).collect();
        Self {
            re: Arc::clone(&self.re),
            im: Arc::new(im),
            shape: self.shape.clone(),
        }
    }

    /// Pointwise modulus / magnitude: `|a + bi| = sqrt(a^2 + b^2)`.
    /// Returns a real-valued `Tensor<T>`.
    pub fn abs(&self) -> FerrotorchResult<Tensor<T>> {
        let data: Vec<T> = self
            .re
            .iter()
            .zip(self.im.iter())
            .map(|(&a, &b)| (a * a + b * b).sqrt())
            .collect();
        Tensor::from_storage(TensorStorage::cpu(data), self.shape.clone(), false)
    }

    /// Phase angle (radians): `atan2(im, re)`.
    pub fn angle(&self) -> FerrotorchResult<Tensor<T>> {
        let data: Vec<T> = self
            .re
            .iter()
            .zip(self.im.iter())
            .map(|(&a, &b)| b.atan2(a))
            .collect();
        Tensor::from_storage(TensorStorage::cpu(data), self.shape.clone(), false)
    }

    /// 2-D complex matrix multiplication.
    ///
    /// Both operands must be 2-D; trailing dim of `self` must match leading
    /// dim of `other`. Computed via four real `mm` calls under the hood:
    /// for `a + bi` and `c + di`,
    ///   `(a + bi) @ (c + di) = (a@c - b@d) + (a@d + b@c)i`.
    /// This keeps the implementation entirely in the existing real `mm`
    /// surface — no new GEMM kernels needed. (#624)
    pub fn matmul(&self, other: &Self) -> FerrotorchResult<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ComplexTensor::matmul: both operands must be 2-D, got {:?} and {:?}",
                    self.shape, other.shape
                ),
            });
        }
        if self.shape[1] != other.shape[0] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ComplexTensor::matmul: inner dims must match, got {} vs {}",
                    self.shape[1], other.shape[0]
                ),
            });
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        // Wrap real and imaginary parts in real Tensors so we can call mm.
        let a_re = Tensor::from_storage(
            TensorStorage::cpu((*self.re).clone()),
            self.shape.clone(),
            false,
        )?;
        let a_im = Tensor::from_storage(
            TensorStorage::cpu((*self.im).clone()),
            self.shape.clone(),
            false,
        )?;
        let b_re = Tensor::from_storage(
            TensorStorage::cpu((*other.re).clone()),
            other.shape.clone(),
            false,
        )?;
        let b_im = Tensor::from_storage(
            TensorStorage::cpu((*other.im).clone()),
            other.shape.clone(),
            false,
        )?;

        let ac = crate::ops::linalg::mm(&a_re, &b_re)?;
        let bd = crate::ops::linalg::mm(&a_im, &b_im)?;
        let ad = crate::ops::linalg::mm(&a_re, &b_im)?;
        let bc = crate::ops::linalg::mm(&a_im, &b_re)?;

        let ac_d = ac.data()?;
        let bd_d = bd.data()?;
        let ad_d = ad.data()?;
        let bc_d = bc.data()?;

        let total = m * n;
        let mut out_re = Vec::with_capacity(total);
        let mut out_im = Vec::with_capacity(total);
        for i in 0..total {
            out_re.push(ac_d[i] - bd_d[i]);
            out_im.push(ad_d[i] + bc_d[i]);
        }
        let _ = k;
        Self::from_re_im(out_re, out_im, vec![m, n])
    }

    /// 1-D complex FFT along the last logical dim. Bridges to the existing
    /// interleaved-real `crate::fft::fft` by round-tripping through
    /// [`Self::to_interleaved`] / [`Self::from_interleaved`]. `n` truncates
    /// or zero-pads the signal length, matching `torch.fft.fft`. (#624)
    pub fn fft(&self, n: Option<usize>) -> FerrotorchResult<Self> {
        let interleaved = self.to_interleaved()?;
        let out = crate::fft::fft(&interleaved, n)?;
        Self::from_interleaved(&out)
    }

    /// 1-D inverse complex FFT — counterpart to [`Self::fft`].
    pub fn ifft(&self, n: Option<usize>) -> FerrotorchResult<Self> {
        let interleaved = self.to_interleaved()?;
        let out = crate::fft::ifft(&interleaved, n)?;
        Self::from_interleaved(&out)
    }

    /// 2-D complex FFT (last two logical dims).
    pub fn fft2(&self) -> FerrotorchResult<Self> {
        let interleaved = self.to_interleaved()?;
        let out = crate::fft::fft2(&interleaved)?;
        Self::from_interleaved(&out)
    }

    /// 2-D inverse complex FFT.
    pub fn ifft2(&self) -> FerrotorchResult<Self> {
        let interleaved = self.to_interleaved()?;
        let out = crate::fft::ifft2(&interleaved)?;
        Self::from_interleaved(&out)
    }

    /// Reshape (must preserve numel; no data copy).
    pub fn reshape(&self, shape: &[usize]) -> FerrotorchResult<Self> {
        let new_total: usize = shape.iter().product::<usize>().max(1);
        if new_total != self.re.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ComplexTensor::reshape: new shape {:?} (numel {}) != current numel {}",
                    shape,
                    new_total,
                    self.re.len()
                ),
            });
        }
        Ok(Self {
            re: Arc::clone(&self.re),
            im: Arc::clone(&self.im),
            shape: shape.to_vec(),
        })
    }

    fn binary<F: Fn(T, T, T, T) -> (T, T)>(
        &self,
        other: &Self,
        f: F,
        op: &str,
    ) -> FerrotorchResult<Self> {
        if self.shape != other.shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ComplexTensor::{op}: shapes {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let n = self.re.len();
        let mut out_re = Vec::with_capacity(n);
        let mut out_im = Vec::with_capacity(n);
        for i in 0..n {
            let (r, im) = f(self.re[i], self.im[i], other.re[i], other.im[i]);
            out_re.push(r);
            out_im.push(im);
        }
        Ok(Self {
            re: Arc::new(out_re),
            im: Arc::new(out_im),
            shape: self.shape.clone(),
        })
    }
}

impl<T: Float> std::fmt::Display for ComplexTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ComplexTensor(shape={:?}, numel={})",
            self.shape,
            self.numel()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn complex_construction_from_re_im() {
        let c = ComplexTensor::<f32>::from_re_im(vec![1.0, 2.0], vec![3.0, 4.0], vec![2]).unwrap();
        assert_eq!(c.numel(), 2);
        assert_eq!(c.re(), &[1.0, 2.0]);
        assert_eq!(c.im(), &[3.0, 4.0]);
    }

    #[test]
    fn complex_from_real_zero_imag() {
        let r = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]), vec![3], false)
            .unwrap();
        let c = ComplexTensor::from_real(&r).unwrap();
        assert_eq!(c.re(), &[1.0, 2.0, 3.0]);
        assert!(c.im().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn complex_zeros() {
        let c = ComplexTensor::<f32>::zeros(&[2, 3]);
        assert_eq!(c.numel(), 6);
        assert!(c.re().iter().all(|&x| x == 0.0));
        assert!(c.im().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn complex_scalar_constructor() {
        let c = ComplexTensor::<f32>::scalar(2.0, 3.0);
        assert_eq!(c.shape(), &[] as &[usize]);
        assert_eq!(c.re()[0], 2.0);
        assert_eq!(c.im()[0], 3.0);
    }

    #[test]
    fn complex_interleaved_roundtrip() {
        // Interleaved [N, 2] → ComplexTensor([N]) → back.
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            false,
        )
        .unwrap();
        let c = ComplexTensor::from_interleaved(&t).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.re(), &[1.0, 3.0, 5.0]);
        assert_eq!(c.im(), &[2.0, 4.0, 6.0]);
        let t2 = c.to_interleaved().unwrap();
        assert_eq!(t.data().unwrap(), t2.data().unwrap());
    }

    #[test]
    fn complex_interleaved_rejects_wrong_trailing_dim() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]), vec![3], false)
            .unwrap();
        let err = ComplexTensor::from_interleaved(&t).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn complex_real_imag_extraction() {
        let c = ComplexTensor::<f32>::from_re_im(vec![1.0, 2.0], vec![3.0, 4.0], vec![2]).unwrap();
        let r = c.real().unwrap();
        let i = c.imag().unwrap();
        assert_eq!(r.data().unwrap(), &[1.0, 2.0]);
        assert_eq!(i.data().unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn complex_pointwise_add() {
        // (1+2i) + (3+4i) = (4+6i)
        let a = ComplexTensor::<f32>::from_re_im(vec![1.0], vec![2.0], vec![1]).unwrap();
        let b = ComplexTensor::<f32>::from_re_im(vec![3.0], vec![4.0], vec![1]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.re()[0], 4.0);
        assert_eq!(c.im()[0], 6.0);
    }

    #[test]
    fn complex_pointwise_sub() {
        let a = ComplexTensor::<f32>::from_re_im(vec![5.0], vec![6.0], vec![1]).unwrap();
        let b = ComplexTensor::<f32>::from_re_im(vec![3.0], vec![2.0], vec![1]).unwrap();
        let c = a.sub(&b).unwrap();
        assert_eq!(c.re()[0], 2.0);
        assert_eq!(c.im()[0], 4.0);
    }

    #[test]
    fn complex_pointwise_mul() {
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 - 8 + (4+6)i = -5 + 10i
        let a = ComplexTensor::<f32>::from_re_im(vec![1.0], vec![2.0], vec![1]).unwrap();
        let b = ComplexTensor::<f32>::from_re_im(vec![3.0], vec![4.0], vec![1]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.re()[0], -5.0);
        assert_eq!(c.im()[0], 10.0);
    }

    #[test]
    fn complex_conj_negates_imag() {
        let c = ComplexTensor::<f32>::from_re_im(vec![1.0, -2.0], vec![3.0, 4.0], vec![2]).unwrap();
        let cc = c.conj();
        assert_eq!(cc.re(), c.re());
        assert_eq!(cc.im(), &[-3.0, -4.0]);
    }

    #[test]
    fn complex_abs_pythagorean() {
        // |3 + 4i| = 5
        let c = ComplexTensor::<f32>::from_re_im(vec![3.0], vec![4.0], vec![1]).unwrap();
        let m = c.abs().unwrap();
        assert!(close(m.data().unwrap()[0], 5.0, 1e-6));
    }

    #[test]
    fn complex_angle_quadrants() {
        // angle(1+0i) = 0; angle(0+1i) = π/2; angle(-1+0i) = π; angle(0-1i) = -π/2
        let c = ComplexTensor::<f32>::from_re_im(
            vec![1.0, 0.0, -1.0, 0.0],
            vec![0.0, 1.0, 0.0, -1.0],
            vec![4],
        )
        .unwrap();
        let a = c.angle().unwrap();
        let d = a.data().unwrap();
        let pi = std::f32::consts::PI;
        assert!(close(d[0], 0.0, 1e-6));
        assert!(close(d[1], pi / 2.0, 1e-6));
        assert!(close(d[2], pi, 1e-6));
        assert!(close(d[3], -pi / 2.0, 1e-6));
    }

    #[test]
    fn complex_reshape_preserves_data() {
        let c = ComplexTensor::<f32>::from_re_im(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![6],
        )
        .unwrap();
        let r = c.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.re(), c.re());
        assert_eq!(r.im(), c.im());
    }

    #[test]
    fn complex_reshape_size_mismatch_errors() {
        let c = ComplexTensor::<f32>::from_re_im(vec![1.0, 2.0], vec![3.0, 4.0], vec![2]).unwrap();
        let err = c.reshape(&[3]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn complex_binary_op_shape_mismatch() {
        let a = ComplexTensor::<f32>::zeros(&[3]);
        let b = ComplexTensor::<f32>::zeros(&[2]);
        let err = a.add(&b).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn complex_clone_shares_arc_buffers() {
        let c = ComplexTensor::<f32>::zeros(&[5]);
        let c2 = c.clone();
        assert!(Arc::ptr_eq(&c.re, &c2.re));
        assert!(Arc::ptr_eq(&c.im, &c2.im));
    }

    // ---------------------------------------------------------------
    // matmul + FFT integration (#624)
    // ---------------------------------------------------------------

    #[test]
    fn complex_matmul_2x2_known_value() {
        // A = [[1+1i, 0], [0, 1+0i]], B = [[1+0i, 0], [0, 2+1i]]
        // A @ B = [[1+1i, 0], [0, 2+1i]]
        let a = ComplexTensor::<f64>::from_re_im(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::<f64>::from_re_im(
            vec![1.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![2, 2],
        )
        .unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.re(), &[1.0, 0.0, 0.0, 2.0]);
        assert_eq!(c.im(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn complex_matmul_rejects_shape_mismatch() {
        let a = ComplexTensor::<f64>::zeros(&[2, 3]);
        let b = ComplexTensor::<f64>::zeros(&[2, 4]); // inner dim mismatch
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn complex_matmul_against_real_path_when_im_is_zero() {
        // Pure-real complex inputs should give the same answer as a real mm.
        let a =
            ComplexTensor::<f64>::from_re_im(vec![1.0, 2.0, 3.0, 4.0], vec![0.0; 4], vec![2, 2])
                .unwrap();
        let b =
            ComplexTensor::<f64>::from_re_im(vec![5.0, 6.0, 7.0, 8.0], vec![0.0; 4], vec![2, 2])
                .unwrap();
        let c = a.matmul(&b).unwrap();
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert_eq!(c.re(), &[19.0, 22.0, 43.0, 50.0]);
        assert!(c.im().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn complex_fft_ifft_roundtrip() {
        // Round-trip through fft + ifft should recover the original complex
        // signal (up to floating-point tolerance) — both routes go through
        // the existing interleaved fft/ifft surface.
        let signal = ComplexTensor::<f64>::from_re_im(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.5, -0.5, 1.0, -1.0],
            vec![4],
        )
        .unwrap();
        let fft = signal.fft(None).unwrap();
        assert_eq!(fft.shape(), &[4]);
        let back = fft.ifft(None).unwrap();
        let tol = 1e-10;
        for i in 0..4 {
            assert!((back.re()[i] - signal.re()[i]).abs() < tol);
            assert!((back.im()[i] - signal.im()[i]).abs() < tol);
        }
    }

    #[test]
    fn complex_fft2_ifft2_roundtrip() {
        // Build a 4x4 complex signal and round-trip through fft2/ifft2.
        let mut re = vec![0.0_f64; 16];
        let mut im = vec![0.0_f64; 16];
        for i in 0..16 {
            re[i] = (i as f64) * 0.5;
            im[i] = (i as f64) * 0.25;
        }
        let signal = ComplexTensor::<f64>::from_re_im(re.clone(), im.clone(), vec![4, 4]).unwrap();
        let fft = signal.fft2().unwrap();
        let back = fft.ifft2().unwrap();
        let tol = 1e-9;
        for i in 0..16 {
            assert!((back.re()[i] - re[i]).abs() < tol);
            assert!((back.im()[i] - im[i]).abs() < tol);
        }
    }
}
