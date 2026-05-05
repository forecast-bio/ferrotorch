/// Re-export ferray's type system for tensor element types.
///
/// ferray's `Element` trait is sealed — only types ferray knows about can be
/// tensor elements. This module re-exports those types and adds convenience
/// traits for float-specific operations needed by the autograd engine.
pub use ferray_core::{DType, Element};

/// Marker trait for float element types that support autograd.
///
/// This is the bound used by `Tensor<T>` operations that require
/// differentiable arithmetic (add, mul, matmul, activations, etc.).
/// Integer and boolean tensors exist but cannot participate in gradient
/// computation.
///
/// # Implementors
///
/// - `f32` — default working precision, best-tested path.
/// - `f64` — double precision for numerical sensitivity / reference runs.
/// - `half::bf16` — 16-bit brain-float used by Llama 3 and friends.
///   Has the same exponent range as `f32` (8-bit) with a truncated
///   mantissa (7-bit), so it trades precision for range. Well-suited
///   to weight storage; kernels that accumulate (matmul, softmax,
///   norms) should upcast to `f32` or `f64` for the accumulator —
///   `Tensor<bf16>::to(Device::Cpu)` plus a `T::from` cast, or the
///   explicit mixed-precision helpers in `ferrotorch-nn`.
pub trait Float: Element + num_traits::Float + std::ops::AddAssign {}

impl Float for f32 {}
impl Float for f64 {}
impl Float for half::bf16 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_is_float() {
        // Compile-time check: bf16 satisfies the Float trait.
        fn assert_float<T: Float>() {}
        assert_float::<f32>();
        assert_float::<f64>();
        assert_float::<half::bf16>();
    }

    #[test]
    fn bf16_element_dtype() {
        assert_eq!(<half::bf16 as Element>::dtype(), DType::BF16);
    }

    #[test]
    // reason: bit-exact round-trip of 0.0 / 1.0 through bf16 → f32; both values
    // are exactly representable in bf16, so equality (not epsilon) is correct.
    #[allow(clippy::float_cmp)]
    fn bf16_zero_one_round_trip() {
        let zero = <half::bf16 as Element>::zero();
        let one = <half::bf16 as Element>::one();
        assert_eq!(zero.to_f32(), 0.0);
        assert_eq!(one.to_f32(), 1.0);
    }

    #[test]
    // reason: 2 + 3 = 5 in bf16 is bit-exact (small integers are exactly
    // representable and integer addition does not lose mantissa bits).
    #[allow(clippy::float_cmp)]
    fn bf16_num_traits_float_ops() {
        use num_traits::Float as _;
        let a = half::bf16::from_f32(2.0);
        let b = half::bf16::from_f32(3.0);
        // Addition composes via num_traits::Float.
        assert_eq!((a + b).to_f32(), 5.0);
        // sqrt rounds through bf16 precision.
        let s = a.sqrt();
        assert!((s.to_f32() - 2.0_f32.sqrt()).abs() < 1e-2);
    }

    #[test]
    // reason: 1 + 2 = 3 in bf16 is bit-exact (small integers are exactly
    // representable and integer addition does not lose mantissa bits).
    #[allow(clippy::float_cmp)]
    fn bf16_add_assign() {
        let mut x = half::bf16::from_f32(1.0);
        x += half::bf16::from_f32(2.0);
        assert_eq!(x.to_f32(), 3.0);
    }

    #[test]
    fn bf16_tensor_construction_and_shape() {
        use crate::creation::from_slice;
        let data: Vec<half::bf16> = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(3.0),
            half::bf16::from_f32(4.0),
        ];
        let t = from_slice(&data, &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
        let got: Vec<f32> = t.data().unwrap().iter().map(|b| b.to_f32()).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bf16_tensor_addition() {
        use crate::creation::from_slice;
        let xs: Vec<half::bf16> = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(3.0),
        ];
        let a = from_slice(&xs, &[3]).unwrap();
        let b = from_slice(&xs, &[3]).unwrap();
        let c = a.add_t(&b).unwrap();
        let got: Vec<f32> = c.data().unwrap().iter().map(|b| b.to_f32()).collect();
        assert_eq!(got, vec![2.0, 4.0, 6.0]);
    }
}
