//! Fallible numeric conversions used across the workspace.
//!
//! Many places in ferrotorch need to convert between numeric types (e.g.
//! `f64` -> `T: Float`, `usize` -> `f32`) where the conversion can fail
//! because the source value isn't representable in the target type
//! (NaN through `bf16`, infinities, out-of-range integers). The naive
//! shape `T::from(v).unwrap()` panics — which is forbidden in library
//! code by `rust-quality` §3 — so callers route through this helper
//! and propagate the error via `?`.
//!
//! This is the **shared** helper for the workspace; downstream crates
//! (e.g. `ferrotorch-vision`) should call this rather than re-author a
//! local copy.

use crate::error::{FerrotorchError, FerrotorchResult};

/// Fallible numeric cast from `T` to `U` via `num_traits::NumCast`.
///
/// Returns `Err(FerrotorchError::InvalidArgument)` with a structured message
/// when the source value cannot be represented in the target type. Common
/// failure modes:
///
/// - `f64::INFINITY` -> any integer type.
/// - A `f32` exceeding `i8::MAX` cast to `i8`.
/// - `f64::NAN` cast to `bf16` (the bf16 NaN is representable; this typically
///   does succeed, but the API is still fallible by contract).
///
/// # Examples
///
/// ```
/// use ferrotorch_core::numeric_cast::cast;
///
/// let x: f64 = 3.5;
/// let y: f32 = cast(x).unwrap();
/// assert!((y - 3.5_f32).abs() < f32::EPSILON);
///
/// // Out-of-range fails cleanly instead of panicking.
/// let huge = f64::INFINITY;
/// assert!(cast::<f64, i32>(huge).is_err());
/// ```
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` with a message naming the
/// source type, target type, and (if `Display`-able) the source value when
/// `NumCast::from` returns `None`.
#[inline]
pub fn cast<T, U>(v: T) -> FerrotorchResult<U>
where
    T: num_traits::ToPrimitive + std::fmt::Debug + Copy,
    U: num_traits::NumCast,
{
    <U as num_traits::NumCast>::from(v).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!(
            "cast from {} to {} failed: value {:?} not representable",
            std::any::type_name::<T>(),
            std::any::type_name::<U>(),
            v,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cast_f64_to_f32_succeeds_for_finite() {
        let x: f32 = cast(3.5_f64).unwrap();
        assert!((x - 3.5_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn cast_f64_inf_to_i32_fails() {
        let r: FerrotorchResult<i32> = cast(f64::INFINITY);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("not representable"), "got: {msg}");
    }

    #[test]
    fn cast_usize_to_f32_succeeds() {
        let x: f32 = cast(42_usize).unwrap();
        assert_eq!(x, 42.0);
    }

    #[test]
    fn cast_to_bf16_round_trip() {
        let x: half::bf16 = cast(1.5_f64).unwrap();
        assert!((x.to_f32() - 1.5).abs() < 0.01);
    }

    #[test]
    fn cast_negative_to_unsigned_fails() {
        let r: FerrotorchResult<u32> = cast(-1_i32);
        assert!(r.is_err());
    }
}
