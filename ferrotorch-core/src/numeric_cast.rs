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
//!
//! # Saturation guard (issue #815)
//!
//! `num_traits::NumCast` for narrow float targets (`half::bf16`,
//! `half::f16`) silently *saturates* out-of-range source values to
//! `±Infinity` instead of returning `None`. That violates the cast
//! contract ("Err on values not representable in target dtype") and
//! produces silent corruption downstream. We therefore wrap the raw
//! `NumCast::from` call with an explicit finiteness check: if the
//! source was a finite value but the result is non-finite, the cast
//! saturated and we return `Err(InvalidArgument)`. Genuine `±Infinity`
//! and `NaN` sources pass through untouched, because they were already
//! non-finite in the source domain.

use crate::error::{FerrotorchError, FerrotorchResult};

/// Fallible numeric cast from `T` to `U` via `num_traits::NumCast`.
///
/// Returns `Err(FerrotorchError::InvalidArgument)` with a structured message
/// when the source value cannot be represented in the target type. Common
/// failure modes:
///
/// - `f64::INFINITY` -> any integer type.
/// - A `f32` exceeding `i8::MAX` cast to `i8`.
/// - A finite `f64` (e.g. `1e300`) cast to `bf16` or `f16`: the source is
///   in-range as `f64` but overflows the narrow target's exponent range.
///   `num_traits` would silently saturate to `±Infinity`; this helper
///   catches that and returns `Err` (issue #815).
///
/// `f64::INFINITY` and `f64::NAN` cast to `bf16` (or any other float
/// target) pass through as `bf16::INFINITY` / `bf16::NAN` because the
/// source itself is non-finite — the cast did not saturate, it preserved
/// non-finiteness.
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
/// Returns `FerrotorchError::InvalidArgument` when:
///
/// - `NumCast::from` returns `None` (source unrepresentable in target), or
/// - `NumCast::from` returns `Some(non_finite)` for a finite source —
///   i.e. the underlying cast silently saturated. See the module-level
///   note about issue #815.
#[inline]
pub fn cast<T, U>(v: T) -> FerrotorchResult<U>
where
    T: num_traits::ToPrimitive + std::fmt::Debug + Copy,
    U: num_traits::NumCast,
{
    let result: U = <U as num_traits::NumCast>::from(v).ok_or_else(|| {
        FerrotorchError::InvalidArgument {
            message: format!(
                "cast from {} to {} failed: value {:?} not representable",
                std::any::type_name::<T>(),
                std::any::type_name::<U>(),
                v,
            ),
        }
    })?;

    // Saturation guard for issue #815. `NumCast` may silently saturate a
    // finite source to `±Infinity` when the target is a narrow float
    // (bf16, f16). Detect this by routing both source and result through
    // `ToPrimitive::to_f64()`:
    //   * Source was non-finite (Inf / NaN) → genuine non-finite passthrough.
    //   * Source was finite, result is non-finite → saturation; return Err.
    //   * Either side cannot project to f64 → cannot diagnose; pass through.
    //
    // For integer targets the result always projects to a finite f64, so
    // the check is a no-op cost in those (common) call sites.
    let src_finite = v.to_f64().is_some_and(f64::is_finite);
    if src_finite {
        if let Some(r) = result.to_f64() {
            if !r.is_finite() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "cast from {} to {} failed: value {:?} saturates to non-finite ({}) and is not representable",
                        std::any::type_name::<T>(),
                        std::any::type_name::<U>(),
                        v,
                        r,
                    ),
                });
            }
        }
    }

    Ok(result)
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
    // reason: 42 is exactly representable in f32 (any integer up to 2^24 is),
    // so usize→f32 round-trip is bit-exact and equality is the correct check.
    #[allow(clippy::float_cmp)]
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

    // ---------------------------------------------------------------------
    // Issue #815 regression tests — saturation guard for narrow floats.
    //
    // `num_traits::NumCast` for `bf16` and `f16` silently saturates an
    // out-of-range finite source to `±Infinity` instead of returning
    // `None`. The `cast` helper layers a finiteness check on top so the
    // contract ("Err on values not representable in target dtype") is
    // honored. These tests pin the post-fix behavior and serve as the
    // regression bar for #815.
    // ---------------------------------------------------------------------

    #[test]
    fn cast_huge_f64_to_bf16_returns_err() {
        // 1e300 is finite as f64 but exceeds bf16's max (~3.39e38, but the
        // exponent range of bf16 is 8-bit and any value above ~3.39e38
        // saturates to bf16::INFINITY in `num_traits`'s impl).
        let r: FerrotorchResult<half::bf16> = cast(1e300_f64);
        assert!(r.is_err(), "expected Err for finite-source saturation");
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("saturates to non-finite") || msg.contains("not representable"),
            "got: {msg}"
        );
    }

    #[test]
    fn cast_huge_f32_to_bf16_returns_err() {
        // f32::MAX (~3.4e38) is finite as f32 but exceeds bf16 representable
        // range; `num_traits` saturates to bf16::INFINITY.
        let r: FerrotorchResult<half::bf16> = cast(f32::MAX);
        assert!(r.is_err(), "expected Err for finite-source saturation");
    }

    #[test]
    fn cast_f64_inf_to_bf16_passes_through() {
        // Source is genuinely non-finite — the cast did not saturate, it
        // preserved Infinity. This must succeed.
        let v: half::bf16 = cast(f64::INFINITY).expect("Inf passthrough");
        assert!(v.is_infinite(), "expected bf16::INFINITY, got {v}");
    }

    #[test]
    fn cast_f64_neg_inf_to_bf16_passes_through() {
        let v: half::bf16 = cast(f64::NEG_INFINITY).expect("-Inf passthrough");
        assert!(v.is_infinite(), "expected bf16::NEG_INFINITY, got {v}");
    }

    #[test]
    fn cast_f64_nan_to_bf16_passes_through() {
        // NaN is non-finite at the source, so passthrough is correct.
        let v: half::bf16 = cast(f64::NAN).expect("NaN passthrough");
        assert!(v.is_nan(), "expected bf16::NAN");
    }

    #[test]
    fn cast_f64_in_range_to_bf16_succeeds() {
        let v: half::bf16 = cast(1.5_f64).expect("in-range");
        assert!((v.to_f32() - 1.5).abs() < 0.01);
    }

    #[test]
    fn cast_huge_f64_to_f16_returns_err() {
        // f16's max is ~65504. 1e30 vastly exceeds it; `num_traits`
        // saturates to f16::INFINITY without the guard.
        let r: FerrotorchResult<half::f16> = cast(1e30_f64);
        assert!(r.is_err(), "expected Err for finite-source saturation");
    }

    #[test]
    fn cast_f64_inf_to_f16_passes_through() {
        let v: half::f16 = cast(f64::INFINITY).expect("Inf passthrough");
        assert!(v.is_infinite(), "expected f16::INFINITY, got {v}");
    }

    #[test]
    fn cast_f64_nan_to_f16_passes_through() {
        let v: half::f16 = cast(f64::NAN).expect("NaN passthrough");
        assert!(v.is_nan(), "expected f16::NAN");
    }

    #[test]
    fn cast_f64_in_range_to_f16_succeeds() {
        let v: half::f16 = cast(1.5_f64).expect("in-range");
        assert!((v.to_f32() - 1.5).abs() < 0.01);
    }

    #[test]
    fn cast_huge_f64_to_f32_returns_err() {
        // f64::MAX is finite as f64 but exceeds f32::MAX (~3.4e38);
        // `num_traits` saturates to f32::INFINITY without the guard.
        let r: FerrotorchResult<f32> = cast(f64::MAX);
        assert!(r.is_err(), "expected Err for finite-source saturation");
    }

    #[test]
    fn cast_f64_inf_to_f32_passes_through() {
        // Genuine Infinity passthrough across the f64 → f32 narrowing too.
        let v: f32 = cast(f64::INFINITY).expect("Inf passthrough");
        assert!(v.is_infinite());
    }
}
