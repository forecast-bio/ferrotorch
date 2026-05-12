//! Typed parameter-state key shared across optimizers.
//!
//! Many of the optimizers in this crate maintain a per-parameter state map
//! keyed by `(group_idx, param_idx)`. Historically that key was the result
//! of `format!("g{group_idx}_p{param_idx}")` — recomputed inside every
//! per-parameter iteration of the hot inner step loop. For a 7B-parameter
//! model running an `Adam`-family optimizer that meant a fresh `String`
//! heap allocation per parameter per step, with the format machinery on
//! top.
//!
//! [`ParamKey`] replaces that allocation-heavy hashing key with an
//! eight-byte `Copy` value that hashes directly without touching the heap.
//! The serialized wire format (used by [`super::OptimizerState`]) stays
//! `"g{group_idx}_p{param_idx}"` so checkpoints written before this
//! migration still round-trip: the conversion happens at the
//! `state_dict` / `load_state_dict` boundary via [`std::fmt::Display`] and
//! [`std::str::FromStr`].
//!
//! Issue: #1122.

use std::fmt;
use std::str::FromStr;

use ferrotorch_core::FerrotorchError;

/// Stable per-parameter identity within a single optimizer instance.
///
/// `ParamKey` is a `Copy` newtype over `(group_idx, param_idx)` used as
/// the lookup key in optimizer-internal `HashMap`s. The 32-bit-per-field
/// representation is more than enough for any realistic model
/// configuration (no optimizer in practice holds 4 billion parameter
/// groups or 4 billion parameters within one group), and it keeps the
/// type a single 8-byte `Copy` value — small enough to pass by value
/// through the lookup chain without heap allocation.
///
/// # Wire format
///
/// On serialization (via [`Display`]) and deserialization (via
/// [`FromStr`]) the value uses the string format `"g{group}_p{param}"`.
/// This matches the legacy `format!("g{}_p{}")` keys, so checkpoints
/// written by older versions of this crate round-trip unchanged.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParamKey {
    /// Index of the parameter group (`Optimizer::param_groups()[group]`).
    pub group: u32,
    /// Index of the parameter within its group
    /// (`Optimizer::param_groups()[group].params()[param]`).
    pub param: u32,
}

impl ParamKey {
    /// Construct a new key from the parameter's `(group_idx, param_idx)`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if either index does not fit in `u32`. No
    /// optimizer in this crate is reachable with indices that large; the
    /// debug assertion exists to catch a programming error early rather
    /// than silently truncating.
    #[inline]
    #[must_use]
    pub const fn new(group: usize, param: usize) -> Self {
        debug_assert!(group <= u32::MAX as usize);
        debug_assert!(param <= u32::MAX as usize);
        Self {
            group: group as u32,
            param: param as u32,
        }
    }
}

impl fmt::Display for ParamKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}_p{}", self.group, self.param)
    }
}

impl From<ParamKey> for String {
    #[inline]
    fn from(key: ParamKey) -> Self {
        key.to_string()
    }
}

impl FromStr for ParamKey {
    type Err = FerrotorchError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let invalid = || FerrotorchError::InvalidArgument {
            message: format!(
                "ParamKey: expected 'g{{group}}_p{{param}}' format, got {s:?}"
            ),
        };
        let rest = s.strip_prefix('g').ok_or_else(invalid)?;
        let (g_str, p_part) = rest.split_once("_p").ok_or_else(invalid)?;
        let group: u32 = g_str.parse().map_err(|_| invalid())?;
        let param: u32 = p_part.parse().map_err(|_| invalid())?;
        Ok(Self { group, param })
    }
}

impl TryFrom<&str> for ParamKey {
    type Error = FerrotorchError;

    #[inline]
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        s.parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_legacy_format() {
        let k = ParamKey::new(0, 0);
        assert_eq!(k.to_string(), "g0_p0");
        let k = ParamKey::new(3, 17);
        assert_eq!(k.to_string(), "g3_p17");
    }

    #[test]
    fn from_str_round_trips_display() {
        for &(g, p) in &[(0usize, 0usize), (1, 2), (3, 17), (100, 99_999)] {
            let k = ParamKey::new(g, p);
            let parsed: ParamKey = k.to_string().parse().unwrap();
            assert_eq!(parsed, k);
        }
    }

    #[test]
    fn from_str_rejects_bad_format() {
        assert!("".parse::<ParamKey>().is_err());
        assert!("0_0".parse::<ParamKey>().is_err());
        assert!("g0p0".parse::<ParamKey>().is_err());
        assert!("g_p".parse::<ParamKey>().is_err());
        assert!("gx_p0".parse::<ParamKey>().is_err());
        assert!("g0_py".parse::<ParamKey>().is_err());
    }

    #[test]
    fn copy_and_hash_eq_consistent() {
        use std::collections::HashMap;

        let mut m: HashMap<ParamKey, i32> = HashMap::new();
        let k = ParamKey::new(2, 5);
        m.insert(k, 42);
        // Copy semantics: `k` is still usable.
        assert_eq!(m.get(&k), Some(&42));
        assert_eq!(m.get(&ParamKey::new(2, 5)), Some(&42));
        assert_eq!(m.get(&ParamKey::new(5, 2)), None);
    }

    #[test]
    fn string_conversion() {
        let k = ParamKey::new(7, 3);
        let s: String = k.into();
        assert_eq!(s, "g7_p3");
        let k2 = ParamKey::try_from("g7_p3").unwrap();
        assert_eq!(k2, k);
    }
}
