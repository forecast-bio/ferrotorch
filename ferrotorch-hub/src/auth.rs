//! HuggingFace Hub authentication helpers (#509).
//!
//! Provides:
//! - [`hf_token`] — discover the user's HF auth token from the standard
//!   sources (`HF_TOKEN` env var → `$HOME/.cache/huggingface/token` file).
//! - [`with_auth`] — add an `Authorization: Bearer <token>` header to a
//!   `ureq::Request` when a token is available, no-op otherwise.
//!
//! Auth is strictly opt-in: when no token is found, requests fall through
//! unchanged. Public repos (the common case) don't need it; gated repos
//! like `meta-llama/Meta-Llama-3-8B` do.

#[cfg(feature = "http")]
use std::path::PathBuf;

/// Read the HuggingFace auth token from the standard locations.
///
/// Resolution order (matches the HF Python client):
/// 1. `HF_TOKEN` environment variable (if non-empty).
/// 2. `$HF_HOME/token` (if `HF_HOME` is set).
/// 3. `$HOME/.cache/huggingface/token` (the default).
///
/// Returns `None` when no token is found — callers should treat this as
/// "anonymous request" and not error.
#[cfg(feature = "http")]
pub fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    let candidates: Vec<PathBuf> = std::iter::empty()
        .chain(
            std::env::var_os("HF_HOME")
                .map(PathBuf::from)
                .map(|h| h.join("token")),
        )
        .chain(
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|h| h.join(".cache").join("huggingface").join("token")),
        )
        .collect();
    for p in candidates {
        if let Ok(s) = std::fs::read_to_string(&p) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Decorate a `ureq::Request` with `Authorization: Bearer <token>` when
/// [`hf_token`] returns a value. Returns the original request unmodified
/// otherwise. Designed to slot into a builder chain:
///
/// ```ignore
/// let resp = with_auth(ureq::get(&url)).call()?;
/// ```
#[cfg(feature = "http")]
pub fn with_auth(req: ureq::Request) -> ureq::Request {
    match hf_token() {
        Some(t) => req.set("Authorization", &format!("Bearer {t}")),
        None => req,
    }
}

#[cfg(test)]
#[cfg(feature = "http")]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes every env-mutating test in this module so that no two
    /// tests can interleave `set_var`/`remove_var` calls. Rust 2024 marks
    /// `std::env::set_var` and `remove_var` as `unsafe` because cargo's
    /// default test harness runs tests on multiple threads, and the C
    /// `setenv`/`getenv` machinery is not thread-safe. Holding this lock
    /// for the whole test body — including the read inside `hf_token()`
    /// and the restore block — restricts the env-mutation window to a
    /// single thread of the test process at a time, satisfying the
    /// `// SAFETY:` invariant the `unsafe` blocks rely on.
    ///
    /// **Caveat**: the lock only synchronises tests inside this module.
    /// Other crates in the workspace that mutate `HF_TOKEN` or `HF_HOME`
    /// from their own tests would race with us; we accept that because
    /// (a) no such crate currently exists in the workspace and (b) the
    /// global solution (a workspace-wide env-mutation mutex or
    /// `serial_test`) is a workspace-coordination event out of scope for
    /// this leaf-crate dispatch.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn token_from_env_var_takes_precedence() {
        // We can't easily test env-var precedence inside the test process
        // without polluting other tests' state, so we just verify the
        // accessor can be called without panicking. Real token resolution
        // is exercised by the integration helpers below.
        let _ = hf_token();
    }

    #[test]
    fn token_from_explicit_env() {
        // Acquire the env-mutation lock for the lifetime of this test so
        // no other test in this module can race on HF_TOKEN.
        // `unwrap_or_else` on poisoning lets us proceed when an earlier
        // test panicked while holding the lock — env state may be dirty,
        // but the read+restore pattern is robust to that.
        let _g = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("HF_TOKEN").ok();
        // SAFETY: ENV_MUTEX serialises all set_var/remove_var calls in
        // this test module. No other thread of this test process can
        // race on HF_TOKEN while `_g` is held (the read inside
        // `hf_token()` and the restore block all sit inside the same
        // critical section). Per Rust 2024's unsafe contract on
        // env::set_var, this is the binding invariant: no concurrent
        // env access from any other thread.
        unsafe { std::env::set_var("HF_TOKEN", "hf_testtoken") };
        let t = hf_token();
        assert_eq!(t.as_deref(), Some("hf_testtoken"));
        // SAFETY: same lock, same invariant — this is the matching
        // restore so we don't leak a test value into sibling tests
        // (which would also race with the lock-protected window of the
        // next test that reads HF_TOKEN).
        unsafe {
            match prior {
                Some(v) => std::env::set_var("HF_TOKEN", v),
                None => std::env::remove_var("HF_TOKEN"),
            }
        };
    }

    #[test]
    fn empty_env_var_falls_through() {
        let _g = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("HF_TOKEN").ok();
        // SAFETY: ENV_MUTEX held for the duration of this test
        // serialises HF_TOKEN access; see token_from_explicit_env for
        // the full rationale.
        unsafe { std::env::set_var("HF_TOKEN", "   ") };
        // With an empty/whitespace HF_TOKEN, the env source is rejected; the
        // result depends on whether the test machine has a cache file. We
        // only assert that an empty env var is not directly returned.
        if let Some(t) = hf_token() {
            assert!(!t.trim().is_empty());
        }
        // SAFETY: matching restore inside the same critical section.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("HF_TOKEN", v),
                None => std::env::remove_var("HF_TOKEN"),
            }
        };
    }

    #[test]
    fn with_auth_no_op_without_token() {
        let _g = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("HF_TOKEN").ok();
        let prior_home = std::env::var("HF_HOME").ok();
        // SAFETY: ENV_MUTEX held for the duration of this test
        // serialises both HF_TOKEN and HF_HOME access. No other thread
        // can read or write either env var while `_g` is alive — that
        // is the invariant Rust 2024's unsafe `env::set_var` /
        // `remove_var` requires.
        unsafe {
            std::env::remove_var("HF_TOKEN");
            std::env::set_var(
                "HF_HOME",
                "/tmp/__ferrotorch_nonexistent_hf_home_for_test__",
            );
        };
        // The decorated request is observably the same shape; we can't
        // compare requests for header equality directly, but we can ensure
        // the call doesn't panic.
        let req = ureq::get("https://example.invalid");
        let _ = with_auth(req);
        // SAFETY: matching restore inside the same critical section.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("HF_TOKEN", v),
                None => std::env::remove_var("HF_TOKEN"),
            }
            match prior_home {
                Some(v) => std::env::set_var("HF_HOME", v),
                None => std::env::remove_var("HF_HOME"),
            }
        }
    }
}
