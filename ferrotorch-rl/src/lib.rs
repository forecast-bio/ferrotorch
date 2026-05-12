// Crate-level lint baseline. Mirrors the ferrotorch-bert / ferrotorch-graph
// posture: deny correctness / idiom / Debug / docs problems; warn pedantic
// stylistic issues. Specific pedantic lints are allowed crate-wide where the
// lint is consistently wrong for ML / numeric kernel code.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Casts: dimension math (`as usize`, `as f32`, `as u32`) is intrinsic to
// tensor indexing.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// Builder-style accessors don't all need `#[must_use]`.
#![allow(clippy::must_use_candidate)]
// `MLP`, `PPO`, `sb3`, `bf16` flagged as missing backticks even inside fences.
#![allow(clippy::doc_markdown)]
// `needless_pass_by_value` would force `&MlpPolicyConfig` everywhere.
#![allow(clippy::needless_pass_by_value)]
// `unnecessary_wraps` flags `Result`-returning helpers that today always
// succeed but are part of an extensible API surface.
#![allow(clippy::unnecessary_wraps)]
// `format!("x={}", x)` vs `format!("x={x}")` churn.
#![allow(clippy::uninlined_format_args)]

//! Reinforcement-learning policy composition for ferrotorch.
//!
//! Phase D.2 of real-artifact-driven development: the
//! `stable-baselines3` `ActorCriticPolicy` (a.k.a. `MlpPolicy`) for
//! discrete-action environments, mirrored byte-for-byte from the
//! pinned `sb3/ppo-CartPole-v1` checkpoint and verified for forward-
//! pass parity against `stable-baselines3` via
//! `scripts/verify_rl_inference.py`.
//!
//! # Architecture (matches sb3 `MlpPolicy` defaults for CartPole-v1)
//!
//! ```text
//! MlpPolicy
//! ├── features_extractor: FlattenExtractor (identity on 1-D obs)
//! ├── mlp_extractor: MlpExtractor
//! │   ├── policy_net: Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh
//! │   └── value_net:  Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh
//! ├── action_net: Linear(64 → n_actions)        ← Categorical logits
//! └── value_net:  Linear(64 → 1)                ← scalar state value
//! ```
//!
//! `share_features_extractor=True` in sb3's default, but the policy /
//! value trunks have *separate* `Linear` weights — the "shared"
//! features extractor is the flatten layer, not the MLP. For a 1-D
//! observation (e.g. CartPole's `[cart_pos, cart_vel, pole_angle,
//! pole_angvel]`) the flatten is a no-op, so the loader treats it as
//! identity.
//!
//! # Loading real weights
//!
//! [`load_ppo_policy`] accepts a path to `model.safetensors` (the
//! ferrotorch mirror of an sb3 zip checkpoint) plus dimensions
//! `(obs_dim, hidden, n_actions)` and returns a populated
//! [`MlpPolicy`] plus a [`DropReport`] documenting any upstream key
//! that was intentionally not consumed. For the canonical
//! `sb3/ppo-CartPole-v1` pin the report should be empty; a non-empty
//! report on a clean pin signals a state-dict-drop bug (#1141 class).
//!
//! # On `log_std`
//!
//! sb3's `ActorCriticPolicy` only emits a `log_std` parameter for
//! continuous-action policies (DiagGaussianDistribution). For the
//! discrete `Categorical` distribution used by CartPole-v1 there is no
//! `log_std`, so the state dict has exactly 12 keys (4 for the policy
//! trunk × 2, 2 for action_net, 2 for value_net).

pub mod mlp_policy;
pub mod safetensors_loader;

pub use mlp_policy::{ActionNet, MlpExtractor, MlpPolicy, MlpPolicyConfig, ValueHead};
pub use safetensors_loader::{DropReport, load_ppo_policy};
