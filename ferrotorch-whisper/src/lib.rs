// Crate-level lint baseline. Mirrors the ferrotorch-bert posture:
// deny correctness / idiom / Debug / docs problems; warn pedantic
// stylistic issues. Specific pedantic lints are allowed crate-wide
// where the lint is consistently wrong for ML/numeric kernel code.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Casts: dimension math (`as usize`, `as f32`, `as u32`) is intrinsic
// to tensor indexing — every kernel call would otherwise need a
// per-call allow.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// Builder-style accessors don't all need `#[must_use]`.
#![allow(clippy::must_use_candidate)]
// Identifiers like `bf16`, `f32`, `LayerNorm`, `Whisper`, `STFT` are
// flagged as missing backticks even when they appear in code-fenced
// text.
#![allow(clippy::doc_markdown)]
// `needless_pass_by_value` would force `&WhisperConfig` signatures
// throughout, hiding intent in the API.
#![allow(clippy::needless_pass_by_value)]
// `unnecessary_wraps` flags `Result`-returning helpers that today
// always succeed but are part of an extensible API surface.
#![allow(clippy::unnecessary_wraps)]
// `uninlined_format_args` flags `format!("x={}", x)` vs
// `format!("x={x}")`. Both are equally clear; the fixup churn is high.
#![allow(clippy::uninlined_format_args)]
// `many_single_char_names` flags conventional ML kernel locals
// (`q`, `k`, `v`, `h`).
#![allow(clippy::many_single_char_names)]
// `similar_names` flags variable pairs that are intentionally similar
// (e.g. `q2` / `q_h`).
#![allow(clippy::similar_names)]
// `module_name_repetitions`: every type starts with `Whisper`
// (matching the HF naming) — the lint would force renames that lose
// the upstream-1:1 mapping.
#![allow(clippy::module_name_repetitions)]

//! Whisper-family audio encoder model composition for ferrotorch.
//!
//! Assembles the encoder half of OpenAI's Whisper model from ferrotorch
//! primitives:
//!
//! ```text
//! WhisperEncoder
//! ├── WhisperConvStem
//! │   ├── Conv1d (conv1: num_mel_bins → d_model, k=3, stride=1, pad=1, bias)
//! │   └── Conv1d (conv2: d_model → d_model,     k=3, stride=2, pad=1, bias)
//! ├── embed_positions (sinusoidal, loaded from state-dict as a parameter)
//! └── WhisperEncoderLayer × N
//!     ├── LayerNorm  (self_attn_layer_norm)                  ← PRE-NORM
//!     ├── WhisperEncoderSelfAttention
//!     │   ├── Linear q_proj    [d_model, d_model] (bias)
//!     │   ├── Linear k_proj    [d_model, d_model] (NO bias)
//!     │   ├── Linear v_proj    [d_model, d_model] (bias)
//!     │   └── Linear out_proj  [d_model, d_model] (bias)
//!     ├── LayerNorm  (final_layer_norm)                      ← PRE-NORM
//!     ├── Linear fc1 [d_model, encoder_ffn_dim] (bias) + GELU
//!     └── Linear fc2 [encoder_ffn_dim, d_model] (bias)
//! └── LayerNorm (layer_norm — final encoder LayerNorm)
//! ```
//!
//! # Audio preprocessing
//!
//! [`audio::log_mel_spectrogram`] turns 16 kHz mono `f32` PCM into the
//! `[1, 80, 3000]` log-mel tensor the encoder consumes. The 80-bin
//! filter bank is shipped as the embedded binary asset
//! `assets/mel_filters_80x201.bin`, byte-for-byte equal to
//! `WhisperFeatureExtractor.mel_filters.T`, so any drift between this
//! module and the reference is in the STFT / log / clip / normalize
//! pipeline — never in the mel scale.
//!
//! # Loading real weights
//!
//! [`WhisperEncoder::load_hf_state_dict`] accepts a `StateDict` whose
//! keys use the HuggingFace `WhisperModel` naming convention. It
//! filters out non-encoder keys (decoder / `proj_out` / etc.) and
//! returns a [`encoder::DropReport`] documenting every drop so the pin
//! script can confirm no encoder key was silently lost. Combined with
//! `ferrotorch_serialize::load_safetensors` and the
//! [`load_whisper_encoder`] helper this gives a direct path from a
//! downloaded `openai/whisper-tiny` checkpoint to an encoder ready to
//! produce `[1, 1500, 384]` hidden states.
//!
//! # Out of scope
//!
//! The decoder (cross-attention, kv-cache, beam search) is intentionally
//! not implemented in this crate. Phase B.2 of real-artifact-driven
//! development is encoder-only.

pub mod attention;
pub mod audio;
pub mod config;
pub mod encoder;
pub mod layer;
pub mod safetensors_loader;

pub use attention::WhisperEncoderSelfAttention;
pub use audio::{log_mel_spectrogram, N_FRAMES, N_MELS, SAMPLE_RATE};
pub use config::{HfWhisperConfig, WhisperConfig};
pub use encoder::{DropReport, WhisperConvStem, WhisperEncoder};
pub use layer::WhisperEncoderLayer;
pub use safetensors_loader::load_whisper_encoder;
