# ferrotorch-whisper

Whisper-family audio encoder model composition for ferrotorch. Composes
`Conv1d`, sinusoidal positional embeddings, pre-norm transformer encoder
layers (Q/K/V/out_proj, GELU FFN), and final LayerNorm from
[`ferrotorch-nn`](../ferrotorch-nn) primitives into the encoder stack of
OpenAI's Whisper model.

## What it provides

- **`WhisperConfig` / `HfWhisperConfig`** — model hyperparameters;
  `HfWhisperConfig` deserializes HuggingFace `config.json` directly.
- **`WhisperConvStem`** — the 2 × `Conv1d` + GELU front-end that maps
  `[B, 80, 3000]` log-mels to `[B, d_model, 1500]`.
- **`WhisperEncoderLayer`** — pre-norm self-attention + GELU FFN block
  matching `WhisperEncoderLayer` in HuggingFace `transformers`.
- **`WhisperEncoder`** — full encoder stack (`conv stem` + sinusoidal
  positional embeddings + N encoder layers + final LayerNorm).
- **`audio::log_mel_spectrogram`** — pure-Rust port of
  `transformers.WhisperFeatureExtractor` that turns 16 kHz mono audio
  into the `[1, 80, 3000]` log-mel tensor the encoder consumes
  (`SAMPLE_RATE = 16_000`, `N_MELS = 80`, `N_FRAMES = 3000`).
- **`load_whisper_encoder`** — SafeTensors loader for HuggingFace
  Whisper checkpoints; returns a `DropReport` (#1141 silent-drop-bug
  guard).

## Quick start

```rust
use ferrotorch_whisper::{WhisperConfig, WhisperEncoder, audio};

let cfg = WhisperConfig::tiny();
let mut encoder: WhisperEncoder<f32> = WhisperEncoder::new(cfg)?;
let _drop = ferrotorch_whisper::load_whisper_encoder(&mut encoder, "/path/to/whisper-tiny")?;

// 16 kHz mono PCM -> [1, 80, 3000] log-mel
let pcm: Vec<f32> = read_wav_16k_mono("clip.wav")?;
let mel: Tensor<f32> = audio::log_mel_spectrogram(&pcm)?;

// [1, 80, 3000] -> [1, 1500, d_model] encoder hidden states
let hidden = encoder.forward_from_mel(&mel)?;
```

## Real-artifact parity

Phase B.2 of real-artifact-driven development: ENCODER-ONLY parity
against [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
via the `ferrotorch/whisper-tiny-encoder` HF mirror (#1149).

Decoder generation (cross-attention, beam search, kv-cache) is
intentionally out of scope. Add a separate crate or extend this one in a
follow-up; the encoder hidden states emitted here are the standard input
contract for downstream Whisper decoders.

## Part of ferrotorch

This crate is one component of the
[ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
