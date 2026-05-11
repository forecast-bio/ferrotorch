# ferrotorch-whisper

Whisper-family audio encoder model composition for ferrotorch. Composes
`Conv1d`, sinusoidal positional embeddings, pre-norm transformer encoder
layers (Q/K/V/out_proj, GELU FFN), and final LayerNorm from
`ferrotorch-nn` primitives into the encoder stack of OpenAI's Whisper
model.

The crate also ships `audio::log_mel_spectrogram`, a pure-Rust port of
`transformers.WhisperFeatureExtractor` that turns 16 kHz mono audio into
the `[1, 80, 3000]` log-mel tensor the encoder consumes.

Phase B.2 of real-artifact-driven development: ENCODER-ONLY parity
against `openai/whisper-tiny` via the mirrored
`ferrotorch/whisper-tiny-encoder` HF mirror. Decoder generation
(cross-attention, beam search, kv-cache) is intentionally out of scope.
