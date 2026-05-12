# ferrotorch-bert

BERT-family encoder-only model composition for ferrotorch — assembles a
standard `BertModel` from ferrotorch primitives (token / position /
token-type embeddings, multi-head self-attention, post-norm residual
blocks, GELU intermediate FFN) and adds a `SentenceTransformer` wrapper
that performs mean-pooling over the attention mask followed by optional
L2 normalization for sentence embeddings.

## What it provides

- **`BertConfig`** — model hyperparameters; `HfBertConfig` deserializes
  HuggingFace `config.json` files directly.
- **`BertEmbeddings`** — token + position + token-type embeddings with
  post-embedding LayerNorm + dropout.
- **`BertSelfAttention` / `BertAttention` / `BertSelfOutput`** —
  multi-head self-attention with post-norm residual.
- **`BertIntermediate` / `BertOutput` / `BertLayer`** — GELU FFN +
  post-norm residual; full `BertLayer` block.
- **`BertEncoder` / `BertModel`** — N-layer transformer stack.
- **`SentenceTransformer`** — `BertModel` + attention-mask-aware mean
  pool + optional L2 normalize (`SentenceTransformer::encode`).
- **`load_bert_model` / `load_sentence_transformer`** — SafeTensors
  loaders that consume HuggingFace `transformers` checkpoints and
  return a `DropReport` listing any upstream key the loader
  intentionally did not consume (#1141 silent-drop-bug guard).

## Quick start

```rust
use ferrotorch_bert::{BertConfig, SentenceTransformer, load_sentence_transformer};

let cfg = BertConfig::all_mini_lm_l6_v2();
let mut model: SentenceTransformer<f32> = SentenceTransformer::new(cfg, /* normalize */ true)?;
let drop_report = load_sentence_transformer(&mut model, "/path/to/all-MiniLM-L6-v2")?;
assert!(drop_report.is_empty(), "loader dropped upstream keys: {drop_report:?}");

let input_ids: Tensor<i64>      = /* tokenized [B, S] */;
let attention_mask: Tensor<i64> = /* [B, S] */;
let token_type_ids: Tensor<i64> = /* [B, S] */;
let embedding = model.encode(&input_ids, &attention_mask, &token_type_ids)?;
// embedding shape: [B, hidden_size]; L2-normalized when `normalize=true`
```

## Real-artifact parity

The first pinned checkpoint is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(22M parameters, 6 layers, 384 hidden, Apache 2.0), mirrored byte-for-byte
to `ferrotorch/all-MiniLM-L6-v2` and registered in
[`ferrotorch-hub`](../ferrotorch-hub).

`examples/text_embedding_dump.rs` + `scripts/verify_text_embedding_inference.py`
verify cosine similarity ≥ 0.999 and max-abs diff ≤ 0.01 against
upstream
`sentence_transformers.SentenceTransformer.encode(..., normalize_embeddings=True)`
output (Phase B.1 of real-artifact-driven development; #1148).

## Part of ferrotorch

This crate is one component of the
[ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
