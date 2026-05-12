# ferrotorch-grammar

Constrained-decoding grammar processors for ferrotorch — turn a JSON
Schema into a per-step token-allow mask so an LLM can only emit text
that satisfies the schema.

Originally lived inside `ferrotorch-llama`; extracted into its own
crate so non-Llama models (e.g. `ferrotorch-bert` decoders, future
encoder-decoder seq2seq models) can reuse the machinery without
pulling in the entire Llama 3 stack (#1120).

## What it provides

- **`Schema`** — internal AST for a (subset-of) JSON Schema document.
  Parsed once via `Schema::from_json_schema(&serde_json::Value)`.
- **`JsonGrammar`** — state machine tracking the partially-emitted
  JSON value. Knows which characters are legal at the current
  position (e.g. after `{` only `"` or `}` are allowed; mid-string
  only non-control chars + escape sequences).
- **`JsonSchemaProcessor`** — public type that wraps a tokenizer
  vocabulary `Vec<String>` and produces a `TokenMask` (one `u32` flag
  per token) on every `compute_mask()` call. Advance with
  `step_token(token_id)` after each sample.
- **`TokenMask`** — `Vec<u32>` of 0/1 flags, one per vocab token.
  Apply to logits via `ferrotorch_cubecl::apply_token_mask_to_gpu`
  (GPU) or simple `if mask[i] == 0 { logits[i] = -inf }` (CPU).

## Quick start

```rust
use ferrotorch_grammar::JsonSchemaProcessor;
use serde_json::json;

let schema = json!({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"}
    },
    "required": ["name", "age"]
});
let vocab: Vec<String> = vec![/* tokenizer vocab */];

let mut proc = JsonSchemaProcessor::new(&schema, vocab)?;

loop {
    let mask = proc.compute_mask();
    let logits = model.forward(/* ... */);
    let token_id = sample_with_mask(logits, &mask);
    if proc.is_complete() { break; }
    proc.step_token(token_id)?;
}
```

## Supported JSON Schema subset

| Keyword            | Status |
|--------------------|--------|
| `type: object`     | yes (with `properties`, `required`) |
| `type: array`      | yes (with `items`)                  |
| `type: string`     | yes (with `enum`)                   |
| `type: integer`    | yes                                 |
| `type: number`     | yes                                 |
| `type: boolean`    | yes                                 |
| `type: null`       | yes                                 |
| `nullable: true`   | yes                                 |
| `oneOf` / `anyOf`  | not yet                             |
| `$ref` / `$defs`   | not yet                             |
| `pattern`          | not yet                             |
| `minLength` / etc. | not yet                             |

Schemas using unsupported keywords return `GrammarError::Schema`.

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda`  | no      | Enables `gpu_dispatch::{PackedVocab, compute_mask_gpu}` for masking that runs on the GPU via ferrotorch-cubecl |

## Part of ferrotorch

This crate is one component of the
[ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
