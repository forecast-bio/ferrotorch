# ferrotorch-tokenize

Text tokenization wrapper around the HuggingFace `tokenizers` crate for ferrotorch.

## What it provides

- **`load_tokenizer`** -- load a `tokenizer.json` file (BPE, WordPiece, Unigram)
- **`encode` / `encode_batch`** -- text to token ids, with optional special tokens
- **`decode`** -- token ids back to text
- **`vocab_size`** -- vocabulary size including added/special tokens
- **`token_to_id` / `id_to_token`** -- single-token lookups

## Quick start

```rust
use ferrotorch_tokenize::{load_tokenizer, encode, decode};

let tok = load_tokenizer("/path/to/tokenizer.json")?;
let ids = encode(&tok, "Hello, world!", true)?;
let text = decode(&tok, &ids, false)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
