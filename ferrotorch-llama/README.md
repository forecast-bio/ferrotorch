# ferrotorch-llama

Llama 3 / Meta LLaMA model composition and weight loading for ferrotorch.

## What it provides

- **`LlamaForCausalLM`** -- full causal LM: decoder stack + lm_head projection
- **`LlamaModel`** -- embedding + N decoder layers + final RMSNorm
- **`LlamaDecoderLayer`** -- grouped-query attention (GQA) with RoPE + SwiGLU MLP
- **`LlamaAttention`** -- multi-head attention with rotary position embeddings
- **`LlamaMLP`** -- SwiGLU gate/up/down projections
- **`LlamaConfig`** -- model hyperparameters (8B, 70B, etc.)
- **`LlamaGpuInferencer`** -- bf16 GPU inference on CUDA (with `cuda` feature)

Weight loading accepts HuggingFace transformers naming convention via `load_hf_state_dict`, working directly with SafeTensors checkpoints.

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda`  | no      | Enables `LlamaGpuInferencer` for bf16 inference via ferrotorch-gpu |

## Quick start

```rust
use ferrotorch_llama::{LlamaForCausalLM, LlamaConfig};
use ferrotorch_serialize::load_safetensors_sharded;

let config = LlamaConfig::llama3_8b();
let mut model = LlamaForCausalLM::new(&config)?;
let state = load_safetensors_sharded("/path/to/Meta-Llama-3-8B")?;
model.load_hf_state_dict(&state)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
