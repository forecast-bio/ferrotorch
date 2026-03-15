# ferrotorch-serialize

State dict and checkpoint serialization for ferrotorch.

## What it provides

- **State dict I/O** -- `save_state_dict`, `load_state_dict` for saving/loading model parameters
- **SafeTensors** -- `save_safetensors`, `load_safetensors` for the SafeTensors format
- **Checkpointing** -- `save_checkpoint`, `load_checkpoint`, `TrainingCheckpoint` for full training state (model + optimizer + epoch)
- **ONNX export** -- `export_onnx`, `export_ir_graph_to_onnx`, `ir_graph_to_onnx` with `OnnxExportConfig`
- **PyTorch import** -- `load_pytorch_state_dict`, `parse_pickle` for loading PyTorch `.pt`/`.pth` files

## Quick start

```rust
use ferrotorch_serialize::{save_state_dict, load_state_dict};

// Save model weights
save_state_dict(&model, "model.safetensors")?;

// Load into a new model
load_state_dict(&mut new_model, "model.safetensors")?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
