# ferrotorch-serialize

State dict and checkpoint serialization for ferrotorch — SafeTensors, GGUF, ONNX export, and PyTorch import.

## What it provides

- **State dict I/O** -- `save_state_dict`, `load_state_dict` for saving/loading model parameters
- **SafeTensors** -- `save_safetensors`, `load_safetensors` for the SafeTensors format
- **Checkpointing** -- `save_checkpoint`, `load_checkpoint`, `TrainingCheckpoint` for full training state (model + optimizer + epoch)
- **GGUF** -- `load_gguf`, `load_gguf_mmap`, `parse_gguf_bytes`, `dequantize_gguf_tensor`, `load_gguf_state_dict` for loading GGUF-quantized models (Q4_0, Q4_1, Q8_0, Q5_0, Q5_1, F16, F32)
- **ONNX export** -- `export_onnx`, `export_ir_graph_to_onnx`, `ir_graph_to_onnx` with `OnnxExportConfig`; exports via the ferrotorch-jit `IrGraph` and emits opset-compatible `.onnx` files without an external protobuf dependency
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
