//! Serialization for ferrotorch models and training checkpoints.
//!
//! Provides a simple JSON+binary format inspired by SafeTensors:
//! - **Header**: JSON metadata (tensor names, shapes, dtypes, byte offsets)
//! - **Body**: raw tensor data bytes, concatenated
//!
//! The format is intentionally simple and can be swapped to real SafeTensors
//! when a stable Rust crate is available.

pub mod checkpoint;
pub mod gguf;
pub mod onnx_export;
pub mod pytorch_export;
pub mod pytorch_import;
pub mod safetensors_io;
pub mod state_dict;

pub use checkpoint::{AsyncCheckpointer, TrainingCheckpoint, load_checkpoint, save_checkpoint};
pub use gguf::{
    GgmlType, GgufFile, GgufMetadata, GgufTensorInfo, GgufValue, dequantize_gguf_tensor, load_gguf,
    load_gguf_mmap, load_gguf_state_dict, load_gguf_state_dict_mmap, parse_gguf_bytes,
};
pub use onnx_export::{
    OnnxExportConfig, export_from_program, export_ir_graph_to_onnx, export_onnx,
    ir_graph_to_onnx,
};
pub use pytorch_export::{save_pytorch, validate_checkpoint};
pub use pytorch_import::{PickleValue, load_pytorch_state_dict, parse_pickle};
pub use safetensors_io::{
    ShardProgress, load_safetensors, load_safetensors_auto, load_safetensors_mmap,
    load_safetensors_sharded, load_safetensors_sharded_filtered, load_safetensors_sharded_mmap,
    load_safetensors_sharded_with_progress, save_safetensors,
};
pub use state_dict::{load_state_dict, save_state_dict};
