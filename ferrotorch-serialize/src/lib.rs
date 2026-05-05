//! Serialization for ferrotorch models and training checkpoints.
//!
//! Provides a simple JSON+binary format inspired by `SafeTensors`:
//! - **Header**: JSON metadata (tensor names, shapes, dtypes, byte offsets)
//! - **Body**: raw tensor data bytes, concatenated
//!
//! The format is intentionally simple and can be swapped to real `SafeTensors`
//! when a stable Rust crate is available.

#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
#![allow(missing_docs)]
// The pedantic group is intentionally noisy; `clippy::pedantic` is a
// double-edged sword in a parser-heavy crate where literal protocol bytes
// (GGUF magic, pickle opcodes, ONNX wire tags) are necessarily integer
// constants and cast operations. We allow a small set of pedantic lints
// crate-wide where the alternative would be hand-typed numeric helpers
// that obscure the format spec rather than clarify it. Each `#[allow]`
// here is justified by the adjacent comment.
//
// `cast_possible_truncation` / `cast_possible_wrap` / `cast_sign_loss`:
//   The on-disk wire formats this crate parses (GGUF, pickle, ONNX
//   protobuf, safetensors) define explicit `u32`/`u64`/`i32`/`i64`/`usize`
//   widths that legitimately need narrowing casts after bounds checks
//   the parser already performs.
// `cast_precision_loss`:
//   Tensor offsets/sizes are `usize` values that we display in error
//   messages and convert to `f64` for human-readable progress reports;
//   the precision loss is acceptable for those use cases.
// `must_use_candidate` / `missing_errors_doc` / `missing_panics_doc`:
//   The public surface here is documented at the function level; the
//   pedantic lints duplicate that and add noise to the diff.
// `module_name_repetitions`:
//   Public types like `OnnxExportConfig` deliberately mirror module
//   names because they appear in user-facing imports.
// `similar_names`:
//   Format constants (`ONNX_FLOAT` vs `ONNX_DOUBLE`, `ATTR_INTS` vs
//   `ATTR_TYPE_INTS`) match the upstream ONNX wire-format names verbatim
//   and renaming them would lose round-trip clarity.
// `too_many_lines`:
//   Pickle / GGUF parsers naturally have long match arms; the alternative
//   of fragmenting them across helpers loses local reasoning about wire
//   bytes and offsets.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::too_many_lines
)]

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
    OnnxExportConfig, export_from_program, export_ir_graph_to_onnx, export_onnx, ir_graph_to_onnx,
};
pub use pytorch_export::{save_pytorch, validate_checkpoint};
pub use pytorch_import::{
    PickleValue, load_pytorch_state_dict, load_pytorch_state_dict_mmap, parse_pickle,
};
pub use safetensors_io::{
    ShardProgress, load_safetensors, load_safetensors_auto, load_safetensors_mmap,
    load_safetensors_sharded, load_safetensors_sharded_filtered, load_safetensors_sharded_mmap,
    load_safetensors_sharded_with_progress, save_safetensors,
};
pub use state_dict::{load_state_dict, save_state_dict};
