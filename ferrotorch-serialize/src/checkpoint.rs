//! Training checkpoint save/load — bundles model state, optimizer state,
//! and training metadata into a single file.
//!
//! ## File format
//!
//! A checkpoint file is a concatenation of three sections, each preceded by
//! an 8-byte little-endian length prefix:
//!
//! ```text
//! [8 bytes: metadata JSON length]
//! [metadata JSON bytes]
//! [8 bytes: optimizer state JSON length]
//! [optimizer state JSON bytes]
//! [8 bytes: state dict binary length]
//! [state dict binary bytes (same format as state_dict::save)]
//! ```

#[cfg(not(target_endian = "little"))]
compile_error!(
    "ferrotorch checkpoint serialization assumes little-endian byte order. \
     Big-endian platforms are not supported."
);

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::StateDict;
use ferrotorch_optim::OptimizerState;

/// A complete training checkpoint for resuming training.
///
/// Marked `#[non_exhaustive]` so the struct can grow new fields (e.g. RNG
/// state, scheduler state, framework version) without forcing every
/// downstream consumer to update their pattern matches or struct
/// literals. Use [`TrainingCheckpoint::new`] to construct values; field
/// access continues to work as before.
#[derive(Debug)]
#[non_exhaustive]
pub struct TrainingCheckpoint<T: Float> {
    /// The model's parameter state dict.
    pub model_state: StateDict<T>,
    /// The optimizer's internal state (momentum buffers, etc.).
    pub optimizer_state: OptimizerState,
    /// The epoch at which training was paused.
    pub epoch: usize,
    /// The global step count.
    pub step: usize,
}

impl<T: Float> TrainingCheckpoint<T> {
    /// Construct a new checkpoint from its mandatory components.
    ///
    /// Required because the struct is `#[non_exhaustive]` and so cannot be
    /// built with struct-literal syntax outside this crate.
    pub fn new(
        model_state: StateDict<T>,
        optimizer_state: OptimizerState,
        epoch: usize,
        step: usize,
    ) -> Self {
        Self {
            model_state,
            optimizer_state,
            epoch,
            step,
        }
    }
}

/// Save a training checkpoint to a file.
pub fn save_checkpoint<T: Float>(
    checkpoint: &TrainingCheckpoint<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to create checkpoint file {}: {e}", path.display()),
    })?;

    // Section 1: metadata JSON (epoch, step).
    let metadata_json = format!(
        r#"{{"epoch":{},"step":{}}}"#,
        checkpoint.epoch, checkpoint.step
    );
    write_section(&mut file, metadata_json.as_bytes())?;

    // Section 2: optimizer state JSON.
    let opt_json = serialize_optimizer_state(&checkpoint.optimizer_state);
    write_section(&mut file, opt_json.as_bytes())?;

    // Section 3: state dict binary (reuse state_dict::save format in memory).
    let sd_bytes = serialize_state_dict_to_bytes(&checkpoint.model_state)?;
    write_section(&mut file, &sd_bytes)?;

    file.flush().map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("flush error: {e}"),
    })?;

    Ok(())
}

/// Load a training checkpoint from a file.
pub fn load_checkpoint<T: Float>(
    path: impl AsRef<Path>,
) -> FerrotorchResult<TrainingCheckpoint<T>> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open checkpoint file {}: {e}", path.display()),
    })?;

    // Section 1: metadata.
    let meta_bytes = read_section(&mut file)?;
    let meta_str =
        std::str::from_utf8(&meta_bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("invalid UTF-8 in metadata: {e}"),
        })?;
    let (epoch, step) = parse_metadata(meta_str)?;

    // Section 2: optimizer state.
    let opt_bytes = read_section(&mut file)?;
    let opt_str =
        std::str::from_utf8(&opt_bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("invalid UTF-8 in optimizer state: {e}"),
        })?;
    let optimizer_state = deserialize_optimizer_state(opt_str)?;

    // Section 3: state dict.
    let sd_bytes = read_section(&mut file)?;
    let model_state = deserialize_state_dict_from_bytes::<T>(&sd_bytes)?;

    Ok(TrainingCheckpoint {
        model_state,
        optimizer_state,
        epoch,
        step,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Write a length-prefixed section.
fn write_section(file: &mut std::fs::File, data: &[u8]) -> FerrotorchResult<()> {
    let len = data.len() as u64;
    file.write_all(&len.to_le_bytes())
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("write error: {e}"),
        })?;
    file.write_all(data)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("write error: {e}"),
        })?;
    Ok(())
}

/// Read a length-prefixed section.
fn read_section(file: &mut std::fs::File) -> FerrotorchResult<Vec<u8>> {
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("read error (section length): {e}"),
        })?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("read error (section body, expected {len} bytes): {e}"),
        })?;
    Ok(buf)
}

/// Parse the metadata JSON: `{"epoch":N,"step":N}`.
fn parse_metadata(s: &str) -> FerrotorchResult<(usize, usize)> {
    let extract = |key: &str| -> FerrotorchResult<usize> {
        let pattern = format!(r#""{key}":"#);
        let start = s
            .find(&pattern)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("missing key \"{key}\" in metadata: {s}"),
            })?
            + pattern.len();
        let rest = &s[start..];
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        rest[..end]
            .trim()
            .parse::<usize>()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to parse \"{key}\": {e}"),
            })
    };
    Ok((extract("epoch")?, extract("step")?))
}

/// Serialize an `OptimizerState` to JSON without serde.
///
/// Format: `{"param_name":{"state_key":[v1,v2,...], ...}, ...}`
fn serialize_optimizer_state(state: &OptimizerState) -> String {
    if state.is_empty() {
        return "{}".to_string();
    }

    let mut outer_parts: Vec<String> = Vec::new();
    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();

    for key in keys {
        let inner_map = &state[key];
        let mut inner_parts: Vec<String> = Vec::new();
        let mut inner_keys: Vec<&String> = inner_map.keys().collect();
        inner_keys.sort();

        for ik in inner_keys {
            let values = &inner_map[ik];
            let vals_str = values
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(",");
            inner_parts.push(format!(r#""{ik}":[{vals_str}]"#));
        }
        outer_parts.push(format!(r#""{}":{{{}}}"#, key, inner_parts.join(",")));
    }

    format!("{{{}}}", outer_parts.join(","))
}

/// Deserialize an `OptimizerState` from JSON without serde.
fn deserialize_optimizer_state(s: &str) -> FerrotorchResult<OptimizerState> {
    let s = s.trim();
    if s == "{}" {
        return Ok(HashMap::new());
    }

    // Strip outer braces.
    if !s.starts_with('{') || !s.ends_with('}') {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("optimizer state is not a JSON object: {s}"),
        });
    }
    let inner = &s[1..s.len() - 1];

    let mut result: OptimizerState = HashMap::new();

    // Split on top-level entries. We need to respect nested braces.
    let entries = split_top_level(inner, '{', '}');

    for entry in entries {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        // "key":{...}
        let colon_pos = entry
            .find(":{")
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("invalid optimizer state entry: {entry}"),
            })?;
        let key = entry[..colon_pos].trim().trim_matches('"').to_string();
        let value_str = &entry[colon_pos + 1..];

        // Parse inner map: {"state_key":[v1,v2,...], ...}
        let inner_map = parse_inner_opt_state(value_str)?;
        result.insert(key, inner_map);
    }

    Ok(result)
}

/// Split a string on commas, but only at top-level (respecting braces).
fn split_top_level(s: &str, open: char, close: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    let mut in_string = false;
    let mut prev_char = '\0';

    for ch in s.chars() {
        if ch == '"' && prev_char != '\\' {
            in_string = !in_string;
        }
        if !in_string {
            if ch == open || ch == '[' {
                depth += 1;
            } else if ch == close || ch == ']' {
                depth -= 1;
            } else if ch == ',' && depth == 0 {
                parts.push(std::mem::take(&mut current));
                prev_char = ch;
                continue;
            }
        }
        current.push(ch);
        prev_char = ch;
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Parse the inner optimizer state map: `{"key":[v1,v2,...], ...}`.
fn parse_inner_opt_state(s: &str) -> FerrotorchResult<HashMap<String, Vec<f64>>> {
    let s = s.trim();
    if s == "{}" {
        return Ok(HashMap::new());
    }
    if !s.starts_with('{') || !s.ends_with('}') {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("invalid inner optimizer state: {s}"),
        });
    }
    let inner = &s[1..s.len() - 1];

    let mut result = HashMap::new();
    let entries = split_top_level(inner, '[', ']');

    for entry in entries {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        // "key":[v1,v2,...]
        let colon_pos = entry
            .find(":[")
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("invalid inner state entry: {entry}"),
            })?;
        let key = entry[..colon_pos].trim().trim_matches('"').to_string();
        let arr_str = &entry[colon_pos + 1..];

        // Parse array: [v1,v2,...]
        let arr_inner = arr_str.trim().trim_start_matches('[').trim_end_matches(']');
        let values: Vec<f64> = if arr_inner.trim().is_empty() {
            Vec::new()
        } else {
            arr_inner
                .split(',')
                .map(|v| {
                    v.trim()
                        .parse::<f64>()
                        .map_err(|e| FerrotorchError::InvalidArgument {
                            message: format!("failed to parse optimizer value \"{v}\": {e}"),
                        })
                })
                .collect::<FerrotorchResult<Vec<f64>>>()?
        };

        result.insert(key, values);
    }

    Ok(result)
}

/// Serialize a `StateDict` to the `state_dict` binary format in memory.
fn serialize_state_dict_to_bytes<T: Float>(state: &StateDict<T>) -> FerrotorchResult<Vec<u8>> {
    // Write to a temp buffer using state_dict::save_state_dict, but into
    // memory. We replicate the logic inline to avoid temp files.
    let mut buf = Vec::new();

    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();

    let elem_size = std::mem::size_of::<T>();
    let mut byte_offset: usize = 0;

    let dtype = crate::state_dict::dtype_tag::<T>();

    // Header.
    for key in &keys {
        let tensor = &state[*key];
        let numel = tensor.numel();
        let byte_length = numel * elem_size;

        let shape_str = format!(
            "[{}]",
            tensor
                .shape()
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let line = format!(
            r#"{{"name":"{key}","shape":{shape_str},"dtype":"{dtype}","byte_offset":{byte_offset},"byte_length":{byte_length}}}"#,
        );
        buf.extend_from_slice(line.as_bytes());
        buf.push(b'\n');
        byte_offset += byte_length;
    }

    // Separator.
    buf.extend_from_slice(b"\n---\n");

    // Body.
    for key in &keys {
        let tensor = &state[*key];
        let data = tensor.data()?;
        // SAFETY: `data: &[T]` where `T: Float` is one of f32/f64/bf16 — all
        // `Copy` POD types whose in-memory representation is a byte sequence
        // with no padding and no `Drop` semantics, so reinterpreting the same
        // bytes as `&[u8]` is sound. The pointer is valid for
        // `size_of_val(data)` bytes (that is the slice's exact byte length),
        // and the returned `&[u8]` borrows from `data`, inheriting its
        // lifetime, so no dangling reference is produced. The crate-level
        // `compile_error!` at top-of-file restricts this code to
        // little-endian targets, matching the on-disk byte order of the
        // state-dict format.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        buf.extend_from_slice(byte_slice);
    }

    Ok(buf)
}

/// Deserialize a `StateDict` from the `state_dict` binary format in memory.
fn deserialize_state_dict_from_bytes<T: Float>(bytes: &[u8]) -> FerrotorchResult<StateDict<T>> {
    // Find the separator.
    let sep = b"\n---\n";
    let sep_pos = bytes
        .windows(sep.len())
        .position(|w| w == sep)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "missing separator in state dict section".into(),
        })?;

    let header_bytes = &bytes[..sep_pos];
    let body = &bytes[sep_pos + sep.len()..];

    let header_str =
        std::str::from_utf8(header_bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("invalid UTF-8 in state dict header: {e}"),
        })?;

    let expected_dtype = crate::state_dict::dtype_tag::<T>();
    let elem_size = std::mem::size_of::<T>();
    let mut state: StateDict<T> = HashMap::new();

    for line in header_str.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let meta = crate::state_dict::parse_meta_line(trimmed)?;

        if meta.dtype != expected_dtype {
            return Err(FerrotorchError::DtypeMismatch {
                expected: expected_dtype.to_string(),
                got: meta.dtype.clone(),
            });
        }

        let end = meta.byte_offset + meta.byte_length;
        if end > body.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor \"{}\" requires bytes [{}..{}] but body has {} bytes",
                    meta.name,
                    meta.byte_offset,
                    end,
                    body.len()
                ),
            });
        }

        let numel = meta.byte_length / elem_size;
        let expected_numel: usize = meta.shape.iter().product();
        if expected_numel != numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "tensor \"{}\" shape {:?} implies {} elements but data has {}",
                    meta.name, meta.shape, expected_numel, numel
                ),
            });
        }

        let byte_slice = &body[meta.byte_offset..end];
        let mut data: Vec<T> = Vec::with_capacity(numel);
        for chunk in byte_slice.chunks_exact(elem_size) {
            // Safe byte-to-float conversion using from_le_bytes; both
            // `try_into` and the numeric `cast` are fallible and propagate
            // a structured `FerrotorchError` instead of panicking on
            // malformed input. `chunk.try_into()` can only fail if
            // `chunks_exact` lied (it doesn't), but we propagate cleanly
            // anyway. `cast::<f64, T>` can fail for narrow `Float`
            // implementations where the source value isn't representable.
            let value: T = match elem_size {
                4 => {
                    let arr: [u8; 4] =
                        chunk
                            .try_into()
                            .map_err(|e| FerrotorchError::InvalidArgument {
                                message: format!(
                                    "malformed checkpoint chunk for tensor \"{}\": {e}",
                                    meta.name
                                ),
                            })?;
                    cast::<f32, T>(f32::from_le_bytes(arr))?
                }
                8 => {
                    let arr: [u8; 8] =
                        chunk
                            .try_into()
                            .map_err(|e| FerrotorchError::InvalidArgument {
                                message: format!(
                                    "malformed checkpoint chunk for tensor \"{}\": {e}",
                                    meta.name
                                ),
                            })?;
                    cast::<f64, T>(f64::from_le_bytes(arr))?
                }
                other => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unsupported element size {other} for tensor \"{}\" \
                             (checkpoint format only supports f32/f64)",
                            meta.name
                        ),
                    });
                }
            };
            data.push(value);
        }

        let storage = ferrotorch_core::TensorStorage::cpu(data);
        let tensor = ferrotorch_core::Tensor::from_storage(storage, meta.shape.clone(), false)?;
        state.insert(meta.name.clone(), tensor);
    }

    Ok(state)
}

// ---------------------------------------------------------------------------
// Async checkpointing
// ---------------------------------------------------------------------------

/// Asynchronous checkpoint saver that writes checkpoints on a background thread.
///
/// # Limitations
///
/// - **Only supports `f32` checkpoints.** The background thread serializes
///   `TrainingCheckpoint<f32>`. Supporting generic `T: Float` would require
///   boxing the checkpoint or a trait-object approach; for now f32 covers the
///   dominant training dtype.
///
/// # Panic handling
///
/// If the background thread panics, the `in_flight` flag is reset via
/// `catch_unwind` so subsequent saves are not blocked.
pub struct AsyncCheckpointer {
    /// Whether a save is currently in progress.
    in_flight: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Join handle for the background thread.
    handle: Option<std::thread::JoinHandle<FerrotorchResult<()>>>,
}

impl std::fmt::Debug for AsyncCheckpointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncCheckpointer")
            .field(
                "in_flight",
                &self.in_flight.load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("has_pending_save", &self.handle.is_some())
            .finish()
    }
}

impl AsyncCheckpointer {
    /// Create a new async checkpointer.
    pub fn new() -> Self {
        Self {
            in_flight: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            handle: None,
        }
    }

    /// Save a checkpoint asynchronously on a background thread.
    ///
    /// If a previous save is still in progress, this call blocks until it
    /// finishes before starting the new save.
    pub fn save(
        &mut self,
        checkpoint: TrainingCheckpoint<f32>,
        path: impl AsRef<Path> + Send + 'static,
    ) -> FerrotorchResult<()> {
        // Wait for any previous save to finish.
        self.wait()?;

        let in_flight = self.in_flight.clone();
        in_flight.store(true, std::sync::atomic::Ordering::SeqCst);

        let handle = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                save_checkpoint(&checkpoint, path)
            }));

            // Always reset in_flight, even on panic.
            in_flight.store(false, std::sync::atomic::Ordering::SeqCst);

            match result {
                Ok(inner) => inner,
                Err(panic_payload) => {
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic in async checkpoint".to_string()
                    };
                    Err(FerrotorchError::InvalidArgument {
                        message: format!("async checkpoint thread panicked: {msg}"),
                    })
                }
            }
        });

        self.handle = Some(handle);
        Ok(())
    }

    /// Block until the in-flight save completes (if any).
    pub fn wait(&mut self) -> FerrotorchResult<()> {
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| FerrotorchError::InvalidArgument {
                    message: "async checkpoint thread panicked".into(),
                })??;
        }
        Ok(())
    }

    /// Whether a save is currently in progress.
    pub fn is_saving(&self) -> bool {
        self.in_flight.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Default for AsyncCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> ferrotorch_core::Tensor<f64> {
        let storage = TensorStorage::cpu(data);
        ferrotorch_core::Tensor::from_storage(storage, shape, false).unwrap()
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut model_state: StateDict<f64> = HashMap::new();
        model_state.insert(
            "fc.weight".to_string(),
            make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        model_state.insert("fc.bias".to_string(), make_tensor(vec![0.5, -0.5], vec![2]));

        let mut optimizer_state: OptimizerState = HashMap::new();
        let mut adam_state = HashMap::new();
        adam_state.insert("m".to_string(), vec![0.1, 0.2, 0.3, 0.4]);
        adam_state.insert("v".to_string(), vec![0.01, 0.02, 0.03, 0.04]);
        optimizer_state.insert("fc.weight".to_string(), adam_state);

        let checkpoint = TrainingCheckpoint {
            model_state,
            optimizer_state,
            epoch: 10,
            step: 5000,
        };

        let dir = std::env::temp_dir().join(format!("ferrotorch_test_ckpt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.ftc");

        save_checkpoint(&checkpoint, &path).unwrap();
        let loaded: TrainingCheckpoint<f64> = load_checkpoint(&path).unwrap();

        // Verify metadata.
        assert_eq!(loaded.epoch, 10);
        assert_eq!(loaded.step, 5000);

        // Verify model state.
        assert_eq!(loaded.model_state.len(), 2);
        let w = &loaded.model_state["fc.weight"];
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        let b = &loaded.model_state["fc.bias"];
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.data().unwrap(), &[0.5, -0.5]);

        // Verify optimizer state.
        assert_eq!(loaded.optimizer_state.len(), 1);
        let opt = &loaded.optimizer_state["fc.weight"];
        assert_eq!(opt["m"], vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(opt["v"], vec![0.01, 0.02, 0.03, 0.04]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_checkpoint_empty_optimizer_state() {
        let mut model_state: StateDict<f64> = HashMap::new();
        model_state.insert("w".to_string(), make_tensor(vec![1.0], vec![1]));

        let checkpoint = TrainingCheckpoint {
            model_state,
            optimizer_state: HashMap::new(),
            epoch: 0,
            step: 0,
        };

        let dir = std::env::temp_dir().join(format!(
            "ferrotorch_test_ckpt_empty_opt_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.ftc");

        save_checkpoint(&checkpoint, &path).unwrap();
        let loaded: TrainingCheckpoint<f64> = load_checkpoint(&path).unwrap();

        assert_eq!(loaded.epoch, 0);
        assert_eq!(loaded.step, 0);
        assert!(loaded.optimizer_state.is_empty());
        assert_eq!(loaded.model_state["w"].data().unwrap(), &[1.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_checkpoint_missing_file() {
        let result = load_checkpoint::<f64>("/nonexistent/checkpoint.ftc");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("failed to open"));
    }

    #[test]
    fn test_optimizer_state_serialization_roundtrip() {
        let mut state: OptimizerState = HashMap::new();

        let mut entry1 = HashMap::new();
        entry1.insert("momentum".to_string(), vec![1.0, 2.0, 3.0]);
        entry1.insert("velocity".to_string(), vec![0.1, 0.2, 0.3]);
        state.insert("layer1.weight".to_string(), entry1);

        let mut entry2 = HashMap::new();
        entry2.insert("momentum".to_string(), vec![4.0, 5.0]);
        state.insert("layer1.bias".to_string(), entry2);

        let json = serialize_optimizer_state(&state);
        let loaded = deserialize_optimizer_state(&json).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["layer1.weight"]["momentum"], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded["layer1.weight"]["velocity"], vec![0.1, 0.2, 0.3]);
        assert_eq!(loaded["layer1.bias"]["momentum"], vec![4.0, 5.0]);
    }

    #[test]
    fn test_metadata_parsing() {
        let (epoch, step) = parse_metadata(r#"{"epoch":42,"step":12345}"#).unwrap();
        assert_eq!(epoch, 42);
        assert_eq!(step, 12345);
    }
}
