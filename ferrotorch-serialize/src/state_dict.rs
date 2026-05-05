//! Save and load `StateDict<T>` to/from a simple JSON+binary format.
//!
//! ## File format
//!
//! ```text
//! <JSON line 1: {"name": "...", "shape": [...], "dtype": "...", "byte_offset": N, "byte_length": N}>
//! <JSON line 2: ...>
//! ...
//! \n---\n
//! <raw bytes for tensor 0><raw bytes for tensor 1>...
//! ```
//!
//! Each JSON line describes one tensor. The binary section contains the raw
//! little-endian bytes for each tensor, concatenated in the order of the header
//! lines. The separator `\n---\n` divides header from body.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "ferrotorch state dict serialization assumes little-endian byte order. \
     Big-endian platforms are not supported."
);

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::StateDict;

/// Magic separator between the JSON header and the binary body.
const SEPARATOR: &[u8] = b"\n---\n";

/// Metadata for a single tensor in the serialized file.
#[derive(Debug)]
pub(crate) struct TensorMeta {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: String,
    pub(crate) byte_offset: usize,
    pub(crate) byte_length: usize,
}

/// Return the dtype tag string for the concrete `Float` type.
pub(crate) fn dtype_tag<T: Float>() -> &'static str {
    if std::mem::size_of::<T>() == 4 {
        "f32"
    } else if std::mem::size_of::<T>() == 8 {
        "f64"
    } else {
        "unknown"
    }
}

/// Save a state dict to a file.
///
/// The tensors are sorted by name for deterministic output. Each tensor's
/// data is written as raw little-endian bytes.
pub fn save_state_dict<T: Float>(
    state: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()> {
    let path = path.as_ref();

    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();

    // Build header lines and compute byte offsets.
    let mut header_lines: Vec<String> = Vec::with_capacity(keys.len());
    let elem_size = std::mem::size_of::<T>();
    let mut byte_offset: usize = 0;

    for key in &keys {
        let tensor = &state[*key];
        let numel = tensor.numel();
        let byte_length = numel * elem_size;

        // Build JSON line manually to avoid a serde dependency.
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
            r#"{{"name":"{}","shape":{},"dtype":"{}","byte_offset":{},"byte_length":{}}}"#,
            key,
            shape_str,
            dtype_tag::<T>(),
            byte_offset,
            byte_length,
        );
        header_lines.push(line);
        byte_offset += byte_length;
    }

    // Write file.
    let mut file = std::fs::File::create(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to create file {}: {e}", path.display()),
    })?;

    // Write header lines.
    for line in &header_lines {
        file.write_all(line.as_bytes())
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("write error: {e}"),
            })?;
        file.write_all(b"\n")
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("write error: {e}"),
            })?;
    }

    // Write separator.
    file.write_all(SEPARATOR)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("write error: {e}"),
        })?;

    // Write tensor data.
    for key in &keys {
        let tensor = &state[*key];
        let data = tensor.data()?;
        // Convert the slice of T to raw bytes (little-endian on LE platforms,
        // which covers x86/ARM — production SafeTensors would enforce LE).
        //
        // SAFETY: `data: &[T]` where `T: Float` is one of f32/f64/bf16 — all
        // `Copy` POD types with a stable bit-level layout and no padding or
        // `Drop` semantics. Reinterpreting the same memory as `&[u8]` is
        // therefore well-defined. The byte length `size_of_val(data)` equals
        // `data.len() * size_of::<T>()`, which is the exact extent the
        // pointer is valid for, and the returned `&[u8]` reborrows `data`,
        // so no dangling reference can outlive the source slice. The
        // crate-level `compile_error!` at top-of-file forbids
        // big-endian targets, matching the on-disk byte order required by
        // this format.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        file.write_all(byte_slice)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("write error: {e}"),
            })?;
    }

    file.flush().map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("flush error: {e}"),
    })?;

    Ok(())
}

/// Load a state dict from a file.
///
/// The dtype of the file must match the requested type `T` (e.g., loading
/// an `f32` file into `StateDict<f64>` is an error).
pub fn load_state_dict<T: Float>(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<T>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open file {}: {e}", path.display()),
    })?;
    let mut reader = BufReader::new(file);

    // Parse header lines until we hit the separator.
    let mut metas: Vec<TensorMeta> = Vec::new();
    let expected_dtype = dtype_tag::<T>();

    loop {
        let mut line = String::new();
        let bytes_read =
            reader
                .read_line(&mut line)
                .map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("read error: {e}"),
                })?;

        if bytes_read == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "unexpected end of file before separator".into(),
            });
        }

        let trimmed = line.trim();

        // Check for separator.
        if trimmed == "---" {
            break;
        }

        // Skip empty lines.
        if trimmed.is_empty() {
            continue;
        }

        // Parse the JSON line manually (no serde).
        let meta = parse_meta_line(trimmed)?;

        // Validate dtype.
        if meta.dtype != expected_dtype {
            return Err(FerrotorchError::DtypeMismatch {
                expected: expected_dtype.to_string(),
                got: meta.dtype.clone(),
            });
        }

        metas.push(meta);
    }

    // Read the entire binary body.
    let mut body = Vec::new();
    reader
        .read_to_end(&mut body)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("read error: {e}"),
        })?;

    // Reconstruct tensors.
    let elem_size = std::mem::size_of::<T>();
    let mut state: StateDict<T> = HashMap::with_capacity(metas.len());

    for meta in &metas {
        let end = meta.byte_offset + meta.byte_length;
        if end > body.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor \"{}\" requires bytes [{}..{}] but body has only {} bytes",
                    meta.name,
                    meta.byte_offset,
                    end,
                    body.len()
                ),
            });
        }

        let numel = meta.byte_length / elem_size;
        if numel * elem_size != meta.byte_length {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor \"{}\" byte_length {} is not a multiple of element size {}",
                    meta.name, meta.byte_length, elem_size
                ),
            });
        }

        // Verify shape matches numel.
        let expected_numel: usize = meta.shape.iter().product();
        if expected_numel != numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "tensor \"{}\" shape {:?} implies {} elements but data has {}",
                    meta.name, meta.shape, expected_numel, numel
                ),
            });
        }

        // Reinterpret bytes as T values.
        let byte_slice = &body[meta.byte_offset..end];
        let mut data: Vec<T> = Vec::with_capacity(numel);
        for chunk in byte_slice.chunks_exact(elem_size) {
            // Safe byte-to-float conversion using from_le_bytes; the
            // `try_into` is fallible for malformed input (a chunk that
            // isn't exactly `elem_size` bytes — should not happen with
            // `chunks_exact`, but we propagate cleanly anyway), and the
            // numeric `cast` is fallible for source values not
            // representable in `T` (e.g. `f64::INFINITY` -> `bf16`-like
            // narrow `Float` impls).
            let value: T = match elem_size {
                4 => {
                    let arr: [u8; 4] =
                        chunk
                            .try_into()
                            .map_err(|e| FerrotorchError::InvalidArgument {
                                message: format!(
                                    "malformed state-dict chunk for tensor \"{}\": {e}",
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
                                    "malformed state-dict chunk for tensor \"{}\": {e}",
                                    meta.name
                                ),
                            })?;
                    cast::<f64, T>(f64::from_le_bytes(arr))?
                }
                other => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unsupported element size {other} for tensor \"{}\" \
                             (state_dict format only supports f32/f64)",
                            meta.name
                        ),
                    });
                }
            };
            data.push(value);
        }

        let storage = TensorStorage::cpu(data);
        let tensor = Tensor::from_storage(storage, meta.shape.clone(), false)?;
        state.insert(meta.name.clone(), tensor);
    }

    Ok(state)
}

/// Parse a single JSON metadata line without serde.
///
/// Expected format:
/// `{"name":"...","shape":[...],"dtype":"...","byte_offset":N,"byte_length":N}`
pub(crate) fn parse_meta_line(line: &str) -> FerrotorchResult<TensorMeta> {
    // Simple key-value extraction from JSON-like string.
    let extract_string = |key: &str| -> FerrotorchResult<String> {
        let pattern = format!(r#""{key}":""#);
        let start = line
            .find(&pattern)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("missing key \"{key}\" in header line: {line}"),
            })?
            + pattern.len();
        let end = line[start..]
            .find('"')
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("unterminated string for key \"{key}\" in: {line}"),
            })?
            + start;
        Ok(line[start..end].to_string())
    };

    let extract_usize = |key: &str| -> FerrotorchResult<usize> {
        let pattern = format!(r#""{key}":"#);
        let start = line
            .find(&pattern)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("missing key \"{key}\" in header line: {line}"),
            })?
            + pattern.len();
        // Find the end: next comma, closing brace, or end of string.
        let rest = &line[start..];
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        let value_str = rest[..end].trim();
        value_str
            .parse::<usize>()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to parse \"{key}\" as usize from \"{value_str}\": {e}"),
            })
    };

    let extract_shape = || -> FerrotorchResult<Vec<usize>> {
        let key = "shape";
        let pattern = format!(r#""{key}":["#);
        let start = line
            .find(&pattern)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("missing key \"{key}\" in header line: {line}"),
            })?
            + pattern.len();
        let end = line[start..]
            .find(']')
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("unterminated array for key \"{key}\" in: {line}"),
            })?
            + start;
        let inner = &line[start..end];
        if inner.trim().is_empty() {
            return Ok(Vec::new());
        }
        inner
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .map_err(|e| FerrotorchError::InvalidArgument {
                        message: format!("failed to parse shape element \"{s}\": {e}"),
                    })
            })
            .collect()
    };

    let name = extract_string("name")?;
    let shape = extract_shape()?;
    let dtype = extract_string("dtype")?;
    let byte_offset = extract_usize("byte_offset")?;
    let byte_length = extract_usize("byte_length")?;

    Ok(TensorMeta {
        name,
        shape,
        dtype,
        byte_offset,
        byte_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;
    use std::collections::HashMap;

    fn make_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    fn make_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    #[test]
    fn test_save_load_roundtrip_f64() {
        let mut state: StateDict<f64> = HashMap::new();
        state.insert(
            "weight".to_string(),
            make_tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        );
        state.insert(
            "bias".to_string(),
            make_tensor_f64(vec![0.1, 0.2, 0.3], vec![3]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_sd_f64");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.fts");

        save_state_dict(&state, &path).unwrap();
        let loaded: StateDict<f64> = load_state_dict(&path).unwrap();

        assert_eq!(loaded.len(), 2);

        // Check weight.
        let w = &loaded["weight"];
        assert_eq!(w.shape(), &[2, 3]);
        let w_data = w.data().unwrap();
        assert_eq!(w_data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Check bias.
        let b = &loaded["bias"];
        assert_eq!(b.shape(), &[3]);
        let b_data = b.data().unwrap();
        assert_eq!(b_data, &[0.1, 0.2, 0.3]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary round-trip value, not π.
    fn test_save_load_roundtrip_f32() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "layer.weight".to_string(),
            make_tensor_f32(vec![1.0, -2.5, 3.14, 0.0], vec![2, 2]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_sd_f32");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.fts");

        save_state_dict(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_state_dict(&path).unwrap();

        let w = &loaded["layer.weight"];
        assert_eq!(w.shape(), &[2, 2]);
        let data = w.data().unwrap();
        assert_eq!(data, &[1.0f32, -2.5, 3.14, 0.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_missing_file() {
        let result = load_state_dict::<f64>("/nonexistent/path/file.fts");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("failed to open file"));
    }

    #[test]
    fn test_shape_preservation_scalar() {
        let mut state: StateDict<f64> = HashMap::new();
        state.insert("scalar".to_string(), make_tensor_f64(vec![42.0], vec![]));

        let dir = std::env::temp_dir().join("ferrotorch_test_sd_scalar");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.fts");

        save_state_dict(&state, &path).unwrap();
        let loaded: StateDict<f64> = load_state_dict(&path).unwrap();

        let s = &loaded["scalar"];
        assert_eq!(s.shape(), &[] as &[usize]);
        assert_eq!(s.data().unwrap(), &[42.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_shape_preservation_high_rank() {
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let mut state: StateDict<f64> = HashMap::new();
        state.insert(
            "conv.weight".to_string(),
            make_tensor_f64(data, vec![2, 3, 2, 2]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_sd_4d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.fts");

        save_state_dict(&state, &path).unwrap();
        let loaded: StateDict<f64> = load_state_dict(&path).unwrap();

        let t = &loaded["conv.weight"];
        assert_eq!(t.shape(), &[2, 3, 2, 2]);
        let loaded_data = t.data().unwrap();
        let expected: Vec<f64> = (0..24).map(f64::from).collect();
        assert_eq!(loaded_data, expected.as_slice());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_empty_state_dict() {
        let state: StateDict<f64> = HashMap::new();

        let dir = std::env::temp_dir().join("ferrotorch_test_sd_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.fts");

        save_state_dict(&state, &path).unwrap();
        let loaded: StateDict<f64> = load_state_dict(&path).unwrap();

        assert!(loaded.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_dtype_mismatch() {
        // Save as f32, try to load as f64.
        let mut state: StateDict<f32> = HashMap::new();
        state.insert("x".to_string(), make_tensor_f32(vec![1.0, 2.0], vec![2]));

        let dir = std::env::temp_dir().join("ferrotorch_test_sd_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.fts");

        save_state_dict(&state, &path).unwrap();

        let result = load_state_dict::<f64>(&path);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("dtype mismatch") || err_msg.contains("expected"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parse_meta_line() {
        let line =
            r#"{"name":"weight","shape":[2,3],"dtype":"f64","byte_offset":0,"byte_length":48}"#;
        let meta = parse_meta_line(line).unwrap();
        assert_eq!(meta.name, "weight");
        assert_eq!(meta.shape, vec![2, 3]);
        assert_eq!(meta.dtype, "f64");
        assert_eq!(meta.byte_offset, 0);
        assert_eq!(meta.byte_length, 48);
    }

    #[test]
    fn test_parse_meta_line_empty_shape() {
        let line = r#"{"name":"scalar","shape":[],"dtype":"f64","byte_offset":0,"byte_length":8}"#;
        let meta = parse_meta_line(line).unwrap();
        assert_eq!(meta.name, "scalar");
        assert!(meta.shape.is_empty());
    }

    #[test]
    fn test_deterministic_ordering() {
        // Keys should be sorted alphabetically in the output file.
        let mut state: StateDict<f64> = HashMap::new();
        state.insert("z_last".to_string(), make_tensor_f64(vec![3.0], vec![1]));
        state.insert("a_first".to_string(), make_tensor_f64(vec![1.0], vec![1]));
        state.insert("m_middle".to_string(), make_tensor_f64(vec![2.0], vec![1]));

        let dir = std::env::temp_dir().join("ferrotorch_test_sd_order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.fts");

        save_state_dict(&state, &path).unwrap();

        // Read raw bytes and extract the header portion (before the separator).
        let raw = std::fs::read(&path).unwrap();
        let sep_pos = raw
            .windows(SEPARATOR.len())
            .position(|w| w == SEPARATOR)
            .unwrap();
        let header = std::str::from_utf8(&raw[..sep_pos]).unwrap();
        let a_pos = header.find("a_first").unwrap();
        let m_pos = header.find("m_middle").unwrap();
        let z_pos = header.find("z_last").unwrap();
        assert!(a_pos < m_pos);
        assert!(m_pos < z_pos);

        std::fs::remove_dir_all(&dir).ok();
    }
}
