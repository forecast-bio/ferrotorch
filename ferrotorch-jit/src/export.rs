//! Model export via `torch.export`-style API.
//!
//! Provides [`export()`] to trace a module and produce an [`ExportedProgram`]
//! that can be serialized, optimized, or passed to a runtime.
//!
//! # Limitations
//!
//! - **`export()` only supports single-input models.** The traced function
//!   receives `inputs[0]` as its sole argument. If `example_inputs` contains
//!   more than one tensor, an error is returned. Multi-input support requires
//!   a different module forward signature (future work).

use std::collections::HashMap;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Module;

use crate::graph::IrGraph;
use crate::trace;

// ---------------------------------------------------------------------------
// ExportedProgram
// ---------------------------------------------------------------------------

/// A traced and exported model program.
///
/// Contains the IR graph, the state dict (model weights), and metadata
/// needed to run inference.
///
/// # State dict dtype limitation
///
/// **The `state_dict` field currently only stores `f32` tensors.** This
/// covers the dominant inference dtype but means that models trained in
/// f64 will have their weights truncated to f32 when exported. A future
/// version will support generic or mixed-precision state dicts.
pub struct ExportedProgram {
    /// The traced IR graph.
    pub graph: IrGraph,
    /// Model weights, stored as f32.
    ///
    /// **Limitation:** Only f32 is supported. Models trained in other dtypes
    /// (f64, f16) will have their weights converted to f32 on export. This
    /// matches the common inference scenario but may lose precision for
    /// f64-trained models.
    pub state_dict: HashMap<String, Vec<f32>>,
    /// Input tensor shapes used during tracing.
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shape produced during tracing.
    pub output_shape: Vec<usize>,
}

impl ExportedProgram {
    /// Run the exported program on new inputs.
    ///
    /// # Limitations
    ///
    /// **Only single-output models are supported.** The graph is expected to
    /// produce exactly one output tensor. Multi-output support (e.g., models
    /// returning a tuple of tensors) is planned for a future release.
    pub fn run<T: Float>(&self, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        crate::interpret(&self.graph, inputs)
    }

    /// Serialize the exported program to a simple JSON representation.
    ///
    /// The format stores the graph metadata and state dict. This is not
    /// meant for production use — prefer ONNX export for interop.
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();

        // Input shapes
        let shapes_str: Vec<String> = self
            .input_shapes
            .iter()
            .map(|s| {
                format!(
                    "[{}]",
                    s.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect();
        parts.push(format!("\"input_shapes\":[{}]", shapes_str.join(",")));

        // Output shape
        parts.push(format!(
            "\"output_shape\":[{}]",
            self.output_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ));

        // Number of graph nodes (summary)
        parts.push(format!("\"num_graph_nodes\":{}", self.graph.nodes.len()));

        // State dict keys
        let mut keys: Vec<&String> = self.state_dict.keys().collect();
        keys.sort();
        let keys_str: Vec<String> = keys
            .iter()
            .map(|k| format!("\"{}\"", escape_json_string(k)))
            .collect();
        parts.push(format!("\"state_dict_keys\":[{}]", keys_str.join(",")));

        format!("{{{}}}", parts.join(","))
    }

    /// Parse a simple JSON representation back into metadata.
    ///
    /// This is a minimal parser that extracts known keys. It does not
    /// reconstruct the full graph (use ONNX for full serialization).
    pub fn parse_json_metadata(json: &str) -> FerrotorchResult<ExportedProgramMetadata> {
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(FerrotorchError::InvalidArgument {
                message: "expected JSON object".into(),
            });
        }

        let num_nodes = extract_json_usize(json, "num_graph_nodes")?;
        let input_shapes = extract_json_nested_arrays(json, "input_shapes")?;
        let output_shape = extract_json_array(json, "output_shape")?;
        let state_dict_keys = extract_json_string_array(json, "state_dict_keys")?;

        Ok(ExportedProgramMetadata {
            num_graph_nodes: num_nodes,
            input_shapes,
            output_shape,
            state_dict_keys,
        })
    }
}

/// Metadata extracted from an [`ExportedProgram`] JSON representation.
#[derive(Debug, Clone)]
pub struct ExportedProgramMetadata {
    pub num_graph_nodes: usize,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub state_dict_keys: Vec<String>,
}

// ---------------------------------------------------------------------------
// Export function
// ---------------------------------------------------------------------------

/// Export a module by tracing it with example inputs.
///
/// Traces the module's `forward` method and produces an [`ExportedProgram`]
/// containing the IR graph and model weights.
///
/// # Limitations
///
/// - **Only single-input is supported.** `example_inputs` must contain
///   exactly one tensor. The module's `forward` method is called with that
///   single tensor. Multi-input modules are not yet supported.
/// - **The state dict is stored as f32.** See [`ExportedProgram::state_dict`].
///
/// # Errors
///
/// Returns an error if:
/// - `example_inputs` does not contain exactly one tensor.
/// - Tracing fails (e.g., no autograd graph).
/// - The module forward fails.
pub fn export<T: Float, M: Module<T>>(
    module: &M,
    example_inputs: &[Tensor<T>],
) -> FerrotorchResult<ExportedProgram> {
    // Validate single-input constraint.
    if example_inputs.len() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "export() currently only supports single-input models. \
                 Got {} example inputs. Pass exactly one tensor in the \
                 example_inputs slice.",
                example_inputs.len()
            ),
        });
    }

    let input_shapes: Vec<Vec<usize>> = example_inputs.iter().map(|t| t.shape().to_vec()).collect();

    // Trace the module's forward function.
    let graph = trace(
        |inputs: &[Tensor<T>]| module.forward(&inputs[0]),
        example_inputs,
    )?;

    // Extract output shape from the graph.
    let output_shape = if let Some(&out_id) = graph.output_values.first() {
        graph
            .values
            .iter()
            .find(|v| v.id == out_id)
            .map(|v| v.shape.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    // Extract state dict as f32.
    let named_params = module.named_parameters();
    let mut state_dict = HashMap::new();
    for (name, param) in named_params {
        let data = param.tensor().data_vec()?;
        let f32_data: Vec<f32> = data.iter().map(|&v| v.to_f64().unwrap() as f32).collect();
        state_dict.insert(name, f32_data);
    }

    Ok(ExportedProgram {
        graph,
        state_dict,
        input_shapes,
        output_shape,
    })
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Escape a string for JSON output.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Control character: emit \uXXXX
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Unescape a JSON string value, including `\uXXXX` sequences.
fn unescape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('b') => out.push('\u{0008}'),
                Some('f') => out.push('\u{000C}'),
                Some('u') => {
                    // Parse \uXXXX
                    let hex: String = chars.by_ref().take(4).collect();
                    if hex.len() == 4 {
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(code) {
                                out.push(c);
                            } else {
                                // Invalid unicode code point, emit replacement
                                out.push('\u{FFFD}');
                            }
                        } else {
                            out.push_str("\\u");
                            out.push_str(&hex);
                        }
                    } else {
                        out.push_str("\\u");
                        out.push_str(&hex);
                    }
                }
                Some(c) => {
                    out.push('\\');
                    out.push(c);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Extract a usize value for a given key from JSON.
///
/// Uses exact key matching with `"key":` delimiter to avoid false matches
/// (e.g., "num_graph_nodes" should not match "extra_num_graph_nodes").
fn extract_json_usize(json: &str, key: &str) -> FerrotorchResult<usize> {
    let pattern = format!("\"{}\":", key);
    let start = json
        .find(&pattern)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("missing key \"{key}\" in JSON"),
        })?
        + pattern.len();
    let rest = &json[start..];
    let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse::<usize>()
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse \"{key}\": {e}"),
        })
}

/// Extract a flat array of usize values from JSON.
fn extract_json_array(json: &str, key: &str) -> FerrotorchResult<Vec<usize>> {
    let pattern = format!("\"{}\":[", key);
    let start = json
        .find(&pattern)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("missing key \"{key}\" in JSON"),
        })?
        + pattern.len();
    let rest = &json[start..];
    let end = rest
        .find(']')
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("unterminated array for key \"{key}\""),
        })?;
    let inner = &rest[..end];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("failed to parse array element: {e}"),
                })
        })
        .collect()
}

/// Extract nested arrays (e.g., `[[2,3],[4]]`) from JSON.
fn extract_json_nested_arrays(json: &str, key: &str) -> FerrotorchResult<Vec<Vec<usize>>> {
    let pattern = format!("\"{}\":[", key);
    let start = json
        .find(&pattern)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("missing key \"{key}\" in JSON"),
        })?
        + pattern.len();
    let rest = &json[start..];

    // Find the matching closing bracket.
    let mut depth = 1;
    let mut end = 0;
    for (i, ch) in rest.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }

    let inner = &rest[..end];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut pos = 0;
    while pos < inner.len() {
        if let Some(arr_start) = inner[pos..].find('[') {
            let abs_start = pos + arr_start + 1;
            if let Some(arr_end) = inner[abs_start..].find(']') {
                let abs_end = abs_start + arr_end;
                let arr_str = &inner[abs_start..abs_end];
                if arr_str.trim().is_empty() {
                    result.push(Vec::new());
                } else {
                    let values: Vec<usize> = arr_str
                        .split(',')
                        .map(|s| s.trim().parse::<usize>())
                        .collect::<Result<_, _>>()
                        .map_err(|e| FerrotorchError::InvalidArgument {
                            message: format!("failed to parse nested array: {e}"),
                        })?;
                    result.push(values);
                }
                pos = abs_end + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Ok(result)
}

/// Extract an array of JSON strings from JSON.
fn extract_json_string_array(json: &str, key: &str) -> FerrotorchResult<Vec<String>> {
    let pattern = format!("\"{}\":[", key);
    let start = json
        .find(&pattern)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("missing key \"{key}\" in JSON"),
        })?
        + pattern.len();
    let rest = &json[start..];
    let end = rest
        .find(']')
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("unterminated array for key \"{key}\""),
        })?;
    let inner = &rest[..end];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut pos = 0;
    while pos < inner.len() {
        if let Some(q_start) = inner[pos..].find('"') {
            let abs_start = pos + q_start + 1;
            // Find closing quote, respecting escapes.
            let mut end_pos = abs_start;
            let chars: Vec<char> = inner[abs_start..].chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '\\' {
                    i += 2; // skip escaped character
                    end_pos += chars[i - 2].len_utf8()
                        + if i - 1 < chars.len() {
                            chars[i - 1].len_utf8()
                        } else {
                            0
                        };
                    continue;
                }
                if chars[i] == '"' {
                    break;
                }
                end_pos += chars[i].len_utf8();
                i += 1;
            }
            let raw_str = &inner[abs_start..end_pos];
            result.push(unescape_json_string(raw_str));
            pos = end_pos + 1;
        } else {
            break;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_json_string() {
        assert_eq!(escape_json_string("hello"), "hello");
        assert_eq!(escape_json_string("he\"llo"), "he\\\"llo");
        assert_eq!(escape_json_string("a\\b"), "a\\\\b");
        assert_eq!(escape_json_string("a\nb"), "a\\nb");
    }

    #[test]
    fn test_unescape_json_string() {
        assert_eq!(unescape_json_string("hello"), "hello");
        assert_eq!(unescape_json_string("he\\\"llo"), "he\"llo");
        assert_eq!(unescape_json_string("a\\\\b"), "a\\b");
        assert_eq!(unescape_json_string("a\\nb"), "a\nb");
    }

    #[test]
    fn test_unescape_unicode() {
        assert_eq!(unescape_json_string("\\u0041"), "A");
        assert_eq!(unescape_json_string("\\u00e9"), "\u{00e9}"); // e-acute
        assert_eq!(unescape_json_string("hello\\u0020world"), "hello world");
    }

    #[test]
    fn test_extract_json_usize() {
        let json = r#"{"num_graph_nodes":42,"other":7}"#;
        assert_eq!(extract_json_usize(json, "num_graph_nodes").unwrap(), 42);
        assert_eq!(extract_json_usize(json, "other").unwrap(), 7);
    }

    #[test]
    fn test_extract_json_array() {
        let json = r#"{"output_shape":[2,3,4]}"#;
        assert_eq!(
            extract_json_array(json, "output_shape").unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_extract_json_array_empty() {
        let json = r#"{"output_shape":[]}"#;
        assert!(extract_json_array(json, "output_shape").unwrap().is_empty());
    }

    #[test]
    fn test_extract_json_nested_arrays() {
        let json = r#"{"input_shapes":[[2,3],[4]]}"#;
        let result = extract_json_nested_arrays(json, "input_shapes").unwrap();
        assert_eq!(result, vec![vec![2, 3], vec![4]]);
    }

    #[test]
    fn test_extract_json_string_array() {
        let json = r#"{"state_dict_keys":["weight","bias"]}"#;
        let result = extract_json_string_array(json, "state_dict_keys").unwrap();
        assert_eq!(result, vec!["weight", "bias"]);
    }

    #[test]
    fn test_exported_program_metadata_roundtrip() {
        let json = r#"{"input_shapes":[[2,3]],"output_shape":[2,5],"num_graph_nodes":3,"state_dict_keys":["fc.weight","fc.bias"]}"#;
        let meta = ExportedProgram::parse_json_metadata(json).unwrap();
        assert_eq!(meta.num_graph_nodes, 3);
        assert_eq!(meta.input_shapes, vec![vec![2, 3]]);
        assert_eq!(meta.output_shape, vec![2, 5]);
        assert_eq!(meta.state_dict_keys, vec!["fc.weight", "fc.bias"]);
    }
}
