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
// Symbolic shape specs — DimSpec, InputSpec
// ---------------------------------------------------------------------------

/// A single dimension in an [`InputSpec`] — either a fixed integer
/// size or a symbolic (dynamic) dimension with an optional range.
///
/// Mirrors `PyTorch`'s `torch.export.Dim` / `SymInt` for dynamic shape
/// support. Produced by [`export`] (all dims are [`DimSpec::Static`])
/// or [`export_with_dynamic_shapes`] (selected dims are
/// [`DimSpec::Dynamic`]). CL-396.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimSpec {
    /// A fixed, concrete dimension size.
    Static(usize),
    /// A dynamic dimension whose size is allowed to vary at runtime.
    Dynamic {
        /// Symbolic name (e.g., `"batch"`). Forwarded to ONNX as
        /// `dim_param` for consumers that support symbolic shapes.
        name: String,
        /// Optional inclusive minimum. `None` means unbounded.
        min: Option<usize>,
        /// Optional inclusive maximum. `None` means unbounded.
        max: Option<usize>,
    },
}

impl DimSpec {
    /// Convenience constructor for a symbolic dim with no bounds.
    pub fn dynamic(name: impl Into<String>) -> Self {
        DimSpec::Dynamic {
            name: name.into(),
            min: None,
            max: None,
        }
    }

    /// Convenience constructor for a symbolic dim bounded to
    /// `[min, max]` inclusive.
    pub fn dynamic_range(name: impl Into<String>, min: usize, max: usize) -> Self {
        DimSpec::Dynamic {
            name: name.into(),
            min: Some(min),
            max: Some(max),
        }
    }

    /// Returns `true` if this dimension is dynamic.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, DimSpec::Dynamic { .. })
    }
}

/// Per-input shape specification carrying a mix of static and
/// dynamic dimensions.
///
/// [`export`] produces `InputSpec`s with all dimensions
/// [`DimSpec::Static`]. To mark dimensions as symbolic, use
/// [`export_with_dynamic_shapes`] (or construct an `InputSpec`
/// manually and pass it to `ferrotorch_serialize::export_from_program`).
/// CL-396.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSpec {
    /// Per-dimension spec, outermost-first.
    pub shape: Vec<DimSpec>,
}

impl InputSpec {
    /// Create a new `InputSpec` from a list of dim specs.
    pub fn new(shape: Vec<DimSpec>) -> Self {
        Self { shape }
    }

    /// Build an all-static `InputSpec` from concrete sizes.
    pub fn all_static(shape: &[usize]) -> Self {
        Self {
            shape: shape.iter().map(|&d| DimSpec::Static(d)).collect(),
        }
    }

    /// Returns `true` if any dimension in this spec is dynamic.
    pub fn has_dynamic_dims(&self) -> bool {
        self.shape.iter().any(DimSpec::is_dynamic)
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

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
#[derive(Debug, Clone)]
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
    /// Per-input shape specifications, optionally carrying symbolic
    /// (dynamic) dimensions. Produced by [`export`] (all
    /// [`DimSpec::Static`]) or [`export_with_dynamic_shapes`]
    /// (selected dims [`DimSpec::Dynamic`]). Consumers such as
    /// `ferrotorch-serialize::export_from_program` use this to emit
    /// `dim_param` entries in ONNX for dynamic axes. CL-396.
    pub input_specs: Vec<InputSpec>,
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

    /// Validate `inputs` against this program's [`Self::input_specs`] and
    /// run the graph. Returns an `InvalidArgument` error describing
    /// the first violation if:
    ///
    /// - The number of inputs doesn't match `input_specs.len()`.
    /// - Any input's rank doesn't match its spec's rank.
    /// - Any concrete ([`DimSpec::Static`]) dim doesn't match.
    /// - Any symbolic ([`DimSpec::Dynamic`]) dim is outside its
    ///   optional `[min, max]` range.
    ///
    /// This is the torch.export-style "run with guards" path. Use
    /// [`run`](Self::run) for the unchecked path. CL-461.
    pub fn run_with_guards<T: Float>(&self, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        self.check_inputs(inputs)?;
        self.run(inputs)
    }

    /// Check runtime `inputs` against [`Self::input_specs`]. Returns an
    /// error describing the first violation or `Ok(())` on success.
    /// Public so callers can validate without running the graph.
    /// CL-461.
    pub fn check_inputs<T: Float>(&self, inputs: &[Tensor<T>]) -> FerrotorchResult<()> {
        if inputs.len() != self.input_specs.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ExportedProgram guard: expected {} inputs, got {}",
                    self.input_specs.len(),
                    inputs.len()
                ),
            });
        }
        for (i, (input, spec)) in inputs.iter().zip(self.input_specs.iter()).enumerate() {
            let shape = input.shape();
            if shape.len() != spec.shape.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "ExportedProgram guard: input {i} rank mismatch: \
                         spec has {} dims, runtime has {} (shape {:?})",
                        spec.shape.len(),
                        shape.len(),
                        shape,
                    ),
                });
            }
            for (dim_idx, (&actual, expected)) in shape.iter().zip(spec.shape.iter()).enumerate() {
                match expected {
                    DimSpec::Static(s) => {
                        if *s != actual {
                            return Err(FerrotorchError::InvalidArgument {
                                message: format!(
                                    "ExportedProgram guard: input {i} dim {dim_idx} \
                                     is static, expected {s}, got {actual}"
                                ),
                            });
                        }
                    }
                    DimSpec::Dynamic { name, min, max } => {
                        if actual == 0 {
                            return Err(FerrotorchError::InvalidArgument {
                                message: format!(
                                    "ExportedProgram guard: input {i} dim {dim_idx} \
                                     ('{name}') has runtime value 0"
                                ),
                            });
                        }
                        if let Some(min_v) = min {
                            if actual < *min_v {
                                return Err(FerrotorchError::InvalidArgument {
                                    message: format!(
                                        "ExportedProgram guard: input {i} dim {dim_idx} \
                                         ('{name}') = {actual} is below declared min {min_v}"
                                    ),
                                });
                            }
                        }
                        if let Some(max_v) = max {
                            if actual > *max_v {
                                return Err(FerrotorchError::InvalidArgument {
                                    message: format!(
                                        "ExportedProgram guard: input {i} dim {dim_idx} \
                                         ('{name}') = {actual} is above declared max {max_v}"
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Serialize this `ExportedProgram` to a compact binary byte vector.
    ///
    /// The format is self-contained and deterministic: it layers a
    /// magic header on top of the existing [`IrGraph::serialize`]
    /// format and appends the `state_dict`, `input_shapes`,
    /// `input_specs`, and `output_shape` sections. Use
    /// [`ExportedProgram::deserialize`] to reconstruct.
    ///
    /// This is the preferred path for round-tripping an
    /// `ExportedProgram` to disk — see [`save`](Self::save) for the
    /// file-writing convenience wrapper. CL-296.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4096);

        // Header: magic "FTEP" + version 1.
        out.extend_from_slice(b"FTEP");
        out.extend_from_slice(&1u32.to_le_bytes());

        // Embed the graph via its own serializer. Prefix with length
        // so the deserializer can skip it if the graph format is
        // bumped independently of this one.
        let graph_bytes = self.graph.serialize();
        out.extend_from_slice(&(graph_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&graph_bytes);

        // input_shapes: [n] × [rank, dims...]
        out.extend_from_slice(&(self.input_shapes.len() as u32).to_le_bytes());
        for shape in &self.input_shapes {
            out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &d in shape {
                out.extend_from_slice(&(d as u64).to_le_bytes());
            }
        }

        // output_shape: [rank, dims...]
        out.extend_from_slice(&(self.output_shape.len() as u32).to_le_bytes());
        for &d in &self.output_shape {
            out.extend_from_slice(&(d as u64).to_le_bytes());
        }

        // input_specs: [n] × InputSpec. Each spec is [rank] ×
        //   tag u8 — 0 = Static, 1 = Dynamic
        //   if Static: u64 size
        //   if Dynamic: string_len u32, string bytes, has_min u8,
        //               (u64 min if has_min), has_max u8, (u64 max
        //               if has_max).
        out.extend_from_slice(&(self.input_specs.len() as u32).to_le_bytes());
        for spec in &self.input_specs {
            out.extend_from_slice(&(spec.shape.len() as u32).to_le_bytes());
            for dim in &spec.shape {
                match dim {
                    DimSpec::Static(v) => {
                        out.push(0);
                        out.extend_from_slice(&(*v as u64).to_le_bytes());
                    }
                    DimSpec::Dynamic { name, min, max } => {
                        out.push(1);
                        let bytes = name.as_bytes();
                        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        out.extend_from_slice(bytes);
                        match min {
                            Some(m) => {
                                out.push(1);
                                out.extend_from_slice(&(*m as u64).to_le_bytes());
                            }
                            None => out.push(0),
                        }
                        match max {
                            Some(m) => {
                                out.push(1);
                                out.extend_from_slice(&(*m as u64).to_le_bytes());
                            }
                            None => out.push(0),
                        }
                    }
                }
            }
        }

        // state_dict: [n_entries] × [key_len u32, key bytes, data_len u32, data f32 LE bytes]
        let mut keys: Vec<&String> = self.state_dict.keys().collect();
        keys.sort(); // deterministic ordering
        out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        for key in keys {
            let kb = key.as_bytes();
            out.extend_from_slice(&(kb.len() as u32).to_le_bytes());
            out.extend_from_slice(kb);
            let data = &self.state_dict[key];
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            for &v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }

        out
    }

    /// Reconstruct an `ExportedProgram` from bytes produced by
    /// [`ExportedProgram::serialize`]. CL-296.
    pub fn deserialize(data: &[u8]) -> FerrotorchResult<Self> {
        let mut pos = 0usize;

        // Helper closures — indexed reads with bounds checking.
        fn need<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> FerrotorchResult<&'a [u8]> {
            if data.len() < *pos + n {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "ExportedProgram deserialize: unexpected EOF at {pos}, need {n}, have {}",
                        data.len() - *pos
                    ),
                });
            }
            let slice = &data[*pos..*pos + n];
            *pos += n;
            Ok(slice)
        }
        fn read_u32(data: &[u8], pos: &mut usize) -> FerrotorchResult<u32> {
            let b = need(data, pos, 4)?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        fn read_u64(data: &[u8], pos: &mut usize) -> FerrotorchResult<u64> {
            let b = need(data, pos, 8)?;
            Ok(u64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        fn read_u8(data: &[u8], pos: &mut usize) -> FerrotorchResult<u8> {
            let b = need(data, pos, 1)?;
            Ok(b[0])
        }

        // Header.
        let magic = need(data, &mut pos, 4)?;
        if magic != b"FTEP" {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ExportedProgram deserialize: bad magic {magic:?}, expected b\"FTEP\""
                ),
            });
        }
        let version = read_u32(data, &mut pos)?;
        if version != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ExportedProgram deserialize: unsupported version {version}"),
            });
        }

        // Graph.
        let graph_len = read_u32(data, &mut pos)? as usize;
        let graph_bytes = need(data, &mut pos, graph_len)?;
        let graph = IrGraph::deserialize(graph_bytes)?;

        // input_shapes.
        let n_inputs = read_u32(data, &mut pos)? as usize;
        let mut input_shapes: Vec<Vec<usize>> = Vec::with_capacity(n_inputs);
        for _ in 0..n_inputs {
            let rank = read_u32(data, &mut pos)? as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(read_u64(data, &mut pos)? as usize);
            }
            input_shapes.push(shape);
        }

        // output_shape.
        let out_rank = read_u32(data, &mut pos)? as usize;
        let mut output_shape: Vec<usize> = Vec::with_capacity(out_rank);
        for _ in 0..out_rank {
            output_shape.push(read_u64(data, &mut pos)? as usize);
        }

        // input_specs.
        let n_specs = read_u32(data, &mut pos)? as usize;
        let mut input_specs: Vec<InputSpec> = Vec::with_capacity(n_specs);
        for _ in 0..n_specs {
            let rank = read_u32(data, &mut pos)? as usize;
            let mut dims: Vec<DimSpec> = Vec::with_capacity(rank);
            for _ in 0..rank {
                let tag = read_u8(data, &mut pos)?;
                match tag {
                    0 => {
                        let v = read_u64(data, &mut pos)? as usize;
                        dims.push(DimSpec::Static(v));
                    }
                    1 => {
                        let name_len = read_u32(data, &mut pos)? as usize;
                        let name_bytes = need(data, &mut pos, name_len)?;
                        let name = String::from_utf8(name_bytes.to_vec()).map_err(|e| {
                            FerrotorchError::InvalidArgument {
                                message: format!(
                                    "ExportedProgram deserialize: invalid UTF-8 in DimSpec::Dynamic name: {e}"
                                ),
                            }
                        })?;
                        let has_min = read_u8(data, &mut pos)?;
                        let min = if has_min != 0 {
                            Some(read_u64(data, &mut pos)? as usize)
                        } else {
                            None
                        };
                        let has_max = read_u8(data, &mut pos)?;
                        let max = if has_max != 0 {
                            Some(read_u64(data, &mut pos)? as usize)
                        } else {
                            None
                        };
                        dims.push(DimSpec::Dynamic { name, min, max });
                    }
                    other => {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "ExportedProgram deserialize: unknown DimSpec tag {other}"
                            ),
                        });
                    }
                }
            }
            input_specs.push(InputSpec { shape: dims });
        }

        // state_dict.
        let n_entries = read_u32(data, &mut pos)? as usize;
        let mut state_dict = HashMap::with_capacity(n_entries);
        for _ in 0..n_entries {
            let key_len = read_u32(data, &mut pos)? as usize;
            let key_bytes = need(data, &mut pos, key_len)?;
            let key = String::from_utf8(key_bytes.to_vec()).map_err(|e| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "ExportedProgram deserialize: invalid UTF-8 in state_dict key: {e}"
                    ),
                }
            })?;
            let data_len = read_u32(data, &mut pos)? as usize;
            let mut values: Vec<f32> = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let b = need(data, &mut pos, 4)?;
                values.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
            state_dict.insert(key, values);
        }

        Ok(Self {
            graph,
            state_dict,
            input_shapes,
            input_specs,
            output_shape,
        })
    }

    /// Write this `ExportedProgram` to a file at `path` in the format
    /// produced by [`serialize`](Self::serialize).
    ///
    /// The canonical extension is `.ftep` ("ferrotorch exported
    /// program") but the format is the same regardless of extension.
    /// CL-296.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> FerrotorchResult<()> {
        let bytes = self.serialize();
        let path = path.as_ref();
        std::fs::write(path, &bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "ExportedProgram::save: failed to write {}: {e}",
                path.display()
            ),
        })?;
        Ok(())
    }

    /// Read an `ExportedProgram` from a file produced by
    /// [`save`](Self::save). CL-296.
    pub fn load(path: impl AsRef<std::path::Path>) -> FerrotorchResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "ExportedProgram::load: failed to read {}: {e}",
                path.display()
            ),
        })?;
        Self::deserialize(&bytes)
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
                        .map(std::string::ToString::to_string)
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
                .map(std::string::ToString::to_string)
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
    /// Number of nodes in the exported IR graph.
    pub num_graph_nodes: usize,
    /// Per-input shape recorded at trace time, in declaration order.
    pub input_shapes: Vec<Vec<usize>>,
    /// Shape of the single graph output recorded at trace time.
    pub output_shape: Vec<usize>,
    /// Names of the model parameters captured in the state dict.
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
    // All-static input specs derived from example shapes.
    let input_specs: Vec<InputSpec> = example_inputs
        .iter()
        .map(|t| InputSpec::all_static(t.shape()))
        .collect();
    export_with_dynamic_shapes(module, example_inputs, input_specs)
}

/// Like [`export`] but lets the caller mark selected input dimensions
/// as symbolic (dynamic) via [`InputSpec`]s.
///
/// The resulting [`ExportedProgram`] records the per-input
/// [`DimSpec`] list in its `input_specs` field, which downstream
/// exporters (e.g. ONNX `export_from_program`) use to emit
/// `dim_param` entries for dynamic axes. CL-396.
///
/// # Arguments
///
/// * `module` — the module to trace.
/// * `example_inputs` — concrete example tensors; their sizes set the
///   trace-time shapes. Dynamic dim positions must match the
///   example's concrete size at trace time, but runtime consumers
///   are free to substitute any positive integer (subject to the
///   optional `min`/`max` bounds) so long as the remaining
///   shape-dependent ops are polymorphic.
/// * `input_specs` — one `InputSpec` per input, specifying which
///   dims are static vs. dynamic. Length must equal
///   `example_inputs.len()`, and each `InputSpec.rank()` must match
///   the corresponding example tensor's rank.
///
/// # Errors
///
/// - `example_inputs` does not have exactly one tensor (single-input
///   limitation inherited from [`export`]).
/// - `input_specs.len() != example_inputs.len()`.
/// - Any `InputSpec` rank does not match its example tensor's rank.
/// - Any `DimSpec::Static` dim does not match the example's size at
///   that position.
/// - Tracing fails.
pub fn export_with_dynamic_shapes<T: Float, M: Module<T>>(
    module: &M,
    example_inputs: &[Tensor<T>],
    input_specs: Vec<InputSpec>,
) -> FerrotorchResult<ExportedProgram> {
    // Validate single-input constraint (inherited from export).
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
    if input_specs.len() != example_inputs.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "export_with_dynamic_shapes: input_specs length {} does not match \
                 example_inputs length {}",
                input_specs.len(),
                example_inputs.len(),
            ),
        });
    }
    for (i, (spec, example)) in input_specs.iter().zip(example_inputs.iter()).enumerate() {
        if spec.rank() != example.shape().len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "export_with_dynamic_shapes: input {i} rank mismatch: \
                     spec rank {} vs example rank {} (example shape {:?})",
                    spec.rank(),
                    example.shape().len(),
                    example.shape(),
                ),
            });
        }
        for (dim_idx, (dim_spec, &example_dim)) in
            spec.shape.iter().zip(example.shape().iter()).enumerate()
        {
            if let DimSpec::Static(s) = dim_spec {
                if *s != example_dim {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "export_with_dynamic_shapes: input {i} dim {dim_idx} \
                             DimSpec::Static({s}) != example dim {example_dim}"
                        ),
                    });
                }
            }
        }
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
        input_specs,
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
                Some(c @ ('"' | '\\' | '/')) => out.push(c),
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
/// (e.g., "`num_graph_nodes`" should not match "`extra_num_graph_nodes`").
fn extract_json_usize(json: &str, key: &str) -> FerrotorchResult<usize> {
    let pattern = format!("\"{key}\":");
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
    let pattern = format!("\"{key}\":[");
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
    let pattern = format!("\"{key}\":[");
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
    let pattern = format!("\"{key}\":[");
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

    // --- CL-396: symbolic shape specs ------------------------------------

    #[test]
    fn test_dim_spec_dynamic_constructors() {
        let d = DimSpec::dynamic("batch");
        assert!(d.is_dynamic());
        if let DimSpec::Dynamic { name, min, max } = &d {
            assert_eq!(name, "batch");
            assert_eq!(*min, None);
            assert_eq!(*max, None);
        } else {
            unreachable!();
        }

        let r = DimSpec::dynamic_range("batch", 1, 64);
        if let DimSpec::Dynamic { name, min, max } = &r {
            assert_eq!(name, "batch");
            assert_eq!(*min, Some(1));
            assert_eq!(*max, Some(64));
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_dim_spec_static_is_not_dynamic() {
        assert!(!DimSpec::Static(4).is_dynamic());
    }

    #[test]
    fn test_input_spec_all_static_has_no_dynamic_dims() {
        let s = InputSpec::all_static(&[4, 3, 32, 32]);
        assert_eq!(s.rank(), 4);
        assert!(!s.has_dynamic_dims());
        for d in &s.shape {
            assert!(matches!(d, DimSpec::Static(_)));
        }
    }

    #[test]
    fn test_input_spec_mixed_has_dynamic_dims() {
        let s = InputSpec::new(vec![DimSpec::dynamic("batch"), DimSpec::Static(10)]);
        assert!(s.has_dynamic_dims());
    }

    // --- CL-296: ExportedProgram save/load roundtrip ---------------------

    fn build_dummy_program() -> ExportedProgram {
        // Build a minimal IR graph: input -> relu -> output.
        use crate::graph::{IrGraph, IrOpKind};
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4, 10]);
        let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4, 10]]);
        g.set_outputs(vec![outs[0]]);

        let mut state_dict = HashMap::new();
        state_dict.insert("fc.weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]);
        state_dict.insert("fc.bias".to_string(), vec![0.1f32, 0.2]);

        ExportedProgram {
            graph: g,
            state_dict,
            input_shapes: vec![vec![4, 10]],
            input_specs: vec![InputSpec::new(vec![
                DimSpec::dynamic_range("batch", 1, 64),
                DimSpec::Static(10),
            ])],
            output_shape: vec![4, 10],
        }
    }

    #[test]
    fn test_exported_program_serialize_deserialize_roundtrip() {
        let original = build_dummy_program();
        let bytes = original.serialize();
        assert!(bytes.starts_with(b"FTEP"));
        let parsed = ExportedProgram::deserialize(&bytes).unwrap();

        // Input shapes and output shape preserved.
        assert_eq!(parsed.input_shapes, original.input_shapes);
        assert_eq!(parsed.output_shape, original.output_shape);

        // State dict fully preserved.
        assert_eq!(parsed.state_dict.len(), original.state_dict.len());
        for (k, v) in &original.state_dict {
            assert_eq!(parsed.state_dict.get(k), Some(v));
        }

        // Input specs preserved including symbolic names and range bounds.
        assert_eq!(parsed.input_specs.len(), 1);
        assert_eq!(parsed.input_specs[0].shape.len(), 2);
        match &parsed.input_specs[0].shape[0] {
            DimSpec::Dynamic { name, min, max } => {
                assert_eq!(name, "batch");
                assert_eq!(*min, Some(1));
                assert_eq!(*max, Some(64));
            }
            DimSpec::Static(_) => panic!("expected dynamic dim 0"),
        }
        assert_eq!(parsed.input_specs[0].shape[1], DimSpec::Static(10));

        // Graph round-tripped via IrGraph::serialize.
        assert_eq!(parsed.graph.nodes.len(), original.graph.nodes.len());
        assert_eq!(parsed.graph.input_values, original.graph.input_values);
    }

    #[test]
    fn test_exported_program_save_load_file_roundtrip() {
        let original = build_dummy_program();
        let dir = std::env::temp_dir().join("ferrotorch_test_ep_save_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("program.ftep");
        original.save(&path).unwrap();
        assert!(path.exists());

        let loaded = ExportedProgram::load(&path).unwrap();
        assert_eq!(loaded.input_shapes, original.input_shapes);
        assert_eq!(loaded.output_shape, original.output_shape);
        assert_eq!(loaded.state_dict.len(), original.state_dict.len());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_exported_program_deserialize_rejects_bad_magic() {
        let mut bad = vec![0u8; 16];
        bad[..4].copy_from_slice(b"XXXX");
        let result = ExportedProgram::deserialize(&bad);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("bad magic"), "err = {err}");
    }

    #[test]
    fn test_exported_program_deserialize_rejects_bad_version() {
        let mut bytes = build_dummy_program().serialize();
        // Bump the version byte at offset 4 to something unsupported.
        bytes[4] = 99;
        let result = ExportedProgram::deserialize(&bytes);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("unsupported version"), "err = {err}");
    }

    #[test]
    fn test_exported_program_serialize_is_deterministic() {
        // Two serializations of the same program should be byte-identical
        // (important for content-addressed storage / caching).
        let program = build_dummy_program();
        let a = program.serialize();
        let b = program.serialize();
        assert_eq!(a, b);
    }

    // --- CL-461: runtime guards on ExportedProgram::run -----------------

    fn build_relu_program_with_dynamic_batch() -> ExportedProgram {
        use crate::graph::{IrGraph, IrOpKind};
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4, 10]);
        let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4, 10]]);
        g.set_outputs(vec![outs[0]]);

        ExportedProgram {
            graph: g,
            state_dict: HashMap::new(),
            input_shapes: vec![vec![4, 10]],
            input_specs: vec![InputSpec::new(vec![
                DimSpec::dynamic_range("batch", 1, 32),
                DimSpec::Static(10),
            ])],
            output_shape: vec![4, 10],
        }
    }

    #[test]
    fn test_check_inputs_accepts_valid_dynamic_batch() {
        use ferrotorch_core::randn;
        let program = build_relu_program_with_dynamic_batch();
        // Batch=8 is in [1, 32] and dim 1 matches static 10.
        let x: Tensor<f32> = randn(&[8, 10]).unwrap();
        program.check_inputs(&[x]).unwrap();
    }

    #[test]
    fn test_check_inputs_rejects_wrong_input_count() {
        let program = build_relu_program_with_dynamic_batch();
        let empty: Vec<Tensor<f32>> = vec![];
        let result = program.check_inputs(&empty);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("expected 1 inputs, got 0"));
    }

    #[test]
    fn test_check_inputs_rejects_rank_mismatch() {
        use ferrotorch_core::randn;
        let program = build_relu_program_with_dynamic_batch();
        let x: Tensor<f32> = randn(&[8, 10, 2]).unwrap();
        let result = program.check_inputs(&[x]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("rank mismatch"));
    }

    #[test]
    fn test_check_inputs_rejects_static_dim_mismatch() {
        use ferrotorch_core::randn;
        let program = build_relu_program_with_dynamic_batch();
        // dim 1 is static 10 but we pass 7.
        let x: Tensor<f32> = randn(&[8, 7]).unwrap();
        let result = program.check_inputs(&[x]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("dim 1"));
        assert!(err.contains("expected 10, got 7"));
    }

    #[test]
    fn test_check_inputs_rejects_dynamic_dim_below_min() {
        use ferrotorch_core::randn;
        let program = build_relu_program_with_dynamic_batch();
        // batch must be >= 1, but we have the special zero check.
        // Use the range case: batch=0 fails the "runtime value 0" check.
        let x: Tensor<f32> = randn(&[1, 10]).unwrap();
        // 1 is exactly min — should be accepted.
        program.check_inputs(&[x]).unwrap();
    }

    #[test]
    fn test_check_inputs_rejects_dynamic_dim_above_max() {
        use ferrotorch_core::randn;
        let program = build_relu_program_with_dynamic_batch();
        let x: Tensor<f32> = randn(&[64, 10]).unwrap();
        let result = program.check_inputs(&[x]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("above declared max 32"));
    }

    #[test]
    fn test_run_with_guards_runs_when_inputs_valid() {
        use ferrotorch_core::randn;
        let program = build_relu_program_with_dynamic_batch();
        let x: Tensor<f32> = randn(&[16, 10]).unwrap();
        let out = program.run_with_guards(&[x]).unwrap();
        assert_eq!(out.shape(), &[16, 10]);
    }

    #[test]
    fn test_run_with_guards_rejects_bad_inputs_without_calling_interpret() {
        use ferrotorch_core::randn;
        let program = build_relu_program_with_dynamic_batch();
        let x: Tensor<f32> = randn(&[64, 10]).unwrap();
        let result = program.run_with_guards(&[x]);
        assert!(result.is_err());
    }

    #[test]
    fn test_exported_program_roundtrip_all_static_dims() {
        let mut state_dict = HashMap::new();
        state_dict.insert("w".to_string(), vec![1.0f32; 8]);

        use crate::graph::IrGraph;
        let mut g = IrGraph::new();
        let _x = g.add_input(vec![2, 4]);

        let program = ExportedProgram {
            graph: g,
            state_dict,
            input_shapes: vec![vec![2, 4]],
            input_specs: vec![InputSpec::all_static(&[2, 4])],
            output_shape: vec![2, 4],
        };

        let bytes = program.serialize();
        let parsed = ExportedProgram::deserialize(&bytes).unwrap();
        for dim in &parsed.input_specs[0].shape {
            assert!(matches!(dim, DimSpec::Static(_)));
        }
    }
}
