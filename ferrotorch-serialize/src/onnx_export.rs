//! Export an [`IrGraph`] as an ONNX model file.
//!
//! [ONNX](https://onnx.ai/) (Open Neural Network Exchange) is a standard
//! protobuf-based format for representing ML models. Models exported as `.onnx`
//! files can be loaded by ONNX Runtime (C++/Python), TensorRT (NVIDIA),
//! CoreML (Apple), and many other inference engines.
//!
//! This module implements a minimal hand-written binary protobuf encoder --
//! no protobuf compiler or code-generation step is needed. We only emit the
//! subset of protobuf fields required to produce a valid ONNX `ModelProto`.
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_core::grad_fns::activation::relu;
//! use ferrotorch_serialize::onnx_export::{export_onnx, OnnxExportConfig};
//!
//! export_onnx(
//!     |inputs| relu(&inputs[0]),
//!     &example_inputs,
//!     "model.onnx",
//!     OnnxExportConfig::default(),
//! )?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use ferrotorch_jit::graph::{IrGraph, IrNodeId, IrOpKind, IrValueId};
use ferrotorch_jit::trace;

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for ONNX model export.
pub struct OnnxExportConfig {
    /// ONNX opset version. Minimum (and default) is 17.
    pub opset_version: usize,
    /// Human-readable model name stored in the ONNX graph.
    pub model_name: String,
    /// Dynamic axes: maps input index → list of (axis, name) pairs.
    ///
    /// Example: `vec![(0, vec![(0, "batch".into())])]` marks axis 0 of input 0
    /// as dynamic with symbolic name "batch".
    pub dynamic_axes: HashMap<usize, Vec<(usize, String)>>,
}

impl Default for OnnxExportConfig {
    fn default() -> Self {
        Self {
            opset_version: 17,
            model_name: "ferrotorch_model".to_string(),
            dynamic_axes: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX data type constants
// ---------------------------------------------------------------------------

/// ONNX `TensorProto.DataType` constants.
const ONNX_FLOAT: i32 = 1;
const ONNX_DOUBLE: i32 = 11;

/// Return the ONNX data-type enum value for `T`.
fn onnx_dtype<T: Float>() -> FerrotorchResult<i32> {
    match std::mem::size_of::<T>() {
        4 => Ok(ONNX_FLOAT),
        8 => Ok(ONNX_DOUBLE),
        other => Err(FerrotorchError::InvalidArgument {
            message: format!("unsupported element size {other} for ONNX export"),
        }),
    }
}

// ===========================================================================
// Minimal protobuf binary encoder
// ===========================================================================

/// A lightweight protobuf wire-format writer.
///
/// Only the wire types needed for ONNX are implemented:
/// - varint (wire type 0)
/// - 32-bit fixed (wire type 5)
/// - 64-bit fixed (wire type 1)
/// - length-delimited (wire type 2): bytes, strings, and embedded messages
struct ProtobufWriter {
    buf: Vec<u8>,
}

#[allow(dead_code)]
impl ProtobufWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    fn bytes(&self) -> &[u8] {
        &self.buf
    }

    // -- low-level primitives -----------------------------------------------

    fn write_varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                self.buf.push(byte);
                return;
            }
            self.buf.push(byte | 0x80);
        }
    }

    fn write_tag(&mut self, field_number: u32, wire_type: u32) {
        self.write_varint(((field_number as u64) << 3) | wire_type as u64);
    }

    // -- typed field writers -------------------------------------------------

    /// Write a `uint64` field (wire type 0 — varint).
    fn write_uint64(&mut self, field_number: u32, value: u64) {
        self.write_tag(field_number, 0);
        self.write_varint(value);
    }

    /// Write an `int64` field (wire type 0 — varint, zig-zag NOT used for
    /// plain `int64` in proto3; the spec says "standard encoding").
    fn write_int64(&mut self, field_number: u32, value: i64) {
        self.write_tag(field_number, 0);
        self.write_varint(value as u64);
    }

    /// Write an `int32` field (wire type 0 — varint).
    ///
    /// The `value as u64` cast sign-extends negative i32 values, which is
    /// correct per protobuf varint encoding for `int32` (not `sint32`).
    /// Negative values produce 10-byte varints because all upper bits are set.
    fn write_int32(&mut self, field_number: u32, value: i32) {
        self.write_tag(field_number, 0);
        self.write_varint(value as u64);
    }

    /// Write a `string` field (wire type 2 — length-delimited).
    fn write_string(&mut self, field_number: u32, s: &str) {
        self.write_tag(field_number, 2);
        self.write_varint(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Write a `bytes` field (wire type 2 — length-delimited).
    fn write_bytes(&mut self, field_number: u32, data: &[u8]) {
        self.write_tag(field_number, 2);
        self.write_varint(data.len() as u64);
        self.buf.extend_from_slice(data);
    }

    /// Write an embedded message field (wire type 2 — length-delimited).
    fn write_message(&mut self, field_number: u32, message: &[u8]) {
        self.write_tag(field_number, 2);
        self.write_varint(message.len() as u64);
        self.buf.extend_from_slice(message);
    }

    /// Write a `float` field (wire type 5 — 32-bit fixed).
    fn write_float(&mut self, field_number: u32, value: f32) {
        self.write_tag(field_number, 5);
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a `double` field (wire type 1 — 64-bit fixed).
    fn write_double(&mut self, field_number: u32, value: f64) {
        self.write_tag(field_number, 1);
        self.buf.extend_from_slice(&value.to_le_bytes());
    }
}

// ===========================================================================
// ONNX protobuf field numbers (from onnx.proto3)
// ===========================================================================

// ModelProto
const MODEL_IR_VERSION: u32 = 1;
const MODEL_GRAPH: u32 = 7;
const MODEL_OPSET_IMPORT: u32 = 8;

// OperatorSetIdProto
const OPSET_DOMAIN: u32 = 1;
const OPSET_VERSION: u32 = 2;

// GraphProto
const GRAPH_NODE: u32 = 1;
const GRAPH_NAME: u32 = 2;
const GRAPH_INITIALIZER: u32 = 5;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;

// NodeProto
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const NODE_NAME: u32 = 3;
const NODE_OP_TYPE: u32 = 4;
// const NODE_DOMAIN: u32 = 7; // unused — default domain is ""
const NODE_ATTRIBUTE: u32 = 5;

// AttributeProto
const ATTR_NAME: u32 = 1;
// const ATTR_F: u32 = 4; // unused
const ATTR_I: u32 = 3;
#[allow(dead_code)]
const ATTR_INTS: u32 = 8;
const ATTR_TYPE: u32 = 20;

// AttributeProto.AttributeType enum values
const ATTR_TYPE_INT: i32 = 2;
#[allow(dead_code)]
const ATTR_TYPE_INTS: i32 = 7;

// TensorProto
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
const TENSOR_NAME: u32 = 8;
const TENSOR_RAW_DATA: u32 = 13;

// ValueInfoProto
const VALUE_INFO_NAME: u32 = 1;
const VALUE_INFO_TYPE: u32 = 2;

// TypeProto
const TYPE_TENSOR: u32 = 1;

// TypeProto.Tensor
const TENSOR_TYPE_ELEM_TYPE: u32 = 1;
const TENSOR_TYPE_SHAPE: u32 = 2;

// TensorShapeProto
const SHAPE_DIM: u32 = 1;

// TensorShapeProto.Dimension
const DIM_VALUE: u32 = 1;
/// Protobuf field tag for symbolic dimension parameter names.
pub const DIM_PARAM: u32 = 2;

// ===========================================================================
// ONNX message builders
// ===========================================================================

/// Encode an `OperatorSetIdProto`.
fn encode_opset(domain: &str, version: u64) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    if !domain.is_empty() {
        w.write_string(OPSET_DOMAIN, domain);
    }
    w.write_uint64(OPSET_VERSION, version);
    w.into_bytes()
}

/// Encode a `TensorShapeProto.Dimension` (dim_value variant).
fn encode_dim(value: u64) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_uint64(DIM_VALUE, value);
    w.into_bytes()
}

/// Encode a `TensorShapeProto`.
fn encode_shape(dims: &[usize]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    for &d in dims {
        let dim_bytes = encode_dim(d as u64);
        w.write_message(SHAPE_DIM, &dim_bytes);
    }
    w.into_bytes()
}

/// Encode a `TensorShapeProto.Dimension` with a symbolic dim_param (string).
pub fn encode_dim_param(param: &str) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_string(DIM_PARAM, param);
    w.into_bytes()
}

/// Specification for a single ONNX dimension — either static or dynamic.
pub enum OnnxDimSpec {
    Static(usize),
    Dynamic(String),
}

/// Encode a `TensorShapeProto` with a mix of static and dynamic dimensions.
pub fn encode_shape_with_dynamic(dims: &[OnnxDimSpec]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    for dim in dims {
        let dim_bytes = match dim {
            OnnxDimSpec::Static(value) => encode_dim(*value as u64),
            OnnxDimSpec::Dynamic(name) => encode_dim_param(name),
        };
        w.write_message(SHAPE_DIM, &dim_bytes);
    }
    w.into_bytes()
}

/// Encode a `ValueInfoProto` with support for dynamic dimensions.
pub fn encode_value_info_dynamic(name: &str, elem_type: i32, dims: &[OnnxDimSpec]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_string(VALUE_INFO_NAME, name);

    // TypeProto.Tensor
    let mut tt = ProtobufWriter::new();
    tt.write_int32(TENSOR_TYPE_ELEM_TYPE, elem_type);
    let shape_bytes = encode_shape_with_dynamic(dims);
    tt.write_message(TENSOR_TYPE_SHAPE, &shape_bytes);
    let tensor_type_bytes = tt.into_bytes();

    // TypeProto
    let mut tp = ProtobufWriter::new();
    tp.write_message(TYPE_TENSOR, &tensor_type_bytes);
    let type_bytes = tp.into_bytes();

    w.write_message(VALUE_INFO_TYPE, &type_bytes);
    w.into_bytes()
}

/// Encode a `TypeProto.Tensor`.
fn encode_tensor_type(elem_type: i32, shape: &[usize]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_int32(TENSOR_TYPE_ELEM_TYPE, elem_type);
    let shape_bytes = encode_shape(shape);
    w.write_message(TENSOR_TYPE_SHAPE, &shape_bytes);
    w.into_bytes()
}

/// Encode a `TypeProto`.
fn encode_type_proto(elem_type: i32, shape: &[usize]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    let tensor_bytes = encode_tensor_type(elem_type, shape);
    w.write_message(TYPE_TENSOR, &tensor_bytes);
    w.into_bytes()
}

/// Encode a `ValueInfoProto`.
fn encode_value_info(name: &str, elem_type: i32, shape: &[usize]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_string(VALUE_INFO_NAME, name);
    let type_bytes = encode_type_proto(elem_type, shape);
    w.write_message(VALUE_INFO_TYPE, &type_bytes);
    w.into_bytes()
}

/// Encode a `TensorProto` (initializer) with raw data bytes.
fn encode_tensor_proto(name: &str, elem_type: i32, dims: &[usize], raw_data: &[u8]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    // dims is a repeated int64 field — write each one individually
    for &d in dims {
        w.write_int64(TENSOR_DIMS, d as i64);
    }
    w.write_int32(TENSOR_DATA_TYPE, elem_type);
    w.write_string(TENSOR_NAME, name);
    w.write_bytes(TENSOR_RAW_DATA, raw_data);
    w.into_bytes()
}

/// Encode an `AttributeProto` with an integer value.
fn encode_attr_int(name: &str, value: i64) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_string(ATTR_NAME, name);
    w.write_int64(ATTR_I, value);
    w.write_int32(ATTR_TYPE, ATTR_TYPE_INT);
    w.into_bytes()
}

/// Encode an `AttributeProto` with a list of integers.
#[allow(dead_code)]
fn encode_attr_ints(name: &str, values: &[i64]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_string(ATTR_NAME, name);
    for &v in values {
        w.write_int64(ATTR_INTS, v);
    }
    w.write_int32(ATTR_TYPE, ATTR_TYPE_INTS);
    w.into_bytes()
}

/// Encode a `NodeProto`.
fn encode_node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: &[Vec<u8>],
) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    for &inp in inputs {
        w.write_string(NODE_INPUT, inp);
    }
    for &out in outputs {
        w.write_string(NODE_OUTPUT, out);
    }
    w.write_string(NODE_NAME, name);
    w.write_string(NODE_OP_TYPE, op_type);
    for attr in attributes {
        w.write_message(NODE_ATTRIBUTE, attr);
    }
    w.into_bytes()
}

/// Encode a `GraphProto`.
fn encode_graph(
    name: &str,
    nodes: &[Vec<u8>],
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    initializers: &[Vec<u8>],
) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    for node in nodes {
        w.write_message(GRAPH_NODE, node);
    }
    w.write_string(GRAPH_NAME, name);
    for init in initializers {
        w.write_message(GRAPH_INITIALIZER, init);
    }
    for inp in inputs {
        w.write_message(GRAPH_INPUT, inp);
    }
    for out in outputs {
        w.write_message(GRAPH_OUTPUT, out);
    }
    w.into_bytes()
}

/// Encode a `ModelProto`.
fn encode_model(ir_version: u64, opset_imports: &[Vec<u8>], graph: &[u8]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    w.write_uint64(MODEL_IR_VERSION, ir_version);
    for opset in opset_imports {
        w.write_message(MODEL_OPSET_IMPORT, opset);
    }
    w.write_message(MODEL_GRAPH, graph);
    w.into_bytes()
}

// ===========================================================================
// IrOpKind -> ONNX op mapping
// ===========================================================================

/// Information needed to create an ONNX `NodeProto` from an IR operation.
#[derive(Debug)]
struct OnnxOpMapping {
    /// ONNX operator name (e.g. "Add", "Relu", "MatMul").
    op_type: &'static str,
    /// Additional attributes to attach to the NodeProto.
    attributes: Vec<Vec<u8>>,
    /// If the op needs an auxiliary constant input (e.g. Reshape needs a
    /// shape tensor), its name and raw bytes. These will be added as
    /// initializer inputs.
    aux_initializer: Option<(String, Vec<u8>, Vec<usize>)>,
}

fn map_ir_op(op: &IrOpKind, node_name: &str, elem_type: i32) -> FerrotorchResult<OnnxOpMapping> {
    match op {
        // Arithmetic
        IrOpKind::Add => Ok(OnnxOpMapping {
            op_type: "Add",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Sub => Ok(OnnxOpMapping {
            op_type: "Sub",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Mul => Ok(OnnxOpMapping {
            op_type: "Mul",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Div => Ok(OnnxOpMapping {
            op_type: "Div",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Neg => Ok(OnnxOpMapping {
            op_type: "Neg",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Pow { .. } => {
            // ONNX Pow takes two tensors; exponent is a scalar initializer.
            // We extract the exponent from the IrOpKind and store it as an
            // auxiliary initializer tensor.
            let IrOpKind::Pow { exponent } = op else {
                unreachable!()
            };
            let exp_name = format!("{node_name}_exponent");
            let raw = if elem_type == ONNX_FLOAT {
                (*exponent as f32).to_le_bytes().to_vec()
            } else {
                exponent.to_le_bytes().to_vec()
            };
            Ok(OnnxOpMapping {
                op_type: "Pow",
                attributes: vec![],
                aux_initializer: Some((exp_name, raw, vec![])),
            })
        }
        IrOpKind::Sqrt => Ok(OnnxOpMapping {
            op_type: "Sqrt",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Abs => Ok(OnnxOpMapping {
            op_type: "Abs",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Exp => Ok(OnnxOpMapping {
            op_type: "Exp",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Log => Ok(OnnxOpMapping {
            op_type: "Log",
            attributes: vec![],
            aux_initializer: None,
        }),

        // Reduction
        IrOpKind::Sum => Ok(OnnxOpMapping {
            op_type: "ReduceSum",
            attributes: vec![encode_attr_int("keepdims", 0)],
            aux_initializer: None,
        }),
        IrOpKind::Mean => Ok(OnnxOpMapping {
            op_type: "ReduceMean",
            attributes: vec![encode_attr_int("keepdims", 0)],
            aux_initializer: None,
        }),
        IrOpKind::Prod => Ok(OnnxOpMapping {
            op_type: "ReduceProd",
            attributes: vec![encode_attr_int("keepdims", 0)],
            aux_initializer: None,
        }),

        // Linalg
        IrOpKind::Matmul | IrOpKind::Mm | IrOpKind::Mv | IrOpKind::Dot => Ok(OnnxOpMapping {
            op_type: "MatMul",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Transpose => Ok(OnnxOpMapping {
            op_type: "Transpose",
            attributes: vec![],
            aux_initializer: None,
        }),

        // Activation
        IrOpKind::Relu => Ok(OnnxOpMapping {
            op_type: "Relu",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Sigmoid => Ok(OnnxOpMapping {
            op_type: "Sigmoid",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Tanh => Ok(OnnxOpMapping {
            op_type: "Tanh",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Gelu => {
            // Gelu is handled as a multi-node decomposition in the
            // emitter loop — map_ir_op is never called for it. This
            // branch exists only for completeness; if it ever fires,
            // we have a bug in the emitter.
            Err(FerrotorchError::InvalidArgument {
                message: "Gelu should be decomposed in the emitter loop, not mapped here; see emit_gelu_decomposed".into(),
            })
        }
        IrOpKind::Silu => {
            // Silu is decomposed into Sigmoid + Mul in the emitter
            // loop — see emit_silu_decomposed. See Gelu above.
            Err(FerrotorchError::InvalidArgument {
                message: "Silu should be decomposed in the emitter loop, not mapped here; see emit_silu_decomposed".into(),
            })
        }
        IrOpKind::Softmax => Ok(OnnxOpMapping {
            op_type: "Softmax",
            attributes: vec![encode_attr_int("axis", -1)],
            aux_initializer: None,
        }),
        IrOpKind::LogSoftmax => Ok(OnnxOpMapping {
            op_type: "LogSoftmax",
            attributes: vec![encode_attr_int("axis", -1)],
            aux_initializer: None,
        }),

        // Shape
        IrOpKind::Reshape { shape } => {
            // ONNX Reshape takes the target shape as a second int64 tensor
            // input. We create an auxiliary initializer for it.
            let shape_name = format!("{node_name}_shape");
            let raw: Vec<u8> = shape
                .iter()
                .flat_map(|&d| (d as i64).to_le_bytes())
                .collect();
            let dims = vec![shape.len()];
            Ok(OnnxOpMapping {
                op_type: "Reshape",
                attributes: vec![],
                // We encode the shape tensor as int64 (data_type=7).
                // We'll handle this specially since it's int64, not the model
                // element type.
                aux_initializer: Some((shape_name, raw, dims)),
            })
        }
        IrOpKind::Flatten => Ok(OnnxOpMapping {
            op_type: "Flatten",
            attributes: vec![],
            aux_initializer: None,
        }),
        IrOpKind::Squeeze { axis } => Ok(OnnxOpMapping {
            op_type: "Squeeze",
            attributes: vec![],
            aux_initializer: {
                // ONNX Squeeze (opset 13+) takes axes as a tensor input.
                let axes_name = format!("{node_name}_axes");
                let raw = (*axis as i64).to_le_bytes().to_vec();
                Some((axes_name, raw, vec![1]))
            },
        }),
        IrOpKind::Unsqueeze { axis } => Ok(OnnxOpMapping {
            op_type: "Unsqueeze",
            attributes: vec![],
            aux_initializer: {
                let axes_name = format!("{node_name}_axes");
                let raw = (*axis as i64).to_le_bytes().to_vec();
                Some((axes_name, raw, vec![1]))
            },
        }),
        IrOpKind::Cat { axis } => Ok(OnnxOpMapping {
            op_type: "Concat",
            attributes: vec![encode_attr_int("axis", *axis as i64)],
            aux_initializer: None,
        }),

        // Linear: maps to ONNX Gemm (General Matrix Multiply).
        // Gemm computes Y = alpha * A @ B + beta * C, with transB=1 for weight^T.
        IrOpKind::Linear => Ok(OnnxOpMapping {
            op_type: "Gemm",
            attributes: vec![encode_attr_int("transB", 1)],
            aux_initializer: None,
        }),

        // Not exportable
        IrOpKind::Input { .. } | IrOpKind::Constant { .. } | IrOpKind::Output => {
            // These are handled specially — not emitted as ONNX nodes.
            Err(FerrotorchError::InvalidArgument {
                message: format!("IR op {op:?} should not be mapped to an ONNX node directly"),
            })
        }

        IrOpKind::FusedElementwise { .. } => Err(FerrotorchError::InvalidArgument {
            message: "fused elementwise ops must be un-fused before ONNX export".into(),
        }),

        IrOpKind::Cond => Err(FerrotorchError::InvalidArgument {
            message: "cond ops must be lowered before ONNX export".into(),
        }),

        IrOpKind::Scan => Err(FerrotorchError::InvalidArgument {
            message: "scan ops must be lowered before ONNX export".into(),
        }),

        IrOpKind::FusedLinearActivation { .. } => Err(FerrotorchError::InvalidArgument {
            message: "fused linear+activation ops must be un-fused before ONNX export".into(),
        }),

        IrOpKind::FusedAttention { .. } => Err(FerrotorchError::InvalidArgument {
            message: "fused attention ops must be lowered before ONNX export".into(),
        }),
    }
}

// ===========================================================================
// Graph conversion: IrGraph -> ONNX bytes
// ===========================================================================

/// Convert an [`IrGraph`] to ONNX protobuf bytes.
///
/// This is the core conversion routine. It walks the graph in topological
/// order and emits ONNX `NodeProto`, `ValueInfoProto`, `TensorProto`
/// (initializer), etc.
pub fn ir_graph_to_onnx(
    graph: &IrGraph,
    config: &OnnxExportConfig,
    elem_type: i32,
) -> FerrotorchResult<Vec<u8>> {
    // Topological ordering so that every value is defined before it is used.
    let topo = graph.topological_order();

    // Map IrValueId -> ONNX string name.
    let value_name = |id: IrValueId| -> String { format!("val_{}", id.0) };

    // Build lookup: IrNodeId -> &IrNode
    let node_map: HashMap<IrNodeId, &_> = graph.nodes.iter().map(|n| (n.id, n)).collect();

    // Build lookup: IrValueId -> &IrValue
    let value_map: HashMap<IrValueId, &_> = graph.values.iter().map(|v| (v.id, v)).collect();

    // Collect the encoded ONNX messages.
    let mut onnx_nodes: Vec<Vec<u8>> = Vec::new();
    let mut onnx_initializers: Vec<Vec<u8>> = Vec::new();
    let mut onnx_inputs: Vec<Vec<u8>> = Vec::new();

    // Track which value names are already declared as graph inputs (to avoid
    // double-declaring initializers that are also inputs).
    let mut declared_inputs: Vec<String> = Vec::new();

    let mut input_counter: usize = 0;

    for &nid in &topo {
        let node = node_map[&nid];

        match &node.op {
            IrOpKind::Input { .. } => {
                // Emit a graph input ValueInfoProto.
                let out_id = node.outputs[0];
                let shape = &value_map[&out_id].shape;
                let name = value_name(out_id);

                // Use dynamic dimension encoding if configured for this input.
                let vi = if let Some(dyn_axes) = config.dynamic_axes.get(&input_counter) {
                    let dims: Vec<OnnxDimSpec> = shape
                        .iter()
                        .enumerate()
                        .map(|(axis, &size)| {
                            if let Some((_, sym_name)) = dyn_axes.iter().find(|(a, _)| *a == axis) {
                                OnnxDimSpec::Dynamic(sym_name.clone())
                            } else {
                                OnnxDimSpec::Static(size)
                            }
                        })
                        .collect();
                    encode_value_info_dynamic(&name, elem_type, &dims)
                } else {
                    encode_value_info(&name, elem_type, shape)
                };

                onnx_inputs.push(vi);
                declared_inputs.push(name);
                input_counter += 1;
            }

            IrOpKind::Constant { data, shape } => {
                // Emit an initializer (TensorProto) and a matching graph
                // input so that ONNX validators are happy.
                let out_id = node.outputs[0];
                let name = value_name(out_id);

                let raw: Vec<u8> = if elem_type == ONNX_FLOAT {
                    data.iter()
                        .flat_map(|&v| (v as f32).to_le_bytes())
                        .collect()
                } else {
                    data.iter().flat_map(|&v| v.to_le_bytes()).collect()
                };

                let tp = encode_tensor_proto(&name, elem_type, shape, &raw);
                onnx_initializers.push(tp);

                // Also add as input (ONNX best practice: initializers should
                // also appear in graph.input).
                let vi = encode_value_info(&name, elem_type, shape);
                onnx_inputs.push(vi);
                declared_inputs.push(name);
            }

            IrOpKind::Output => {
                // Output nodes don't create ONNX nodes.
            }

            // Silu decomposed into Sigmoid + Mul so the ONNX output
            // is portable to any opset >= 13. CL-375.
            IrOpKind::Silu => {
                let node_name = format!("node_{}", nid.0);
                let input_name = value_name(node.inputs[0]);
                let output_name = value_name(node.outputs[0]);
                let sigmoid_out = format!("{node_name}_sigmoid");

                // x -> Sigmoid -> tmp
                let sigmoid_node = encode_node(
                    &format!("{node_name}_sigmoid"),
                    "Sigmoid",
                    &[input_name.as_str()],
                    &[sigmoid_out.as_str()],
                    &[],
                );
                // (x, tmp) -> Mul -> y
                let mul_node = encode_node(
                    &format!("{node_name}_mul"),
                    "Mul",
                    &[input_name.as_str(), sigmoid_out.as_str()],
                    &[output_name.as_str()],
                    &[],
                );
                onnx_nodes.push(sigmoid_node);
                onnx_nodes.push(mul_node);
            }

            // Gelu decomposed into the erf-based formula
            //     y = x * 0.5 * (1 + Erf(x / sqrt(2)))
            // which maps to five standard ONNX ops (Div, Erf, Add,
            // Mul, Mul) supported since opset 13. CL-375.
            IrOpKind::Gelu => {
                let node_name = format!("node_{}", nid.0);
                let input_name = value_name(node.inputs[0]);
                let output_name = value_name(node.outputs[0]);

                // Intermediate value names.
                let sqrt2_name = format!("{node_name}_sqrt2");
                let half_name = format!("{node_name}_half");
                let one_name = format!("{node_name}_one");
                let div_out = format!("{node_name}_div");
                let erf_out = format!("{node_name}_erf");
                let add_out = format!("{node_name}_add");
                let half_mul_out = format!("{node_name}_halfmul");

                // Constants — we create three initializers: sqrt(2),
                // 0.5, and 1.0. They use `elem_type` so they match
                // the model's dtype. Scalars are encoded as 1-element
                // tensors with a 1D [1] shape (an empty shape is also
                // valid ONNX but less portable across runtimes).
                let scalar_dims = vec![1usize];
                let sqrt2_raw: Vec<u8> = if elem_type == ONNX_FLOAT {
                    (std::f64::consts::SQRT_2 as f32).to_le_bytes().to_vec()
                } else {
                    std::f64::consts::SQRT_2.to_le_bytes().to_vec()
                };
                let half_raw: Vec<u8> = if elem_type == ONNX_FLOAT {
                    (0.5f32).to_le_bytes().to_vec()
                } else {
                    (0.5f64).to_le_bytes().to_vec()
                };
                let one_raw: Vec<u8> = if elem_type == ONNX_FLOAT {
                    (1.0f32).to_le_bytes().to_vec()
                } else {
                    (1.0f64).to_le_bytes().to_vec()
                };
                for (cname, craw) in [
                    (&sqrt2_name, &sqrt2_raw),
                    (&half_name, &half_raw),
                    (&one_name, &one_raw),
                ] {
                    let tp = encode_tensor_proto(cname, elem_type, &scalar_dims, craw);
                    onnx_initializers.push(tp);
                    let vi = encode_value_info(cname, elem_type, &scalar_dims);
                    onnx_inputs.push(vi);
                    declared_inputs.push(cname.clone());
                }

                // Emit the five ONNX nodes.
                let div_node = encode_node(
                    &format!("{node_name}_div"),
                    "Div",
                    &[input_name.as_str(), sqrt2_name.as_str()],
                    &[div_out.as_str()],
                    &[],
                );
                let erf_node = encode_node(
                    &format!("{node_name}_erf"),
                    "Erf",
                    &[div_out.as_str()],
                    &[erf_out.as_str()],
                    &[],
                );
                let add_node = encode_node(
                    &format!("{node_name}_add"),
                    "Add",
                    &[erf_out.as_str(), one_name.as_str()],
                    &[add_out.as_str()],
                    &[],
                );
                let halfmul_node = encode_node(
                    &format!("{node_name}_halfmul"),
                    "Mul",
                    &[input_name.as_str(), half_name.as_str()],
                    &[half_mul_out.as_str()],
                    &[],
                );
                let final_mul_node = encode_node(
                    &format!("{node_name}_finalmul"),
                    "Mul",
                    &[half_mul_out.as_str(), add_out.as_str()],
                    &[output_name.as_str()],
                    &[],
                );

                onnx_nodes.push(div_node);
                onnx_nodes.push(erf_node);
                onnx_nodes.push(add_node);
                onnx_nodes.push(halfmul_node);
                onnx_nodes.push(final_mul_node);
            }

            op => {
                // Regular computation node.
                let node_name = format!("node_{}", nid.0);
                let mapping = map_ir_op(op, &node_name, elem_type)?;

                // Collect input names.
                let input_names: Vec<String> =
                    node.inputs.iter().map(|&id| value_name(id)).collect();
                let mut all_input_names = input_names.clone();

                // Handle auxiliary initializers (e.g., Reshape shape tensor).
                if let Some((aux_name, raw, dims)) = &mapping.aux_initializer {
                    // Determine the data type for the auxiliary tensor.
                    // Reshape shape and Squeeze/Unsqueeze axes are int64 (type 7).
                    let aux_dtype = match mapping.op_type {
                        "Reshape" | "Squeeze" | "Unsqueeze" => 7, // INT64
                        _ => elem_type,
                    };
                    let tp = encode_tensor_proto(aux_name, aux_dtype, dims, raw);
                    onnx_initializers.push(tp);

                    let vi = encode_value_info(aux_name, aux_dtype, dims);
                    onnx_inputs.push(vi);
                    declared_inputs.push(aux_name.clone());

                    all_input_names.push(aux_name.clone());
                }

                // Collect output names.
                let output_names: Vec<String> =
                    node.outputs.iter().map(|&id| value_name(id)).collect();

                let input_refs: Vec<&str> = all_input_names.iter().map(|s| s.as_str()).collect();
                let output_refs: Vec<&str> = output_names.iter().map(|s| s.as_str()).collect();

                let encoded = encode_node(
                    &node_name,
                    mapping.op_type,
                    &input_refs,
                    &output_refs,
                    &mapping.attributes,
                );
                onnx_nodes.push(encoded);
            }
        }
    }

    // Graph outputs.
    let onnx_outputs: Vec<Vec<u8>> = graph
        .output_values
        .iter()
        .map(|&id| {
            let shape = &value_map[&id].shape;
            encode_value_info(&value_name(id), elem_type, shape)
        })
        .collect();

    // Assemble the graph.
    let graph_bytes = encode_graph(
        &config.model_name,
        &onnx_nodes,
        &onnx_inputs,
        &onnx_outputs,
        &onnx_initializers,
    );

    // Opset imports.
    let opset = encode_opset("", config.opset_version as u64);

    // Model (ONNX IR version 8 corresponds to opset 17+).
    let model_bytes = encode_model(8, &[opset], &graph_bytes);

    Ok(model_bytes)
}

// ===========================================================================
// Public API
// ===========================================================================

/// Trace a function and export the resulting graph as an ONNX model file.
///
/// The function `trace_fn` is traced with the provided `example_inputs` to
/// produce an [`IrGraph`], which is then serialized to the ONNX protobuf
/// binary format and written to `path`.
///
/// # Arguments
///
/// * `trace_fn` - The model forward function. It receives a slice of tensors
///   and must return a single output tensor.
/// * `example_inputs` - Concrete input tensors (at least one must have
///   `requires_grad = true` for tracing to work).
/// * `path` - Output file path (conventionally `*.onnx`).
/// * `config` - Export configuration (opset version, model name).
///
/// # Errors
///
/// Returns an error if:
/// - The opset version is less than 17.
/// - Tracing fails (e.g. no autograd graph).
/// - An unsupported IR operation is encountered.
/// - The output file cannot be written.
pub fn export_onnx<T: Float>(
    trace_fn: impl Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
    example_inputs: &[Tensor<T>],
    path: impl AsRef<Path>,
    config: OnnxExportConfig,
) -> FerrotorchResult<()> {
    // Validate opset version.
    if config.opset_version < 17 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ONNX opset version must be >= 17, got {}",
                config.opset_version
            ),
        });
    }

    let elem_type = onnx_dtype::<T>()?;

    // Trace the function.
    let graph = trace(trace_fn, example_inputs)?;

    // Convert to ONNX bytes.
    let onnx_bytes = ir_graph_to_onnx(&graph, &config, elem_type)?;

    // Write to file.
    let path = path.as_ref();
    std::fs::write(path, &onnx_bytes).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to write ONNX file {}: {e}", path.display()),
    })?;

    Ok(())
}

/// Export an already-constructed [`IrGraph`] as an ONNX model file.
///
/// Use this when you already have a traced or manually constructed graph and
/// do not need to re-trace.
pub fn export_ir_graph_to_onnx(
    graph: &IrGraph,
    path: impl AsRef<Path>,
    config: OnnxExportConfig,
    elem_type: i32,
) -> FerrotorchResult<()> {
    if config.opset_version < 17 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ONNX opset version must be >= 17, got {}",
                config.opset_version
            ),
        });
    }

    let onnx_bytes = ir_graph_to_onnx(graph, &config, elem_type)?;

    let path = path.as_ref();
    std::fs::write(path, &onnx_bytes).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to write ONNX file {}: {e}", path.display()),
    })?;

    Ok(())
}

// ===========================================================================
// ExportedProgram -> ONNX
// ===========================================================================

/// Export an [`ExportedProgram`](ferrotorch_jit::export::ExportedProgram)
/// as an ONNX model file.
///
/// The program's `graph` field is written out with the given config,
/// matching the behavior of [`export_ir_graph_to_onnx`]. The program's
/// f32 state dict is not emitted as separate ONNX metadata — the
/// graph's `Constant` nodes already carry the weights.
///
/// Any dynamic dimensions declared in `program.input_specs` are
/// forwarded into the ONNX output as `dim_param` entries via the
/// [`OnnxExportConfig::dynamic_axes`] path — existing entries on the
/// caller's `config` are merged with the program-level specs, with
/// caller entries taking precedence on conflict (input index +
/// axis). CL-396.
///
/// Uses [`ONNX_FLOAT`] as the element type since the current
/// [`ExportedProgram`] state dict is f32-only.
///
/// # Arguments
///
/// * `program` — The exported program to convert.
/// * `path` — Output file path (conventionally `*.onnx`).
/// * `config` — Export configuration (opset version, model name,
///   dynamic axes).
///
/// # Errors
///
/// Returns an error if the graph contains operations that cannot be
/// mapped to ONNX, or if the file cannot be written.
pub fn export_from_program(
    program: &ferrotorch_jit::export::ExportedProgram,
    path: impl AsRef<Path>,
    mut config: OnnxExportConfig,
) -> FerrotorchResult<()> {
    if config.opset_version < 17 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ONNX opset version must be >= 17, got {}",
                config.opset_version
            ),
        });
    }
    // ExportedProgram state dict is f32-only today; match that.
    let elem_type = ONNX_FLOAT;

    // Merge program-level dynamic dims into the caller's
    // OnnxExportConfig. Caller-provided entries win on conflict.
    for (input_idx, spec) in program.input_specs.iter().enumerate() {
        let mut program_axes: Vec<(usize, String)> = Vec::new();
        for (axis, dim) in spec.shape.iter().enumerate() {
            if let ferrotorch_jit::export::DimSpec::Dynamic { name, .. } = dim {
                program_axes.push((axis, name.clone()));
            }
        }
        if program_axes.is_empty() {
            continue;
        }
        let entry = config.dynamic_axes.entry(input_idx).or_default();
        for (axis, name) in program_axes {
            if !entry.iter().any(|(a, _)| *a == axis) {
                entry.push((axis, name));
            }
        }
    }

    let onnx_bytes = ir_graph_to_onnx(&program.graph, &config, elem_type)?;

    let path = path.as_ref();
    std::fs::write(path, &onnx_bytes).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to write ONNX file {}: {e}", path.display()),
    })?;

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_jit::graph::{IrGraph, IrOpKind};

    // -----------------------------------------------------------------------
    // Helper: build a simple IrGraph without tracing
    // -----------------------------------------------------------------------

    /// Build a graph: input0 + input1 -> output
    fn make_add_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2, 3]);
        let y = g.add_input(vec![2, 3]);
        let (_, outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![2, 3]]);
        g.set_outputs(vec![outs[0]]);
        g
    }

    /// Build a graph with a constant: input0 + const -> output
    fn make_add_const_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![1.0, 2.0, 3.0], vec![3]);
        let (_, outs) = g.add_node(IrOpKind::Add, vec![x, c], vec![vec![3]]);
        g.set_outputs(vec![outs[0]]);
        g
    }

    /// Build a graph: relu(input0 + input1) -> output
    fn make_add_relu_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let y = g.add_input(vec![4]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);
        g
    }

    /// Build a graph: matmul(input0, input1) -> output
    fn make_matmul_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![2, 3]);
        let b = g.add_input(vec![3, 4]);
        let (_, outs) = g.add_node(IrOpKind::Matmul, vec![a, b], vec![vec![2, 4]]);
        g.set_outputs(vec![outs[0]]);
        g
    }

    /// Build a graph: softmax(input0) -> output
    fn make_softmax_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2, 5]);
        let (_, outs) = g.add_node(IrOpKind::Softmax, vec![x], vec![vec![2, 5]]);
        g.set_outputs(vec![outs[0]]);
        g
    }

    /// Build a graph: reshape(input0, [6]) -> output
    fn make_reshape_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2, 3]);
        let (_, outs) = g.add_node(IrOpKind::Reshape { shape: vec![6] }, vec![x], vec![vec![6]]);
        g.set_outputs(vec![outs[0]]);
        g
    }

    // -----------------------------------------------------------------------
    // Test: ProtobufWriter varint encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_varint_encoding() {
        let mut w = ProtobufWriter::new();

        // Single-byte varints.
        w.write_varint(0);
        assert_eq!(w.bytes(), &[0x00]);

        let mut w = ProtobufWriter::new();
        w.write_varint(1);
        assert_eq!(w.bytes(), &[0x01]);

        let mut w = ProtobufWriter::new();
        w.write_varint(127);
        assert_eq!(w.bytes(), &[0x7F]);

        // Two-byte varint.
        let mut w = ProtobufWriter::new();
        w.write_varint(128);
        assert_eq!(w.bytes(), &[0x80, 0x01]);

        // Larger value: 300 = 0b100101100 -> [0xAC, 0x02]
        let mut w = ProtobufWriter::new();
        w.write_varint(300);
        assert_eq!(w.bytes(), &[0xAC, 0x02]);
    }

    // -----------------------------------------------------------------------
    // Test: export a simple add graph and verify protobuf structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_add_graph_produces_valid_bytes() {
        let graph = make_add_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_FLOAT).unwrap();

        // ONNX files are just protobuf. The first byte should be a valid
        // protobuf tag. Field 1 (ir_version) with wire type 0 (varint) = 0x08.
        assert!(!bytes.is_empty());
        assert_eq!(
            bytes[0], 0x08,
            "first byte should be tag for field 1 varint"
        );

        // ir_version should be 8.
        assert_eq!(bytes[1], 0x08, "ir_version should be 8");

        // The bytes should contain the model name.
        let model_bytes = String::from_utf8_lossy(&bytes);
        assert!(
            model_bytes.contains("ferrotorch_model"),
            "model bytes should contain the graph name"
        );
    }

    // -----------------------------------------------------------------------
    // Test: export with constants (weights)
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_with_constants() {
        let graph = make_add_const_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_FLOAT).unwrap();

        assert!(!bytes.is_empty());

        // The output should contain the "Add" op type string.
        let as_str = String::from_utf8_lossy(&bytes);
        assert!(as_str.contains("Add"), "should contain Add op");

        // Should contain raw constant data (1.0f32 = 0x3F800000).
        let needle = 1.0f32.to_le_bytes();
        assert!(
            bytes.windows(4).any(|w| w == needle),
            "should contain raw bytes for 1.0f32"
        );
    }

    // -----------------------------------------------------------------------
    // Test: export with multiple ops (add + relu)
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_add_relu_graph() {
        let graph = make_add_relu_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_FLOAT).unwrap();

        let as_str = String::from_utf8_lossy(&bytes);
        assert!(as_str.contains("Add"), "should contain Add op");
        assert!(as_str.contains("Relu"), "should contain Relu op");
    }

    // -----------------------------------------------------------------------
    // Test: export MatMul graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_matmul_graph() {
        let graph = make_matmul_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_FLOAT).unwrap();

        let as_str = String::from_utf8_lossy(&bytes);
        assert!(as_str.contains("MatMul"), "should contain MatMul op");
    }

    // -----------------------------------------------------------------------
    // Test: export Softmax graph (with axis attribute)
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_softmax_graph() {
        let graph = make_softmax_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_FLOAT).unwrap();

        let as_str = String::from_utf8_lossy(&bytes);
        assert!(as_str.contains("Softmax"), "should contain Softmax op");
        // The axis attribute name should be present.
        assert!(as_str.contains("axis"), "should contain axis attribute");
    }

    // -----------------------------------------------------------------------
    // Test: export Reshape graph (with shape initializer)
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_reshape_graph() {
        let graph = make_reshape_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_FLOAT).unwrap();

        let as_str = String::from_utf8_lossy(&bytes);
        assert!(as_str.contains("Reshape"), "should contain Reshape op");
        // The shape initializer name should be present.
        assert!(
            as_str.contains("_shape"),
            "should contain shape initializer name"
        );
    }

    // -----------------------------------------------------------------------
    // Test: opset version validation (reject < 17)
    // -----------------------------------------------------------------------

    #[test]
    fn test_opset_version_reject_below_17() {
        let graph = make_add_graph();
        let config = OnnxExportConfig {
            opset_version: 13,
            model_name: "test".into(),
            ..Default::default()
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_onnx_opset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_opset.onnx");

        let result = export_ir_graph_to_onnx(&graph, &path, config, ONNX_FLOAT);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("opset version must be >= 17"),
            "error should mention opset version: {msg}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: opset version 17 is accepted
    // -----------------------------------------------------------------------

    #[test]
    fn test_opset_version_17_accepted() {
        let graph = make_add_graph();
        let config = OnnxExportConfig {
            opset_version: 17,
            model_name: "test".into(),
            ..Default::default()
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_onnx_opset17");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("opset17.onnx");

        let result = export_ir_graph_to_onnx(&graph, &path, config, ONNX_FLOAT);
        assert!(result.is_ok());

        // Verify file was written.
        assert!(path.exists());
        let file_bytes = std::fs::read(&path).unwrap();
        assert!(!file_bytes.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: round-trip structure — export, read back, verify protobuf fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_structure() {
        let graph = make_add_graph();
        let config = OnnxExportConfig {
            opset_version: 17,
            model_name: "roundtrip_test".into(),
            ..Default::default()
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_onnx_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.onnx");

        export_ir_graph_to_onnx(&graph, &path, config, ONNX_FLOAT).unwrap();

        let bytes = std::fs::read(&path).unwrap();

        // Verify overall structure by parsing the top-level protobuf manually.
        // Field 1 (ir_version, varint) should be first.
        assert_eq!(bytes[0], 0x08); // tag: field 1, wire type 0
        assert_eq!(bytes[1], 0x08); // value: 8

        // The file should contain the model name string.
        assert!(
            bytes
                .windows(b"roundtrip_test".len())
                .any(|w| w == b"roundtrip_test"),
            "should contain model name"
        );

        // The file should contain value names (val_0, val_1, etc.).
        assert!(
            bytes.windows(5).any(|w| w == b"val_0"),
            "should contain val_0"
        );
        assert!(
            bytes.windows(5).any(|w| w == b"val_1"),
            "should contain val_1"
        );

        // Should contain the "Add" op type.
        assert!(
            bytes.windows(3).any(|w| w == b"Add"),
            "should contain Add op"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: f64 element type
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_f64_graph() {
        let graph = make_add_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_DOUBLE).unwrap();

        // The bytes should contain the data type 11 (DOUBLE).
        // In protobuf, TensorType elem_type field 1 with value 11 is encoded
        // as tag 0x08, value 0x0B.
        assert!(!bytes.is_empty());
        assert!(
            bytes.windows(2).any(|w| w == [0x08, 0x0B]),
            "should contain DOUBLE data type encoding (0x08, 0x0B)"
        );
    }

    // -----------------------------------------------------------------------
    // Test: export with constant data in f64
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_constants_f64() {
        let graph = make_add_const_graph();
        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_DOUBLE).unwrap();

        // Should contain raw f64 bytes for 1.0 (0x3FF0000000000000 LE).
        let needle = 1.0f64.to_le_bytes();
        assert!(
            bytes.windows(8).any(|w| w == needle),
            "should contain raw bytes for 1.0f64"
        );
    }

    // -----------------------------------------------------------------------
    // Test: all basic op types are mapped
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_basic_ops_have_mappings() {
        let ops = [
            IrOpKind::Add,
            IrOpKind::Sub,
            IrOpKind::Mul,
            IrOpKind::Div,
            IrOpKind::Neg,
            IrOpKind::Pow { exponent: 2.0 },
            IrOpKind::Sqrt,
            IrOpKind::Abs,
            IrOpKind::Sum,
            IrOpKind::Mean,
            IrOpKind::Prod,
            IrOpKind::Matmul,
            IrOpKind::Mm,
            IrOpKind::Mv,
            IrOpKind::Dot,
            IrOpKind::Transpose,
            IrOpKind::Relu,
            IrOpKind::Sigmoid,
            IrOpKind::Tanh,
            // Gelu and Silu are intentionally absent — they are
            // decomposed into multiple ONNX nodes in the emitter
            // loop, not mapped by `map_ir_op` directly. See the
            // dedicated decomposition tests below.
            IrOpKind::Softmax,
            IrOpKind::LogSoftmax,
            IrOpKind::Reshape { shape: vec![6] },
            IrOpKind::Flatten,
            IrOpKind::Squeeze { axis: 0 },
            IrOpKind::Unsqueeze { axis: 0 },
            IrOpKind::Cat { axis: 0 },
        ];

        for op in &ops {
            let result = map_ir_op(op, "test_node", ONNX_FLOAT);
            assert!(
                result.is_ok(),
                "op {:?} should have an ONNX mapping, got error: {:?}",
                op,
                result.err()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Silu decomposition: Sigmoid + Mul (CL-375)
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_decomposed_into_sigmoid_and_mul() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, outs) = g.add_node(IrOpKind::Silu, vec![x], vec![vec![3]]);
        g.set_outputs(vec![outs[0]]);

        let bytes = ir_graph_to_onnx(&g, &OnnxExportConfig::default(), ONNX_FLOAT).unwrap();
        let as_str = String::from_utf8_lossy(&bytes);

        // Standard ONNX ops — no custom "Silu" op type should appear.
        assert!(as_str.contains("Sigmoid"), "should contain Sigmoid op");
        assert!(as_str.contains("Mul"), "should contain Mul op");
        assert!(
            !as_str.contains("\u{0}Silu"),
            "should NOT contain the non-standard Silu op type"
        );
    }

    #[test]
    fn test_map_ir_op_rejects_silu_direct_call() {
        // Direct calls to map_ir_op with Silu should error — it's
        // handled exclusively in the emitter loop.
        let result = map_ir_op(&IrOpKind::Silu, "test", ONNX_FLOAT);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("decomposed"));
    }

    // -----------------------------------------------------------------------
    // Gelu decomposition: Div + Erf + Add + Mul + Mul (CL-375)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gelu_decomposed_via_erf_formula() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, outs) = g.add_node(IrOpKind::Gelu, vec![x], vec![vec![4]]);
        g.set_outputs(vec![outs[0]]);

        let bytes = ir_graph_to_onnx(&g, &OnnxExportConfig::default(), ONNX_FLOAT).unwrap();
        let as_str = String::from_utf8_lossy(&bytes);

        // All five decomposed ONNX ops should be present.
        assert!(as_str.contains("Div"), "should contain Div op");
        assert!(as_str.contains("Erf"), "should contain Erf op");
        assert!(as_str.contains("Add"), "should contain Add op");
        assert!(as_str.contains("Mul"), "should contain Mul op");
        // And the original Gelu name should not appear as an op type.
        assert!(
            !as_str.contains("\u{0}Gelu"),
            "should NOT contain the standalone Gelu op type"
        );
    }

    #[test]
    fn test_map_ir_op_rejects_gelu_direct_call() {
        let result = map_ir_op(&IrOpKind::Gelu, "test", ONNX_FLOAT);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("decomposed"));
    }

    #[test]
    fn test_gelu_decomposition_emits_three_initializers() {
        // sqrt(2), 0.5, and 1.0 should each appear as initializers.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2]);
        let (_, outs) = g.add_node(IrOpKind::Gelu, vec![x], vec![vec![2]]);
        g.set_outputs(vec![outs[0]]);

        let bytes = ir_graph_to_onnx(&g, &OnnxExportConfig::default(), ONNX_FLOAT).unwrap();
        let as_str = String::from_utf8_lossy(&bytes);

        // Every initializer is named "<node>_sqrt2", "<node>_half", "<node>_one".
        assert!(as_str.contains("sqrt2"), "sqrt2 initializer missing");
        assert!(as_str.contains("half"), "half initializer missing");
        assert!(as_str.contains("one"), "one initializer missing");
    }

    // -----------------------------------------------------------------------
    // export_from_program (CL-375 — re-enabled)
    // -----------------------------------------------------------------------

    #[test]
    fn test_export_from_program_writes_valid_file() {
        use ferrotorch_jit::export::ExportedProgram;

        // Build a minimal IrGraph directly and wrap it in an
        // ExportedProgram. This avoids depending on the tracer for
        // the test, which has stricter requirements on the input
        // function.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        g.set_outputs(vec![outs[0]]);

        let program = ExportedProgram {
            graph: g,
            state_dict: std::collections::HashMap::new(),
            input_shapes: vec![vec![4]],
            input_specs: vec![ferrotorch_jit::export::InputSpec::all_static(&[4])],
            output_shape: vec![4],
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_export_from_program");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prog.onnx");

        export_from_program(&program, &path, OnnxExportConfig::default()).unwrap();

        // File should exist and be non-empty.
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(!bytes.is_empty());
        let as_str = String::from_utf8_lossy(&bytes);
        assert!(
            as_str.contains("Relu"),
            "exported bytes should contain Relu op"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_export_from_program_forwards_dynamic_input_specs() {
        use ferrotorch_jit::export::{DimSpec, ExportedProgram, InputSpec};

        // Simple Relu graph with input [batch, 10] where batch is
        // symbolic. The ExportedProgram.input_specs should be
        // forwarded into the ONNX output's dim_param field.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4, 10]);
        let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4, 10]]);
        g.set_outputs(vec![outs[0]]);

        let program = ExportedProgram {
            graph: g,
            state_dict: std::collections::HashMap::new(),
            input_shapes: vec![vec![4, 10]],
            input_specs: vec![InputSpec::new(vec![
                DimSpec::dynamic("batch"),
                DimSpec::Static(10),
            ])],
            output_shape: vec![4, 10],
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_export_from_program_dynamic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dyn.onnx");

        export_from_program(&program, &path, OnnxExportConfig::default()).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let as_str = String::from_utf8_lossy(&bytes);
        // The symbolic dim name should appear in the output as a
        // dim_param entry.
        assert!(
            as_str.contains("batch"),
            "dynamic dim name 'batch' should appear in ONNX output"
        );
        assert!(as_str.contains("Relu"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_export_from_program_caller_dynamic_axes_take_precedence() {
        use ferrotorch_jit::export::{DimSpec, ExportedProgram, InputSpec};

        // Program declares axis 0 as "batch"; caller overrides with
        // "N" on the same axis — caller should win.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4, 10]);
        let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4, 10]]);
        g.set_outputs(vec![outs[0]]);

        let program = ExportedProgram {
            graph: g,
            state_dict: std::collections::HashMap::new(),
            input_shapes: vec![vec![4, 10]],
            input_specs: vec![InputSpec::new(vec![
                DimSpec::dynamic("batch"),
                DimSpec::Static(10),
            ])],
            output_shape: vec![4, 10],
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_export_from_program_caller_override");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("override.onnx");

        let mut config = OnnxExportConfig::default();
        config.dynamic_axes.insert(0, vec![(0, "N".to_string())]);
        export_from_program(&program, &path, config).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let as_str = String::from_utf8_lossy(&bytes);
        assert!(
            as_str.contains("N"),
            "caller override 'N' should be present"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_export_from_program_rejects_opset_below_17() {
        use ferrotorch_jit::export::ExportedProgram;

        let mut g = IrGraph::new();
        let x = g.add_input(vec![2]);
        g.set_outputs(vec![x]);

        let program = ExportedProgram {
            graph: g,
            state_dict: std::collections::HashMap::new(),
            input_shapes: vec![vec![2]],
            input_specs: vec![ferrotorch_jit::export::InputSpec::all_static(&[2])],
            output_shape: vec![2],
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_export_from_program_opset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.onnx");
        let config = OnnxExportConfig {
            opset_version: 16,
            ..OnnxExportConfig::default()
        };
        let result = export_from_program(&program, &path, config);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("opset version"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: FusedElementwise is rejected
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_elementwise_rejected() {
        let op = IrOpKind::FusedElementwise {
            ops: vec![IrOpKind::Add, IrOpKind::Relu],
        };
        let result = map_ir_op(&op, "test", ONNX_FLOAT);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("un-fused"));
    }

    // -----------------------------------------------------------------------
    // Test: ReduceSum has keepdims=0 attribute
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_sum_attributes() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2, 3]);
        let (_, outs) = g.add_node(IrOpKind::Sum, vec![x], vec![vec![]]);
        g.set_outputs(vec![outs[0]]);

        let config = OnnxExportConfig::default();
        let bytes = ir_graph_to_onnx(&g, &config, ONNX_FLOAT).unwrap();

        let as_str = String::from_utf8_lossy(&bytes);
        assert!(as_str.contains("ReduceSum"), "should contain ReduceSum op");
        assert!(
            as_str.contains("keepdims"),
            "should contain keepdims attribute"
        );
    }

    // -----------------------------------------------------------------------
    // Test: write to file and verify file exists
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_onnx_file() {
        let graph = make_add_relu_graph();
        let config = OnnxExportConfig {
            opset_version: 17,
            model_name: "file_test".into(),
            ..Default::default()
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_onnx_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.onnx");

        export_ir_graph_to_onnx(&graph, &path, config, ONNX_FLOAT).unwrap();

        assert!(path.exists());
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0, "file should not be empty");

        // Read back and verify the start.
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[0], 0x08); // field 1, varint
        assert_eq!(bytes[1], 0x08); // ir_version = 8

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: custom model name
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_model_name() {
        let graph = make_add_graph();
        let config = OnnxExportConfig {
            opset_version: 17,
            model_name: "my_custom_model".into(),
            ..Default::default()
        };
        let bytes = ir_graph_to_onnx(&graph, &config, ONNX_FLOAT).unwrap();

        assert!(
            bytes
                .windows(b"my_custom_model".len())
                .any(|w| w == b"my_custom_model"),
            "should contain custom model name"
        );
    }

    // -----------------------------------------------------------------------
    // Test: higher opset version is accepted
    // -----------------------------------------------------------------------

    #[test]
    fn test_opset_version_21_accepted() {
        let graph = make_add_graph();
        let config = OnnxExportConfig {
            opset_version: 21,
            model_name: "test".into(),
            ..Default::default()
        };

        let dir = std::env::temp_dir().join("ferrotorch_test_onnx_opset21");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("opset21.onnx");

        let result = export_ir_graph_to_onnx(&graph, &path, config, ONNX_FLOAT);
        assert!(result.is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    // export_from_program tests disabled — ExportedProgram API was rewritten.
    // Re-enable when export_from_program is re-implemented.
    /*
        #[test]
        fn test_export_from_program_static() {
            use ferrotorch_jit::export::{
                DType, DimSpec, ExportMetadata, ExportedProgram, InputSpec, OutputSpec,
            };

            // Build a simple program: y = a + b
            let mut g = IrGraph::new();
            let a = g.add_input(vec![2, 3]);
            let b = g.add_input(vec![2, 3]);
            let (_, outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![2, 3]]);
            g.set_outputs(vec![outs[0]]);

            let program = ExportedProgram::from_parts(
                g,
                vec![
                    InputSpec {
                        name: "a".into(),
                        shape: vec![DimSpec::Static(2), DimSpec::Static(3)],
                        dtype: DType::Float32,
                    },
                    InputSpec {
                        name: "b".into(),
                        shape: vec![DimSpec::Static(2), DimSpec::Static(3)],
                        dtype: DType::Float32,
                    },
                ],
                vec![OutputSpec {
                    name: "out".into(),
                    shape: vec![2, 3],
                    dtype: DType::Float32,
                }],
                std::collections::HashMap::new(),
                Vec::new(),
                ExportMetadata::default(),
            );

            let dir = std::env::temp_dir().join("ferrotorch_test_onnx_from_prog");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("from_program.onnx");

            export_from_program(&program, &path).unwrap();
            assert!(path.exists());

            let bytes = std::fs::read(&path).unwrap();
            let as_str = String::from_utf8_lossy(&bytes);
            assert!(as_str.contains("Add"), "should contain Add op");

            std::fs::remove_dir_all(&dir).ok();
        }

        // -----------------------------------------------------------------------
        // Test: export_from_program preserves dynamic dims as dim_param
        // -----------------------------------------------------------------------

        #[test]
        fn test_export_from_program_dynamic_dims() {
            use ferrotorch_jit::export::{
                DType, DimSpec, ExportMetadata, ExportedProgram, InputSpec, OutputSpec,
            };

            let mut g = IrGraph::new();
            let a = g.add_input(vec![4, 3]);
            let b = g.add_input(vec![4, 3]);
            let (_, outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![4, 3]]);
            g.set_outputs(vec![outs[0]]);

            let program = ExportedProgram::from_parts(
                g,
                vec![
                    InputSpec {
                        name: "a".into(),
                        shape: vec![
                            DimSpec::Dynamic {
                                name: "batch".into(),
                                min: 1,
                                max: 64,
                            },
                            DimSpec::Static(3),
                        ],
                        dtype: DType::Float32,
                    },
                    InputSpec {
                        name: "b".into(),
                        shape: vec![
                            DimSpec::Dynamic {
                                name: "batch".into(),
                                min: 1,
                                max: 64,
                            },
                            DimSpec::Static(3),
                        ],
                        dtype: DType::Float32,
                    },
                ],
                vec![OutputSpec {
                    name: "out".into(),
                    shape: vec![4, 3],
                    dtype: DType::Float32,
                }],
                std::collections::HashMap::new(),
                Vec::new(),
                ExportMetadata::default(),
            );

            let dir = std::env::temp_dir().join("ferrotorch_test_onnx_from_prog_dyn");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("dynamic.onnx");

            export_from_program(&program, &path).unwrap();

            let bytes = std::fs::read(&path).unwrap();
            // The dynamic dim name "batch" should appear in the output.
            assert!(
                bytes.windows(5).any(|w| w == b"batch"),
                "should contain dynamic dim name 'batch'"
            );

            std::fs::remove_dir_all(&dir).ok();
        }
    */
}
