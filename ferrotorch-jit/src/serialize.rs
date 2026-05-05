//! Binary serialization and deserialization for [`IrGraph`].
//!
//! Uses a simple custom binary format (no external dependencies) with
//! little-endian byte order throughout.
//!
//! # Binary format
//!
//! | Offset | Field | Type |
//! |--------|-------|------|
//! | 0 | Magic bytes `b"FTIR"` | `[u8; 4]` |
//! | 4 | Version (currently 2) | `u32` |
//! | 8 | Value count | `u32` |
//! | ... | Values (variable) | see below |
//! | ... | Node count | `u32` |
//! | ... | Nodes (variable) | see below |
//! | ... | Input value count | `u32` |
//! | ... | Input value ids | `u32` each |
//! | ... | Output value count | `u32` |
//! | ... | Output value ids | `u32` each |
//!
//! # Versioning
//!
//! - **v1** — original format; values had `id`, `shape`, `producer` only.
//!   No dtype was encoded; readers default v1 values to [`Dtype::F32`].
//! - **v2** — adds a single dtype tag byte after each value's `producer`
//!   field (`0` = `F32`, `1` = `F64`). All other fields are unchanged
//!   relative to v1.

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};

use crate::graph::{Dtype, IrGraph, IrNode, IrNodeId, IrOpKind, IrValue, IrValueId};

// ---------------------------------------------------------------------------
// Magic & version
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"FTIR";
/// Current writer version (v2 carries dtype). Readers also accept v1 by
/// defaulting every value's dtype to [`Dtype::F32`].
const VERSION: u32 = 2;
const VERSION_V1: u32 = 1;

// ---------------------------------------------------------------------------
// Dtype tag bytes (v2+)
// ---------------------------------------------------------------------------

const DTYPE_TAG_F32: u8 = 0;
const DTYPE_TAG_F64: u8 = 1;

fn dtype_to_tag(d: Dtype) -> u8 {
    match d {
        Dtype::F32 => DTYPE_TAG_F32,
        Dtype::F64 => DTYPE_TAG_F64,
    }
}

fn tag_to_dtype(tag: u8) -> FerrotorchResult<Dtype> {
    match tag {
        DTYPE_TAG_F32 => Ok(Dtype::F32),
        DTYPE_TAG_F64 => Ok(Dtype::F64),
        other => Err(FerrotorchError::InvalidArgument {
            message: format!("IR deserialize: unknown dtype tag {other}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Op-kind tag bytes
// ---------------------------------------------------------------------------

const TAG_INPUT: u8 = 0;
const TAG_CONSTANT: u8 = 1;
const TAG_OUTPUT: u8 = 2;

const TAG_ADD: u8 = 10;
const TAG_SUB: u8 = 11;
const TAG_MUL: u8 = 12;
const TAG_DIV: u8 = 13;
const TAG_NEG: u8 = 14;
const TAG_POW: u8 = 15;
const TAG_SQRT: u8 = 16;
const TAG_ABS: u8 = 17;
const TAG_EXP: u8 = 18;
const TAG_LOG: u8 = 19;

const TAG_SUM: u8 = 20;
const TAG_MEAN: u8 = 21;
const TAG_PROD: u8 = 22;

const TAG_MATMUL: u8 = 30;
const TAG_MM: u8 = 31;
const TAG_MV: u8 = 32;
const TAG_DOT: u8 = 33;
const TAG_TRANSPOSE: u8 = 34;
const TAG_LINEAR: u8 = 35;

const TAG_RELU: u8 = 40;
const TAG_SIGMOID: u8 = 41;
const TAG_TANH: u8 = 42;
const TAG_GELU: u8 = 43;
const TAG_SILU: u8 = 44;
const TAG_SOFTMAX: u8 = 45;
const TAG_LOG_SOFTMAX: u8 = 46;

const TAG_RESHAPE: u8 = 50;
const TAG_FLATTEN: u8 = 51;
const TAG_SQUEEZE: u8 = 52;
const TAG_UNSQUEEZE: u8 = 53;
const TAG_CAT: u8 = 54;

const TAG_FUSED_ELEMENTWISE: u8 = 60;

const TAG_COND: u8 = 70;
const TAG_SCAN: u8 = 71;

// ---------------------------------------------------------------------------
// Writer helpers
// ---------------------------------------------------------------------------

/// A simple write-cursor backed by a `Vec<u8>`.
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_usize_as_u32(&mut self, v: usize) {
        self.write_u32(v as u32);
    }

    fn write_isize_as_i32(&mut self, v: isize) {
        self.buf.extend_from_slice(&(v as i32).to_le_bytes());
    }

    fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

// ---------------------------------------------------------------------------
// Reader helpers
// ---------------------------------------------------------------------------

/// A simple read-cursor over a byte slice.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_exact(&mut self, n: usize) -> FerrotorchResult<&'a [u8]> {
        if self.remaining() < n {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "IR deserialize: unexpected EOF at offset {} (need {} bytes, have {})",
                    self.pos,
                    n,
                    self.remaining()
                ),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> FerrotorchResult<u8> {
        let bytes = self.read_exact(1)?;
        Ok(bytes[0])
    }

    fn read_u32(&mut self) -> FerrotorchResult<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> FerrotorchResult<i32> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> FerrotorchResult<f64> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_u32_as_usize(&mut self) -> FerrotorchResult<usize> {
        self.read_u32().map(|v| v as usize)
    }

    fn read_i32_as_isize(&mut self) -> FerrotorchResult<isize> {
        self.read_i32().map(|v| v as isize)
    }
}

// ---------------------------------------------------------------------------
// Op-kind serialization
// ---------------------------------------------------------------------------

fn write_op_kind(w: &mut Writer, op: &IrOpKind) {
    match op {
        IrOpKind::Input { index } => {
            w.write_u8(TAG_INPUT);
            w.write_usize_as_u32(*index);
        }
        IrOpKind::Constant { data, shape } => {
            w.write_u8(TAG_CONSTANT);
            w.write_usize_as_u32(data.len());
            for &v in data {
                w.write_f64(v);
            }
            w.write_usize_as_u32(shape.len());
            for &s in shape {
                w.write_usize_as_u32(s);
            }
        }
        IrOpKind::Output => w.write_u8(TAG_OUTPUT),

        IrOpKind::Add => w.write_u8(TAG_ADD),
        IrOpKind::Sub => w.write_u8(TAG_SUB),
        IrOpKind::Mul => w.write_u8(TAG_MUL),
        IrOpKind::Div => w.write_u8(TAG_DIV),
        IrOpKind::Neg => w.write_u8(TAG_NEG),
        IrOpKind::Pow { exponent } => {
            w.write_u8(TAG_POW);
            w.write_f64(*exponent);
        }
        IrOpKind::Sqrt => w.write_u8(TAG_SQRT),
        IrOpKind::Abs => w.write_u8(TAG_ABS),
        IrOpKind::Exp => w.write_u8(TAG_EXP),
        IrOpKind::Log => w.write_u8(TAG_LOG),

        IrOpKind::Sum => w.write_u8(TAG_SUM),
        IrOpKind::Mean => w.write_u8(TAG_MEAN),
        IrOpKind::Prod => w.write_u8(TAG_PROD),

        IrOpKind::Matmul => w.write_u8(TAG_MATMUL),
        IrOpKind::Mm => w.write_u8(TAG_MM),
        IrOpKind::Mv => w.write_u8(TAG_MV),
        IrOpKind::Dot => w.write_u8(TAG_DOT),
        IrOpKind::Transpose => w.write_u8(TAG_TRANSPOSE),
        IrOpKind::Linear => w.write_u8(TAG_LINEAR),

        IrOpKind::Relu => w.write_u8(TAG_RELU),
        IrOpKind::Sigmoid => w.write_u8(TAG_SIGMOID),
        IrOpKind::Tanh => w.write_u8(TAG_TANH),
        IrOpKind::Gelu => w.write_u8(TAG_GELU),
        IrOpKind::Silu => w.write_u8(TAG_SILU),
        IrOpKind::Softmax => w.write_u8(TAG_SOFTMAX),
        IrOpKind::LogSoftmax => w.write_u8(TAG_LOG_SOFTMAX),

        IrOpKind::Reshape { shape } => {
            w.write_u8(TAG_RESHAPE);
            w.write_usize_as_u32(shape.len());
            for &s in shape {
                w.write_isize_as_i32(s);
            }
        }
        IrOpKind::Flatten => w.write_u8(TAG_FLATTEN),
        IrOpKind::Squeeze { axis } => {
            w.write_u8(TAG_SQUEEZE);
            w.write_usize_as_u32(*axis);
        }
        IrOpKind::Unsqueeze { axis } => {
            w.write_u8(TAG_UNSQUEEZE);
            w.write_usize_as_u32(*axis);
        }
        IrOpKind::Cat { axis } => {
            w.write_u8(TAG_CAT);
            w.write_usize_as_u32(*axis);
        }

        IrOpKind::Cond => w.write_u8(TAG_COND),
        IrOpKind::Scan => w.write_u8(TAG_SCAN),

        IrOpKind::FusedElementwise { ops } => {
            w.write_u8(TAG_FUSED_ELEMENTWISE);
            w.write_usize_as_u32(ops.len());
            for op in ops {
                write_op_kind(w, op);
            }
        }
        IrOpKind::FusedLinearActivation { activation } => {
            // Tag 0x30 for fused linear+activation
            w.write_u8(0x30);
            write_op_kind(w, activation);
        }
        IrOpKind::FusedAttention { head_dim } => {
            // Tag 0x31 for fused attention
            w.write_u8(0x31);
            w.write_usize_as_u32(*head_dim);
        }
    }
}

fn read_op_kind(r: &mut Reader<'_>) -> FerrotorchResult<IrOpKind> {
    let tag = r.read_u8()?;
    match tag {
        TAG_INPUT => {
            let index = r.read_u32_as_usize()?;
            Ok(IrOpKind::Input { index })
        }
        TAG_CONSTANT => {
            let data_len = r.read_u32_as_usize()?;
            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                data.push(r.read_f64()?);
            }
            let shape_len = r.read_u32_as_usize()?;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                shape.push(r.read_u32_as_usize()?);
            }
            Ok(IrOpKind::Constant { data, shape })
        }
        TAG_OUTPUT => Ok(IrOpKind::Output),

        TAG_ADD => Ok(IrOpKind::Add),
        TAG_SUB => Ok(IrOpKind::Sub),
        TAG_MUL => Ok(IrOpKind::Mul),
        TAG_DIV => Ok(IrOpKind::Div),
        TAG_NEG => Ok(IrOpKind::Neg),
        TAG_POW => {
            let exponent = r.read_f64()?;
            Ok(IrOpKind::Pow { exponent })
        }
        TAG_SQRT => Ok(IrOpKind::Sqrt),
        TAG_ABS => Ok(IrOpKind::Abs),
        TAG_EXP => Ok(IrOpKind::Exp),
        TAG_LOG => Ok(IrOpKind::Log),

        TAG_SUM => Ok(IrOpKind::Sum),
        TAG_MEAN => Ok(IrOpKind::Mean),
        TAG_PROD => Ok(IrOpKind::Prod),

        TAG_MATMUL => Ok(IrOpKind::Matmul),
        TAG_MM => Ok(IrOpKind::Mm),
        TAG_MV => Ok(IrOpKind::Mv),
        TAG_DOT => Ok(IrOpKind::Dot),
        TAG_TRANSPOSE => Ok(IrOpKind::Transpose),
        TAG_LINEAR => Ok(IrOpKind::Linear),

        TAG_RELU => Ok(IrOpKind::Relu),
        TAG_SIGMOID => Ok(IrOpKind::Sigmoid),
        TAG_TANH => Ok(IrOpKind::Tanh),
        TAG_GELU => Ok(IrOpKind::Gelu),
        TAG_SILU => Ok(IrOpKind::Silu),
        TAG_SOFTMAX => Ok(IrOpKind::Softmax),
        TAG_LOG_SOFTMAX => Ok(IrOpKind::LogSoftmax),

        TAG_RESHAPE => {
            let shape_len = r.read_u32_as_usize()?;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                shape.push(r.read_i32_as_isize()?);
            }
            Ok(IrOpKind::Reshape { shape })
        }
        TAG_FLATTEN => Ok(IrOpKind::Flatten),
        TAG_SQUEEZE => {
            let axis = r.read_u32_as_usize()?;
            Ok(IrOpKind::Squeeze { axis })
        }
        TAG_UNSQUEEZE => {
            let axis = r.read_u32_as_usize()?;
            Ok(IrOpKind::Unsqueeze { axis })
        }
        TAG_CAT => {
            let axis = r.read_u32_as_usize()?;
            Ok(IrOpKind::Cat { axis })
        }

        TAG_COND => Ok(IrOpKind::Cond),
        TAG_SCAN => Ok(IrOpKind::Scan),

        TAG_FUSED_ELEMENTWISE => {
            let count = r.read_u32_as_usize()?;
            let mut ops = Vec::with_capacity(count);
            for _ in 0..count {
                ops.push(read_op_kind(r)?);
            }
            Ok(IrOpKind::FusedElementwise { ops })
        }

        other => Err(FerrotorchError::InvalidArgument {
            message: format!("IR deserialize: unknown op-kind tag {other}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// IrGraph serialization
// ---------------------------------------------------------------------------

impl IrGraph {
    /// Serialize this IR graph to a binary byte vector.
    ///
    /// The format is self-contained and can be written to a file or sent over
    /// the network. Use [`IrGraph::deserialize`] to reconstruct the graph.
    pub fn serialize(&self) -> Vec<u8> {
        let mut w = Writer::new();

        // Header.
        w.write_bytes(MAGIC);
        w.write_u32(VERSION);

        // Values.
        w.write_usize_as_u32(self.values.len());
        for val in &self.values {
            w.write_usize_as_u32(val.id.0);
            w.write_usize_as_u32(val.shape.len());
            for &s in &val.shape {
                w.write_usize_as_u32(s);
            }
            match val.producer {
                Some(node_id) => {
                    w.write_u8(1);
                    w.write_usize_as_u32(node_id.0);
                }
                None => {
                    w.write_u8(0);
                }
            }
            // v2: dtype tag byte.
            w.write_u8(dtype_to_tag(val.dtype));
        }

        // Nodes.
        w.write_usize_as_u32(self.nodes.len());
        for node in &self.nodes {
            w.write_usize_as_u32(node.id.0);
            write_op_kind(&mut w, &node.op);
            w.write_usize_as_u32(node.inputs.len());
            for &inp in &node.inputs {
                w.write_usize_as_u32(inp.0);
            }
            w.write_usize_as_u32(node.outputs.len());
            for &out in &node.outputs {
                w.write_usize_as_u32(out.0);
            }
        }

        // Graph-level inputs.
        w.write_usize_as_u32(self.input_values.len());
        for &iv in &self.input_values {
            w.write_usize_as_u32(iv.0);
        }

        // Graph-level outputs.
        w.write_usize_as_u32(self.output_values.len());
        for &ov in &self.output_values {
            w.write_usize_as_u32(ov.0);
        }

        w.into_vec()
    }

    /// Deserialize an IR graph from a binary byte slice previously produced by
    /// [`IrGraph::serialize`].
    pub fn deserialize(data: &[u8]) -> FerrotorchResult<IrGraph> {
        let mut r = Reader::new(data);

        // Header.
        let magic = r.read_exact(4)?;
        if magic != MAGIC {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "IR deserialize: invalid magic bytes {magic:?} (expected {MAGIC:?})"
                ),
            });
        }

        let version = r.read_u32()?;
        if version != VERSION && version != VERSION_V1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "IR deserialize: unsupported version {version} \
                     (this build accepts {VERSION_V1} or {VERSION})"
                ),
            });
        }
        let has_dtype = version >= VERSION;

        // Values.
        let value_count = r.read_u32_as_usize()?;
        let mut values = Vec::with_capacity(value_count);
        let mut max_value_id: usize = 0;
        for _ in 0..value_count {
            let id = r.read_u32_as_usize()?;
            if id >= max_value_id {
                max_value_id = id + 1;
            }
            let shape_len = r.read_u32_as_usize()?;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                shape.push(r.read_u32_as_usize()?);
            }
            let has_producer = r.read_u8()?;
            let producer = if has_producer != 0 {
                Some(IrNodeId(r.read_u32_as_usize()?))
            } else {
                None
            };
            // v2 carries an explicit dtype tag; v1 streams default to F32.
            let dtype = if has_dtype {
                tag_to_dtype(r.read_u8()?)?
            } else {
                Dtype::F32
            };
            values.push(IrValue {
                id: IrValueId(id),
                shape,
                producer,
                dtype,
            });
        }

        // Nodes.
        let node_count = r.read_u32_as_usize()?;
        let mut nodes = Vec::with_capacity(node_count);
        let mut max_node_id: usize = 0;
        for _ in 0..node_count {
            let id = r.read_u32_as_usize()?;
            if id >= max_node_id {
                max_node_id = id + 1;
            }
            let op = read_op_kind(&mut r)?;
            let input_count = r.read_u32_as_usize()?;
            let mut inputs = Vec::with_capacity(input_count);
            for _ in 0..input_count {
                inputs.push(IrValueId(r.read_u32_as_usize()?));
            }
            let output_count = r.read_u32_as_usize()?;
            let mut outputs = Vec::with_capacity(output_count);
            for _ in 0..output_count {
                outputs.push(IrValueId(r.read_u32_as_usize()?));
            }
            nodes.push(IrNode {
                id: IrNodeId(id),
                op,
                inputs,
                outputs,
            });
        }

        // Graph-level inputs.
        let input_values_count = r.read_u32_as_usize()?;
        let mut input_values = Vec::with_capacity(input_values_count);
        for _ in 0..input_values_count {
            input_values.push(IrValueId(r.read_u32_as_usize()?));
        }

        // Graph-level outputs.
        let output_values_count = r.read_u32_as_usize()?;
        let mut output_values = Vec::with_capacity(output_values_count);
        for _ in 0..output_values_count {
            output_values.push(IrValueId(r.read_u32_as_usize()?));
        }

        Ok(IrGraph {
            nodes,
            values,
            input_values,
            output_values,
            next_value_id: max_value_id,
            next_node_id: max_node_id,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple graph: input -> add(input, input) -> relu -> output,
    /// serialize and deserialize, then verify counts and structure.
    #[test]
    fn test_round_trip_simple_graph() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![2, 3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![2, 3]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![2, 3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let bytes = g.serialize();
        let g2 = IrGraph::deserialize(&bytes).expect("round-trip failed");

        assert_eq!(g2.node_count(), g.node_count());
        assert_eq!(g2.value_count(), g.value_count());
        assert_eq!(g2.input_values.len(), g.input_values.len());
        assert_eq!(g2.output_values.len(), g.output_values.len());

        // Verify that op kinds match.
        for (orig, deser) in g.nodes.iter().zip(g2.nodes.iter()) {
            assert_eq!(orig.id.0, deser.id.0);
            assert_eq!(orig.op, deser.op);
            assert_eq!(orig.inputs.len(), deser.inputs.len());
            assert_eq!(orig.outputs.len(), deser.outputs.len());
        }

        // Verify value shapes match.
        for (orig, deser) in g.values.iter().zip(g2.values.iter()) {
            assert_eq!(orig.id.0, deser.id.0);
            assert_eq!(orig.shape, deser.shape);
            assert_eq!(orig.producer.map(|p| p.0), deser.producer.map(|p| p.0));
        }
    }

    /// Constants must preserve their data values across round-trip.
    #[test]
    #[allow(clippy::approx_constant)] // 3.14159 is an arbitrary round-trip value, not π.
    fn test_round_trip_constants() {
        let mut g = IrGraph::new();

        let data = vec![1.5, -2.7, 3.14159, 0.0, f64::INFINITY];
        let shape = vec![5];
        let c = g.add_constant(data.clone(), shape.clone());
        g.set_outputs(vec![c]);

        let bytes = g.serialize();
        let g2 = IrGraph::deserialize(&bytes).expect("round-trip failed");

        // Find the Constant node in the deserialized graph.
        let const_node = g2
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Constant { .. }))
            .expect("no Constant node found");

        match &const_node.op {
            IrOpKind::Constant {
                data: d2,
                shape: s2,
            } => {
                assert_eq!(d2.len(), data.len());
                for (a, b) in d2.iter().zip(data.iter()) {
                    // Use to_bits for exact comparison (handles NaN, Inf).
                    assert_eq!(a.to_bits(), b.to_bits(), "constant data mismatch");
                }
                assert_eq!(s2, &shape);
            }
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    /// Invalid magic bytes must produce an error.
    #[test]
    fn test_invalid_magic_bytes() {
        let bad = b"NOPE____extra";
        let result = IrGraph::deserialize(bad);
        assert!(result.is_err());

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("invalid magic bytes"),
            "unexpected error message: {err_msg}"
        );
    }

    /// An empty graph should round-trip correctly.
    #[test]
    fn test_round_trip_empty_graph() {
        let g = IrGraph::new();

        let bytes = g.serialize();
        let g2 = IrGraph::deserialize(&bytes).expect("round-trip failed");

        assert_eq!(g2.node_count(), 0);
        assert_eq!(g2.value_count(), 0);
        assert_eq!(g2.input_values.len(), 0);
        assert_eq!(g2.output_values.len(), 0);
    }

    /// Round-trip a graph containing every op-kind variant to exercise all
    /// serialization arms.
    #[test]
    fn test_round_trip_all_op_kinds() {
        let ops: Vec<IrOpKind> = vec![
            IrOpKind::Input { index: 0 },
            IrOpKind::Constant {
                data: vec![1.0, 2.0],
                shape: vec![2],
            },
            IrOpKind::Output,
            IrOpKind::Add,
            IrOpKind::Sub,
            IrOpKind::Mul,
            IrOpKind::Div,
            IrOpKind::Neg,
            IrOpKind::Pow { exponent: 2.5 },
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
            IrOpKind::Linear,
            IrOpKind::Relu,
            IrOpKind::Sigmoid,
            IrOpKind::Tanh,
            IrOpKind::Gelu,
            IrOpKind::Silu,
            IrOpKind::Softmax,
            IrOpKind::LogSoftmax,
            IrOpKind::Reshape { shape: vec![-1, 3] },
            IrOpKind::Flatten,
            IrOpKind::Squeeze { axis: 1 },
            IrOpKind::Unsqueeze { axis: 0 },
            IrOpKind::Cat { axis: 2 },
            IrOpKind::FusedElementwise {
                ops: vec![IrOpKind::Add, IrOpKind::Relu],
            },
        ];

        // Build a graph with one node per op kind. Each node gets a
        // dummy output value so the graph is well-formed enough to
        // serialize.
        let mut g = IrGraph::new();
        for op in &ops {
            g.add_node(op.clone(), Vec::new(), vec![vec![1]]);
        }

        let bytes = g.serialize();
        let g2 = IrGraph::deserialize(&bytes).expect("round-trip failed");

        assert_eq!(g2.node_count(), ops.len());
        for (orig, deser) in g.nodes.iter().zip(g2.nodes.iter()) {
            assert_eq!(orig.op, deser.op, "op mismatch for node {}", orig.id.0);
        }
    }

    /// Mixed-dtype values must survive a serialize/deserialize round-trip
    /// at v2 (the current writer version).
    #[test]
    fn test_round_trip_dtype_v2() {
        let mut g = IrGraph::new();
        let _x_f32 = g.add_input_with_dtype(vec![4], Dtype::F32);
        let _x_f64 = g.add_input_with_dtype(vec![4], Dtype::F64);
        let _c_f64 = g.add_constant_with_dtype(vec![1.0; 4], vec![4], Dtype::F64);

        let bytes = g.serialize();
        let g2 = IrGraph::deserialize(&bytes).expect("round-trip failed");

        assert_eq!(g2.value_count(), g.value_count());
        for (orig, deser) in g.values.iter().zip(g2.values.iter()) {
            assert_eq!(
                orig.dtype, deser.dtype,
                "dtype mismatch for value id {}",
                orig.id.0
            );
        }
    }

    /// A v1-formatted byte stream (no dtype byte) must still deserialize
    /// successfully, with every value defaulting to `Dtype::F32`.
    #[test]
    fn test_v1_backward_compat_defaults_f32() {
        // Hand-craft a minimal v1 stream: magic + version=1 + value_count=1 +
        // (id=0, shape_len=1, shape=[2], has_producer=0)
        // + node_count=0 + input_count=0 + output_count=0.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION_V1.to_le_bytes());
        // value_count = 1
        buf.extend_from_slice(&1u32.to_le_bytes());
        // id = 0
        buf.extend_from_slice(&0u32.to_le_bytes());
        // shape_len = 1
        buf.extend_from_slice(&1u32.to_le_bytes());
        // shape[0] = 2
        buf.extend_from_slice(&2u32.to_le_bytes());
        // has_producer = 0 (no producer)
        buf.push(0);
        // node_count = 0
        buf.extend_from_slice(&0u32.to_le_bytes());
        // input_count = 0
        buf.extend_from_slice(&0u32.to_le_bytes());
        // output_count = 0
        buf.extend_from_slice(&0u32.to_le_bytes());

        let g = IrGraph::deserialize(&buf).expect("v1 deserialize failed");
        assert_eq!(g.value_count(), 1);
        assert_eq!(g.values[0].dtype, Dtype::F32);
        assert_eq!(g.values[0].shape, vec![2]);
    }

    /// An unknown dtype tag must surface as an error rather than a panic.
    #[test]
    fn test_unknown_dtype_tag_errors() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // value_count = 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // id
        buf.extend_from_slice(&0u32.to_le_bytes()); // shape_len = 0
        buf.push(0); // has_producer = 0
        buf.push(0xFF); // bogus dtype tag

        let result = IrGraph::deserialize(&buf);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("unknown dtype tag"),
            "unexpected error message: {msg}"
        );
    }

    /// Truncated data must produce an error (not a panic).
    #[test]
    fn test_truncated_data() {
        let g = IrGraph::new();
        let bytes = g.serialize();

        // Chop off at various points.
        for truncated_len in [0, 1, 4, 7] {
            if truncated_len < bytes.len() {
                let result = IrGraph::deserialize(&bytes[..truncated_len]);
                assert!(
                    result.is_err(),
                    "expected error for truncated_len={truncated_len}"
                );
            }
        }
    }
}
