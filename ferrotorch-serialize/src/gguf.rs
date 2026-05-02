//! GGUF (GGML Universal Format) parser for loading llama.cpp quantized models.
//!
//! GGUF is the standard binary format used by llama.cpp for quantized LLM
//! weights. This module implements:
//!
//! 1. Header and metadata parsing (version 3).
//! 2. Tensor info extraction (name, shape, quantization type, data offset).
//! 3. Dequantization of Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, F16, and F32
//!    tensor data to `f32`.
//! 4. A public [`load_gguf_state_dict`] function that produces a
//!    `StateDict<f32>`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_nn::StateDict;
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GGUF magic number: "GGUF" in little-endian = 0x46475547.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default alignment for the data section (GGUF v3).
const DEFAULT_ALIGNMENT: usize = 32;

// ---------------------------------------------------------------------------
// GGML quantization types
// ---------------------------------------------------------------------------

/// Quantization types defined by the GGML format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 and 5 are deprecated / unused
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
}

impl GgmlType {
    /// Parse a `u32` into a `GgmlType`.
    fn from_u32(v: u32) -> FerrotorchResult<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            other => Err(gguf_err(&format!("unsupported GGML type: {other}"))),
        }
    }

    /// Block size (number of elements per quantization block).
    fn block_size(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
        }
    }

    /// Byte size of a single quantization block.
    fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18, // 2 (f16 scale) + 16 (32 nibbles)
            Self::Q4_1 => 20, // 2 (f16 scale) + 2 (f16 min) + 16 (32 nibbles)
            Self::Q5_0 => 22, // 2 (f16 scale) + 4 (high bits) + 16 (32 nibbles low)
            Self::Q5_1 => 24, // 2 (f16 scale) + 2 (f16 min) + 4 (high bits) + 16
            Self::Q8_0 => 34, // 2 (f16 scale) + 32 (int8 values)
            Self::Q8_1 => 40, // 4 (f32 scale) + 4 (f32 min) + 32 (int8 values)
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF value types
// ---------------------------------------------------------------------------

/// GGUF metadata value type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> FerrotorchResult<Self> {
        match v {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(gguf_err(&format!("unsupported GGUF value type: {other}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A single metadata value in a GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// Parsed metadata section of a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub entries: HashMap<String, GgufValue>,
}

/// Information about a single tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

/// A fully parsed GGUF file: header, metadata, tensor descriptors, and raw data.
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// GGUF format version (currently 3).
    pub version: u32,
    /// Key-value metadata (model architecture, tokenizer, quantization info, etc.).
    pub metadata: GgufMetadata,
    /// Tensor descriptors (name, shape, type, offset into data section).
    pub tensors: Vec<GgufTensorInfo>,
    /// Raw data section bytes (tensor data lives here at their declared offsets).
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Binary reader helper
// ---------------------------------------------------------------------------

/// Cursor into a byte slice for sequential little-endian reads.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> FerrotorchResult<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(gguf_err(&format!(
                "unexpected end of data at offset {} (need {n} bytes, {} remaining)",
                self.pos,
                self.remaining()
            )));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> FerrotorchResult<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> FerrotorchResult<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> FerrotorchResult<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> FerrotorchResult<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> FerrotorchResult<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> FerrotorchResult<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32(&mut self) -> FerrotorchResult<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> FerrotorchResult<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> FerrotorchResult<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f64(&mut self) -> FerrotorchResult<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a GGUF string: u64 length followed by that many UTF-8 bytes.
    fn read_gguf_string(&mut self) -> FerrotorchResult<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|_| gguf_err("non-UTF-8 string in GGUF metadata"))
    }

    /// Align position to the given boundary.
    fn align_to(&mut self, alignment: usize) {
        let rem = self.pos % alignment;
        if rem != 0 {
            self.pos += alignment - rem;
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Read a single GGUF metadata value of the given type.
fn read_gguf_value(reader: &mut Reader<'_>, vtype: GgufValueType) -> FerrotorchResult<GgufValue> {
    match vtype {
        GgufValueType::Uint8 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        GgufValueType::Int8 => Ok(GgufValue::Int8(reader.read_i8()?)),
        GgufValueType::Uint16 => Ok(GgufValue::Uint16(reader.read_u16()?)),
        GgufValueType::Int16 => Ok(GgufValue::Int16(reader.read_i16()?)),
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(reader.read_u32()?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(reader.read_i32()?)),
        GgufValueType::Float32 => Ok(GgufValue::Float32(reader.read_f32()?)),
        GgufValueType::Bool => {
            let v = reader.read_u8()?;
            Ok(GgufValue::Bool(v != 0))
        }
        GgufValueType::String => Ok(GgufValue::String(reader.read_gguf_string()?)),
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(reader.read_u64()?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(reader.read_i64()?)),
        GgufValueType::Float64 => Ok(GgufValue::Float64(reader.read_f64()?)),
        GgufValueType::Array => {
            let elem_type = GgufValueType::from_u32(reader.read_u32()?)?;
            let count = reader.read_u64()? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_gguf_value(reader, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
    }
}

/// Load and parse a GGUF file from disk.
///
/// Returns a [`GgufFile`] containing the parsed header, metadata, tensor
/// descriptors, and the raw data section.
///
/// # Errors
///
/// Returns an error if the file cannot be read, has an invalid magic number,
/// or is otherwise malformed.
pub fn load_gguf(path: impl AsRef<Path>) -> FerrotorchResult<GgufFile> {
    let path = path.as_ref();
    let file_bytes = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read GGUF file {}: {e}", path.display()),
    })?;

    parse_gguf_bytes(&file_bytes)
}

/// Memory-mapped GGUF loader (#609). Mirrors [`load_gguf`] but uses
/// `mmap2::Mmap` instead of `std::fs::read` to avoid allocating a heap
/// buffer for the whole file before parsing.
///
/// On large checkpoints (e.g. a 20 GB Llama 70B GGUF) this halves peak
/// resident-set size at load time: instead of `[file bytes Vec][data
/// section Vec][decoded tensors]` we have `[mmap region][data section
/// Vec][decoded tensors]`, with the mmap region only paged in on demand
/// and dropped before this function returns.
///
/// # Safety
///
/// The mmap is dropped before the function returns. The header /
/// metadata / tensor-infos are parsed in place from the mmap region; the
/// data section is copied into the returned `GgufFile.data: Vec<u8>` so
/// the GgufFile is fully owned and outlives the mmap. The file must not
/// be mutated while the mmap is live.
pub fn load_gguf_mmap(path: impl AsRef<Path>) -> FerrotorchResult<GgufFile> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open GGUF file {}: {e}", path.display()),
    })?;
    // SAFETY: the mmap is dropped before this function returns. The
    // returned `GgufFile.data` is a `Vec<u8>` populated from the mmap
    // region inside `parse_gguf_bytes`, so no borrowed bytes escape.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to mmap GGUF file {}: {e}", path.display()),
    })?;
    parse_gguf_bytes(&mmap[..])
}

/// Parse GGUF from an in-memory byte slice.
///
/// This is the core parsing routine used by both [`load_gguf`] and the test
/// suite.
pub fn parse_gguf_bytes(data: &[u8]) -> FerrotorchResult<GgufFile> {
    let mut r = Reader::new(data);

    // -- Header --
    let magic = r.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(gguf_err(&format!(
            "invalid GGUF magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
        )));
    }

    let version = r.read_u32()?;
    if !(2..=3).contains(&version) {
        return Err(gguf_err(&format!(
            "unsupported GGUF version {version} (expected 2 or 3)"
        )));
    }

    let tensor_count = r.read_u64()? as usize;
    let metadata_kv_count = r.read_u64()? as usize;

    // -- Metadata --
    let mut entries = HashMap::with_capacity(metadata_kv_count);
    for _ in 0..metadata_kv_count {
        let key = r.read_gguf_string()?;
        let vtype = GgufValueType::from_u32(r.read_u32()?)?;
        let value = read_gguf_value(&mut r, vtype)?;
        entries.insert(key, value);
    }

    // Check for custom alignment in metadata.
    let alignment = match entries.get("general.alignment") {
        Some(GgufValue::Uint32(a)) => *a as usize,
        Some(GgufValue::Uint64(a)) => *a as usize,
        _ => DEFAULT_ALIGNMENT,
    };

    // -- Tensor infos --
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.read_gguf_string()?;
        let n_dims = r.read_u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.read_u64()?);
        }
        let ggml_type = GgmlType::from_u32(r.read_u32()?)?;
        let offset = r.read_u64()?;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    // -- Alignment padding --
    r.align_to(alignment);

    // -- Data section --
    let data_start = r.pos;
    let data = data[data_start..].to_vec();

    Ok(GgufFile {
        version,
        metadata: GgufMetadata { entries },
        tensors,
        data,
    })
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16 -> normal f32.
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = ((e + 127) as u32) & 0xff;
            let f32_mantissa = m << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
        }
    } else if exponent == 31 {
        if mantissa == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
        }
    } else {
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        let f32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
    }
}

/// Dequantize Q4_0 data to f32.
///
/// Q4_0: block size 32, each block = 2 bytes (f16 scale) + 16 bytes (32 nibbles).
/// `dequant(nibble) = (nibble - 8) * scale`
fn dequantize_q4_0(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let num_blocks = num_elements / BLOCK_SIZE;

    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(gguf_err(&format!(
            "Q4_0: need {} bytes for {num_blocks} blocks, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let scale = f16_to_f32(data[block_start], data[block_start + 1]);

        for j in 0..16 {
            let byte = data[block_start + 2 + j];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output.push(lo * scale);
            output.push(hi * scale);
        }
    }

    Ok(output)
}

/// Dequantize Q4_1 data to f32.
///
/// Q4_1: block size 32, each block = 2 bytes (f16 scale) + 2 bytes (f16 min) +
/// 16 bytes (32 nibbles).
/// `dequant(nibble) = nibble * scale + min`
fn dequantize_q4_1(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;
    let num_blocks = num_elements / BLOCK_SIZE;

    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(gguf_err(&format!(
            "Q4_1: need {} bytes for {num_blocks} blocks, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let scale = f16_to_f32(data[block_start], data[block_start + 1]);
        let min = f16_to_f32(data[block_start + 2], data[block_start + 3]);

        for j in 0..16 {
            let byte = data[block_start + 4 + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            output.push(lo * scale + min);
            output.push(hi * scale + min);
        }
    }

    Ok(output)
}

/// Dequantize Q5_0 data to f32.
///
/// Q5_0: block size 32, each block = 2 bytes (f16 scale) + 4 bytes (32 high bits)
/// + 16 bytes (32 low nibbles). Each element is 5 bits: 4 low bits from nibble
/// + 1 high bit from the bit field. `dequant(val5) = (val5 - 16) * scale`
fn dequantize_q5_0(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 22;
    let num_blocks = num_elements / BLOCK_SIZE;

    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(gguf_err(&format!(
            "Q5_0: need {} bytes for {num_blocks} blocks, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let scale = f16_to_f32(data[block_start], data[block_start + 1]);

        // 4 bytes of high bits, packed as a u32.
        let qh = u32::from_le_bytes([
            data[block_start + 2],
            data[block_start + 3],
            data[block_start + 4],
            data[block_start + 5],
        ]);

        for j in 0..16 {
            let byte = data[block_start + 6 + j];
            let lo_nibble = (byte & 0x0F) as u32;
            let hi_nibble = ((byte >> 4) & 0x0F) as u32;

            let lo_high_bit = (qh >> (j * 2)) & 1;
            let hi_high_bit = (qh >> (j * 2 + 1)) & 1;

            let lo_val = (lo_nibble | (lo_high_bit << 4)) as f32 - 16.0;
            let hi_val = (hi_nibble | (hi_high_bit << 4)) as f32 - 16.0;

            output.push(lo_val * scale);
            output.push(hi_val * scale);
        }
    }

    Ok(output)
}

/// Dequantize Q5_1 data to f32.
///
/// Q5_1: block size 32, each block = 2 bytes (f16 scale) + 2 bytes (f16 min) +
/// 4 bytes (32 high bits) + 16 bytes (32 low nibbles).
/// `dequant(val5) = val5 * scale + min`
fn dequantize_q5_1(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 24;
    let num_blocks = num_elements / BLOCK_SIZE;

    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(gguf_err(&format!(
            "Q5_1: need {} bytes for {num_blocks} blocks, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let scale = f16_to_f32(data[block_start], data[block_start + 1]);
        let min = f16_to_f32(data[block_start + 2], data[block_start + 3]);

        let qh = u32::from_le_bytes([
            data[block_start + 4],
            data[block_start + 5],
            data[block_start + 6],
            data[block_start + 7],
        ]);

        for j in 0..16 {
            let byte = data[block_start + 8 + j];
            let lo_nibble = (byte & 0x0F) as u32;
            let hi_nibble = ((byte >> 4) & 0x0F) as u32;

            let lo_high_bit = (qh >> (j * 2)) & 1;
            let hi_high_bit = (qh >> (j * 2 + 1)) & 1;

            let lo_val = (lo_nibble | (lo_high_bit << 4)) as f32;
            let hi_val = (hi_nibble | (hi_high_bit << 4)) as f32;

            output.push(lo_val * scale + min);
            output.push(hi_val * scale + min);
        }
    }

    Ok(output)
}

/// Dequantize Q8_0 data to f32.
///
/// Q8_0: block size 32, each block = 2 bytes (f16 scale) + 32 bytes (int8 values).
/// `dequant(val) = val * scale`
fn dequantize_q8_0(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let num_blocks = num_elements / BLOCK_SIZE;

    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(gguf_err(&format!(
            "Q8_0: need {} bytes for {num_blocks} blocks, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let scale = f16_to_f32(data[block_start], data[block_start + 1]);

        for j in 0..32 {
            let val = data[block_start + 2 + j] as i8;
            output.push(val as f32 * scale);
        }
    }

    Ok(output)
}

/// Dequantize Q8_1 data to f32.
///
/// Q8_1: block size 32, each block = 4 bytes (f32 scale) + 4 bytes (f32 min) +
/// 32 bytes (int8 values).
/// `dequant(val) = val * scale + min`
fn dequantize_q8_1(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 40;
    let num_blocks = num_elements / BLOCK_SIZE;

    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(gguf_err(&format!(
            "Q8_1: need {} bytes for {num_blocks} blocks, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            data[block_start],
            data[block_start + 1],
            data[block_start + 2],
            data[block_start + 3],
        ]);
        let min = f32::from_le_bytes([
            data[block_start + 4],
            data[block_start + 5],
            data[block_start + 6],
            data[block_start + 7],
        ]);

        for j in 0..32 {
            let val = data[block_start + 8 + j] as i8;
            output.push(val as f32 * scale + min);
        }
    }

    Ok(output)
}

/// Dequantize F16 data to f32.
fn dequantize_f16(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    if data.len() < num_elements * 2 {
        return Err(gguf_err(&format!(
            "F16: need {} bytes for {num_elements} elements, got {}",
            num_elements * 2,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        output.push(f16_to_f32(data[i * 2], data[i * 2 + 1]));
    }
    Ok(output)
}

/// Dequantize F32 data (no-op copy).
fn dequantize_f32(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    if data.len() < num_elements * 4 {
        return Err(gguf_err(&format!(
            "F32: need {} bytes for {num_elements} elements, got {}",
            num_elements * 4,
            data.len()
        )));
    }

    let mut output = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let off = i * 4;
        output.push(f32::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ]));
    }
    Ok(output)
}

/// Dequantize raw GGML tensor data of the given type to `Vec<f32>`.
fn dequantize_data(
    data: &[u8],
    ggml_type: GgmlType,
    num_elements: usize,
) -> FerrotorchResult<Vec<f32>> {
    match ggml_type {
        GgmlType::F32 => dequantize_f32(data, num_elements),
        GgmlType::F16 => dequantize_f16(data, num_elements),
        GgmlType::Q4_0 => dequantize_q4_0(data, num_elements),
        GgmlType::Q4_1 => dequantize_q4_1(data, num_elements),
        GgmlType::Q5_0 => dequantize_q5_0(data, num_elements),
        GgmlType::Q5_1 => dequantize_q5_1(data, num_elements),
        GgmlType::Q8_0 => dequantize_q8_0(data, num_elements),
        GgmlType::Q8_1 => dequantize_q8_1(data, num_elements),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Dequantize a single tensor from a parsed GGUF file to `Tensor<f32>`.
///
/// Looks up the tensor by name, reads its raw bytes from the data section,
/// dequantizes according to the tensor's GGML type, and returns a
/// `Tensor<f32>` with the original shape.
///
/// # Errors
///
/// Returns an error if the tensor name is not found, or the data section is
/// too small for the declared tensor.
pub fn dequantize_gguf_tensor(file: &GgufFile, tensor_name: &str) -> FerrotorchResult<Tensor<f32>> {
    let info = file
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .ok_or_else(|| gguf_err(&format!("tensor \"{tensor_name}\" not found in GGUF file")))?;

    let num_elements: u64 = if info.dims.is_empty() {
        1
    } else {
        info.dims.iter().product()
    };
    let num_elements = num_elements as usize;

    // Compute byte range within the data section.
    let block_size = info.ggml_type.block_size();
    let block_bytes = info.ggml_type.block_bytes();
    let num_blocks = num_elements.div_ceil(block_size);
    let byte_len = num_blocks * block_bytes;
    let offset = info.offset as usize;

    if offset + byte_len > file.data.len() {
        return Err(gguf_err(&format!(
            "tensor \"{}\" requires bytes [{}..{}] but data section has {} bytes",
            info.name,
            offset,
            offset + byte_len,
            file.data.len()
        )));
    }

    let raw = &file.data[offset..offset + byte_len];
    let values = dequantize_data(raw, info.ggml_type, num_elements)?;

    let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    let storage = TensorStorage::cpu(values);
    Tensor::from_storage(storage, shape, false)
}

/// Load all tensors from a GGUF file as a `StateDict<f32>`.
///
/// Each tensor is dequantized to `f32` regardless of its original GGML
/// quantization type. Tensor names are used as state dict keys.
///
/// # Errors
///
/// Returns an error if the file cannot be read or any tensor fails to
/// dequantize.
pub fn load_gguf_state_dict(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<f32>> {
    let file = load_gguf(path)?;
    let mut state: StateDict<f32> = HashMap::with_capacity(file.tensors.len());

    // Collect tensor names first to avoid borrow conflict.
    let names: Vec<String> = file.tensors.iter().map(|t| t.name.clone()).collect();

    for name in &names {
        let tensor = dequantize_gguf_tensor(&file, name)?;
        state.insert(name.clone(), tensor);
    }

    Ok(state)
}

/// Memory-mapped variant of [`load_gguf_state_dict`] (#609). Same return
/// contract but uses [`load_gguf_mmap`] to halve peak RSS during the
/// initial parse.
pub fn load_gguf_state_dict_mmap(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<f32>> {
    let file = load_gguf_mmap(path)?;
    let mut state: StateDict<f32> = HashMap::with_capacity(file.tensors.len());

    let names: Vec<String> = file.tensors.iter().map(|t| t.name.clone()).collect();

    for name in &names {
        let tensor = dequantize_gguf_tensor(&file, name)?;
        state.insert(name.clone(), tensor);
    }

    Ok(state)
}

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn gguf_err(msg: &str) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("GGUF: {msg}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test helpers: build synthetic GGUF bytes --

    /// Write a GGUF string (u64 len + bytes) into a buffer.
    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Write a GGUF value type tag.
    fn write_value_type(buf: &mut Vec<u8>, vtype: u32) {
        buf.extend_from_slice(&vtype.to_le_bytes());
    }

    /// Build a minimal valid GGUF file with given metadata and tensor data.
    ///
    /// Returns the raw bytes that can be parsed with `parse_gguf_bytes`.
    fn build_gguf(
        metadata: &[(&str, u32, &[u8])], // (key, value_type, value_bytes)
        tensors: &[(&str, &[u64], u32, &[u8])], // (name, dims, ggml_type, data)
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // Metadata KV count
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        // Metadata entries
        for (key, vtype, value_bytes) in metadata {
            write_gguf_string(&mut buf, key);
            write_value_type(&mut buf, *vtype);
            buf.extend_from_slice(value_bytes);
        }

        // Compute data offsets and write tensor infos.
        // Each tensor's data follows the previous one contiguously.
        let mut data_offset: u64 = 0;
        for (name, dims, ggml_type, data_bytes) in tensors {
            write_gguf_string(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&ggml_type.to_le_bytes());
            buf.extend_from_slice(&data_offset.to_le_bytes());
            data_offset += data_bytes.len() as u64;
        }

        // Alignment padding to DEFAULT_ALIGNMENT.
        let rem = buf.len() % DEFAULT_ALIGNMENT;
        if rem != 0 {
            buf.resize(buf.len() + (DEFAULT_ALIGNMENT - rem), 0);
        }

        // Data section.
        for (_name, _dims, _ggml_type, data_bytes) in tensors {
            buf.extend_from_slice(data_bytes);
        }

        buf
    }

    /// Encode an f16 value from an f32 (simplified: only handles normals and zero).
    fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mantissa = bits & 0x7f_ffff;

        if val == 0.0 {
            let h = (sign as u16) << 15;
            return h.to_le_bytes();
        }

        let h_exp = (exp + 15).clamp(0, 31) as u16;
        let h_mantissa = (mantissa >> 13) as u16;
        let h = ((sign as u16) << 15) | (h_exp << 10) | h_mantissa;
        h.to_le_bytes()
    }

    // -- Parse tests --

    #[test]
    fn test_invalid_magic() {
        let mut data = vec![0u8; 32];
        // Wrong magic.
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let result = parse_gguf_bytes(&data);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("invalid GGUF magic"), "got: {msg}");
    }

    #[test]
    fn test_parse_empty_gguf() {
        let bytes = build_gguf(&[], &[]);
        let file = parse_gguf_bytes(&bytes).unwrap();
        assert_eq!(file.version, 3);
        assert!(file.metadata.entries.is_empty());
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn test_metadata_string() {
        let mut val_bytes = Vec::new();
        // String value: u64 len + bytes.
        let s = "llama";
        val_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        val_bytes.extend_from_slice(s.as_bytes());

        let bytes = build_gguf(
            &[("general.architecture", 8, &val_bytes)], // 8 = String
            &[],
        );
        let file = parse_gguf_bytes(&bytes).unwrap();
        assert_eq!(
            file.metadata.entries.get("general.architecture"),
            Some(&GgufValue::String("llama".to_string()))
        );
    }

    #[test]
    fn test_metadata_uint32() {
        let val_bytes = 42u32.to_le_bytes();
        let bytes = build_gguf(
            &[("llama.block_count", 4, &val_bytes)], // 4 = Uint32
            &[],
        );
        let file = parse_gguf_bytes(&bytes).unwrap();
        assert_eq!(
            file.metadata.entries.get("llama.block_count"),
            Some(&GgufValue::Uint32(42))
        );
    }

    #[test]
    fn test_metadata_array() {
        // Array of Uint32: type_tag(u32) + count(u64) + elements
        let mut val_bytes = Vec::new();
        val_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type = Uint32
        val_bytes.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        val_bytes.extend_from_slice(&10u32.to_le_bytes());
        val_bytes.extend_from_slice(&20u32.to_le_bytes());
        val_bytes.extend_from_slice(&30u32.to_le_bytes());

        let bytes = build_gguf(
            &[("my.array", 9, &val_bytes)], // 9 = Array
            &[],
        );
        let file = parse_gguf_bytes(&bytes).unwrap();
        assert_eq!(
            file.metadata.entries.get("my.array"),
            Some(&GgufValue::Array(vec![
                GgufValue::Uint32(10),
                GgufValue::Uint32(20),
                GgufValue::Uint32(30),
            ]))
        );
    }

    // -- F32 passthrough --

    #[test]
    fn test_f32_passthrough() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let bytes = build_gguf(
            &[],
            &[("weight", &[4], 0, &data)], // type 0 = F32
        );
        let file = parse_gguf_bytes(&bytes).unwrap();
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "weight");
        assert_eq!(file.tensors[0].dims, vec![4]);

        let tensor = dequantize_gguf_tensor(&file, "weight").unwrap();
        let tensor_data: &[f32] = tensor.data().unwrap();
        assert_eq!(tensor_data, &[1.0, 2.0, 3.0, 4.0]);
    }

    // -- Q4_0 dequantization --

    #[test]
    fn test_dequantize_q4_0_known_values() {
        // Build one Q4_0 block (32 elements):
        //   scale = 1.0 (f16), nibbles encode known values.
        let scale_bytes = f32_to_f16_bytes(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);

        // 16 data bytes: each byte encodes two nibbles.
        // Byte 0x88 => lo=8, hi=8 => (8-8)*1.0=0.0, (8-8)*1.0=0.0
        // Byte 0x97 => lo=7, hi=9 => (7-8)*1.0=-1.0, (9-8)*1.0=1.0
        // Byte 0xA6 => lo=6, hi=10 => (6-8)*1.0=-2.0, (10-8)*1.0=2.0
        block.push(0x88); // elements 0,1 => 0.0, 0.0
        block.push(0x97); // elements 2,3 => -1.0, 1.0
        block.push(0xA6); // elements 4,5 => -2.0, 2.0
        // Fill remaining 13 bytes with 0x88 (all zeros).
        block.extend(std::iter::repeat_n(0x88, 13));

        let result = dequantize_q4_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 1e-3);
        assert!((result[1] - 0.0).abs() < 1e-3);
        assert!((result[2] - -1.0).abs() < 1e-3);
        assert!((result[3] - 1.0).abs() < 1e-3);
        assert!((result[4] - -2.0).abs() < 1e-3);
        assert!((result[5] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_dequantize_q4_0_scale_factor() {
        // scale = 0.5 => values halved.
        let scale_bytes = f32_to_f16_bytes(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        // Byte 0x97 => lo=7, hi=9 => (7-8)*0.5=-0.5, (9-8)*0.5=0.5
        block.push(0x97);
        block.extend(std::iter::repeat_n(0x88, 15));

        let result = dequantize_q4_0(&block, 32).unwrap();
        assert!((result[0] - -0.5).abs() < 1e-3);
        assert!((result[1] - 0.5).abs() < 1e-3);
    }

    // -- Q8_0 dequantization --

    #[test]
    fn test_dequantize_q8_0_identity() {
        // scale = 1.0, int8 values should pass through directly.
        let scale_bytes = f32_to_f16_bytes(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);

        // 32 int8 values: 0, 1, 2, ..., 31 (capped to i8 range).
        for i in 0..32u8 {
            block.push(i);
        }

        let result = dequantize_q8_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        for (i, &r) in result.iter().enumerate().take(32) {
            assert!(
                (r - i as f32).abs() < 1e-3,
                "element {i}: expected {}, got {r}",
                i as f32,
            );
        }
    }

    #[test]
    fn test_dequantize_q8_0_negative_values() {
        let scale_bytes = f32_to_f16_bytes(2.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);

        // -1 as u8 = 0xFF, -2 as u8 = 0xFE, etc.
        block.push(0xFF); // -1
        block.push(0xFE); // -2
        block.extend(std::iter::repeat_n(0x00, 30));

        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!((result[0] - -2.0).abs() < 1e-3); // -1 * 2.0
        assert!((result[1] - -4.0).abs() < 1e-3); // -2 * 2.0
    }

    // -- F16 dequantization --

    #[test]
    fn test_dequantize_f16() {
        let mut data = Vec::new();
        for val in &[0.0f32, 1.0, -1.0, 0.5] {
            data.extend_from_slice(&f32_to_f16_bytes(*val));
        }

        let result = dequantize_f16(&data, 4).unwrap();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.0).abs() < 1e-3);
        assert!((result[1] - 1.0).abs() < 1e-3);
        assert!((result[2] - -1.0).abs() < 1e-3);
        assert!((result[3] - 0.5).abs() < 1e-3);
    }

    // -- Full round-trip: build GGUF, parse, load state dict --

    #[test]
    fn test_load_gguf_state_dict_f32() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let bytes = build_gguf(
            &[],
            &[
                ("layer.weight", &[2, 3], 0, &data), // F32, shape [2, 3]
            ],
        );

        // Write to a temp file, then load.
        let dir = std::env::temp_dir().join("ferrotorch_gguf_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_f32.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let state = load_gguf_state_dict(&path).unwrap();
        assert!(state.contains_key("layer.weight"));
        let tensor = &state["layer.weight"];
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_gguf_mmap_matches_read_path() {
        // Same fixture as test_load_gguf_state_dict_f32 but loaded via the
        // mmap-backed path; values must match byte-for-byte.
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let bytes = build_gguf(&[], &[("layer.weight", &[2, 3], 0, &data)]);

        let dir = std::env::temp_dir().join("ferrotorch_gguf_mmap_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("mmap.gguf");
        std::fs::write(&path, &bytes).unwrap();

        // load_gguf_mmap returns the same parsed file as load_gguf.
        let from_read = load_gguf(&path).unwrap();
        let from_mmap = load_gguf_mmap(&path).unwrap();
        assert_eq!(from_read.tensors.len(), from_mmap.tensors.len());
        assert_eq!(from_read.data, from_mmap.data);

        // Sharded state-dict round-trip is equal too.
        let dict_read = load_gguf_state_dict(&path).unwrap();
        let dict_mmap = load_gguf_state_dict_mmap(&path).unwrap();
        assert_eq!(dict_read.len(), dict_mmap.len());
        for (k, v_read) in &dict_read {
            let v_mmap = &dict_mmap[k];
            assert_eq!(v_read.shape(), v_mmap.shape());
            assert_eq!(v_read.data().unwrap(), v_mmap.data().unwrap());
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_gguf_mmap_rejects_missing_file() {
        let path = std::env::temp_dir().join("ferrotorch_gguf_mmap_does_not_exist");
        let _ = std::fs::remove_file(&path);
        let err = load_gguf_mmap(&path).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn test_load_gguf_state_dict_q4_0() {
        // Build a Q4_0 tensor with shape [32] (one block).
        let scale_bytes = f32_to_f16_bytes(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        // all nibbles = 8 => (8-8)*1.0 = 0.0
        block.extend(std::iter::repeat_n(0x88, 16));

        let bytes = build_gguf(
            &[],
            &[("q4_tensor", &[32], 2, &block)], // type 2 = Q4_0
        );

        let file = parse_gguf_bytes(&bytes).unwrap();
        let tensor = dequantize_gguf_tensor(&file, "q4_tensor").unwrap();
        assert_eq!(tensor.shape(), &[32]);
        // All elements should be 0.0 since all nibbles are 8 and (8-8)*scale=0.
        for &v in tensor.data().unwrap().iter() {
            assert!((v - 0.0).abs() < 1e-5, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_load_gguf_state_dict_q8_0() {
        // Build a Q8_0 tensor with shape [32] (one block).
        let scale_bytes = f32_to_f16_bytes(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        for i in 0..32u8 {
            block.push(i);
        }

        let bytes = build_gguf(
            &[],
            &[("q8_tensor", &[32], 8, &block)], // type 8 = Q8_0
        );

        let file = parse_gguf_bytes(&bytes).unwrap();
        let tensor = dequantize_gguf_tensor(&file, "q8_tensor").unwrap();
        assert_eq!(tensor.shape(), &[32]);
        let td = tensor.data().unwrap();
        for (i, &t) in td.iter().enumerate().take(32) {
            assert!(
                (t - i as f32).abs() < 1e-3,
                "element {i}: expected {}, got {t}",
                i as f32,
            );
        }
    }

    #[test]
    fn test_tensor_not_found() {
        let bytes = build_gguf(&[], &[]);
        let file = parse_gguf_bytes(&bytes).unwrap();
        let result = dequantize_gguf_tensor(&file, "nonexistent");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not found"), "got: {msg}");
    }

    #[test]
    fn test_multiple_metadata_types() {
        let uint32_val = 7u32.to_le_bytes();
        let bool_val = [1u8]; // true

        let mut string_val = Vec::new();
        let s = "test-model";
        string_val.extend_from_slice(&(s.len() as u64).to_le_bytes());
        string_val.extend_from_slice(s.as_bytes());

        let bytes = build_gguf(
            &[
                ("layers", 4, &uint32_val),       // Uint32
                ("general.name", 8, &string_val), // String
                ("use_cache", 7, &bool_val),      // Bool
            ],
            &[],
        );

        let file = parse_gguf_bytes(&bytes).unwrap();
        assert_eq!(
            file.metadata.entries.get("layers"),
            Some(&GgufValue::Uint32(7))
        );
        assert_eq!(
            file.metadata.entries.get("general.name"),
            Some(&GgufValue::String("test-model".to_string()))
        );
        assert_eq!(
            file.metadata.entries.get("use_cache"),
            Some(&GgufValue::Bool(true))
        );
    }

    #[test]
    fn test_multiple_tensors_state_dict() {
        let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut bytes_a = Vec::new();
        for v in &data_a {
            bytes_a.extend_from_slice(&v.to_le_bytes());
        }

        let data_b: Vec<f32> = vec![5.0, 6.0];
        let mut bytes_b = Vec::new();
        for v in &data_b {
            bytes_b.extend_from_slice(&v.to_le_bytes());
        }

        let bytes = build_gguf(
            &[],
            &[
                ("attn.weight", &[2, 2], 0, &bytes_a),
                ("attn.bias", &[2], 0, &bytes_b),
            ],
        );

        let file = parse_gguf_bytes(&bytes).unwrap();
        assert_eq!(file.tensors.len(), 2);

        let w = dequantize_gguf_tensor(&file, "attn.weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        let b = dequantize_gguf_tensor(&file, "attn.bias").unwrap();
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.data().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_ggml_type_block_properties() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F32.block_bytes(), 4);
        assert_eq!(GgmlType::F16.block_size(), 1);
        assert_eq!(GgmlType::F16.block_bytes(), 2);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.block_bytes(), 18);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q8_0.block_bytes(), 34);
    }

    #[test]
    fn test_f16_to_f32_special_values() {
        // Zero.
        let z = f16_to_f32(0, 0);
        assert_eq!(z, 0.0);

        // Negative zero.
        let nz = f16_to_f32(0x00, 0x80);
        assert!(nz == 0.0 && nz.is_sign_negative());

        // Infinity.
        let inf = f16_to_f32(0x00, 0x7C);
        assert!(inf.is_infinite() && inf.is_sign_positive());

        // NaN.
        let nan = f16_to_f32(0x01, 0x7C);
        assert!(nan.is_nan());
    }
}
