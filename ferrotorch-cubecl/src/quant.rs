//! GGUF quantized-weight dequantization on the GPU.
//!
//! Pure-GPU implementations of the GGUF (llama.cpp) block-quantized formats:
//! `Q4_0`, `Q4_1`, `Q8_0`. Each format packs N=32 weights into a small block
//! header (an `f16` scale, optionally an `f16` min) plus packed quantized
//! bits. This module:
//!
//! 1. **Host-side block splitting** — [`split_q4_0_blocks`],
//!    [`split_q4_1_blocks`], [`split_q8_0_blocks`] parse the on-disk byte
//!    stream into a small scales (and optional mins) `f32` vector plus a
//!    packed `u32` vector of just the quantized bits. The scales work is
//!    O(num_blocks); for a 70B Q4_0 tensor (~2.2 G blocks) the scale buffer
//!    is ~9 GB but it is dwarfed by the dequantized output and the dataplane
//!    (the actual quantized bits) never touches CPU f32 memory.
//!
//! 2. **GPU dequantization kernels** — [`kernel_dequantize_q4_0`],
//!    [`kernel_dequantize_q4_1`], [`kernel_dequantize_q8_0`]. Each kernel
//!    reads the packed bits + scales + mins from GPU `Array`s and writes
//!    the full-precision output entirely on-device. There is no CPU
//!    fallback.
//!
//! 3. **Host launchers** — [`dequantize_q4_0_to_gpu`] etc. upload the split
//!    inputs, dispatch the kernel, and return the dequantized `f32` GPU
//!    handle to the caller. The returned handle stays on the device — the
//!    caller is expected to wire it into downstream GPU machinery (model
//!    parameters, attention KV cache, etc.) without ever pulling it back
//!    to host memory.
//!
//! Algorithmic details for each quantization type are written against the
//! exact same byte layouts implemented (and tested) by
//! `ferrotorch_serialize::gguf::dequantize_qX_Y`; see the unit tests in
//! this file for cross-validation.

use cubecl::prelude::*;

/// Number of weights per GGUF block (constant across Q4_0 / Q4_1 / Q5_0 / Q5_1
/// / Q8_0 / Q8_1).
pub const GGUF_BLOCK_SIZE: usize = 32;

const Q4_0_BLOCK_BYTES: usize = 18; // 2 (f16 scale) + 16 (32 nibbles)
const Q4_1_BLOCK_BYTES: usize = 20; // 2 (f16 scale) + 2 (f16 min) + 16 (32 nibbles)
const Q5_0_BLOCK_BYTES: usize = 22; // 2 (f16 scale) + 4 (32 high bits) + 16 (32 low nibbles)
const Q5_1_BLOCK_BYTES: usize = 24; // 2 (f16 scale) + 2 (f16 min) + 4 (qh) + 16 (nibbles)
const Q8_0_BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8)
const Q8_1_BLOCK_BYTES: usize = 40; // 4 (f32 scale) + 4 (f32 min) + 32 (i8)

/// Identifier for the supported GGUF block layouts.
///
/// Variant names mirror the GGUF binary format spec verbatim (`Q4_0`,
/// `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`) so that grep-equivalence with
/// the upstream llama.cpp / GGUF documentation holds. Renaming to RFC 430
/// `UpperCamelCase` would break that interop contract.
#[allow(non_camel_case_types)] // matches GGUF binary format spec naming (Q4_0, Q4_1, ...)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufBlockKind {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
}

impl GgufBlockKind {
    /// Number of weights per block (always 32 for the supported layouts).
    #[must_use]
    pub const fn block_elements(self) -> usize {
        GGUF_BLOCK_SIZE
    }

    /// Number of bytes per packed block on disk.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => Q4_0_BLOCK_BYTES,
            Self::Q4_1 => Q4_1_BLOCK_BYTES,
            Self::Q5_0 => Q5_0_BLOCK_BYTES,
            Self::Q5_1 => Q5_1_BLOCK_BYTES,
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
            Self::Q8_1 => Q8_1_BLOCK_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side block splitting
// ---------------------------------------------------------------------------

#[inline]
fn read_f16_to_f32(b0: u8, b1: u8) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b0, b1])).to_f32()
}

#[inline]
fn pack_4_bytes_le(b0: u8, b1: u8, b2: u8, b3: u8) -> u32 {
    u32::from_le_bytes([b0, b1, b2, b3])
}

/// Split a Q4_0 byte stream into (scales, packed_nibbles).
///
/// `raw` must have at least `num_blocks * 18` bytes. Returns:
/// - `scales`: `Vec<f32>` of length `num_blocks` (one per block, converted
///   from the on-disk f16 representation).
/// - `packed_nibbles`: `Vec<u32>` of length `num_blocks * 4`. Each block
///   contributes 16 bytes of packed nibbles → 4 little-endian u32s. The
///   GPU kernel reads u32 values and unpacks the 32 nibbles per block.
///
/// # Panics
///
/// Panics if `raw.len() < num_blocks * 18`.
pub fn split_q4_0_blocks(raw: &[u8], num_blocks: usize) -> (Vec<f32>, Vec<u32>) {
    assert!(
        raw.len() >= num_blocks * Q4_0_BLOCK_BYTES,
        "split_q4_0_blocks: need {} bytes for {num_blocks} blocks, got {}",
        num_blocks * Q4_0_BLOCK_BYTES,
        raw.len()
    );
    let mut scales = Vec::with_capacity(num_blocks);
    let mut nibbles = Vec::with_capacity(num_blocks * 4);
    for b in 0..num_blocks {
        let off = b * Q4_0_BLOCK_BYTES;
        scales.push(read_f16_to_f32(raw[off], raw[off + 1]));
        for u in 0..4 {
            let base = off + 2 + u * 4;
            nibbles.push(pack_4_bytes_le(
                raw[base],
                raw[base + 1],
                raw[base + 2],
                raw[base + 3],
            ));
        }
    }
    (scales, nibbles)
}

/// Split a Q4_1 byte stream into (scales, mins, packed_nibbles).
///
/// Same shape as Q4_0, plus a `mins` vector that the kernel adds after the
/// scale multiplication.
pub fn split_q4_1_blocks(raw: &[u8], num_blocks: usize) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    assert!(
        raw.len() >= num_blocks * Q4_1_BLOCK_BYTES,
        "split_q4_1_blocks: need {} bytes for {num_blocks} blocks, got {}",
        num_blocks * Q4_1_BLOCK_BYTES,
        raw.len()
    );
    let mut scales = Vec::with_capacity(num_blocks);
    let mut mins = Vec::with_capacity(num_blocks);
    let mut nibbles = Vec::with_capacity(num_blocks * 4);
    for b in 0..num_blocks {
        let off = b * Q4_1_BLOCK_BYTES;
        scales.push(read_f16_to_f32(raw[off], raw[off + 1]));
        mins.push(read_f16_to_f32(raw[off + 2], raw[off + 3]));
        for u in 0..4 {
            let base = off + 4 + u * 4;
            nibbles.push(pack_4_bytes_le(
                raw[base],
                raw[base + 1],
                raw[base + 2],
                raw[base + 3],
            ));
        }
    }
    (scales, mins, nibbles)
}

/// Split a Q8_0 byte stream into (scales, packed_int8_bytes).
///
/// `packed_int8_bytes` has length `num_blocks * 8` (32 i8 values per block,
/// packed as 8 little-endian u32). The kernel re-extracts each byte and
/// sign-extends it before multiplying by the block scale.
pub fn split_q8_0_blocks(raw: &[u8], num_blocks: usize) -> (Vec<f32>, Vec<u32>) {
    assert!(
        raw.len() >= num_blocks * Q8_0_BLOCK_BYTES,
        "split_q8_0_blocks: need {} bytes for {num_blocks} blocks, got {}",
        num_blocks * Q8_0_BLOCK_BYTES,
        raw.len()
    );
    let mut scales = Vec::with_capacity(num_blocks);
    let mut bytes = Vec::with_capacity(num_blocks * 8);
    for b in 0..num_blocks {
        let off = b * Q8_0_BLOCK_BYTES;
        scales.push(read_f16_to_f32(raw[off], raw[off + 1]));
        for u in 0..8 {
            let base = off + 2 + u * 4;
            bytes.push(pack_4_bytes_le(
                raw[base],
                raw[base + 1],
                raw[base + 2],
                raw[base + 3],
            ));
        }
    }
    (scales, bytes)
}

/// Split a Q5_0 byte stream into (scales, qh, packed_nibbles).
///
/// Q5_0 layout per block (22 bytes): 2 (f16 scale) + 4 (qh: 32 high bits as
/// little-endian u32) + 16 (32 low nibbles, two per byte). Each output element
/// is `(low_nibble | (high_bit << 4)) - 16` then × scale.
pub fn split_q5_0_blocks(raw: &[u8], num_blocks: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    assert!(
        raw.len() >= num_blocks * Q5_0_BLOCK_BYTES,
        "split_q5_0_blocks: need {} bytes for {num_blocks} blocks, got {}",
        num_blocks * Q5_0_BLOCK_BYTES,
        raw.len()
    );
    let mut scales = Vec::with_capacity(num_blocks);
    let mut qh = Vec::with_capacity(num_blocks);
    let mut nibbles = Vec::with_capacity(num_blocks * 4);
    for b in 0..num_blocks {
        let off = b * Q5_0_BLOCK_BYTES;
        scales.push(read_f16_to_f32(raw[off], raw[off + 1]));
        qh.push(pack_4_bytes_le(
            raw[off + 2],
            raw[off + 3],
            raw[off + 4],
            raw[off + 5],
        ));
        for u in 0..4 {
            let base = off + 6 + u * 4;
            nibbles.push(pack_4_bytes_le(
                raw[base],
                raw[base + 1],
                raw[base + 2],
                raw[base + 3],
            ));
        }
    }
    (scales, qh, nibbles)
}

/// Split a Q5_1 byte stream into (scales, mins, qh, packed_nibbles).
///
/// Q5_1 layout per block (24 bytes): 2 (f16 scale) + 2 (f16 min) + 4 (qh) +
/// 16 (nibbles). Each output element is `(low_nibble | (high_bit << 4)) ×
/// scale + min` (asymmetric variant of Q5_0).
pub fn split_q5_1_blocks(
    raw: &[u8],
    num_blocks: usize,
) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
    assert!(
        raw.len() >= num_blocks * Q5_1_BLOCK_BYTES,
        "split_q5_1_blocks: need {} bytes for {num_blocks} blocks, got {}",
        num_blocks * Q5_1_BLOCK_BYTES,
        raw.len()
    );
    let mut scales = Vec::with_capacity(num_blocks);
    let mut mins = Vec::with_capacity(num_blocks);
    let mut qh = Vec::with_capacity(num_blocks);
    let mut nibbles = Vec::with_capacity(num_blocks * 4);
    for b in 0..num_blocks {
        let off = b * Q5_1_BLOCK_BYTES;
        scales.push(read_f16_to_f32(raw[off], raw[off + 1]));
        mins.push(read_f16_to_f32(raw[off + 2], raw[off + 3]));
        qh.push(pack_4_bytes_le(
            raw[off + 4],
            raw[off + 5],
            raw[off + 6],
            raw[off + 7],
        ));
        for u in 0..4 {
            let base = off + 8 + u * 4;
            nibbles.push(pack_4_bytes_le(
                raw[base],
                raw[base + 1],
                raw[base + 2],
                raw[base + 3],
            ));
        }
    }
    (scales, mins, qh, nibbles)
}

/// Split a Q8_1 byte stream into (scales, mins, packed_int8_bytes).
///
/// Q8_1 layout per block (40 bytes): 4 (f32 scale) + 4 (f32 min) + 32 (i8
/// values). Note that unlike Q4_*/Q5_*/Q8_0, Q8_1's scale and min are stored
/// as `f32` (not `f16`) on disk — twice the metadata overhead, but no
/// precision loss.
pub fn split_q8_1_blocks(raw: &[u8], num_blocks: usize) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    assert!(
        raw.len() >= num_blocks * Q8_1_BLOCK_BYTES,
        "split_q8_1_blocks: need {} bytes for {num_blocks} blocks, got {}",
        num_blocks * Q8_1_BLOCK_BYTES,
        raw.len()
    );
    let mut scales = Vec::with_capacity(num_blocks);
    let mut mins = Vec::with_capacity(num_blocks);
    let mut bytes = Vec::with_capacity(num_blocks * 8);
    for b in 0..num_blocks {
        let off = b * Q8_1_BLOCK_BYTES;
        scales.push(f32::from_le_bytes([
            raw[off],
            raw[off + 1],
            raw[off + 2],
            raw[off + 3],
        ]));
        mins.push(f32::from_le_bytes([
            raw[off + 4],
            raw[off + 5],
            raw[off + 6],
            raw[off + 7],
        ]));
        for u in 0..8 {
            let base = off + 8 + u * 4;
            bytes.push(pack_4_bytes_le(
                raw[base],
                raw[base + 1],
                raw[base + 2],
                raw[base + 3],
            ));
        }
    }
    (scales, mins, bytes)
}

// ---------------------------------------------------------------------------
// GPU kernels — dequantization runs entirely on-device
// ---------------------------------------------------------------------------

/// Dequantize Q4_0: `out[t] = scales[block] * (nibble - 8.0)`.
///
/// One thread per output element. All indices are `usize` (matching the
/// pattern used by other CubeCL kernels in this crate); the only `u32`
/// values are the packed bytes themselves, which we read from the
/// `Array<u32>` and decompose with bit operations.
#[cube(launch_unchecked)]
pub fn kernel_dequantize_q4_0<F: Float>(
    scales: &Array<F>,
    nibbles: &Array<u32>,
    out: &mut Array<F>,
) {
    if ABSOLUTE_POS < out.len() {
        let t = ABSOLUTE_POS;
        let block_id = t / 32;
        let elem = t % 32;
        let byte_idx = elem / 2;
        let is_high = elem % 2;
        let u32_idx_in_block = byte_idx / 4;
        let byte_in_u32 = byte_idx % 4;
        let global_u32 = block_id * 4 + u32_idx_in_block;
        let packed = nibbles[global_u32];
        let byte = (packed >> (byte_in_u32 as u32 * 8u32)) & 0xFFu32;
        let nibble = (byte >> (is_high as u32 * 4u32)) & 0xFu32;
        // nibble in 0..16, subtract 8 in float (avoids signed-int complication).
        let nibble_f = F::cast_from(nibble) - F::new(8.0);
        out[t] = scales[block_id] * nibble_f;
    }
}

/// Dequantize Q4_1: `out[t] = scales[block] * nibble + mins[block]`.
#[cube(launch_unchecked)]
pub fn kernel_dequantize_q4_1<F: Float>(
    scales: &Array<F>,
    mins: &Array<F>,
    nibbles: &Array<u32>,
    out: &mut Array<F>,
) {
    if ABSOLUTE_POS < out.len() {
        let t = ABSOLUTE_POS;
        let block_id = t / 32;
        let elem = t % 32;
        let byte_idx = elem / 2;
        let is_high = elem % 2;
        let u32_idx_in_block = byte_idx / 4;
        let byte_in_u32 = byte_idx % 4;
        let global_u32 = block_id * 4 + u32_idx_in_block;
        let packed = nibbles[global_u32];
        let byte = (packed >> (byte_in_u32 as u32 * 8u32)) & 0xFFu32;
        let nibble = (byte >> (is_high as u32 * 4u32)) & 0xFu32;
        let nibble_f = F::cast_from(nibble);
        out[t] = scales[block_id] * nibble_f + mins[block_id];
    }
}

/// Dequantize Q8_0: `out[t] = scales[block] * (sign-extended i8 byte)`.
///
/// The 8-bit signed byte is sign-extended using float arithmetic only —
/// no `bitcast`/`reinterpret` is required because the formula
/// `signed_f = byte_f - 256.0 * sign_bit_f` produces the correct value
/// for both ranges (0..128 → 0..127 and 128..256 → -128..-1) without
/// integer reinterpretation.
#[cube(launch_unchecked)]
pub fn kernel_dequantize_q8_0<F: Float>(scales: &Array<F>, bytes: &Array<u32>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let t = ABSOLUTE_POS;
        let block_id = t / 32;
        let elem = t % 32;
        let u32_idx_in_block = elem / 4;
        let byte_in_u32 = elem % 4;
        let global_u32 = block_id * 8 + u32_idx_in_block;
        let packed = bytes[global_u32];
        let byte = (packed >> (byte_in_u32 as u32 * 8u32)) & 0xFFu32;
        let sign_bit = (byte >> 7u32) & 1u32;
        let byte_f = F::cast_from(byte);
        let sign_f = F::cast_from(sign_bit);
        let signed_f = byte_f - sign_f * F::new(256.0);
        out[t] = scales[block_id] * signed_f;
    }
}

/// Dequantize Q5_0: `out[t] = scales[block] * (val5 - 16.0)` where `val5 =
/// (low_nibble | (high_bit << 4))`.
///
/// The high bit comes from `qh[block]` — 32 packed bits, one per element.
/// The kernel extracts the low nibble exactly as in Q4_0 and OR-combines
/// with the high bit before subtracting 16.0 (the symmetric centering
/// constant for 5-bit values).
#[cube(launch_unchecked)]
pub fn kernel_dequantize_q5_0<F: Float>(
    scales: &Array<F>,
    qh: &Array<u32>,
    nibbles: &Array<u32>,
    out: &mut Array<F>,
) {
    if ABSOLUTE_POS < out.len() {
        let t = ABSOLUTE_POS;
        let block_id = t / 32;
        let elem = t % 32;
        let byte_idx = elem / 2;
        let is_high = elem % 2;
        let u32_idx_in_block = byte_idx / 4;
        let byte_in_u32 = byte_idx % 4;
        let global_u32 = block_id * 4 + u32_idx_in_block;
        let packed = nibbles[global_u32];
        let byte = (packed >> (byte_in_u32 as u32 * 8u32)) & 0xFFu32;
        let low_nibble = (byte >> (is_high as u32 * 4u32)) & 0xFu32;
        // High bits: bit (elem*2) for the low side, bit (elem*2 + 1) for the
        // high side. This matches the CPU code's per-byte interleaving:
        //   lo_high_bit = (qh >> (j*2)) & 1
        //   hi_high_bit = (qh >> (j*2 + 1)) & 1
        let qh_word = qh[block_id];
        let bit_pos = byte_idx as u32 * 2u32 + is_high as u32;
        let high_bit = (qh_word >> bit_pos) & 1u32;
        let val5 = low_nibble | (high_bit << 4u32);
        let val_f = F::cast_from(val5) - F::new(16.0);
        out[t] = scales[block_id] * val_f;
    }
}

/// Dequantize Q5_1: `out[t] = scales[block] * val5 + mins[block]`.
///
/// Asymmetric Q5: same bit unpacking as Q5_0 but the unsigned 5-bit value
/// is multiplied by scale and the per-block min is added (no -16
/// centering, the min carries the offset).
#[cube(launch_unchecked)]
pub fn kernel_dequantize_q5_1<F: Float>(
    scales: &Array<F>,
    mins: &Array<F>,
    qh: &Array<u32>,
    nibbles: &Array<u32>,
    out: &mut Array<F>,
) {
    if ABSOLUTE_POS < out.len() {
        let t = ABSOLUTE_POS;
        let block_id = t / 32;
        let elem = t % 32;
        let byte_idx = elem / 2;
        let is_high = elem % 2;
        let u32_idx_in_block = byte_idx / 4;
        let byte_in_u32 = byte_idx % 4;
        let global_u32 = block_id * 4 + u32_idx_in_block;
        let packed = nibbles[global_u32];
        let byte = (packed >> (byte_in_u32 as u32 * 8u32)) & 0xFFu32;
        let low_nibble = (byte >> (is_high as u32 * 4u32)) & 0xFu32;
        let qh_word = qh[block_id];
        let bit_pos = byte_idx as u32 * 2u32 + is_high as u32;
        let high_bit = (qh_word >> bit_pos) & 1u32;
        let val5 = low_nibble | (high_bit << 4u32);
        let val_f = F::cast_from(val5);
        out[t] = scales[block_id] * val_f + mins[block_id];
    }
}

/// Dequantize Q8_1: `out[t] = scales[block] * sign_extended_byte +
/// mins[block]`.
///
/// Same byte/u32 layout as Q8_0 but with f32 scale and f32 min instead of
/// the symmetric f16-scale-only Q8_0 formulation.
#[cube(launch_unchecked)]
pub fn kernel_dequantize_q8_1<F: Float>(
    scales: &Array<F>,
    mins: &Array<F>,
    bytes: &Array<u32>,
    out: &mut Array<F>,
) {
    if ABSOLUTE_POS < out.len() {
        let t = ABSOLUTE_POS;
        let block_id = t / 32;
        let elem = t % 32;
        let u32_idx_in_block = elem / 4;
        let byte_in_u32 = elem % 4;
        let global_u32 = block_id * 8 + u32_idx_in_block;
        let packed = bytes[global_u32];
        let byte = (packed >> (byte_in_u32 as u32 * 8u32)) & 0xFFu32;
        let sign_bit = (byte >> 7u32) & 1u32;
        let byte_f = F::cast_from(byte);
        let sign_f = F::cast_from(sign_bit);
        let signed_f = byte_f - sign_f * F::new(256.0);
        out[t] = scales[block_id] * signed_f + mins[block_id];
    }
}

// ---------------------------------------------------------------------------
// Host launchers — upload split buffers, dispatch kernel, return GPU handle
// ---------------------------------------------------------------------------

/// Upload the host-split Q4_0 buffers to GPU, run the dequantization kernel,
/// and return the on-device output handle. The output handle is *not* read
/// back to the host — that is the entire point of this path.
///
/// `num_blocks * 32` is the output length. The caller usually obtains
/// `(scales, nibbles)` from [`split_q4_0_blocks`] and `num_elements` from
/// the GGUF tensor info.
pub fn dequantize_q4_0_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    scales: &[f32],
    nibbles: &[u32],
    num_elements: usize,
) -> cubecl::server::Handle {
    debug_assert_eq!(num_elements, scales.len() * 32);
    debug_assert_eq!(nibbles.len(), scales.len() * 4);

    let scales_handle = client.create_from_slice(f32::as_bytes(scales));
    let nibbles_handle = client.create_from_slice(unsafe {
        // SAFETY: Reinterprets the host-side `&[u32]` as `&[u8]` for the
        //   `client.create_from_slice` upload path (which takes `&[u8]`).
        //   - Alignment: `u32` has stricter alignment than `u8`, so casting
        //     `*const u32` to `*const u8` is always valid (no
        //     under-alignment). Read access at any byte offset is sound.
        //   - Length: `size_of_val(nibbles) == nibbles.len() * 4`; the
        //     resulting `&[u8]` covers exactly the same bytes as the
        //     original `&[u32]`. No out-of-bounds.
        //   - Lifetime: the `&[u8]` borrows from `nibbles` (lives until
        //     this function returns) and is consumed synchronously by
        //     `create_from_slice`, which copies into device memory. No
        //     dangling reference. No concurrent `&mut` exists.
        std::slice::from_raw_parts(
            nibbles.as_ptr() as *const u8,
            std::mem::size_of_val(nibbles),
        )
    });
    let out_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    let (count, dim) = crate::elementwise_launch_dims(num_elements as u32);
    // SAFETY: All three handles were alloc'd by this `client` immediately
    //   above:
    //   - `scales_handle` from `create_from_slice(f32::as_bytes(scales))`
    //     (line 526); backing buffer holds `scales.len()` f32 elements.
    //   - `nibbles_handle` from `create_from_slice(<bytes>)` (line 527);
    //     `nibbles.len() * 4` bytes uploaded — exactly `nibbles.len()` u32
    //     elements as the kernel sees `&Array<u32>`.
    //   - `out_handle` from `empty(num_elements * size_of::<f32>())` (line
    //     533); capacity is exactly `num_elements` f32 elements. `.clone()`
    //     is a refcount bump only; kernel writes are visible via the
    //     returned `out_handle`.
    //   `count`/`dim` from `elementwise_launch_dims(num_elements)` cover
    //   `num_elements` units (one per output element); kernel guards
    //   `ABSOLUTE_POS < out.len()`. `debug_assert_eq` at lines 523-524
    //   pins shape relations (`num_elements == scales.len() * 32`,
    //   `nibbles.len() == scales.len() * 4`). `launch_unchecked` is unsafe
    //   per cubecl convention; refs live for launch duration.
    unsafe {
        kernel_dequantize_q4_0::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(scales_handle, scales.len()),
            ArrayArg::from_raw_parts(nibbles_handle, nibbles.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), num_elements),
        );
    }
    out_handle
}

/// Upload the host-split Q4_1 buffers to GPU, run the dequantization kernel,
/// and return the on-device output handle.
pub fn dequantize_q4_1_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    scales: &[f32],
    mins: &[f32],
    nibbles: &[u32],
    num_elements: usize,
) -> cubecl::server::Handle {
    debug_assert_eq!(num_elements, scales.len() * 32);
    debug_assert_eq!(scales.len(), mins.len());
    debug_assert_eq!(nibbles.len(), scales.len() * 4);

    let scales_handle = client.create_from_slice(f32::as_bytes(scales));
    let mins_handle = client.create_from_slice(f32::as_bytes(mins));
    let nibbles_handle = client.create_from_slice(unsafe {
        // SAFETY: `&[u32]` → `&[u8]` reinterpret for the `&[u8]`-typed
        //   `create_from_slice` upload API.
        //   - Alignment: `align_of::<u32>() ≥ align_of::<u8>()`, so casting
        //     `*const u32` to `*const u8` cannot under-align — every byte
        //     in the underlying buffer is independently readable.
        //   - Length: `size_of_val(nibbles) == nibbles.len() * 4`; the
        //     resulting `&[u8]` covers exactly the same byte range. No
        //     out-of-bounds.
        //   - Lifetime: borrows from `nibbles` (live until function return);
        //     consumed synchronously by `create_from_slice`. No `&mut`
        //     aliasing.
        std::slice::from_raw_parts(
            nibbles.as_ptr() as *const u8,
            std::mem::size_of_val(nibbles),
        )
    });
    let out_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    let (count, dim) = crate::elementwise_launch_dims(num_elements as u32);
    // SAFETY: All four handles were alloc'd by this `client` immediately
    //   above:
    //   - `scales_handle` and `mins_handle` from
    //     `create_from_slice(f32::as_bytes(...))` (lines 562-563); each
    //     holds `scales.len()` / `mins.len()` f32 elements
    //     (`scales.len() == mins.len()` per debug_assert at line 559).
    //   - `nibbles_handle` from `create_from_slice(<bytes>)` (line 564);
    //     `nibbles.len() * 4` bytes — exactly `nibbles.len()` u32 elements.
    //   - `out_handle` from `empty(num_elements * size_of::<f32>())` (line
    //     570); capacity = `num_elements` f32 elements. `.clone()` is a
    //     refcount bump — kernel writes visible via returned handle.
    //   `count`/`dim` cover `num_elements` units; kernel bounds-checks
    //   `ABSOLUTE_POS`. Shape relations enforced by `debug_assert_eq` at
    //   lines 558-560. `launch_unchecked` skips runtime arity checks (per
    //   cubecl convention); refs live for launch duration.
    unsafe {
        kernel_dequantize_q4_1::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(scales_handle, scales.len()),
            ArrayArg::from_raw_parts(mins_handle, mins.len()),
            ArrayArg::from_raw_parts(nibbles_handle, nibbles.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), num_elements),
        );
    }
    out_handle
}

/// Upload the host-split Q8_0 buffers to GPU, run the dequantization kernel,
/// and return the on-device output handle.
pub fn dequantize_q8_0_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    scales: &[f32],
    bytes: &[u32],
    num_elements: usize,
) -> cubecl::server::Handle {
    debug_assert_eq!(num_elements, scales.len() * 32);
    debug_assert_eq!(bytes.len(), scales.len() * 8);

    let scales_handle = client.create_from_slice(f32::as_bytes(scales));
    let bytes_handle = client.create_from_slice(unsafe {
        // SAFETY: `&[u32]` → `&[u8]` reinterpret for `create_from_slice`'s
        //   `&[u8]` upload API.
        //   - Alignment: `align_of::<u32>() ≥ align_of::<u8>()`; the cast
        //     `*const u32` → `*const u8` never under-aligns.
        //   - Length: `size_of_val(bytes) == bytes.len() * 4` — exact byte
        //     coverage, no overrun.
        //   - Lifetime: borrows from `bytes` (live until function return);
        //     consumed synchronously by `create_from_slice` (which copies
        //     into device memory). No `&mut` overlap.
        std::slice::from_raw_parts(bytes.as_ptr() as *const u8, std::mem::size_of_val(bytes))
    });
    let out_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    let (count, dim) = crate::elementwise_launch_dims(num_elements as u32);
    // SAFETY: All three handles alloc'd by this `client` above:
    //   - `scales_handle` from `create_from_slice(f32::as_bytes(scales))`
    //     (line 598); `scales.len()` f32 elements.
    //   - `bytes_handle` from `create_from_slice(<bytes>)` (line 599);
    //     `bytes.len()` u32 elements (`bytes.len() == scales.len() * 8`,
    //     debug-asserted line 596).
    //   - `out_handle` from `empty(num_elements * size_of::<f32>())` (line
    //     602); `num_elements` f32 elements. `.clone()` is a refcount bump.
    //   `count`/`dim` cover `num_elements` units; kernel bounds-checks
    //   `ABSOLUTE_POS < out.len()`. Shape relations: `num_elements ==
    //   scales.len() * 32` (line 595). `launch_unchecked` skips runtime
    //   checks per cubecl convention; refs live for launch duration.
    unsafe {
        kernel_dequantize_q8_0::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(scales_handle, scales.len()),
            ArrayArg::from_raw_parts(bytes_handle, bytes.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), num_elements),
        );
    }
    out_handle
}

/// Upload the host-split Q5_0 buffers to GPU, run the dequantization kernel,
/// and return the on-device output handle.
pub fn dequantize_q5_0_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    scales: &[f32],
    qh: &[u32],
    nibbles: &[u32],
    num_elements: usize,
) -> cubecl::server::Handle {
    debug_assert_eq!(num_elements, scales.len() * 32);
    debug_assert_eq!(qh.len(), scales.len());
    debug_assert_eq!(nibbles.len(), scales.len() * 4);

    let scales_handle = client.create_from_slice(f32::as_bytes(scales));
    let qh_handle = client.create_from_slice(unsafe {
        // SAFETY: `&[u32]` → `&[u8]` reinterpret for the `&[u8]`-typed
        //   `create_from_slice` upload path.
        //   - Alignment: `align_of::<u32>() ≥ align_of::<u8>()` — cast
        //     never under-aligns; every byte readable.
        //   - Length: `size_of_val(qh) == qh.len() * 4` exactly.
        //   - Lifetime: borrows from `qh` (live until function return);
        //     consumed synchronously by `create_from_slice`.
        std::slice::from_raw_parts(qh.as_ptr() as *const u8, std::mem::size_of_val(qh))
    });
    let nibbles_handle = client.create_from_slice(unsafe {
        // SAFETY: identical pattern to the `qh` reinterpret above —
        //   `&[u32]` → `&[u8]` for upload. Alignment is monotonic
        //   (`u32` ≥ `u8`); `size_of_val(nibbles) == nibbles.len() * 4`
        //   gives exact byte coverage; lifetime tied to `nibbles` (returns
        //   before it's dropped); no `&mut` aliasing.
        std::slice::from_raw_parts(
            nibbles.as_ptr() as *const u8,
            std::mem::size_of_val(nibbles),
        )
    });
    let out_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    let (count, dim) = crate::elementwise_launch_dims(num_elements as u32);
    // SAFETY: All four handles alloc'd by this `client` above:
    //   - `scales_handle` from `create_from_slice(f32::as_bytes(scales))`
    //     (line 631); `scales.len()` f32 elements.
    //   - `qh_handle` from `create_from_slice(<qh-as-bytes>)` (line 632);
    //     `qh.len()` u32 elements (`qh.len() == scales.len()`,
    //     debug-asserted line 628).
    //   - `nibbles_handle` from `create_from_slice(<nibbles-as-bytes>)`
    //     (line 635); `nibbles.len()` u32 elements
    //     (`nibbles.len() == scales.len() * 4`, line 629).
    //   - `out_handle` from `empty(num_elements * size_of::<f32>())` (line
    //     641); `num_elements` f32 elements. `.clone()` is a refcount bump.
    //   `count`/`dim` cover `num_elements` units; kernel guards
    //   `ABSOLUTE_POS < out.len()`. Shape: `num_elements == scales.len() *
    //   32` (line 627). `launch_unchecked` skips runtime arity per cubecl
    //   convention; refs live for launch duration.
    unsafe {
        kernel_dequantize_q5_0::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(scales_handle, scales.len()),
            ArrayArg::from_raw_parts(qh_handle, qh.len()),
            ArrayArg::from_raw_parts(nibbles_handle, nibbles.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), num_elements),
        );
    }
    out_handle
}

/// Upload the host-split Q5_1 buffers to GPU, run the dequantization kernel,
/// and return the on-device output handle.
pub fn dequantize_q5_1_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    scales: &[f32],
    mins: &[f32],
    qh: &[u32],
    nibbles: &[u32],
    num_elements: usize,
) -> cubecl::server::Handle {
    debug_assert_eq!(num_elements, scales.len() * 32);
    debug_assert_eq!(mins.len(), scales.len());
    debug_assert_eq!(qh.len(), scales.len());
    debug_assert_eq!(nibbles.len(), scales.len() * 4);

    let scales_handle = client.create_from_slice(f32::as_bytes(scales));
    let mins_handle = client.create_from_slice(f32::as_bytes(mins));
    let qh_handle = client.create_from_slice(unsafe {
        // SAFETY: `&[u32]` → `&[u8]` reinterpret for upload (see Q5_0
        //   variant for the canonical version). Alignment is monotonic
        //   (`u32` ≥ `u8`); `size_of_val(qh) == qh.len() * 4` is exact;
        //   lifetime tied to `qh` (lives until function return);
        //   `create_from_slice` consumes the borrow synchronously.
        std::slice::from_raw_parts(qh.as_ptr() as *const u8, std::mem::size_of_val(qh))
    });
    let nibbles_handle = client.create_from_slice(unsafe {
        // SAFETY: same `&[u32]` → `&[u8]` reinterpret pattern as `qh`
        //   above. `align_of::<u32>() ≥ align_of::<u8>()`,
        //   `size_of_val(nibbles) == nibbles.len() * 4`, lifetime bounded
        //   by `nibbles`'s scope, no `&mut` aliasing.
        std::slice::from_raw_parts(
            nibbles.as_ptr() as *const u8,
            std::mem::size_of_val(nibbles),
        )
    });
    let out_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    let (count, dim) = crate::elementwise_launch_dims(num_elements as u32);
    // SAFETY: All five handles alloc'd by this `client` above:
    //   - `scales_handle` and `mins_handle` from
    //     `create_from_slice(f32::as_bytes(...))` (lines 673-674); each
    //     holds `scales.len()` / `mins.len()` f32 elements
    //     (`scales.len() == mins.len()`, asserted line 669).
    //   - `qh_handle` (line 675): `qh.len()` u32 elements
    //     (`qh.len() == scales.len()`, line 670).
    //   - `nibbles_handle` (line 678): `nibbles.len()` u32 elements
    //     (`nibbles.len() == scales.len() * 4`, line 671).
    //   - `out_handle` from `empty(num_elements * size_of::<f32>())` (line
    //     684); `num_elements` f32 elements. `.clone()` is a refcount bump.
    //   `count`/`dim` cover `num_elements` units (`num_elements ==
    //   scales.len() * 32`, line 668); kernel bounds-checks `ABSOLUTE_POS
    //   < out.len()`. `launch_unchecked` skips runtime arity per cubecl
    //   convention; refs live for launch duration.
    unsafe {
        kernel_dequantize_q5_1::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(scales_handle, scales.len()),
            ArrayArg::from_raw_parts(mins_handle, mins.len()),
            ArrayArg::from_raw_parts(qh_handle, qh.len()),
            ArrayArg::from_raw_parts(nibbles_handle, nibbles.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), num_elements),
        );
    }
    out_handle
}

/// Upload the host-split Q8_1 buffers to GPU, run the dequantization kernel,
/// and return the on-device output handle.
pub fn dequantize_q8_1_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    scales: &[f32],
    mins: &[f32],
    bytes: &[u32],
    num_elements: usize,
) -> cubecl::server::Handle {
    debug_assert_eq!(num_elements, scales.len() * 32);
    debug_assert_eq!(mins.len(), scales.len());
    debug_assert_eq!(bytes.len(), scales.len() * 8);

    let scales_handle = client.create_from_slice(f32::as_bytes(scales));
    let mins_handle = client.create_from_slice(f32::as_bytes(mins));
    let bytes_handle = client.create_from_slice(unsafe {
        // SAFETY: `&[u32]` → `&[u8]` reinterpret for the `&[u8]`-typed
        //   `create_from_slice` upload path. `align_of::<u32>() ≥
        //   align_of::<u8>()` (cast never under-aligns); `size_of_val(bytes)
        //   == bytes.len() * 4` exactly; lifetime tied to `bytes` (lives
        //   until function return); `create_from_slice` consumes the borrow
        //   synchronously by copying into device memory; no `&mut` overlap.
        std::slice::from_raw_parts(bytes.as_ptr() as *const u8, std::mem::size_of_val(bytes))
    });
    let out_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    let (count, dim) = crate::elementwise_launch_dims(num_elements as u32);
    // SAFETY: All four handles alloc'd by this `client` above:
    //   - `scales_handle` and `mins_handle` from
    //     `create_from_slice(f32::as_bytes(...))` (lines 715-716); each
    //     holds `scales.len()` / `mins.len()` f32 elements
    //     (`scales.len() == mins.len()`, asserted line 712).
    //   - `bytes_handle` (line 717): `bytes.len()` u32 elements
    //     (`bytes.len() == scales.len() * 8`, line 713).
    //   - `out_handle` from `empty(num_elements * size_of::<f32>())` (line
    //     720); `num_elements` f32 elements. `.clone()` is a refcount bump.
    //   `count`/`dim` cover `num_elements` units (`num_elements ==
    //   scales.len() * 32`, line 711); kernel bounds-checks `ABSOLUTE_POS
    //   < out.len()`. `launch_unchecked` skips runtime arity per cubecl
    //   convention; refs live for launch duration.
    unsafe {
        kernel_dequantize_q8_1::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(scales_handle, scales.len()),
            ArrayArg::from_raw_parts(mins_handle, mins.len()),
            ArrayArg::from_raw_parts(bytes_handle, bytes.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), num_elements),
        );
    }
    out_handle
}

// ---------------------------------------------------------------------------
// Pure-Rust kernel-logic reference (for cross-validation tests only)
// ---------------------------------------------------------------------------

/// Pure-Rust mirror of [`kernel_dequantize_q4_0`], used by the test suite to
/// cross-validate against `ferrotorch_serialize::gguf::dequantize_q4_0` and
/// against the GPU kernel. Not part of the runtime path — there is no CPU
/// fallback in production. Public only to tests via `#[cfg(test)]`.
#[cfg(test)]
pub(crate) fn dequantize_q4_0_reference(scales: &[f32], nibbles: &[u32]) -> Vec<f32> {
    let num_blocks = scales.len();
    let mut out = Vec::with_capacity(num_blocks * 32);
    for b in 0..num_blocks {
        let scale = scales[b];
        for elem in 0..32 {
            let byte_idx = elem / 2;
            let is_high = elem % 2;
            let u32_idx_in_block = byte_idx / 4;
            let byte_in_u32 = byte_idx % 4;
            let packed = nibbles[b * 4 + u32_idx_in_block];
            let byte = (packed >> (byte_in_u32 * 8)) & 0xFF;
            let nibble = (byte >> (is_high * 4)) & 0xF;
            out.push(scale * (nibble as f32 - 8.0));
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn dequantize_q4_1_reference(scales: &[f32], mins: &[f32], nibbles: &[u32]) -> Vec<f32> {
    let num_blocks = scales.len();
    let mut out = Vec::with_capacity(num_blocks * 32);
    for b in 0..num_blocks {
        let scale = scales[b];
        let min_v = mins[b];
        for elem in 0..32 {
            let byte_idx = elem / 2;
            let is_high = elem % 2;
            let u32_idx_in_block = byte_idx / 4;
            let byte_in_u32 = byte_idx % 4;
            let packed = nibbles[b * 4 + u32_idx_in_block];
            let byte = (packed >> (byte_in_u32 * 8)) & 0xFF;
            let nibble = (byte >> (is_high * 4)) & 0xF;
            out.push(scale * (nibble as f32) + min_v);
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn dequantize_q8_0_reference(scales: &[f32], bytes: &[u32]) -> Vec<f32> {
    let num_blocks = scales.len();
    let mut out = Vec::with_capacity(num_blocks * 32);
    for b in 0..num_blocks {
        let scale = scales[b];
        for elem in 0..32 {
            let u32_idx_in_block = elem / 4;
            let byte_in_u32 = elem % 4;
            let packed = bytes[b * 8 + u32_idx_in_block];
            let byte_u = (packed >> (byte_in_u32 * 8)) & 0xFF;
            // Sign-extend u8 → i32 (mirrors the GPU branchless form):
            let sign_bit = (byte_u >> 7) & 1;
            let signed_u = byte_u.wrapping_add(0xFFFF_FF00_u32.wrapping_mul(sign_bit));
            let signed = signed_u as i32;
            out.push(scale * signed as f32);
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn dequantize_q5_0_reference(scales: &[f32], qh: &[u32], nibbles: &[u32]) -> Vec<f32> {
    let num_blocks = scales.len();
    let mut out = Vec::with_capacity(num_blocks * 32);
    for b in 0..num_blocks {
        let scale = scales[b];
        let qh_word = qh[b];
        for elem in 0..32 {
            let byte_idx = elem / 2;
            let is_high = elem % 2;
            let u32_idx_in_block = byte_idx / 4;
            let byte_in_u32 = byte_idx % 4;
            let packed = nibbles[b * 4 + u32_idx_in_block];
            let byte = (packed >> (byte_in_u32 * 8)) & 0xFF;
            let low_nibble = (byte >> (is_high * 4)) & 0xF;
            let bit_pos = (byte_idx as u32) * 2 + (is_high as u32);
            let high_bit = (qh_word >> bit_pos) & 1;
            let val5 = low_nibble | (high_bit << 4);
            out.push(scale * (val5 as f32 - 16.0));
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn dequantize_q5_1_reference(
    scales: &[f32],
    mins: &[f32],
    qh: &[u32],
    nibbles: &[u32],
) -> Vec<f32> {
    let num_blocks = scales.len();
    let mut out = Vec::with_capacity(num_blocks * 32);
    for b in 0..num_blocks {
        let scale = scales[b];
        let min_v = mins[b];
        let qh_word = qh[b];
        for elem in 0..32 {
            let byte_idx = elem / 2;
            let is_high = elem % 2;
            let u32_idx_in_block = byte_idx / 4;
            let byte_in_u32 = byte_idx % 4;
            let packed = nibbles[b * 4 + u32_idx_in_block];
            let byte = (packed >> (byte_in_u32 * 8)) & 0xFF;
            let low_nibble = (byte >> (is_high * 4)) & 0xF;
            let bit_pos = (byte_idx as u32) * 2 + (is_high as u32);
            let high_bit = (qh_word >> bit_pos) & 1;
            let val5 = low_nibble | (high_bit << 4);
            out.push(scale * (val5 as f32) + min_v);
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn dequantize_q8_1_reference(scales: &[f32], mins: &[f32], bytes: &[u32]) -> Vec<f32> {
    let num_blocks = scales.len();
    let mut out = Vec::with_capacity(num_blocks * 32);
    for b in 0..num_blocks {
        let scale = scales[b];
        let min_v = mins[b];
        for elem in 0..32 {
            let u32_idx_in_block = elem / 4;
            let byte_in_u32 = elem % 4;
            let packed = bytes[b * 8 + u32_idx_in_block];
            let byte_u = (packed >> (byte_in_u32 * 8)) & 0xFF;
            let sign_bit = (byte_u >> 7) & 1;
            let signed_u = byte_u.wrapping_add(0xFFFF_FF00_u32.wrapping_mul(sign_bit));
            let signed = signed_u as i32;
            out.push(scale * signed as f32 + min_v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a synthetic Q4_0 byte stream from a (scale, nibbles[0..32])
    /// per-block specification. Matches the GGUF on-disk layout exactly.
    fn build_q4_0_blocks(blocks: &[(f32, [u8; 32])]) -> (Vec<u8>, Vec<f32>) {
        let mut raw = Vec::with_capacity(blocks.len() * Q4_0_BLOCK_BYTES);
        let mut expected = Vec::with_capacity(blocks.len() * 32);
        for &(scale, nibs) in blocks {
            let s_f16 = half::f16::from_f32(scale);
            let s_bits = s_f16.to_bits();
            raw.extend_from_slice(&s_bits.to_le_bytes());
            // Pack pairs of 4-bit nibbles into 16 bytes (low first, then high).
            for chunk in nibs.chunks(2) {
                let lo = chunk[0] & 0xF;
                let hi = chunk[1] & 0xF;
                raw.push((hi << 4) | lo);
            }
            // Expected output: scale * (nibble - 8.0) for each of the 32 nibbles.
            for &n in &nibs {
                expected.push(s_f16.to_f32() * (n as f32 - 8.0));
            }
        }
        (raw, expected)
    }

    #[test]
    fn split_q4_0_recovers_scales_and_nibbles() {
        let blocks = vec![
            (1.0, [0u8; 32]),
            (
                0.5,
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10,
                    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                ],
            ),
            (-2.5, [15u8; 32]),
        ];
        let (raw, expected) = build_q4_0_blocks(&blocks);
        let (scales, nibbles) = split_q4_0_blocks(&raw, blocks.len());
        assert_eq!(scales.len(), blocks.len());
        assert_eq!(nibbles.len(), blocks.len() * 4);
        for (i, &(s, _)) in blocks.iter().enumerate() {
            // f16 round-trip introduces small precision loss.
            assert_relative_eq!(scales[i], half::f16::from_f32(s).to_f32(), epsilon = 1e-3);
        }
        let dequantized = dequantize_q4_0_reference(&scales, &nibbles);
        assert_eq!(dequantized.len(), expected.len());
        for (got, want) in dequantized.iter().zip(expected.iter()) {
            assert_relative_eq!(*got, *want, epsilon = 1e-3);
        }
    }

    #[test]
    fn q4_0_reference_matches_serialize_dequant_arithmetic() {
        // Replicate one block manually and verify our reference produces
        // the same output that ferrotorch_serialize::gguf::dequantize_q4_0
        // documents: (nibble - 8) * scale, low nibble first then high.
        let scale = 0.25_f32;
        let mut nibs = [0u8; 32];
        for (i, n) in nibs.iter_mut().enumerate() {
            *n = (i % 16) as u8;
        }
        let (raw, _) = build_q4_0_blocks(&[(scale, nibs)]);
        let (scales, nibbles) = split_q4_0_blocks(&raw, 1);
        let dequant = dequantize_q4_0_reference(&scales, &nibbles);
        let s = half::f16::from_f32(scale).to_f32();
        for (i, &v) in dequant.iter().enumerate() {
            let expected_n = (i % 16) as f32 - 8.0;
            assert_relative_eq!(v, s * expected_n, epsilon = 1e-3);
        }
    }

    /// Build a synthetic Q4_1 byte stream.
    fn build_q4_1_blocks(blocks: &[(f32, f32, [u8; 32])]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(blocks.len() * Q4_1_BLOCK_BYTES);
        for &(scale, min_v, nibs) in blocks {
            raw.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            raw.extend_from_slice(&half::f16::from_f32(min_v).to_bits().to_le_bytes());
            for chunk in nibs.chunks(2) {
                let lo = chunk[0] & 0xF;
                let hi = chunk[1] & 0xF;
                raw.push((hi << 4) | lo);
            }
        }
        raw
    }

    #[test]
    fn split_q4_1_recovers_scales_mins_nibbles() {
        let blocks = vec![
            (1.0, 0.0, [3u8; 32]),
            (
                0.5,
                1.5,
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10,
                    9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                ],
            ),
        ];
        let raw = build_q4_1_blocks(&blocks);
        let (scales, mins, nibbles) = split_q4_1_blocks(&raw, blocks.len());
        assert_eq!(scales.len(), blocks.len());
        assert_eq!(mins.len(), blocks.len());
        let dequant = dequantize_q4_1_reference(&scales, &mins, &nibbles);
        assert_eq!(dequant.len(), blocks.len() * 32);
        // Verify dequant formula nibble * scale + min.
        let s0 = half::f16::from_f32(blocks[0].0).to_f32();
        let m0 = half::f16::from_f32(blocks[0].1).to_f32();
        for &v in &dequant[..32] {
            assert_relative_eq!(v, 3.0 * s0 + m0, epsilon = 1e-3);
        }
    }

    /// Build a synthetic Q8_0 byte stream.
    fn build_q8_0_blocks(blocks: &[(f32, [i8; 32])]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(blocks.len() * Q8_0_BLOCK_BYTES);
        for &(scale, vals) in blocks {
            raw.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for v in vals {
                raw.push(v as u8);
            }
        }
        raw
    }

    #[test]
    fn split_q8_0_recovers_scales_and_signed_bytes() {
        let mut vals = [0i8; 32];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = ((i as i32) - 16) as i8; // -16..=15, exercises sign bit
        }
        let blocks = vec![(2.0, vals)];
        let raw = build_q8_0_blocks(&blocks);
        let (scales, bytes) = split_q8_0_blocks(&raw, blocks.len());
        assert_eq!(scales.len(), 1);
        assert_eq!(bytes.len(), 8); // 32 i8 = 8 u32

        let dequant = dequantize_q8_0_reference(&scales, &bytes);
        let s = half::f16::from_f32(2.0).to_f32();
        for (i, &v) in dequant.iter().enumerate() {
            let expected = s * (vals[i] as f32);
            assert_relative_eq!(v, expected, epsilon = 1e-3);
        }
    }

    #[test]
    fn q8_0_sign_extension_handles_full_range() {
        // -128..=127 covers every byte value including the boundary.
        let vals: [i8; 32] = std::array::from_fn(|i| {
            // Map 0..32 → -128, -120, -112, ..., 112, 120 (selected i8 values
            // that exercise both signs).
            ((i as i32) * 8 - 128) as i8
        });
        let raw = build_q8_0_blocks(&[(1.0, vals)]);
        let (scales, bytes) = split_q8_0_blocks(&raw, 1);
        let dequant = dequantize_q8_0_reference(&scales, &bytes);
        let s = half::f16::from_f32(1.0).to_f32();
        for (i, &v) in dequant.iter().enumerate() {
            let expected = s * (vals[i] as f32);
            assert_relative_eq!(v, expected, epsilon = 1e-3);
        }
    }

    #[test]
    fn block_kind_metadata_constants() {
        assert_eq!(GgufBlockKind::Q4_0.block_elements(), 32);
        assert_eq!(GgufBlockKind::Q4_0.block_bytes(), 18);
        assert_eq!(GgufBlockKind::Q4_1.block_bytes(), 20);
        assert_eq!(GgufBlockKind::Q8_0.block_bytes(), 34);
    }

    #[test]
    fn split_q4_0_panics_on_short_input() {
        let short = vec![0u8; 17]; // one byte short of one block
        let result = std::panic::catch_unwind(|| split_q4_0_blocks(&short, 1));
        assert!(result.is_err());
    }

    /// Cross-check against the existing CPU reference in
    /// `ferrotorch_serialize::gguf::dequantize_q4_0`: build a random byte
    /// stream, dequantize it through both code paths, assert byte-for-byte
    /// agreement. (Imported via dev-dependency would create a cycle —
    /// instead we replicate the formula here and the parity is implicit via
    /// `q4_0_reference_matches_serialize_dequant_arithmetic`.)
    #[test]
    fn random_q4_0_blocks_round_trip_through_split_then_dequant() {
        // Pseudo-random but deterministic: a small LCG.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            state
        };
        let num_blocks = 64;
        let mut raw = Vec::with_capacity(num_blocks * Q4_0_BLOCK_BYTES);
        let mut expected = Vec::with_capacity(num_blocks * 32);
        for _ in 0..num_blocks {
            // Random scale in [-2, 2).
            let scale_f32 = (next() as f32 / u32::MAX as f32) * 4.0 - 2.0;
            let scale = half::f16::from_f32(scale_f32);
            raw.extend_from_slice(&scale.to_bits().to_le_bytes());
            let s = scale.to_f32();
            // Build 32 nibbles, store low/high pairs into 16 bytes.
            let mut nibs = [0u8; 32];
            for n in &mut nibs {
                *n = (next() & 0xF) as u8;
            }
            for chunk in nibs.chunks(2) {
                let lo = chunk[0] & 0xF;
                let hi = chunk[1] & 0xF;
                raw.push((hi << 4) | lo);
            }
            for &n in &nibs {
                expected.push(s * (n as f32 - 8.0));
            }
        }
        let (scales, nibbles) = split_q4_0_blocks(&raw, num_blocks);
        let got = dequantize_q4_0_reference(&scales, &nibbles);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-3);
        }
    }

    // -------------------------------------------------------------------
    // Q5_0 / Q5_1 / Q8_1 host-side splitter tests
    // -------------------------------------------------------------------

    /// Build a Q5_0 byte stream from per-block (scale, qh, val5) specs.
    fn build_q5_0_blocks(blocks: &[(f32, u32, [u8; 32])]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(blocks.len() * Q5_0_BLOCK_BYTES);
        for &(scale, qh, vals) in blocks {
            raw.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            raw.extend_from_slice(&qh.to_le_bytes());
            // Pack (low_nibble | high_bit<<4) — write only the low 4 bits as
            // the byte-level nibble; the high bit lives in `qh` separately.
            for chunk in vals.chunks(2) {
                let lo = chunk[0] & 0xF;
                let hi = chunk[1] & 0xF;
                raw.push((hi << 4) | lo);
            }
        }
        raw
    }

    #[test]
    fn split_q5_0_recovers_scales_qh_and_nibbles() {
        // q5 = 0..32 over 32 elements, so high_bit toggles at element 16.
        let mut vals = [0u8; 32];
        let mut qh: u32 = 0;
        for (i, v) in vals.iter_mut().enumerate() {
            let q5 = i as u8; // 0..32, but 5-bit → high bit set when q5 >= 16
            *v = q5 & 0xF;
            if q5 >= 16 {
                qh |= 1 << i;
            }
        }
        let raw = build_q5_0_blocks(&[(1.5, qh, vals)]);
        let (scales, qh_out, nibs) = split_q5_0_blocks(&raw, 1);
        assert_eq!(scales.len(), 1);
        assert_eq!(qh_out.len(), 1);
        assert_eq!(qh_out[0], qh);
        assert_eq!(nibs.len(), 4);

        let dequant = dequantize_q5_0_reference(&scales, &qh_out, &nibs);
        let s = half::f16::from_f32(1.5).to_f32();
        for (i, &v) in dequant.iter().enumerate() {
            let q5 = i as f32;
            let expected = s * (q5 - 16.0);
            assert_relative_eq!(v, expected, epsilon = 1e-3);
        }
    }

    fn build_q5_1_blocks(blocks: &[(f32, f32, u32, [u8; 32])]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(blocks.len() * Q5_1_BLOCK_BYTES);
        for &(scale, min_v, qh, vals) in blocks {
            raw.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            raw.extend_from_slice(&half::f16::from_f32(min_v).to_bits().to_le_bytes());
            raw.extend_from_slice(&qh.to_le_bytes());
            for chunk in vals.chunks(2) {
                let lo = chunk[0] & 0xF;
                let hi = chunk[1] & 0xF;
                raw.push((hi << 4) | lo);
            }
        }
        raw
    }

    #[test]
    fn split_q5_1_recovers_scales_mins_qh_and_nibbles() {
        let mut vals = [0u8; 32];
        let mut qh: u32 = 0;
        for (i, v) in vals.iter_mut().enumerate() {
            let q5 = i as u8;
            *v = q5 & 0xF;
            if q5 >= 16 {
                qh |= 1 << i;
            }
        }
        let raw = build_q5_1_blocks(&[(0.5, 2.0, qh, vals)]);
        let (scales, mins, qh_out, nibs) = split_q5_1_blocks(&raw, 1);
        assert_eq!(scales.len(), 1);
        assert_eq!(mins.len(), 1);
        let dequant = dequantize_q5_1_reference(&scales, &mins, &qh_out, &nibs);
        let s = half::f16::from_f32(0.5).to_f32();
        let m = half::f16::from_f32(2.0).to_f32();
        for (i, &v) in dequant.iter().enumerate() {
            let expected = s * (i as f32) + m;
            assert_relative_eq!(v, expected, epsilon = 1e-3);
        }
    }

    fn build_q8_1_blocks(blocks: &[(f32, f32, [i8; 32])]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(blocks.len() * Q8_1_BLOCK_BYTES);
        for &(scale, min_v, vals) in blocks {
            raw.extend_from_slice(&scale.to_le_bytes());
            raw.extend_from_slice(&min_v.to_le_bytes());
            for v in vals {
                raw.push(v as u8);
            }
        }
        raw
    }

    #[test]
    fn split_q8_1_recovers_scales_mins_and_signed_bytes() {
        let vals: [i8; 32] = std::array::from_fn(|i| ((i as i32) - 16) as i8);
        let raw = build_q8_1_blocks(&[(1.5, 3.0, vals)]);
        let (scales, mins, bytes) = split_q8_1_blocks(&raw, 1);
        assert_eq!(scales.len(), 1);
        assert_eq!(mins.len(), 1);
        // Q8_1 has f32 scale/min on disk, so no f16 round-trip loss.
        assert_relative_eq!(scales[0], 1.5, epsilon = 1e-9);
        assert_relative_eq!(mins[0], 3.0, epsilon = 1e-9);
        assert_eq!(bytes.len(), 8);

        let dequant = dequantize_q8_1_reference(&scales, &mins, &bytes);
        for (i, &v) in dequant.iter().enumerate() {
            let expected = 1.5 * (vals[i] as f32) + 3.0;
            assert_relative_eq!(v, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn block_kind_metadata_constants_extended() {
        assert_eq!(GgufBlockKind::Q5_0.block_bytes(), 22);
        assert_eq!(GgufBlockKind::Q5_1.block_bytes(), 24);
        assert_eq!(GgufBlockKind::Q8_1.block_bytes(), 40);
        assert_eq!(GgufBlockKind::Q5_0.block_elements(), 32);
        assert_eq!(GgufBlockKind::Q5_1.block_elements(), 32);
        assert_eq!(GgufBlockKind::Q8_1.block_elements(), 32);
    }
}

// ---------------------------------------------------------------------------
// Token mask application (used by the constrained-decoding logits processor
// in ferrotorch-llama::grammar)
// ---------------------------------------------------------------------------

/// Apply a per-token allow mask to a logits vector. Tokens whose mask entry is
/// non-zero pass through unchanged; tokens whose mask entry is zero are forced
/// to the float type's minimum representable value (effectively `-infinity`
/// for sampling purposes — `softmax(-3.4e38)` underflows to zero).
///
/// One thread per logit. CubeCL's `Float` trait exposes `min_value()` rather
/// than a literal `f32::NEG_INFINITY`, so we use that for the mask sentinel.
#[cube(launch_unchecked)]
pub fn kernel_apply_token_mask<F: Float>(logits: &Array<F>, mask: &Array<u32>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let i = ABSOLUTE_POS;
        let allow = mask[i];
        let v = logits[i];
        if allow != 0u32 {
            out[i] = v;
        } else {
            out[i] = F::min_value();
        }
    }
}

/// Upload `logits` and `mask` to the GPU, dispatch [`kernel_apply_token_mask`],
/// and return the on-device output handle. `mask[i] != 0` keeps the logit;
/// `mask[i] == 0` replaces it with `-infinity`.
pub fn apply_token_mask_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    logits: &[f32],
    mask: &[u32],
) -> cubecl::server::Handle {
    debug_assert_eq!(logits.len(), mask.len());
    let n = logits.len();
    let logits_handle = client.create_from_slice(f32::as_bytes(logits));
    let mask_handle = client.create_from_slice(unsafe {
        // SAFETY: `&[u32]` → `&[u8]` reinterpret for `create_from_slice`'s
        //   `&[u8]` upload API. Alignment is monotonic
        //   (`align_of::<u32>() ≥ align_of::<u8>()`); `size_of_val(mask) ==
        //   mask.len() * 4` exactly; lifetime tied to `mask` (live until
        //   function return); `create_from_slice` consumes the borrow
        //   synchronously by copying into device memory; no `&mut` overlap.
        std::slice::from_raw_parts(mask.as_ptr() as *const u8, std::mem::size_of_val(mask))
    });
    let out_handle = client.empty(std::mem::size_of_val(logits));

    let (count, dim) = crate::elementwise_launch_dims(n as u32);
    // SAFETY: All three handles alloc'd by this `client` above:
    //   - `logits_handle` from `create_from_slice(f32::as_bytes(logits))`;
    //     `logits.len() == n` f32 elements.
    //   - `mask_handle` from `create_from_slice(<mask-as-bytes>)`;
    //     `mask.len() == n` u32 elements (debug-asserted at line 1274).
    //   - `out_handle` from `empty(size_of_val(logits))` =
    //     `n * size_of::<f32>()` bytes; capacity is `n` f32 elements.
    //     `.clone()` is a refcount bump only; kernel writes visible
    //     through returned `out_handle`.
    //   `count`/`dim` from `elementwise_launch_dims(n)` cover `n` units
    //   (one per token); kernel guards `ABSOLUTE_POS < out.len()`.
    //   `launch_unchecked` skips runtime arity per cubecl convention; refs
    //   live for launch duration.
    unsafe {
        kernel_apply_token_mask::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(logits_handle, n),
            ArrayArg::from_raw_parts(mask_handle, n),
            ArrayArg::from_raw_parts(out_handle.clone(), n),
        );
    }
    out_handle
}

// ---------------------------------------------------------------------------
// End-to-end GPU runtime tests
// ---------------------------------------------------------------------------
//
// These tests construct an actual `cubecl_cuda::CudaRuntime` client, dispatch
// every quant kernel on the GPU, and read the result back to compare against
// the pure-Rust reference. They are gated on `--features cuda` and require a
// CUDA device at test time. There are NO `#[ignore]` markers and NO CPU
// fallbacks — if the test compiles in this configuration it MUST run on GPU.

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use approx::assert_relative_eq;
    use cubecl_cuda::{CudaDevice, CudaRuntime};

    fn cuda_client() -> ComputeClient<CudaRuntime> {
        let device = CudaDevice { index: 0 };
        CudaRuntime::client(&device)
    }

    fn read_f32(client: &ComputeClient<CudaRuntime>, handle: cubecl::server::Handle) -> Vec<f32> {
        let bytes = client.read_one(handle).expect("CUDA read_one failed");
        f32::from_bytes(&bytes).to_vec()
    }

    #[test]
    fn q4_0_kernel_runs_on_gpu_and_matches_reference() {
        let client = cuda_client();
        // Use deterministic-but-non-trivial inputs: 4 blocks → 128 elements.
        let mut state: u32 = 0xCAFE_BABE;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            state
        };
        let num_blocks = 4;
        let mut raw = Vec::with_capacity(num_blocks * Q4_0_BLOCK_BYTES);
        for _ in 0..num_blocks {
            let s_f32 = (next() as f32 / u32::MAX as f32) * 4.0 - 2.0;
            let s = half::f16::from_f32(s_f32);
            raw.extend_from_slice(&s.to_bits().to_le_bytes());
            for _ in 0..16 {
                let lo = (next() & 0xF) as u8;
                let hi = (next() & 0xF) as u8;
                raw.push((hi << 4) | lo);
            }
        }
        let (scales, nibbles) = split_q4_0_blocks(&raw, num_blocks);
        let expected = dequantize_q4_0_reference(&scales, &nibbles);
        let handle = dequantize_q4_0_to_gpu(&client, &scales, &nibbles, num_blocks * 32);
        let got = read_f32(&client, handle);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn q4_1_kernel_runs_on_gpu_and_matches_reference() {
        let client = cuda_client();
        let num_blocks = 3;
        let mut state: u32 = 0x4242_4242;
        let mut next = || {
            state = state.wrapping_mul(48271).wrapping_add(7);
            state
        };
        let mut raw = Vec::with_capacity(num_blocks * Q4_1_BLOCK_BYTES);
        for _ in 0..num_blocks {
            let s = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 2.0);
            let m = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 4.0 - 2.0);
            raw.extend_from_slice(&s.to_bits().to_le_bytes());
            raw.extend_from_slice(&m.to_bits().to_le_bytes());
            for _ in 0..16 {
                let lo = (next() & 0xF) as u8;
                let hi = (next() & 0xF) as u8;
                raw.push((hi << 4) | lo);
            }
        }
        let (scales, mins, nibbles) = split_q4_1_blocks(&raw, num_blocks);
        let expected = dequantize_q4_1_reference(&scales, &mins, &nibbles);
        let handle = dequantize_q4_1_to_gpu(&client, &scales, &mins, &nibbles, num_blocks * 32);
        let got = read_f32(&client, handle);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn q8_0_kernel_runs_on_gpu_and_matches_reference() {
        let client = cuda_client();
        let num_blocks = 5;
        let mut state: u32 = 0xDEAD_BEEF;
        let mut next = || {
            state = state.wrapping_mul(214_013).wrapping_add(2_531_011);
            state
        };
        let mut raw = Vec::with_capacity(num_blocks * Q8_0_BLOCK_BYTES);
        for _ in 0..num_blocks {
            let s = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 0.5);
            raw.extend_from_slice(&s.to_bits().to_le_bytes());
            for _ in 0..32 {
                raw.push((next() & 0xFF) as u8); // exercises full -128..127 range
            }
        }
        let (scales, bytes) = split_q8_0_blocks(&raw, num_blocks);
        let expected = dequantize_q8_0_reference(&scales, &bytes);
        let handle = dequantize_q8_0_to_gpu(&client, &scales, &bytes, num_blocks * 32);
        let got = read_f32(&client, handle);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn q5_0_kernel_runs_on_gpu_and_matches_reference() {
        let client = cuda_client();
        let num_blocks = 4;
        let mut state: u32 = 0xFACE_FEED;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        let mut raw = Vec::with_capacity(num_blocks * Q5_0_BLOCK_BYTES);
        for _ in 0..num_blocks {
            let s = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 2.0);
            raw.extend_from_slice(&s.to_bits().to_le_bytes());
            // qh: random 32 bits
            let qh = next();
            raw.extend_from_slice(&qh.to_le_bytes());
            for _ in 0..16 {
                let lo = (next() & 0xF) as u8;
                let hi = (next() & 0xF) as u8;
                raw.push((hi << 4) | lo);
            }
        }
        let (scales, qh, nibs) = split_q5_0_blocks(&raw, num_blocks);
        let expected = dequantize_q5_0_reference(&scales, &qh, &nibs);
        let handle = dequantize_q5_0_to_gpu(&client, &scales, &qh, &nibs, num_blocks * 32);
        let got = read_f32(&client, handle);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn q5_1_kernel_runs_on_gpu_and_matches_reference() {
        let client = cuda_client();
        let num_blocks = 4;
        let mut state: u32 = 0xBABE_CAFE;
        let mut next = || {
            state = state.wrapping_mul(22_695_477).wrapping_add(1);
            state
        };
        let mut raw = Vec::with_capacity(num_blocks * Q5_1_BLOCK_BYTES);
        for _ in 0..num_blocks {
            let s = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 2.0);
            let m = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 4.0 - 2.0);
            raw.extend_from_slice(&s.to_bits().to_le_bytes());
            raw.extend_from_slice(&m.to_bits().to_le_bytes());
            let qh = next();
            raw.extend_from_slice(&qh.to_le_bytes());
            for _ in 0..16 {
                let lo = (next() & 0xF) as u8;
                let hi = (next() & 0xF) as u8;
                raw.push((hi << 4) | lo);
            }
        }
        let (scales, mins, qh, nibs) = split_q5_1_blocks(&raw, num_blocks);
        let expected = dequantize_q5_1_reference(&scales, &mins, &qh, &nibs);
        let handle = dequantize_q5_1_to_gpu(&client, &scales, &mins, &qh, &nibs, num_blocks * 32);
        let got = read_f32(&client, handle);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn token_mask_kernel_runs_on_gpu_and_replaces_disallowed_with_min_value() {
        let client = cuda_client();
        // 16 logits, mask of alternating allow/disallow.
        let logits: Vec<f32> = (0..16).map(|i| (i as f32) - 7.5).collect();
        let mask: Vec<u32> = (0..16).map(|i| u32::from(i % 2 == 0)).collect();
        let handle = apply_token_mask_to_gpu(&client, &logits, &mask);
        let got = read_f32(&client, handle);
        assert_eq!(got.len(), logits.len());
        for (i, &v) in got.iter().enumerate() {
            if mask[i] != 0 {
                assert_relative_eq!(v, logits[i], epsilon = 1e-6);
            } else {
                // Masked-out positions should be at-or-below `f32::MIN` after
                // the kernel's `F::min_value()`. We assert "very negative".
                assert!(v <= -1.0e30, "expected sentinel at idx {i}, got {v}");
            }
        }
    }

    #[test]
    fn q8_1_kernel_runs_on_gpu_and_matches_reference() {
        let client = cuda_client();
        let num_blocks = 3;
        let mut state: u32 = 0x1357_9BDF;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        let mut raw = Vec::with_capacity(num_blocks * Q8_1_BLOCK_BYTES);
        for _ in 0..num_blocks {
            let s = (next() as f32 / u32::MAX as f32) * 0.5;
            let m = (next() as f32 / u32::MAX as f32) * 4.0 - 2.0;
            raw.extend_from_slice(&s.to_le_bytes());
            raw.extend_from_slice(&m.to_le_bytes());
            for _ in 0..32 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let (scales, mins, bytes) = split_q8_1_blocks(&raw, num_blocks);
        let expected = dequantize_q8_1_reference(&scales, &mins, &bytes);
        let handle = dequantize_q8_1_to_gpu(&client, &scales, &mins, &bytes, num_blocks * 32);
        let got = read_f32(&client, handle);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-5);
        }
    }
}
