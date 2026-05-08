#!/usr/bin/env python3
"""Generate a minimal valid GGUF v3 binary for conformance tests.

Emits:  ferrotorch-serialize/tests/conformance/fixtures/serialize_gguf.bin

GGUF v3 wire layout (all integers little-endian):
  Magic     : 4 bytes  (b"GGUF")
  Version   : u32      (3)
  tensor_count     : u64
  metadata_kv_count: u64
  [metadata KV ...] : key(u64+bytes) + value_type(u32) + value
  [tensor info ...] : name(u64+bytes) + n_dims(u32) + dims(u64*) + ggml_type(u32) + offset(u64)
  [alignment pad]
  [tensor data ...]

GGUF value types:
  0=u8  1=i8  2=u16  3=i16  4=u32  5=i32  6=f32  7=bool  8=string
  9=array  10=u64  11=i64  12=f64

GGUF ggml types:
  0=F32  1=F16  2=Q4_0  3=Q4_1  6=Q5_0  7=Q5_1  8=Q8_0  9=Q8_1

Q8_0 block layout (34 bytes per block, 32 elements per block):
  2 bytes: f16 scale
  32 bytes: int8 quantized values (elements = int8 * scale)
"""

import os
import struct
from pathlib import Path

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
GGUF_VERSION = 3

# Value type tags
VTYPE_UINT32 = 4
VTYPE_STRING = 8

# GGML type tags
GGML_F32 = 0
GGML_Q8_0 = 8

# Q8_0 constants
Q8_0_BLOCK_SIZE = 32     # elements per block
Q8_0_BLOCK_BYTES = 34    # 2 (f16 scale) + 32 (int8 values)


def pack_gguf_string(s: str) -> bytes:
    """Pack a GGUF string: u64 length + UTF-8 bytes."""
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def f32_to_f16_bytes(val: float) -> bytes:
    """Convert an f32 to IEEE 754 f16 bytes (little-endian)."""
    # Use struct.pack with 'e' format (half-precision float, Python 3.6+).
    return struct.pack("<e", val)


def encode_q8_0_block(values: list[float], scale: float) -> bytes:
    """Encode one Q8_0 block: f16 scale + 32 int8 quantized values."""
    assert len(values) == Q8_0_BLOCK_SIZE
    block = f32_to_f16_bytes(scale)
    for v in values:
        if scale != 0.0:
            q = round(v / scale)
        else:
            q = 0
        q = max(-128, min(127, q))
        block += struct.pack("b", q)
    return block


def make_f32_tensor_data(values: list[float]) -> bytes:
    """Encode f32 tensor data as raw little-endian floats."""
    return struct.pack(f"<{len(values)}f", *values)


def make_q8_0_tensor_data(values: list[float]) -> bytes:
    """Encode a sequence of f32 values as Q8_0 blocks.

    Pads with zeros to the next multiple of Q8_0_BLOCK_SIZE.
    Returns raw Q8_0 block bytes.
    """
    # Pad to multiple of block size.
    pad = (-len(values)) % Q8_0_BLOCK_SIZE
    values = values + [0.0] * pad

    result = b""
    for i in range(0, len(values), Q8_0_BLOCK_SIZE):
        block_vals = values[i : i + Q8_0_BLOCK_SIZE]
        max_abs = max(abs(v) for v in block_vals)
        # Scale: max_abs maps to 127 (int8 max).
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        result += encode_q8_0_block(block_vals, scale)
    return result


def build_gguf_binary() -> bytes:
    """Build the complete GGUF v3 binary."""
    buf = bytearray()

    # -------------------------------------------------------------------------
    # Tensor definitions
    # -------------------------------------------------------------------------
    # weights.0: shape [4, 4], F32, 16 elements, deterministic row-major values
    w0_shape = [4, 4]
    w0_data_vals = [float(i) for i in range(16)]   # 0.0 .. 15.0
    w0_data = make_f32_tensor_data(w0_data_vals)

    # weights.1: shape [8], F32, 8 elements
    w1_shape = [8]
    w1_data_vals = [float(i + 0.5) for i in range(8)]   # 0.5, 1.5, ..., 7.5
    w1_data = make_f32_tensor_data(w1_data_vals)

    # weights.q8: shape [32], Q8_0, 32 elements (one block).
    # Values are a simple linear ramp scaled to [-1, 1].
    wq_shape = [32]
    wq_data_vals = [i / 16.0 - 1.0 for i in range(32)]  # -1.0 .. 0.9375
    wq_data = make_q8_0_tensor_data(wq_data_vals)

    tensors = [
        ("weights.0", w0_shape, GGML_F32, w0_data),
        ("weights.1", w1_shape, GGML_F32, w1_data),
        ("weights.q8", wq_shape, GGML_Q8_0, wq_data),
    ]

    # -------------------------------------------------------------------------
    # Metadata definitions
    # -------------------------------------------------------------------------
    def make_string_value(s: str) -> bytes:
        return pack_gguf_string(s)

    meta_arch_key = "general.architecture"
    meta_arch_val = make_string_value("llama")

    meta_name_key = "general.name"
    meta_name_val = make_string_value("test_model")

    metadata = [
        (meta_arch_key, VTYPE_STRING, meta_arch_val),
        (meta_name_key, VTYPE_STRING, meta_name_val),
    ]

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    buf += GGUF_MAGIC
    buf += struct.pack("<I", GGUF_VERSION)             # version u32
    buf += struct.pack("<Q", len(tensors))             # tensor_count u64
    buf += struct.pack("<Q", len(metadata))            # metadata_kv_count u64

    # -------------------------------------------------------------------------
    # Metadata KV entries
    # -------------------------------------------------------------------------
    for key, vtype, val_bytes in metadata:
        buf += pack_gguf_string(key)
        buf += struct.pack("<I", vtype)                # value type u32
        buf += val_bytes

    # -------------------------------------------------------------------------
    # Tensor info entries
    # -------------------------------------------------------------------------
    # Compute data offsets (contiguous in data section).
    data_offset = 0
    for name, shape, ggml_type, data_bytes in tensors:
        buf += pack_gguf_string(name)
        buf += struct.pack("<I", len(shape))           # n_dims u32
        for d in shape:
            buf += struct.pack("<Q", d)               # each dim u64
        buf += struct.pack("<I", ggml_type)            # ggml_type u32
        buf += struct.pack("<Q", data_offset)         # offset u64
        data_offset += len(data_bytes)

    # -------------------------------------------------------------------------
    # Alignment padding
    # -------------------------------------------------------------------------
    rem = len(buf) % DEFAULT_ALIGNMENT
    if rem != 0:
        buf += b"\x00" * (DEFAULT_ALIGNMENT - rem)

    # -------------------------------------------------------------------------
    # Data section
    # -------------------------------------------------------------------------
    for _name, _shape, _ggml_type, data_bytes in tensors:
        buf += data_bytes

    return bytes(buf)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "ferrotorch-serialize" / "tests" / "conformance" / "fixtures" / "serialize_gguf.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    binary = build_gguf_binary()
    out_path.write_bytes(binary)
    print(f"Wrote {len(binary)} bytes to {out_path}")

    # Sanity check: first 4 bytes must be the magic.
    assert binary[:4] == GGUF_MAGIC, "Magic mismatch!"
    assert len(binary) >= 256, f"Binary too small: {len(binary)} bytes"
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
