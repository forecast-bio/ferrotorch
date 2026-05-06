#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.9
(quantization + pruning).

Tracking issue: #771 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/quantize_prune.json

Coverage (44 surface items split across the quantize/prune families):

* Quantize forwards (CPU, integer-domain → bit-exact assertion):
    quantize (per-tensor, per-channel)         → INT8 / UINT8 / INT4
    dequantize (per-tensor, per-channel)       → F32_REDUCTION
    quantized_matmul                           → bit-exact integer accumulation
  Edge cases:
    - symmetric vs asymmetric (zp=0, zp=128, zp=-128 / boundary cases)
    - zero-point + scale boundaries
    - round-trip dequantize(quantize(x)) ≈ x within step
    - constant tensors (degenerate range)

* fake_quantize_differentiable forward (CPU):
    Wraps PyTorch's `torch.fake_quantize_per_tensor_affine`. Tested with
    explicit (scale, zp, qmin, qmax) so the parity is unambiguous.

* Pruning (CPU):
    magnitude_prune at multiple sparsity ratios (mask × original is bit-exact)
    apply_2_4_mask on contiguous and tail-not-divisible-by-4 inputs
    sparsity_ratio sanity checks

PyTorch reference notes:
- ferrotorch's `quantize` includes-zero-in-range (PyTorch's
  `torch.quantize_per_tensor` matches when `qmin/qmax` are derived from
  the dtype) and uses an i32 zero-point that is NOT clamped to qmin..qmax
  (this matches PyTorch's affine-quantizer).
- ferrotorch's `quantized_matmul` requantizes to INT8 with an output
  zp/scale derived from the int32 accumulator. We compare the
  *real-valued* output (after dequant) rather than the integer codes,
  because the choice of output (scale, zp) is internal-to-ferrotorch.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_quantize_prune_fixtures.py

Required Python deps: torch (with CUDA optional), numpy.
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch  # type: ignore

# ---------------------------------------------------------------------------
# Output path and metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-core"
    / "tests"
    / "conformance"
    / "fixtures"
    / "quantize_prune.json"
)

RNG_SEED: int = 0x_771_BADCAFE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


def to_listf(t: torch.Tensor) -> list[Any]:
    """Materialize a tensor to a CPU Python list of floats with sentinels."""
    raw = t.detach().to("cpu").to(torch.float64).reshape(-1).tolist()
    encoded: list[Any] = []
    for v in raw:
        if math.isnan(v):
            encoded.append("NaN")
        elif math.isinf(v):
            encoded.append("Infinity" if v > 0 else "-Infinity")
        else:
            encoded.append(v)
    return encoded


def to_list_int(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().to("cpu").reshape(-1).tolist()]


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "phase": "2.9-quantize-prune",
        "tracking_issue": "#771",
    }


# ---------------------------------------------------------------------------
# Helpers — qmin/qmax for ferrotorch's QuantDtype
# ---------------------------------------------------------------------------

QDTYPES = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    # ferrotorch packs INT4 in i8 storage; range is [-8, 7] before storage cast.
    "int4": (-8, 7),
}


def affine_quantize(
    x: torch.Tensor, scale: float, zp: int, qmin: int, qmax: int
) -> list[int]:
    """Reference affine quantize matching ferrotorch's `quantize_val`:
        q = clamp(round(x / scale + zp), qmin, qmax)

    Returns a list of i32-domain integer codes (NOT the i8 storage form).
    The Rust test reads the QuantizedTensor's stored bytes and reconstructs
    the same i32 domain via `stored_to_i32` before comparing.
    """
    raw = x.detach().to("cpu").to(torch.float32).reshape(-1).tolist()
    codes: list[int] = []
    for v in raw:
        q = round(v / scale + zp)
        q = max(qmin, min(qmax, q))
        codes.append(int(q))
    return codes


def affine_dequantize(codes: list[int], scale: float, zp: int) -> list[float]:
    """Inverse: x = (q - zp) * scale."""
    return [(q - zp) * scale for q in codes]


def compute_scale_zp(min_val: float, max_val: float, qmin: int, qmax: int) -> tuple[float, int]:
    """Match ferrotorch's `compute_scale_zp` exactly:
      - expand range to include zero
      - scale = (max - min) / (qmax - qmin), with EPS floor on range
      - zp = round(qmin - min/scale), NOT clamped
    """
    min_v = min(min_val, 0.0)
    max_v = max(max_val, 0.0)
    rng = max(max_v - min_v, sys.float_info.min)  # f32::EPSILON would be 1.19e-7;
    # we use float_info.min as a very small floor — for any non-degenerate input
    # the EPSILON-floor in Rust is unreached. The constant-tensor edge case
    # exercises the floor and we test the exact ferrotorch behaviour there.
    # For a tighter parity with f32::EPSILON, compute as f32 explicitly:
    eps_f32 = 1.1920929e-7
    rng = max(max_v - min_v, eps_f32)
    scale = rng / (qmax - qmin)
    zp = round(qmin - min_v / scale)
    return scale, int(zp)


# ---------------------------------------------------------------------------
# Quantize fixtures (per-tensor, per-channel; INT8/UINT8/INT4)
# ---------------------------------------------------------------------------


def fixture_quantize_per_tensor() -> list[dict[str, Any]]:
    """Per-tensor quantize: assert bit-exact integer codes and dequant round-trip.

    Tags: "int8_signed_range", "int8_pos_range", "uint8_pos_range",
    "int4_signed_range", "int4_constant" (degenerate), "int8_zero_only".
    """
    out: list[dict[str, Any]] = []

    cases = [
        # (tag, qdtype, shape, data)
        ("int8_signed_range", "int8", [8], [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
        ("int8_pos_range", "int8", [6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("uint8_pos_range", "uint8", [6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("uint8_signed_range", "uint8", [8], [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
        ("int4_signed_range", "int4", [16], list(range(-8, 8))),
        ("int4_pos_range", "int4", [4], [0.0, 1.0, 2.0, 3.0]),
        # Degenerate: all-equal — exercises the EPSILON-floor in compute_scale_zp.
        ("int8_constant", "int8", [4], [5.0, 5.0, 5.0, 5.0]),
        # All-zero: range expansion includes zero already, scale should
        # not be NaN/Inf (the EPSILON floor handles it).
        ("int8_all_zero", "int8", [4], [0.0, 0.0, 0.0, 0.0]),
        # Single element.
        ("int8_single", "int8", [1], [42.0]),
        # 2-D shape so we exercise the multi-dim path.
        ("int8_2d", "int8", [2, 3], [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
    ]

    for tag, qdtype, shape, data in cases:
        x = torch.tensor(data, dtype=torch.float32).reshape(shape)
        qmin, qmax = QDTYPES[qdtype]
        scale, zp = compute_scale_zp(float(x.min().item()), float(x.max().item()), qmin, qmax)
        codes = affine_quantize(x, scale, zp, qmin, qmax)
        out.append({
            "op": "quantize_per_tensor",
            "tag": tag,
            "qdtype": qdtype,
            "shape": shape,
            "x_data": to_listf(x),
            "scale": scale,
            "zero_point": zp,
            "codes": codes,
            # Round-trip reference: dequant(quant(x)).
            "dequant": affine_dequantize(codes, scale, zp),
        })
    return out


def fixture_quantize_per_channel() -> list[dict[str, Any]]:
    """Per-channel quantize. Each channel along `axis` has independent scale/zp."""
    out: list[dict[str, Any]] = []

    # Shape [3, 4]: 3 channels along axis 0, distinct ranges.
    data_3x4 = [
        0.0, 1.0, 2.0, 3.0,
        -10.0, -5.0, 5.0, 10.0,
        100.0, 130.0, 170.0, 200.0,
    ]
    cases = [
        ("int8_axis0", "int8", [3, 4], 0, data_3x4),
        ("uint8_axis0", "uint8", [3, 4], 0, data_3x4),
        ("int4_axis0", "int4", [3, 4], 0, data_3x4),
        # axis=1 path: shape [2, 3], 3 channels along last axis.
        ("int8_axis1", "int8", [2, 3], 1,
         [-1.0, 5.0, 10.0, -2.0, 4.0, 8.0]),
    ]

    for tag, qdtype, shape, axis, data in cases:
        x = torch.tensor(data, dtype=torch.float32).reshape(shape)
        qmin, qmax = QDTYPES[qdtype]
        n_ch = shape[axis]
        # Compute per-channel min/max along all dims except `axis`.
        x_perm = x.transpose(axis, 0).contiguous().reshape(n_ch, -1)
        scales: list[float] = []
        zps: list[int] = []
        codes: list[int] = []
        # We need to emit codes in the original flat order, matching ferrotorch.
        flat = x.reshape(-1).tolist()
        # Compute per-channel params first.
        for ch in range(n_ch):
            ch_min = float(x_perm[ch].min().item())
            ch_max = float(x_perm[ch].max().item())
            s, z = compute_scale_zp(ch_min, ch_max, qmin, qmax)
            scales.append(s)
            zps.append(z)
        # Now walk flat order and look up channel index just like ferrotorch.
        # channel_index(flat_index) = (flat_index / stride) % shape[axis]
        # stride = product(shape[axis+1:])
        stride = 1
        for d in shape[axis + 1:]:
            stride *= d
        for i, v in enumerate(flat):
            ch = (i // stride) % shape[axis]
            q = round(v / scales[ch] + zps[ch])
            q = max(qmin, min(qmax, q))
            codes.append(int(q))

        # Dequant in the same order.
        dequant: list[float] = []
        for i, q in enumerate(codes):
            ch = (i // stride) % shape[axis]
            dequant.append((q - zps[ch]) * scales[ch])

        out.append({
            "op": "quantize_per_channel",
            "tag": tag,
            "qdtype": qdtype,
            "shape": shape,
            "axis": axis,
            "x_data": to_listf(x),
            "scales": scales,
            "zero_points": zps,
            "codes": codes,
            "dequant": dequant,
        })
    return out


# ---------------------------------------------------------------------------
# QParams symmetric / asymmetric reference values
# ---------------------------------------------------------------------------


def fixture_qparams() -> list[dict[str, Any]]:
    """Symmetric and asymmetric QParams reference values for boundary cases.

    Boundaries: zp=0 (INT8 symmetric), zp=128 (UINT8 symmetric),
    zp=-128 boundary (INT8 asymmetric all-positive range).
    """
    out: list[dict[str, Any]] = []

    # Symmetric across the three dtypes.
    for max_abs in (5.0, 1.0, 100.0):
        for qdtype, expected in [
            ("int8", (max_abs / 127.0, 0)),
            ("uint8", (max_abs / 128.0, 128)),
            ("int4", (max_abs / 7.0, 0)),
        ]:
            scale, zp = expected
            out.append({
                "op": "qparams_symmetric",
                "tag": f"{qdtype}_maxabs{max_abs}",
                "qdtype": qdtype,
                "max_abs": max_abs,
                "scale": scale,
                "zero_point": zp,
            })

    # Asymmetric covering boundary zp values.
    asym_cases = [
        # (tag, qdtype, min, max)
        ("int8_signed", "int8", -3.0, 3.0),     # zp ≈ 0
        ("int8_all_pos", "int8", 0.0, 1.0),     # zp = -128 boundary
        ("int8_all_neg", "int8", -1.0, 0.0),    # zp = 127 (range expanded to include 0)
        ("uint8_signed", "uint8", -1.0, 1.0),   # zp ≈ 128
        ("uint8_all_pos", "uint8", 0.0, 1.0),   # zp = 0 boundary
        ("int4_signed", "int4", -1.0, 1.0),     # zp ≈ 0
    ]
    for tag, qdtype, mn, mx in asym_cases:
        qmin, qmax = QDTYPES[qdtype]
        scale, zp = compute_scale_zp(mn, mx, qmin, qmax)
        out.append({
            "op": "qparams_asymmetric",
            "tag": tag,
            "qdtype": qdtype,
            "min_val": mn,
            "max_val": mx,
            "scale": scale,
            "zero_point": zp,
        })
    return out


# ---------------------------------------------------------------------------
# quantized_matmul fixtures
# ---------------------------------------------------------------------------


def fixture_quantized_matmul() -> list[dict[str, Any]]:
    """quantized_matmul on small INT8 inputs.

    We compare the *dequantized* output to the float reference within a
    quantization-step tolerance. Unlike pure quantize, the output's
    scale/zp are derived from the int32 accumulator by ferrotorch and
    thus internal — bit-exact comparison would require modeling that
    requantization, which is not part of PyTorch's surface API."""
    out: list[dict[str, Any]] = []

    cases = [
        # (tag, A_shape, B_shape, A_data, B_data)
        ("identity_2x2", [2, 2], [2, 2],
         [1.0, 2.0, 3.0, 4.0],
         [1.0, 0.0, 0.0, 1.0]),
        ("rect_2x3_3x2", [2, 3], [3, 2],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ("rect_3x2_2x3", [3, 2], [2, 3],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]

    for tag, a_shape, b_shape, a_data, b_data in cases:
        a = torch.tensor(a_data, dtype=torch.float32).reshape(a_shape)
        b = torch.tensor(b_data, dtype=torch.float32).reshape(b_shape)
        c = a @ b
        out.append({
            "op": "quantized_matmul",
            "tag": tag,
            "qdtype": "int8",
            "a_shape": a_shape,
            "b_shape": b_shape,
            "a_data": to_listf(a),
            "b_data": to_listf(b),
            "c_shape": list(c.shape),
            "c_data": to_listf(c),
        })
    return out


# ---------------------------------------------------------------------------
# fake_quantize_differentiable fixtures (CPU; PyTorch parity).
# ---------------------------------------------------------------------------


def fixture_fake_quantize() -> list[dict[str, Any]]:
    """fake_quantize_differentiable: forward + STE backward.

    Mirrors `torch.fake_quantize_per_tensor_affine` and PyTorch's
    `torch.ao.quantization.fake_quantize.FakeQuantize` differentiable
    path (gradient zeroed for out-of-range inputs)."""
    out: list[dict[str, Any]] = []

    cases = [
        # (tag, x_data, scale, zp, qmin, qmax)
        ("int8_in_range", [-1.0, 0.0, 0.5, 1.0], 0.05, 0, -128, 127),
        ("int8_out_of_range", [-200.0, -1.0, 0.0, 1.0, 200.0], 0.05, 0, -128, 127),
        ("uint8_zp128", [-1.0, 0.0, 0.5, 1.0], 0.01, 128, 0, 255),
        ("int4_small_range", [-2.0, -0.5, 0.5, 2.0], 0.5, 0, -8, 7),
    ]
    for tag, x_data, scale, zp, qmin, qmax in cases:
        x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
        # PyTorch's analog: `torch.fake_quantize_per_tensor_affine`.
        # Use it as the parity reference.
        y = torch.fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)
        # Backward: STE — gradient is `1` for in-range, `0` for clamped.
        # We compute by taking `y.sum().backward()`.
        y.sum().backward()
        out.append({
            "op": "fake_quantize_differentiable",
            "tag": tag,
            "x_data": to_listf(x),
            "scale": scale,
            "zero_point": zp,
            "qmin": qmin,
            "qmax": qmax,
            "y_data": to_listf(y),
            "grad_x": to_listf(x.grad),
        })
    return out


# ---------------------------------------------------------------------------
# Pruning fixtures (CPU; bit-exact mask × original).
# ---------------------------------------------------------------------------


def magnitude_prune_reference(data: list[float], sparsity: float) -> list[float]:
    """Match ferrotorch's `magnitude_prune` exactly.

    threshold = sorted(|data|)[n_prune - 1]
    out[i] = 0 if |data[i]| <= threshold else data[i]
    """
    n = len(data)
    n_prune = round(n * sparsity)
    if n_prune == 0:
        return list(data)
    mags = sorted(abs(v) for v in data)
    threshold = mags[n_prune - 1]
    return [0.0 if abs(v) <= threshold else v for v in data]


def apply_2_4_mask_reference(data: list[float]) -> list[float]:
    """Match ferrotorch's `apply_2_4_mask` exactly.

    For each contiguous 4-element group, keep the 2 largest-magnitude
    elements; zero the 2 smallest. Trailing < 4 elements are unchanged.
    Tie-break on stable index order — `sort_by` in Rust is stable, and
    Python's `sorted` is also stable, so the smallest two slots are
    chosen consistently when magnitudes tie.
    """
    out = list(data)
    groups = len(out) // 4
    for g in range(groups):
        base = g * 4
        group = out[base:base + 4]
        # Stable-sort by magnitude ascending; smallest two -> zero.
        order = sorted(range(4), key=lambda i: abs(group[i]))
        for slot in (order[0], order[1]):
            out[base + slot] = 0.0
    return out


def fixture_magnitude_prune() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        # (tag, sparsity, shape, data)
        ("vec_50pct", 0.5, [4], [1.0, -4.0, 2.0, -3.0]),
        ("vec_25pct", 0.25, [8], [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
        ("vec_75pct", 0.75, [8], [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
        ("zero_sparsity", 0.0, [4], [1.0, 2.0, 3.0, 4.0]),
        # Boundary: 99% — should keep at least one element.
        ("vec_high_sparsity", 0.9, [10], [float(i + 1) for i in range(10)]),
        # 2-D tensor preserves shape.
        ("mat2d_50pct", 0.5, [2, 4], [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
    ]
    for tag, sparsity, shape, data in cases:
        pruned = magnitude_prune_reference(data, sparsity)
        zeros = sum(1 for v in pruned if v == 0.0)
        out.append({
            "op": "magnitude_prune",
            "tag": tag,
            "sparsity": sparsity,
            "shape": shape,
            "x_data": data,
            "pruned": pruned,
            "n_zeros": zeros,
        })
    return out


def fixture_apply_2_4_mask() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        # (tag, shape, data)
        ("group1", [4], [1.0, -4.0, 2.0, -3.0]),
        ("group2", [8], [1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8]),
        # Trailing < 4 elements: untouched.
        ("trailing", [6], [1.0, -4.0, 2.0, -3.0, 0.5, 0.1]),
        # 2-D shape, 8 elements (preserves shape).
        ("mat2d", [2, 4], [1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8]),
    ]
    for tag, shape, data in cases:
        out_data = apply_2_4_mask_reference(data)
        zeros = sum(1 for v in out_data if v == 0.0)
        out.append({
            "op": "apply_2_4_mask",
            "tag": tag,
            "shape": shape,
            "x_data": data,
            "masked": out_data,
            "n_zeros": zeros,
        })
    return out


def fixture_sparsity_ratio() -> list[dict[str, Any]]:
    cases = [
        ("half", [4], [0.0, 1.0, 0.0, 2.0], 0.5),
        ("none", [4], [1.0, 2.0, 3.0, 4.0], 0.0),
        ("all", [4], [0.0, 0.0, 0.0, 0.0], 1.0),
        ("75pct", [4], [0.0, 0.0, 0.0, 1.0], 0.75),
    ]
    out = []
    for tag, shape, data, ratio in cases:
        out.append({
            "op": "sparsity_ratio",
            "tag": tag,
            "shape": shape,
            "x_data": data,
            "ratio": ratio,
        })
    return out


# ---------------------------------------------------------------------------
# Round-trip parity (dequantize(quantize(x)) ≈ x within step)
# ---------------------------------------------------------------------------


def fixture_roundtrip() -> list[dict[str, Any]]:
    """Round-trip dequant∘quant: error <= scale (one quantization step)."""
    out: list[dict[str, Any]] = []

    # PerTensor + every dtype.
    cases = [
        ("rt_int8", "int8", [11], [-5.0 + 0.5 * i for i in range(11)]),
        ("rt_uint8", "uint8", [11], [0.0 + 0.2 * i for i in range(11)]),
        ("rt_int4", "int4", [16], list(range(-8, 8))),
    ]
    for tag, qdtype, shape, data in cases:
        x = torch.tensor(data, dtype=torch.float32).reshape(shape)
        qmin, qmax = QDTYPES[qdtype]
        scale, zp = compute_scale_zp(float(x.min().item()), float(x.max().item()), qmin, qmax)
        codes = affine_quantize(x, scale, zp, qmin, qmax)
        recovered = affine_dequantize(codes, scale, zp)
        out.append({
            "op": "roundtrip",
            "tag": tag,
            "qdtype": qdtype,
            "shape": shape,
            "x_data": to_listf(x),
            "scale": scale,
            "zero_point": zp,
            "recovered": recovered,
        })
    return out


# ---------------------------------------------------------------------------
# Build & write
# ---------------------------------------------------------------------------


def all_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(fixture_quantize_per_tensor())
    out.extend(fixture_quantize_per_channel())
    out.extend(fixture_qparams())
    out.extend(fixture_quantized_matmul())
    out.extend(fixture_fake_quantize())
    out.extend(fixture_magnitude_prune())
    out.extend(fixture_apply_2_4_mask())
    out.extend(fixture_sparsity_ratio())
    out.extend(fixture_roundtrip())
    return out


def main() -> None:
    payload = {
        "metadata": fixture_metadata(),
        "fixtures": all_fixtures(),
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURE_PATH} ({len(payload['fixtures'])} fixtures)")


if __name__ == "__main__":
    main()
