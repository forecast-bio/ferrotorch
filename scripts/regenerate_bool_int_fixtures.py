#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.13
(bool tensors and int tensors).

Tracking issue: #775 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/bool_int.json

Coverage (the 42 surface items grouped by type and category):

* `BoolTensor` — logical ops (and, or, xor, not), reductions (any, all,
  count_true), comparisons returning bool (gt, lt, ge, le, eq_t, ne),
  constructors (zeros, ones, from_vec, from_slice, from_predicate),
  shape introspection (shape, numel, ndim, data), reshape, and the
  bool→float conversion (`to_float`).

* `IntTensor<I>` — constructors (from_vec, from_slice, zeros, arange,
  scalar), shape introspection (shape, numel, ndim, data, dtype_name),
  cast (i32↔i64, in-range and out-of-range to surface the OOB error),
  reshape.

PyTorch parity boundary:
  - `BoolTensor` ↔ `torch.tensor(..., dtype=torch.bool)`.
  - `IntTensor<i32>` ↔ `torch.tensor(..., dtype=torch.int32)`.
  - `IntTensor<i64>` ↔ `torch.tensor(..., dtype=torch.int64)`.
  - Comparisons on `Tensor<f32/f64>` returning `BoolTensor` ↔
    `torch.gt(a, b)` / `torch.lt(a, b)` etc.
  - `BoolTensor::to_float::<T>()` ↔ `mask.to(dtype=torch.float32)` (true=1.0,
    false=0.0).

Edge cases REQUIRED by the dispatch:
  - Empty bool tensor: `all() == True`, `any() == False` (PyTorch convention
    for the identity element of empty reductions).
  - All-true / all-false fast paths.
  - Integer overflow: PyTorch wraps in modular arithmetic for int64; we
    don't do arithmetic at the API level on `IntTensor`, but the cast
    helper checks bounds — the OOB case is exercised separately.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_bool_int_fixtures.py

Required Python deps: torch (CPU is sufficient; CUDA not needed because
ferrotorch's `BoolTensor` and `IntTensor` are CPU-resident by design).
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
    / "bool_int.json"
)

# `BoolTensor` and `IntTensor` are CPU-resident by definition (see the
# `bool_tensor.rs` / `int_tensor.rs` doc headers). PyTorch fixtures stay
# on CPU; there is no `cuda:0` lane to mirror because ferrotorch has no
# GPU dispatch for these types.
DEVICES: list[str] = ["cpu"]

RNG_SEED: int = 0xB00_1A7  # bool/int — phase 2.13
torch.manual_seed(RNG_SEED)


def to_list_bool(t: torch.Tensor) -> list[bool]:
    return [bool(v) for v in t.detach().to("cpu").reshape(-1).tolist()]


def to_list_int(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().to("cpu").reshape(-1).tolist()]


def to_list_f64(t: torch.Tensor) -> list[Any]:
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


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "devices": DEVICES,
        "phase": "2.13-bool-int",
        "tracking_issue": "#775",
    }


# ---------------------------------------------------------------------------
# BoolTensor — constructors, predicates, shape introspection
# ---------------------------------------------------------------------------


def fixture_bool_constructors() -> list[dict[str, Any]]:
    """`zeros` / `ones` / `from_vec` constructors. PyTorch reference is
    `torch.zeros(..., dtype=torch.bool)` / `torch.ones(..., dtype=torch.bool)`.
    Each fixture records the expected flat data + shape so the Rust test
    can assert numel / ndim / shape / data() against the reference."""
    out: list[dict[str, Any]] = []
    shapes: list[tuple[list[int], str]] = [
        ([4], "vec1d"),
        ([2, 3], "mat2d"),
        ([2, 2, 3], "ten3d"),
        ([0], "empty1d"),
    ]
    for shape, tag in shapes:
        z = torch.zeros(shape, dtype=torch.bool)
        o = torch.ones(shape, dtype=torch.bool)
        out.append({
            "op": "bool_zeros",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "out_data": to_list_bool(z),
            "ndim": z.ndim,
            "numel": z.numel(),
        })
        out.append({
            "op": "bool_ones",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "out_data": to_list_bool(o),
            "ndim": o.ndim,
            "numel": o.numel(),
        })
    # from_vec / from_slice: explicit data + shape pairs. We pick
    # patterns that exercise mixed-truth, all-true, all-false, and a
    # 0-d "scalar" (shape=[], numel=1).
    patterns: list[tuple[list[bool], list[int], str]] = [
        ([True, False, True, False], [4], "alt4"),
        ([True, True, True, True], [4], "alltrue4"),
        ([False, False, False, False], [4], "allfalse4"),
        ([True, False, True, False, True, False], [2, 3], "mixed2x3"),
        ([True], [], "scalar_true"),
        ([False], [], "scalar_false"),
    ]
    for data, shape, tag in patterns:
        out.append({
            "op": "bool_from_vec",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "in_data": data,
            "out_data": data,  # round-trip
            "ndim": len(shape),
            "numel": max(1, math.prod(shape)) if shape else 1,
        })
        out.append({
            "op": "bool_from_slice",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "in_data": data,
            "out_data": data,
            "ndim": len(shape),
            "numel": max(1, math.prod(shape)) if shape else 1,
        })
    return out


def fixture_bool_from_predicate() -> list[dict[str, Any]]:
    """`BoolTensor::from_predicate(t, |x| pred)` — build a mask from a
    `Tensor<T>` and a closure. PyTorch reference is `pred(t)` evaluated
    element-wise.

    We test three predicates against ferrotorch's static set: `> 0`,
    `< 0.5`, `is_finite` (the latter exercises the NaN/Inf path).
    """
    out: list[dict[str, Any]] = []
    cases: list[tuple[list[float], list[int], str]] = [
        ([-1.0, 0.0, 1.0, 2.0], [4], "pos1d"),
        ([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [2, 3], "pos2d"),
        ([float("nan"), 1.0, float("inf"), -float("inf"), 0.0], [5], "finite1d"),
    ]
    for dtype_name, torch_dtype in (("float32", torch.float32), ("float64", torch.float64)):
        for data, shape, tag in cases:
            a = torch.tensor(data, dtype=torch_dtype).reshape(shape)
            # Three predicates are encoded by `predicate` field; the test
            # picks the matching closure. We always emit the gt0 case;
            # the other predicates only apply where they make sense.
            mask_gt0 = (a > 0).to(torch.bool)
            out.append({
                "op": "bool_from_predicate",
                "tag": f"{tag}_gt0",
                "predicate": "gt0",
                "dtype": dtype_name,
                "device": "cpu",
                "shape": shape,
                "a_data": to_list_f64(a),
                "out_data": to_list_bool(mask_gt0),
            })
            mask_lt_half = (a < 0.5).to(torch.bool)
            # NaN-safe: torch's `<` returns False where either side is NaN.
            out.append({
                "op": "bool_from_predicate",
                "tag": f"{tag}_lt_half",
                "predicate": "lt_half",
                "dtype": dtype_name,
                "device": "cpu",
                "shape": shape,
                "a_data": to_list_f64(a),
                "out_data": to_list_bool(mask_lt_half),
            })
            if "finite" in tag:
                mask_finite = torch.isfinite(a).to(torch.bool)
                out.append({
                    "op": "bool_from_predicate",
                    "tag": f"{tag}_is_finite",
                    "predicate": "is_finite",
                    "dtype": dtype_name,
                    "device": "cpu",
                    "shape": shape,
                    "a_data": to_list_f64(a),
                    "out_data": to_list_bool(mask_finite),
                })
    return out


# ---------------------------------------------------------------------------
# BoolTensor — logical ops (not, and, or, xor)
# ---------------------------------------------------------------------------


def fixture_bool_logical() -> list[dict[str, Any]]:
    """Pointwise NOT/AND/OR/XOR. PyTorch references are
    `torch.logical_not / logical_and / logical_or / logical_xor`."""
    out: list[dict[str, Any]] = []
    pairs: list[tuple[list[bool], list[bool], list[int], str]] = [
        # `__init__` 4-element vectors
        ([True, False, True, False], [True, True, False, False], [4], "vec4"),
        # 2x3 matrix
        (
            [True, False, True, False, True, False],
            [True, True, False, False, True, True],
            [2, 3],
            "mat2x3",
        ),
        # All-true vs all-false — fast paths
        ([True, True, True], [False, False, False], [3], "ones_vs_zeros"),
        ([True, True, True, True], [True, True, True, True], [4], "all_true"),
        ([False, False, False, False], [False, False, False, False], [4], "all_false"),
        # Empty
        ([], [], [0], "empty"),
    ]
    for a_data, b_data, shape, tag in pairs:
        a = torch.tensor(a_data, dtype=torch.bool).reshape(shape)
        b = torch.tensor(b_data, dtype=torch.bool).reshape(shape)
        out.append({
            "op": "bool_not",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "a_data": a_data,
            "out_data": to_list_bool(torch.logical_not(a)),
        })
        out.append({
            "op": "bool_and",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "a_data": a_data,
            "b_data": b_data,
            "out_data": to_list_bool(torch.logical_and(a, b)),
        })
        out.append({
            "op": "bool_or",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "a_data": a_data,
            "b_data": b_data,
            "out_data": to_list_bool(torch.logical_or(a, b)),
        })
        out.append({
            "op": "bool_xor",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "a_data": a_data,
            "b_data": b_data,
            "out_data": to_list_bool(torch.logical_xor(a, b)),
        })
    return out


# ---------------------------------------------------------------------------
# BoolTensor — reductions (any / all / count_true)
# ---------------------------------------------------------------------------


def fixture_bool_reductions() -> list[dict[str, Any]]:
    """`any` / `all` / `count_true`. PyTorch:
       any  ↔ tensor.any().item()
       all  ↔ tensor.all().item()
       count_true ↔ tensor.sum().item()  (bool → int promotion).

    Empty case: PyTorch's reduction identity convention — any([]) = False,
    all([]) = True, sum([]) = 0. ferrotorch's iter-`any`/`all` match the
    Rust stdlib semantics, which agree with PyTorch for these specific
    identities."""
    out: list[dict[str, Any]] = []
    cases: list[tuple[list[bool], list[int], str]] = [
        ([True, False, True], [3], "mixed"),
        ([True, True, True], [3], "alltrue"),
        ([False, False, False], [3], "allfalse"),
        ([True], [1], "single_true"),
        ([False], [1], "single_false"),
        ([], [0], "empty"),  # any=False, all=True
        # 2D
        ([True, False, True, False], [2, 2], "mat2x2_mixed"),
        ([False, False, False, False, False, False], [2, 3], "mat2x3_allfalse"),
    ]
    for data, shape, tag in cases:
        if data:
            a = torch.tensor(data, dtype=torch.bool).reshape(shape)
        else:
            a = torch.empty(0, dtype=torch.bool)
        out.append({
            "op": "bool_any",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "a_data": data,
            "out_scalar_bool": bool(a.any().item()) if a.numel() > 0 else False,
        })
        out.append({
            "op": "bool_all",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "a_data": data,
            "out_scalar_bool": bool(a.all().item()) if a.numel() > 0 else True,
        })
        out.append({
            "op": "bool_count_true",
            "tag": tag,
            "device": "cpu",
            "shape": shape,
            "a_data": data,
            "out_scalar_uint": int(a.sum().item()) if a.numel() > 0 else 0,
        })
    return out


# ---------------------------------------------------------------------------
# BoolTensor — comparisons returning a BoolTensor (gt/lt/ge/le/eq/ne)
# ---------------------------------------------------------------------------


def fixture_bool_compare() -> list[dict[str, Any]]:
    """`BoolTensor::gt(a, b)` etc., where a/b are `Tensor<T>`. PyTorch
    reference is `torch.gt(a, b)` / `torch.lt(a, b)` / ... returning a
    bool tensor. NaN-safe contract: PyTorch returns False for any
    comparison involving NaN (except `ne`, which returns True)."""
    out: list[dict[str, Any]] = []
    pairs: list[tuple[list[float], list[float], list[int], str]] = [
        ([1.0, 2.0, 3.0, 4.0], [0.0, 3.0, 3.0, 5.0], [4], "mixed4"),
        ([1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [3], "ge_le_mix"),
        ([1.0, 2.0, 3.0], [1.0, 5.0, 3.0], [3], "eq_ne_mix"),
        # NaN-safe path
        ([float("nan"), 1.0, 2.0], [1.0, 1.0, float("nan")], [3], "nan_compare"),
        # 2D
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0], [2, 3], "mat2x3"),
    ]
    op_to_torch = {
        "bool_gt": lambda a, b: torch.gt(a, b),
        "bool_lt": lambda a, b: torch.lt(a, b),
        "bool_ge": lambda a, b: torch.ge(a, b),
        "bool_le": lambda a, b: torch.le(a, b),
        "bool_eq_t": lambda a, b: torch.eq(a, b),
        "bool_ne": lambda a, b: torch.ne(a, b),
    }
    for dtype_name, torch_dtype in (("float32", torch.float32), ("float64", torch.float64)):
        for a_data, b_data, shape, tag in pairs:
            a = torch.tensor(a_data, dtype=torch_dtype).reshape(shape)
            b = torch.tensor(b_data, dtype=torch_dtype).reshape(shape)
            for op, fn in op_to_torch.items():
                out.append({
                    "op": op,
                    "tag": tag,
                    "dtype": dtype_name,
                    "device": "cpu",
                    "shape": shape,
                    "a_data": to_list_f64(a),
                    "b_data": to_list_f64(b),
                    "out_data": to_list_bool(fn(a, b)),
                })
    return out


# ---------------------------------------------------------------------------
# BoolTensor — reshape, to_float
# ---------------------------------------------------------------------------


def fixture_bool_reshape() -> list[dict[str, Any]]:
    """Reshape preserves data; PyTorch reference is `t.reshape(new_shape)`
    on a bool tensor."""
    out: list[dict[str, Any]] = []
    cases: list[tuple[list[bool], list[int], list[int], str]] = [
        ([True, False, True, False, True, False], [6], [2, 3], "1d_to_2d"),
        ([True, False, True, False, True, False], [2, 3], [3, 2], "2d_to_2d"),
        ([True, False, True, False], [4], [2, 2], "1d_to_2x2"),
        ([True], [1], [], "1d_to_scalar"),
    ]
    for data, in_shape, new_shape, tag in cases:
        a = torch.tensor(data, dtype=torch.bool).reshape(in_shape)
        r = a.reshape(new_shape) if new_shape else a.reshape(())
        out.append({
            "op": "bool_reshape",
            "tag": tag,
            "device": "cpu",
            "in_shape": in_shape,
            "new_shape": new_shape,
            "a_data": data,
            "out_data": to_list_bool(r),
        })
    return out


def fixture_bool_to_float() -> list[dict[str, Any]]:
    """`BoolTensor::to_float<T>()` → Tensor<T> with true=1.0, false=0.0."""
    out: list[dict[str, Any]] = []
    cases: list[tuple[list[bool], list[int], str]] = [
        ([True, False, True, False], [4], "vec4"),
        ([True, True, False], [3], "vec3"),
        ([False, False, False, False], [4], "allfalse"),
        ([True, True, True], [3], "alltrue"),
    ]
    for dtype_name in ("float32", "float64"):
        for data, shape, tag in cases:
            torch_dtype = torch.float32 if dtype_name == "float32" else torch.float64
            a = torch.tensor(data, dtype=torch.bool).reshape(shape)
            f = a.to(dtype=torch_dtype)
            out.append({
                "op": "bool_to_float",
                "tag": tag,
                "dtype": dtype_name,
                "device": "cpu",
                "shape": shape,
                "a_data": data,
                "out_data": to_list_f64(f),
            })
    return out


# ---------------------------------------------------------------------------
# IntTensor — constructors, scalar, arange, zeros
# ---------------------------------------------------------------------------


def fixture_int_constructors() -> list[dict[str, Any]]:
    """`IntTensor::from_vec / from_slice / zeros / scalar` and
    `IntTensor::arange(n)`. PyTorch references are
    `torch.tensor(..., dtype=torch.int{32,64})` and `torch.arange(n)`."""
    out: list[dict[str, Any]] = []
    shapes: list[tuple[list[int], str]] = [
        ([4], "vec1d"),
        ([2, 3], "mat2d"),
        ([2, 2, 3], "ten3d"),
        ([0], "empty1d"),
    ]
    patterns: list[tuple[list[int], list[int], str]] = [
        ([1, 2, 3, 4], [4], "small_vec"),
        ([0, -1, 100, -100], [4], "signed_vec"),
        ([1, 2, 3, 4, 5, 6], [2, 3], "mat2x3"),
    ]
    for dtype_name in ("i32", "i64"):
        torch_dtype = torch.int32 if dtype_name == "i32" else torch.int64
        # zeros
        for shape, tag in shapes:
            z = torch.zeros(shape, dtype=torch_dtype)
            out.append({
                "op": "int_zeros",
                "tag": tag,
                "dtype": dtype_name,
                "device": "cpu",
                "shape": shape,
                "out_data": to_list_int(z),
                "ndim": z.ndim,
                "numel": z.numel(),
            })
        # from_vec / from_slice
        for data, shape, tag in patterns:
            a = torch.tensor(data, dtype=torch_dtype).reshape(shape)
            out.append({
                "op": "int_from_vec",
                "tag": tag,
                "dtype": dtype_name,
                "device": "cpu",
                "shape": shape,
                "in_data": data,
                "out_data": to_list_int(a),
                "ndim": len(shape),
                "numel": max(1, math.prod(shape)),
            })
            out.append({
                "op": "int_from_slice",
                "tag": tag,
                "dtype": dtype_name,
                "device": "cpu",
                "shape": shape,
                "in_data": data,
                "out_data": to_list_int(a),
                "ndim": len(shape),
                "numel": max(1, math.prod(shape)),
            })
        # arange(n)
        for n in (0, 1, 5, 10):
            ar = torch.arange(n, dtype=torch_dtype)
            out.append({
                "op": "int_arange",
                "tag": f"n{n}",
                "dtype": dtype_name,
                "device": "cpu",
                "n": n,
                "out_data": to_list_int(ar),
                "ndim": 1,
                "numel": n,
            })
        # scalar(v): 0-d tensor with single value
        for v in (0, 1, -1, 42, -100):
            sc = torch.tensor(v, dtype=torch_dtype)
            out.append({
                "op": "int_scalar",
                "tag": f"v{v}",
                "dtype": dtype_name,
                "device": "cpu",
                "scalar": v,
                "shape": [],
                "out_data": [v],
                "ndim": 0,
                "numel": 1,
            })
    return out


# ---------------------------------------------------------------------------
# IntTensor — reshape, dtype_name, cast (i32 ↔ i64), shape introspection
# ---------------------------------------------------------------------------


def fixture_int_reshape() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases: list[tuple[list[int], list[int], list[int], str]] = [
        ([1, 2, 3, 4, 5, 6], [6], [2, 3], "1d_to_2d"),
        ([1, 2, 3, 4, 5, 6], [2, 3], [3, 2], "2d_to_2d"),
        ([1, 2, 3, 4], [4], [2, 2], "1d_to_2x2"),
        ([42], [1], [], "1d_to_scalar"),
    ]
    for dtype_name in ("i32", "i64"):
        torch_dtype = torch.int32 if dtype_name == "i32" else torch.int64
        for data, in_shape, new_shape, tag in cases:
            a = torch.tensor(data, dtype=torch_dtype).reshape(in_shape)
            r = a.reshape(new_shape) if new_shape else a.reshape(())
            out.append({
                "op": "int_reshape",
                "tag": tag,
                "dtype": dtype_name,
                "device": "cpu",
                "in_shape": in_shape,
                "new_shape": new_shape,
                "a_data": data,
                "out_data": to_list_int(r),
            })
    return out


def fixture_int_cast() -> list[dict[str, Any]]:
    """`IntTensor::cast<J>()` — i32 ↔ i64. We exercise both the in-range
    happy path and the out-of-range guard (i64::MAX cast to i32 must
    Err)."""
    out: list[dict[str, Any]] = []
    # In-range: works in both directions.
    in_range_data: list[tuple[list[int], list[int], str]] = [
        ([1, -1, 100], [3], "small"),
        ([0, -1, 1, 1000, -1000], [5], "mixed_signs"),
        ([(-2**30), (2**30 - 1)], [2], "near_i32_extremes"),
    ]
    for src, dst in (("i32", "i64"), ("i64", "i32")):
        torch_src = torch.int32 if src == "i32" else torch.int64
        torch_dst = torch.int32 if dst == "i32" else torch.int64
        for data, shape, tag in in_range_data:
            a = torch.tensor(data, dtype=torch_src).reshape(shape)
            c = a.to(dtype=torch_dst)
            out.append({
                "op": "int_cast",
                "tag": f"{src}_to_{dst}_{tag}",
                "src_dtype": src,
                "dst_dtype": dst,
                "device": "cpu",
                "shape": shape,
                "a_data": data,
                "out_data": to_list_int(c),
                "expect_err": False,
            })
    # Out-of-range: i64::MAX cast to i32 must error in ferrotorch.
    out.append({
        "op": "int_cast",
        "tag": "i64_to_i32_max_oob",
        "src_dtype": "i64",
        "dst_dtype": "i32",
        "device": "cpu",
        "shape": [1],
        "a_data": [(2**63) - 1],
        "out_data": [],  # unused on err path
        "expect_err": True,
    })
    out.append({
        "op": "int_cast",
        "tag": "i64_to_i32_min_oob",
        "src_dtype": "i64",
        "dst_dtype": "i32",
        "device": "cpu",
        "shape": [1],
        "a_data": [-(2**63)],
        "out_data": [],
        "expect_err": True,
    })
    return out


def fixture_int_dtype_name() -> list[dict[str, Any]]:
    """`IntTensor::dtype_name()` — returns `"i32"` or `"i64"` literally.
    PyTorch's analog is `tensor.dtype` (returns `torch.int32` /
    `torch.int64`). We map them by name here so the Rust test can
    string-compare."""
    out: list[dict[str, Any]] = []
    for dtype_name in ("i32", "i64"):
        out.append({
            "op": "int_dtype_name",
            "tag": dtype_name,
            "dtype": dtype_name,
            "device": "cpu",
            "expected_name": dtype_name,
        })
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures += fixture_bool_constructors()
    fixtures += fixture_bool_from_predicate()
    fixtures += fixture_bool_logical()
    fixtures += fixture_bool_reductions()
    fixtures += fixture_bool_compare()
    fixtures += fixture_bool_reshape()
    fixtures += fixture_bool_to_float()
    fixtures += fixture_int_constructors()
    fixtures += fixture_int_reshape()
    fixtures += fixture_int_cast()
    fixtures += fixture_int_dtype_name()

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
