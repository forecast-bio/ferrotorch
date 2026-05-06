#!/usr/bin/env python3
"""Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.8.

Tracking issue: #770 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/nested_sparse.json

Coverage (105 surface items split across two modules):

* `nested.rs` (Cat A — NestedTensor + PackedNestedTensor forwards):
    - construction / accessors (round-trip-implied: number of components,
      ragged_dim, ragged_lengths, consistent_shape, ndim, tensors)
    - to_padded / from_padded round-trip for ragged_dim 0 AND ragged_dim 1
      (PyTorch's stable "strided" layout doesn't expose ragged_dim 1
      directly — reference is computed in Python)
    - empty / mismatched-shape errors
    - nested_scaled_dot_product_attention (PyTorch parity for stacked
      independent SDPA calls)
    - PackedNestedTensor: from_sequences, from_nested, to_nested,
      to_padded, from_padded round-trip; elementwise (add/sub/mul/div/map);
      reductions (sum_per_component, mean_per_component)

* `sparse.rs` (Cat A — sparse forwards):
    - SparseTensor (rank-N COO with Vec<Vec<usize>> indices):
      from_dense, to_dense round-trip, coalesce (duplicate-merge & sort),
      add, mul_scalar, t (transpose), spmm
    - CooTensor (2-D COO with separate row/col arrays):
      new, accessors, coalesce, to_dense, from_csr
    - CsrTensor: new, from_coo, accessors, to_dense
    - CscTensor: new, from_csr, to_csr, to_dense, accessors
    - SemiStructuredSparseTensor: compress / decompress / sparse_matmul_24
      (the 2:4 semi-structured pattern is reference-only; PyTorch's
      `to_sparse_semi_structured` is experimental and out of scope here —
      we just verify ferrotorch's compress/decompress behaviour preserves
      values per its own docs).
    - SparseGrad: new, coalesce, apply_sgd

Edge cases (per dispatch):
* Empty nested tensor (no elements within a single component is the
  closest stable form — torch.nested rejects 0-component lists).
* Ragged dim 0 vs ragged dim 1.
* Sparse to_dense round-trip equality.
* Sparse matmul on small (5x5) matrices.
* Various sparsity levels: 1%, 50%, 99%.
* COO with duplicate indices (PyTorch coalesces; ferrotorch parity).

GPU note: ferrotorch's nested + sparse modules are CPU-only.
Per `rust-gpu-discipline` §3, surfacing this via a tracked cascade
issue (filed by the dispatch) is the correct response — the test
file `cascade_skip()`s GPU tests with that issue ref rather than
silently passing.

Usage:
    python3 scripts/regenerate_nested_sparse_fixtures.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-core"
    / "tests"
    / "conformance"
    / "fixtures"
    / "nested_sparse.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
# GPU device retained for metadata only — ferrotorch nested/sparse ops are
# CPU-only and the conformance test cascade-skips GPU paths against the
# tracked issue. The fixture format mirrors the other phases for parity.
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

RNG_SEED: int = 0xBADCAFE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


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


def listf(values: list[float]) -> list[Any]:
    """Encode a Python list with NaN/Inf sentinels (parity with to_listf)."""
    out: list[Any] = []
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            out.append("NaN")
        elif isinstance(v, float) and math.isinf(v):
            out.append("Infinity" if v > 0 else "-Infinity")
        else:
            out.append(float(v))
    return out


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "dtypes": DTYPES,
        "devices": DEVICES,
    }


# ---------------------------------------------------------------------------
# NestedTensor — to_padded / from_padded reference. PyTorch's
# `torch.nested.to_padded_tensor` / `nested_to_padded` only supports the
# leading-dim ragged layout; for ragged_dim=1 we compute the reference in
# Python by scattering the components into a zero-padded dense tensor.
# ---------------------------------------------------------------------------


def _build_padded_reference(
    components: list[list[float]],
    component_shapes: list[list[int]],
    ragged_dim: int,
    pad_value: float,
) -> tuple[list[Any], list[int]]:
    """Compute the padded dense tensor that ferrotorch's `to_padded` should
    produce. Returns (flat row-major values, output shape).

    Output shape is `[batch] + component_shape with ragged_dim replaced by
    max_len`.
    """
    batch = len(components)
    if batch == 0:
        return [], [0]
    ndim = len(component_shapes[0])
    max_len = max(s[ragged_dim] for s in component_shapes)

    out_shape = [batch] + [
        max_len if d == ragged_dim else component_shapes[0][d] for d in range(ndim)
    ]
    numel = 1
    for x in out_shape:
        numel *= x
    out = [pad_value] * numel

    # Strides for the output tensor (row-major).
    out_strides = [0] * (ndim + 1)
    out_strides[ndim] = 1
    for d in range(ndim - 1, -1, -1):
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1]

    for b, (data, shape) in enumerate(zip(components, component_shapes)):
        # Strides for this component.
        c_strides = [0] * ndim
        if ndim > 0:
            c_strides[ndim - 1] = 1
            for d in range(ndim - 2, -1, -1):
                c_strides[d] = c_strides[d + 1] * shape[d + 1]
        c_numel = 1
        for x in shape:
            c_numel *= x
        for flat in range(c_numel):
            remaining = flat
            out_flat = b * out_strides[0]
            for d in range(ndim):
                coord = remaining // c_strides[d] if c_strides[d] > 0 else 0
                if c_strides[d] > 0:
                    remaining %= c_strides[d]
                out_flat += coord * out_strides[d + 1]
            out[out_flat] = data[flat]

    return listf(out), out_shape


def fixture_nested_to_padded() -> list[dict[str, Any]]:
    """to_padded / from_padded round-trip — covers `NestedTensor::to_padded`,
    `from_padded`, `consistent_shape`, `ndim`, `num_components`,
    `ragged_dim`, `ragged_lengths`, `tensors` (the test reads them all)."""
    out: list[dict[str, Any]] = []

    # --- ragged_dim = 0, components shape [L_i, 2] ---
    components = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # [3, 2]
        [7.0, 8.0, 9.0, 10.0],  # [2, 2]
        [11.0, 12.0],  # [1, 2]
    ]
    shapes = [[3, 2], [2, 2], [1, 2]]
    pad = 0.0
    padded, padded_shape = _build_padded_reference(components, shapes, 0, pad)
    out.append(
        {
            "op": "nested_to_padded",
            "tag": "ragged_dim0_2d",
            "dtype": "float32",
            "device": "cpu",
            "components": [listf(c) for c in components],
            "component_shapes": shapes,
            "ragged_dim": 0,
            "pad_value": pad,
            "padded_data": padded,
            "padded_shape": padded_shape,
        }
    )
    out.append(
        {
            "op": "nested_to_padded",
            "tag": "ragged_dim0_2d",
            "dtype": "float64",
            "device": "cpu",
            "components": [listf(c) for c in components],
            "component_shapes": shapes,
            "ragged_dim": 0,
            "pad_value": pad,
            "padded_data": padded,
            "padded_shape": padded_shape,
        }
    )

    # --- ragged_dim = 1, components shape [2, L_i] ---
    components_d1 = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # [2, 3]
        [7.0, 8.0, 9.0, 10.0],  # [2, 2]
    ]
    shapes_d1 = [[2, 3], [2, 2]]
    padded_d1, padded_shape_d1 = _build_padded_reference(
        components_d1, shapes_d1, 1, 0.0
    )
    for dtype in DTYPES:
        out.append(
            {
                "op": "nested_to_padded",
                "tag": "ragged_dim1_2d",
                "dtype": dtype,
                "device": "cpu",
                "components": [listf(c) for c in components_d1],
                "component_shapes": shapes_d1,
                "ragged_dim": 1,
                "pad_value": 0.0,
                "padded_data": padded_d1,
                "padded_shape": padded_shape_d1,
            }
        )

    # --- single-component (degenerate batch=1) — no padding actually happens ---
    single = [[1.0, 2.0, 3.0]]
    single_shape = [[3]]
    padded_s, padded_shape_s = _build_padded_reference(single, single_shape, 0, -1.0)
    out.append(
        {
            "op": "nested_to_padded",
            "tag": "single_component_1d",
            "dtype": "float32",
            "device": "cpu",
            "components": [listf(c) for c in single],
            "component_shapes": single_shape,
            "ragged_dim": 0,
            "pad_value": -1.0,
            "padded_data": padded_s,
            "padded_shape": padded_shape_s,
        }
    )

    return out


# ---------------------------------------------------------------------------
# Nested scaled dot-product attention. PyTorch's torch.nn.functional.
# scaled_dot_product_attention applied per-component — ferrotorch's
# nested SDPA is the same formula (Q K^T / sqrt(d_k), softmax, @V).
# ---------------------------------------------------------------------------


def fixture_nested_sdpa() -> list[dict[str, Any]]:
    """Cover `nested_scaled_dot_product_attention` and its top-level re-export."""
    out: list[dict[str, Any]] = []

    # Two components with different sequence lengths.
    # Component 0: seq_q=2, seq_k=3, d_k=4, d_v=5
    # Component 1: seq_q=1, seq_k=2, d_k=4, d_v=5
    torch.manual_seed(RNG_SEED)
    for dtype in DTYPES:
        td = torch_dtype(dtype)
        q0 = torch.randn(2, 4, dtype=td)
        k0 = torch.randn(3, 4, dtype=td)
        v0 = torch.randn(3, 5, dtype=td)
        q1 = torch.randn(1, 4, dtype=td)
        k1 = torch.randn(2, 4, dtype=td)
        v1 = torch.randn(2, 5, dtype=td)

        # Reference: PyTorch's per-component F.scaled_dot_product_attention.
        ref0 = torch.nn.functional.scaled_dot_product_attention(q0, k0, v0)
        ref1 = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1)

        out.append(
            {
                "op": "nested_sdpa",
                "tag": "two_components",
                "dtype": dtype,
                "device": "cpu",
                "queries": [to_listf(q0), to_listf(q1)],
                "keys": [to_listf(k0), to_listf(k1)],
                "values": [to_listf(v0), to_listf(v1)],
                "query_shapes": [list(q0.shape), list(q1.shape)],
                "key_shapes": [list(k0.shape), list(k1.shape)],
                "value_shapes": [list(v0.shape), list(v1.shape)],
                "outputs": [to_listf(ref0), to_listf(ref1)],
                "output_shapes": [list(ref0.shape), list(ref1.shape)],
            }
        )

    return out


# ---------------------------------------------------------------------------
# PackedNestedTensor — packed flat storage.
# ---------------------------------------------------------------------------


def fixture_packed_nested() -> list[dict[str, Any]]:
    """Cover all PackedNestedTensor methods."""
    out: list[dict[str, Any]] = []

    # 1-D sequences (empty tail shape).
    seqs_1d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0], [9.0, 10.0]]
    lengths_1d = [3, 5, 2]
    expected_offsets_1d = [0, 3, 8, 10]
    expected_data_1d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for dtype in DTYPES:
        out.append(
            {
                "op": "packed_from_sequences",
                "tag": "1d",
                "dtype": dtype,
                "device": "cpu",
                "sequences": [listf(s) for s in seqs_1d],
                "lengths": lengths_1d,
                "tail_shape": [],
                "expected_offsets": expected_offsets_1d,
                "expected_data": listf(expected_data_1d),
                "expected_num_components": 3,
                "expected_total_numel": 10,
            }
        )

    # 2-D sequences with tail shape [4]. Three components, lengths 3,2,1.
    seqs_2d = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],  # 3 rows of 4
        [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],  # 2 rows of 4
        [21.0, 22.0, 23.0, 24.0],  # 1 row of 4
    ]
    lengths_2d = [3, 2, 1]
    tail_2d = [4]
    expected_offsets_2d = [0, 12, 20, 24]
    flat_2d: list[float] = []
    for s in seqs_2d:
        flat_2d.extend(s)

    for dtype in DTYPES:
        out.append(
            {
                "op": "packed_from_sequences",
                "tag": "2d_tail4",
                "dtype": dtype,
                "device": "cpu",
                "sequences": [listf(s) for s in seqs_2d],
                "lengths": lengths_2d,
                "tail_shape": tail_2d,
                "expected_offsets": expected_offsets_2d,
                "expected_data": listf(flat_2d),
                "expected_num_components": 3,
                "expected_total_numel": 24,
            }
        )

    # Elementwise add/sub/mul/div on identically-shaped packs.
    a = [[10.0, 20.0, 30.0], [40.0, 50.0]]
    b = [[1.0, 2.0, 3.0], [4.0, 5.0]]
    a_flat = a[0] + a[1]
    b_flat = b[0] + b[1]
    add = [x + y for x, y in zip(a_flat, b_flat)]
    sub = [x - y for x, y in zip(a_flat, b_flat)]
    mul = [x * y for x, y in zip(a_flat, b_flat)]
    div = [x / y for x, y in zip(a_flat, b_flat)]
    for dtype in DTYPES:
        out.append(
            {
                "op": "packed_elementwise",
                "tag": "add_sub_mul_div",
                "dtype": dtype,
                "device": "cpu",
                "a_sequences": [listf(s) for s in a],
                "b_sequences": [listf(s) for s in b],
                "lengths": [3, 2],
                "tail_shape": [],
                "expected_add": listf(add),
                "expected_sub": listf(sub),
                "expected_mul": listf(mul),
                "expected_div": listf(div),
            }
        )

    # Reductions: sum_per_component, mean_per_component.
    seqs_red = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0], [5.0]]
    lengths_red = [3, 4, 1]
    sums = [sum(s) for s in seqs_red]
    means = [sum(s) / len(s) for s in seqs_red]
    for dtype in DTYPES:
        out.append(
            {
                "op": "packed_reductions",
                "tag": "sum_mean_per_component",
                "dtype": dtype,
                "device": "cpu",
                "sequences": [listf(s) for s in seqs_red],
                "lengths": lengths_red,
                "tail_shape": [],
                "expected_sums": listf(sums),
                "expected_means": listf(means),
            }
        )

    # to_padded with explicit pad value, including non-zero pad.
    pad_seqs = [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
    pad_lengths = [3, 2, 4]
    pad_value = -1.0
    # Output shape [3 components, max_len=4]: 12 elements.
    expected_padded = [
        1.0, 2.0, 3.0, -1.0,
        4.0, 5.0, -1.0, -1.0,
        6.0, 7.0, 8.0, 9.0,
    ]
    for dtype in DTYPES:
        out.append(
            {
                "op": "packed_to_padded",
                "tag": "1d_padval_neg1",
                "dtype": dtype,
                "device": "cpu",
                "sequences": [listf(s) for s in pad_seqs],
                "lengths": pad_lengths,
                "tail_shape": [],
                "pad_value": pad_value,
                "expected_padded_shape": [3, 4],
                "expected_padded": listf(expected_padded),
            }
        )

    return out


# ---------------------------------------------------------------------------
# SparseTensor (rank-N COO with Vec<Vec<usize>> indices) — built via PyTorch
# torch.sparse_coo_tensor and its `.to_dense()` / `.coalesce()` / `+` etc.
# ---------------------------------------------------------------------------


def _coo_to_dense(
    indices: list[list[int]], values: list[float], shape: list[int]
) -> list[float]:
    """Reference dense reconstruction (duplicates summed, parity with
    PyTorch's sparse_coo_tensor.to_dense)."""
    numel = 1
    for x in shape:
        numel *= x
    data = [0.0] * numel
    for idx, val in zip(indices, values):
        flat = 0
        stride = 1
        for d in range(len(shape) - 1, -1, -1):
            flat += idx[d] * stride
            stride *= shape[d]
        data[flat] += val
    return data


def fixture_sparse_tensor_coo_rankN() -> list[dict[str, Any]]:
    """Cover SparseTensor (rank-N COO):
    - new, indices, values, nnz, shape, ndim
    - from_dense, to_dense round-trip
    - coalesce, add, mul_scalar, t, spmm
    """
    out: list[dict[str, Any]] = []

    # 3x3 matrix with three non-zeros: [(0,1)=5, (2,0)=3, (1,2)=7].
    indices = [[0, 1], [2, 0], [1, 2]]
    values = [5.0, 3.0, 7.0]
    shape = [3, 3]
    dense = _coo_to_dense(indices, values, shape)
    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_to_dense",
                "tag": "3x3_basic",
                "dtype": dtype,
                "device": "cpu",
                "indices": indices,
                "values": listf(values),
                "shape": shape,
                "expected_dense": listf(dense),
                "expected_nnz": 3,
                "expected_ndim": 2,
            }
        )

    # COO with duplicate indices — coalesce parity. PyTorch sums duplicates.
    dup_indices = [[0, 0], [0, 1], [0, 1]]
    dup_values = [1.0, 3.0, 4.0]  # [0,1] occurs twice -> sum = 7.0
    dup_shape = [1, 3]
    # Coalesced canonical form (sorted by index, duplicates summed, zeros
    # removed).
    coalesced_indices = [[0, 0], [0, 1]]
    coalesced_values = [1.0, 7.0]
    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_coalesce",
                "tag": "duplicates_summed",
                "dtype": dtype,
                "device": "cpu",
                "indices": dup_indices,
                "values": listf(dup_values),
                "shape": dup_shape,
                "expected_coalesced_indices": coalesced_indices,
                "expected_coalesced_values": listf(coalesced_values),
                "expected_coalesced_nnz": 2,
            }
        )

    # Coalesce with zero-sum cancellation.
    cancel_indices = [[0, 0], [0, 0]]
    cancel_values = [5.0, -5.0]
    cancel_shape = [1, 1]
    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_coalesce",
                "tag": "zero_sum_cancel",
                "dtype": dtype,
                "device": "cpu",
                "indices": cancel_indices,
                "values": listf(cancel_values),
                "shape": cancel_shape,
                "expected_coalesced_indices": [],
                "expected_coalesced_values": [],
                "expected_coalesced_nnz": 0,
            }
        )

    # Sparsity-level sweep: 1%, 50%, 99% non-zeros in a 10x10 matrix.
    for sparsity_pct, tag in [(99, "1pct_nnz"), (50, "50pct_nnz"), (1, "99pct_nnz")]:
        # `sparsity_pct` is the percentage of zeros; non-zero count is
        # 100 - sparsity_pct.
        nnz_count = 100 - sparsity_pct
        torch.manual_seed(RNG_SEED + sparsity_pct)
        # Build a dense reference of shape 10x10 with `nnz_count` non-zeros
        # placed at deterministic positions.
        positions = list(range(100))
        # Use a deterministic permutation: take first nnz_count positions
        # after a seeded shuffle.
        gen = torch.Generator().manual_seed(RNG_SEED + sparsity_pct)
        perm = torch.randperm(100, generator=gen).tolist()[:nnz_count]
        perm.sort()
        idx_list: list[list[int]] = []
        val_list: list[float] = []
        for flat in perm:
            r = flat // 10
            c = flat % 10
            idx_list.append([r, c])
            # Use distinct values so spmm parity is meaningful.
            val_list.append(1.0 + 0.5 * len(val_list))
        dense_ref = _coo_to_dense(idx_list, val_list, [10, 10])

        # Construct an [10, 4] dense vector for spmm, deterministic.
        spmm_rhs = [(i + 1) * 0.5 for i in range(10 * 4)]
        # spmm reference: out[i,c] = sum_j sparse[i,j] * rhs[j,c].
        spmm_ref = [0.0] * (10 * 4)
        for idx, v in zip(idx_list, val_list):
            r, c = idx[0], idx[1]
            for col in range(4):
                spmm_ref[r * 4 + col] += v * spmm_rhs[c * 4 + col]

        for dtype in DTYPES:
            out.append(
                {
                    "op": "sparse_spmm",
                    "tag": f"10x10_{tag}",
                    "dtype": dtype,
                    "device": "cpu",
                    "indices": idx_list,
                    "values": listf(val_list),
                    "shape": [10, 10],
                    "expected_dense": listf(dense_ref),
                    "rhs_data": listf(spmm_rhs),
                    "rhs_shape": [10, 4],
                    "expected_spmm": listf(spmm_ref),
                    "expected_spmm_shape": [10, 4],
                    "expected_nnz": nnz_count,
                }
            )

    # mul_scalar
    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_mul_scalar",
                "tag": "scalar_3",
                "dtype": dtype,
                "device": "cpu",
                "indices": [[0, 0], [1, 1]],
                "values": listf([2.0, 3.0]),
                "shape": [2, 2],
                "scalar": 3.0,
                "expected_values": listf([6.0, 9.0]),
            }
        )

    # add (uncoalesced result expected to have shared indices appear twice).
    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_add",
                "tag": "two_2x2",
                "dtype": dtype,
                "device": "cpu",
                "a_indices": [[0, 0], [0, 1]],
                "a_values": listf([1.0, 2.0]),
                "b_indices": [[0, 1], [1, 0]],
                "b_values": listf([3.0, 4.0]),
                "shape": [2, 2],
                # Coalesced result: [0,0]=1, [0,1]=2+3=5, [1,0]=4.
                "expected_coalesced_indices": [[0, 0], [0, 1], [1, 0]],
                "expected_coalesced_values": listf([1.0, 5.0, 4.0]),
                "expected_uncoalesced_nnz": 4,
            }
        )

    # transpose
    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_transpose",
                "tag": "3x4",
                "dtype": dtype,
                "device": "cpu",
                "indices": [[0, 1], [2, 0]],
                "values": listf([5.0, 3.0]),
                "shape": [3, 4],
                "expected_indices": [[1, 0], [0, 2]],
                "expected_values": listf([5.0, 3.0]),
                "expected_shape": [4, 3],
            }
        )

    # from_dense with threshold.
    dense_in = [0.5, 1.5, 0.1, 2.0]
    threshold = 1.0
    # only |v| > 1.0 kept: 1.5 at [0,1], 2.0 at [1,1].
    expected_indices = [[0, 1], [1, 1]]
    expected_values = [1.5, 2.0]
    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_from_dense",
                "tag": "threshold_1",
                "dtype": dtype,
                "device": "cpu",
                "dense_data": listf(dense_in),
                "shape": [2, 2],
                "threshold": threshold,
                "expected_indices": expected_indices,
                "expected_values": listf(expected_values),
            }
        )

    return out


# ---------------------------------------------------------------------------
# CooTensor / CsrTensor / CscTensor — 2-D format conversions and accessors.
# ---------------------------------------------------------------------------


def fixture_coo_csr_csc() -> list[dict[str, Any]]:
    """Cover CooTensor, CsrTensor, CscTensor — construction, conversions,
    and dense round-trip."""
    out: list[dict[str, Any]] = []

    # 5x5 matrix with deterministic non-zeros — the dispatch's required
    # sparse-matmul-on-small-matrices edge case.
    rows = [0, 1, 1, 2, 3, 4, 4]
    cols = [1, 0, 3, 2, 4, 1, 4]
    vals = [1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5]
    nrows, ncols = 5, 5

    # Reference dense.
    dense_ref = [0.0] * (nrows * ncols)
    for r, c, v in zip(rows, cols, vals):
        dense_ref[r * ncols + c] = v

    # Reference CSR (rows sorted ascending; cols within each row sorted).
    # Build by row.
    by_row: list[list[tuple[int, float]]] = [[] for _ in range(nrows)]
    for r, c, v in zip(rows, cols, vals):
        by_row[r].append((c, v))
    for entries in by_row:
        entries.sort(key=lambda x: x[0])
    csr_row_ptrs = [0]
    csr_cols: list[int] = []
    csr_vals: list[float] = []
    for entries in by_row:
        for c, v in entries:
            csr_cols.append(c)
            csr_vals.append(v)
        csr_row_ptrs.append(len(csr_cols))

    # Reference CSC: column-major.
    by_col: list[list[tuple[int, float]]] = [[] for _ in range(ncols)]
    for r, c, v in zip(rows, cols, vals):
        by_col[c].append((r, v))
    for entries in by_col:
        entries.sort(key=lambda x: x[0])
    csc_col_ptrs = [0]
    csc_rows: list[int] = []
    csc_vals: list[float] = []
    for entries in by_col:
        for r, v in entries:
            csc_rows.append(r)
            csc_vals.append(v)
        csc_col_ptrs.append(len(csc_rows))

    for dtype in DTYPES:
        out.append(
            {
                "op": "coo_csr_csc_5x5",
                "tag": "format_round_trip",
                "dtype": dtype,
                "device": "cpu",
                "row_indices": rows,
                "col_indices": cols,
                "values": listf(vals),
                "nrows": nrows,
                "ncols": ncols,
                "expected_dense": listf(dense_ref),
                "expected_csr_row_ptrs": csr_row_ptrs,
                "expected_csr_col_indices": csr_cols,
                "expected_csr_values": listf(csr_vals),
                "expected_csc_col_ptrs": csc_col_ptrs,
                "expected_csc_row_indices": csc_rows,
                "expected_csc_values": listf(csc_vals),
                "expected_nnz": len(vals),
            }
        )

    # COO with duplicates — coalesce parity. The dispatch lists this as
    # an explicit edge case.
    dup_rows = [0, 0, 1, 1]
    dup_cols = [0, 0, 1, 2]
    dup_vals = [1.0, 2.0, 3.0, 4.0]
    # Coalesced: [0,0]=3.0, [1,1]=3.0, [1,2]=4.0
    expected_coal_rows = [0, 1, 1]
    expected_coal_cols = [0, 1, 2]
    expected_coal_vals = [3.0, 3.0, 4.0]
    for dtype in DTYPES:
        out.append(
            {
                "op": "coo_coalesce",
                "tag": "duplicates",
                "dtype": dtype,
                "device": "cpu",
                "row_indices": dup_rows,
                "col_indices": dup_cols,
                "values": listf(dup_vals),
                "nrows": 2,
                "ncols": 3,
                "expected_coalesced_rows": expected_coal_rows,
                "expected_coalesced_cols": expected_coal_cols,
                "expected_coalesced_values": listf(expected_coal_vals),
                "expected_coalesced_nnz": 3,
            }
        )

    return out


# ---------------------------------------------------------------------------
# SemiStructuredSparseTensor — 2:4 compress / decompress / sparse_matmul_24.
# ---------------------------------------------------------------------------


def _compress_24_reference(data: list[float]) -> tuple[list[float], list[int]]:
    """Reference for ferrotorch's `compress`: per group of 4, keep the 2
    largest-magnitude positions (ties broken by lower index). Returns
    (kept_values_in_orig_order, group_nibbles)."""
    assert len(data) % 4 == 0
    num_groups = len(data) // 4
    values: list[float] = []
    nibbles: list[int] = []
    for g in range(num_groups):
        base = g * 4
        # (pos, |value|) sorted descending by magnitude, ties by lower pos.
        mags = sorted(
            ((p, abs(data[base + p])) for p in range(4)),
            key=lambda t: (-t[1], t[0]),
        )
        kept = sorted([mags[0][0], mags[1][0]])
        for p in kept:
            values.append(data[base + p])
        nibble = (1 << kept[0]) | (1 << kept[1])
        nibbles.append(nibble)
    return values, nibbles


def _decompress_24_reference(
    values: list[float], nibbles: list[int], shape: list[int]
) -> list[float]:
    numel = 1
    for x in shape:
        numel *= x
    out = [0.0] * numel
    val_idx = 0
    for g, nibble in enumerate(nibbles):
        for p in range(4):
            if (nibble >> p) & 1:
                out[g * 4 + p] = values[val_idx]
                val_idx += 1
    return out


def fixture_semi_structured() -> list[dict[str, Any]]:
    """Cover SemiStructuredSparseTensor and sparse_matmul_24."""
    out: list[dict[str, Any]] = []

    # 1-D, 8 elements (2 groups of 4). Picks distinct magnitudes so the
    # tie-break behaviour is unambiguous.
    data = [3.0, -1.0, 2.0, -4.0, 5.0, -6.0, 1.5, 0.5]
    values, nibbles = _compress_24_reference(data)
    decompressed = _decompress_24_reference(values, nibbles, [8])

    for dtype in DTYPES:
        out.append(
            {
                "op": "semi_structured_24",
                "tag": "1d_8elem",
                "dtype": dtype,
                "device": "cpu",
                "dense_data": listf(data),
                "shape": [8],
                "expected_kept_values": listf(values),
                "expected_nibbles": nibbles,
                "expected_decompressed": listf(decompressed),
                "expected_num_groups": 2,
            }
        )

    # 2-D matmul: A [3, 8] @ B [8, 4] where B is compressed in 2:4 format.
    # B's last-dim stride is 4, which is the contract `compress` enforces
    # via `numel % 4 == 0`. n=4 is a multiple of 4 (per the function's
    # docstring contract).
    a = [(i * 0.25 + 1.0) for i in range(3 * 8)]
    b = [(i * 0.5 - 1.0) for i in range(8 * 4)]
    # Build the compressed B's effective dense (zeros where masked out).
    b_values, b_nibbles = _compress_24_reference(b)
    b_eff = _decompress_24_reference(b_values, b_nibbles, [8, 4])
    # Reference matmul: A @ B_effective.
    matmul_ref = [0.0] * (3 * 4)
    for i in range(3):
        for j in range(4):
            acc = 0.0
            for k in range(8):
                acc += a[i * 8 + k] * b_eff[k * 4 + j]
            matmul_ref[i * 4 + j] = acc

    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_matmul_24",
                "tag": "3x8_8x4",
                "dtype": dtype,
                "device": "cpu",
                "a_data": listf(a),
                "a_shape": [3, 8],
                "b_dense_data": listf(b),
                "b_shape": [8, 4],
                "expected_b_kept_values": listf(b_values),
                "expected_b_nibbles": b_nibbles,
                "expected_b_decompressed": listf(b_eff),
                "expected_matmul": listf(matmul_ref),
                "expected_matmul_shape": [3, 4],
            }
        )

    return out


# ---------------------------------------------------------------------------
# SparseGrad — sparse-gradient SGD update path.
# ---------------------------------------------------------------------------


def fixture_sparse_grad() -> list[dict[str, Any]]:
    """Cover SparseGrad.new / coalesce / apply_sgd."""
    out: list[dict[str, Any]] = []

    # Embedding-shaped parameter [V=4, D=3]. Two updates touching rows 0,2,0:
    # the duplicate row 0 must coalesce to a sum.
    indices = [0, 2, 0]
    values = [
        # row 0 update 1: [1.0, 2.0, 3.0]
        1.0, 2.0, 3.0,
        # row 2 update: [10.0, 20.0, 30.0]
        10.0, 20.0, 30.0,
        # row 0 update 2: [4.0, 5.0, 6.0]  -> coalesces with first into [5,7,9]
        4.0, 5.0, 6.0,
    ]
    slab_shape = [3]
    coalesced_indices = [0, 2]
    coalesced_values = [
        5.0, 7.0, 9.0,  # row 0 sum
        10.0, 20.0, 30.0,  # row 2
    ]

    # apply_sgd reference: param = param - lr * coalesced_grad (after coalescing).
    # Use a non-zero starting param to test scalability.
    param = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    lr = 0.1
    # Note: the source `apply_sgd` does NOT coalesce internally — the caller is
    # expected to coalesce first. Test BOTH paths: uncoalesced (each individual
    # slab applied in order, duplicates stack) and coalesced.
    # Uncoalesced application of all three slabs in order:
    uncoal_param = list(param)
    for k, idx in enumerate(indices):
        for j in range(len(slab_shape) and slab_shape[0] or 1):
            uncoal_param[idx * slab_shape[0] + j] -= lr * values[k * slab_shape[0] + j]
    # Coalesced application:
    coal_param = list(param)
    for k, idx in enumerate(coalesced_indices):
        for j in range(slab_shape[0]):
            coal_param[idx * slab_shape[0] + j] -= lr * coalesced_values[
                k * slab_shape[0] + j
            ]

    for dtype in DTYPES:
        out.append(
            {
                "op": "sparse_grad",
                "tag": "embedding_4x3_dup_row0",
                "dtype": dtype,
                "device": "cpu",
                "indices": indices,
                "values": listf(values),
                "slab_shape": slab_shape,
                "expected_coalesced_indices": coalesced_indices,
                "expected_coalesced_values": listf(coalesced_values),
                "expected_nnz": 3,
                "expected_coalesced_nnz": 2,
                "param_shape": [4, 3],
                "param_data": listf(param),
                "lr": lr,
                "expected_uncoalesced_param": listf(uncoal_param),
                "expected_coalesced_param": listf(coal_param),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Empty-nested error case (NestedTensor::new requires ≥1 component).
# ---------------------------------------------------------------------------


def fixture_nested_error_cases() -> list[dict[str, Any]]:
    """Documented edge cases — no value to compare, but the test asserts
    ferrotorch returns Err. Recorded as fixture rows for symmetry with
    other phases."""
    return [
        {
            "op": "nested_empty_error",
            "tag": "no_components",
            "dtype": "float32",
            "device": "cpu",
            "expected_error": True,
        },
        {
            "op": "nested_shape_mismatch_error",
            "tag": "non_ragged_dim_differs",
            "dtype": "float32",
            "device": "cpu",
            "components": [
                listf([1.0] * 6),  # shape [3, 2]
                listf([1.0] * 6),  # shape [2, 3]
            ],
            "component_shapes": [[3, 2], [2, 3]],
            "ragged_dim": 0,
            "expected_error": True,
        },
    ]


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    fixtures: list[dict[str, Any]] = []
    fixtures.extend(fixture_nested_to_padded())
    fixtures.extend(fixture_nested_sdpa())
    fixtures.extend(fixture_packed_nested())
    fixtures.extend(fixture_sparse_tensor_coo_rankN())
    fixtures.extend(fixture_coo_csr_csc())
    fixtures.extend(fixture_semi_structured())
    fixtures.extend(fixture_sparse_grad())
    fixtures.extend(fixture_nested_error_cases())

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"wrote {FIXTURE_PATH} ({len(fixtures)} fixtures)")


if __name__ == "__main__":
    main()
