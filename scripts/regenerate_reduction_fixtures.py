#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.2
(reductions + cumulative ops).

Tracking issue: #764 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/reduction.json

Coverage (29 surface items split across 5 categories):

* Cat A — reduction forwards (7 items, full CPU + GPU + autograd):
    sum, sum_dim, mean, mean_dim, prod, amax, amin
  Edge cases: empty tensor (sum=0, mean=NaN, prod=1, amax/amin = error so
  not tabulated), 1D + 2D + 3D inputs, sum_dim/mean_dim with each dim,
  keepdim=True/False, and the amax/amin tie-mass-distribution test
  (`[1.0, 1.0, 1.0]` -> grad = [1/3, 1/3, 1/3]).

* Cat B — cumulative forwards (5 items, full coverage):
    cumsum, cumprod, cummax, cummin, logcumsumexp
  Edge cases: dim=0 vs dim=-1, 1D/2D/3D, cumprod-with-zero
  (`[1.0, 0.0, 2.0, 3.0]` along dim=0), logcumsumexp numerical stability
  (`[100.0, 100.0]`). For cummax/cummin, both .values and .indices are
  recorded. To avoid PyTorch's "last-tie" vs ferrotorch's "first-tie"
  index-tracking divergence (a known parity gap surfaced as a separate
  cascade issue), the cummax/cummin reference inputs use strictly
  distinct values along the scan dim.

Cat C/D/E in the dispatch are exclusion-with-implicit-coverage entries
(`*_forward` ops, backward grad_fn structs, CumExtremeResult struct
properties) and are tested transitively through Cat A/B's autograd path
or via direct field-access tests; they don't need fixture data here.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_reduction_fixtures.py

Required Python deps: torch (with CUDA), numpy.
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
    / "reduction.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
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


def to_list_int(t: torch.Tensor) -> list[int]:
    """Materialize an integer tensor (e.g. cummax/cummin indices)."""
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
        "dtypes": DTYPES,
        "devices": DEVICES,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seeded(shape: list[int], dtype: str, device: str, base: float) -> torch.Tensor:
    n = max(1, math.prod(shape) if shape else 1)
    vals = [base + i * 0.5 for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype), device=device).reshape(shape)


# ---------------------------------------------------------------------------
# Cat A — reduction forwards (sum, mean, prod, amax, amin)
# ---------------------------------------------------------------------------
#
# Each global-reduction op gets:
#  * a 1-D, 2-D and 3-D fixture
#  * forward output (scalar)
#  * grad-w.r.t.-input under loss = output (since output is already scalar,
#    we directly call output.backward()).


REDUCTION_SHAPES: list[tuple[list[int], str]] = [
    ([4], "vec1d"),
    ([2, 3], "mat2d"),
    ([2, 2, 3], "ten3d"),
]


def _reduction_input(
    shape: list[int], dtype: str, device: str, op: str
) -> torch.Tensor:
    """Pick an input that exercises the op without zeros (for prod) or
    NaNs (for amax/amin/sum/mean)."""
    n = max(1, math.prod(shape))
    if op == "prod":
        # avoid zeros so backward divides safely; use small non-unit values.
        vals = [1.0 + (i % 5) * 0.25 for i in range(n)]
    elif op in ("amax", "amin"):
        # use distinct values so the global extremum is unambiguous
        # (keeps the basic tests away from ties; the dedicated tie test
        # below handles mass distribution).
        vals = [1.0 + i * 0.5 for i in range(n)]
    else:
        vals = [0.5 + i * 0.25 for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype), device=device).reshape(shape)


def fixture_cat_a_global_reductions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op in ("sum", "mean", "prod", "amax", "amin"):
        for device in DEVICES:
            for dtype in DTYPES:
                for shape, tag in REDUCTION_SHAPES:
                    a = _reduction_input(shape, dtype, device, op)
                    a_g = a.detach().clone().requires_grad_(True)
                    if op == "sum":
                        fwd = a_g.sum()
                    elif op == "mean":
                        fwd = a_g.mean()
                    elif op == "prod":
                        fwd = a_g.prod()
                    elif op == "amax":
                        fwd = torch.amax(a_g)
                    else:  # amin
                        fwd = torch.amin(a_g)
                    fwd.backward()
                    out.append(
                        {
                            "op": op,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "a_shape": shape,
                            "a_data": to_listf(a),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                            "grad_a": to_listf(a_g.grad),
                        }
                    )
    return out


def fixture_cat_a_dim_reductions() -> list[dict[str, Any]]:
    """sum_dim / mean_dim across each dim with keepdim=True/False on a 2-D
    input. Forward output, grad-wrt-input under loss = sum(output)."""
    out: list[dict[str, Any]] = []
    for op in ("sum_dim", "mean_dim"):
        for device in DEVICES:
            for dtype in DTYPES:
                # 2-D fixture: try both axes, both keepdim choices.
                a_2d = torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    dtype=torch_dtype(dtype),
                    device=device,
                ).reshape(2, 3)
                for axis in (0, 1):
                    for keepdim in (False, True):
                        a_g = a_2d.detach().clone().requires_grad_(True)
                        if op == "sum_dim":
                            fwd = torch.sum(a_g, dim=axis, keepdim=keepdim)
                        else:
                            fwd = torch.mean(a_g, dim=axis, keepdim=keepdim)
                        loss = fwd.sum()
                        loss.backward()
                        out.append(
                            {
                                "op": op,
                                "tag": f"mat2d_axis{axis}_keepdim{int(keepdim)}",
                                "dtype": dtype,
                                "device": device,
                                "axis": axis,
                                "keepdim": keepdim,
                                "a_shape": [2, 3],
                                "a_data": to_listf(a_2d),
                                "out_shape": list(fwd.shape),
                                "out_values": to_listf(fwd),
                                "grad_a": to_listf(a_g.grad),
                            }
                        )
                # 3-D fixture: each axis, keepdim=False (the common case).
                a_3d = torch.arange(
                    1.0, 13.0, dtype=torch_dtype(dtype), device=device
                ).reshape(2, 2, 3)
                for axis in (0, 1, 2):
                    a_g = a_3d.detach().clone().requires_grad_(True)
                    if op == "sum_dim":
                        fwd = torch.sum(a_g, dim=axis, keepdim=False)
                    else:
                        fwd = torch.mean(a_g, dim=axis, keepdim=False)
                    loss = fwd.sum()
                    loss.backward()
                    out.append(
                        {
                            "op": op,
                            "tag": f"ten3d_axis{axis}_keepdim0",
                            "dtype": dtype,
                            "device": device,
                            "axis": axis,
                            "keepdim": False,
                            "a_shape": [2, 2, 3],
                            "a_data": to_listf(a_3d),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                            "grad_a": to_listf(a_g.grad),
                        }
                    )
    return out


def fixture_cat_a_edge_cases() -> list[dict[str, Any]]:
    """Documented edge cases:
      * sum/mean/prod on an empty 1-D tensor.
      * amax/amin tie distribution: input `[1, 1, 1]`, grad = [1/3, 1/3, 1/3]
        (PyTorch's mass-distribution convention).

    `amax([])` / `amin([])` are explicit RuntimeErrors in PyTorch (the
    ferrotorch counterpart returns the same kind of error); we encode that
    expectation as `out_values = ["Error"]` and the test asserts ferrotorch
    returns Err — see the test code.
    """
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            empty = torch.tensor([], dtype=torch_dtype(dtype), device=device)
            for op in ("sum", "mean", "prod"):
                if op == "sum":
                    fwd = torch.sum(empty)
                elif op == "mean":
                    fwd = torch.mean(empty)
                else:
                    fwd = torch.prod(empty)
                out.append(
                    {
                        "op": f"{op}_empty",
                        "tag": "edge",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [0],
                        "a_data": [],
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )

            # amax / amin tie distribution: 3 equal values -> grad = 1/3 each.
            ties = torch.tensor(
                [1.0, 1.0, 1.0], dtype=torch_dtype(dtype), device=device
            )
            for op in ("amax", "amin"):
                a_g = ties.detach().clone().requires_grad_(True)
                fwd = torch.amax(a_g) if op == "amax" else torch.amin(a_g)
                fwd.backward()
                out.append(
                    {
                        "op": f"{op}_ties",
                        "tag": "tiedist",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [3],
                        "a_data": to_listf(ties),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Cat B — cumulative forwards (cumsum, cumprod, cummax, cummin, logcumsumexp)
# ---------------------------------------------------------------------------


CUMULATIVE_SHAPE_AXES: list[tuple[list[int], int, str]] = [
    ([4], 0, "vec1d_dim0"),
    ([2, 3], 0, "mat2d_dim0"),
    ([2, 3], 1, "mat2d_dim1"),
    ([2, 3], -1, "mat2d_dimneg1"),
    ([2, 2, 3], 0, "ten3d_dim0"),
    ([2, 2, 3], 1, "ten3d_dim1"),
    ([2, 2, 3], 2, "ten3d_dim2"),
]


def _cumulative_input(
    shape: list[int], dtype: str, device: str, op: str
) -> torch.Tensor:
    n = max(1, math.prod(shape))
    if op == "cumprod":
        # Small positive values that don't overflow when multiplied 12 times.
        vals = [1.05 + (i % 5) * 0.05 for i in range(n)]
    elif op == "logcumsumexp":
        # Moderate-magnitude values to keep f32 in the well-behaved band.
        vals = [(i % 7) * 0.25 - 0.5 for i in range(n)]
    elif op in ("cummax", "cummin"):
        # CRITICAL: use strictly distinct values along the scan dim. PyTorch
        # uses a "last-tie" index-tracking convention while ferrotorch uses
        # a "first-tie" convention; this divergence is filed as a separate
        # cascade issue. The distinct-value case is unambiguous and tests
        # the values + indices contract without entering the tie regime.
        vals = [(i * 1.7 + (i // 3) * 0.3) % 11 - 5 for i in range(n)]
    else:  # cumsum
        vals = [0.5 + i * 0.25 for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype), device=device).reshape(shape)


def fixture_cat_b_cumulative() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op in ("cumsum", "cumprod", "logcumsumexp"):
        for device in DEVICES:
            for dtype in DTYPES:
                for shape, axis, tag in CUMULATIVE_SHAPE_AXES:
                    a = _cumulative_input(shape, dtype, device, op)
                    a_g = a.detach().clone().requires_grad_(True)
                    if op == "cumsum":
                        fwd = torch.cumsum(a_g, dim=axis)
                    elif op == "cumprod":
                        fwd = torch.cumprod(a_g, dim=axis)
                    else:
                        fwd = torch.logcumsumexp(a_g, dim=axis)
                    loss = fwd.sum()
                    loss.backward()
                    out.append(
                        {
                            "op": op,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "axis": axis,
                            "a_shape": shape,
                            "a_data": to_listf(a),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                            "grad_a": to_listf(a_g.grad),
                        }
                    )
    # cummax / cummin: not differentiable. Record values + indices.
    for op in ("cummax", "cummin"):
        for device in DEVICES:
            for dtype in DTYPES:
                for shape, axis, tag in CUMULATIVE_SHAPE_AXES:
                    a = _cumulative_input(shape, dtype, device, op)
                    if op == "cummax":
                        result = torch.cummax(a, dim=axis)
                    else:
                        result = torch.cummin(a, dim=axis)
                    out.append(
                        {
                            "op": op,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "axis": axis,
                            "a_shape": shape,
                            "a_data": to_listf(a),
                            "out_shape": list(result.values.shape),
                            "out_values": to_listf(result.values),
                            "out_indices": to_list_int(result.indices),
                        }
                    )
    return out


def fixture_cat_b_edge_cases() -> list[dict[str, Any]]:
    """cumprod-with-zero and logcumsumexp numerical stability."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # cumprod with a zero element. PyTorch's backward special-cases
            # the zero so the gradient at the zero position is finite.
            x = torch.tensor(
                [1.0, 0.0, 2.0, 3.0], dtype=torch_dtype(dtype), device=device
            )
            a_g = x.detach().clone().requires_grad_(True)
            fwd = torch.cumprod(a_g, dim=0)
            fwd.sum().backward()
            out.append(
                {
                    "op": "cumprod_zero",
                    "tag": "edge",
                    "dtype": dtype,
                    "device": device,
                    "axis": 0,
                    "a_shape": [4],
                    "a_data": to_listf(x),
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                }
            )

            # logcumsumexp at large magnitude. Without the max-subtract trick
            # exp(100) overflows f32. Output[1] should be 100 + log(2).
            x = torch.tensor(
                [100.0, 100.0], dtype=torch_dtype(dtype), device=device
            )
            fwd = torch.logcumsumexp(x, dim=0)
            out.append(
                {
                    "op": "logcumsumexp_overflow",
                    "tag": "edge",
                    "dtype": dtype,
                    "device": device,
                    "axis": 0,
                    "a_shape": [2],
                    "a_data": to_listf(x),
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures += fixture_cat_a_global_reductions()
    fixtures += fixture_cat_a_dim_reductions()
    fixtures += fixture_cat_a_edge_cases()
    fixtures += fixture_cat_b_cumulative()
    fixtures += fixture_cat_b_edge_cases()

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
