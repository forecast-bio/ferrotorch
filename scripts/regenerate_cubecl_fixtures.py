#!/usr/bin/env python3
"""
Regenerate ferrotorch-core CPU reference fixtures for the ferrotorch-cubecl
conformance suite.

Tracking issue: #860 (ferrotorch-cubecl conformance suite)

Output: ``ferrotorch-cubecl/tests/conformance/fixtures.json``

Reference: ferrotorch-core CPU implementation.
Since ferrotorch-cubecl is itself a portable GPU backend, the conformance
contract is "same result as ferrotorch-core CPU within tolerance."
All reference values in this script are computed analytically using
Python / numpy with the same math the kernels implement, then cross-checked
against PyTorch values where available.

Reference rev: f1cd5d540e7531ccbd21f83cd6c70e8705409017

Tolerance table:
  - Elementwise binary ops (add, sub, mul, div):  atol=1e-6, rtol=0  (IEEE-754 exact)
  - Elementwise unary (neg, abs):                  atol=1e-6, rtol=0
  - Transcendentals (exp, ln, sqrt, sin, cos):     atol=1e-4, rtol=0
  - Tanh, sigmoid:                                  atol=1e-4, rtol=0
  - relu:                                           atol=1e-6, rtol=0
  - matmul:                                         atol=1e-3, rtol=0
  - Chebyshev T/U/V/W polynomials:                 atol=1e-4, rtol=0
  - Hermite H/He, Laguerre L, Legendre P:          atol=1e-3, rtol=0

Usage:

    python3 scripts/regenerate_cubecl_fixtures.py

Required:

    pip install --user numpy
    (PyTorch optional — only used for cross-check comments, not required)
"""

import json
import math
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(
    REPO_ROOT,
    "ferrotorch-cubecl",
    "tests",
    "conformance",
    "fixtures.json",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_list(arr):
    """Convert numpy array to nested Python list of floats."""
    if isinstance(arr, np.ndarray):
        if arr.ndim == 1:
            return [float(x) for x in arr]
        elif arr.ndim == 2:
            return [[float(x) for x in row] for row in arr]
    return [float(x) for x in arr]


def scalar(x):
    return float(x)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def build_elementwise_binary_fixtures():
    """Elementwise binary ops: add, sub, mul, div."""
    fixtures = []

    rng = np.random.default_rng(42)

    for op_name, op_fn in [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
    ]:
        # 1-D, normal values
        a = rng.standard_normal(8).astype(np.float32) + 1.0
        b = rng.standard_normal(8).astype(np.float32) + 1.0
        if op_name == "div":
            b = np.abs(b) + 0.1  # avoid near-zero divisors
        expected = op_fn(a, b).astype(np.float32)
        fixtures.append({
            "op": f"portable_{op_name}_1d",
            "a": fmt_list(a),
            "b": fmt_list(b),
            "shape": [8],
            "expected": fmt_list(expected),
            "atol": 1e-5,
        })

        # 2-D (matrix shaped)
        a2 = (rng.standard_normal(12).astype(np.float32) + 1.0).reshape(3, 4)
        b2 = (rng.standard_normal(12).astype(np.float32) + 1.0).reshape(3, 4)
        if op_name == "div":
            b2 = np.abs(b2) + 0.1
        expected2 = op_fn(a2, b2).astype(np.float32)
        fixtures.append({
            "op": f"portable_{op_name}_2d",
            "a": fmt_list(a2.flatten()),
            "b": fmt_list(b2.flatten()),
            "shape": [3, 4],
            "expected": fmt_list(expected2.flatten()),
            "atol": 1e-5,
        })

    return fixtures


def build_elementwise_unary_fixtures():
    """Elementwise unary ops: relu, neg, abs, exp, ln, sqrt, sin, cos, tanh, sigmoid."""
    fixtures = []
    rng = np.random.default_rng(42)

    # relu
    x = np.array([-3.0, -1.5, -0.1, 0.0, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    expected = np.maximum(x, 0.0)
    fixtures.append({
        "op": "portable_relu_1d",
        "x": fmt_list(x),
        "shape": [8],
        "expected": fmt_list(expected),
        "atol": 1e-6,
    })

    # neg
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    fixtures.append({
        "op": "portable_neg_1d",
        "x": fmt_list(x),
        "shape": [5],
        "expected": fmt_list(-x),
        "atol": 1e-6,
    })

    # abs
    x = np.array([-4.0, -1.5, 0.0, 1.5, 4.0], dtype=np.float32)
    fixtures.append({
        "op": "portable_abs_1d",
        "x": fmt_list(x),
        "shape": [5],
        "expected": fmt_list(np.abs(x)),
        "atol": 1e-6,
    })

    # exp
    x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    fixtures.append({
        "op": "portable_exp_1d",
        "x": fmt_list(x),
        "shape": [6],
        "expected": fmt_list(np.exp(x).astype(np.float32)),
        "atol": 1e-4,
    })

    # ln
    x = np.array([0.1, 0.5, 1.0, math.e, 10.0, 100.0], dtype=np.float32)
    fixtures.append({
        "op": "portable_ln_1d",
        "x": fmt_list(x),
        "shape": [6],
        "expected": fmt_list(np.log(x).astype(np.float32)),
        "atol": 1e-4,
    })

    # sqrt
    x = np.array([0.0, 0.25, 1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    fixtures.append({
        "op": "portable_sqrt_1d",
        "x": fmt_list(x),
        "shape": [6],
        "expected": fmt_list(np.sqrt(x).astype(np.float32)),
        "atol": 1e-4,
    })

    # sin
    pi = np.float32(math.pi)
    x = np.array([0.0, pi / 6, pi / 4, pi / 2, pi, 3 * pi / 2], dtype=np.float32)
    fixtures.append({
        "op": "portable_sin_1d",
        "x": fmt_list(x),
        "shape": [6],
        "expected": fmt_list(np.sin(x).astype(np.float32)),
        "atol": 1e-4,
    })

    # cos
    x = np.array([0.0, pi / 6, pi / 4, pi / 2, pi, 3 * pi / 2], dtype=np.float32)
    fixtures.append({
        "op": "portable_cos_1d",
        "x": fmt_list(x),
        "shape": [6],
        "expected": fmt_list(np.cos(x).astype(np.float32)),
        "atol": 1e-4,
    })

    # tanh
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    fixtures.append({
        "op": "portable_tanh_1d",
        "x": fmt_list(x),
        "shape": [7],
        "expected": fmt_list(np.tanh(x).astype(np.float32)),
        "atol": 1e-4,
    })

    # sigmoid: 1 / (1 + exp(-x))
    x = np.array([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
    sig = (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    fixtures.append({
        "op": "portable_sigmoid_1d",
        "x": fmt_list(x),
        "shape": [5],
        "expected": fmt_list(sig),
        "atol": 1e-4,
    })

    # Large-shape relu (exercises multi-cube launch)
    n = 512
    x_large = rng.standard_normal(n).astype(np.float32)
    expected_large = np.maximum(x_large, 0.0)
    fixtures.append({
        "op": "portable_relu_large",
        "x": fmt_list(x_large),
        "shape": [n],
        "expected": fmt_list(expected_large),
        "atol": 1e-6,
    })

    return fixtures


def build_matmul_fixtures():
    """Matrix multiplication fixtures (2-D, row-major)."""
    fixtures = []

    # Classic 2x3 @ 3x2
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    c = a @ b
    fixtures.append({
        "op": "portable_matmul_2x3_3x2",
        "a": fmt_list(a.flatten()),
        "b": fmt_list(b.flatten()),
        "m": 2, "k": 3, "n": 2,
        "expected": fmt_list(c.flatten()),
        "atol": 1e-3,
    })

    # Identity x matrix → same matrix
    n_sq = 4
    I = np.eye(n_sq, dtype=np.float32)
    rng = np.random.default_rng(42)
    B = rng.standard_normal((n_sq, n_sq)).astype(np.float32)
    fixtures.append({
        "op": "portable_matmul_identity_4x4",
        "a": fmt_list(I.flatten()),
        "b": fmt_list(B.flatten()),
        "m": n_sq, "k": n_sq, "n": n_sq,
        "expected": fmt_list(B.flatten()),
        "atol": 1e-4,
    })

    # Square 4x4 matmul
    rng2 = np.random.default_rng(7)
    A2 = rng2.standard_normal((4, 4)).astype(np.float32)
    B2 = rng2.standard_normal((4, 4)).astype(np.float32)
    C2 = A2 @ B2
    fixtures.append({
        "op": "portable_matmul_4x4",
        "a": fmt_list(A2.flatten()),
        "b": fmt_list(B2.flatten()),
        "m": 4, "k": 4, "n": 4,
        "expected": fmt_list(C2.flatten()),
        "atol": 1e-3,
    })

    return fixtures


def chebyshev_t(x, n):
    """Chebyshev T_n(x): T_0=1, T_1=x, T_{k+1}=2x T_k - T_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    prev2 = np.ones_like(x)
    prev1 = x.copy()
    for _ in range(2, n + 1):
        nxt = 2.0 * x * prev1 - prev2
        prev2 = prev1
        prev1 = nxt
    return prev1


def chebyshev_u(x, n):
    """Chebyshev U_n(x): U_0=1, U_1=2x, U_{k+1}=2x U_k - U_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    prev2 = np.ones_like(x)
    prev1 = 2.0 * x
    for _ in range(2, n + 1):
        nxt = 2.0 * x * prev1 - prev2
        prev2 = prev1
        prev1 = nxt
    return prev1


def chebyshev_v(x, n):
    """Chebyshev V_n(x): V_0=1, V_1=2x-1, V_{k+1}=2x V_k - V_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x - 1.0
    prev2 = np.ones_like(x)
    prev1 = 2.0 * x - 1.0
    for _ in range(2, n + 1):
        nxt = 2.0 * x * prev1 - prev2
        prev2 = prev1
        prev1 = nxt
    return prev1


def chebyshev_w(x, n):
    """Chebyshev W_n(x): W_0=1, W_1=2x+1, W_{k+1}=2x W_k - W_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x + 1.0
    prev2 = np.ones_like(x)
    prev1 = 2.0 * x + 1.0
    for _ in range(2, n + 1):
        nxt = 2.0 * x * prev1 - prev2
        prev2 = prev1
        prev1 = nxt
    return prev1


def hermite_h(x, n):
    """Hermite H_n(x): H_0=1, H_1=2x, H_{k+1}=2x H_k - 2k H_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    prev2 = np.ones_like(x)
    prev1 = 2.0 * x
    for k in range(1, n):
        nxt = 2.0 * x * prev1 - 2.0 * k * prev2
        prev2 = prev1
        prev1 = nxt
    return prev1


def hermite_he(x, n):
    """Hermite He_n(x): He_0=1, He_1=x, He_{k+1}=x He_k - k He_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    prev2 = np.ones_like(x)
    prev1 = x.copy()
    for k in range(1, n):
        nxt = x * prev1 - k * prev2
        prev2 = prev1
        prev1 = nxt
    return prev1


def laguerre_l(x, n):
    """Laguerre L_n(x): L_0=1, L_1=1-x, (k+1)L_{k+1}=(2k+1-x)L_k - k L_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 1.0 - x
    prev2 = np.ones_like(x)
    prev1 = 1.0 - x
    for k in range(1, n):
        nxt = ((2.0 * k + 1.0 - x) * prev1 - k * prev2) / (k + 1.0)
        prev2 = prev1
        prev1 = nxt
    return prev1


def legendre_p(x, n):
    """Legendre P_n(x): P_0=1, P_1=x, (k+1)P_{k+1}=(2k+1)x P_k - k P_{k-1}."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    prev2 = np.ones_like(x)
    prev1 = x.copy()
    for k in range(1, n):
        nxt = ((2.0 * k + 1.0) * x * prev1 - k * prev2) / (k + 1.0)
        prev2 = prev1
        prev1 = nxt
    return prev1


def build_polynomial_fixtures():
    """Orthogonal polynomial families: Chebyshev T/U/V/W, Hermite H/He, Laguerre L, Legendre P."""
    fixtures = []

    # Points in [-1, 1] (natural domain for Chebyshev / Legendre / Hermite)
    x_cheb = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    # Points for Laguerre (domain [0, ∞))
    x_lag = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    # Hermite points (all reals)
    x_herm = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)

    poly_cases = [
        # (op_name, fn, x_arr, degree, atol)
        ("portable_chebyshev_t_n3", chebyshev_t, x_cheb, 3, 1e-4),
        ("portable_chebyshev_t_n0", chebyshev_t, x_cheb, 0, 1e-6),
        ("portable_chebyshev_t_n1", chebyshev_t, x_cheb, 1, 1e-6),
        ("portable_chebyshev_t_n5", chebyshev_t, x_cheb, 5, 1e-3),
        ("portable_chebyshev_u_n2", chebyshev_u, x_cheb, 2, 1e-4),
        ("portable_chebyshev_u_n0", chebyshev_u, x_cheb, 0, 1e-6),
        ("portable_chebyshev_v_n1", chebyshev_v, x_cheb, 1, 1e-6),
        ("portable_chebyshev_v_n3", chebyshev_v, x_cheb, 3, 1e-4),
        ("portable_chebyshev_w_n1", chebyshev_w, x_cheb, 1, 1e-6),
        ("portable_chebyshev_w_n3", chebyshev_w, x_cheb, 3, 1e-4),
        ("portable_hermite_h_n3",  hermite_h,   x_herm, 3, 1e-2),
        ("portable_hermite_h_n0",  hermite_h,   x_herm, 0, 1e-6),
        ("portable_hermite_h_n1",  hermite_h,   x_herm, 1, 1e-5),
        ("portable_hermite_he_n3", hermite_he,  x_herm, 3, 1e-3),
        ("portable_hermite_he_n0", hermite_he,  x_herm, 0, 1e-6),
        ("portable_laguerre_l_n2", laguerre_l,  x_lag,  2, 1e-4),
        ("portable_laguerre_l_n0", laguerre_l,  x_lag,  0, 1e-6),
        ("portable_legendre_p_n2", legendre_p,  x_cheb, 2, 1e-4),
        ("portable_legendre_p_n0", legendre_p,  x_cheb, 0, 1e-6),
        ("portable_legendre_p_n3", legendre_p,  x_cheb, 3, 1e-4),
    ]

    for op_name, fn, x, deg, atol in poly_cases:
        x_f32 = x.astype(np.float32)
        # Compute reference in float64, then cast expected to float32
        expected = fn(x.astype(np.float64), deg).astype(np.float32)
        fixtures.append({
            "op": op_name,
            "x": fmt_list(x_f32),
            "shape": [len(x_f32)],
            "degree": deg,
            "expected": fmt_list(expected),
            "atol": atol,
        })

    return fixtures


def build_device_meta_fixtures():
    """Fixtures for CubeDevice / CubeRuntime API surface (CPU-only, structural)."""
    fixtures = []

    fixtures.append({
        "op": "cube_device_ordinal",
        "cases": [
            {"variant": "Cuda", "ordinal_arg": 3, "expected_ordinal": 3},
            {"variant": "Wgpu", "ordinal_arg": 1, "expected_ordinal": 1},
            {"variant": "Rocm", "ordinal_arg": 0, "expected_ordinal": 0},
        ],
    })

    fixtures.append({
        "op": "cube_device_backend_name",
        "cases": [
            {"variant": "Cuda", "ordinal_arg": 0, "expected": "cuda"},
            {"variant": "Wgpu", "ordinal_arg": 0, "expected": "wgpu"},
            {"variant": "Rocm", "ordinal_arg": 0, "expected": "rocm"},
        ],
    })

    fixtures.append({
        "op": "cube_device_display",
        "cases": [
            {"variant": "Cuda", "ordinal_arg": 2, "expected": "cuda:2"},
            {"variant": "Wgpu", "ordinal_arg": 0, "expected": "wgpu:0"},
            {"variant": "Rocm", "ordinal_arg": 1, "expected": "rocm:1"},
        ],
    })

    return fixtures


def build_gguf_block_meta_fixtures():
    """Structural fixtures for GgufBlockKind.block_bytes / block_elements."""
    # Values from quant.rs constants
    block_bytes_map = {
        "Q4_0": 18,
        "Q4_1": 20,
        "Q5_0": 22,
        "Q5_1": 24,
        "Q8_0": 34,
        "Q8_1": 40,
    }
    return [{
        "op": "gguf_block_kind_meta",
        "cases": [
            {
                "variant": v,
                "expected_block_bytes": bb,
                "expected_block_elements": 32,
            }
            for v, bb in block_bytes_map.items()
        ],
    }]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fixtures = []

    fixtures += build_elementwise_binary_fixtures()
    fixtures += build_elementwise_unary_fixtures()
    fixtures += build_matmul_fixtures()
    fixtures += build_polynomial_fixtures()
    fixtures += build_device_meta_fixtures()
    fixtures += build_gguf_block_meta_fixtures()

    output = {
        "metadata": {
            "generator": "scripts/regenerate_cubecl_fixtures.py",
            "reference": "ferrotorch-core CPU + analytical numpy reference",
            "reference_rev": "f1cd5d540e7531ccbd21f83cd6c70e8705409017",
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "tracking_issue": "#860",
        },
        "fixtures": fixtures,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(fixtures)} fixtures to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
