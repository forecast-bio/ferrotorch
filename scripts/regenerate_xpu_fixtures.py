#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for the ferrotorch-xpu conformance suite.

Tracking issue: #827 (XPU conformance suite).

Output: ``ferrotorch-xpu/tests/conformance/fixtures/fixtures.json``.

Reference: ``torch.xpu.*`` (Intel XPU backend), pinned to ``torch == 2.11.0``.

On this test box (WSL2, no Intel GPU) ``torch.xpu.is_available()`` returns
``False`` and ``torch.xpu.device_count()`` returns ``0``. Most fixtures are
generated from the CPU-side torch functions and the XPU-availability flags.
Live-XPU operations (e.g. actually executing kernels on an XPU device) are
cascade_skip-tagged in the Rust test with reason "no Intel XPU on test box".

The fixtures that CAN be generated offline:
  * xpu_is_available   — torch.xpu.is_available()  → False on this box
  * xpu_device_count   — torch.xpu.device_count()  → 0 on this box
  * XpuDevice_metadata — ordinal, device() string, is_available() shape
  * make_xpu_tensor    — shape/dtype/device metadata contract (no XPU needed)
  * ops_cpu_reference  — CPU-side torch reference values for every op the crate
                         wraps (add, sub, mul, div, matmul, neg, abs, relu, exp,
                         ln, sqrt, sin, cos, tanh, sigmoid, polynomial families)

The Rust conformance test exercises ferrotorch-xpu's stub path (wgpu feature
disabled by default in the test environment) against these reference values and
the API-shape guarantees (DeviceUnavailable returns, ordinal queries, etc.).

Usage:
    python3 scripts/regenerate_xpu_fixtures.py

Required Python deps:
    torch==2.11.0
    numpy
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root & output path
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT = REPO_ROOT / "ferrotorch-xpu" / "tests" / "conformance" / "fixtures" / "fixtures.json"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Regenerate ferrotorch-xpu conformance fixtures."
)
parser.add_argument(
    "--seed", type=int, default=42, help="RNG seed for deterministic fixtures."
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Import torch
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    print("ERROR: torch is not installed. Run: pip install torch==2.11.0", file=sys.stderr)
    sys.exit(1)

torch_version = torch.__version__

# ---------------------------------------------------------------------------
# torch.xpu availability probe (what IS knowable without an Intel GPU)
# ---------------------------------------------------------------------------

xpu_available = torch.xpu.is_available() if hasattr(torch, "xpu") else False
xpu_device_count = torch.xpu.device_count() if hasattr(torch, "xpu") else 0
has_xpu_module = hasattr(torch, "xpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def t2list(t: "torch.Tensor") -> list:
    """Convert a float32 tensor to a flat Python list (always 1-D)."""
    return t.detach().cpu().reshape(-1).tolist()


def make_cpu_input(values: list[float]) -> "torch.Tensor":
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Build fixtures list
# ---------------------------------------------------------------------------

torch.manual_seed(args.seed)

fixtures: list[dict] = []

# ---- XPU availability / device count (torch.xpu surface) ------------------

fixtures.append({
    "op": "xpu_is_available",
    "result": xpu_available,
    "note": "torch.xpu.is_available() on this machine",
})

fixtures.append({
    "op": "xpu_device_count",
    "result": xpu_device_count,
    "note": "torch.xpu.device_count() on this machine",
})

fixtures.append({
    "op": "has_xpu_module",
    "result": has_xpu_module,
    "note": "hasattr(torch, 'xpu') — module presence probe",
})

# ---- XpuDevice metadata shapes (no GPU required) ---------------------------

fixtures.append({
    "op": "xpu_device_metadata",
    "ordinal": 0,
    "device_str": "xpu:0",
    "is_available_when_wgpu_disabled": False,
    "note": (
        "XpuDevice::new(0) errors with DeviceUnavailable when wgpu feature is "
        "absent; is_available() returns false; device() returns Device::Xpu(0) "
        "rendered as 'xpu:0' via Display."
    ),
})

# ---- CPU reference values for every op the crate wraps --------------------
# The Rust-side test runs ferrotorch-xpu's stub stubs (wgpu disabled) and
# verifies the *error contract* (DeviceUnavailable). The CPU torch values here
# serve as the reference that a live XPU path would need to match within
# tolerance. They are labelled with ``cascade_skip_live_xpu = true`` in the
# Rust test because no Intel GPU is present.

# Binary elementwise ops
for op_name, op_fn in [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
]:
    for (av, bv) in [
        ([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]),
        ([-1.5, 0.0, 2.5], [3.0, 1.0, -0.5]),
    ]:
        a = make_cpu_input(av)
        b = make_cpu_input(bv)
        result = op_fn(a, b)
        fixtures.append({
            "op": f"xpu_{op_name}",
            "a": av,
            "b": bv,
            "expected": t2list(result),
            "shape": list(result.shape),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })

# Matmul: 2-D only (crate is 2-D-only by design)
a_mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
b_mat = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
c_mat = a_mat @ b_mat
fixtures.append({
    "op": "xpu_matmul",
    "a": t2list(a_mat),
    "b": t2list(b_mat),
    "a_shape": list(a_mat.shape),
    "b_shape": list(b_mat.shape),
    "expected": t2list(c_mat),
    "expected_shape": list(c_mat.shape),
    "cascade_skip_live_xpu": True,
    "cascade_skip_reason": "no Intel XPU on test box",
})

# Unary elementwise ops
unary_inputs = {
    "xpu_neg":     [1.0, -2.0, 0.0, 3.5],
    "xpu_abs":     [-3.0, 1.0, 0.0, -0.5],
    "xpu_relu":    [-3.0, -1.0, 0.0, 1.0, 3.0],
    "xpu_exp":     [0.0, 1.0, 2.0, -1.0],
    "xpu_ln":      [1.0, math.e, 10.0, 0.5],
    "xpu_sqrt":    [0.0, 1.0, 4.0, 9.0],
    "xpu_sin":     [0.0, math.pi / 6, math.pi / 2, math.pi],
    "xpu_cos":     [0.0, math.pi / 3, math.pi / 2, math.pi],
    "xpu_tanh":    [-2.0, -1.0, 0.0, 1.0, 2.0],
    "xpu_sigmoid": [-4.0, -1.0, 0.0, 1.0, 4.0],
}

for op_name, vals in unary_inputs.items():
    x = make_cpu_input(vals)
    torch_op_name = op_name.replace("xpu_", "")
    if torch_op_name == "ln":
        result = torch.log(x)
    elif torch_op_name == "relu":
        result = torch.relu(x)
    elif torch_op_name == "sigmoid":
        result = torch.sigmoid(x)
    else:
        result = getattr(torch, torch_op_name)(x)
    fixtures.append({
        "op": op_name,
        "x": vals,
        "expected": t2list(result),
        "shape": list(result.shape),
        "cascade_skip_live_xpu": True,
        "cascade_skip_reason": "no Intel XPU on test box",
    })

# Polynomial families (orthogonal polynomials via torch.special)
# T_n, U_n, V_n, W_n (Chebyshev), H_n (Hermite physicists'),
# He_n (Hermite probabilists'), L_n (Laguerre), P_n (Legendre)
poly_inputs = [-1.0, -0.5, 0.0, 0.5, 1.0]
x_poly = make_cpu_input(poly_inputs)

for n in [1, 2, 3]:
    # Chebyshev T_n (first kind): torch.special.chebyshev_polynomial_t
    try:
        result = torch.special.chebyshev_polynomial_t(x_poly, n)
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_t",
            "x": poly_inputs,
            "n": n,
            "expected": t2list(result),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        # Fallback: compute T_n analytically for small n
        def chebyshev_t(xv: list[float], nv: int) -> list[float]:
            return [math.cos(nv * math.acos(max(-1.0, min(1.0, xi)))) for xi in xv]
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_t",
            "x": poly_inputs,
            "n": n,
            "expected": chebyshev_t(poly_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

    # Chebyshev U_n (second kind)
    try:
        result_u = torch.special.chebyshev_polynomial_u(x_poly, n)
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_u",
            "x": poly_inputs,
            "n": n,
            "expected": t2list(result_u),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        # U_n(cos θ) = sin((n+1)θ)/sin(θ), with U_n(±1) handled as limit
        def chebyshev_u(xv: list[float], nv: int) -> list[float]:
            out = []
            for xi in xv:
                xi = max(-1.0, min(1.0, xi))
                if abs(abs(xi) - 1.0) < 1e-9:
                    out.append(float((nv + 1) * (1 if xi > 0 else ((-1) ** nv))))
                else:
                    theta = math.acos(xi)
                    out.append(math.sin((nv + 1) * theta) / math.sin(theta))
            return out
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_u",
            "x": poly_inputs,
            "n": n,
            "expected": chebyshev_u(poly_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

    # Chebyshev V_n (third kind): V_n(x) = T_n(x) + U_{n-1}(x) [with U_{-1}=0]
    try:
        result_v = torch.special.chebyshev_polynomial_v(x_poly, n)
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_v",
            "x": poly_inputs,
            "n": n,
            "expected": t2list(result_v),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        def chebyshev_v(xv: list[float], nv: int) -> list[float]:
            tn = chebyshev_t(xv, nv)
            un_1 = chebyshev_u(xv, nv - 1) if nv >= 1 else [0.0] * len(xv)
            return [t + u for t, u in zip(tn, un_1)]
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_v",
            "x": poly_inputs,
            "n": n,
            "expected": chebyshev_v(poly_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

    # Chebyshev W_n (fourth kind): W_n(x) = U_n(x) - U_{n-1}(x) [with U_{-1}=0]
    try:
        result_w = torch.special.chebyshev_polynomial_w(x_poly, n)
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_w",
            "x": poly_inputs,
            "n": n,
            "expected": t2list(result_w),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        def chebyshev_w(xv: list[float], nv: int) -> list[float]:
            un = chebyshev_u(xv, nv)
            un_1 = chebyshev_u(xv, nv - 1) if nv >= 1 else [0.0] * len(xv)
            return [u - u1 for u, u1 in zip(un, un_1)]
        fixtures.append({
            "op": "xpu_chebyshev_polynomial_w",
            "x": poly_inputs,
            "n": n,
            "expected": chebyshev_w(poly_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

    # Hermite H_n (physicists')
    try:
        result_h = torch.special.hermite_polynomial_h(x_poly, n)
        fixtures.append({
            "op": "xpu_hermite_polynomial_h",
            "x": poly_inputs,
            "n": n,
            "expected": t2list(result_h),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        # H_0=1, H_1=2x, H_n=2x*H_{n-1} - 2(n-1)*H_{n-2}
        def hermite_h(xv: list[float], nv: int) -> list[float]:
            def h_at(xi: float, nv2: int) -> float:
                if nv2 == 0:
                    return 1.0
                if nv2 == 1:
                    return 2.0 * xi
                h0, h1 = 1.0, 2.0 * xi
                for k in range(2, nv2 + 1):
                    h0, h1 = h1, 2.0 * xi * h1 - 2.0 * (k - 1) * h0
                return h1
            return [h_at(xi, nv) for xi in xv]
        fixtures.append({
            "op": "xpu_hermite_polynomial_h",
            "x": poly_inputs,
            "n": n,
            "expected": hermite_h(poly_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

    # Hermite He_n (probabilists')
    try:
        result_he = torch.special.hermite_polynomial_he(x_poly, n)
        fixtures.append({
            "op": "xpu_hermite_polynomial_he",
            "x": poly_inputs,
            "n": n,
            "expected": t2list(result_he),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        # He_0=1, He_1=x, He_n=x*He_{n-1} - (n-1)*He_{n-2}
        def hermite_he(xv: list[float], nv: int) -> list[float]:
            def he_at(xi: float, nv2: int) -> float:
                if nv2 == 0:
                    return 1.0
                if nv2 == 1:
                    return xi
                h0, h1 = 1.0, xi
                for k in range(2, nv2 + 1):
                    h0, h1 = h1, xi * h1 - (k - 1) * h0
                return h1
            return [he_at(xi, nv) for xi in xv]
        fixtures.append({
            "op": "xpu_hermite_polynomial_he",
            "x": poly_inputs,
            "n": n,
            "expected": hermite_he(poly_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

    # Laguerre L_n (on non-negative inputs: 0..4)
    laguerre_inputs = [0.0, 0.5, 1.0, 2.0, 4.0]
    x_lag = make_cpu_input(laguerre_inputs)
    try:
        result_l = torch.special.laguerre_polynomial_l(x_lag, n)
        fixtures.append({
            "op": "xpu_laguerre_polynomial_l",
            "x": laguerre_inputs,
            "n": n,
            "expected": t2list(result_l),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        # L_0=1, L_1=1-x, L_n=((2n-1-x)*L_{n-1} - (n-1)*L_{n-2})/n
        def laguerre_l(xv: list[float], nv: int) -> list[float]:
            def l_at(xi: float, nv2: int) -> float:
                if nv2 == 0:
                    return 1.0
                if nv2 == 1:
                    return 1.0 - xi
                l0, l1 = 1.0, 1.0 - xi
                for k in range(2, nv2 + 1):
                    l0, l1 = l1, ((2 * k - 1 - xi) * l1 - (k - 1) * l0) / k
                return l1
            return [l_at(xi, nv) for xi in xv]
        fixtures.append({
            "op": "xpu_laguerre_polynomial_l",
            "x": laguerre_inputs,
            "n": n,
            "expected": laguerre_l(laguerre_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

    # Legendre P_n
    try:
        result_p = torch.special.legendre_polynomial_p(x_poly, n)
        fixtures.append({
            "op": "xpu_legendre_polynomial_p",
            "x": poly_inputs,
            "n": n,
            "expected": t2list(result_p),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
        })
    except AttributeError:
        # Bonnet recurrence: P_0=1, P_1=x, P_n=((2n-1)x*P_{n-1}-(n-1)*P_{n-2})/n
        def legendre_p(xv: list[float], nv: int) -> list[float]:
            def p_at(xi: float, nv2: int) -> float:
                if nv2 == 0:
                    return 1.0
                if nv2 == 1:
                    return xi
                p0, p1 = 1.0, xi
                for k in range(2, nv2 + 1):
                    p0, p1 = p1, ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k
                return p1
            return [p_at(xi, nv) for xi in xv]
        fixtures.append({
            "op": "xpu_legendre_polynomial_p",
            "x": poly_inputs,
            "n": n,
            "expected": legendre_p(poly_inputs, n),
            "cascade_skip_live_xpu": True,
            "cascade_skip_reason": "no Intel XPU on test box",
            "computed_analytically": True,
        })

# ---- Device-mismatch contract fixture ----------------------------------------
# Verifies that xpu_add rejects a CPU tensor with DeviceUnavailable/DeviceMismatch.
fixtures.append({
    "op": "xpu_add_rejects_cpu_input",
    "note": (
        "xpu_add(cpu_tensor, cpu_tensor, xpu) must return Err(DeviceMismatch) "
        "when both inputs are on CPU. When wgpu is disabled, the op returns "
        "Err(DeviceUnavailable) before it even checks device placement."
    ),
    "expected_error_variants": ["DeviceMismatch", "DeviceUnavailable"],
})

# ---- Shape/kind validation fixture -------------------------------------------
fixtures.append({
    "op": "xpu_matmul_rejects_1d_input",
    "note": (
        "xpu_matmul([3], [3]) must return Err(ShapeMismatch). 1-D × 1-D is "
        "not a valid 2-D matrix multiply in ferrotorch-xpu. Mirrors the existing "
        "unit test xpu_matmul_rejects_non_2d_inputs."
    ),
    "expected_error_variant": "ShapeMismatch",
})

# ---- is_available contract fixture -------------------------------------------
# torch.xpu.is_available() on this box
fixtures.append({
    "op": "xpu_no_backend_stub",
    "wgpu_feature_enabled": False,
    "xpu_device_new_result": "Err(DeviceUnavailable)",
    "is_available_result": False,
    "note": (
        "Without the wgpu feature, XpuDevice::new returns DeviceUnavailable "
        "and is_available() returns false. Matches torch.xpu.is_available() "
        "on a box with no Intel XPU."
    ),
})

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

output = {
    "metadata": {
        "torch_version": torch_version,
        "xpu_available": xpu_available,
        "xpu_device_count": xpu_device_count,
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": args.seed,
        "note": (
            "Fixtures generated on a WSL2 box with no Intel XPU. "
            "Live-XPU fixtures are tagged cascade_skip_live_xpu=True with "
            "reason 'no Intel XPU on test box'. They serve as reference values "
            "for a future environment where torch.xpu.is_available() == True."
        ),
    },
    "fixtures": fixtures,
}

with open(OUTPUT, "w", encoding="utf-8") as fh:
    json.dump(output, fh, indent=2)
    fh.write("\n")

print(f"Wrote {len(fixtures)} fixtures to {OUTPUT}")
print(f"  torch version:     {torch_version}")
print(f"  xpu_available:     {xpu_available}")
print(f"  xpu_device_count:  {xpu_device_count}")
