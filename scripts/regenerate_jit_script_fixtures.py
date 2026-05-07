#!/usr/bin/env python3
"""
Regenerate reference fixtures for the ferrotorch-jit-script conformance suite.

Tracking issue: #826 (Conformance Buildout C1 — ferrotorch-jit-script).

Output: ``ferrotorch-jit-script/tests/conformance/fixtures.json``.

Pin: torch == 2.11.0

Background
----------
``torch.jit.script`` is an annotation that compiles a Python function into
a TorchScript graph.  ferrotorch's ``#[script]`` is its Rust analogue —
a proc-macro that rewrites a Rust function body so it is captured via
``ferrotorch_jit::trace`` and returned as a ``TracedModule<T>``.

The parity contract is *structural*, not numeric:

  "calling forward_multi on the TracedModule with input X produces the same
   numeric output that calling the original function on X would produce."

This script records ``(input_values, expected_output)`` pairs by running the
equivalent PyTorch expressions directly (not via ``torch.jit.script``, which
is a Python-only construct).  The ferrotorch Rust tests load these pairs,
construct tensors from ``input_values``, run the traced module, and assert
the output matches ``expected_output`` within tolerance.

Spec-only cases (compile-error paths, unrecognized return types) have no
numeric PyTorch reference; they are recorded in the fixture with
``cascade_skip`` set to the sentinel string ``"spec-only marker, no PyTorch
reference"`` so the Rust tests can skip them gracefully.

Usage
-----
    python3 scripts/regenerate_jit_script_fixtures.py

Required Python deps:

    torch==2.11.0   (CPU-only build is sufficient; no CUDA needed)
    numpy           (for moment statistics, though not strictly used here)

The script exits 0 on success and writes
``ferrotorch-jit-script/tests/conformance/fixtures.json`` in the repo root.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import sys

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    print(
        "ERROR: 'torch' is not installed. Install with:\n"
        "    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu",
        file=sys.stderr,
    )
    sys.exit(1)

REQUIRED_TORCH = "2.11.0"
actual = torch.__version__
# Accept exact match or patch-level variations (e.g. 2.11.0+cpu)
if not actual.startswith(REQUIRED_TORCH):
    print(
        f"WARNING: torch version is {actual!r}, expected {REQUIRED_TORCH!r}. "
        "Fixtures may drift if versions differ.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _tensor(values: list[float], dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def _tolist(t: torch.Tensor) -> list[float]:
    return t.tolist()


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

SPEC_ONLY_SKIP = "spec-only marker, no PyTorch reference"


def build_fixtures() -> list[dict]:
    fixtures: list[dict] = []

    # ------------------------------------------------------------------
    # case: two_arg_weighted_sum_f32
    # Mirrors: #[script] fn weighted_sum(a, w) -> sum(a * w)
    # PyTorch: (a * w).sum()
    # ------------------------------------------------------------------
    a = _tensor([1.0, 2.0, 3.0])
    w = _tensor([4.0, 5.0, 6.0])
    expected = (a * w).sum()
    fixtures.append(
        {
            "case": "two_arg_weighted_sum_f32",
            "op": "script",
            "description": (
                "Weighted element-wise sum: sum(a * w). Tests that #[script] "
                "produces a TracedModule whose forward_multi returns the correct scalar."
            ),
            "input_a": _tolist(a),
            "input_w": _tolist(w),
            "shape_a": list(a.shape),
            "shape_w": list(w.shape),
            "dtype": "float32",
            "expected_output": _tolist(expected.unsqueeze(0)),
            "expected_shape": [1],
            "torch_reference": (
                "a = torch.tensor([1.0,2.0,3.0]); w = torch.tensor([4.0,5.0,6.0]); "
                f"(a*w).sum().item() == {expected.item()}"
            ),
        }
    )

    # ------------------------------------------------------------------
    # case: three_arg_add_f32
    # Mirrors: #[script] fn three_arg_add(a, b, c) -> a + b + c
    # ------------------------------------------------------------------
    a2 = _tensor([1.0, 2.0])
    b2 = _tensor([3.0, 4.0])
    c2 = _tensor([5.0, 6.0])
    expected2 = a2 + b2 + c2
    fixtures.append(
        {
            "case": "three_arg_add_f32",
            "op": "script",
            "description": (
                "Three-argument addition: a + b + c element-wise. "
                "Tests that #[script] handles multi-arg functions correctly."
            ),
            "input_a": _tolist(a2),
            "input_b": _tolist(b2),
            "input_c": _tolist(c2),
            "shape_a": list(a2.shape),
            "shape_b": list(b2.shape),
            "shape_c": list(c2.shape),
            "dtype": "float32",
            "expected_output": _tolist(expected2),
            "expected_shape": list(expected2.shape),
            "torch_reference": (
                "a=torch.tensor([1.,2.]); b=torch.tensor([3.,4.]); "
                f"c=torch.tensor([5.,6.]); (a+b+c).tolist() == {_tolist(expected2)}"
            ),
        }
    )

    # ------------------------------------------------------------------
    # case: module_save_load_roundtrip_f32
    # Mirrors: weighted_sum(a=[2,3], w=[4,5]) -> TracedModule -> bytes -> TracedModule
    # ------------------------------------------------------------------
    a3 = _tensor([2.0, 3.0])
    w3 = _tensor([4.0, 5.0])
    expected3 = (a3 * w3).sum()
    fixtures.append(
        {
            "case": "module_save_load_roundtrip_f32",
            "op": "script",
            "description": (
                "TracedModule serialization round-trip: to_bytes then from_bytes "
                "should produce a module that re-executes correctly."
            ),
            "input_a": _tolist(a3),
            "input_w": _tolist(w3),
            "shape_a": list(a3.shape),
            "shape_w": list(w3.shape),
            "dtype": "float32",
            "expected_output": _tolist(expected3.unsqueeze(0)),
            "expected_shape": [1],
            "torch_reference": (
                "a=torch.tensor([2.,3.]); w=torch.tensor([4.,5.]); "
                f"(a*w).sum().item() == {expected3.item()}"
            ),
        }
    )

    # ------------------------------------------------------------------
    # case: scalar_type_preservation_f64
    # Mirrors: weighted_sum_f64(a, w) — same arithmetic but in f64.
    # This is the regression guard for the silent-f32-fallback bug.
    # ------------------------------------------------------------------
    a4 = _tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    w4 = _tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    expected4 = (a4 * w4).sum()
    fixtures.append(
        {
            "case": "scalar_type_preservation_f64",
            "op": "script",
            "description": (
                "The macro must preserve the scalar type T: a function returning "
                "Tensor<f64> must produce TracedModule<f64>, not TracedModule<f32>. "
                "This is the regression guard for the silent-f32-fallback bug."
            ),
            "input_a": _tolist(a4),
            "input_w": _tolist(w4),
            "shape_a": list(a4.shape),
            "shape_w": list(w4.shape),
            "dtype": "float64",
            "expected_output": [expected4.item()],
            "expected_shape": [1],
            "torch_reference": (
                "a=torch.tensor([1.,2.,3.], dtype=torch.float64); "
                "w=torch.tensor([4.,5.,6.], dtype=torch.float64); "
                f"(a*w).sum().item() == {expected4.item()}"
            ),
        }
    )

    # ------------------------------------------------------------------
    # case: module_reuse_f32
    # Tests that the same TracedModule can be called multiple times.
    # ------------------------------------------------------------------
    a5a = _tensor([1.0, 1.0])
    w5a = _tensor([2.0, 2.0])
    a5b = _tensor([3.0, 4.0])
    w5b = _tensor([1.0, 2.0])
    expected5a = (a5a * w5a).sum()
    expected5b = (a5b * w5b).sum()
    fixtures.append(
        {
            "case": "module_reuse_f32",
            "op": "script",
            "description": (
                "A TracedModule can be re-executed multiple times with different "
                "inputs, returning different outputs each time."
            ),
            "input_a_first": _tolist(a5a),
            "input_w_first": _tolist(w5a),
            "input_a_second": _tolist(a5b),
            "input_w_second": _tolist(w5b),
            "shape": list(a5a.shape),
            "dtype": "float32",
            "expected_output_first": _tolist(expected5a.unsqueeze(0)),
            "expected_output_second": _tolist(expected5b.unsqueeze(0)),
            "torch_reference": (
                f"sum([1*2,1*2])=={expected5a.item()}; "
                f"sum([3*1,4*2])=={expected5b.item()}"
            ),
        }
    )

    # ------------------------------------------------------------------
    # Spec-only cases: compile-error paths
    # These have no PyTorch runtime reference; they are documented here
    # so the surface coverage gate knows they exist.
    # ------------------------------------------------------------------
    fixtures.append(
        {
            "case": "unrecognized_return_type_emits_compile_error",
            "op": "script_error",
            "description": (
                "spec-only: the macro must emit compile_error! when the return "
                "type is not Tensor<T>, FerrotorchResult<Tensor<T>>, or "
                "Result<Tensor<T>,_>. Verified via compile_fail tests only — "
                "no runtime fixture."
            ),
            "cascade_skip": SPEC_ONLY_SKIP,
        }
    )
    fixtures.append(
        {
            "case": "missing_return_type_emits_compile_error",
            "op": "script_error",
            "description": (
                "spec-only: the macro must emit compile_error! when the annotated "
                "function has no return type. Verified via compile_fail tests only "
                "— no runtime fixture."
            ),
            "cascade_skip": SPEC_ONLY_SKIP,
        }
    )

    return fixtures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path to write the fixtures JSON. "
            "Defaults to ferrotorch-jit-script/tests/conformance/fixtures.json "
            "relative to the repo root."
        ),
    )
    args = parser.parse_args()

    # Locate repo root (the directory containing this script's parent).
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_out = (
        repo_root
        / "ferrotorch-jit-script"
        / "tests"
        / "conformance"
        / "fixtures.json"
    )
    out_path = pathlib.Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = build_fixtures()

    payload = {
        "metadata": {
            "torch_version": torch.__version__,
            "python_executable": sys.executable,
            "python_platform": platform.platform(),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "description": (
                "Reference fixtures for ferrotorch-jit-script conformance suite. "
                "torch.jit.script is a code-transformation annotation with no numeric "
                "output — parity is structural (module is callable, forward returns "
                "correct values). Fixtures encode (input_values, expected_output) pairs "
                "derived from equivalent PyTorch arithmetic expressions."
            ),
        },
        "fixtures": fixtures,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    n_live = sum(1 for fx in fixtures if "cascade_skip" not in fx)
    n_skip = sum(1 for fx in fixtures if "cascade_skip" in fx)
    print(
        f"Written {out_path}\n"
        f"  {len(fixtures)} total fixtures: {n_live} live, {n_skip} cascade-skip"
    )


if __name__ == "__main__":
    main()
