#!/usr/bin/env python3
"""Verify ferrotorch-jit trace + AoT-compile parity on the pinned MLP
(#1170, Phase G.4).

Drives the `cargo run --example jit_trace_dump` binary, then compares
each rust-side `.bin` against the torch-reference output frozen in the
pin (`ferrotorch/jit-trace-parity-v1::_value_parity_output.bin`).

Three-stage gate (mirrors the comments in registry.rs + pin script):

  1. **eager vs reference** — ferrotorch's eager MLP forward against
     torch's eager MLP forward. Picks up the float32-BLAS floor.
     Threshold: max_abs <= 1e-4, cosine_sim >= 0.99999.

  2. **traced vs eager** — `ferrotorch_jit::trace` reproduces the
     same autograd ops via the IR interpreter. Pure graph walk — no
     reordering — so bit-tight up to f32 rounding.
     Threshold: max_abs <= 1e-5, cosine_sim >= 0.99999.

  3. **compiled vs eager** — `ferrotorch_jit::compile` adds constant
     folding / DCE / operator fusion / memory planning on top of the
     trace. Fusion may re-order non-associative sums (the
     FusedLinearActivation pattern collapses Linear + Relu into one
     interpreter step), so the tolerance is slightly looser.
     Threshold: max_abs <= 1e-4, cosine_sim >= 0.9999.

Exit code:
  * 0 — every stage PASSES
  * 1 — any stage FAILS or the rust dump command errors out

Usage:
  python3 scripts/verify_jit_inference.py
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Tolerances — frozen per the architect spec at the top of #1170.
# ---------------------------------------------------------------------------

EAGER_MAX_ABS = 1e-4
EAGER_COS_MIN = 0.99999
TRACED_MAX_ABS = 1e-5
TRACED_COS_MIN = 0.99999
COMPILED_MAX_ABS = 1e-4
COMPILED_COS_MIN = 0.9999

REPO_ROOT = Path(__file__).resolve().parent.parent
PIN_FIXTURE_DIR = Path("/tmp/ferrotorch_pin_jit_trace_parity_v1")
HF_REPO_ID = "ferrotorch/jit-trace-parity-v1"


# ---------------------------------------------------------------------------
# Binary I/O — matches the pin script's `dump_f32` and the rust example's
# `write_f32_tensor`: `[u32 ndim][u32 × ndim shape][f32 le data]`.
# ---------------------------------------------------------------------------


def read_f32_tensor(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with path.open("rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
        numel = 1
        for d in shape:
            numel *= d
        data = np.frombuffer(f.read(numel * 4), dtype="<f4")
    return shape, data.reshape(shape).copy()


# ---------------------------------------------------------------------------
# Comparison primitives.
# ---------------------------------------------------------------------------


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        # Both zero -> trivially identical; one zero -> degenerate.
        return 1.0 if na == nb else 0.0
    return float(np.dot(af, bf) / (na * nb))


def evaluate(
    stage: str, rust_path: Path, ref: np.ndarray, max_abs_tol: float, cos_min: float
) -> bool:
    if not rust_path.exists():
        print(f"[verify] {stage:10s}  FAIL — rust dump {rust_path} missing")
        return False
    rust_shape, rust = read_f32_tensor(rust_path)
    if rust.shape != ref.shape:
        print(
            f"[verify] {stage:10s}  FAIL — shape mismatch rust={rust_shape} ref={ref.shape}"
        )
        return False
    ma = max_abs(rust, ref)
    cs = cosine_sim(rust, ref)
    ok = (ma <= max_abs_tol) and (cs >= cos_min)
    verdict = "PASS" if ok else "FAIL"
    print(
        f"[verify] {stage:10s}  {verdict} — max_abs={ma:.3e} (<= {max_abs_tol:.0e}) "
        f"cosine_sim={cs:.6f} (>= {cos_min:g})"
    )
    return ok


# ---------------------------------------------------------------------------
# Fixture preparation.
# ---------------------------------------------------------------------------


def ensure_fixtures(out_dir: Path) -> Path:
    """Ensure `_value_parity_{input,output}.bin` are available, return dir."""
    needed = ("_value_parity_input.bin", "_value_parity_output.bin")
    # 1. Local copy from the pin script's WORK_DIR.
    if all((PIN_FIXTURE_DIR / n).exists() for n in needed):
        return PIN_FIXTURE_DIR
    # 2. Download via huggingface_hub.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"[verify] huggingface_hub unavailable ({exc!r}); cannot fetch fixtures")
        sys.exit(1)
    for fname in needed:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=fname, repo_type="model")
        shutil.copy2(path, out_dir / fname)
    return out_dir


# ---------------------------------------------------------------------------
# Rust dump driver.
# ---------------------------------------------------------------------------


def run_rust_dump(fixture_dir: Path, dump_dir: Path) -> int:
    cmd: list[str] = [
        "cargo",
        "run",
        "-p",
        "ferrotorch-jit",
        "--release",
        "--example",
        "jit_trace_dump",
        "--",
        "--model",
        "jit-trace-parity-v1",
        "--fixture-dir",
        str(fixture_dir),
        "--output-dir",
        str(dump_dir),
    ]
    print(f"[verify] running rust dump: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return res.returncode


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrotorch_jit_verify_") as tmp:
        tmp_dir = Path(tmp)
        fixture_dir = ensure_fixtures(tmp_dir)
        dump_dir = tmp_dir / "rust_dump"
        dump_dir.mkdir(parents=True, exist_ok=True)
        rc = run_rust_dump(fixture_dir, dump_dir)
        if rc != 0:
            print(f"[verify] rust dump exited with status {rc}")
            return 1

        ref_shape, ref = read_f32_tensor(fixture_dir / "_value_parity_output.bin")
        print(f"[verify] torch reference output shape={ref_shape}")
        print(
            f"[verify] torch reference row 0 = "
            f"{ref.ravel()[: min(ref.size, 4)].tolist()}"
        )

        results: list[bool] = []
        results.append(
            evaluate(
                "eager",
                dump_dir / "eager.bin",
                ref,
                EAGER_MAX_ABS,
                EAGER_COS_MIN,
            )
        )
        # traced/compiled compare against ferrotorch's own eager output, which
        # is the load-bearing semantic (the spec frames bit-tightness for the
        # graph walk specifically).
        _, eager_arr = read_f32_tensor(dump_dir / "eager.bin")
        results.append(
            evaluate(
                "traced",
                dump_dir / "traced.bin",
                eager_arr,
                TRACED_MAX_ABS,
                TRACED_COS_MIN,
            )
        )
        results.append(
            evaluate(
                "compiled",
                dump_dir / "compiled.bin",
                eager_arr,
                COMPILED_MAX_ABS,
                COMPILED_COS_MIN,
            )
        )

        passed = all(results)
        verdict = "PASS" if passed else "FAIL"
        print(f"\n[verify] OVERALL: {verdict}")
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
