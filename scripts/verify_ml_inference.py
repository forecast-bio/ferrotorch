#!/usr/bin/env python3
"""Verify ferrotorch-ml's sklearn-equivalent ops against scikit-learn,
using the deterministic fixtures pinned at
`ferrotorch/ml-sklearn-parity-v1`.

Phase D.3 of real-artifact-driven development (#1159). Companion to:
  * `scripts/pin_pretrained_ml_fixtures.py` (the pin)
  * `ferrotorch-ml/examples/ml_op_dump.rs`
  * `ferrotorch-ml/tests/conformance_sklearn_parity.rs`

For each of the 5 configs this script:

  1. Downloads the per-config subfolder
     (`<config>/meta.json` + per-config input/output files) from the
     HF mirror via `huggingface_hub.hf_hub_download`.
  2. Invokes the matching Rust example:
       `cargo run -p ferrotorch-ml --release --example ml_op_dump --
        --config <name> --input-<key> <path> --output <tmp>`
  3. Reads the Rust-side output (.bin for numeric configs; stdout JSON
     for index-only configs) and compares it to the reference using the
     per-config tolerance:

     * pca_n4                   — cosine_sim ≥ 0.9999 PER PRINCIPAL
                                  COMPONENT; max_abs ≤ 1e-5 after
                                  per-PC sign alignment.
     * standard_scaler          — max_abs ≤ 1e-6.
     * one_hot_encoder          — exact integer equality.
     * kfold_5                  — SET-equality: each test fold has size
                                  10; union of all test folds == [0, 50).
     * train_test_split_80_20   — SET-equality: train/test sizes are
                                  80/20; train ∪ test == [0, 100);
                                  no overlap; rust's label vectors
                                  match the test/train X rows.

Usage:
  python3 scripts/verify_ml_inference.py \
      [--configs pca_n4,standard_scaler,...]
      [--quiet]

The Rust example is pre-built by this script on first invocation.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("/tmp/ferrotorch_verify_ml")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_REPO_ID = "ferrotorch/ml-sklearn-parity-v1"


# ---------------------------------------------------------------------------
# Config table — mirror of pin_pretrained_ml_fixtures.py.
# ---------------------------------------------------------------------------


@dataclass
class ConfigSpec:
    name: str
    equality_mode: str  # "COSINE_SIM_PER_PC" | "MAX_ABS" | "EXACT" | "SET"
    # Files the harness needs from the mirror (per-config inputs/outputs
    # under `<config>/`). Always includes meta.json — the verifier loads
    # tolerance fields out of that for forwards compat.
    mirror_files: tuple[str, ...]
    # Inputs passed to the rust example as --input-<key> <path>.
    rust_inputs: tuple[tuple[str, str], ...]  # (key, mirror filename)
    # If True, rust writes a .bin to --output; if False, the verdict
    # comes purely from stdout JSON (kfold_5, train_test_split).
    has_output_bin: bool


CONFIGS: list[ConfigSpec] = [
    ConfigSpec(
        name="pca_n4",
        equality_mode="COSINE_SIM_PER_PC",
        mirror_files=("meta.json", "input_X.bin", "output_Y.bin"),
        rust_inputs=(("X", "input_X.bin"),),
        has_output_bin=True,
    ),
    ConfigSpec(
        name="standard_scaler",
        equality_mode="MAX_ABS",
        mirror_files=("meta.json", "input_X.bin", "output_Y.bin"),
        rust_inputs=(("X", "input_X.bin"),),
        has_output_bin=True,
    ),
    ConfigSpec(
        name="one_hot_encoder",
        equality_mode="EXACT",
        mirror_files=("meta.json", "input_X_indices.bin", "output_Y.bin"),
        rust_inputs=(("X-indices", "input_X_indices.bin"),),
        has_output_bin=True,
    ),
    ConfigSpec(
        name="kfold_5",
        equality_mode="SET",
        mirror_files=("meta.json", "fold_indices.json"),
        rust_inputs=(),
        has_output_bin=False,
    ),
    ConfigSpec(
        name="train_test_split_80_20",
        equality_mode="SET",
        mirror_files=(
            "meta.json", "input_X.bin", "input_y.bin", "split_indices.json",
        ),
        rust_inputs=(("X", "input_X.bin"), ("y", "input_y.bin")),
        has_output_bin=False,
    ),
]


# ---------------------------------------------------------------------------
# Multi-tensor binary format.
# ---------------------------------------------------------------------------


def read_multi_tensor_f32(path: Path) -> list[np.ndarray]:
    raw = path.read_bytes()
    off = 0
    if len(raw) < 4:
        raise ValueError(f"{path}: file too short for num_tensors header")
    (n,) = struct.unpack_from("<I", raw, off)
    off += 4
    out: list[np.ndarray] = []
    for ti in range(n):
        (ndim,) = struct.unpack_from("<I", raw, off)
        off += 4
        shape = struct.unpack_from(f"<{ndim}I", raw, off)
        off += 4 * ndim
        numel = 1
        for s in shape:
            numel *= int(s)
        arr = np.frombuffer(raw, dtype="<f4", count=numel, offset=off).reshape(shape)
        off += 4 * numel
        out.append(arr.astype(np.float32, copy=True))
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes after {n} tensors")
    return out


# ---------------------------------------------------------------------------
# Fixture download.
# ---------------------------------------------------------------------------


def fetch_fixture(spec: ConfigSpec) -> dict[str, Path]:
    """Download every file the verifier needs into the HF cache and return
    a map of filename → local path."""
    files: dict[str, Path] = {}
    for fn in spec.mirror_files:
        local = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{spec.name}/{fn}")
        files[fn] = Path(local)
    return files


# ---------------------------------------------------------------------------
# Cargo example dispatch.
# ---------------------------------------------------------------------------


def build_rust_example_once() -> None:
    cmd = [
        "cargo", "build", "-p", "ferrotorch-ml", "--release",
        "--example", "ml_op_dump",
    ]
    print(f"  building Rust example once: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"cargo build failed ({proc.returncode})")


def run_rust_dump(
    spec: ConfigSpec,
    fixture_files: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    cmd = [
        "cargo", "run", "-q", "-p", "ferrotorch-ml", "--release",
        "--example", "ml_op_dump", "--",
        "--config", spec.name,
    ]
    for key, fn in spec.rust_inputs:
        cmd += [f"--input-{key}", str(fixture_files[fn])]
    if spec.has_output_bin:
        cmd += ["--output", str(output_path)]

    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"rust dump failed for {spec.name} ({proc.returncode})")

    json_line: str | None = None
    for line in proc.stdout.splitlines():
        t = line.strip()
        if t.startswith("{") and t.endswith("}"):
            json_line = t
    if json_line is None:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"{spec.name}: rust dump did not print a JSON verdict")
    return json.loads(json_line)


# ---------------------------------------------------------------------------
# Per-config comparators.
# ---------------------------------------------------------------------------


@dataclass
class ConfigVerdict:
    name: str
    equality_mode: str
    passed: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


def _verify_pca_n4(spec: ConfigSpec, files: dict[str, Path], rust_bin: Path) -> ConfigVerdict:
    ref_tensors = read_multi_tensor_f32(files["output_Y.bin"])
    rust_tensors = read_multi_tensor_f32(rust_bin)
    if len(ref_tensors) != 1 or len(rust_tensors) != 1:
        return ConfigVerdict(
            name=spec.name, equality_mode=spec.equality_mode, passed=False,
            summary=f"expected 1 tensor each, got ref={len(ref_tensors)} rust={len(rust_tensors)}",
        )
    ref = ref_tensors[0]
    rust = rust_tensors[0]
    if ref.shape != rust.shape or ref.shape != (100, 4):
        return ConfigVerdict(
            name=spec.name, equality_mode=spec.equality_mode, passed=False,
            summary=f"shape mismatch ref={ref.shape} rust={rust.shape}",
        )

    # Per-PC cosine similarity (signs can flip independently per PC).
    cos_per_pc: list[float] = []
    max_abs_aligned = 0.0
    for k in range(4):
        a = ref[:, k].astype(np.float64)
        b = rust[:, k].astype(np.float64)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            cos_per_pc.append(0.0)
            continue
        cos = float(np.dot(a, b) / (na * nb))
        cos_per_pc.append(cos)
        # Align sign per PC for max_abs comparison.
        sign = 1.0 if cos >= 0.0 else -1.0
        diff = a - sign * b
        max_abs_aligned = max(max_abs_aligned, float(np.abs(diff).max()))

    cos_min = min(abs(c) for c in cos_per_pc)
    cos_min_floor = 0.9999
    max_abs_cap = 1e-5

    failures: list[str] = []
    if cos_min < cos_min_floor:
        failures.append(
            f"min |cos_per_pc|={cos_min:.6f} < {cos_min_floor} "
            f"(per-PC cos={[f'{c:.6f}' for c in cos_per_pc]})"
        )
    if max_abs_aligned > max_abs_cap:
        failures.append(
            f"max_abs_after_sign_align={max_abs_aligned:.3e} > {max_abs_cap:.0e}"
        )

    summary = (
        f"per-PC |cos|={[f'{abs(c):.6f}' for c in cos_per_pc]}, "
        f"max_abs_aligned={max_abs_aligned:.3e}"
    )
    if failures:
        summary += " — FAIL: " + "; ".join(failures)
    return ConfigVerdict(
        name=spec.name, equality_mode=spec.equality_mode,
        passed=not failures, summary=summary,
        detail={
            "cos_per_pc": cos_per_pc,
            "max_abs_aligned": max_abs_aligned,
            "shape": list(rust.shape),
        },
    )


def _verify_standard_scaler(spec: ConfigSpec, files: dict[str, Path], rust_bin: Path) -> ConfigVerdict:
    ref = read_multi_tensor_f32(files["output_Y.bin"])[0]
    rust = read_multi_tensor_f32(rust_bin)[0]
    if ref.shape != rust.shape or ref.shape != (100, 10):
        return ConfigVerdict(
            name=spec.name, equality_mode=spec.equality_mode, passed=False,
            summary=f"shape mismatch ref={ref.shape} rust={rust.shape}",
        )
    diff = np.abs(rust.astype(np.float64) - ref.astype(np.float64))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    cap = 1e-6
    passed = max_abs <= cap
    summary = f"max_abs={max_abs:.3e} (cap {cap:.0e}), mean_abs={mean_abs:.3e}"
    if not passed:
        summary += f" — FAIL: max_abs > {cap:.0e}"
    return ConfigVerdict(
        name=spec.name, equality_mode=spec.equality_mode,
        passed=passed, summary=summary,
        detail={"max_abs": max_abs, "mean_abs": mean_abs, "shape": list(rust.shape)},
    )


def _verify_one_hot(spec: ConfigSpec, files: dict[str, Path], rust_bin: Path) -> ConfigVerdict:
    ref = read_multi_tensor_f32(files["output_Y.bin"])[0]
    rust = read_multi_tensor_f32(rust_bin)[0]
    if ref.shape != rust.shape or ref.shape != (5, 3):
        return ConfigVerdict(
            name=spec.name, equality_mode=spec.equality_mode, passed=False,
            summary=f"shape mismatch ref={ref.shape} rust={rust.shape}",
        )
    # Both sides are exactly 0.0 / 1.0; exact equality is required.
    diff = np.abs(rust - ref)
    max_abs = float(diff.max())
    passed = max_abs == 0.0
    summary = f"max_abs={max_abs} (exact integer required)"
    if not passed:
        summary += " — FAIL: ferrotorch one-hot does not exactly match sklearn"
    return ConfigVerdict(
        name=spec.name, equality_mode=spec.equality_mode,
        passed=passed, summary=summary,
        detail={"max_abs": max_abs, "shape": list(rust.shape)},
    )


def _verify_kfold(spec: ConfigSpec, files: dict[str, Path], rust_verdict: dict[str, Any]) -> ConfigVerdict:
    ref = json.loads(files["fold_indices.json"].read_text())
    expected_size = int(ref["n_samples"]) // int(ref["n_splits"])
    n_samples = int(ref["n_samples"])
    rust_folds = rust_verdict.get("folds", [])
    failures: list[str] = []

    if len(rust_folds) != int(ref["n_splits"]):
        failures.append(f"n_folds rust={len(rust_folds)} != ref={int(ref['n_splits'])}")
        return ConfigVerdict(
            name=spec.name, equality_mode=spec.equality_mode, passed=False,
            summary="; ".join(failures),
        )

    # SET-equality invariants (per Phase D.3 spec).
    test_union: set[int] = set()
    train_union: set[int] = set()
    for k, fold in enumerate(rust_folds):
        test_set = set(int(v) for v in fold["test"])
        train_set = set(int(v) for v in fold["train"])
        if len(test_set) != expected_size:
            failures.append(
                f"fold {k}: |test|={len(test_set)} != {expected_size}"
            )
        if len(test_set) != len(fold["test"]):
            failures.append(f"fold {k}: duplicates within test")
        if test_set & train_set:
            failures.append(
                f"fold {k}: train ∩ test = {sorted(test_set & train_set)[:5]} (not disjoint)"
            )
        if test_set | train_set != set(range(n_samples)):
            failures.append(
                f"fold {k}: train ∪ test != [0, {n_samples})"
            )
        # Test sets must be pairwise disjoint across folds (covering each
        # index exactly once).
        if test_set & test_union:
            failures.append(
                f"fold {k}: test overlaps prior fold(s) on {sorted(test_set & test_union)[:5]}"
            )
        test_union |= test_set
        train_union |= train_set

    if test_union != set(range(n_samples)):
        failures.append(
            f"union of all test sets != [0, {n_samples}); missing="
            f"{sorted(set(range(n_samples)) - test_union)[:5]}"
        )

    summary = f"5 folds × size={expected_size}, test-union covers [0, {n_samples})"
    if failures:
        summary += " — FAIL: " + "; ".join(failures[:3])
    return ConfigVerdict(
        name=spec.name, equality_mode=spec.equality_mode,
        passed=not failures, summary=summary,
        detail={"num_folds": len(rust_folds), "expected_size": expected_size,
                "n_samples": n_samples,
                "failures": failures[:20]},
    )


def _verify_train_test_split(
    spec: ConfigSpec, files: dict[str, Path], rust_verdict: dict[str, Any],
) -> ConfigVerdict:
    ref = json.loads(files["split_indices.json"].read_text())
    n_samples = int(ref["n_samples"])
    n_train_expected = int(ref["n_train"])
    n_test_expected = int(ref["n_test"])

    rust_train = [int(v) for v in rust_verdict.get("train_indices", [])]
    rust_test = [int(v) for v in rust_verdict.get("test_indices", [])]
    failures: list[str] = []

    if len(rust_train) != n_train_expected:
        failures.append(f"|train|={len(rust_train)} != {n_train_expected}")
    if len(rust_test) != n_test_expected:
        failures.append(f"|test|={len(rust_test)} != {n_test_expected}")

    train_set = set(rust_train)
    test_set = set(rust_test)
    if len(train_set) != len(rust_train):
        failures.append("duplicates within train_indices")
    if len(test_set) != len(rust_test):
        failures.append("duplicates within test_indices")
    if train_set & test_set:
        failures.append(
            f"train ∩ test = {sorted(train_set & test_set)[:5]} (not disjoint)"
        )
    if train_set | test_set != set(range(n_samples)):
        missing = sorted(set(range(n_samples)) - (train_set | test_set))
        failures.append(
            f"train ∪ test != [0, {n_samples}); missing={missing[:5]}"
        )

    # Note: label consistency is asserted INSIDE the rust example
    # (`y_test[k] == y[test_indices[k]]`) — if the rust binary returned a
    # verdict at all then that invariant held. We re-check the count
    # invariants here for belt-and-suspenders.

    summary = (
        f"train={len(rust_train)}/test={len(rust_test)}, "
        f"union covers [0, {n_samples})"
    )
    if failures:
        summary += " — FAIL: " + "; ".join(failures[:3])
    return ConfigVerdict(
        name=spec.name, equality_mode=spec.equality_mode,
        passed=not failures, summary=summary,
        detail={
            "n_train": len(rust_train),
            "n_test": len(rust_test),
            "n_samples": n_samples,
            "failures": failures[:20],
        },
    )


VERIFIERS = {
    "pca_n4": _verify_pca_n4,
    "standard_scaler": _verify_standard_scaler,
    "one_hot_encoder": _verify_one_hot,
}


def verify_one(spec: ConfigSpec, quiet: bool) -> ConfigVerdict:
    print(f"\n=== {spec.name} (equality={spec.equality_mode}) ===", flush=True)

    # -- 1. Fetch fixture. -------------------------------------------------
    files = fetch_fixture(spec)
    print(f"  fixture: {sorted(p.name for p in files.values())}")

    # -- 2. Run ferrotorch. -----------------------------------------------
    if spec.has_output_bin:
        rust_bin = CACHE_DIR / f"{spec.name}_rust.bin"
        if rust_bin.exists():
            rust_bin.unlink()
    else:
        rust_bin = CACHE_DIR / f"{spec.name}_unused.bin"

    rust_verdict = run_rust_dump(spec, files, rust_bin)
    if not quiet:
        print(f"  rust verdict: {rust_verdict}")

    # -- 3. Per-config comparison. ----------------------------------------
    if spec.name == "kfold_5":
        v = _verify_kfold(spec, files, rust_verdict)
    elif spec.name == "train_test_split_80_20":
        v = _verify_train_test_split(spec, files, rust_verdict)
    else:
        v = VERIFIERS[spec.name](spec, files, rust_bin)

    if not quiet:
        print(f"  {v.summary}")
    return v


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs", default=",".join(c.name for c in CONFIGS),
        help="Comma-separated subset of config names to verify.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final per-config verdict line.")
    args = p.parse_args()

    by_name = {c.name: c for c in CONFIGS}
    requested = [n.strip() for n in args.configs.split(",") if n.strip()]
    for r in requested:
        if r not in by_name:
            print(f"unknown config {r!r}. Known: {list(by_name)}", file=sys.stderr)
            return 2

    build_rust_example_once()

    verdicts: list[ConfigVerdict] = []
    for name in requested:
        spec = by_name[name]
        try:
            v = verify_one(spec, quiet=args.quiet)
        except Exception as e:  # noqa: BLE001
            v = ConfigVerdict(
                name=name, equality_mode=spec.equality_mode,
                passed=False, summary=f"exception: {e!r}",
                detail={"exception": repr(e)},
            )
        verdicts.append(v)

    print("\n=== VERDICTS ===")
    any_fail = False
    for v in verdicts:
        tag = "PASS" if v.passed else "FAIL"
        if not v.passed:
            any_fail = True
        print(f"{v.name}: {tag} — {v.summary}")

    report = {
        v.name: {
            "equality_mode": v.equality_mode,
            "passed": v.passed,
            "summary": v.summary,
            "detail": v.detail,
        }
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_ml_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
