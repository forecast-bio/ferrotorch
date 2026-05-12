#!/usr/bin/env python3
"""Verify `ferrotorch-serialize`'s four format loaders/exporters
(`.pth` / SafeTensors / GGUF / ONNX) against the canonical references
pinned at `ferrotorch/serialize-parity-v1`.

Phase G.3 of real-artifact-driven development (#1169). Companion to:
  * `scripts/pin_pretrained_serialize_fixtures.py` (the pin)
  * `ferrotorch-serialize/examples/serialize_parity_dump.rs` (rust dump)
  * `ferrotorch-serialize/tests/conformance_format_parity.rs` (cargo gate)

For each target the harness:

  1. Downloads (or reads from `--local-fixture-root`) the per-target
     fixture folder from the HF mirror via `huggingface_hub.hf_hub_download`.
  2. Runs the rust dump example (`serialize_parity_dump --target <T>`).
  3. Compares rust dump against the python reference per the per-target
     tolerance, which is a HARD floor.

Tolerances (hard floors — loosening forbidden by dispatch):
  * pth_load               : max_abs == 0     (byte-exact)
  * safetensors_round_trip : max_abs == 0     (byte-exact)
  * gguf_load              : max_abs <= 1e-4  (Q8_0 dequant noise floor)
  * onnx_export            : max_abs <= 1e-5  AND cosine_sim >= 0.9999
                             between (rust-emitted ONNX run via
                             onnxruntime) and (ferrotorch's own forward).

Usage:
  python3 scripts/verify_serialize_inference.py [--targets pth,safetensors,...]
                                                [--local-fixture-root <dir>]
                                                [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = Path("/tmp/ferrotorch_verify_serialize")
HF_REPO_ID = "ferrotorch/serialize-parity-v1"

TOL_GGUF_MAX_ABS = 1e-4
TOL_ONNX_MAX_ABS = 1e-5
TOL_ONNX_COS_SIM = 0.9999

TARGETS = ["pth", "safetensors", "gguf", "onnx"]

# Per-target fixture subfolder names on the HF mirror.
SUBDIR = {
    "pth": "resnet18-pth",
    "safetensors": "safetensors-rt",
    "gguf": "gguf",
    "onnx": "onnx-mlp",
}


def read_bin(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    """Read `[u32 ndim][u32 shape...][f32 bytes]` little-endian file."""
    with path.open("rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
        numel = int(np.prod(shape)) if shape else 1
        buf = f.read(numel * 4)
        if len(buf) != numel * 4:
            raise RuntimeError(
                f"truncated {path}: got {len(buf)} bytes, expected {numel * 4}"
            )
        data = np.frombuffer(buf, dtype="<f4").copy()
    return shape, (data.reshape(shape) if shape else data)


def fetch_target(target: str, local_root: Path | None) -> Path:
    """Materialize one target's fixture folder, either from a local
    staging dir or the HF mirror, into a clean stage dir."""
    sub = SUBDIR[target]
    stage = CACHE_ROOT / sub
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    if local_root is not None:
        src = local_root / sub
        if not src.is_dir():
            raise RuntimeError(f"missing local fixture dir: {src}")
        # Copy everything in src into stage (preserve subdirectories).
        for p in src.rglob("*"):
            rel = p.relative_to(src)
            dst = stage / rel
            if p.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
        return stage
    # HF download.
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f for f in list_repo_files(HF_REPO_ID) if f.startswith(sub + "/")
    ]
    if not files:
        raise RuntimeError(f"no files for target {target!r} (subdir {sub!r}) in {HF_REPO_ID}")
    for f in files:
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=f)
        rel = f[len(sub) + 1 :]
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        shutil.copyfile(cached, dst)
    return stage


def run_rust_dump(target: str, fixture_dir: Path, output_dir: Path, quiet: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo", "run", "--quiet", "--release",
        "-p", "ferrotorch-serialize",
        "--example", "serialize_parity_dump",
        "--",
        "--target", target,
        "--fixture-dir", str(fixture_dir),
        "--output-dir", str(output_dir),
    ]
    if not quiet:
        print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  STDOUT: {r.stdout}\n  STDERR: {r.stderr}", flush=True)
        raise RuntimeError(f"rust dump for {target!r} failed (exit {r.returncode})")
    if not quiet:
        # Only show last 6 lines of dump stdout
        for ln in r.stdout.strip().split("\n")[-6:]:
            print(f"    {ln}", flush=True)


# ---------------------------------------------------------------------------
# Target verifiers
# ---------------------------------------------------------------------------


def verify_pth(fixture_dir: Path, rust_dir: Path) -> tuple[bool, str]:
    ref_dir = fixture_dir / "reference_state_dict"
    rust_files = sorted(rust_dir.glob("*.bin"))
    ref_files = sorted(ref_dir.glob("*.bin"))
    if len(rust_files) != len(ref_files):
        return False, f"tensor count mismatch: rust={len(rust_files)} ref={len(ref_files)}"

    rust_names = {p.name for p in rust_files}
    ref_names = {p.name for p in ref_files}
    if rust_names != ref_names:
        only_rust = sorted(rust_names - ref_names)[:5]
        only_ref = sorted(ref_names - rust_names)[:5]
        return False, (
            f"key sets differ — only_rust={only_rust} only_ref={only_ref}"
        )

    worst_diff = 0.0
    worst_name = ""
    for p in rust_files:
        rs, rd = read_bin(p)
        fs, fd = read_bin(ref_dir / p.name)
        if rs != fs:
            return False, f"shape mismatch on {p.name}: rust={rs} ref={fs}"
        diff = float(np.abs(rd - fd).max()) if rd.size else 0.0
        if diff > worst_diff:
            worst_diff = diff
            worst_name = p.name
    if worst_diff > 0.0:
        return False, f"non-zero max_abs={worst_diff:.6e} on {worst_name} (expected 0.0)"
    return True, f"byte-exact across {len(rust_files)} tensors"


def verify_safetensors(fixture_dir: Path, rust_dir: Path) -> tuple[bool, str]:
    # Same byte-exact contract as .pth.
    return verify_pth(fixture_dir, rust_dir)


def verify_gguf(fixture_dir: Path, rust_dir: Path) -> tuple[bool, str]:
    ref_dir = fixture_dir / "reference_dequant"
    names_path = fixture_dir / "sampled_tensor_names.json"
    names: list[str] = json.loads(names_path.read_text())
    worst_diff = 0.0
    worst_name = ""
    n_compared = 0
    for name in names:
        rp = rust_dir / f"{name}.bin"
        fp = ref_dir / f"{name}.bin"
        if not rp.exists():
            return False, f"missing rust output {rp}"
        if not fp.exists():
            return False, f"missing reference {fp}"
        rs, rd = read_bin(rp)
        fs, fd = read_bin(fp)
        if rs != fs:
            return False, f"shape mismatch on {name}: rust={rs} ref={fs}"
        diff = float(np.abs(rd - fd).max()) if rd.size else 0.0
        if diff > worst_diff:
            worst_diff = diff
            worst_name = name
        n_compared += 1
    if worst_diff > TOL_GGUF_MAX_ABS:
        return False, (
            f"max_abs={worst_diff:.6e} on {worst_name} exceeds floor {TOL_GGUF_MAX_ABS:.0e}"
        )
    return True, (
        f"max_abs={worst_diff:.6e} on {worst_name or '(equal)'} "
        f"<= {TOL_GGUF_MAX_ABS:.0e} across {n_compared} tensors"
    )


def verify_onnx(fixture_dir: Path, rust_dir: Path) -> tuple[bool, str]:
    import onnxruntime as ort  # lazy import — only needed for this target

    onnx_path = rust_dir / "mlp.onnx"
    if not onnx_path.exists():
        return False, f"rust did not emit {onnx_path}"

    # Load the rust-emitted ONNX in Python, run inference on each
    # fixed input, and compare against both ferrotorch's own forward
    # (rust dump output) AND torch's reference (python pin output).
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    cases = ["zeros", "ones", "random"]
    overall_max_abs = 0.0
    overall_min_cos = 1.0
    worst_case = ""
    for case in cases:
        _, x = read_bin(fixture_dir / f"input_{case}.bin")
        _, torch_y = read_bin(fixture_dir / f"torch_forward_{case}.bin")
        _, ft_y = read_bin(rust_dir / f"ferrotorch_forward_{case}.bin")
        ort_y = sess.run(None, {input_name: x.astype(np.float32)})[0]

        # The "rust-emitted ONNX (run via onnxruntime) vs ferrotorch's
        # own forward" is the primary contract — the ONNX exporter is
        # what we're verifying. The torch reference provides a third
        # corroborating point but is not the gating comparison.
        d_ort_ft = float(np.abs(ort_y.flatten() - ft_y.flatten()).max())
        d_ort_torch = float(np.abs(ort_y.flatten() - torch_y.flatten()).max())
        cos = float(
            np.dot(ort_y.flatten(), ft_y.flatten())
            / (np.linalg.norm(ort_y) * np.linalg.norm(ft_y) + 1e-30)
        )
        primary_diff = max(d_ort_ft, d_ort_torch)
        if primary_diff > overall_max_abs:
            overall_max_abs = primary_diff
            worst_case = case
        overall_min_cos = min(overall_min_cos, cos)

    if overall_max_abs > TOL_ONNX_MAX_ABS:
        return False, (
            f"max_abs={overall_max_abs:.6e} on {worst_case} exceeds floor "
            f"{TOL_ONNX_MAX_ABS:.0e}"
        )
    if overall_min_cos < TOL_ONNX_COS_SIM:
        return False, (
            f"cosine_sim={overall_min_cos:.6f} on {worst_case} below floor "
            f"{TOL_ONNX_COS_SIM:.4f}"
        )
    return True, (
        f"max_abs={overall_max_abs:.6e} <= {TOL_ONNX_MAX_ABS:.0e}, "
        f"cos_sim={overall_min_cos:.6f} >= {TOL_ONNX_COS_SIM:.4f} across {len(cases)} inputs"
    )


VERIFIERS = {
    "pth": ("pth_load", verify_pth),
    "safetensors": ("safetensors_round_trip", verify_safetensors),
    "gguf": ("gguf_load", verify_gguf),
    "onnx": ("onnx_export", verify_onnx),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--targets", default=",".join(TARGETS),
        help=f"Comma-separated targets (default: {','.join(TARGETS)})",
    )
    p.add_argument(
        "--local-fixture-root", default=None,
        help="Read fixtures from a local staging dir instead of HF.",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    local_root = Path(args.local_fixture_root) if args.local_fixture_root else None

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    rust_out_root = CACHE_ROOT / "rust_out"
    rust_out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== ferrotorch-serialize format-parity verify (#1169) ===")
    print(f"  targets:           {targets}")
    print(f"  local_fixture_root:{local_root}")
    print(f"  cache_root:        {CACHE_ROOT}")

    results: dict[str, tuple[bool, str]] = {}
    for target in targets:
        if target not in VERIFIERS:
            print(f"  unknown target {target!r}", flush=True)
            results[target] = (False, "unknown target")
            continue
        label, verifier = VERIFIERS[target]
        print(f"\n--- {label} ({target}) ---", flush=True)
        try:
            fixture_dir = fetch_target(target, local_root)
            print(f"  fixture: {fixture_dir}", flush=True)
            rust_dir = rust_out_root / target
            run_rust_dump(target, fixture_dir, rust_dir, quiet=args.quiet)
            ok, msg = verifier(fixture_dir, rust_dir)
            results[target] = (ok, msg)
            verdict = "PASS" if ok else "FAIL"
            print(f"  [{verdict}] {label}: {msg}", flush=True)
        except Exception as e:
            results[target] = (False, f"exception: {type(e).__name__}: {e}")
            print(f"  [FAIL] {label}: exception: {type(e).__name__}: {e}", flush=True)

    print("\n=== SUMMARY ===")
    all_pass = True
    for target, (ok, msg) in results.items():
        label, _ = VERIFIERS.get(target, (target, None))
        verdict = "PASS" if ok else "FAIL"
        print(f"  {label}: {verdict} — {msg}")
        if not ok:
            all_pass = False
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
