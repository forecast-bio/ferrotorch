#!/usr/bin/env python3
"""Verify ferrotorch pretrained audio-encoder inference against the
transformers reference.

Companion to `scripts/verify_text_embedding_inference.py` but for
Whisper-family audio encoders. For each pinned encoder in the
`ferrotorch/*` HF org this script:

  1. Downloads the mirror's `config.json`, `model.safetensors`, and the
     frozen parity probe (`_value_parity_{audio,mel,encoder_output}.bin`)
     via `huggingface_hub`.
  2. Invokes the Rust binary
     (`cargo run -p ferrotorch-whisper --release --example whisper_encoder_dump`)
     pointing `--mel` at the frozen mel spectrogram and `--model` at the
     ferrotorch repo (the example fetches `config.json` +
     `model.safetensors` through `ferrotorch-hub`).
  3. Reads the dumped `[1, 1500, 384]` f32 encoder output and compares
     it elementwise against `_value_parity_encoder_output.bin`.
  4. Computes:
       - `cosine_sim` — `(rust @ tv) / (||rust|| * ||tv||)`
       - `max_abs`    — `max(abs(rust - tv))`
     and compares each against the per-model tolerance in `TOL`.
  5. Prints a one-line verdict per model and a JSON report.

The PASS bar is `cosine_sim >= 0.999, max_abs <= 0.5`. This is the same
floor the causal-LM harness uses — the f32 accumulation noise between
ferrotorch and transformers running the same byte-for-byte weights sits
well below 0.5 in encoder-output scale.

Usage:
  python3 scripts/verify_audio_encoder_inference.py [--models whisper-tiny-encoder,...]
                                                    [--quiet]
                                                    [--self-test]

The Rust example must be pre-built (this script will also build it on
first invocation):
  cargo build -p ferrotorch-whisper --release --example whisper_encoder_dump
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
CACHE_DIR = Path("/tmp/ferrotorch_verify_audio_encoder")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Per-model tolerances. Same floor as the causal-LM harness; the only
# divergence between ferrotorch and transformers running the same
# byte-for-byte weights is f32 accumulation noise from a different
# op-order in the attention + FFN stack.
TOL: dict[str, dict[str, Any]] = {
    "whisper-tiny-encoder": dict(
        cosine_sim_min=0.999,
        max_abs=0.5,
        encoder_shape=(1, 1500, 384),
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_dump_f32(path: Path) -> np.ndarray:
    """Read `[u32 ndim][u32 × ndim shape][f32 × prod(shape)]` little-endian."""
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"dump {path} truncated (< 4 bytes)")
    (ndim,) = struct.unpack_from("<I", raw, 0)
    off = 4
    if len(raw) < off + 4 * ndim:
        raise ValueError(
            f"dump {path}: header claims ndim={ndim} but only {len(raw)} bytes total"
        )
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = 1
    for s in shape:
        n *= int(s)
    expect = off + 4 * n
    if len(raw) != expect:
        raise ValueError(
            f"dump {path}: header claims shape={shape} (expects {expect} bytes) "
            f"but file is {len(raw)} bytes"
        )
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def fetch_mirror(model_name: str) -> dict[str, Path]:
    """Download the mirror's files via huggingface_hub. Returns a map of
    filename → local path."""
    repo_id = f"ferrotorch/{model_name}"
    out: dict[str, Path] = {}
    for fn in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "_value_parity_audio.bin",
        "_value_parity_mel.bin",
        "_value_parity_encoder_output.bin",
    ):
        try:
            local = hf_hub_download(repo_id=repo_id, filename=fn)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"failed to download {fn} from {repo_id}: {e}")
        out[fn] = Path(local)
    return out


def run_rust_dump(model_name: str, mel_path: Path, output_bin: Path) -> dict[str, Any]:
    """Invoke the Rust example and parse its stdout JSON verdict line."""
    cmd = [
        "cargo", "run", "-p", "ferrotorch-whisper", "--release",
        "--example", "whisper_encoder_dump", "--",
        "--model", model_name,
        "--mel", str(mel_path),
        "--output", str(output_bin),
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"rust dump failed ({proc.returncode}); stderr above")
    json_line: str | None = None
    for line in proc.stdout.splitlines():
        t = line.strip()
        if t.startswith("{") and t.endswith("}"):
            json_line = t
    if json_line is None:
        sys.stderr.write(proc.stdout)
        raise RuntimeError("rust dump did not print a JSON verdict line")
    return json.loads(json_line)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

@dataclass
class ModelVerdict:
    name: str
    passed: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


def verify_one(name: str, quiet: bool) -> ModelVerdict:
    print(f"\n=== {name} ===", flush=True)
    tol = TOL[name]

    # -- 1. Fetch the mirror. ---------------------------------------------
    print(f"  downloading ferrotorch/{name} …", flush=True)
    files = fetch_mirror(name)
    print(f"  cached: {sorted(p.name for p in files.values())}")

    # -- 2. Read the reference encoder output. ----------------------------
    ref = read_dump_f32(files["_value_parity_encoder_output.bin"])
    expect_shape = tol["encoder_shape"]
    if ref.shape != expect_shape:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"reference shape {ref.shape} != expected {expect_shape}",
        )
    print(
        f"  reference encoder output: shape={list(ref.shape)} "
        f"||ref||={float(np.linalg.norm(ref)):.6f}",
        flush=True,
    )

    # -- 3. Run ferrotorch. -----------------------------------------------
    output_bin = CACHE_DIR / f"{name}_rust_dump.bin"
    verdict = run_rust_dump(name, files["_value_parity_mel.bin"], output_bin)
    rust = read_dump_f32(output_bin)
    if rust.shape != ref.shape:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"shape mismatch: rust={list(rust.shape)} vs ref={list(ref.shape)}",
        )

    # -- 4. Metrics. ------------------------------------------------------
    diff = rust - ref
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    rust_norm = float(np.linalg.norm(rust))
    cos = cosine_similarity(rust, ref)

    failures: list[str] = []
    if cos < tol["cosine_sim_min"]:
        failures.append(f"cosine_sim={cos:.6f} < {tol['cosine_sim_min']}")
    if max_abs > tol["max_abs"]:
        failures.append(f"max_abs={max_abs:.6f} > {tol['max_abs']}")

    passed = not failures
    summary = (
        f"cosine_sim={cos:.6f}, max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, "
        f"||rust||={rust_norm:.6f}"
    )
    if failures:
        summary += " — FAIL: " + "; ".join(failures)
    if not quiet:
        print(f"  metrics: {summary}")

    return ModelVerdict(
        name=name, passed=passed, summary=summary,
        detail=dict(
            shape=list(rust.shape),
            cosine_sim=cos,
            max_abs=max_abs,
            mean_abs=mean_abs,
            rust_norm=rust_norm,
            ref_norm=float(np.linalg.norm(ref)),
            rust_dump_verdict=verdict,
            failures=failures,
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", default=",".join(TOL.keys()),
        help="Comma-separated subset of model names to verify.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final per-model verdict line.")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in TOL:
            print(f"unknown model {m!r}. Known: {list(TOL)}", file=sys.stderr)
            return 2

    verdicts: list[ModelVerdict] = []
    for m in models:
        try:
            v = verify_one(m, quiet=args.quiet)
        except Exception as e:  # noqa: BLE001
            v = ModelVerdict(
                name=m, passed=False, summary=f"exception: {e!r}",
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
            "passed": v.passed,
            "summary": v.summary,
            "detail": v.detail,
        }
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_audio_encoder_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")
    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_read_dump_f32(tmp: Path) -> None:
    path = tmp / "_self_test_dump.bin"
    shape = (1, 4)
    data = np.arange(4, dtype="<f4").reshape(shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", d))
        f.write(data.tobytes(order="C"))
    got = read_dump_f32(path)
    assert got.shape == shape, (got.shape, shape)
    assert np.allclose(got, data), (got, data)
    print("_test_read_dump_f32: ok")


def _test_cosine() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9
    c = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, c)) < 1e-9
    d = -a
    assert abs(cosine_similarity(a, d) + 1.0) < 1e-9
    print("_test_cosine: ok")


def _self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _test_read_dump_f32(Path(td))
    _test_cosine()
    print("self-test: all assertions passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
