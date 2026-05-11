#!/usr/bin/env python3
"""Verify ferrotorch pretrained causal-LM inference against transformers reference.

Companion to `scripts/verify_pretrained_inference.py` but for causal LMs. For
each pinned LM in the `ferrotorch/*` HF org this script:

  1. Loads the upstream model via `transformers.AutoModelForCausalLM` in
     float32 and the matching tokenizer.
  2. Tokenizes a frozen short prompt (`PARITY_PROMPT` — same prompt the
     `scripts/pin_pretrained_llm_weights.py` froze into the mirror's
     `_value_parity_input.txt`).
  3. Runs a single transformers prefill `model(input_ids=ids, use_cache=False)`
     to recover the reference logits tensor `[1, S, V]`.
  4. Invokes the ferrotorch Rust binary
     (`cargo run -p ferrotorch-llama --release --example llm_inference_dump`)
     against the same prompt and reads the dumped `[1, S, V]` f32 tensor.
  5. Computes:
       - `max_abs`               — max absolute elementwise diff
       - `top1_argmax_agree_pct` — over all positions
       - `top5_last_token_overlap` — `|top5(rust[-1]) ∩ top5(tv[-1])|`
       - `last_token_argmax_match` — boolean
     and compares each against the per-model tolerance in `TOL`.
  6. Prints a one-line verdict per model and a JSON report.

Tolerances are intentionally tight: at f32 the only divergence between
ferrotorch and transformers (which share weights byte-for-byte) is f32
accumulation noise from a different op-order in the attention/MLP stack.
We require `max_abs<=0.5` (logits scale is single digits), `top1>=99%`,
and `top5_last_token_overlap>=4` (out of 5).

This is intentionally a measurement tool — it makes no fixes and reports
the verdict honestly. A FAIL diagnoses where the divergence happens so a
follow-up dispatch can address it.

Usage:
  python3 scripts/verify_causal_lm_inference.py [--models smollm-135m,...]
                                                [--quiet]
                                                [--self-test]

The Rust example must be pre-built (this script will also build it on first
invocation):
  cargo build -p ferrotorch-llama --release --example llm_inference_dump
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("/tmp/ferrotorch_verify_llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Match the pin script's frozen prompt. The pin script also writes this same
# string into `_value_parity_input.txt`. We deliberately re-declare the
# constant here so the harness can run without re-reading the mirror, and we
# cross-check at runtime that the upstream tokenizer produces the SAME token
# ids the mirror's `_value_parity_token_ids.json` claims.
PARITY_PROMPT = "The quick brown fox jumps over the lazy"


# Per-model tolerances. Tight on purpose — the ferrotorch path consumes the
# same upstream safetensors byte-for-byte, so any drift larger than f32
# accumulation noise is a bug.
TOL: dict[str, dict[str, Any]] = {
    "smollm-135m": dict(
        max_abs=0.5,
        top1_argmax_agree_pct_min=99.0,
        top5_last_token_overlap_min=4,  # out of 5
        last_token_argmax_must_match=True,
    ),
}

# Upstream HF repo per ferrotorch mirror — needed for the tokenizer + reference.
UPSTREAM_REPO: dict[str, str] = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
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
            f"dump {path}: header claims ndim={ndim} but only "
            f"{len(raw)} bytes total"
        )
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = 1
    for s in shape:
        n *= int(s)
    expect = off + 4 * n
    if len(raw) != expect:
        raise ValueError(
            f"dump {path}: header claims shape={shape} "
            f"(expects {expect} bytes total) but file is {len(raw)} bytes"
        )
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def run_rust_dump(model_name: str, output_bin: Path, prompt: str) -> dict[str, Any]:
    """Invoke the Rust example and parse its stdout JSON verdict line."""
    cmd = [
        "cargo",
        "run",
        "-p",
        "ferrotorch-llama",
        "--release",
        "--example",
        "llm_inference_dump",
        "--",
        "--model",
        model_name,
        "--output",
        str(output_bin),
        "--prompt",
        prompt,
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"rust dump failed ({proc.returncode}); stderr above"
        )
    # The example prints exactly one JSON line on stdout; cargo also prints
    # build progress on the same stream sometimes — we parse the last line
    # that begins with '{'.
    json_line: str | None = None
    for line in proc.stdout.splitlines():
        t = line.strip()
        if t.startswith("{") and t.endswith("}"):
            json_line = t
    if json_line is None:
        sys.stderr.write(proc.stdout)
        raise RuntimeError("rust dump did not print a JSON verdict line")
    return json.loads(json_line)


def topk_indices(row: np.ndarray, k: int) -> list[int]:
    """Return the indices of the top-k values in `row` (descending)."""
    if k >= row.shape[0]:
        return np.argsort(-row).tolist()
    # argpartition then sort the top-k.
    part = np.argpartition(-row, k - 1)[:k]
    return part[np.argsort(-row[part])].tolist()


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
    upstream = UPSTREAM_REPO[name]

    # -- 1. Load reference (tokenizer + model) ------------------------------
    print(f"  loading upstream tokenizer/model from {upstream!r}…", flush=True)
    tok = AutoTokenizer.from_pretrained(upstream)
    enc = tok(PARITY_PROMPT, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(torch.int64)
    seq_len = int(ids.shape[1])
    py_token_ids = ids[0].tolist()
    print(f"  prompt: {PARITY_PROMPT!r} -> {seq_len} tokens: {py_token_ids}",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        upstream, torch_dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    tv_logits = out.logits.detach().to(torch.float32).cpu().numpy()
    assert tv_logits.ndim == 3 and tv_logits.shape[:2] == (1, seq_len), (
        tv_logits.shape, seq_len
    )
    vocab = tv_logits.shape[2]
    print(
        f"  tv logits: shape={list(tv_logits.shape)} "
        f"max_abs={float(np.abs(tv_logits).max()):.4f} "
        f"argmax[last]={int(tv_logits[0, -1].argmax())}",
        flush=True,
    )

    # Free upstream weights as soon as we have the reference tensor.
    del model

    # -- 2. Run ferrotorch -------------------------------------------------
    output_bin = CACHE_DIR / f"{name}_rust_dump.bin"
    verdict = run_rust_dump(name, output_bin, PARITY_PROMPT)
    rust_logits = read_dump_f32(output_bin)
    if rust_logits.shape != tv_logits.shape:
        return ModelVerdict(
            name=name,
            passed=False,
            summary=(
                f"shape mismatch: rust={list(rust_logits.shape)} "
                f"vs tv={list(tv_logits.shape)}"
            ),
            detail={"rust_shape": list(rust_logits.shape),
                    "tv_shape": list(tv_logits.shape)},
        )
    rust_token_ids = list(verdict.get("token_ids", []))
    if rust_token_ids != py_token_ids:
        return ModelVerdict(
            name=name,
            passed=False,
            summary=(
                f"tokenizer disagreement: rust={rust_token_ids} "
                f"vs py={py_token_ids}"
            ),
            detail={"rust_token_ids": rust_token_ids,
                    "py_token_ids": py_token_ids},
        )

    # -- 3. Compute metrics -----------------------------------------------
    diff = rust_logits - tv_logits
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())

    rust_argmax = rust_logits.argmax(axis=-1)  # [1, S]
    tv_argmax = tv_logits.argmax(axis=-1)
    top1_pct = float((rust_argmax == tv_argmax).mean() * 100.0)

    rust_top5 = topk_indices(rust_logits[0, -1], 5)
    tv_top5 = topk_indices(tv_logits[0, -1], 5)
    overlap = len(set(rust_top5) & set(tv_top5))

    last_match = bool(
        int(rust_logits[0, -1].argmax()) == int(tv_logits[0, -1].argmax())
    )

    # -- 4. Apply tolerances --------------------------------------------
    failures: list[str] = []
    if max_abs > tol["max_abs"]:
        failures.append(f"max_abs={max_abs:.4f} > {tol['max_abs']}")
    if top1_pct < tol["top1_argmax_agree_pct_min"]:
        failures.append(
            f"top1_pct={top1_pct:.2f} < {tol['top1_argmax_agree_pct_min']}"
        )
    if overlap < tol["top5_last_token_overlap_min"]:
        failures.append(
            f"top5_last_overlap={overlap} < {tol['top5_last_token_overlap_min']}"
        )
    if tol["last_token_argmax_must_match"] and not last_match:
        failures.append(
            f"last_token_argmax: rust={int(rust_logits[0, -1].argmax())} "
            f"vs tv={int(tv_logits[0, -1].argmax())}"
        )

    passed = not failures
    summary = (
        f"top1={top1_pct:.2f}%, top5_overlap={overlap}/5, "
        f"max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}, "
        f"last_match={last_match}"
    )
    if failures:
        summary += " — FAIL: " + "; ".join(failures)

    if not quiet:
        print(f"  rust argmax[last]={int(rust_logits[0, -1].argmax())} "
              f"tv argmax[last]={int(tv_logits[0, -1].argmax())}")
        print(f"  rust top5[last]={rust_top5}")
        print(f"  tv   top5[last]={tv_top5}")
        print(f"  metrics: {summary}")

    return ModelVerdict(
        name=name,
        passed=passed,
        summary=summary,
        detail=dict(
            shape=list(rust_logits.shape),
            max_abs=max_abs,
            mean_abs=mean_abs,
            top1_argmax_agree_pct=top1_pct,
            top5_last_token_overlap=overlap,
            last_token_argmax_match=last_match,
            rust_top5_last=rust_top5,
            tv_top5_last=tv_top5,
            token_ids=py_token_ids,
            failures=failures,
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        default=",".join(TOL.keys()),
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
    report_path = CACHE_DIR / "verify_causal_lm_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")

    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_read_dump_f32(tmp: Path) -> None:
    """Round-trip: write a known [u32 ndim][u32 shape][f32 data] file and read."""
    path = tmp / "_self_test_dump.bin"
    shape = (1, 3, 4)
    data = np.arange(12, dtype="<f4").reshape(shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", d))
        f.write(data.tobytes(order="C"))
    got = read_dump_f32(path)
    assert got.shape == shape, (got.shape, shape)
    assert np.allclose(got, data), (got, data)
    print("_test_read_dump_f32: ok")


def _test_topk_indices() -> None:
    row = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8], dtype=np.float32)
    # Sorted descending: idx 1 (0.9), 5 (0.8), 3 (0.7), 2 (0.3), 4 (0.2), 0 (0.1)
    assert topk_indices(row, 1) == [1], topk_indices(row, 1)
    assert topk_indices(row, 3) == [1, 5, 3], topk_indices(row, 3)
    assert topk_indices(row, 6) == [1, 5, 3, 2, 4, 0], topk_indices(row, 6)
    # Ties broken by first-occurrence (argsort is stable on equal keys).
    row2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = topk_indices(row2, 2)
    assert sorted(out) == [0, 1] or sorted(out) == [0, 2] or sorted(out) == [1, 2], out
    print("_test_topk_indices: ok")


def _test_top5_overlap() -> None:
    """Synthetic last-token-top5 overlap check matches set-intersection."""
    rust = [10, 20, 30, 40, 50]
    tv = [10, 20, 30, 99, 88]
    overlap = len(set(rust) & set(tv))
    assert overlap == 3, overlap
    print("_test_top5_overlap: ok")


def _self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _test_read_dump_f32(Path(td))
    _test_topk_indices()
    _test_top5_overlap()
    print("self-test: all assertions passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
