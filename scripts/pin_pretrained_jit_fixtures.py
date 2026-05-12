#!/usr/bin/env python3
"""Pin the ferrotorch-jit trace/AoT-compile parity fixtures (#1170, Phase G.4).

The ferrotorch-jit tracer captures an autograd graph by executing a
closure that takes a slice of `Tensor<T>` arguments. The supported IR
ops (per `ferrotorch-jit/src/graph.rs::IrOpKind`) do *not* include
Conv2d / BatchNorm / LayerNorm — so ResNet-18 (the obvious next pin
after #1139) is *not* traceable. The most complex pretrained model the
tracer can handle today is a feed-forward MLP built out of `Linear` +
elementwise activations.

This pin produces a deterministic 2-layer MLP

    x -> Linear(in -> hidden, bias=True)
      -> ReLU
      -> Linear(hidden -> out, bias=True)
      -> output

with `in=8, hidden=16, out=4` (small enough that f32 accumulation is
trivially tight, large enough that any weight transpose / bias-broadcast
bug surfaces in `max_abs`).

The mirror ships:
  * `model.safetensors` — `{l1_weight, l1_bias, l2_weight, l2_bias}`
    in the standard PyTorch nn.Linear layout
    (`weight: [out, in]`, `bias: [out]`)
  * `_value_parity_input.bin`  — [N=5, in=8] f32 deterministic inputs
  * `_value_parity_output.bin` — [N=5, out=4] f32 eager reference outputs
  * `config.json` — architecture + tolerance hints

All three artifacts are pinned to `ferrotorch/jit-trace-parity-v1`.

Run via:
  python3 scripts/pin_pretrained_jit_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK_DIR = Path("/tmp/ferrotorch_pin_jit_trace_parity_v1")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Architecture — frozen so the rust example can use the same dims.
IN_FEATURES = 8
HIDDEN = 16
OUT_FEATURES = 4
N_SAMPLES = 5


class MLP(nn.Module):
    """The exact reference architecture the tracer will reproduce.

    Same `nn.Linear` convention as `ferrotorch-nn::Linear`
    (`weight: [out, in]`, `bias: [out]`), so the state_dict ports 1:1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(IN_FEATURES, HIDDEN, bias=True)
        self.l2 = nn.Linear(HIDDEN, OUT_FEATURES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


def dump_f32(path: Path, t: torch.Tensor) -> None:
    arr = t.detach().cpu().to(torch.float32).numpy()
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)

    # -- 1. Build the model with deterministic weights. ---------------------
    model = MLP()
    # The default kaiming init under `torch.manual_seed(42)` is already
    # deterministic, but we explicitly fix every weight to ensure that
    # subsequent torch-version bumps cannot drift this pin.
    with torch.no_grad():
        rng = np.random.RandomState(42)
        l1_w = rng.uniform(-0.3, 0.3, size=(HIDDEN, IN_FEATURES)).astype(np.float32)
        l1_b = rng.uniform(-0.1, 0.1, size=(HIDDEN,)).astype(np.float32)
        l2_w = rng.uniform(-0.3, 0.3, size=(OUT_FEATURES, HIDDEN)).astype(np.float32)
        l2_b = rng.uniform(-0.1, 0.1, size=(OUT_FEATURES,)).astype(np.float32)
        model.l1.weight.copy_(torch.from_numpy(l1_w))
        model.l1.bias.copy_(torch.from_numpy(l1_b))
        model.l2.weight.copy_(torch.from_numpy(l2_w))
        model.l2.bias.copy_(torch.from_numpy(l2_b))
    model.eval()

    # -- 2. Build the deterministic input batch. -----------------------------
    rng = np.random.RandomState(123)
    inputs = rng.standard_normal(size=(N_SAMPLES, IN_FEATURES)).astype(np.float32)
    inputs_t = torch.from_numpy(inputs)

    # -- 3. Eager forward — frozen reference output. -------------------------
    with torch.no_grad():
        ref_out = model(inputs_t)
    print(
        f"[pin] eager forward: input shape={list(inputs_t.shape)} "
        f"output shape={list(ref_out.shape)}",
        flush=True,
    )
    print(
        f"[pin] eager output sample row 0: {ref_out[0].numpy().tolist()}",
        flush=True,
    )

    # -- 4. Save state_dict + fixtures. --------------------------------------
    from safetensors.torch import save_file

    state = model.state_dict()
    # Map to flat keys (l1_weight, l1_bias, …) so the rust loader does
    # not need to walk an nn.Module key tree — it can just look up the
    # four flat strings.
    flat = {
        "l1_weight": state["l1.weight"].detach().contiguous(),
        "l1_bias": state["l1.bias"].detach().contiguous(),
        "l2_weight": state["l2.weight"].detach().contiguous(),
        "l2_bias": state["l2.bias"].detach().contiguous(),
    }
    print("[pin] state_dict keys:", flush=True)
    for k, v in flat.items():
        print(f"[pin]   {k}: shape={tuple(v.shape)} dtype={v.dtype}", flush=True)

    weights_path = WORK_DIR / "model.safetensors"
    save_file(flat, str(weights_path))
    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    print(f"[pin] wrote {weights_path} ({weights_path.stat().st_size} bytes)", flush=True)
    print(f"[pin] SHA-256: {sha}", flush=True)

    input_path = WORK_DIR / "_value_parity_input.bin"
    output_path = WORK_DIR / "_value_parity_output.bin"
    dump_f32(input_path, inputs_t)
    dump_f32(output_path, ref_out)
    print(
        f"[pin] dumped parity fixtures: input={input_path.stat().st_size}B "
        f"output={output_path.stat().st_size}B",
        flush=True,
    )

    cfg = {
        "architecture": "MLP",
        "layers": "Linear(in -> hidden, bias=True) -> ReLU -> Linear(hidden -> out, bias=True)",
        "in_features": IN_FEATURES,
        "hidden": HIDDEN,
        "out_features": OUT_FEATURES,
        "n_samples": N_SAMPLES,
        "seed_weights": 42,
        "seed_inputs": 123,
        # The Rust eager and traced paths both run on f32 CPU. Tracing is a
        # pure shape-preserving graph walk over the same autograd ops the
        # eager Linear executes, so traced vs eager is bit-identical up to
        # f32 rounding. Eager vs reference picks up float32 accumulation
        # differences between torch's BLAS and ferrotorch's mm_differentiable
        # impl.
        "tolerance": {
            "eager_vs_reference_max_abs": 1e-4,
            "eager_vs_reference_cosine_sim_min": 0.99999,
            "traced_vs_eager_max_abs": 1e-5,
            "compiled_vs_eager_max_abs": 1e-4,
        },
    }
    cfg_path = WORK_DIR / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # -- 5. Upload to HF (best-effort; tolerate offline). -------------------
    upload = os.environ.get("FERROTORCH_PIN_UPLOAD", "1") != "0"
    if upload:
        try:
            from huggingface_hub import HfApi, create_repo, upload_file
        except ImportError:
            print(
                "[pin] huggingface_hub not installed — skipping upload "
                "(set FERROTORCH_PIN_UPLOAD=0 to silence)",
                file=sys.stderr,
                flush=True,
            )
        else:
            repo_id = "ferrotorch/jit-trace-parity-v1"
            try:
                create_repo(repo_id, repo_type="model", exist_ok=True)
                for relative, local in [
                    ("model.safetensors", weights_path),
                    ("_value_parity_input.bin", input_path),
                    ("_value_parity_output.bin", output_path),
                    ("config.json", cfg_path),
                ]:
                    upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=relative,
                        repo_id=repo_id,
                        repo_type="model",
                    )
                api = HfApi()
                files = api.list_repo_files(repo_id=repo_id, repo_type="model")
                print(f"[pin] uploaded to {repo_id}. Repo files:", flush=True)
                for fname in files:
                    print(f"[pin]   - {fname}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[pin] HF upload failed: {exc!r}", file=sys.stderr, flush=True)
                return 2
    else:
        print("[pin] FERROTORCH_PIN_UPLOAD=0, skipping upload", flush=True)

    print(f"\n[pin] DONE. SHA-256 for registry.rs pin: {sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
