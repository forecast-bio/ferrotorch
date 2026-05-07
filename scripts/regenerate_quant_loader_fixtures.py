#!/usr/bin/env python3
"""Regenerate reference fixtures for ferrotorch-llama GPTQ/AWQ/HQQ integration tests.

Tracking issue: #633 (Sprint C.9)

This script downloads three small quantized TinyLlama checkpoints from the
HuggingFace Hub, runs a fixed forward pass through each one using the
appropriate HF reference pipeline, and saves the expected last-token logits
to ``ferrotorch-llama/tests/fixtures/quant_loader_expected.json``.

The Rust integration tests in
``ferrotorch-llama/tests/integration_quant_loaders.rs`` load this JSON and
compare ferrotorch's output within F32_TRANSCENDENTAL = 1e-5 tolerance.

## Dependencies

    pip install --user \\
        transformers==4.50.3 \\
        auto-gptq==0.7.1 \\
        autoawq==0.2.6 \\
        hqq==0.2.1 \\
        torch accelerate

## Usage

    python3 scripts/regenerate_quant_loader_fixtures.py

## Outputs

    ferrotorch-llama/tests/fixtures/quant_loader_expected.json

## Checkpoints used

| Scheme | Repo | Why |
|--------|------|-----|
| GPTQ   | TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ | smallest public GPTQ TinyLlama |
| AWQ    | TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ  | smallest public AWQ TinyLlama  |
| HQQ    | mobiuslabsgmbh/TinyLlama-1.1B-Chat-v1.0-HQQ | public HQQ TinyLlama  |

## Fixed prompt

Token ids ``[1, 15043, 29892, 3186, 29991]`` correspond to ``"Hello, world!"``
tokenized with the TinyLlama SentencePiece tokenizer (BOS=1).  The same ids
are hard-coded in ``PROMPT_IDS`` in the Rust test file.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_OUT_PATH = _REPO_ROOT / "ferrotorch-llama" / "tests" / "fixtures" / "quant_loader_expected.json"

# ---------------------------------------------------------------------------
# Version pin enforcement
# ---------------------------------------------------------------------------

def _require_version(pkg_name: str, module, required_prefix: str, install_hint: str) -> None:
    ver = getattr(module, "__version__", "unknown")
    if not ver.startswith(required_prefix):
        print(
            f"ERROR: {pkg_name}=={ver} installed; require {required_prefix}x.\n"
            f"  Run: {install_hint}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"{pkg_name} {ver} — OK")


try:
    import torch  # type: ignore
except ImportError as exc:
    print(f"ERROR: torch not installed.\n  Run: pip install --user torch\n{exc}", file=sys.stderr)
    sys.exit(1)

try:
    import transformers  # type: ignore
    _require_version(
        "transformers",
        transformers,
        "4.50.",
        "pip install --user transformers==4.50.3",
    )
except ImportError as exc:
    print(f"ERROR: transformers not installed.\n  Run: pip install --user transformers==4.50.3\n{exc}", file=sys.stderr)
    sys.exit(1)

try:
    from auto_gptq import AutoGPTQForCausalLM  # type: ignore
    import auto_gptq  # type: ignore
    _require_version(
        "auto-gptq",
        auto_gptq,
        "0.7.",
        "pip install --user auto-gptq==0.7.1",
    )
except ImportError as exc:
    print(f"ERROR: auto-gptq not installed.\n  Run: pip install --user auto-gptq==0.7.1\n{exc}", file=sys.stderr)
    sys.exit(1)

try:
    from awq import AutoAWQForCausalLM  # type: ignore
    import awq  # type: ignore
    _require_version(
        "autoawq",
        awq,
        "0.2.",
        "pip install --user autoawq==0.2.6",
    )
except ImportError as exc:
    print(f"ERROR: autoawq not installed.\n  Run: pip install --user autoawq==0.2.6\n{exc}", file=sys.stderr)
    sys.exit(1)

try:
    import hqq  # type: ignore
    from hqq.engine.hf import HQQModelForCausalLM  # type: ignore
    _require_version(
        "hqq",
        hqq,
        "0.2.",
        "pip install --user hqq==0.2.1",
    )
except ImportError as exc:
    print(f"ERROR: hqq not installed.\n  Run: pip install --user hqq==0.2.1\n{exc}", file=sys.stderr)
    sys.exit(1)

from transformers import AutoTokenizer  # type: ignore

# ---------------------------------------------------------------------------
# Checkpoint coordinates
# ---------------------------------------------------------------------------

GPTQ_REPO = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
AWQ_REPO = "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ"
HQQ_REPO = "mobiuslabsgmbh/TinyLlama-1.1B-Chat-v1.0-HQQ"

# Fixed prompt token ids — must match PROMPT_IDS in integration_quant_loaders.rs.
# "Hello, world!" with BOS=1, TinyLlama SentencePiece tokenizer.
PROMPT_IDS: list[int] = [1, 15043, 29892, 3186, 29991]

DEVICE = "cpu"  # CPU-only; ferrotorch integration tests run on CPU.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ids_to_tensor(ids: list[int]) -> "torch.Tensor":
    """Wrap a list of token ids into a [1, seq_len] int64 tensor on CPU."""
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def last_token_logits(logits_tensor: "torch.Tensor") -> list[float]:
    """Extract the last-position logit vector as a plain Python float list.

    Args:
        logits_tensor: ``[1, seq_len, vocab_size]`` float32 tensor.

    Returns:
        List of ``vocab_size`` floats for the last sequence position.
    """
    # logits_tensor is [batch=1, seq_len, vocab_size]
    last = logits_tensor[0, -1, :]  # [vocab_size]
    return last.float().tolist()


# ---------------------------------------------------------------------------
# GPTQ reference forward
# ---------------------------------------------------------------------------

def run_gptq(repo: str) -> dict[str, Any]:
    """Download and run a GPTQ forward pass.  Returns fixture data."""
    print(f"\n[GPTQ] Loading {repo} ...")
    model = AutoGPTQForCausalLM.from_quantized(
        repo,
        device_map=DEVICE,
        use_safetensors=True,
        trust_remote_code=False,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )
    model.eval()

    input_ids = ids_to_tensor(PROMPT_IDS)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = last_token_logits(out.logits)
    print(f"[GPTQ] Forward OK — vocab_size={len(logits)}, "
          f"last_logit[0]={logits[0]:.6f}")

    return {
        "repo": repo,
        "prompt_ids": PROMPT_IDS,
        "last_token_logits": logits,
        "vocab_size": len(logits),
    }


# ---------------------------------------------------------------------------
# AWQ reference forward
# ---------------------------------------------------------------------------

def run_awq(repo: str) -> dict[str, Any]:
    """Download and run an AWQ forward pass.  Returns fixture data."""
    print(f"\n[AWQ] Loading {repo} ...")
    model = AutoAWQForCausalLM.from_quantized(
        repo,
        device_map=DEVICE,
        trust_remote_code=False,
        fuse_layers=False,
    )
    model.eval()

    input_ids = ids_to_tensor(PROMPT_IDS)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = last_token_logits(out.logits)
    print(f"[AWQ] Forward OK — vocab_size={len(logits)}, "
          f"last_logit[0]={logits[0]:.6f}")

    return {
        "repo": repo,
        "prompt_ids": PROMPT_IDS,
        "last_token_logits": logits,
        "vocab_size": len(logits),
    }


# ---------------------------------------------------------------------------
# HQQ reference forward
# ---------------------------------------------------------------------------

def run_hqq(repo: str) -> dict[str, Any]:
    """Download and run an HQQ forward pass.  Returns fixture data."""
    print(f"\n[HQQ] Loading {repo} ...")
    model = HQQModelForCausalLM.from_quantized(repo)
    model.eval()

    input_ids = ids_to_tensor(PROMPT_IDS)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = last_token_logits(out.logits)
    print(f"[HQQ] Forward OK — vocab_size={len(logits)}, "
          f"last_logit[0]={logits[0]:.6f}")

    return {
        "repo": repo,
        "prompt_ids": PROMPT_IDS,
        "last_token_logits": logits,
        "vocab_size": len(logits),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("regenerate_quant_loader_fixtures.py — Sprint C.9 (#633)")
    print("=" * 60)

    gptq_data = run_gptq(GPTQ_REPO)
    awq_data = run_awq(AWQ_REPO)
    hqq_data = run_hqq(HQQ_REPO)

    output = {
        "metadata": {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "auto_gptq_version": auto_gptq.__version__,
            "autoawq_version": awq.__version__,
            "hqq_version": hqq.__version__,
            "device": DEVICE,
            "prompt_ids": PROMPT_IDS,
            "tolerance": 1e-5,
            "tolerance_name": "F32_TRANSCENDENTAL",
            "note": (
                "last_token_logits contains the full vocab-size float32 logit "
                "vector for the last prompt token, run on CPU with the reference "
                "HF pipeline.  ferrotorch dequantizes on CPU and must match "
                "within F32_TRANSCENDENTAL = 1e-5 absolute error."
            ),
        },
        "gptq": gptq_data,
        "awq": awq_data,
        "hqq": hqq_data,
    }

    with open(_OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"\nWrote fixture to {_OUT_PATH}")
    print(f"  GPTQ vocab_size: {gptq_data['vocab_size']}")
    print(f"  AWQ  vocab_size: {awq_data['vocab_size']}")
    print(f"  HQQ  vocab_size: {hqq_data['vocab_size']}")


if __name__ == "__main__":
    main()
