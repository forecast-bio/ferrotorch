#!/usr/bin/env python3
"""Regenerate reference fixtures for ferrotorch-llama conformance suite.

Tracking issue: #844

Reference: transformers==4.50.3, torch==2.11.0+cu130
Pin verification: this script asserts the installed versions before running.

Outputs:
    ferrotorch-llama/tests/conformance/fixtures/llama.json

Coverage:
- RMSNorm forward (matches transformers.models.llama.modeling_llama.LlamaRMSNorm)
- RoPE cos/sin tables and apply_rotary_pos_emb
- LlamaMLP (SwiGLU: silu(gate) * up then down)
- LlamaAttention forward (GQA, num_heads=4, num_kv_heads=2)
- LlamaDecoderLayer forward (pre-norm + residual pattern)
- LlamaForCausalLM forward_from_ids (end-to-end, logits shape + values)

Design:
- Tiny 2-layer model: vocab=64, hidden=32, intermediate=64,
  heads=4, kv_heads=2, rope_theta=10000, rms_eps=1e-5
- Seed=42 for reproducibility — always produces the same weights
- All weights initialised with torch.randn rescaled to 0.02 std
  (same as HuggingFace default init) to avoid overflow/NaN in attention
- No model downloads — purely synthetic seeded weights
- CPU only (ferrotorch CPU conformance; GPU tested separately in gpu_smoke.rs)

Usage:
    python3 scripts/regenerate_llama_fixtures.py
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Version pin enforcement.
# ---------------------------------------------------------------------------
import torch  # type: ignore

try:
    import transformers  # type: ignore
    _transformers_ver = transformers.__version__
except ImportError as exc:
    print(
        f"ERROR: transformers not installed. "
        f"Run: pip install --user transformers==4.50.3\n{exc}",
        file=sys.stderr,
    )
    sys.exit(1)

_REQUIRED_TRANSFORMERS_MAJOR_MINOR = (4, 50)
_ver_parts = tuple(int(x) for x in _transformers_ver.split(".")[:2])
if _ver_parts != _REQUIRED_TRANSFORMERS_MAJOR_MINOR:
    print(
        f"ERROR: transformers=={_transformers_ver} installed; "
        f"require 4.50.x. "
        f"Run: pip install --user transformers==4.50.3",
        file=sys.stderr,
    )
    sys.exit(1)

# Use transformers' own LlamaConfig/modeling to build the reference forward.
from transformers import LlamaConfig as HfLlamaConfig  # type: ignore
from transformers.models.llama.modeling_llama import (  # type: ignore
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-llama"
    / "tests"
    / "conformance"
    / "fixtures"
    / "llama.json"
)

RNG_SEED: int = 42
torch.manual_seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Tiny synthetic Llama config — same dimensions as the tiny_config() helper
# in ferrotorch-llama/src/model.rs unit tests, extended to vocab=64 so
# the lm_head exercises non-trivial embedding projections.
# ---------------------------------------------------------------------------
TINY_CFG = HfLlamaConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    max_position_embeddings=64,
    hidden_act="silu",
    tie_word_embeddings=False,
    # Required by transformers ≥ 4.45 for LlamaRotaryEmbedding construction.
    rope_scaling=None,
    attention_bias=False,
    mlp_bias=False,
    _attn_implementation="eager",
)

SEQ_LEN: int = 4  # short sequence, fast
BATCH: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_list(t: torch.Tensor) -> list[Any]:
    """Materialise a tensor to a Python list with NaN/Inf sentinels."""
    raw = t.detach().float().cpu().reshape(-1).tolist()
    out: list[Any] = []
    for v in raw:
        if isinstance(v, float) and math.isnan(v):
            out.append("NaN")
        elif isinstance(v, float) and math.isinf(v):
            out.append("Infinity" if v > 0 else "-Infinity")
        else:
            out.append(v)
    return out


def metadata() -> dict[str, Any]:
    return {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "tiny_config": {
            "vocab_size": TINY_CFG.vocab_size,
            "hidden_size": TINY_CFG.hidden_size,
            "intermediate_size": TINY_CFG.intermediate_size,
            "num_hidden_layers": TINY_CFG.num_hidden_layers,
            "num_attention_heads": TINY_CFG.num_attention_heads,
            "num_key_value_heads": TINY_CFG.num_key_value_heads,
            "rms_norm_eps": TINY_CFG.rms_norm_eps,
            "rope_theta": TINY_CFG.rope_theta,
            "max_position_embeddings": TINY_CFG.max_position_embeddings,
        },
    }


def _make_weight(shape: tuple[int, ...], seed_offset: int = 0) -> torch.Tensor:
    """Return a seeded deterministic weight with std=0.02 (HF default init)."""
    torch.manual_seed(RNG_SEED + seed_offset)
    return torch.randn(*shape, dtype=torch.float32) * 0.02


# ---------------------------------------------------------------------------
# Layer 1 — RMSNorm
# ---------------------------------------------------------------------------


def fixture_rms_norm() -> list[dict[str, Any]]:
    """LlamaRMSNorm: y = x / rms(x) * weight, rms(x) = sqrt(mean(x^2) + eps)."""
    out: list[dict[str, Any]] = []
    norm = LlamaRMSNorm(TINY_CFG.hidden_size, eps=TINY_CFG.rms_norm_eps)
    torch.manual_seed(RNG_SEED)
    w = _make_weight((TINY_CFG.hidden_size,), seed_offset=0)
    norm.weight = torch.nn.Parameter(w.clone())

    for tag, shape in [
        ("1d_token", (TINY_CFG.hidden_size,)),
        ("2d_seq", (SEQ_LEN, TINY_CFG.hidden_size)),
        ("3d_batch_seq", (BATCH, SEQ_LEN, TINY_CFG.hidden_size)),
    ]:
        torch.manual_seed(RNG_SEED + 1)
        x = torch.randn(*shape, dtype=torch.float32) * 0.5
        with torch.no_grad():
            y = norm(x)
        out.append(
            {
                "op": "rms_norm",
                "tag": tag,
                "input_shape": list(shape),
                "weight": to_list(norm.weight),
                "eps": TINY_CFG.rms_norm_eps,
                "input": to_list(x),
                "expected": to_list(y),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Layer 2 — RoPE
# ---------------------------------------------------------------------------


def fixture_rope() -> list[dict[str, Any]]:
    """RoPE: apply_rotary_pos_emb matches LlamaRotaryEmbedding + apply_rotary_pos_emb."""
    out: list[dict[str, Any]] = []

    head_dim = TINY_CFG.hidden_size // TINY_CFG.num_attention_heads  # 8
    # Build the rotary embedding module.
    rope = LlamaRotaryEmbedding(config=TINY_CFG)

    for seq_len in [1, SEQ_LEN]:
        torch.manual_seed(RNG_SEED + 10)
        # q: [batch, num_heads, seq, head_dim]
        q = torch.randn(BATCH, TINY_CFG.num_attention_heads, seq_len, head_dim) * 0.1
        k = torch.randn(BATCH, TINY_CFG.num_key_value_heads, seq_len, head_dim) * 0.1
        # Position ids: [batch, seq_len]
        position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]

        with torch.no_grad():
            cos, sin = rope(q, position_ids)
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # type: ignore
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        out.append(
            {
                "op": "rope_apply",
                "tag": f"seq{seq_len}",
                "head_dim": head_dim,
                "rope_theta": TINY_CFG.rope_theta,
                "max_position_embeddings": TINY_CFG.max_position_embeddings,
                "num_heads": TINY_CFG.num_attention_heads,
                "num_kv_heads": TINY_CFG.num_key_value_heads,
                "seq_len": seq_len,
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "q_input": to_list(q),
                "k_input": to_list(k),
                "position_ids": position_ids.tolist(),
                "cos": to_list(cos),
                "sin": to_list(sin),
                "q_rotated": to_list(q_rot),
                "k_rotated": to_list(k_rot),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Layer 3 — MLP (SwiGLU)
# ---------------------------------------------------------------------------


def fixture_mlp() -> list[dict[str, Any]]:
    """LlamaMLP: down(silu(gate(x)) * up(x))."""
    out: list[dict[str, Any]] = []

    mlp = LlamaMLP(TINY_CFG)
    # Seed weights deterministically.
    torch.manual_seed(RNG_SEED + 20)
    mlp.gate_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.intermediate_size, TINY_CFG.hidden_size), 20)
    )
    mlp.up_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.intermediate_size, TINY_CFG.hidden_size), 21)
    )
    mlp.down_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size, TINY_CFG.intermediate_size), 22)
    )
    mlp.eval()

    for tag, shape in [
        ("2d_seq", (SEQ_LEN, TINY_CFG.hidden_size)),
        ("3d_batch_seq", (BATCH, SEQ_LEN, TINY_CFG.hidden_size)),
    ]:
        torch.manual_seed(RNG_SEED + 23)
        x = torch.randn(*shape, dtype=torch.float32) * 0.1
        with torch.no_grad():
            y = mlp(x)
        out.append(
            {
                "op": "mlp_forward",
                "tag": tag,
                "input_shape": list(shape),
                "hidden_size": TINY_CFG.hidden_size,
                "intermediate_size": TINY_CFG.intermediate_size,
                "gate_proj_weight": to_list(mlp.gate_proj.weight),
                "up_proj_weight": to_list(mlp.up_proj.weight),
                "down_proj_weight": to_list(mlp.down_proj.weight),
                "input": to_list(x),
                "expected": to_list(y),
                "expected_shape": list(y.shape),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Layer 4 — Attention (GQA)
# ---------------------------------------------------------------------------


def fixture_attention() -> list[dict[str, Any]]:
    """LlamaAttention forward: GQA with num_heads=4, num_kv_heads=2.

    In transformers 4.50.x LlamaAttention.forward requires (hidden_states,
    position_embeddings, attention_mask, ...) where position_embeddings is
    the (cos, sin) tuple from LlamaRotaryEmbedding. We compute it from the
    rope module and pass it explicitly.
    """
    out: list[dict[str, Any]] = []

    attn = LlamaAttention(TINY_CFG, layer_idx=0)
    rope = LlamaRotaryEmbedding(config=TINY_CFG)
    torch.manual_seed(RNG_SEED + 30)
    head_dim = TINY_CFG.hidden_size // TINY_CFG.num_attention_heads
    kv_dim = TINY_CFG.num_key_value_heads * head_dim

    attn.q_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size, TINY_CFG.hidden_size), 30)
    )
    attn.k_proj.weight = torch.nn.Parameter(
        _make_weight((kv_dim, TINY_CFG.hidden_size), 31)
    )
    attn.v_proj.weight = torch.nn.Parameter(
        _make_weight((kv_dim, TINY_CFG.hidden_size), 32)
    )
    attn.o_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size, TINY_CFG.hidden_size), 33)
    )
    attn.eval()

    for seq_len in [1, SEQ_LEN]:
        torch.manual_seed(RNG_SEED + 34)
        x = torch.randn(BATCH, seq_len, TINY_CFG.hidden_size, dtype=torch.float32) * 0.1
        position_ids = torch.arange(seq_len).unsqueeze(0)
        with torch.no_grad():
            # Build position_embeddings = (cos, sin) via the rope module.
            # rope.forward(x, position_ids) expects x shaped [B, num_heads, S, head_dim]
            # but only uses x for dtype/device — we use a dummy Q-shaped tensor.
            dummy_q = torch.zeros(
                BATCH, TINY_CFG.num_attention_heads, seq_len, head_dim,
                dtype=torch.float32,
            )
            cos, sin = rope(dummy_q, position_ids)
            # transformers 4.50.x returns a 2-tuple (hidden_states, attn_weights)
            # when output_attentions=False (the default).
            attn_out = attn(
                hidden_states=x,
                position_embeddings=(cos, sin),
                attention_mask=None,
                past_key_value=None,
            )
            out_tensor = attn_out[0]
        out.append(
            {
                "op": "attention_forward",
                "tag": f"seq{seq_len}_gqa",
                "batch": BATCH,
                "seq_len": seq_len,
                "hidden_size": TINY_CFG.hidden_size,
                "num_heads": TINY_CFG.num_attention_heads,
                "num_kv_heads": TINY_CFG.num_key_value_heads,
                "head_dim": head_dim,
                "q_proj_weight": to_list(attn.q_proj.weight),
                "k_proj_weight": to_list(attn.k_proj.weight),
                "v_proj_weight": to_list(attn.v_proj.weight),
                "o_proj_weight": to_list(attn.o_proj.weight),
                "input": to_list(x),
                "input_shape": [BATCH, seq_len, TINY_CFG.hidden_size],
                "expected": to_list(out_tensor),
                "expected_shape": list(out_tensor.shape),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Layer 5 — DecoderLayer
# ---------------------------------------------------------------------------


def fixture_decoder_layer() -> list[dict[str, Any]]:
    """LlamaDecoderLayer: pre-norm + attention + residual + pre-norm + MLP + residual.

    In transformers 4.50.x, LlamaDecoderLayer.forward requires explicit
    position_embeddings=(cos, sin). We compute these from LlamaRotaryEmbedding
    before invoking the layer.
    """
    out: list[dict[str, Any]] = []

    layer = LlamaDecoderLayer(TINY_CFG, layer_idx=0)
    rope = LlamaRotaryEmbedding(config=TINY_CFG)
    torch.manual_seed(RNG_SEED + 40)
    head_dim = TINY_CFG.hidden_size // TINY_CFG.num_attention_heads
    kv_dim = TINY_CFG.num_key_value_heads * head_dim

    # Seed all sub-module weights deterministically.
    layer.input_layernorm.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size,), 40)
    )
    layer.post_attention_layernorm.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size,), 41)
    )
    layer.self_attn.q_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size, TINY_CFG.hidden_size), 42)
    )
    layer.self_attn.k_proj.weight = torch.nn.Parameter(
        _make_weight((kv_dim, TINY_CFG.hidden_size), 43)
    )
    layer.self_attn.v_proj.weight = torch.nn.Parameter(
        _make_weight((kv_dim, TINY_CFG.hidden_size), 44)
    )
    layer.self_attn.o_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size, TINY_CFG.hidden_size), 45)
    )
    layer.mlp.gate_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.intermediate_size, TINY_CFG.hidden_size), 46)
    )
    layer.mlp.up_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.intermediate_size, TINY_CFG.hidden_size), 47)
    )
    layer.mlp.down_proj.weight = torch.nn.Parameter(
        _make_weight((TINY_CFG.hidden_size, TINY_CFG.intermediate_size), 48)
    )
    layer.eval()

    torch.manual_seed(RNG_SEED + 49)
    x = torch.randn(BATCH, SEQ_LEN, TINY_CFG.hidden_size, dtype=torch.float32) * 0.1
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    with torch.no_grad():
        # Compute rope cos/sin for position_embeddings kwarg (required in 4.50.x).
        dummy_q = torch.zeros(
            BATCH, TINY_CFG.num_attention_heads, SEQ_LEN, head_dim, dtype=torch.float32
        )
        cos, sin = rope(dummy_q, position_ids)
        result = layer(
            hidden_states=x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
    # In 4.50.x with use_cache=False, output_attentions=False the tuple has
    # exactly one element: the updated hidden states.
    y = result[0]

    def _wlist(param: torch.nn.Parameter) -> list[Any]:
        return to_list(param)

    out.append(
        {
            "op": "decoder_layer_forward",
            "tag": f"seq{SEQ_LEN}_2norm_attn_mlp",
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "hidden_size": TINY_CFG.hidden_size,
            "num_heads": TINY_CFG.num_attention_heads,
            "num_kv_heads": TINY_CFG.num_key_value_heads,
            "intermediate_size": TINY_CFG.intermediate_size,
            "rms_norm_eps": TINY_CFG.rms_norm_eps,
            # Weights (for Rust-side state-dict injection).
            "input_layernorm_weight": _wlist(layer.input_layernorm.weight),
            "post_attention_layernorm_weight": _wlist(layer.post_attention_layernorm.weight),
            "q_proj_weight": _wlist(layer.self_attn.q_proj.weight),
            "k_proj_weight": _wlist(layer.self_attn.k_proj.weight),
            "v_proj_weight": _wlist(layer.self_attn.v_proj.weight),
            "o_proj_weight": _wlist(layer.self_attn.o_proj.weight),
            "gate_proj_weight": _wlist(layer.mlp.gate_proj.weight),
            "up_proj_weight": _wlist(layer.mlp.up_proj.weight),
            "down_proj_weight": _wlist(layer.mlp.down_proj.weight),
            # Input / output.
            "input": to_list(x),
            "input_shape": [BATCH, SEQ_LEN, TINY_CFG.hidden_size],
            "expected": to_list(y),
            "expected_shape": list(y.shape),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Layer 6 — Full LlamaForCausalLM forward_from_ids
# ---------------------------------------------------------------------------


def fixture_causal_lm() -> list[dict[str, Any]]:
    """LlamaForCausalLM end-to-end: token ids → logits [1, seq, vocab]."""
    out: list[dict[str, Any]] = []

    torch.manual_seed(RNG_SEED + 50)
    model = LlamaForCausalLM(TINY_CFG)
    # Reinitialise every parameter with std=0.02 for reproducibility.
    # (LlamaForCausalLM._init_weights uses the config's initializer_range.)
    torch.manual_seed(RNG_SEED + 50)
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            torch.nn.init.ones_(param)  # norm weights
    model.eval()

    # Gather state dict for Rust-side loading.
    sd: dict[str, list[Any]] = {}
    for name, param in model.named_parameters():
        sd[name] = to_list(param.data)

    token_ids = [1, 5, 7, 9]  # four tokens, all in [0, vocab_size)
    ids_tensor = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids=ids_tensor).logits  # [1, seq, vocab]

    # Record last-token logits for a targeted greedy test.
    last_logits = logits[0, -1, :]  # [vocab_size]
    greedy_next = int(last_logits.argmax().item())

    out.append(
        {
            "op": "causal_lm_forward",
            "tag": "end_to_end_seq4",
            "batch": BATCH,
            "seq_len": len(token_ids),
            "vocab_size": TINY_CFG.vocab_size,
            "hidden_size": TINY_CFG.hidden_size,
            "token_ids": token_ids,
            # Full state dict (flat lists, HF naming).
            "state_dict": sd,
            # Expected outputs.
            "expected_logits": to_list(logits),
            "expected_logits_shape": list(logits.shape),
            "expected_last_logits": to_list(last_logits),
            "greedy_next_token": greedy_next,
        }
    )

    # Edge case: single-token prefix.
    single_ids = [3]
    ids_tensor_1 = torch.tensor([single_ids], dtype=torch.long)
    with torch.no_grad():
        logits_1 = model(input_ids=ids_tensor_1).logits

    out.append(
        {
            "op": "causal_lm_forward",
            "tag": "single_token",
            "batch": BATCH,
            "seq_len": 1,
            "vocab_size": TINY_CFG.vocab_size,
            "hidden_size": TINY_CFG.hidden_size,
            "token_ids": single_ids,
            "state_dict": sd,  # same weights
            "expected_logits": to_list(logits_1),
            "expected_logits_shape": list(logits_1.shape),
            "expected_last_logits": to_list(logits_1[0, -1, :]),
            "greedy_next_token": int(logits_1[0, -1, :].argmax().item()),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Layer 7 — Config validation edge cases (no torch, pure metadata)
# ---------------------------------------------------------------------------


def fixture_config_validation() -> list[dict[str, Any]]:
    """LlamaConfig validation: zero fields and non-divisible heads.

    These don't need a reference value — the test just verifies ferrotorch
    returns Err for these inputs. Recorded as fixture rows for symmetry.
    """
    return [
        {
            "op": "config_validation",
            "tag": "zero_hidden_size",
            "vocab_size": 32,
            "hidden_size": 0,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "expected_error": True,
        },
        {
            "op": "config_validation",
            "tag": "non_divisible_heads",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 5,  # 32 % 5 != 0
            "num_key_value_heads": 5,
            "expected_error": True,
        },
        {
            "op": "config_validation",
            "tag": "kv_heads_not_dividing_attn_heads",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 3,  # 4 % 3 != 0
            "expected_error": True,
        },
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    fixtures: list[dict[str, Any]] = []

    print("Generating RMSNorm fixtures...", flush=True)
    fixtures.extend(fixture_rms_norm())

    print("Generating RoPE fixtures...", flush=True)
    fixtures.extend(fixture_rope())

    print("Generating MLP fixtures...", flush=True)
    fixtures.extend(fixture_mlp())

    print("Generating Attention fixtures...", flush=True)
    fixtures.extend(fixture_attention())

    print("Generating DecoderLayer fixtures...", flush=True)
    fixtures.extend(fixture_decoder_layer())

    print("Generating CausalLM end-to-end fixtures...", flush=True)
    fixtures.extend(fixture_causal_lm())

    print("Adding config validation fixtures...", flush=True)
    fixtures.extend(fixture_config_validation())

    payload = {
        "metadata": metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"Wrote {FIXTURE_PATH} ({len(fixtures)} fixtures, transformers=={transformers.__version__})")


if __name__ == "__main__":
    main()
