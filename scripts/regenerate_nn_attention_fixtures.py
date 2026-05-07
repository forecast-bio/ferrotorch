#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-nn C9.4
(attention + grad_fns conformance).

Tracking issue: crosslink #806 (C9.4 sub-phase).

Output:
    ferrotorch-nn/tests/conformance/fixtures/nn_attention.json

Coverage:
  Module 1 — attention.MultiheadAttention
    * small MHA forward (seq=8, d_model=16, nhead=4), seeded weights
    * GQA forward (nhead=4, nkv=2)
    * helper fns: reshape_to_heads, transpose_heads_to_2d, repeat_kv
  Module 2 — flash_attention
    * flash_attention forward (causal + non-causal) vs standard_attention
    * standard_attention forward as reference
  Module 3 — flex_attention (composite cross-check against flex_attention.json)
    * flex_attention baseline forward
    * causal_score_mod / alibi_score_mod smoke
  Module 4 — paged_attention
    * PagePool alloc/free cycle (no numeric output — structural)
    * PagedKVCache append + retrieval
    * PagedAttentionManager multi-sequence
  Module 5 — transformer
    * RotaryPositionEmbedding apply (interleaved + half_rotation)
    * SwiGLU forward
    * KVCache update + retrieval
    * TransformerEncoderLayer forward (pre-norm)
    * TransformerDecoderLayer forward_with_memory
  Module 6 — grad_fns (via ferrotorch-core re-exports)
    * softmax backward (gradient through conformance_nn test)
    * layer-norm backward
    * cross-entropy backward

Tolerances:
  F32 CPU matmul:  1e-4
  F32 CPU pointwise: 1e-5
  F64 CPU matmul:  1e-9

Usage from WSL (preferred):

    python3 scripts/regenerate_nn_attention_fixtures.py

Required Python deps: torch >= 2.0, numpy.
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

# ---------------------------------------------------------------------------
# Output path and metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-nn"
    / "tests"
    / "conformance"
    / "fixtures"
    / "nn_attention.json"
)

RNG_SEED: int = 0xC9_4BEE
torch.manual_seed(RNG_SEED)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def to_listf(t: torch.Tensor) -> list[Any]:
    """Materialize a tensor to a CPU Python list, encoding special floats."""
    raw = t.detach().to("cpu").to(torch.float64).reshape(-1).tolist()
    out: list[Any] = []
    for v in raw:
        if isinstance(v, float):
            if math.isnan(v):
                out.append("NaN")
            elif math.isinf(v):
                out.append("Infinity" if v > 0 else "-Infinity")
            else:
                out.append(v)
        else:
            out.append(v)
    return out


def seeded_randn(
    *shape: int,
    dtype: torch.dtype = torch.float32,
    seed: int = RNG_SEED,
) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=g)


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": hex(RNG_SEED),
        "phase": "C9.4",
        "tracking_issue": "#806",
    }


# ===========================================================================
# Module 1 — MultiheadAttention
# ===========================================================================

def _mha_seeded_weights(
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    bias: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Generate seeded weight tensors for an MHA layer."""
    g = torch.Generator()
    g.manual_seed(seed)
    kv_dim = (embed_dim // num_heads) * num_kv_heads
    weights: dict[str, torch.Tensor] = {
        "q_proj": torch.randn(embed_dim, embed_dim, generator=g),
        "k_proj": torch.randn(kv_dim, embed_dim, generator=g),
        "v_proj": torch.randn(kv_dim, embed_dim, generator=g),
        "out_proj": torch.randn(embed_dim, embed_dim, generator=g),
    }
    if bias:
        weights["q_bias"] = torch.zeros(embed_dim)
        weights["k_bias"] = torch.zeros(kv_dim)
        weights["v_bias"] = torch.zeros(kv_dim)
        weights["out_bias"] = torch.zeros(embed_dim)
    return weights


def _manual_mha_forward(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    causal_mask: bool,
) -> torch.Tensor:
    """
    Manual MHA forward matching ferrotorch's MultiheadAttention.

    Input x: [batch, seq, embed_dim]
    Output: [batch, seq, embed_dim]
    """
    batch, seq, d = x.shape
    head_dim = embed_dim // num_heads
    kv_dim = head_dim * num_kv_heads
    group_size = num_heads // num_kv_heads

    # Project Q, K, V
    q = x @ weights["q_proj"].T  # [B, S, D]
    k = x @ weights["k_proj"].T  # [B, S, kv_dim]
    v = x @ weights["v_proj"].T  # [B, S, kv_dim]

    if "q_bias" in weights:
        q = q + weights["q_bias"]
        k = k + weights["k_bias"]
        v = v + weights["v_bias"]

    # Reshape to [B, heads, seq, head_dim]
    q = q.reshape(batch, seq, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

    # Repeat K/V for GQA
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, sq, sk]

    if causal_mask:
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)  # [B, H, sq, dv]

    # Merge heads: [B, sq, D]
    attn_out = attn_out.transpose(1, 2).reshape(batch, seq, embed_dim)

    # Output projection
    out = attn_out @ weights["out_proj"].T
    if "out_bias" in weights:
        out = out + weights["out_bias"]
    return out


def fixture_mha_small_forward() -> list[dict[str, Any]]:
    """MHA: small (seq=8, d_model=16, nhead=4) seeded forward."""
    fixtures: list[dict[str, Any]] = []
    for dtype_name in ("float32", "float64"):
        dtype = torch_dtype(dtype_name)
        embed_dim, num_heads = 16, 4
        batch, seq = 2, 8
        seed = RNG_SEED + 1

        weights = _mha_seeded_weights(embed_dim, num_heads, num_heads, bias=True, seed=seed)
        weights = {k: v.to(dtype) for k, v in weights.items()}

        x = seeded_randn(batch, seq, embed_dim, dtype=dtype, seed=seed + 10)

        out = _manual_mha_forward(x, weights, embed_dim, num_heads, num_heads, causal_mask=False)

        row: dict[str, Any] = {
            "op": "mha_forward",
            "tag": f"small_bias_seq8_{dtype_name}",
            "dtype": dtype_name,
            "batch": batch,
            "seq": seq,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_heads,
            "bias": True,
            "causal_mask": False,
            "seed": seed,
            "input_data": to_listf(x),
            "expected_output": to_listf(out),
        }
        for k, v in weights.items():
            row[f"weight_{k}"] = to_listf(v)
        fixtures.append(row)

    return fixtures


def fixture_mha_causal_forward() -> list[dict[str, Any]]:
    """MHA: causal forward (seq=8, d_model=16, nhead=4)."""
    fixtures: list[dict[str, Any]] = []
    dtype_name = "float32"
    dtype = torch_dtype(dtype_name)
    embed_dim, num_heads = 16, 4
    batch, seq = 1, 8
    seed = RNG_SEED + 2

    weights = _mha_seeded_weights(embed_dim, num_heads, num_heads, bias=False, seed=seed)
    weights = {k: v.to(dtype) for k, v in weights.items()}
    x = seeded_randn(batch, seq, embed_dim, dtype=dtype, seed=seed + 10)

    out = _manual_mha_forward(x, weights, embed_dim, num_heads, num_heads, causal_mask=True)
    row: dict[str, Any] = {
        "op": "mha_forward",
        "tag": f"causal_seq8_{dtype_name}",
        "dtype": dtype_name,
        "batch": batch,
        "seq": seq,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_heads,
        "bias": False,
        "causal_mask": True,
        "seed": seed,
        "input_data": to_listf(x),
        "expected_output": to_listf(out),
    }
    for k, v in weights.items():
        row[f"weight_{k}"] = to_listf(v)
    fixtures.append(row)
    return fixtures


def fixture_mha_gqa_forward() -> list[dict[str, Any]]:
    """MHA-GQA: 4 query heads, 2 KV heads (group_size=2)."""
    fixtures: list[dict[str, Any]] = []
    dtype_name = "float32"
    dtype = torch_dtype(dtype_name)
    embed_dim, num_heads, num_kv_heads = 16, 4, 2
    batch, seq = 1, 4
    seed = RNG_SEED + 3

    weights = _mha_seeded_weights(embed_dim, num_heads, num_kv_heads, bias=False, seed=seed)
    weights = {k: v.to(dtype) for k, v in weights.items()}
    x = seeded_randn(batch, seq, embed_dim, dtype=dtype, seed=seed + 10)

    out = _manual_mha_forward(x, weights, embed_dim, num_heads, num_kv_heads, causal_mask=False)
    row: dict[str, Any] = {
        "op": "mha_forward",
        "tag": f"gqa_4q2kv_{dtype_name}",
        "dtype": dtype_name,
        "batch": batch,
        "seq": seq,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "bias": False,
        "causal_mask": False,
        "seed": seed,
        "input_data": to_listf(x),
        "expected_output": to_listf(out),
    }
    for k, v in weights.items():
        row[f"weight_{k}"] = to_listf(v)
    fixtures.append(row)
    return fixtures


def fixture_reshape_to_heads() -> list[dict[str, Any]]:
    """reshape_to_heads: verify shape transformation [B, S, H*D] → [B, H, S, D]."""
    fixtures: list[dict[str, Any]] = []
    batch, seq, num_heads, head_dim = 2, 6, 4, 8
    embed_dim = num_heads * head_dim
    x = seeded_randn(batch, seq, embed_dim, seed=RNG_SEED + 20)
    expected = x.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)
    fixtures.append({
        "op": "reshape_to_heads",
        "tag": "basic",
        "dtype": "float32",
        "batch": batch,
        "seq": seq,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "input_data": to_listf(x),
        "expected_output": to_listf(expected),
    })
    return fixtures


def fixture_repeat_kv() -> list[dict[str, Any]]:
    """repeat_kv: verify [B, H_kv, S, D] → [B, H_q, S, D] via group expansion."""
    fixtures: list[dict[str, Any]] = []
    batch, kv_heads, seq, head_dim, group_size = 2, 2, 4, 8, 3
    kv = seeded_randn(batch, kv_heads, seq, head_dim, seed=RNG_SEED + 30)
    expected = kv.repeat_interleave(group_size, dim=1)
    fixtures.append({
        "op": "repeat_kv",
        "tag": "group3",
        "dtype": "float32",
        "batch": batch,
        "kv_heads": kv_heads,
        "seq": seq,
        "head_dim": head_dim,
        "group_size": group_size,
        "input_data": to_listf(kv),
        "expected_output": to_listf(expected),
    })
    return fixtures


# ===========================================================================
# Module 2 — flash_attention / standard_attention
# ===========================================================================

def _standard_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Vanilla scaled-dot-product attention: [B, N_q, D] inputs."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.bmm(q, k.transpose(-2, -1)) * scale  # [B, N_q, N_k]
    if causal:
        n_q, n_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(n_q, n_k, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, v)


def fixture_flash_attention() -> list[dict[str, Any]]:
    """
    flash_attention / standard_attention: numerics must agree on small shapes
    (flash_attention implements exact online-softmax, same math as standard).
    Fixtures store the standard_attention reference output; ferrotorch's
    flash_attention must match within F32_MATMUL = 1e-4.
    """
    fixtures: list[dict[str, Any]] = []
    configs = [
        # (batch, n_q, n_k, d, d_v, causal, tag)
        (2, 8, 8, 16, 16, False, "basic_noncausal"),
        (2, 8, 8, 16, 16, True, "basic_causal"),
        (1, 4, 4, 8, 8, True, "tiny_causal"),
        (2, 6, 4, 8, 12, False, "rect_noncausal"),
    ]
    for dtype_name in ("float32", "float64"):
        dtype = torch_dtype(dtype_name)
        for batch, n_q, n_k, d, d_v, causal, tag in configs:
            seed = RNG_SEED + hash(tag) % 1000
            q = seeded_randn(batch, n_q, d, dtype=dtype, seed=seed)
            k = seeded_randn(batch, n_k, d, dtype=dtype, seed=seed + 1)
            v = seeded_randn(batch, n_k, d_v, dtype=dtype, seed=seed + 2)
            out = _standard_sdpa(q, k, v, causal)
            fixtures.append({
                "op": "flash_attention",
                "tag": f"{tag}_{dtype_name}",
                "dtype": dtype_name,
                "batch": batch,
                "n_q": n_q,
                "n_k": n_k,
                "d": d,
                "d_v": d_v,
                "causal": causal,
                "q_data": to_listf(q),
                "k_data": to_listf(k),
                "v_data": to_listf(v),
                "expected_output": to_listf(out),
            })
    return fixtures


def fixture_standard_attention() -> list[dict[str, Any]]:
    """standard_attention: direct reference for the non-tiled path."""
    fixtures: list[dict[str, Any]] = []
    dtype_name = "float32"
    dtype = torch_dtype(dtype_name)
    batch, n, d = 2, 6, 8
    q = seeded_randn(batch, n, d, dtype=dtype, seed=RNG_SEED + 50)
    k = seeded_randn(batch, n, d, dtype=dtype, seed=RNG_SEED + 51)
    v = seeded_randn(batch, n, d, dtype=dtype, seed=RNG_SEED + 52)
    out = _standard_sdpa(q, k, v, causal=False)
    fixtures.append({
        "op": "standard_attention",
        "tag": f"basic_{dtype_name}",
        "dtype": dtype_name,
        "batch": batch,
        "n": n,
        "d": d,
        "q_data": to_listf(q),
        "k_data": to_listf(k),
        "v_data": to_listf(v),
        "expected_output": to_listf(out),
    })
    return fixtures


# ===========================================================================
# Module 3 — flex_attention (composite cross-checks)
# ===========================================================================

def _manual_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Manual flex attention: [B, H, N_q, D] inputs."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        scores = scores + bias
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def fixture_flex_attention() -> list[dict[str, Any]]:
    """
    flex_attention baseline + causal composite checks.
    Uses [B, H, N_q, D] layout matching ferrotorch's flex_attention.
    """
    fixtures: list[dict[str, Any]] = []
    configs = [
        # (batch, heads, n_q, n_k, d, variant)
        (1, 2, 4, 4, 8, "baseline"),
        (2, 2, 4, 4, 8, "causal"),
        (1, 1, 4, 4, 8, "alibi"),
    ]
    dtype_name = "float32"
    dtype = torch_dtype(dtype_name)

    for batch, heads, n_q, n_k, d, variant in configs:
        seed = RNG_SEED + hash(variant) % 500
        q = seeded_randn(batch, heads, n_q, d, dtype=dtype, seed=seed)
        k = seeded_randn(batch, heads, n_k, d, dtype=dtype, seed=seed + 1)
        v = seeded_randn(batch, heads, n_k, d, dtype=dtype, seed=seed + 2)

        bias_t: torch.Tensor | None = None
        mask_t: torch.Tensor | None = None

        if variant == "causal":
            mask_t = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool))
        elif variant == "alibi":
            slope = 0.5
            rows = torch.arange(n_q, dtype=dtype).unsqueeze(1)
            cols = torch.arange(n_k, dtype=dtype).unsqueeze(0)
            bias_t = -slope * (rows - cols).abs()

        out = _manual_flex(q, k, v, bias=bias_t, mask=mask_t)

        row: dict[str, Any] = {
            "op": "flex_attention",
            "tag": f"nn_{variant}_{dtype_name}",
            "variant": variant,
            "dtype": dtype_name,
            "batch": batch,
            "heads": heads,
            "n_q": n_q,
            "n_k": n_k,
            "d": d,
            "q_data": to_listf(q),
            "k_data": to_listf(k),
            "v_data": to_listf(v),
            "expected_output": to_listf(out),
        }
        if bias_t is not None:
            row["bias"] = to_listf(bias_t)
        if mask_t is not None:
            row["mask_2d"] = [bool(b) for b in mask_t.reshape(-1).tolist()]
        fixtures.append(row)
    return fixtures


# ===========================================================================
# Module 4 — paged_attention (structural / shape checks)
# ===========================================================================

def fixture_paged_attention_structural() -> list[dict[str, Any]]:
    """
    PagedAttention structural checks: record expected shapes and values
    from a KV append + retrieval cycle. No neural-network numerics —
    these are deterministic raw-value copies.
    """
    fixtures: list[dict[str, Any]] = []

    page_size, num_heads, head_dim = 4, 2, 8
    num_tokens = 6  # will span 2 pages

    seed = RNG_SEED + 100
    g = torch.Generator()
    g.manual_seed(seed)
    k_data = torch.randn(num_tokens, num_heads, head_dim, generator=g)
    v_data = torch.randn(num_tokens, num_heads, head_dim, generator=g)

    fixtures.append({
        "op": "paged_kv_append_retrieve",
        "tag": "two_page_span",
        "dtype": "float32",
        "page_size": page_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_tokens": num_tokens,
        "k_data": to_listf(k_data),
        "v_data": to_listf(v_data),
        # ferrotorch's get_kv returns the concatenated K/V in token order —
        # expected == input for a freshly-built cache.
        "expected_k": to_listf(k_data),
        "expected_v": to_listf(v_data),
    })

    return fixtures


def fixture_paged_attention_manager() -> list[dict[str, Any]]:
    """PagedAttentionManager: multi-sequence append."""
    fixtures: list[dict[str, Any]] = []

    page_size, num_heads, head_dim, num_pages = 8, 2, 8, 4
    tokens_a, tokens_b = 5, 3

    g = torch.Generator()
    g.manual_seed(RNG_SEED + 200)
    ka = torch.randn(tokens_a, num_heads, head_dim, generator=g)
    va = torch.randn(tokens_a, num_heads, head_dim, generator=g)
    kb = torch.randn(tokens_b, num_heads, head_dim, generator=g)
    vb = torch.randn(tokens_b, num_heads, head_dim, generator=g)

    fixtures.append({
        "op": "paged_manager_multi_seq",
        "tag": "two_sequences",
        "dtype": "float32",
        "page_size": page_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_pages": num_pages,
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "ka_data": to_listf(ka),
        "va_data": to_listf(va),
        "kb_data": to_listf(kb),
        "vb_data": to_listf(vb),
        "expected_ka": to_listf(ka),
        "expected_va": to_listf(va),
        "expected_kb": to_listf(kb),
        "expected_vb": to_listf(vb),
    })
    return fixtures


# ===========================================================================
# Module 5 — transformer
# ===========================================================================

def _rope_apply_interleaved(
    x: torch.Tensor,
    seq_offset: int,
    base: float,
) -> torch.Tensor:
    """RoPE interleaved (pairs (x[2i], x[2i+1]))."""
    d = x.shape[-1]
    half = d // 2
    seq_len = x.shape[-2]

    thetas = torch.tensor([1.0 / (base ** (2 * i / d)) for i in range(half)], dtype=x.dtype)
    positions = torch.arange(seq_offset, seq_offset + seq_len, dtype=x.dtype)
    angles = positions.unsqueeze(1) * thetas.unsqueeze(0)  # [S, D/2]

    cos_v = angles.cos()
    sin_v = angles.sin()

    # Reshape for interleaved: x is [... , S, D]
    flat = x.reshape(*x.shape[:-1], half, 2)
    x0 = flat[..., 0]  # [... , S, D/2]
    x1 = flat[..., 1]

    out0 = x0 * cos_v - x1 * sin_v
    out1 = x0 * sin_v + x1 * cos_v
    return torch.stack([out0, out1], dim=-1).reshape(*x.shape)


def _rope_apply_half(
    x: torch.Tensor,
    seq_offset: int,
    base: float,
) -> torch.Tensor:
    """RoPE half-rotation (pairs (x[i], x[i+d/2]))."""
    d = x.shape[-1]
    half = d // 2
    seq_len = x.shape[-2]

    thetas = torch.tensor([1.0 / (base ** (2 * i / d)) for i in range(half)], dtype=x.dtype)
    positions = torch.arange(seq_offset, seq_offset + seq_len, dtype=x.dtype)
    angles = positions.unsqueeze(1) * thetas.unsqueeze(0)  # [S, D/2]

    cos_v = angles.cos()
    sin_v = angles.sin()

    x_first = x[..., :half]
    x_second = x[..., half:]

    out_first = x_first * cos_v - x_second * sin_v
    out_second = x_first * sin_v + x_second * cos_v
    return torch.cat([out_first, out_second], dim=-1)


def fixture_rope() -> list[dict[str, Any]]:
    """RotaryPositionEmbedding apply fixtures."""
    fixtures: list[dict[str, Any]] = []
    configs = [
        # (batch, seq, dim, convention, seq_offset)
        (1, 4, 8, "interleaved", 0),
        (2, 4, 8, "half_rotation", 0),
        (1, 3, 8, "interleaved", 2),  # seq_offset test
    ]
    dtype = torch.float32
    base = 10000.0

    for batch, seq, dim, convention, seq_offset in configs:
        seed = RNG_SEED + hash(convention + str(seq_offset)) % 300
        x = seeded_randn(batch, seq, dim, dtype=dtype, seed=seed)
        if convention == "interleaved":
            expected = _rope_apply_interleaved(x, seq_offset, base)
        else:
            expected = _rope_apply_half(x, seq_offset, base)

        fixtures.append({
            "op": "rope_apply",
            "tag": f"{convention}_offset{seq_offset}",
            "dtype": "float32",
            "batch": batch,
            "seq": seq,
            "dim": dim,
            "convention": convention,
            "seq_offset": seq_offset,
            "base": base,
            "input_data": to_listf(x),
            "expected_output": to_listf(expected),
        })
    return fixtures


def fixture_swiglu() -> list[dict[str, Any]]:
    """SwiGLU forward: silu(w1(x)) * w2(x), then w3."""
    fixtures: list[dict[str, Any]] = []
    in_features, hidden, batch, seq = 8, 16, 2, 4
    seed = RNG_SEED + 400

    g = torch.Generator()
    g.manual_seed(seed)
    dtype = torch.float32

    w1 = torch.randn(hidden, in_features, generator=g, dtype=dtype)
    w2 = torch.randn(hidden, in_features, generator=g, dtype=dtype)
    w3 = torch.randn(in_features, hidden, generator=g, dtype=dtype)
    b1 = torch.zeros(hidden, dtype=dtype)
    b2 = torch.zeros(hidden, dtype=dtype)
    b3 = torch.zeros(in_features, dtype=dtype)

    x = torch.randn(batch, seq, in_features, generator=g, dtype=dtype)

    gate = F.silu(x @ w1.T + b1)
    up = x @ w2.T + b2
    out = (gate * up) @ w3.T + b3

    fixtures.append({
        "op": "swiglu_forward",
        "tag": "basic",
        "dtype": "float32",
        "batch": batch,
        "seq": seq,
        "in_features": in_features,
        "hidden": hidden,
        "w1": to_listf(w1),
        "w2": to_listf(w2),
        "w3": to_listf(w3),
        "b1": to_listf(b1),
        "b2": to_listf(b2),
        "b3": to_listf(b3),
        "input_data": to_listf(x),
        "expected_output": to_listf(out),
    })
    return fixtures


def fixture_kvcache() -> list[dict[str, Any]]:
    """KVCache update + retrieval: verifies concatenation of past + new tokens."""
    fixtures: list[dict[str, Any]] = []
    max_seq, batch, num_heads, head_dim = 16, 1, 2, 8
    tokens_1, tokens_2 = 3, 2
    seed = RNG_SEED + 500

    g = torch.Generator()
    g.manual_seed(seed)
    dtype = torch.float32

    k1 = torch.randn(batch, tokens_1, num_heads, head_dim, generator=g, dtype=dtype)
    v1 = torch.randn(batch, tokens_1, num_heads, head_dim, generator=g, dtype=dtype)
    k2 = torch.randn(batch, tokens_2, num_heads, head_dim, generator=g, dtype=dtype)
    v2 = torch.randn(batch, tokens_2, num_heads, head_dim, generator=g, dtype=dtype)

    # After two updates: concatenation along seq dim
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)

    fixtures.append({
        "op": "kvcache_update_retrieve",
        "tag": "two_step",
        "dtype": "float32",
        "max_seq": max_seq,
        "batch": batch,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "tokens_1": tokens_1,
        "tokens_2": tokens_2,
        "k1_data": to_listf(k1),
        "v1_data": to_listf(v1),
        "k2_data": to_listf(k2),
        "v2_data": to_listf(v2),
        "expected_k": to_listf(k_cat),
        "expected_v": to_listf(v_cat),
    })
    return fixtures


def _manual_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / (var + eps).sqrt()
    return weight * x_norm + bias


def _manual_ffn(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    return F.relu(x @ w1.T + b1) @ w2.T + b2


def fixture_encoder_layer() -> list[dict[str, Any]]:
    """
    TransformerEncoderLayer: pre-norm forward.

    Architecture: norm1 -> self-attn -> residual -> norm2 -> ffn -> residual
    """
    fixtures: list[dict[str, Any]] = []
    batch, seq, d_model, nhead, d_ff = 2, 4, 16, 4, 32
    seed = RNG_SEED + 600
    dtype = torch.float32

    g = torch.Generator()
    g.manual_seed(seed)

    # Attention weights
    attn_w = _mha_seeded_weights(d_model, nhead, nhead, bias=False, seed=seed)
    attn_w = {k: v.to(dtype) for k, v in attn_w.items()}

    # Layer norm 1
    ln1_w = torch.ones(d_model, dtype=dtype)
    ln1_b = torch.zeros(d_model, dtype=dtype)

    # FFN weights
    ffn_w1 = torch.randn(d_ff, d_model, generator=g, dtype=dtype)
    ffn_b1 = torch.zeros(d_ff, dtype=dtype)
    ffn_w2 = torch.randn(d_model, d_ff, generator=g, dtype=dtype)
    ffn_b2 = torch.zeros(d_model, dtype=dtype)

    # Layer norm 2
    ln2_w = torch.ones(d_model, dtype=dtype)
    ln2_b = torch.zeros(d_model, dtype=dtype)

    x = torch.randn(batch, seq, d_model, generator=g, dtype=dtype)

    # Pre-norm forward
    normed = _manual_layer_norm(x, ln1_w, ln1_b)
    attn_out = _manual_mha_forward(normed, attn_w, d_model, nhead, nhead, causal_mask=False)
    x2 = x + attn_out

    normed2 = _manual_layer_norm(x2, ln2_w, ln2_b)
    ffn_out = _manual_ffn(normed2, ffn_w1, ffn_b1, ffn_w2, ffn_b2)
    out = x2 + ffn_out

    row: dict[str, Any] = {
        "op": "encoder_layer_forward",
        "tag": "prenorm_basic",
        "dtype": "float32",
        "batch": batch,
        "seq": seq,
        "d_model": d_model,
        "nhead": nhead,
        "d_ff": d_ff,
        "ln1_w": to_listf(ln1_w),
        "ln1_b": to_listf(ln1_b),
        "ffn_w1": to_listf(ffn_w1),
        "ffn_b1": to_listf(ffn_b1),
        "ffn_w2": to_listf(ffn_w2),
        "ffn_b2": to_listf(ffn_b2),
        "ln2_w": to_listf(ln2_w),
        "ln2_b": to_listf(ln2_b),
        "input_data": to_listf(x),
        "expected_output": to_listf(out),
    }
    for k, v in attn_w.items():
        row[f"attn_{k}"] = to_listf(v)
    fixtures.append(row)
    return fixtures


def fixture_decoder_layer() -> list[dict[str, Any]]:
    """
    TransformerDecoderLayer: forward_with_memory.

    Architecture (pre-norm decoder):
      norm1 -> self-attn(x, x, x) -> residual
      norm2 -> cross-attn(x, mem, mem) -> residual
      norm3 -> ffn -> residual
    """
    fixtures: list[dict[str, Any]] = []
    batch, seq_q, seq_k, d_model, nhead, d_ff = 1, 3, 5, 16, 4, 32
    seed = RNG_SEED + 700
    dtype = torch.float32

    g = torch.Generator()
    g.manual_seed(seed)

    attn_self_w = _mha_seeded_weights(d_model, nhead, nhead, bias=False, seed=seed)
    attn_cross_w = _mha_seeded_weights(d_model, nhead, nhead, bias=False, seed=seed + 1)
    attn_self_w = {k: v.to(dtype) for k, v in attn_self_w.items()}
    attn_cross_w = {k: v.to(dtype) for k, v in attn_cross_w.items()}

    ln1_w, ln1_b = torch.ones(d_model), torch.zeros(d_model)
    ln2_w, ln2_b = torch.ones(d_model), torch.zeros(d_model)
    ln3_w, ln3_b = torch.ones(d_model), torch.zeros(d_model)

    ffn_w1 = torch.randn(d_ff, d_model, generator=g, dtype=dtype)
    ffn_b1 = torch.zeros(d_ff)
    ffn_w2 = torch.randn(d_model, d_ff, generator=g, dtype=dtype)
    ffn_b2 = torch.zeros(d_model)

    x = torch.randn(batch, seq_q, d_model, generator=g, dtype=dtype)
    # memory has seq_k tokens but is broadcast-projected via cross-attn
    mem = torch.randn(batch, seq_k, d_model, generator=g, dtype=dtype)

    # --- pre-norm decoder step ---
    # self-attn block
    normed_x = _manual_layer_norm(x, ln1_w, ln1_b)
    self_attn_out = _manual_mha_forward(normed_x, attn_self_w, d_model, nhead, nhead, causal_mask=False)
    x2 = x + self_attn_out

    # cross-attn block — ferrotorch passes (query, key, value) = (x2, mem, mem)
    normed_x2 = _manual_layer_norm(x2, ln2_w, ln2_b)
    normed_mem = mem  # memory is not normalized in the simple decoder
    # For cross-attn: Q from decoder, K/V from encoder memory
    # We use a simplified cross-attn: project Q from normed_x2, K/V from normed_mem
    q_cross = normed_x2 @ attn_cross_w["q_proj"].T   # [B, sq, D]
    k_cross = normed_mem @ attn_cross_w["k_proj"].T   # [B, sk, D]
    v_cross = normed_mem @ attn_cross_w["v_proj"].T   # [B, sk, D]

    bsz = batch
    head_dim = d_model // nhead
    # [B, H, seq, dim]
    q_c = q_cross.reshape(bsz, seq_q, nhead, head_dim).permute(0, 2, 1, 3)
    k_c = k_cross.reshape(bsz, seq_k, nhead, head_dim).permute(0, 2, 1, 3)
    v_c = v_cross.reshape(bsz, seq_k, nhead, head_dim).permute(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(head_dim)
    cross_scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
    cross_weights = torch.softmax(cross_scores, dim=-1)
    cross_attn_val = torch.matmul(cross_weights, v_c).permute(0, 2, 1, 3).reshape(bsz, seq_q, d_model)
    cross_attn_out = cross_attn_val @ attn_cross_w["out_proj"].T
    x3 = x2 + cross_attn_out

    # ffn block
    normed_x3 = _manual_layer_norm(x3, ln3_w, ln3_b)
    ffn_out = _manual_ffn(normed_x3, ffn_w1, ffn_b1, ffn_w2, ffn_b2)
    out = x3 + ffn_out

    row: dict[str, Any] = {
        "op": "decoder_layer_forward_with_memory",
        "tag": "prenorm_basic",
        "dtype": "float32",
        "batch": batch,
        "seq_q": seq_q,
        "seq_k": seq_k,
        "d_model": d_model,
        "nhead": nhead,
        "d_ff": d_ff,
        "ln1_w": to_listf(ln1_w),
        "ln1_b": to_listf(ln1_b),
        "ln2_w": to_listf(ln2_w),
        "ln2_b": to_listf(ln2_b),
        "ln3_w": to_listf(ln3_w),
        "ln3_b": to_listf(ln3_b),
        "ffn_w1": to_listf(ffn_w1),
        "ffn_b1": to_listf(ffn_b1),
        "ffn_w2": to_listf(ffn_w2),
        "ffn_b2": to_listf(ffn_b2),
        "input_data": to_listf(x),
        "memory_data": to_listf(mem),
        "expected_output": to_listf(out),
    }
    for k, v in attn_self_w.items():
        row[f"self_attn_{k}"] = to_listf(v)
    for k, v in attn_cross_w.items():
        row[f"cross_attn_{k}"] = to_listf(v)
    fixtures.append(row)
    return fixtures


# ===========================================================================
# Module 6 — grad_fns (backward via ferrotorch-core re-exports)
# ===========================================================================

def fixture_softmax_backward() -> list[dict[str, Any]]:
    """softmax backward: grad_input from sum-reduction loss."""
    fixtures: list[dict[str, Any]] = []
    for dtype_name in ("float32", "float64"):
        dtype = torch_dtype(dtype_name)
        batch, seq, d = 2, 4, 8
        x = seeded_randn(batch, seq, d, dtype=dtype, seed=RNG_SEED + 800).requires_grad_(True)
        out = torch.softmax(x, dim=-1)
        loss = out.sum()
        loss.backward()
        fixtures.append({
            "op": "softmax_backward",
            "tag": f"sum_loss_{dtype_name}",
            "dtype": dtype_name,
            "batch": batch,
            "seq": seq,
            "d": d,
            "input_data": to_listf(x.detach()),
            "grad_input": to_listf(x.grad),
        })
    return fixtures


def fixture_layernorm_backward() -> list[dict[str, Any]]:
    """layer_norm backward: grad_input, grad_weight, grad_bias."""
    fixtures: list[dict[str, Any]] = []
    dtype_name = "float32"
    dtype = torch_dtype(dtype_name)
    batch, seq, d = 2, 3, 8
    x = seeded_randn(batch, seq, d, dtype=dtype, seed=RNG_SEED + 900).requires_grad_(True)
    w = torch.ones(d, dtype=dtype, requires_grad=True)
    b = torch.zeros(d, dtype=dtype, requires_grad=True)
    out = F.layer_norm(x, [d], w, b, eps=1e-5)
    loss = out.sum()
    loss.backward()
    fixtures.append({
        "op": "layernorm_backward",
        "tag": f"sum_loss_{dtype_name}",
        "dtype": dtype_name,
        "batch": batch,
        "seq": seq,
        "d": d,
        "input_data": to_listf(x.detach()),
        "weight_data": to_listf(w.detach()),
        "bias_data": to_listf(b.detach()),
        "grad_input": to_listf(x.grad),
        "grad_weight": to_listf(w.grad),
        "grad_bias": to_listf(b.grad),
    })
    return fixtures


def fixture_cross_entropy_backward() -> list[dict[str, Any]]:
    """cross_entropy backward: grad_input via mean-reduction loss."""
    fixtures: list[dict[str, Any]] = []
    dtype_name = "float32"
    dtype = torch_dtype(dtype_name)
    batch, num_classes = 4, 8
    x = seeded_randn(batch, num_classes, dtype=dtype, seed=RNG_SEED + 1000).requires_grad_(True)
    g = torch.Generator()
    g.manual_seed(RNG_SEED + 1001)
    targets = torch.randint(0, num_classes, (batch,), generator=g)
    loss = F.cross_entropy(x, targets, reduction="mean")
    loss.backward()
    fixtures.append({
        "op": "cross_entropy_backward",
        "tag": f"mean_{dtype_name}",
        "dtype": dtype_name,
        "batch": batch,
        "num_classes": num_classes,
        "input_data": to_listf(x.detach()),
        "targets": targets.tolist(),
        "grad_input": to_listf(x.grad),
    })
    return fixtures


# ===========================================================================
# Top-level entry
# ===========================================================================

def main() -> int:
    all_fixtures: list[dict[str, Any]] = []

    # Module 1 — MultiheadAttention
    all_fixtures.extend(fixture_mha_small_forward())
    all_fixtures.extend(fixture_mha_causal_forward())
    all_fixtures.extend(fixture_mha_gqa_forward())
    all_fixtures.extend(fixture_reshape_to_heads())
    all_fixtures.extend(fixture_repeat_kv())

    # Module 2 — flash_attention / standard_attention
    all_fixtures.extend(fixture_flash_attention())
    all_fixtures.extend(fixture_standard_attention())

    # Module 3 — flex_attention composite
    all_fixtures.extend(fixture_flex_attention())

    # Module 4 — paged_attention structural
    all_fixtures.extend(fixture_paged_attention_structural())
    all_fixtures.extend(fixture_paged_attention_manager())

    # Module 5 — transformer
    all_fixtures.extend(fixture_rope())
    all_fixtures.extend(fixture_swiglu())
    all_fixtures.extend(fixture_kvcache())
    all_fixtures.extend(fixture_encoder_layer())
    all_fixtures.extend(fixture_decoder_layer())

    # Module 6 — grad_fns
    all_fixtures.extend(fixture_softmax_backward())
    all_fixtures.extend(fixture_layernorm_backward())
    all_fixtures.extend(fixture_cross_entropy_backward())

    payload: dict[str, Any] = {
        "metadata": fixture_metadata(),
        "fixtures": all_fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"wrote {len(all_fixtures)} fixtures → {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
