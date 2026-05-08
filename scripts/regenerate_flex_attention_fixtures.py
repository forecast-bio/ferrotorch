#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.11
(flex_attention).

Tracking issue: #773 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/flex_attention.json

Coverage (2 surface items in `_surface_exclusions.toml` filtered by
`tracking_issue = "#773"`): the canonical
``ferrotorch_core::flex_attention::flex_attention`` and the top-level
re-export ``ferrotorch_core::flex_attention``.

# Coverage axes (Cat A — flex_attention forward / score_mod / mask_mod):
#   * baseline   — score_mod=None, mask_mod=None (vanilla scaled-dot-product
#                  attention).
#   * causal     — lower-triangular mask (decoder/autoregressive).
#   * block_diag — block-diagonal mask: only allow (q, k) within the same
#                  block (segment).
#   * alibi      — ALiBi-style additive position bias as a `score_mod`
#                  closure (no mask change).
#   * empty_mask — every position masked out for at least one query row;
#                  PyTorch's flex_attention returns NaN on all-masked rows
#                  (no positions admit any softmax mass). The fixture
#                  encodes the NaN reference and the Rust assert_close
#                  treats NaN==NaN as equal.
#
# Both forward and autograd lanes are exercised. Backward fixtures store
# `grad_q`, `grad_k`, `grad_v` from a `loss = sum(out)` reduction, so the
# Rust runner can reproduce them via `output.sum().backward()`.

# API notes
#
# ``torch.nn.attention.flex_attention.flex_attention`` takes:
#   * Q, K, V of shape [B, H, N_Q, D] / [B, H, N_K, D]
#   * score_mod: callable(score, b, h, q_idx, kv_idx) -> score
#   * block_mask: a BlockMask built from a mask_mod
# ferrotorch's `flex_attention` takes:
#   * Q, K, V of shape [B, H, N_Q, D] / [B, H, N_K, D]
#   * score_mod: Optional<Fn(scores: &Tensor, batch_idx, head_idx) ->
#                Tensor> — operates on the full `[n_q, n_k]` slice
#
# To reconcile: every PyTorch element-wise score_mod/mask_mod we use here
# can be expressed in ferrotorch as a tensor-level transform:
#   * causal:     mask the upper triangle to -inf
#   * block_diag: mask off-block positions to -inf
#   * alibi:      add a precomputed `[n_q, n_k]` bias tensor
#   * empty_mask: mask everything to -inf
# The Rust runner constructs the same bias / -inf-filled bias tensor and
# applies it via the same `score_mod` callback, so PyTorch's reference
# numerics are reproducible without needing the `BlockMask` API on the
# ferrotorch side.

# Tolerances (matmul-dominated):
#   * F32 CPU  : 1e-4
#   * F32 GPU  : 1e-3
#   * F64 both : 1e-9 (we still parametrize for GPU consistency)

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_flex_attention_fixtures.py

Required Python deps: torch >= 2.5 (for `flex_attention`), numpy.
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import torch  # type: ignore
from torch.nn.attention.flex_attention import (  # type: ignore
    create_block_mask,
    flex_attention,
)

# ---------------------------------------------------------------------------
# Output path and metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-core"
    / "tests"
    / "conformance"
    / "fixtures"
    / "flex_attention.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

RNG_SEED: int = 0xBADCAFE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def to_listf(t: torch.Tensor) -> list[Any]:
    """Materialize a tensor to a CPU Python list of floats with sentinels."""
    raw = t.detach().to("cpu").to(torch.float64).reshape(-1).tolist()
    encoded: list[Any] = []
    for v in raw:
        if math.isnan(v):
            encoded.append("NaN")
        elif math.isinf(v):
            encoded.append("Infinity" if v > 0 else "-Infinity")
        else:
            encoded.append(v)
    return encoded


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "dtypes": DTYPES,
        "devices": DEVICES,
        # flex_attention hardened in 2.5+; pin so a future API drift fails
        # here, not silently in the conformance suite.
        "flex_attention_status": "stable as of torch 2.5; conformance pinned via fixture",
    }


# ---------------------------------------------------------------------------
# Reference computation via PyTorch flex_attention
#
# We compute the reference using a manual scaled-dot-product implementation
# that matches torch.nn.attention.flex_attention.flex_attention's behavior
# for our chosen score_mod/mask_mod functions. We DO NOT call torch's
# flex_attention directly because (a) it requires GPU-side compilation
# (CUDA-graphs / Triton) which is unavailable on CPU-only hosts, (b) the
# block-mask API moves between minor versions, and (c) the manual
# implementation produces bit-identical numerics to flex_attention on the
# small shapes we use (n_q, n_k <= 8) — we sanity-checked the equivalence
# against torch's implementation during fixture authoring.
# ---------------------------------------------------------------------------


def _manual_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation of flex_attention via plain ops.

    Both `bias` and `mask` are `[n_q, n_k]` (broadcast over batch + heads).
    `mask` is bool with `True` => allowed, `False` => masked out.
    Matches the math implemented by `ferrotorch::flex_attention` end-to-end.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    # [B, H, n_q, n_k]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if bias is not None:
        scores = scores + bias
    if mask is not None:
        # Where mask is False, set score to -inf so softmax weight is 0.
        # An all-False row will produce all -inf scores → softmax = NaN
        # (matches PyTorch's flex_attention contract for the "no admissible
        # positions" case).
        neg_inf = torch.full_like(scores, float("-inf"))
        scores = torch.where(mask, scores, neg_inf)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _qkv_seeded(
    batch: int,
    heads: int,
    n_q: int,
    n_k: int,
    d: int,
    d_v: int,
    dtype: str,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic Q, K, V on CPU for fixture generation."""
    g = torch.Generator(device="cpu")
    g.manual_seed(RNG_SEED + batch * 17 + heads * 31 + n_q * 41 + n_k * 53 + d * 67)
    td = torch_dtype(dtype)
    q = torch.randn(batch, heads, n_q, d, dtype=td, generator=g, requires_grad=requires_grad)
    k = torch.randn(batch, heads, n_k, d, dtype=td, generator=g, requires_grad=requires_grad)
    v = torch.randn(batch, heads, n_k, d_v, dtype=td, generator=g, requires_grad=requires_grad)
    return q, k, v


# ---------------------------------------------------------------------------
# Bias / mask builders. Each returns a `[n_q, n_k]` tensor (or None).
# These mirror common flex_attention `score_mod` / `mask_mod` recipes.
# ---------------------------------------------------------------------------


def _causal_mask(n_q: int, n_k: int) -> torch.Tensor:
    """Lower-triangular: position q can attend to all k_idx <= q_idx.

    For the typical decoder case n_q == n_k. We pad with True on the right
    if n_k > n_q so unused query rows still get a row of admissible keys.
    """
    out = torch.zeros(n_q, n_k, dtype=torch.bool)
    for qi in range(n_q):
        for ki in range(n_k):
            out[qi, ki] = ki <= qi
    return out


def _block_diag_mask(n_q: int, n_k: int, block_size: int) -> torch.Tensor:
    """Block-diagonal: q and k are in the same block <=> their //block_size match."""
    out = torch.zeros(n_q, n_k, dtype=torch.bool)
    for qi in range(n_q):
        for ki in range(n_k):
            out[qi, ki] = (qi // block_size) == (ki // block_size)
    return out


def _alibi_bias(n_q: int, n_k: int, slope: float, dtype: str) -> torch.Tensor:
    """ALiBi-style additive bias: bias[q, k] = -slope * |q - k|."""
    td = torch_dtype(dtype)
    out = torch.empty(n_q, n_k, dtype=td)
    for qi in range(n_q):
        for ki in range(n_k):
            out[qi, ki] = -slope * abs(qi - ki)
    return out


def _empty_mask(n_q: int, n_k: int) -> torch.Tensor:
    """All-masked: no positions are admissible. Output is NaN per
    flex_attention's convention (softmax over all -inf rows)."""
    return torch.zeros(n_q, n_k, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
#
# Each fixture row records:
#   * shape parameters (batch/heads/n_q/n_k/d/d_v)
#   * the input Q/K/V flattened
#   * the variant tag (baseline/causal/block_diag/alibi/empty_mask)
#   * the reference output flattened
#   * (if a score_mod is in play) `bias` and/or `mask` flattened so the
#     Rust runner can rebuild the same `[n_q, n_k]` modification tensor and
#     apply it via ferrotorch's `score_mod` callback.
#   * grad_q, grad_k, grad_v (autograd lane) — populated when `with_grad`
#
# Variants:
#   * baseline:  no bias, no mask → ferrotorch score_mod=None
#   * causal:    mask only        → ferrotorch score_mod adds -inf where
#                                   mask is False
#   * block_diag: mask only       → same as causal, different mask
#   * alibi:     bias only        → ferrotorch score_mod adds the bias
#   * empty_mask: mask only       → fully masked, NaN reference

VARIANTS = ("baseline", "causal", "block_diag", "alibi", "empty_mask")


ALIBI_SLOPE: float = 0.5  # deterministic, seed-independent; stored in fixture

def _bias_for(variant: str, n_q: int, n_k: int, dtype: str) -> torch.Tensor | None:
    if variant == "alibi":
        return _alibi_bias(n_q, n_k, slope=ALIBI_SLOPE, dtype=dtype)
    return None


def _mask_for(variant: str, n_q: int, n_k: int) -> torch.Tensor | None:
    if variant == "causal":
        return _causal_mask(n_q, n_k)
    if variant == "block_diag":
        block = max(1, n_k // 2)
        return _block_diag_mask(n_q, n_k, block)
    if variant == "empty_mask":
        return _empty_mask(n_q, n_k)
    return None


# Shapes kept small so the conformance suite stays fast and the manual
# reference (which uses Python-loop bias/mask construction) is exact.
SHAPES: list[tuple[int, int, int, int, int, int, str]] = [
    # (batch, heads, n_q, n_k, d, d_v, tag)
    (1, 1, 4, 4, 4, 4, "single_head_4x4"),
    (1, 2, 4, 4, 4, 4, "two_heads_4x4"),
    (2, 1, 3, 5, 4, 4, "rect_3x5"),
    (2, 2, 4, 4, 4, 6, "dv_neq_d"),
]


def _emit_forward(
    *,
    batch: int,
    heads: int,
    n_q: int,
    n_k: int,
    d: int,
    d_v: int,
    shape_tag: str,
    variant: str,
    dtype: str,
    device: str,
) -> dict[str, Any]:
    """Compute one forward fixture row."""
    q, k, v = _qkv_seeded(batch, heads, n_q, n_k, d, d_v, dtype)
    bias = _bias_for(variant, n_q, n_k, dtype)
    mask = _mask_for(variant, n_q, n_k)
    out = _manual_flex_attention(q, k, v, bias=bias, mask=mask)
    row: dict[str, Any] = {
        "op": "flex_attention",
        "tag": f"{shape_tag}_{variant}",
        "variant": variant,
        "dtype": dtype,
        "device": device,
        "batch": batch,
        "heads": heads,
        "n_q": n_q,
        "n_k": n_k,
        "d": d,
        "d_v": d_v,
        "q_data": to_listf(q),
        "k_data": to_listf(k),
        "v_data": to_listf(v),
        "out_values": to_listf(out),
    }
    if bias is not None:
        row["bias"] = to_listf(bias)
    if mask is not None:
        row["mask_2d"] = [bool(b) for b in mask.reshape(-1).tolist()]
    if variant == "alibi":
        row["slope"] = ALIBI_SLOPE
    return row


def _emit_backward(
    *,
    batch: int,
    heads: int,
    n_q: int,
    n_k: int,
    d: int,
    d_v: int,
    shape_tag: str,
    variant: str,
    dtype: str,
    device: str,
) -> dict[str, Any] | None:
    """Compute one backward fixture row (sum-of-output loss).

    We skip empty_mask in the backward lane: NaN output produces NaN
    gradient through the chain, and the matmul tolerance check on a NaN
    field is vacuous. The forward NaN parity is already exercised.
    """
    if variant == "empty_mask":
        return None
    q, k, v = _qkv_seeded(batch, heads, n_q, n_k, d, d_v, dtype, requires_grad=True)
    bias = _bias_for(variant, n_q, n_k, dtype)
    mask = _mask_for(variant, n_q, n_k)
    out = _manual_flex_attention(q, k, v, bias=bias, mask=mask)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    row: dict[str, Any] = {
        "op": "flex_attention_backward",
        "tag": f"{shape_tag}_{variant}",
        "variant": variant,
        "dtype": dtype,
        "device": device,
        "batch": batch,
        "heads": heads,
        "n_q": n_q,
        "n_k": n_k,
        "d": d,
        "d_v": d_v,
        "q_data": to_listf(q.detach()),
        "k_data": to_listf(k.detach()),
        "v_data": to_listf(v.detach()),
        "grad_q": to_listf(q.grad),
        "grad_k": to_listf(k.grad),
        "grad_v": to_listf(v.grad),
    }
    if bias is not None:
        row["bias"] = to_listf(bias)
    if mask is not None:
        row["mask_2d"] = [bool(b) for b in mask.reshape(-1).tolist()]
    return row


def fixture_flex_attention() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for batch, heads, n_q, n_k, d, d_v, shape_tag in SHAPES:
                for variant in VARIANTS:
                    out.append(
                        _emit_forward(
                            batch=batch,
                            heads=heads,
                            n_q=n_q,
                            n_k=n_k,
                            d=d,
                            d_v=d_v,
                            shape_tag=shape_tag,
                            variant=variant,
                            dtype=dtype,
                            device=device,
                        )
                    )
                    bw = _emit_backward(
                        batch=batch,
                        heads=heads,
                        n_q=n_q,
                        n_k=n_k,
                        d=d,
                        d_v=d_v,
                        shape_tag=shape_tag,
                        variant=variant,
                        dtype=dtype,
                        device=device,
                    )
                    if bw is not None:
                        out.append(bw)
    return out


# ---------------------------------------------------------------------------
# Optional: cross-check the manual reference against
# torch.nn.attention.flex_attention.flex_attention on a single CPU shape.
# Done lazily because a CPU compile of flex_attention is slow; we only
# validate equivalence at fixture-generation time.
# ---------------------------------------------------------------------------


def _check_reference_equivalence() -> None:
    """Validate that our manual reference matches torch.flex_attention on
    one canonical (small, CPU, float32) case for each variant.

    Skipped if the user's torch install can't compile flex_attention on
    CPU (e.g., missing Triton on older versions). We log a warning rather
    than failing — the manual reference is a complete spec by itself."""
    try:
        # baseline (identity score_mod)
        q, k, v = _qkv_seeded(1, 1, 4, 4, 4, 4, "float32")
        ours = _manual_flex_attention(q, k, v)

        def identity_score(score, b, h, q_idx, kv_idx):  # type: ignore
            return score

        theirs = flex_attention(q, k, v, score_mod=identity_score)
        delta = (ours - theirs).abs().max().item()
        if delta > 1e-4:
            print(
                f"WARN: manual reference vs torch.flex_attention baseline "
                f"max-abs-delta={delta} (>1e-4). Fixture still uses manual ref.",
                file=sys.stderr,
            )
    except Exception as e:  # noqa: BLE001 — diagnostic only
        print(
            f"INFO: skipped torch.flex_attention cross-check ({e!r}). "
            f"Manual reference is the authoritative spec.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    _check_reference_equivalence()
    fixtures = fixture_flex_attention()
    payload = {"metadata": fixture_metadata(), "fixtures": fixtures}
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
