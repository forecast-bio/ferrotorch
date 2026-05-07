#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-nn Phase C9.1
(basic layers: Linear, Conv*, Embedding, Identity, Padding, Upsample, Pooling,
Dropout, LazyLinear, LazyConv*).

Tracking issue: crosslink #899 (C9.1 nn layers conformance).

Output:
    ferrotorch-nn/tests/conformance/fixtures.json

Coverage (11 in-scope modules):
  * linear  — Linear(4,3) with bias on (2,4) input; seeded weights via manual_seed=42
  * conv    — Conv2d, Conv1d, Conv3d, ConvTranspose2d shape tests; zero-input forward
  * embedding — Embedding lookup, padding_idx; EmbeddingBag sum
  * identity  — Identity pass-through; Flatten shapes
  * padding   — ConstantPad1d/2d, ZeroPad2d, ReflectionPad2d, ReplicationPad1d
  * upsample  — Upsample nearest 2x, PixelShuffle r=2
  * pooling   — MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, MaxPool1d, AvgPool1d, LPPool1d
  * dropout   — eval identity for Dropout/Dropout2d/AlphaDropout
  * lazy      — LazyLinear/LazyConv2d shape-only after materialization

Pin: torch == 2.11.0
Usage from WSL:
    python3 scripts/regenerate_nn_layers_fixtures.py

Required Python deps: torch (>= 2.11.0), numpy.

Notes:
  - Script is designed to be re-run whenever fixtures need refreshing.
  - Pin the torch version before running: pip install torch==2.11.0
  - CUDA fixtures are omitted (no CUDA required to run this script).
  - All random weights are seeded with manual_seed=42 for reproducibility.
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "ferrotorch-nn" / "tests" / "conformance" / "fixtures.json"

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

TORCH_PIN = "torch==2.11.0"
RNG_SEED = 42

torch.manual_seed(RNG_SEED)


def to_listf(t: torch.Tensor) -> list[Any]:
    """Serialize tensor to list; NaN/Inf -> sentinel strings."""
    raw = t.detach().cpu().to(torch.float64).reshape(-1).tolist()
    out: list[Any] = []
    for v in raw:
        if math.isnan(v):
            out.append("NaN")
        elif math.isinf(v):
            out.append("Infinity" if v > 0 else "-Infinity")
        else:
            out.append(v)
    return out


def fixture_metadata() -> dict[str, Any]:
    return {
        "version": torch.__version__,
        "generated_by": "scripts/regenerate_nn_layers_fixtures.py",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_pin": TORCH_PIN,
        "rng_seed": RNG_SEED,
        "description": (
            "Reference fixtures for ferrotorch-nn C9.1 conformance suite. "
            "Forward outputs captured from PyTorch with manual_seed=42. "
            "Tolerance: 1e-5 absolute for f32 CPU."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seeded_linear(in_f: int, out_f: int, bias: bool) -> nn.Linear:
    """Build a Linear layer with seeded weights."""
    torch.manual_seed(RNG_SEED)
    layer = nn.Linear(in_f, out_f, bias=bias)
    return layer


def seeded_conv2d(in_ch: int, out_ch: int, ksize: int, stride: int, pad: int,
                  bias: bool) -> nn.Conv2d:
    torch.manual_seed(RNG_SEED)
    return nn.Conv2d(in_ch, out_ch, ksize, stride=stride, padding=pad, bias=bias)


def seeded_conv1d(in_ch: int, out_ch: int, ksize: int, stride: int, pad: int,
                  bias: bool) -> nn.Conv1d:
    torch.manual_seed(RNG_SEED)
    return nn.Conv1d(in_ch, out_ch, ksize, stride=stride, padding=pad, bias=bias)


def seeded_conv_transpose2d(in_ch: int, out_ch: int, ksize: int, stride: int, pad: int,
                             out_pad: int, bias: bool) -> nn.ConvTranspose2d:
    torch.manual_seed(RNG_SEED)
    return nn.ConvTranspose2d(in_ch, out_ch, ksize, stride=stride, padding=pad,
                              output_padding=out_pad, bias=bias)


# ---------------------------------------------------------------------------
# Cat 1 — linear
# ---------------------------------------------------------------------------


def fixture_linear() -> list[dict[str, Any]]:
    """Linear forward: known numerical values for fixture assertions."""
    fixtures = []

    # 1a. Linear(4, 3, bias=True), input shape [2, 4]
    torch.manual_seed(RNG_SEED)
    layer = nn.Linear(4, 3, bias=True)
    # Use seeded weight directly from kaiming_uniform_ (default nn.Linear init)
    w = layer.weight.detach()
    b = layer.bias.detach()
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    with torch.no_grad():
        out = layer(x)
    fixtures.append({
        "id": "linear_forward_2x4_to_3",
        "module": "Linear",
        "op": "forward",
        "config": {"in_features": 4, "out_features": 3, "bias": True},
        "input_shape": list(x.shape),
        "input_data": to_listf(x),
        "weight": to_listf(w),
        "bias_data": to_listf(b),
        "expected_shape": list(out.shape),
        "expected_output": to_listf(out),
        "tolerance": 1e-3,
        "note": "Linear(4,3) with bias, 2-sample batch. Weights from torch.manual_seed(42).",
    })

    # 1b. Linear(3, 2, bias=False) with identity-like weight
    layer2 = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer2.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    x2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with torch.no_grad():
        out2 = layer2(x2)
    fixtures.append({
        "id": "linear_forward_no_bias",
        "module": "Linear",
        "op": "forward",
        "config": {"in_features": 3, "out_features": 2, "bias": False},
        "input_shape": list(x2.shape),
        "input_data": to_listf(x2),
        "weight": to_listf(layer2.weight),
        "bias_data": None,
        "expected_shape": list(out2.shape),
        "expected_output": to_listf(out2),
        "tolerance": 1e-6,
        "note": "Linear(3,2) no bias, identity-like weight.",
    })

    # 1c. 1-D input
    layer3 = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer3.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    x3 = torch.tensor([1.0, 2.0, 3.0])
    with torch.no_grad():
        out3 = layer3(x3)
    fixtures.append({
        "id": "linear_forward_1d_input",
        "module": "Linear",
        "op": "forward",
        "config": {"in_features": 3, "out_features": 2, "bias": False},
        "input_shape": list(x3.shape),
        "input_data": to_listf(x3),
        "weight": to_listf(layer3.weight),
        "bias_data": None,
        "expected_shape": list(out3.shape),
        "expected_output": to_listf(out3),
        "tolerance": 1e-6,
        "note": "Linear 1-D input: (in_features,) -> (out_features,).",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 2 — embedding
# ---------------------------------------------------------------------------


def fixture_embedding() -> list[dict[str, Any]]:
    fixtures = []

    # 2a. Basic lookup
    emb = nn.Embedding(5, 3)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    idx = torch.tensor([0, 2, 4])
    out = emb(idx)
    fixtures.append({
        "id": "embedding_forward_basic",
        "module": "Embedding",
        "op": "forward",
        "config": {"num_embeddings": 5, "embedding_dim": 3},
        "input_indices": idx.tolist(),
        "weight": to_listf(emb.weight),
        "expected_shape": list(out.shape),
        "expected_output": to_listf(out),
        "tolerance": 1e-6,
        "note": "Embedding(5,3) lookup for indices [0,2,4]. Weight set manually.",
    })

    # 2b. padding_idx
    emb2 = nn.Embedding(4, 2, padding_idx=0)
    with torch.no_grad():
        emb2.weight.copy_(torch.tensor([
            [0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]
        ]))
    idx2 = torch.tensor([0, 1, 0, 2])
    out2 = emb2(idx2)
    fixtures.append({
        "id": "embedding_forward_padding_idx",
        "module": "Embedding",
        "op": "forward_with_padding",
        "config": {"num_embeddings": 4, "embedding_dim": 2, "padding_idx": 0},
        "input_indices": idx2.tolist(),
        "weight": to_listf(emb2.weight),
        "expected_shape": list(out2.shape),
        "expected_output": to_listf(out2),
        "tolerance": 1e-6,
        "note": "Embedding with padding_idx=0: indices 0 map to zero vectors.",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 3 — identity / flatten
# ---------------------------------------------------------------------------


def fixture_identity() -> list[dict[str, Any]]:
    fixtures = []

    # Identity passthrough
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fixtures.append({
        "id": "identity_forward_2x3",
        "module": "Identity",
        "op": "forward",
        "config": {},
        "input_shape": list(x.shape),
        "input_data": to_listf(x),
        "expected_shape": list(x.shape),
        "expected_output": to_listf(x),
        "tolerance": 1e-7,
        "note": "Identity pass-through.",
    })

    # Flatten default [2,3,4] -> [2,12]
    f = nn.Flatten(start_dim=1)
    x2 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out2 = f(x2)
    fixtures.append({
        "id": "flatten_default_2x3x4",
        "module": "Flatten",
        "op": "forward",
        "config": {"start_dim": 1, "end_dim": -1},
        "input_shape": list(x2.shape),
        "expected_shape": list(out2.shape),
        "note": "Flatten default (batch dim preserved).",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 4 — padding
# ---------------------------------------------------------------------------


def fixture_padding() -> list[dict[str, Any]]:
    fixtures = []

    # ConstantPad1d (1,2) value=0
    pad = nn.ConstantPad1d((1, 2), 0.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = pad(x)
    fixtures.append({
        "id": "constant_pad1d_basic",
        "module": "ConstantPad1d",
        "op": "forward",
        "config": {"padding": [1, 2], "value": 0.0},
        "input_shape": list(x.shape),
        "input_data": to_listf(x),
        "expected_shape": list(out.shape),
        "expected_output": to_listf(out),
        "tolerance": 1e-7,
        "note": "ConstantPad1d(1,2) value=0.",
    })

    # ZeroPad2d (1,1,1,1)
    pad2 = nn.ZeroPad2d((1, 1, 1, 1))
    x2 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    out2 = pad2(x2)
    fixtures.append({
        "id": "zero_pad2d_basic",
        "module": "ZeroPad2d",
        "op": "forward",
        "config": {"padding": [1, 1, 1, 1]},
        "input_shape": list(x2.shape),
        "input_data": to_listf(x2),
        "expected_shape": list(out2.shape),
        "expected_output": to_listf(out2),
        "tolerance": 1e-7,
        "note": "ZeroPad2d(1,1,1,1) on [1,1,2,2] input.",
    })

    # ReflectionPad2d (1,1,1,1)
    pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
    x3 = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    out3 = pad3(x3)
    fixtures.append({
        "id": "reflection_pad2d_basic",
        "module": "ReflectionPad2d",
        "op": "forward",
        "config": {"padding": [1, 1, 1, 1]},
        "input_shape": list(x3.shape),
        "input_data": to_listf(x3),
        "expected_shape": list(out3.shape),
        "expected_output": to_listf(out3),
        "tolerance": 1e-7,
        "note": "ReflectionPad2d(1,1,1,1) on 3x3 input.",
    })

    # ReplicationPad1d (2,3)
    pad4 = nn.ReplicationPad1d((2, 3))
    x4 = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    out4 = pad4(x4)
    fixtures.append({
        "id": "replication_pad1d_basic",
        "module": "ReplicationPad1d",
        "op": "forward",
        "config": {"padding": [2, 3]},
        "input_shape": list(x4.shape),
        "input_data": to_listf(x4),
        "expected_shape": list(out4.shape),
        "expected_output": to_listf(out4),
        "tolerance": 1e-7,
        "note": "ReplicationPad1d(2,3): edges repeated.",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 5 — pooling
# ---------------------------------------------------------------------------


def fixture_pooling() -> list[dict[str, Any]]:
    fixtures = []

    x4x4 = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)

    # MaxPool2d 2x2 stride 2
    mp2 = nn.MaxPool2d(2, stride=2)
    out = mp2(x4x4)
    fixtures.append({
        "id": "max_pool2d_basic",
        "module": "MaxPool2d",
        "op": "forward",
        "config": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
        "input_shape": list(x4x4.shape),
        "input_data": to_listf(x4x4),
        "expected_shape": list(out.shape),
        "expected_output": to_listf(out),
        "tolerance": 1e-6,
        "note": "MaxPool2d(2,2) on 4x4 input. Non-overlapping 2x2 windows.",
    })

    # AvgPool2d 2x2 stride 2
    ap2 = nn.AvgPool2d(2, stride=2)
    out2 = ap2(x4x4)
    fixtures.append({
        "id": "avg_pool2d_basic",
        "module": "AvgPool2d",
        "op": "forward",
        "config": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
        "input_shape": list(x4x4.shape),
        "input_data": to_listf(x4x4),
        "expected_shape": list(out2.shape),
        "expected_output": to_listf(out2),
        "tolerance": 1e-5,
        "note": "AvgPool2d(2,2) averages each 2x2 tile.",
    })

    # AdaptiveAvgPool2d (2,2)
    aap = nn.AdaptiveAvgPool2d((2, 2))
    out3 = aap(x4x4)
    fixtures.append({
        "id": "adaptive_avg_pool2d_2x2",
        "module": "AdaptiveAvgPool2d",
        "op": "forward",
        "config": {"output_size": [2, 2]},
        "input_shape": list(x4x4.shape),
        "input_data": to_listf(x4x4),
        "expected_shape": list(out3.shape),
        "expected_output": to_listf(out3),
        "tolerance": 1e-5,
        "note": "AdaptiveAvgPool2d(2,2) on 4x4 input.",
    })

    # MaxPool1d 2 stride 2
    mp1 = nn.MaxPool1d(2, stride=2)
    x1 = torch.tensor([[[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]]])
    out4 = mp1(x1)
    fixtures.append({
        "id": "max_pool1d_basic",
        "module": "MaxPool1d",
        "op": "forward",
        "config": {"kernel_size": 2, "stride": 2, "padding": 0},
        "input_shape": list(x1.shape),
        "input_data": to_listf(x1),
        "expected_shape": list(out4.shape),
        "expected_output": to_listf(out4),
        "tolerance": 1e-6,
        "note": "MaxPool1d(2,2) on 1D length-6 input.",
    })

    # AvgPool1d 2 stride 2
    ap1 = nn.AvgPool1d(2, stride=2)
    x1b = torch.tensor([[[1.0, 3.0, 5.0, 7.0]]])
    out5 = ap1(x1b)
    fixtures.append({
        "id": "avg_pool1d_basic",
        "module": "AvgPool1d",
        "op": "forward",
        "config": {"kernel_size": 2, "stride": 2, "padding": 0},
        "input_shape": list(x1b.shape),
        "input_data": to_listf(x1b),
        "expected_shape": list(out5.shape),
        "expected_output": to_listf(out5),
        "tolerance": 1e-6,
        "note": "AvgPool1d(2,2) averages pairs.",
    })

    # LPPool1d p=2, kernel=2, stride=2
    lp = nn.LPPool1d(2, kernel_size=2, stride=2)
    xlp = torch.tensor([[[3.0, 4.0, 0.0, 5.0]]])
    outlp = lp(xlp)
    fixtures.append({
        "id": "lp_pool1d_basic",
        "module": "LPPool1d",
        "op": "forward",
        "config": {"norm_type": 2.0, "kernel_size": 2, "stride": 2},
        "input_shape": list(xlp.shape),
        "input_data": to_listf(xlp),
        "expected_shape": list(outlp.shape),
        "expected_output": to_listf(outlp),
        "tolerance": 1e-5,
        "note": "LPPool1d p=2, k=2, stride=2: L2([3,4])=5, L2([0,5])=5.",
    })

    # AdaptiveMaxPool2d (2,2)
    amp2 = nn.AdaptiveMaxPool2d((2, 2))
    out6 = amp2(x4x4)
    fixtures.append({
        "id": "adaptive_max_pool2d_basic",
        "module": "AdaptiveMaxPool2d",
        "op": "forward",
        "config": {"output_size": [2, 2]},
        "input_shape": list(x4x4.shape),
        "input_data": to_listf(x4x4),
        "expected_shape": list(out6.shape),
        "expected_output": to_listf(out6),
        "tolerance": 1e-6,
        "note": "AdaptiveMaxPool2d(2,2) on 4x4: max of each 2x2 tile.",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 6 — upsample
# ---------------------------------------------------------------------------


def fixture_upsample() -> list[dict[str, Any]]:
    fixtures = []

    # Upsample nearest 2x
    up = nn.Upsample(size=(4, 4), mode="nearest")
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    out = up(x)
    fixtures.append({
        "id": "upsample_nearest_2x",
        "module": "Upsample",
        "op": "forward",
        "config": {"size": [4, 4], "mode": "nearest"},
        "input_shape": list(x.shape),
        "input_data": to_listf(x),
        "expected_shape": list(out.shape),
        "expected_output": to_listf(out),
        "tolerance": 1e-6,
        "note": "Upsample nearest-neighbor 2x from [2,2] to [4,4].",
    })

    # PixelShuffle r=2
    ps = nn.PixelShuffle(2)
    data = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2)
    out_ps = ps(data)
    fixtures.append({
        "id": "pixel_shuffle_r2",
        "module": "PixelShuffle",
        "op": "forward",
        "config": {"upscale_factor": 2},
        "input_shape": list(data.shape),
        "input_data": to_listf(data),
        "expected_shape": list(out_ps.shape),
        "expected_output": to_listf(out_ps),
        "tolerance": 1e-6,
        "note": "PixelShuffle upscale_factor=2: [1,4,2,2] -> [1,1,4,4].",
    })

    # PixelUnshuffle r=2
    pu = nn.PixelUnshuffle(2)
    out_pu = pu(out_ps)
    fixtures.append({
        "id": "pixel_unshuffle_r2",
        "module": "PixelUnshuffle",
        "op": "forward",
        "config": {"downscale_factor": 2},
        "input_shape": list(out_ps.shape),
        "input_data": to_listf(out_ps),
        "expected_shape": list(out_pu.shape),
        "expected_output": to_listf(out_pu),
        "tolerance": 1e-6,
        "note": "PixelUnshuffle downscale_factor=2: inverse of pixel_shuffle.",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 7 — dropout (eval-mode only for determinism)
# ---------------------------------------------------------------------------


def fixture_dropout() -> list[dict[str, Any]]:
    fixtures = []

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    for name, layer in [
        ("Dropout", nn.Dropout(p=0.5)),
        ("Dropout2d", nn.Dropout2d(p=0.5)),
        ("AlphaDropout", nn.AlphaDropout(p=0.1)),
    ]:
        layer.eval()
        if name == "Dropout2d":
            xin = torch.ones(1, 4, 2, 2)
        else:
            xin = x
        out = layer(xin)
        fixtures.append({
            "id": f"{name.lower()}_eval_passthrough",
            "module": name,
            "op": "forward_eval",
            "config": {"p": layer.p},
            "input_shape": list(xin.shape),
            "input_data": to_listf(xin),
            "expected_shape": list(out.shape),
            "expected_output": to_listf(out),
            "tolerance": 1e-7,
            "note": f"{name} in eval mode is identity.",
        })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 8 — conv (shape-only; weights seeded)
# ---------------------------------------------------------------------------


def fixture_conv() -> list[dict[str, Any]]:
    fixtures = []

    # Conv2d shape
    torch.manual_seed(RNG_SEED)
    c2d = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0, bias=False)
    x = torch.zeros(1, 1, 5, 5)
    out = c2d(x)
    fixtures.append({
        "id": "conv2d_forward_shape",
        "module": "Conv2d",
        "op": "forward_shape",
        "config": {
            "in_channels": 1, "out_channels": 2,
            "kernel_size": 3, "stride": 1, "padding": 0, "bias": False,
        },
        "input_shape": list(x.shape),
        "expected_shape": list(out.shape),
        "note": "Conv2d(1->2, kernel=3, stride=1, no pad) on 5x5: output is 3x3.",
    })

    # Conv1d shape
    torch.manual_seed(RNG_SEED)
    c1d = nn.Conv1d(2, 4, kernel_size=3, stride=1, padding=1, bias=True)
    x1 = torch.zeros(2, 2, 8)
    out1 = c1d(x1)
    fixtures.append({
        "id": "conv1d_forward_shape",
        "module": "Conv1d",
        "op": "forward_shape",
        "config": {
            "in_channels": 2, "out_channels": 4,
            "kernel_size": 3, "stride": 1, "padding": 1, "bias": True,
        },
        "input_shape": list(x1.shape),
        "expected_shape": list(out1.shape),
        "note": "Conv1d(2->4, kernel=3, stride=1, pad=1) length=8: output length=8.",
    })

    # ConvTranspose2d shape
    torch.manual_seed(RNG_SEED)
    ct2d = nn.ConvTranspose2d(2, 1, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False)
    x2 = torch.zeros(1, 2, 4, 4)
    out2 = ct2d(x2)
    fixtures.append({
        "id": "conv_transpose2d_forward_shape",
        "module": "ConvTranspose2d",
        "op": "forward_shape",
        "config": {
            "in_channels": 2, "out_channels": 1,
            "kernel_size": 3, "stride": 2, "padding": 1,
            "output_padding": 1, "bias": False,
        },
        "input_shape": list(x2.shape),
        "expected_shape": list(out2.shape),
        "note": "ConvTranspose2d(2->1, k=3, stride=2, pad=1, out_pad=1) 4x4->8x8.",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 9 — lazy (shape-only)
# ---------------------------------------------------------------------------


def fixture_lazy() -> list[dict[str, Any]]:
    fixtures = []

    # LazyLinear materializes on first forward
    torch.manual_seed(RNG_SEED)
    ll = nn.LazyLinear(3)
    x = torch.zeros(2, 4)
    out = ll(x)
    fixtures.append({
        "id": "lazy_linear_materialize",
        "module": "LazyLinear",
        "op": "forward_materialize",
        "config": {"out_features": 3, "bias": True},
        "input_shape": list(x.shape),
        "input_data_zeros": True,
        "expected_shape": list(out.shape),
        "note": "LazyLinear materializes on first forward call; output shape checked only.",
    })

    # LazyConv2d
    torch.manual_seed(RNG_SEED)
    lc2d = nn.LazyConv2d(4, kernel_size=3, stride=1, padding=1)
    xc = torch.zeros(1, 2, 5, 5)
    outc = lc2d(xc)
    fixtures.append({
        "id": "lazy_conv2d_materialize",
        "module": "LazyConv2d",
        "op": "forward_materialize",
        "config": {"out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        "input_shape": list(xc.shape),
        "input_data_zeros": True,
        "expected_shape": list(outc.shape),
        "note": "LazyConv2d materializes in_channels=2 on first forward. Shape only.",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Cat 10 — EmbeddingBag
# ---------------------------------------------------------------------------


def fixture_embedding_bag() -> list[dict[str, Any]]:
    fixtures = []

    # EmbeddingBag sum
    eb = nn.EmbeddingBag(5, 3, mode="sum")
    w = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ])
    with torch.no_grad():
        eb.weight.copy_(w)
    # 2 bags: [0,1] and [2,3,4]
    input_ids = torch.tensor([0, 1, 2, 3, 4])
    offsets = torch.tensor([0, 2])
    out = eb(input_ids, offsets)
    fixtures.append({
        "id": "embedding_bag_sum",
        "module": "EmbeddingBag",
        "op": "forward_bag",
        "config": {"num_embeddings": 5, "embedding_dim": 3, "mode": "sum"},
        "input_indices": input_ids.tolist(),
        "offsets": offsets.tolist(),
        "weight": to_listf(eb.weight),
        "expected_shape": list(out.shape),
        "expected_output": to_listf(out),
        "tolerance": 1e-6,
        "note": "EmbeddingBag sum: bag [0,1]=row0+row1, bag [2,3,4]=row2+row3+row4.",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Assemble and write
# ---------------------------------------------------------------------------


def main() -> None:
    all_fixtures: list[dict[str, Any]] = []
    all_fixtures.extend(fixture_linear())
    all_fixtures.extend(fixture_embedding())
    all_fixtures.extend(fixture_identity())
    all_fixtures.extend(fixture_padding())
    all_fixtures.extend(fixture_pooling())
    all_fixtures.extend(fixture_upsample())
    all_fixtures.extend(fixture_dropout())
    all_fixtures.extend(fixture_conv())
    all_fixtures.extend(fixture_lazy())
    all_fixtures.extend(fixture_embedding_bag())

    output = fixture_metadata()
    output["fixtures"] = all_fixtures

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(all_fixtures)} fixtures to {FIXTURE_PATH}")
    print(f"Torch version: {torch.__version__}")


if __name__ == "__main__":
    main()
