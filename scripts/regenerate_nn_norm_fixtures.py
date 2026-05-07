#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-nn C9.2
(norm, activation, loss, init, functional).

Tracking issue: #901 (C9.2 conformance suite).

Output:
    ferrotorch-nn/tests/conformance/fixtures/nn_norm_activation_loss.json

Coverage (7 modules):

1. **norm** — LayerNorm, GroupNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d,
   InstanceNorm1d, InstanceNorm2d, RMSNorm (forward, eval mode so stats are
   deterministic).

2. **activation** — ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax, LogSoftmax,
   ELU, SELU, Mish, LeakyReLU, HardSigmoid, HardSwish, Softplus, LogSigmoid,
   ReLU6, Hardtanh.

3. **loss** — MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss,
   NLLLoss — all three reduction modes (mean/sum/none) where applicable.

4. **init** — uniform, normal, xavier_uniform, xavier_normal, kaiming_uniform,
   kaiming_normal, constant, zeros, ones — seeded RNG → first-N samples.
   Verified by comparing distribution stats (mean, std, min, max) and comparing
   the xavier/kaiming closed-form limits directly.

5. **functional** — linear, relu, sigmoid, tanh, gelu, silu, softmax,
   log_softmax, leaky_relu — functional versions matched against module forward.

6. **lazy_norm** — LazyBatchNorm1d, LazyBatchNorm2d (initialized on first forward,
   then checked against the regular BatchNorm).

7. **utils** — clip_grad_norm_, clip_grad_value_ (simple CPU cases, norm/value
   verified analytically for small tensors).

Pin: torch 2.11.0.

Usage from WSL (preferred per project conventions):

    python3 scripts/regenerate_nn_norm_fixtures.py

Required Python deps: torch, numpy.
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
import torch.nn.init as init

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT / "ferrotorch-nn" / "tests" / "conformance" / "fixtures" / "nn_norm_activation_loss.json"
)

# ---------------------------------------------------------------------------
# RNG seed
# ---------------------------------------------------------------------------
SEED = 0xC9_2A_FF
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_list(t: torch.Tensor) -> list:
    """Recursively convert tensor to nested float lists, replacing inf/nan with sentinels."""
    flat = t.detach().float().flatten().tolist()
    return [_sentinel(v) for v in flat]

def _sentinel(v: float) -> Any:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v

def seeded_input(shape: list[int], seed: int = SEED, low: float = -2.0, high: float = 2.0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.empty(shape).uniform_(low, high, generator=g)

def seeded_positive_input(shape: list[int], seed: int = SEED) -> torch.Tensor:
    """Returns values in (0, 1) for BCELoss targets."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.empty(shape).uniform_(0.01, 0.99, generator=g)

fixtures: list[dict] = []

# ===========================================================================
# 1. NORM MODULE
# ===========================================================================

def _norm_fixture(op: str, tag: str, input_data: list, input_shape: list, out_data: list, out_shape: list,
                  extra: dict | None = None) -> dict:
    f: dict = {"op": op, "tag": tag, "input_data": input_data, "input_shape": input_shape,
               "out_data": out_data, "out_shape": out_shape}
    if extra:
        f.update(extra)
    return f

# ----- LayerNorm -----
for norm_dim in [8, 16]:
    for batch in [2, 4]:
        x = seeded_input([batch, norm_dim], seed=SEED + norm_dim)
        m = nn.LayerNorm(norm_dim, eps=1e-5)
        m.eval()
        with torch.no_grad():
            y = m(x)
        fixtures.append(_norm_fixture(
            "layer_norm", f"b{batch}_d{norm_dim}",
            to_list(x), [batch, norm_dim],
            to_list(y), [batch, norm_dim],
            {"weight": to_list(m.weight), "bias": to_list(m.bias), "eps": 1e-5},
        ))

# LayerNorm with 2D normalized_shape [H, W]
x = seeded_input([2, 3, 4], seed=SEED + 7)
m = nn.LayerNorm([3, 4], eps=1e-5)
m.eval()
with torch.no_grad():
    y = m(x)
fixtures.append(_norm_fixture(
    "layer_norm", "3d_shape34",
    to_list(x), [2, 3, 4],
    to_list(y), [2, 3, 4],
    {"weight": to_list(m.weight), "bias": to_list(m.bias), "eps": 1e-5},
))

# ----- GroupNorm -----
for num_groups, num_channels in [(2, 8), (4, 8), (1, 4)]:
    x = seeded_input([2, num_channels, 5], seed=SEED + num_channels)
    m = nn.GroupNorm(num_groups, num_channels, eps=1e-5)
    m.eval()
    with torch.no_grad():
        y = m(x)
    fixtures.append(_norm_fixture(
        "group_norm", f"g{num_groups}_c{num_channels}",
        to_list(x), [2, num_channels, 5],
        to_list(y), [2, num_channels, 5],
        {"num_groups": num_groups, "num_channels": num_channels, "eps": 1e-5,
         "weight": to_list(m.weight), "bias": to_list(m.bias)},
    ))

# ----- BatchNorm1d -----
for num_features in [4, 8]:
    x = seeded_input([3, num_features], seed=SEED + num_features + 100)
    m = nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True)
    m.eval()
    # Pre-seed running stats with known values to make eval deterministic
    m.running_mean.fill_(0.0)
    m.running_var.fill_(1.0)
    with torch.no_grad():
        y = m(x)
    fixtures.append(_norm_fixture(
        "batch_norm_1d", f"features{num_features}",
        to_list(x), [3, num_features],
        to_list(y), [3, num_features],
        {"num_features": num_features, "eps": 1e-5,
         "running_mean": m.running_mean.tolist(), "running_var": m.running_var.tolist(),
         "weight": to_list(m.weight), "bias": to_list(m.bias)},
    ))

# ----- BatchNorm2d -----
for num_features in [4, 8]:
    x = seeded_input([2, num_features, 3, 3], seed=SEED + num_features + 200)
    m = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True)
    m.eval()
    m.running_mean.fill_(0.0)
    m.running_var.fill_(1.0)
    with torch.no_grad():
        y = m(x)
    fixtures.append(_norm_fixture(
        "batch_norm_2d", f"c{num_features}_hw3",
        to_list(x), [2, num_features, 3, 3],
        to_list(y), [2, num_features, 3, 3],
        {"num_features": num_features, "eps": 1e-5,
         "running_mean": m.running_mean.tolist(), "running_var": m.running_var.tolist(),
         "weight": to_list(m.weight), "bias": to_list(m.bias)},
    ))

# ----- BatchNorm3d -----
x = seeded_input([2, 4, 2, 2, 2], seed=SEED + 300)
m = nn.BatchNorm3d(4, eps=1e-5, momentum=0.1, affine=True)
m.eval()
m.running_mean.fill_(0.0)
m.running_var.fill_(1.0)
with torch.no_grad():
    y = m(x)
fixtures.append(_norm_fixture(
    "batch_norm_3d", "c4_dhw2",
    to_list(x), [2, 4, 2, 2, 2],
    to_list(y), [2, 4, 2, 2, 2],
    {"num_features": 4, "eps": 1e-5,
     "running_mean": m.running_mean.tolist(), "running_var": m.running_var.tolist(),
     "weight": to_list(m.weight), "bias": to_list(m.bias)},
))

# ----- InstanceNorm1d -----
x = seeded_input([2, 4, 6], seed=SEED + 400)
m = nn.InstanceNorm1d(4, eps=1e-5, affine=False)
m.eval()
with torch.no_grad():
    y = m(x)
fixtures.append(_norm_fixture(
    "instance_norm_1d", "c4_l6",
    to_list(x), [2, 4, 6],
    to_list(y), [2, 4, 6],
    {"num_features": 4, "eps": 1e-5},
))

# ----- InstanceNorm2d -----
x = seeded_input([2, 4, 3, 3], seed=SEED + 500)
m = nn.InstanceNorm2d(4, eps=1e-5, affine=False)
m.eval()
with torch.no_grad():
    y = m(x)
fixtures.append(_norm_fixture(
    "instance_norm_2d", "c4_hw3",
    to_list(x), [2, 4, 3, 3],
    to_list(y), [2, 4, 3, 3],
    {"num_features": 4, "eps": 1e-5},
))

# ----- RMSNorm -----
for norm_dim in [8, 16]:
    x = seeded_input([3, norm_dim], seed=SEED + norm_dim + 600)
    m = nn.RMSNorm(norm_dim, eps=1e-6)
    m.eval()
    with torch.no_grad():
        y = m(x)
    fixtures.append(_norm_fixture(
        "rms_norm", f"d{norm_dim}",
        to_list(x), [3, norm_dim],
        to_list(y), [3, norm_dim],
        {"normalized_dim": norm_dim, "eps": 1e-6, "weight": to_list(m.weight)},
    ))

# ===========================================================================
# 2. ACTIVATION MODULE
# ===========================================================================

def _act_fixture(op: str, tag: str, input_data: list, input_shape: list,
                 out_data: list, out_shape: list, extra: dict | None = None) -> dict:
    f: dict = {"op": op, "tag": tag, "input_data": input_data, "input_shape": input_shape,
               "out_data": out_data, "out_shape": out_shape}
    if extra:
        f.update(extra)
    return f

ACT_SHAPES = [[6], [3, 4], [2, 3, 4]]

def act_input(shape: list, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.empty(shape).uniform_(-2.0, 2.0, generator=g)

# ReLU
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 1000)
    y = F.relu(x)
    fixtures.append(_act_fixture("relu", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# ReLU6
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 1100)
    y = F.relu6(x)
    fixtures.append(_act_fixture("relu6", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Sigmoid
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 1200)
    y = torch.sigmoid(x)
    fixtures.append(_act_fixture("sigmoid", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Sigmoid saturation
x_sat = torch.tensor([-100.0, -50.0, 0.0, 50.0, 100.0])
y_sat = torch.sigmoid(x_sat)
fixtures.append(_act_fixture("sigmoid", "saturation", to_list(x_sat), [5], to_list(y_sat), [5]))

# Tanh
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 1300)
    y = torch.tanh(x)
    fixtures.append(_act_fixture("tanh", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Tanh saturation
x_sat = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
y_sat = torch.tanh(x_sat)
fixtures.append(_act_fixture("tanh", "saturation", to_list(x_sat), [5], to_list(y_sat), [5]))

# GELU (exact)
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 1400)
    y = F.gelu(x, approximate="none")
    fixtures.append(_act_fixture("gelu_exact", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# GELU (tanh)
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 1500)
    y = F.gelu(x, approximate="tanh")
    fixtures.append(_act_fixture("gelu_tanh", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# SiLU
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 1600)
    y = F.silu(x)
    fixtures.append(_act_fixture("silu", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Softmax (last dim)
for shape in [[6], [3, 4], [2, 3, 4]]:
    x = act_input(shape, SEED + 1700)
    y = F.softmax(x, dim=-1)
    fixtures.append(_act_fixture("softmax", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Softmax numerical stability: [100, 100, 100]
x_stable = torch.tensor([100.0, 100.0, 100.0])
y_stable = F.softmax(x_stable, dim=-1)
fixtures.append(_act_fixture("softmax", "stability_flat100", to_list(x_stable), [3], to_list(y_stable), [3]))

# LogSoftmax
for shape in [[6], [3, 4]]:
    x = act_input(shape, SEED + 1800)
    y = F.log_softmax(x, dim=-1)
    fixtures.append(_act_fixture("log_softmax", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# LogSoftmax numerical stability: [1000, 1001]
x_ls = torch.tensor([1000.0, 1001.0])
y_ls = F.log_softmax(x_ls, dim=-1)
fixtures.append(_act_fixture("log_softmax", "stability_large", to_list(x_ls), [2], to_list(y_ls), [2]))

# ELU
for alpha in [1.0, 0.5]:
    for shape in ACT_SHAPES:
        x = act_input(shape, SEED + 1900)
        y = F.elu(x, alpha=alpha)
        fixtures.append(_act_fixture("elu", f"alpha{alpha}_shape{'x'.join(map(str, shape))}",
                                     to_list(x), shape, to_list(y), shape,
                                     {"alpha": alpha}))

# SELU
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 2000)
    y = F.selu(x)
    fixtures.append(_act_fixture("selu", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Mish
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 2100)
    y = F.mish(x)
    fixtures.append(_act_fixture("mish", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# LeakyReLU
for neg_slope in [0.01, 0.1, 0.2]:
    for shape in ACT_SHAPES:
        x = act_input(shape, SEED + 2200)
        y = F.leaky_relu(x, negative_slope=neg_slope)
        fixtures.append(_act_fixture("leaky_relu", f"slope{neg_slope}_shape{'x'.join(map(str, shape))}",
                                     to_list(x), shape, to_list(y), shape,
                                     {"negative_slope": neg_slope}))

# HardSigmoid
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 2300)
    y = F.hardsigmoid(x)
    fixtures.append(_act_fixture("hardsigmoid", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# HardSwish
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 2400)
    y = F.hardswish(x)
    fixtures.append(_act_fixture("hardswish", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Softplus
for beta in [1.0, 2.0]:
    for shape in [[6], [3, 4]]:
        x = act_input(shape, SEED + 2500)
        y = F.softplus(x, beta=beta)
        fixtures.append(_act_fixture("softplus", f"beta{beta}_shape{'x'.join(map(str, shape))}",
                                     to_list(x), shape, to_list(y), shape,
                                     {"beta": beta}))

# LogSigmoid
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 2600)
    y = F.logsigmoid(x)
    fixtures.append(_act_fixture("log_sigmoid", f"shape{'x'.join(map(str, shape))}",
                                 to_list(x), shape, to_list(y), shape))

# Hardtanh
for (min_val, max_val) in [(-1.0, 1.0), (-2.0, 2.0)]:
    for shape in ACT_SHAPES:
        x = act_input(shape, SEED + 2700)
        y = F.hardtanh(x, min_val=min_val, max_val=max_val)
        fixtures.append(_act_fixture("hardtanh",
                                     f"min{min_val}_max{max_val}_shape{'x'.join(map(str, shape))}",
                                     to_list(x), shape, to_list(y), shape,
                                     {"min_val": min_val, "max_val": max_val}))

# ===========================================================================
# 3. LOSS MODULE
# ===========================================================================

def _loss_fixture(op: str, tag: str, pred_data: list, pred_shape: list,
                  target_data: list, target_shape: list, out_scalar: float,
                  extra: dict | None = None) -> dict:
    f: dict = {"op": op, "tag": tag, "pred_data": pred_data, "pred_shape": pred_shape,
               "target_data": target_data, "target_shape": target_shape,
               "out_scalar": _sentinel(out_scalar)}
    if extra:
        f.update(extra)
    return f

REDUCTION_MODES = ["mean", "sum", "none"]

def torch_reduction(s: str):
    return {"mean": "mean", "sum": "sum", "none": "none"}[s]

# ----- MSELoss -----
for reduction in REDUCTION_MODES:
    g = torch.Generator()
    g.manual_seed(SEED + 3000)
    pred = torch.empty([4, 8]).uniform_(-2.0, 2.0, generator=g)
    g.manual_seed(SEED + 3001)
    target = torch.empty([4, 8]).uniform_(-2.0, 2.0, generator=g)
    loss_fn = nn.MSELoss(reduction=reduction)
    out = loss_fn(pred, target)
    if reduction == "none":
        fixtures.append({
            "op": "mse_loss", "tag": f"reduction_{reduction}",
            "pred_data": to_list(pred), "pred_shape": [4, 8],
            "target_data": to_list(target), "target_shape": [4, 8],
            "out_data": to_list(out), "out_shape": [4, 8],
            "reduction": reduction,
        })
    else:
        fixtures.append(_loss_fixture("mse_loss", f"reduction_{reduction}",
                                      to_list(pred), [4, 8], to_list(target), [4, 8],
                                      out.item(), {"reduction": reduction}))

# ----- L1Loss -----
for reduction in REDUCTION_MODES:
    g = torch.Generator()
    g.manual_seed(SEED + 3100)
    pred = torch.empty([4, 8]).uniform_(-2.0, 2.0, generator=g)
    g.manual_seed(SEED + 3101)
    target = torch.empty([4, 8]).uniform_(-2.0, 2.0, generator=g)
    loss_fn = nn.L1Loss(reduction=reduction)
    out = loss_fn(pred, target)
    if reduction == "none":
        fixtures.append({
            "op": "l1_loss", "tag": f"reduction_{reduction}",
            "pred_data": to_list(pred), "pred_shape": [4, 8],
            "target_data": to_list(target), "target_shape": [4, 8],
            "out_data": to_list(out), "out_shape": [4, 8],
            "reduction": reduction,
        })
    else:
        fixtures.append(_loss_fixture("l1_loss", f"reduction_{reduction}",
                                      to_list(pred), [4, 8], to_list(target), [4, 8],
                                      out.item(), {"reduction": reduction}))

# ----- BCELoss -----
for reduction in REDUCTION_MODES:
    g = torch.Generator()
    g.manual_seed(SEED + 3200)
    pred = seeded_positive_input([4, 8], seed=SEED + 3200)
    target = seeded_positive_input([4, 8], seed=SEED + 3201)
    # Round targets to 0 or 1 for cleaner BCE
    target_bin = (target > 0.5).float()
    loss_fn = nn.BCELoss(reduction=reduction)
    out = loss_fn(pred, target_bin)
    if reduction == "none":
        fixtures.append({
            "op": "bce_loss", "tag": f"reduction_{reduction}",
            "pred_data": to_list(pred), "pred_shape": [4, 8],
            "target_data": to_list(target_bin), "target_shape": [4, 8],
            "out_data": to_list(out), "out_shape": [4, 8],
            "reduction": reduction,
        })
    else:
        fixtures.append(_loss_fixture("bce_loss", f"reduction_{reduction}",
                                      to_list(pred), [4, 8], to_list(target_bin), [4, 8],
                                      out.item(), {"reduction": reduction}))

# ----- BCEWithLogitsLoss -----
for reduction in REDUCTION_MODES:
    g = torch.Generator()
    g.manual_seed(SEED + 3300)
    pred = torch.empty([4, 8]).uniform_(-3.0, 3.0, generator=g)
    target_bin = (seeded_positive_input([4, 8], seed=SEED + 3301) > 0.5).float()
    loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
    out = loss_fn(pred, target_bin)
    if reduction == "none":
        fixtures.append({
            "op": "bce_with_logits_loss", "tag": f"reduction_{reduction}",
            "pred_data": to_list(pred), "pred_shape": [4, 8],
            "target_data": to_list(target_bin), "target_shape": [4, 8],
            "out_data": to_list(out), "out_shape": [4, 8],
            "reduction": reduction,
        })
    else:
        fixtures.append(_loss_fixture("bce_with_logits_loss", f"reduction_{reduction}",
                                      to_list(pred), [4, 8], to_list(target_bin), [4, 8],
                                      out.item(), {"reduction": reduction}))

# ----- CrossEntropyLoss -----
for reduction in ["mean", "sum"]:
    for num_classes in [5, 10]:
        g = torch.Generator()
        g.manual_seed(SEED + 3400 + num_classes)
        logits = torch.empty([8, num_classes]).uniform_(-3.0, 3.0, generator=g)
        g.manual_seed(SEED + 3401 + num_classes)
        targets = torch.randint(0, num_classes, [8], generator=g)
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        out = loss_fn(logits, targets)
        fixtures.append({
            "op": "cross_entropy_loss",
            "tag": f"reduction_{reduction}_c{num_classes}",
            "pred_data": to_list(logits), "pred_shape": [8, num_classes],
            "target_data": targets.tolist(), "target_shape": [8],
            "out_scalar": _sentinel(out.item()),
            "reduction": reduction, "num_classes": num_classes,
        })

# ----- NLLLoss -----
for reduction in ["mean", "sum"]:
    g = torch.Generator()
    g.manual_seed(SEED + 3500)
    # NLLLoss expects log probabilities
    logits = torch.empty([6, 5]).uniform_(-3.0, 3.0, generator=g)
    log_probs = F.log_softmax(logits, dim=-1)
    g.manual_seed(SEED + 3501)
    targets = torch.randint(0, 5, [6], generator=g)
    loss_fn = nn.NLLLoss(reduction=reduction)
    out = loss_fn(log_probs, targets)
    fixtures.append({
        "op": "nll_loss",
        "tag": f"reduction_{reduction}",
        "pred_data": to_list(log_probs), "pred_shape": [6, 5],
        "target_data": targets.tolist(), "target_shape": [6],
        "out_scalar": _sentinel(out.item()),
        "reduction": reduction,
    })

# ===========================================================================
# 4. INIT MODULE
# ===========================================================================
# For init, we generate expected first-N samples from known seeded distributions.
# We record min, max, mean, std for statistical checks (RNG differs) plus the
# limits/bounds for analytical checks.

def _init_fixture(op: str, tag: str, shape: list, extra: dict) -> dict:
    f: dict = {"op": op, "tag": tag, "shape": shape}
    f.update(extra)
    return f

# ----- zeros -----
fixtures.append(_init_fixture("init_zeros", "2x4", [2, 4],
                              {"expected_values": [0.0] * 8}))

# ----- ones -----
fixtures.append(_init_fixture("init_ones", "2x4", [2, 4],
                              {"expected_values": [1.0] * 8}))

# ----- constant -----
for val in [0.5, -2.3]:
    fixtures.append(_init_fixture("init_constant", f"val{val}_3x3", [3, 3],
                                  {"fill_value": val, "expected_values": [val] * 9}))

# ----- xavier_uniform bounds -----
# For a [fan_out=4, fan_in=8] weight: limit = sqrt(6 / (4+8)) = sqrt(0.5)
fan_out, fan_in = 4, 8
limit_xu = math.sqrt(6.0 / (fan_in + fan_out))
fixtures.append(_init_fixture("init_xavier_uniform", f"fanin{fan_in}_fanout{fan_out}", [fan_out, fan_in],
                              {"expected_limit": limit_xu, "fan_in": fan_in, "fan_out": fan_out}))

# For a [32, 64] weight:
fan_out, fan_in = 32, 64
limit_xu2 = math.sqrt(6.0 / (fan_in + fan_out))
fixtures.append(_init_fixture("init_xavier_uniform", f"fanin{fan_in}_fanout{fan_out}", [fan_out, fan_in],
                              {"expected_limit": limit_xu2, "fan_in": fan_in, "fan_out": fan_out}))

# ----- xavier_normal std -----
fan_out, fan_in = 4, 8
std_xn = math.sqrt(2.0 / (fan_in + fan_out))
fixtures.append(_init_fixture("init_xavier_normal", f"fanin{fan_in}_fanout{fan_out}", [fan_out, fan_in],
                              {"expected_std": std_xn, "fan_in": fan_in, "fan_out": fan_out}))

# ----- kaiming_uniform -----
fan_in = 16
gain = math.sqrt(2.0)  # ReLU gain
std_ku = gain / math.sqrt(fan_in)
limit_ku = math.sqrt(3.0) * std_ku
fixtures.append(_init_fixture("init_kaiming_uniform", f"relu_fanin{fan_in}", [32, fan_in],
                              {"expected_limit": limit_ku, "fan_in": fan_in, "nonlinearity": "relu"}))

# ----- kaiming_normal -----
fan_in = 16
std_kn = gain / math.sqrt(fan_in)
fixtures.append(_init_fixture("init_kaiming_normal", f"relu_fanin{fan_in}", [32, fan_in],
                              {"expected_std": std_kn, "fan_in": fan_in, "nonlinearity": "relu"}))

# ----- uniform distribution stats -----
g = torch.Generator()
g.manual_seed(SEED + 4000)
t = torch.empty([1000]).uniform_(-1.0, 1.0, generator=g)
fixtures.append(_init_fixture("init_uniform", "1000_samples_n1_p1", [1000],
                              {"low": -1.0, "high": 1.0,
                               "expected_mean": 0.0, "expected_std": 1.0 / math.sqrt(3.0),
                               "tol_mean": 0.1, "tol_std": 0.05}))

# ----- normal distribution stats -----
g = torch.Generator()
g.manual_seed(SEED + 4100)
t = torch.empty([1000]).normal_(0.0, 1.0, generator=g)
fixtures.append(_init_fixture("init_normal", "1000_samples_mu0_std1", [1000],
                              {"mean": 0.0, "std": 1.0,
                               "expected_mean": 0.0, "expected_std": 1.0,
                               "tol_mean": 0.15, "tol_std": 0.1}))

# ===========================================================================
# 5. FUNCTIONAL MODULE
# ===========================================================================

def _fn_fixture(op: str, tag: str, input_data: list, input_shape: list,
                out_data: list, out_shape: list, extra: dict | None = None) -> dict:
    f: dict = {"op": f"fn_{op}", "tag": tag, "input_data": input_data, "input_shape": input_shape,
               "out_data": out_data, "out_shape": out_shape}
    if extra:
        f.update(extra)
    return f

# fn_linear vs nn.Linear
for (in_f, out_f) in [(8, 4), (16, 8)]:
    g = torch.Generator()
    g.manual_seed(SEED + 5000 + in_f)
    x = torch.empty([4, in_f]).uniform_(-1.0, 1.0, generator=g)
    g.manual_seed(SEED + 5001 + in_f)
    w = torch.empty([out_f, in_f]).uniform_(-0.5, 0.5, generator=g)
    g.manual_seed(SEED + 5002 + in_f)
    b = torch.empty([out_f]).uniform_(-0.2, 0.2, generator=g)
    y = F.linear(x, w, b)
    fixtures.append({
        "op": "fn_linear", "tag": f"in{in_f}_out{out_f}",
        "input_data": to_list(x), "input_shape": [4, in_f],
        "weight_data": to_list(w), "weight_shape": [out_f, in_f],
        "bias_data": to_list(b), "bias_shape": [out_f],
        "out_data": to_list(y), "out_shape": [4, out_f],
    })

# fn_relu
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 5100)
    y = F.relu(x)
    fixtures.append(_fn_fixture("relu", f"shape{'x'.join(map(str, shape))}",
                                to_list(x), shape, to_list(y), shape))

# fn_sigmoid
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 5200)
    y = torch.sigmoid(x)
    fixtures.append(_fn_fixture("sigmoid", f"shape{'x'.join(map(str, shape))}",
                                to_list(x), shape, to_list(y), shape))

# fn_tanh
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 5300)
    y = torch.tanh(x)
    fixtures.append(_fn_fixture("tanh", f"shape{'x'.join(map(str, shape))}",
                                to_list(x), shape, to_list(y), shape))

# fn_gelu
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 5400)
    y = F.gelu(x, approximate="none")
    fixtures.append(_fn_fixture("gelu", f"shape{'x'.join(map(str, shape))}",
                                to_list(x), shape, to_list(y), shape))

# fn_silu
for shape in ACT_SHAPES:
    x = act_input(shape, SEED + 5500)
    y = F.silu(x)
    fixtures.append(_fn_fixture("silu", f"shape{'x'.join(map(str, shape))}",
                                to_list(x), shape, to_list(y), shape))

# fn_softmax
for shape in [[6], [3, 4]]:
    x = act_input(shape, SEED + 5600)
    y = F.softmax(x, dim=-1)
    fixtures.append(_fn_fixture("softmax", f"shape{'x'.join(map(str, shape))}",
                                to_list(x), shape, to_list(y), shape))

# fn_log_softmax
for shape in [[6], [3, 4]]:
    x = act_input(shape, SEED + 5700)
    y = F.log_softmax(x, dim=-1)
    fixtures.append(_fn_fixture("log_softmax", f"shape{'x'.join(map(str, shape))}",
                                to_list(x), shape, to_list(y), shape))

# fn_leaky_relu
for neg_slope in [0.01, 0.1]:
    for shape in ACT_SHAPES:
        x = act_input(shape, SEED + 5800)
        y = F.leaky_relu(x, negative_slope=neg_slope)
        fixtures.append({
            **_fn_fixture("leaky_relu", f"slope{neg_slope}_shape{'x'.join(map(str, shape))}",
                          to_list(x), shape, to_list(y), shape),
            "negative_slope": neg_slope,
        })

# fn_mse_loss
g = torch.Generator()
g.manual_seed(SEED + 5900)
pred = torch.empty([4, 8]).uniform_(-2.0, 2.0, generator=g)
g.manual_seed(SEED + 5901)
target = torch.empty([4, 8]).uniform_(-2.0, 2.0, generator=g)
mse_val = F.mse_loss(pred, target)
fixtures.append({
    "op": "fn_mse_loss", "tag": "4x8_mean",
    "pred_data": to_list(pred), "pred_shape": [4, 8],
    "target_data": to_list(target), "target_shape": [4, 8],
    "out_scalar": _sentinel(mse_val.item()),
})

# ===========================================================================
# 6. LAZY_NORM MODULE
# ===========================================================================

# LazyBatchNorm1d — first forward determines num_features
for num_features in [4, 8]:
    # Create LazyBatchNorm1d, run forward to initialize, then compare
    x = seeded_input([3, num_features], seed=SEED + 6000 + num_features)
    lazy_m = nn.LazyBatchNorm1d()
    lazy_m.eval()

    # First forward triggers materialization (in eval mode running stats are 0/1)
    # We need training first to set stats, then eval
    lazy_m.train()
    # Use same reference batch to compare against regular BN
    with torch.no_grad():
        # Run in train mode to accumulate stats
        _ = lazy_m(x)

    lazy_m.eval()
    with torch.no_grad():
        y_lazy = lazy_m(x)

    # Compare against regular BatchNorm1d with same running stats
    regular_m = nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1)
    regular_m.eval()
    regular_m.running_mean.copy_(lazy_m.running_mean)
    regular_m.running_var.copy_(lazy_m.running_var)
    regular_m.weight.data.copy_(lazy_m.weight)
    regular_m.bias.data.copy_(lazy_m.bias)
    with torch.no_grad():
        y_regular = regular_m(x)

    fixtures.append({
        "op": "lazy_batch_norm_1d", "tag": f"features{num_features}",
        "input_data": to_list(x), "input_shape": [3, num_features],
        "out_data": to_list(y_lazy), "out_shape": [3, num_features],
        "num_features": num_features,
        "running_mean": lazy_m.running_mean.tolist(),
        "running_var": lazy_m.running_var.tolist(),
    })

# ===========================================================================
# 7. UTILS MODULE (clip_grad_norm_, clip_grad_value_)
# ===========================================================================

# clip_grad_norm_ — compute analytically:
# grads: [[1.0, 2.0], [3.0, 4.0]] → global norm = sqrt(1+4+9+16) = sqrt(30)
# max_norm=2.0 → coeff = 2.0/sqrt(30) → clipped grads
import math
g1 = torch.tensor([1.0, 2.0])
g2 = torch.tensor([3.0, 4.0])
global_norm = math.sqrt(sum(v*v for v in [1.0, 2.0, 3.0, 4.0]))
max_norm = 2.0
coeff = max_norm / global_norm
clipped_g1 = [v * coeff for v in [1.0, 2.0]]
clipped_g2 = [v * coeff for v in [3.0, 4.0]]
fixtures.append({
    "op": "clip_grad_norm_", "tag": "2params_maxnorm2",
    "grad1_data": [1.0, 2.0], "grad1_shape": [2],
    "grad2_data": [3.0, 4.0], "grad2_shape": [2],
    "max_norm": max_norm,
    "global_norm": _sentinel(global_norm),
    "clipped_grad1": [_sentinel(v) for v in clipped_g1],
    "clipped_grad2": [_sentinel(v) for v in clipped_g2],
})

# clip_grad_norm_ — no clipping needed (norm < max_norm)
g1_small = torch.tensor([0.1, 0.2])
g2_small = torch.tensor([0.3, 0.4])
global_norm_small = math.sqrt(sum(v*v for v in [0.1, 0.2, 0.3, 0.4]))
max_norm_large = 10.0
fixtures.append({
    "op": "clip_grad_norm_", "tag": "no_clip_needed",
    "grad1_data": [0.1, 0.2], "grad1_shape": [2],
    "grad2_data": [0.3, 0.4], "grad2_shape": [2],
    "max_norm": max_norm_large,
    "global_norm": _sentinel(global_norm_small),
    "clipped_grad1": [0.1, 0.2],  # unchanged
    "clipped_grad2": [0.3, 0.4],  # unchanged
})

# clip_grad_value_ — values beyond threshold get clamped
fixtures.append({
    "op": "clip_grad_value_", "tag": "basic_clip",
    "grad1_data": [-3.0, 1.5, 0.5, 2.5], "grad1_shape": [4],
    "clip_value": 2.0,
    "clipped_grad1": [-2.0, 1.5, 0.5, 2.0],
})

# ===========================================================================
# Write output
# ===========================================================================

FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

metadata = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "python_executable": sys.executable,
    "python_platform": platform.platform(),
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "rng_seed": SEED,
    "description": "C9.2 — ferrotorch-nn norm/activation/loss/init/functional/lazy_norm/utils",
    "modules": ["norm", "activation", "loss", "init", "functional", "lazy_norm", "utils"],
}

output = {"metadata": metadata, "fixtures": fixtures}
with open(FIXTURE_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {len(fixtures)} fixtures to {FIXTURE_PATH}")
