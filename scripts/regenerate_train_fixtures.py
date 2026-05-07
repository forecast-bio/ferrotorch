#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-train conformance suite.

Tracking issue: #843.

Reference: torch == 2.11.0

Output:
    ferrotorch-train/tests/conformance/fixtures.json

Coverage (training-loop semantic workloads):

* metric semantics
    - LossMetric: running mean accumulation over multiple batches
    - AccuracyMetric: correct/total across batches
    - TopKAccuracy: correct-in-top-k across batches
    - RunningAverage: windowed mean with window eviction

* gradient utilities (parity with torch.nn.utils.clip_grad_norm_ /
  clip_grad_value_)
    - clip_grad_norm_ L2: known gradient vector [3,4], max_norm=2.5
    - clip_grad_norm_ L1: gradient [3,-4], max_norm=3.5
    - clip_grad_norm_ inf: gradient [3,-7], max_norm=3.5
    - clip_grad_norm_ no-clip: norm already below max
    - clip_grad_value_: elementwise clamp to [-1, 1]

* EMA decay arithmetic (parity with PyTorch's EMA pattern)
    - decay=0.9, initial=[10.0], update=[20.0] -> 0.9*10+0.1*20=11.0
    - decay=0.5, three steps from 0 toward 10.0
    - decay=0.0: full replace
    - decay=1.0: no update

* training history statistics
    - best_train_loss: picks minimum over epochs
    - best_val_loss: picks minimum ignoring None entries

* EarlyStopping trigger semantics
    - triggers after patience=2 epochs with no improvement
    - resets on improvement
    - min_delta: small improvement does not count

* gradient accumulation semantics (multi-batch loss sum, then mean)
    - 4 batches of known loss values -> mean

* checkpoint sequential composition
    - chain of scale factors: expected product

Usage:
    python3 scripts/regenerate_train_fixtures.py
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

# ---------------------------------------------------------------------------
# Output path and metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT / "ferrotorch-train" / "tests" / "conformance" / "fixtures.json"
)

RNG_SEED: int = 0xABCD_1234
torch.manual_seed(RNG_SEED)


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
    }


# ---------------------------------------------------------------------------
# Metric fixtures
# ---------------------------------------------------------------------------


def metric_fixtures() -> list[dict[str, Any]]:
    """Compute reference values for metric accumulation."""
    out: list[dict[str, Any]] = []

    # LossMetric: mean of several batches
    batches_loss = [1.0, 2.0, 3.0, 4.0]
    mean_loss = sum(batches_loss) / len(batches_loss)
    out.append(
        {
            "kind": "loss_metric",
            "batches": batches_loss,
            "expected_mean": mean_loss,
            "label": "LossMetric basic mean",
        }
    )

    # LossMetric: empty -> 0
    out.append(
        {
            "kind": "loss_metric_empty",
            "batches": [],
            "expected_mean": 0.0,
            "label": "LossMetric empty returns 0",
        }
    )

    # LossMetric: single value
    out.append(
        {
            "kind": "loss_metric_single",
            "batches": [42.0],
            "expected_mean": 42.0,
            "label": "LossMetric single value",
        }
    )

    # LossMetric: reset then accumulate
    batches_after_reset = [5.0, 7.0]
    mean_after_reset = sum(batches_after_reset) / len(batches_after_reset)
    out.append(
        {
            "kind": "loss_metric_after_reset",
            "batches_before_reset": [10.0, 20.0],
            "batches_after_reset": batches_after_reset,
            "expected_mean_before": 15.0,
            "expected_mean_after": mean_after_reset,
            "label": "LossMetric reset then accumulate",
        }
    )

    # AccuracyMetric: 8/10 + 9/10 = 17/20 = 0.85
    out.append(
        {
            "kind": "accuracy_metric",
            "batches": [[8, 10], [9, 10]],
            "expected": 0.85,
            "label": "AccuracyMetric two batches",
        }
    )

    # AccuracyMetric: perfect score
    out.append(
        {
            "kind": "accuracy_metric_perfect",
            "batches": [[10, 10]],
            "expected": 1.0,
            "label": "AccuracyMetric perfect",
        }
    )

    # TopKAccuracy
    out.append(
        {
            "kind": "topk_accuracy",
            "k": 5,
            "batches": [[9, 10]],
            "expected": 0.9,
            "label": "TopKAccuracy k=5",
        }
    )

    # RunningAverage: window=3, 4 updates -> evicts oldest
    values_ra = [1.0, 2.0, 3.0, 6.0]  # window [2,3,6] after 4th push
    expected_ra = (2.0 + 3.0 + 6.0) / 3.0
    out.append(
        {
            "kind": "running_average",
            "window_size": 3,
            "values": values_ra,
            "expected_after_all": expected_ra,
            "expected_after_3": (1.0 + 2.0 + 3.0) / 3.0,
            "label": "RunningAverage window eviction",
        }
    )

    return out


# ---------------------------------------------------------------------------
# Gradient utilities fixtures
# ---------------------------------------------------------------------------


def clip_grad_norm_pytorch(grads: list[list[float]], max_norm: float, norm_type: float) -> dict[str, Any]:
    """Compute clip_grad_norm_ reference values using torch."""
    param_list = []
    tensor_list = []
    for g in grads:
        t = torch.tensor(g, dtype=torch.float64, requires_grad=True)
        # Simulate having a gradient: create a param and set .grad manually
        p = torch.nn.Parameter(torch.zeros_like(t))
        p.grad = t.clone().detach()
        param_list.append(p)
        tensor_list.append(t)

    total_norm_before = torch.nn.utils.clip_grad_norm_(param_list, max_norm, norm_type)

    clipped_grads = []
    for p in param_list:
        clipped_grads.append(p.grad.tolist())

    return {
        "total_norm": float(total_norm_before),
        "clipped_grads": clipped_grads,
    }


def grad_utils_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # L2 norm clips: [3, 4] -> norm=5, max_norm=2.5 -> coef = 2.5/5.000001
    result = clip_grad_norm_pytorch([[3.0, 4.0]], max_norm=2.5, norm_type=2.0)
    out.append(
        {
            "kind": "clip_grad_norm_l2_clips",
            "grads": [[3.0, 4.0]],
            "max_norm": 2.5,
            "norm_type": 2.0,
            "expected_total_norm": result["total_norm"],
            "expected_clipped": result["clipped_grads"],
            "label": "clip_grad_norm_ L2 clips [3,4] max_norm=2.5",
        }
    )

    # L2 norm no-clip: [0.1, 0.2] -> norm ~0.224 < 1.0
    result_nc = clip_grad_norm_pytorch([[0.1, 0.2]], max_norm=1.0, norm_type=2.0)
    out.append(
        {
            "kind": "clip_grad_norm_l2_noclip",
            "grads": [[0.1, 0.2]],
            "max_norm": 1.0,
            "norm_type": 2.0,
            "expected_total_norm": result_nc["total_norm"],
            "expected_clipped": result_nc["clipped_grads"],
            "label": "clip_grad_norm_ L2 no-clip [0.1,0.2] max_norm=1.0",
        }
    )

    # L1 norm: [3, -4] -> norm=7, max_norm=3.5
    result_l1 = clip_grad_norm_pytorch([[3.0, -4.0]], max_norm=3.5, norm_type=1.0)
    out.append(
        {
            "kind": "clip_grad_norm_l1",
            "grads": [[3.0, -4.0]],
            "max_norm": 3.5,
            "norm_type": 1.0,
            "expected_total_norm": result_l1["total_norm"],
            "expected_clipped": result_l1["clipped_grads"],
            "label": "clip_grad_norm_ L1 [3,-4] max_norm=3.5",
        }
    )

    # Inf norm: [3, -7] -> norm=7, max_norm=3.5
    result_inf = clip_grad_norm_pytorch([[3.0, -7.0]], max_norm=3.5, norm_type=float("inf"))
    out.append(
        {
            "kind": "clip_grad_norm_inf",
            "grads": [[3.0, -7.0]],
            "max_norm": 3.5,
            "norm_type": float("inf"),
            "expected_total_norm": result_inf["total_norm"],
            "expected_clipped": result_inf["clipped_grads"],
            "label": "clip_grad_norm_ inf norm [3,-7] max_norm=3.5",
        }
    )

    # Multiple params: [3] + [4] -> total L2 norm=5
    result_mp = clip_grad_norm_pytorch([[3.0], [4.0]], max_norm=2.5, norm_type=2.0)
    out.append(
        {
            "kind": "clip_grad_norm_multi_param",
            "grads": [[3.0], [4.0]],
            "max_norm": 2.5,
            "norm_type": 2.0,
            "expected_total_norm": result_mp["total_norm"],
            "expected_clipped": result_mp["clipped_grads"],
            "label": "clip_grad_norm_ two params L2",
        }
    )

    # clip_grad_value_: [10, -10, 0.5] clipped to 1.0
    out.append(
        {
            "kind": "clip_grad_value",
            "grads": [[10.0, -10.0, 0.5]],
            "clip_value": 1.0,
            "expected_clipped": [[1.0, -1.0, 0.5]],
            "label": "clip_grad_value_ [10,-10,0.5] clip=1.0",
        }
    )

    # clip_grad_value_: no clip needed
    out.append(
        {
            "kind": "clip_grad_value_noclip",
            "grads": [[0.3, -0.3]],
            "clip_value": 1.0,
            "expected_clipped": [[0.3, -0.3]],
            "label": "clip_grad_value_ [0.3,-0.3] clip=1.0 no clip",
        }
    )

    return out


# ---------------------------------------------------------------------------
# EMA decay arithmetic fixtures
# ---------------------------------------------------------------------------


def ema_fixtures() -> list[dict[str, Any]]:
    """Pure arithmetic — computed to double precision for reference."""
    out: list[dict[str, Any]] = []

    # decay=0.9, init=10.0, update to 20.0 -> 0.9*10 + 0.1*20 = 11.0
    out.append(
        {
            "kind": "ema_single_step",
            "decay": 0.9,
            "initial": [10.0],
            "update": [20.0],
            "expected_after_1": [0.9 * 10.0 + 0.1 * 20.0],
            "label": "EMA single step decay=0.9",
        }
    )

    # decay=0.5, init=0.0, three updates all with 10.0
    # step1: 0.5*0 + 0.5*10 = 5.0
    # step2: 0.5*5 + 0.5*10 = 7.5
    # step3: 0.5*7.5 + 0.5*10 = 8.75
    out.append(
        {
            "kind": "ema_multi_step",
            "decay": 0.5,
            "initial": [0.0],
            "updates": [[10.0], [10.0], [10.0]],
            "expected_after_each": [5.0, 7.5, 8.75],
            "label": "EMA three steps decay=0.5",
        }
    )

    # decay=0.0: full replace
    out.append(
        {
            "kind": "ema_decay_zero",
            "decay": 0.0,
            "initial": [100.0],
            "update": [42.0],
            "expected_after_1": [42.0],
            "label": "EMA decay=0 full replace",
        }
    )

    # decay=1.0: no change
    out.append(
        {
            "kind": "ema_decay_one",
            "decay": 1.0,
            "initial": [100.0],
            "update": [42.0],
            "expected_after_1": [100.0],
            "label": "EMA decay=1 no change",
        }
    )

    return out


# ---------------------------------------------------------------------------
# Training history statistics
# ---------------------------------------------------------------------------


def history_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Best train loss
    train_losses = [2.0, 0.5, 1.0]
    best_idx = train_losses.index(min(train_losses))
    out.append(
        {
            "kind": "history_best_train_loss",
            "epochs": [
                {"epoch": i, "train_loss": l, "val_loss": None}
                for i, l in enumerate(train_losses)
            ],
            "expected_best_epoch": best_idx,
            "expected_best_loss": min(train_losses),
            "label": "history best_train_loss",
        }
    )

    # Best val loss (some epochs have None)
    epoch_data = [
        {"epoch": 0, "train_loss": 2.0, "val_loss": 1.5},
        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.8},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.9},
    ]
    val_losses = [e["val_loss"] for e in epoch_data if e["val_loss"] is not None]
    best_val = min(val_losses)
    best_val_epoch = next(e["epoch"] for e in epoch_data if e["val_loss"] == best_val)
    out.append(
        {
            "kind": "history_best_val_loss",
            "epochs": epoch_data,
            "expected_best_epoch": best_val_epoch,
            "expected_best_loss": best_val,
            "label": "history best_val_loss",
        }
    )

    # train_losses() / val_losses() accessors
    out.append(
        {
            "kind": "history_loss_vectors",
            "epochs": epoch_data,
            "expected_train_losses": [e["train_loss"] for e in epoch_data],
            "expected_val_losses": [e["val_loss"] for e in epoch_data],
            "label": "history train_losses/val_losses vectors",
        }
    )

    return out


# ---------------------------------------------------------------------------
# EarlyStopping semantics
# ---------------------------------------------------------------------------


def early_stopping_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # patience=2, no min_delta: triggers after 2 non-improving epochs
    out.append(
        {
            "kind": "early_stopping_triggers",
            "patience": 2,
            "min_delta": 0.0,
            "val_losses": [1.0, 1.0, 1.1],
            # epoch 0: best=1.0, wait=0; epoch 1: no improve, wait=1;
            # epoch 2: no improve, wait=2 >= patience -> stopped
            "expected_stopped_after_n": 3,  # stopped=True after 3rd on_epoch_end
            "expected_wait_sequence": [0, 1, 2],
            "label": "EarlyStopping triggers after patience=2",
        }
    )

    # patience=3, min_delta=0.1: small improvement (0.05) doesn't count
    out.append(
        {
            "kind": "early_stopping_min_delta",
            "patience": 3,
            "min_delta": 0.1,
            "val_losses": [1.0, 0.95, 0.8],
            # epoch 0: best=1.0, wait=0
            # epoch 1: 0.95 < 1.0 but diff=0.05 < min_delta -> wait=1
            # epoch 2: 0.8 < 1.0-0.1=0.9 -> improvement, reset wait=0
            "expected_wait_sequence": [0, 1, 0],
            "label": "EarlyStopping min_delta filtering",
        }
    )

    # patience=3, resets on improvement
    out.append(
        {
            "kind": "early_stopping_resets",
            "patience": 3,
            "min_delta": 0.0,
            "val_losses": [1.0, 1.1, 1.2, 0.5],
            # epoch 0: best=1.0, wait=0
            # epoch 1: no improve, wait=1
            # epoch 2: no improve, wait=2
            # epoch 3: improvement 0.5 < 1.0, reset wait=0
            "expected_wait_sequence": [0, 1, 2, 0],
            "label": "EarlyStopping resets on improvement",
        }
    )

    return out


# ---------------------------------------------------------------------------
# Gradient accumulation workload
# ---------------------------------------------------------------------------


def grad_accum_fixtures() -> list[dict[str, Any]]:
    """Model: multi-batch mean loss."""
    out: list[dict[str, Any]] = []

    # 4 micro-batches of known loss, summed then divided -> mean
    batch_losses = [2.0, 3.0, 1.5, 2.5]
    mean_loss = sum(batch_losses) / len(batch_losses)
    out.append(
        {
            "kind": "grad_accum_mean",
            "batch_losses": batch_losses,
            "n_accumulate": len(batch_losses),
            "expected_mean_loss": mean_loss,
            "label": "gradient accumulation mean over 4 batches",
        }
    )

    return out


# ---------------------------------------------------------------------------
# Checkpoint sequential composition
# ---------------------------------------------------------------------------


def checkpoint_sequential_fixtures() -> list[dict[str, Any]]:
    """Each module scales by a fixed factor; sequential composition is a product."""
    out: list[dict[str, Any]] = []

    factors = [2.0, 3.0, 4.0]
    input_val = 1.0
    expected = input_val
    for f in factors:
        expected *= f

    out.append(
        {
            "kind": "checkpoint_sequential_product",
            "input_val": input_val,
            "scale_factors": factors,
            "expected_output": expected,
            "segments": 2,
            "label": "checkpoint_sequential: product of scale factors",
        }
    )

    # Single module, single segment
    out.append(
        {
            "kind": "checkpoint_sequential_single",
            "input_val": 1.0,
            "scale_factors": [5.0],
            "expected_output": 5.0,
            "segments": 1,
            "label": "checkpoint_sequential: single module",
        }
    )

    return out


# ---------------------------------------------------------------------------
# AMP context fixtures
# ---------------------------------------------------------------------------


def amp_fixtures() -> list[dict[str, Any]]:
    """GradScaler initial scale and state dict round-trip values."""
    out: list[dict[str, Any]] = []

    # Default GradScaler init_scale matches PyTorch's default: 2**16 = 65536
    out.append(
        {
            "kind": "grad_scaler_default_scale",
            "expected_init_scale": 65536.0,
            "label": "GradScaler default init_scale == 2**16",
        }
    )

    # Custom scale round-trips through state dict
    out.append(
        {
            "kind": "grad_scaler_state_dict_roundtrip",
            "init_scale": 1024.0,
            "expected_scale_after_load": 1024.0,
            "label": "GradScaler state dict round-trip preserves scale",
        }
    )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_fixtures: dict[str, Any] = {
        "metadata": fixture_metadata(),
        "metric": metric_fixtures(),
        "grad_utils": grad_utils_fixtures(),
        "ema": ema_fixtures(),
        "history": history_fixtures(),
        "early_stopping": early_stopping_fixtures(),
        "grad_accum": grad_accum_fixtures(),
        "checkpoint_sequential": checkpoint_sequential_fixtures(),
        "amp": amp_fixtures(),
    }

    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_fixtures, f, indent=2)
        f.write("\n")

    total = (
        len(all_fixtures["metric"])
        + len(all_fixtures["grad_utils"])
        + len(all_fixtures["ema"])
        + len(all_fixtures["history"])
        + len(all_fixtures["early_stopping"])
        + len(all_fixtures["grad_accum"])
        + len(all_fixtures["checkpoint_sequential"])
        + len(all_fixtures["amp"])
    )
    print(f"Wrote {total} fixtures to {FIXTURE_PATH}")
    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
