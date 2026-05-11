#!/usr/bin/env python3
"""Pin pretrained torchvision weights to the `ferrotorch/*` HuggingFace org.

Closes ferrotorch issue #1130.

For each of the five torchvision-canonical detection / segmentation models
this script:

1. Downloads the upstream PyTorch state_dict via `torchvision.models`.
2. Re-keys it into the parameter layout that
   `ferrotorch_vision::models::{detection,segmentation}::<factory>::<f32>(...)`
   exposes through `Module::named_parameters()`. The ground-truth layout
   is captured by `ferrotorch-vision/examples/dump_torchvision_keys.rs`
   and consumed here (`/tmp/ferrotorch_keys.json`), so the mapping cannot
   drift away from the Rust side silently.
3. Hard-fails if any ferrotorch parameter key is left unfilled OR any
   torchvision key is left dangling that is not in the per-model
   `INTENTIONAL_DROP` set. This is the whole point: `strict=false` on the
   Rust loader silently drops mismatches; this script catches them at
   conversion time, where the failure is actionable.
4. Saves the converted weights as `model.safetensors` and uploads them
   to `huggingface.co/ferrotorch/<model>` alongside a README that
   includes the verbatim upstream BSD-3-Clause license (torchvision 0.21
   LICENSE — see https://github.com/pytorch/vision/blob/v0.21.0/LICENSE).

Run:
    cargo run --release -p ferrotorch-vision --example dump_torchvision_keys \
        > /tmp/ferrotorch_keys.json
    python3 scripts/pin_pretrained_weights.py [--dry-run] [--skip-upload]
                                              [--models name1,name2,...]
                                              [--keys /tmp/ferrotorch_keys.json]

`--dry-run` writes everything under `--out-dir` but never touches HF.
`--skip-upload` writes safetensors locally but does not upload (useful
when iterating on a mapping). Default `--out-dir` is
`/tmp/ferrotorch_pretrained_weights/<model>/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

import torch
import torchvision  # noqa: F401  -- ensures the model factories are importable
import torchvision.models.detection as tv_detection
import torchvision.models.segmentation as tv_segmentation
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# torchvision 0.21 LICENSE (BSD 3-Clause), quoted verbatim.
#
# Reproduced from `torchvision-0.21.0.dist-info/LICENSE` for the upstream
# repository at https://github.com/pytorch/vision/blob/v0.21.0/LICENSE.
# This text is included in every uploaded model README per the BSD-3
# redistribution-in-binary-form clause.
# ---------------------------------------------------------------------------
TORCHVISION_LICENSE = """\
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# ---------------------------------------------------------------------------
# Mapping helpers.
# ---------------------------------------------------------------------------

# torchvision SSD300_VGG16: backbone.features.<idx> torchvision indices map
# to the 10 VGG conv layers in ferrotorch features_stage1[0..10].  The
# torchvision indices interleave ReLU/pool modules, so the conv-only
# subsequence is fixed.
SSD_STAGE1_TV_INDICES = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]


def _check_shape(name: str, tensor: torch.Tensor, expected: list) -> None:
    """Assert tensor shape matches the ferrotorch dump's expected shape."""
    actual = list(tensor.shape)
    if actual != expected:
        raise SystemExit(
            f"shape mismatch for '{name}': torchvision tensor {actual} "
            f"vs ferrotorch expects {expected}. The Rust dump and the upstream "
            f"checkpoint disagree on this parameter — investigate before "
            f"continuing."
        )


def map_ssd300_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str]]:
    """Map SSD300+VGG16 torchvision state_dict to ferrotorch keys.

    Returns `(mapped_sd, used_tv_keys, filled_ft_keys)`.
    """
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()

    def put(ft: str, tv: str) -> None:
        if tv not in tv_sd:
            raise SystemExit(
                f"ssd300_vgg16: torchvision key '{tv}' missing from state_dict "
                f"(needed for ferrotorch '{ft}')"
            )
        _check_shape(ft, tv_sd[tv], ft_keys[ft])
        out[ft] = tv_sd[tv]
        used_tv.add(tv)
        filled_ft.add(ft)

    # features_stage1: 10 VGG conv layers (no BN).
    for i, ti in enumerate(SSD_STAGE1_TV_INDICES):
        put(f"features_stage1.{i}.conv.weight", f"backbone.features.{ti}.weight")
        put(f"features_stage1.{i}.conv.bias", f"backbone.features.{ti}.bias")

    # L2Norm — torchvision stores it as `backbone.scale_weight` (shape [512]),
    # ferrotorch as `l2_norm.weight` (same shape).
    put("l2_norm.weight", "backbone.scale_weight")

    # features_stage2: VGG conv5_{1,2,3} = backbone.extra.0.{1,3,5}.
    for i, ti in enumerate([1, 3, 5]):
        put(f"features_stage2.{i}.conv.weight", f"backbone.extra.0.{ti}.weight")
        put(f"features_stage2.{i}.conv.bias", f"backbone.extra.0.{ti}.bias")

    # conv6: backbone.extra.0.7.1 (atrous 3x3, 1024 out)
    put("conv6.conv.weight", "backbone.extra.0.7.1.weight")
    put("conv6.conv.bias", "backbone.extra.0.7.1.bias")
    # conv7: backbone.extra.0.7.3 (1x1, 1024 out)
    put("conv7.conv.weight", "backbone.extra.0.7.3.weight")
    put("conv7.conv.bias", "backbone.extra.0.7.3.bias")

    # extra blocks: torchvision indices 1..4, each has subconvs .0 (1x1) and .2 (3x3).
    for block in range(4):
        ti_block = block + 1  # torchvision uses .1, .2, .3, .4
        put(f"extra.{block}.0.conv.weight", f"backbone.extra.{ti_block}.0.weight")
        put(f"extra.{block}.0.conv.bias", f"backbone.extra.{ti_block}.0.bias")
        put(f"extra.{block}.1.conv.weight", f"backbone.extra.{ti_block}.2.weight")
        put(f"extra.{block}.1.conv.bias", f"backbone.extra.{ti_block}.2.bias")

    # Heads: 6 cls + 6 reg.
    for i in range(6):
        put(f"head.cls_heads.{i}.weight",
            f"head.classification_head.module_list.{i}.weight")
        put(f"head.cls_heads.{i}.bias",
            f"head.classification_head.module_list.{i}.bias")
        put(f"head.reg_heads.{i}.weight",
            f"head.regression_head.module_list.{i}.weight")
        put(f"head.reg_heads.{i}.bias",
            f"head.regression_head.module_list.{i}.bias")

    return out, used_tv, filled_ft


def _map_resnet50_backbone(
    tv_sd: dict[str, torch.Tensor],
    tv_prefix: str,
    ft_prefix: str,
    ft_keys: dict[str, list],
    out: dict[str, torch.Tensor],
    used_tv: set[str],
    filled_ft: set[str],
) -> None:
    """Map the shared ResNet50 backbone subtree.

    torchvision detection wraps the backbone as `<tv_prefix>.body.<rest>`
    and includes BN running_mean/running_var/num_batches_tracked. Segmentation
    omits the `.body` indirection — pass tv_prefix accordingly.

    ferrotorch exposes `<ft_prefix>.<rest>` via `named_parameters()` with
    no running stats (those live in `BatchNorm2d`'s `Mutex<Vec<f64>>` and
    are not yet plumbed through `named_buffers`; see #995). Running stats
    *are still included* in the safetensors output (as plain tensors keyed
    by the ferrotorch param-path with `.running_mean` / `.running_var`
    suffix) so they're available once #995 lands without re-uploading.
    """
    # Conv1 + bn1.
    pairs = [
        ("conv1.weight", "conv1.weight", "param"),
        ("bn1.weight", "bn1.weight", "param"),
        ("bn1.bias", "bn1.bias", "param"),
        ("bn1.running_mean", "bn1.running_mean", "buffer"),
        ("bn1.running_var", "bn1.running_var", "buffer"),
    ]
    for ft_sub, tv_sub, kind in pairs:
        tv_key = f"{tv_prefix}.{tv_sub}"
        if tv_key not in tv_sd:
            raise SystemExit(f"resnet50 backbone: missing torchvision key '{tv_key}'")
        ft_key = f"{ft_prefix}.{ft_sub}"
        if kind == "param":
            _check_shape(ft_key, tv_sd[tv_key], ft_keys[ft_key])
            filled_ft.add(ft_key)
        out[ft_key] = tv_sd[tv_key]
        used_tv.add(tv_key)

    # Optionally consume num_batches_tracked (not stored in our safetensors —
    # ferrotorch BatchNorm2d does not expose it; mark used_tv so we don't
    # flag it as an unmapped torchvision key).
    nbt = f"{tv_prefix}.bn1.num_batches_tracked"
    if nbt in tv_sd:
        used_tv.add(nbt)

    # Stages: layer1..layer4 with bottleneck blocks.
    BLOCKS_PER_LAYER = {1: 3, 2: 4, 3: 6, 4: 3}
    for layer in range(1, 5):
        for block in range(BLOCKS_PER_LAYER[layer]):
            base_tv = f"{tv_prefix}.layer{layer}.{block}"
            base_ft = f"{ft_prefix}.layer{layer}.{block}"
            for conv_i in (1, 2, 3):
                # convN.weight
                tv_k = f"{base_tv}.conv{conv_i}.weight"
                ft_k = f"{base_ft}.conv{conv_i}.weight"
                _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
                out[ft_k] = tv_sd[tv_k]
                used_tv.add(tv_k)
                filled_ft.add(ft_k)
                # bnN.{weight,bias} parameters, plus running stats as buffers.
                for bn_field in ("weight", "bias"):
                    tv_k = f"{base_tv}.bn{conv_i}.{bn_field}"
                    ft_k = f"{base_ft}.bn{conv_i}.{bn_field}"
                    _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
                    out[ft_k] = tv_sd[tv_k]
                    used_tv.add(tv_k)
                    filled_ft.add(ft_k)
                for stat in ("running_mean", "running_var"):
                    tv_k = f"{base_tv}.bn{conv_i}.{stat}"
                    ft_k = f"{base_ft}.bn{conv_i}.{stat}"
                    if tv_k in tv_sd:
                        out[ft_k] = tv_sd[tv_k]
                        used_tv.add(tv_k)
                nbt = f"{base_tv}.bn{conv_i}.num_batches_tracked"
                if nbt in tv_sd:
                    used_tv.add(nbt)
            # downsample present only in the first block of each layer (or
            # whenever stride/channels change).
            ds_w = f"{base_tv}.downsample.0.weight"
            if ds_w in tv_sd:
                ft_k = f"{base_ft}.downsample.0.weight"
                _check_shape(ft_k, tv_sd[ds_w], ft_keys[ft_k])
                out[ft_k] = tv_sd[ds_w]
                used_tv.add(ds_w)
                filled_ft.add(ft_k)
                for bn_field in ("weight", "bias"):
                    tv_k = f"{base_tv}.downsample.1.{bn_field}"
                    ft_k = f"{base_ft}.downsample.1.{bn_field}"
                    _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
                    out[ft_k] = tv_sd[tv_k]
                    used_tv.add(tv_k)
                    filled_ft.add(ft_k)
                for stat in ("running_mean", "running_var"):
                    tv_k = f"{base_tv}.downsample.1.{stat}"
                    ft_k = f"{base_ft}.downsample.1.{stat}"
                    if tv_k in tv_sd:
                        out[ft_k] = tv_sd[tv_k]
                        used_tv.add(tv_k)
                nbt = f"{base_tv}.downsample.1.num_batches_tracked"
                if nbt in tv_sd:
                    used_tv.add(nbt)


def _fill_resnet_fc_random(
    ft_keys: dict[str, list],
    ft_prefix: str,
    out: dict[str, torch.Tensor],
    filled_ft: set[str],
) -> None:
    """ferrotorch ResNet50 backbone unconditionally includes an `fc` head;
    torchvision's detection / segmentation models discard it. We fill those
    two keys with Kaiming-initialised tensors so `strict=true` smoke tests
    can re-load the safetensors and confirm every key is present. (The
    detection / segmentation forward paths never consult `fc`.)
    """
    fc_w_key = f"{ft_prefix}.fc.weight"
    fc_b_key = f"{ft_prefix}.fc.bias"
    if fc_w_key in ft_keys:
        fc_w_shape = ft_keys[fc_w_key]
        torch.manual_seed(0)  # deterministic so SHA is reproducible
        # Kaiming-normal init — matches torch.nn.Linear default for fan_in=in_features.
        in_features = fc_w_shape[1]
        std = (2.0 / in_features) ** 0.5
        out[fc_w_key] = torch.randn(*fc_w_shape) * std
        filled_ft.add(fc_w_key)
    if fc_b_key in ft_keys:
        out[fc_b_key] = torch.zeros(*ft_keys[fc_b_key])
        filled_ft.add(fc_b_key)


def _map_fpn_with_bias(
    tv_sd: dict[str, torch.Tensor],
    tv_prefix: str,
    ft_prefix: str,
    ft_keys: dict[str, list],
    out: dict[str, torch.Tensor],
    used_tv: set[str],
    filled_ft: set[str],
    intentional_drop_tv: set[str],
) -> None:
    """Map torchvision FPN weights AND biases into ferrotorch's FPN.

    Both torchvision (`nn.Conv2d(..., bias=True)`) and ferrotorch
    (`Conv2d::new(..., bias=true)`) emit bias parameters on FPN
    lateral (`inner_blocks.{i}.0`) and output (`layer_blocks.{i}.0`)
    convolutions, so the mapping is now one-to-one.

    Before #1141 fix, ferrotorch FPN used `bias=false`; the biases were
    recorded in `intentional_drop_tv` and silently dropped — that drop
    was the actual root cause of #1141 (FPN max-abs-diff vs torchvision
    on a real COCO image was ~0.77 at p2 / 3.4 at p6, propagating to
    multi-unit divergence in RPN cls_logits and a 924/1000 mismatch
    in post-NMS proposals).
    """
    # torchvision: inner_blocks index 0..3 corresponds to C2..C5
    # ferrotorch:  lateral2..lateral5 (matching the same C2..C5).
    # `intentional_drop_tv` is no longer mutated here — both weights and
    # biases now map cleanly. Param kept for backwards-compat with the
    # call sites (which still pass it for the dropouts in other heads).
    _ = intentional_drop_tv
    for i, level in enumerate([2, 3, 4, 5]):
        tv_w = f"{tv_prefix}.inner_blocks.{i}.0.weight"
        ft_w = f"{ft_prefix}.lateral{level}.weight"
        _check_shape(ft_w, tv_sd[tv_w], ft_keys[ft_w])
        out[ft_w] = tv_sd[tv_w]
        used_tv.add(tv_w)
        filled_ft.add(ft_w)
        tv_b = f"{tv_prefix}.inner_blocks.{i}.0.bias"
        ft_b = f"{ft_prefix}.lateral{level}.bias"
        if tv_b not in tv_sd:
            raise SystemExit(f"FPN bias missing in torchvision state_dict: {tv_b}")
        _check_shape(ft_b, tv_sd[tv_b], ft_keys[ft_b])
        out[ft_b] = tv_sd[tv_b]
        used_tv.add(tv_b)
        filled_ft.add(ft_b)

        tv_w = f"{tv_prefix}.layer_blocks.{i}.0.weight"
        ft_w = f"{ft_prefix}.output{level}.weight"
        _check_shape(ft_w, tv_sd[tv_w], ft_keys[ft_w])
        out[ft_w] = tv_sd[tv_w]
        used_tv.add(tv_w)
        filled_ft.add(ft_w)
        tv_b = f"{tv_prefix}.layer_blocks.{i}.0.bias"
        ft_b = f"{ft_prefix}.output{level}.bias"
        if tv_b not in tv_sd:
            raise SystemExit(f"FPN bias missing in torchvision state_dict: {tv_b}")
        _check_shape(ft_b, tv_sd[tv_b], ft_keys[ft_b])
        out[ft_b] = tv_sd[tv_b]
        used_tv.add(tv_b)
        filled_ft.add(ft_b)


def _map_fasterrcnn_heads(
    tv_sd: dict[str, torch.Tensor],
    ft_prefix: str,
    ft_keys: dict[str, list],
    out: dict[str, torch.Tensor],
    used_tv: set[str],
    filled_ft: set[str],
) -> None:
    """Map FasterRCNN's RPN + ROI heads (shared between FasterRCNN and MaskRCNN)."""
    pairs = [
        # RPN — torchvision wraps the 3x3 conv in a Sequential ((Conv2d, ReLU))
        # at module index 0.
        (f"{ft_prefix}rpn.head.conv.weight", "rpn.head.conv.0.0.weight"),
        (f"{ft_prefix}rpn.head.conv.bias",   "rpn.head.conv.0.0.bias"),
        (f"{ft_prefix}rpn.head.cls_logits.weight", "rpn.head.cls_logits.weight"),
        (f"{ft_prefix}rpn.head.cls_logits.bias",   "rpn.head.cls_logits.bias"),
        (f"{ft_prefix}rpn.head.bbox_pred.weight",  "rpn.head.bbox_pred.weight"),
        (f"{ft_prefix}rpn.head.bbox_pred.bias",    "rpn.head.bbox_pred.bias"),
        # ROI box head.
        (f"{ft_prefix}head.fc6.weight",         "roi_heads.box_head.fc6.weight"),
        (f"{ft_prefix}head.fc6.bias",           "roi_heads.box_head.fc6.bias"),
        (f"{ft_prefix}head.fc7.weight",         "roi_heads.box_head.fc7.weight"),
        (f"{ft_prefix}head.fc7.bias",           "roi_heads.box_head.fc7.bias"),
        (f"{ft_prefix}head.cls_score.weight",   "roi_heads.box_predictor.cls_score.weight"),
        (f"{ft_prefix}head.cls_score.bias",     "roi_heads.box_predictor.cls_score.bias"),
        (f"{ft_prefix}head.bbox_pred.weight",   "roi_heads.box_predictor.bbox_pred.weight"),
        (f"{ft_prefix}head.bbox_pred.bias",     "roi_heads.box_predictor.bbox_pred.bias"),
    ]
    for ft_k, tv_k in pairs:
        if tv_k not in tv_sd:
            raise SystemExit(f"fasterrcnn heads: missing torchvision key '{tv_k}'")
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)


def map_fasterrcnn_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map FasterRCNN ResNet50+FPN state_dict to ferrotorch keys."""
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    _map_resnet50_backbone(
        tv_sd, "backbone.body", "backbone", ft_keys,
        out, used_tv, filled_ft,
    )
    _fill_resnet_fc_random(ft_keys, "backbone", out, filled_ft)
    _map_fpn_with_bias(
        tv_sd, "backbone.fpn", "fpn", ft_keys,
        out, used_tv, filled_ft, intentional_drop_tv,
    )
    _map_fasterrcnn_heads(tv_sd, "", ft_keys, out, used_tv, filled_ft)

    return out, used_tv, filled_ft, intentional_drop_tv


def _map_keypointrcnn_box_head(
    tv_sd: dict[str, torch.Tensor],
    ft_prefix: str,
    ft_keys: dict[str, list],
    out: dict[str, torch.Tensor],
    used_tv: set[str],
    filled_ft: set[str],
) -> None:
    """Map FasterRCNN's RPN + ROI box head for KeypointRCNN.

    Same as `_map_fasterrcnn_heads` except the box predictor for keypointrcnn
    has `num_classes=2` (bg + person), so `cls_score` is `[2, 1024]` and
    `bbox_pred` is `[8, 1024]`. Both `_check_shape` calls inside use the
    ferrotorch-side recorded shape, so we just need to point at the same
    torchvision keys.
    """
    pairs = [
        # RPN — torchvision wraps the 3x3 conv in a Sequential ((Conv2d, ReLU))
        # at module index 0.
        (f"{ft_prefix}rpn.head.conv.weight", "rpn.head.conv.0.0.weight"),
        (f"{ft_prefix}rpn.head.conv.bias",   "rpn.head.conv.0.0.bias"),
        (f"{ft_prefix}rpn.head.cls_logits.weight", "rpn.head.cls_logits.weight"),
        (f"{ft_prefix}rpn.head.cls_logits.bias",   "rpn.head.cls_logits.bias"),
        (f"{ft_prefix}rpn.head.bbox_pred.weight",  "rpn.head.bbox_pred.weight"),
        (f"{ft_prefix}rpn.head.bbox_pred.bias",    "rpn.head.bbox_pred.bias"),
        # ROI box head (2 classes for keypointrcnn).
        (f"{ft_prefix}head.fc6.weight",         "roi_heads.box_head.fc6.weight"),
        (f"{ft_prefix}head.fc6.bias",           "roi_heads.box_head.fc6.bias"),
        (f"{ft_prefix}head.fc7.weight",         "roi_heads.box_head.fc7.weight"),
        (f"{ft_prefix}head.fc7.bias",           "roi_heads.box_head.fc7.bias"),
        (f"{ft_prefix}head.cls_score.weight",   "roi_heads.box_predictor.cls_score.weight"),
        (f"{ft_prefix}head.cls_score.bias",     "roi_heads.box_predictor.cls_score.bias"),
        (f"{ft_prefix}head.bbox_pred.weight",   "roi_heads.box_predictor.bbox_pred.weight"),
        (f"{ft_prefix}head.bbox_pred.bias",     "roi_heads.box_predictor.bbox_pred.bias"),
    ]
    for ft_k, tv_k in pairs:
        if tv_k not in tv_sd:
            raise SystemExit(f"keypointrcnn box head: missing torchvision key '{tv_k}'")
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)


def map_keypointrcnn_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map KeypointRCNN ResNet50+FPN state_dict to ferrotorch keys.

    Same backbone+FPN+RPN as MaskRCNN, but:
      - 2-class box predictor (bg + person) — keys are unchanged, shapes
        differ from the 91-class MaskRCNN/FasterRCNN.
      - No mask head.
      - 8-conv KeypointRCNNHeads at `roi_heads.keypoint_head.{0,2,4,6,8,10,12,14}`
        (the odd indices are ReLUs in the torchvision Sequential, no
        parameters). Mapped to ferrotorch `keypoint_head.conv{0,2,...,14}`.
      - Single ConvTranspose2d at `roi_heads.keypoint_predictor.kps_score_lowres`,
        mapped to ferrotorch `keypoint_predictor.kps_score_lowres`.
    """
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    _map_resnet50_backbone(
        tv_sd, "backbone.body", "faster_rcnn.backbone", ft_keys,
        out, used_tv, filled_ft,
    )
    _fill_resnet_fc_random(ft_keys, "faster_rcnn.backbone", out, filled_ft)
    _map_fpn_with_bias(
        tv_sd, "backbone.fpn", "faster_rcnn.fpn", ft_keys,
        out, used_tv, filled_ft, intentional_drop_tv,
    )
    _map_keypointrcnn_box_head(
        tv_sd, "faster_rcnn.", ft_keys, out, used_tv, filled_ft,
    )

    # KeypointRCNNHeads: 8 conv layers at even indices in the torchvision
    # `nn.Sequential`. The odd indices are ReLU (no params).
    for i in (0, 2, 4, 6, 8, 10, 12, 14):
        tv_w = f"roi_heads.keypoint_head.{i}.weight"
        tv_b = f"roi_heads.keypoint_head.{i}.bias"
        ft_w = f"keypoint_head.conv{i}.weight"
        ft_b = f"keypoint_head.conv{i}.bias"
        if tv_w not in tv_sd or tv_b not in tv_sd:
            raise SystemExit(
                f"keypointrcnn: torchvision key '{tv_w}' or '{tv_b}' missing"
            )
        _check_shape(ft_w, tv_sd[tv_w], ft_keys[ft_w])
        _check_shape(ft_b, tv_sd[tv_b], ft_keys[ft_b])
        out[ft_w] = tv_sd[tv_w]
        out[ft_b] = tv_sd[tv_b]
        used_tv.update({tv_w, tv_b})
        filled_ft.update({ft_w, ft_b})

    # KeypointRCNNPredictor: single ConvTranspose2d.
    pairs = [
        (
            "keypoint_predictor.kps_score_lowres.weight",
            "roi_heads.keypoint_predictor.kps_score_lowres.weight",
        ),
        (
            "keypoint_predictor.kps_score_lowres.bias",
            "roi_heads.keypoint_predictor.kps_score_lowres.bias",
        ),
    ]
    for ft_k, tv_k in pairs:
        if tv_k not in tv_sd:
            raise SystemExit(f"keypointrcnn predictor: missing '{tv_k}'")
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)

    return out, used_tv, filled_ft, intentional_drop_tv


def map_maskrcnn_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map MaskRCNN ResNet50+FPN state_dict to ferrotorch keys."""
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    _map_resnet50_backbone(
        tv_sd, "backbone.body", "faster_rcnn.backbone", ft_keys,
        out, used_tv, filled_ft,
    )
    _fill_resnet_fc_random(ft_keys, "faster_rcnn.backbone", out, filled_ft)
    _map_fpn_with_bias(
        tv_sd, "backbone.fpn", "faster_rcnn.fpn", ft_keys,
        out, used_tv, filled_ft, intentional_drop_tv,
    )
    _map_fasterrcnn_heads(tv_sd, "faster_rcnn.", ft_keys, out, used_tv, filled_ft)

    # Mask head: 4-conv FCN.
    for i in range(4):
        tv_w = f"roi_heads.mask_head.{i}.0.weight"
        tv_b = f"roi_heads.mask_head.{i}.0.bias"
        ft_w = f"mask_head.conv{i + 1}.weight"
        ft_b = f"mask_head.conv{i + 1}.bias"
        _check_shape(ft_w, tv_sd[tv_w], ft_keys[ft_w])
        _check_shape(ft_b, tv_sd[tv_b], ft_keys[ft_b])
        out[ft_w] = tv_sd[tv_w]
        out[ft_b] = tv_sd[tv_b]
        used_tv.update({tv_w, tv_b})
        filled_ft.update({ft_w, ft_b})

    # Mask predictor: deconv + 1x1 conv_logits.
    pairs = [
        ("mask_predictor.deconv.weight",       "roi_heads.mask_predictor.conv5_mask.weight"),
        ("mask_predictor.deconv.bias",         "roi_heads.mask_predictor.conv5_mask.bias"),
        ("mask_predictor.conv_logits.weight",  "roi_heads.mask_predictor.mask_fcn_logits.weight"),
        ("mask_predictor.conv_logits.bias",    "roi_heads.mask_predictor.mask_fcn_logits.bias"),
    ]
    for ft_k, tv_k in pairs:
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)

    return out, used_tv, filled_ft, intentional_drop_tv


def map_deeplabv3_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map DeepLabV3 ResNet50 state_dict to ferrotorch keys.

    torchvision segmentation models prefix backbone keys with `backbone.`
    (no `.body.`). The aux classifier is OPTIONAL in ferrotorch and not
    present in our DeepLabV3 implementation; we mark `aux_classifier.*`
    as intentionally dropped.
    """
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    _map_resnet50_backbone(
        tv_sd, "backbone", "backbone", ft_keys,
        out, used_tv, filled_ft,
    )
    _fill_resnet_fc_random(ft_keys, "backbone", out, filled_ft)

    # ASPP: classifier.0.convs.{0..4}.* and classifier.0.project.*
    # ferrotorch: head.aspp.{0..4}.*  +  head.aspp.project*  +
    #             head.{conv_intermediate, bn_intermediate, classifier}
    # branch 0: 1x1 conv (has .conv wrapper)
    pairs_b0 = [
        ("head.aspp.0.conv.weight", "classifier.0.convs.0.0.weight", "param"),
        ("head.aspp.0.bn.weight",   "classifier.0.convs.0.1.weight", "param"),
        ("head.aspp.0.bn.bias",     "classifier.0.convs.0.1.bias",   "param"),
        ("head.aspp.0.bn.running_mean", "classifier.0.convs.0.1.running_mean", "buffer"),
        ("head.aspp.0.bn.running_var",  "classifier.0.convs.0.1.running_var",  "buffer"),
    ]
    # branches 1..3: 3x3 atrous (no .conv wrapper)
    pairs_atrous = []
    for i in (1, 2, 3):
        pairs_atrous.extend([
            (f"head.aspp.{i}.weight",    f"classifier.0.convs.{i}.0.weight", "param"),
            (f"head.aspp.{i}.bn.weight", f"classifier.0.convs.{i}.1.weight", "param"),
            (f"head.aspp.{i}.bn.bias",   f"classifier.0.convs.{i}.1.bias",   "param"),
            (f"head.aspp.{i}.bn.running_mean", f"classifier.0.convs.{i}.1.running_mean", "buffer"),
            (f"head.aspp.{i}.bn.running_var",  f"classifier.0.convs.{i}.1.running_var",  "buffer"),
        ])
    # branch 4: ASPPPooling (1x1 conv after global avgpool, has .conv wrapper).
    # In torchvision module 4 is ASPPPooling = Sequential(AdaptiveAvgPool2d, Conv2d, BN, ReLU)
    # so indices .1 (conv) and .2 (BN).
    pairs_b4 = [
        ("head.aspp.4.conv.weight",     "classifier.0.convs.4.1.weight", "param"),
        ("head.aspp.4.bn.weight",       "classifier.0.convs.4.2.weight", "param"),
        ("head.aspp.4.bn.bias",         "classifier.0.convs.4.2.bias",   "param"),
        ("head.aspp.4.bn.running_mean", "classifier.0.convs.4.2.running_mean", "buffer"),
        ("head.aspp.4.bn.running_var",  "classifier.0.convs.4.2.running_var",  "buffer"),
    ]
    # Project: classifier.0.project = Sequential(Conv2d(1280→256, 1x1), BN, ReLU, Dropout)
    pairs_project = [
        ("head.aspp.project.weight",       "classifier.0.project.0.weight",       "param"),
        ("head.aspp.project_bn.weight",    "classifier.0.project.1.weight",       "param"),
        ("head.aspp.project_bn.bias",      "classifier.0.project.1.bias",         "param"),
        ("head.aspp.project_bn.running_mean", "classifier.0.project.1.running_mean", "buffer"),
        ("head.aspp.project_bn.running_var",  "classifier.0.project.1.running_var",  "buffer"),
    ]
    # Intermediate 3x3 + BN + ReLU, then 1x1 classifier (no BN/ReLU).
    pairs_tail = [
        ("head.conv_intermediate.weight",       "classifier.1.weight", "param"),
        ("head.bn_intermediate.weight",         "classifier.2.weight", "param"),
        ("head.bn_intermediate.bias",           "classifier.2.bias",   "param"),
        ("head.bn_intermediate.running_mean",   "classifier.2.running_mean", "buffer"),
        ("head.bn_intermediate.running_var",    "classifier.2.running_var",  "buffer"),
        ("head.classifier.weight",              "classifier.4.weight", "param"),
        ("head.classifier.bias",                "classifier.4.bias",   "param"),
    ]

    for group in (pairs_b0, pairs_atrous, pairs_b4, pairs_project, pairs_tail):
        for ft_k, tv_k, kind in group:
            if tv_k not in tv_sd:
                raise SystemExit(f"deeplabv3: missing torchvision key '{tv_k}'")
            if kind == "param":
                _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
                filled_ft.add(ft_k)
            out[ft_k] = tv_sd[tv_k]
            used_tv.add(tv_k)

    # num_batches_tracked is not stored on ferrotorch side — record as used.
    for nbt_key in [k for k in tv_sd if k.endswith("num_batches_tracked")]:
        used_tv.add(nbt_key)

    # aux_classifier is intentionally dropped (not present in ferrotorch model).
    for k in tv_sd:
        if k.startswith("aux_classifier."):
            intentional_drop_tv.add(k)

    return out, used_tv, filled_ft, intentional_drop_tv


def map_retinanet_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map RetinaNet ResNet50+FPN state_dict to ferrotorch keys.

    Distinct from FasterRCNN/MaskRCNN:
    - FPN has only 3 lateral inputs (inner_blocks.0..2 → lateral3..5).
    - FPN extra blocks: `LastLevelP6P7` — `extra_blocks.p6.{w,b}` and
      `extra_blocks.p7.{w,b}` mapped to `fpn.p6.*` and `fpn.p7.*`.
    - Classification / regression heads: torchvision wraps the 4 inner convs
      in `Conv2dNormActivation` so the keys are
      `head.classification_head.conv.<i>.0.{weight,bias}`. The final
      cls_logits/bbox_reg are bare `nn.Conv2d`.
    """
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    # ResNet-50 backbone — same body wrapping as FasterRCNN.
    _map_resnet50_backbone(
        tv_sd, "backbone.body", "backbone", ft_keys,
        out, used_tv, filled_ft,
    )
    _fill_resnet_fc_random(ft_keys, "backbone", out, filled_ft)

    # FPN — only 3 lateral / output blocks (inner_blocks.0..2 → C3..C5 /
    # lateral3..5). _map_fpn_with_bias hard-codes 4 levels so we inline the
    # 3-level mapping here.
    fpn_tv = "backbone.fpn"
    fpn_ft = "fpn"
    for i, level in enumerate([3, 4, 5]):
        for kind in ("weight", "bias"):
            tv_w = f"{fpn_tv}.inner_blocks.{i}.0.{kind}"
            ft_w = f"{fpn_ft}.lateral{level}.{kind}"
            if tv_w not in tv_sd:
                raise SystemExit(
                    f"retinanet: torchvision key '{tv_w}' missing "
                    f"(needed for '{ft_w}')"
                )
            _check_shape(ft_w, tv_sd[tv_w], ft_keys[ft_w])
            out[ft_w] = tv_sd[tv_w]
            used_tv.add(tv_w)
            filled_ft.add(ft_w)

            tv_o = f"{fpn_tv}.layer_blocks.{i}.0.{kind}"
            ft_o = f"{fpn_ft}.output{level}.{kind}"
            if tv_o not in tv_sd:
                raise SystemExit(
                    f"retinanet: torchvision key '{tv_o}' missing "
                    f"(needed for '{ft_o}')"
                )
            _check_shape(ft_o, tv_sd[tv_o], ft_keys[ft_o])
            out[ft_o] = tv_sd[tv_o]
            used_tv.add(tv_o)
            filled_ft.add(ft_o)

    # LastLevelP6P7: extra_blocks.p6/p7.
    for px in ("p6", "p7"):
        for kind in ("weight", "bias"):
            tv_k = f"{fpn_tv}.extra_blocks.{px}.{kind}"
            ft_k = f"{fpn_ft}.{px}.{kind}"
            if tv_k not in tv_sd:
                raise SystemExit(
                    f"retinanet: torchvision key '{tv_k}' missing "
                    f"(needed for '{ft_k}')"
                )
            _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
            out[ft_k] = tv_sd[tv_k]
            used_tv.add(tv_k)
            filled_ft.add(ft_k)

    # Classification head: 4 conv layers + final cls_logits.
    for i in range(4):
        for kind in ("weight", "bias"):
            tv_k = f"head.classification_head.conv.{i}.0.{kind}"
            ft_k = f"classification_head.conv.{i}.{kind}"
            _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
            out[ft_k] = tv_sd[tv_k]
            used_tv.add(tv_k)
            filled_ft.add(ft_k)
    for kind in ("weight", "bias"):
        tv_k = f"head.classification_head.cls_logits.{kind}"
        ft_k = f"classification_head.cls_logits.{kind}"
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)

    # Regression head: 4 conv layers + final bbox_reg.
    for i in range(4):
        for kind in ("weight", "bias"):
            tv_k = f"head.regression_head.conv.{i}.0.{kind}"
            ft_k = f"regression_head.conv.{i}.{kind}"
            _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
            out[ft_k] = tv_sd[tv_k]
            used_tv.add(tv_k)
            filled_ft.add(ft_k)
    for kind in ("weight", "bias"):
        tv_k = f"head.regression_head.bbox_reg.{kind}"
        ft_k = f"regression_head.bbox_reg.{kind}"
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)

    return out, used_tv, filled_ft, intentional_drop_tv


def map_fcos_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map FCOS ResNet50+FPN state_dict to ferrotorch keys.

    Architecture sharing with RetinaNet:
    - ResNet-50 backbone via `backbone.body.*`.
    - FPN P3..P7 with `inner_blocks` / `layer_blocks` / `extra_blocks.{p6,p7}`.

    Differences from RetinaNet:
    - Heads use 4× (Conv + GroupNorm + ReLU) instead of 4× Conv. torchvision
      lays this out as `nn.Sequential(Conv, GN, ReLU, Conv, GN, ReLU, ...)`,
      so the Conv lives at indices 0, 3, 6, 9 and GroupNorm at 1, 4, 7, 10.
      Ferrotorch named_parameters mirror this indexing so the mapping is
      almost identity.
    - The regression head has TWO output convs sharing the trunk:
      `bbox_reg` (4 channels) and `bbox_ctrness` (1 channel — the
      centerness branch unique to FCOS).
    - No `cls_logits` bias prior init in pretrained (torchvision overwrites it
      anyway).
    """
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    # ResNet-50 backbone — same body wrapping as RetinaNet/FasterRCNN.
    _map_resnet50_backbone(
        tv_sd, "backbone.body", "backbone", ft_keys,
        out, used_tv, filled_ft,
    )
    _fill_resnet_fc_random(ft_keys, "backbone", out, filled_ft)

    # FPN — 3 lateral / output blocks (inner_blocks.0..2 → lateral3..5),
    # then LastLevelP6P7 extras. Identical to RetinaNet (#1143).
    fpn_tv = "backbone.fpn"
    fpn_ft = "fpn"
    for i, level in enumerate([3, 4, 5]):
        for kind in ("weight", "bias"):
            tv_w = f"{fpn_tv}.inner_blocks.{i}.0.{kind}"
            ft_w = f"{fpn_ft}.lateral{level}.{kind}"
            if tv_w not in tv_sd:
                raise SystemExit(
                    f"fcos: torchvision key '{tv_w}' missing "
                    f"(needed for '{ft_w}')"
                )
            _check_shape(ft_w, tv_sd[tv_w], ft_keys[ft_w])
            out[ft_w] = tv_sd[tv_w]
            used_tv.add(tv_w)
            filled_ft.add(ft_w)

            tv_o = f"{fpn_tv}.layer_blocks.{i}.0.{kind}"
            ft_o = f"{fpn_ft}.output{level}.{kind}"
            if tv_o not in tv_sd:
                raise SystemExit(
                    f"fcos: torchvision key '{tv_o}' missing "
                    f"(needed for '{ft_o}')"
                )
            _check_shape(ft_o, tv_sd[tv_o], ft_keys[ft_o])
            out[ft_o] = tv_sd[tv_o]
            used_tv.add(tv_o)
            filled_ft.add(ft_o)

    for px in ("p6", "p7"):
        for kind in ("weight", "bias"):
            tv_k = f"{fpn_tv}.extra_blocks.{px}.{kind}"
            ft_k = f"{fpn_ft}.{px}.{kind}"
            if tv_k not in tv_sd:
                raise SystemExit(
                    f"fcos: torchvision key '{tv_k}' missing "
                    f"(needed for '{ft_k}')"
                )
            _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
            out[ft_k] = tv_sd[tv_k]
            used_tv.add(tv_k)
            filled_ft.add(ft_k)

    # Head trunks: Conv at idx 0,3,6,9 + GroupNorm at idx 1,4,7,10 (ReLU at
    # 2,5,8,11 contributes no params). Same indexing on both sides — the
    # mapping is identity.
    head_indices = [0, 1, 3, 4, 6, 7, 9, 10]
    for head_name in ("classification_head", "regression_head"):
        for idx in head_indices:
            for kind in ("weight", "bias"):
                tv_k = f"head.{head_name}.conv.{idx}.{kind}"
                ft_k = f"{head_name}.conv.{idx}.{kind}"
                if tv_k not in tv_sd:
                    raise SystemExit(
                        f"fcos: torchvision key '{tv_k}' missing "
                        f"(needed for '{ft_k}')"
                    )
                _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
                out[ft_k] = tv_sd[tv_k]
                used_tv.add(tv_k)
                filled_ft.add(ft_k)

    # Final output convs.
    for kind in ("weight", "bias"):
        tv_k = f"head.classification_head.cls_logits.{kind}"
        ft_k = f"classification_head.cls_logits.{kind}"
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)

        tv_k = f"head.regression_head.bbox_reg.{kind}"
        ft_k = f"regression_head.bbox_reg.{kind}"
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)

        tv_k = f"head.regression_head.bbox_ctrness.{kind}"
        ft_k = f"regression_head.bbox_ctrness.{kind}"
        _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)
        filled_ft.add(ft_k)

    return out, used_tv, filled_ft, intentional_drop_tv


def map_fcn_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map FCN ResNet50 state_dict to ferrotorch keys.

    Both sides expose the same `classifier.{0,1,4}.*` keys in the FCN head;
    only difference is running stats (not in ferrotorch named_parameters).
    aux_classifier is intentionally dropped (matches ferrotorch FCN model).
    """
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    _map_resnet50_backbone(
        tv_sd, "backbone", "backbone", ft_keys,
        out, used_tv, filled_ft,
    )
    _fill_resnet_fc_random(ft_keys, "backbone", out, filled_ft)

    pairs = [
        ("classifier.0.weight",             "classifier.0.weight", "param"),
        ("classifier.1.weight",             "classifier.1.weight", "param"),
        ("classifier.1.bias",               "classifier.1.bias",   "param"),
        ("classifier.1.running_mean",       "classifier.1.running_mean", "buffer"),
        ("classifier.1.running_var",        "classifier.1.running_var",  "buffer"),
        ("classifier.4.weight",             "classifier.4.weight", "param"),
        ("classifier.4.bias",               "classifier.4.bias",   "param"),
    ]
    for ft_k, tv_k, kind in pairs:
        if tv_k not in tv_sd:
            raise SystemExit(f"fcn: missing torchvision key '{tv_k}'")
        if kind == "param":
            _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
            filled_ft.add(ft_k)
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)

    for nbt_key in [k for k in tv_sd if k.endswith("num_batches_tracked")]:
        used_tv.add(nbt_key)
    for k in tv_sd:
        if k.startswith("aux_classifier."):
            intentional_drop_tv.add(k)

    return out, used_tv, filled_ft, intentional_drop_tv


def map_lraspp_keys(
    tv_sd: dict[str, torch.Tensor],
    ft_keys: dict[str, list],
) -> tuple[dict[str, torch.Tensor], set[str], set[str], set[str]]:
    """Map LRASPP MobileNetV3-Large state_dict to ferrotorch keys (#1146).

    torchvision's `lraspp_mobilenet_v3_large` wraps the backbone in
    `IntermediateLayerGetter({"4": "low", "16": "high"})`, which strips
    the `features.` indirection: torchvision keys are
    `backbone.<i>.{...}` for i in 0..16 (and the classifier head
    skipped entirely). ferrotorch keeps the `backbone.features.<i>.{...}`
    layout exposed by `MobileNetV3Large::named_parameters()`, so we
    re-insert the missing `features.` segment.

    NO INTENTIONAL DROPS — every torchvision key MUST be either mapped
    or covered by the BN-stat handling. The FPN-bias-drop bug from
    #1141 was a silent-drop failure mode; this mapper hard-fails if
    any torchvision key is left dangling.
    """
    out: dict[str, torch.Tensor] = {}
    used_tv: set[str] = set()
    filled_ft: set[str] = set()
    intentional_drop_tv: set[str] = set()

    def put(ft_k: str, tv_k: str, kind: str) -> None:
        if tv_k not in tv_sd:
            raise SystemExit(f"lraspp: torchvision key '{tv_k}' missing")
        if kind == "param":
            _check_shape(ft_k, tv_sd[tv_k], ft_keys[ft_k])
            filled_ft.add(ft_k)
        out[ft_k] = tv_sd[tv_k]
        used_tv.add(tv_k)

    # Backbone (MobileNetV3-Large dilated). 17 indices: 0=stem (3-children
    # Conv2dNormActivation), 1..15 = InvertedResidual, 16 = head conv.
    # ferrotorch retains `features.` whereas torchvision drops it via
    # IntermediateLayerGetter — re-insert.
    backbone_tv_keys = [k for k in tv_sd if k.startswith("backbone.")]
    for tv_k in backbone_tv_keys:
        # Strip the leading "backbone." then re-prefix with
        # "backbone.features.".
        suffix = tv_k[len("backbone."):]
        ft_k = f"backbone.features.{suffix}"
        # Sort by kind: BN stats / num_batches_tracked are buffers; the
        # rest are params.
        if suffix.endswith("num_batches_tracked"):
            # BN num_batches_tracked is not exposed as a ferrotorch
            # parameter — record as used so the coverage check doesn't
            # complain. Skip emitting it (matches the FCN/DeepLabV3
            # pattern via the trailing nbt sweep).
            used_tv.add(tv_k)
            continue
        if suffix.endswith("running_mean") or suffix.endswith("running_var"):
            kind = "buffer"
        else:
            kind = "param"
            if ft_k not in ft_keys:
                raise SystemExit(
                    f"lraspp: ferrotorch key '{ft_k}' not found for "
                    f"torchvision '{tv_k}' (shape {list(tv_sd[tv_k].shape)})"
                )
        put(ft_k, tv_k, kind)

    # LRASPP head (`classifier.*` on both sides, identical structure).
    head_pairs = [
        ("classifier.cbr.0.weight",          "classifier.cbr.0.weight",          "param"),
        ("classifier.cbr.1.weight",          "classifier.cbr.1.weight",          "param"),
        ("classifier.cbr.1.bias",            "classifier.cbr.1.bias",            "param"),
        ("classifier.cbr.1.running_mean",    "classifier.cbr.1.running_mean",    "buffer"),
        ("classifier.cbr.1.running_var",     "classifier.cbr.1.running_var",     "buffer"),
        ("classifier.scale.1.weight",        "classifier.scale.1.weight",        "param"),
        ("classifier.low_classifier.weight", "classifier.low_classifier.weight", "param"),
        ("classifier.low_classifier.bias",   "classifier.low_classifier.bias",   "param"),
        ("classifier.high_classifier.weight","classifier.high_classifier.weight","param"),
        ("classifier.high_classifier.bias",  "classifier.high_classifier.bias",  "param"),
    ]
    for ft_k, tv_k, kind in head_pairs:
        put(ft_k, tv_k, kind)

    # Sweep up any remaining num_batches_tracked under classifier.* (the
    # cbr BN has one — also marked used via the explicit pair above? No:
    # we listed running_mean/var, NOT num_batches_tracked. Mark it).
    for nbt_key in [k for k in tv_sd if k.endswith("num_batches_tracked")]:
        used_tv.add(nbt_key)

    return out, used_tv, filled_ft, intentional_drop_tv


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "ssd300_vgg16": dict(
        factory=lambda: tv_detection.ssd300_vgg16(weights="COCO_V1"),
        weights_enum="COCO_V1",
        num_classes=91,
        mapper=map_ssd300_keys,
        has_intentional_drops=False,  # SSD has no FPN/aux dropouts
        param_count=35_641_826,
        description=(
            "SSD300 with VGG-16 backbone, pretrained on COCO. Re-keyed from "
            "torchvision 0.21 `ssd300_vgg16` (`SSD300_VGG16_Weights.COCO_V1`)."
        ),
    ),
    "fasterrcnn_resnet50_fpn": dict(
        factory=lambda: tv_detection.fasterrcnn_resnet50_fpn(weights="COCO_V1"),
        weights_enum="COCO_V1",
        num_classes=91,
        mapper=map_fasterrcnn_keys,
        has_intentional_drops=True,  # FPN biases
        param_count=41_755_286,
        description=(
            "Faster R-CNN with ResNet-50 + FPN backbone, pretrained on COCO. "
            "Re-keyed from torchvision 0.21 `fasterrcnn_resnet50_fpn` "
            "(`FasterRCNN_ResNet50_FPN_Weights.COCO_V1`)."
        ),
    ),
    "maskrcnn_resnet50_fpn": dict(
        factory=lambda: tv_detection.maskrcnn_resnet50_fpn(weights="COCO_V1"),
        weights_enum="COCO_V1",
        num_classes=91,
        mapper=map_maskrcnn_keys,
        has_intentional_drops=True,  # FPN biases
        param_count=44_401_393,
        description=(
            "Mask R-CNN with ResNet-50 + FPN backbone, pretrained on COCO. "
            "Re-keyed from torchvision 0.21 `maskrcnn_resnet50_fpn` "
            "(`MaskRCNN_ResNet50_FPN_Weights.COCO_V1`)."
        ),
    ),
    "deeplabv3_resnet50": dict(
        factory=lambda: tv_segmentation.deeplabv3_resnet50(
            weights="COCO_WITH_VOC_LABELS_V1"
        ),
        weights_enum="COCO_WITH_VOC_LABELS_V1",
        num_classes=21,
        mapper=map_deeplabv3_keys,
        has_intentional_drops=True,  # aux_classifier.*
        param_count=42_004_074,
        description=(
            "DeepLabV3 with ResNet-50 backbone, pretrained on a COCO subset "
            "with Pascal VOC labels (21 classes). Re-keyed from torchvision "
            "0.21 `deeplabv3_resnet50` "
            "(`DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`)."
        ),
    ),
    "fcn_resnet50": dict(
        factory=lambda: tv_segmentation.fcn_resnet50(
            weights="COCO_WITH_VOC_LABELS_V1"
        ),
        weights_enum="COCO_WITH_VOC_LABELS_V1",
        num_classes=21,
        mapper=map_fcn_keys,
        has_intentional_drops=True,  # aux_classifier.*
        param_count=35_322_218,
        description=(
            "FCN with ResNet-50 backbone, pretrained on a COCO subset with "
            "Pascal VOC labels (21 classes). Re-keyed from torchvision 0.21 "
            "`fcn_resnet50` (`FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`)."
        ),
    ),
    "retinanet_resnet50_fpn": dict(
        factory=lambda: tv_detection.retinanet_resnet50_fpn(weights="COCO_V1"),
        weights_enum="COCO_V1",
        num_classes=91,
        mapper=map_retinanet_keys,
        has_intentional_drops=False,
        param_count=34_014_999,
        description=(
            "RetinaNet with ResNet-50 + FPN(P3-P7) backbone, pretrained on "
            "COCO. Re-keyed from torchvision 0.21 `retinanet_resnet50_fpn` "
            "(`RetinaNet_ResNet50_FPN_Weights.COCO_V1`). 9 anchors/location, "
            "shared 4-conv class/reg heads, sigmoid scoring."
        ),
    ),
    "fcos_resnet50_fpn": dict(
        factory=lambda: tv_detection.fcos_resnet50_fpn(weights="COCO_V1"),
        weights_enum="COCO_V1",
        num_classes=91,
        mapper=map_fcos_keys,
        has_intentional_drops=False,
        param_count=32_269_600,
        description=(
            "FCOS with ResNet-50 + FPN(P3-P7) backbone, pretrained on COCO. "
            "Re-keyed from torchvision 0.21 `fcos_resnet50_fpn` "
            "(`FCOS_ResNet50_FPN_Weights.COCO_V1`). Anchor-free one-stage "
            "detector: single anchor/location + centerness branch, GroupNorm "
            "in 4-conv shared trunks for both class and regression heads, "
            "sigmoid scoring gated by centerness."
        ),
    ),
    "lraspp_mobilenet_v3_large": dict(
        factory=lambda: tv_segmentation.lraspp_mobilenet_v3_large(
            weights="COCO_WITH_VOC_LABELS_V1"
        ),
        weights_enum="COCO_WITH_VOC_LABELS_V1",
        num_classes=21,
        mapper=map_lraspp_keys,
        # No intentional drops: every torchvision key maps. The
        # `IntermediateLayerGetter` flattening (backbone.X vs
        # backbone.features.X) is handled inside the mapper, so we
        # require strict coverage.
        has_intentional_drops=False,
        param_count=3_221_538,
        description=(
            "LRASPP with MobileNetV3-Large dilated backbone, pretrained on a "
            "COCO subset with Pascal VOC labels (21 classes). Re-keyed from "
            "torchvision 0.21 `lraspp_mobilenet_v3_large` "
            "(`LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1`). "
            "#1146 — Phase A.4 of real-artifact-driven development."
        ),
    ),
    "keypointrcnn_resnet50_fpn": dict(
        factory=lambda: tv_detection.keypointrcnn_resnet50_fpn(weights="COCO_V1"),
        weights_enum="COCO_V1",
        num_classes=2,
        mapper=map_keypointrcnn_keys,
        # FPN bias drops same as fasterrcnn/maskrcnn (the post-#1141 ferrotorch
        # FPN is bias=True so this is actually NO drops — but keep True so the
        # README explainer covers the documented FPN-bias provenance).
        has_intentional_drops=True,
        param_count=59_137_258,
        description=(
            "Keypoint R-CNN with ResNet-50 + FPN backbone for COCO person "
            "keypoint detection. Re-keyed from torchvision 0.21 "
            "`keypointrcnn_resnet50_fpn` "
            "(`KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`). Same FasterRCNN "
            "body as #1141 (with FPN biases) but with num_classes=2 "
            "(background + person) for the box predictor, plus an 8-conv "
            "KeypointRCNNHeads (256→512→...→512) and a single-deconv "
            "KeypointRCNNPredictor outputting 17 keypoint heatmap channels."
        ),
    ),
}


def render_readme(name: str, sha: str, info: dict, intentional_drop_summary: str) -> str:
    """Render the per-repo README that ships alongside model.safetensors."""
    return textwrap.dedent(f"""\
        ---
        license: bsd-3-clause
        tags:
          - vision
          - pytorch
          - torchvision
          - ferrotorch
        ---

        # `ferrotorch/{name}`

        {info['description']}

        ## Provenance

        * Upstream factory: `torchvision.models.{info.get('category', '')}{name}`
          with `weights="{info['weights_enum']}"` (torchvision 0.21).
        * Conversion script: [`ferrotorch/scripts/pin_pretrained_weights.py`](https://github.com/dollspace/ferrotorch/blob/main/scripts/pin_pretrained_weights.py).
        * Ferrotorch issue: <https://github.com/dollspace/ferrotorch/issues/1130>.
        * Number of trainable parameters in upstream torchvision model: **{info['param_count']:,}**.
        * SHA-256 of `model.safetensors` (this file is pinned in
          `ferrotorch-hub/src/registry.rs`): `{sha}`.

        ## How to load

        ```rust
        use ferrotorch_vision::models::registry::get_model;
        let model = get_model("{name}", /* pretrained = */ true, /* num_classes = */ {info['num_classes']}).unwrap();
        ```

        The loader downloads this file, verifies SHA-256, then calls
        `Module::load_state_dict(state_dict, strict=false)`. `strict=false`
        is required because ferrotorch's `Module::named_parameters()` does
        not yet expose `BatchNorm2d` running statistics
        (`running_mean` / `running_var`), so those keys in this safetensors
        file are intentionally ignored at load time until ferrotorch
        issue #995 closes. They are still included here so re-uploading
        is unnecessary once that work lands.

        ## Conversion notes

        {intentional_drop_summary}

        ## Upstream license (verbatim, torchvision 0.21 `LICENSE`)

        ```
{textwrap.indent(TORCHVISION_LICENSE, '        ')}
        ```
    """)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_one(
    name: str,
    info: dict,
    ft_keys_full: dict,
    out_root: Path,
) -> tuple[str, Path, str]:
    """Build, map, save one model. Returns (sha256, model_path, drop_summary)."""
    print(f"\n=== {name} ===", flush=True)
    # ferrotorch param shapes for this model.
    ft_entry = ft_keys_full[name]
    ft_keys = {k: shape for k, shape in ft_entry["parameters"]}

    # Pull the torchvision state_dict.
    print(f"  downloading torchvision {name} weights…", flush=True)
    m = info["factory"]()
    tv_sd = m.state_dict()
    print(f"  upstream keys: {len(tv_sd)}; ferrotorch params: {len(ft_keys)}",
          flush=True)

    # Run the mapper.
    result = info["mapper"](tv_sd, ft_keys)
    if len(result) == 3:
        mapped, used_tv, filled_ft = result
        intentional_drop_tv = set()
    else:
        mapped, used_tv, filled_ft, intentional_drop_tv = result

    # Coverage check 1: every ferrotorch param key must be filled.
    missing_ft = set(ft_keys) - filled_ft
    if missing_ft:
        raise SystemExit(
            f"{name}: {len(missing_ft)} ferrotorch parameter keys left "
            f"unfilled — refusing to upload. Sample: "
            f"{sorted(missing_ft)[:5]}"
        )

    # Coverage check 2: every torchvision key must be either used or
    # explicitly intentional-dropped.
    unused_tv = set(tv_sd) - used_tv - intentional_drop_tv
    if unused_tv:
        raise SystemExit(
            f"{name}: {len(unused_tv)} torchvision keys are neither mapped nor "
            f"in INTENTIONAL_DROP; refusing to upload (this is the failure "
            f"mode the script exists to catch). Sample: "
            f"{sorted(unused_tv)[:10]}"
        )

    print(f"  ferrotorch params filled: {len(filled_ft)}/{len(ft_keys)}", flush=True)
    print(f"  torchvision keys used:    {len(used_tv)}/{len(tv_sd)}", flush=True)
    print(f"  intentional drops:        {len(intentional_drop_tv)}", flush=True)

    # Save.
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.safetensors"

    # safetensors requires contiguous tensors and contiguous-by-stride for
    # fp16/bf16; convert to contiguous fp32 before saving.
    to_save = {k: v.detach().to(torch.float32).contiguous() for k, v in mapped.items()}
    save_file(to_save, str(model_path))

    sha = sha256_of(model_path)
    print(f"  wrote {model_path}  sha256={sha}", flush=True)

    # Drop summary for README.
    if intentional_drop_tv:
        drop_summary = (
            "The following upstream torchvision keys are intentionally "
            "dropped because the ferrotorch architecture does not have a "
            "corresponding parameter slot:\n\n"
            + "\n".join(f"  * `{k}`" for k in sorted(intentional_drop_tv))
            + "\n\nFor FPN bias drops this is a known mismatch between "
            "torchvision's `nn.Conv2d(..., bias=True)` FPN convolutions and "
            "ferrotorch's `bias=False` FPN convolutions. For "
            "`aux_classifier.*` drops the ferrotorch DeepLabV3 / FCN "
            "implementations do not expose an aux head."
        )
    else:
        drop_summary = "All upstream torchvision keys were mapped 1:1 to ferrotorch parameter slots."

    # README.
    readme_path = out_dir / "README.md"
    readme_path.write_text(render_readme(name, sha, info, drop_summary))
    print(f"  wrote {readme_path}", flush=True)

    return sha, model_path, drop_summary


def hf_upload(name: str, model_path: Path, readme_path: Path) -> None:
    """Upload model.safetensors + README.md to ferrotorch/<name>."""
    from huggingface_hub import HfApi  # local import — only needed for upload
    api = HfApi()
    repo_id = f"ferrotorch/{name}"
    print(f"  uploading to https://huggingface.co/{repo_id}", flush=True)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="model.safetensors",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: pin pretrained safetensors for {name} (#1130)",
    )
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"docs: add README + license for {name} (#1130)",
    )
    print(f"  upload done.", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keys", default="/tmp/ferrotorch_keys.json",
                   help="Path to dump_torchvision_keys output JSON.")
    p.add_argument("--out-dir", default="/tmp/ferrotorch_pretrained_weights",
                   help="Where to write model.safetensors + README.md per model.")
    p.add_argument("--dry-run", action="store_true",
                   help="Convert + save locally but do not upload to HF.")
    p.add_argument("--skip-upload", action="store_true",
                   help="Same as --dry-run (kept for backward compat).")
    p.add_argument("--models", default="",
                   help="Comma-separated subset of models to run (default: all 5).")
    args = p.parse_args()

    ft_keys_path = Path(args.keys)
    if not ft_keys_path.exists():
        print(f"missing {ft_keys_path}: run "
              f"`cargo run --release -p ferrotorch-vision --example "
              f"dump_torchvision_keys > {ft_keys_path}` first.",
              file=sys.stderr)
        return 2
    ft_keys_full = json.loads(ft_keys_path.read_text())

    selected = list(MODELS) if not args.models else [m.strip() for m in args.models.split(",")]
    for s in selected:
        if s not in MODELS:
            print(f"unknown model '{s}'. Available: {list(MODELS)}", file=sys.stderr)
            return 2

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str]] = []
    for name in selected:
        sha, model_path, _drop_summary = convert_one(name, MODELS[name], ft_keys_full, out_root)
        summary.append((name, sha))
        if not (args.dry_run or args.skip_upload):
            hf_upload(name, model_path, model_path.parent / "README.md")

    print("\n=== SUMMARY ===")
    for name, sha in summary:
        print(f"  {name:32s}  sha256={sha}")

    # Pretty-print the registry-ready snippet.
    print("\n=== Drop-in registry SHA pins (for ferrotorch-hub/src/registry.rs) ===")
    for name, sha in summary:
        print(f'  // {name}')
        print(f'  weights_sha256: "{sha}",')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
