#!/usr/bin/env python3
"""#1141 round-4 per-stage RPN probe — torchvision side.

Loads `fasterrcnn_resnet50_fpn` pretrained, registers forward hooks on the
FPN body and RPN head per-level outputs, runs the same preprocessing as
the Rust probe on the same image, and dumps the intermediate tensors so
we can diff against the Rust dump stage by stage.

Then loads the Rust safetensors dump (produced by
`examples/probe_rpn_stages_1141.rs`) and prints a per-stage / per-level
max-abs-diff and mean-abs-diff table.

Usage:
  python3 scripts/probe_rpn_stages_1141.py \
      --image /tmp/ferrotorch_verify_images/coco_000000087038.jpg \
      --rust-dump /tmp/ferrotorch_probe_1141_rust.safetensors

The first stage to break the f32 round-off floor (~1e-5) is the bug site.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.image_list import ImageList

LEVEL_KEYS = ["p2", "p3", "p4", "p5", "p6"]


def preprocess_fasterrcnn(chw: torch.Tensor) -> torch.Tensor:
    """Same recipe as ferrotorch-vision/examples/inference_dump.rs."""
    _, h, w = chw.shape
    s_min = 800.0 / min(h, w)
    s_max = 1333.0 / max(h, w)
    scale = min(s_min, s_max)
    out_h = round(h * scale)
    out_w = round(w * scale)
    bchw = F.interpolate(chw.unsqueeze(0), size=(out_h, out_w),
                         mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normed = (bchw - mean) / std
    stride = 32
    pad_h = ((out_h + stride - 1) // stride) * stride
    pad_w = ((out_w + stride - 1) // stride) * stride
    if pad_h == out_h and pad_w == out_w:
        return normed
    padded = torch.zeros(1, 3, pad_h, pad_w)
    padded[:, :, :out_h, :out_w] = normed
    return padded


def load_image_chw(path: Path) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class TvCapture:
    """Forward hooks that record FPN feature outputs and RPN per-level
    cls / bbox outputs."""

    def __init__(self) -> None:
        self.fpn_outs: Dict[str, torch.Tensor] = {}  # raw, by torchvision key
        self.rpn_cls: List[torch.Tensor] = []
        self.rpn_bbox: List[torch.Tensor] = []
        self.anchors_concat: torch.Tensor | None = None

    def attach(self, model: torch.nn.Module) -> None:
        # FPN: the FPN body returns an OrderedDict {"0": p2, "1": p3, "2": p4, "3": p5, "pool": p6}.
        fpn = model.backbone.fpn

        def fpn_hook(_module, _inp, out):
            # `out` is an OrderedDict.
            self.fpn_outs = {k: v.detach() for k, v in out.items()}

        fpn.register_forward_hook(fpn_hook)

        # RPN head: forward(features) returns (objectness, bbox_regression),
        # both as List[Tensor] per level. We snapshot the full lists.
        rpn_head = model.rpn.head

        def head_hook(_module, _inp, out):
            objectness, bbox = out
            self.rpn_cls = [t.detach() for t in objectness]
            self.rpn_bbox = [t.detach() for t in bbox]

        rpn_head.register_forward_hook(head_hook)

        # Anchor generator: capture concatenated anchors per image.
        anchor_gen = model.rpn.anchor_generator

        def anchor_hook(_module, _inp, out):
            # `out` is List[Tensor] of length B; we use the first image's.
            self.anchors_concat = out[0].detach()

        anchor_gen.register_forward_hook(anchor_hook)


def patch_transform(model: torch.nn.Module) -> None:
    class Noop(torch.nn.Module):
        def __init__(self, parent: torch.nn.Module):
            super().__init__()
            for k in ("image_mean", "image_std", "min_size", "max_size", "size_divisible"):
                if hasattr(parent, k):
                    setattr(self, k, getattr(parent, k))

        def forward(self, images, targets=None):
            stacked = torch.stack(images, dim=0)
            image_sizes = [(t.shape[-2], t.shape[-1]) for t in images]
            return ImageList(stacked, image_sizes), targets

        def postprocess(self, result, image_shapes, original_image_sizes):
            return result

    model.transform = Noop(model.transform)


def run_torchvision_probe(input_bchw: torch.Tensor) -> TvCapture:
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    m = fasterrcnn_resnet50_fpn(weights=weights).eval()
    patch_transform(m)
    cap = TvCapture()
    cap.attach(m)
    with torch.no_grad():
        _ = m([input_bchw[0]])
    return cap


def load_rust_dump(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def stat(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, str]:
    if a.shape != b.shape:
        return float("inf"), float("inf"), f"shape mismatch rust={a.shape} tv={b.shape}"
    d = np.abs(a - b)
    return float(d.max()), float(d.mean()), ""


def decode_boxes_np(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Mirror BoxCoder weights=(1,1,1,1), one-sided clip log(1000/16)."""
    log_max = float(np.log(1000.0 / 16.0))
    x1 = anchors[:, 0]
    y1 = anchors[:, 1]
    x2 = anchors[:, 2]
    y2 = anchors[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = np.minimum(deltas[:, 2], log_max)
    dh = np.minimum(deltas[:, 3], log_max)
    px = dx * w + cx
    py = dy * h + cy
    pw = np.exp(dw) * w
    ph = np.exp(dh) * h
    out = np.empty_like(anchors)
    out[:, 0] = px - 0.5 * pw
    out[:, 1] = py - 0.5 * ph
    out[:, 2] = px + 0.5 * pw
    out[:, 3] = py + 0.5 * ph
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--rust-dump", required=True)
    args = ap.parse_args()

    img_path = Path(args.image)
    rust_path = Path(args.rust_dump)
    meta_path = rust_path.with_suffix(".json")
    if not meta_path.exists():
        print(f"WARN: no metadata JSON at {meta_path}", file=sys.stderr)

    # --- Build the same preprocessed tensor ---
    chw = load_image_chw(img_path)
    input_bchw = preprocess_fasterrcnn(chw)
    img_h, img_w = input_bchw.shape[-2:]
    print(f"input shape: {tuple(input_bchw.shape)}  (H={img_h}, W={img_w})")

    # --- Torchvision capture ---
    cap = run_torchvision_probe(input_bchw)
    # torchvision FPN keys: "0".."3", "pool" — map to our p2..p6.
    tv_fpn = {
        "p2": cap.fpn_outs["0"].numpy(),
        "p3": cap.fpn_outs["1"].numpy(),
        "p4": cap.fpn_outs["2"].numpy(),
        "p5": cap.fpn_outs["3"].numpy(),
        "p6": cap.fpn_outs["pool"].numpy(),
    }
    tv_cls = [t.numpy() for t in cap.rpn_cls]   # list of [1, A, H, W]
    tv_bbox = [t.numpy() for t in cap.rpn_bbox]  # list of [1, A*4, H, W]
    tv_anchors_concat = cap.anchors_concat.numpy() if cap.anchors_concat is not None else None

    # --- Rust dump ---
    rd = load_rust_dump(rust_path)
    print("rust keys:", sorted(rd.keys()))

    print("\n=== Stage A: FPN features per level ===")
    print(f"{'level':<6} {'rust shape':<24} {'tv shape':<24} {'max_abs':>10} {'mean_abs':>10}  {'note'}")
    for k in LEVEL_KEYS:
        rk = f"fpn_{k}"
        if rk not in rd:
            print(f"{k:<6} MISSING in rust dump")
            continue
        rv = rd[rk]
        tv = tv_fpn[k]
        ma, me, note = stat(rv, tv)
        print(f"{k:<6} {str(rv.shape):<24} {str(tv.shape):<24} {ma:>10.4g} {me:>10.4g}  {note}")

    print("\n=== Stage B: RPN cls_logits per level ===")
    print(f"{'level':<6} {'rust shape':<24} {'tv shape':<24} {'max_abs':>10} {'mean_abs':>10}  {'note'}")
    for i, k in enumerate(LEVEL_KEYS):
        rk = f"cls_{k}"
        if rk not in rd:
            print(f"{k:<6} MISSING in rust dump")
            continue
        rv = rd[rk]
        tv = tv_cls[i]
        ma, me, note = stat(rv, tv)
        print(f"{k:<6} {str(rv.shape):<24} {str(tv.shape):<24} {ma:>10.4g} {me:>10.4g}  {note}")

    print("\n=== Stage C: RPN bbox_deltas per level ===")
    print(f"{'level':<6} {'rust shape':<24} {'tv shape':<24} {'max_abs':>10} {'mean_abs':>10}  {'note'}")
    for i, k in enumerate(LEVEL_KEYS):
        rk = f"bbox_{k}"
        if rk not in rd:
            print(f"{k:<6} MISSING in rust dump")
            continue
        rv = rd[rk]
        tv = tv_bbox[i]
        ma, me, note = stat(rv, tv)
        print(f"{k:<6} {str(rv.shape):<24} {str(tv.shape):<24} {ma:>10.4g} {me:>10.4g}  {note}")

    # Reconstruct tv anchors per level so we can also diff anchors and
    # decoded boxes. The anchor generator hooked output is concatenated
    # across levels in the same order the RPN flattens them: per-level
    # (H*W*A, 4), concatenated in level order.
    if tv_anchors_concat is not None:
        # Per-level counts.
        per_lvl_counts: List[int] = []
        for i, k in enumerate(LEVEL_KEYS):
            shape = tv_cls[i].shape  # [1, A, H, W]
            a = shape[1]
            h = shape[2]
            w = shape[3]
            per_lvl_counts.append(a * h * w)
        offsets = np.cumsum([0] + per_lvl_counts)
        tv_anchors_per_level = {}
        for i, k in enumerate(LEVEL_KEYS):
            tv_anchors_per_level[k] = tv_anchors_concat[offsets[i]:offsets[i + 1]]

        print("\n=== Stage D: Anchor coordinates per level ===")
        print(f"{'level':<6} {'rust shape':<24} {'tv shape':<24} {'max_abs':>10} {'mean_abs':>10}  {'note'}")
        for k in LEVEL_KEYS:
            rk = f"anchors_{k}"
            if rk not in rd:
                print(f"{k:<6} MISSING in rust dump")
                continue
            rv = rd[rk]
            tv = tv_anchors_per_level[k]
            ma, me, note = stat(rv, tv)
            print(f"{k:<6} {str(rv.shape):<24} {str(tv.shape):<24} {ma:>10.4g} {me:>10.4g}  {note}")

        print("\n=== Stage E: Decoded proposals per level (pre-NMS, all anchors) ===")
        print(f"{'level':<6} {'rust shape':<24} {'tv shape':<24} {'max_abs':>10} {'mean_abs':>10}  {'note'}")
        for i, k in enumerate(LEVEL_KEYS):
            rk = f"decoded_{k}"
            if rk not in rd:
                print(f"{k:<6} MISSING in rust dump")
                continue
            rv = rd[rk]
            # Reconstruct tv decoded boxes:
            # tv_bbox is [1, A*4, H, W]. We need to flatten the same way:
            # for fh, fw, ai → (ai*4 + d). Use a Python equivalent of the
            # rust loop in the probe binary.
            bbox_tv = tv_bbox[i][0]  # [A*4, H, W]
            cls_shape = tv_cls[i].shape
            a = cls_shape[1]
            h = cls_shape[2]
            w = cls_shape[3]
            # Rearrange [A*4, H, W] → [H, W, A, 4] then reshape [H*W*A, 4]
            bbox_perm = bbox_tv.reshape(a, 4, h, w).transpose(2, 3, 0, 1).reshape(-1, 4)
            anc_tv = tv_anchors_per_level[k]
            decoded_tv = decode_boxes_np(anc_tv, bbox_perm)
            ma, me, note = stat(rv, decoded_tv.astype(np.float32))
            print(f"{k:<6} {str(rv.shape):<24} {str(decoded_tv.shape):<24} {ma:>10.4g} {me:>10.4g}  {note}")

    # Post-NMS proposal count comparison (best-effort, use post-NMS proposals
    # from rust dump and compare to torchvision proposals).
    if "proposals_post_nms" in rd:
        rust_props = rd["proposals_post_nms"]
        print(f"\n=== Stage F: rust post-NMS proposals = {rust_props.shape[0]} ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
