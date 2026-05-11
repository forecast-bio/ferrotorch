#!/usr/bin/env python3
"""Per-stage probe for the SD-1.5 UNet — diffusers reference side (#1151).

Loads the pinned `ferrotorch/sd-v1-5-unet` mirror, runs the diffusers
`UNet2DConditionModel.forward` on the frozen parity probe, and dumps
each interesting intermediate tensor to disk in the same f32 dump
format (`[u32 ndim][u32 × ndim shape][f32 data]`) used everywhere else
in the harness.

The Rust-side counterpart (`unet_probe_dump.rs`) writes the same set
of names into a parallel directory; the compare step is a separate
small script.

Stages dumped:
    01_time_emb                     [1, 1280]
    02_conv_in                      [1, 320, 64, 64]
    03_down_block_0_out             [1, 320, 32, 32]
    04_down_block_1_out             [1, 640, 16, 16]
    05_down_block_2_out             [1, 1280,  8,  8]
    06_down_block_3_out             [1, 1280,  8,  8]
    07_mid_out                      [1, 1280,  8,  8]
    08_up_block_0_out               [1, 1280, 16, 16]
    09_up_block_1_out               [1, 1280, 32, 32]
    10_up_block_2_out               [1,  640, 64, 64]
    11_up_block_3_out               [1,  320, 64, 64]
    12_conv_norm_out                [1,  320, 64, 64]
    13_predicted_noise              [1,    4, 64, 64]
"""
from __future__ import annotations
import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from huggingface_hub import hf_hub_download

REPO_ID = "ferrotorch/sd-v1-5-unet"
PROBE_DIR_DEFAULT = Path("/tmp/ferrotorch_probe_1151_tv")


def write_dump_f32(path: Path, x: torch.Tensor) -> None:
    arr = x.detach().to(torch.float32).cpu().contiguous().numpy()
    shape = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", d))
        f.write(arr.tobytes(order="C"))


def read_dump_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (ndim,) = struct.unpack_from("<I", raw, 0)
    off = 4
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = int(np.prod(shape))
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROBE_DIR_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # --- 1. Fetch mirror artifacts (config + weights + parity probe). ---
    files: dict[str, Path] = {}
    for fn in (
        "config.json",
        "model.safetensors",
        "_value_parity_noisy_latent.bin",
        "_value_parity_timestep.bin",
        "_value_parity_text_embedding.bin",
        "_value_parity_predicted_noise.bin",
    ):
        files[fn] = Path(hf_hub_download(repo_id=REPO_ID, filename=fn))

    repo_dir = files["config.json"].parent

    # --- 2. Load UNet via diffusers (fp32 on CPU for parity). ----------
    # Mirror stores weights as `model.safetensors`; diffusers expects
    # `diffusion_pytorch_model.safetensors`. Build the unet from the
    # config and `load_state_dict` from our safetensors directly.
    import json
    from safetensors.torch import load_file
    cfg = json.loads(files["config.json"].read_text())
    # Strip the leading `_class_name` / `_diffusers_version` private keys
    # so `from_config` accepts the rest.
    public_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    unet: UNet2DConditionModel = UNet2DConditionModel.from_config(public_cfg)
    sd = load_file(str(files["model.safetensors"]))
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    missing[:5]={missing[:5]}")
    if unexpected:
        print(f"    unexpected[:5]={unexpected[:5]}")
    unet = unet.to(torch.float32)
    unet.eval()
    torch.set_grad_enabled(False)

    # --- 3. Load fixed inputs. -----------------------------------------
    latent = torch.from_numpy(read_dump_f32(files["_value_parity_noisy_latent.bin"]))
    ts_arr = read_dump_f32(files["_value_parity_timestep.bin"]).reshape(-1)
    timestep = torch.from_numpy(ts_arr)
    text = torch.from_numpy(read_dump_f32(files["_value_parity_text_embedding.bin"]))
    print(f"  latent: {tuple(latent.shape)}  ||={latent.norm():.4f}")
    print(f"  timestep: {timestep.tolist()}")
    print(f"  text: {tuple(text.shape)}  ||={text.norm():.4f}")

    # --- 4. Register hooks at the canonical per-stage boundaries. ------
    bag: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def _h(_mod, _in, out):
            # CrossAttnDownBlock returns (hidden_states, res_samples).
            # Transformer2DModel returns a Transformer2DModelOutput with a
            # `.sample` attribute (a dataclass — not a tuple).
            if isinstance(out, tuple):
                bag[name] = out[0].detach().clone()
            elif hasattr(out, "sample"):
                bag[name] = out.sample.detach().clone()
            else:
                bag[name] = out.detach().clone()
        return _h

    # Stage 1: time_emb is computed inline in forward, not as a module.
    # We'll recompute it ourselves to match diffusers exactly.

    # Stage 2: conv_in output
    h_conv_in = unet.conv_in.register_forward_hook(hook("02_conv_in"))

    # Stages 3-6: down blocks
    db_hooks = []
    for i, blk in enumerate(unet.down_blocks):
        db_hooks.append(blk.register_forward_hook(hook(f"0{3+i}_down_block_{i}_out")))

    # Fine-grained hooks inside down_blocks[0] (the first divergent block).
    db0 = unet.down_blocks[0]
    fine_hooks = []
    for j, (r, a) in enumerate(zip(db0.resnets, db0.attentions)):
        fine_hooks.append(r.register_forward_hook(hook(f"03a_db0_resnet{j}_out")))
        fine_hooks.append(a.register_forward_hook(hook(f"03a_db0_attn{j}_out")))
    if getattr(db0, "downsamplers", None) is not None:
        for d in db0.downsamplers:
            fine_hooks.append(d.register_forward_hook(hook("03a_db0_downsample_out")))

    # Deep-dive on the first Transformer2DModel inside db0.attentions[0]:
    # capture sub-step outputs to localize the bug.
    t2d0 = db0.attentions[0]
    deep_hooks = []
    deep_hooks.append(t2d0.norm.register_forward_hook(hook("03b_t2d_norm_out")))
    deep_hooks.append(t2d0.proj_in.register_forward_hook(hook("03b_t2d_proj_in_out")))
    deep_hooks.append(t2d0.proj_out.register_forward_hook(hook("03b_t2d_proj_out")))
    blk0 = t2d0.transformer_blocks[0]
    deep_hooks.append(blk0.norm1.register_forward_hook(hook("03b_t2d_b0_norm1_out")))
    deep_hooks.append(blk0.attn1.register_forward_hook(hook("03b_t2d_b0_attn1_out")))
    # Hook inside attn1 for q/k/v projections.
    deep_hooks.append(blk0.attn1.to_q.register_forward_hook(hook("03c_attn1_q_proj")))
    deep_hooks.append(blk0.attn1.to_k.register_forward_hook(hook("03c_attn1_k_proj")))
    deep_hooks.append(blk0.attn1.to_v.register_forward_hook(hook("03c_attn1_v_proj")))
    # to_out is a Sequential; its [0] is the Linear we want.
    deep_hooks.append(blk0.attn1.to_out[0].register_forward_hook(hook("03c_attn1_to_out")))
    # Capture the merged-heads tensor (the input to to_out[0]) via a pre-hook.
    def pre_hook(name: str):
        def _h(_mod, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            bag[name] = x.detach().clone()
        return _h
    deep_hooks.append(blk0.attn1.to_out[0].register_forward_pre_hook(pre_hook("03c_attn1_merged")))
    deep_hooks.append(blk0.norm2.register_forward_hook(hook("03b_t2d_b0_norm2_out")))
    deep_hooks.append(blk0.attn2.register_forward_hook(hook("03b_t2d_b0_attn2_out")))
    deep_hooks.append(blk0.norm3.register_forward_hook(hook("03b_t2d_b0_norm3_out")))
    deep_hooks.append(blk0.ff.register_forward_hook(hook("03b_t2d_b0_ff_out")))

    # Stage 7: mid block
    h_mid = unet.mid_block.register_forward_hook(hook("07_mid_out"))

    # Stages 8-11: up blocks
    ub_hooks = []
    for i, blk in enumerate(unet.up_blocks):
        ub_hooks.append(blk.register_forward_hook(hook(f"0{8+i}_up_block_{i}_out" if 8+i < 10 else f"{8+i}_up_block_{i}_out")))

    # Stage 12: conv_norm_out
    h_cno = unet.conv_norm_out.register_forward_hook(hook("12_conv_norm_out"))

    # --- 5. Run forward. -----------------------------------------------
    # Compute time_emb ourselves (mirrors diffusers internal recipe).
    t_proj = unet.time_proj(timestep)              # [1, 320]
    t_emb = unet.time_embedding(t_proj.to(torch.float32))  # [1, 1280]
    bag["01_time_emb"] = t_emb.clone()

    out = unet(latent, timestep, encoder_hidden_states=text).sample
    bag["13_predicted_noise"] = out.detach().clone()

    h_conv_in.remove()
    for hh in db_hooks: hh.remove()
    h_mid.remove()
    for hh in ub_hooks: hh.remove()
    h_cno.remove()

    # --- 6. Save the per-stage tensors. --------------------------------
    print("\n  --- per-stage tv reference ---")
    for name in sorted(bag.keys()):
        t = bag[name]
        write_dump_f32(args.out / f"{name}.bin", t)
        print(f"  {name}: shape={tuple(t.shape)}  norm={t.norm().item():.4f}  max_abs={t.abs().max().item():.4f}")

    # --- 7. Quick sanity: compare predicted noise to the pinned ref. ---
    pinned_ref = read_dump_f32(files["_value_parity_predicted_noise.bin"])
    our_pred = bag["13_predicted_noise"].cpu().numpy()
    delta_max = float(np.abs(our_pred - pinned_ref).max())
    print(f"\n  predicted_noise vs pinned _value_parity_predicted_noise.bin: max_abs={delta_max:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
