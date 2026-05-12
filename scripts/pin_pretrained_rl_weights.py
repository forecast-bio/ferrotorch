#!/usr/bin/env python3
"""Pin a PPO-CartPole-v1 reference checkpoint to `ferrotorch/ppo-cartpole-v1` (#1158).

Pipeline:

  1. Pull the canonical `sb3/ppo-CartPole-v1` checkpoint (`.zip`)
     from the HuggingFace Hub via `huggingface_sb3.load_from_hub`.
  2. Load it with `stable_baselines3.PPO.load(...)`. The policy is a
     standard `ActorCriticPolicy` (a.k.a. `MlpPolicy`):
       * `FlattenExtractor` (identity on 1-D obs)
       * `MlpExtractor` with two separate Tanh-MLP trunks (policy and
         value), each `Linear(4 → 64) → Tanh → Linear(64 → 64) → Tanh`.
       * `action_net = Linear(64 → 2)` for Categorical action logits.
       * `value_net = Linear(64 → 1)` for the scalar state value.
  3. Save `policy.state_dict()` verbatim as `model.safetensors` —
     ferrotorch-rl's `MlpPolicy::named_parameters` returns exactly the
     sb3 key layout so the loader needs no key remap.
  4. Run one forward over a frozen observation
     `[[0.1, -0.05, 0.02, 0.1]]` (a representative CartPole state —
     non-trivial values in every dimension so the harness catches any
     transposed-weight or wrong-trunk bug) and dump:
       * `_value_parity_obs.bin`            — [1, 4] f32 (input)
       * `_value_parity_action_logits.bin`  — [1, 2] f32 (ref output)
       * `_value_parity_value.bin`          — [1, 1] f32 (ref output)
     in the standard `[u32 ndim][u32 × ndim shape][f32 data]`
     little-endian format the other ferrotorch dumps use.
  5. Upload all four artifacts plus a `config.json` describing the
     architecture to `huggingface.co/ferrotorch/ppo-cartpole-v1`.
  6. Print the SHA-256 of `model.safetensors` so the caller can update
     `ferrotorch-hub/src/registry.rs`.

Run via:
  python3 scripts/pin_pretrained_rl_weights.py

Skip the HF upload with `FERROTORCH_PIN_UPLOAD=0`.

License: `sb3/ppo-CartPole-v1` is Apache 2.0 (carried over from
stable-baselines3); the upstream README states the policy was trained
with the default `MlpPolicy` config and no novel data, so a
byte-for-byte mirror to the `ferrotorch` org keeps the same license.
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

WORK_DIR = Path("/tmp/ferrotorch_pin_ppo_cartpole_v1")
WORK_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "ferrotorch/ppo-cartpole-v1"
UPSTREAM_REPO = "sb3/ppo-CartPole-v1"
UPSTREAM_FILE = "ppo-CartPole-v1.zip"

# Frozen parity-probe observation. Non-trivial in every dim so a swapped
# trunk (policy/value) or transposed weight will fail loudly. Matches
# the values consumed by `scripts/verify_rl_inference.py` and by
# `ferrotorch-rl/examples/ppo_policy_dump.rs`'s default obs path.
PARITY_OBS = np.array([[0.1, -0.05, 0.02, 0.1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Binary dump helpers — same `[u32 ndim][u32 × ndim shape][f32 data]`
# little-endian format the rest of the project uses.
# ---------------------------------------------------------------------------
def dump_f32(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def main() -> int:
    # -- 1. Fetch upstream sb3 zip. -----------------------------------------
    print(f"[pin] fetching {UPSTREAM_REPO}/{UPSTREAM_FILE}…", flush=True)
    zip_path = load_from_hub(repo_id=UPSTREAM_REPO, filename=UPSTREAM_FILE)
    print(f"[pin] cached at {zip_path}", flush=True)

    # -- 2. Load policy. -----------------------------------------------------
    # PPO.load emits a `UserWarning: Could not deserialize object learning_rate`
    # because sb3 zips a pickled lr_schedule object that depends on the
    # exact Python version. We only need the trained weights, so the
    # warning is harmless for parity work.
    print("[pin] loading PPO model (cpu, weights only)…", flush=True)
    model = PPO.load(zip_path, device="cpu")
    policy = model.policy

    print("[pin] policy class:", type(policy).__name__, flush=True)
    print("[pin] features_extractor:", type(policy.features_extractor).__name__, flush=True)
    print("[pin] mlp_extractor:", policy.mlp_extractor, flush=True)
    print("[pin] action_net:", policy.action_net, flush=True)
    print("[pin] value_net:", policy.value_net, flush=True)

    # Cross-check architecture so the pin refuses to upload an
    # off-distribution checkpoint.
    state = policy.state_dict()
    expected_keys = {
        "mlp_extractor.policy_net.0.weight": (64, 4),
        "mlp_extractor.policy_net.0.bias": (64,),
        "mlp_extractor.policy_net.2.weight": (64, 64),
        "mlp_extractor.policy_net.2.bias": (64,),
        "mlp_extractor.value_net.0.weight": (64, 4),
        "mlp_extractor.value_net.0.bias": (64,),
        "mlp_extractor.value_net.2.weight": (64, 64),
        "mlp_extractor.value_net.2.bias": (64,),
        "action_net.weight": (2, 64),
        "action_net.bias": (2,),
        "value_net.weight": (1, 64),
        "value_net.bias": (1,),
    }
    print("[pin] state_dict keys (sorted):", flush=True)
    for k in sorted(state.keys()):
        print(f"[pin]   {k}: shape={tuple(state[k].shape)} dtype={state[k].dtype}", flush=True)

    missing = sorted(set(expected_keys) - set(state))
    extra = sorted(set(state) - set(expected_keys))
    if missing:
        print(f"[pin] FATAL: missing keys: {missing}", file=sys.stderr, flush=True)
        return 2
    if extra:
        # log_std would imply continuous actions; we refuse to pin
        # anything but the discrete CartPole-v1 layout.
        print(f"[pin] FATAL: unexpected upstream keys: {extra}", file=sys.stderr, flush=True)
        return 2
    for k, exp_shape in expected_keys.items():
        got = tuple(state[k].shape)
        if got != exp_shape:
            print(
                f"[pin] FATAL: shape mismatch for {k}: got {got} expected {exp_shape}",
                file=sys.stderr,
                flush=True,
            )
            return 2

    # All keys present, dtypes float32. Verify dtype.
    for k, v in state.items():
        if v.dtype != torch.float32:
            print(
                f"[pin] FATAL: dtype mismatch for {k}: got {v.dtype} expected torch.float32",
                file=sys.stderr,
                flush=True,
            )
            return 2

    # -- 3. Save state_dict to safetensors (verbatim, no key remap). --------
    from safetensors.torch import save_file

    weights_path = WORK_DIR / "model.safetensors"
    save_file(
        {k: v.detach().contiguous().cpu() for k, v in state.items()},
        str(weights_path),
    )
    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    print(f"[pin] wrote {weights_path} ({weights_path.stat().st_size} bytes)", flush=True)
    print(f"[pin] SHA-256: {sha}", flush=True)

    # -- 4. Eval-mode parity forward. ---------------------------------------
    obs = torch.tensor(PARITY_OBS, dtype=torch.float32)
    policy.set_training_mode(False)
    with torch.no_grad():
        features = policy.extract_features(obs)
        # sb3's FlattenExtractor on 1-D obs returns a single tensor.
        # On some sb3 versions a (pi_features, vf_features) tuple is
        # returned when `share_features_extractor=False`; defend
        # against both.
        if isinstance(features, tuple):
            pi_features, vf_features = features
        else:
            pi_features = features
            vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        action_logits = policy.action_net(latent_pi)  # [1, 2]
        value = policy.value_net(latent_vf)            # [1, 1]
    print(
        f"[pin] parity forward: obs={PARITY_OBS.tolist()} "
        f"action_logits={action_logits.numpy().tolist()} "
        f"value={value.numpy().tolist()}",
        flush=True,
    )

    obs_path = WORK_DIR / "_value_parity_obs.bin"
    logits_path = WORK_DIR / "_value_parity_action_logits.bin"
    value_path = WORK_DIR / "_value_parity_value.bin"
    dump_f32(obs_path, PARITY_OBS)
    dump_f32(logits_path, action_logits.cpu().numpy())
    dump_f32(value_path, value.cpu().numpy())
    print(
        f"[pin] dumped parity fixtures: obs={obs_path.stat().st_size}B, "
        f"action_logits={logits_path.stat().st_size}B, "
        f"value={value_path.stat().st_size}B",
        flush=True,
    )

    # -- 5. Upload to HF (best-effort; tolerate offline). -------------------
    upload = os.environ.get("FERROTORCH_PIN_UPLOAD", "1") != "0"
    if upload:
        try:
            from huggingface_hub import HfApi, create_repo, upload_file
        except ImportError:
            print(
                "[pin] huggingface_hub not installed — skipping upload "
                "(set FERROTORCH_PIN_UPLOAD=0 to silence)",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                create_repo(REPO_ID, repo_type="model", exist_ok=True)
                for relative, local in [
                    ("model.safetensors", weights_path),
                    ("_value_parity_obs.bin", obs_path),
                    ("_value_parity_action_logits.bin", logits_path),
                    ("_value_parity_value.bin", value_path),
                ]:
                    upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=relative,
                        repo_id=REPO_ID,
                        repo_type="model",
                    )
                # Minimal config.json so the hub download is happy.
                import json
                cfg = {
                    "architecture": "ActorCriticPolicy (sb3 MlpPolicy)",
                    "upstream_repo": UPSTREAM_REPO,
                    "upstream_file": UPSTREAM_FILE,
                    "env_id": "CartPole-v1",
                    "obs_dim": 4,
                    "hidden": 64,
                    "n_actions": 2,
                    "activation": "Tanh",
                    "discrete_action_distribution": "Categorical",
                    "log_std_present": False,
                    "license": "Apache-2.0",
                    "sb3_version": "2.8.0",
                    "parity_obs": PARITY_OBS.tolist(),
                }
                cfg_path = WORK_DIR / "config.json"
                cfg_path.write_text(json.dumps(cfg, indent=2))
                upload_file(
                    path_or_fileobj=str(cfg_path),
                    path_in_repo="config.json",
                    repo_id=REPO_ID,
                    repo_type="model",
                )
                api = HfApi()
                files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
                print(f"[pin] uploaded to {REPO_ID}. Repo files:", flush=True)
                for fname in files:
                    print(f"[pin]   - {fname}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[pin] HF upload failed: {exc!r}", file=sys.stderr, flush=True)
                return 2
    else:
        print("[pin] FERROTORCH_PIN_UPLOAD=0, skipping upload", flush=True)

    print(f"\n[pin] DONE. SHA-256 for registry.rs pin: {sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
