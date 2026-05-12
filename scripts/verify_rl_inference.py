#!/usr/bin/env python3
"""Verify ferrotorch pretrained PPO MlpPolicy inference against the
`stable-baselines3` reference (#1158).

For each pinned RL policy in the `ferrotorch/*` HF org this script:

  1. Pulls the upstream sb3 zip via `huggingface_sb3.load_from_hub`
     and loads it with `stable_baselines3.PPO.load(...)`.
  2. Runs one eval-mode forward over a frozen observation (the same
     `_value_parity_obs.bin` the pin script froze on the mirror) and
     captures the reference `action_logits` `[B, n_actions]` and
     `value` `[B, 1]`.
  3. Invokes the Rust binary
     (`cargo run -p ferrotorch-rl --release --example ppo_policy_dump`)
     against the same observation, then reads `<prefix>_action_logits.bin`
     and `<prefix>_value.bin`.
  4. Computes:
       - `cosine_sim` — over a flat (logits ⊕ value) concatenation,
         and separately per head.
       - `max_abs`    — `max(abs(rust - tv))` per head.
     and compares each against the per-model tolerance in `TOL`.
  5. Prints a one-line verdict per model and a JSON report.

Tolerances are intentionally TIGHT: rust loads the same upstream
safetensors byte-for-byte and the MLP is 4×64×64×{2,1}-ish — f32
accumulation noise alone should not exceed `max_abs <= 1e-5`,
`cosine_sim >= 0.99999`.

Usage:
  python3 scripts/verify_rl_inference.py [--models ppo-cartpole-v1,...]
                                         [--quiet]
                                         [--self-test]

The Rust example must be pre-built (this script will also build it on
first invocation):
  cargo build -p ferrotorch-rl --release --example ppo_policy_dump
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("/tmp/ferrotorch_verify_rl")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Frozen parity-probe observation (must match the pin script + the HF
# mirror's `_value_parity_obs.bin`).
PARITY_OBS = np.array([[0.1, -0.05, 0.02, 0.1]], dtype=np.float32)

# Per-model tolerances. Tight on purpose — see module docstring.
TOL: dict[str, dict[str, Any]] = {
    "ppo-cartpole-v1": dict(
        cosine_sim_min=0.99999,
        max_abs=1e-5,
        obs_dim=4,
        hidden=64,
        n_actions=2,
    ),
}

# Upstream sb3 source per ferrotorch mirror.
UPSTREAM: dict[str, dict[str, str]] = {
    "ppo-cartpole-v1": dict(
        repo="sb3/ppo-CartPole-v1",
        zip_file="ppo-CartPole-v1.zip",
    ),
}


# ---------------------------------------------------------------------------
# Binary dump helpers.
# ---------------------------------------------------------------------------
def read_dump_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"dump {path} truncated (< 4 bytes)")
    (ndim,) = struct.unpack_from("<I", raw, 0)
    off = 4
    if len(raw) < off + 4 * ndim:
        raise ValueError(
            f"dump {path}: header claims ndim={ndim} but only {len(raw)} bytes total"
        )
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = 1
    for s in shape:
        n *= int(s)
    expect = off + 4 * n
    if len(raw) != expect:
        raise ValueError(
            f"dump {path}: header claims shape={shape} (expects {expect} bytes) "
            f"but file is {len(raw)} bytes"
        )
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Rust binary invocation.
# ---------------------------------------------------------------------------
def run_rust_dump(
    model_name: str,
    output_prefix: Path,
    obs_dim: int,
    hidden: int,
    n_actions: int,
    obs_bin: Path | None = None,
) -> dict[str, Any]:
    cmd = [
        "cargo", "run", "-p", "ferrotorch-rl", "--release",
        "--example", "ppo_policy_dump", "--",
        "--model", model_name,
        "--obs-dim", str(obs_dim),
        "--hidden", str(hidden),
        "--n-actions", str(n_actions),
        "--output-prefix", str(output_prefix),
    ]
    if obs_bin is not None:
        cmd.extend(["--obs-bin", str(obs_bin)])
    print(f"  running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"rust dump failed ({proc.returncode}); stderr above")
    json_line: str | None = None
    for line in proc.stdout.splitlines():
        t = line.strip()
        if t.startswith("{") and t.endswith("}"):
            json_line = t
    if json_line is None:
        sys.stderr.write(proc.stdout)
        raise RuntimeError("rust dump did not print a JSON verdict line")
    return json.loads(json_line)


# ---------------------------------------------------------------------------
# Per-model evaluation.
# ---------------------------------------------------------------------------
@dataclass
class ModelVerdict:
    name: str
    passed: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


def reference_forward(upstream_cfg: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Run the upstream sb3 PPO policy on `PARITY_OBS` and return
    `(action_logits, value)` as float32 numpy arrays.
    """
    zip_path = load_from_hub(repo_id=upstream_cfg["repo"], filename=upstream_cfg["zip_file"])
    model = PPO.load(zip_path, device="cpu")
    policy = model.policy
    obs = torch.tensor(PARITY_OBS, dtype=torch.float32)
    policy.set_training_mode(False)
    with torch.no_grad():
        features = policy.extract_features(obs)
        if isinstance(features, tuple):
            pi_features, vf_features = features
        else:
            pi_features = features
            vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        action_logits = policy.action_net(latent_pi)
        value = policy.value_net(latent_vf)
    return action_logits.cpu().numpy().astype(np.float32), value.cpu().numpy().astype(np.float32)


def verify_one(name: str, quiet: bool) -> ModelVerdict:
    print(f"\n=== {name} ===", flush=True)
    tol = TOL[name]
    upstream = UPSTREAM[name]

    # -- 1. Reference forward (sb3). ----------------------------------------
    print(f"  loading upstream sb3 model {upstream['repo']!r}…", flush=True)
    ref_logits, ref_value = reference_forward(upstream)
    print(
        f"  ref action_logits: shape={list(ref_logits.shape)} data={ref_logits.tolist()}",
        flush=True,
    )
    print(
        f"  ref value:         shape={list(ref_value.shape)} data={ref_value.tolist()}",
        flush=True,
    )
    if ref_logits.shape != (1, tol["n_actions"]):
        return ModelVerdict(
            name=name, passed=False,
            summary=f"ref action_logits shape {list(ref_logits.shape)} "
                    f"!= [1, n_actions={tol['n_actions']}]",
        )
    if ref_value.shape != (1, 1):
        return ModelVerdict(
            name=name, passed=False,
            summary=f"ref value shape {list(ref_value.shape)} != [1, 1]",
        )

    # -- 2. Fetch the mirror's `_value_parity_obs.bin` and pass it
    #       through to the Rust example. The ferrotorch hub downloader
    #       only fetches `config.json` + `model.safetensors` (+ optional
    #       tokenizer files); the parity probe is a custom file, so we
    #       resolve it here via `huggingface_hub.hf_hub_download`. This
    #       also doubles as a sanity check that the mirror has the
    #       parity probe at the expected path.
    from huggingface_hub import hf_hub_download
    repo_id = f"ferrotorch/{name}"
    obs_bin = Path(
        hf_hub_download(repo_id, "_value_parity_obs.bin")
    )

    # Cross-check the mirror's obs matches the value the harness froze
    # in `PARITY_OBS` — drifting these out of sync would silently
    # invalidate the test.
    mirror_obs = read_dump_f32(obs_bin)
    if not np.allclose(mirror_obs, PARITY_OBS, atol=0.0, rtol=0.0):
        return ModelVerdict(
            name=name, passed=False,
            summary=f"mirror _value_parity_obs.bin = {mirror_obs.tolist()} "
                    f"!= harness PARITY_OBS = {PARITY_OBS.tolist()}",
        )

    prefix = CACHE_DIR / f"{name}_rust"
    verdict = run_rust_dump(
        name, prefix, tol["obs_dim"], tol["hidden"], tol["n_actions"],
        obs_bin=obs_bin,
    )
    rust_logits = read_dump_f32(prefix.parent / f"{prefix.name}_action_logits.bin")
    rust_value = read_dump_f32(prefix.parent / f"{prefix.name}_value.bin")
    print(
        f"  rust action_logits: shape={list(rust_logits.shape)} data={rust_logits.tolist()}",
        flush=True,
    )
    print(
        f"  rust value:         shape={list(rust_value.shape)} data={rust_value.tolist()}",
        flush=True,
    )
    if rust_logits.shape != ref_logits.shape:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"action_logits shape mismatch: rust={list(rust_logits.shape)} "
                    f"vs ref={list(ref_logits.shape)}",
        )
    if rust_value.shape != ref_value.shape:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"value shape mismatch: rust={list(rust_value.shape)} "
                    f"vs ref={list(ref_value.shape)}",
        )

    # -- 3. Metrics. ---------------------------------------------------------
    logits_diff = rust_logits - ref_logits
    value_diff = rust_value - ref_value
    logits_max_abs = float(np.abs(logits_diff).max())
    value_max_abs = float(np.abs(value_diff).max())
    logits_cos = cosine_similarity(rust_logits, ref_logits)
    value_cos = cosine_similarity(rust_value, ref_value)
    # Combined: concat heads (so the report has a single "overall" number).
    combined_rust = np.concatenate([rust_logits.reshape(-1), rust_value.reshape(-1)])
    combined_ref = np.concatenate([ref_logits.reshape(-1), ref_value.reshape(-1)])
    combined_cos = cosine_similarity(combined_rust, combined_ref)
    combined_max_abs = float(np.abs(combined_rust - combined_ref).max())

    failures: list[str] = []
    if logits_cos < tol["cosine_sim_min"]:
        failures.append(
            f"action_logits cosine_sim={logits_cos:.7f} < {tol['cosine_sim_min']}"
        )
    if logits_max_abs > tol["max_abs"]:
        failures.append(
            f"action_logits max_abs={logits_max_abs:.7e} > {tol['max_abs']:.0e}"
        )
    if value_cos < tol["cosine_sim_min"]:
        failures.append(f"value cosine_sim={value_cos:.7f} < {tol['cosine_sim_min']}")
    if value_max_abs > tol["max_abs"]:
        failures.append(
            f"value max_abs={value_max_abs:.7e} > {tol['max_abs']:.0e}"
        )

    passed = not failures
    summary = (
        f"action_logits max_abs={logits_max_abs:.3e} cos={logits_cos:.7f}, "
        f"value max_abs={value_max_abs:.3e} cos={value_cos:.7f}, "
        f"combined cos={combined_cos:.7f}"
    )
    if failures:
        summary += " — FAIL: " + "; ".join(failures)
    if not quiet:
        print(f"  metrics: {summary}", flush=True)

    return ModelVerdict(
        name=name, passed=passed, summary=summary,
        detail=dict(
            action_logits_shape=list(rust_logits.shape),
            value_shape=list(rust_value.shape),
            action_logits_max_abs=logits_max_abs,
            action_logits_cosine_sim=logits_cos,
            value_max_abs=value_max_abs,
            value_cosine_sim=value_cos,
            combined_max_abs=combined_max_abs,
            combined_cosine_sim=combined_cos,
            rust_verdict=verdict,
            failures=failures,
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", default=",".join(TOL.keys()),
        help="Comma-separated subset of model names to verify.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final per-model verdict line.")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in TOL:
            print(f"unknown model {m!r}. Known: {list(TOL)}", file=sys.stderr)
            return 2

    verdicts: list[ModelVerdict] = []
    for m in models:
        try:
            v = verify_one(m, quiet=args.quiet)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            v = ModelVerdict(
                name=m, passed=False, summary=f"exception: {e!r}",
                detail={"exception": repr(e)},
            )
        verdicts.append(v)

    print("\n=== VERDICTS ===")
    any_fail = False
    for v in verdicts:
        tag = "PASS" if v.passed else "FAIL"
        if not v.passed:
            any_fail = True
        print(f"{v.name}: {tag} — {v.summary}")

    report = {
        v.name: {"passed": v.passed, "summary": v.summary, "detail": v.detail}
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_rl_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")
    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "self.bin"
        shape = (1, 4)
        data = np.arange(4, dtype="<f4").reshape(shape)
        with path.open("wb") as f:
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(data.tobytes(order="C"))
        got = read_dump_f32(path)
        assert got.shape == shape, (got.shape, shape)
        assert np.allclose(got, data), (got, data)
    a = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9
    assert abs(cosine_similarity(a, -a) + 1.0) < 1e-9
    print("self-test: all assertions passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
