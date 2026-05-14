#!/usr/bin/env python3
"""Reference-numerics generator for the ferrotorch SD-1.5 VAE encoder
parity test.

Counterpart of `scripts/verify_sd_pipeline_inference.py` but scoped to
the encoder half of `AutoencoderKL`. Loads
`runwayml/stable-diffusion-v1-5`'s VAE via `diffusers`, encodes a
frozen reference image, and dumps the encoder outputs (mean, clamped
logvar, mode latent) as float32 binary fixtures the ferrotorch parity
test consumes.

Used in two modes:

  * `--pin`: produces the on-disk fixtures the future
    `ferrotorch/sd-v1-5-vae-encoder` HF mirror will ship alongside the
    safetensors. Wires into the existing
    `scripts/pin_pretrained_diffusion_weights.py` flow once the mirror
    is registered (#1150 pattern).

  * `--verify`: re-runs the diffusers reference against an existing
    fixture set and asserts the numerics are stable across `diffusers`
    versions. CI-friendly — fails the diffusers upgrade if the encoder
    drifts.

The reference input is a deterministic striped RGB image in [-1, 1],
matching `striped_image()` in
`ferrotorch-diffusion/tests/conformance_vae_encoder.rs`.

Tolerance baselines (chosen to match the existing SD pipeline
conformance harness):
    cosine_sim >= 0.999, max_abs <= 0.5

Usage:
    python3 scripts/verify_vae_encoder.py --pin
    python3 scripts/verify_vae_encoder.py --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Imports gated behind a helpful error message so a missing dep doesn't
# crash with an opaque ModuleNotFoundError.
try:
    import torch
    from diffusers import AutoencoderKL
except ImportError as e:  # pragma: no cover — operator-side script
    print(
        f"verify_vae_encoder: missing dep ({e}). Install with:\n"
        "    pip install diffusers==0.38.0 torch numpy\n",
        file=sys.stderr,
    )
    raise SystemExit(2) from e

SD15_REPO = "runwayml/stable-diffusion-v1-5"
SD15_VAE_SUBFOLDER = "vae"
TORCH_DTYPE = torch.float32  # f32 throughout for reference exactness
SEED = 42

# Tolerances matching the existing SD pipeline conformance harness.
COS_TOL = 0.999
MAX_ABS_TOL = 0.5


def striped_image(b: int = 1, h: int = 512, w: int = 512) -> torch.Tensor:
    """Build the deterministic [-1, 1] striped RGB input.

    Matches `striped_image()` in
    `ferrotorch-diffusion/tests/conformance_vae_encoder.rs` byte-for-byte
    when both are evaluated in f32. The vertical gradient per channel
    keeps the input out of the constant-zero regime that would silently
    mask bugs in the post-norm layers.
    """
    img = torch.zeros((b, 3, h, w), dtype=TORCH_DTYPE)
    for c in range(3):
        for y in range(h):
            base = (y / h) * 2.0 - 1.0
            img[:, c, y, :] = max(-1.0, min(1.0, base + c * 0.05))
    return img


def load_vae() -> AutoencoderKL:
    """Load the SD-1.5 VAE from HuggingFace (full encoder + decoder)."""
    vae = AutoencoderKL.from_pretrained(
        SD15_REPO,
        subfolder=SD15_VAE_SUBFOLDER,
        torch_dtype=TORCH_DTYPE,
    )
    vae.eval()
    return vae


def encode_reference(vae: AutoencoderKL, img: torch.Tensor) -> dict[str, np.ndarray]:
    """Run the diffusers encoder and return the dump dict.

    All outputs are f32 numpy arrays for round-trip via the
    ferrotorch_serialize fixture format (which our existing pin scripts
    also use for `_value_parity_*.bin`).
    """
    with torch.no_grad():
        out = vae.encode(img)
    latent_dist = out.latent_dist
    mean = latent_dist.mean.detach().cpu().numpy().astype(np.float32)
    # diffusers clamps logvar to [-30, 20] inside DiagonalGaussianDistribution.__init__
    logvar = (
        latent_dist.logvar.detach()
        .clamp(-30.0, 20.0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    # `.mode()` returns the mean — the deterministic latent. This is
    # what the ferrotorch `dist.mode()` API mirrors and what the parity
    # test will compare element-wise.
    mode = mean.copy()
    return {"mean": mean, "logvar_clamped": logvar, "mode": mode}


def write_fixtures(out_dir: Path, dump: dict[str, np.ndarray]) -> None:
    """Write each entry as a raw f32 little-endian .bin file.

    Format matches `pin_pretrained_diffusion_weights.py`'s
    `_value_parity_*.bin` convention so the existing fixture-loader
    helpers in the rust parity tests will work unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in dump.items():
        # Sanity: dtype must be exactly f32 so the rust parity test can
        # `from_le_bytes` 4-byte chunks without ambiguity.
        assert arr.dtype == np.float32, (
            f"verify_vae_encoder: dump '{name}' has dtype {arr.dtype}, expected float32"
        )
        path = out_dir / f"_value_parity_{name}.bin"
        path.write_bytes(arr.tobytes(order="C"))
        print(f"  wrote {path.relative_to(out_dir.parent)}  shape={list(arr.shape)}")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    num = float(np.dot(af, bf))
    den = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if den < 1e-12:
        # Both near-zero — treat as identical.
        return 1.0 if np.allclose(af, bf, atol=1e-6) else 0.0
    return num / den


def verify_against_existing(fixture_dir: Path) -> int:
    """Re-run the encoder and compare against on-disk fixtures."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    vae = load_vae()
    img = striped_image()
    fresh = encode_reference(vae, img)

    failures = 0
    for name, ref in fresh.items():
        path = fixture_dir / f"_value_parity_{name}.bin"
        if not path.is_file():
            print(f"  MISSING: {path}")
            failures += 1
            continue
        on_disk = np.frombuffer(path.read_bytes(), dtype=np.float32).reshape(ref.shape)
        cs = cosine_sim(ref, on_disk)
        ma = float(np.max(np.abs(ref - on_disk)))
        status = "PASS" if (cs >= COS_TOL and ma <= MAX_ABS_TOL) else "FAIL"
        print(f"  {name}: cosine={cs:.6f}  max_abs={ma:.6f}  -> {status}")
        if status == "FAIL":
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pin",
        action="store_true",
        help="Produce fresh reference fixtures (use when registering the HF mirror).",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Re-run the reference and verify on-disk fixtures match.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "ferrotorch-diffusion"
        / "tests"
        / "fixtures"
        / "vae_encoder",
        help="Where to read/write the _value_parity_*.bin fixtures.",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.pin:
        print(
            f"verify_vae_encoder: loading {SD15_REPO}/{SD15_VAE_SUBFOLDER} "
            f"({TORCH_DTYPE}) and pinning fixtures to {args.fixture_dir}"
        )
        vae = load_vae()
        img = striped_image()
        dump = encode_reference(vae, img)
        write_fixtures(args.fixture_dir, dump)
        print("verify_vae_encoder: PIN OK")
        return 0

    if args.verify:
        print(f"verify_vae_encoder: verifying against fixtures in {args.fixture_dir}")
        failures = verify_against_existing(args.fixture_dir)
        if failures:
            print(f"verify_vae_encoder: VERIFY FAIL ({failures} mismatch)")
            return 1
        print("verify_vae_encoder: VERIFY OK")
        return 0

    # Unreachable — argparse `required=True` guarantees one of the modes.
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
