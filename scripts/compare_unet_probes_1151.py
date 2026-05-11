#!/usr/bin/env python3
"""Compare per-stage UNet probes (#1151).

Reads the two dump directories produced by
`scripts/probe_unet_stages_1151.py` (TV / diffusers reference) and
`ferrotorch-diffusion/examples/unet_probe_dump.rs` (rust ferrotorch),
and prints a sorted-by-stage table:

  stage  shape  max_abs  mean_abs  cosine  ||rust||/||tv||

The FIRST stage where `max_abs` jumps above ~1e-3 is the bug site.
"""
from __future__ import annotations
import argparse
import struct
from pathlib import Path

import numpy as np


def read_dump_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (ndim,) = struct.unpack_from("<I", raw, 0)
    off = 4
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = int(np.prod(shape))
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.reshape(-1), b.reshape(-1)
    na, nb = float(np.linalg.norm(af)), float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv", type=Path, default=Path("/tmp/ferrotorch_probe_1151_tv"))
    ap.add_argument("--rust", type=Path, default=Path("/tmp/ferrotorch_probe_1151_rust"))
    args = ap.parse_args()

    names = sorted({p.name for p in args.tv.glob("*.bin")} & {p.name for p in args.rust.glob("*.bin")})
    print(f"{'stage':<28} {'shape':<22} {'max_abs':>10} {'mean_abs':>10} {'cosine':>9} {'||r||/||t||':>11}")
    print("-" * 95)
    for name in names:
        t = read_dump_f32(args.tv / name)
        r = read_dump_f32(args.rust / name)
        if t.shape != r.shape:
            print(f"{name:<28}  SHAPE MISMATCH tv={t.shape} rust={r.shape}")
            continue
        diff = np.abs(t - r)
        nt, nr = float(np.linalg.norm(t)), float(np.linalg.norm(r))
        ratio = nr / nt if nt > 0 else float("nan")
        print(
            f"{name:<28} {str(tuple(t.shape)):<22} "
            f"{diff.max():>10.4f} {diff.mean():>10.4f} "
            f"{cos(t, r):>9.4f} {ratio:>11.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
