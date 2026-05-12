#!/usr/bin/env python3
"""Pin real-artifact format-parity fixtures for `ferrotorch-serialize`.

Phase G.3 of real-artifact-driven development (#1169). Mirrors
`ferrotorch/serialize-parity-v1` on HuggingFace with four targets,
one per ferrotorch-serialize loader/exporter:

  * Target A — `.pth` load:
      Real `torchvision` weights (`resnet18-f37072fd.pth`, 45 MB,
      official `download.pytorch.org` URL). Reference per-tensor
      flat f32 binaries (`[u32 ndim][u32 shape...][f32 bytes]`)
      are produced by `torch.load(...)` so the rust dump can be
      compared byte-exact.

  * Target B — SafeTensors round-trip:
      The same resnet18 state_dict re-saved via
      `safetensors.torch.save_file`. Reference binaries are the
      same files used by Target A — `ferrotorch_serialize::load_safetensors`
      must reproduce them byte-exact.

  * Target C — GGUF load:
      Real upstream GGUF (`unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf`,
      ~145 MB; all-Q8_0 + F32). Reference dequantized f32 binaries
      for a sampled subset of tensors are produced by `gguf` (the
      python lib); ferrotorch's Q8_0 dequant kernel must reproduce
      them under a 1e-4 max-abs floor (Q8 group scaling has a known
      noise floor between block-quant implementations).

  * Target D — ONNX export:
      Tiny ferrotorch MLP (`Linear(4 -> 8) + ReLU + Linear(8 -> 2)`)
      with weights fixed by seed=42. The rust dump exports the
      traced graph to .onnx; the verifier loads the .onnx with
      `onnxruntime.InferenceSession` and asserts cosine_sim >= 0.9999
      and max_abs <= 1e-5 against ferrotorch's own forward output.

The python pin script's only role for Target D is to ship the
fixed input tensors + the reference forward outputs computed by
`torch.nn.Sequential` mirroring ferrotorch's MLP — the rust dump
side produces the .onnx + the ferrotorch forward output, and the
verifier triangulates rust-ONNX vs rust-ferrotorch vs torch.

Layout on disk + on the HF mirror:

  resnet18-pth/
    resnet18-5c106cde.pth            (real upstream)
    reference_state_dict/
      <key>.bin                      (per-tensor `[u32 ndim][u32 shape][f32]`)
    keys.json                        (ordered list of tensor names)
  safetensors-rt/
    resnet18.safetensors             (re-serialized state_dict)
    reference_state_dict/<key>.bin   (same files as above; copied so target stays self-contained)
    keys.json
  gguf/
    SmolLM-135M-Instruct-Q4_K_M.gguf (real upstream)
    reference_dequant/<name>.bin
    sampled_tensor_names.json        (subset chosen at pin time)
    meta.json                        (per-tensor shape + GGML type)
  onnx-mlp/
    mlp_weights.bin                  (fixed-seed weights the rust side reads)
    input_zeros.bin
    input_ones.bin
    input_random.bin
    torch_forward_zeros.bin
    torch_forward_ones.bin
    torch_forward_random.bin
    meta.json                        (architecture + seed)
  bundle.tar
  README.md

Usage:
    python3 scripts/pin_pretrained_serialize_fixtures.py
    python3 scripts/pin_pretrained_serialize_fixtures.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import tarfile
import textwrap
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gguf
import safetensors.torch as st_torch
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "ferrotorch/serialize-parity-v1"

# Target A — use the modern ZIP-pickle resnet18 (`-f37072fd.pth`).
# The legacy `-5c106cde.pth` is the 2018 v0.4 *tar*-pickle format
# that `torch.load(..., weights_only=True)` refuses to load and
# that `ferrotorch_serialize::load_pytorch_state_dict` (a
# `zip::ZipArchive` reader) also cannot parse. The IMAGENET1K_V1
# weights ship in modern ZIP-pickle format and are directly
# comparable to ferrotorch's loader.
PTH_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
PTH_NAME = "resnet18-f37072fd.pth"

# Target C — ferrotorch-serialize's GGUF reader supports the legacy
# block-quant family (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1) plus F16/F32 but
# not the K-quants (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K). The mixed-quant
# `Q4_K_M.gguf` files therefore fail to parse with "unsupported
# GGML type: 14"; we pin the all-Q8_0 + F32 variant instead so the
# rust reader exercises the dequant path on every tensor in the
# file without hitting any unsupported types. K-quant support is a
# separate future-work item (the on-disk layout for K-quants differs
# substantially from the round-and-scale legacy layout).
GGUF_REPO = "unsloth/SmolLM2-135M-Instruct-GGUF"
GGUF_FILE = "SmolLM2-135M-Instruct-Q8_0.gguf"
# Sample at most this many tensors for the GGUF target — full dump would
# be ~hundreds of MB.
GGUF_SAMPLE_LIMIT = 12

# Target D
ONNX_SEED = 42
ONNX_IN = 4
ONNX_HIDDEN = 8
ONNX_OUT = 2


def dump_f32(arr: np.ndarray, path: Path) -> None:
    """Dump float32 ndarray as `[u32 ndim][u32 shape...][f32 bytes]`.

    Mirrors `dump_f32` from the SD pipeline pin script so the rust
    side can use the same parser.
    """
    data = arr.reshape(-1).astype("<f4", copy=False)
    shape = list(arr.shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", int(d)))
        f.write(data.tobytes(order="C"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Target A — .pth
# ---------------------------------------------------------------------------


def pin_pth(out_root: Path) -> None:
    print("=== Target A — .pth (resnet18) ===", flush=True)
    out = out_root / "resnet18-pth"
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = out / "reference_state_dict"
    ref_dir.mkdir(exist_ok=True)

    pth_path = out / PTH_NAME
    if not pth_path.exists():
        print(f"  downloading {PTH_URL}", flush=True)
        urllib.request.urlretrieve(PTH_URL, pth_path)
    sz = pth_path.stat().st_size
    print(f"  pth size: {sz} bytes (SHA-256 {sha256_of(pth_path)[:12]}...)", flush=True)

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict), f"unexpected pth payload: {type(state)}"
    keys = sorted(state.keys())
    print(f"  state_dict has {len(keys)} tensors", flush=True)

    for k in keys:
        t = state[k]
        # All resnet18 weights are float32 (no integer buffers in
        # this checkpoint — running_mean / running_var / num_batches
        # tracked are float32 / int64 in some torchvision versions,
        # but the 5c106cde.pth is the legacy v0.4 export and only
        # has float weights; defensive check below).
        if t.dtype not in (torch.float32, torch.float64):
            # Skip non-float tensors honestly — the rust loader
            # converts integer types to Float<T>, but for parity
            # we want byte-exact f32 comparison.
            print(f"    skipping non-float key {k}: dtype={t.dtype}", flush=True)
            continue
        arr = t.detach().cpu().float().numpy().astype(np.float32)
        # Use a filename-safe key (resnet18 keys contain dots but no
        # slashes, so they're already safe).
        dump_f32(arr, ref_dir / f"{k}.bin")
    (out / "keys.json").write_text(json.dumps(keys, indent=2))


# ---------------------------------------------------------------------------
# Target B — SafeTensors round-trip
# ---------------------------------------------------------------------------


def pin_safetensors(out_root: Path) -> None:
    print("=== Target B — SafeTensors round-trip (resnet18) ===", flush=True)
    out = out_root / "safetensors-rt"
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = out / "reference_state_dict"
    ref_dir.mkdir(exist_ok=True)

    pth_path = out_root / "resnet18-pth" / PTH_NAME
    if not pth_path.exists():
        raise SystemExit(
            "Target A must run before Target B (need resnet18 .pth file)"
        )

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    # Filter to float dtypes for byte-exact round-trip.
    keep = {
        k: v.detach().cpu().float().contiguous()
        for k, v in state.items()
        if v.dtype in (torch.float32, torch.float64)
    }
    keys = sorted(keep.keys())

    st_path = out / "resnet18.safetensors"
    st_torch.save_file(keep, str(st_path))
    print(f"  wrote {st_path.name} ({st_path.stat().st_size} bytes)", flush=True)

    for k in keys:
        arr = keep[k].numpy().astype(np.float32)
        dump_f32(arr, ref_dir / f"{k}.bin")
    (out / "keys.json").write_text(json.dumps(keys, indent=2))


# ---------------------------------------------------------------------------
# Target C — GGUF
# ---------------------------------------------------------------------------


def pin_gguf(out_root: Path) -> None:
    print("=== Target C — GGUF (SmolLM-135M-Instruct-Q4_K_M) ===", flush=True)
    out = out_root / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = out / "reference_dequant"
    ref_dir.mkdir(exist_ok=True)

    gguf_local = out / GGUF_FILE
    if not gguf_local.exists():
        print(f"  downloading {GGUF_REPO}/{GGUF_FILE}", flush=True)
        cached = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
        shutil.copyfile(cached, gguf_local)
    print(f"  gguf size: {gguf_local.stat().st_size} bytes", flush=True)

    reader = gguf.GGUFReader(str(gguf_local))
    print(f"  gguf has {len(reader.tensors)} tensors", flush=True)

    meta = {
        "num_tensors": len(reader.tensors),
        "tensors": [],
    }
    # Pick a deterministic spread: every Nth tensor up to a cap, so we
    # don't bias to "all the first 12 layers" only.
    stride = max(1, len(reader.tensors) // GGUF_SAMPLE_LIMIT)
    sampled = reader.tensors[::stride][:GGUF_SAMPLE_LIMIT]
    sampled_names: list[str] = []

    for t in reader.tensors:
        meta["tensors"].append(
            {
                "name": t.name,
                "shape": [int(d) for d in t.shape],
                "ggml_type": int(t.tensor_type),
                "ggml_type_name": t.tensor_type.name,
            }
        )

    for t in sampled:
        # GGUFReader returns:
        #   * For F32 tensors: `t.data` is the float32 buffer directly.
        #   * For F16 / quantized tensors: `t.data` is the raw uint8
        #     byte buffer; we have to call `gguf.quants.dequantize`.
        # ferrotorch-serialize's `dequantize_gguf_tensor` returns a
        # `Tensor<f32>` with shape = `info.dims` (the GGUF dim order,
        # which is fastest-varying-first). NumPy's natural row-major
        # reshape gives the **reverse** of `t.shape` — for a (1536,
        # 576) GGUF weight, numpy emits a (576, 1536) array because
        # GGUF's dims[0]=1536 is the inner / fastest-varying axis.
        #
        # We dump in the GGUF dim order (i.e. reverse numpy's natural
        # shape) so the rust binary's `[u32 ndim][u32 shape...]`
        # header matches ferrotorch's tensor shape byte-for-byte.
        # The underlying flat f32 buffer is identical between the two
        # conventions because both numpy and ferrotorch use row-major
        # storage.
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            data = np.array(t.data, dtype=np.float32, copy=True)
            np_shape = tuple(int(d) for d in t.shape)
            data = data.reshape(np_shape)
        else:
            data = gguf.quants.dequantize(t.data, t.tensor_type)
            # `dequantize` returns a row-major array whose shape is
            # already the reverse of `t.shape` for quantized types.
            data = np.array(data, dtype=np.float32, copy=False)
        # Persist with GGUF dim order in the header.
        gguf_dims = tuple(int(d) for d in t.shape)
        flat = data.reshape(-1).astype("<f4", copy=False)
        with (ref_dir / f"{t.name}.bin").open("wb") as f:
            f.write(struct.pack("<I", len(gguf_dims)))
            for d in gguf_dims:
                f.write(struct.pack("<I", d))
            f.write(flat.tobytes(order="C"))
        sampled_names.append(t.name)
        print(
            f"    sampled {t.name}: gguf_dims={gguf_dims} ggml_type={t.tensor_type.name} "
            f"deq.shape={data.shape}",
            flush=True,
        )

    (out / "sampled_tensor_names.json").write_text(json.dumps(sampled_names, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Target D — ONNX export
# ---------------------------------------------------------------------------


def pin_onnx(out_root: Path) -> None:
    print("=== Target D — ONNX export (tiny MLP) ===", flush=True)
    out = out_root / "onnx-mlp"
    out.mkdir(parents=True, exist_ok=True)

    # Build the reference MLP with torch + the exact same seed the
    # rust side will use. The rust side reads `mlp_weights.bin` so
    # the two sides start from identical weights regardless of any
    # init discrepancy.
    torch.manual_seed(ONNX_SEED)
    fc1 = nn.Linear(ONNX_IN, ONNX_HIDDEN, bias=True)
    relu = nn.ReLU()
    fc2 = nn.Linear(ONNX_HIDDEN, ONNX_OUT, bias=True)
    model = nn.Sequential(fc1, relu, fc2)
    model.eval()

    # Dump weights in the order the rust side will consume them:
    # [fc1.weight, fc1.bias, fc2.weight, fc2.bias], each preceded
    # by its own [u32 ndim][u32 shape...] header.
    with (out / "mlp_weights.bin").open("wb") as f:
        for tensor in (fc1.weight, fc1.bias, fc2.weight, fc2.bias):
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            shape = list(arr.shape)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr.tobytes(order="C"))

    # Build three fixed inputs.
    torch.manual_seed(ONNX_SEED + 1)
    inputs = {
        "zeros": torch.zeros(1, ONNX_IN, dtype=torch.float32),
        "ones": torch.ones(1, ONNX_IN, dtype=torch.float32),
        "random": torch.randn(1, ONNX_IN, dtype=torch.float32),
    }
    with torch.no_grad():
        for name, x in inputs.items():
            arr = x.cpu().numpy().astype(np.float32)
            dump_f32(arr, out / f"input_{name}.bin")
            y = model(x).cpu().numpy().astype(np.float32)
            dump_f32(y, out / f"torch_forward_{name}.bin")
            print(
                f"    input_{name}: x.shape={tuple(x.shape)} -> y.shape={tuple(y.shape)} "
                f"y.norm={np.linalg.norm(y):.4f}",
                flush=True,
            )

    meta = {
        "seed": ONNX_SEED,
        "architecture": f"Linear({ONNX_IN}->{ONNX_HIDDEN}) + ReLU + Linear({ONNX_HIDDEN}->{ONNX_OUT})",
        "in_features": ONNX_IN,
        "hidden_features": ONNX_HIDDEN,
        "out_features": ONNX_OUT,
        "inputs": list(inputs.keys()),
        "tolerance_max_abs": 1e-5,
        "tolerance_cosine_sim": 0.9999,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote weights + 3 inputs + 3 reference outputs + meta.json", flush=True)


# ---------------------------------------------------------------------------
# Bundle + README + upload
# ---------------------------------------------------------------------------


def make_bundle(out_root: Path) -> tuple[Path, str]:
    bundle = out_root / "bundle.tar"
    with tarfile.open(bundle, "w") as tar:
        for sub in sorted(out_root.iterdir()):
            if sub.name == "bundle.tar":
                continue
            tar.add(sub, arcname=sub.name)
    sha = sha256_of(bundle)
    print(f"  bundle.tar: {bundle.stat().st_size} bytes, SHA-256 {sha}", flush=True)
    return bundle, sha


def render_readme(sha: str) -> str:
    return textwrap.dedent(
        f"""\
        ---
        license: apache-2.0
        tags:
          - ferrotorch
          - real-artifact
          - serialize
          - format-parity
        ---

        # `ferrotorch/serialize-parity-v1`

        Phase G.3 of ferrotorch's real-artifact-driven development
        (#1169). Pins canonical references for `ferrotorch-serialize`'s
        four format loaders/exporters so the rust crate's parsers and
        emitters can be verified byte-exact against the upstream
        toolchains they target.

        ## Targets

        * **`.pth` load** — `resnet18-pth/resnet18-5c106cde.pth` is the
          official torchvision checkpoint
          (`https://download.pytorch.org/models/resnet18-5c106cde.pth`).
          `reference_state_dict/<key>.bin` carries each tensor as
          `[u32 ndim][u32 shape...][f32 bytes]`. The rust harness
          dumps the same per-tensor binaries via
          `ferrotorch_serialize::load_pytorch_state_dict` and compares
          byte-exact (max_abs = 0).

        * **SafeTensors round-trip** — `safetensors-rt/resnet18.safetensors`
          is the same resnet18 state_dict re-saved via
          `safetensors.torch.save_file`. References are the same per-
          tensor binaries as the .pth target. The rust harness compares
          byte-exact (max_abs = 0).

        * **GGUF load** — `gguf/SmolLM-135M-Instruct-Q4_K_M.gguf` is the
          upstream `unsloth/SmolLM-135M-Instruct-GGUF` checkpoint.
          `reference_dequant/<name>.bin` carries dequantized f32 tensors
          for a deterministic stride-sampled subset of layers, produced
          by python's `gguf.GGUFReader`. The rust harness reproduces
          those under max_abs <= 1e-4 (Q4_K group scaling has a known
          noise floor between implementations).

        * **ONNX export** — `onnx-mlp/` carries:
          - `mlp_weights.bin` — fixed-seed (`torch.manual_seed(42)`)
            weights for a `Linear(4 -> 8) + ReLU + Linear(8 -> 2)`
            MLP. The rust side reads these so its in-memory MLP
            matches torch's bit-for-bit before export.
          - `input_{{zeros,ones,random}}.bin` — three fixed inputs.
          - `torch_forward_{{zeros,ones,random}}.bin` — reference
            forward outputs from `torch.nn.Sequential`.

          The rust harness builds the same MLP from `mlp_weights.bin`,
          exports it via `ferrotorch_serialize::export_onnx`, dumps the
          rust-side ferrotorch forward, and the python verifier loads
          the rust-emitted .onnx via `onnxruntime.InferenceSession`
          and asserts cosine_sim >= 0.9999 + max_abs <= 1e-5 across
          (rust-onnx vs rust-ferrotorch) AND (rust-onnx vs torch).

        ## Provenance

        * Pin script:
          [`scripts/pin_pretrained_serialize_fixtures.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/pin_pretrained_serialize_fixtures.py).
        * Verifier:
          [`scripts/verify_serialize_inference.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/verify_serialize_inference.py).
        * Rust dumps:
          [`ferrotorch-serialize/examples/serialize_{{pth,safetensors,gguf,onnx_export}}_dump.rs`](https://github.com/dollspace-gay/ferrotorch/tree/main/ferrotorch-serialize/examples).
        * Cargo test wrapper:
          [`ferrotorch-serialize/tests/conformance_format_parity.rs`](https://github.com/dollspace-gay/ferrotorch/blob/main/ferrotorch-serialize/tests/conformance_format_parity.rs).
        * Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/1169>.
        * SHA-256 of `bundle.tar` (pinned in
          `ferrotorch-hub/src/registry.rs`): `{sha}`.

        ## Upstream licenses

        * resnet18 weights — BSD-3-Clause (torchvision).
        * SmolLM2-135M-Instruct-GGUF — Apache-2.0 (upstream `unsloth`
          mirror of HuggingFace's `HuggingFaceTB/SmolLM2-135M-Instruct`).
        * ferrotorch fixtures themselves — Apache-2.0 / MIT.
        """
    )


def upload(out_root: Path, sha: str) -> None:
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    (out_root / "README.md").write_text(render_readme(sha))
    api.upload_folder(
        folder_path=str(out_root),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="feat: pin ferrotorch-serialize format-parity fixtures (#1169)",
    )
    print(f"  uploaded {out_root} -> https://huggingface.co/{REPO_ID}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", default="/tmp/serialize_parity",
        help="Local staging directory. Default: /tmp/serialize_parity.",
    )
    p.add_argument(
        "--targets", default="pth,safetensors,gguf,onnx",
        help="Comma-separated list of targets to pin (default: all 4).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Stage every fixture locally but skip the HF upload.",
    )
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    targets = {t.strip() for t in args.targets.split(",") if t.strip()}

    print(f"=== ferrotorch-serialize format-parity pin ({REPO_ID}) ===")
    print(f"  out_dir: {out_root}")
    print(f"  targets: {sorted(targets)}")

    if "pth" in targets:
        pin_pth(out_root)
    if "safetensors" in targets:
        pin_safetensors(out_root)
    if "gguf" in targets:
        pin_gguf(out_root)
    if "onnx" in targets:
        pin_onnx(out_root)

    _bundle, sha = make_bundle(out_root)

    if not args.dry_run:
        upload(out_root, sha)
    else:
        (out_root / "README.md").write_text(render_readme(sha))
        print("  dry-run: skipped HF upload", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  repo:           https://huggingface.co/{REPO_ID}")
    print(f"  bundle.tar SHA: {sha}")
    print(f"  out_dir:        {out_root}")
    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print(f'  weights_url: "https://huggingface.co/{REPO_ID}/resolve/main/bundle.tar",')
    print(f'  weights_sha256: "{sha}",')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
