#!/usr/bin/env python3
"""Pin a pretrained Whisper audio-encoder checkpoint to the `ferrotorch/*` HF org.

Phase B.2 of real-artifact-driven development (closes ferrotorch issue
#1149).

Mirrors `scripts/pin_pretrained_text_weights.py` but for audio
encoder-only models. For the chosen model this script:

1. Downloads the upstream HF safetensors + tokenizer + feature-extractor
   preprocessor config + the model config.
2. Verifies the safetensors `encoder.*` key list matches the layout
   `ferrotorch_whisper::WhisperEncoder::load_hf_state_dict` consumes.
   Every encoder key must either map onto a parameter or appear in the
   documented (always-empty) drop list — silent state-dict drops are
   refused.
3. Generates a fixed parity probe:
     - `_value_parity_audio.bin`: deterministic synthetic 30-second
       waveform (sum of sine waves at three frequencies) as the
       `[u32 ndim][u32 shape][f32 audio]` little-endian dump. Choosing
       a synthetic signal makes the parity probe a stable, network-free
       artifact — we don't depend on librispeech / etc. for re-pin.
     - `_value_parity_mel.bin`: the
       `WhisperFeatureExtractor.from_pretrained(<repo>)(audio, ...)`
       output `[1, 80, 3000]` in the same dump format. This is the
       reference log-mel — the Rust-side `ferrotorch_whisper::audio` is
       compared against it.
     - `_value_parity_encoder_output.bin`: float32 encoder hidden
       states `[1, 1500, 384]` from a fresh
       `WhisperModel.from_pretrained(<repo>).encoder(input_features=mel)`
       forward pass on those mel features.
4. Re-packs the encoder-only subset of the safetensors so the pinned
   `model.safetensors` carries ONLY the keys ferrotorch consumes (no
   decoder / `proj_out` payload). This shrinks the upload to ~one-quarter
   of the full Whisper-tiny size and makes the pin asymmetric — a future
   change that needs the decoder cannot accidentally use this mirror.
5. Uploads `model.safetensors`, `config.json`,
   `preprocessor_config.json`, the parity probe files, and a README to
   `huggingface.co/ferrotorch/<name>`.
6. Hashes the uploaded `model.safetensors` with SHA-256 and prints a
   registry-ready snippet for `ferrotorch-hub/src/registry.rs`.

Usage:
    python3 scripts/pin_pretrained_whisper_weights.py \
        [--model whisper-tiny-encoder] \
        [--dry-run] [--skip-upload] \
        [--out-dir /tmp/ferrotorch_pretrained_whisper_weights]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import WhisperFeatureExtractor, WhisperModel


# ---------------------------------------------------------------------------
# MIT LICENSE text (Whisper is MIT-licensed; we redistribute weights
# byte-for-byte for the encoder slice).
# ---------------------------------------------------------------------------
MIT_LICENSE_NOTICE = """\
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


@dataclass
class WhisperModelInfo:
    """One pinnable Whisper-encoder model entry."""

    name: str
    upstream_repo: str
    description: str
    license: str
    param_count: int


MODELS: dict[str, WhisperModelInfo] = {
    "whisper-tiny-encoder": WhisperModelInfo(
        name="whisper-tiny-encoder",
        upstream_repo="openai/whisper-tiny",
        description=(
            "whisper-tiny encoder (openai/whisper-tiny). 4-layer 6-head "
            "Transformer audio encoder, d_model=384, encoder_ffn_dim=1536, "
            "num_mel_bins=80, max_source_positions=1500. MIT-licensed. "
            "Pinned encoder-only — decoder/proj_out weights are dropped "
            "from this mirror. Real-artifact baseline for audio encoder "
            "parity vs transformers (#1149)."
        ),
        license="mit",
        # Encoder-only parameter count:
        # conv1 = 80*384*3 + 384 = 92544
        # conv2 = 384*384*3 + 384 = 442752
        # embed_positions = 1500 * 384 = 576000
        # per-layer (×4):
        #   self_attn_layer_norm = 2 * 384 = 768
        #   q_proj = 384*384 + 384 = 147840
        #   k_proj = 384*384 = 147456 (NO BIAS)
        #   v_proj = 384*384 + 384 = 147840
        #   out_proj = 384*384 + 384 = 147840
        #   final_layer_norm = 2 * 384 = 768
        #   fc1 = 384*1536 + 1536 = 591360
        #   fc2 = 1536*384 + 384 = 590208
        # layer_norm = 2 * 384 = 768
        # = 92544 + 442752 + 576000
        #   + 4 * (768 + 147840 + 147456 + 147840 + 147840 + 768 +
        #          591360 + 590208)
        #   + 768
        # = 1111296 + 4*1774080 + 768
        # = 1111296 + 7096320 + 768
        # = 8208384
        param_count=8_208_384,
    ),
}


# ---------------------------------------------------------------------------
# Expected ferrotorch-whisper encoder state-dict key set, parameterised
# by config. Mirrors `WhisperEncoder::named_parameters()` exactly.
# ---------------------------------------------------------------------------

def expected_encoder_keys_and_shapes(cfg: dict, prefix: str) -> dict[str, list[int]]:
    """Per-parameter shape pin. Refuses any checkpoint whose layout
    diverges from what the loader will consume.

    `prefix` is the upstream HF key prefix that lives in front of the
    encoder. For `WhisperForConditionalGeneration` checkpoints it is
    `"model.encoder."`; for a bare `WhisperModel` it is `"encoder."`.
    The pin always re-packs into the `encoder.*` form so the mirror's
    layout is normalised regardless of upstream variant.
    """
    d = cfg["d_model"]
    ff = cfg["encoder_ffn_dim"]
    n_mel = cfg["num_mel_bins"]
    n_pos = cfg["max_source_positions"]
    n_layers = cfg["encoder_layers"]

    shapes: dict[str, list[int]] = {
        f"{prefix}conv1.weight": [d, n_mel, 3],
        f"{prefix}conv1.bias": [d],
        f"{prefix}conv2.weight": [d, d, 3],
        f"{prefix}conv2.bias": [d],
        f"{prefix}embed_positions.weight": [n_pos, d],
        f"{prefix}layer_norm.weight": [d],
        f"{prefix}layer_norm.bias": [d],
    }
    for i in range(n_layers):
        p = f"{prefix}layers.{i}"
        shapes[f"{p}.self_attn_layer_norm.weight"] = [d]
        shapes[f"{p}.self_attn_layer_norm.bias"] = [d]
        shapes[f"{p}.self_attn.q_proj.weight"] = [d, d]
        shapes[f"{p}.self_attn.q_proj.bias"] = [d]
        shapes[f"{p}.self_attn.k_proj.weight"] = [d, d]
        # NO k_proj.bias — Whisper convention.
        shapes[f"{p}.self_attn.v_proj.weight"] = [d, d]
        shapes[f"{p}.self_attn.v_proj.bias"] = [d]
        shapes[f"{p}.self_attn.out_proj.weight"] = [d, d]
        shapes[f"{p}.self_attn.out_proj.bias"] = [d]
        shapes[f"{p}.final_layer_norm.weight"] = [d]
        shapes[f"{p}.final_layer_norm.bias"] = [d]
        shapes[f"{p}.fc1.weight"] = [ff, d]
        shapes[f"{p}.fc1.bias"] = [ff]
        shapes[f"{p}.fc2.weight"] = [d, ff]
        shapes[f"{p}.fc2.bias"] = [d]
    return shapes


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_f32(data: np.ndarray, path: Path) -> None:
    """Dump a float32 ndarray in the `[u32 ndim][u32 shape][f32]`
    little-endian format the Rust dump example writes."""
    arr = data.reshape(-1).astype("<f4", copy=False)
    shape = list(data.shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def synthetic_audio() -> np.ndarray:
    """Deterministic 30-second audio: sum of three pure tones modulated
    by a slow envelope. f32 in [-1, 1], 16 kHz mono, 480 000 samples."""
    sr = 16000
    n = sr * 30
    t = np.arange(n, dtype=np.float64) / sr
    # 220 Hz + 440 Hz + 880 Hz with a 0.25 Hz amplitude envelope. The
    # combination produces non-trivial mel-band activity across the
    # whole 30-second window — silence would let the floor clamp the
    # log-mel to its minimum, hiding STFT drift.
    sig = (
        0.30 * np.sin(2 * np.pi * 220.0 * t)
        + 0.20 * np.sin(2 * np.pi * 440.0 * t)
        + 0.10 * np.sin(2 * np.pi * 880.0 * t)
    )
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 0.25 * t))
    sig = sig * env
    # Normalize to peak 0.9.
    peak = float(np.max(np.abs(sig)))
    if peak > 0:
        sig = sig * (0.9 / peak)
    return sig.astype(np.float32, copy=False)


def convert_one(info: WhisperModelInfo, out_root: Path) -> tuple[str, Path]:
    """Download, verify, write parity probe. Returns (sha256, model_dir)."""
    print(f"\n=== {info.name} <- {info.upstream_repo} ===", flush=True)

    out_dir = out_root / info.name
    out_dir.mkdir(parents=True, exist_ok=True)

    upstream_files = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ]
    local_paths: dict[str, Path] = {}
    for fn in upstream_files:
        try:
            p = hf_hub_download(repo_id=info.upstream_repo, filename=fn)
        except Exception as e:
            raise SystemExit(
                f"{info.name}: failed to download upstream {fn} from "
                f"{info.upstream_repo}: {e}"
            )
        target = out_dir / fn
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(p).read_bytes())
        local_paths[fn] = target
        print(f"  fetched {fn} -> {target}", flush=True)

    cfg = json.loads(local_paths["config.json"].read_text())
    arch = cfg.get("architectures", [])
    print(
        f"  config: arch={arch} d_model={cfg['d_model']} "
        f"encoder_layers={cfg['encoder_layers']} heads={cfg['encoder_attention_heads']} "
        f"ffn={cfg['encoder_ffn_dim']} mel={cfg['num_mel_bins']} "
        f"max_src_pos={cfg['max_source_positions']} "
        f"act={cfg.get('activation_function')}",
        flush=True,
    )

    # ---- Verify safetensors layout (encoder subset). -----------------
    # Detect whether the upstream uses the `model.encoder.*` (full
    # `WhisperForConditionalGeneration`) or `encoder.*` (bare
    # `WhisperModel`) layout. We always re-pack to `encoder.*` so the
    # ferrotorch mirror has one canonical layout.
    with safe_open(local_paths["model.safetensors"], framework="pt") as f:
        full_keys = set(f.keys())
        full_shapes: dict[str, list[int]] = {
            k: list(f.get_slice(k).get_shape()) for k in full_keys
        }
        full_tensors: dict[str, torch.Tensor] = {
            k: f.get_tensor(k) for k in full_keys
        }
    has_model_prefix = any(k.startswith("model.encoder.") for k in full_keys)
    has_bare_prefix = any(k.startswith("encoder.") for k in full_keys)
    if has_model_prefix and not has_bare_prefix:
        upstream_prefix = "model.encoder."
    elif has_bare_prefix and not has_model_prefix:
        upstream_prefix = "encoder."
    else:
        raise SystemExit(
            f"{info.name}: cannot infer encoder prefix — upstream keys mix "
            f"`encoder.*` and `model.encoder.*` (or have neither). Refusing "
            f"to pin until the layout is disambiguated."
        )
    print(f"  upstream encoder prefix: {upstream_prefix!r}", flush=True)

    expected_shapes = expected_encoder_keys_and_shapes(cfg, upstream_prefix)
    encoder_keys = {k for k in full_keys if k.startswith(upstream_prefix)}
    missing = set(expected_shapes) - encoder_keys
    if missing:
        raise SystemExit(
            f"{info.name}: ferrotorch-whisper expects {len(missing)} encoder "
            f"keys absent from the upstream safetensors. Sample: "
            f"{sorted(missing)[:5]}"
        )
    unexpected_enc = encoder_keys - set(expected_shapes)
    if unexpected_enc:
        raise SystemExit(
            f"{info.name}: upstream safetensors has {len(unexpected_enc)} "
            f"`{upstream_prefix}*` keys ferrotorch-whisper does NOT consume. "
            f"Refusing to pin (we will not silently drop encoder parameters). "
            f"Sample: {sorted(unexpected_enc)[:5]}"
        )
    for k, exp in expected_shapes.items():
        got = full_shapes.get(k)
        if got != exp:
            raise SystemExit(
                f"{info.name}: shape mismatch for '{k}': upstream {got} vs "
                f"ferrotorch expects {exp}. Refusing to pin."
            )
    non_encoder = sorted(full_keys - encoder_keys)
    print(
        f"  state-dict cross-check OK: {len(expected_shapes)}/{len(expected_shapes)} "
        f"encoder keys mapped; will drop {len(non_encoder)} non-encoder keys "
        f"from the mirror (first few: {non_encoder[:3]}).",
        flush=True,
    )

    # ---- Re-pack encoder-only safetensors with canonical `encoder.*`
    #      prefix. -----------------------------------------------------
    encoder_tensors: dict[str, torch.Tensor] = {}
    for k in expected_shapes:
        # Strip the upstream prefix and re-attach the canonical one.
        canon = "encoder." + k[len(upstream_prefix):]
        # Clone to a contiguous tensor so safe_save accepts the slice.
        encoder_tensors[canon] = full_tensors[k].contiguous().clone()
    encoder_path = out_dir / "model.safetensors"
    save_file(encoder_tensors, str(encoder_path))
    print(
        f"  re-packed encoder-only model.safetensors "
        f"({encoder_path.stat().st_size} bytes, {len(encoder_tensors)} keys)",
        flush=True,
    )

    # ---- Generate parity probe. --------------------------------------
    print("  generating value-parity probe…", flush=True)
    audio = synthetic_audio()
    print(
        f"  synthetic audio: shape={audio.shape} dtype={audio.dtype} "
        f"min={audio.min():.4f} max={audio.max():.4f}",
        flush=True,
    )
    fe = WhisperFeatureExtractor.from_pretrained(info.upstream_repo)
    mel = fe(audio, sampling_rate=16000, return_tensors="np").input_features
    print(
        f"  mel: shape={mel.shape} dtype={mel.dtype} "
        f"min={mel.min():.4f} max={mel.max():.4f} "
        f"mean={mel.mean():.4f} std={mel.std():.4f}",
        flush=True,
    )
    if mel.shape != (1, 80, 3000):
        raise SystemExit(
            f"{info.name}: mel shape {mel.shape} != expected (1, 80, 3000)"
        )

    # Run the upstream encoder (without the decoder).
    model = WhisperModel.from_pretrained(info.upstream_repo, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        enc_out = model.encoder(input_features=torch.from_numpy(mel)).last_hidden_state
    enc_np = enc_out.cpu().numpy().astype(np.float32)
    print(
        f"  encoder output: shape={enc_np.shape} "
        f"min={enc_np.min():.4f} max={enc_np.max():.4f} "
        f"mean={enc_np.mean():.4f} std={enc_np.std():.4f}",
        flush=True,
    )
    if enc_np.shape != (1, 1500, 384):
        raise SystemExit(
            f"{info.name}: encoder output shape {enc_np.shape} != (1, 1500, 384)"
        )

    parity_audio = out_dir / "_value_parity_audio.bin"
    dump_f32(audio.reshape(1, -1), parity_audio)
    parity_mel = out_dir / "_value_parity_mel.bin"
    dump_f32(mel, parity_mel)
    parity_enc = out_dir / "_value_parity_encoder_output.bin"
    dump_f32(enc_np, parity_enc)
    print(
        f"  wrote {parity_audio.name} ({parity_audio.stat().st_size} bytes), "
        f"{parity_mel.name} ({parity_mel.stat().st_size} bytes), "
        f"{parity_enc.name} ({parity_enc.stat().st_size} bytes)",
        flush=True,
    )

    # ---- SHA of the re-packed encoder-only safetensors. --------------
    sha = sha256_of(encoder_path)
    print(f"  encoder-only model.safetensors SHA-256: {sha}", flush=True)

    # ---- README. -----------------------------------------------------
    readme_path = out_dir / "README.md"
    readme_path.write_text(render_readme(info, cfg, sha, non_encoder))
    print(f"  wrote {readme_path}", flush=True)

    return sha, out_dir


def render_readme(info: WhisperModelInfo, cfg: dict, sha: str, non_encoder: list[str]) -> str:
    return textwrap.dedent(f"""\
        ---
        license: {info.license}
        tags:
          - automatic-speech-recognition
          - audio
          - whisper
          - ferrotorch
        ---

        # `ferrotorch/{info.name}`

        {info.description}

        ## Provenance

        * Upstream: `{info.upstream_repo}` ({info.license}).
        * Conversion script:
          [`ferrotorch/scripts/pin_pretrained_whisper_weights.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/pin_pretrained_whisper_weights.py).
        * Ferrotorch issue: <https://github.com/dollspace-gay/ferrotorch/issues/1149>.
        * SHA-256 of `model.safetensors` (this file is pinned in
          `ferrotorch-hub/src/registry.rs`): `{sha}`.
        * Number of trainable parameters in the encoder slice:
          **{info.param_count:,}**.
        * Config snapshot: d_model={cfg['d_model']},
          encoder_layers={cfg['encoder_layers']},
          encoder_attention_heads={cfg['encoder_attention_heads']},
          encoder_ffn_dim={cfg['encoder_ffn_dim']},
          num_mel_bins={cfg['num_mel_bins']},
          max_source_positions={cfg['max_source_positions']},
          activation_function={cfg.get('activation_function', 'gelu')!r}.
        * Non-encoder keys dropped from the upstream checkpoint (this
          mirror is encoder-only): {len(non_encoder)} total, first few:
          `{non_encoder[:3]}`.

        ## Value-parity probe

        Three extra files are uploaded so the ferrotorch-side harness can
        reproduce the parity verdict without re-running the upstream
        Whisper model:

        * `_value_parity_audio.bin` — deterministic synthetic 30-second
          audio (sum of three sine waves with a slow envelope),
          16 kHz mono float32, shape `[1, 480000]`.
        * `_value_parity_mel.bin` — `WhisperFeatureExtractor(audio)`
          output `[1, 80, 3000]` float32 from the upstream feature
          extractor. The Rust-side `ferrotorch_whisper::audio` is
          compared against this.
        * `_value_parity_encoder_output.bin` — float32 encoder hidden
          states `[1, 1500, 384]` from
          `WhisperModel.encoder(input_features=mel).last_hidden_state`.
          Format: `[u32 ndim][u32 × ndim shape][f32 × prod(shape)]`
          little-endian (matches every other ferrotorch dump).

        ## How to load

        ```rust
        use ferrotorch_whisper::{{
            HfWhisperConfig, WhisperConfig, load_whisper_encoder,
        }};
        use ferrotorch_hub::{{HubCache, hf_download_model}};

        let cache = HubCache::with_default_dir();
        let repo_dir = hf_download_model("ferrotorch/{info.name}", "main", &cache)?;
        let hf_cfg = HfWhisperConfig::from_file(repo_dir.join("config.json"))?;
        let cfg = WhisperConfig::from_hf(&hf_cfg)?;
        let (encoder, _drop_report) = load_whisper_encoder::<f32>(
            &repo_dir.join("model.safetensors"),
            cfg,
            /* strict = */ false,
        )?;
        ```

        ## Upstream license

        ```
{textwrap.indent(MIT_LICENSE_NOTICE, '        ')}
        ```
    """)


def hf_upload(info: WhisperModelInfo, out_dir: Path) -> None:
    api = HfApi()
    repo_id = f"ferrotorch/{info.name}"
    print(f"  uploading to https://huggingface.co/{repo_id}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    files = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "_value_parity_audio.bin",
        "_value_parity_mel.bin",
        "_value_parity_encoder_output.bin",
        "README.md",
    ]
    for fname in files:
        p = out_dir / fname
        if not p.exists():
            print(f"    skip (missing locally): {fname}", flush=True)
            continue
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"feat: pin encoder-only artifact for {info.name} (#1149)",
        )
        print(f"    uploaded {fname}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="whisper-tiny-encoder",
        help="Which model to pin (key in MODELS). Default: whisper-tiny-encoder.",
    )
    p.add_argument(
        "--out-dir", default="/tmp/ferrotorch_pretrained_whisper_weights",
        help="Staging directory.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Stage everything locally but do not upload.")
    p.add_argument("--skip-upload", action="store_true",
                   help="Alias for --dry-run.")
    args = p.parse_args()

    if args.model not in MODELS:
        print(f"unknown model '{args.model}'. Known: {list(MODELS)}",
              file=sys.stderr)
        return 2

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    info = MODELS[args.model]
    sha, out_dir = convert_one(info, out_root)
    if not (args.dry_run or args.skip_upload):
        hf_upload(info, out_dir)

    print("\n=== SUMMARY ===")
    print(f"  {info.name:24s}  sha256={sha}")
    print(f"  hf:   https://huggingface.co/ferrotorch/{info.name}")
    print(f"  dir:  {out_dir}")
    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print(f"  // {info.name}: {info.upstream_repo}")
    print(f'  weights_url: "https://huggingface.co/ferrotorch/{info.name}/resolve/main/model.safetensors",')
    print(f'  weights_sha256: "{sha}",')
    print(f"  num_parameters: {info.param_count},")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
