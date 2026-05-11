#!/usr/bin/env python3
"""Pin a pretrained causal LM checkpoint to the `ferrotorch/*` HuggingFace org.

Closes ferrotorch issue #1147 (Phase A.5 of real-artifact-driven development).

Mirrors the vision-side `scripts/pin_pretrained_weights.py` but for causal
LMs. For the chosen small Llama-architecture model this script:

1. Downloads the upstream HF safetensors + tokenizer + config.
2. Verifies the safetensors key list matches the layout that
   `ferrotorch_llama::LlamaForCausalLM::load_hf_state_dict` expects
   (HF Llama convention). Every key must map; the only intentionally
   absent key is `lm_head.weight` when `tie_word_embeddings=True`
   (the loader copies `model.embed_tokens.weight` into the lm_head).
3. Generates a fixed parity probe:
     - `_value_parity_input.txt`: the verbatim prompt the harness will run.
     - `_value_parity_output.bin`: logits from a fresh
       `transformers.AutoModelForCausalLM.from_pretrained(..., torch_dtype=float32)`
       forward pass on those tokens. Single prefill, no cache.
   These artifacts let the Rust-side `cargo test` re-derive the
   same comparison numbers without re-running the transformers model
   in CI (the value_parity.bin is uploaded to the mirror).
4. Uploads `model.safetensors`, `config.json`, `tokenizer.json`,
   `tokenizer_config.json`, `special_tokens_map.json`, the parity
   probe files, and a README to `huggingface.co/ferrotorch/<name>`.
5. Hashes the uploaded `model.safetensors` with SHA-256 and prints a
   registry-ready snippet for `ferrotorch-hub/src/registry.rs`.

Run:
    python3 scripts/pin_pretrained_llm_weights.py
        [--model smollm-135m]
        [--dry-run]
        [--skip-upload]
        [--out-dir /tmp/ferrotorch_pretrained_llm_weights]

`--dry-run` writes everything under `--out-dir` but never touches HF.
`--skip-upload` is identical (kept for backwards compatibility with the
vision pin script).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Apache 2.0 LICENSE text (verbatim) — required in every uploaded README
# because the upstream model (HuggingFaceTB/SmolLM-135M) is released under
# Apache 2.0 and we redistribute the weights byte-for-byte.
# Source: https://www.apache.org/licenses/LICENSE-2.0.txt
# ---------------------------------------------------------------------------
APACHE_2_0_LICENSE_NOTICE = """\
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# ---------------------------------------------------------------------------
# Fixed parity probe prompt — the harness will tokenize this exactly. Choose
# a prompt that exercises a non-trivial token sequence (~10 tokens) but is
# short enough that a single prefill is fast on CPU.
# ---------------------------------------------------------------------------
PARITY_PROMPT = "The quick brown fox jumps over the lazy"


# ---------------------------------------------------------------------------
# Per-model metadata. Keys here become both the `ferrotorch/<name>` repo and
# the registry name.
# ---------------------------------------------------------------------------
@dataclass
class LmModelInfo:
    """One pinnable causal-LM model entry."""

    # Local short name (also the `ferrotorch/<name>` repo and registry key).
    name: str
    # Upstream HF repo id (e.g. `HuggingFaceTB/SmolLM-135M`).
    upstream_repo: str
    # Human description (lands in registry.rs and the per-repo README).
    description: str
    # Upstream license tag.
    license: str
    # Parameter count expected for the registry pin.
    param_count: int


MODELS: dict[str, LmModelInfo] = {
    "smollm-135m": LmModelInfo(
        name="smollm-135m",
        upstream_repo="HuggingFaceTB/SmolLM-135M",
        description=(
            "SmolLM-135M (HuggingFaceTB/SmolLM-135M). "
            "Llama-architecture causal LM, 135M parameters, "
            "30 layers / 9 q-heads / 3 kv-heads (GQA), hidden=576, "
            "intermediate=1536, vocab=49152, tie_word_embeddings=true, "
            "rope_theta=10000.0. Apache 2.0 license. Pinned as the "
            "real-artifact baseline for causal LM parity vs "
            "`transformers==4.50.3` (#1147)."
        ),
        license="apache-2.0",
        # Real upstream parameter count: 134_515_008
        # = embed(49152*576) + 30 * (
        #     attn(576*576 + 192*576 + 192*576 + 576*576)
        #   + mlp(3 * 576*1536)
        #   + 2*norm(576)
        #   ) + final_norm(576)
        # = 28311552 + 30 * (884736 + 2654208 + 1152) + 576
        # = 28311552 + 30 * 3540096 + 576
        # = 28311552 + 106202880 + 576 = 134_515_008
        # Plus lm_head is TIED to embeddings (no extra parameters).
        param_count=134_515_008,
    ),
}


# ---------------------------------------------------------------------------
# Expected ferrotorch-llama HF state-dict key set, parameterised by the
# config. Matches `LlamaForCausalLM::load_hf_state_dict`.
# ---------------------------------------------------------------------------


def expected_hf_keys(cfg: dict) -> set[str]:
    """Return the set of HF state-dict keys ferrotorch-llama consumes."""
    n_layers = cfg["num_hidden_layers"]
    keys: set[str] = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(n_layers):
        keys.add(f"model.layers.{i}.input_layernorm.weight")
        keys.add(f"model.layers.{i}.post_attention_layernorm.weight")
        for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
            keys.add(f"model.layers.{i}.self_attn.{sub}.weight")
        for sub in ("gate_proj", "up_proj", "down_proj"):
            keys.add(f"model.layers.{i}.mlp.{sub}.weight")
    if not cfg.get("tie_word_embeddings", False):
        keys.add("lm_head.weight")
    return keys


def expected_param_shapes(cfg: dict) -> dict[str, list[int]]:
    """Per-parameter shape pinning so the pin script can refuse a checkpoint
    whose dimensions disagree with the config (architectural drift)."""
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = hidden // n_heads
    kv_dim = n_kv * head_dim
    n_layers = cfg["num_hidden_layers"]
    shapes: dict[str, list[int]] = {
        "model.embed_tokens.weight": [vocab, hidden],
        "model.norm.weight": [hidden],
    }
    for i in range(n_layers):
        shapes[f"model.layers.{i}.input_layernorm.weight"] = [hidden]
        shapes[f"model.layers.{i}.post_attention_layernorm.weight"] = [hidden]
        shapes[f"model.layers.{i}.self_attn.q_proj.weight"] = [hidden, hidden]
        shapes[f"model.layers.{i}.self_attn.k_proj.weight"] = [kv_dim, hidden]
        shapes[f"model.layers.{i}.self_attn.v_proj.weight"] = [kv_dim, hidden]
        shapes[f"model.layers.{i}.self_attn.o_proj.weight"] = [hidden, hidden]
        shapes[f"model.layers.{i}.mlp.gate_proj.weight"] = [inter, hidden]
        shapes[f"model.layers.{i}.mlp.up_proj.weight"] = [inter, hidden]
        shapes[f"model.layers.{i}.mlp.down_proj.weight"] = [hidden, inter]
    if not cfg.get("tie_word_embeddings", False):
        shapes["lm_head.weight"] = [vocab, hidden]
    return shapes


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_f32_3d(t: torch.Tensor, path: Path) -> None:
    """Dump a 3-D float32 tensor `[1, S, V]` in the same little-endian format
    the Rust example writes:
      [u32 ndim][u32 × ndim shape][f32 × prod(shape) data]
    """
    import struct

    assert t.dtype == torch.float32 and t.ndim == 3, (t.dtype, t.shape)
    flat = t.detach().contiguous().cpu().numpy().reshape(-1).astype("<f4")
    with path.open("wb") as f:
        f.write(struct.pack("<I", t.ndim))
        for d in t.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(flat.tobytes(order="C"))


def convert_one(
    info: LmModelInfo,
    out_root: Path,
) -> tuple[str, Path]:
    """Download, verify, write parity probe. Returns (sha256, model_dir)."""
    print(f"\n=== {info.name} <- {info.upstream_repo} ===", flush=True)

    # --- 1. Download upstream files into the staging dir. ---
    out_dir = out_root / info.name
    out_dir.mkdir(parents=True, exist_ok=True)

    upstream_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
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
        target.write_bytes(Path(p).read_bytes())
        local_paths[fn] = target
        print(f"  fetched {fn} -> {target}", flush=True)

    # --- 2. Parse config + verify key list. ---
    cfg = json.loads(local_paths["config.json"].read_text())
    arch = cfg.get("architectures", [])
    if "LlamaForCausalLM" not in arch:
        raise SystemExit(
            f"{info.name}: upstream architecture {arch!r} is not "
            f"LlamaForCausalLM — ferrotorch-llama cannot load it."
        )
    print(
        f"  config: arch={arch} hidden={cfg['hidden_size']} "
        f"layers={cfg['num_hidden_layers']} heads={cfg['num_attention_heads']} "
        f"kv={cfg.get('num_key_value_heads', cfg['num_attention_heads'])} "
        f"vocab={cfg['vocab_size']} tie={cfg.get('tie_word_embeddings', False)}",
        flush=True,
    )

    expected_keys = expected_hf_keys(cfg)
    expected_shapes = expected_param_shapes(cfg)
    with safe_open(local_paths["model.safetensors"], framework="pt") as f:
        actual_keys = set(f.keys())
        actual_shapes: dict[str, list[int]] = {
            k: list(f.get_slice(k).get_shape()) for k in actual_keys
        }

    missing = expected_keys - actual_keys
    if missing:
        raise SystemExit(
            f"{info.name}: ferrotorch-llama expects {len(missing)} keys "
            f"that are absent from the upstream safetensors. Sample: "
            f"{sorted(missing)[:5]}"
        )
    # tied-embedding upstream commonly omits `lm_head.weight`; the loader
    # copies `model.embed_tokens.weight` in. Allow that, but reject any OTHER
    # unmapped key.
    extra = actual_keys - expected_keys
    if cfg.get("tie_word_embeddings", False):
        extra.discard("lm_head.weight")
    if extra:
        raise SystemExit(
            f"{info.name}: upstream safetensors has {len(extra)} keys that "
            f"ferrotorch-llama does NOT consume — refusing to pin. Sample: "
            f"{sorted(extra)[:5]}. Either ferrotorch-llama is missing "
            f"parameters or the upstream layout drifted; investigate."
        )
    for k, exp in expected_shapes.items():
        got = actual_shapes.get(k)
        if got is None:
            continue  # already handled by the missing-keys check above
        if got != exp:
            raise SystemExit(
                f"{info.name}: shape mismatch for '{k}': upstream {got} vs "
                f"ferrotorch expects {exp}. Refusing to pin."
            )
    print(
        f"  state-dict cross-check OK: "
        f"{len(actual_keys)}/{len(expected_keys)} keys matched, "
        f"all shapes agree.",
        flush=True,
    )

    # --- 3. Generate the parity probe.
    # Tokenize PARITY_PROMPT with the upstream tokenizer, run a fresh
    # transformers forward pass in float32, and dump the logits in the
    # ferrotorch [u32 ndim][u32 shape][f32 data] format.
    print("  generating value-parity probe…", flush=True)
    tok = AutoTokenizer.from_pretrained(info.upstream_repo)
    enc = tok(PARITY_PROMPT, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(torch.int64)  # [1, seq_len]
    seq_len = int(ids.shape[1])
    print(f"  prompt: {PARITY_PROMPT!r} -> {seq_len} tokens: {ids[0].tolist()}",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        info.upstream_repo, torch_dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits.detach().to(torch.float32)  # [1, S, V]
    assert logits.shape == (1, seq_len, cfg["vocab_size"]), logits.shape
    print(
        f"  reference logits: shape={list(logits.shape)} "
        f"max_abs={logits.abs().max().item():.4f} "
        f"argmax[last]={int(logits[0, -1].argmax())}",
        flush=True,
    )

    parity_in = out_dir / "_value_parity_input.txt"
    parity_in.write_text(PARITY_PROMPT + "\n")
    parity_out = out_dir / "_value_parity_output.bin"
    dump_f32_3d(logits, parity_out)
    parity_ids_path = out_dir / "_value_parity_token_ids.json"
    parity_ids_path.write_text(json.dumps(ids[0].tolist()))
    print(
        f"  wrote {parity_in.name}, {parity_out.name} "
        f"({parity_out.stat().st_size} bytes), {parity_ids_path.name}",
        flush=True,
    )

    # --- 4. Compute the safetensors SHA. We pin the upstream-equivalent file.
    # safetensors is identical to the upstream — we redistribute byte-for-byte.
    sha = sha256_of(local_paths["model.safetensors"])
    print(f"  model.safetensors SHA-256: {sha}", flush=True)

    # --- 5. README.
    readme_path = out_dir / "README.md"
    readme_path.write_text(render_readme(info, cfg, sha))
    print(f"  wrote {readme_path}", flush=True)

    return sha, out_dir


def render_readme(info: LmModelInfo, cfg: dict, sha: str) -> str:
    return textwrap.dedent(f"""\
        ---
        license: {info.license}
        tags:
          - text-generation
          - llama
          - ferrotorch
        ---

        # `ferrotorch/{info.name}`

        {info.description}

        ## Provenance

        * Upstream: `{info.upstream_repo}` ({info.license}).
        * Conversion script: [`ferrotorch/scripts/pin_pretrained_llm_weights.py`](https://github.com/dollspace/ferrotorch/blob/main/scripts/pin_pretrained_llm_weights.py).
        * Ferrotorch issue: <https://github.com/dollspace/ferrotorch/issues/1147>.
        * Number of trainable parameters: **{info.param_count:,}**.
        * SHA-256 of `model.safetensors` (this file is pinned in
          `ferrotorch-hub/src/registry.rs`): `{sha}`.
        * Config snapshot: hidden={cfg['hidden_size']}, layers={cfg['num_hidden_layers']},
          heads={cfg['num_attention_heads']}, kv_heads={cfg.get('num_key_value_heads', cfg['num_attention_heads'])},
          intermediate={cfg['intermediate_size']}, vocab={cfg['vocab_size']},
          tie_word_embeddings={cfg.get('tie_word_embeddings', False)},
          rope_theta={cfg.get('rope_theta', 10000.0)},
          rms_norm_eps={cfg.get('rms_norm_eps', 1e-5)}.

        ## Value-parity probe

        Two extra files are uploaded so the ferrotorch-side harness can
        reproduce the parity verdict without re-running the upstream
        transformers model:

        * `_value_parity_input.txt` — the verbatim prompt string the
          harness tokenizes (`"{PARITY_PROMPT}"`).
        * `_value_parity_token_ids.json` — the tokenizer's output for that
          prompt (with the upstream tokenizer's `add_special_tokens=True`).
        * `_value_parity_output.bin` — float32 logits dumped from a fresh
          `transformers.AutoModelForCausalLM.from_pretrained(..., torch_dtype=float32)`
          single-prefill forward pass on those token ids (no cache).
          Format: `[u32 ndim][u32 × ndim shape][f32 × prod(shape) data]`
          little-endian; identical layout to the vision-side dumps.

        ## How to load

        ```rust
        use ferrotorch_hub::load_pretrained;
        use ferrotorch_llama::{{LlamaConfig, LlamaForCausalLM}};
        use ferrotorch_hub::HfTransformerConfig;

        let state = load_pretrained::<f32>("{info.name}")?;
        let hf_cfg = HfTransformerConfig::from_file("config.json")?;
        let cfg = LlamaConfig::from_hf(&hf_cfg)?;
        let mut model = LlamaForCausalLM::<f32>::new(cfg)?;
        model.load_hf_state_dict(&state, /* strict = */ true)?;
        ```

        ## Upstream license

        ```
{textwrap.indent(APACHE_2_0_LICENSE_NOTICE, '        ')}
        ```
    """)


def hf_upload(info: LmModelInfo, out_dir: Path) -> None:
    api = HfApi()
    repo_id = f"ferrotorch/{info.name}"
    print(f"  uploading to https://huggingface.co/{repo_id}", flush=True)
    # Ensure the repo exists (no-op if it already does).
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "_value_parity_input.txt",
        "_value_parity_token_ids.json",
        "_value_parity_output.bin",
        "README.md",
    ]:
        p = out_dir / fname
        if not p.exists():
            print(f"    skip (missing locally): {fname}", flush=True)
            continue
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"feat: pin causal LM artifact for {info.name} (#1147)",
        )
        print(f"    uploaded {fname}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="smollm-135m",
                   help="Which model to pin (key in MODELS). Default: smollm-135m.")
    p.add_argument("--out-dir", default="/tmp/ferrotorch_pretrained_llm_weights",
                   help="Staging directory.")
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
    print(f'  // {info.name}: {info.upstream_repo}')
    print(f'  weights_url: "https://huggingface.co/ferrotorch/{info.name}/resolve/main/model.safetensors",')
    print(f'  weights_sha256: "{sha}",')
    print(f'  num_parameters: {info.param_count},')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
