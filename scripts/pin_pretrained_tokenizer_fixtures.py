#!/usr/bin/env python3
"""Pin a HuggingFace tokenizer parity fixture set to the
`ferrotorch/tokenizer-parity-v1` HF mirror.

Phase G.2 of real-artifact-driven development (#1168). For every
canonical tokenizer family below, this script runs Python's
`transformers.AutoTokenizer` over a fixed corpus of test strings and
snapshots:

  * `<family>/tokenizer.json`             — the tokenizer config the
                                            rust side will read back.
  * `<family>/tokenizer_config.json`      — full tokenizer config
                                            (when the upstream repo
                                            ships one), keeps the
                                            chat-template string if
                                            present.
  * `<family>/strings.json`               — the fixed list of test
                                            input strings.
  * `<family>/token_ids.json`             — reference encode results
                                            per string for both
                                            `add_special_tokens` true
                                            and false.
  * `<family>/decoded.json`               — reference decode results
                                            per string for the
                                            ids-with-special variant,
                                            both `skip_special_tokens`
                                            false and true. Also the
                                            decode-no-special variant.
  * `<family>/chat_template.json`         — for tokenizers that ship a
                                            chat template: the rendered
                                            conversation string for
                                            `add_generation_prompt`
                                            false and true, plus the
                                            template string and the
                                            BOS/EOS tokens.
  * `<family>/meta.json`                  — provenance: upstream repo,
                                            transformers/tokenizers
                                            versions, vocab sizes.

Output corpus covers ASCII, long, unicode, whitespace-heavy, code,
template control tokens, and edge cases (empty string, single
character, leading/trailing spaces).

Canonical tokenizer families:

  1. llama3   — `meta-llama/Meta-Llama-3-8B-Instruct`      (BPE,         BOS 128000, chat tpl)
  2. clip     — `openai/clip-vit-large-patch14`            (BPE,         BOS 49406)
  3. bert     — `bert-base-uncased`                        (WordPiece,   CLS 101)
  4. gpt2     — `gpt2`                                     (BPE,         no BOS by default)
  5. smollm   — `HuggingFaceTB/SmolLM-135M-Instruct`       (BPE,         BOS 1, chat tpl)

The Instruct variants are used for Llama 3 and SmolLM specifically so the
chat-template path is exercised by real fixtures — the base variants
of both ship the same `tokenizer.json` BPE but no `chat_template` field
in `tokenizer_config.json`, so they would not test the rust
`apply_chat_template` rendering path.

Usage:
  python3 scripts/pin_pretrained_tokenizer_fixtures.py \
      [--out-dir /tmp/ferrotorch_tokenizer_fixtures] \
      [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import transformers
import tokenizers as tokenizers_pkg
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer as FastTokenizer
from transformers import AutoTokenizer


HF_REPO_ID = "ferrotorch/tokenizer-parity-v1"


# Fixed corpus of 20 test strings, covering a representative mix of
# ASCII, unicode, whitespace, code, and tokenizer-control sequences.
# Pinning these here (rather than reading from a separate file) means
# the script is the single source of truth for the corpus.
TEST_STRINGS: list[str] = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog. " * 3,
    "日本語のテスト \U0001f389 émoji",
    "\n\tindented\n  text\n",
    "def foo(x):\n    return x + 1",
    "<|begin_of_text|>Hello<|end_of_text|>",
    "[CLS] sentence A [SEP] sentence B [SEP]",
    "",
    "a",
    "   leading and trailing   ",
    "Mixed 123 with NUMBERS 4567 and symbols !@#$%^&*()",
    "Newline\n\n\nthree",
    "Tab\ttab\ttab",
    "Quote \"double\" and 'single' and `backtick`",
    "URL: https://example.com/path?query=value&other=1",
    "Email: alice@example.com, BOB@FOO.IO",
    "中文测试 with English mixed",
    "Repeating aaaaaaaaaaaa and bbbbbbbbbbbb",
    "Emoji rain \U0001f308\U0001f308\U0001f308 and stars ✨✨",
    "Code: `int main() { return 0; }`",
]


# A fixed conversation used for chat-template parity. We render
# `add_generation_prompt` false and true so both the closed-form
# transcript and the open "assistant turn primed" form are pinned.
CHAT_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]


@dataclass(frozen=True)
class FamilySpec:
    """A canonical tokenizer family to fixture."""
    name: str
    repo_id: str
    tokenizer_kind: str  # human-readable label, kept in meta.json
    # tokenizer.json is required for every family in this set. Some
    # families additionally need tokenizer_config.json for the chat
    # template; this list captures every file we want to mirror.
    files_to_mirror: tuple[str, ...]


SPECS: tuple[FamilySpec, ...] = (
    FamilySpec(
        name="llama3",
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer_kind="BPE (Llama 3, tiktoken-derived)",
        files_to_mirror=("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
    ),
    FamilySpec(
        name="clip",
        repo_id="openai/clip-vit-large-patch14",
        tokenizer_kind="BPE (CLIP, lowercased)",
        files_to_mirror=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    FamilySpec(
        name="bert",
        repo_id="bert-base-uncased",
        tokenizer_kind="WordPiece (BERT, lowercased)",
        files_to_mirror=(
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ),
    ),
    FamilySpec(
        name="gpt2",
        repo_id="gpt2",
        tokenizer_kind="BPE (GPT-2)",
        files_to_mirror=(
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ),
    ),
    FamilySpec(
        name="smollm",
        repo_id="HuggingFaceTB/SmolLM-135M-Instruct",
        tokenizer_kind="BPE (SmolLM, GPT-2 family)",
        files_to_mirror=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
    ),
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_download(repo_id: str, filename: str) -> Path | None:
    """Best-effort: not every repo ships every optional file (e.g.
    `vocab.json` only exists for slow tokenizers). Return None on miss
    so we can keep going."""
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as e:
        print(f"    [skip] {repo_id}:{filename} — {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


def mirror_tokenizer_files(spec: FamilySpec, family_dir: Path) -> list[str]:
    """Pull tokenizer files from the upstream repo into family_dir.

    Returns the list of basenames actually mirrored."""
    mirrored: list[str] = []
    family_dir.mkdir(parents=True, exist_ok=True)
    for fname in spec.files_to_mirror:
        src = try_download(spec.repo_id, fname)
        if src is None:
            continue
        dst = family_dir / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # Copy (not symlink) so the bundle.tar is self-contained.
        shutil.copyfile(src, dst)
        mirrored.append(fname)
    return mirrored


def generate_family(spec: FamilySpec, out_root: Path) -> dict[str, Any]:
    """Fixture one family: mirror files + run AutoTokenizer to produce
    reference encodings, decodings, and chat-template renders."""
    print(f"\n--- family {spec.name} ({spec.repo_id}) ---", flush=True)
    family_dir = out_root / spec.name
    family_dir.mkdir(parents=True, exist_ok=True)

    # --- Mirror tokenizer files -----------------------------------------
    mirrored = mirror_tokenizer_files(spec, family_dir)
    print(f"  mirrored: {mirrored}", flush=True)

    # --- Load reference tokenizers --------------------------------------
    #
    # Two tokenizers, two responsibilities:
    #
    # * `fast` (`tokenizers.Tokenizer.from_file`) is the exact same
    #   library code the ferrotorch-tokenize rust crate wraps; we use
    #   it as the encode/decode reference because that is what rust
    #   actually round-trips. `transformers.AutoTokenizer.decode`
    #   layers extra post-processing on top of this (notably
    #   `clean_up_tokenization_spaces` for Llama 3, and a slow Python
    #   CLIPTokenizer that strips `</w>` markers and inserts spaces)
    #   that the rust crate does NOT — and SHOULD NOT — emulate.
    #   Matching `tokenizers.Tokenizer` is the architecturally correct
    #   parity contract for this crate.
    # * `auto` (`transformers.AutoTokenizer.from_pretrained`) is used
    #   ONLY for `apply_chat_template`, which is a `transformers`-layer
    #   feature (the Jinja2 templating, BOS/EOS injection, generation
    #   prompt). The rust side reproduces it via minijinja; the
    #   rendered string is what we exact-match against.
    tokenizer_json_path = family_dir / "tokenizer.json"
    fast = FastTokenizer.from_file(str(tokenizer_json_path))
    auto = AutoTokenizer.from_pretrained(spec.repo_id)
    vocab_size = fast.get_vocab_size(with_added_tokens=True)
    print(f"  vocab_size: {vocab_size}", flush=True)

    # --- Encode every test string both ways -----------------------------
    encode_with_special: list[list[int]] = []
    encode_no_special: list[list[int]] = []
    for s in TEST_STRINGS:
        ids_ws = fast.encode(s, add_special_tokens=True).ids
        ids_ns = fast.encode(s, add_special_tokens=False).ids
        encode_with_special.append(list(ids_ws))
        encode_no_special.append(list(ids_ns))

    # --- Decode round-trips for both encodings --------------------------
    # ferrotorch_tokenize::decode wraps `tokenizers::Tokenizer::decode`
    # one-for-one; using `fast.decode` as the reference asserts the
    # rust wrapper does not add or drop any post-processing relative to
    # the Python binding of the same library.
    #
    #   (encode_with_special, skip=false) → decoded_ws_keep
    #   (encode_with_special, skip=true)  → decoded_ws_skip
    #   (encode_no_special,   skip=false) → decoded_ns
    decoded_with_special_keep: list[str] = []
    decoded_with_special_skip: list[str] = []
    decoded_no_special: list[str] = []
    for ids_ws, ids_ns in zip(encode_with_special, encode_no_special):
        decoded_with_special_keep.append(fast.decode(ids_ws, skip_special_tokens=False))
        decoded_with_special_skip.append(fast.decode(ids_ws, skip_special_tokens=True))
        decoded_no_special.append(fast.decode(ids_ns, skip_special_tokens=False))

    # --- Chat template (if upstream defines one) ------------------------
    # `apply_chat_template` is a `transformers.AutoTokenizer` feature
    # (not a `tokenizers.Tokenizer` feature) — it renders a Jinja2
    # template stored in `tokenizer_config.json`. We use `auto` for
    # both the template-string lookup and the rendered reference; the
    # rust side reproduces this with minijinja.
    chat_block: dict[str, Any] = {"has_chat_template": False}
    chat_template_str: str | None = getattr(auto, "chat_template", None)
    if chat_template_str:
        rendered_no_gen = auto.apply_chat_template(
            CHAT_MESSAGES,
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered_with_gen = auto.apply_chat_template(
            CHAT_MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_block = {
            "has_chat_template": True,
            "messages": CHAT_MESSAGES,
            "template": chat_template_str,
            "bos_token": auto.bos_token,
            "eos_token": auto.eos_token,
            "rendered_no_generation_prompt": rendered_no_gen,
            "rendered_with_generation_prompt": rendered_with_gen,
        }
        print(
            f"  chat_template: yes; "
            f"len(no_gen)={len(rendered_no_gen)} "
            f"len(with_gen)={len(rendered_with_gen)}",
            flush=True,
        )
    else:
        print("  chat_template: none (upstream tokenizer has no chat_template)", flush=True)

    # --- Persist --------------------------------------------------------
    (family_dir / "strings.json").write_text(json.dumps(TEST_STRINGS, ensure_ascii=False, indent=2))
    (family_dir / "token_ids.json").write_text(
        json.dumps(
            {
                "encode_with_special": encode_with_special,
                "encode_no_special": encode_no_special,
            },
            indent=2,
        )
    )
    (family_dir / "decoded.json").write_text(
        json.dumps(
            {
                "decode_with_special_keep": decoded_with_special_keep,
                "decode_with_special_skip": decoded_with_special_skip,
                "decode_no_special": decoded_no_special,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    (family_dir / "chat_template.json").write_text(
        json.dumps(chat_block, ensure_ascii=False, indent=2)
    )

    meta = {
        "family": spec.name,
        "upstream_repo": spec.repo_id,
        "tokenizer_kind": spec.tokenizer_kind,
        "mirrored_files": mirrored,
        "num_test_strings": len(TEST_STRINGS),
        "vocab_size": int(vocab_size),
        "encode_decode_reference": "tokenizers.Tokenizer.from_file",
        "chat_template_reference": "transformers.AutoTokenizer.apply_chat_template",
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers_pkg.__version__,
        "has_chat_template": chat_block["has_chat_template"],
        "bos_token": auto.bos_token,
        "eos_token": auto.eos_token,
        "pad_token": auto.pad_token,
        "unk_token": auto.unk_token,
    }
    (family_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {len(list(family_dir.glob('*')))} files to {family_dir}", flush=True)
    return meta


def build_bundle(out_root: Path) -> Path:
    tar_path = out_root / "bundle.tar"
    with tarfile.open(tar_path, "w") as tar:
        for sub in sorted(out_root.iterdir()):
            if sub.is_dir():
                tar.add(sub, arcname=sub.name)
    return tar_path


def render_readme(bundle_sha: str, metas: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for m in metas:
        rows.append(
            f"| `{m['family']}` | `{m['upstream_repo']}` | {m['tokenizer_kind']} "
            f"| {m['vocab_size']} | {'yes' if m['has_chat_template'] else 'no'} |"
        )
    table = "\n".join(rows)
    return textwrap.dedent(
        f"""\
        ---
        license: apache-2.0
        tags:
          - tokenizer
          - parity
          - ferrotorch
          - real-artifact
        ---

        # `ferrotorch/tokenizer-parity-v1`

        HuggingFace tokenizer parity fixtures pinned for the ferrotorch
        real-artifact harness (Phase G.2, #1168).

        ## Provenance

        * Upstream tokenizers: see the `upstream_repo` column below.
        * Generator script:
          [`scripts/pin_pretrained_tokenizer_fixtures.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/pin_pretrained_tokenizer_fixtures.py).
        * SHA-256 of `bundle.tar` (pinned in
          `ferrotorch-hub/src/registry.rs`): `{bundle_sha}`.

        ## Families

        | family   | upstream                              | kind                    | vocab  | chat tpl |
        |----------|---------------------------------------|-------------------------|--------|----------|
        {table}

        ## Layout

        Each `<family>/` subfolder ships:

        * `tokenizer.json`            — upstream tokenizer config (fast
                                        tokenizers format).
        * `tokenizer_config.json`     — full config with chat template
                                        (when upstream ships one).
        * `special_tokens_map.json`   — special-token mapping (when
                                        upstream ships one).
        * Additional family-specific files (`vocab.json`, `merges.txt`,
          `vocab.txt`) when upstream ships them — these let rust
          tooling that bypasses `tokenizer.json` still round-trip.
        * `strings.json`              — the 20-element fixed test
                                        corpus (same list for every
                                        family).
        * `token_ids.json`            — `{{ encode_with_special[20],
                                            encode_no_special[20] }}`
                                        — Python reference encodings.
        * `decoded.json`              — `{{ decode_with_special_keep[20],
                                            decode_with_special_skip[20],
                                            decode_no_special[20] }}`
                                        — Python reference decodes.
        * `chat_template.json`        — for families with a chat
                                        template: rendered system+user
                                        +assistant conversation with
                                        and without
                                        `add_generation_prompt`.
        * `meta.json`                 — versions and provenance.

        ## How the rust side consumes this

        The rust dump example
        [`ferrotorch-tokenize/examples/tokenizer_parity_dump.rs`](https://github.com/dollspace-gay/ferrotorch/blob/main/ferrotorch-tokenize/examples/tokenizer_parity_dump.rs)
        loads `tokenizer.json` (and optionally `tokenizer_config.json`)
        from the local family folder, then re-runs encode/decode/chat
        template against the corpus and writes the rust-side outputs
        next to the references. The python harness
        [`scripts/verify_tokenizer_inference.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/verify_tokenizer_inference.py)
        compares every output with **exact integer / string equality**
        — there is no tolerance, divergence on any string surfaces a
        real bug.

        ## Upstream licenses

        Each upstream tokenizer carries its own license. See:

        * meta-llama/Meta-Llama-3-8B  : Meta Llama 3 Community License
        * openai/clip-vit-large-patch14 : MIT
        * bert-base-uncased             : Apache 2.0
        * gpt2                          : MIT
        * HuggingFaceTB/SmolLM-135M     : Apache 2.0

        Only the tokenizer config and vocabulary metadata are mirrored;
        none of these files contains model weights.
        """
    )


def hf_upload(out_root: Path) -> None:
    api = HfApi()
    print(f"\nuploading to https://huggingface.co/{HF_REPO_ID} ...", flush=True)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_root),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="feat: pin ferrotorch-tokenize parity fixtures v1 (#1168)",
    )
    print("upload complete.", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="/tmp/ferrotorch_tokenizer_fixtures",
        help="Staging directory.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage everything locally but do not upload to HF.",
    )
    args = p.parse_args()

    out_root = Path(args.out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    print(f"=== ferrotorch tokenizer parity pin ({HF_REPO_ID}) ===")
    print(f"  staging: {out_root}")
    print(f"  families: {[s.name for s in SPECS]}")

    metas: list[dict[str, Any]] = []
    for spec in SPECS:
        metas.append(generate_family(spec, out_root))

    bundle_path = build_bundle(out_root)
    bundle_sha = sha256_of(bundle_path)
    print(f"\nbundle.tar: {bundle_path.stat().st_size} bytes, SHA-256 {bundle_sha}", flush=True)

    (out_root / "README.md").write_text(render_readme(bundle_sha, metas))

    if not args.dry_run:
        hf_upload(out_root)
    else:
        print("  dry-run: skipped HF upload", flush=True)

    print("\n=== SUMMARY ===")
    for m in metas:
        print(
            f"  {m['family']:8} vocab={m['vocab_size']:>7} "
            f"chat_tpl={'yes' if m['has_chat_template'] else 'no '} "
            f"({m['upstream_repo']})"
        )
    print(f"\nlocal stage:   {out_root}")
    print(f"bundle:        {bundle_path}")
    print(f"bundle sha256: {bundle_sha}")
    print(f"hf:            https://huggingface.co/{HF_REPO_ID}")

    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print("  ModelInfo {")
    print('      name: "tokenizer-parity-v1",')
    print(
        '      description: "Phase G.2 HF tokenizer parity fixtures: 5 canonical '
        "tokenizer families (Llama 3 BPE Instruct, CLIP BPE, BERT WordPiece, "
        "GPT-2 BPE, SmolLM BPE Instruct). 20 fixed test strings per family + a "
        "fixed system+user+assistant chat conversation rendered with and "
        "without add_generation_prompt. Encode/decode references come from "
        "tokenizers.Tokenizer (the exact library the ferrotorch-tokenize rust "
        "crate wraps); chat-template renders come from transformers."
        "AutoTokenizer (the rust side reproduces apply_chat_template via "
        "minijinja). Mixed upstream licenses; real-artifact baseline for "
        'ferrotorch-tokenize parity vs HF (#1168).",'
    )
    print(f'      weights_url: "https://huggingface.co/{HF_REPO_ID}/resolve/main/bundle.tar",')
    print(f'      weights_sha256: "{bundle_sha}",')
    print("      format: WeightsFormat::FerrotorchStateDict,")
    print("      num_parameters: 0,")
    print("  },")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
