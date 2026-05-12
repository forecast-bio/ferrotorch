#!/usr/bin/env python3
"""Verify ferrotorch-tokenize encode/decode/chat-template parity vs the
frozen `transformers.AutoTokenizer` reference pinned at
`ferrotorch/tokenizer-parity-v1` (#1168, Phase G.2).

Pipeline parity is verified exactly per (family, string) pair so any
divergence pinpoints a real bug or HF-version skew. There is **no
tolerance**: integer token IDs and decoded strings must match byte for
byte.

Per family the harness:

  1. Pulls the family folder from
     `ferrotorch/tokenizer-parity-v1` via `hf_hub_download` (every
     file enumerated in `meta.json` plus the four fixture JSONs).
  2. Runs `ferrotorch-tokenize/examples/tokenizer_parity_dump` with
     `--family <name> --fixture-dir <dl> --output-dir <rust_out>`.
  3. Reads the rust outputs (`token_ids.json`, `decoded.json`,
     `chat_template.json`) and compares them element-wise against the
     reference JSONs.

Per-string check failures are surfaced with the failing string prefix,
the rust ids, and the reference ids so a single divergence is enough
context to diagnose without re-running the rust binary.

Usage:
    python3 scripts/verify_tokenizer_inference.py [--families llama3,clip,...]
                                                  [--keep-dumps]
                                                  [--local-stage-dir DIR]

`--local-stage-dir` overrides the HF download path; useful while
iterating on the pin script before pushing the mirror.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ROOT = Path(__file__).resolve().parent.parent
DUMP_DIR = Path("/tmp/ferrotorch_verify_tokenizer")
TRAJ_REPO = "ferrotorch/tokenizer-parity-v1"

ALL_FAMILIES: tuple[str, ...] = ("llama3", "clip", "bert", "gpt2", "smollm")


@dataclass
class FamilyVerdict:
    family: str
    passed: bool
    summary: str
    failures: list[str] = field(default_factory=list)


def fetch_family_fixtures(family: str, dest_dir: Path) -> Path:
    """Download every fixture file for `family` from the mirror into a
    per-family subdir under `dest_dir`. Returns the family folder
    path."""
    family_dir = dest_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    repo_files = [f for f in list_repo_files(TRAJ_REPO) if f.startswith(f"{family}/")]
    if not repo_files:
        raise RuntimeError(f"no files in {TRAJ_REPO} under {family}/")
    for f in repo_files:
        cached = hf_hub_download(repo_id=TRAJ_REPO, filename=f)
        dst = family_dir / Path(f).name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # symlink to the cached blob; rust's File::open follows symlinks.
        dst.symlink_to(Path(cached).resolve())
    return family_dir


def stage_local_family(local_stage: Path, family: str, dest_dir: Path) -> Path:
    """Materialize a family folder from a local staging directory (the
    pin script's `--out-dir`). Used when iterating before the mirror is
    uploaded."""
    src = local_stage / family
    if not src.exists():
        raise RuntimeError(f"local stage {src} does not exist")
    family_dir = dest_dir / family
    if family_dir.exists():
        shutil.rmtree(family_dir)
    family_dir.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copyfile(f, family_dir / f.name)
    return family_dir


def run_rust_dump(family: str, fixture_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Build and run the rust tokenizer parity dump for a family."""
    cmd = [
        "cargo", "run", "-q", "-p", "ferrotorch-tokenize", "--release",
        "--example", "tokenizer_parity_dump", "--",
        "--family", family,
        "--fixture-dir", str(fixture_dir),
        "--output-dir", str(output_dir),
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"rust tokenizer dump failed for {family} ({proc.returncode})")
    json_line: str | None = None
    for line in proc.stdout.splitlines():
        t = line.strip()
        if t.startswith("{") and t.endswith("}"):
            json_line = t
    if json_line is None:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"rust dump for {family} did not emit a JSON verdict line")
    return json.loads(json_line)


def _short(s: str, n: int = 40) -> str:
    """Truncate-with-ellipsis for error context; escapes newlines."""
    s_repr = s.replace("\n", "\\n").replace("\t", "\\t")
    if len(s_repr) <= n:
        return s_repr
    return s_repr[:n] + "..."


def compare_family(
    family: str,
    fixture_dir: Path,
    rust_out_dir: Path,
    verbose: bool = False,
) -> FamilyVerdict:
    """Element-wise diff of token IDs, decoded strings, and chat
    template renders for one family."""
    failures: list[str] = []

    strings = json.loads((fixture_dir / "strings.json").read_text(encoding="utf-8"))
    ref_ids = json.loads((fixture_dir / "token_ids.json").read_text(encoding="utf-8"))
    ref_dec = json.loads((fixture_dir / "decoded.json").read_text(encoding="utf-8"))
    ref_chat = json.loads((fixture_dir / "chat_template.json").read_text(encoding="utf-8"))

    rust_ids = json.loads((rust_out_dir / "token_ids.json").read_text(encoding="utf-8"))
    rust_dec = json.loads((rust_out_dir / "decoded.json").read_text(encoding="utf-8"))
    rust_chat = json.loads((rust_out_dir / "chat_template.json").read_text(encoding="utf-8"))

    n = len(strings)

    # --- 1. Encode parity ---------------------------------------------------
    for key in ("encode_with_special", "encode_no_special"):
        r = rust_ids[key]
        p = ref_ids[key]
        if len(r) != n or len(p) != n:
            failures.append(
                f"{key}: length mismatch — rust={len(r)} py={len(p)} expected={n}"
            )
            continue
        for i in range(n):
            if r[i] != p[i]:
                failures.append(
                    f"{key}[{i}] mismatch on input {_short(strings[i])!r}: "
                    f"rust={r[i][:20]}{'...' if len(r[i]) > 20 else ''} "
                    f"py={p[i][:20]}{'...' if len(p[i]) > 20 else ''} "
                    f"(rust_len={len(r[i])} py_len={len(p[i])})"
                )

    # --- 2. Decode parity ---------------------------------------------------
    for key in ("decode_with_special_keep", "decode_with_special_skip", "decode_no_special"):
        r = rust_dec[key]
        p = ref_dec[key]
        if len(r) != n or len(p) != n:
            failures.append(
                f"{key}: length mismatch — rust={len(r)} py={len(p)} expected={n}"
            )
            continue
        for i in range(n):
            if r[i] != p[i]:
                failures.append(
                    f"{key}[{i}] mismatch on input {_short(strings[i])!r}: "
                    f"rust={_short(r[i], 60)!r} py={_short(p[i], 60)!r}"
                )

    # --- 3. Chat template parity -------------------------------------------
    if ref_chat["has_chat_template"] != rust_chat["has_chat_template"]:
        failures.append(
            f"has_chat_template disagreement: rust={rust_chat['has_chat_template']} "
            f"py={ref_chat['has_chat_template']}"
        )
    elif ref_chat["has_chat_template"]:
        # Compare both renders verbatim.
        for key in ("rendered_no_generation_prompt", "rendered_with_generation_prompt"):
            r = rust_chat.get(key, "")
            p = ref_chat.get(key, "")
            if r != p:
                failures.append(
                    f"chat_template.{key} mismatch:\n"
                    f"  rust = {_short(r, 200)!r}\n"
                    f"  py   = {_short(p, 200)!r}"
                )

    passed = not failures
    summary = (
        f"{n} strings × (encode×2, decode×3) "
        f"+ chat_template({'yes' if ref_chat['has_chat_template'] else 'no'})"
    )
    if verbose:
        print(f"  {family}: {'PASS' if passed else 'FAIL'} {summary}", flush=True)
        for f in failures[:5]:
            print(f"    - {f}", flush=True)
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more failures", flush=True)
    return FamilyVerdict(family=family, passed=passed, summary=summary, failures=failures)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--families",
        default=",".join(ALL_FAMILIES),
        help="Comma-separated families to verify. Default: all 5.",
    )
    p.add_argument(
        "--keep-dumps", action="store_true",
        help="Keep the rust dump directory after the run.",
    )
    p.add_argument(
        "--local-stage-dir",
        default="",
        help="If set, use this local pin-script staging directory instead "
             "of pulling from the HF mirror. Used while iterating before "
             "upload.",
    )
    args = p.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    bad = [f for f in families if f not in ALL_FAMILIES]
    if bad:
        print(f"unknown families: {bad}; supported: {ALL_FAMILIES}", file=sys.stderr)
        return 2

    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    fixtures_dir = DUMP_DIR / "fixtures"
    rust_dir = DUMP_DIR / "rust_out"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    rust_dir.mkdir(parents=True, exist_ok=True)

    print("=== ferrotorch-tokenize HF parity (Phase G.2, #1168) ===")
    if args.local_stage_dir:
        print(f"  local-stage-dir: {args.local_stage_dir} (skipping HF download)")
    else:
        print(f"  source mirror:   {TRAJ_REPO}")
    print(f"  families: {families}")

    verdicts: list[FamilyVerdict] = []
    for family in families:
        print(f"\n--- family {family} ---")
        if args.local_stage_dir:
            fixture_dir = stage_local_family(
                Path(args.local_stage_dir), family, fixtures_dir
            )
        else:
            fixture_dir = fetch_family_fixtures(family, fixtures_dir)
        rust_out = rust_dir / family
        rust_out.mkdir(parents=True, exist_ok=True)
        rust_verdict = run_rust_dump(family, fixture_dir, rust_out)
        print(f"  rust verdict: {rust_verdict}")
        v = compare_family(family, fixture_dir, rust_out, verbose=True)
        verdicts.append(v)

    # --- Summary ----------------------------------------------------------
    n_pass = sum(1 for v in verdicts if v.passed)
    print(f"\n=== TOTAL: {n_pass}/{len(verdicts)} families passed ===")
    for v in verdicts:
        marker = "PASS" if v.passed else "FAIL"
        print(f"  {marker} {v.family}: {v.summary}")
        if not v.passed:
            for f in v.failures[:10]:
                print(f"    - {f}")
            if len(v.failures) > 10:
                print(f"    ... and {len(v.failures) - 10} more failures")

    overall = n_pass == len(verdicts)
    print()
    print(f"tokenizer-parity: {'PASS' if overall else 'FAIL'}")

    if args.keep_dumps:
        print(f"\nrust dumps preserved at {rust_dir}")
        print(f"fixtures preserved at  {fixtures_dir}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
