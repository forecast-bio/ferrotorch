#!/usr/bin/env python3
"""Pin scikit-learn reference outputs for the ferrotorch-ml sklearn parity
harness to the `ferrotorch/ml-sklearn-parity-v1` HF mirror.

Phase D.3 of real-artifact-driven development (#1159): build deterministic
reference outputs for 5 canonical tabular ops in scikit-learn, freeze the
exact inputs + outputs, and ship one per-config subfolder so the harness
can byte-compare ferrotorch-ml against scikit-learn without re-running
sklearn at verification time.

Configurations (5 total — all of Phase D.3's spec):

  * pca_n4                    — sklearn.decomposition.PCA(n_components=4).fit_transform
                                Input X: np.random.RandomState(42).randn(100, 10), f32
                                Output: [100, 4] projected coords. Sign-flip tolerated
                                per-component (PCs may flip across implementations).
  * standard_scaler           — sklearn.preprocessing.StandardScaler().fit_transform
                                Input X: same as pca_n4
                                Output: [100, 10] zero-mean, unit-variance (biased /n)
  * one_hot_encoder           — sklearn.preprocessing.OneHotEncoder(sparse_output=False)
                                Input: np.array([['a'],['b'],['a'],['c'],['b']])
                                Output: [5, 3] one-hot matrix. Categories alphabetically
                                ordered → 'a'=col 0, 'b'=col 1, 'c'=col 2.
  * kfold_5                   — sklearn.model_selection.KFold(n_splits=5,
                                shuffle=True, random_state=42).split(np.arange(50))
                                Output: 5 (train_idx, test_idx) pairs. PRNG-dependent
                                so SET-equality only (Option B from #1156): each test
                                set has |fold|, union of all test sets == [0, 50).
  * train_test_split_80_20    — sklearn.model_selection.train_test_split(
                                X[100, 10], y[100], test_size=0.2, random_state=42)
                                Output: (X_train[80,10], X_test[20,10], y_train[80],
                                y_test[20]). SET-equality on the split + label
                                consistency (test labels match test X rows).

Per config the pin emits one subfolder containing:

  * meta.json       — config + dtype + shapes + sklearn version
  * input_*.bin     — fixed input(s), f32 LE multi-tensor format
                      (for one_hot_encoder: input_categories.bin as Array2<usize>
                      pre-encoded to {a:0,b:1,c:2}, encoded as f32 for transport)
  * output_*.bin    — sklearn reference output(s), same format

  * for kfold_5:    fold_indices.json — pure JSON, list of (train, test) lists
  * for tts:        split_indices.json — train_indices, test_indices lists +
                                          ground truth labels[100] for the
                                          SET-equality check

Then everything is bundled into a single `bundle.tar` artifact so registry.rs
can checksum a single file. The verify harness pulls individual files via
`hf_hub_download` and does not consume the tar.

Multi-tensor binary layout per .bin (little-endian; same as the dataloader
and optimizer pin scripts):
  [u32 num_tensors]
  per-tensor: [u32 ndim][u32 × ndim shape][f32 × prod(shape)]

Usage:
  python3 scripts/pin_pretrained_ml_fixtures.py \
      [--out-dir /tmp/ferrotorch_ml_fixtures] \
      [--dry-run] [--only kfold_5,...]

Required:
  pip install --user scikit-learn==1.5.2 numpy huggingface_hub
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tarfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from huggingface_hub import HfApi
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HF_REPO_ID = "ferrotorch/ml-sklearn-parity-v1"


# ---------------------------------------------------------------------------
# Configuration matrix — 5 configs total.
# ---------------------------------------------------------------------------


@dataclass
class ConfigSpec:
    name: str
    description: str


SPECS: list[ConfigSpec] = [
    ConfigSpec("pca_n4", "PCA(n_components=4).fit_transform on np.random.randn(100,10)"),
    ConfigSpec("standard_scaler", "StandardScaler().fit_transform on np.random.randn(100,10)"),
    ConfigSpec("one_hot_encoder", "OneHotEncoder(sparse_output=False).fit_transform on 5 cat strings"),
    ConfigSpec("kfold_5", "KFold(5, shuffle=True, random_state=42).split(arange(50))"),
    ConfigSpec("train_test_split_80_20", "train_test_split(X[100,10],y[100],test_size=0.2,rs=42)"),
]


# ---------------------------------------------------------------------------
# Multi-tensor binary format — mirror of dataloader / optimizer pin scripts.
# ---------------------------------------------------------------------------


def dump_multi_tensor_f32(path: Path, tensors: list[np.ndarray]) -> None:
    """Write a `[u32 num_tensors]` + per-tensor `[u32 ndim][u32 shape][f32]`
    little-endian dump."""
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(tensors)))
        for arr in tensors:
            arr32 = np.ascontiguousarray(arr, dtype="<f4")
            shape = list(arr32.shape)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr32.tobytes(order="C"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fixed input dataset — deterministic.
# ---------------------------------------------------------------------------


def make_X_100x10() -> np.ndarray:
    """The reference 100x10 input matrix for PCA / StandardScaler /
    train_test_split. ferrotorch-ml's rust example reads this from the
    pinned input_X.bin file rather than re-generating with rust's
    `rand` (numpy's RandomState is not portable to rust's PRNG).

    `np.random.RandomState(42).randn(100, 10)` is f64 by default; we
    materialise to f32 because the pinned binary format and the rust
    side both use f32.
    """
    rng = np.random.RandomState(42)
    return rng.randn(100, 10).astype(np.float32, copy=True)


def make_y_100() -> np.ndarray:
    """Reference 100-element label vector for train_test_split. Integer
    labels stored as f32 for transport via the multi-tensor format. We
    use i % 4 to keep ranges small + verifiable."""
    return np.array([i % 4 for i in range(100)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-config sklearn invocations.
# ---------------------------------------------------------------------------


def emit_pca_n4(out_dir: Path) -> dict[str, Any]:
    X = make_X_100x10()
    n_components = 4
    pca = PCA(n_components=n_components)
    # sklearn PCA uses f64 internally; cast input to f64 for stability, then
    # cast output back to f32 (this matches what a careful client would do).
    X64 = X.astype(np.float64, copy=True)
    out = pca.fit_transform(X64).astype(np.float32, copy=True)

    if out.shape != (100, n_components):
        raise RuntimeError(f"pca_n4: unexpected output shape {out.shape}")

    dump_multi_tensor_f32(out_dir / "input_X.bin", [X])
    dump_multi_tensor_f32(out_dir / "output_Y.bin", [out])

    meta = {
        "name": "pca_n4",
        "op": "sklearn.decomposition.PCA(n_components=4).fit_transform",
        "n_samples": 100,
        "n_features": 10,
        "n_components": n_components,
        "input_shape": [100, 10],
        "output_shape": [100, n_components],
        "dtype": "float32",
        "input_seed_source": "np.random.RandomState(42).randn(100, 10).astype(f32)",
        "equality_mode": "COSINE_SIM_PER_PC",  # sign-flip tolerated per PC
        "tolerance": {"cosine_sim_min": 0.9999, "max_abs_after_sign_align": 1e-5},
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def emit_standard_scaler(out_dir: Path) -> dict[str, Any]:
    X = make_X_100x10()
    sc = StandardScaler()
    out = sc.fit_transform(X.astype(np.float64, copy=True)).astype(np.float32, copy=True)

    dump_multi_tensor_f32(out_dir / "input_X.bin", [X])
    dump_multi_tensor_f32(out_dir / "output_Y.bin", [out])

    meta = {
        "name": "standard_scaler",
        "op": "sklearn.preprocessing.StandardScaler().fit_transform",
        "n_samples": 100,
        "n_features": 10,
        "input_shape": [100, 10],
        "output_shape": [100, 10],
        "dtype": "float32",
        "input_seed_source": "np.random.RandomState(42).randn(100, 10).astype(f32)",
        "equality_mode": "MAX_ABS",
        "tolerance": {"max_abs": 1e-6},
        "variance_estimator": "biased (/n) — sklearn default == ferrolearn default",
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def emit_one_hot_encoder(out_dir: Path) -> dict[str, Any]:
    # Raw categorical input — 5 samples, 1 column, 3 unique values.
    X_str = np.array([["a"], ["b"], ["a"], ["c"], ["b"]])
    enc = OneHotEncoder(sparse_output=False)
    out = enc.fit_transform(X_str).astype(np.float32, copy=True)

    # Categories are alphabetically sorted by sklearn → ['a','b','c'].
    cats = enc.categories_[0].tolist()
    if cats != ["a", "b", "c"]:
        raise RuntimeError(
            f"one_hot_encoder: expected sklearn categories=['a','b','c'], got {cats}"
        )

    # Pre-encode the string input as Array2<usize> using the same alphabetical
    # ordering so the rust side (which calls ferrolearn's OneHotEncoder on
    # Array2<usize>) gets identical category indices without needing string
    # parsing. We transport as f32 (lossless for the values 0..2).
    encode = {"a": 0, "b": 1, "c": 2}
    X_idx = np.array([[encode[v[0]]] for v in X_str], dtype=np.float32)

    dump_multi_tensor_f32(out_dir / "input_X_indices.bin", [X_idx])
    dump_multi_tensor_f32(out_dir / "output_Y.bin", [out])

    meta = {
        "name": "one_hot_encoder",
        "op": "sklearn.preprocessing.OneHotEncoder(sparse_output=False).fit_transform",
        "n_samples": 5,
        "n_features": 1,
        "categories": cats,
        "input_shape": [5, 1],
        "output_shape": [5, 3],
        "dtype": "float32",
        "input_source": "np.array([['a'],['b'],['a'],['c'],['b']])",
        "category_encoding": {
            "note": "input_X_indices.bin holds Array2<usize>-style indices "
                    "transported as f32. Use {'a':0,'b':1,'c':2}.",
            "map": encode,
        },
        "equality_mode": "EXACT",
        "tolerance": {"max_abs": 0.0, "exact_integer": True},
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def emit_kfold_5(out_dir: Path) -> dict[str, Any]:
    n_samples = 50
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(n_samples)
    folds = []
    for train_idx, test_idx in kf.split(indices):
        folds.append({
            "train": [int(v) for v in train_idx],
            "test": [int(v) for v in test_idx],
        })

    # Sanity: each test fold has 10 elements; union of all test sets is [0, 50).
    union: set[int] = set()
    for f in folds:
        if len(f["test"]) != n_samples // n_splits:
            raise RuntimeError(
                f"kfold_5: sklearn returned test fold of size {len(f['test'])}, "
                f"expected {n_samples // n_splits}"
            )
        for v in f["test"]:
            if v in union:
                raise RuntimeError(f"kfold_5: index {v} appears in two test folds")
            union.add(v)
    if union != set(range(n_samples)):
        raise RuntimeError("kfold_5: union of test folds != [0, 50)")

    # Pin as JSON, not .bin — folds are integer index lists, exact equality
    # only, and SET-comparison doesn't benefit from a multi-tensor format.
    fold_data = {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": 42,
        "folds": folds,
        "equality_mode": "SET",
    }
    (out_dir / "fold_indices.json").write_text(json.dumps(fold_data, indent=2))

    meta = {
        "name": "kfold_5",
        "op": "sklearn.model_selection.KFold(n_splits=5, shuffle=True, "
              "random_state=42).split(arange(50))",
        "n_samples": n_samples,
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": 42,
        "expected_fold_size": n_samples // n_splits,
        "equality_mode": "SET",
        "tolerance": {
            "note": "rust SmallRng != numpy RNG, so shuffle order differs. "
                    "SET-equality: each test fold has exact size; union of "
                    "all test folds == [0, n_samples).",
        },
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def emit_train_test_split_80_20(out_dir: Path) -> dict[str, Any]:
    X = make_X_100x10()
    y = make_y_100()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    if X_train.shape != (80, 10) or X_test.shape != (20, 10):
        raise RuntimeError(
            f"train_test_split: unexpected shapes "
            f"X_train={X_train.shape}, X_test={X_test.shape}"
        )

    # Persist the input dataset so the rust side reads the exact same X / y
    # (lookup by row index would be ambiguous with a different RNG order
    # — both sides must materialise from the same fixed inputs).
    dump_multi_tensor_f32(out_dir / "input_X.bin", [X])
    dump_multi_tensor_f32(out_dir / "input_y.bin", [y])

    # Persist sklearn's split arrays so a curious human can inspect them;
    # the SET-equality harness uses split_indices.json to know which row
    # indices ended up where on the sklearn side.
    dump_multi_tensor_f32(out_dir / "output_X_train.bin", [X_train])
    dump_multi_tensor_f32(out_dir / "output_X_test.bin", [X_test])
    dump_multi_tensor_f32(out_dir / "output_y_train.bin", [y_train])
    dump_multi_tensor_f32(out_dir / "output_y_test.bin", [y_test])

    # Reverse-engineer which row indices sklearn placed in each set so the
    # harness can do SET-equality independent of the row-order each PRNG
    # picks (rust vs numpy).
    y_to_idx_lookup: dict[float, list[int]] = {}
    for i, v in enumerate(y.tolist()):
        y_to_idx_lookup.setdefault(float(v), []).append(i)
    # Match by exact row equality (X is fixed and unique per row).
    sklearn_train_idx: list[int] = []
    sklearn_test_idx: list[int] = []
    for row in X_train:
        match = None
        for i in range(X.shape[0]):
            if np.array_equal(X[i], row):
                match = i
                break
        if match is None:
            raise RuntimeError("train_test_split: failed to reverse-locate train row")
        sklearn_train_idx.append(match)
    for row in X_test:
        match = None
        for i in range(X.shape[0]):
            if np.array_equal(X[i], row):
                match = i
                break
        if match is None:
            raise RuntimeError("train_test_split: failed to reverse-locate test row")
        sklearn_test_idx.append(match)
    # Sanity: disjoint, covers all 100.
    if set(sklearn_train_idx) & set(sklearn_test_idx):
        raise RuntimeError("train_test_split: sklearn train/test indices overlap")
    if set(sklearn_train_idx) | set(sklearn_test_idx) != set(range(100)):
        raise RuntimeError("train_test_split: sklearn indices do not cover [0, 100)")

    (out_dir / "split_indices.json").write_text(json.dumps({
        "n_samples": 100,
        "test_size": 0.2,
        "random_state": 42,
        "n_train": 80,
        "n_test": 20,
        # Recovered sklearn split (used only for inspection — SET-equality
        # checks the *ferrotorch* split against the full [0, 100) coverage,
        # not against sklearn's specific order/identity).
        "sklearn_train_indices": sklearn_train_idx,
        "sklearn_test_indices": sklearn_test_idx,
        "equality_mode": "SET",
    }, indent=2))

    meta = {
        "name": "train_test_split_80_20",
        "op": "sklearn.model_selection.train_test_split(X[100,10], y[100], "
              "test_size=0.2, random_state=42)",
        "n_samples": 100,
        "n_features": 10,
        "test_size": 0.2,
        "random_state": 42,
        "n_train": 80,
        "n_test": 20,
        "equality_mode": "SET",
        "tolerance": {
            "note": "rust SmallRng != numpy RNG: SET-equality only. Sizes must "
                    "match exactly (80/20); union of train+test == [0, 100); "
                    "test labels must match test X rows (label consistency).",
        },
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


EMITTERS = {
    "pca_n4": emit_pca_n4,
    "standard_scaler": emit_standard_scaler,
    "one_hot_encoder": emit_one_hot_encoder,
    "kfold_5": emit_kfold_5,
    "train_test_split_80_20": emit_train_test_split_80_20,
}


# ---------------------------------------------------------------------------
# Bundle + upload.
# ---------------------------------------------------------------------------


def write_readme(out_root: Path, metas: list[dict[str, Any]]) -> None:
    config_lines = []
    for m in metas:
        config_lines.append(
            f"  * `{m['name']}` — {m['op']} "
            f"(equality_mode={m['equality_mode']})"
        )
    readme = textwrap.dedent(f"""
        ---
        license: bsd-3-clause
        tags:
        - test-fixtures
        - sklearn
        - tabular
        ---

        # ferrotorch / ml-sklearn-parity-v1

        scikit-learn reference outputs for ferrotorch-ml's tabular
        operations, generated by running the 5-config matrix on a fixed
        deterministic dataset and snapshotting the inputs + outputs as
        `.bin` (multi-tensor f32) and `.json` (integer indices) files.

        Phase D.3 of real-artifact-driven development (#1159). Companion to:
          * `scripts/pin_pretrained_ml_fixtures.py` (this pin)
          * `scripts/verify_ml_inference.py` (the harness)
          * `ferrotorch-ml/examples/ml_op_dump.rs`
          * `ferrotorch-ml/tests/conformance_sklearn_parity.rs`

        sklearn version: {metas[0]['sklearn_version']}.

        ## Configurations

        {chr(10).join(config_lines)}

        ## Layout

        One subfolder per configuration:

        ```
        <config_name>/
          meta.json
          input_*.bin        # one or more input tensors (f32 LE multi-tensor)
          output_*.bin       # sklearn reference output(s) (f32 LE multi-tensor)
          fold_indices.json  # kfold_5 only — integer fold index lists
          split_indices.json # train_test_split_80_20 only — split indices
        ```

        ## Binary format

        Each `.bin` file is a little-endian multi-tensor dump (same as
        ferrotorch/dataloader-batches-v1 and ferrotorch/optimizer-trajectories-v1):

        ```
        [u32 num_tensors]
        per-tensor:
          [u32 ndim] [u32 × ndim shape] [f32 × prod(shape)]
        ```

        ## Equality semantics

        * `pca_n4` — cosine_sim ≥ 0.9999 PER PRINCIPAL COMPONENT (PCs may
          flip sign across implementations; the harness aligns each PC's
          sign before computing max_abs).
        * `standard_scaler` — max_abs ≤ 1e-6 (essentially exact f32
          arithmetic; sklearn + ferrolearn both use biased variance /n).
        * `one_hot_encoder` — exact integer equality.
        * `kfold_5` — SET-equality. rust's `rand` crate (SmallRng) and
          numpy's PRNG cannot byte-match the shuffle permutation; each
          test fold must have exact size 10, and the union of all test
          folds must equal [0, 50).
        * `train_test_split_80_20` — SET-equality. Sizes must be exactly
          80/20; union of train+test indices == [0, 100); test labels
          must match test X rows (label consistency invariant).

        ## License

        BSD-3-Clause (scikit-learn inherits BSD-3-Clause; the reference
        outputs are deterministic projections of public-domain numpy
        random state, so the BSD-3-Clause notice flows through).
    """).strip()
    (out_root / "README.md").write_text(readme)


def hf_upload(out_root: Path) -> None:
    api = HfApi()
    print(f"\nuploading to https://huggingface.co/{HF_REPO_ID} ...", flush=True)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_root),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="feat: pin sklearn parity fixtures v1 (#1159)",
    )
    print("upload complete.", flush=True)


def build_bundle(out_root: Path) -> Path:
    """Write a single `bundle.tar` so registry.rs can checksum one
    artifact. The verify script downloads individual files via
    `hf_hub_download` and does not consume this tar."""
    tar_path = out_root / "bundle.tar"
    with tarfile.open(tar_path, "w") as tar:
        for sub in sorted(out_root.iterdir()):
            if sub.is_dir():
                tar.add(sub, arcname=sub.name)
    return tar_path


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="/tmp/ferrotorch_ml_fixtures",
        help="Staging directory.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Stage everything locally but do not upload to HF.",
    )
    p.add_argument(
        "--only", default="",
        help="Comma-separated subset of config names to regenerate (debug).",
    )
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    specs = [s for s in SPECS if not only or s.name in only]
    if not specs:
        print("no specs match --only filter", file=sys.stderr)
        return 2

    metas: list[dict[str, Any]] = []
    for spec in specs:
        sub = out_root / spec.name
        sub.mkdir(parents=True, exist_ok=True)
        print(f"\n=== emitting {spec.name} ===", flush=True)
        meta = EMITTERS[spec.name](sub)
        metas.append(meta)
        print(f"  done. files: {sorted(p.name for p in sub.iterdir())}")

    write_readme(out_root, metas)
    bundle_path = build_bundle(out_root)
    bundle_sha = sha256_of(bundle_path)

    if not args.dry_run:
        hf_upload(out_root)

    print("\n=== SUMMARY ===")
    for m in metas:
        print(f"  {m['name']:24s}  equality={m['equality_mode']}")
    print(f"\nlocal stage:   {out_root}")
    print(f"bundle:        {bundle_path}")
    print(f"bundle sha256: {bundle_sha}")
    print(f"hf:            https://huggingface.co/{HF_REPO_ID}")

    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print('  ModelInfo {')
    print('      name: "ml-sklearn-parity-v1",')
    print(
        '      description: "Phase D.3 sklearn parity fixtures: PCA(n=4), '
        'StandardScaler, OneHotEncoder, KFold(5,shuffle,rs=42), '
        'train_test_split(0.2,rs=42). 5 configs over a fixed deterministic '
        'dataset (np.random.RandomState(42).randn(100,10) + i%4 labels). '
        'BSD-3-Clause; real-artifact baseline for ferrotorch-ml vs '
        'scikit-learn (#1159).",'
    )
    print(f'      weights_url: "https://huggingface.co/{HF_REPO_ID}/resolve/main/bundle.tar",')
    print(f'      weights_sha256: "{bundle_sha}",')
    print('      format: WeightsFormat::FerrotorchStateDict,')
    print('      num_parameters: 0,')
    print('  },')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
