#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for the ferrotorch-data conformance suite.

Reference: torch == 2.11.0
Tracking issue: #838 (conformance phase — ferrotorch-data)

Output: ``ferrotorch-data/tests/conformance/fixtures/data.json``

Coverage:
  * Dataset variants: VecDataset (via range(N)), TensorDataset, ConcatDataset,
    ChainDataset (via torch.utils.data.ConcatDataset / ChainDataset).
  * Sampler variants: SequentialSampler, RandomSampler, BatchSampler,
    WeightedRandomSampler, DistributedSampler.
  * DataLoader: sequential, shuffle, drop_last, single-worker only
    (multi-worker requires OS process spawning — cascade_skip).
  * collate_fn: default_collate (stack tensors along dim 0).
  * Transforms: Normalize channel-wise.

RNG strategy: torch.manual_seed(42) before all shuffle/random operations.

Usage (WSL):

    python3 scripts/regenerate_data_fixtures.py

Required deps:

    torch==2.11.0
    numpy
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import torch.utils.data as tud

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-data"
    / "tests"
    / "conformance"
    / "fixtures"
    / "data.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(data: list, shape: list[int]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32).reshape(shape)


def _to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().tolist()


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

def gen_vec_dataset() -> list[dict[str, Any]]:
    """
    Simple range(N) dataset — ferrotorch VecDataset<i32> analogue.
    Fixtures: length, individual items by index, OOB error expected.
    """
    n = 10
    data = list(range(n))
    ds = tud.TensorDataset(torch.arange(n, dtype=torch.int64))
    fixtures = []

    # length
    fixtures.append({
        "kind": "vec_dataset",
        "subtest": "len",
        "n": n,
        "expected_len": n,
    })

    # get item
    for idx in [0, 4, 9]:
        fixtures.append({
            "kind": "vec_dataset",
            "subtest": "get",
            "n": n,
            "index": idx,
            "expected_value": data[idx],
        })

    # OOB
    fixtures.append({
        "kind": "vec_dataset",
        "subtest": "get_oob",
        "n": n,
        "index": n,
        "expected_error": "IndexOutOfBounds",
    })

    return fixtures


def gen_tensor_dataset() -> list[dict[str, Any]]:
    """
    TensorDataset with two tensors: x [N, 2], y [N].
    Mirrors torch.utils.data.TensorDataset.
    """
    torch.manual_seed(42)
    n = 5
    x = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
    y = torch.arange(n, dtype=torch.float32)
    ds = tud.TensorDataset(x, y)

    fixtures = []

    # length
    fixtures.append({
        "kind": "tensor_dataset",
        "subtest": "len",
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "expected_len": n,
    })

    # get item
    for idx in [0, 2, 4]:
        x_item, y_item = ds[idx]
        fixtures.append({
            "kind": "tensor_dataset",
            "subtest": "get",
            "index": idx,
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "x_data": _to_list(x),
            "y_data": _to_list(y),
            "expected_x": _to_list(x_item),
            "expected_y": _to_list(y_item),
        })

    return fixtures


def gen_concat_dataset() -> list[dict[str, Any]]:
    """
    ConcatDataset: two VecDatasets of different sizes.
    Mirrors torch.utils.data.ConcatDataset.
    """
    a = tud.TensorDataset(torch.tensor([10, 20, 30], dtype=torch.int64))
    b = tud.TensorDataset(torch.tensor([40, 50], dtype=torch.int64))
    ds = tud.ConcatDataset([a, b])

    expected_items = [10, 20, 30, 40, 50]
    fixtures = []

    fixtures.append({
        "kind": "concat_dataset",
        "subtest": "len",
        "sizes": [3, 2],
        "expected_len": 5,
    })

    for idx in range(5):
        item, = ds[idx]  # TensorDataset returns tuple
        fixtures.append({
            "kind": "concat_dataset",
            "subtest": "get",
            "sizes": [3, 2],
            "index": idx,
            "expected_value": expected_items[idx],
        })

    return fixtures


def gen_chain_dataset() -> list[dict[str, Any]]:
    """
    ChainDataset: two IterableDataset concatenated.
    ferrotorch ChainDataset is iterable + map-style.
    We record the full iteration order.
    """
    # Use a simple range iterable as our reference.
    class RangeIterDS(tud.IterableDataset):
        def __init__(self, values):
            self._values = values
        def __iter__(self):
            return iter(self._values)

    a_vals = [1, 2, 3]
    b_vals = [4, 5]
    a = RangeIterDS(a_vals)
    b = RangeIterDS(b_vals)
    ds = tud.ChainDataset([a, b])

    expected_items = list(ds)
    return [{
        "kind": "chain_dataset",
        "subtest": "iter_order",
        "a_values": a_vals,
        "b_values": b_vals,
        "expected_items": expected_items,
    }]


# ---------------------------------------------------------------------------
# Sampler fixtures
# ---------------------------------------------------------------------------

def gen_sequential_sampler() -> list[dict[str, Any]]:
    n = 8
    sampler = tud.SequentialSampler(range(n))
    indices = list(sampler)
    return [{
        "kind": "sequential_sampler",
        "subtest": "indices",
        "n": n,
        "expected_indices": indices,
    }]


def gen_random_sampler() -> list[dict[str, Any]]:
    n = 10
    torch.manual_seed(42)
    sampler = tud.RandomSampler(range(n), generator=torch.Generator().manual_seed(42))
    indices = list(sampler)

    fixtures = []
    fixtures.append({
        "kind": "random_sampler",
        "subtest": "is_permutation",
        "n": n,
        "seed": 42,
        "indices": indices,
        "note": (
            "ferrotorch uses a different PRNG (xorshift64) — exact values "
            "will differ. Assert only that the output is a permutation of "
            "0..n. The fixture records PyTorch's specific output for reference "
            "but the Rust test asserts permutation-completeness only."
        ),
    })
    return fixtures


def gen_batch_sampler() -> list[dict[str, Any]]:
    fixtures = []

    # Case 1: sequential 10 items, batch_size=3, drop_last=False
    inner = tud.SequentialSampler(range(10))
    bs = tud.BatchSampler(inner, batch_size=3, drop_last=False)
    fixtures.append({
        "kind": "batch_sampler",
        "subtest": "sequential_no_drop",
        "n": 10,
        "batch_size": 3,
        "drop_last": False,
        "expected_batches": [list(b) for b in bs],
    })

    # Case 2: sequential 10 items, batch_size=3, drop_last=True
    inner = tud.SequentialSampler(range(10))
    bs = tud.BatchSampler(inner, batch_size=3, drop_last=True)
    fixtures.append({
        "kind": "batch_sampler",
        "subtest": "sequential_drop_last",
        "n": 10,
        "batch_size": 3,
        "drop_last": True,
        "expected_batches": [list(b) for b in bs],
    })

    # Case 3: exact division, batch_size=5
    inner = tud.SequentialSampler(range(15))
    bs = tud.BatchSampler(inner, batch_size=5, drop_last=False)
    fixtures.append({
        "kind": "batch_sampler",
        "subtest": "exact_division",
        "n": 15,
        "batch_size": 5,
        "drop_last": False,
        "expected_batches": [list(b) for b in bs],
    })

    return fixtures


def gen_weighted_random_sampler() -> list[dict[str, Any]]:
    """
    WeightedRandomSampler: record that heavy-weight index dominates.
    We cannot test exact values due to PRNG differences; instead record
    the weight configuration and assert that the high-weight index
    appears in the majority of drawn samples.
    """
    torch.manual_seed(42)
    weights = torch.tensor([1.0, 1.0, 100.0, 1.0])
    n_samples = 200
    sampler = tud.WeightedRandomSampler(
        weights, num_samples=n_samples, replacement=True,
        generator=torch.Generator().manual_seed(42)
    )
    indices = list(sampler)
    count_heavy = sum(1 for i in indices if i == 2)

    return [{
        "kind": "weighted_random_sampler",
        "subtest": "heavy_bias",
        "weights": weights.tolist(),
        "n_samples": n_samples,
        "heavy_index": 2,
        "heavy_count_reference": count_heavy,
        "note": (
            "ferrotorch uses a different PRNG. Assert heavy_index count > "
            "n_samples * 0.7 (heavy weight is 100× others)."
        ),
    }]


def gen_distributed_sampler() -> list[dict[str, Any]]:
    """
    DistributedSampler: records per-rank indices for num_replicas=3, n=10.
    """
    fixtures = []
    n = 10
    num_replicas = 3
    for rank in range(num_replicas):
        torch.manual_seed(42)
        sampler = tud.DistributedSampler(
            range(n),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=0,
        )
        indices = list(sampler)
        fixtures.append({
            "kind": "distributed_sampler",
            "subtest": "sequential_partition",
            "n": n,
            "num_replicas": num_replicas,
            "rank": rank,
            "expected_indices": indices,
            "expected_per_rank_len": len(indices),
        })

    return fixtures


# ---------------------------------------------------------------------------
# DataLoader fixtures
# ---------------------------------------------------------------------------

def gen_dataloader_sequential() -> list[dict[str, Any]]:
    """
    DataLoader: sequential order, batch_size=3, n=8, no shuffle, no drop_last.
    """
    n = 8
    batch_size = 3
    ds = tud.TensorDataset(torch.arange(n, dtype=torch.float32))
    loader = tud.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    batches = [t[0].tolist() for t in loader]  # TensorDataset yields tuples

    return [{
        "kind": "dataloader",
        "subtest": "sequential",
        "n": n,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "expected_batches": batches,
    }]


def gen_dataloader_drop_last() -> list[dict[str, Any]]:
    n = 10
    batch_size = 3
    ds = tud.TensorDataset(torch.arange(n, dtype=torch.float32))
    loader = tud.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
    batches = [t[0].tolist() for t in loader]

    return [{
        "kind": "dataloader",
        "subtest": "drop_last",
        "n": n,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": True,
        "expected_batches": batches,
        "expected_num_batches": len(batches),
    }]


def gen_dataloader_shuffle() -> list[dict[str, Any]]:
    """
    DataLoader shuffle: record that the full set of samples is covered and
    order differs from sequential. Cannot test exact order due to PRNG diff.
    """
    n = 10
    batch_size = 10
    torch.manual_seed(42)
    ds = tud.TensorDataset(torch.arange(n, dtype=torch.float32))
    loader = tud.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )
    batches = [t[0].tolist() for t in loader]
    all_indices = [int(v) for b in batches for v in b]

    return [{
        "kind": "dataloader",
        "subtest": "shuffle_coverage",
        "n": n,
        "batch_size": batch_size,
        "shuffle": True,
        "note": (
            "ferrotorch uses a different PRNG. Assert that the full epoch "
            "covers all n samples exactly once (permutation completeness)."
        ),
        "expected_sorted_indices": sorted(all_indices),
    }]


def gen_dataloader_num_batches() -> list[dict[str, Any]]:
    """
    DataLoader batch count arithmetic: n/batch_size round-up and round-down.
    """
    cases = [
        (10, 3, False, 4),  # ceil(10/3)
        (10, 3, True, 3),   # floor(10/3)
        (9, 3, False, 3),   # exact
        (9, 3, True, 3),    # exact, drop_last makes no difference
        (1, 5, False, 1),   # single tiny batch
        (0, 5, False, 0),   # empty dataset
    ]
    fixtures = []
    for n, bs, dl, expected in cases:
        fixtures.append({
            "kind": "dataloader",
            "subtest": "num_batches",
            "n": n,
            "batch_size": bs,
            "drop_last": dl,
            "expected_num_batches": expected,
        })
    return fixtures


# ---------------------------------------------------------------------------
# Collate fixtures
# ---------------------------------------------------------------------------

def gen_default_collate() -> list[dict[str, Any]]:
    """
    default_collate: stack list of 1-D tensors along new dim 0.
    Mirrors torch.utils.data.default_collate([t1, t2, ...]).
    """
    fixtures = []

    # Case 1: list of [3] tensors → [2, 3] batch
    torch.manual_seed(42)
    samples = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
    batch = torch.stack(samples, dim=0)
    fixtures.append({
        "kind": "collate",
        "subtest": "stack_1d",
        "samples": [s.tolist() for s in samples],
        "input_shape": list(samples[0].shape),
        "expected_shape": list(batch.shape),
        "expected_data": _to_list(batch),
    })

    # Case 2: list of [2, 2] tensors → [3, 2, 2]
    samples_2d = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
    ]
    batch_2d = torch.stack(samples_2d, dim=0)
    fixtures.append({
        "kind": "collate",
        "subtest": "stack_2d",
        "num_samples": 3,
        "input_shape": [2, 2],
        "expected_shape": list(batch_2d.shape),
        "expected_data": _to_list(batch_2d),
    })

    return fixtures


# ---------------------------------------------------------------------------
# Transform fixtures
# ---------------------------------------------------------------------------

def gen_normalize() -> list[dict[str, Any]]:
    """
    Normalize: channel-wise (input[c] - mean[c]) / std[c].
    Mirrors torchvision.transforms.Normalize semantics on a [C, H, W] tensor.
    (We do not require torchvision — compute directly.)
    """
    import torchvision.transforms.functional as F  # type: ignore

    # Case 1: [2, 3] tensor (2 channels, 3 elements each)
    # Channel 0: [2, 4, 6], mean=4, std=2 => [-1, 0, 1]
    # Channel 1: [10, 20, 30], mean=20, std=10 => [-1, 0, 1]
    t = torch.tensor([[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]])
    mean = [4.0, 20.0]
    std = [2.0, 10.0]
    out = F.normalize(t, mean=mean, std=std)

    fixtures = [{
        "kind": "normalize",
        "subtest": "two_channel",
        "input": t.tolist(),
        "input_shape": list(t.shape),
        "mean": mean,
        "std": std,
        "expected": out.tolist(),
        "expected_shape": list(out.shape),
    }]

    # Case 2: identity (mean=0, std=1)
    t2 = torch.tensor([[1.0, 2.0, 3.0]])
    out2 = F.normalize(t2, mean=[0.0], std=[1.0])
    fixtures.append({
        "kind": "normalize",
        "subtest": "identity",
        "input": t2.tolist(),
        "mean": [0.0],
        "std": [1.0],
        "expected": out2.tolist(),
    })

    return fixtures


def gen_normalize_no_torchvision() -> list[dict[str, Any]]:
    """Fallback: compute Normalize without torchvision."""
    # Case 1: two channels
    t = torch.tensor([[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]])
    mean = torch.tensor([4.0, 20.0]).view(2, 1)
    std = torch.tensor([2.0, 10.0]).view(2, 1)
    out = (t - mean) / std

    fixtures = [{
        "kind": "normalize",
        "subtest": "two_channel",
        "input": t.tolist(),
        "input_shape": list(t.shape),
        "mean": [4.0, 20.0],
        "std": [2.0, 10.0],
        "expected": out.tolist(),
        "expected_shape": list(out.shape),
    }]

    # Case 2: identity
    t2 = torch.tensor([[1.0, 2.0, 3.0]])
    out2 = t2  # (t - 0) / 1 = t
    fixtures.append({
        "kind": "normalize",
        "subtest": "identity",
        "input": t2.tolist(),
        "mean": [0.0],
        "std": [1.0],
        "expected": out2.tolist(),
    })

    return fixtures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)

    all_fixtures: list[dict[str, Any]] = []

    all_fixtures.extend(gen_vec_dataset())
    all_fixtures.extend(gen_tensor_dataset())
    all_fixtures.extend(gen_concat_dataset())
    all_fixtures.extend(gen_chain_dataset())
    all_fixtures.extend(gen_sequential_sampler())
    all_fixtures.extend(gen_random_sampler())
    all_fixtures.extend(gen_batch_sampler())
    all_fixtures.extend(gen_weighted_random_sampler())
    all_fixtures.extend(gen_distributed_sampler())
    all_fixtures.extend(gen_dataloader_sequential())
    all_fixtures.extend(gen_dataloader_drop_last())
    all_fixtures.extend(gen_dataloader_shuffle())
    all_fixtures.extend(gen_dataloader_num_batches())
    all_fixtures.extend(gen_default_collate())

    try:
        all_fixtures.extend(gen_normalize())
    except Exception:
        all_fixtures.extend(gen_normalize_no_torchvision())

    doc = {
        "metadata": {
            "torch_version": torch.__version__,
            "python_executable": sys.executable,
            "python_platform": platform.platform(),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "rng_seed": 42,
            "reference": "torch==2.11.0",
            "note": (
                "Multi-worker DataLoader fixtures are omitted: "
                "multi-worker requires OS process spawning, "
                "single-worker conformance only."
            ),
        },
        "fixtures": all_fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {len(all_fixtures)} fixtures to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
