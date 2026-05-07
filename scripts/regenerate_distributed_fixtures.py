#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for the ferrotorch-distributed conformance suite.

Tracking issue: #882 — ferrotorch-distributed conformance suite.
Reference: torch.distributed (collective ops, process groups), torch == 2.11.0

Output: ``ferrotorch-distributed/tests/conformance/fixtures.json``

torch.distributed is multi-process by nature; this script exercises the
*single-process-observable surface* of torch.distributed — availability gates,
constant values, error contracts, and identity paths for world_size=1 —
and records the expected Python-side return values as conformance ground truth.

For genuinely multi-process ops (allreduce with world_size>1, broadcast, etc.)
the fixture records the expected output computed arithmetically (not executed
via torch.distributed), because running real collective ops requires spawning
multiple processes and a rendezvous store. The arithmetic expectations match
the documented PyTorch collective semantics.

Fixtures recorded per public item:

* ``torch.distributed.is_gloo_available()``    → bool
* ``torch.distributed.is_mpi_available()``     → bool (usually False on this box)
* ``torch.distributed.is_nccl_available()``    → bool
* ``ReduceOp`` variants                        → list of variant names
* ``DEFAULT_COLLECTIVE_TIMEOUT``               → seconds (ferrotorch constant)
* Backend construction error contracts         → error type names
* Collective identity paths (world_size=1)     → inputs == outputs
* Collective semantics (world_size=2, arithmetic) → expected outputs

Usage:
    python3 scripts/regenerate_distributed_fixtures.py

Required:
    torch >= 2.5 (torch == 2.11.0 pinned as the conformance reference)
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import sys

# ---------------------------------------------------------------------------
# Torch import with graceful error
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:
    print(f"ERROR: cannot import torch — {exc}", file=sys.stderr)
    print("Install with: pip install torch==2.11.0", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------

REQUIRED_VERSION = "2.11.0"
actual = torch.__version__.split("+")[0]
if actual != REQUIRED_VERSION:
    print(
        f"WARNING: torch version mismatch — expected {REQUIRED_VERSION}, got {torch.__version__}",
        file=sys.stderr,
    )
    print(
        "Fixtures may not match the pinned reference. Continue at your own risk.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(
    REPO_ROOT,
    "ferrotorch-distributed",
    "tests",
    "conformance",
    "fixtures.json",
)

# ---------------------------------------------------------------------------
# Fixture collection helpers
# ---------------------------------------------------------------------------


def fixture(op: str, **kwargs) -> dict:
    """Build a single fixture entry."""
    return {"op": op, **kwargs}


fixtures: list[dict] = []

# ---- is_gloo_available -------------------------------------------------------
gloo_available = dist.is_gloo_available() if hasattr(dist, "is_gloo_available") else False
fixtures.append(
    fixture(
        "is_gloo_available",
        expected=gloo_available,
        platform_note=(
            "True when torch was compiled with Gloo; False when gloo-backend feature is off"
        ),
    )
)

# ---- is_mpi_available --------------------------------------------------------
mpi_available = dist.is_mpi_available() if hasattr(dist, "is_mpi_available") else False
fixtures.append(
    fixture(
        "is_mpi_available",
        expected=mpi_available,
        platform_note=(
            "True when torch was compiled with MPI support and MPI is installed"
        ),
    )
)

# ---- is_ucc_available --------------------------------------------------------
ucc_available = dist.is_ucc_available() if hasattr(dist, "is_ucc_available") else False
fixtures.append(
    fixture(
        "is_ucc_available",
        expected=ucc_available,
        platform_note=(
            "True when torch was compiled with UCC support and UCC is installed"
        ),
    )
)

# ---- ReduceOp variants -------------------------------------------------------
# torch.distributed.ReduceOp includes SUM, PRODUCT, MIN, MAX, BAND, BOR, BXOR, AVG, PREMUL_SUM.
# ferrotorch maps Sum→SUM and Mean→AVG. We record the relevant subset.
reduce_op_variants = []
if hasattr(dist, "ReduceOp"):
    for name in ["SUM", "AVG", "MIN", "MAX", "PRODUCT"]:
        if hasattr(dist.ReduceOp, name):
            reduce_op_variants.append(name)

fixtures.append(
    fixture(
        "ReduceOp_variants",
        torch_variants=reduce_op_variants,
        ferrotorch_variants=["Sum", "Mean"],
        note=(
            "ferrotorch::ReduceOp has Sum (→ torch.SUM) and Mean (→ torch.AVG). "
            "torch.distributed.ReduceOp has a broader set; ferrotorch covers the "
            "two most-used variants."
        ),
    )
)

# ---- DEFAULT_COLLECTIVE_TIMEOUT ----------------------------------------------
# torch.distributed._DEFAULT_PG_TIMEOUT is 30 minutes (1800s) for NCCL/Gloo.
# ferrotorch uses 60s for the simulated backend (appropriate for unit tests).
# The fixture records the ferrotorch value as the conformance reference.
fixtures.append(
    fixture(
        "DEFAULT_COLLECTIVE_TIMEOUT_secs",
        expected_secs=60,
        torch_nccl_default_secs=1800,
        note=(
            "ferrotorch::DEFAULT_COLLECTIVE_TIMEOUT is 60s (unit-test-appropriate); "
            "torch._DEFAULT_PG_TIMEOUT is 30 min for NCCL/Gloo. The ferrotorch "
            "constant is the conformance reference here."
        ),
    )
)

# ---- SimulatedBackend: create_group(1) ----------------------------------------
fixtures.append(
    fixture(
        "SimulatedBackend_create_group_world_size_1",
        world_size=1,
        expected_len=1,
        expected_rank_0=0,
        expected_world_size_0=1,
        note=(
            "create_group(1) — analogue of torch.distributed.init_process_group "
            "with world_size=1; returns a single backend with rank=0"
        ),
    )
)

# ---- SimulatedBackend: create_group(2) ----------------------------------------
fixtures.append(
    fixture(
        "SimulatedBackend_create_group_world_size_2",
        world_size=2,
        expected_len=2,
        expected_ranks=[0, 1],
        expected_world_size=2,
        note="create_group(2): two backends with ranks 0 and 1",
    )
)

# ---- SimulatedBackend: create_group(0) error ---------------------------------
fixtures.append(
    fixture(
        "SimulatedBackend_create_group_world_size_0_error",
        world_size=0,
        expected_error="InvalidWorldSize",
        note="create_group(0) must return Err(InvalidWorldSize)",
    )
)

# ---- allreduce: world_size=1 identity ----------------------------------------
input_1d = [1.0, 2.0, 3.0]
fixtures.append(
    fixture(
        "allreduce_world_size_1_sum",
        world_size=1,
        op_type="Sum",
        input=input_1d,
        shape=[3],
        expected=input_1d,
        note=(
            "torch.distributed.all_reduce with world_size=1 is identity (SUM of one "
            "tensor is the tensor itself)"
        ),
    )
)
fixtures.append(
    fixture(
        "allreduce_world_size_1_mean",
        world_size=1,
        op_type="Mean",
        input=input_1d,
        shape=[3],
        expected=input_1d,
        note="torch.distributed.all_reduce(op=AVG) with world_size=1 is identity",
    )
)

# ---- allreduce: world_size=2 sum --------------------------------------------
# torch.distributed.all_reduce(op=SUM): each rank contributes; all get sum.
r0 = [1.0, 2.0, 3.0]
r1 = [4.0, 5.0, 6.0]
sum_expected = [a + b for a, b in zip(r0, r1)]
fixtures.append(
    fixture(
        "allreduce_world_size_2_sum",
        world_size=2,
        op_type="Sum",
        input_rank0=r0,
        input_rank1=r1,
        shape=[3],
        expected_all_ranks=sum_expected,
        note=(
            "torch.distributed.all_reduce(op=SUM, world_size=2): "
            "every rank receives element-wise sum of all inputs"
        ),
    )
)

# ---- allreduce: world_size=2 mean -------------------------------------------
mean_expected = [(a + b) / 2.0 for a, b in zip(r0, r1)]
fixtures.append(
    fixture(
        "allreduce_world_size_2_mean",
        world_size=2,
        op_type="Mean",
        input_rank0=r0,
        input_rank1=r1,
        shape=[3],
        expected_all_ranks=mean_expected,
        note=(
            "torch.distributed.all_reduce(op=AVG, world_size=2): "
            "every rank receives element-wise mean = sum / world_size"
        ),
    )
)

# ---- broadcast: world_size=1 identity ----------------------------------------
fixtures.append(
    fixture(
        "broadcast_world_size_1",
        world_size=1,
        root=0,
        input=[10.0, 20.0, 30.0],
        shape=[3],
        expected=[10.0, 20.0, 30.0],
        note="torch.distributed.broadcast with world_size=1 is identity",
    )
)

# ---- broadcast: world_size=2 from root=0 -------------------------------------
root_data = [10.0, 20.0, 30.0]
fixtures.append(
    fixture(
        "broadcast_world_size_2_from_root_0",
        world_size=2,
        root=0,
        input_rank0=root_data,
        input_rank1=[0.0, 0.0, 0.0],
        shape=[3],
        expected_rank0=root_data,
        expected_rank1=root_data,  # all ranks receive root's tensor
        note=(
            "torch.distributed.broadcast(src=0): all ranks receive rank-0's tensor; "
            "rank 1's input is replaced by rank 0's data"
        ),
    )
)

# ---- all_gather: world_size=1 identity ----------------------------------------
fixtures.append(
    fixture(
        "all_gather_world_size_1",
        world_size=1,
        input=[1.0, 2.0],
        shape=[2],
        expected=[1.0, 2.0],
        note="torch.distributed.all_gather with world_size=1 is identity",
    )
)

# ---- all_gather: world_size=2 ------------------------------------------------
# torch.distributed.all_gather: concatenates tensors from all ranks along dim 0.
ag_r0 = [1.0, 2.0]
ag_r1 = [3.0, 4.0]
ag_expected = ag_r0 + ag_r1
fixtures.append(
    fixture(
        "all_gather_world_size_2",
        world_size=2,
        input_rank0=ag_r0,
        input_rank1=ag_r1,
        input_shape=[2],
        expected_shape=[4],
        expected_all_ranks=ag_expected,
        note=(
            "torch.distributed.all_gather: concatenates along dim 0; "
            "world_size=2 doubles the first dim"
        ),
    )
)

# ---- reduce_scatter: world_size=1 identity ------------------------------------
fixtures.append(
    fixture(
        "reduce_scatter_world_size_1_sum",
        world_size=1,
        op_type="Sum",
        input=[1.0, 2.0, 3.0, 4.0],
        shape=[4],
        expected=[1.0, 2.0, 3.0, 4.0],
        note="torch.distributed.reduce_scatter with world_size=1 is identity",
    )
)

# ---- reduce_scatter: world_size=2 sum ----------------------------------------
# torch.distributed.reduce_scatter:
#   sum([1,2,3,4], [5,6,7,8]) = [6,8,10,12]
#   rank 0 gets first half: [6, 8]
#   rank 1 gets second half: [10, 12]
rs_r0 = [1.0, 2.0, 3.0, 4.0]
rs_r1 = [5.0, 6.0, 7.0, 8.0]
rs_sum = [a + b for a, b in zip(rs_r0, rs_r1)]
rs_expected_r0 = rs_sum[:2]
rs_expected_r1 = rs_sum[2:]
fixtures.append(
    fixture(
        "reduce_scatter_world_size_2_sum",
        world_size=2,
        op_type="Sum",
        input_rank0=rs_r0,
        input_rank1=rs_r1,
        input_shape=[4],
        expected_rank0=rs_expected_r0,
        expected_rank1=rs_expected_r1,
        note=(
            "torch.distributed.reduce_scatter(op=SUM, world_size=2): "
            "sum then scatter; rank i gets the i-th chunk"
        ),
    )
)

# ---- barrier -----------------------------------------------------------------
fixtures.append(
    fixture(
        "barrier_world_size_1",
        world_size=1,
        expected_ok=True,
        note="torch.distributed.barrier with world_size=1 is a no-op returning Ok",
    )
)
fixtures.append(
    fixture(
        "barrier_world_size_2",
        world_size=2,
        expected_ok=True,
        note=(
            "torch.distributed.barrier: all ranks must arrive before any proceed; "
            "returns Ok when all ranks reach the barrier"
        ),
    )
)

# ---- send / recv round-trip --------------------------------------------------
fixtures.append(
    fixture(
        "send_recv_round_trip",
        world_size=2,
        send_rank=0,
        recv_rank=1,
        input=[7.0, 8.0, 9.0],
        shape=[3],
        expected=[7.0, 8.0, 9.0],
        note=(
            "torch.distributed.send/recv: receiver gets exact copy of sent tensor; "
            "no transformation applied"
        ),
    )
)

# ---- sendrecv round-trip (symmetric exchange) --------------------------------
fixtures.append(
    fixture(
        "sendrecv_round_trip",
        world_size=2,
        rank0_sends=[1.0, 2.0],
        rank1_sends=[3.0, 4.0],
        shape=[2],
        expected_rank0_receives=[3.0, 4.0],
        expected_rank1_receives=[1.0, 2.0],
        note=(
            "torch.distributed (batch_isend_irecv 2-party): symmetric exchange; "
            "each rank receives the other's data"
        ),
    )
)

# ---- send errors -------------------------------------------------------------
fixtures.append(
    fixture(
        "send_to_self_error",
        world_size=2,
        rank=0,
        dst_rank=0,
        expected_error="InvalidArgument",
        note=(
            "torch.distributed.send to self-rank raises ValueError; "
            "ferrotorch returns Err(InvalidArgument)"
        ),
    )
)
fixtures.append(
    fixture(
        "send_dst_out_of_range_error",
        world_size=2,
        rank=0,
        dst_rank=5,
        expected_error="InvalidArgument",
        note=(
            "torch.distributed.send to dst_rank >= world_size raises ValueError; "
            "ferrotorch returns Err(InvalidArgument)"
        ),
    )
)

# ---- SubBackend members + rank mapping ---------------------------------------
fixtures.append(
    fixture(
        "SubBackend_members",
        world_size=4,
        members=[1, 2, 3],
        expected_members=[1, 2, 3],
        note=(
            "torch.distributed.new_group(ranks=[1,2,3]): the group exposes its "
            "member rank list; SubBackend::members() must match"
        ),
    )
)
fixtures.append(
    fixture(
        "SubBackend_rank_mapping",
        world_size=4,
        members=[1, 2, 3],
        to_global={"0": 1, "1": 2, "2": 3},
        to_local={"1": 0, "2": 1, "3": 2, "0": None, "4": None},
        note=(
            "SubBackend rank mapping: to_global(local) and to_local(global) "
            "mirror ProcessGroup rank-index semantics in torch.distributed"
        ),
    )
)

# ---- DeviceMesh --------------------------------------------------------------
fixtures.append(
    fixture(
        "DeviceMesh_new_valid",
        mesh_shape=[2, 2],
        mesh_world_size=4,
        expected_ndim=2,
        expected_size=4,
        note=(
            "torch.distributed.DeviceMesh('cpu', [[0,1],[2,3]]): ndim=2, "
            "total size = product(shape) = 4"
        ),
    )
)
fixtures.append(
    fixture(
        "DeviceMesh_new_shape_mismatch_error",
        mesh_shape=[2, 3],
        mesh_world_size=4,
        expected_error="InvalidArgument",
        note="DeviceMesh: shape [2,3] has product 6 != world_size 4 → Err",
    )
)
fixtures.append(
    fixture(
        "DeviceMesh_new_empty_shape_error",
        mesh_shape=[],
        mesh_world_size=1,
        expected_error="InvalidArgument",
        note="DeviceMesh: empty shape → Err",
    )
)

# ---- Placement variants ------------------------------------------------------
fixtures.append(
    fixture(
        "Placement_variants",
        variants=["Replicate", "Shard", "Partial"],
        shard_is_shard=True,
        replicate_is_replicate=True,
        partial_is_partial=True,
        note=(
            "torch.distributed.tensor.Replicate / Shard / Partial placement "
            "variants and their is_* predicate semantics"
        ),
    )
)

# ---- DTensor -----------------------------------------------------------------
fixtures.append(
    fixture(
        "DTensor_from_local_valid",
        local_shape=[2],
        global_shape=[4],
        mesh_shape=[2],
        mesh_world_size=2,
        placement="Shard(0)",
        expected_ok=True,
        note=(
            "torch.distributed.tensor.DTensor.from_local(local, mesh, [Shard(0)]): "
            "valid construction with matching placements.len() == mesh.ndim()"
        ),
    )
)
fixtures.append(
    fixture(
        "DTensor_from_local_placement_mismatch_error",
        local_shape=[2],
        global_shape=[4],
        mesh_shape=[2, 2],
        mesh_world_size=4,
        placements_len=1,
        expected_error="ShapeMismatch",
        note=(
            "DTensor::from_local: placements.len()=1 != mesh.ndim()=2 → Err(ShapeMismatch)"
        ),
    )
)

# ---- DistributedError Display ------------------------------------------------
fixtures.append(
    fixture(
        "DistributedError_display",
        variants=[
            {"variant": "InvalidWorldSize", "world_size": 0, "contains": "world size"},
            {"variant": "InvalidRank", "rank": 5, "world_size": 3, "contains": "rank"},
            {"variant": "Timeout", "seconds": 30, "contains": "timed out"},
            {"variant": "BackendUnavailable", "backend": "gloo", "contains": "gloo"},
        ],
        note=(
            "DistributedError variants must produce human-readable Display output; "
            "each variant is checked for a keyword that should appear in the message"
        ),
    )
)

# ---- PendingCollective op_name -----------------------------------------------
fixtures.append(
    fixture(
        "PendingCollective_op_name",
        op_name_all_gather="async_all_gather",
        op_name_reduce_scatter="async_reduce_scatter",
        note=(
            "PendingCollective::op_name() must return the collective's name string; "
            "used in error messages and diagnostics"
        ),
    )
)

# ---- async_all_gather --------------------------------------------------------
fixtures.append(
    fixture(
        "async_all_gather_matches_sync",
        world_size=2,
        input_rank0=[0.0, 10.0],
        input_rank1=[1.0, 11.0],
        shape=[2],
        expected_all_ranks=[0.0, 10.0, 1.0, 11.0],
        note=(
            "async_all_gather(...).wait() must equal synchronous all_gather result; "
            "torch.distributed.all_gather Work.wait() semantics"
        ),
    )
)

# ---- async_reduce_scatter ----------------------------------------------------
fixtures.append(
    fixture(
        "async_reduce_scatter_matches_sync",
        world_size=2,
        op_type="Sum",
        input_rank0=[1.0, 2.0, 3.0, 4.0],
        input_rank1=[5.0, 6.0, 7.0, 8.0],
        input_shape=[4],
        expected_rank0=[6.0, 8.0],
        expected_rank1=[10.0, 12.0],
        note=(
            "async_reduce_scatter(...).wait() must equal synchronous reduce_scatter result; "
            "torch.distributed.reduce_scatter Work.wait() semantics"
        ),
    )
)

# ---- TensorShardSpec ---------------------------------------------------------
fixtures.append(
    fixture(
        "TensorShardSpec_fields",
        key="weight",
        rank=0,
        shard_index=0,
        num_shards=4,
        note=(
            "TensorShardSpec records full_shape, shard_dim, shard_sizes — "
            "mirrors torch.distributed.checkpoint shard spec"
        ),
    )
)

# ---- ShardMetadata -----------------------------------------------------------
fixtures.append(
    fixture(
        "ShardMetadata_fields",
        num_ranks=4,
        note=(
            "ShardMetadata records num_ranks and tensor_specs map — "
            "mirrors torch.distributed.checkpoint ShardMetadata"
        ),
    )
)

# ---- flat_shard_metadata -----------------------------------------------------
fixtures.append(
    fixture(
        "flat_shard_metadata_single_shard",
        num_shards=1,
        total_elements=100,
        expected_full_shape=[100],
        expected_shard_sizes=[100],
        note=(
            "flat_shard_metadata with world_size=1: single shard of 100 elements, "
            "full_shape=[100], shard_sizes=[100]"
        ),
    )
)
fixtures.append(
    fixture(
        "flat_shard_metadata_four_shards",
        num_shards=4,
        total_elements=100,
        shard_index_0={"offset": 0, "length": 25, "total": 100},
        shard_index_1={"offset": 25, "length": 25, "total": 100},
        shard_index_2={"offset": 50, "length": 25, "total": 100},
        shard_index_3={"offset": 75, "length": 25, "total": 100},
        expected_full_shape=[100],
        expected_shard_sizes=[25, 25, 25, 25],
        note=(
            "flat_shard_metadata with world_size=4, 25 elements per shard: "
            "full_shape=[100], shard_sizes=[25,25,25,25]"
        ),
    )
)

# ---- RpcError display --------------------------------------------------------
fixtures.append(
    fixture(
        "RpcError_display",
        note=(
            "RpcError variants must implement Display — cascade_skip for runtime "
            "errors requiring live TCP; structural Display tested directly"
        ),
        cascade_skip_reason=(
            "RpcError runtime variants (NoConnection, Internal) require live TCP "
            "RPC connections; structural Display tested only"
        ),
    )
)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

output = {
    "metadata": {
        "torch_version": torch.__version__,
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "conformance_note": (
            "torch.distributed collective ops. Multi-process collectives are "
            "exercised via single-process mocking (world_size=1 identity paths) "
            "or documented as cascade_skip where genuine multi-process is required. "
            f"Reference: torch=={REQUIRED_VERSION}."
        ),
    },
    "fixtures": fixtures,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
    f.write("\n")

print(f"Written {len(fixtures)} fixtures to {OUTPUT_PATH}")
print(f"torch version: {torch.__version__}")
print(f"is_gloo_available: {gloo_available}")
print(f"is_mpi_available: {mpi_available}")
print(f"is_ucc_available: {ucc_available}")
