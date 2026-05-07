#!/usr/bin/env python3
"""Regenerate reference fixtures for ferrotorch-gpu C8.1 lifecycle conformance.

Tracking: C8.1 (GPU lifecycle + infrastructure: allocator, pool, transfer,
stream, memory_guard, device, module_cache).

Output:
    ferrotorch-gpu/tests/conformance/fixtures_lifecycle.json

Coverage:

* allocator.rs — round_size / get_allocation_size pure-arithmetic fixtures.
* pool.rs      — round_len / pool take+return / stream-aware reuse contracts.
* transfer.rs  — H2D + D2H round-trip bit-exact parity (f32, f64, empty, large).
* stream.rs    — StreamPool lifecycle; StreamGuard set/restore; CudaEventWrapper
                 record/sync contracts.
* memory_guard — budget enforcement; pressure level transitions; hook lifecycle.
* device.rs    — GpuDevice construction contracts.
* module_cache — get_or_compile repeated-call identity.

These fixtures use Python/numpy as the ground-truth oracle for the pure-arithmetic
cases (round_size, round_len, H2D+D2H bit-exact parity). The GPU-touching fixtures
(stream lifecycle, module_cache, memory_guard) encode behavioral contracts that are
verified structurally — the fixture records the *expected behavior description*
rather than a numeric value, because the actual computation happens on the CUDA device
at conformance-test time with no Python involvement.

Pin: cudarc 0.19.x (see workspace Cargo.toml).

Usage:
    python3 scripts/regenerate_gpu_lifecycle_fixtures.py

The script requires numpy (for f32/f64 bit-exact round-trip construction) but does
NOT require PyTorch or a CUDA device — all GPU-touching assertions are structural.
"""

from __future__ import annotations

import datetime
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-gpu"
    / "tests"
    / "conformance"
    / "fixtures_lifecycle.json"
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not found; skipping bit-exact f32/f64 validation.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIN_BLOCK_SIZE = 512
SMALL_SIZE = 1 << 20   # 1 MiB
SMALL_BUFFER = 2 << 20  # 2 MiB
MIN_LARGE_ALLOC = 10 << 20  # 10 MiB
LARGE_BUFFER = 20 << 20  # 20 MiB
ROUND_LARGE = 2 << 20   # 2 MiB
ROUND_ELEMENTS = 256


def round_size(size: int) -> int:
    """Mirror of allocator::round_size."""
    if size < MIN_BLOCK_SIZE:
        return MIN_BLOCK_SIZE
    return (size + MIN_BLOCK_SIZE - 1) & ~(MIN_BLOCK_SIZE - 1)


def get_allocation_size(size: int) -> int:
    """Mirror of allocator::get_allocation_size."""
    if size <= SMALL_SIZE:
        return SMALL_BUFFER
    elif size < MIN_LARGE_ALLOC:
        return LARGE_BUFFER
    else:
        return (size + ROUND_LARGE - 1) & ~(ROUND_LARGE - 1)


def round_len(length: int) -> int:
    """Mirror of pool::round_len."""
    if length == 0:
        return 0
    remainder = length % ROUND_ELEMENTS
    if remainder == 0:
        return length
    return length + (ROUND_ELEMENTS - remainder)


def fl32(values: list[float]) -> list[float]:
    """Round-trip through f32 to match GPU precision."""
    if not HAS_NUMPY:
        return values
    arr = np.array(values, dtype=np.float32)
    return arr.tolist()


def fl64(values: list[float]) -> list[float]:
    """Round-trip through f64 (no precision loss)."""
    if not HAS_NUMPY:
        return values
    arr = np.array(values, dtype=np.float64)
    return arr.tolist()


# ---------------------------------------------------------------------------
# Fixture generators
# ---------------------------------------------------------------------------

def allocator_fixtures() -> list[dict[str, Any]]:
    """round_size and get_allocation_size arithmetic fixtures."""
    cases = []

    for size, expected in [
        (0, MIN_BLOCK_SIZE),
        (1, MIN_BLOCK_SIZE),
        (511, MIN_BLOCK_SIZE),
        (512, MIN_BLOCK_SIZE),
        (513, 1024),
        (1024, 1024),
        (1025, 1536),
    ]:
        computed = round_size(size)
        assert computed == expected, f"round_size({size}): {computed} != {expected}"
        cases.append({
            "id": f"alloc_round_size_{size}",
            "module": "allocator",
            "op": "round_size",
            "inputs": {"bytes": size},
            "expected_output": expected,
            "note": f"round_size({size}) == {expected}",
        })

    for size, expected in [
        (512, SMALL_BUFFER),
        (SMALL_SIZE, SMALL_BUFFER),
        (SMALL_SIZE + 1, LARGE_BUFFER),
        (MIN_LARGE_ALLOC - 1, LARGE_BUFFER),
        (MIN_LARGE_ALLOC, MIN_LARGE_ALLOC),
        (MIN_LARGE_ALLOC + 1, MIN_LARGE_ALLOC + ROUND_LARGE),
        (30 << 20, 30 << 20),
    ]:
        computed = get_allocation_size(size)
        assert computed == expected, f"get_allocation_size({size}): {computed} != {expected}"
        cases.append({
            "id": f"alloc_get_alloc_size_{size}",
            "module": "allocator",
            "op": "get_allocation_size",
            "inputs": {"size": size},
            "expected_output": expected,
            "note": f"get_allocation_size({size}) == {expected}",
        })

    return cases


def pool_fixtures() -> list[dict[str, Any]]:
    """round_len and pool behavioral contracts."""
    cases = []

    for length, expected in [
        (0, 0),
        (1, 256),
        (255, 256),
        (256, 256),
        (257, 512),
        (512, 512),
        (1000, 1024),
    ]:
        computed = round_len(length)
        assert computed == expected, f"round_len({length}): {computed} != {expected}"
        cases.append({
            "id": f"pool_round_len_{length}",
            "module": "pool",
            "op": "round_len",
            "inputs": {"len": length},
            "expected_output": expected,
            "note": f"round_len({length}) == {expected}",
        })

    # Behavioral contracts (non-numeric assertions)
    cases += [
        {
            "id": "pool_take_miss_returns_none",
            "module": "pool",
            "op": "pool_take on empty pool",
            "inputs": {"device": 9901, "rounded_len": 256, "dtype": "u64"},
            "expected_output": None,
            "note": "pool_take must return None when no buffer is cached for (device, len, type)",
        },
        {
            "id": "pool_return_then_take",
            "module": "pool",
            "op": "pool_return followed by pool_take",
            "inputs": {"device": 9902, "rounded_len": 256, "value": 12345, "dtype": "u64"},
            "expected_output": 12345,
            "note": "pool_take after pool_return must return the stored value",
        },
        {
            "id": "pool_stream_aware_reject_wrong_stream",
            "module": "pool",
            "op": "pool_take_stream mismatched stream",
            "inputs": {"device": 9903, "alloc_stream": 100, "query_stream": 200},
            "expected_output": None,
            "note": "pool_take_stream returns None when alloc_stream != query_stream",
        },
        {
            "id": "pool_stream_aware_accept_correct_stream",
            "module": "pool",
            "op": "pool_take_stream matching stream",
            "inputs": {"device": 9904, "alloc_stream": 100, "query_stream": 100},
            "expected_output": "present",
            "note": "pool_take_stream returns the value when streams match and no cross-stream use recorded",
        },
        {
            "id": "pool_record_stream_prevents_stream_reuse",
            "module": "pool",
            "op": "record_stream blocks stream-aware take",
            "inputs": {
                "device": 9905,
                "alloc_stream": 300,
                "blocking_stream": 400,
            },
            "expected_stream_aware_output": None,
            "expected_plain_output": "present",
            "note": "record_stream marks cross-stream use; pool_take_stream returns None but plain pool_take still works",
        },
        {
            "id": "pool_empty_cache_device_specific",
            "module": "pool",
            "op": "empty_cache clears only target device",
            "inputs": {"device_to_clear": 9906, "device_to_keep": 9907},
            "note": "empty_cache(device_a) must not remove entries for device_b",
        },
        {
            "id": "pool_empty_cache_all",
            "module": "pool",
            "op": "empty_cache_all clears all devices",
            "inputs": {"devices": [9908, 9909]},
            "note": "empty_cache_all must remove all cached entries",
        },
    ]

    return cases


def transfer_fixtures() -> list[dict[str, Any]]:
    """H2D + D2H round-trip and alloc_zeros fixtures."""
    cases = []

    # f32 round-trip (bit-exact)
    f32_data = fl32([1.0, 2.0, 3.0, 4.0, 5.0])
    cases.append({
        "id": "transfer_h2d_d2h_f32",
        "module": "transfer",
        "op": "cpu_to_gpu + gpu_to_cpu",
        "inputs": {"dtype": "f32", "data": f32_data, "len": len(f32_data)},
        "expected_output": f32_data,
        "tolerance": 0.0,
        "note": "H2D + D2H round-trip is bit-exact for f32 via cudarc clone_htod/clone_dtoh",
    })

    # f64 round-trip (bit-exact)
    f64_data = fl64([1.0, -2.5, 0.0, 1e300, -1e300])
    cases.append({
        "id": "transfer_h2d_d2h_f64",
        "module": "transfer",
        "op": "cpu_to_gpu + gpu_to_cpu",
        "inputs": {"dtype": "f64", "data": f64_data, "len": len(f64_data)},
        "expected_output": f64_data,
        "tolerance": 0.0,
        "note": "H2D + D2H round-trip is bit-exact for f64",
    })

    # Empty transfer
    cases.append({
        "id": "transfer_h2d_d2h_empty",
        "module": "transfer",
        "op": "cpu_to_gpu + gpu_to_cpu empty",
        "inputs": {"dtype": "f32", "data": [], "len": 0},
        "expected_output": [],
        "tolerance": 0.0,
        "note": "Empty H2D + D2H round-trip returns an empty Vec",
    })

    # Large transfer
    n = 1_000_000
    large_data = fl32([float(i) for i in range(min(n, 1000))])  # fixture stores first 1000
    cases.append({
        "id": "transfer_h2d_d2h_large_first_1000",
        "module": "transfer",
        "op": "cpu_to_gpu + gpu_to_cpu large",
        "inputs": {
            "dtype": "f32",
            "n": n,
            "pattern": "ascending_integers",
            "first_1000": large_data,
        },
        "expected_first_1000": large_data,
        "tolerance": 0.0,
        "note": f"Large transfer of {n} f32 elements; first 1000 values verified",
    })

    # alloc_zeros contracts
    cases.append({
        "id": "transfer_alloc_zeros_f32",
        "module": "transfer",
        "op": "alloc_zeros_f32",
        "inputs": {"len": 1024},
        "expected_output": "all_zero",
        "note": "alloc_zeros_f32(1024) must return a buffer where every element is 0.0f32",
    })

    cases.append({
        "id": "transfer_alloc_zeros_f64",
        "module": "transfer",
        "op": "alloc_zeros_f64",
        "inputs": {"len": 512},
        "expected_output": "all_zero",
        "note": "alloc_zeros_f64(512) must return a buffer where every element is 0.0f64",
    })

    cases.append({
        "id": "transfer_pool_reuse_zeros",
        "module": "transfer",
        "op": "alloc_zeros_f32 pool reuse",
        "inputs": {"len": 512},
        "expected_output": "all_zero",
        "note": "Pool-hit alloc_zeros_f32 must still be all-zero (memset_zeros in pool-hit path)",
    })

    # Pinned round-trip
    pinned_data = fl32([10.0, 20.0, 30.0])
    cases.append({
        "id": "transfer_pinned_round_trip",
        "module": "transfer",
        "op": "cpu_to_gpu_pinned + gpu_to_cpu",
        "inputs": {"dtype": "f32", "data": pinned_data},
        "expected_output": pinned_data,
        "tolerance": 0.0,
        "note": "Pinned H2D uses DMA; bit-exact round-trip expected",
    })

    # Device mismatch rejected
    cases.append({
        "id": "transfer_device_mismatch",
        "module": "transfer",
        "op": "gpu_to_cpu device mismatch",
        "inputs": {"buffer_device": 99, "query_device": 0},
        "expected_error": "DeviceMismatch",
        "note": "gpu_to_cpu must return GpuError::DeviceMismatch when ordinals differ",
    })

    return cases


def allocator_integration_fixtures() -> list[dict[str, Any]]:
    """CudaAllocator behavioral contracts (CUDA-device-required; structural)."""
    return [
        {
            "id": "alloc_starts_at_zero",
            "module": "allocator",
            "op": "memory_allocated at creation",
            "inputs": {"device": 0},
            "expected_allocated_bytes": 0,
            "expected_peak_bytes": 0,
            "note": "Freshly created CudaAllocator must have 0 allocated and 0 peak bytes",
        },
        {
            "id": "alloc_zeros_increases_bytes",
            "module": "allocator",
            "op": "alloc_zeros f32 256",
            "inputs": {"count": 256, "dtype": "f32", "expected_bytes": 256 * 4},
            "expected_allocated_bytes": 256 * 4,
            "note": "alloc_zeros(256 f32) must set memory_allocated to 256*4=1024 bytes",
        },
        {
            "id": "alloc_free_decreases_bytes",
            "module": "allocator",
            "op": "alloc_zeros f32 128 + free",
            "inputs": {"count": 128, "dtype": "f32", "expected_bytes": 128 * 4},
            "expected_allocated_after_alloc": 128 * 4,
            "expected_allocated_after_free": 0,
            "note": "free() must reduce memory_allocated back to 0",
        },
        {
            "id": "alloc_peak_tracks_max",
            "module": "allocator",
            "op": "two allocs; peak stays after partial free",
            "inputs": {"counts": [100, 200], "dtype": "f32"},
            "note": "max_memory_allocated must not decrease when memory is freed",
        },
        {
            "id": "alloc_reset_peak",
            "module": "allocator",
            "op": "reset_peak_stats after free",
            "inputs": {"count": 512, "dtype": "f32"},
            "note": "After free + reset_peak_stats, max_memory_allocated must equal current (0)",
        },
        {
            "id": "alloc_copy_tracks_bytes",
            "module": "allocator",
            "op": "alloc_copy f64 slice",
            "inputs": {"data": [1.0, 2.0, 3.0, 4.0], "dtype": "f64"},
            "expected_allocated_bytes": 4 * 8,
            "note": "alloc_copy must track 4*sizeof(f64)=32 bytes",
        },
        {
            "id": "alloc_empty_buffer",
            "module": "allocator",
            "op": "alloc_zeros 0 elements",
            "inputs": {"count": 0, "dtype": "f32"},
            "expected_allocated_bytes": 0,
            "note": "Zero-element alloc must not change memory_allocated",
        },
        {
            "id": "alloc_cache_find_after_insert_free",
            "module": "allocator",
            "op": "cache_insert + cache_free + cache_find",
            "inputs": {"stream": 1, "size": 2048, "driver_size": 4096, "ptr": 4096},
            "note": "cache_insert then cache_free then cache_find must return a hit (1 hit, 1 miss)",
        },
        {
            "id": "alloc_empty_cache_clears_free_pool",
            "module": "allocator",
            "op": "empty_cache clears free_block_count",
            "inputs": {},
            "note": "empty_cache() must set free_block_count to 0",
        },
    ]


def stream_fixtures() -> list[dict[str, Any]]:
    """Stream lifecycle behavioral contracts."""
    return [
        {
            "id": "stream_pool_lazy_init",
            "module": "stream",
            "op": "StreamPool::pool_size after first get_stream",
            "inputs": {"device_ordinal": 0},
            "expected_pool_size_min": 1,
            "expected_pool_size_max": 8,
            "note": "StreamPool must lazily create streams; pool_size must be in [1, 8] after first access",
        },
        {
            "id": "stream_pool_round_robin",
            "module": "stream",
            "op": "StreamPool round-robin wrap",
            "inputs": {"device_ordinal": 0},
            "note": "After pool_size get_stream calls, the next call must return the same Arc as the first",
        },
        {
            "id": "stream_pool_invalid_device",
            "module": "stream",
            "op": "StreamPool::get_stream ordinal=MAX_DEVICES+1",
            "inputs": {"device_ordinal": 9999},
            "expected_error": "InvalidDevice",
            "note": "get_stream with ordinal >= MAX_DEVICES (64) must return GpuError::InvalidDevice",
        },
        {
            "id": "stream_guard_set_restore",
            "module": "stream",
            "op": "StreamGuard set + drop restores previous",
            "inputs": {},
            "note": "StreamGuard::new sets the current stream; drop restores the previous stream",
        },
        {
            "id": "stream_guard_clears_when_no_previous",
            "module": "stream",
            "op": "StreamGuard drop with no previous clears",
            "inputs": {},
            "note": "When no previous stream existed, StreamGuard drop calls clear_current_stream",
        },
        {
            "id": "stream_event_record_sync",
            "module": "stream",
            "op": "CudaEventWrapper record + synchronize",
            "inputs": {},
            "note": "Record on default stream with no pending work; synchronize must complete immediately",
        },
        {
            "id": "stream_event_query_after_sync",
            "module": "stream",
            "op": "CudaEventWrapper::query after synchronize",
            "inputs": {},
            "expected_complete": True,
            "note": "query() after synchronize() must return Ok(true)",
        },
        {
            "id": "stream_event_elapsed_us_nonneg",
            "module": "stream",
            "op": "CudaEventWrapper::elapsed_us",
            "inputs": {},
            "expected_min_elapsed_us": 0,
            "note": "elapsed_us between two sequential events must be >= 0",
        },
        {
            "id": "stream_priority_range_invariant",
            "module": "stream",
            "op": "get_stream_priority_range",
            "inputs": {"device_ordinal": 0},
            "note": "get_stream_priority_range must return (least, greatest) with greatest <= least (CUDA convention: lower int = higher priority)",
        },
        {
            "id": "stream_priority_all_three_levels",
            "module": "stream",
            "op": "new_stream_with_priority High/Normal/Low",
            "inputs": {},
            "note": "Creating streams at all three priority levels must succeed and produce distinct Arcs",
        },
        {
            "id": "stream_priority_pool_populates",
            "module": "stream",
            "op": "StreamPool::get_priority_stream",
            "inputs": {"device_ordinal": 0},
            "note": "get_priority_stream for High and Low must populate the priority pool (size > 0)",
        },
    ]


def memory_guard_fixtures() -> list[dict[str, Any]]:
    """MemoryGuard behavioral contracts."""
    return [
        {
            "id": "guard_zero_alloc_zero_bytes",
            "module": "memory_guard",
            "op": "new allocator stats at zero",
            "inputs": {},
            "expected_used_bytes": 0,
            "note": "Freshly built MemoryGuard must have stats().used_bytes == 0",
        },
        {
            "id": "guard_alloc_increases_used",
            "module": "memory_guard",
            "op": "safe_alloc f32 128",
            "inputs": {"count": 128, "dtype": "f32", "expected_bytes": 128 * 4},
            "expected_used_bytes": 128 * 4,
            "note": "safe_alloc(128 f32) must increase stats().used_bytes by 128*4=512",
        },
        {
            "id": "guard_free_decreases_used",
            "module": "memory_guard",
            "op": "safe_alloc + free f32 64",
            "inputs": {"count": 64, "dtype": "f32"},
            "expected_used_after_free": 0,
            "note": "free() must set stats().used_bytes back to 0",
        },
        {
            "id": "guard_budget_enforced",
            "module": "memory_guard",
            "op": "safe_alloc over budget",
            "inputs": {"budget_bytes": 1024, "alloc_count": 1000, "dtype": "f32"},
            "expected_error": "BudgetExceeded",
            "note": "safe_alloc requesting more than budget must return GpuError::BudgetExceeded",
        },
        {
            "id": "guard_pressure_none_when_no_budget",
            "module": "memory_guard",
            "op": "pressure_level with budget=0",
            "inputs": {"budget_bytes": 0},
            "expected_pressure": "None",
            "note": "pressure_level must return PressureLevel::None when budget is 0 (unlimited)",
        },
        {
            "id": "guard_pressure_critical_at_100pct",
            "module": "memory_guard",
            "op": "pressure_level at full usage",
            "inputs": {
                "budget_bytes": 1024,
                "used_bytes_after_alloc": 1024,
            },
            "expected_pressure": "Critical",
            "note": "pressure_level must return Critical when used_bytes >= budget_bytes",
        },
        {
            "id": "guard_hook_fires_before_budget_error",
            "module": "memory_guard",
            "op": "register_hook + safe_alloc_with_hooks",
            "inputs": {
                "budget_bytes": 512,
                "alloc_count": 200,
                "dtype": "f32",
                "hook_name": "free_1kib",
                "hook_frees_bytes": 1024,
            },
            "note": "A hook that frees enough memory must allow an otherwise-over-budget alloc to succeed via safe_alloc_with_hooks",
        },
        {
            "id": "guard_remove_hook",
            "module": "memory_guard",
            "op": "register_hook + remove_hook",
            "inputs": {"hook_name": "test_hook"},
            "expected_remove_result": True,
            "note": "remove_hook returns true when a hook with the given name was found and removed",
        },
        {
            "id": "guard_set_budget_and_check",
            "module": "memory_guard",
            "op": "set_budget + budget()",
            "inputs": {"budget_bytes": 1048576},
            "expected_budget": 1048576,
            "note": "set_budget(N) followed by budget() must return N",
        },
        {
            "id": "guard_set_oom_policy",
            "module": "memory_guard",
            "op": "set_oom_policy(RetryAfterFree)",
            "inputs": {},
            "note": "set_oom_policy must accept OomPolicy variants without error",
        },
        {
            "id": "guard_guarded_device_memory_info",
            "module": "memory_guard",
            "op": "MemoryGuardedDevice::memory_info",
            "inputs": {},
            "note": "memory_info() must return Ok((free, total)) with total > 0 on a real GPU",
        },
    ]


def device_fixtures() -> list[dict[str, Any]]:
    """GpuDevice lifecycle contracts."""
    return [
        {
            "id": "device_new_ordinal_zero",
            "module": "device",
            "op": "GpuDevice::new(0)",
            "inputs": {"ordinal": 0},
            "expected_ordinal": 0,
            "note": "GpuDevice::new(0) must create a device with ordinal() == 0",
        },
        {
            "id": "device_stream_returns_arc",
            "module": "device",
            "op": "GpuDevice::stream()",
            "inputs": {},
            "note": "device.stream() must return an Arc<CudaStream> without panicking",
        },
        {
            "id": "device_default_stream_is_stable",
            "module": "device",
            "op": "GpuDevice::default_stream() called twice",
            "inputs": {},
            "note": "Two calls to default_stream() must return the same Arc (ptr equality)",
        },
        {
            "id": "device_invalid_ordinal_errors",
            "module": "device",
            "op": "GpuDevice::new(9999)",
            "inputs": {"ordinal": 9999},
            "expected_error": "Driver",
            "note": "new() with an out-of-range ordinal must return Err (CUDA driver error)",
        },
    ]


def module_cache_fixtures() -> list[dict[str, Any]]:
    """module_cache::get_or_compile contracts."""
    return [
        {
            "id": "module_cache_repeated_calls_identity",
            "module": "module_cache",
            "op": "get_or_compile x2 via gpu_add",
            "inputs": {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "dtype": "f32",
            },
            "expected_output": [5.0, 7.0, 9.0],
            "tolerance": 1e-6,
            "note": "Two gpu_add calls exercise get_or_compile; both must produce identical correct results",
        },
        {
            "id": "module_cache_second_call_faster",
            "module": "module_cache",
            "op": "timing: 2nd get_or_compile < 1st",
            "inputs": {
                "kernel": "mul_kernel",
                "n": 1024,
                "dtype": "f32",
            },
            "note": "Second call skips PTX compilation; structural timing test (no strict ratio — logs for manual inspection)",
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_fixtures: list[dict[str, Any]] = []
    all_fixtures += allocator_fixtures()
    all_fixtures += pool_fixtures()
    all_fixtures += transfer_fixtures()
    all_fixtures += allocator_integration_fixtures()
    all_fixtures += stream_fixtures()
    all_fixtures += memory_guard_fixtures()
    all_fixtures += device_fixtures()
    all_fixtures += module_cache_fixtures()

    output = {
        "version": "cudarc-0.19.x",
        "generated_by": "scripts/regenerate_gpu_lifecycle_fixtures.py",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "description": (
            "Reference fixtures for ferrotorch-gpu C8.1 lifecycle conformance tests. "
            "Arithmetic fixtures are validated against Python reference implementations. "
            "GPU-touching fixtures encode behavioral contracts verified structurally at "
            "conformance-test time against the real CUDA device."
        ),
        "fixture_count": len(all_fixtures),
        "fixtures": all_fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(all_fixtures)} fixtures to {FIXTURE_PATH}")
    print("Fixture counts by module:")
    by_module: dict[str, int] = {}
    for fx in all_fixtures:
        mod = fx.get("module", "unknown")
        by_module[mod] = by_module.get(mod, 0) + 1
    for mod, count in sorted(by_module.items()):
        print(f"  {mod}: {count}")


if __name__ == "__main__":
    main()
