#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for the ferrotorch-mps conformance suite.

Tracking issue: conformance suite for ferrotorch-mps (torch.mps parity).
Reference: torch.mps — Apple Metal Performance Shaders backend, torch == 2.11.0

Output: ``ferrotorch-mps/tests/conformance/fixtures.json``

torch.mps is Apple-Silicon-specific. On Linux/WSL the MPS backend is
unconditionally unavailable. This script therefore exercises the
*device-detection and lifecycle surface* of torch.mps — the same items
ferrotorch-mps exposes — and records the expected Python-side return values.
These become the conformance ground truth for the Rust implementation.

Fixtures recorded per public item:

* ``torch.mps.is_available()``   → bool (always False on Linux)
* ``torch.mps.device_count()``   → int (always 0 on Linux)
* ``torch.device('mps', 0)``     → device type/index metadata
* MpsDevice construction error   → error type name string from torch

Usage:
    python3 scripts/regenerate_mps_fixtures.py

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
except ImportError as exc:
    print(f"ERROR: cannot import torch — {exc}", file=sys.stderr)
    print("Install with: pip install torch==2.11.0", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(
    REPO_ROOT, "ferrotorch-mps", "tests", "conformance", "fixtures.json"
)

# ---------------------------------------------------------------------------
# Fixture collection helpers
# ---------------------------------------------------------------------------


def fixture(op: str, **kwargs) -> dict:
    """Build a single fixture entry."""
    return {"op": op, **kwargs}


fixtures: list[dict] = []

# ---- is_mps_available -------------------------------------------------------
# torch.mps.is_available() returns False on Linux and on any non-Apple-Silicon
# platform. This is the ground truth the Rust side must match.
mps_available = torch.backends.mps.is_available()
fixtures.append(
    fixture(
        "is_mps_available",
        expected=mps_available,
        platform_note="always False on non-Apple platforms; always False in ferrotorch stub",
    )
)

# ---- mps_device_count -------------------------------------------------------
# torch.mps does not expose device_count() directly in all torch versions;
# the canonical form is torch.mps.device_count() (added in 2.0+).
# On non-MPS platforms this returns 0.
try:
    device_count = torch.mps.device_count()
except AttributeError:
    # Fallback for very old torch builds — treat as 0 on non-MPS host.
    device_count = 0

fixtures.append(
    fixture(
        "mps_device_count",
        expected=device_count,
        platform_note="always 0 on non-Apple platforms; always 0 in ferrotorch stub",
    )
)

# ---- MpsDevice::new (construction) ------------------------------------------
# torch.device('mps') raises RuntimeError on non-Apple when you try to *use*
# it for tensor allocation. Construction of torch.device('mps') itself always
# succeeds (it's just a string-plus-index struct), but MpsDevice::new in
# ferrotorch returns DeviceUnavailable immediately — which is the *honest*
# shape: fail fast rather than let the invalid handle propagate.
#
# Record both PyTorch's device-string behaviour and the expected ferrotorch
# error so the Rust conformance test can verify each side.
torch_device_mps_type = torch.device("mps").type  # "mps"
torch_device_mps_index = torch.device("mps").index  # None (default ordinal)
torch_device_mps_0_index = torch.device("mps", 0).index  # 0

fixtures.append(
    fixture(
        "MpsDevice_new",
        torch_device_type=torch_device_mps_type,
        torch_device_default_index=torch_device_mps_index,
        torch_device_ordinal_0_index=torch_device_mps_0_index,
        expected_ferrotorch_error="DeviceUnavailable",
        note="torch.device('mps') construction is always valid; "
        "ferrotorch MpsDevice::new always returns Err(DeviceUnavailable) until #451",
    )
)

# ---- MpsDevice::count (associated fn alias for mps_device_count) ------------
fixtures.append(
    fixture(
        "MpsDevice_count",
        expected=device_count,
        note="Same as mps_device_count(); both must return 0 on non-MPS platforms",
    )
)

# ---- MpsDevice::ordinal -------------------------------------------------------
# MpsDevice cannot be constructed (::new always errors), so we record the
# expected shape: ordinal() should return the usize passed to ::new. This is
# a structural/contract fixture — the Rust test verifies the value via the
# struct field encoding since construction always fails until #451.
fixtures.append(
    fixture(
        "MpsDevice_ordinal",
        ordinal_input=0,
        expected_ordinal=0,
        note="ordinal() round-trips the value from ::new; "
        "test is structural (via Display) since ::new always errors",
    )
)

# ---- init_mps_backend -------------------------------------------------------
# torch.mps has no direct init_backend() function; the closest analog is
# torch.mps.synchronize() which raises on non-MPS or simply initializes the
# runtime context. We record availability as the expected error signal.
fixtures.append(
    fixture(
        "init_mps_backend",
        mps_available_at_fixture_time=mps_available,
        expected_ferrotorch_error="DeviceUnavailable",
        note="ferrotorch init_mps_backend always returns Err(DeviceUnavailable) until #451; "
        "torch has no direct equivalent — synchronize() would raise on non-MPS",
    )
)

# ---- MpsDevice Display ("mps:{ordinal}") ------------------------------------
# torch.device('mps', 3) formats as "device(type='mps', index=3)" in Python.
# ferrotorch's MpsDevice Display produces "mps:{ordinal}".
# Record the torch representation and the expected ferrotorch Display string
# so the conformance test can assert the right format.
for ordinal in [0, 1, 2]:
    td = torch.device("mps", ordinal)
    fixtures.append(
        fixture(
            "MpsDevice_display",
            ordinal=ordinal,
            torch_repr=str(td),
            expected_ferrotorch_display=f"mps:{ordinal}",
            note="ferrotorch Display mirrors torch.mps device ordinal in 'mps:N' form",
        )
    )

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

output = {
    "metadata": {
        "torch_version": torch.__version__,
        "mps_available": mps_available,
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "conformance_note": (
            "torch.mps is Apple-Silicon-specific. On Linux/WSL the backend is "
            "always unavailable. Fixtures record device-detection and lifecycle "
            "behaviour which is meaningful on all platforms."
        ),
    },
    "fixtures": fixtures,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as fh:
    json.dump(output, fh, indent=2)
    fh.write("\n")

print(f"Written {len(fixtures)} fixtures to {OUTPUT_PATH}")
print(f"  torch version : {torch.__version__}")
print(f"  mps_available : {mps_available}")
print(f"  device_count  : {device_count}")
