#!/usr/bin/env python3
"""Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.7
(FFT, complex tensors, signal processing).

Tracking issue: #769 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/fft.json

Coverage (102 canonical-path surface items per #769):

* Cat A — FFT forwards: fft, ifft, rfft, irfft, fft2, ifft2, rfft2 (via
  fftn/rfftn), irfft2, fftn, ifftn, rfftn, irfftn, fftshift, ifftshift,
  fftfreq, rfftfreq, hfft, ihfft.
* Cat A — Complex tensor: ComplexTensor::{from_re_im, from_real, zeros,
  scalar, from_interleaved, to_interleaved, real, imag, re, im, shape,
  ndim, numel, add, sub, mul, conj, abs, angle, matmul, fft, ifft, fft2,
  ifft2, reshape}.
* Cat A — Signal/window: bartlett, blackman, hann, hanning, hamming,
  kaiser, cosine, exponential, gaussian, general_cosine, general_hamming,
  nuttall, parzen, taylor, tukey (both signal::* and signal::windows::*).
* Cat B — autograd: fft_differentiable, ifft_differentiable,
  rfft_differentiable, irfft_differentiable, fftn_differentiable,
  ifftn_differentiable, rfftn_differentiable, irfftn_differentiable,
  hfft_differentiable, ihfft_differentiable. Each Backward struct (e.g.
  FftBackward) is implicit-coverage via the matching `*_differentiable`
  fixture.

Tolerances (per dispatch table in #769):
  F32_FFT_CPU = 1e-4, F32_FFT_GPU = 1e-3, F64_FFT = 1e-9
  fftshift / ifftshift / window functions are bit-exact on CPU (no FP
  arithmetic in shift; windows generated against ferray-window so we
  apply F64_FFT to absorb the implementation-difference budget).

Edge cases per the dispatch:
  * Even AND odd-length FFT
  * Real vs complex input (rfft/irfft round-trip)
  * Multi-dim FFT (axes=-1, axes=0, axes=(-2,-1))
  * Hermitian symmetry round-trip (ifft(rfft(real)) ≈ real)
  * Window symmetry (w[i] == w[N-1-i])
  * Empty input (FFT of empty → empty) -- documented; ferrotorch rejects
    n=0, so we don't include it as a positive fixture.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_fft_fixtures.py

Required Python deps: torch (with CUDA).
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch  # type: ignore

# ---------------------------------------------------------------------------
# Output path and metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-core"
    / "tests"
    / "conformance"
    / "fixtures"
    / "fft.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

RNG_SEED: int = 0x77F73C09  # 0x77 = phase 2.7
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def to_listf(t: torch.Tensor) -> list[Any]:
    """Materialize a tensor to a CPU Python list of floats with sentinels.

    Complex tensors are converted to interleaved real `[re, im, re, im, ...]`
    with the trailing dim of 2 — matching ferrotorch's complex-as-trailing-2
    convention used by `fft::*`.
    """
    if t.is_complex():
        # Pack into interleaved [re, im] pairs as a real tensor with a
        # trailing dim of 2. `resolve_conj()` materializes any pending
        # conjugation (e.g. from `torch.fft.ihfft` outputs that PyTorch
        # represents lazily as a conjugated view).
        t = torch.view_as_real(t.resolve_conj())
    raw = t.detach().to("cpu").to(torch.float64).reshape(-1).tolist()
    encoded: list[Any] = []
    for v in raw:
        if math.isnan(v):
            encoded.append("NaN")
        elif math.isinf(v):
            encoded.append("Infinity" if v > 0 else "-Infinity")
        else:
            encoded.append(v)
    return encoded


def shape_with_complex_trailing(t: torch.Tensor) -> list[int]:
    """Return the shape ferrotorch sees for a tensor.

    Real tensors keep their shape. Complex tensors are stored with an
    extra trailing dim of 2 (real, imag).
    """
    if t.is_complex():
        return list(t.shape) + [2]
    return list(t.shape)


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "dtypes": DTYPES,
        "devices": DEVICES,
    }


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def _real_input(shape: list[int], dtype: str, device: str, seed: int) -> torch.Tensor:
    """Deterministic real-valued input for FFT."""
    g = torch.Generator(device=device)
    g.manual_seed(RNG_SEED ^ seed)
    return torch.randn(*shape, dtype=torch_dtype(dtype), device=device, generator=g)


def _complex_input(shape: list[int], dtype: str, device: str, seed: int) -> torch.Tensor:
    """Deterministic complex-valued input. PyTorch returns a true complex tensor."""
    g = torch.Generator(device=device)
    g.manual_seed(RNG_SEED ^ seed)
    real = torch.randn(*shape, dtype=torch_dtype(dtype), device=device, generator=g)
    g.manual_seed(RNG_SEED ^ (seed + 1))
    imag = torch.randn(*shape, dtype=torch_dtype(dtype), device=device, generator=g)
    cdtype = torch.complex64 if dtype == "float32" else torch.complex128
    return torch.complex(real, imag).to(cdtype)


# ---------------------------------------------------------------------------
# Cat A — FFT forwards (1-D)
#
# `op` codes encode the API surface element and the arg shape:
#     fft        — input shape `[..., n, 2]` (interleaved complex)
#     ifft       — same
#     rfft       — input shape `[..., n]` (real)
#     irfft      — input shape `[..., n/2+1, 2]` (Hermitian complex)
#     fft2       — input shape `[..., rows, cols, 2]`
#     ifft2      — same
#     fftn       — N-D complex
#     ifftn      — N-D complex
#     rfftn      — N-D real
#     irfftn     — inverse N-D real
#     hfft       — Hermitian complex → real
#     ihfft      — real → Hermitian complex
#     fftshift   — bit-exact data movement
#     ifftshift  — bit-exact data movement
#     fftfreq    — frequency helper (closed-form)
#     rfftfreq   — frequency helper (closed-form)
# ---------------------------------------------------------------------------


# (tag, n, fft_n_arg) — fft_n_arg goes to torch.fft.fft as `n=`. None means
# default. We exercise both even and odd lengths.
FFT_1D_CASES: list[tuple[str, int, int | None]] = [
    ("len_8_default", 8, None),
    ("len_7_default", 7, None),  # odd
    ("len_8_pad_to_16", 8, 16),
    ("len_8_truncate_to_4", 8, 4),
    ("len_16_default", 16, None),
]


def fixture_fft_1d() -> list[dict[str, Any]]:
    """1-D complex FFT (and IFFT) along the last logical axis.

    Input shape [batch, n, 2] (interleaved). Output shape matches.
    """
    out: list[dict[str, Any]] = []
    for op_name in ("fft", "ifft"):
        torch_op = getattr(torch.fft, op_name)
        for tag, n, fft_n_arg in FFT_1D_CASES:
            for device in DEVICES:
                for dtype in DTYPES:
                    # Batch=2 to exercise the loop.
                    inp_complex = _complex_input([2, n], dtype, device, 0x101)
                    fwd = torch_op(inp_complex, n=fft_n_arg)
                    out.append(
                        {
                            "op": op_name,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "n_arg": fft_n_arg,
                            "a_shape": shape_with_complex_trailing(inp_complex),
                            "a_data": to_listf(inp_complex),
                            "out_shape": shape_with_complex_trailing(fwd),
                            "out_values": to_listf(fwd),
                        }
                    )
    return out


def fixture_rfft_1d() -> list[dict[str, Any]]:
    """1-D real-to-complex FFT (rfft) and inverse (irfft)."""
    out: list[dict[str, Any]] = []
    for tag, n, fft_n_arg in FFT_1D_CASES:
        for device in DEVICES:
            for dtype in DTYPES:
                inp_real = _real_input([2, n], dtype, device, 0x102)
                fwd = torch.fft.rfft(inp_real, n=fft_n_arg)
                out.append(
                    {
                        "op": "rfft",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": fft_n_arg,
                        "a_shape": shape_with_complex_trailing(inp_real),
                        "a_data": to_listf(inp_real),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                    }
                )
                # irfft of the rfft round-trip (use forward result as input).
                # Pin output_n to the original length so reconstruction is
                # exact (within tolerance).
                rev = torch.fft.irfft(fwd, n=n)
                out.append(
                    {
                        "op": "irfft",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": n,
                        "a_shape": shape_with_complex_trailing(fwd),
                        "a_data": to_listf(fwd),
                        "out_shape": shape_with_complex_trailing(rev),
                        "out_values": to_listf(rev),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Cat A — FFT forwards (2-D + N-D)
# ---------------------------------------------------------------------------


FFT_2D_CASES: list[tuple[str, int, int]] = [
    ("rows4_cols4", 4, 4),
    ("rows3_cols4", 3, 4),  # rows odd
    ("rows4_cols5", 4, 5),  # cols odd
    ("rows8_cols8", 8, 8),
]


def fixture_fft_2d() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op_name in ("fft2", "ifft2"):
        torch_op = getattr(torch.fft, op_name)
        for tag, rows, cols in FFT_2D_CASES:
            for device in DEVICES:
                for dtype in DTYPES:
                    inp_complex = _complex_input([rows, cols], dtype, device, 0x201)
                    fwd = torch_op(inp_complex)
                    out.append(
                        {
                            "op": op_name,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "rows": rows,
                            "cols": cols,
                            "a_shape": shape_with_complex_trailing(inp_complex),
                            "a_data": to_listf(inp_complex),
                            "out_shape": shape_with_complex_trailing(fwd),
                            "out_values": to_listf(fwd),
                        }
                    )
    return out


# (tag, shape, axes-or-None, s-or-None)
FFTN_CASES: list[tuple[str, list[int], list[int] | None, list[int] | None]] = [
    ("ndim_2_default", [3, 4], None, None),
    ("ndim_3_default", [2, 3, 4], None, None),
    ("ndim_3_axes_neg1", [2, 3, 4], [-1], None),
    ("ndim_3_axes_0", [4, 3, 2], [0], None),
    ("ndim_3_axes_n2_n1", [2, 3, 4], [-2, -1], None),
    ("ndim_2_with_s", [3, 4], None, [4, 4]),
]


def fixture_fftn() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op_name in ("fftn", "ifftn"):
        torch_op = getattr(torch.fft, op_name)
        for tag, shape, axes, s in FFTN_CASES:
            for device in DEVICES:
                for dtype in DTYPES:
                    inp = _complex_input(shape, dtype, device, 0x301)
                    kwargs: dict[str, Any] = {}
                    if axes is not None:
                        kwargs["dim"] = tuple(axes)
                    if s is not None:
                        kwargs["s"] = tuple(s)
                    fwd = torch_op(inp, **kwargs)
                    out.append(
                        {
                            "op": op_name,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "axes": axes,
                            "s": s,
                            "a_shape": shape_with_complex_trailing(inp),
                            "a_data": to_listf(inp),
                            "out_shape": shape_with_complex_trailing(fwd),
                            "out_values": to_listf(fwd),
                        }
                    )
    return out


def fixture_rfftn() -> list[dict[str, Any]]:
    """N-D real FFT and inverse round-trip pair."""
    out: list[dict[str, Any]] = []
    for tag, shape, axes, s in FFTN_CASES:
        for device in DEVICES:
            for dtype in DTYPES:
                inp_real = _real_input(shape, dtype, device, 0x302)
                kwargs: dict[str, Any] = {}
                if axes is not None:
                    kwargs["dim"] = tuple(axes)
                if s is not None:
                    kwargs["s"] = tuple(s)
                fwd = torch.fft.rfftn(inp_real, **kwargs)
                out.append(
                    {
                        "op": "rfftn",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "axes": axes,
                        "s": s,
                        "a_shape": shape_with_complex_trailing(inp_real),
                        "a_data": to_listf(inp_real),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                    }
                )
                # irfftn round-trip: use forward output and pin `s` to
                # the original real-input shape segment.
                back_kwargs: dict[str, Any] = dict(kwargs)
                if axes is not None:
                    back_kwargs["s"] = tuple(shape[i] for i in axes)
                else:
                    back_kwargs["s"] = tuple(shape)
                rev = torch.fft.irfftn(fwd, **back_kwargs)
                out.append(
                    {
                        "op": "irfftn",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "axes": axes,
                        "s": list(back_kwargs["s"]),
                        "a_shape": shape_with_complex_trailing(fwd),
                        "a_data": to_listf(fwd),
                        "out_shape": shape_with_complex_trailing(rev),
                        "out_values": to_listf(rev),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Cat A — Hermitian FFT
# ---------------------------------------------------------------------------


def fixture_hfft() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # hfft: complex Hermitian [n/2+1] -> real [n].  ihfft: real [n] -> complex.
    for tag, n in [("len_8", 8), ("len_7", 7), ("len_16", 16)]:
        for device in DEVICES:
            for dtype in DTYPES:
                # ihfft: real -> Hermitian complex.
                real_inp = _real_input([n], dtype, device, 0x401)
                ih = torch.fft.ihfft(real_inp)
                out.append(
                    {
                        "op": "ihfft",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": None,
                        "a_shape": [n],
                        "a_data": to_listf(real_inp),
                        "out_shape": shape_with_complex_trailing(ih),
                        "out_values": to_listf(ih),
                    }
                )
                # hfft: take the ihfft output and round-trip back, pinning n.
                hb = torch.fft.hfft(ih, n=n)
                out.append(
                    {
                        "op": "hfft",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": n,
                        "a_shape": shape_with_complex_trailing(ih),
                        "a_data": to_listf(ih),
                        "out_shape": [n],
                        "out_values": to_listf(hb),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Cat A — fftshift / ifftshift (bit-exact data movement)
# ---------------------------------------------------------------------------


SHIFT_CASES: list[tuple[str, list[int], list[int] | None]] = [
    ("len_8_all_axes", [8], None),
    ("len_7_all_axes", [7], None),  # odd
    ("shape_2x4_default", [2, 4], None),
    ("shape_2x4_axes_n1", [2, 4], [-1]),
    ("shape_3x5_axes_0", [3, 5], [0]),
    ("shape_2x3x4_default", [2, 3, 4], None),
]


def fixture_fftshift() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op_name in ("fftshift", "ifftshift"):
        torch_op = getattr(torch.fft, op_name)
        for tag, shape, axes in SHIFT_CASES:
            for device in DEVICES:
                for dtype in DTYPES:
                    inp = _real_input(shape, dtype, device, 0x501)
                    kwargs: dict[str, Any] = {}
                    if axes is not None:
                        kwargs["dim"] = tuple(axes)
                    fwd = torch_op(inp, **kwargs)
                    out.append(
                        {
                            "op": op_name,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "axes": axes,
                            "a_shape": shape,
                            "a_data": to_listf(inp),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                        }
                    )
    return out


# ---------------------------------------------------------------------------
# Cat A — fftfreq / rfftfreq (closed-form, bit-exact)
# ---------------------------------------------------------------------------


def fixture_fftfreq() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tag, n, d in [
        ("n8_d1", 8, 1.0),
        ("n7_d1", 7, 1.0),  # odd
        ("n16_d0p1", 16, 0.1),  # custom spacing
    ]:
        # fftfreq / rfftfreq always return CPU f64 in ferrotorch — we still
        # generate one fixture per device for symmetry but only `cpu` is
        # consumed by the test runner.
        f = torch.fft.fftfreq(n, d=d, dtype=torch.float64)
        rf = torch.fft.rfftfreq(n, d=d, dtype=torch.float64)
        out.append(
            {
                "op": "fftfreq",
                "tag": tag,
                "dtype": "float64",
                "device": "cpu",
                "n": n,
                "d": d,
                "out_shape": [n],
                "out_values": to_listf(f),
            }
        )
        out.append(
            {
                "op": "rfftfreq",
                "tag": tag,
                "dtype": "float64",
                "device": "cpu",
                "n": n,
                "d": d,
                "out_shape": [n // 2 + 1],
                "out_values": to_listf(rf),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Cat A — Signal windows
# ---------------------------------------------------------------------------
#
# ferrotorch's `signal::*` and `signal::windows::*` always return CPU f64
# tensors — we only generate CPU fixtures.


def fixture_windows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    M = 16  # window length

    # Windows with no extra parameters.
    simple = [
        ("bartlett", torch.signal.windows.bartlett),
        ("blackman", torch.signal.windows.blackman),
        ("hann", torch.signal.windows.hann),
        ("hamming", torch.signal.windows.hamming),
        ("nuttall", torch.signal.windows.nuttall),
        ("cosine", torch.signal.windows.cosine),
    ]
    for name, fn in simple:
        w = fn(M, sym=True, dtype=torch.float64)
        out.append(
            {
                "op": f"window_{name}",
                "tag": "len_16",
                "dtype": "float64",
                "device": "cpu",
                "m": M,
                "out_shape": [M],
                "out_values": to_listf(w),
            }
        )

    # Kaiser: shape parameter beta. We test 0 (rectangular) and 8.6.
    for beta_tag, beta in [("beta_0", 0.0), ("beta_8p6", 8.6)]:
        w = torch.signal.windows.kaiser(M, beta=beta, sym=True, dtype=torch.float64)
        out.append(
            {
                "op": "window_kaiser",
                "tag": beta_tag,
                "dtype": "float64",
                "device": "cpu",
                "m": M,
                "beta": beta,
                "out_shape": [M],
                "out_values": to_listf(w),
            }
        )

    # Exponential: tau parameter.
    w = torch.signal.windows.exponential(
        M, center=None, tau=1.0, sym=True, dtype=torch.float64
    )
    out.append(
        {
            "op": "window_exponential",
            "tag": "tau_1",
            "dtype": "float64",
            "device": "cpu",
            "m": M,
            "tau": 1.0,
            "out_shape": [M],
            "out_values": to_listf(w),
        }
    )

    # Gaussian: std parameter.
    w = torch.signal.windows.gaussian(M, std=2.0, sym=True, dtype=torch.float64)
    out.append(
        {
            "op": "window_gaussian",
            "tag": "std_2",
            "dtype": "float64",
            "device": "cpu",
            "m": M,
            "std": 2.0,
            "out_shape": [M],
            "out_values": to_listf(w),
        }
    )

    # general_cosine: coefficient list.
    coeffs = [0.5, 0.5]
    w = torch.signal.windows.general_cosine(M, a=coeffs, sym=True, dtype=torch.float64)
    out.append(
        {
            "op": "window_general_cosine",
            "tag": "hann_coeffs",
            "dtype": "float64",
            "device": "cpu",
            "m": M,
            "coeffs": coeffs,
            "out_shape": [M],
            "out_values": to_listf(w),
        }
    )

    # general_hamming: alpha parameter.
    w = torch.signal.windows.general_hamming(M, alpha=0.54, sym=True, dtype=torch.float64)
    out.append(
        {
            "op": "window_general_hamming",
            "tag": "alpha_0p54",
            "dtype": "float64",
            "device": "cpu",
            "m": M,
            "alpha": 0.54,
            "out_shape": [M],
            "out_values": to_listf(w),
        }
    )

    # Windows not in torch.signal.windows in this PyTorch version
    # (parzen, taylor, tukey, hanning). Reference values come from scipy
    # — ferrotorch's `signal::*` calls into ferray-window which mirrors
    # scipy/numpy semantics. We compare against scipy's reference, then
    # the conformance test applies a slightly looser tolerance because
    # ferray-window may differ from scipy at the LSB level.
    try:
        import scipy.signal.windows as scw  # type: ignore
        import numpy as np  # type: ignore

        # parzen
        w_np = scw.parzen(M, sym=True).astype(np.float64)
        out.append(
            {
                "op": "window_parzen",
                "tag": "len_16",
                "dtype": "float64",
                "device": "cpu",
                "m": M,
                "out_shape": [M],
                "out_values": w_np.tolist(),
            }
        )

        # taylor (nbar=4, sll=30, norm=True)
        w_np = scw.taylor(M, nbar=4, sll=30, norm=True, sym=True).astype(np.float64)
        out.append(
            {
                "op": "window_taylor",
                "tag": "nbar4_sll30_norm",
                "dtype": "float64",
                "device": "cpu",
                "m": M,
                "nbar": 4,
                "sll": 30.0,
                "norm": True,
                "out_shape": [M],
                "out_values": w_np.tolist(),
            }
        )

        # tukey (alpha=0.5)
        w_np = scw.tukey(M, alpha=0.5, sym=True).astype(np.float64)
        out.append(
            {
                "op": "window_tukey",
                "tag": "alpha_0p5",
                "dtype": "float64",
                "device": "cpu",
                "m": M,
                "alpha": 0.5,
                "out_shape": [M],
                "out_values": w_np.tolist(),
            }
        )

        # hanning is an alias of hann; emit a separate fixture so the
        # conformance test can pin both surface paths.
        w_np = scw.hann(M, sym=True).astype(np.float64)
        out.append(
            {
                "op": "window_hanning",
                "tag": "len_16",
                "dtype": "float64",
                "device": "cpu",
                "m": M,
                "out_shape": [M],
                "out_values": w_np.tolist(),
            }
        )
    except ImportError as exc:
        raise SystemExit(
            "scipy is required for parzen/taylor/tukey/hanning fixtures "
            "(not exposed in this torch.signal.windows). "
            "Install with: pip install scipy"
        ) from exc

    return out


# ---------------------------------------------------------------------------
# Cat A — Complex tensor ops
# ---------------------------------------------------------------------------
#
# ComplexTensor is structure-of-arrays; the conformance test exercises
# every method by directly constructing real/imag pairs and validating
# against PyTorch's complex op outputs.


def fixture_complex_ops() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Use a 2x3 complex tensor for the algebraic ops.
    a_re = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a_im = [-1.0, 0.5, 2.0, -2.0, 0.0, 1.5]
    b_re = [0.5, 1.5, -1.0, 2.0, 1.0, 3.0]
    b_im = [1.0, 0.0, 1.0, -1.5, 0.5, 2.0]
    shape = [2, 3]
    for dtype in ("float32", "float64"):
        td = torch_dtype(dtype)
        a = torch.complex(
            torch.tensor(a_re, dtype=td).reshape(shape),
            torch.tensor(a_im, dtype=td).reshape(shape),
        )
        b = torch.complex(
            torch.tensor(b_re, dtype=td).reshape(shape),
            torch.tensor(b_im, dtype=td).reshape(shape),
        )
        for op_name, fwd in [
            ("complex_add", a + b),
            ("complex_sub", a - b),
            ("complex_mul", a * b),
            ("complex_conj", torch.conj(a).resolve_conj()),
            ("complex_abs", torch.abs(a)),
            ("complex_angle", torch.angle(a)),
        ]:
            out.append(
                {
                    "op": op_name,
                    "tag": "shape_2x3",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_re": a_re,
                    "a_im": a_im,
                    "b_re": b_re if op_name in ("complex_add", "complex_sub", "complex_mul") else None,
                    "b_im": b_im if op_name in ("complex_add", "complex_sub", "complex_mul") else None,
                    "shape": shape,
                    "out_shape": shape_with_complex_trailing(fwd),
                    "out_values": to_listf(fwd),
                }
            )

    # Complex matmul: 2x3 @ 3x4 = 2x4
    a_re_m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a_im_m = [0.5, -1.0, 2.0, 0.0, 1.0, -0.5]
    b_re_m = [0.0, 1.0, -1.0, 2.0, 0.5, 0.5, 1.0, 0.0, -0.5, 1.5, 2.0, -1.0]
    b_im_m = [1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.5, 0.0, 0.0, 0.5, -1.5, 1.0]
    a_shape_m = [2, 3]
    b_shape_m = [3, 4]
    for dtype in ("float32", "float64"):
        td = torch_dtype(dtype)
        a = torch.complex(
            torch.tensor(a_re_m, dtype=td).reshape(a_shape_m),
            torch.tensor(a_im_m, dtype=td).reshape(a_shape_m),
        )
        b = torch.complex(
            torch.tensor(b_re_m, dtype=td).reshape(b_shape_m),
            torch.tensor(b_im_m, dtype=td).reshape(b_shape_m),
        )
        c = a @ b
        out.append(
            {
                "op": "complex_matmul",
                "tag": "2x3_at_3x4",
                "dtype": dtype,
                "device": "cpu",
                "a_re": a_re_m,
                "a_im": a_im_m,
                "b_re": b_re_m,
                "b_im": b_im_m,
                "a_shape": a_shape_m,
                "b_shape": b_shape_m,
                "out_shape": shape_with_complex_trailing(c),
                "out_values": to_listf(c),
            }
        )

    # Complex 1-D fft (via ComplexTensor::fft, n=None and n=8)
    cplx_re = [1.0, 2.0, 3.0, 4.0]
    cplx_im = [0.5, -1.0, 0.0, 1.5]
    cplx_shape = [4]
    for dtype in ("float32", "float64"):
        td = torch_dtype(dtype)
        a = torch.complex(
            torch.tensor(cplx_re, dtype=td),
            torch.tensor(cplx_im, dtype=td),
        )
        for op_name, n_arg, fwd in [
            ("complex_fft_default", None, torch.fft.fft(a)),
            ("complex_ifft_default", None, torch.fft.ifft(a)),
        ]:
            out.append(
                {
                    "op": op_name,
                    "tag": "len_4",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_re": cplx_re,
                    "a_im": cplx_im,
                    "shape": cplx_shape,
                    "n_arg": n_arg,
                    "out_shape": shape_with_complex_trailing(fwd),
                    "out_values": to_listf(fwd),
                }
            )

    # Complex 2-D fft (ComplexTensor::fft2 / ifft2)
    cplx2_re = list(range(1, 9))  # 1..8
    cplx2_im = [0.0, 0.5, -0.5, 1.0, -1.0, 0.5, 1.5, -0.5]
    cplx2_shape = [2, 4]
    for dtype in ("float32", "float64"):
        td = torch_dtype(dtype)
        a = torch.complex(
            torch.tensor(cplx2_re, dtype=td).reshape(cplx2_shape),
            torch.tensor(cplx2_im, dtype=td).reshape(cplx2_shape),
        )
        for op_name, fwd in [
            ("complex_fft2_default", torch.fft.fft2(a)),
            ("complex_ifft2_default", torch.fft.ifft2(a)),
        ]:
            out.append(
                {
                    "op": op_name,
                    "tag": "shape_2x4",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_re": cplx2_re,
                    "a_im": cplx2_im,
                    "shape": cplx2_shape,
                    "out_shape": shape_with_complex_trailing(fwd),
                    "out_values": to_listf(fwd),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Cat B — autograd: *_differentiable forward + backward grad
# ---------------------------------------------------------------------------
#
# Loss = real-sum over the output (matches the rest of the conformance suite's
# loss convention). For complex outputs torch.fft returns complex; loss
# is `out.real.sum() + out.imag.sum()` (the real-and-imag-sum proxy that
# our `real_sum` reduction in the test runner mirrors when reading
# back ferrotorch's `[..., 2]` interleaved tensor).


def _real_imag_sum_grad(t: torch.Tensor) -> torch.Tensor:
    """Loss = real(t).sum() + imag(t).sum()  for complex t,
    or t.sum() for real t. Returns a 0-d real tensor."""
    if t.is_complex():
        return t.real.sum() + t.imag.sum()
    return t.sum()


def fixture_fft_differentiable() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # 1-D fft / ifft
    for tag, n in [("len_8", 8), ("len_7", 7)]:
        for device in DEVICES:
            for dtype in DTYPES:
                # Complex input requires_grad
                inp_complex = _complex_input([2, n], dtype, device, 0x601)
                inp_g = inp_complex.detach().clone().requires_grad_(True)
                fwd = torch.fft.fft(inp_g)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                grad_a = inp_g.grad
                out.append(
                    {
                        "op": "fft_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": None,
                        "a_shape": shape_with_complex_trailing(inp_complex),
                        "a_data": to_listf(inp_complex),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(grad_a),
                    }
                )

                inp_g = inp_complex.detach().clone().requires_grad_(True)
                fwd = torch.fft.ifft(inp_g)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "ifft_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": None,
                        "a_shape": shape_with_complex_trailing(inp_complex),
                        "a_data": to_listf(inp_complex),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

    # rfft / irfft
    for tag, n in [("len_8", 8), ("len_6", 6)]:
        for device in DEVICES:
            for dtype in DTYPES:
                inp_real = _real_input([2, n], dtype, device, 0x602)
                inp_g = inp_real.detach().clone().requires_grad_(True)
                fwd = torch.fft.rfft(inp_g)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "rfft_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": None,
                        "a_shape": shape_with_complex_trailing(inp_real),
                        "a_data": to_listf(inp_real),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

                # irfft: input is the rfft output (complex Hermitian).
                inp_complex_h = torch.fft.rfft(_real_input([2, n], dtype, device, 0x603))
                inp_g = inp_complex_h.detach().clone().requires_grad_(True)
                fwd = torch.fft.irfft(inp_g, n=n)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "irfft_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": n,
                        "a_shape": shape_with_complex_trailing(inp_complex_h),
                        "a_data": to_listf(inp_complex_h),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

    # fftn / ifftn (use a 2-D input)
    for tag, shape in [("ndim_2_3x4", [3, 4]), ("ndim_2_4x4", [4, 4])]:
        for device in DEVICES:
            for dtype in DTYPES:
                inp_complex = _complex_input(shape, dtype, device, 0x604)
                inp_g = inp_complex.detach().clone().requires_grad_(True)
                fwd = torch.fft.fftn(inp_g)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "fftn_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "axes": None,
                        "s": None,
                        "a_shape": shape_with_complex_trailing(inp_complex),
                        "a_data": to_listf(inp_complex),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

                inp_g = inp_complex.detach().clone().requires_grad_(True)
                fwd = torch.fft.ifftn(inp_g)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "ifftn_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "axes": None,
                        "s": None,
                        "a_shape": shape_with_complex_trailing(inp_complex),
                        "a_data": to_listf(inp_complex),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

    # rfftn / irfftn (use a 2-D real input)
    for tag, shape in [("ndim_2_3x4", [3, 4]), ("ndim_2_4x4", [4, 4])]:
        for device in DEVICES:
            for dtype in DTYPES:
                inp_real = _real_input(shape, dtype, device, 0x605)
                inp_g = inp_real.detach().clone().requires_grad_(True)
                fwd = torch.fft.rfftn(inp_g)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "rfftn_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "axes": None,
                        "s": None,
                        "a_shape": shape_with_complex_trailing(inp_real),
                        "a_data": to_listf(inp_real),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

                # irfftn round-trip
                inp_complex_h = torch.fft.rfftn(
                    _real_input(shape, dtype, device, 0x606)
                )
                inp_g = inp_complex_h.detach().clone().requires_grad_(True)
                fwd = torch.fft.irfftn(inp_g, s=tuple(shape))
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "irfftn_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "axes": None,
                        "s": shape,
                        "a_shape": shape_with_complex_trailing(inp_complex_h),
                        "a_data": to_listf(inp_complex_h),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

    # hfft / ihfft
    for tag, n in [("len_8", 8), ("len_6", 6)]:
        for device in DEVICES:
            for dtype in DTYPES:
                # ihfft input = real
                inp_real = _real_input([n], dtype, device, 0x607)
                inp_g = inp_real.detach().clone().requires_grad_(True)
                fwd = torch.fft.ihfft(inp_g)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "ihfft_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": None,
                        "a_shape": [n],
                        "a_data": to_listf(inp_real),
                        "out_shape": shape_with_complex_trailing(fwd),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )

                # hfft input = Hermitian complex
                inp_complex_h = torch.fft.ihfft(
                    _real_input([n], dtype, device, 0x608)
                )
                inp_g = inp_complex_h.detach().clone().requires_grad_(True)
                fwd = torch.fft.hfft(inp_g, n=n)
                loss = _real_imag_sum_grad(fwd)
                loss.backward()
                out.append(
                    {
                        "op": "hfft_differentiable",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "n_arg": n,
                        "a_shape": shape_with_complex_trailing(inp_complex_h),
                        "a_data": to_listf(inp_complex_h),
                        "out_shape": [n],
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(inp_g.grad),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures.extend(fixture_fft_1d())
    fixtures.extend(fixture_rfft_1d())
    fixtures.extend(fixture_fft_2d())
    fixtures.extend(fixture_fftn())
    fixtures.extend(fixture_rfftn())
    fixtures.extend(fixture_hfft())
    fixtures.extend(fixture_fftshift())
    fixtures.extend(fixture_fftfreq())
    fixtures.extend(fixture_windows())
    fixtures.extend(fixture_complex_ops())
    fixtures.extend(fixture_fft_differentiable())

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
