#!/usr/bin/env python3
"""Pin a torch.distributions reference fixture set to the
`ferrotorch/distributions-parity-v1` HF mirror.

Phase G.1 of real-artifact-driven development (#1167): for every
canonical (distribution, config) tuple, run `torch.distributions.<Dist>`
under `torch.manual_seed(42)` and snapshot:

  * `<dist>_params.json`        — the parameters as JSON (loc, scale, ...)
  * `<dist>_sample.bin`         — `[N, *event_shape]` f32 samples
                                  (provenance — Monte Carlo reference;
                                  byte-comparison across PRNGs is NOT
                                  meaningful, so the verifier compares
                                  *moments* only)
  * `<dist>_test_points.bin`    — fixed `[M, *event_shape]` test points
                                  spanning the support (or valid one-hot
                                  for discretes)
  * `<dist>_log_prob.bin`       — `[M]` reference log_prob at those points
  * `<dist>_entropy.bin`        — `[1]` reference scalar (or shape-vec) entropy
  * `<dist>_ref_moments.json`   — the sample mean / sample variance of the
                                  torch reference (so the verifier can
                                  read torch's MC noise floor rather than
                                  re-running torch)

And for canonical KL pairs:

  * `kl_<dist1>_<dist2>_p_params.json`
  * `kl_<dist1>_<dist2>_q_params.json`
  * `kl_<dist1>_<dist2>.bin`    — `[1]` reference KL divergence

Binary format (little-endian, single-tensor):

```
[u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]
```

Usage:
  python3 scripts/pin_pretrained_distributions_fixtures.py \
      [--out-dir /tmp/ferrotorch_distributions_fixtures] \
      [--dry-run] [--only normal_standard,beta_25]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tarfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributions as D


HF_REPO_ID = "ferrotorch/distributions-parity-v1"
SEED = 42
N_SAMPLES = 10_000
TEST_POINTS_DEFAULT = 100


# ---------------------------------------------------------------------------
# Binary format
# ---------------------------------------------------------------------------


def dump_single_tensor_f32(path: Path, arr: np.ndarray) -> None:
    """Write `[u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]`."""
    arr32 = np.ascontiguousarray(arr, dtype="<f4")
    shape = list(arr32.shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr32.tobytes(order="C"))


def read_single_tensor_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    off = 0
    (ndim,) = struct.unpack_from("<I", raw, off)
    off += 4
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    numel = 1
    for s in shape:
        numel *= int(s)
    arr = np.frombuffer(raw, dtype="<f4", count=numel, offset=off).reshape(shape)
    return arr.astype(np.float32, copy=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DistSpec: one fixture config + the closure that builds the torch distribution.
# ---------------------------------------------------------------------------


@dataclass
class DistSpec:
    """One distribution config to pin."""

    name: str
    family: str                       # ferrotorch class name, e.g. "Normal"
    params: dict[str, Any]            # primitive-typed config (lists / floats)
    factory: Callable[[], D.Distribution]
    # Fixed test points; if None, derived from sample percentiles in [5, 95]%.
    test_points: list[Any] | None = None
    # If True the distribution is discrete and test points are integer-valued.
    discrete: bool = False
    # If True we skip entropy (some distributions lack closed-form entropy).
    skip_entropy: bool = False
    # Event shape (default: scalar []).
    event_shape: list[int] = field(default_factory=list)
    # Optional entropy override (torch.Tensor). Used when the factory's
    # distribution doesn't expose `.entropy()` directly (e.g.
    # `TransformedDistribution` raises NotImplementedError in torch) but a
    # mathematically equivalent reference *can* be computed. The pin script
    # writes the override to entropy.bin instead of calling `dist.entropy()`.
    entropy_override: Callable[[], torch.Tensor] | None = None


def _ld(*xs: float) -> torch.Tensor:
    """Build an f32 tensor with shape `[len(xs)]`."""
    return torch.tensor(xs, dtype=torch.float32)


def _s(x: float) -> torch.Tensor:
    """Build an f32 scalar tensor."""
    return torch.tensor(x, dtype=torch.float32)


def _make_specs() -> list[DistSpec]:
    """Build the canonical config matrix. Confined to distributions that
    actually exist in ferrotorch-distributions (per the public surface
    inventory in lib.rs); torch-only families are skipped here so the
    fixtures stay 1:1 with what ferrotorch can verify."""
    specs: list[DistSpec] = []

    # ----- Univariate continuous -----
    specs.append(DistSpec(
        name="normal_standard",
        family="Normal",
        params={"loc": 0.0, "scale": 1.0},
        factory=lambda: D.Normal(_s(0.0), _s(1.0)),
        test_points=np.linspace(-4.0, 4.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="normal_shifted",
        family="Normal",
        params={"loc": 2.0, "scale": 0.5},
        factory=lambda: D.Normal(_s(2.0), _s(0.5)),
        test_points=np.linspace(0.0, 4.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="beta_25",
        family="Beta",
        params={"concentration1": 2.0, "concentration0": 5.0},
        factory=lambda: D.Beta(_s(2.0), _s(5.0)),
        test_points=np.linspace(0.01, 0.99, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="gamma_21",
        family="Gamma",
        params={"concentration": 2.0, "rate": 1.0},
        factory=lambda: D.Gamma(_s(2.0), _s(1.0)),
        test_points=np.linspace(0.1, 10.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="cauchy_standard",
        family="Cauchy",
        params={"loc": 0.0, "scale": 1.0},
        factory=lambda: D.Cauchy(_s(0.0), _s(1.0)),
        test_points=np.linspace(-5.0, 5.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="exponential_1p5",
        family="Exponential",
        params={"rate": 1.5},
        factory=lambda: D.Exponential(_s(1.5)),
        test_points=np.linspace(0.01, 5.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="uniform_neg2_3",
        family="Uniform",
        params={"low": -2.0, "high": 3.0},
        factory=lambda: D.Uniform(_s(-2.0), _s(3.0)),
        test_points=np.linspace(-1.9, 2.9, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="lognormal_0_p5",
        family="LogNormal",
        params={"loc": 0.0, "scale": 0.5},
        factory=lambda: D.LogNormal(_s(0.0), _s(0.5)),
        test_points=np.linspace(0.1, 5.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="laplace_0_1",
        family="Laplace",
        params={"loc": 0.0, "scale": 1.0},
        factory=lambda: D.Laplace(_s(0.0), _s(1.0)),
        test_points=np.linspace(-4.0, 4.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="halfnormal_1",
        family="HalfNormal",
        params={"scale": 1.0},
        factory=lambda: D.HalfNormal(_s(1.0)),
        test_points=np.linspace(0.01, 4.0, TEST_POINTS_DEFAULT).tolist(),
    ))
    specs.append(DistSpec(
        name="studentt_df5",
        family="StudentT",
        params={"df": 5.0, "loc": 0.0, "scale": 1.0},
        factory=lambda: D.StudentT(_s(5.0), _s(0.0), _s(1.0)),
        test_points=np.linspace(-4.0, 4.0, TEST_POINTS_DEFAULT).tolist(),
    ))

    # ----- Univariate discrete -----
    specs.append(DistSpec(
        name="bernoulli_p3",
        family="Bernoulli",
        params={"probs": 0.3},
        factory=lambda: D.Bernoulli(_s(0.3)),
        test_points=[0.0, 1.0],
        discrete=True,
    ))
    specs.append(DistSpec(
        name="poisson_3",
        family="Poisson",
        params={"rate": 3.0},
        factory=lambda: D.Poisson(_s(3.0)),
        test_points=[float(k) for k in range(0, 15)],
        discrete=True,
    ))
    specs.append(DistSpec(
        name="categorical_k4",
        family="Categorical",
        params={"probs": [0.1, 0.3, 0.4, 0.2]},
        factory=lambda: D.Categorical(probs=_ld(0.1, 0.3, 0.4, 0.2)),
        test_points=[0.0, 1.0, 2.0, 3.0],
        discrete=True,
    ))

    # ----- Multivariate -----
    specs.append(DistSpec(
        name="dirichlet_k4",
        family="Dirichlet",
        params={"concentration": [1.0, 2.0, 3.0, 4.0]},
        factory=lambda: D.Dirichlet(_ld(1.0, 2.0, 3.0, 4.0)),
        # 5 test points: each row a valid simplex vector summing to 1.
        test_points=[
            [0.25, 0.25, 0.25, 0.25],
            [0.10, 0.20, 0.30, 0.40],
            [0.40, 0.30, 0.20, 0.10],
            [0.05, 0.45, 0.40, 0.10],
            [0.01, 0.01, 0.49, 0.49],
        ],
        event_shape=[4],
    ))
    # 3-D MultivariateNormal with a small SPD cov.
    mvn_loc = [0.0, 0.0, 0.0]
    mvn_cov = [
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]
    specs.append(DistSpec(
        name="mvn_3d",
        family="MultivariateNormal",
        params={"loc": mvn_loc, "covariance_matrix": mvn_cov},
        factory=lambda: D.MultivariateNormal(
            torch.tensor(mvn_loc, dtype=torch.float32),
            covariance_matrix=torch.tensor(mvn_cov, dtype=torch.float32),
        ),
        test_points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        event_shape=[3],
    ))

    # ----- Transformed (#1109) -----
    # TransformedDistribution(Normal(0, 1), [AffineTransform(loc=2, scale=3)])
    # is mathematically equivalent to Normal(2, 3) for sample / log_prob /
    # entropy. We pin against the actual `D.TransformedDistribution` for
    # sample + log_prob (torch implements both), and use the equivalent
    # Normal(2, 3).entropy() as the analytical reference for entropy.bin —
    # torch raises NotImplementedError on `td.entropy()` directly, but the
    # identity H(Y) = H(X) + E[log|det J|] gives a closed-form result for
    # this chain (log|det J| = log|3|, constant), so we ship a real
    # discriminating reference rather than skipping entropy.
    def _tn_factory() -> D.Distribution:
        from torch.distributions.transforms import AffineTransform
        base = D.Normal(_s(0.0), _s(1.0))
        affine = AffineTransform(loc=_s(2.0), scale=_s(3.0))
        return D.TransformedDistribution(base, [affine])

    def _tn_entropy_ref() -> torch.Tensor:
        # entropy(TD(Normal(0,1), Affine(2,3))) = entropy(Normal(2, 3))
        return D.Normal(_s(2.0), _s(3.0)).entropy()

    specs.append(DistSpec(
        name="transformed_normal_affine",
        family="TransformedDistribution",
        params={
            "base": {"family": "Normal", "loc": 0.0, "scale": 1.0},
            "transforms": [{"family": "Affine", "loc": 2.0, "scale": 3.0}],
        },
        factory=_tn_factory,
        test_points=np.linspace(-4.0, 8.0, TEST_POINTS_DEFAULT).tolist(),
        entropy_override=_tn_entropy_ref,
    ))

    # ----- Multinomial -----
    # Multinomial test points must sum to total_count (no MC noise on the
    # event-axis sum). Pick 4 valid count vectors.
    mn_probs = [0.2, 0.3, 0.5]
    specs.append(DistSpec(
        name="multinomial_k3_n20",
        family="Multinomial",
        params={"total_count": 20, "probs": mn_probs},
        factory=lambda: D.Multinomial(20, probs=_ld(*mn_probs)),
        test_points=[
            [4.0, 6.0, 10.0],
            [4.0, 4.0, 12.0],
            [6.0, 6.0, 8.0],
            [2.0, 8.0, 10.0],
        ],
        event_shape=[3],
        discrete=True,
    ))

    return specs


# KL pairs (P, Q) where ferrotorch has the closed-form formula. Pairs are
# stored as (name, p_factory, q_factory, p_params, q_params).
@dataclass
class KlSpec:
    name: str
    p_params: dict[str, Any]
    q_params: dict[str, Any]
    p_factory: Callable[[], D.Distribution]
    q_factory: Callable[[], D.Distribution]


def _make_kl_specs() -> list[KlSpec]:
    specs: list[KlSpec] = []
    specs.append(KlSpec(
        name="kl_normal_normal",
        p_params={"loc": 0.0, "scale": 1.0},
        q_params={"loc": 1.0, "scale": 2.0},
        p_factory=lambda: D.Normal(_s(0.0), _s(1.0)),
        q_factory=lambda: D.Normal(_s(1.0), _s(2.0)),
    ))
    specs.append(KlSpec(
        name="kl_bernoulli_bernoulli",
        p_params={"probs": 0.3},
        q_params={"probs": 0.5},
        p_factory=lambda: D.Bernoulli(_s(0.3)),
        q_factory=lambda: D.Bernoulli(_s(0.5)),
    ))
    specs.append(KlSpec(
        name="kl_uniform_uniform",
        p_params={"low": -1.0, "high": 1.0},
        q_params={"low": -2.0, "high": 2.0},
        p_factory=lambda: D.Uniform(_s(-1.0), _s(1.0)),
        q_factory=lambda: D.Uniform(_s(-2.0), _s(2.0)),
    ))
    specs.append(KlSpec(
        name="kl_categorical_categorical",
        p_params={"probs": [0.1, 0.3, 0.4, 0.2]},
        q_params={"probs": [0.25, 0.25, 0.25, 0.25]},
        p_factory=lambda: D.Categorical(probs=_ld(0.1, 0.3, 0.4, 0.2)),
        q_factory=lambda: D.Categorical(probs=_ld(0.25, 0.25, 0.25, 0.25)),
    ))
    specs.append(KlSpec(
        name="kl_laplace_laplace",
        p_params={"loc": 0.0, "scale": 1.0},
        q_params={"loc": 0.5, "scale": 2.0},
        p_factory=lambda: D.Laplace(_s(0.0), _s(1.0)),
        q_factory=lambda: D.Laplace(_s(0.5), _s(2.0)),
    ))
    specs.append(KlSpec(
        name="kl_exponential_exponential",
        p_params={"rate": 1.5},
        q_params={"rate": 1.0},
        p_factory=lambda: D.Exponential(_s(1.5)),
        q_factory=lambda: D.Exponential(_s(1.0)),
    ))
    specs.append(KlSpec(
        name="kl_gamma_gamma",
        p_params={"concentration": 2.0, "rate": 1.0},
        q_params={"concentration": 3.0, "rate": 2.0},
        p_factory=lambda: D.Gamma(_s(2.0), _s(1.0)),
        q_factory=lambda: D.Gamma(_s(3.0), _s(2.0)),
    ))
    specs.append(KlSpec(
        name="kl_poisson_poisson",
        p_params={"rate": 3.0},
        q_params={"rate": 5.0},
        p_factory=lambda: D.Poisson(_s(3.0)),
        q_factory=lambda: D.Poisson(_s(5.0)),
    ))
    return specs


# ---------------------------------------------------------------------------
# Generation per spec.
# ---------------------------------------------------------------------------


def _flatten_sample_for_moments(sample: torch.Tensor, event_shape: list[int]) -> np.ndarray:
    """Flatten leading sample dims into one axis; keep event_shape intact.

    For scalar event shapes the result is `[N]`. For event_shape=[K] the
    result is `[N, K]` and moments are computed along axis 0.
    """
    if not event_shape:
        return sample.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = 1
    for s in sample.shape[: sample.ndim - len(event_shape)]:
        n *= int(s)
    return sample.detach().cpu().numpy().astype(np.float32).reshape((n, *event_shape))


def generate_spec(spec: DistSpec, out_dir: Path) -> dict[str, Any]:
    """Pin one DistSpec to `<out_dir>/<spec.name>/`."""
    sub = out_dir / spec.name
    sub.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    dist = spec.factory()

    # ----- Sample N points. -----
    sample = dist.sample(torch.Size([N_SAMPLES]))
    sample_np = _flatten_sample_for_moments(sample, spec.event_shape)
    # Compute moments along the sample axis (axis 0).
    sample_mean = sample_np.astype(np.float64).mean(axis=0)
    sample_var = sample_np.astype(np.float64).var(axis=0, ddof=0)

    # Persist full sample tensor for provenance (compare-moments policy).
    dump_single_tensor_f32(sub / "sample.bin", sample_np)

    # ----- Test points. -----
    if spec.test_points is None:
        raise RuntimeError(f"{spec.name}: test_points must be supplied explicitly")
    test_arr = np.asarray(spec.test_points, dtype=np.float32)
    if spec.event_shape:
        # event_shape=[K]: test_points is `[M, K]`.
        if test_arr.ndim != 1 + len(spec.event_shape):
            raise RuntimeError(
                f"{spec.name}: test_points ndim {test_arr.ndim} doesn't match "
                f"1 + len(event_shape) {1 + len(spec.event_shape)}"
            )
    else:
        if test_arr.ndim != 1:
            raise RuntimeError(
                f"{spec.name}: scalar-event test_points must be 1-D, got "
                f"{test_arr.shape}"
            )
    dump_single_tensor_f32(sub / "test_points.bin", test_arr)

    # ----- log_prob. -----
    tp_torch = torch.tensor(test_arr, dtype=torch.float32)
    with torch.no_grad():
        log_prob = dist.log_prob(tp_torch).detach().cpu().numpy().astype(np.float32)
    dump_single_tensor_f32(sub / "log_prob.bin", log_prob)

    # ----- entropy. -----
    entropy_val: float | None = None
    if not spec.skip_entropy:
        try:
            with torch.no_grad():
                if spec.entropy_override is not None:
                    ent = spec.entropy_override()
                else:
                    ent = dist.entropy()
            ent_np = ent.detach().cpu().numpy().astype(np.float32)
            # Most univariate cases yield a scalar; some multivariates yield
            # a [B]-shaped tensor. Always store as a tensor so the verifier
            # can read it generically.
            if ent_np.ndim == 0:
                ent_np = ent_np.reshape((1,))
            dump_single_tensor_f32(sub / "entropy.bin", ent_np)
            entropy_val = float(ent_np.reshape(-1)[0])
        except NotImplementedError:
            # Some distributions in torch (e.g. Multinomial) don't expose
            # entropy.
            spec.skip_entropy = True

    # ----- Moments + params manifest. -----
    moments = {
        "sample_mean": sample_mean.reshape(-1).tolist(),
        "sample_var": sample_var.reshape(-1).tolist(),
        "n_samples": int(N_SAMPLES),
        "seed": SEED,
        "event_shape": list(spec.event_shape),
    }
    (sub / "ref_moments.json").write_text(json.dumps(moments, indent=2))

    meta = {
        "name": spec.name,
        "family": spec.family,
        "params": spec.params,
        "event_shape": list(spec.event_shape),
        "discrete": spec.discrete,
        "n_samples": int(N_SAMPLES),
        "seed": SEED,
        "test_points_shape": list(test_arr.shape),
        "log_prob_shape": list(log_prob.shape),
        "entropy_available": not spec.skip_entropy,
        "entropy_scalar": entropy_val,
        "torch_version": torch.__version__,
        "format": (
            "Each .bin file is `[u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]`"
            " little-endian. Single tensor per file (unlike the multi-tensor"
            " optimizer pin)."
        ),
    }
    (sub / "params.json").write_text(json.dumps(meta, indent=2))
    print(
        f"  {spec.name:<26s} family={spec.family} "
        f"event={spec.event_shape} mean={sample_mean.reshape(-1)[0]:+.4f} "
        f"var={sample_var.reshape(-1)[0]:.4f} "
        f"log_prob[0]={log_prob.reshape(-1)[0]:+.6f} "
        f"entropy={entropy_val if entropy_val is not None else 'n/a'}"
    )
    return meta


def generate_kl_spec(spec: KlSpec, out_dir: Path) -> dict[str, Any]:
    sub = out_dir / spec.name
    sub.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    p = spec.p_factory()
    q = spec.q_factory()
    kl = D.kl_divergence(p, q).detach().cpu().numpy().astype(np.float32)
    if kl.ndim == 0:
        kl = kl.reshape((1,))
    dump_single_tensor_f32(sub / "kl.bin", kl)
    meta = {
        "name": spec.name,
        "p_params": spec.p_params,
        "q_params": spec.q_params,
        "kl_scalar": float(kl.reshape(-1)[0]),
        "torch_version": torch.__version__,
        "format": "kl.bin holds `[1]` (or broadcast batch shape) f32 KL divergence",
    }
    (sub / "params.json").write_text(json.dumps(meta, indent=2))
    print(f"  {spec.name:<26s} KL = {float(kl.reshape(-1)[0]):.6f}")
    return meta


# ---------------------------------------------------------------------------
# Bundle + upload.
# ---------------------------------------------------------------------------


def write_readme(out_root: Path, metas: list[dict[str, Any]], kl_metas: list[dict[str, Any]]) -> None:
    dist_lines = []
    for m in metas:
        param_str = ", ".join(f"{k}={v}" for k, v in m["params"].items())
        dist_lines.append(f"  * `{m['name']}` — `{m['family']}({param_str})`")
    kl_lines = []
    for m in kl_metas:
        kl_lines.append(f"  * `{m['name']}` — KL = `{m['kl_scalar']:.6f}`")
    readme = textwrap.dedent(f"""
        ---
        license: apache-2.0
        tags:
        - test-fixtures
        - distributions
        - pytorch
        ---

        # ferrotorch / distributions-parity-v1

        Reference fixtures for ferrotorch-distributions's parity vs
        `torch.distributions`, generated under `torch.manual_seed({SEED})`
        with `N={N_SAMPLES}` samples per config plus fixed test points
        for log_prob / entropy / KL divergence.

        Phase G.1 of real-artifact-driven development (#1167). Companion
        to:
          * `scripts/pin_pretrained_distributions_fixtures.py` (this pin)
          * `scripts/verify_distributions_inference.py` (the harness)
          * `ferrotorch-distributions/examples/distributions_dump.rs`
          * `ferrotorch-distributions/tests/conformance_torch_parity.rs`

        ## Why moment-based sample comparison

        ferrotorch's tensor PRNG (`ferrotorch_core::creation::rand` /
        `randn`) is a time-seeded xorshift, not `torch.Generator`'s
        Philox. Sample sequences cannot byte-match across the two PRNGs.
        The harness therefore compares *sample moments* (mean, variance)
        with a Monte-Carlo noise budget at N=10000:
          * sample mean: max_abs <= 0.05
          * sample var:  max_abs <= 0.1
        Analytic outputs (`log_prob`, `entropy`, KL) are deterministic
        and use tight tolerances (max_abs <= 1e-4).

        ## Configurations

        ### Distributions
        {chr(10).join(dist_lines)}

        ### KL pairs
        {chr(10).join(kl_lines)}

        ## Layout

        One subfolder per config:

        ```
        <config_name>/
          params.json
          test_points.bin     # fixed test points where log_prob is evaluated
          sample.bin          # [N, *event_shape] torch reference samples
          log_prob.bin        # [M] reference log_prob at test_points
          entropy.bin         # [1 or B] reference entropy (some skip)
          ref_moments.json    # sample mean + variance of the torch sample
        ```

        For KL configs:
        ```
        <kl_config_name>/
          params.json
          kl.bin              # reference KL divergence
        ```

        ## Binary format

        ```
        [u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]
        ```

        Little-endian single-tensor.

        ## License

        Apache 2.0. Synthetic fixtures generated by this repo's pin
        script; no upstream weights / data.
    """).strip()
    (out_root / "README.md").write_text(readme)


def build_bundle(out_root: Path) -> Path:
    tar_path = out_root / "bundle.tar"
    with tarfile.open(tar_path, "w") as tar:
        for sub in sorted(out_root.iterdir()):
            if sub.is_dir():
                tar.add(sub, arcname=sub.name)
    return tar_path


def hf_upload(out_root: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\nuploading to https://huggingface.co/{HF_REPO_ID} ...", flush=True)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_root),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="feat: pin distributions-parity fixtures v1 (#1167)",
    )
    print("upload complete.", flush=True)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="/tmp/ferrotorch_distributions_fixtures",
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
    specs = _make_specs()
    kl_specs = _make_kl_specs()
    if only:
        specs = [s for s in specs if s.name in only]
        kl_specs = [s for s in kl_specs if s.name in only]
    if not specs and not kl_specs:
        print(f"no specs match --only filter {sorted(only)!r}", file=sys.stderr)
        return 2

    print(f"=== Distribution specs ({len(specs)}) ===")
    metas: list[dict[str, Any]] = []
    for spec in specs:
        metas.append(generate_spec(spec, out_root))

    print(f"\n=== KL specs ({len(kl_specs)}) ===")
    kl_metas: list[dict[str, Any]] = []
    for ks in kl_specs:
        kl_metas.append(generate_kl_spec(ks, out_root))

    write_readme(out_root, metas, kl_metas)
    bundle_path = build_bundle(out_root)
    bundle_sha = sha256_of(bundle_path)

    if not args.dry_run:
        hf_upload(out_root)

    print("\n=== SUMMARY ===")
    print(f"  distributions: {len(metas)}")
    print(f"  KL pairs:      {len(kl_metas)}")
    print(f"\nlocal stage:  {out_root}")
    print(f"bundle:       {bundle_path}")
    print(f"bundle sha256: {bundle_sha}")
    print(f"hf:           https://huggingface.co/{HF_REPO_ID}")

    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print('  ModelInfo {')
    print('      name: "distributions-parity-v1",')
    print('      description: "Phase G.1 torch.distributions parity fixtures: ~17 canonical distribution configs (Normal, Beta, Gamma, Cauchy, Dirichlet, Exponential, Bernoulli, Categorical, Uniform, LogNormal, Multinomial, Poisson, StudentT, MultivariateNormal, HalfNormal, Laplace) plus 8 KL pairs (Normal-Normal, Bernoulli-Bernoulli, Uniform-Uniform, Categorical-Categorical, Laplace-Laplace, Exponential-Exponential, Gamma-Gamma, Poisson-Poisson). Reference samples (N=10000 with torch.manual_seed(42)), log_prob at fixed test points, entropy, and KL divergence from torch.distributions. Apache 2.0; real-artifact baseline for ferrotorch-distributions parity vs torch.distributions (#1167).",')
    print(f'      weights_url: "https://huggingface.co/{HF_REPO_ID}/resolve/main/bundle.tar",')
    print(f'      weights_sha256: "{bundle_sha}",')
    print('      format: WeightsFormat::FerrotorchStateDict,')
    print('      num_parameters: 0,')
    print('  },')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
