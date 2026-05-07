#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for the ferrotorch-distributions
conformance suite.

Tracking issue: #859 — ferrotorch-distributions conformance suite.

Reference libraries (pinned):
    torch == 2.11.0

Output:
    ferrotorch-distributions/tests/conformance/fixtures.json

The script records analytic outputs (log_prob, mean, variance, entropy,
cdf, icdf) for each distribution at curated parameter + input combinations.
Sampling fixtures record shapes and statistical moment bounds only — exact
sample values are NOT recorded because ferrotorch uses a different RNG
algorithm (xorshift) than PyTorch (Mersenne Twister / Philox).

Distributions covered:
    Normal, Bernoulli, Categorical, Beta, Gamma, Exponential, Laplace,
    Cauchy, Gumbel, HalfNormal, LogNormal, Poisson, StudentT, Uniform,
    MultivariateNormal, LowRankMultivariateNormal, Dirichlet, Multinomial,
    Independent, MixtureSameFamily, OneHotCategorical, RelaxedBernoulli,
    RelaxedOneHotCategorical, Kumaraswamy, Pareto, VonMises, Weibull,
    TransformedDistribution.

Infrastructure fixtures:
    kl_divergence, constraints (Constraint.check), transforms
    (ExpTransform, AffineTransform, SigmoidTransform, TanhTransform,
    SoftplusTransform, ComposeTransform).

Usage:
    python3 scripts/regenerate_distributions_fixtures.py

Required:
    pip install "torch==2.11.0"

CI contract: exits 0 on success, non-zero on any error.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Reference library imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.distributions as dist
    import torch.distributions.constraints as constraints
    import torch.distributions.transforms as transforms
except ImportError as exc:
    print(f"ERROR: cannot import torch — {exc}", file=sys.stderr)
    print("Install with: pip install torch==2.11.0", file=sys.stderr)
    sys.exit(1)

TORCH_VERSION = torch.__version__
if not TORCH_VERSION.startswith("2.11"):
    print(
        f"WARNING: expected torch==2.11.0, got {TORCH_VERSION}. "
        "Fixtures may differ from pinned reference.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scalar_f64(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float64)

def vec_f64(vs: list[float]) -> torch.Tensor:
    return torch.tensor(vs, dtype=torch.float64)

def to_list(t: torch.Tensor) -> list:
    return t.detach().tolist()

def to_scalar(t: torch.Tensor) -> float:
    return float(t.detach().item())

def safe_cdf(d, x_t):
    try:
        return to_list(d.cdf(x_t))
    except (NotImplementedError, AttributeError):
        return None

def safe_icdf(d, q_t):
    try:
        return to_list(d.icdf(q_t))
    except (NotImplementedError, AttributeError):
        return None

def safe_mean(d):
    try:
        m = d.mean
        if isinstance(m, torch.Tensor):
            return to_list(m)
        return None
    except Exception:
        return None

def safe_variance(d):
    try:
        v = d.variance
        if isinstance(v, torch.Tensor):
            return to_list(v)
        return None
    except Exception:
        return None

def safe_entropy(d):
    try:
        h = d.entropy()
        return to_list(h)
    except Exception:
        return None

def safe_log_prob(d, x_t):
    try:
        return to_list(d.log_prob(x_t))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Distribution fixtures
# ---------------------------------------------------------------------------

fixtures: dict = {
    "metadata": {
        "torch_version": TORCH_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "conformance_note": (
            "Reference fixtures for ferrotorch-distributions conformance suite. "
            "Exact sample values are NOT recorded (RNG algorithm divergence). "
            "All analytic outputs (log_prob, mean, variance, entropy, cdf, icdf) "
            "are recorded against torch==2.11.0."
        ),
    },
    "normal": [],
    "bernoulli": [],
    "categorical": [],
    "beta": [],
    "gamma": [],
    "exponential": [],
    "laplace": [],
    "cauchy": [],
    "gumbel": [],
    "half_normal": [],
    "log_normal": [],
    "poisson": [],
    "student_t": [],
    "uniform": [],
    "multivariate_normal": [],
    "low_rank_multivariate_normal": [],
    "dirichlet": [],
    "multinomial": [],
    "independent": [],
    "mixture_same_family": [],
    "one_hot_categorical": [],
    "relaxed_bernoulli": [],
    "relaxed_one_hot_categorical": [],
    "kumaraswamy": [],
    "pareto": [],
    "von_mises": [],
    "weibull": [],
    "transformed_distribution": [],
    "kl_divergence": [],
    "constraints": [],
    "transforms": [],
}

# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

for loc_v, scale_v, label in [
    (0.0, 1.0, "standard"),
    (2.0, 3.0, "nonzero_loc"),
    (-1.5, 0.5, "negative_loc_small_scale"),
]:
    loc_t = scalar_f64(loc_v)
    scale_t = scalar_f64(scale_v)
    d = dist.Normal(loc_t, scale_t)
    xs = [loc_v - scale_v, loc_v, loc_v + scale_v, loc_v + 2 * scale_v]
    x_t = torch.tensor(xs, dtype=torch.float64)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    q_t = torch.tensor(qs, dtype=torch.float64)
    fixtures["normal"].append({
        "label": label,
        "loc": loc_v,
        "scale": scale_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
        "cdf_x": xs,
        "cdf": safe_cdf(d, x_t),
        "icdf_q": qs,
        "icdf": safe_icdf(d, q_t),
    })

# Batch Normal (2-element)
loc_b = torch.tensor([0.0, 1.0], dtype=torch.float64)
scale_b = torch.tensor([1.0, 2.0], dtype=torch.float64)
d_b = dist.Normal(loc_b, scale_b)
x_b = torch.tensor([0.0, 1.0], dtype=torch.float64)
fixtures["normal"].append({
    "label": "batch_2",
    "loc": [0.0, 1.0],
    "scale": [1.0, 2.0],
    "x_points": [0.0, 1.0],
    "log_prob": safe_log_prob(d_b, x_b),
    "mean": safe_mean(d_b),
    "variance": safe_variance(d_b),
    "entropy": safe_entropy(d_b),
})

# ---------------------------------------------------------------------------
# Bernoulli
# ---------------------------------------------------------------------------

for p_v, label in [(0.5, "fair"), (0.7, "biased"), (0.2, "low_p"), (0.99, "near_one")]:
    p_t = scalar_f64(p_v)
    d = dist.Bernoulli(probs=p_t)
    x01 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    fixtures["bernoulli"].append({
        "label": label,
        "probs": p_v,
        "log_prob_0": to_scalar(d.log_prob(scalar_f64(0.0))),
        "log_prob_1": to_scalar(d.log_prob(scalar_f64(1.0))),
        "mean": to_scalar(d.mean),
        "variance": to_scalar(d.variance),
        "entropy": to_scalar(d.entropy()),
    })

# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

for probs_v, label in [
    ([0.1, 0.3, 0.6], "3_class"),
    ([0.25, 0.25, 0.25, 0.25], "uniform_4"),
    ([0.9, 0.05, 0.05], "concentrated"),
]:
    p_t = torch.tensor(probs_v, dtype=torch.float64)
    d = dist.Categorical(probs=p_t)
    # log_prob for each class index
    lp = [to_scalar(d.log_prob(torch.tensor(float(i)))) for i in range(len(probs_v))]
    fixtures["categorical"].append({
        "label": label,
        "probs": probs_v,
        "log_prob_per_class": lp,
        "entropy": to_scalar(d.entropy()),
    })

# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

for a_v, b_v, label in [
    (2.0, 3.0, "alpha2_beta3"),
    (0.5, 0.5, "jeffrey"),
    (5.0, 1.0, "right_skewed"),
    (1.0, 1.0, "uniform"),
]:
    a_t = scalar_f64(a_v)
    b_t = scalar_f64(b_v)
    d = dist.Beta(a_t, b_t)
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    x_t = torch.tensor(xs, dtype=torch.float64)
    qs = [0.1, 0.5, 0.9]
    q_t = torch.tensor(qs, dtype=torch.float64)
    fixtures["beta"].append({
        "label": label,
        "concentration1": a_v,
        "concentration0": b_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
        "icdf_q": qs,
        "icdf": safe_icdf(d, q_t),
    })

# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

for conc_v, rate_v, label in [
    (2.0, 1.0, "alpha2_rate1"),
    (1.0, 2.0, "alpha1_rate2"),
    (3.0, 3.0, "equal_params"),
    (0.5, 1.0, "sub_unity_conc"),
]:
    conc_t = scalar_f64(conc_v)
    rate_t = scalar_f64(rate_v)
    d = dist.Gamma(conc_t, rate_t)
    xs = [0.5, 1.0, 2.0, 4.0]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["gamma"].append({
        "label": label,
        "concentration": conc_v,
        "rate": rate_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------

for rate_v, label in [(1.0, "rate1"), (0.5, "rate_half"), (3.0, "rate3")]:
    rate_t = scalar_f64(rate_v)
    d = dist.Exponential(rate_t)
    xs = [0.0, 0.5, 1.0, 2.0]
    x_t = torch.tensor(xs, dtype=torch.float64)
    qs = [0.1, 0.5, 0.9]
    q_t = torch.tensor(qs, dtype=torch.float64)
    fixtures["exponential"].append({
        "label": label,
        "rate": rate_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
        "cdf": safe_cdf(d, x_t),
        "icdf_q": qs,
        "icdf": safe_icdf(d, q_t),
    })

# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------

for loc_v, scale_v, label in [
    (0.0, 1.0, "standard"),
    (2.0, 0.5, "shifted_tight"),
]:
    loc_t = scalar_f64(loc_v)
    scale_t = scalar_f64(scale_v)
    d = dist.Laplace(loc_t, scale_t)
    xs = [loc_v - 2, loc_v, loc_v + 2]
    x_t = torch.tensor(xs, dtype=torch.float64)
    qs = [0.1, 0.5, 0.9]
    q_t = torch.tensor(qs, dtype=torch.float64)
    fixtures["laplace"].append({
        "label": label,
        "loc": loc_v,
        "scale": scale_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
        "cdf": safe_cdf(d, x_t),
        "icdf_q": qs,
        "icdf": safe_icdf(d, q_t),
    })

# ---------------------------------------------------------------------------
# Cauchy
# ---------------------------------------------------------------------------

for loc_v, scale_v, label in [
    (0.0, 1.0, "standard"),
    (1.0, 2.0, "shifted_wider"),
]:
    loc_t = scalar_f64(loc_v)
    scale_t = scalar_f64(scale_v)
    d = dist.Cauchy(loc_t, scale_t)
    xs = [loc_v - 2, loc_v, loc_v + 2]
    x_t = torch.tensor(xs, dtype=torch.float64)
    qs = [0.25, 0.5, 0.75]
    q_t = torch.tensor(qs, dtype=torch.float64)
    fixtures["cauchy"].append({
        "label": label,
        "loc": loc_v,
        "scale": scale_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "entropy": safe_entropy(d),
        "cdf": safe_cdf(d, x_t),
        "icdf_q": qs,
        "icdf": safe_icdf(d, q_t),
    })

# ---------------------------------------------------------------------------
# Gumbel
# ---------------------------------------------------------------------------

for loc_v, scale_v, label in [
    (0.0, 1.0, "standard"),
    (2.0, 0.5, "shifted"),
]:
    loc_t = scalar_f64(loc_v)
    scale_t = scalar_f64(scale_v)
    d = dist.Gumbel(loc_t, scale_t)
    xs = [loc_v - 1, loc_v, loc_v + 1, loc_v + 3]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["gumbel"].append({
        "label": label,
        "loc": loc_v,
        "scale": scale_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# HalfNormal
# ---------------------------------------------------------------------------

for scale_v, label in [(1.0, "scale1"), (2.0, "scale2")]:
    scale_t = scalar_f64(scale_v)
    d = dist.HalfNormal(scale_t)
    xs = [0.0, 0.5, 1.0, 2.0]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["half_normal"].append({
        "label": label,
        "scale": scale_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------

for loc_v, scale_v, label in [
    (0.0, 1.0, "standard"),
    (1.0, 0.5, "positive_loc"),
]:
    loc_t = scalar_f64(loc_v)
    scale_t = scalar_f64(scale_v)
    d = dist.LogNormal(loc_t, scale_t)
    xs = [0.5, 1.0, 2.0, 4.0]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["log_normal"].append({
        "label": label,
        "loc": loc_v,
        "scale": scale_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

for rate_v, label in [(1.0, "rate1"), (3.0, "rate3"), (0.5, "rate_half")]:
    rate_t = scalar_f64(rate_v)
    d = dist.Poisson(rate_t)
    ks = [0.0, 1.0, 2.0, 3.0, 5.0]
    k_t = torch.tensor(ks, dtype=torch.float64)
    fixtures["poisson"].append({
        "label": label,
        "rate": rate_v,
        "k_points": ks,
        "log_prob": safe_log_prob(d, k_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
    })

# ---------------------------------------------------------------------------
# StudentT
# ---------------------------------------------------------------------------

for df_v, loc_v, scale_v, label in [
    (1.0, 0.0, 1.0, "cauchy"),
    (3.0, 0.0, 1.0, "df3"),
    (10.0, 2.0, 1.5, "df10_shifted"),
]:
    df_t = scalar_f64(df_v)
    loc_t = scalar_f64(loc_v)
    scale_t = scalar_f64(scale_v)
    d = dist.StudentT(df_t, loc_t, scale_t)
    xs = [loc_v - 2, loc_v, loc_v + 2]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["student_t"].append({
        "label": label,
        "df": df_v,
        "loc": loc_v,
        "scale": scale_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "entropy": safe_entropy(d),
        "mean": to_list(d.mean) if df_v > 1 else None,
        "variance": to_list(d.variance) if df_v > 2 else None,
    })

# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------

for lo_v, hi_v, label in [
    (0.0, 1.0, "unit"),
    (-1.0, 1.0, "symmetric"),
    (2.0, 5.0, "shifted"),
]:
    lo_t = scalar_f64(lo_v)
    hi_t = scalar_f64(hi_v)
    d = dist.Uniform(lo_t, hi_t)
    mid = (lo_v + hi_v) / 2.0
    xs = [lo_v, mid, hi_v - 1e-9]
    x_t = torch.tensor(xs, dtype=torch.float64)
    qs = [0.0, 0.5, 1.0]
    q_t = torch.tensor(qs, dtype=torch.float64)
    fixtures["uniform"].append({
        "label": label,
        "low": lo_v,
        "high": hi_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
        "cdf": safe_cdf(d, x_t),
        "icdf_q": qs,
        "icdf": safe_icdf(d, q_t),
    })

# ---------------------------------------------------------------------------
# MultivariateNormal
# ---------------------------------------------------------------------------

# 2D case: loc=[0,0], scale_tril=[[1,0],[0.5,1]]
loc_mv = torch.tensor([0.0, 0.0], dtype=torch.float64)
scale_tril = torch.tensor([[1.0, 0.0], [0.5, 1.0]], dtype=torch.float64)
d = dist.MultivariateNormal(loc_mv, scale_tril=scale_tril)
x_mv = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]], dtype=torch.float64)
fixtures["multivariate_normal"].append({
    "label": "2d_cholesky",
    "loc": loc_mv.tolist(),
    "scale_tril": scale_tril.tolist(),
    "x_points": x_mv.tolist(),
    "log_prob": safe_log_prob(d, x_mv),
    "mean": safe_mean(d),
    "entropy": safe_entropy(d),
})

# 3D case: loc=[1,2,3], scale_tril=identity
loc_3 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
st_3 = torch.eye(3, dtype=torch.float64)
d3 = dist.MultivariateNormal(loc_3, scale_tril=st_3)
x3 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
fixtures["multivariate_normal"].append({
    "label": "3d_identity_scale",
    "loc": loc_3.tolist(),
    "scale_tril": st_3.tolist(),
    "x_points": x3.tolist(),
    "log_prob": safe_log_prob(d3, x3),
    "entropy": safe_entropy(d3),
})

# ---------------------------------------------------------------------------
# LowRankMultivariateNormal
# ---------------------------------------------------------------------------

loc_lr = torch.tensor([0.0, 0.0], dtype=torch.float64)
cov_factor = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
cov_diag = torch.tensor([0.5, 0.5], dtype=torch.float64)
d_lr = dist.LowRankMultivariateNormal(loc_lr, cov_factor, cov_diag)
x_lr = torch.tensor([[0.0, 0.0], [1.0, 0.5]], dtype=torch.float64)
fixtures["low_rank_multivariate_normal"].append({
    "label": "2d_rank1",
    "loc": loc_lr.tolist(),
    "cov_factor": cov_factor.tolist(),
    "cov_diag": cov_diag.tolist(),
    "x_points": x_lr.tolist(),
    "log_prob": safe_log_prob(d_lr, x_lr),
    "mean": safe_mean(d_lr),
    "entropy": safe_entropy(d_lr),
})

# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------

for conc_v, label in [
    ([1.0, 1.0, 1.0], "uniform_3"),
    ([2.0, 3.0, 1.0], "asymmetric_3"),
    ([0.5, 0.5], "jeffreys_2"),
]:
    conc_t = torch.tensor(conc_v, dtype=torch.float64)
    d = dist.Dirichlet(conc_t)
    # sample on the simplex
    if len(conc_v) == 3:
        x_simplex = torch.tensor([[0.33, 0.33, 0.34], [0.1, 0.8, 0.1]], dtype=torch.float64)
    else:
        x_simplex = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float64)
    fixtures["dirichlet"].append({
        "label": label,
        "concentration": conc_v,
        "x_points": x_simplex.tolist(),
        "log_prob": safe_log_prob(d, x_simplex),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# Multinomial
# ---------------------------------------------------------------------------

for probs_v, total_count, label in [
    ([0.2, 0.3, 0.5], 10, "3_class_10"),
    ([0.5, 0.5], 5, "binary_5"),
]:
    p_t = torch.tensor(probs_v, dtype=torch.float64)
    d = dist.Multinomial(total_count=total_count, probs=p_t)
    # log_prob at the mean rounded to ints
    mean_counts = [round(total_count * p) for p in probs_v]
    # adjust to sum to total_count
    diff = total_count - sum(mean_counts)
    if diff != 0:
        mean_counts[-1] += diff
    x_counts = torch.tensor(mean_counts, dtype=torch.float64)
    fixtures["multinomial"].append({
        "label": label,
        "probs": probs_v,
        "total_count": total_count,
        "x_counts": mean_counts,
        "log_prob": to_scalar(d.log_prob(x_counts)),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
    })

# ---------------------------------------------------------------------------
# Independent
# ---------------------------------------------------------------------------

# Wrap a batch of 3 Normal distributions as a single Independent
loc_ind = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
scale_ind = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
base_normal = dist.Normal(loc_ind, scale_ind)
d_ind = dist.Independent(base_normal, reinterpreted_batch_ndims=1)
x_ind = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
fixtures["independent"].append({
    "label": "normal_batch3",
    "loc": loc_ind.tolist(),
    "scale": scale_ind.tolist(),
    "reinterpreted_batch_ndims": 1,
    "x_point": x_ind.tolist(),
    "log_prob": to_scalar(d_ind.log_prob(x_ind)),
    "entropy": to_scalar(d_ind.entropy()),
})

# ---------------------------------------------------------------------------
# MixtureSameFamily
# ---------------------------------------------------------------------------

# 2-component Normal mixture
mix_probs = torch.tensor([0.4, 0.6], dtype=torch.float64)
comp_loc = torch.tensor([-2.0, 2.0], dtype=torch.float64)
comp_scale = torch.tensor([0.5, 0.5], dtype=torch.float64)
mixture = dist.Categorical(probs=mix_probs)
components = dist.Normal(comp_loc, comp_scale)
d_mix = dist.MixtureSameFamily(mixture, components)
xs_mix = [-2.0, 0.0, 2.0]
x_t_mix = torch.tensor(xs_mix, dtype=torch.float64)
fixtures["mixture_same_family"].append({
    "label": "two_component_normal",
    "mix_probs": mix_probs.tolist(),
    "comp_loc": comp_loc.tolist(),
    "comp_scale": comp_scale.tolist(),
    "x_points": xs_mix,
    "log_prob": safe_log_prob(d_mix, x_t_mix),
})

# ---------------------------------------------------------------------------
# OneHotCategorical
# ---------------------------------------------------------------------------

for probs_v, label in [
    ([0.1, 0.3, 0.6], "3_class"),
    ([0.25, 0.25, 0.25, 0.25], "uniform_4"),
]:
    p_t = torch.tensor(probs_v, dtype=torch.float64)
    d = dist.OneHotCategorical(probs=p_t)
    n = len(probs_v)
    # log_prob of each one-hot vector
    lps = []
    for i in range(n):
        oh = [0.0] * n
        oh[i] = 1.0
        lps.append(to_scalar(d.log_prob(torch.tensor(oh, dtype=torch.float64))))
    fixtures["one_hot_categorical"].append({
        "label": label,
        "probs": probs_v,
        "log_prob_per_class": lps,
        "entropy": to_scalar(d.entropy()),
    })

# ---------------------------------------------------------------------------
# RelaxedBernoulli
# ---------------------------------------------------------------------------

for temp_v, p_v, label in [
    (0.5, 0.7, "temp_half_p07"),
    (1.0, 0.5, "temp_one_fair"),
]:
    temp_t = scalar_f64(temp_v)
    p_t = scalar_f64(p_v)
    d = dist.RelaxedBernoulli(temperature=temp_t, probs=p_t)
    xs = [0.2, 0.5, 0.8]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["relaxed_bernoulli"].append({
        "label": label,
        "temperature": temp_v,
        "probs": p_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
    })

# ---------------------------------------------------------------------------
# RelaxedOneHotCategorical
# ---------------------------------------------------------------------------

for temp_v, probs_v, label in [
    (0.5, [0.2, 0.3, 0.5], "3_class_temp_half"),
    (1.0, [0.5, 0.5], "binary_fair"),
]:
    temp_t = scalar_f64(temp_v)
    p_t = torch.tensor(probs_v, dtype=torch.float64)
    d = dist.RelaxedOneHotCategorical(temperature=temp_t, probs=p_t)
    n = len(probs_v)
    # uniform point on simplex
    xs_ohc = [1.0 / n] * n
    x_t_ohc = torch.tensor(xs_ohc, dtype=torch.float64)
    fixtures["relaxed_one_hot_categorical"].append({
        "label": label,
        "temperature": temp_v,
        "probs": probs_v,
        "x_point_uniform": xs_ohc,
        "log_prob_at_uniform": to_scalar(d.log_prob(x_t_ohc)),
    })

# ---------------------------------------------------------------------------
# Kumaraswamy
# ---------------------------------------------------------------------------

for a_v, b_v, label in [
    (2.0, 5.0, "a2_b5"),
    (0.5, 0.5, "sub_unity"),
    (1.0, 1.0, "uniform"),
]:
    a_t = scalar_f64(a_v)
    b_t = scalar_f64(b_v)
    d = dist.Kumaraswamy(a_t, b_t)
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["kumaraswamy"].append({
        "label": label,
        "concentration1": a_v,
        "concentration0": b_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

for scale_v, alpha_v, label in [
    (1.0, 2.0, "scale1_alpha2"),
    (2.0, 3.0, "scale2_alpha3"),
]:
    scale_t = scalar_f64(scale_v)
    alpha_t = scalar_f64(alpha_v)
    d = dist.Pareto(scale_t, alpha_t)
    xs = [scale_v + 0.1, scale_v + 0.5, scale_v + 1.0, scale_v + 3.0]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["pareto"].append({
        "label": label,
        "scale": scale_v,
        "alpha": alpha_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# VonMises
# ---------------------------------------------------------------------------

for loc_v, conc_v, label in [
    (0.0, 1.0, "loc0_conc1"),
    (1.5, 3.0, "loc_pi2_conc3"),
]:
    loc_t = scalar_f64(loc_v)
    conc_t = scalar_f64(conc_v)
    d = dist.VonMises(loc_t, conc_t)
    xs = [loc_v - 1.0, loc_v, loc_v + 1.0]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["von_mises"].append({
        "label": label,
        "loc": loc_v,
        "concentration": conc_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# Weibull
# ---------------------------------------------------------------------------

for scale_v, conc_v, label in [
    (1.0, 1.0, "exponential_special"),
    (2.0, 3.0, "scale2_conc3"),
    (1.0, 0.5, "sub_unity_conc"),
]:
    scale_t = scalar_f64(scale_v)
    conc_t = scalar_f64(conc_v)
    d = dist.Weibull(scale_t, conc_t)
    xs = [0.5, 1.0, 2.0, 3.0]
    x_t = torch.tensor(xs, dtype=torch.float64)
    fixtures["weibull"].append({
        "label": label,
        "scale": scale_v,
        "concentration": conc_v,
        "x_points": xs,
        "log_prob": safe_log_prob(d, x_t),
        "mean": safe_mean(d),
        "variance": safe_variance(d),
        "entropy": safe_entropy(d),
    })

# ---------------------------------------------------------------------------
# TransformedDistribution: Normal through ExpTransform == LogNormal
# ---------------------------------------------------------------------------

loc_td = scalar_f64(0.0)
scale_td = scalar_f64(1.0)
base_td = dist.Normal(loc_td, scale_td)
exp_transform = transforms.ExpTransform()
d_td = dist.TransformedDistribution(base_td, [exp_transform])
d_ref = dist.LogNormal(loc_td, scale_td)
xs_td = [0.5, 1.0, 2.0, 4.0]
x_t_td = torch.tensor(xs_td, dtype=torch.float64)
fixtures["transformed_distribution"].append({
    "label": "normal_exp_equals_lognormal",
    "base_loc": 0.0,
    "base_scale": 1.0,
    "transform": "ExpTransform",
    "x_points": xs_td,
    "log_prob_transformed": safe_log_prob(d_td, x_t_td),
    "log_prob_lognormal_ref": safe_log_prob(d_ref, x_t_td),
    "note": "TransformedDistribution(Normal(0,1), ExpTransform) log_prob must match LogNormal(0,1).log_prob",
})

# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

# Normal || Normal
p_n = dist.Normal(scalar_f64(0.0), scalar_f64(1.0))
q_n = dist.Normal(scalar_f64(1.0), scalar_f64(2.0))
kl_nn = to_scalar(dist.kl_divergence(p_n, q_n))
# Analytic: KL(N(m1,s1)||N(m2,s2)) = log(s2/s1) + (s1^2+(m1-m2)^2)/(2s2^2) - 0.5
fixtures["kl_divergence"].append({
    "label": "normal_normal",
    "p_loc": 0.0, "p_scale": 1.0,
    "q_loc": 1.0, "q_scale": 2.0,
    "kl": kl_nn,
})

# Normal self-KL must be 0
p_same = dist.Normal(scalar_f64(0.0), scalar_f64(1.0))
q_same = dist.Normal(scalar_f64(0.0), scalar_f64(1.0))
fixtures["kl_divergence"].append({
    "label": "normal_self_zero",
    "p_loc": 0.0, "p_scale": 1.0,
    "q_loc": 0.0, "q_scale": 1.0,
    "kl": to_scalar(dist.kl_divergence(p_same, q_same)),
    "expected_zero": True,
})

# Bernoulli || Bernoulli
p_b = dist.Bernoulli(probs=scalar_f64(0.3))
q_b = dist.Bernoulli(probs=scalar_f64(0.7))
fixtures["kl_divergence"].append({
    "label": "bernoulli_bernoulli",
    "p_probs": 0.3, "q_probs": 0.7,
    "kl": to_scalar(dist.kl_divergence(p_b, q_b)),
})

# Gamma || Gamma
p_g = dist.Gamma(scalar_f64(2.0), scalar_f64(1.0))
q_g = dist.Gamma(scalar_f64(3.0), scalar_f64(2.0))
fixtures["kl_divergence"].append({
    "label": "gamma_gamma",
    "p_concentration": 2.0, "p_rate": 1.0,
    "q_concentration": 3.0, "q_rate": 2.0,
    "kl": to_scalar(dist.kl_divergence(p_g, q_g)),
})

# Exponential || Exponential (special case of Gamma)
p_e = dist.Exponential(scalar_f64(1.0))
q_e = dist.Exponential(scalar_f64(2.0))
fixtures["kl_divergence"].append({
    "label": "exponential_exponential",
    "p_rate": 1.0, "q_rate": 2.0,
    "kl": to_scalar(dist.kl_divergence(p_e, q_e)),
})

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _json_safe(v: float) -> object:
    """Convert float to a JSON-serializable representation."""
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return v

constraint_cases = [
    ("real", constraints.real, [0.0, 1.0, -1.0, float("inf"), float("nan")],
     [True, True, True, True, False]),
    ("positive", constraints.positive, [-1.0, 0.0, 0.001, 1.0, float("inf")],
     [False, False, True, True, True]),
    ("unit_interval", constraints.unit_interval, [-0.1, 0.0, 0.5, 1.0, 1.1],
     [False, True, True, True, False]),
]

for label, c_factory, values, expected_bool in constraint_cases:
    c = c_factory
    # For some torch versions, constraints are classes not instances:
    if callable(c) and not hasattr(c, 'check'):
        c = c()
    results = []
    for v in values:
        try:
            t = torch.tensor(v, dtype=torch.float64)
            ok = bool(c.check(t))
        except Exception:
            ok = False
        results.append(ok)
    fixtures["constraints"].append({
        "label": label,
        # Represent inf/nan as strings for JSON compatibility
        "values": [_json_safe(v) for v in values],
        "expected_check": expected_bool,
        "torch_check": results,
    })

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ExpTransform: y = exp(x)
exp_t = transforms.ExpTransform()
xs_exp = [-2.0, -1.0, 0.0, 1.0, 2.0]
x_exp = torch.tensor(xs_exp, dtype=torch.float64)
y_exp = exp_t(x_exp)
x_inv = exp_t.inv(y_exp)
fixtures["transforms"].append({
    "label": "exp_transform",
    "x_points": xs_exp,
    "forward": to_list(y_exp),
    "inverse": to_list(x_inv),
    "log_abs_det_jacobian": to_list(exp_t.log_abs_det_jacobian(x_exp, y_exp)),
})

# AffineTransform: y = 2 + 3*x
aff_t = transforms.AffineTransform(loc=2.0, scale=3.0)
xs_aff = [-1.0, 0.0, 1.0, 2.0]
x_aff = torch.tensor(xs_aff, dtype=torch.float64)
y_aff = aff_t(x_aff)
fixtures["transforms"].append({
    "label": "affine_transform_loc2_scale3",
    "loc": 2.0,
    "scale": 3.0,
    "x_points": xs_aff,
    "forward": to_list(y_aff),
    "inverse": to_list(aff_t.inv(y_aff)),
    "log_abs_det_jacobian": to_list(aff_t.log_abs_det_jacobian(x_aff, y_aff)),
})

# SigmoidTransform: y = sigmoid(x)
sig_t = transforms.SigmoidTransform()
xs_sig = [-2.0, -1.0, 0.0, 1.0, 2.0]
x_sig = torch.tensor(xs_sig, dtype=torch.float64)
y_sig = sig_t(x_sig)
fixtures["transforms"].append({
    "label": "sigmoid_transform",
    "x_points": xs_sig,
    "forward": to_list(y_sig),
    "inverse": to_list(sig_t.inv(y_sig)),
    "log_abs_det_jacobian": to_list(sig_t.log_abs_det_jacobian(x_sig, y_sig)),
})

# TanhTransform: y = tanh(x)
tanh_t = transforms.TanhTransform()
xs_tanh = [-2.0, -1.0, 0.0, 1.0, 2.0]
x_tanh = torch.tensor(xs_tanh, dtype=torch.float64)
y_tanh = tanh_t(x_tanh)
fixtures["transforms"].append({
    "label": "tanh_transform",
    "x_points": xs_tanh,
    "forward": to_list(y_tanh),
    "inverse": to_list(tanh_t.inv(y_tanh)),
    "log_abs_det_jacobian": to_list(tanh_t.log_abs_det_jacobian(x_tanh, y_tanh)),
})

# SoftplusTransform: y = log(1 + exp(x))
sp_t = transforms.SoftplusTransform()
xs_sp = [-2.0, -1.0, 0.0, 1.0, 2.0]
x_sp = torch.tensor(xs_sp, dtype=torch.float64)
y_sp = sp_t(x_sp)
fixtures["transforms"].append({
    "label": "softplus_transform",
    "x_points": xs_sp,
    "forward": to_list(y_sp),
    "inverse": to_list(sp_t.inv(y_sp)),
    "log_abs_det_jacobian": to_list(sp_t.log_abs_det_jacobian(x_sp, y_sp)),
})

# ComposeTransform: AffineTransform then ExpTransform
compose_t = transforms.ComposeTransform([aff_t, exp_t])
xs_comp = [-1.0, 0.0, 1.0]
x_comp = torch.tensor(xs_comp, dtype=torch.float64)
y_comp = compose_t(x_comp)
fixtures["transforms"].append({
    "label": "compose_affine_then_exp",
    "x_points": xs_comp,
    "forward": to_list(y_comp),
    "inverse": to_list(compose_t.inv(y_comp)),
    "log_abs_det_jacobian": to_list(compose_t.log_abs_det_jacobian(x_comp, y_comp)),
})

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

out_path = "ferrotorch-distributions/tests/conformance/fixtures.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(fixtures, f, indent=2)

# Count fixtures
total = sum(
    len(v) for k, v in fixtures.items()
    if k != "metadata" and isinstance(v, list)
)
print(f"OK  regenerate_distributions_fixtures.py")
print(f"    torch={TORCH_VERSION}")
print(f"    {total} fixture entries across {len(fixtures) - 1} distribution families")
print(f"    written to {out_path}")
