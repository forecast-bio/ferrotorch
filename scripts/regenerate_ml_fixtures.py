#!/usr/bin/env python3
"""
Regenerate scikit-learn reference fixtures for the ferrotorch-ml conformance suite.

Tracking issue: #840 (ferrotorch-ml sklearn-parity conformance).

Output: ``ferrotorch-ml/tests/conformance/fixtures.json``

Reference library: scikit-learn 1.5.x (pin: 1.5.2)
All inputs use ``random_state=42`` for full reproducibility.

How fixtures are generated per module:

* **adapter** — round-trip identity checks; no external reference needed.
  The fixture records (input_flat, shape) → expected_flat for 1-D and 2-D
  cases. sklearn is not needed here; the reference is numeric identity.

* **metrics/regression** — sklearn.metrics.{r2_score, mean_squared_error,
  root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
  median_absolute_error, max_error, explained_variance_score} called with
  deterministic y_true / y_pred pairs.

* **metrics/classification** — sklearn.metrics.{accuracy_score,
  precision_score, recall_score, f1_score, roc_auc_score, log_loss,
  confusion_matrix, hamming_loss, balanced_accuracy_score,
  matthews_corrcoef, cohen_kappa_score, brier_score_loss,
  top_k_accuracy_score, zero_one_loss, average_precision_score}

* **metrics/ranking** — sklearn.metrics.{dcg_score, ndcg_score,
  coverage_error, label_ranking_average_precision_score,
  label_ranking_loss}

* **metrics/clustering** — sklearn.metrics.{adjusted_rand_score,
  adjusted_mutual_info_score, normalized_mutual_info_score,
  homogeneity_score, completeness_score, v_measure_score,
  fowlkes_mallows_score, silhouette_score, davies_bouldin_score,
  calinski_harabasz_score}

* **datasets** — shape + label-range checks for make_classification,
  make_regression, make_blobs, make_moons, make_circles, load_iris,
  load_wine, load_breast_cancer. No numeric comparison to sklearn output
  because ferrolearn_datasets is a separate Rust reimplementation; we
  verify structural contracts (shape, dtype, label range).

Usage:

    python3 scripts/regenerate_ml_fixtures.py

Required:

    pip install --user scikit-learn==1.5.2 numpy
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn import __version__ as _sklearn_version
from sklearn.metrics import (
    # regression
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
    # classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    hamming_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    top_k_accuracy_score,
    zero_one_loss,
    average_precision_score,
    # ranking
    dcg_score,
    ndcg_score,
    coverage_error,
    label_ranking_average_precision_score,
    label_ranking_loss,
    # clustering
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

assert _sklearn_version.startswith("1.5"), (
    f"Expected scikit-learn 1.5.x, got {_sklearn_version}. "
    "Run: pip install --user scikit-learn==1.5.2"
)

RNG = np.random.default_rng(42)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ferrotorch-ml",
    "tests",
    "conformance",
    "fixtures.json",
)

fixtures = []


def add(op, **fields):
    """Append a fixture record."""
    fixtures.append({"op": op, **fields})


# ---------------------------------------------------------------------------
# Adapter round-trip fixtures (no sklearn dependency — numeric identity)
# ---------------------------------------------------------------------------

# 1-D round-trip
data_1d = [1.0, 2.0, 3.0, 4.0, 5.0]
add("adapter_tensor_to_array1",
    input_flat=data_1d, shape=[5], expected_flat=data_1d)
add("adapter_array1_to_tensor",
    input_flat=data_1d, shape=[5], expected_flat=data_1d)

# 2-D round-trip (row-major)
data_2d = list(map(float, range(6)))  # [0,1,2,3,4,5]
add("adapter_tensor_to_array2",
    input_flat=data_2d, shape=[2, 3], expected_flat=data_2d)
add("adapter_array2_to_tensor",
    input_flat=data_2d, shape=[2, 3], expected_flat=data_2d)

# usize label round-trip
labels_usize = [0, 1, 2, 1, 0]
labels_float = [float(x) for x in labels_usize]
add("adapter_array1_usize_to_tensor",
    input_labels=labels_usize, expected_flat=labels_float)
add("adapter_tensor_to_array1_usize",
    input_flat=labels_float, expected_labels=labels_usize)

# ---------------------------------------------------------------------------
# Helper: small deterministic regression inputs
# ---------------------------------------------------------------------------

def reg_pair(n=20, noise=0.1, seed=42):
    """Return (y_true_list, y_pred_list) using numpy RNG."""
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0.5, 5.0, n)
    y_pred = y_true + rng.normal(0, noise, n)
    return y_true.tolist(), y_pred.tolist()


y_true_reg, y_pred_reg = reg_pair()

# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

add("r2_score",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(r2_score(y_true_reg, y_pred_reg)),
    tol=1e-9)

add("mean_squared_error",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(mean_squared_error(y_true_reg, y_pred_reg)),
    tol=1e-9)

# sklearn 1.5 root_mean_squared_error is mean_squared_error(squared=False)
_mse_val = mean_squared_error(y_true_reg, y_pred_reg)
add("root_mean_squared_error",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(np.sqrt(_mse_val)),
    tol=1e-9)

add("mean_absolute_error",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(mean_absolute_error(y_true_reg, y_pred_reg)),
    tol=1e-9)

# MAPE — use non-zero y_true to avoid division issues
add("mean_absolute_percentage_error",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(mean_absolute_percentage_error(y_true_reg, y_pred_reg)),
    tol=1e-9)

add("median_absolute_error",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(median_absolute_error(y_true_reg, y_pred_reg)),
    tol=1e-9)

add("max_error",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(max_error(y_true_reg, y_pred_reg)),
    tol=1e-9)

add("explained_variance_score",
    y_true=y_true_reg, y_pred=y_pred_reg,
    expected=float(explained_variance_score(y_true_reg, y_pred_reg)),
    tol=1e-9)

# Edge case: perfect predictions → R² = 1, MSE = 0, RMSE = 0, MAE = 0
y_perfect = [1.0, 2.0, 3.0, 4.0]
add("r2_score_perfect",
    y_true=y_perfect, y_pred=y_perfect,
    expected=1.0, tol=1e-12)
add("mean_squared_error_perfect",
    y_true=y_perfect, y_pred=y_perfect,
    expected=0.0, tol=1e-12)
add("mean_absolute_error_perfect",
    y_true=y_perfect, y_pred=y_perfect,
    expected=0.0, tol=1e-12)
add("root_mean_squared_error_perfect",
    y_true=y_perfect, y_pred=y_perfect,
    expected=0.0, tol=1e-12)

# Edge case: constant-mean baseline → R² = 0
y_mean_pred = [float(np.mean(y_perfect))] * len(y_perfect)
add("r2_score_constant_baseline",
    y_true=y_perfect, y_pred=y_mean_pred,
    expected=0.0, tol=1e-12)

# ---------------------------------------------------------------------------
# Helper: classification pairs (binary and multi-class)
# ---------------------------------------------------------------------------

# Binary
y_true_bin = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
y_pred_bin = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
y_score_bin = [0.1, 0.6, 0.8, 0.3, 0.9, 0.2, 0.7, 0.55, 0.4, 0.85]

# Multi-class (3 classes)
y_true_mc = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
y_pred_mc = [0, 1, 1, 2, 2, 1, 0, 0, 2, 0]

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

add("accuracy_score",
    y_true=y_true_bin, y_pred=y_pred_bin,
    expected=float(accuracy_score(y_true_bin, y_pred_bin)),
    tol=1e-12)

add("precision_score_binary",
    y_true=y_true_bin, y_pred=y_pred_bin, average="binary",
    expected=float(precision_score(y_true_bin, y_pred_bin, average="binary",
                                   zero_division=0)),
    tol=1e-12)

add("recall_score_binary",
    y_true=y_true_bin, y_pred=y_pred_bin, average="binary",
    expected=float(recall_score(y_true_bin, y_pred_bin, average="binary",
                                zero_division=0)),
    tol=1e-12)

add("f1_score_binary",
    y_true=y_true_bin, y_pred=y_pred_bin, average="binary",
    expected=float(f1_score(y_true_bin, y_pred_bin, average="binary",
                            zero_division=0)),
    tol=1e-12)

add("precision_score_macro",
    y_true=y_true_mc, y_pred=y_pred_mc, average="macro",
    expected=float(precision_score(y_true_mc, y_pred_mc, average="macro",
                                   zero_division=0)),
    tol=1e-12)

add("recall_score_macro",
    y_true=y_true_mc, y_pred=y_pred_mc, average="macro",
    expected=float(recall_score(y_true_mc, y_pred_mc, average="macro",
                                zero_division=0)),
    tol=1e-12)

add("f1_score_macro",
    y_true=y_true_mc, y_pred=y_pred_mc, average="macro",
    expected=float(f1_score(y_true_mc, y_pred_mc, average="macro",
                            zero_division=0)),
    tol=1e-12)

add("roc_auc_score",
    y_true=y_true_bin, y_score=y_score_bin,
    expected=float(roc_auc_score(y_true_bin, y_score_bin)),
    tol=1e-12)

# log_loss: 2 samples, 2 classes
y_ll_true = [0, 1]
y_ll_prob = [[0.9, 0.1], [0.1, 0.9]]
add("log_loss",
    y_true=y_ll_true, y_prob=y_ll_prob,
    expected=float(log_loss(y_ll_true, y_ll_prob)),
    tol=1e-9)

# confusion matrix: binary
cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()
add("confusion_matrix",
    y_true=y_true_bin, y_pred=y_pred_bin,
    expected=cm)

add("hamming_loss",
    y_true=y_true_bin, y_pred=y_pred_bin,
    expected=float(hamming_loss(y_true_bin, y_pred_bin)),
    tol=1e-12)

add("balanced_accuracy_score",
    y_true=y_true_bin, y_pred=y_pred_bin, adjusted=False,
    expected=float(balanced_accuracy_score(y_true_bin, y_pred_bin,
                                           adjusted=False)),
    tol=1e-12)

add("matthews_corrcoef",
    y_true=y_true_bin, y_pred=y_pred_bin,
    expected=float(matthews_corrcoef(y_true_bin, y_pred_bin)),
    tol=1e-12)

add("cohen_kappa_score",
    y_true=y_true_bin, y_pred=y_pred_bin,
    expected=float(cohen_kappa_score(y_true_bin, y_pred_bin)),
    tol=1e-12)

add("brier_score_loss",
    y_true=y_true_bin, y_prob=y_score_bin,
    expected=float(brier_score_loss(y_true_bin, y_score_bin)),
    tol=1e-12)

# top-k: 3 samples, 3 classes
y_tk_true = [0, 2, 1]
y_tk_score = [
    [0.6, 0.3, 0.1],
    [0.1, 0.2, 0.7],
    [0.2, 0.5, 0.3],
]
add("top_k_accuracy_score_k1",
    y_true=y_tk_true, y_score=y_tk_score, k=1,
    expected=float(top_k_accuracy_score(y_tk_true, y_tk_score, k=1)),
    tol=1e-12)
add("top_k_accuracy_score_k2",
    y_true=y_tk_true, y_score=y_tk_score, k=2,
    expected=float(top_k_accuracy_score(y_tk_true, y_tk_score, k=2)),
    tol=1e-12)

add("zero_one_loss_normalized",
    y_true=y_true_bin, y_pred=y_pred_bin, normalize=True,
    expected=float(zero_one_loss(y_true_bin, y_pred_bin, normalize=True)),
    tol=1e-12)
add("zero_one_loss_count",
    y_true=y_true_bin, y_pred=y_pred_bin, normalize=False,
    expected=float(zero_one_loss(y_true_bin, y_pred_bin, normalize=False)),
    tol=1e-12)

add("average_precision_score",
    y_true=y_true_bin, y_score=y_score_bin,
    expected=float(average_precision_score(y_true_bin, y_score_bin)),
    tol=1e-12)

# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

# Single-query: 1-D inputs
y_rel = [3.0, 2.0, 1.0, 0.0]
y_sc  = [3.0, 2.0, 1.0, 0.0]  # perfect ranking
add("ndcg_score_perfect",
    y_true=y_rel, y_score=y_sc, k=None,
    expected=float(ndcg_score(
        np.array([y_rel]), np.array([y_sc]))),
    tol=1e-9)

y_sc_imperfect = [2.0, 3.0, 1.0, 0.0]  # slight mis-rank
add("ndcg_score_imperfect",
    y_true=y_rel, y_score=y_sc_imperfect, k=None,
    expected=float(ndcg_score(
        np.array([y_rel]), np.array([y_sc_imperfect]))),
    tol=1e-9)

add("dcg_score",
    y_true=y_rel, y_score=y_sc, k=None,
    expected=float(dcg_score(
        np.array([y_rel]), np.array([y_sc]))),
    tol=1e-9)

# Multi-label: 2-D inputs [N, K]
y_lrap_true = np.array([[1, 0, 0], [0, 1, 0]])
y_lrap_score = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])
add("label_ranking_average_precision_score",
    y_true=y_lrap_true.tolist(), y_score=y_lrap_score.tolist(),
    expected=float(label_ranking_average_precision_score(
        y_lrap_true, y_lrap_score)),
    tol=1e-9)

add("label_ranking_loss",
    y_true=y_lrap_true.tolist(), y_score=y_lrap_score.tolist(),
    expected=float(label_ranking_loss(y_lrap_true, y_lrap_score)),
    tol=1e-9)

y_cov_true = np.array([[1, 0, 0, 1]])
y_cov_score = np.array([[0.9, 0.4, 0.5, 0.1]])
add("coverage_error",
    y_true=y_cov_true.tolist(), y_score=y_cov_score.tolist(),
    expected=float(coverage_error(y_cov_true, y_cov_score)),
    tol=1e-9)

# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

labels_cl = [0, 0, 1, 1, 2, 2]

add("adjusted_rand_score_perfect",
    labels_true=labels_cl, labels_pred=labels_cl,
    expected=1.0, tol=1e-12)

_ari_mixed_pred = [0, 1, 1, 0, 2, 2]
add("adjusted_rand_score_mixed",
    labels_true=labels_cl, labels_pred=_ari_mixed_pred,
    expected=float(adjusted_rand_score(labels_cl, _ari_mixed_pred)),
    tol=1e-12)

add("adjusted_mutual_info_score",
    labels_true=labels_cl, labels_pred=labels_cl,
    expected=float(adjusted_mutual_info_score(labels_cl, labels_cl)),
    tol=1e-9)

add("normalized_mutual_info_score",
    labels_true=labels_cl, labels_pred=labels_cl,
    expected=1.0, tol=1e-9)

add("homogeneity_score_perfect",
    labels_true=labels_cl, labels_pred=labels_cl,
    expected=1.0, tol=1e-9)

add("completeness_score_perfect",
    labels_true=labels_cl, labels_pred=labels_cl,
    expected=1.0, tol=1e-9)

add("v_measure_score_perfect",
    labels_true=labels_cl, labels_pred=labels_cl,
    expected=1.0, tol=1e-9)

add("fowlkes_mallows_score_perfect",
    labels_true=labels_cl, labels_pred=labels_cl,
    expected=1.0, tol=1e-12)

# Silhouette — well-separated 2-D clusters
x_sil = np.array([
    [0.0, 0.0], [0.1, 0.1],
    [10.0, 10.0], [10.1, 10.1],
])
labels_sil = [0, 0, 1, 1]
add("silhouette_score",
    x=x_sil.tolist(), labels=labels_sil,
    expected=float(silhouette_score(x_sil, labels_sil)),
    tol=1e-9)

# Davies-Bouldin — lower is better
x_db = np.array([
    [0.0, 0.0], [0.1, 0.1],
    [5.0, 5.0], [5.1, 5.1],
])
labels_db = [0, 0, 1, 1]
add("davies_bouldin_score",
    x=x_db.tolist(), labels=labels_db,
    expected=float(davies_bouldin_score(x_db, labels_db)),
    tol=1e-9)

# Calinski-Harabasz — higher is better
add("calinski_harabasz_score",
    x=x_db.tolist(), labels=labels_db,
    expected=float(calinski_harabasz_score(x_db, labels_db)),
    tol=1e-9)

# ---------------------------------------------------------------------------
# Dataset shape / contract fixtures (no numeric sklearn comparison;
# the contracts are shape, dtype, and label-range invariants)
# ---------------------------------------------------------------------------

add("dataset_make_classification",
    n_samples=50, n_features=4, n_classes=3, random_state=42,
    expected_x_shape=[50, 4],
    expected_y_shape=[50],
    label_min=0, label_max=2)

add("dataset_make_regression",
    n_samples=80, n_features=5, n_informative=3,
    noise=0.1, random_state=7,
    expected_x_shape=[80, 5],
    expected_y_shape=[80])

add("dataset_make_blobs",
    n_samples=60, n_features=2, centers=3, cluster_std=1.0,
    random_state=1,
    expected_x_shape=[60, 2],
    expected_y_shape=[60],
    label_min=0, label_max=2)

add("dataset_make_moons",
    n_samples=40, noise=0.05, random_state=2,
    expected_x_shape=[40, 2],
    expected_y_shape=[40],
    label_min=0, label_max=1)

add("dataset_make_circles",
    n_samples=40, noise=0.02, factor=0.5, random_state=3,
    expected_x_shape=[40, 2],
    expected_y_shape=[40],
    label_min=0, label_max=1)

add("dataset_load_iris",
    expected_x_shape=[150, 4],
    expected_y_shape=[150],
    label_min=0, label_max=2)

add("dataset_load_wine",
    expected_x_shape=[178, 13],
    expected_y_shape=[178],
    label_min=0, label_max=2)

add("dataset_load_breast_cancer",
    expected_x_shape=[569, 30],
    expected_y_shape=[569],
    label_min=0, label_max=1)

# ---------------------------------------------------------------------------
# Metadata and output
# ---------------------------------------------------------------------------

metadata = {
    "sklearn_version": _sklearn_version,
    "numpy_version": np.__version__,
    "python_executable": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "random_state": 42,
    "description": (
        "scikit-learn 1.5.x reference fixtures for ferrotorch-ml conformance. "
        "Tracking issue #840."
    ),
}

out_dir = os.path.dirname(OUTPUT_PATH)
os.makedirs(out_dir, exist_ok=True)

payload = {"metadata": metadata, "fixtures": fixtures}

with open(OUTPUT_PATH, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

print(
    f"Wrote {len(fixtures)} fixtures to {OUTPUT_PATH}  "
    f"(sklearn {_sklearn_version})"
)
