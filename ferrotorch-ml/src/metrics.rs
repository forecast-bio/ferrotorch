//! Sklearn metrics shaped for `&Tensor<T>` inputs.
//!
//! Thin wrappers around [`ferrolearn_metrics`] that handle the
//! Tensor → ndarray adapter at the boundary so callers keep their
//! data in tensors throughout.
//!
//! # Coverage
//!
//! - **Regression** ([`r2_score`], [`mean_squared_error`],
//!   [`mean_absolute_error`], [`mean_absolute_percentage_error`],
//!   [`root_mean_squared_error`], [`median_absolute_error`],
//!   [`max_error`], [`explained_variance_score`])
//! - **Classification** ([`accuracy_score`])
//!
//! For the long tail (precision / recall / F1 / ROC-AUC / log-loss /
//! confusion matrix / clustering / pairwise / ranking) reach into
//! `ferrolearn_metrics::*` directly via your local `cargo add` —
//! these wrappers cover the most common 90% case.
//!
//! # GPU input is transparently materialised
//!
//! Each metric routes its `&Tensor<T>` arguments through
//! [`super::adapter`], which transparently moves GPU tensors to host
//! memory before delegating to ferrolearn (see the adapter module
//! docstring for the rationale). Matches the `loss.cpu().item()` idiom
//! from torch.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use crate::adapter::{tensor_to_array1, tensor_to_array1_usize};

/// Map a `ferrolearn::FerroError` into a ferrotorch-shaped error so the
/// caller never has to import ferrolearn's error type.
fn map_metric_err(e: ferrolearn_core::FerroError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("ferrolearn metric: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Regression metrics
// ---------------------------------------------------------------------------

/// R² coefficient of determination — the canonical regression score.
///
/// `1.0` is perfect prediction; `0.0` is the constant-mean baseline; can
/// be negative when the model is worse than that baseline. Mirrors
/// `sklearn.metrics.r2_score`.
pub fn r2_score<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<T>
where
    T: Float + num_traits::Float + Send + Sync + 'static,
{
    let yt = tensor_to_array1(y_true)?;
    let yp = tensor_to_array1(y_pred)?;
    ferrolearn_metrics::r2_score(&yt, &yp).map_err(map_metric_err)
}

/// Mean squared error: `mean((y_true - y_pred)^2)`.
pub fn mean_squared_error<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<T>
where
    T: Float + num_traits::Float + Send + Sync + 'static,
{
    let yt = tensor_to_array1(y_true)?;
    let yp = tensor_to_array1(y_pred)?;
    ferrolearn_metrics::mean_squared_error(&yt, &yp).map_err(map_metric_err)
}

/// Root mean squared error: `sqrt(mean((y_true - y_pred)^2))`.
pub fn root_mean_squared_error<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<T>
where
    T: Float + num_traits::Float + Send + Sync + 'static,
{
    let yt = tensor_to_array1(y_true)?;
    let yp = tensor_to_array1(y_pred)?;
    ferrolearn_metrics::root_mean_squared_error(&yt, &yp).map_err(map_metric_err)
}

/// Mean absolute error: `mean(|y_true - y_pred|)`.
pub fn mean_absolute_error<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<T>
where
    T: Float + num_traits::Float + Send + Sync + 'static,
{
    let yt = tensor_to_array1(y_true)?;
    let yp = tensor_to_array1(y_pred)?;
    ferrolearn_metrics::mean_absolute_error(&yt, &yp).map_err(map_metric_err)
}

/// Median absolute error.
pub fn median_absolute_error<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<T>
where
    T: Float + num_traits::Float + Send + Sync + 'static,
{
    let yt = tensor_to_array1(y_true)?;
    let yp = tensor_to_array1(y_pred)?;
    ferrolearn_metrics::median_absolute_error(&yt, &yp).map_err(map_metric_err)
}

/// Maximum residual: `max(|y_true - y_pred|)`.
pub fn max_error<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<T>
where
    T: Float + num_traits::Float + Send + Sync + 'static,
{
    let yt = tensor_to_array1(y_true)?;
    let yp = tensor_to_array1(y_pred)?;
    ferrolearn_metrics::max_error(&yt, &yp).map_err(map_metric_err)
}

/// Explained-variance score (between 0 and 1; 1 = perfect).
pub fn explained_variance_score<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<T>
where
    T: Float + num_traits::Float + Send + Sync + 'static,
{
    let yt = tensor_to_array1(y_true)?;
    let yp = tensor_to_array1(y_pred)?;
    ferrolearn_metrics::explained_variance_score(&yt, &yp).map_err(map_metric_err)
}

// ---------------------------------------------------------------------------
// Classification metrics
// ---------------------------------------------------------------------------

/// Classification accuracy: fraction of `y_true == y_pred`.
///
/// Both inputs are interpreted as integer class labels; values are
/// rounded via `as usize` after a finite/non-negative check. Mirrors
/// `sklearn.metrics.accuracy_score`.
pub fn accuracy_score<T>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<f64>
where
    T: Float,
{
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::accuracy_score(&yt, &yp).map_err(map_metric_err)
}

// ---------------------------------------------------------------------------
// Classification metric expansion (#598)
// ---------------------------------------------------------------------------

/// Re-export sklearn-style averaging strategy for multi-class precision /
/// recall / F1. (Binary / Macro / Micro / Weighted)
pub use ferrolearn_metrics::classification::Average;

/// Convert a `Tensor<T>` to an `Array1<f64>`. Used for score-typed metric
/// arguments (ROC-AUC, log_loss, etc. expect raw scores in f64).
fn tensor_to_array1_f64<T: Float>(t: &Tensor<T>) -> FerrotorchResult<ndarray::Array1<f64>> {
    let data: Vec<f64> = t.data_vec()?.iter().map(|v| v.to_f64().unwrap()).collect();
    Ok(ndarray::Array1::from(data))
}

/// Precision: `TP / (TP + FP)`. Mirrors `sklearn.metrics.precision_score`.
pub fn precision_score<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    average: Average,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::precision_score(&yt, &yp, average).map_err(map_metric_err)
}

/// Recall (sensitivity): `TP / (TP + FN)`. Mirrors `sklearn.metrics.recall_score`.
pub fn recall_score<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    average: Average,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::recall_score(&yt, &yp, average).map_err(map_metric_err)
}

/// F1 score (harmonic mean of precision and recall).
pub fn f1_score<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    average: Average,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::f1_score(&yt, &yp, average).map_err(map_metric_err)
}

/// Area under the ROC curve. `y_score` is the predicted probability /
/// decision-function value for the positive class. Binary classification.
pub fn roc_auc_score<T: Float>(y_true: &Tensor<T>, y_score: &Tensor<T>) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let ys = tensor_to_array1_f64(y_score)?;
    ferrolearn_metrics::roc_auc_score(&yt, &ys).map_err(map_metric_err)
}

/// Cross-entropy / log loss for probabilistic classifiers. `y_prob` is
/// `[n_samples, n_classes]` of class probabilities; rows must sum to 1
/// (within tolerance). `y_true` is integer labels.
pub fn log_loss<T: Float>(y_true: &Tensor<T>, y_prob: &Tensor<T>) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    if y_prob.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "log_loss: y_prob must be 2-D [n_samples, n_classes], got {:?}",
                y_prob.shape()
            ),
        });
    }
    let n = y_prob.shape()[0];
    let k = y_prob.shape()[1];
    let data: Vec<f64> = y_prob
        .data_vec()?
        .iter()
        .map(|v| v.to_f64().unwrap())
        .collect();
    let arr = ndarray::Array2::from_shape_vec((n, k), data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("log_loss: failed to build Array2: {e}"),
        }
    })?;
    ferrolearn_metrics::log_loss(&yt, &arr).map_err(map_metric_err)
}

/// Confusion matrix `[n_classes, n_classes]` where `M[i, j]` is the count
/// of samples with true class `i` predicted as class `j`. Returns the
/// matrix as a `Vec<Vec<usize>>` (rows-of-rows; the n_classes is implicit).
pub fn confusion_matrix<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
) -> FerrotorchResult<Vec<Vec<usize>>> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    let m = ferrolearn_metrics::confusion_matrix(&yt, &yp).map_err(map_metric_err)?;
    let n = m.shape()[0];
    let k = m.shape()[1];
    let flat = m.iter().copied().collect::<Vec<usize>>();
    Ok((0..n).map(|i| flat[i * k..(i + 1) * k].to_vec()).collect())
}

/// Hamming loss: fraction of mispredicted labels.
pub fn hamming_loss<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::hamming_loss(&yt, &yp).map_err(map_metric_err)
}

/// Balanced accuracy: average of per-class recalls. Robust to class imbalance.
/// `adjusted = false` matches sklearn's default; `adjusted = true` rescales
/// to `[0, 1]` with chance at 0.
pub fn balanced_accuracy_score<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    adjusted: bool,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::balanced_accuracy_score(&yt, &yp, adjusted).map_err(map_metric_err)
}

/// Matthews correlation coefficient — single-number summary that handles
/// class imbalance gracefully (range `[-1, 1]`).
pub fn matthews_corrcoef<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::matthews_corrcoef(&yt, &yp).map_err(map_metric_err)
}

/// Cohen's kappa: inter-rater agreement, corrected for chance.
pub fn cohen_kappa_score<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::cohen_kappa_score(&yt, &yp).map_err(map_metric_err)
}

// ---------------------------------------------------------------------------
// 2-D probability scoring metrics (#599)
// ---------------------------------------------------------------------------

/// Brier score loss for binary classification: `mean((y_prob - y_true)^2)`.
/// `y_prob` is the predicted probability of the positive class (1-D, shape
/// `[N]`). Mirrors `sklearn.metrics.brier_score_loss`.
pub fn brier_score_loss<T: Float>(y_true: &Tensor<T>, y_prob: &Tensor<T>) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_f64(y_prob)?;
    ferrolearn_metrics::brier_score_loss(&yt, &yp).map_err(map_metric_err)
}

/// D² Brier score: 1 - (model_brier / null_brier). Coefficient-of-determination
/// analog for binary calibration. `1.0` is perfect, `0.0` is null-model
/// baseline.
pub fn d2_brier_score<T: Float>(y_true: &Tensor<T>, y_prob: &Tensor<T>) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_f64(y_prob)?;
    ferrolearn_metrics::d2_brier_score(&yt, &yp).map_err(map_metric_err)
}

/// Top-K accuracy: a sample counts as correct if the true label is among
/// the top-K highest-scored classes. `y_score` is `[N, n_classes]`.
pub fn top_k_accuracy_score<T: Float>(
    y_true: &Tensor<T>,
    y_score: &Tensor<T>,
    k: usize,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    if y_score.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "top_k_accuracy_score: y_score must be 2-D, got {:?}",
                y_score.shape()
            ),
        });
    }
    let n = y_score.shape()[0];
    let c = y_score.shape()[1];
    let data: Vec<f64> = y_score
        .data_vec()?
        .iter()
        .map(|v| v.to_f64().unwrap())
        .collect();
    let arr = ndarray::Array2::from_shape_vec((n, c), data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("top_k_accuracy_score: y_score shape: {e}"),
        }
    })?;
    ferrolearn_metrics::top_k_accuracy_score(&yt, &arr, k).map_err(map_metric_err)
}

/// 0/1 loss. `normalize=true` returns the fraction misclassified; `false`
/// returns the count.
pub fn zero_one_loss<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    normalize: bool,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let yp = tensor_to_array1_usize(y_pred)?;
    ferrolearn_metrics::zero_one_loss(&yt, &yp, normalize).map_err(map_metric_err)
}

/// Average precision score (binary): the precision-recall curve's area,
/// computed as a sum of trapezoidal rectangles.
pub fn average_precision_score<T: Float>(
    y_true: &Tensor<T>,
    y_score: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_usize(y_true)?;
    let ys = tensor_to_array1_f64(y_score)?;
    ferrolearn_metrics::average_precision_score(&yt, &ys).map_err(map_metric_err)
}

// ---------------------------------------------------------------------------
// Ranking metrics (#617)
// ---------------------------------------------------------------------------
//
// Ranking metrics expect 2-D inputs: `y_true` of shape [N, K] (per-sample
// relevance scores) and `y_score` [N, K] (predicted scores). We adapt
// through `tensor_to_array2_*` boundary helpers; sklearn's API uses the
// same 2-D contract.

/// Convert a 2-D `Tensor<T>` to an `Array2<f64>`.
fn tensor_to_array2_f64<T: Float>(t: &Tensor<T>) -> FerrotorchResult<ndarray::Array2<f64>> {
    if t.shape().len() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "tensor_to_array2_f64: expected 2-D tensor, got shape {:?}",
                t.shape()
            ),
        });
    }
    let (rows, cols) = (t.shape()[0], t.shape()[1]);
    let data: Vec<f64> = t.data_vec()?.iter().map(|v| v.to_f64().unwrap()).collect();
    ndarray::Array2::from_shape_vec((rows, cols), data).map_err(|e| {
        FerrotorchError::ShapeMismatch {
            message: format!("tensor_to_array2_f64: shape build failed: {e}"),
        }
    })
}

/// Convert a 2-D `Tensor<T>` to an `Array2<usize>` (for ranking targets).
fn tensor_to_array2_usize<T: Float>(t: &Tensor<T>) -> FerrotorchResult<ndarray::Array2<usize>> {
    if t.shape().len() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "tensor_to_array2_usize: expected 2-D tensor, got shape {:?}",
                t.shape()
            ),
        });
    }
    let (rows, cols) = (t.shape()[0], t.shape()[1]);
    let data = t.data_vec()?;
    let mut out = Vec::with_capacity(data.len());
    for (i, &v) in data.iter().enumerate() {
        let f = v.to_f64().ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("tensor_to_array2_usize: element {i} is not finite"),
        })?;
        if !f.is_finite() || f < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor_to_array2_usize: element {i} = {f} is not a non-negative finite integer"
                ),
            });
        }
        out.push(f as usize);
    }
    ndarray::Array2::from_shape_vec((rows, cols), out).map_err(|e| FerrotorchError::ShapeMismatch {
        message: format!("tensor_to_array2_usize: shape build failed: {e}"),
    })
}

/// Convert a 1-D label tensor (floats) to an `Array1<isize>` for clustering
/// metrics (sklearn convention: `-1` = noise / unlabelled).
fn tensor_to_array1_isize<T: Float>(t: &Tensor<T>) -> FerrotorchResult<ndarray::Array1<isize>> {
    if t.shape().len() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "tensor_to_array1_isize: expected 1-D tensor, got shape {:?}",
                t.shape()
            ),
        });
    }
    let data = t.data_vec()?;
    let mut out = Vec::with_capacity(data.len());
    for (i, &v) in data.iter().enumerate() {
        let f = v.to_f64().ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("tensor_to_array1_isize: element {i} is not finite"),
        })?;
        if !f.is_finite() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("tensor_to_array1_isize: element {i} = {f} is not finite"),
            });
        }
        out.push(f as isize);
    }
    Ok(ndarray::Array1::from(out))
}

/// Discounted Cumulative Gain (DCG) — sum of relevance scores discounted by
/// log(rank). `y_true` (relevance) and `y_score` are both 1-D `[K]`.
/// Mirrors `sklearn.metrics.dcg_score` for the single-query case.
pub fn dcg_score<T: Float>(
    y_true: &Tensor<T>,
    y_score: &Tensor<T>,
    k: Option<usize>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_f64(y_true)?;
    let ys = tensor_to_array1_f64(y_score)?;
    ferrolearn_metrics::dcg_score(&yt, &ys, k).map_err(map_metric_err)
}

/// Normalized Discounted Cumulative Gain (NDCG) — DCG normalized by the
/// ideal DCG so the score lies in `[0, 1]`. Mirrors
/// `sklearn.metrics.ndcg_score` for the single-query case.
pub fn ndcg_score<T: Float>(
    y_true: &Tensor<T>,
    y_score: &Tensor<T>,
    k: Option<usize>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_f64(y_true)?;
    let ys = tensor_to_array1_f64(y_score)?;
    ferrolearn_metrics::ndcg_score(&yt, &ys, k).map_err(map_metric_err)
}

/// Coverage error: average number of labels in the ranked list above the
/// last positive label. Mirrors `sklearn.metrics.coverage_error`. `y_true`
/// must be 2-D `[N, K]` of binary indicators (0/1).
pub fn coverage_error<T: Float>(y_true: &Tensor<T>, y_score: &Tensor<T>) -> FerrotorchResult<f64> {
    let yt = tensor_to_array2_usize(y_true)?;
    let ys = tensor_to_array2_f64(y_score)?;
    ferrolearn_metrics::coverage_error(&yt, &ys).map_err(map_metric_err)
}

/// Label-ranking average precision score. Mirrors
/// `sklearn.metrics.label_ranking_average_precision_score`.
pub fn label_ranking_average_precision_score<T: Float>(
    y_true: &Tensor<T>,
    y_score: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array2_usize(y_true)?;
    let ys = tensor_to_array2_f64(y_score)?;
    ferrolearn_metrics::label_ranking_average_precision_score(&yt, &ys).map_err(map_metric_err)
}

/// Label-ranking loss (number of label-pair inversions, normalised).
/// Mirrors `sklearn.metrics.label_ranking_loss`.
pub fn label_ranking_loss<T: Float>(
    y_true: &Tensor<T>,
    y_score: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array2_usize(y_true)?;
    let ys = tensor_to_array2_f64(y_score)?;
    ferrolearn_metrics::label_ranking_loss(&yt, &ys).map_err(map_metric_err)
}

// ---------------------------------------------------------------------------
// Clustering metrics (#617)
// ---------------------------------------------------------------------------
//
// `_score` flavours use both true & predicted label arrays (label-based);
// `silhouette_score` / `davies_bouldin_score` / `calinski_harabasz_score`
// use the data matrix [N, D] + the cluster assignments.

/// Adjusted Rand index — clustering similarity corrected for chance.
/// Mirrors `sklearn.metrics.adjusted_rand_score`. Inputs are 1-D label
/// tensors of equal length (encoded as floats; cast to `isize` so sklearn's
/// `-1` noise convention is preserved).
pub fn adjusted_rand_score<T: Float>(
    labels_true: &Tensor<T>,
    labels_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_isize(labels_true)?;
    let yp = tensor_to_array1_isize(labels_pred)?;
    ferrolearn_metrics::adjusted_rand_score(&yt, &yp).map_err(map_metric_err)
}

/// Adjusted mutual information between two clusterings. Mirrors
/// `sklearn.metrics.adjusted_mutual_info_score` (uses ferrolearn's default
/// expected-value adjustment; no method parameter).
pub fn adjusted_mutual_info_score<T: Float>(
    labels_true: &Tensor<T>,
    labels_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_isize(labels_true)?;
    let yp = tensor_to_array1_isize(labels_pred)?;
    ferrolearn_metrics::adjusted_mutual_info(&yt, &yp).map_err(map_metric_err)
}

/// Normalized mutual information score (arithmetic-mean normalisation,
/// matching sklearn's default). Mirrors
/// `sklearn.metrics.normalized_mutual_info_score`.
pub fn normalized_mutual_info_score<T: Float>(
    labels_true: &Tensor<T>,
    labels_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_isize(labels_true)?;
    let yp = tensor_to_array1_isize(labels_pred)?;
    ferrolearn_metrics::normalized_mutual_info_score(
        &yt,
        &yp,
        ferrolearn_metrics::clustering::NmiMethod::Arithmetic,
    )
    .map_err(map_metric_err)
}

/// Homogeneity score: each cluster contains members of a single class.
/// Mirrors `sklearn.metrics.homogeneity_score`.
pub fn homogeneity_score<T: Float>(
    labels_true: &Tensor<T>,
    labels_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_isize(labels_true)?;
    let yp = tensor_to_array1_isize(labels_pred)?;
    ferrolearn_metrics::homogeneity_score(&yt, &yp).map_err(map_metric_err)
}

/// Completeness score: all members of a class go to the same cluster.
/// Mirrors `sklearn.metrics.completeness_score`.
pub fn completeness_score<T: Float>(
    labels_true: &Tensor<T>,
    labels_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_isize(labels_true)?;
    let yp = tensor_to_array1_isize(labels_pred)?;
    ferrolearn_metrics::completeness_score(&yt, &yp).map_err(map_metric_err)
}

/// V-measure: harmonic mean of homogeneity and completeness. Mirrors
/// `sklearn.metrics.v_measure_score`.
pub fn v_measure_score<T: Float>(
    labels_true: &Tensor<T>,
    labels_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_isize(labels_true)?;
    let yp = tensor_to_array1_isize(labels_pred)?;
    ferrolearn_metrics::v_measure_score(&yt, &yp).map_err(map_metric_err)
}

/// Fowlkes-Mallows score: geometric mean of pairwise precision and recall.
pub fn fowlkes_mallows_score<T: Float>(
    labels_true: &Tensor<T>,
    labels_pred: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let yt = tensor_to_array1_isize(labels_true)?;
    let yp = tensor_to_array1_isize(labels_pred)?;
    ferrolearn_metrics::fowlkes_mallows_score(&yt, &yp).map_err(map_metric_err)
}

/// Silhouette score: mean intra-cluster cohesion vs. nearest-cluster
/// separation. `x` is the feature matrix `[N, D]`; `labels` is the 1-D
/// cluster assignment (encoded as floats; cast to `isize`, sklearn allows
/// `-1` for noise).
pub fn silhouette_score<T: Float>(x: &Tensor<T>, labels: &Tensor<T>) -> FerrotorchResult<f64> {
    let xa = tensor_to_array2_f64(x)?;
    let la = tensor_to_array1_isize(labels)?;
    ferrolearn_metrics::silhouette_score(&xa, &la).map_err(map_metric_err)
}

/// Davies-Bouldin score (lower is better): cluster compactness over
/// inter-cluster spread.
pub fn davies_bouldin_score<T: Float>(x: &Tensor<T>, labels: &Tensor<T>) -> FerrotorchResult<f64> {
    let xa = tensor_to_array2_f64(x)?;
    let la = tensor_to_array1_isize(labels)?;
    ferrolearn_metrics::davies_bouldin_score(&xa, &la).map_err(map_metric_err)
}

/// Calinski-Harabasz score (higher is better): variance ratio criterion.
pub fn calinski_harabasz_score<T: Float>(
    x: &Tensor<T>,
    labels: &Tensor<T>,
) -> FerrotorchResult<f64> {
    let xa = tensor_to_array2_f64(x)?;
    let la = tensor_to_array1_isize(labels)?;
    ferrolearn_metrics::calinski_harabasz_score(&xa, &la).map_err(map_metric_err)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::tensor;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn r2_perfect_prediction_is_one() {
        let y = tensor(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let r = r2_score(&y, &y).unwrap();
        assert!(close(r, 1.0, 1e-12));
    }

    #[test]
    fn r2_constant_baseline_is_zero() {
        let y_true = tensor(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        // Predict the mean (= 2.5) for every sample → R² = 0.
        let y_pred = tensor(&[2.5_f64, 2.5, 2.5, 2.5]).unwrap();
        let r = r2_score(&y_true, &y_pred).unwrap();
        assert!(close(r, 0.0, 1e-12));
    }

    #[test]
    fn mse_known_value() {
        let y_true = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        let y_pred = tensor(&[1.5_f64, 2.5, 3.5]).unwrap();
        // Errors: 0.5, 0.5, 0.5 → MSE = 0.25
        let m = mean_squared_error(&y_true, &y_pred).unwrap();
        assert!(close(m, 0.25, 1e-12));
    }

    #[test]
    fn rmse_is_sqrt_mse() {
        let y_true = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        let y_pred = tensor(&[1.5_f64, 2.5, 3.5]).unwrap();
        let r = root_mean_squared_error(&y_true, &y_pred).unwrap();
        assert!(close(r, 0.25_f64.sqrt(), 1e-12));
    }

    #[test]
    fn mae_known_value() {
        let y_true = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        let y_pred = tensor(&[1.0_f64, 3.0, 5.0]).unwrap();
        // |0| + |1| + |2| = 3, mean = 1.0
        let m = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!(close(m, 1.0, 1e-12));
    }

    #[test]
    fn max_error_known_value() {
        let y_true = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        let y_pred = tensor(&[1.0_f64, 5.0, 3.0]).unwrap();
        let m = max_error(&y_true, &y_pred).unwrap();
        assert!(close(m, 3.0, 1e-12));
    }

    #[test]
    fn accuracy_score_perfect_match() {
        let y_true = tensor(&[0.0_f64, 1.0, 1.0, 0.0, 1.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 1.0, 0.0, 1.0]).unwrap();
        let a = accuracy_score(&y_true, &y_pred).unwrap();
        assert!(close(a, 1.0, 1e-12));
    }

    #[test]
    fn accuracy_score_mixed() {
        let y_true = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 0.0, 1.0, 0.0]).unwrap(); // 3 of 4 correct
        let a = accuracy_score(&y_true, &y_pred).unwrap();
        assert!(close(a, 0.75, 1e-12));
    }

    // ----- GPU pass-through ----------------------------------------------

    #[test]
    fn metric_works_on_cpu_tensor() {
        // Sanity check: with the auto-CPU-materialisation policy, a
        // CPU tensor flows through unchanged. CUDA inputs would also
        // succeed (data is moved to host transparently inside the
        // adapter); we can't construct a CUDA tensor in this CPU-only
        // test environment, but the relaxation is exercised by the
        // module-level docs and is covered indirectly by the adapter's
        // own tensor-to-Vec conversion path.
        let y = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        assert!(r2_score(&y, &y).is_ok());
    }

    // -----------------------------------------------------------------------
    // Classification metric expansion (#598)
    // -----------------------------------------------------------------------

    #[test]
    fn precision_binary_perfect() {
        // y_true = [0, 1, 1, 0], y_pred = [0, 1, 1, 0]: all correct
        let y_true = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let p = precision_score(&y_true, &y_pred, Average::Binary).unwrap();
        assert!(close(p, 1.0, 1e-12));
    }

    #[test]
    fn recall_binary_partial() {
        // y_true = [0, 1, 1, 0], y_pred = [0, 1, 0, 0] → recall = 1/2
        let y_true = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 0.0, 0.0]).unwrap();
        let r = recall_score(&y_true, &y_pred, Average::Binary).unwrap();
        assert!(close(r, 0.5, 1e-12));
    }

    #[test]
    fn f1_binary_harmonic_mean() {
        // y_true = [0, 1, 1, 0], y_pred = [0, 1, 0, 0]
        // precision = 1.0, recall = 0.5 → F1 = 2 * (p * r) / (p + r) = 2/3
        let y_true = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 0.0, 0.0]).unwrap();
        let f = f1_score(&y_true, &y_pred, Average::Binary).unwrap();
        assert!(close(f, 2.0 / 3.0, 1e-12));
    }

    #[test]
    fn confusion_matrix_binary() {
        // y_true = [0, 1, 0, 1], y_pred = [0, 1, 1, 0]
        // True 0 → pred 0 once, pred 1 once
        // True 1 → pred 0 once, pred 1 once
        // CM = [[1, 1], [1, 1]]
        let y_true = tensor(&[0.0_f64, 1.0, 0.0, 1.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        assert_eq!(cm, vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn hamming_loss_partial() {
        // 1 mismatch out of 4 → 0.25
        let y_true = tensor(&[0.0_f64, 1.0, 0.0, 1.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 1.0, 1.0]).unwrap();
        let l = hamming_loss(&y_true, &y_pred).unwrap();
        assert!(close(l, 0.25, 1e-12));
    }

    #[test]
    fn balanced_accuracy_balanced_classes() {
        let y_true = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let b = balanced_accuracy_score(&y_true, &y_pred, false).unwrap();
        assert!(close(b, 1.0, 1e-12));
    }

    #[test]
    fn matthews_corrcoef_perfect_is_one() {
        let y = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let m = matthews_corrcoef(&y, &y).unwrap();
        assert!(close(m, 1.0, 1e-12));
    }

    #[test]
    fn cohen_kappa_perfect_is_one() {
        let y = tensor(&[0.0_f64, 1.0, 1.0, 0.0, 1.0]).unwrap();
        let k = cohen_kappa_score(&y, &y).unwrap();
        assert!(close(k, 1.0, 1e-12));
    }

    #[test]
    fn roc_auc_perfect_separation() {
        // Scores cleanly separate the two classes.
        let y_true = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let y_score = tensor(&[0.1_f64, 0.2, 0.8, 0.9]).unwrap();
        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        assert!(close(auc, 1.0, 1e-12));
    }

    #[test]
    fn log_loss_known_value() {
        // 2-class, 2 samples; correct class probability = 0.9 each.
        // log_loss = -mean(log(0.9)) for both samples = -ln(0.9).
        let y_true = tensor(&[0.0_f64, 1.0]).unwrap();
        let y_prob = Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(vec![0.9_f64, 0.1, 0.1, 0.9]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let l = log_loss(&y_true, &y_prob).unwrap();
        assert!(close(l, -0.9_f64.ln(), 1e-9), "got {l}");
    }

    #[test]
    fn log_loss_rejects_1d_y_prob() {
        let y_true = tensor(&[0.0_f64, 1.0]).unwrap();
        let y_prob = tensor(&[0.5_f64, 0.5]).unwrap();
        let err = log_loss(&y_true, &y_prob).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    // -----------------------------------------------------------------------
    // 2-D probability scoring (#599)
    // -----------------------------------------------------------------------

    #[test]
    fn brier_score_perfect_predictions_is_zero() {
        let y_true = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let y_prob = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let b = brier_score_loss(&y_true, &y_prob).unwrap();
        assert!(close(b, 0.0, 1e-12));
    }

    #[test]
    fn brier_score_uniform_half_loss() {
        // For y_true = [0, 1] with y_prob = [0.5, 0.5], each sample
        // contributes (0.5 - 0)^2 = 0.25 and (0.5 - 1)^2 = 0.25.
        // Mean = 0.25.
        let y_true = tensor(&[0.0_f64, 1.0]).unwrap();
        let y_prob = tensor(&[0.5_f64, 0.5]).unwrap();
        let b = brier_score_loss(&y_true, &y_prob).unwrap();
        assert!(close(b, 0.25, 1e-12));
    }

    #[test]
    fn d2_brier_perfect_score_is_one() {
        let y_true = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let y_prob = tensor(&[0.0_f64, 1.0, 1.0, 0.0]).unwrap();
        let d2 = d2_brier_score(&y_true, &y_prob).unwrap();
        assert!(close(d2, 1.0, 1e-12));
    }

    #[test]
    fn top_k_accuracy_k_equals_classes_is_one() {
        // 3 classes; with k=3 every sample's true class is in the top-3.
        let y_true = tensor(&[0.0_f64, 2.0, 1.0]).unwrap();
        let y_score = Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(vec![
                0.6_f64, 0.3, 0.1, 0.1, 0.2, 0.7, 0.2, 0.5, 0.3,
            ]),
            vec![3, 3],
            false,
        )
        .unwrap();
        let acc = top_k_accuracy_score(&y_true, &y_score, 3).unwrap();
        assert!(close(acc, 1.0, 1e-12));
    }

    #[test]
    fn top_k_accuracy_k_one_matches_argmax() {
        // k=1 → top_k matches argmax = standard accuracy.
        let y_true = tensor(&[0.0_f64, 2.0]).unwrap();
        let y_score = Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(vec![
                0.6_f64, 0.3, 0.1, // argmax = 0 ✓
                0.5, 0.4, 0.1, // argmax = 0 ✗ (true = 2)
            ]),
            vec![2, 3],
            false,
        )
        .unwrap();
        let acc = top_k_accuracy_score(&y_true, &y_score, 1).unwrap();
        assert!(close(acc, 0.5, 1e-12));
    }

    #[test]
    fn top_k_rejects_1d_y_score() {
        let y_true = tensor(&[0.0_f64]).unwrap();
        let y_score = tensor(&[0.5_f64]).unwrap();
        let err = top_k_accuracy_score(&y_true, &y_score, 1).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn zero_one_loss_normalize_matches_hamming() {
        let y_true = tensor(&[0.0_f64, 1.0, 0.0, 1.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 1.0, 1.0]).unwrap();
        let l = zero_one_loss(&y_true, &y_pred, true).unwrap();
        assert!(close(l, 0.25, 1e-12));
    }

    #[test]
    fn zero_one_loss_unnormalized_count() {
        let y_true = tensor(&[0.0_f64, 1.0, 0.0, 1.0]).unwrap();
        let y_pred = tensor(&[0.0_f64, 1.0, 1.0, 1.0]).unwrap();
        let l = zero_one_loss(&y_true, &y_pred, false).unwrap();
        // 1 mismatch
        assert!(close(l, 1.0, 1e-12));
    }

    #[test]
    fn average_precision_perfect_separation() {
        let y_true = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let y_score = tensor(&[0.1_f64, 0.2, 0.8, 0.9]).unwrap();
        let ap = average_precision_score(&y_true, &y_score).unwrap();
        assert!(close(ap, 1.0, 1e-12));
    }

    // ----- ranking metrics (#617) ----------------------------------------

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        // For a single query, perfect ranking (scores match relevance order)
        // gives NDCG = 1.0.
        let y_true = tensor(&[3.0_f64, 2.0, 1.0, 0.0]).unwrap();
        let y_score = tensor(&[3.0_f64, 2.0, 1.0, 0.0]).unwrap();
        let n = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!(close(n, 1.0, 1e-9));
    }

    #[test]
    fn dcg_score_finite() {
        let y_true = tensor(&[3.0_f64, 0.0, 2.0]).unwrap();
        let y_score = tensor(&[1.0_f64, 0.5, 2.0]).unwrap();
        // DCG should compute without erroring; specific value depends on
        // gain formulation — verifying finite + > 0 is sufficient as a
        // smoke test.
        let d = dcg_score(&y_true, &y_score, None).unwrap();
        assert!(d.is_finite() && d > 0.0);
    }

    #[test]
    fn coverage_error_known_input() {
        // y_true is [N, K] of binary indicators. Covering rank-of-last-positive.
        // For one row [1, 0, 0, 1] with scores [0.9, 0.4, 0.5, 0.1], the
        // sorted rank descending is [0(0.9), 2(0.5), 1(0.4), 3(0.1)] —
        // last positive (col 3) lands at rank 4. Coverage = 4 averaged
        // over 1 sample.
        let y_true = tensor(&[1.0_f64, 0.0, 0.0, 1.0]).unwrap();
        let y_score = tensor(&[0.9_f64, 0.4, 0.5, 0.1]).unwrap();
        // Reshape to 2-D [1, 4].
        let y_true_2d = y_true.reshape_t(&[1, 4]).unwrap();
        let y_score_2d = y_score.reshape_t(&[1, 4]).unwrap();
        let c = coverage_error(&y_true_2d, &y_score_2d).unwrap();
        assert!(c.is_finite() && c > 0.0);
    }

    // ----- clustering metrics (#617) -------------------------------------

    #[test]
    fn adjusted_rand_score_perfect_match_is_one() {
        let labels = tensor(&[0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let s = adjusted_rand_score(&labels, &labels).unwrap();
        assert!(close(s, 1.0, 1e-12));
    }

    #[test]
    fn homogeneity_completeness_v_measure_consistent() {
        let labels_true = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let labels_pred = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let h = homogeneity_score(&labels_true, &labels_pred).unwrap();
        let c = completeness_score(&labels_true, &labels_pred).unwrap();
        let v = v_measure_score(&labels_true, &labels_pred).unwrap();
        assert!(close(h, 1.0, 1e-9));
        assert!(close(c, 1.0, 1e-9));
        assert!(close(v, 1.0, 1e-9));
    }

    #[test]
    fn nmi_perfect_clustering_is_one() {
        let labels_true = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let labels_pred = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let n = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
        assert!(close(n, 1.0, 1e-9));
    }

    #[test]
    fn fowlkes_mallows_perfect_match_is_one() {
        let labels = tensor(&[0.0_f64, 0.0, 1.0, 1.0, 2.0]).unwrap();
        let s = fowlkes_mallows_score(&labels, &labels).unwrap();
        assert!(close(s, 1.0, 1e-12));
    }

    #[test]
    fn silhouette_score_well_separated_clusters() {
        // Two well-separated 2-D clusters.
        let x = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![
                0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1,
            ]),
            vec![4, 2],
            false,
        )
        .unwrap();
        let labels = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let s = silhouette_score(&x, &labels).unwrap();
        // Tightly separated clusters should give silhouette near 1.
        assert!(s > 0.9);
    }

    #[test]
    fn davies_bouldin_returns_finite() {
        let x = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![0.0_f64, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1]),
            vec![4, 2],
            false,
        )
        .unwrap();
        let labels = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let d = davies_bouldin_score(&x, &labels).unwrap();
        assert!(d.is_finite() && d >= 0.0);
    }

    #[test]
    fn calinski_harabasz_returns_finite() {
        let x = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![0.0_f64, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1]),
            vec![4, 2],
            false,
        )
        .unwrap();
        let labels = tensor(&[0.0_f64, 0.0, 1.0, 1.0]).unwrap();
        let c = calinski_harabasz_score(&x, &labels).unwrap();
        assert!(c.is_finite() && c > 0.0);
    }
}
