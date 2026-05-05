//! Dataset generators returning `Tensor` pairs.
//!
//! Wraps the most common [`ferrolearn_datasets`] generators with a
//! `(Tensor<T>, Tensor<T>)` return so they slot directly into a
//! tensor-shaped pipeline. The returned tensors live on
//! [`Device::Cpu`](ferrotorch_core::Device::Cpu); move them to GPU
//! explicitly with `.to(...)` if you need them there.
//!
//! # Coverage
//!
//! Synthetic feature/label generators:
//! - [`make_classification`], [`make_regression`], [`make_blobs`],
//!   [`make_moons`], [`make_circles`]
//!
//! Toy datasets (small, hand-curated):
//! - [`load_iris`], [`load_wine`], [`load_breast_cancer`]
//!
//! All return `(X: Tensor<F> [n_samples, n_features], y: Tensor<F>
//! [n_samples])`. For classification problems `y` carries integer class
//! labels encoded as floats — round-trip via
//! [`super::adapter::tensor_to_array1_usize`] to feed
//! `ferrolearn-metrics::accuracy_score`.
//!
//! For the longer tail (`make_friedman1/2/3`, `make_low_rank_matrix`,
//! `make_spd_matrix`, etc.) call `ferrolearn_datasets::*` directly and
//! convert via [`super::adapter`].

use ndarray::{Array1, Array2};

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use crate::adapter::{array1_to_tensor, array1_usize_to_tensor, array2_to_tensor};

fn map_dataset_err(e: ferrolearn_core::FerroError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("ferrolearn dataset: {e}"),
    }
}

/// Convert a `(Array2<F>, Array1<usize>)` ferrolearn classification
/// output to `(Tensor<F>, Tensor<F>)` with class labels encoded as
/// floats in `y`.
fn pack_xy_classify<F: Float>(
    xy: (Array2<F>, Array1<usize>),
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)> {
    let (x_arr, y_arr) = xy;
    Ok((array2_to_tensor(x_arr)?, array1_usize_to_tensor(y_arr)?))
}

/// Convert a `(Array2<F>, Array1<F>)` ferrolearn regression output to
/// `(Tensor<F>, Tensor<F>)`.
fn pack_xy_regress<F: Float>(
    xy: (Array2<F>, Array1<F>),
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)> {
    let (x_arr, y_arr) = xy;
    Ok((array2_to_tensor(x_arr)?, array1_to_tensor(y_arr)?))
}

// ---------------------------------------------------------------------------
// Synthetic generators
// ---------------------------------------------------------------------------

/// Generate a random `n`-class classification problem.
///
/// Returns `(X: [n_samples, n_features], y: [n_samples])` where `y`
/// carries integer class labels encoded as floats. Mirrors
/// `sklearn.datasets.make_classification`.
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::make_classification;
///
/// let (x, y) = make_classification::<f64>(50, 4, 3, Some(42)).unwrap();
/// assert_eq!(x.shape(), &[50, 4]);
/// assert_eq!(y.shape(), &[50]);
/// // Labels are integer-valued in [0, n_classes).
/// for &v in y.data().unwrap() {
///     assert!((0.0..3.0).contains(&v));
///     assert_eq!(v.fract(), 0.0);
/// }
/// ```
pub fn make_classification<F>(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::make_classification::<F>(
        n_samples,
        n_features,
        n_classes,
        random_state,
    )
    .map_err(map_dataset_err)?;
    pack_xy_classify(xy)
}

/// Generate a random regression problem with `n_informative` informative
/// features (the rest are noise) and additive Gaussian noise of
/// magnitude `noise`.
///
/// Mirrors `sklearn.datasets.make_regression`.
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::make_regression;
///
/// // 80 samples, 5 features (3 informative + 2 noise), small noise.
/// let (x, y) = make_regression::<f64>(80, 5, 3, 0.1_f64, Some(7)).unwrap();
/// assert_eq!(x.shape(), &[80, 5]);
/// assert_eq!(y.shape(), &[80]);
/// // Targets are real-valued, so just check they're finite.
/// for &v in y.data().unwrap() {
///     assert!(v.is_finite());
/// }
/// ```
pub fn make_regression<F>(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: F,
    random_state: Option<u64>,
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::make_regression::<F>(
        n_samples,
        n_features,
        n_informative,
        noise,
        random_state,
    )
    .map_err(map_dataset_err)?;
    pack_xy_regress(xy)
}

/// Generate isotropic Gaussian blobs for clustering. `cluster_std`
/// controls the per-cluster spread (1.0 is a reasonable default).
///
/// Mirrors `sklearn.datasets.make_blobs`.
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::make_blobs;
///
/// let (x, y) = make_blobs::<f64>(60, 2, 3, 1.0_f64, Some(1)).unwrap();
/// assert_eq!(x.shape(), &[60, 2]);
/// assert_eq!(y.shape(), &[60]);
/// // Cluster IDs span [0, centers).
/// for &v in y.data().unwrap() {
///     assert!((0.0..3.0).contains(&v));
/// }
/// ```
pub fn make_blobs<F>(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: F,
    random_state: Option<u64>,
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::make_blobs::<F>(
        n_samples,
        n_features,
        centers,
        cluster_std,
        random_state,
    )
    .map_err(map_dataset_err)?;
    pack_xy_classify(xy)
}

/// Generate the two-moons binary-classification toy dataset.
///
/// Mirrors `sklearn.datasets.make_moons`.
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::make_moons;
///
/// let (x, y) = make_moons::<f64>(40, 0.05_f64, Some(2)).unwrap();
/// assert_eq!(x.shape(), &[40, 2]);
/// assert_eq!(y.shape(), &[40]);
/// // Binary task: labels are exactly 0 or 1.
/// for &v in y.data().unwrap() {
///     assert!(v == 0.0 || v == 1.0);
/// }
/// ```
pub fn make_moons<F>(
    n_samples: usize,
    noise: F,
    random_state: Option<u64>,
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::make_moons::<F>(n_samples, noise, random_state)
        .map_err(map_dataset_err)?;
    pack_xy_classify(xy)
}

/// Generate the concentric-circles binary-classification toy dataset.
///
/// Mirrors `sklearn.datasets.make_circles`.
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::make_circles;
///
/// let (x, y) = make_circles::<f64>(40, 0.02_f64, 0.5_f64, Some(3)).unwrap();
/// assert_eq!(x.shape(), &[40, 2]);
/// assert_eq!(y.shape(), &[40]);
/// // Binary task: labels are exactly 0 or 1.
/// for &v in y.data().unwrap() {
///     assert!(v == 0.0 || v == 1.0);
/// }
/// ```
pub fn make_circles<F>(
    n_samples: usize,
    noise: F,
    factor: F,
    random_state: Option<u64>,
) -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::make_circles::<F>(n_samples, noise, factor, random_state)
        .map_err(map_dataset_err)?;
    pack_xy_classify(xy)
}

// ---------------------------------------------------------------------------
// Toy datasets
// ---------------------------------------------------------------------------

/// Iris flower classification (150 samples × 4 features × 3 classes).
///
/// The dataset is shipped inline by `ferrolearn-datasets` (no network
/// or filesystem access).
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::load_iris;
///
/// let (x, y) = load_iris::<f64>().unwrap();
/// assert_eq!(x.shape(), &[150, 4]);
/// assert_eq!(y.shape(), &[150]);
/// // Three classes: labels in {0, 1, 2}.
/// for &v in y.data().unwrap() {
///     assert!((0.0..3.0).contains(&v));
/// }
/// ```
pub fn load_iris<F>() -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::load_iris::<F>().map_err(map_dataset_err)?;
    pack_xy_classify(xy)
}

/// Wine cultivar classification (178 samples × 13 features × 3 classes).
///
/// The dataset is shipped inline by `ferrolearn-datasets` (no network
/// or filesystem access).
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::load_wine;
///
/// let (x, y) = load_wine::<f64>().unwrap();
/// assert_eq!(x.shape(), &[178, 13]);
/// assert_eq!(y.shape(), &[178]);
/// // Three classes: labels in {0, 1, 2}.
/// for &v in y.data().unwrap() {
///     assert!((0.0..3.0).contains(&v));
/// }
/// ```
pub fn load_wine<F>() -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::load_wine::<F>().map_err(map_dataset_err)?;
    pack_xy_classify(xy)
}

/// Breast-cancer Wisconsin diagnostic dataset (569 samples × 30 features
/// × 2 classes).
///
/// The dataset is shipped inline by `ferrolearn-datasets` (no network
/// or filesystem access).
///
/// # Examples
///
/// ```
/// use ferrotorch_ml::datasets::load_breast_cancer;
///
/// let (x, y) = load_breast_cancer::<f64>().unwrap();
/// assert_eq!(x.shape(), &[569, 30]);
/// assert_eq!(y.shape(), &[569]);
/// // Binary task: labels are exactly 0 or 1.
/// for &v in y.data().unwrap() {
///     assert!(v == 0.0 || v == 1.0);
/// }
/// ```
pub fn load_breast_cancer<F>() -> FerrotorchResult<(Tensor<F>, Tensor<F>)>
where
    F: Float + num_traits::Float + Send + Sync + 'static,
{
    let xy = ferrolearn_datasets::load_breast_cancer::<F>().map_err(map_dataset_err)?;
    pack_xy_classify(xy)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_classification_returns_correct_shapes() {
        let (x, y) = make_classification::<f64>(100, 5, 3, Some(42)).unwrap();
        assert_eq!(x.shape(), &[100, 5]);
        assert_eq!(y.shape(), &[100]);
    }

    #[test]
    fn make_classification_y_labels_are_in_range() {
        let (_x, y) = make_classification::<f64>(50, 4, 3, Some(0)).unwrap();
        for &v in y.data().unwrap() {
            assert!((0.0..3.0).contains(&v), "label {v} out of range [0, 3)");
            assert!(v.fract() == 0.0, "label {v} is not integer-valued");
        }
    }

    #[test]
    fn make_regression_returns_correct_shapes() {
        // 80 samples, 5 features (3 informative + 2 noise), small noise.
        let (x, y) = make_regression::<f64>(80, 5, 3, 0.1, Some(7)).unwrap();
        assert_eq!(x.shape(), &[80, 5]);
        assert_eq!(y.shape(), &[80]);
    }

    #[test]
    fn make_blobs_three_centers_two_features() {
        let (x, y) = make_blobs::<f64>(60, 2, 3, 1.0, Some(1)).unwrap();
        assert_eq!(x.shape(), &[60, 2]);
        assert_eq!(y.shape(), &[60]);
        // Labels span [0, 3).
        for &v in y.data().unwrap() {
            assert!((0.0..3.0).contains(&v));
        }
    }

    #[test]
    fn make_moons_is_binary() {
        let (_x, y) = make_moons::<f64>(40, 0.05, Some(2)).unwrap();
        for &v in y.data().unwrap() {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn make_circles_is_binary() {
        let (x, y) = make_circles::<f64>(40, 0.02, 0.5, Some(3)).unwrap();
        assert_eq!(x.shape(), &[40, 2]);
        for &v in y.data().unwrap() {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn iris_has_known_shape() {
        let (x, y) = load_iris::<f64>().unwrap();
        assert_eq!(x.shape(), &[150, 4]);
        assert_eq!(y.shape(), &[150]);
        // 3 classes (0, 1, 2).
        for &v in y.data().unwrap() {
            assert!((0.0..3.0).contains(&v));
        }
    }

    #[test]
    fn wine_has_known_shape() {
        let (x, y) = load_wine::<f64>().unwrap();
        assert_eq!(x.shape(), &[178, 13]);
        assert_eq!(y.shape(), &[178]);
    }

    #[test]
    fn breast_cancer_has_known_shape() {
        let (x, y) = load_breast_cancer::<f64>().unwrap();
        assert_eq!(x.shape(), &[569, 30]);
        assert_eq!(y.shape(), &[569]);
        // Binary task.
        for &v in y.data().unwrap() {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    // ----- End-to-end: dataset → metric -----------------------------------

    #[test]
    fn iris_self_classify_is_perfect_accuracy() {
        // Sanity: predicting the labels exactly should give 100% accuracy
        // through the metric path. Verifies the (X, y) packing + the
        // metric adapter cooperate.
        use crate::metrics::accuracy_score;
        let (_x, y) = load_iris::<f64>().unwrap();
        let acc = accuracy_score(&y, &y).unwrap();
        assert!((acc - 1.0).abs() < 1e-12);
    }
}
