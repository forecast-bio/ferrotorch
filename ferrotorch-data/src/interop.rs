//! Tabular-data interop with Apache Arrow and Polars.
//!
//! This module is **feature-gated**:
//!
//! - `arrow` enables [`tensor_to_arrow_array`], [`tensor_from_arrow_array`],
//!   and [`record_batch_to_tensor`].
//! - `polars` (which implies `arrow`) additionally enables
//!   [`dataframe_to_tensor`].
//!
//! All conversions allocate a new buffer (`memcpy` from one ecosystem to
//! the other). The cost is a single contiguous copy, not per-element.
//!
//! # GPU discipline
//!
//! Arrow and Polars are **CPU formats**. Every conversion in this module
//! produces a CPU-resident `Tensor<T>`; if you want the result on a GPU
//! you must move it explicitly with [`Tensor::to`](ferrotorch_core::Tensor::to).
//! There is no automatic GPU upload — that would obscure data movement
//! and conflict with `/rust-gpu-discipline`. Conversely, `tensor_to_arrow_array`
//! requires a CPU tensor; passing a CUDA tensor returns
//! [`FerrotorchError::NotImplementedOnCuda`] rather than silently
//! downloading.
//!
//! # Why so few helpers?
//!
//! The four entry points cover **two primitives** (`tensor_to_arrow_array`
//! / `tensor_from_arrow_array`) and **two RecordBatch / DataFrame
//! aggregators** (`record_batch_to_tensor` / `dataframe_to_tensor`).
//! Building a `ferrotorch_data::Dataset` on top is a one-liner —
//! materialise the matrix once with these helpers, then wrap it in
//! [`TensorDataset`]. Keeping the kernels small and explicit avoids
//! committing to a single-row vs single-column policy in the API.

#![cfg(feature = "arrow")]

use arrow::array::{Array, ArrayRef, PrimitiveArray, RecordBatch};
use arrow::buffer::Buffer;
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType};
use ferray_numpy_interop::arrow_conv::ArrowElement;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

// ---------------------------------------------------------------------------
// Tensor <-> Arrow PrimitiveArray
// ---------------------------------------------------------------------------

/// Convert a CPU tensor to a 1-D Arrow [`PrimitiveArray`] of length
/// `tensor.numel()`. The shape is **not** preserved — pass the shape
/// alongside the array if you need a round-trip.
///
/// Allocates one contiguous buffer.
///
/// # Errors
/// Returns [`FerrotorchError::NotImplementedOnCuda`] when called on a
/// CUDA tensor; move the tensor to CPU explicitly with `.to(Device::Cpu)`
/// before calling this.
pub fn tensor_to_arrow_array<T>(
    tensor: &Tensor<T>,
) -> FerrotorchResult<PrimitiveArray<T::ArrowType>>
where
    T: Float + ArrowElement + ArrowNativeType,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    if tensor.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "tensor_to_arrow_array",
        });
    }
    // Build the Arrow PrimitiveArray directly from the tensor's flat data.
    // ferray-numpy-interop's `arrayd_to_arrow_flat` does the same thing
    // but goes through a ferray ArrayD round-trip; we know T statically
    // and can short-circuit.
    let data = tensor.data_vec()?;
    let buffer = Buffer::from_vec(data);
    Ok(PrimitiveArray::<T::ArrowType>::new(buffer.into(), None))
}

/// Convert a 1-D Arrow [`PrimitiveArray`] to a tensor of the requested
/// `shape`. `shape.iter().product()` must equal the array's length.
///
/// Returns a CPU-resident tensor; move to GPU with `.to(Device::Cuda)` if
/// needed.
///
/// Allocates one contiguous buffer (Arrow → ferray copy via
/// `arrayd_from_arrow_flat`, then ferray → ferrotorch via
/// `TensorStorage::cpu`).
///
/// # Errors
/// - `ShapeMismatch` if `shape` doesn't match the array length.
/// - `Ferray(InvalidValue)` if the Arrow array contains nulls (ferrotorch
///   tensors don't support nulls).
pub fn tensor_from_arrow_array<T>(
    arr: &PrimitiveArray<T::ArrowType>,
    shape: &[usize],
) -> FerrotorchResult<Tensor<T>>
where
    T: Float + ArrowElement + ArrowNativeType,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    if arr.null_count() > 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tensor_from_arrow_array: Arrow array contains {} nulls (ferrotorch tensors do \
                 not support nulls)",
                arr.null_count()
            ),
        });
    }
    let expected: usize = shape.iter().product();
    if arr.len() != expected {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "tensor_from_arrow_array: arrow length {} does not match shape {shape:?} \
                 (product {expected})",
                arr.len()
            ),
        });
    }
    let data: Vec<T> = arr.values().iter().copied().collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

// ---------------------------------------------------------------------------
// Arrow RecordBatch -> Tensor (homogeneous numeric columns)
// ---------------------------------------------------------------------------

/// Convert an Arrow [`RecordBatch`] of `n_cols` homogeneous numeric
/// columns into a 2-D tensor of shape `[n_rows, n_cols]`.
///
/// Every column must be a [`PrimitiveArray<T::ArrowType>`] of length
/// `n_rows`. The output is in row-major (C-order) layout:
/// `out[i, j] == column_j[i]`.
///
/// To dispatch on column dtype at runtime, call this with a different
/// `T` per branch. To handle a heterogeneous schema, project / cast the
/// RecordBatch to a single dtype first.
///
/// # Errors
/// - `InvalidArgument` if the schema has zero columns or any column is
///   not a [`PrimitiveArray<T::ArrowType>`].
/// - Forwarded from [`tensor_from_arrow_array`] for null / dtype issues.
pub fn record_batch_to_tensor<T>(rb: &RecordBatch) -> FerrotorchResult<Tensor<T>>
where
    T: Float + ArrowElement + ArrowNativeType,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    let n_cols = rb.num_columns();
    if n_cols == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "record_batch_to_tensor: RecordBatch has zero columns".into(),
        });
    }
    let n_rows = rb.num_rows();

    // Materialise each column into a Vec<T>, then interleave into row-major.
    let mut columns: Vec<Vec<T>> = Vec::with_capacity(n_cols);
    for (col_idx, arr_ref) in rb.columns().iter().enumerate() {
        let arr = arr_ref
            .as_any()
            .downcast_ref::<PrimitiveArray<T::ArrowType>>()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!(
                    "record_batch_to_tensor: column {col_idx} is not a {:?} primitive array",
                    T::ArrowType::DATA_TYPE
                ),
            })?;
        if arr.null_count() > 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "record_batch_to_tensor: column {col_idx} contains {} nulls",
                    arr.null_count()
                ),
            });
        }
        if arr.len() != n_rows {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "record_batch_to_tensor: column {col_idx} has {} rows, expected {n_rows}",
                    arr.len()
                ),
            });
        }
        // arr.values() is &ScalarBuffer<T>; iterate to materialise a Vec<T>.
        columns.push(arr.values().iter().copied().collect());
    }

    // Interleave column-major source into row-major output.
    let mut out: Vec<T> = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for col in &columns {
            out.push(col[i]);
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n_rows, n_cols], false)
}

/// Reflexive helper: returns the Arrow `ArrayRef` form of a tensor. Useful
/// when you want to emit a single column rather than a `PrimitiveArray<T>`.
pub fn tensor_to_arrow_arrayref<T>(tensor: &Tensor<T>) -> FerrotorchResult<ArrayRef>
where
    T: Float + ArrowElement + ArrowNativeType,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    let prim = tensor_to_arrow_array(tensor)?;
    Ok(std::sync::Arc::new(prim) as ArrayRef)
}

// ---------------------------------------------------------------------------
// Polars DataFrame -> Tensor
// ---------------------------------------------------------------------------

/// Convert a Polars [`DataFrame`] of `n_cols` homogeneous numeric columns
/// into a 2-D tensor of shape `[n_rows, n_cols]`.
///
/// Implementation: each column is converted to an Arrow `PrimitiveArray<T>`
/// via the column's `.to_arrow(0, CompatLevel::newest())` (Polars stores
/// columns as chunked Arrow arrays internally). Casting between numeric
/// dtypes is performed by Polars before the Arrow conversion.
///
/// Same row-major layout as [`record_batch_to_tensor`].
///
/// # Errors
/// - `InvalidArgument` if the DataFrame is empty, has multi-chunk columns
///   that fail to rechunk, or any column rejects the `T`-cast.
/// - `ShapeMismatch` for inconsistent column lengths (shouldn't happen for
///   a well-formed DataFrame but checked defensively).
#[cfg(feature = "polars")]
pub fn dataframe_to_tensor<T>(df: &polars::frame::DataFrame) -> FerrotorchResult<Tensor<T>>
where
    T: Float + ArrowElement + ArrowNativeType,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    use polars::datatypes::DataType;
    use polars::prelude::IntoSeries;

    let n_cols = df.width();
    if n_cols == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "dataframe_to_tensor: DataFrame has zero columns".into(),
        });
    }
    let n_rows = df.height();

    // Pick the target Polars DataType from the requested T. We support
    // f32 and f64 in this initial cut — extending to integer T is
    // straightforward but outside Phase 2B's scope.
    let target_dt = match (T::ArrowType::DATA_TYPE, std::mem::size_of::<T>()) {
        (arrow::datatypes::DataType::Float32, 4) => DataType::Float32,
        (arrow::datatypes::DataType::Float64, 8) => DataType::Float64,
        (other, _) => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "dataframe_to_tensor: only f32/f64 supported, got Arrow dtype {other:?}"
                ),
            });
        }
    };

    let mut columns: Vec<Vec<T>> = Vec::with_capacity(n_cols);
    for (col_idx, col) in df.iter().enumerate() {
        let s = if col.dtype() == &target_dt {
            col.clone().into_series()
        } else {
            col.cast(&target_dt).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!(
                    "dataframe_to_tensor: column {col_idx} ({}) cannot cast to {target_dt:?}: {e}",
                    col.name()
                ),
            })?
        };
        if s.null_count() > 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "dataframe_to_tensor: column {col_idx} contains {} nulls",
                    s.null_count()
                ),
            });
        }
        // Materialise via Polars' AnyValue iteration, casting to T.
        // Numeric-only by construction (we caught the dtype above).
        let mut v: Vec<T> = Vec::with_capacity(n_rows);
        for av in s.iter() {
            let f64v: f64 =
                av.try_extract::<f64>()
                    .map_err(|e| FerrotorchError::InvalidArgument {
                        message: format!(
                            "dataframe_to_tensor: column {col_idx} value extract failed: {e}"
                        ),
                    })?;
            v.push(
                T::from(f64v).ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "dataframe_to_tensor: column {col_idx} value {f64v} cannot be cast to T"
                    ),
                })?,
            );
        }
        if v.len() != n_rows {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "dataframe_to_tensor: column {col_idx} has {} rows, expected {n_rows}",
                    v.len()
                ),
            });
        }
        columns.push(v);
    }

    // Row-major interleave.
    let mut out: Vec<T> = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for col in &columns {
            out.push(col[i]);
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n_rows, n_cols], false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // ----- Tensor <-> PrimitiveArray round-trip --------------------------

    #[test]
    fn tensor_to_arrow_round_trip_f64() {
        let t = ferrotorch_core::tensor(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let arr = tensor_to_arrow_array(&t).unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.values(), &[1.0, 2.0, 3.0, 4.0]);

        let back: Tensor<f64> = tensor_from_arrow_array(&arr, &[4]).unwrap();
        assert_eq!(back.shape(), &[4]);
        assert_eq!(back.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tensor_to_arrow_round_trip_f32() {
        let t = ferrotorch_core::tensor(&[1.5_f32, 2.5, 3.5]).unwrap();
        let arr = tensor_to_arrow_array(&t).unwrap();
        let back: Tensor<f32> = tensor_from_arrow_array(&arr, &[3]).unwrap();
        assert_eq!(back.data().unwrap(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn tensor_to_arrow_preserves_2d_via_explicit_shape() {
        // Shape isn't carried in the PrimitiveArray; user passes it on
        // the way back.
        let data = (0..6).map(|x| x as f64).collect::<Vec<_>>();
        let t = Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![2, 3], false).unwrap();
        let arr = tensor_to_arrow_array(&t).unwrap();
        let back: Tensor<f64> = tensor_from_arrow_array(&arr, &[2, 3]).unwrap();
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.data().unwrap(), &data);
    }

    #[test]
    fn tensor_from_arrow_rejects_shape_mismatch() {
        let t = ferrotorch_core::tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        let arr = tensor_to_arrow_array(&t).unwrap();
        let err = tensor_from_arrow_array::<f64>(&arr, &[5]).unwrap_err();
        // ferray reports shape mismatch; we propagate it via FerrotorchError::Ferray.
        match err {
            FerrotorchError::Ferray(_) | FerrotorchError::ShapeMismatch { .. } => {}
            other => panic!("expected shape mismatch, got {other:?}"),
        }
    }

    #[test]
    fn tensor_from_arrow_rejects_nulls() {
        // Build a Float64Array with one null entry.
        let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let err = tensor_from_arrow_array::<f64>(&arr, &[3]).unwrap_err();
        match err {
            FerrotorchError::Ferray(_) | FerrotorchError::InvalidArgument { .. } => {}
            other => panic!("expected null-rejection error, got {other:?}"),
        }
    }

    // ----- ArrayRef helper -----------------------------------------------

    #[test]
    fn tensor_to_arrayref_returns_dyn_array() {
        let t = ferrotorch_core::tensor(&[1.0_f32, 2.0]).unwrap();
        let arr_ref = tensor_to_arrow_arrayref(&t).unwrap();
        assert_eq!(arr_ref.len(), 2);
        // Downcast back to Float32Array to verify dtype.
        let casted = arr_ref.as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(casted.values(), &[1.0, 2.0]);
    }

    // ----- record_batch_to_tensor ----------------------------------------

    #[test]
    fn record_batch_to_tensor_assembles_matrix() {
        // 3 rows × 2 cols of f64.
        let col_a = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let col_b = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let rb = RecordBatch::try_new(schema, vec![Arc::new(col_a), Arc::new(col_b)]).unwrap();

        let t: Tensor<f64> = record_batch_to_tensor(&rb).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        // Row-major: out[0, 0]=1, [0, 1]=10, [1, 0]=2, [1, 1]=20, ...
        assert_eq!(t.data().unwrap(), &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }

    #[test]
    fn record_batch_to_tensor_rejects_zero_columns() {
        let rb = RecordBatch::new_empty(Arc::new(Schema::empty()));
        let err = record_batch_to_tensor::<f64>(&rb).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn record_batch_to_tensor_rejects_dtype_mismatch() {
        // RecordBatch has Float32 column; we ask for f64.
        let col = Float32Array::from(vec![1.0_f32, 2.0]);
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        let rb = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();
        let err = record_batch_to_tensor::<f64>(&rb).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn record_batch_to_tensor_rejects_null_column() {
        let col = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, true)]));
        let rb = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();
        let err = record_batch_to_tensor::<f64>(&rb).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    // ----- GPU discipline -------------------------------------------------

    #[test]
    fn tensor_to_arrow_accepts_cpu_tensor() {
        // Sanity: CPU path is reachable; GPU path would return
        // NotImplementedOnCuda but we can't construct a CUDA tensor in
        // this CPU-only test environment.
        let t = ferrotorch_core::tensor(&[1.0_f64]).unwrap();
        assert!(tensor_to_arrow_array(&t).is_ok());
    }

    // ----- dataframe_to_tensor (polars feature only) ---------------------

    #[cfg(feature = "polars")]
    mod polars_tests {
        use super::*;
        use polars::prelude::*;

        #[test]
        fn dataframe_to_tensor_f64_basic() {
            let df = df!(
                "a" => &[1.0_f64, 2.0, 3.0],
                "b" => &[10.0_f64, 20.0, 30.0],
            )
            .unwrap();
            let t: Tensor<f64> = dataframe_to_tensor(&df).unwrap();
            assert_eq!(t.shape(), &[3, 2]);
            assert_eq!(t.data().unwrap(), &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        }

        #[test]
        fn dataframe_to_tensor_f32_with_cast() {
            // Source column is i64; target is f32. dataframe_to_tensor
            // should cast through Polars before extraction.
            let df = df!("a" => &[1_i64, 2, 3]).unwrap();
            let t: Tensor<f32> = dataframe_to_tensor(&df).unwrap();
            assert_eq!(t.shape(), &[3, 1]);
            assert_eq!(t.data().unwrap(), &[1.0_f32, 2.0, 3.0]);
        }

        #[test]
        fn dataframe_to_tensor_rejects_nulls() {
            let s = Series::new("a".into(), &[Some(1.0_f64), None, Some(3.0)]);
            let df = DataFrame::new(vec![s.into()]).unwrap();
            assert!(dataframe_to_tensor::<f64>(&df).is_err());
        }

        #[test]
        fn dataframe_to_tensor_rejects_empty() {
            let df = DataFrame::empty();
            assert!(dataframe_to_tensor::<f64>(&df).is_err());
        }
    }
}
