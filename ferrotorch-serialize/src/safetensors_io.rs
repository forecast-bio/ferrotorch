//! Save and load `StateDict<T>` using the
//! [SafeTensors](https://huggingface.co/docs/safetensors/) format.
//!
//! Files produced by this module are fully compatible with Python's
//! `safetensors` library, enabling seamless model exchange between Rust and
//! the HuggingFace ecosystem.

use std::collections::HashMap;
use std::path::Path;

use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::StateDict;

/// Return the `safetensors::Dtype` that corresponds to the concrete `Float`
/// type `T`.
fn st_dtype<T: Float>() -> FerrotorchResult<Dtype> {
    let size = std::mem::size_of::<T>();
    match size {
        4 => Ok(Dtype::F32),
        8 => Ok(Dtype::F64),
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "unsupported element size {} for safetensors serialization",
                size
            ),
        }),
    }
}

/// Return the expected `safetensors::Dtype` for the concrete `Float` type `T`,
/// used during loading to validate the file contents.
fn expected_dtype<T: Float>() -> FerrotorchResult<Dtype> {
    st_dtype::<T>()
}

/// Convert a slice of `T` to its raw little-endian byte representation.
///
/// # Safety
///
/// This reinterprets the memory of a `&[T]` as `&[u8]`. This is safe for
/// `f32` and `f64` on little-endian platforms (x86, ARM), which is the same
/// assumption that the SafeTensors format makes.
fn as_le_bytes<T: Float>(data: &[T]) -> &[u8] {
    let elem_size = std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * elem_size) }
}

/// Save a state dict using the SafeTensors format (HuggingFace standard).
///
/// The tensors are sorted by name for deterministic output. The resulting
/// file can be loaded by Python's `safetensors` library:
///
/// ```python
/// from safetensors import safe_open
/// with safe_open("model.safetensors", framework="numpy") as f:
///     weight = f.get_tensor("weight")
/// ```
pub fn save_safetensors<T: Float>(
    state: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()> {
    let path = path.as_ref();
    let dtype = st_dtype::<T>()?;

    // Collect tensor data so the byte slices live long enough.
    // We need to hold onto the data references while building TensorViews.
    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();

    // Build (name, TensorView) pairs. We need the tensor data to outlive the
    // TensorView, so we collect data slices first.
    struct TensorEntry<'a> {
        name: &'a str,
        shape: Vec<usize>,
        data: &'a [u8],
    }

    let mut entries: Vec<TensorEntry<'_>> = Vec::with_capacity(keys.len());
    for key in &keys {
        let tensor = &state[*key];
        let data_slice = tensor
            .data()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read tensor \"{key}\": {e}"),
            })?;
        let byte_data = as_le_bytes(data_slice);
        entries.push(TensorEntry {
            name: key.as_str(),
            shape: tensor.shape().to_vec(),
            data: byte_data,
        });
    }

    // Build TensorView objects. The safetensors crate requires Vec<(String, TensorView)>
    // or any IntoIterator<Item = (S, V)> where V: View.
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|entry| {
            let view = TensorView::new(dtype, entry.shape.clone(), entry.data).map_err(|e| {
                FerrotorchError::InvalidArgument {
                    message: format!("failed to create TensorView for \"{}\": {e}", entry.name),
                }
            });
            view.map(|v| (entry.name.to_string(), v))
        })
        .collect::<FerrotorchResult<Vec<_>>>()?;

    serialize_to_file(views, &None, path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to write safetensors file {}: {e}", path.display()),
    })?;

    Ok(())
}

/// Load a state dict from a SafeTensors file.
///
/// The dtype stored in the file must match the requested type `T`. For
/// example, loading an `F32` file into `StateDict<f64>` produces an error.
pub fn load_safetensors<T: Float>(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<T>> {
    let path = path.as_ref();
    let expected = expected_dtype::<T>()?;
    let elem_size = std::mem::size_of::<T>();

    let file_data = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read safetensors file {}: {e}", path.display()),
    })?;

    let st =
        SafeTensors::deserialize(&file_data).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse safetensors file {}: {e}", path.display()),
        })?;

    let tensor_list = st.tensors();
    let mut state: StateDict<T> = HashMap::with_capacity(tensor_list.len());

    for (name, view) in &tensor_list {
        // Validate dtype.
        if view.dtype() != expected {
            return Err(FerrotorchError::DtypeMismatch {
                expected: format!("{:?}", expected),
                got: format!("{:?}", view.dtype()),
            });
        }

        let shape = view.shape().to_vec();
        let byte_data = view.data();
        let numel: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        // Validate byte length.
        let expected_bytes = numel * elem_size;
        if byte_data.len() != expected_bytes {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor \"{}\" has {} bytes but shape {:?} with dtype {:?} requires {} bytes",
                    name,
                    byte_data.len(),
                    shape,
                    expected,
                    expected_bytes,
                ),
            });
        }

        // Reinterpret bytes as T values (little-endian assumption, same as
        // the safetensors specification).
        let data: Vec<T> = byte_data
            .chunks_exact(elem_size)
            .map(|chunk| {
                let mut bytes = [0u8; 8];
                bytes[..elem_size].copy_from_slice(chunk);
                unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const T) }
            })
            .collect();

        let storage = TensorStorage::cpu(data);
        let tensor = Tensor::from_storage(storage, shape, false)?;
        state.insert(name.clone(), tensor);
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;
    use std::collections::HashMap;

    fn make_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    fn make_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    #[test]
    fn test_save_load_roundtrip_f32() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        );
        state.insert(
            "bias".to_string(),
            make_tensor_f32(vec![0.1, 0.2, 0.3], vec![3]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_f32");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();

        assert_eq!(loaded.len(), 2);

        let w = &loaded["weight"];
        assert_eq!(w.shape(), &[2, 3]);
        let w_data = w.data().unwrap();
        assert_eq!(w_data, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let b = &loaded["bias"];
        assert_eq!(b.shape(), &[3]);
        let b_data = b.data().unwrap();
        assert_eq!(b_data, &[0.1f32, 0.2, 0.3]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_load_roundtrip_f64() {
        let mut state: StateDict<f64> = HashMap::new();
        state.insert(
            "layer.weight".to_string(),
            make_tensor_f64(vec![1.0, -2.5, 3.14, 0.0, 99.9, -0.001], vec![3, 2]),
        );
        state.insert(
            "layer.bias".to_string(),
            make_tensor_f64(vec![0.5, -0.5, 1.0], vec![3]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_f64");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f64> = load_safetensors(&path).unwrap();

        assert_eq!(loaded.len(), 2);

        let w = &loaded["layer.weight"];
        assert_eq!(w.shape(), &[3, 2]);
        let w_data = w.data().unwrap();
        assert_eq!(w_data, &[1.0f64, -2.5, 3.14, 0.0, 99.9, -0.001]);

        let b = &loaded["layer.bias"];
        assert_eq!(b.shape(), &[3]);
        let b_data = b.data().unwrap();
        assert_eq!(b_data, &[0.5f64, -0.5, 1.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_correct_tensor_names_and_shapes() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "encoder.0.weight".to_string(),
            make_tensor_f32(vec![1.0; 12], vec![3, 4]),
        );
        state.insert(
            "encoder.0.bias".to_string(),
            make_tensor_f32(vec![0.0; 3], vec![3]),
        );
        state.insert(
            "decoder.weight".to_string(),
            make_tensor_f32(vec![2.0; 8], vec![4, 2]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_names");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();

        // Read back and verify via the raw safetensors crate that names/shapes
        // are correct (independent of our load function).
        let file_data = std::fs::read(&path).unwrap();
        let st = SafeTensors::deserialize(&file_data).unwrap();

        let mut names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
        names.sort();
        assert_eq!(
            names,
            vec!["decoder.weight", "encoder.0.bias", "encoder.0.weight"],
        );

        let enc_w = st.tensor("encoder.0.weight").unwrap();
        assert_eq!(enc_w.shape(), &[3, 4]);
        assert_eq!(enc_w.dtype(), Dtype::F32);

        let enc_b = st.tensor("encoder.0.bias").unwrap();
        assert_eq!(enc_b.shape(), &[3]);

        let dec_w = st.tensor("decoder.weight").unwrap();
        assert_eq!(dec_w.shape(), &[4, 2]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_readable_by_safetensors_crate() {
        // Verify the file we produce is valid safetensors by deserializing it
        // directly with the safetensors crate (not our wrapper).
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "x".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_valid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();

        let file_data = std::fs::read(&path).unwrap();
        let st = SafeTensors::deserialize(&file_data).unwrap();

        assert_eq!(st.len(), 1);
        let tv = st.tensor("x").unwrap();
        assert_eq!(tv.dtype(), Dtype::F32);
        assert_eq!(tv.shape(), &[3]);
        // Verify the raw bytes decode correctly.
        let bytes = tv.data();
        assert_eq!(bytes.len(), 3 * 4); // 3 elements * 4 bytes each
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0f32, 2.0, 3.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_missing_file() {
        let result = load_safetensors::<f32>("/nonexistent/path/model.safetensors");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("failed to read"));
    }

    #[test]
    fn test_dtype_mismatch() {
        // Save as f32, try to load as f64.
        let mut state: StateDict<f32> = HashMap::new();
        state.insert("x".to_string(), make_tensor_f32(vec![1.0, 2.0], vec![2]));

        let dir = std::env::temp_dir().join("ferrotorch_test_st_dtype_mm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();

        let result = load_safetensors::<f64>(&path);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_empty_state_dict() {
        let state: StateDict<f32> = HashMap::new();

        let dir = std::env::temp_dir().join("ferrotorch_test_st_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();
        assert!(loaded.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_high_rank_tensor() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "conv.weight".to_string(),
            make_tensor_f32(data.clone(), vec![2, 3, 2, 2]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_4d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();

        let t = &loaded["conv.weight"];
        assert_eq!(t.shape(), &[2, 3, 2, 2]);
        let loaded_data = t.data().unwrap();
        assert_eq!(loaded_data, data.as_slice());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_1d_tensor() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert("vec".to_string(), make_tensor_f32(vec![42.0], vec![1]));

        let dir = std::env::temp_dir().join("ferrotorch_test_st_1d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();

        let v = &loaded["vec"];
        assert_eq!(v.shape(), &[1]);
        assert_eq!(v.data().unwrap(), &[42.0f32]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
