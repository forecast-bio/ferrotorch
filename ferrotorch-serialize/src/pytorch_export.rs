//! Export a `StateDict<T>` as a PyTorch-compatible `.pt` / `.pth` file.
//!
//! [CL-328] Serialization Expansion: `PyTorch` export writer.
//!
//! A `.pt` file is a ZIP archive containing:
//! - `archive/data.pkl` -- pickle protocol 2 bytecodes describing the state dict
//!   structure (an `OrderedDict` of tensor rebuild instructions).
//! - `archive/data/0`, `archive/data/1`, ... -- raw tensor byte blobs referenced
//!   by the pickle via `PERSISTENT_LOAD`.
//!
//! This module mirrors [`pytorch_import`](crate::pytorch_import) but in the
//! write direction: given a `StateDict<T>`, it produces a ZIP file readable by
//! `torch.load()`.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "ferrotorch pytorch export assumes little-endian byte order. \
     Big-endian platforms are not supported."
);

use std::io::Write;
use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::StateDict;

// ---------------------------------------------------------------------------
// Pickle protocol 2 opcodes (write direction)
// ---------------------------------------------------------------------------

const PROTO: u8 = 0x80;
const EMPTY_DICT: u8 = 0x7d;
const EMPTY_LIST: u8 = 0x5d;
const EMPTY_TUPLE: u8 = 0x29;
const MARK: u8 = 0x28;
const TUPLE: u8 = 0x74;
const TUPLE1: u8 = 0x85;
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;
const BINPUT: u8 = 0x71;
const GLOBAL: u8 = 0x63;
const SHORT_BINUNICODE: u8 = 0x8c;
const BININT1: u8 = 0x4b;
const BININT: u8 = 0x4a;
const BININT2: u8 = 0x4d;
const REDUCE: u8 = 0x52;
const BUILD: u8 = 0x62;
const APPENDS: u8 = 0x65;
const STOP: u8 = 0x2e;
const BINPERSID: u8 = 0x51;

// ---------------------------------------------------------------------------
// Pickle writer helpers
// ---------------------------------------------------------------------------

/// A minimal pickle protocol 2 writer.
struct PickleWriter {
    buf: Vec<u8>,
    memo_idx: u8,
}

impl PickleWriter {
    fn new() -> Self {
        let mut pw = Self {
            buf: Vec::with_capacity(4096),
            memo_idx: 0,
        };
        // Protocol 2 header.
        pw.buf.push(PROTO);
        pw.buf.push(2);
        pw
    }

    fn finish(mut self) -> Vec<u8> {
        self.buf.push(STOP);
        self.buf
    }

    /// Emit GLOBAL opcode: `c<module>\n<name>\n`
    fn emit_global(&mut self, module: &str, name: &str) {
        self.buf.push(GLOBAL);
        self.buf.extend_from_slice(module.as_bytes());
        self.buf.push(b'\n');
        self.buf.extend_from_slice(name.as_bytes());
        self.buf.push(b'\n');
    }

    /// Emit a `SHORT_BINUNICODE` string (len < 256).
    fn emit_short_binunicode(&mut self, s: &str) {
        assert!(s.len() < 256, "string too long for SHORT_BINUNICODE");
        self.buf.push(SHORT_BINUNICODE);
        self.buf.push(s.len() as u8);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Emit BINPUT memo assignment.
    fn emit_binput(&mut self) -> u8 {
        let idx = self.memo_idx;
        self.buf.push(BINPUT);
        self.buf.push(idx);
        self.memo_idx += 1;
        idx
    }

    /// Emit an integer that fits the most compact encoding.
    fn emit_int(&mut self, value: i64) {
        if (0..256).contains(&value) {
            self.buf.push(BININT1);
            self.buf.push(value as u8);
        } else if (0..65536).contains(&value) {
            self.buf.push(BININT2);
            self.buf.extend_from_slice(&(value as u16).to_le_bytes());
        } else {
            self.buf.push(BININT);
            self.buf.extend_from_slice(&(value as i32).to_le_bytes());
        }
    }

    fn emit_empty_tuple(&mut self) {
        self.buf.push(EMPTY_TUPLE);
    }

    fn emit_empty_dict(&mut self) {
        self.buf.push(EMPTY_DICT);
    }

    fn emit_empty_list(&mut self) {
        self.buf.push(EMPTY_LIST);
    }

    fn emit_mark(&mut self) {
        self.buf.push(MARK);
    }

    fn emit_tuple(&mut self) {
        self.buf.push(TUPLE);
    }

    fn emit_tuple1(&mut self) {
        self.buf.push(TUPLE1);
    }

    fn emit_tuple2(&mut self) {
        self.buf.push(TUPLE2);
    }

    fn emit_tuple3(&mut self) {
        self.buf.push(TUPLE3);
    }

    fn emit_reduce(&mut self) {
        self.buf.push(REDUCE);
    }

    fn emit_build(&mut self) {
        self.buf.push(BUILD);
    }

    fn emit_appends(&mut self) {
        self.buf.push(APPENDS);
    }

    fn emit_binpersid(&mut self) {
        self.buf.push(BINPERSID);
    }
}

// ---------------------------------------------------------------------------
// PyTorch dtype helpers
// ---------------------------------------------------------------------------

/// `PyTorch` storage class name for the given `Float` type and byte width.
// The `4` and `_` arms map to the same string deliberately: PyTorch only
// has `FloatStorage` (f32) and `DoubleStorage` (f64); narrower `Float`
// implementations (bf16) fall back to the f32 storage class for
// compatibility with downstream readers that don't speak bf16.
#[allow(clippy::match_same_arms)]
fn pytorch_storage_type<T: Float>() -> &'static str {
    match std::mem::size_of::<T>() {
        4 => "FloatStorage",
        8 => "DoubleStorage",
        _ => "FloatStorage",
    }
}

/// `PyTorch` dtype string for use in `rebuild_tensor_v2` metadata.
// Same rationale as `pytorch_storage_type`: bf16 falls back to
// `torch.float32` for PyTorch consumer compatibility.
#[allow(clippy::match_same_arms)]
fn pytorch_dtype_str<T: Float>() -> &'static str {
    match std::mem::size_of::<T>() {
        4 => "torch.float32",
        8 => "torch.float64",
        _ => "torch.float32",
    }
}

// ---------------------------------------------------------------------------
// Per-tensor metadata for pickle generation
// ---------------------------------------------------------------------------

struct TensorEntry<'a> {
    name: &'a str,
    storage_idx: usize,
    shape: &'a [usize],
    numel: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Save a `StateDict<T>` as a PyTorch-compatible `.pt` file.
///
/// The output file is a ZIP archive containing `archive/data.pkl` (pickle
/// protocol 2) and `archive/data/N` storage blobs, identical in layout to
/// files produced by `torch.save(state_dict, path)`.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written.
pub fn save_pytorch<T: Float>(
    state: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()> {
    let path = path.as_ref();

    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();

    let elem_size = std::mem::size_of::<T>();

    let entries: Vec<TensorEntry<'_>> = keys
        .iter()
        .enumerate()
        .map(|(idx, key)| {
            let tensor = &state[*key];
            TensorEntry {
                name: key,
                storage_idx: idx,
                shape: tensor.shape(),
                numel: tensor.numel(),
            }
        })
        .collect();

    // -----------------------------------------------------------------------
    // Build the pickle bytes (archive/data.pkl)
    // -----------------------------------------------------------------------
    let pkl_bytes = build_state_dict_pickle::<T>(&entries);

    // -----------------------------------------------------------------------
    // Write the ZIP archive
    // -----------------------------------------------------------------------
    let file = std::fs::File::create(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to create file {}: {e}", path.display()),
    })?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    // Write data.pkl
    zip.start_file("archive/data.pkl", options)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ZIP write error: {e}"),
        })?;
    zip.write_all(&pkl_bytes)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ZIP write error: {e}"),
        })?;

    // Write tensor storage blobs: archive/data/0, archive/data/1, ...
    for (idx, key) in keys.iter().enumerate() {
        let tensor = &state[*key];
        let data = tensor.data()?;
        debug_assert_eq!(
            std::mem::size_of_val(data),
            tensor.numel() * elem_size,
            "tensor byte size mismatch for {key}"
        );
        // SAFETY: `data: &[T]` where `T: Float` is one of f32/f64/bf16 —
        // all `Copy` POD types with a stable bit-level layout, no padding,
        // and no `Drop` semantics, so reinterpreting their bytes as `&[u8]`
        // is sound. `elem_size = size_of::<T>()` and the `debug_assert_eq!`
        // immediately above this block confirms `tensor.numel() * elem_size
        // == size_of_val(data)`, so the byte length is the exact extent the
        // source pointer is valid for. The returned slice reborrows `data`
        // and cannot outlive it. PyTorch's `.pt` format is little-endian by
        // contract; that matches the `compile_error!` at the top of this
        // module.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), tensor.numel() * elem_size)
        };

        let entry_name = format!("archive/data/{idx}");
        zip.start_file(&entry_name, options)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("ZIP write error: {e}"),
            })?;
        zip.write_all(byte_slice)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("ZIP write error: {e}"),
            })?;
    }

    // Write dtype metadata file (matches PyTorch's archive/data_type_id).
    let dtype_str = pytorch_dtype_str::<T>();
    zip.start_file("archive/data_type_id", options)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ZIP write error: {e}"),
        })?;
    zip.write_all(dtype_str.as_bytes())
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ZIP write error: {e}"),
        })?;

    zip.finish().map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("ZIP finalize error: {e}"),
    })?;

    Ok(())
}

/// Build the `data.pkl` pickle bytes for a state dict.
///
/// Emits pickle protocol 2 bytecodes that encode:
/// ```text
/// collections.OrderedDict([
///   ("key0", torch._utils._rebuild_tensor_v2(
///       PERSISTENT_LOAD(('storage', torch.FloatStorage, '0', 'cpu', numel)),
///       0, (dim0, dim1, ...), (stride0, stride1, ...), False, OrderedDict()
///   )),
///   ...
/// ])
/// ```
fn build_state_dict_pickle<T: Float>(entries: &[TensorEntry<'_>]) -> Vec<u8> {
    let storage_type = pytorch_storage_type::<T>();
    let mut pw = PickleWriter::new();

    // Push OrderedDict callable: GLOBAL 'collections' 'OrderedDict'
    pw.emit_global("collections", "OrderedDict");
    pw.emit_binput(); // memo 0

    // The argument to OrderedDict() is a single-element tuple containing a list
    // of (key, tensor) pairs.
    pw.emit_empty_tuple();
    pw.emit_reduce(); // OrderedDict()
    pw.emit_binput(); // memo 1

    // Now BUILD with state = list of (key, value) tuples.
    pw.emit_empty_list();
    pw.emit_binput(); // memo 2

    // MARK + (key, tensor)... + APPENDS
    pw.emit_mark();

    for entry in entries {
        // (key, tensor_rebuild) tuple
        //
        // key string
        pw.emit_short_binunicode(entry.name);

        // tensor rebuild:
        // GLOBAL 'torch._utils' '_rebuild_tensor_v2'
        // REDUCE(
        //   PERSISTENT_LOAD(('storage', torch.{dtype}Storage, '{idx}', 'cpu', numel)),
        //   storage_offset,
        //   shape_tuple,
        //   stride_tuple,
        //   False,
        //   OrderedDict()
        // )

        pw.emit_global("torch._utils", "_rebuild_tensor_v2");
        pw.emit_binput();

        // Build the arguments tuple for _rebuild_tensor_v2
        pw.emit_mark();

        // arg 0: PERSISTENT_LOAD(('storage', StorageClass, key, 'cpu', numel))
        pw.emit_mark();
        pw.emit_short_binunicode("storage");
        pw.emit_global("torch", storage_type);
        pw.emit_binput();
        pw.emit_short_binunicode(&entry.storage_idx.to_string());
        pw.emit_short_binunicode("cpu");
        pw.emit_int(entry.numel as i64);
        pw.emit_tuple(); // close the persistent-load tuple
        pw.emit_binpersid();

        // arg 1: storage_offset = 0
        pw.emit_int(0);

        // arg 2: shape tuple
        emit_shape_tuple(&mut pw, entry.shape);

        // arg 3: stride tuple (row-major / C-contiguous strides)
        let strides = compute_contiguous_strides(entry.shape);
        emit_shape_tuple(&mut pw, &strides);

        // arg 4: requires_grad = False (BININT1 0)
        pw.emit_int(0);

        // arg 5: empty dict {} (tensor metadata — empty for standard tensors)
        pw.emit_empty_dict();

        pw.emit_tuple(); // close the _rebuild_tensor_v2 args
        pw.emit_reduce(); // _rebuild_tensor_v2(...)
        pw.emit_binput();

        // Now make the (key, value) pair tuple
        pw.emit_tuple2();
    }

    pw.emit_appends(); // append all (key, value) pairs to the list

    pw.emit_build(); // OrderedDict.__setstate__(list)
    pw.emit_binput();

    pw.finish()
}

/// Emit a shape/stride tuple using the most compact tuple opcode.
fn emit_shape_tuple(pw: &mut PickleWriter, dims: &[usize]) {
    match dims.len() {
        0 => pw.emit_empty_tuple(),
        1 => {
            pw.emit_int(dims[0] as i64);
            pw.emit_tuple1();
        }
        2 => {
            pw.emit_int(dims[0] as i64);
            pw.emit_int(dims[1] as i64);
            pw.emit_tuple2();
        }
        3 => {
            pw.emit_int(dims[0] as i64);
            pw.emit_int(dims[1] as i64);
            pw.emit_int(dims[2] as i64);
            pw.emit_tuple3();
        }
        _ => {
            pw.emit_mark();
            for &d in dims {
                pw.emit_int(d as i64);
            }
            pw.emit_tuple();
        }
    }
}

/// Compute C-contiguous (row-major) strides for a given shape.
fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Validate checkpoint integrity
// ---------------------------------------------------------------------------

/// Validate a `.pt` / `.pth` checkpoint file by checking CRC32 integrity of
/// every entry in the ZIP archive.
///
/// Returns `Ok(())` if all entries pass the CRC32 check, or an error
/// describing the first corrupt entry found.
pub fn validate_checkpoint(path: impl AsRef<Path>) -> FerrotorchResult<()> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open checkpoint file {}: {e}", path.display()),
    })?;

    let mut archive = zip::ZipArchive::new(file).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read ZIP archive {}: {e}", path.display()),
    })?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read ZIP entry {i}: {e}"),
            })?;

        let entry_name = entry.name().to_string();

        // Read the entire entry to force decompression + CRC32 validation.
        // The `zip` crate's `read_to_end` checks the CRC32 footer on Stored
        // entries and validates the inflated checksum on Deflated entries.
        let mut buf = Vec::with_capacity(entry.size() as usize);
        std::io::Read::read_to_end(&mut entry, &mut buf).map_err(|e| {
            FerrotorchError::InvalidArgument {
                message: format!("CRC32 integrity check failed for entry \"{entry_name}\": {e}"),
            }
        })?;

        // Compute our own CRC32 and compare against the stored one.
        let expected_crc = entry.crc32();
        let actual_crc = crc32_hash(&buf);
        if expected_crc != actual_crc {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CRC32 mismatch for entry \"{entry_name}\": \
                     expected 0x{expected_crc:08X}, got 0x{actual_crc:08X}"
                ),
            });
        }
    }

    Ok(())
}

/// Compute CRC32 (ISO 3309 / ITU-T V.42) of a byte slice.
///
/// Uses the standard polynomial 0xEDB88320 (reflected form).
fn crc32_hash(data: &[u8]) -> u32 {
    // Build lookup table at compile time is not possible without const generics
    // magic, but a simple runtime table is fine for a validation utility.
    let table: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut i = 0u32;
        while i < 256 {
            let mut crc = i;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            t[i as usize] = crc;
            i += 1;
        }
        t
    };

    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    !crc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};
    use std::collections::HashMap;
    use std::io::Read;

    fn make_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    fn make_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    // -----------------------------------------------------------------------
    // Test: save_pytorch creates a valid ZIP with the right entries
    // -----------------------------------------------------------------------

    #[test]
    fn test_save_pytorch_zip_structure() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "layer.weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        state.insert(
            "layer.bias".to_string(),
            make_tensor_f32(vec![0.5, -0.5], vec![2]),
        );

        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_struct_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        // Open as ZIP and verify entries exist.
        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();

        assert!(names.contains(&"archive/data.pkl".to_string()));
        assert!(names.contains(&"archive/data/0".to_string()));
        assert!(names.contains(&"archive/data/1".to_string()));

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: pickle contains expected structure markers
    // -----------------------------------------------------------------------

    #[test]
    fn test_save_pytorch_pickle_structure() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "fc.weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        );

        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_pkl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        // Read the pickle bytes from the ZIP.
        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut pkl_entry = archive.by_name("archive/data.pkl").unwrap();
        let mut pkl_bytes = Vec::new();
        pkl_entry.read_to_end(&mut pkl_bytes).unwrap();

        // Protocol 2 header.
        assert_eq!(pkl_bytes[0], 0x80);
        assert_eq!(pkl_bytes[1], 0x02);

        // Last byte should be STOP.
        assert_eq!(*pkl_bytes.last().unwrap(), 0x2e);

        // Pickle should contain references to torch._utils, _rebuild_tensor_v2,
        // collections.OrderedDict, etc.
        let pkl_str = String::from_utf8_lossy(&pkl_bytes);
        assert!(pkl_str.contains("collections"));
        assert!(pkl_str.contains("OrderedDict"));
        assert!(pkl_str.contains("torch._utils"));
        assert!(pkl_str.contains("_rebuild_tensor_v2"));
        assert!(pkl_str.contains("FloatStorage"));
        assert!(pkl_str.contains("fc.weight"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: tensor storage blobs contain correct data
    // -----------------------------------------------------------------------

    #[test]
    fn test_save_pytorch_tensor_data() {
        let mut state: StateDict<f32> = HashMap::new();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        state.insert(
            "weight".to_string(),
            make_tensor_f32(data.clone(), vec![2, 2]),
        );

        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_data_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut entry = archive.by_name("archive/data/0").unwrap();
        let mut blob = Vec::new();
        entry.read_to_end(&mut blob).unwrap();

        assert_eq!(blob.len(), 4 * 4); // 4 floats * 4 bytes
        let values: Vec<f32> = blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, data);

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: f64 state dict
    // -----------------------------------------------------------------------

    #[test]
    fn test_save_pytorch_f64() {
        let mut state: StateDict<f64> = HashMap::new();
        state.insert(
            "param".to_string(),
            make_tensor_f64(vec![1.0, 2.0, 3.0], vec![3]),
        );

        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_f64_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        // Pickle should reference DoubleStorage.
        let mut pkl_entry = archive.by_name("archive/data.pkl").unwrap();
        let mut pkl_bytes = Vec::new();
        pkl_entry.read_to_end(&mut pkl_bytes).unwrap();
        let pkl_str = String::from_utf8_lossy(&pkl_bytes);
        assert!(pkl_str.contains("DoubleStorage"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: empty state dict
    // -----------------------------------------------------------------------

    #[test]
    fn test_save_pytorch_empty() {
        let state: StateDict<f32> = HashMap::new();

        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        // Should still be a valid ZIP with data.pkl.
        let file = std::fs::File::open(&path).unwrap();
        let archive = zip::ZipArchive::new(file).unwrap();
        assert!(!archive.is_empty()); // at least data.pkl

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: roundtrip — save_pytorch then load_pytorch_state_dict
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_save_load() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "layer1.weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        );
        state.insert(
            "layer1.bias".to_string(),
            make_tensor_f32(vec![0.1, 0.2], vec![2]),
        );

        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        // Load it back with our importer.
        let loaded: StateDict<f32> = crate::pytorch_import::load_pytorch_state_dict(&path).unwrap();

        assert_eq!(loaded.len(), 2);

        let w = &loaded["layer1.weight"];
        assert_eq!(w.shape(), &[2, 3]);
        let w_data = w.data().unwrap();
        assert_eq!(w_data, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let b = &loaded["layer1.bias"];
        assert_eq!(b.shape(), &[2]);
        let b_data = b.data().unwrap();
        assert_eq!(b_data, &[0.1f32, 0.2]);

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: contiguous strides computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_contiguous_strides() {
        assert_eq!(compute_contiguous_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_contiguous_strides(&[5]), vec![1]);
        assert!(compute_contiguous_strides(&[]).is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: CRC32 computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_crc32_known_values() {
        // CRC32 of empty is 0x00000000.
        assert_eq!(crc32_hash(b""), 0x0000_0000);

        // CRC32 of "123456789" is 0xCBF43926 (well-known test vector).
        assert_eq!(crc32_hash(b"123456789"), 0xCBF4_3926);
    }

    // -----------------------------------------------------------------------
    // Test: validate_checkpoint on a valid file
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_checkpoint_valid() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert("x".to_string(), make_tensor_f32(vec![1.0, 2.0], vec![2]));

        let dir = std::env::temp_dir().join(format!(
            "ferrotorch_test_pt_validate_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        // Should pass validation.
        validate_checkpoint(&path).unwrap();

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: validate_checkpoint on a nonexistent file
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_checkpoint_missing_file() {
        let result = validate_checkpoint("/nonexistent/model.pt");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Test: validate_checkpoint on a corrupt file
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_checkpoint_corrupt() {
        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_corrupt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrupt.pt");

        // Write some garbage that isn't a valid ZIP.
        std::fs::write(&path, b"this is not a ZIP file").unwrap();

        let result = validate_checkpoint(&path);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test: deterministic ordering of keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_deterministic_key_order() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert("z".to_string(), make_tensor_f32(vec![3.0], vec![1]));
        state.insert("a".to_string(), make_tensor_f32(vec![1.0], vec![1]));
        state.insert("m".to_string(), make_tensor_f32(vec![2.0], vec![1]));

        let dir =
            std::env::temp_dir().join(format!("ferrotorch_test_pt_order_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");

        save_pytorch(&state, &path).unwrap();

        // Read pickle and verify key order in the bytestream.
        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut pkl_entry = archive.by_name("archive/data.pkl").unwrap();
        let mut pkl_bytes = Vec::new();
        pkl_entry.read_to_end(&mut pkl_bytes).unwrap();
        let pkl_str = String::from_utf8_lossy(&pkl_bytes);

        let pos_a = pkl_str
            .find("\"a\"")
            .or_else(|| pkl_str.find("\x01a"))
            .unwrap_or(pkl_str.find('a').unwrap());
        let pos_m = pkl_str.rfind('m').unwrap();
        let pos_z = pkl_str.rfind('z').unwrap();
        // "a" should appear before "m" and "m" before "z" in pickle output
        assert!(pos_a < pos_m);
        assert!(pos_m < pos_z);

        std::fs::remove_dir_all(&dir).ok();
    }
}
