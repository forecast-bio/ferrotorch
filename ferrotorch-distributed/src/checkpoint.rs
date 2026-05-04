//! Distributed checkpointing with per-rank shard saving, loading, and resharding.
//!
//! Each rank saves its own shard as a SafeTensors file under a shared checkpoint
//! directory. Rank 0 additionally writes a `metadata.json` file describing how
//! tensors are distributed across ranks. When loading, if the world size has
//! changed, the resharding logic automatically splits or merges shards so that
//! each new rank receives the correct slice of each tensor.
//!
//! # Async checkpointing
//!
//! [`AsyncCheckpointer`] copies tensor data to CPU memory in a background thread
//! and writes to disk without blocking the training loop. Call
//! [`save_async`](AsyncCheckpointer::save_async) to start a checkpoint, and
//! [`CheckpointFuture::wait`] when you need to ensure the write completed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{FerrotorchError, Float, Tensor};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to distributed checkpointing.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DistCheckpointError {
    #[error("I/O error: {message}")]
    Io { message: String },

    #[error("serialization error: {message}")]
    Serialization { message: String },

    #[error("metadata error: {message}")]
    Metadata { message: String },

    #[error("shard file missing: {path}")]
    MissingShard { path: String },

    #[error("tensor error: {message}")]
    Tensor { message: String },

    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },

    #[error("async checkpoint failed: {message}")]
    AsyncFailed { message: String },
}

impl From<DistCheckpointError> for FerrotorchError {
    fn from(e: DistCheckpointError) -> Self {
        FerrotorchError::InvalidArgument {
            message: e.to_string(),
        }
    }
}

impl From<std::io::Error> for DistCheckpointError {
    fn from(e: std::io::Error) -> Self {
        DistCheckpointError::Io {
            message: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata types
// ---------------------------------------------------------------------------

/// Describes how a single tensor is sharded across ranks.
///
/// Marked `#[non_exhaustive]` so future fields (e.g., dtype, ordering,
/// per-rank ranges for non-uniform shards) can be added without a major
/// version bump. Construct via in-crate helpers (e.g.
/// [`flat_shard_metadata`]) and access fields by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TensorShardSpec {
    /// Full (unsharded) shape of the tensor.
    pub full_shape: Vec<usize>,
    /// Which dimension is split across ranks.
    pub shard_dim: usize,
    /// Size along `shard_dim` for each rank. The sum must equal
    /// `full_shape[shard_dim]`.
    pub shard_sizes: Vec<usize>,
}

/// Metadata for a distributed checkpoint: how many ranks participated and
/// how each tensor is sharded.
///
/// Marked `#[non_exhaustive]` to allow forward-compatible additions
/// (versioning, dtype catalog, optimizer-state metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ShardMetadata {
    /// Number of ranks that produced this checkpoint.
    pub num_ranks: usize,
    /// Per-tensor sharding specification, keyed by tensor name.
    pub tensor_specs: HashMap<String, TensorShardSpec>,
}

/// Handle for a distributed checkpoint directory.
///
/// Wraps the directory path and the shard metadata. Typically created by
/// [`save_distributed`] (which writes files) or by reading an existing
/// checkpoint's `metadata.json`.
///
/// Marked `#[non_exhaustive]` for forward-compatible additions
/// (e.g., async-write handles, optimizer-state cache).
#[non_exhaustive]
pub struct DistributedCheckpoint {
    /// Directory where shard files are stored.
    pub checkpoint_dir: PathBuf,
    /// Metadata about how tensors are sharded across ranks.
    pub shard_metadata: ShardMetadata,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// SafeTensors dtype for a given `Float` type.
fn st_dtype<T: Float>() -> Result<Dtype, DistCheckpointError> {
    match std::mem::size_of::<T>() {
        4 => Ok(Dtype::F32),
        8 => Ok(Dtype::F64),
        other => Err(DistCheckpointError::InvalidArgument {
            message: format!("unsupported element size {other} for safetensors serialization"),
        }),
    }
}

/// Reinterpret a `&[T]` as a byte slice (little-endian platforms).
fn as_le_bytes<T: Float>(data: &[T]) -> &[u8] {
    // SAFETY: byte-reinterpret of a `&[T]` of `Float` (f32/f64/bf16/f16) into
    // `&[u8]`.
    //
    // - VALIDITY: every byte pattern is a valid `u8`, so reading the
    //   underlying bytes never produces an invalid value (unlike e.g. `bool`
    //   or `char`).
    // - LIFETIME: the resulting `&[u8]` borrows the same allocation as
    //   `data` and is returned tied to `data`'s lifetime — the borrow
    //   checker therefore prevents `data` from being dropped or mutated
    //   while the byte slice is live.
    // - LENGTH: `mem::size_of_val(data) == data.len() * mem::size_of::<T>()`,
    //   so the resulting slice covers exactly the same memory as the input.
    // - ALIGNMENT: `*const u8` has alignment 1, which is always satisfied
    //   by an allocation aligned to `T` (alignment of `T` is a multiple of
    //   1 for every primitive `Float` type).
    // - ALIASING: this returns a shared borrow; no `&mut [u8]` aliasing is
    //   possible while the borrow lives.
    // - ENDIANNESS: ferrotorch targets little-endian platforms (the
    //   workspace's MSRV-supported targets are all LE); on a hypothetical
    //   big-endian build the byte order would not match SafeTensors'
    //   on-disk LE convention, but no such target is currently supported.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// Build a shard file path for a given rank inside the checkpoint directory.
fn shard_path(dir: &Path, rank: usize) -> PathBuf {
    dir.join(format!("rank_{rank}.safetensors"))
}

/// Path to the metadata JSON file.
fn metadata_path(dir: &Path) -> PathBuf {
    dir.join("metadata.json")
}

/// Save a `HashMap<String, Tensor<T>>` as a single SafeTensors file.
fn save_tensors_to_file<T: Float>(
    tensors: &HashMap<String, Tensor<T>>,
    path: &Path,
) -> Result<(), DistCheckpointError> {
    let dtype = st_dtype::<T>()?;

    let mut keys: Vec<&str> = tensors.keys().map(String::as_str).collect();
    keys.sort_unstable();

    struct Entry<'a> {
        name: String,
        shape: Vec<usize>,
        data: &'a [u8],
    }

    let mut entries: Vec<Entry<'_>> = Vec::with_capacity(keys.len());
    for key in &keys {
        let tensor = &tensors[*key];
        let data_slice = tensor.data().map_err(|e| DistCheckpointError::Tensor {
            message: format!("failed to read tensor \"{key}\": {e}"),
        })?;
        entries.push(Entry {
            name: (*key).to_owned(),
            shape: tensor.shape().to_vec(),
            data: as_le_bytes(data_slice),
        });
    }

    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|entry| {
            TensorView::new(dtype, entry.shape.clone(), entry.data)
                .map(|v| (entry.name.clone(), v))
                .map_err(|e| DistCheckpointError::Serialization {
                    message: format!("TensorView for \"{}\": {e}", entry.name),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    serialize_to_file(views, &None, path).map_err(|e| DistCheckpointError::Serialization {
        message: format!("safetensors write to {}: {e}", path.display()),
    })?;

    Ok(())
}

/// Load a SafeTensors file into `HashMap<String, Tensor<T>>`.
fn load_tensors_from_file<T: Float>(
    path: &Path,
) -> Result<HashMap<String, Tensor<T>>, DistCheckpointError> {
    let elem_size = std::mem::size_of::<T>();
    let expected = st_dtype::<T>()?;

    let file_data = std::fs::read(path).map_err(|e| DistCheckpointError::Io {
        message: format!("reading {}: {e}", path.display()),
    })?;

    let st =
        SafeTensors::deserialize(&file_data).map_err(|e| DistCheckpointError::Serialization {
            message: format!("parsing {}: {e}", path.display()),
        })?;

    let tensor_list = st.tensors();
    let mut result: HashMap<String, Tensor<T>> = HashMap::with_capacity(tensor_list.len());

    for (name, view) in &tensor_list {
        if view.dtype() != expected {
            return Err(DistCheckpointError::Tensor {
                message: format!(
                    "tensor \"{name}\" has dtype {:?}, expected {:?}",
                    view.dtype(),
                    expected
                ),
            });
        }

        let shape = view.shape().to_vec();
        let byte_data = view.data();
        let numel: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let expected_bytes = numel * elem_size;

        if byte_data.len() != expected_bytes {
            return Err(DistCheckpointError::Tensor {
                message: format!(
                    "tensor \"{name}\" has {} bytes but shape {shape:?} requires {expected_bytes}",
                    byte_data.len()
                ),
            });
        }

        let data: Vec<T> = byte_data
            .chunks_exact(elem_size)
            .map(|chunk| {
                let mut bytes = [0u8; 8];
                bytes[..elem_size].copy_from_slice(chunk);
                // SAFETY: `read_unaligned` of `T` from a stack buffer.
                //
                // - VALIDITY: `T: Float` is constrained to f32/f64/bf16/f16,
                //   for which every 4- or 8-byte pattern is a valid value
                //   (NaN/inf included). No invalid bit patterns exist for
                //   IEEE-754 floats — unlike `bool` (must be 0/1) or
                //   `NonZero<...>`.
                // - LENGTH: `elem_size == size_of::<T>()` (verified at the
                //   call site by `chunks_exact(elem_size)`); since
                //   `size_of::<T>() <= 8` for every supported float and
                //   `bytes` is a `[u8; 8]`, the read stays within bounds.
                // - ALIGNMENT: `read_unaligned` does not require
                //   `bytes.as_ptr()` to be aligned to `T` — that's the
                //   whole point of the function. Using `read` instead
                //   would be UB here because `[u8; 8]` is 1-aligned.
                // - LIFETIME: `bytes` is a stack-local array; the read
                //   produces an owned `T` by value, so no dangling
                //   reference can outlive the closure.
                // - PROVENANCE: `bytes.as_ptr()` is derived directly from
                //   the live stack array `bytes`; the cast to `*const T`
                //   preserves provenance under the strict-provenance model.
                // - ENDIANNESS: SafeTensors stores tensors little-endian;
                //   ferrotorch's supported targets are LE, so the byte
                //   pattern read here matches the on-disk pattern. On a
                //   hypothetical BE host the values would be byte-swapped,
                //   but no such target is currently supported.
                unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const T) }
            })
            .collect();

        let storage = TensorStorage::cpu(data);
        let tensor = Tensor::from_storage(storage, shape, false).map_err(|e| {
            DistCheckpointError::Tensor {
                message: format!("creating tensor \"{name}\": {e}"),
            }
        })?;
        result.insert(name.clone(), tensor);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Save / Load
// ---------------------------------------------------------------------------

/// Save a distributed checkpoint.
///
/// Each rank saves its own shard of the state dict to
/// `dir/rank_<rank>.safetensors`. Rank 0 additionally writes
/// `dir/metadata.json` containing the [`ShardMetadata`] so that future loads
/// (potentially with a different world size) know how the data is distributed.
///
/// The `state_dict` should contain this rank's shard of each tensor.
/// `shard_spec` describes the full shapes and how they are sharded.
///
/// # Errors
///
/// Returns an error if the directory cannot be created, if tensor data cannot
/// be read (e.g., GPU tensor without prior `.cpu()`), or if serialization
/// fails.
pub fn save_distributed<T: Float>(
    state_dict: &HashMap<String, Tensor<T>>,
    dir: &Path,
    rank: usize,
    world_size: usize,
    shard_spec: &ShardMetadata,
) -> Result<(), DistCheckpointError> {
    // Validate basic inputs.
    if world_size == 0 {
        return Err(DistCheckpointError::InvalidArgument {
            message: "world_size must be >= 1".into(),
        });
    }
    if rank >= world_size {
        return Err(DistCheckpointError::InvalidArgument {
            message: format!("rank {rank} >= world_size {world_size}"),
        });
    }

    // Create the checkpoint directory if it doesn't exist.
    std::fs::create_dir_all(dir)?;

    // Save this rank's shard.
    let path = shard_path(dir, rank);
    save_tensors_to_file(state_dict, &path)?;

    // Rank 0 writes the metadata file.
    if rank == 0 {
        let json = serde_json::to_string_pretty(shard_spec).map_err(|e| {
            DistCheckpointError::Serialization {
                message: format!("serializing metadata: {e}"),
            }
        })?;
        std::fs::write(metadata_path(dir), json)?;
    }

    Ok(())
}

/// Load a distributed checkpoint for a specific rank.
///
/// Reads `dir/metadata.json` to discover the original sharding layout. If the
/// current `world_size` matches the saved metadata, each rank simply loads its
/// own shard file. If world sizes differ, automatic resharding is performed
/// via [`reshard`].
///
/// # Errors
///
/// Returns an error if metadata or shard files are missing, if tensors have
/// unexpected dtypes, or if resharding fails.
pub fn load_distributed<T: Float>(
    dir: &Path,
    rank: usize,
    world_size: usize,
) -> Result<HashMap<String, Tensor<T>>, DistCheckpointError> {
    if world_size == 0 {
        return Err(DistCheckpointError::InvalidArgument {
            message: "world_size must be >= 1".into(),
        });
    }
    if rank >= world_size {
        return Err(DistCheckpointError::InvalidArgument {
            message: format!("rank {rank} >= world_size {world_size}"),
        });
    }

    // Read metadata.
    let meta_path = metadata_path(dir);
    let meta_json = std::fs::read_to_string(&meta_path).map_err(|e| DistCheckpointError::Io {
        message: format!("reading {}: {e}", meta_path.display()),
    })?;
    let metadata: ShardMetadata =
        serde_json::from_str(&meta_json).map_err(|e| DistCheckpointError::Serialization {
            message: format!("parsing metadata: {e}"),
        })?;

    let old_world_size = metadata.num_ranks;

    if old_world_size == world_size {
        // Same world size — just load this rank's shard directly.
        let path = shard_path(dir, rank);
        if !path.exists() {
            // If there are no tensor specs, this is a legitimately empty
            // checkpoint — return an empty map.
            if metadata.tensor_specs.is_empty() {
                return Ok(HashMap::new());
            }
            return Err(DistCheckpointError::MissingShard {
                path: path.display().to_string(),
            });
        }
        load_tensors_from_file(&path)
    } else {
        // Different world size — need to reshard.
        reshard(dir, old_world_size, world_size, rank)
    }
}

// ---------------------------------------------------------------------------
// Resharding
// ---------------------------------------------------------------------------

/// Reshard a checkpoint from `old_world_size` ranks to `new_world_size` ranks,
/// returning the data for `new_rank`.
///
/// For each tensor described in the checkpoint metadata, this function:
///
/// 1. Reconstructs the full (unsharded) tensor by loading and concatenating
///    all old shard files along the shard dimension.
/// 2. Splits the full tensor along the shard dimension into `new_world_size`
///    pieces and returns the piece for `new_rank`.
///
/// This handles both scale-up (e.g., 4 GPUs to 8) and scale-down (e.g., 8
/// GPUs to 4) as well as arbitrary remappings.
///
/// # Errors
///
/// Returns an error if shard files are missing, if tensors have unexpected
/// shapes, or if the full tensor cannot be evenly divided for resharding.
pub fn reshard<T: Float>(
    dir: &Path,
    old_world_size: usize,
    new_world_size: usize,
    new_rank: usize,
) -> Result<HashMap<String, Tensor<T>>, DistCheckpointError> {
    if new_world_size == 0 {
        return Err(DistCheckpointError::InvalidArgument {
            message: "new_world_size must be >= 1".into(),
        });
    }
    if new_rank >= new_world_size {
        return Err(DistCheckpointError::InvalidArgument {
            message: format!("new_rank {new_rank} >= new_world_size {new_world_size}"),
        });
    }
    if old_world_size == 0 {
        return Err(DistCheckpointError::InvalidArgument {
            message: "old_world_size must be >= 1".into(),
        });
    }

    // Read metadata.
    let meta_path = metadata_path(dir);
    let meta_json = std::fs::read_to_string(&meta_path).map_err(|e| DistCheckpointError::Io {
        message: format!("reading {}: {e}", meta_path.display()),
    })?;
    let metadata: ShardMetadata =
        serde_json::from_str(&meta_json).map_err(|e| DistCheckpointError::Serialization {
            message: format!("parsing metadata: {e}"),
        })?;

    // Load all old shards into memory.
    let mut old_shards: Vec<HashMap<String, Tensor<T>>> = Vec::with_capacity(old_world_size);
    for old_rank in 0..old_world_size {
        let path = shard_path(dir, old_rank);
        if !path.exists() {
            return Err(DistCheckpointError::MissingShard {
                path: path.display().to_string(),
            });
        }
        old_shards.push(load_tensors_from_file(&path)?);
    }

    // For each tensor in the metadata, reconstruct full tensor then re-split.
    let mut result: HashMap<String, Tensor<T>> = HashMap::new();

    for (name, spec) in &metadata.tensor_specs {
        let shard_dim = spec.shard_dim;
        let full_shape = &spec.full_shape;

        // Collect shard data from each old rank for this tensor.
        let mut shard_datas: Vec<Vec<T>> = Vec::with_capacity(old_world_size);
        let mut shard_shapes: Vec<Vec<usize>> = Vec::with_capacity(old_world_size);

        for (old_rank, shard) in old_shards.iter().enumerate().take(old_world_size) {
            let tensor = shard.get(name).ok_or_else(|| DistCheckpointError::Tensor {
                message: format!("tensor \"{name}\" missing from rank {old_rank} shard"),
            })?;
            shard_datas.push(tensor.data_vec().map_err(|e| DistCheckpointError::Tensor {
                message: format!("reading tensor \"{name}\" from rank {old_rank}: {e}"),
            })?);
            shard_shapes.push(tensor.shape().to_vec());
        }

        // Reconstruct the full tensor by concatenating along shard_dim.
        let full_data = concat_along_dim(&shard_datas, &shard_shapes, shard_dim, full_shape)?;

        // Now split the full tensor along shard_dim for the new world size.
        let full_dim_size = full_shape[shard_dim];
        let new_shard_sizes = compute_shard_sizes(full_dim_size, new_world_size);
        let new_offset: usize = new_shard_sizes[..new_rank].iter().sum();
        let new_size = new_shard_sizes[new_rank];

        // Extract this rank's slice from the full data.
        let mut new_shape = full_shape.clone();
        new_shape[shard_dim] = new_size;

        let new_data = slice_along_dim(&full_data, full_shape, shard_dim, new_offset, new_size);

        let tensor =
            Tensor::from_storage(TensorStorage::cpu(new_data), new_shape, false).map_err(|e| {
                DistCheckpointError::Tensor {
                    message: format!("creating resharded tensor \"{name}\": {e}"),
                }
            })?;

        result.insert(name.clone(), tensor);
    }

    Ok(result)
}

/// Compute how to divide `total` elements across `num_parts` as evenly as
/// possible. The first `total % num_parts` parts get one extra element.
fn compute_shard_sizes(total: usize, num_parts: usize) -> Vec<usize> {
    let base = total / num_parts;
    let remainder = total % num_parts;
    (0..num_parts)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}

/// Concatenate tensor data from multiple shards along a given dimension.
///
/// Each entry in `shard_datas` is a flat (row-major) data array for that shard.
/// `shard_shapes` gives the shape of each shard. `full_shape` is the expected
/// output shape after concatenation.
fn concat_along_dim<T: Float>(
    shard_datas: &[Vec<T>],
    shard_shapes: &[Vec<usize>],
    dim: usize,
    full_shape: &[usize],
) -> Result<Vec<T>, DistCheckpointError> {
    let ndim = full_shape.len();
    if dim >= ndim {
        return Err(DistCheckpointError::InvalidArgument {
            message: format!("shard_dim {dim} >= ndim {ndim}"),
        });
    }

    let full_numel: usize = full_shape.iter().product();
    let mut full_data = vec![<T as num_traits::Zero>::zero(); full_numel];

    // For concatenation along `dim`, we think of the tensor as having three
    // logical parts:
    //   outer = product of dims before `dim`
    //   middle = dim (varies per shard)
    //   inner = product of dims after `dim`
    let outer: usize = full_shape[..dim].iter().product();
    let inner: usize = full_shape[dim + 1..].iter().product();
    let full_middle = full_shape[dim];

    // Walk through shards and copy their data into the right offsets.
    let mut dim_offset = 0;
    for (shard_idx, shard_data) in shard_datas.iter().enumerate() {
        let shard_middle = shard_shapes[shard_idx][dim];

        // Validate shard shape dimensions except along the shard dim.
        for d in 0..ndim {
            if d != dim && shard_shapes[shard_idx][d] != full_shape[d] {
                return Err(DistCheckpointError::Tensor {
                    message: format!(
                        "shard {shard_idx} has shape {:?} but expected dim {d} to be {} (full shape {full_shape:?})",
                        shard_shapes[shard_idx], full_shape[d]
                    ),
                });
            }
        }

        for o in 0..outer {
            let src_start = o * shard_middle * inner;
            let dst_start = o * full_middle * inner + dim_offset * inner;
            let count = shard_middle * inner;

            full_data[dst_start..dst_start + count]
                .copy_from_slice(&shard_data[src_start..src_start + count]);
        }

        dim_offset += shard_middle;
    }

    if dim_offset != full_middle {
        return Err(DistCheckpointError::Tensor {
            message: format!(
                "shard sizes along dim {dim} sum to {dim_offset}, expected {full_middle}"
            ),
        });
    }

    Ok(full_data)
}

/// Extract a contiguous slice of a tensor along a given dimension.
///
/// Returns the flat data for `shape` with `shape[dim]` replaced by `size`,
/// starting at offset `offset` along that dimension.
fn slice_along_dim<T: Float>(
    data: &[T],
    shape: &[usize],
    dim: usize,
    offset: usize,
    size: usize,
) -> Vec<T> {
    let outer: usize = shape[..dim].iter().product();
    let full_middle = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let out_numel = outer * size * inner;
    let mut result = Vec::with_capacity(out_numel);

    for o in 0..outer {
        let src_start = o * full_middle * inner + offset * inner;
        let count = size * inner;
        result.extend_from_slice(&data[src_start..src_start + count]);
    }

    result
}

// ---------------------------------------------------------------------------
// Async checkpointing
// ---------------------------------------------------------------------------

/// Result of an asynchronous checkpoint operation.
///
/// The checkpoint is being written in a background thread. Call [`wait`](Self::wait)
/// to block until completion and retrieve any error.
pub struct CheckpointFuture {
    handle: Option<thread::JoinHandle<Result<(), DistCheckpointError>>>,
    /// Cached result after the thread has been joined.
    result: Option<Result<(), DistCheckpointError>>,
}

impl CheckpointFuture {
    /// Block until the checkpoint write completes.
    ///
    /// Returns `Ok(())` if the write succeeded, or the error encountered
    /// during serialization/I/O.
    ///
    /// Calling `wait()` multiple times is safe — subsequent calls return
    /// the cached result.
    pub fn wait(&mut self) -> Result<(), DistCheckpointError> {
        if let Some(handle) = self.handle.take() {
            let res = handle
                .join()
                .map_err(|_| DistCheckpointError::AsyncFailed {
                    message: "background checkpoint thread panicked".into(),
                })?;
            self.result = Some(res);
        }

        match &self.result {
            Some(Ok(())) => Ok(()),
            Some(Err(e)) => Err(DistCheckpointError::AsyncFailed {
                message: format!("{e}"),
            }),
            None => Err(DistCheckpointError::AsyncFailed {
                message: "no checkpoint was started".into(),
            }),
        }
    }

    /// Returns `true` if the background write has completed (success or failure).
    pub fn is_done(&self) -> bool {
        if self.result.is_some() {
            return true;
        }
        match &self.handle {
            Some(h) => h.is_finished(),
            None => true,
        }
    }
}

/// Asynchronous checkpointer that writes shards to disk in a background thread.
///
/// Before writing, tensor data is staged (copied) into CPU memory so that
/// training can continue on GPU without blocking. The actual file I/O happens
/// in a separate thread.
///
/// # Usage
///
/// ```ignore
/// let ckpt = AsyncCheckpointer::new(
///     checkpoint_dir.clone(),
///     rank,
///     world_size,
///     shard_metadata.clone(),
/// );
///
/// // Non-blocking: training continues immediately.
/// let mut future = ckpt.save_async(&state_dict)?;
///
/// // ... continue training ...
///
/// // When you need to be sure it finished:
/// future.wait()?;
/// ```
pub struct AsyncCheckpointer {
    dir: PathBuf,
    rank: usize,
    world_size: usize,
    shard_spec: ShardMetadata,
    /// Guard against concurrent saves: only one checkpoint at a time.
    in_flight: Arc<Mutex<bool>>,
}

impl AsyncCheckpointer {
    /// Create a new async checkpointer.
    ///
    /// - `dir`: directory where shard files will be written.
    /// - `rank`: this process's rank.
    /// - `world_size`: total number of ranks.
    /// - `shard_spec`: metadata describing how tensors are sharded.
    pub fn new(dir: PathBuf, rank: usize, world_size: usize, shard_spec: ShardMetadata) -> Self {
        Self {
            dir,
            rank,
            world_size,
            shard_spec,
            in_flight: Arc::new(Mutex::new(false)),
        }
    }

    /// The checkpoint directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// This process's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of ranks.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Start an asynchronous checkpoint.
    ///
    /// This immediately copies all tensor data to CPU-owned `Vec`s (staging),
    /// then spawns a background thread that serializes and writes the shard
    /// file. For GPU tensors, the staging step transfers data to host memory.
    ///
    /// Returns a [`CheckpointFuture`] that can be polled or waited on.
    ///
    /// # Errors
    ///
    /// Returns an error immediately if another async save is already in flight,
    /// or if staging (GPU-to-CPU copy) fails.
    pub fn save_async(
        &self,
        state_dict: &HashMap<String, Tensor<f32>>,
    ) -> Result<CheckpointFuture, DistCheckpointError> {
        // Check that no other save is in progress.
        {
            let mut guard =
                self.in_flight
                    .lock()
                    .map_err(|e| DistCheckpointError::AsyncFailed {
                        message: format!("lock poisoned: {e}"),
                    })?;
            if *guard {
                return Err(DistCheckpointError::AsyncFailed {
                    message: "another async checkpoint is already in flight".into(),
                });
            }
            *guard = true;
        }

        // Stage: copy all tensor data to CPU-owned Vecs. This is the only part
        // that touches GPU memory and must happen on the calling thread.
        let mut staged: HashMap<String, (Vec<f32>, Vec<usize>)> =
            HashMap::with_capacity(state_dict.len());

        for (name, tensor) in state_dict {
            let data = tensor.data_vec().map_err(|e| {
                // Release the lock on error.
                if let Ok(mut g) = self.in_flight.lock() {
                    *g = false;
                }
                DistCheckpointError::Tensor {
                    message: format!("staging tensor \"{name}\": {e}"),
                }
            })?;
            let shape = tensor.shape().to_vec();
            staged.insert(name.clone(), (data, shape));
        }

        // Capture everything the background thread needs.
        let dir = self.dir.clone();
        let rank = self.rank;
        let shard_spec = self.shard_spec.clone();
        let in_flight = Arc::clone(&self.in_flight);

        let handle = thread::spawn(move || {
            let result = (|| -> Result<(), DistCheckpointError> {
                // Rebuild tensors from staged data.
                let mut tensors: HashMap<String, Tensor<f32>> =
                    HashMap::with_capacity(staged.len());
                for (name, (data, shape)) in staged {
                    let tensor = Tensor::from_storage(TensorStorage::cpu(data), shape, false)
                        .map_err(|e| DistCheckpointError::Tensor {
                            message: format!("rebuilding tensor \"{name}\": {e}"),
                        })?;
                    tensors.insert(name, tensor);
                }

                // Write to disk.
                std::fs::create_dir_all(&dir)?;
                let path = shard_path(&dir, rank);
                save_tensors_to_file(&tensors, &path)?;

                // Rank 0 writes metadata.
                if rank == 0 {
                    let json = serde_json::to_string_pretty(&shard_spec).map_err(|e| {
                        DistCheckpointError::Serialization {
                            message: format!("serializing metadata: {e}"),
                        }
                    })?;
                    std::fs::write(metadata_path(&dir), json)?;
                }

                Ok(())
            })();

            // Release the in-flight guard.
            if let Ok(mut g) = in_flight.lock() {
                *g = false;
            }

            result
        });

        Ok(CheckpointFuture {
            handle: Some(handle),
            result: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience: auto-generate ShardMetadata from a flat state dict
// ---------------------------------------------------------------------------

/// Build a [`ShardMetadata`] where every tensor is sharded along dimension 0
/// with equal shard sizes.
///
/// This is the common case for FSDP where parameters are simply chunked along
/// the flattened (dim-0) axis.
pub fn flat_shard_metadata(
    state_dict: &HashMap<String, Tensor<f32>>,
    world_size: usize,
) -> ShardMetadata {
    let mut tensor_specs = HashMap::new();
    for (name, tensor) in state_dict {
        let shape = tensor.shape();
        // For FSDP-style flat sharding, the shard tensor is 1-D with shape
        // [chunk_size]. The full tensor has shape [chunk_size * world_size].
        let shard_numel = shape.iter().product::<usize>();
        let full_numel = shard_numel * world_size;
        let shard_sizes = vec![shard_numel; world_size];
        tensor_specs.insert(
            name.clone(),
            TensorShardSpec {
                full_shape: vec![full_numel],
                shard_dim: 0,
                shard_sizes,
            },
        );
    }
    ShardMetadata {
        num_ranks: world_size,
        tensor_specs,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;
    use std::collections::HashMap;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join("ferrotorch_test_dist_ckpt")
            .join(name)
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    // --- save_distributed / load_distributed roundtrip ---

    #[test]
    fn test_save_load_single_rank() {
        let dir = temp_dir("single_rank");
        cleanup(&dir);

        let mut state: HashMap<String, Tensor<f32>> = HashMap::new();
        state.insert(
            "weight".into(),
            make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
        );
        state.insert("bias".into(), make_tensor(vec![0.1, 0.2], vec![2]));

        let spec = ShardMetadata {
            num_ranks: 1,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "weight".into(),
                    TensorShardSpec {
                        full_shape: vec![4],
                        shard_dim: 0,
                        shard_sizes: vec![4],
                    },
                );
                m.insert(
                    "bias".into(),
                    TensorShardSpec {
                        full_shape: vec![2],
                        shard_dim: 0,
                        shard_sizes: vec![2],
                    },
                );
                m
            },
        };

        save_distributed(&state, &dir, 0, 1, &spec).unwrap();
        let loaded = load_distributed::<f32>(&dir, 0, 1).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["weight"].data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded["bias"].data().unwrap(), &[0.1, 0.2]);

        cleanup(&dir);
    }

    #[test]
    fn test_save_load_two_ranks() {
        let dir = temp_dir("two_ranks");
        cleanup(&dir);

        // Rank 0 shard: first half of weight.
        let mut state0: HashMap<String, Tensor<f32>> = HashMap::new();
        state0.insert("weight".into(), make_tensor(vec![1.0, 2.0], vec![2]));

        // Rank 1 shard: second half of weight.
        let mut state1: HashMap<String, Tensor<f32>> = HashMap::new();
        state1.insert("weight".into(), make_tensor(vec![3.0, 4.0], vec![2]));

        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "weight".into(),
                    TensorShardSpec {
                        full_shape: vec![4],
                        shard_dim: 0,
                        shard_sizes: vec![2, 2],
                    },
                );
                m
            },
        };

        save_distributed(&state0, &dir, 0, 2, &spec).unwrap();
        save_distributed(&state1, &dir, 1, 2, &spec).unwrap();

        // Load each rank's shard (same world size).
        let loaded0 = load_distributed::<f32>(&dir, 0, 2).unwrap();
        let loaded1 = load_distributed::<f32>(&dir, 1, 2).unwrap();

        assert_eq!(loaded0["weight"].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(loaded1["weight"].data().unwrap(), &[3.0, 4.0]);

        cleanup(&dir);
    }

    // --- Resharding tests ---

    #[test]
    fn test_reshard_2_to_4() {
        // Saved with 2 ranks, load with 4 ranks.
        let dir = temp_dir("reshard_2_to_4");
        cleanup(&dir);

        // Full tensor: [1, 2, 3, 4, 5, 6, 7, 8], shape [8].
        // 2 ranks: rank 0 = [1,2,3,4], rank 1 = [5,6,7,8].
        let mut state0: HashMap<String, Tensor<f32>> = HashMap::new();
        state0.insert("w".into(), make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]));

        let mut state1: HashMap<String, Tensor<f32>> = HashMap::new();
        state1.insert("w".into(), make_tensor(vec![5.0, 6.0, 7.0, 8.0], vec![4]));

        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![8],
                        shard_dim: 0,
                        shard_sizes: vec![4, 4],
                    },
                );
                m
            },
        };

        save_distributed(&state0, &dir, 0, 2, &spec).unwrap();
        save_distributed(&state1, &dir, 1, 2, &spec).unwrap();

        // Reshard to 4 ranks. Each rank gets 2 elements.
        let r0 = reshard::<f32>(&dir, 2, 4, 0).unwrap();
        let r1 = reshard::<f32>(&dir, 2, 4, 1).unwrap();
        let r2 = reshard::<f32>(&dir, 2, 4, 2).unwrap();
        let r3 = reshard::<f32>(&dir, 2, 4, 3).unwrap();

        assert_eq!(r0["w"].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(r1["w"].data().unwrap(), &[3.0, 4.0]);
        assert_eq!(r2["w"].data().unwrap(), &[5.0, 6.0]);
        assert_eq!(r3["w"].data().unwrap(), &[7.0, 8.0]);

        cleanup(&dir);
    }

    #[test]
    fn test_reshard_4_to_2() {
        // Saved with 4 ranks, load with 2.
        let dir = temp_dir("reshard_4_to_2");
        cleanup(&dir);

        let spec = ShardMetadata {
            num_ranks: 4,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![8],
                        shard_dim: 0,
                        shard_sizes: vec![2, 2, 2, 2],
                    },
                );
                m
            },
        };

        for rank in 0..4 {
            let start = rank as f32 * 2.0 + 1.0;
            let mut state: HashMap<String, Tensor<f32>> = HashMap::new();
            state.insert("w".into(), make_tensor(vec![start, start + 1.0], vec![2]));
            save_distributed(&state, &dir, rank, 4, &spec).unwrap();
        }

        // Reshard to 2 ranks. Each rank gets 4 elements.
        let r0 = reshard::<f32>(&dir, 4, 2, 0).unwrap();
        let r1 = reshard::<f32>(&dir, 4, 2, 1).unwrap();

        assert_eq!(r0["w"].data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r1["w"].data().unwrap(), &[5.0, 6.0, 7.0, 8.0]);

        cleanup(&dir);
    }

    #[test]
    fn test_reshard_2d_tensor() {
        // Full tensor: [[1,2,3],[4,5,6],[7,8,9],[10,11,12]], shape [4, 3].
        // Shard along dim 0: rank 0 gets rows 0-1, rank 1 gets rows 2-3.
        let dir = temp_dir("reshard_2d");
        cleanup(&dir);

        let mut state0: HashMap<String, Tensor<f32>> = HashMap::new();
        state0.insert(
            "w".into(),
            make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        );

        let mut state1: HashMap<String, Tensor<f32>> = HashMap::new();
        state1.insert(
            "w".into(),
            make_tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]),
        );

        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![4, 3],
                        shard_dim: 0,
                        shard_sizes: vec![2, 2],
                    },
                );
                m
            },
        };

        save_distributed(&state0, &dir, 0, 2, &spec).unwrap();
        save_distributed(&state1, &dir, 1, 2, &spec).unwrap();

        // Reshard to 4 ranks: each gets 1 row.
        let r0 = reshard::<f32>(&dir, 2, 4, 0).unwrap();
        let r1 = reshard::<f32>(&dir, 2, 4, 1).unwrap();
        let r2 = reshard::<f32>(&dir, 2, 4, 2).unwrap();
        let r3 = reshard::<f32>(&dir, 2, 4, 3).unwrap();

        assert_eq!(r0["w"].shape(), &[1, 3]);
        assert_eq!(r0["w"].data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(r1["w"].shape(), &[1, 3]);
        assert_eq!(r1["w"].data().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(r2["w"].shape(), &[1, 3]);
        assert_eq!(r2["w"].data().unwrap(), &[7.0, 8.0, 9.0]);
        assert_eq!(r3["w"].shape(), &[1, 3]);
        assert_eq!(r3["w"].data().unwrap(), &[10.0, 11.0, 12.0]);

        cleanup(&dir);
    }

    #[test]
    fn test_reshard_dim1() {
        // Full tensor: [[1,2,3,4],[5,6,7,8]], shape [2, 4].
        // Shard along dim 1: rank 0 gets cols 0-1, rank 1 gets cols 2-3.
        let dir = temp_dir("reshard_dim1");
        cleanup(&dir);

        let mut state0: HashMap<String, Tensor<f32>> = HashMap::new();
        state0.insert(
            "w".into(),
            make_tensor(vec![1.0, 2.0, 5.0, 6.0], vec![2, 2]),
        );

        let mut state1: HashMap<String, Tensor<f32>> = HashMap::new();
        state1.insert(
            "w".into(),
            make_tensor(vec![3.0, 4.0, 7.0, 8.0], vec![2, 2]),
        );

        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![2, 4],
                        shard_dim: 1,
                        shard_sizes: vec![2, 2],
                    },
                );
                m
            },
        };

        save_distributed(&state0, &dir, 0, 2, &spec).unwrap();
        save_distributed(&state1, &dir, 1, 2, &spec).unwrap();

        // Reshard to 4 ranks: each gets 1 column.
        let r0 = reshard::<f32>(&dir, 2, 4, 0).unwrap();
        let r1 = reshard::<f32>(&dir, 2, 4, 1).unwrap();
        let r2 = reshard::<f32>(&dir, 2, 4, 2).unwrap();
        let r3 = reshard::<f32>(&dir, 2, 4, 3).unwrap();

        assert_eq!(r0["w"].shape(), &[2, 1]);
        assert_eq!(r0["w"].data().unwrap(), &[1.0, 5.0]);
        assert_eq!(r1["w"].shape(), &[2, 1]);
        assert_eq!(r1["w"].data().unwrap(), &[2.0, 6.0]);
        assert_eq!(r2["w"].shape(), &[2, 1]);
        assert_eq!(r2["w"].data().unwrap(), &[3.0, 7.0]);
        assert_eq!(r3["w"].shape(), &[2, 1]);
        assert_eq!(r3["w"].data().unwrap(), &[4.0, 8.0]);

        cleanup(&dir);
    }

    #[test]
    fn test_reshard_3_to_2_uneven() {
        // Full tensor: [1,2,3,4,5,6,7,8,9], shape [9].
        // 3 ranks with equal shards of 3 each.
        let dir = temp_dir("reshard_3_to_2");
        cleanup(&dir);

        let spec = ShardMetadata {
            num_ranks: 3,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![9],
                        shard_dim: 0,
                        shard_sizes: vec![3, 3, 3],
                    },
                );
                m
            },
        };

        for rank in 0..3usize {
            let start = rank as f32 * 3.0 + 1.0;
            let mut state: HashMap<String, Tensor<f32>> = HashMap::new();
            state.insert(
                "w".into(),
                make_tensor(vec![start, start + 1.0, start + 2.0], vec![3]),
            );
            save_distributed(&state, &dir, rank, 3, &spec).unwrap();
        }

        // Reshard to 2 ranks: 9 / 2 = 4 rem 1, so rank 0 gets 5, rank 1 gets 4.
        let r0 = reshard::<f32>(&dir, 3, 2, 0).unwrap();
        let r1 = reshard::<f32>(&dir, 3, 2, 1).unwrap();

        assert_eq!(r0["w"].data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(r1["w"].data().unwrap(), &[6.0, 7.0, 8.0, 9.0]);

        cleanup(&dir);
    }

    // --- load_distributed triggers resharding ---

    #[test]
    fn test_load_distributed_reshards_when_world_size_differs() {
        let dir = temp_dir("load_reshard");
        cleanup(&dir);

        // Save with 2 ranks.
        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![4],
                        shard_dim: 0,
                        shard_sizes: vec![2, 2],
                    },
                );
                m
            },
        };

        let mut s0: HashMap<String, Tensor<f32>> = HashMap::new();
        s0.insert("w".into(), make_tensor(vec![1.0, 2.0], vec![2]));
        save_distributed(&s0, &dir, 0, 2, &spec).unwrap();

        let mut s1: HashMap<String, Tensor<f32>> = HashMap::new();
        s1.insert("w".into(), make_tensor(vec![3.0, 4.0], vec![2]));
        save_distributed(&s1, &dir, 1, 2, &spec).unwrap();

        // Load with 4 ranks — triggers resharding.
        let r0 = load_distributed::<f32>(&dir, 0, 4).unwrap();
        let r1 = load_distributed::<f32>(&dir, 1, 4).unwrap();
        let r2 = load_distributed::<f32>(&dir, 2, 4).unwrap();
        let r3 = load_distributed::<f32>(&dir, 3, 4).unwrap();

        assert_eq!(r0["w"].data().unwrap(), &[1.0]);
        assert_eq!(r1["w"].data().unwrap(), &[2.0]);
        assert_eq!(r2["w"].data().unwrap(), &[3.0]);
        assert_eq!(r3["w"].data().unwrap(), &[4.0]);

        cleanup(&dir);
    }

    // --- Metadata serialization ---

    #[test]
    fn test_metadata_roundtrip() {
        let spec = ShardMetadata {
            num_ranks: 4,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "layer.weight".into(),
                    TensorShardSpec {
                        full_shape: vec![256, 512],
                        shard_dim: 0,
                        shard_sizes: vec![64, 64, 64, 64],
                    },
                );
                m.insert(
                    "layer.bias".into(),
                    TensorShardSpec {
                        full_shape: vec![256],
                        shard_dim: 0,
                        shard_sizes: vec![64, 64, 64, 64],
                    },
                );
                m
            },
        };

        let json = serde_json::to_string_pretty(&spec).unwrap();
        let loaded: ShardMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.num_ranks, 4);
        assert_eq!(loaded.tensor_specs.len(), 2);
        assert_eq!(
            loaded.tensor_specs["layer.weight"].full_shape,
            vec![256, 512]
        );
        assert_eq!(loaded.tensor_specs["layer.weight"].shard_dim, 0);
        assert_eq!(
            loaded.tensor_specs["layer.weight"].shard_sizes,
            vec![64, 64, 64, 64]
        );
    }

    // --- compute_shard_sizes ---

    #[test]
    fn test_compute_shard_sizes_even() {
        assert_eq!(compute_shard_sizes(8, 4), vec![2, 2, 2, 2]);
        assert_eq!(compute_shard_sizes(12, 3), vec![4, 4, 4]);
    }

    #[test]
    fn test_compute_shard_sizes_uneven() {
        // 9 / 2 = 4 rem 1 -> [5, 4]
        assert_eq!(compute_shard_sizes(9, 2), vec![5, 4]);
        // 10 / 3 = 3 rem 1 -> [4, 3, 3]
        assert_eq!(compute_shard_sizes(10, 3), vec![4, 3, 3]);
        // 7 / 4 = 1 rem 3 -> [2, 2, 2, 1]
        assert_eq!(compute_shard_sizes(7, 4), vec![2, 2, 2, 1]);
    }

    // --- concat_along_dim ---

    #[test]
    fn test_concat_1d() {
        let data0 = vec![1.0f32, 2.0];
        let data1 = vec![3.0f32, 4.0, 5.0];
        let full_shape = vec![5];

        let result =
            concat_along_dim(&[data0, data1], &[vec![2], vec![3]], 0, &full_shape).unwrap();

        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concat_2d_dim0() {
        // Two 1x3 matrices -> 2x3 matrix.
        let data0 = vec![1.0f32, 2.0, 3.0];
        let data1 = vec![4.0f32, 5.0, 6.0];
        let full_shape = vec![2, 3];

        let result =
            concat_along_dim(&[data0, data1], &[vec![1, 3], vec![1, 3]], 0, &full_shape).unwrap();

        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_2d_dim1() {
        // Two 2x1 matrices -> 2x2 matrix.
        // data0 = [[1],[3]], data1 = [[2],[4]]
        let data0 = vec![1.0f32, 3.0];
        let data1 = vec![2.0f32, 4.0];
        let full_shape = vec![2, 2];

        let result =
            concat_along_dim(&[data0, data1], &[vec![2, 1], vec![2, 1]], 1, &full_shape).unwrap();

        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // --- slice_along_dim ---

    #[test]
    fn test_slice_1d() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];

        let s0 = slice_along_dim(&data, &shape, 0, 0, 2);
        assert_eq!(s0, vec![1.0, 2.0]);

        let s1 = slice_along_dim(&data, &shape, 0, 2, 3);
        assert_eq!(s1, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_slice_2d_dim0() {
        // [[1,2,3],[4,5,6],[7,8,9],[10,11,12]], shape [4,3]
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let shape = vec![4, 3];

        let s = slice_along_dim(&data, &shape, 0, 1, 2);
        assert_eq!(s, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_slice_2d_dim1() {
        // [[1,2,3,4],[5,6,7,8]], shape [2,4]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = vec![2, 4];

        let s = slice_along_dim(&data, &shape, 1, 1, 2);
        assert_eq!(s, vec![2.0, 3.0, 6.0, 7.0]);
    }

    // --- flat_shard_metadata ---

    #[test]
    fn test_flat_shard_metadata() {
        let mut state: HashMap<String, Tensor<f32>> = HashMap::new();
        state.insert("w".into(), make_tensor(vec![1.0, 2.0, 3.0], vec![3]));

        let meta = flat_shard_metadata(&state, 4);
        assert_eq!(meta.num_ranks, 4);

        let spec = &meta.tensor_specs["w"];
        assert_eq!(spec.full_shape, vec![12]); // 3 * 4
        assert_eq!(spec.shard_dim, 0);
        assert_eq!(spec.shard_sizes, vec![3, 3, 3, 3]);
    }

    // --- AsyncCheckpointer ---

    #[test]
    fn test_async_checkpoint_basic() {
        let dir = temp_dir("async_basic");
        cleanup(&dir);

        let mut state: HashMap<String, Tensor<f32>> = HashMap::new();
        state.insert("w".into(), make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]));

        let spec = ShardMetadata {
            num_ranks: 1,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![4],
                        shard_dim: 0,
                        shard_sizes: vec![4],
                    },
                );
                m
            },
        };

        let ckpt = AsyncCheckpointer::new(dir.clone(), 0, 1, spec);
        let mut future = ckpt.save_async(&state).unwrap();
        future.wait().unwrap();

        // Verify the file was written correctly.
        let loaded = load_distributed::<f32>(&dir, 0, 1).unwrap();
        assert_eq!(loaded["w"].data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        cleanup(&dir);
    }

    #[test]
    fn test_async_checkpoint_wait_idempotent() {
        let dir = temp_dir("async_idempotent");
        cleanup(&dir);

        let mut state: HashMap<String, Tensor<f32>> = HashMap::new();
        state.insert("x".into(), make_tensor(vec![42.0], vec![1]));

        let spec = ShardMetadata {
            num_ranks: 1,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "x".into(),
                    TensorShardSpec {
                        full_shape: vec![1],
                        shard_dim: 0,
                        shard_sizes: vec![1],
                    },
                );
                m
            },
        };

        let ckpt = AsyncCheckpointer::new(dir.clone(), 0, 1, spec);
        let mut future = ckpt.save_async(&state).unwrap();

        // Wait twice — should not panic.
        future.wait().unwrap();
        future.wait().unwrap();

        cleanup(&dir);
    }

    #[test]
    fn test_async_checkpoint_is_done() {
        let dir = temp_dir("async_is_done");
        cleanup(&dir);

        let mut state: HashMap<String, Tensor<f32>> = HashMap::new();
        state.insert("x".into(), make_tensor(vec![1.0], vec![1]));

        let spec = ShardMetadata {
            num_ranks: 1,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "x".into(),
                    TensorShardSpec {
                        full_shape: vec![1],
                        shard_dim: 0,
                        shard_sizes: vec![1],
                    },
                );
                m
            },
        };

        let ckpt = AsyncCheckpointer::new(dir.clone(), 0, 1, spec);
        let mut future = ckpt.save_async(&state).unwrap();
        future.wait().unwrap();
        assert!(future.is_done());

        cleanup(&dir);
    }

    // --- Error cases ---

    #[test]
    fn test_save_invalid_rank() {
        let dir = temp_dir("invalid_rank");
        let state: HashMap<String, Tensor<f32>> = HashMap::new();
        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: HashMap::new(),
        };

        let result = save_distributed(&state, &dir, 5, 2, &spec);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_missing_metadata() {
        let dir = temp_dir("missing_meta");
        cleanup(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let result = load_distributed::<f32>(&dir, 0, 1);
        assert!(result.is_err());

        cleanup(&dir);
    }

    #[test]
    fn test_load_missing_shard() {
        let dir = temp_dir("missing_shard");
        cleanup(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Write metadata but no shard file.
        let spec = ShardMetadata {
            num_ranks: 1,
            tensor_specs: HashMap::new(),
        };
        let json = serde_json::to_string_pretty(&spec).unwrap();
        std::fs::write(metadata_path(&dir), json).unwrap();

        // This should work (empty tensor_specs, no shard needed for the direct load).
        // But if we try to load with a different world_size triggering reshard,
        // it needs the shard files.
        let loaded = load_distributed::<f32>(&dir, 0, 1).unwrap();
        assert!(loaded.is_empty());

        cleanup(&dir);
    }

    // --- Multiple tensors ---

    #[test]
    fn test_reshard_multiple_tensors() {
        let dir = temp_dir("reshard_multi");
        cleanup(&dir);

        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "weight".into(),
                    TensorShardSpec {
                        full_shape: vec![4],
                        shard_dim: 0,
                        shard_sizes: vec![2, 2],
                    },
                );
                m.insert(
                    "bias".into(),
                    TensorShardSpec {
                        full_shape: vec![6],
                        shard_dim: 0,
                        shard_sizes: vec![3, 3],
                    },
                );
                m
            },
        };

        let mut s0: HashMap<String, Tensor<f32>> = HashMap::new();
        s0.insert("weight".into(), make_tensor(vec![1.0, 2.0], vec![2]));
        s0.insert("bias".into(), make_tensor(vec![10.0, 20.0, 30.0], vec![3]));

        let mut s1: HashMap<String, Tensor<f32>> = HashMap::new();
        s1.insert("weight".into(), make_tensor(vec![3.0, 4.0], vec![2]));
        s1.insert("bias".into(), make_tensor(vec![40.0, 50.0, 60.0], vec![3]));

        save_distributed(&s0, &dir, 0, 2, &spec).unwrap();
        save_distributed(&s1, &dir, 1, 2, &spec).unwrap();

        // Reshard to 1 rank (consolidate).
        let r0 = load_distributed::<f32>(&dir, 0, 1).unwrap();

        assert_eq!(r0["weight"].data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            r0["bias"].data().unwrap(),
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        );

        cleanup(&dir);
    }

    // --- DistributedCheckpoint struct ---

    #[test]
    fn test_distributed_checkpoint_struct() {
        let ckpt = DistributedCheckpoint {
            checkpoint_dir: PathBuf::from("/tmp/test"),
            shard_metadata: ShardMetadata {
                num_ranks: 2,
                tensor_specs: HashMap::new(),
            },
        };
        assert_eq!(ckpt.checkpoint_dir, PathBuf::from("/tmp/test"));
        assert_eq!(ckpt.shard_metadata.num_ranks, 2);
    }

    // Edge case: reshard to same world_size (no-op path through reshard fn)
    #[test]
    fn test_reshard_same_world_size() {
        let dir = temp_dir("reshard_same");
        cleanup(&dir);

        let spec = ShardMetadata {
            num_ranks: 2,
            tensor_specs: {
                let mut m = HashMap::new();
                m.insert(
                    "w".into(),
                    TensorShardSpec {
                        full_shape: vec![4],
                        shard_dim: 0,
                        shard_sizes: vec![2, 2],
                    },
                );
                m
            },
        };

        let mut s0: HashMap<String, Tensor<f32>> = HashMap::new();
        s0.insert("w".into(), make_tensor(vec![1.0, 2.0], vec![2]));
        save_distributed(&s0, &dir, 0, 2, &spec).unwrap();

        let mut s1: HashMap<String, Tensor<f32>> = HashMap::new();
        s1.insert("w".into(), make_tensor(vec![3.0, 4.0], vec![2]));
        save_distributed(&s1, &dir, 1, 2, &spec).unwrap();

        // Reshard with same world_size — should produce identical results.
        let r0 = reshard::<f32>(&dir, 2, 2, 0).unwrap();
        let r1 = reshard::<f32>(&dir, 2, 2, 1).unwrap();

        assert_eq!(r0["w"].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(r1["w"].data().unwrap(), &[3.0, 4.0]);

        cleanup(&dir);
    }
}
