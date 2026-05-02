# ferrotorch-data

Data loading, batching, transforms, and device transfer for ferrotorch.

## What it provides

- **`Dataset` trait** — random-access datasets with `len`/`get`
- **`VecDataset`** — in-memory dataset from vectors
- **`MappedDataset`** — lazy map transform over an existing dataset
- **`DataLoader`** — parallel batched iteration with:
  - Rayon-based `num_workers` for parallel per-sample loading
  - `prefetch_factor` — background thread pipeline that hides I/O latency
  - `shuffle`, `drop_last`, `seed`, custom `collate_fn`
  - Automatic device transfer via `.device()` + `ToDevice` trait
  - **`pin_memory(true)`** — page-locked host memory + DMA for ~2x faster CPU→GPU transfers (CL-378)
- **Samplers** — `SequentialSampler`, `RandomSampler`, `DistributedSampler`, custom sampler injection via `Sampler` trait
- **Transforms** — `ToTensor`, `Normalize`, `RandomCrop`, `RandomHorizontalFlip`, `Compose`, `Transform` trait with seedable RNG
- **Interop** — NumPy `.npy`/`.npz` and Apache Arrow / Parquet readers via `interop::{load_numpy, load_npz, ArrowDataset}`

## Quick start

```rust
use std::sync::Arc;
use ferrotorch_core::Device;
use ferrotorch_data::{DataLoader, VecDataset};

let dataset = Arc::new(VecDataset::new(vec![1, 2, 3, 4, 5]));
let loader = DataLoader::new(dataset, 2)
    .shuffle(true)
    .seed(42)
    .num_workers(4)
    .prefetch_factor(2);

for batch in loader.iter(0 /* epoch */) {
    let batch = batch?;
    // batch: Vec<D::Sample>
}
```

## GPU pipeline with pin_memory

```rust
use ferrotorch_data::{DataLoader, ToDevice};

let loader = DataLoader::new(dataset, 64)
    .shuffle(true)
    .num_workers(8)
    .prefetch_factor(4)
    .device(Device::Cuda(0))     // auto-transfer batches to GPU
    .pin_memory(true);           // use pinned host buffers + DMA

// The transfer happens inside the prefetch pipeline, so it overlaps
// with the consumer's training step.
```

Sample types implement the `ToDevice` trait with optional `to_device_pinned`
override for custom tensor structures. A blanket `impl<T: Float> ToDevice
for Tensor<T>` is provided for loaders that yield raw tensors.

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
