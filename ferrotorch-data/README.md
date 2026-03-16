# ferrotorch-data

Data loading, batching, and transforms for ferrotorch.

## What it provides

- **`Dataset` trait** -- `Dataset` and `IterableDataset` for random-access and streaming data
- **`VecDataset`** -- in-memory dataset from vectors
- **`MappedDataset`** -- lazy map transform over an existing dataset
- **`DataLoader`** -- parallel batched iteration with rayon-based `num_workers`, custom `collate_fn` support
- **Samplers** -- `SequentialSampler`, `RandomSampler`, `DistributedSampler` (multi-rank), custom sampler injection, and the `Sampler` trait
- **Transforms** -- `ToTensor`, `Normalize`, `RandomCrop`, `RandomHorizontalFlip`, `Compose`, and the `Transform` trait with seedable RNG (`manual_seed`)

## Quick start

```rust
use ferrotorch_data::{VecDataset, DataLoader, RandomSampler};

let dataset = VecDataset::new(inputs, targets);
let sampler = RandomSampler::new(dataset.len());
let loader = DataLoader::new(dataset, 32, sampler);

for batch in loader.iter() {
    // batch.data, batch.target
}
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
