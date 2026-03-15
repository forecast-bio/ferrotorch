# ferrotorch-vision

Vision models, datasets, and transforms for ferrotorch.

## What it provides

- **Datasets** -- `Mnist`, `Cifar10`, `Cifar100` with automatic download/extraction and `Split` (train/test)
- **Image I/O** -- `read_image`, `read_image_as_tensor`, `write_image`, `write_tensor_as_image`, `raw_image_to_tensor`
- **Vision transforms** -- `CenterCrop`, `Resize`, `VisionNormalize`, `VisionToTensor` with `IMAGENET_MEAN`/`IMAGENET_STD`
- **Model registry** -- `register_model`, `get_model`, `list_models`, `ModelRegistry`, `ModelConstructor`
- **Feature extraction** -- `create_feature_extractor`, `FeatureExtractor` for intermediate layer outputs

## Quick start

```rust
use ferrotorch_vision::{Mnist, Split, VisionToTensor, VisionNormalize};

let dataset = Mnist::new("./data", Split::Train)?;
let sample = dataset.get(0)?;
let image_tensor = VisionToTensor.transform(&sample.image)?;
let normalized = VisionNormalize::new(vec![0.1307], vec![0.3081])
    .transform(&image_tensor)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
