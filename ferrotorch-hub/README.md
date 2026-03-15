# ferrotorch-hub

Pretrained model registry, download, and caching for ferrotorch.

## What it provides

- **`list_models`** -- browse all registered pretrained model architectures
- **`get_model_info`** -- query metadata (name, description, parameter count, weights format)
- **`load_pretrained`** -- load pretrained state dicts from the local cache
- **`download_weights`** -- fetch model weights from a URL and cache locally
- **`HubCache`** -- manage the on-disk cache directory (`~/.cache/ferrotorch/hub`)

Mirrors the workflow of `torch.hub` and `torchvision.models` with pretrained weight support.

## Quick start

```rust
use ferrotorch_hub::{list_models, get_model_info, load_pretrained};

// Browse available models.
for model in list_models() {
    println!("{}: {} ({} params)", model.name, model.description, model.num_parameters);
}

// Load pretrained weights (requires cached weights on disk).
let state_dict = load_pretrained::<f32>("resnet50").unwrap();
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
