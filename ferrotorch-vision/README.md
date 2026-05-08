# ferrotorch-vision

Vision models, datasets, and transforms for ferrotorch — covering classification, detection, and segmentation.

## What it provides

### Model architectures

**Classification** (8 architectures with full forward parity):

| Architecture      | Variants              |
|-------------------|-----------------------|
| ResNet            | 18, 34, 50            |
| VGG               | 11, 16                |
| EfficientNet      | B0                    |
| MobileNetV2       | standard              |
| MobileNetV3       | Small                 |
| ConvNeXt          | Tiny                  |
| Swin Transformer  | Tiny                  |
| ViT               | B/16                  |
| DenseNet          | 121                   |
| InceptionV3       | standard              |

**Object detection** (3 architectures):

| Architecture      | Notes                                     |
|-------------------|-------------------------------------------|
| Faster R-CNN      | ResNet-50 backbone with FPN               |
| Mask R-CNN        | Faster R-CNN + mask head                  |
| SSD300            | Single Shot Detector, 300x300 input       |

**Segmentation** (2 architectures):

| Architecture      | Notes                                     |
|-------------------|-------------------------------------------|
| DeepLabV3         | ASPP-based segmentation                   |
| FCN               | Fully Convolutional Network               |

### Datasets

- **`Mnist`**, **`Cifar10`**, **`Cifar100`** — automatic download/extraction with `Split` (train/test)
- **`ImageFolder`** — load arbitrary directory-based classification datasets

### Image I/O

`read_image`, `read_image_as_tensor`, `write_image`, `write_tensor_as_image`, `raw_image_to_tensor`

### Transforms

`CenterCrop`, `Resize`, `VisionNormalize`, `VisionToTensor` with `IMAGENET_MEAN`/`IMAGENET_STD`

### Model registry

`register_model`, `get_model`, `list_models`, `ModelRegistry`, `ModelConstructor`

### Feature extraction

`create_feature_extractor`, `FeatureExtractor` for intermediate layer outputs

## Quick start

```rust
use ferrotorch_vision::{Mnist, Split, VisionToTensor, VisionNormalize};
use ferrotorch_vision::models::get_model;

// Load a dataset
let dataset = Mnist::new("./data", Split::Train)?;
let sample = dataset.get(0)?;
let image_tensor = VisionToTensor.transform(&sample.image)?;
let normalized = VisionNormalize::new(vec![0.1307], vec![0.3081])
    .transform(&image_tensor)?;

// Instantiate a classification model
let model = get_model::<f32>("resnet50", 1000)?;

// Instantiate a detection model
let detector = get_model::<f32>("faster_rcnn", 91)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
