//! One-shot helper to dump ferrotorch-vision parameter / buffer keys + shapes
//! for the 5 torchvision-canonical models being pinned in #1130.
//!
//! Run with:
//!   cargo run --example dump_torchvision_keys -p ferrotorch-vision --release \
//!       > /tmp/ferrotorch_keys.json
//!
//! Output is a JSON dict { "<model_name>": { "parameters": [[name, shape], ...],
//! "buffers": [[name, shape], ...] }, ... }
//!
//! This is a generation-only tool — it is intentionally not a test. The output
//! is consumed by `scripts/pin_pretrained_weights.py` to build deterministic
//! state_dict mappings without us having to mirror the architecture in two
//! places.

use ferrotorch_nn::Module;
use ferrotorch_vision::models::detection::{
    fasterrcnn_resnet50_fpn, fcos_resnet50_fpn, keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn, ssd300_vgg16,
};
use ferrotorch_vision::models::segmentation::{
    deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large,
};

fn dump<T: ferrotorch_core::Float, M: Module<T>>(model: &M) -> serde_json::Value {
    let params: Vec<_> = model
        .named_parameters()
        .into_iter()
        .map(|(name, p)| serde_json::json!([name, p.shape()]))
        .collect();
    let buffers: Vec<_> = model
        .named_buffers()
        .into_iter()
        .map(|(name, b)| serde_json::json!([name, b.shape()]))
        .collect();
    serde_json::json!({
        "parameters": params,
        "buffers": buffers,
    })
}

fn main() {
    let ssd = ssd300_vgg16::<f32>(91).expect("ssd300_vgg16 build");
    let frcnn = fasterrcnn_resnet50_fpn::<f32>(91).expect("fasterrcnn_resnet50_fpn build");
    let mrcnn = maskrcnn_resnet50_fpn::<f32>(91).expect("maskrcnn_resnet50_fpn build");
    let dl3 = deeplabv3_resnet50::<f32>(21).expect("deeplabv3_resnet50 build");
    let fcn = fcn_resnet50::<f32>(21).expect("fcn_resnet50 build");
    // #1143: RetinaNet uses 91 classes for the COCO_V1 checkpoint (no
    // explicit background; sigmoid per-class scoring).
    let retina = retinanet_resnet50_fpn::<f32>(91).expect("retinanet_resnet50_fpn build");
    // #1144: FCOS COCO_V1 — same 91-class convention; sigmoid scoring gated
    // by a separate centerness branch.
    let fcos = fcos_resnet50_fpn::<f32>(91).expect("fcos_resnet50_fpn build");
    // #1145: Keypoint R-CNN COCO_V1 — 2 classes (bg + person), 17 keypoints.
    let krcnn = keypointrcnn_resnet50_fpn::<f32>().expect("keypointrcnn_resnet50_fpn build");
    // #1146: LRASPP MobileNetV3-Large COCO_WITH_VOC_LABELS_V1 — 21 classes
    // (Pascal VOC background + 20 foreground).
    let lraspp =
        lraspp_mobilenet_v3_large::<f32>(21).expect("lraspp_mobilenet_v3_large build");

    let out = serde_json::json!({
        "ssd300_vgg16": dump(&ssd),
        "fasterrcnn_resnet50_fpn": dump(&frcnn),
        "maskrcnn_resnet50_fpn": dump(&mrcnn),
        "deeplabv3_resnet50": dump(&dl3),
        "fcn_resnet50": dump(&fcn),
        "retinanet_resnet50_fpn": dump(&retina),
        "fcos_resnet50_fpn": dump(&fcos),
        "keypointrcnn_resnet50_fpn": dump(&krcnn),
        "lraspp_mobilenet_v3_large": dump(&lraspp),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
