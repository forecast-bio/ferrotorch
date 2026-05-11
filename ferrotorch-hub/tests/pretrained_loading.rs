//! Network-gated end-to-end smoke tests for the five torchvision-canonical
//! detection / segmentation models pinned in #1130.
//!
//! For each of `ssd300_vgg16`, `fasterrcnn_resnet50_fpn`,
//! `maskrcnn_resnet50_fpn`, `deeplabv3_resnet50`, `fcn_resnet50` we:
//!
//! 1. **Download + SHA-256 verify** the safetensors hosted at
//!    `huggingface.co/ferrotorch/<name>`. Any mutation of the pinned SHA
//!    in [`ferrotorch_hub::registry`] causes
//!    [`ferrotorch_hub::download_weights`] to return
//!    `Err(InvalidArgument)` — this is the sabotage probe exercised in
//!    Stage 8 of #1130.
//! 2. **Parse** through [`ferrotorch_hub::load_pretrained::<f32>`], i.e.
//!    SafeTensors → `StateDict<f32>`. The returned state dict carries the
//!    re-keyed ferrotorch parameter names + BN running stats (the running
//!    stats are silently ignored by the production `strict=false` loader
//!    in `ferrotorch-vision::models::registry::maybe_load_pretrained`
//!    until ferrotorch issue #995 lands `named_buffers()` exposure for
//!    BN, at which point this same safetensors file becomes fully
//!    consumable without re-uploading).
//! 3. **Canary-key assertion**: every model has a hand-picked set of
//!    *expected* ferrotorch parameter keys and shapes — including
//!    backbone, intermediate (FPN / ASPP), and head — that prove the
//!    file actually contains the re-keyed torchvision weights rather
//!    than a placeholder or a wrongly-mapped tensor. This is the
//!    "strict=true re-load coverage" criterion from the #1130 pre-flight.
//!
//! Why canary keys instead of a full forward pass? `ferrotorch-hub` cannot
//! depend on `ferrotorch-vision` (vision regular-deps on hub), so we
//! verify at the StateDict layer where the test crate already has access.
//! The forward-pass check is one cargo invocation deeper — see
//! `ferrotorch-vision/tests/conformance_vision_*` for runtime conformance.
//!
//! Gated behind `--ignored` and the `http` feature so offline CI does
//! not download ~830 MB on every `cargo test`. Run explicitly with:
//!
//! ```bash
//! cargo test -p ferrotorch-hub --features http -- --ignored test_pretrained
//! ```
//!
//! The `#[ignore]` reason on each test is the same — "requires network;
//! gated for offline CI". If the upstream `huggingface.co/ferrotorch/*`
//! repos or the pinned SHA changes, these tests fail loudly — and that
//! signal is the whole point of pinning.

#![cfg(feature = "http")]

use ferrotorch_hub::{HubCache, download_weights, get_model_info, load_pretrained};
use ferrotorch_nn::StateDict;

/// Returns true if the given `state_dict` contains `key` and the tensor's
/// shape equals `expected`. Used by all five smoke tests to assert that
/// the canary keys carry the right tensors (not random noise of the wrong
/// rank).
fn assert_key_shape(state_dict: &StateDict<f32>, key: &str, expected: &[usize]) {
    let t = state_dict
        .get(key)
        .unwrap_or_else(|| panic!("state_dict missing canary key '{key}'"));
    assert_eq!(
        t.shape(),
        expected,
        "canary shape mismatch for '{key}': got {:?} expected {expected:?}",
        t.shape()
    );
    // Real, non-zero data: at least one of the first 100 elements is nonzero.
    // This guards against ever uploading a zeroed-out tensor.
    let buf = t
        .data_vec()
        .expect("canary tensor must be readable via data_vec");
    let any_nonzero = buf.iter().take(100).any(|v| *v != 0.0);
    assert!(
        any_nonzero,
        "canary key '{key}' is all-zero across the first 100 elements; \
         the safetensors file likely contains placeholder data, not real \
         pretrained weights"
    );
}

/// Each entry: `(canary_key, expected_shape)`. These are sampled across
/// backbone / intermediate (FPN, ASPP) / head so a mapping bug at any
/// stage of the converter would surface immediately.
struct Canary {
    name: &'static str,
    canaries: &'static [(&'static str, &'static [usize])],
    /// Expected number of distinct keys in the safetensors (parameters +
    /// BN running stats). Computed from the dump + Python converter; a
    /// drift here proves the mapper changed and we should regenerate.
    expected_keys_in_safetensors: usize,
}

const SSD300_CANARIES: Canary = Canary {
    name: "ssd300_vgg16",
    canaries: &[
        // Backbone first conv (matches torchvision `backbone.features.0.weight`).
        ("features_stage1.0.conv.weight", &[64, 3, 3, 3]),
        // L2Norm scale on conv4_3 output.
        ("l2_norm.weight", &[512]),
        // Extra block conv (matches torchvision `backbone.extra.1.0.weight`).
        ("extra.0.0.conv.weight", &[256, 1024, 1, 1]),
        // Classification head, level 0: 91 classes × 4 anchors = 364 channels.
        ("head.cls_heads.0.weight", &[364, 512, 3, 3]),
        // Regression head, level 5: 4 box coords × 4 anchors = 16 channels.
        ("head.reg_heads.5.weight", &[16, 256, 3, 3]),
    ],
    expected_keys_in_safetensors: 71,
};

const FASTERRCNN_CANARIES: Canary = Canary {
    name: "fasterrcnn_resnet50_fpn",
    canaries: &[
        // ResNet-50 stem conv1.
        ("backbone.conv1.weight", &[64, 3, 7, 7]),
        // First bottleneck in layer3.
        ("backbone.layer3.0.conv2.weight", &[256, 256, 3, 3]),
        // FPN lateral on C5 (2048 → 256, 1×1).
        ("fpn.lateral5.weight", &[256, 2048, 1, 1]),
        // FPN output P3 (3×3 smoother).
        ("fpn.output3.weight", &[256, 256, 3, 3]),
        // RPN classification logits (3 anchors per loc, 1×1).
        ("rpn.head.cls_logits.weight", &[3, 256, 1, 1]),
        // ROI box head fc6 (256·7·7 = 12544 → 1024).
        ("head.fc6.weight", &[1024, 12544]),
        // ROI bbox predictor (4 coords × 91 classes = 364).
        ("head.bbox_pred.weight", &[364, 1024]),
    ],
    // 183 ferrotorch params + 4 BN running stats per BN layer (53 BN layers,
    // each contributes mean+var, plus the 53 weight/bias pairs are already
    // counted) = 183 + 53*2 = 289. Plus 8 intentionally-dropped FPN biases
    // are NOT in the safetensors. Computed empirically from the converter.
    expected_keys_in_safetensors: 289,
};

const MASKRCNN_CANARIES: Canary = Canary {
    name: "maskrcnn_resnet50_fpn",
    canaries: &[
        // Same backbone but under `faster_rcnn.` prefix.
        ("faster_rcnn.backbone.conv1.weight", &[64, 3, 7, 7]),
        ("faster_rcnn.fpn.lateral5.weight", &[256, 2048, 1, 1]),
        ("faster_rcnn.rpn.head.cls_logits.weight", &[3, 256, 1, 1]),
        ("faster_rcnn.head.fc6.weight", &[1024, 12544]),
        // Mask head — 4 sequential 3×3 256→256 convs.
        ("mask_head.conv1.weight", &[256, 256, 3, 3]),
        ("mask_head.conv4.weight", &[256, 256, 3, 3]),
        // Mask predictor: 2×2 deconv 256→256, then 1×1 conv to 91 classes.
        ("mask_predictor.deconv.weight", &[256, 256, 2, 2]),
        ("mask_predictor.conv_logits.weight", &[91, 256, 1, 1]),
    ],
    // 195 params + 53*2 running stats = 301.
    expected_keys_in_safetensors: 301,
};

const DEEPLABV3_CANARIES: Canary = Canary {
    name: "deeplabv3_resnet50",
    canaries: &[
        // Stem conv.
        ("backbone.conv1.weight", &[64, 3, 7, 7]),
        // ASPP branch 0 (1×1 conv inside .conv wrapper).
        ("head.aspp.0.conv.weight", &[256, 2048, 1, 1]),
        // ASPP branch 1 (3×3 atrous, NO .conv wrapper).
        ("head.aspp.1.weight", &[256, 2048, 3, 3]),
        // ASPP pooling branch (branch 4: 1×1 after global avgpool).
        ("head.aspp.4.conv.weight", &[256, 2048, 1, 1]),
        // ASPP project: 5*256 = 1280 → 256.
        ("head.aspp.project.weight", &[256, 1280, 1, 1]),
        // Final 1×1 classifier head (21 VOC classes).
        ("head.classifier.weight", &[21, 256, 1, 1]),
    ],
    // 182 params + 53 backbone-BN*2 + 7 head-BN*2 = 182 + 106 + 14 = 302.
    expected_keys_in_safetensors: 302,
};

const FCN_CANARIES: Canary = Canary {
    name: "fcn_resnet50",
    canaries: &[
        ("backbone.conv1.weight", &[64, 3, 7, 7]),
        ("backbone.layer4.2.conv3.weight", &[2048, 512, 1, 1]),
        // FCN head: 2048 → 512 (3×3) → BN → ReLU → Dropout → 512 → 21 (1×1).
        ("classifier.0.weight", &[512, 2048, 3, 3]),
        ("classifier.4.weight", &[21, 512, 1, 1]),
    ],
    // 164 params + 53*2 running stats + 1 head-BN*2 = 272.
    expected_keys_in_safetensors: 272,
};

/// Common probe: download → SHA verify → SafeTensors parse → canary key shapes.
fn run_smoke(canary: &Canary) {
    let info = get_model_info(canary.name)
        .unwrap_or_else(|| panic!("'{}' must be in the hub registry", canary.name));

    // Step 1: download + SHA verify. The placeholder digest "0"*64 short-
    // circuits to `Err(InvalidArgument)` (#739 audit). A real digest means
    // the bytes the HF mirror serves hash to exactly what the registry
    // pins — that property *is* the security contract this test guards.
    let dir = tempfile::tempdir().expect("tempdir for download cache");
    let cache = HubCache::new(dir.path());
    let path = download_weights(info, &cache).unwrap_or_else(|e| {
        panic!(
            "{}: download + SHA verify must succeed (got {e}). Either the \
             registry pin is stale or the upstream HF repo was rewritten — \
             rerun scripts/pin_pretrained_weights.py to regenerate.",
            canary.name
        )
    });
    let bytes = std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("{}: stat downloaded file: {e}", canary.name))
        .len();
    assert!(
        bytes > 10_000_000,
        "{}: downloaded file is suspiciously small ({bytes} bytes); the \
         smallest pinned model is ~130 MB so anything under 10 MB is wrong.",
        canary.name
    );

    // Step 2: SafeTensors → StateDict<f32>.
    let state_dict = load_pretrained::<f32>(canary.name).unwrap_or_else(|e| {
        panic!(
            "{}: load_pretrained::<f32> must succeed (got {e}). The \
             safetensors file parsed OK at upload time, so a failure here \
             likely means the file is corrupted in the HF cache — clear \
             ~/.cache/ferrotorch and rerun.",
            canary.name
        )
    });
    let n_keys = state_dict.len();
    assert_eq!(
        n_keys, canary.expected_keys_in_safetensors,
        "{}: expected {} keys in safetensors, got {}. \
         If you've intentionally changed the converter mapping, update \
         expected_keys_in_safetensors in this test.",
        canary.name, canary.expected_keys_in_safetensors, n_keys
    );

    // Step 3: every canary key must be present with the right shape + nonzero data.
    for (key, expected_shape) in canary.canaries {
        assert_key_shape(&state_dict, key, expected_shape);
    }
}

#[test]
#[ignore = "requires network; gated for offline CI"]
fn test_pretrained_ssd300_vgg16() {
    run_smoke(&SSD300_CANARIES);
}

#[test]
#[ignore = "requires network; gated for offline CI"]
fn test_pretrained_fasterrcnn_resnet50_fpn() {
    run_smoke(&FASTERRCNN_CANARIES);
}

#[test]
#[ignore = "requires network; gated for offline CI"]
fn test_pretrained_maskrcnn_resnet50_fpn() {
    run_smoke(&MASKRCNN_CANARIES);
}

#[test]
#[ignore = "requires network; gated for offline CI"]
fn test_pretrained_deeplabv3_resnet50() {
    run_smoke(&DEEPLABV3_CANARIES);
}

#[test]
#[ignore = "requires network; gated for offline CI"]
fn test_pretrained_fcn_resnet50() {
    run_smoke(&FCN_CANARIES);
}
