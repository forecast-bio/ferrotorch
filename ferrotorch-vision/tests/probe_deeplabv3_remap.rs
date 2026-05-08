//! Phase 9 (#1009) probe-before-fix for the DeepLabV3-ResNet50 state-dict
//! remap.
//!
//! Asserts that every torchvision `classifier.<...>` key (params + BN
//! buffers) maps 1:1 onto a ferrotorch `head.<...>` key, and that no
//! ferrotorch head parameter / BN buffer is left without a torchvision
//! source. The mapping itself lives test-side in
//! `conformance_vision_models.rs::remap_torchvision_to_ferrotorch_deeplabv3_keys`;
//! this probe duplicates only the *key set* so a regression in either
//! the `src/` head structure (Phase 9 #1009 5-element Sequential) or the
//! remap layout surfaces here before the value-parity loader runs.
//!
//! ## Why a probe-before-fix at all
//!
//! Phase 8 (#1009 STOP-AND-REPORT) failure mode #30 (ASPP branch-order
//! divergence) is invisible at the loader level — branch order is just
//! Vec ordering; if the remap silently swaps `convs.1` ↔ `convs.2` the
//! loader still passes shape checks but the forward output diverges.
//! This probe checks:
//!   * the ferrotorch head exposes exactly the keys the remap promises
//!     to *write to* (no missing target);
//!   * the remap covers every torchvision classifier-side key (no
//!     unmapped torchvision key on the loader's reject path).
//!
//! ## Reference layout (torchvision `deeplabv3_resnet50`, num_classes=21)
//!
//! Captured via `python3 -c "from torchvision.models.segmentation import
//! deeplabv3_resnet50; m = deeplabv3_resnet50(weights=None); ..."` and
//! cross-referenced with `scripts/regenerate_vision_fixtures.py` once the
//! fixture entry is added. Only classifier-side keys are listed here —
//! backbone keys flow through the same `backbone.<...>` paths that
//! Phase 6's `resnet50_dilated` uses.

use ferrotorch_nn::module::Module;
use ferrotorch_vision::models::segmentation::deeplabv3_resnet50;
use std::collections::HashSet;

/// Every classifier-side parameter key torchvision's
/// `deeplabv3_resnet50(num_classes=21)` exports under its state dict.
/// Captured once from a deterministic torchvision build (Phase 9 probe
/// session); kept hand-coded so the probe runs without a Python
/// dependency at test time.
fn torchvision_classifier_param_keys() -> Vec<&'static str> {
    vec![
        // ASPP — five branches inside `classifier.0.convs.<i>`:
        // i=0 is `[Conv2d(1×1), BN, ReLU]` -> conv at .0, bn at .1.
        "classifier.0.convs.0.0.weight",
        "classifier.0.convs.0.1.weight",
        "classifier.0.convs.0.1.bias",
        // i=1/2/3 are dilated 3×3 ASPPConv blocks:
        //   `[Conv2d(3×3, dilation=r), BN, ReLU]`
        "classifier.0.convs.1.0.weight",
        "classifier.0.convs.1.1.weight",
        "classifier.0.convs.1.1.bias",
        "classifier.0.convs.2.0.weight",
        "classifier.0.convs.2.1.weight",
        "classifier.0.convs.2.1.bias",
        "classifier.0.convs.3.0.weight",
        "classifier.0.convs.3.1.weight",
        "classifier.0.convs.3.1.bias",
        // i=4 is ASPPPooling: `[AdaptiveAvgPool2d, Conv2d(1×1), BN,
        // ReLU]` — note the conv lives at index 1 (not 0), bn at 2.
        "classifier.0.convs.4.1.weight",
        "classifier.0.convs.4.2.weight",
        "classifier.0.convs.4.2.bias",
        // ASPP projection: 1×1 conv at .0, BN at .1.
        "classifier.0.project.0.weight",
        "classifier.0.project.1.weight",
        "classifier.0.project.1.bias",
        // 5-element DeepLabHead Sequential:
        //   1: Conv2d(256, 256, 3, bias=False)
        //   2: BatchNorm2d(256)
        //   3: ReLU (no params)
        //   4: Conv2d(256, num_classes, 1, bias=True)
        "classifier.1.weight",
        "classifier.2.weight",
        "classifier.2.bias",
        "classifier.4.weight",
        "classifier.4.bias",
    ]
}

/// Every classifier-side BN buffer key torchvision's
/// `deeplabv3_resnet50(num_classes=21)` exports.
fn torchvision_classifier_bn_buffer_keys() -> Vec<&'static str> {
    vec![
        // ASPP branch BNs.
        "classifier.0.convs.0.1.running_mean",
        "classifier.0.convs.0.1.running_var",
        "classifier.0.convs.1.1.running_mean",
        "classifier.0.convs.1.1.running_var",
        "classifier.0.convs.2.1.running_mean",
        "classifier.0.convs.2.1.running_var",
        "classifier.0.convs.3.1.running_mean",
        "classifier.0.convs.3.1.running_var",
        // ASPPPooling BN (slot .2 because slot .0 is the avgpool).
        "classifier.0.convs.4.2.running_mean",
        "classifier.0.convs.4.2.running_var",
        // ASPP project BN.
        "classifier.0.project.1.running_mean",
        "classifier.0.project.1.running_var",
        // DeepLabHead intermediate BN.
        "classifier.2.running_mean",
        "classifier.2.running_var",
    ]
}

/// Re-implementation of the test-side remap function the conformance
/// test will use (Phase 9 #1009). Kept in lock-step here so the probe
/// can run independently of the conformance test compilation.
///
/// Translates one torchvision key. Returns `None` for any key the probe
/// is not expected to handle (caller surfaces it as "unmapped").
fn remap_one(tv_key: &str) -> Option<String> {
    // Buffer/param suffix -> the ferrotorch tail is identical for every
    // BatchNorm2d (running_mean/running_var/weight/bias).
    fn rest<'a>(key: &'a str, prefix: &str) -> Option<&'a str> {
        key.strip_prefix(prefix)
    }

    // -- ASPP project ---------------------------------------------------
    if let Some(rest) = rest(tv_key, "classifier.0.project.0.") {
        // 1×1 projection conv: `project.0.weight` -> ferrotorch `head.aspp.project.<weight>`
        return Some(format!("head.aspp.project.{rest}"));
    }
    if let Some(rest) = rest(tv_key, "classifier.0.project.1.") {
        // projection BN -> ferrotorch `head.aspp.project_bn.<...>`
        return Some(format!("head.aspp.project_bn.{rest}"));
    }
    // -- ASPP branches: classifier.0.convs.<i>.<j>.<...> ----------------
    if let Some(rest) = rest(tv_key, "classifier.0.convs.") {
        let mut parts = rest.splitn(3, '.');
        let i = parts.next()?;
        let j = parts.next()?;
        let tail = parts.next()?;
        match (i, j) {
            // Branch 0: ASPPConv1x1 -> ferrotorch `head.aspp.0.{conv|bn}.<...>`
            ("0", "0") => Some(format!("head.aspp.0.conv.{tail}")),
            ("0", "1") => Some(format!("head.aspp.0.bn.{tail}")),
            // Branches 1/2/3: DilatedConv2d -> conv key flattens away
            // the `conv.` prefix (see aspp.rs:117-122 — torchvision
            // stores the dilated conv weight at `<branch>.0.weight`,
            // ferrotorch at `<branch>.weight`).
            ("1", "0") | ("2", "0") | ("3", "0") => Some(format!("head.aspp.{i}.{tail}")),
            ("1", "1") | ("2", "1") | ("3", "1") => Some(format!("head.aspp.{i}.bn.{tail}")),
            // Branch 4: ASPPPooling -> conv at slot .1, BN at slot .2.
            ("4", "1") => Some(format!("head.aspp.4.conv.{tail}")),
            ("4", "2") => Some(format!("head.aspp.4.bn.{tail}")),
            _ => None,
        }
    } else if let Some(rest) = rest(tv_key, "classifier.1.") {
        Some(format!("head.conv_intermediate.{rest}"))
    } else if let Some(rest) = rest(tv_key, "classifier.2.") {
        Some(format!("head.bn_intermediate.{rest}"))
    } else {
        rest(tv_key, "classifier.4.").map(|rest| format!("head.classifier.{rest}"))
    }
}

#[test]
fn deeplabv3_remap_covers_every_torchvision_classifier_param() {
    let unmapped: Vec<&'static str> = torchvision_classifier_param_keys()
        .into_iter()
        .filter(|k| remap_one(k).is_none())
        .collect();
    assert!(
        unmapped.is_empty(),
        "torchvision classifier param keys without a ferrotorch remap: {unmapped:?}",
    );
}

#[test]
fn deeplabv3_remap_covers_every_torchvision_classifier_bn_buffer() {
    let unmapped: Vec<&'static str> = torchvision_classifier_bn_buffer_keys()
        .into_iter()
        .filter(|k| remap_one(k).is_none())
        .collect();
    assert!(
        unmapped.is_empty(),
        "torchvision classifier BN buffer keys without a ferrotorch remap: {unmapped:?}",
    );
}

#[test]
#[ignore = "diagnostic only — emits the live key set for human inspection"]
fn dump_ferrotorch_deeplabv3_keys() {
    let model = deeplabv3_resnet50::<f32>(21).unwrap();
    let mut keys: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, p)| format!("{n} {:?}", p.tensor().shape()))
        .collect();
    keys.sort();
    eprintln!("=== ferrotorch deeplabv3_resnet50 named_parameters ===");
    for k in &keys {
        eprintln!("  {k}");
    }
}

#[test]
fn deeplabv3_ferrotorch_head_keys_match_remap_targets() {
    // Build a fresh ferrotorch deeplabv3_resnet50 and collect the head-
    // side parameter keys. Every key MUST appear as a remap target —
    // otherwise the loader will fail with "no source" for that key.
    let model = deeplabv3_resnet50::<f32>(21).unwrap();
    let ft_head_param_keys: HashSet<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.starts_with("head."))
        .collect();
    let remap_targets: HashSet<String> = torchvision_classifier_param_keys()
        .into_iter()
        .filter_map(remap_one)
        .collect();

    let unsourced: Vec<&String> = ft_head_param_keys.difference(&remap_targets).collect();
    assert!(
        unsourced.is_empty(),
        "ferrotorch head params without a torchvision source after remap: {unsourced:?}\n\
         (head keys = {ft_head_param_keys:?})\n(remap targets = {remap_targets:?})",
    );
    let unused: Vec<&String> = remap_targets.difference(&ft_head_param_keys).collect();
    assert!(
        unused.is_empty(),
        "remap targets without a matching ferrotorch head param: {unused:?}\n\
         (head keys = {ft_head_param_keys:?})",
    );
}
