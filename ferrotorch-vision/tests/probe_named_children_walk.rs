//! Probe-before-fix for Phase 4 (#995) — proves that the
//! `Module::named_descendants_dyn()` walk on a freshly-constructed
//! [`ResNet`] yields the dot-separated paths the BN-buffer dispatch
//! loop in `tests/conformance_vision_models.rs` expects.
//!
//! This file is the safety net for failure mode #18 (mass workaround
//! propagation): if the override pattern does not actually surface
//! the BN paths the loader will look up, the propagation across all
//! ~17 vision models would silently keep the loader on its Phase 1A
//! fallback path. The probe runs against *real* `resnet50()` from
//! `ferrotorch-vision`, not a hand-rolled stand-in, so a regression
//! anywhere in the override chain (ResNet → Bottleneck → BatchNorm)
//! surfaces immediately.
//!
//! ## Probe-before-fix protocol
//!
//! Before the override lands, `expected_paths` are absent from
//! `named_descendants_dyn()` (the default-`empty` `named_children` of
//! every leaf and every parent leaves the walk empty), and this probe
//! FAILS. After the override lands on every concrete vision module
//! that owns sub-Modules, this probe PASSes. Both runs are recorded
//! in the Phase 4 evidence section so the architect can independently
//! re-execute them.

use ferrotorch_nn::module::Module;
use ferrotorch_vision::models::resnet::resnet50;

/// Sentinel BN paths a real torchvision-shaped `resnet50` exposes.
/// Lifted from the `param_keys` / `buffer_keys` of the
/// `resnet50_value_parity` fixture descriptor, then trimmed to the
/// `<bn-path>` half (drops the `.running_mean` / `.running_var`
/// suffix). Only a representative subset is asserted — full coverage
/// is the loader's contract; the probe just has to detect the
/// "Phase 1A all-skipped" failure mode.
fn expected_resnet50_named_descendant_paths() -> Vec<&'static str> {
    vec![
        // Stem.
        "conv1",
        "bn1",
        "maxpool",
        // First residual stage — all four BNs in the first block plus
        // the projection BN on the downsample branch.
        "layer1.0",
        "layer1.0.conv1",
        "layer1.0.bn1",
        "layer1.0.conv2",
        "layer1.0.bn2",
        "layer1.0.conv3",
        "layer1.0.bn3",
        "layer1.0.downsample.0",
        "layer1.0.downsample.1",
        // Stride-2 transition into layer2.
        "layer2.0.bn1",
        "layer2.0.bn3",
        // Mid-network sanity.
        "layer3.5.bn1",
        "layer3.5.bn3",
        // Last residual stage.
        "layer4.2.bn1",
        "layer4.2.bn3",
        // Head.
        "avgpool",
        "fc",
    ]
}

#[test]
fn probe_resnet50_named_descendants_walk_exposes_bn_paths() {
    let model = resnet50::<f32>(1000).expect("resnet50 construction");

    let walked: Vec<String> = model
        .named_descendants_dyn()
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    // Diagnostic: surface the full walk on failure so a regression
    // shows the exact path set produced (helps the architect's
    // independent-verification step).
    eprintln!(
        "probe_resnet50_named_descendants_walk_exposes_bn_paths: \
         walked {} paths: {:?}",
        walked.len(),
        walked
    );

    assert!(
        !walked.is_empty(),
        "ResNet50 named_descendants_dyn() returned an empty Vec — \
         the named_children override is missing on ResNet, Bottleneck, \
         and/or one of their leaf sub-modules. The Phase 2 BN-buffer \
         loader will fall through to the Phase 1A skip branch for \
         every fixture key (see #995 dispatch summary)."
    );

    let walked_set: std::collections::HashSet<&str> = walked.iter().map(String::as_str).collect();

    let mut missing: Vec<&str> = Vec::new();
    for expected in expected_resnet50_named_descendant_paths() {
        if !walked_set.contains(expected) {
            missing.push(expected);
        }
    }

    assert!(
        missing.is_empty(),
        "ResNet50 named_descendants_dyn() is missing expected paths: \
         {:?} (walked={} total). Each missing path is a BN or stem/head \
         module the Phase 2 loader needs to reach via path → module index.",
        missing,
        walked.len(),
    );
}

/// Companion check: at least one of the BN paths we expect must
/// downcast back to `BatchNorm2d<f32>` via `Module::as_any`. If
/// the walk produced the names but the modules themselves did not
/// opt into the downcast hook, the loader's
/// `BnBufferDispatchOutcome::SkippedNoAsAny` path fires and the
/// running stats are still NOT applied. Probing one canonical
/// path keeps this independent of the full ~50-key sweep.
#[test]
fn probe_resnet50_bn_paths_downcast_to_batchnorm2d() {
    use ferrotorch_nn::norm::BatchNorm2d;

    let model = resnet50::<f32>(1000).expect("resnet50 construction");
    let mut path_to_module: std::collections::HashMap<String, &dyn Module<f32>> =
        std::collections::HashMap::new();
    for (name, child) in model.named_descendants_dyn() {
        path_to_module.insert(name, child);
    }

    let path = "layer1.0.bn1";
    let bn_module = path_to_module
        .get(path)
        .copied()
        .unwrap_or_else(|| panic!("named_descendants_dyn missing canonical BN path {path}"));
    let any = bn_module.as_any().unwrap_or_else(|| {
        panic!(
            "BN module at {path} returned None from Module::as_any — \
             dispatch_bn_buffer will treat it as SkippedNoAsAny and \
             never apply running stats (#995)"
        )
    });
    assert!(
        any.downcast_ref::<BatchNorm2d<f32>>().is_some(),
        "BN module at {path} did not downcast to BatchNorm2d<f32> — \
         Phase 2 invariant violation",
    );
}
