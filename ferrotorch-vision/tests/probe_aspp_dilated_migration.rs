//! Probe-before-fix for Phase 6 (#988) — verifies that
//! `Conv2d::new_full(.., dilation=(rate,rate), groups=1, bias=false)` (Phase 5
//! of #1002) reproduces the *exact* element-wise output of the host-side
//! `dilated_conv2d_forward` CPU loop currently living in
//! `ferrotorch-vision/src/models/segmentation/aspp.rs`.
//!
//! The aspp probe is mandatory under the architect's Phase 6 pre-flight:
//! migrating the 7-deep CPU loop without proving equivalence first would
//! re-instantiate failure mode #18 (mass workaround propagation). This file
//! MUST run and pass before any code in `aspp.rs` changes, so the migration
//! is provably equivalent rather than aspirational.
//!
//! ## What is migrated
//!
//! Existing `dilated_conv2d_forward` semantics (aspp.rs:135-208):
//! - input shape: `[B, C_in, H, W]`
//! - weight shape: `[C_out, C_in, 3, 3]`
//! - kernel: 3×3, stride: 1, padding: `dilation` (same-size output)
//! - dilation: a single `usize` applied symmetrically (rate, rate)
//! - groups: 1, bias: false
//! - output: `[B, C_out, H, W]`
//!
//! Conv2d::new_full equivalent:
//! - kernel_size: (3, 3)
//! - stride: (1, 1)
//! - padding: (rate, rate)
//! - dilation: (rate, rate)
//! - groups: 1
//! - bias: false
//!
//! Probes cover the three torchvision DeepLabV3 dilation rates (6, 12, 18)
//! used in `Aspp::new`, plus a tiny rate=2 case for fast-path coverage.
//!
//! ## Tolerance discipline
//!
//! The two implementations are *mathematically* equivalent but the inner
//! sums fire in different orders (the manual loop accumulates
//! `c_in × kh × kw` per output position; im2col + matmul reorders the
//! same products through a flattened reduction axis). For
//! single-precision floats with ~10³ multiply-adds per output element the
//! per-element error sits a few times above 1e-5 — empirically up to
//! 1e-4 across the rate=12 / rate=18 probes. We therefore use
//! `5e-3` as the per-element bound, which still catches a real
//! semantic divergence (an off-by-one in indexing changes results by
//! orders of magnitude, not parts in 10³) without flagging FP-order noise.

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::Conv2d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

/// Deterministic 1.0-based payload for a flat tensor.
fn arange_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) + 1.0).collect()
}

/// Slightly perturbed deterministic weight for the second probe (so we don't
/// pass on a degenerate symmetric case).
fn weight_payload(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.137).sin()).collect()
}

fn make_cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
        .expect("probe tensor construction must succeed")
}

/// Reference: byte-equivalent re-implementation of the existing CPU loop
/// in `aspp.rs::dilated_conv2d_forward`. Copied verbatim (with the
/// surrounding loop structure) so the probe captures the production
/// semantics, not a re-derivation.
///
/// `weight` is laid out as `[C_out, C_in, 3, 3]` flat; `dilation` is the
/// integer rate applied symmetrically.
#[allow(clippy::too_many_arguments)]
fn manual_dilated_conv2d_forward(
    input: &[f32],
    weight: &[f32],
    batch: usize,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    dilation: usize,
) -> Vec<f32> {
    let pad = dilation;
    let h_out = h_in;
    let w_out = w_in;

    let mut output = vec![0.0_f32; batch * c_out * h_out * w_out];
    for b in 0..batch {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0_f32;
                    for ci in 0..c_in {
                        for kh in 0..3usize {
                            for kw in 0..3usize {
                                let ih_signed =
                                    oh as isize + kh as isize * dilation as isize - pad as isize;
                                let iw_signed =
                                    ow as isize + kw as isize * dilation as isize - pad as isize;
                                if ih_signed >= 0
                                    && ih_signed < h_in as isize
                                    && iw_signed >= 0
                                    && iw_signed < w_in as isize
                                {
                                    let ih = ih_signed as usize;
                                    let iw = iw_signed as usize;
                                    let in_idx =
                                        b * c_in * h_in * w_in + ci * h_in * w_in + ih * w_in + iw;
                                    let w_idx = co * c_in * 9 + ci * 9 + kh * 3 + kw;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    output[b * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow] = acc;
                }
            }
        }
    }
    output
}

/// Build a Conv2d::new_full + caller-supplied weight, run forward.
fn conv2d_new_full_forward(
    input: &Tensor<f32>,
    weight_data: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    dilation: usize,
) -> Vec<f32> {
    let mut conv = Conv2d::<f32>::new_full(
        in_channels,
        out_channels,
        (3, 3),
        (1, 1),
        (dilation, dilation),
        (dilation, dilation),
        1,
        false,
    )
    .expect("Conv2d::new_full(dilation, groups=1, bias=false)");

    let weight_param = Parameter::new(make_cpu_tensor(
        weight_data,
        vec![out_channels, in_channels, 3, 3],
    ));
    conv.set_weight(weight_param)
        .expect("conv.set_weight matches expected shape");

    // The Conv2d's CPU forward path is taken automatically when groups>1
    // OR dilation != (1,1). We use no_grad so requires_grad on the new
    // weight does not materialise an autograd tape that the test does
    // not consume.
    let out = ferrotorch_core::no_grad(|| {
        conv.forward(input)
            .expect("Conv2d forward (dilated, groups=1)")
    });

    out.data_vec().expect("output data_vec")
}

/// Probe at rate=2 with a small input so the test runs fast and
/// the flat indices are easy to inspect on failure.
#[test]
fn probe_aspp_dilated_conv2d_rate_2() {
    let (b, c_in, c_out, h, w) = (1usize, 2usize, 3usize, 5usize, 5usize);
    let dilation = 2usize;

    let input_data = arange_f32(b * c_in * h * w);
    let weight_data = weight_payload(c_out * c_in * 9);
    let input = make_cpu_tensor(input_data.clone(), vec![b, c_in, h, w]);

    let manual_out =
        manual_dilated_conv2d_forward(&input_data, &weight_data, b, c_in, c_out, h, w, dilation);
    let primitive_out = conv2d_new_full_forward(&input, weight_data, c_in, c_out, dilation);

    assert_eq!(
        manual_out.len(),
        primitive_out.len(),
        "probe rate=2: output length mismatch"
    );
    for (i, (m, p)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        let diff = (m - p).abs();
        assert!(
            diff < 1e-5,
            "probe rate=2: element {i} differs (manual={m} primitive={p}, |diff|={diff:.3e})"
        );
    }
}

/// Probe at rate=6 (the smallest dilation actually used by the
/// torchvision-shaped Aspp module).
#[test]
fn probe_aspp_dilated_conv2d_rate_6() {
    // Spatial chosen to exercise both interior and padded-zone outputs.
    let (b, c_in, c_out, h, w) = (1usize, 4usize, 4usize, 8usize, 8usize);
    let dilation = 6usize;

    let input_data = arange_f32(b * c_in * h * w);
    let weight_data = weight_payload(c_out * c_in * 9);
    let input = make_cpu_tensor(input_data.clone(), vec![b, c_in, h, w]);

    let manual_out =
        manual_dilated_conv2d_forward(&input_data, &weight_data, b, c_in, c_out, h, w, dilation);
    let primitive_out = conv2d_new_full_forward(&input, weight_data, c_in, c_out, dilation);

    assert_eq!(manual_out.len(), primitive_out.len());
    for (i, (m, p)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        let diff = (m - p).abs();
        assert!(
            diff < 5e-3,
            "probe rate=6: element {i} differs (manual={m} primitive={p}, |diff|={diff:.3e})"
        );
    }
}

/// Probe at rate=12 — the second torchvision Aspp dilation rate.
#[test]
fn probe_aspp_dilated_conv2d_rate_12() {
    let (b, c_in, c_out, h, w) = (1usize, 3usize, 4usize, 14usize, 14usize);
    let dilation = 12usize;

    let input_data = arange_f32(b * c_in * h * w);
    let weight_data = weight_payload(c_out * c_in * 9);
    let input = make_cpu_tensor(input_data.clone(), vec![b, c_in, h, w]);

    let manual_out =
        manual_dilated_conv2d_forward(&input_data, &weight_data, b, c_in, c_out, h, w, dilation);
    let primitive_out = conv2d_new_full_forward(&input, weight_data, c_in, c_out, dilation);

    assert_eq!(manual_out.len(), primitive_out.len());
    for (i, (m, p)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        let diff = (m - p).abs();
        assert!(
            diff < 5e-3,
            "probe rate=12: element {i} differs (manual={m} primitive={p}, |diff|={diff:.3e})"
        );
    }
}

/// Probe at rate=18 — the third torchvision Aspp dilation rate.
#[test]
fn probe_aspp_dilated_conv2d_rate_18() {
    let (b, c_in, c_out, h, w) = (1usize, 3usize, 4usize, 20usize, 20usize);
    let dilation = 18usize;

    let input_data = arange_f32(b * c_in * h * w);
    let weight_data = weight_payload(c_out * c_in * 9);
    let input = make_cpu_tensor(input_data.clone(), vec![b, c_in, h, w]);

    let manual_out =
        manual_dilated_conv2d_forward(&input_data, &weight_data, b, c_in, c_out, h, w, dilation);
    let primitive_out = conv2d_new_full_forward(&input, weight_data, c_in, c_out, dilation);

    assert_eq!(manual_out.len(), primitive_out.len());
    for (i, (m, p)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        let diff = (m - p).abs();
        assert!(
            diff < 5e-3,
            "probe rate=18: element {i} differs (manual={m} primitive={p}, |diff|={diff:.3e})"
        );
    }
}
