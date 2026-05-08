//! Phase 5: Conv2d::new_full forward + backward + validation tests
//! covering groups + dilation (Issue #1002).
//!
//! All references in this file are HAND-COMPUTED or computed from a NAIVE
//! direct-loop implementation that is *independent* of the production CPU
//! path under test (avoids failure mode #12 — Tautological reference).
//! Numerical tolerances are tight (1e-3) because both the reference and
//! the implementation use direct f32 sums.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Conv2d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

/// Naive grouped + dilated convolution reference.
///
/// Iterates explicitly over the index space `(b, g, oc_in_g, oh, ow, ic_in_g, kh, kw)`
/// and accumulates `input * weight` into the output, with the kernel taps
/// spaced by `dilation`. This loop deliberately does NOT use `im2col`,
/// `mm`, or any helper from the implementation — it is the cleanroom
/// reference that the production `Conv2d::forward` is checked against.
#[allow(clippy::too_many_arguments)]
fn naive_grouped_dilated_conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) -> (Vec<f32>, usize, usize) {
    let in_per_g = in_channels / groups;
    let out_per_g = out_channels / groups;
    let eff_kh = dh * (kh - 1) + 1;
    let eff_kw = dw * (kw - 1) + 1;
    let h_out = (h + 2 * ph - eff_kh) / sh + 1;
    let w_out = (w + 2 * pw - eff_kw) / sw + 1;
    let mut out = vec![0.0f32; batch * out_channels * h_out * w_out];

    for b in 0..batch {
        for g in 0..groups {
            for ocg in 0..out_per_g {
                let oc = g * out_per_g + ocg;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0f32;
                        for icg in 0..in_per_g {
                            let ic = g * in_per_g + icg;
                            for kkh in 0..kh {
                                for kkw in 0..kw {
                                    // Padded input coordinates.
                                    let ih_p = oh * sh + kkh * dh;
                                    let iw_p = ow * sw + kkw * dw;
                                    if ih_p < ph || iw_p < pw {
                                        continue;
                                    }
                                    let ih = ih_p - ph;
                                    let iw = iw_p - pw;
                                    if ih >= h || iw >= w {
                                        continue;
                                    }
                                    let in_v =
                                        input[b * in_channels * h * w + ic * h * w + ih * w + iw];
                                    // Weight layout: [out_channels, in_per_group, kh, kw].
                                    let w_v = weight
                                        [oc * in_per_g * kh * kw + icg * kh * kw + kkh * kw + kkw];
                                    acc += in_v * w_v;
                                }
                            }
                        }
                        if let Some(bv) = bias {
                            acc += bv[oc];
                        }
                        out[b * out_channels * h_out * w_out
                            + oc * h_out * w_out
                            + oh * w_out
                            + ow] = acc;
                    }
                }
            }
        }
    }
    (out, h_out, w_out)
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{ctx}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        let rel = if y.abs() > 1e-6 { d / y.abs() } else { d };
        assert!(
            d < tol || rel < tol,
            "{ctx}: index {i} got={x} want={y} diff={d} rel={rel}"
        );
    }
}

// =====================================================================
// Forward correctness: groups=1, dilation=(1,1) — bit-equal regression
// =====================================================================

#[test]
fn forward_dense_unchanged_vs_old_signature() {
    // new_full(...,(1,1),1,bias) must produce identical output to new(...).
    let in_c = 3;
    let out_c = 4;
    let batch = 2;
    let h = 5;
    let w = 5;
    let kh = 3;
    let kw = 3;

    // Deterministic input and weight to avoid Kaiming randomness divergence.
    let input: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|i| (i as f32) * 0.05)
        .collect();

    let mut conv_a = Conv2d::<f32>::new(in_c, out_c, (kh, kw), (1, 1), (1, 1), false).unwrap();
    let mut conv_b =
        Conv2d::<f32>::new_full(in_c, out_c, (kh, kw), (1, 1), (1, 1), (1, 1), 1, false).unwrap();
    let wp_a = Parameter::from_slice(&weight, &[out_c, in_c, kh, kw]).unwrap();
    let wp_b = Parameter::from_slice(&weight, &[out_c, in_c, kh, kw]).unwrap();
    conv_a.set_weight(wp_a).unwrap();
    conv_b.set_weight(wp_b).unwrap();

    let x = t(&input, &[batch, in_c, h, w]);
    let y_a = conv_a.forward(&x).unwrap();
    let y_b = conv_b.forward(&x).unwrap();
    assert_eq!(y_a.shape(), y_b.shape());
    assert_close(
        y_a.data().unwrap(),
        y_b.data().unwrap(),
        1e-6,
        "dense new() vs new_full() bit-equality",
    );
}

// =====================================================================
// Forward correctness: groups=2, dilation=(1,1)
// =====================================================================

#[test]
fn forward_groups2_no_dilation_matches_naive() {
    let in_c = 4;
    let out_c = 6;
    let groups = 2;
    let kh = 3;
    let kw = 3;
    let h = 6;
    let w = 6;
    let batch = 1;

    let input: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| (i as f32) * 0.07 - 2.0)
        .collect();
    // Weight shape with groups: [out_c, in_c/groups, kh, kw].
    let in_per_g = in_c / groups;
    let weight: Vec<f32> = (0..out_c * in_per_g * kh * kw)
        .map(|i| (i as f32) * 0.03 - 0.5)
        .collect();

    let mut conv =
        Conv2d::<f32>::new_full(in_c, out_c, (kh, kw), (1, 1), (0, 0), (1, 1), groups, false)
            .unwrap();
    let wp = Parameter::from_slice(&weight, &[out_c, in_per_g, kh, kw]).unwrap();
    conv.set_weight(wp).unwrap();

    let x = t(&input, &[batch, in_c, h, w]);
    let y = conv.forward(&x).unwrap();

    let (expected, h_out, w_out) = naive_grouped_dilated_conv2d(
        &input, &weight, None, batch, in_c, out_c, h, w, kh, kw, 1, 1, 0, 0, 1, 1, groups,
    );
    assert_eq!(y.shape(), &[batch, out_c, h_out, w_out]);
    assert_close(y.data().unwrap(), &expected, 1e-3, "groups=2 dil=(1,1)");
}

// =====================================================================
// Forward correctness: depthwise (groups == in_channels)
// =====================================================================

#[test]
fn forward_depthwise_matches_naive() {
    let in_c = 5;
    let groups = in_c;
    let out_c = in_c; // 1 filter per group
    let kh = 3;
    let kw = 3;
    let h = 7;
    let w = 7;
    let batch = 1;

    let input: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();
    // Weight shape [out_c, 1, kh, kw] when depthwise.
    let weight: Vec<f32> = (0..out_c * kh * kw)
        .map(|i| ((i as f32) * 0.05).cos())
        .collect();

    let mut conv =
        Conv2d::<f32>::new_full(in_c, out_c, (kh, kw), (1, 1), (1, 1), (1, 1), groups, false)
            .unwrap();
    let wp = Parameter::from_slice(&weight, &[out_c, 1, kh, kw]).unwrap();
    conv.set_weight(wp).unwrap();

    let x = t(&input, &[batch, in_c, h, w]);
    let y = conv.forward(&x).unwrap();

    let (expected, h_out, w_out) = naive_grouped_dilated_conv2d(
        &input, &weight, None, batch, in_c, out_c, h, w, kh, kw, 1, 1, 1, 1, 1, 1, groups,
    );
    assert_eq!(y.shape(), &[batch, out_c, h_out, w_out]);
    assert_close(y.data().unwrap(), &expected, 1e-3, "depthwise");
}

// =====================================================================
// Forward correctness: groups=1, dilation=(2,2)
// =====================================================================

#[test]
fn forward_dilation2_matches_naive_and_spatial_size() {
    let in_c = 2;
    let out_c = 3;
    let kh = 3;
    let kw = 3;
    let h = 7;
    let w = 7;
    let batch = 1;
    let dh = 2;
    let dw = 2;

    let input: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|i| (i as f32) * 0.05)
        .collect();

    // With kh=3 dh=2 -> eff_kh = 5. h_out = (7 + 0 - 5)/1 + 1 = 3.
    let mut conv =
        Conv2d::<f32>::new_full(in_c, out_c, (kh, kw), (1, 1), (0, 0), (dh, dw), 1, false).unwrap();
    let wp = Parameter::from_slice(&weight, &[out_c, in_c, kh, kw]).unwrap();
    conv.set_weight(wp).unwrap();

    let x = t(&input, &[batch, in_c, h, w]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(
        y.shape(),
        &[batch, out_c, 3, 3],
        "dilation=2 must yield H_out=W_out=3 for H=W=7,k=3"
    );

    let (expected, _, _) = naive_grouped_dilated_conv2d(
        &input, &weight, None, batch, in_c, out_c, h, w, kh, kw, 1, 1, 0, 0, dh, dw, 1,
    );
    assert_close(y.data().unwrap(), &expected, 1e-3, "dilation=2");
}

// =====================================================================
// Forward correctness: groups=2, dilation=(2,2) (combined)
// =====================================================================

#[test]
fn forward_groups2_dilation2_matches_naive() {
    let in_c = 4;
    let out_c = 4;
    let groups = 2;
    let in_per_g = in_c / groups;
    let kh = 2;
    let kw = 2;
    let h = 5;
    let w = 5;
    let batch = 1;
    let dh = 2;
    let dw = 2;

    let input: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| (i as f32) * 0.02 + 0.1)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_per_g * kh * kw)
        .map(|i| (i as f32) * 0.07)
        .collect();

    let mut conv = Conv2d::<f32>::new_full(
        in_c,
        out_c,
        (kh, kw),
        (1, 1),
        (0, 0),
        (dh, dw),
        groups,
        false,
    )
    .unwrap();
    let wp = Parameter::from_slice(&weight, &[out_c, in_per_g, kh, kw]).unwrap();
    conv.set_weight(wp).unwrap();

    let x = t(&input, &[batch, in_c, h, w]);
    let y = conv.forward(&x).unwrap();

    let (expected, h_out, w_out) = naive_grouped_dilated_conv2d(
        &input, &weight, None, batch, in_c, out_c, h, w, kh, kw, 1, 1, 0, 0, dh, dw, groups,
    );
    assert_eq!(y.shape(), &[batch, out_c, h_out, w_out]);
    assert_close(y.data().unwrap(), &expected, 1e-3, "groups=2 dil=2");
}

// =====================================================================
// Validation: rejects bad parameters
// =====================================================================

#[test]
fn validation_groups_must_divide_in_channels() {
    // groups=3 with in_channels=4 -> 4 % 3 != 0
    let r = Conv2d::<f32>::new_full(4, 6, (3, 3), (1, 1), (0, 0), (1, 1), 3, false);
    let err = r.expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("must divide in_channels=4"),
        "error must name the offending field; got: {msg}"
    );
}

#[test]
fn validation_groups_must_divide_out_channels() {
    // in_channels=4, groups=2 ok; out_channels=5 not divisible.
    let r = Conv2d::<f32>::new_full(4, 5, (3, 3), (1, 1), (0, 0), (1, 1), 2, false);
    let err = r.expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("must divide out_channels=5"),
        "error must name the offending field; got: {msg}"
    );
}

#[test]
fn validation_dilation_zero_rejected() {
    let r = Conv2d::<f32>::new_full(2, 2, (3, 3), (1, 1), (0, 0), (0, 1), 1, false);
    let err = r.expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("dilation"),
        "error must mention dilation; got: {msg}"
    );
}

#[test]
fn validation_groups_zero_rejected() {
    let r = Conv2d::<f32>::new_full(2, 2, (3, 3), (1, 1), (0, 0), (1, 1), 0, false);
    let err = r.expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("groups"),
        "error must mention groups; got: {msg}"
    );
}

// =====================================================================
// Backward correctness: finite differences
// =====================================================================

/// Build a fresh Conv2d with the given config and weights, weight requires_grad=true.
#[allow(clippy::too_many_arguments)]
fn build_conv(
    in_c: usize,
    out_c: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    weight: &[f32],
    weight_shape: &[usize],
) -> Conv2d<f32> {
    let mut conv = Conv2d::<f32>::new_full(
        in_c, out_c, kernel, stride, padding, dilation, groups, false,
    )
    .unwrap();
    let wp = Parameter::from_slice(weight, weight_shape).unwrap();
    conv.set_weight(wp).unwrap();
    conv
}

/// Run forward with given input data, return flat output.
fn forward_flat(conv: &Conv2d<f32>, input_data: &[f32], input_shape: &[usize]) -> Vec<f32> {
    let x = t(input_data, input_shape);
    conv.forward(&x).unwrap().data().unwrap().to_vec()
}

/// Run forward (with input as leaf, requires_grad=true), backprop a one-hot
/// grad_output on `out_idx`, return (grad_input, grad_weight) flat.
#[allow(clippy::too_many_arguments)]
fn analytic_grad_one_hot(
    in_c: usize,
    out_c: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    input_data: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    out_idx: usize,
) -> (Vec<f32>, Vec<f32>) {
    // Build with weight requires_grad=true via Parameter::from_slice (default true).
    let conv = build_conv(
        in_c,
        out_c,
        kernel,
        stride,
        padding,
        dilation,
        groups,
        weight,
        weight_shape,
    );
    let input = leaf(input_data, input_shape);
    let output = conv.forward(&input).unwrap();
    let out_total = output.shape().iter().product::<usize>();
    let mut go = vec![0.0f32; out_total];
    go[out_idx] = 1.0;
    let go_t = t(&go, output.shape());
    let grads = output.grad_fn().unwrap().backward(&go_t).unwrap();
    let grad_input = grads[0].as_ref().unwrap().data().unwrap().to_vec();
    let grad_weight = grads[1].as_ref().unwrap().data().unwrap().to_vec();
    (grad_input, grad_weight)
}

/// Central-difference numerical grad for output element `out_idx` w.r.t.
/// every input and every weight scalar.
#[allow(clippy::too_many_arguments)]
fn numerical_grad_one_hot(
    in_c: usize,
    out_c: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    input_data: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    eps: f32,
    out_idx: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_in = input_data.len();
    let n_w = weight.len();
    let mut g_in = vec![0.0f32; n_in];
    let mut g_w = vec![0.0f32; n_w];
    let conv_base = build_conv(
        in_c,
        out_c,
        kernel,
        stride,
        padding,
        dilation,
        groups,
        weight,
        weight_shape,
    );

    for i in 0..n_in {
        let mut plus = input_data.to_vec();
        plus[i] += eps;
        let mut minus = input_data.to_vec();
        minus[i] -= eps;
        let yp = forward_flat(&conv_base, &plus, input_shape);
        let ym = forward_flat(&conv_base, &minus, input_shape);
        g_in[i] = (yp[out_idx] - ym[out_idx]) / (2.0 * eps);
    }

    for i in 0..n_w {
        let mut plus = weight.to_vec();
        plus[i] += eps;
        let mut minus = weight.to_vec();
        minus[i] -= eps;
        let conv_p = build_conv(
            in_c,
            out_c,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            &plus,
            weight_shape,
        );
        let conv_m = build_conv(
            in_c,
            out_c,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            &minus,
            weight_shape,
        );
        let yp = forward_flat(&conv_p, input_data, input_shape);
        let ym = forward_flat(&conv_m, input_data, input_shape);
        g_w[i] = (yp[out_idx] - ym[out_idx]) / (2.0 * eps);
    }
    (g_in, g_w)
}

#[test]
fn backward_groups2_finite_difference() {
    let in_c = 4;
    let out_c = 4;
    let groups = 2;
    let in_per_g = in_c / groups;
    let kh = 2;
    let kw = 2;
    let h = 4;
    let w = 4;
    let batch = 1;

    let input: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| (i as f32) * 0.05 - 0.7)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_per_g * kh * kw)
        .map(|i| (i as f32) * 0.03 - 0.2)
        .collect();

    // Output shape: H_out = (4-2)/1 + 1 = 3 -> [1,4,3,3], 36 elements.
    // Pick element in group 1 (oc=2, oh=1, ow=1) -> idx = 2*9 + 1*3 + 1 = 22.
    let out_idx = 22;
    let (g_in, g_w) = analytic_grad_one_hot(
        in_c,
        out_c,
        (kh, kw),
        (1, 1),
        (0, 0),
        (1, 1),
        groups,
        &input,
        &[batch, in_c, h, w],
        &weight,
        &[out_c, in_per_g, kh, kw],
        out_idx,
    );

    let (g_in_n, g_w_n) = numerical_grad_one_hot(
        in_c,
        out_c,
        (kh, kw),
        (1, 1),
        (0, 0),
        (1, 1),
        groups,
        &input,
        &[batch, in_c, h, w],
        &weight,
        &[out_c, in_per_g, kh, kw],
        1e-2,
        out_idx,
    );

    assert_close(&g_in, &g_in_n, 1e-2, "grad_input grouped vs FD");
    assert_close(&g_w, &g_w_n, 1e-2, "grad_weight grouped vs FD");
}

#[test]
fn backward_dilation2_finite_difference() {
    let in_c = 2;
    let out_c = 2;
    let kh = 3;
    let kw = 3;
    let h = 6;
    let w = 6;
    let batch = 1;
    let dh = 2;
    let dw = 2;

    let input: Vec<f32> = (0..batch * in_c * h * w)
        .map(|i| (i as f32) * 0.04 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|i| (i as f32) * 0.02 - 0.1)
        .collect();

    // Output shape with k=3 dh=2 padding=0: H_out = (6 - 5)/1 + 1 = 2 -> [1,2,2,2]
    let out_idx = 5;
    let (g_in, g_w) = analytic_grad_one_hot(
        in_c,
        out_c,
        (kh, kw),
        (1, 1),
        (0, 0),
        (dh, dw),
        1,
        &input,
        &[batch, in_c, h, w],
        &weight,
        &[out_c, in_c, kh, kw],
        out_idx,
    );

    let (g_in_n, g_w_n) = numerical_grad_one_hot(
        in_c,
        out_c,
        (kh, kw),
        (1, 1),
        (0, 0),
        (dh, dw),
        1,
        &input,
        &[batch, in_c, h, w],
        &weight,
        &[out_c, in_c, kh, kw],
        1e-2,
        out_idx,
    );

    assert_close(&g_in, &g_in_n, 1e-2, "grad_input dilated vs FD");
    assert_close(&g_w, &g_w_n, 1e-2, "grad_weight dilated vs FD");
}
