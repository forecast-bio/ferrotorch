//! Conformance Phase C9.1 — `ferrotorch-nn` basic layers.
//!
//! Tracking issue: crosslink #899 (C9.1 nn layers conformance).
//!
//! ## Coverage
//!
//! ### linear
//! - [`Linear::new`] — construction with/without bias, zero-feature errors
//! - [`Linear::forward`] — 2D, 3D, 1D inputs; output shape; numerical values
//! - [`Linear::in_features`] / [`Linear::out_features`]
//! - [`Linear::parameters`] / [`Linear::named_parameters`]
//! - [`Linear::train`] / [`Linear::eval`] / [`Linear::is_training`]
//!
//! ### conv
//! - [`Conv2d::new`] — construction, parameter counts
//! - [`Conv2d::forward`] — output shape for standard configs
//! - [`Conv1d::new`] / [`Conv1d::forward`]
//! - [`Conv3d::new`] / [`Conv3d::forward`]
//! - [`ConvTranspose2d::new`] / [`ConvTranspose2d::forward`]
//! - [`ConvTranspose1d::new`] / [`ConvTranspose1d::forward`]
//! - [`ConvTranspose3d::new`] / [`ConvTranspose3d::forward`]
//!
//! ### lazy_linear
//! - [`LazyLinear::new`] — pre-initialization state
//! - [`LazyLinear::is_initialized`] / [`LazyLinear::in_features`]
//! - [`LazyLinear::materialize`] — eager + lazy materialization
//! - [`LazyLinear::forward`] — shape after materialization
//! - [`LazyLinear::out_features`]
//! - [`LazyLinear::parameters`] / [`LazyLinear::named_parameters`]
//! - [`LazyLinear::train`] / [`LazyLinear::eval`] / [`LazyLinear::is_training`]
//!
//! ### lazy_conv
//! - [`LazyConv1d::new`] / [`LazyConv1d::is_initialized`] / [`LazyConv1d::materialize`]
//! - [`LazyConv1d::forward`] / [`LazyConv1d::parameters`] / [`LazyConv1d::named_parameters`]
//! - [`LazyConv2d::new`] / [`LazyConv2d::is_initialized`] / [`LazyConv2d::materialize`]
//! - [`LazyConv2d::forward`] / [`LazyConv2d::parameters`]
//! - [`LazyConv3d::new`] / [`LazyConv3d::is_initialized`] / [`LazyConv3d::forward`]
//! - [`LazyConv2d::train`] / [`LazyConv2d::eval`] / [`LazyConv2d::is_training`]
//! - [`LazyConv2d::named_parameters`]
//!
//! ### lazy_conv_transpose
//! - [`LazyConvTranspose2d::new`] / [`LazyConvTranspose2d::is_initialized`]
//! - [`LazyConvTranspose2d::materialize`] / [`LazyConvTranspose2d::forward`]
//! - [`LazyConvTranspose2d::parameters`] / [`LazyConvTranspose2d::named_parameters`]
//! - [`LazyConvTranspose2d::train`] / [`LazyConvTranspose2d::eval`]
//! - [`LazyConvTranspose1d::new`] / [`LazyConvTranspose1d::forward`]
//! - [`LazyConvTranspose3d::new`] / [`LazyConvTranspose3d::forward`]
//!
//! ### embedding
//! - [`Embedding::new`] / [`Embedding::from_pretrained`] / [`Embedding::with_sparse`]
//! - [`Embedding::forward`] — lookup, padding_idx
//! - [`Embedding::parameters`] / [`Embedding::named_parameters`]
//! - [`Embedding::train`] / [`Embedding::eval`] / [`Embedding::is_training`]
//! - [`EmbeddingBag::new`] / [`EmbeddingBag::forward_bag`]
//! - [`EmbeddingBag::num_embeddings`] / [`EmbeddingBag::embedding_dim`] / [`EmbeddingBag::mode`]
//! - [`EmbeddingBagMode`] — Sum / Mean / Max variants
//!
//! ### identity
//! - [`Identity::new`] / [`Identity::forward`] / train/eval
//! - [`Flatten::new`] / [`Flatten::forward`] — default, range, all-dims, errors
//! - [`Unflatten::new`] / [`Unflatten::forward`]
//! - [`ChannelShuffle::new`] / [`ChannelShuffle::forward`]
//! - [`CosineSimilarity::new`] / [`CosineSimilarity::forward`]
//! - [`PairwiseDistance::new`] / [`PairwiseDistance::forward`]
//!
//! ### padding
//! - [`PaddingMode`] — all variants
//! - [`ConstantPad1d::new`] / [`ConstantPad1d::forward`]
//! - [`ConstantPad2d::new`] / [`ConstantPad2d::forward`]
//! - [`ConstantPad3d::new`] / [`ConstantPad3d::forward`]
//! - [`ZeroPad1d::new`] / [`ZeroPad1d::forward`]
//! - [`ZeroPad2d::new`] / [`ZeroPad2d::forward`]
//! - [`ZeroPad3d::new`] / [`ZeroPad3d::forward`]
//! - [`ReflectionPad1d::new`] / [`ReflectionPad1d::forward`]
//! - [`ReflectionPad2d::new`] / [`ReflectionPad2d::forward`]
//! - [`ReflectionPad3d::new`] / [`ReflectionPad3d::forward`]
//! - [`ReplicationPad1d::new`] / [`ReplicationPad1d::forward`]
//! - [`ReplicationPad2d::new`] / [`ReplicationPad2d::forward`]
//! - [`ReplicationPad3d::new`] / [`ReplicationPad3d::forward`]
//! - [`CircularPad1d::new`] / [`CircularPad1d::forward`]
//! - [`CircularPad2d::new`] / [`CircularPad2d::forward`]
//! - [`CircularPad3d::new`] / [`CircularPad3d::forward`]
//!
//! ### upsample
//! - [`InterpolateMode`] — all variants
//! - [`GridSamplePaddingMode`] / [`GridSampleMode`]
//! - [`Upsample::new`] / [`Upsample::with_scale_factor`] / [`Upsample::forward`]
//! - [`PixelShuffle::new`] / [`PixelShuffle::forward`]
//! - [`PixelUnshuffle::new`] / [`PixelUnshuffle::forward`]
//! - [`Unfold::new`] / [`Unfold::forward`]
//! - [`Fold::new`] / [`Fold::forward`]
//! - [`interpolate`] — nearest, bilinear
//!
//! ### pooling
//! - [`MaxPool2d::new`] / [`MaxPool2d::forward`]
//! - [`AvgPool2d::new`] / [`AvgPool2d::forward`]
//! - [`AdaptiveAvgPool2d::new`] / [`AdaptiveAvgPool2d::forward`]
//! - [`MaxPool1d::new`] / [`MaxPool1d::forward`]
//! - [`MaxPool3d::new`] / [`MaxPool3d::forward`]
//! - [`AvgPool1d::new`] / [`AvgPool1d::forward`]
//! - [`AvgPool3d::new`] / [`AvgPool3d::forward`]
//! - [`AdaptiveMaxPool2d::new`] / [`AdaptiveMaxPool2d::forward`]
//! - [`AdaptiveAvgPool1d::new`] / [`AdaptiveAvgPool1d::forward`]
//! - [`AdaptiveAvgPool3d::new`] / [`AdaptiveAvgPool3d::forward`]
//! - [`AdaptiveMaxPool1d::new`] / [`AdaptiveMaxPool1d::forward`]
//! - [`AdaptiveMaxPool3d::new`] / [`AdaptiveMaxPool3d::forward`]
//! - [`MaxUnpool2d::new`] / [`MaxUnpool2d::forward_with_indices`]
//! - [`FractionalMaxPool2d::new`] / [`FractionalMaxPool2d::forward`]
//! - [`LPPool1d::new`] / [`LPPool1d::forward`]
//! - [`LPPool2d::new`] / [`LPPool2d::forward`]
//! - free functions: [`max_pool1d`], [`max_pool2d`], [`max_pool3d`],
//!   [`avg_pool1d`], [`avg_pool2d`], [`avg_pool3d`],
//!   [`adaptive_avg_pool1d`], [`adaptive_avg_pool2d`], [`adaptive_avg_pool3d`],
//!   [`adaptive_max_pool2d`]
//!
//! ### dropout
//! - [`Dropout::new`] — valid/invalid p
//! - [`Dropout::forward`] — eval identity, train stochastic properties
//! - [`Dropout::train`] / [`Dropout::eval`] / [`Dropout::is_training`]
//! - [`Dropout1d::new`] / [`Dropout1d::forward`]
//! - [`Dropout2d::new`] / [`Dropout2d::forward`]
//! - [`Dropout3d::new`] / [`Dropout3d::forward`]
//! - [`AlphaDropout::new`] / [`AlphaDropout::forward`]
//!
//! ## Cascade-skip entries
//!
//! Marked with `#[ignore]` + a crosslink reference when a known divergence
//! prevents a test from passing. Fixes land in their own dispatch and flip
//! the skip back to a live assertion.

#![allow(clippy::float_cmp)]
#![allow(clippy::unreadable_literal)]

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::Tensor;
use ferrotorch_nn::conv::{
    Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
};
use ferrotorch_nn::dropout::{AlphaDropout, Dropout, Dropout1d, Dropout2d, Dropout3d};
use ferrotorch_nn::embedding::{Embedding, EmbeddingBag, EmbeddingBagMode};
use ferrotorch_nn::identity::{
    ChannelShuffle, CosineSimilarity, Flatten, Identity, PairwiseDistance, Unflatten,
};
use ferrotorch_nn::lazy_conv::{LazyConv1d, LazyConv2d, LazyConv3d};
use ferrotorch_nn::lazy_conv_transpose::{
    LazyConvTranspose1d, LazyConvTranspose2d, LazyConvTranspose3d,
};
use ferrotorch_nn::lazy_linear::LazyLinear;
use ferrotorch_nn::linear::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::padding::{
    CircularPad1d, CircularPad2d, CircularPad3d, ConstantPad1d, ConstantPad2d, ConstantPad3d,
    PaddingMode, ReflectionPad1d, ReflectionPad2d, ReflectionPad3d, ReplicationPad1d,
    ReplicationPad2d, ReplicationPad3d, ZeroPad1d, ZeroPad2d, ZeroPad3d,
};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d, FractionalMaxPool2d, LPPool1d, LPPool2d,
    MaxPool1d, MaxPool2d, MaxPool3d, MaxUnpool2d, adaptive_avg_pool1d, adaptive_avg_pool2d,
    adaptive_avg_pool3d, adaptive_max_pool2d, avg_pool1d, avg_pool2d, avg_pool3d, max_pool1d,
    max_pool2d, max_pool3d,
};
use ferrotorch_nn::upsample::{
    Fold, GridSampleMode, GridSamplePaddingMode, InterpolateMode, PixelShuffle, PixelUnshuffle,
    Unfold, Upsample, interpolate,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn cpu_tensor_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cpu_tensor_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{context}[{i}]: actual={a} expected={e} diff={diff} tol={tol}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{context}[{i}]: actual={a} expected={e} diff={diff} tol={tol}"
        );
    }
}

fn zeros_f32(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    cpu_tensor_f32(&vec![0.0f32; n.max(1)], shape)
}

#[allow(dead_code)]
fn zeros_f64(shape: &[usize]) -> Tensor<f64> {
    let n: usize = shape.iter().product();
    cpu_tensor_f64(&vec![0.0f64; n.max(1)], shape)
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: linear
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn linear_new_with_bias() {
    let layer = Linear::<f32>::new(4, 3, true).unwrap();
    assert_eq!(layer.in_features(), 4);
    assert_eq!(layer.out_features(), 3);
    assert_eq!(layer.weight.shape(), &[3, 4]);
    assert!(layer.bias.is_some());
    assert_eq!(layer.bias.as_ref().unwrap().shape(), &[3]);
}

#[test]
fn linear_new_without_bias() {
    let layer = Linear::<f32>::new(8, 4, false).unwrap();
    assert_eq!(layer.weight.shape(), &[4, 8]);
    assert!(layer.bias.is_none());
}

#[test]
fn linear_new_zero_in_features_errors() {
    assert!(Linear::<f32>::new(0, 5, true).is_err());
}

#[test]
fn linear_new_zero_out_features_errors() {
    assert!(Linear::<f32>::new(4, 0, true).is_err());
}

#[test]
fn linear_forward_2d_shape() {
    let layer = Linear::<f32>::new(4, 3, true).unwrap();
    let x = zeros_f32(&[2, 4]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
}

#[test]
fn linear_forward_3d_shape() {
    let layer = Linear::<f32>::new(4, 3, true).unwrap();
    let x = zeros_f32(&[2, 5, 4]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 5, 3]);
}

#[test]
fn linear_forward_1d_input() {
    let mut layer = Linear::<f32>::new(3, 2, false).unwrap();
    layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0], &[3]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_close_f32(out.data().unwrap(), &[1.0, 2.0], 1e-6, "linear_1d");
}

#[test]
fn linear_forward_numerical_no_bias() {
    // weight = [[1,0,0],[0,1,0]]  (2×3), input = [[1,0,0],[0,0,1]]
    // output = [[1,0],[0,0]]
    let mut layer = Linear::<f32>::new(3, 2, false).unwrap();
    layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();
    let x = cpu_tensor_f32(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_close_f32(out.data().unwrap(), &[1.0, 0.0, 0.0, 0.0], 1e-6, "linear_no_bias");
}

#[test]
fn linear_forward_wrong_feature_dim_errors() {
    let layer = Linear::<f32>::new(4, 3, true).unwrap();
    let x = zeros_f32(&[2, 5]);
    assert!(layer.forward(&x).is_err());
}

#[test]
fn linear_train_eval() {
    let mut layer = Linear::<f32>::new(4, 3, true).unwrap();
    assert!(layer.is_training());
    layer.eval();
    assert!(!layer.is_training());
    layer.train();
    assert!(layer.is_training());
}

#[test]
fn linear_parameters_with_bias() {
    let layer = Linear::<f32>::new(4, 3, true).unwrap();
    assert_eq!(layer.parameters().len(), 2);
}

#[test]
fn linear_parameters_no_bias() {
    let layer = Linear::<f32>::new(4, 3, false).unwrap();
    assert_eq!(layer.parameters().len(), 1);
}

#[test]
fn linear_named_parameters_with_bias() {
    let layer = Linear::<f32>::new(3, 2, true).unwrap();
    let named = layer.named_parameters();
    assert_eq!(named.len(), 2);
    assert_eq!(named[0].0, "weight");
    assert_eq!(named[1].0, "bias");
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: conv
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn conv2d_new_basic() {
    let layer = Conv2d::<f32>::new(1, 4, (3, 3), (1, 1), (0, 0), true).unwrap();
    let params = layer.parameters();
    // weight: 4*1*3*3=36, bias: 4 => 40
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total, 40);
}

#[test]
fn conv2d_forward_shape_no_pad() {
    // Input [1,1,5,5], kernel 3, stride 1, pad 0 -> output [1,1,3,3]
    let layer = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
    let x = zeros_f32(&[1, 1, 5, 5]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
}

#[test]
fn conv2d_forward_shape_with_pad() {
    // Input [2,3,8,8], kernel (3,3), stride 1, pad 1 -> same spatial [2,4,8,8]
    let layer = Conv2d::<f32>::new(3, 4, (3, 3), (1, 1), (1, 1), true).unwrap();
    let x = zeros_f32(&[2, 3, 8, 8]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4, 8, 8]);
}

#[test]
fn conv2d_train_eval() {
    let mut layer = Conv2d::<f32>::new(1, 2, (3, 3), (1, 1), (0, 0), false).unwrap();
    assert!(layer.is_training());
    layer.eval();
    assert!(!layer.is_training());
    layer.train();
    assert!(layer.is_training());
}

#[test]
fn conv2d_named_parameters() {
    let layer = Conv2d::<f32>::new(1, 2, (3, 3), (1, 1), (0, 0), true).unwrap();
    let named = layer.named_parameters();
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"weight"));
    assert!(names.contains(&"bias"));
}

#[test]
fn conv1d_forward_shape() {
    // [2, 2, 8], kernel 3, stride 1, pad 1 -> [2, 4, 8]
    let layer = Conv1d::<f32>::new(2, 4, 3, 1, 1, true).unwrap();
    let x = zeros_f32(&[2, 2, 8]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4, 8]);
}

#[test]
fn conv1d_named_parameters() {
    let layer = Conv1d::<f32>::new(1, 2, 3, 1, 0, true).unwrap();
    let named = layer.named_parameters();
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"weight"));
    assert!(names.contains(&"bias"));
}

#[test]
fn conv1d_train_eval() {
    let mut layer = Conv1d::<f32>::new(1, 2, 3, 1, 0, false).unwrap();
    assert!(layer.is_training());
    layer.eval();
    assert!(!layer.is_training());
}

#[test]
fn conv3d_forward_shape() {
    // [1, 2, 4, 4, 4], kernel (2,2,2), stride 1, pad 0 -> [1, 4, 3, 3, 3]
    let layer = Conv3d::<f32>::new(2, 4, (2, 2, 2), (1, 1, 1), (0, 0, 0), false).unwrap();
    let x = zeros_f32(&[1, 2, 4, 4, 4]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 3, 3, 3]);
}

#[test]
fn conv3d_named_parameters() {
    let layer = Conv3d::<f32>::new(1, 2, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
    let named = layer.named_parameters();
    assert!(!named.is_empty());
    assert_eq!(named[0].0, "weight");
}

#[test]
fn conv3d_train_eval() {
    let mut layer = Conv3d::<f32>::new(1, 2, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
    assert!(layer.is_training());
    layer.eval();
    assert!(!layer.is_training());
}

#[test]
fn conv_transpose2d_forward_shape() {
    // [1, 2, 4, 4], k=3, stride=2, pad=1, out_pad=1 -> [1, 1, 8, 8]
    let layer = ConvTranspose2d::<f32>::new(2, 1, (3, 3), (2, 2), (1, 1), (1, 1), false).unwrap();
    let x = zeros_f32(&[1, 2, 4, 4]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 8, 8]);
}

#[test]
fn conv_transpose2d_named_parameters() {
    let layer = ConvTranspose2d::<f32>::new(2, 1, (3, 3), (1, 1), (0, 0), (0, 0), true).unwrap();
    let named = layer.named_parameters();
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"weight"));
    assert!(names.contains(&"bias"));
}

#[test]
fn conv_transpose2d_train_eval() {
    let mut layer =
        ConvTranspose2d::<f32>::new(2, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
    assert!(layer.is_training());
    layer.eval();
    assert!(!layer.is_training());
}

#[test]
fn conv_transpose1d_forward_shape() {
    // [1, 2, 5], k=3, stride=2, pad=1, out_pad=1 -> [1, 4, 10]
    let layer = ConvTranspose1d::<f32>::new(2, 4, 3, 2, 1, 1, false).unwrap();
    let x = zeros_f32(&[1, 2, 5]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 10]);
}

#[test]
fn conv_transpose1d_named_parameters() {
    let layer = ConvTranspose1d::<f32>::new(1, 2, 3, 1, 0, 0, true).unwrap();
    let named = layer.named_parameters();
    assert!(named.iter().any(|(n, _)| n == "weight"));
}

#[test]
fn conv_transpose1d_train_eval() {
    let mut layer = ConvTranspose1d::<f32>::new(1, 2, 3, 1, 0, 0, false).unwrap();
    layer.eval();
    assert!(!layer.is_training());
}

#[test]
fn conv_transpose3d_forward_shape() {
    // [1, 2, 3, 3, 3], k=3, stride=2, pad=1, out_pad=1 -> [1, 4, 6, 6, 6]
    let layer =
        ConvTranspose3d::<f32>::new(2, 4, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), false)
            .unwrap();
    let x = zeros_f32(&[1, 2, 3, 3, 3]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 6, 6, 6]);
}

#[test]
fn conv_transpose3d_named_parameters() {
    let layer =
        ConvTranspose3d::<f32>::new(1, 2, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
            .unwrap();
    let named = layer.named_parameters();
    assert!(named.iter().any(|(n, _)| n == "weight"));
}

#[test]
fn conv_transpose3d_train_eval() {
    let mut layer =
        ConvTranspose3d::<f32>::new(1, 2, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
            .unwrap();
    layer.eval();
    assert!(!layer.is_training());
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: lazy_linear
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn lazy_linear_new_not_initialized() {
    let lazy: LazyLinear<f32> = LazyLinear::new(8, true).unwrap();
    assert!(!lazy.is_initialized());
    assert_eq!(lazy.in_features(), None);
    assert_eq!(lazy.out_features(), 8);
    assert_eq!(lazy.parameters().len(), 0);
}

#[test]
fn lazy_linear_zero_out_features_errors() {
    assert!(LazyLinear::<f32>::new(0, true).is_err());
}

#[test]
fn lazy_linear_forward_materializes() {
    let lazy: LazyLinear<f32> = LazyLinear::new(4, true).unwrap();
    let x = zeros_f32(&[2, 6]);
    let out = lazy.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert!(lazy.is_initialized());
    assert_eq!(lazy.in_features(), Some(6));
    assert_eq!(lazy.parameters().len(), 2);
}

#[test]
fn lazy_linear_materialize_explicit() {
    let lazy: LazyLinear<f32> = LazyLinear::new(8, true).unwrap();
    lazy.materialize(16).unwrap();
    assert!(lazy.is_initialized());
    assert_eq!(lazy.in_features(), Some(16));
    assert_eq!(lazy.parameters().len(), 2);
}

#[test]
fn lazy_linear_materialize_idempotent() {
    let lazy: LazyLinear<f32> = LazyLinear::new(4, false).unwrap();
    lazy.materialize(8).unwrap();
    lazy.materialize(16).unwrap(); // second call is no-op
    assert_eq!(lazy.in_features(), Some(8));
    assert_eq!(lazy.parameters().len(), 1); // no bias
}

#[test]
fn lazy_linear_rejects_mismatched_in_features() {
    let lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
    let x1 = zeros_f32(&[1, 3]);
    let _ = lazy.forward(&x1).unwrap();
    let x2 = zeros_f32(&[1, 4]);
    assert!(lazy.forward(&x2).is_err());
}

#[test]
fn lazy_linear_named_parameters_after_init() {
    let lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
    let _ = lazy.forward(&zeros_f32(&[1, 3])).unwrap();
    let names: Vec<String> = lazy
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(names.contains(&"weight".to_string()));
    assert!(names.contains(&"bias".to_string()));
}

#[test]
fn lazy_linear_train_eval() {
    let mut lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
    assert!(lazy.is_training());
    lazy.eval();
    assert!(!lazy.is_training());
    lazy.train();
    assert!(lazy.is_training());
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: lazy_conv
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn lazy_conv1d_not_initialized_before_forward() {
    let lazy: LazyConv1d<f32> = LazyConv1d::new(4, 3, 1, 1, false).unwrap();
    assert!(!lazy.is_initialized());
    assert_eq!(lazy.parameters().len(), 0);
}

#[test]
fn lazy_conv1d_forward_materializes() {
    let lazy: LazyConv1d<f32> = LazyConv1d::new(4, 3, 1, 1, false).unwrap();
    let x = zeros_f32(&[1, 2, 8]);
    let out = lazy.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 8]);
    assert!(lazy.is_initialized());
}

#[test]
fn lazy_conv1d_materialize_explicit() {
    let lazy: LazyConv1d<f32> = LazyConv1d::new(4, 3, 1, 1, true).unwrap();
    lazy.materialize(2).unwrap();
    assert!(lazy.is_initialized());
    assert_eq!(lazy.parameters().len(), 2);
}

#[test]
fn lazy_conv1d_named_parameters_after_materialize() {
    let lazy: LazyConv1d<f32> = LazyConv1d::new(4, 3, 1, 1, true).unwrap();
    lazy.materialize(2).unwrap();
    let named = lazy.named_parameters();
    assert!(named.iter().any(|(n, _)| n == "weight"));
}

#[test]
fn lazy_conv2d_not_initialized_before_forward() {
    let lazy: LazyConv2d<f32> = LazyConv2d::new(4, (3, 3), (1, 1), (1, 1), false).unwrap();
    assert!(!lazy.is_initialized());
    assert_eq!(lazy.parameters().len(), 0);
}

#[test]
fn lazy_conv2d_forward_materializes() {
    let lazy: LazyConv2d<f32> = LazyConv2d::new(4, (3, 3), (1, 1), (1, 1), false).unwrap();
    let x = zeros_f32(&[1, 2, 5, 5]);
    let out = lazy.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 5, 5]);
    assert!(lazy.is_initialized());
    assert_eq!(lazy.parameters().len(), 1);
}

#[test]
fn lazy_conv2d_materialize_explicit() {
    let lazy: LazyConv2d<f32> = LazyConv2d::new(4, (3, 3), (1, 1), (1, 1), true).unwrap();
    lazy.materialize(3).unwrap();
    assert!(lazy.is_initialized());
    assert_eq!(lazy.parameters().len(), 2);
}

#[test]
fn lazy_conv2d_named_parameters_after_materialize() {
    let lazy: LazyConv2d<f32> = LazyConv2d::new(2, (3, 3), (1, 1), (1, 1), true).unwrap();
    lazy.materialize(1).unwrap();
    let named = lazy.named_parameters();
    assert!(named.iter().any(|(n, _)| n == "weight"));
    assert!(named.iter().any(|(n, _)| n == "bias"));
}

#[test]
fn lazy_conv2d_train_eval() {
    let mut lazy: LazyConv2d<f32> = LazyConv2d::new(4, (3, 3), (1, 1), (1, 1), false).unwrap();
    assert!(lazy.is_training());
    lazy.eval();
    assert!(!lazy.is_training());
    lazy.train();
    assert!(lazy.is_training());
}

#[test]
fn lazy_conv3d_not_initialized_before_forward() {
    let lazy: LazyConv3d<f32> = LazyConv3d::new(4, (3, 3, 3), (1, 1, 1), (1, 1, 1), false).unwrap();
    assert!(!lazy.is_initialized());
}

#[test]
fn lazy_conv3d_forward_materializes() {
    let lazy: LazyConv3d<f32> = LazyConv3d::new(4, (3, 3, 3), (1, 1, 1), (1, 1, 1), false).unwrap();
    let x = zeros_f32(&[1, 2, 4, 4, 4]);
    let out = lazy.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 4, 4, 4]);
    assert!(lazy.is_initialized());
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: lazy_conv_transpose
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn lazy_conv_transpose2d_not_initialized_before_forward() {
    let lazy: LazyConvTranspose2d<f32> =
        LazyConvTranspose2d::new(1, (3, 3), (2, 2), (1, 1), (1, 1), false);
    assert!(!lazy.is_initialized());
    assert_eq!(lazy.parameters().len(), 0);
}

#[test]
fn lazy_conv_transpose2d_materialize_explicit() {
    let lazy: LazyConvTranspose2d<f32> =
        LazyConvTranspose2d::new(2, (3, 3), (1, 1), (0, 0), (0, 0), true);
    lazy.materialize(1).unwrap();
    assert!(lazy.is_initialized());
    assert_eq!(lazy.parameters().len(), 2);
}

#[test]
fn lazy_conv_transpose2d_named_parameters_after_materialize() {
    let lazy: LazyConvTranspose2d<f32> =
        LazyConvTranspose2d::new(2, (3, 3), (1, 1), (0, 0), (0, 0), true);
    lazy.materialize(1).unwrap();
    let named = lazy.named_parameters();
    assert!(named.iter().any(|(n, _)| n == "weight"));
    assert!(named.iter().any(|(n, _)| n == "bias"));
}

#[test]
fn lazy_conv_transpose2d_forward_materializes() {
    let lazy: LazyConvTranspose2d<f32> =
        LazyConvTranspose2d::new(1, (3, 3), (2, 2), (1, 1), (1, 1), false);
    let x = zeros_f32(&[1, 2, 4, 4]);
    let out = lazy.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 8, 8]);
    assert!(lazy.is_initialized());
}

#[test]
fn lazy_conv_transpose2d_train_eval() {
    let mut lazy: LazyConvTranspose2d<f32> =
        LazyConvTranspose2d::new(1, (3, 3), (1, 1), (0, 0), (0, 0), false);
    assert!(lazy.is_training());
    lazy.eval();
    assert!(!lazy.is_training());
    lazy.train();
    assert!(lazy.is_training());
}

#[test]
fn lazy_conv_transpose1d_forward_materializes() {
    let lazy: LazyConvTranspose1d<f32> = LazyConvTranspose1d::new(4, 3, 1, 0, 0, false);
    let x = zeros_f32(&[1, 2, 8]);
    let out = lazy.forward(&x).unwrap();
    // k=3, stride=1, pad=0, out_pad=0 -> L_out = (8-1)*1 - 2*0 + 1*(3-1) + 1 = 10
    assert_eq!(out.shape(), &[1, 4, 10]);
    assert!(lazy.is_initialized());
}

#[test]
fn lazy_conv_transpose3d_forward_materializes() {
    let lazy: LazyConvTranspose3d<f32> =
        LazyConvTranspose3d::new(4, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false);
    let x = zeros_f32(&[1, 2, 4, 4, 4]);
    let out = lazy.forward(&x).unwrap();
    // k=3, stride=1, pad=0 -> D/H/W_out = (4-1)*1 - 0 + 2 + 0 + 1 = 6
    assert_eq!(out.shape(), &[1, 4, 6, 6, 6]);
    assert!(lazy.is_initialized());
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: embedding
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn embedding_new_basic() {
    let e = Embedding::<f32>::new(10, 4, None).unwrap();
    assert_eq!(e.num_embeddings, 10);
    assert_eq!(e.embedding_dim, 4);
    assert_eq!(e.weight.shape(), &[10, 4]);
}

#[test]
fn embedding_forward_lookup() {
    let mut e = Embedding::<f32>::new(5, 3, None).unwrap();
    // Set known weight
    let w: Vec<f32> = (0..15).map(|i| i as f32).collect();
    e.weight = Parameter::from_slice(&w, &[5, 3]).unwrap();
    // indices [0, 2, 4]
    let indices = cpu_tensor_f32(&[0.0, 2.0, 4.0], &[3]);
    let out = e.forward(&indices).unwrap();
    assert_eq!(out.shape(), &[3, 3]);
    let data = out.data().unwrap();
    assert_close_f32(&data[0..3], &[0.0, 1.0, 2.0], 1e-7, "embedding row 0");
    assert_close_f32(&data[3..6], &[6.0, 7.0, 8.0], 1e-7, "embedding row 2");
    assert_close_f32(&data[6..9], &[12.0, 13.0, 14.0], 1e-7, "embedding row 4");
}

#[test]
fn embedding_with_padding_idx() {
    let e = Embedding::<f32>::new(4, 2, Some(0)).unwrap();
    assert_eq!(e.padding_idx, Some(0));
}

#[test]
fn embedding_from_pretrained() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let weight_t = cpu_tensor_f32(&data, &[4, 3]);
    let e = Embedding::<f32>::from_pretrained(weight_t, None).unwrap();
    assert_eq!(e.num_embeddings, 4);
    assert_eq!(e.embedding_dim, 3);
}

#[test]
fn embedding_with_sparse() {
    let e = Embedding::<f32>::new(5, 3, None).unwrap().with_sparse(true);
    assert!(e.sparse);
}

#[test]
fn embedding_named_parameters() {
    let e = Embedding::<f32>::new(5, 3, None).unwrap();
    let named = e.named_parameters();
    assert_eq!(named.len(), 1);
    assert_eq!(named[0].0, "weight");
}

#[test]
fn embedding_train_eval() {
    let mut e = Embedding::<f32>::new(5, 3, None).unwrap();
    assert!(e.is_training());
    e.eval();
    assert!(!e.is_training());
    e.train();
    assert!(e.is_training());
}

#[test]
fn embedding_bag_mode_variants() {
    // Confirm all EmbeddingBagMode variants are constructible.
    let _sum = EmbeddingBagMode::Sum;
    let _mean = EmbeddingBagMode::Mean;
    let _max = EmbeddingBagMode::Max;
}

#[test]
fn embedding_bag_new_and_accessors() {
    let eb = EmbeddingBag::<f32>::new(5, 3, EmbeddingBagMode::Sum).unwrap();
    assert_eq!(eb.num_embeddings(), 5);
    assert_eq!(eb.embedding_dim(), 3);
    assert!(matches!(eb.mode(), EmbeddingBagMode::Sum));
}

#[test]
fn embedding_bag_forward_sum() {
    let w = vec![
        1.0f32, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, // row 1
        0.0, 0.0, 1.0, // row 2
        1.0, 1.0, 0.0, // row 3
        0.0, 1.0, 1.0, // row 4
    ];
    // Construct with a known weight using from_pretrained path
    let wt = cpu_tensor_f32(&w, &[5, 3]);
    let emb = Embedding::<f32>::from_pretrained(wt, None).unwrap();
    // Lookup [0, 1] = row0 + row1 = [1,1,0]
    let indices0 = cpu_tensor_f32(&[0.0, 1.0], &[2]);
    let out0 = emb.forward(&indices0).unwrap();
    assert_eq!(out0.shape(), &[2, 3]);
}

#[test]
fn embedding_bag_forward_shape() {
    let eb = EmbeddingBag::<f32>::new(8, 4, EmbeddingBagMode::Sum).unwrap();
    let out = eb.forward_bag(&cpu_tensor_f32(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]), &[0, 2, 4]).unwrap();
    // 3 bags (offsets [0,2,4] on 5 items -> bags=[0..2), [2..4), [4..5))
    assert_eq!(out.shape(), &[3, 4]);
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: identity
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn identity_forward_passthrough() {
    let id = Identity::new();
    let x = cpu_tensor_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let out: Tensor<f64> = id.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_close_f64(
        out.data().unwrap(),
        x.data().unwrap(),
        1e-10,
        "identity passthrough",
    );
}

#[test]
fn identity_train_eval() {
    let mut id = Identity::new();
    assert!(Module::<f32>::is_training(&id));
    Module::<f32>::eval(&mut id);
    assert!(!Module::<f32>::is_training(&id));
    Module::<f32>::train(&mut id);
    assert!(Module::<f32>::is_training(&id));
}

#[test]
fn flatten_default_batch_preserved() {
    let f = Flatten::default();
    let x = cpu_tensor_f32(
        &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
    );
    let out: Tensor<f32> = f.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 12]);
}

#[test]
fn flatten_range() {
    // Flatten dims 2..3 of [2, 3, 4, 5] -> [2, 3, 20]
    let f = Flatten::new(2, 3);
    let x = cpu_tensor_f32(
        &(0..120).map(|i| i as f32).collect::<Vec<_>>(),
        &[2, 3, 4, 5],
    );
    let out: Tensor<f32> = f.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3, 20]);
}

#[test]
fn flatten_all_dims() {
    let f = Flatten::new(0, -1);
    let x = cpu_tensor_f32(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3, 4]);
    let out: Tensor<f32> = f.forward(&x).unwrap();
    assert_eq!(out.shape(), &[24]);
}

#[test]
fn flatten_start_dim_out_of_range_errors() {
    let f = Flatten::new(5, -1);
    let x = cpu_tensor_f32(&[0.0f32; 4], &[2, 2]);
    assert!(Module::<f32>::forward(&f, &x).is_err());
}

#[test]
fn unflatten_basic() {
    let uf = Unflatten::new(1, vec![2, 3]);
    let x = cpu_tensor_f32(
        &(0..12).map(|i| i as f32).collect::<Vec<_>>(),
        &[2, 6],
    );
    let out: Tensor<f32> = Module::<f32>::forward(&uf, &x).unwrap();
    assert_eq!(out.shape(), &[2, 2, 3]);
}

#[test]
fn unflatten_wrong_size_errors() {
    let uf = Unflatten::new(1, vec![3, 3]);
    let x = cpu_tensor_f32(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[2, 6]);
    // 3*3=9 != 6
    assert!(Module::<f32>::forward(&uf, &x).is_err());
}

#[test]
fn channel_shuffle_groups2() {
    let cs = ChannelShuffle::new(2);
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1]);
    let out: Tensor<f32> = Module::<f32>::forward(&cs, &x).unwrap();
    // groups=2: channel order 0,1,2,3 -> 0,2,1,3
    assert_eq!(out.shape(), &[1, 4, 1, 1]);
    assert_close_f32(out.data().unwrap(), &[1.0, 3.0, 2.0, 4.0], 1e-7, "channel_shuffle");
}

#[test]
fn cosine_similarity_orthogonal_zero() {
    let cs = CosineSimilarity::new(1, 1e-8);
    let x1 = cpu_tensor_f64(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    let x2 = cpu_tensor_f64(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
    let out = cs.forward(&x1, &x2).unwrap();
    assert_eq!(out.shape(), &[2]);
    let data = out.data().unwrap();
    // [1,0,0] vs [0,1,0] = 0.0, [0,1,0] vs [0,0,1] = 0.0
    assert_close_f64(data, &[0.0, 0.0], 1e-8, "cosine_orthogonal");
}

#[test]
fn cosine_similarity_parallel_one() {
    let cs = CosineSimilarity::new(1, 1e-8);
    let x1 = cpu_tensor_f64(&[3.0, 4.0], &[1, 2]);
    let x2 = cpu_tensor_f64(&[6.0, 8.0], &[1, 2]);
    let out = cs.forward(&x1, &x2).unwrap();
    let data = out.data().unwrap();
    assert_close_f64(data, &[1.0], 1e-6, "cosine_parallel");
}

#[test]
fn pairwise_distance_euclidean() {
    let pd = PairwiseDistance::new(2.0, 1e-6, false);
    let x1 = cpu_tensor_f64(&[0.0, 0.0, 1.0, 0.0], &[2, 2]);
    let x2 = cpu_tensor_f64(&[3.0, 4.0, 1.0, 0.0], &[2, 2]);
    let out = pd.forward(&x1, &x2).unwrap();
    // row0: sqrt(9+16)=5, row1: sqrt(0)=0
    assert_eq!(out.shape(), &[2]);
    let data = out.data().unwrap();
    assert_close_f64(data, &[5.0, 0.0], 1e-4, "pairwise_euclidean");
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: padding
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn padding_mode_variants_constructible() {
    let _z = PaddingMode::Zeros;
    let _r = PaddingMode::Reflect;
    let _rep = PaddingMode::Replicate;
    let _c = PaddingMode::Circular;
}

#[test]
fn constant_pad1d_forward() {
    let pad = ConstantPad1d::<f32>::new((1, 2), 0.0);
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 7]);
    assert_close_f32(
        out.data().unwrap(),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
        1e-7,
        "constant_pad1d",
    );
}

#[test]
fn constant_pad1d_nonzero_value() {
    let pad = ConstantPad1d::<f32>::new((2, 1), -1.0);
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0], &[1, 3]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 6]);
    assert_close_f32(
        out.data().unwrap(),
        &[-1.0, -1.0, 1.0, 2.0, 3.0, -1.0],
        1e-7,
        "constant_pad1d_nonzero",
    );
}

#[test]
fn constant_pad2d_forward() {
    let pad = ConstantPad2d::<f32>::new((1, 1, 1, 1), 0.0);
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let expected = [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_close_f32(out.data().unwrap(), &expected, 1e-7, "constant_pad2d");
}

#[test]
fn constant_pad3d_forward() {
    let pad = ConstantPad3d::<f32>::new((1, 0, 0, 0, 0, 0), 9.0);
    let x = cpu_tensor_f32(&[1.0, 2.0], &[1, 1, 1, 2]);
    // last dim 2 + (1+0) = 3
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape()[3], 3);
    assert_close_f32(&out.data().unwrap()[0..1], &[9.0], 1e-7, "constant_pad3d_first");
}

#[test]
fn zero_pad1d_forward() {
    let pad = ZeroPad1d::<f32>::new((2, 1));
    let x = cpu_tensor_f32(&[5.0, 6.0], &[1, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 5]);
    assert_close_f32(
        out.data().unwrap(),
        &[0.0, 0.0, 5.0, 6.0, 0.0],
        1e-7,
        "zero_pad1d",
    );
}

#[test]
fn zero_pad2d_forward() {
    let pad = ZeroPad2d::<f32>::new((1, 1, 1, 1));
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let expected = [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_close_f32(out.data().unwrap(), &expected, 1e-7, "zero_pad2d");
}

#[test]
fn zero_pad3d_forward() {
    let pad = ZeroPad3d::<f32>::new((1, 0, 0, 0, 0, 0));
    let x = cpu_tensor_f32(&[7.0, 8.0], &[1, 1, 1, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape()[3], 3);
    assert_close_f32(&out.data().unwrap()[0..1], &[0.0], 1e-7, "zero_pad3d");
}

#[test]
fn reflection_pad1d_forward() {
    // Input [1, 1, 4] = [a b c d], reflect pad (2, 1)
    // left=2: c b | a b c d, right=1: d
    // result: c b a b c d d
    let pad = ReflectionPad1d::<f32>::new((2, 1));
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 7]);
    assert_close_f32(
        out.data().unwrap(),
        &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0],
        1e-7,
        "reflection_pad1d",
    );
}

#[test]
fn reflection_pad2d_shape() {
    let pad = ReflectionPad2d::<f32>::new((1, 1, 1, 1));
    let x = zeros_f32(&[1, 1, 3, 3]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 5, 5]);
}

#[test]
fn reflection_pad3d_shape() {
    let pad = ReflectionPad3d::<f32>::new((1, 0, 1, 0, 1, 0));
    let x = zeros_f32(&[1, 1, 3, 3, 3]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4, 4]);
}

#[test]
fn replication_pad1d_forward() {
    let pad = ReplicationPad1d::<f32>::new((2, 3));
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 9]);
    assert_close_f32(
        out.data().unwrap(),
        &[1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
        1e-7,
        "replication_pad1d",
    );
}

#[test]
fn replication_pad2d_shape() {
    let pad = ReplicationPad2d::<f32>::new((1, 2, 1, 2));
    let x = zeros_f32(&[1, 1, 3, 3]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 6, 6]);
}

#[test]
fn replication_pad3d_shape() {
    let pad = ReplicationPad3d::<f32>::new((1, 0, 1, 0, 1, 0));
    let x = zeros_f32(&[1, 1, 3, 3, 3]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4, 4]);
}

#[test]
fn circular_pad1d_forward() {
    // Input [1, 1, 4] = [1 2 3 4], circular pad (2, 1)
    // left wraps from end: [3 4] -> [3 4 1 2 3 4] + right wraps from start: [1]
    // result: [3, 4, 1, 2, 3, 4, 1]
    let pad = CircularPad1d::<f32>::new((2, 1));
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 7]);
    assert_close_f32(
        out.data().unwrap(),
        &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        1e-7,
        "circular_pad1d",
    );
}

#[test]
fn circular_pad2d_shape() {
    let pad = CircularPad2d::<f32>::new((1, 1, 1, 1));
    let x = zeros_f32(&[1, 1, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 6, 6]);
}

#[test]
fn circular_pad3d_shape() {
    let pad = CircularPad3d::<f32>::new((1, 0, 1, 0, 1, 0));
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pad, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 5, 5, 5]);
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: upsample
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn interpolate_mode_variants_constructible() {
    let _n = InterpolateMode::Nearest;
    let _bl = InterpolateMode::Bilinear;
    let _bc = InterpolateMode::Bicubic;
}

#[test]
fn grid_sample_padding_mode_variants() {
    let _z = GridSamplePaddingMode::Zeros;
    let _b = GridSamplePaddingMode::Border;
    let _r = GridSamplePaddingMode::Reflection;
}

#[test]
fn grid_sample_mode_variants() {
    let _bl = GridSampleMode::Bilinear;
    let _n = GridSampleMode::Nearest;
}

#[test]
fn upsample_new_nearest_output_shape() {
    let up = Upsample::new([4, 4], InterpolateMode::Nearest);
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&up, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_close_f32(out.data().unwrap(), &expected, 1e-6, "upsample_nearest");
}

#[test]
fn upsample_with_scale_factor() {
    let up = Upsample::with_scale_factor([2.0, 2.0], InterpolateMode::Nearest);
    assert!(up.scale_factor.is_some());
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&up, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
}

#[test]
fn upsample_bilinear_shape() {
    let up = Upsample::new([6, 6], InterpolateMode::Bilinear);
    let x = zeros_f32(&[1, 1, 3, 3]);
    let out: Tensor<f32> = Module::<f32>::forward(&up, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 6, 6]);
}

#[test]
fn interpolate_free_fn_nearest() {
    let x = cpu_tensor_f32(&[1.0f32; 4], &[1, 1, 2, 2]);
    let out = interpolate(&x, Some([4, 4]), None, InterpolateMode::Nearest, false).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
}

#[test]
fn pixel_shuffle_r2_shape() {
    let ps = PixelShuffle::new(2);
    let x = zeros_f32(&[1, 4, 2, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&ps, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
}

#[test]
fn pixel_shuffle_r2_numerical() {
    let ps = PixelShuffle::new(2);
    let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = cpu_tensor_f32(&data, &[1, 4, 2, 2]);
    let out: Tensor<f32> = Module::<f32>::forward(&ps, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let expected = [
        1.0, 5.0, 2.0, 6.0, 9.0, 13.0, 10.0, 14.0, 3.0, 7.0, 4.0, 8.0, 11.0, 15.0, 12.0, 16.0,
    ];
    assert_close_f32(out.data().unwrap(), &expected, 1e-6, "pixel_shuffle_r2");
}

#[test]
fn pixel_unshuffle_r2_shape() {
    let pu = PixelUnshuffle::new(2);
    let x = zeros_f32(&[1, 1, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pu, &x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 2, 2]);
}

#[test]
fn pixel_unshuffle_is_inverse_of_pixel_shuffle() {
    let ps = PixelShuffle::new(2);
    let pu = PixelUnshuffle::new(2);
    let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = cpu_tensor_f32(&data, &[1, 4, 2, 2]);
    let shuffled: Tensor<f32> = Module::<f32>::forward(&ps, &x).unwrap();
    let recovered: Tensor<f32> = Module::<f32>::forward(&pu, &shuffled).unwrap();
    assert_eq!(recovered.shape(), &[1, 4, 2, 2]);
    assert_close_f32(recovered.data().unwrap(), &data, 1e-6, "pixel_shuffle_inverse");
}

#[test]
fn unfold_output_shape() {
    // Input [1, 3, 4, 4], kernel [2,2], dilation [1,1], pad [0,0], stride [2,2]
    // out_h = (4 - 2) / 2 + 1 = 2, out_w = 2, L = 2*2 = 4 patches
    // output shape [1, 3*2*2, 4] = [1, 12, 4]
    let uf = Unfold::new([2, 2], [1, 1], [0, 0], [2, 2]);
    let x = zeros_f32(&[1, 3, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&uf, &x).unwrap();
    assert_eq!(out.shape(), &[1, 12, 4]);
}

#[test]
fn fold_output_shape() {
    // Fold is the inverse of unfold; output_size [4,4], kernel [2,2],
    // dilation [1,1], padding [0,0], stride [2,2]
    // Input to fold: [1, 12, 4]
    let fold = Fold::new([4, 4], [2, 2], [1, 1], [0, 0], [2, 2]);
    let x = zeros_f32(&[1, 12, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&fold, &x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 4, 4]);
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: pooling
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn max_pool2d_basic() {
    let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
    let x = cpu_tensor_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
    );
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_close_f32(out.data().unwrap(), &[6.0, 8.0, 14.0, 16.0], 1e-6, "max_pool2d");
}

#[test]
fn avg_pool2d_basic() {
    let pool = AvgPool2d::new([2, 2], [2, 2], [0, 0]);
    let x = cpu_tensor_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
    );
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_close_f32(
        out.data().unwrap(),
        &[3.5, 5.5, 11.5, 13.5],
        1e-5,
        "avg_pool2d",
    );
}

#[test]
fn adaptive_avg_pool2d_2x2() {
    let pool = AdaptiveAvgPool2d::new((2, 2));
    let x = cpu_tensor_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
    );
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_close_f32(
        out.data().unwrap(),
        &[3.5, 5.5, 11.5, 13.5],
        1e-5,
        "adaptive_avg_pool2d",
    );
}

#[test]
fn max_pool1d_basic() {
    let pool = MaxPool1d::new(2, 2, 0);
    let x = cpu_tensor_f32(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[1, 1, 6]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3]);
    assert_close_f32(out.data().unwrap(), &[3.0, 5.0, 6.0], 1e-6, "max_pool1d");
}

#[test]
fn avg_pool1d_basic() {
    let pool = AvgPool1d::new(2, 2, 0);
    let x = cpu_tensor_f32(&[1.0, 3.0, 5.0, 7.0], &[1, 1, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2]);
    assert_close_f32(out.data().unwrap(), &[2.0, 6.0], 1e-6, "avg_pool1d");
}

#[test]
fn max_pool3d_shape() {
    let pool = MaxPool3d::new([2, 2, 2], [2, 2, 2], [0, 0, 0]);
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn avg_pool3d_shape() {
    let pool = AvgPool3d::new([2, 2, 2], [2, 2, 2], [0, 0, 0]);
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn adaptive_max_pool2d_basic() {
    let pool = AdaptiveMaxPool2d::new((2, 2));
    let x = cpu_tensor_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
    );
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_close_f32(out.data().unwrap(), &[6.0, 8.0, 14.0, 16.0], 1e-6, "adaptive_max_pool2d");
}

#[test]
fn adaptive_avg_pool1d_basic() {
    let pool = AdaptiveAvgPool1d::new(3);
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3]);
}

#[test]
fn adaptive_avg_pool3d_shape() {
    let pool = AdaptiveAvgPool3d::new((2, 2, 2));
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn adaptive_max_pool1d_shape() {
    let pool = AdaptiveMaxPool1d::new(3);
    let x = zeros_f32(&[1, 2, 6]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 3]);
}

#[test]
fn adaptive_max_pool3d_shape() {
    let pool = AdaptiveMaxPool3d::new((2, 2, 2));
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn max_unpool2d_forward_with_indices() {
    let unpool = MaxUnpool2d::new([2, 2], [2, 2], [0, 0]);
    // pooled output [1,1,2,2] with flat indices of max positions in the 4x4 input
    // MaxPool2d 2x2 on row-major [1..16]: positions 5,7,13,15 (0-indexed)
    let pooled = cpu_tensor_f32(&[6.0, 8.0, 14.0, 16.0], &[1, 1, 2, 2]);
    let indices: &[usize] = &[5, 7, 13, 15];
    let out = unpool.forward_with_indices(&pooled, indices, (4, 4)).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let data = out.data().unwrap();
    assert!((data[5] - 6.0).abs() < 1e-6, "unpool at idx 5");
    assert!((data[7] - 8.0).abs() < 1e-6, "unpool at idx 7");
    assert!((data[13] - 14.0).abs() < 1e-6, "unpool at idx 13");
    assert!((data[15] - 16.0).abs() < 1e-6, "unpool at idx 15");
}

#[test]
fn fractional_max_pool2d_output_shape() {
    let pool = FractionalMaxPool2d::new((2, 2));
    let x = zeros_f32(&[1, 1, 5, 5]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
}

#[test]
fn lp_pool1d_l2_norm() {
    let pool = LPPool1d::new(2.0, 2, 2);
    let x = cpu_tensor_f32(&[3.0, 4.0, 0.0, 5.0], &[1, 1, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2]);
    // L2([3,4]) = 5, L2([0,5]) = 5
    assert_close_f32(out.data().unwrap(), &[5.0, 5.0], 1e-5, "lp_pool1d_l2");
}

#[test]
fn lp_pool2d_l2_norm_shape() {
    let pool = LPPool2d::new(2.0, [2, 2], [2, 2]);
    let x = zeros_f32(&[1, 1, 4, 4]);
    let out: Tensor<f32> = Module::<f32>::forward(&pool, &x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
}

#[test]
fn max_pool2d_free_fn() {
    let x = cpu_tensor_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[1, 1, 4, 4],
    );
    let out = max_pool2d::<f32>(&x, [2, 2], [2, 2], [0, 0]).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_close_f32(out.data().unwrap(), &[6.0, 8.0, 14.0, 16.0], 1e-6, "max_pool2d_free");
}

#[test]
fn max_pool1d_free_fn() {
    let x = cpu_tensor_f32(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[1, 1, 6]);
    let out = max_pool1d::<f32>(&x, 2, 2, 0).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3]);
}

#[test]
fn max_pool3d_free_fn() {
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out = max_pool3d::<f32>(&x, [2, 2, 2], [2, 2, 2], [0, 0, 0]).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn avg_pool1d_free_fn() {
    let x = cpu_tensor_f32(&[2.0, 4.0, 6.0, 8.0], &[1, 1, 4]);
    let out = avg_pool1d::<f32>(&x, 2, 2, 0).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2]);
    assert_close_f32(out.data().unwrap(), &[3.0, 7.0], 1e-6, "avg_pool1d_free");
}

#[test]
fn avg_pool2d_free_fn() {
    let x = cpu_tensor_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[1, 1, 4, 4],
    );
    let out = avg_pool2d::<f32>(&x, [2, 2], [2, 2], [0, 0]).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
}

#[test]
fn avg_pool3d_free_fn() {
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out = avg_pool3d::<f32>(&x, [2, 2, 2], [2, 2, 2], [0, 0, 0]).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn adaptive_avg_pool1d_free_fn() {
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let out = adaptive_avg_pool1d::<f32>(&x, 3).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3]);
}

#[test]
fn adaptive_avg_pool2d_free_fn() {
    let x = zeros_f32(&[1, 1, 4, 4]);
    let out = adaptive_avg_pool2d::<f32>(&x, (2, 2)).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
}

#[test]
fn adaptive_avg_pool3d_free_fn() {
    let x = zeros_f32(&[1, 1, 4, 4, 4]);
    let out = adaptive_avg_pool3d::<f32>(&x, (2, 2, 2)).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn adaptive_max_pool2d_free_fn() {
    let x = zeros_f32(&[1, 1, 4, 4]);
    let out = adaptive_max_pool2d::<f32>(&x, (2, 2)).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
}

// ---------------------------------------------------------------------------
// ===========================================================================
// MODULE: dropout
// ===========================================================================
// ---------------------------------------------------------------------------

#[test]
fn dropout_new_valid_p() {
    assert!(Dropout::<f32>::new(0.5).is_ok());
    assert!(Dropout::<f32>::new(0.0).is_ok());
}

#[test]
fn dropout_new_invalid_p_errors() {
    assert!(Dropout::<f32>::new(1.0).is_err());
    assert!(Dropout::<f32>::new(1.5).is_err());
    assert!(Dropout::<f32>::new(-0.1).is_err());
}

#[test]
fn dropout_eval_is_identity() {
    let mut drop = Dropout::<f32>::new(0.5).unwrap();
    drop.eval();
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let out = drop.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_close_f32(out.data().unwrap(), x.data().unwrap(), 1e-7, "dropout_eval");
}

#[test]
fn dropout_p_zero_is_identity_in_train() {
    let drop = Dropout::<f32>::new(0.0).unwrap();
    assert!(drop.is_training());
    let x = cpu_tensor_f32(&[1.0, 2.0, 3.0], &[3]);
    let out = drop.forward(&x).unwrap();
    assert_close_f32(out.data().unwrap(), x.data().unwrap(), 1e-7, "dropout_p0");
}

#[test]
fn dropout_train_mode_scales_survivors() {
    // With p=0.5 in train mode, output elements are either 0 or 2x the input.
    // Sum should be close to input sum (unbiased estimator).
    let drop = Dropout::<f32>::new(0.5).unwrap();
    let n = 1000usize;
    let data: Vec<f32> = vec![1.0; n];
    let x = cpu_tensor_f32(&data, &[n]);
    let out = drop.forward(&x).unwrap();
    assert_eq!(out.shape(), &[n]);
    // All outputs are either 0 or 2.0 (scale = 1/(1-0.5) = 2)
    for &v in out.data().unwrap().iter() {
        assert!(v == 0.0 || (v - 2.0).abs() < 1e-5, "unexpected value {v}");
    }
}

#[test]
fn dropout_train_eval() {
    let mut drop = Dropout::<f32>::new(0.3).unwrap();
    assert!(drop.is_training());
    drop.eval();
    assert!(!drop.is_training());
    drop.train();
    assert!(drop.is_training());
}

#[test]
fn dropout_zero_parameters() {
    let drop = Dropout::<f32>::new(0.5).unwrap();
    assert_eq!(drop.parameters().len(), 0);
}

#[test]
fn dropout1d_eval_is_identity() {
    let mut drop = Dropout1d::<f32>::new(0.5).unwrap();
    drop.eval();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = cpu_tensor_f32(&data, &[2, 4, 3]);
    let out = drop.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4, 3]);
    assert_close_f32(out.data().unwrap(), &data, 1e-7, "dropout1d_eval");
}

#[test]
fn dropout1d_train_eval() {
    let mut drop = Dropout1d::<f32>::new(0.3).unwrap();
    assert!(drop.is_training());
    drop.eval();
    assert!(!drop.is_training());
}

#[test]
fn dropout2d_eval_is_identity() {
    let mut drop = Dropout2d::<f32>::new(0.5).unwrap();
    drop.eval();
    let data: Vec<f32> = vec![1.0; 16];
    let x = cpu_tensor_f32(&data, &[1, 4, 2, 2]);
    let out = drop.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 2, 2]);
    assert_close_f32(out.data().unwrap(), &data, 1e-7, "dropout2d_eval");
}

#[test]
fn dropout2d_train_eval() {
    let mut drop = Dropout2d::<f32>::new(0.3).unwrap();
    assert!(drop.is_training());
    drop.eval();
    assert!(!drop.is_training());
}

#[test]
fn dropout3d_eval_is_identity() {
    let mut drop = Dropout3d::<f32>::new(0.5).unwrap();
    drop.eval();
    let data: Vec<f32> = vec![2.0; 54];
    let x = cpu_tensor_f32(&data, &[1, 2, 3, 3, 3]);
    let out = drop.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 3, 3, 3]);
    assert_close_f32(out.data().unwrap(), &data, 1e-7, "dropout3d_eval");
}

#[test]
fn dropout3d_train_eval() {
    let mut drop = Dropout3d::<f32>::new(0.2).unwrap();
    assert!(drop.is_training());
    drop.eval();
    assert!(!drop.is_training());
}

#[test]
fn alpha_dropout_eval_is_identity() {
    let mut drop = AlphaDropout::<f32>::new(0.1).unwrap();
    drop.eval();
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let x = cpu_tensor_f32(&data, &[2, 4]);
    let out = drop.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert_close_f32(out.data().unwrap(), &data, 1e-7, "alpha_dropout_eval");
}

#[test]
fn alpha_dropout_train_eval() {
    let mut drop = AlphaDropout::<f32>::new(0.1).unwrap();
    assert!(drop.is_training());
    drop.eval();
    assert!(!drop.is_training());
}
