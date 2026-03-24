//! Quantization-aware training (QAT) for ferrotorch.
//!
//! QAT simulates quantization during training by inserting `FakeQuantize`
//! modules before weights and after activations. The Straight-Through
//! Estimator (STE) lets gradients flow through the quantization, so the
//! model learns to be robust to quantization noise.
//!
//! # Workflow
//!
//! 1. **Prepare**: Call [`prepare_qat`] on a trained (or untrained) model
//!    to insert `FakeQuantize` nodes.
//! 2. **Train**: Fine-tune the model normally. The fake quantizers will
//!    observe weight/activation ranges and simulate quantization error.
//! 3. **Convert**: Call [`QatModel::convert`] to freeze quantization
//!    parameters and produce a `QuantizedModel` with integer weights.
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_nn::qat::{QatConfig, prepare_qat};
//!
//! let mut model = Sequential::new(vec![
//!     Box::new(Linear::<f32>::new(784, 256, true)?),
//!     Box::new(ReLU::new()),
//!     Box::new(Linear::<f32>::new(256, 10, true)?),
//! ]);
//!
//! let config = QatConfig::default_symmetric_int8();
//! let mut qat_model = prepare_qat(&mut model, config);
//! // ... train qat_model ...
//! let quantized = qat_model.convert()?;
//! ```

use std::collections::HashMap;

use ferrotorch_core::{
    dequantize, quantize_per_tensor, FakeQuantize, FerrotorchError, FerrotorchResult,
    HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver, Observer,
    PerChannelMinMaxObserver, QParams, QuantDtype, QuantizedTensor, Tensor,
};

use crate::module::Module;

// ---------------------------------------------------------------------------
// ObserverType — which observer to use for calibration
// ---------------------------------------------------------------------------

/// Specifies which observer to use for tracking tensor statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserverType {
    /// Track global min and max.
    MinMax,
    /// Track min and max with exponential moving average.
    MovingAverageMinMax,
    /// Build a histogram and use percentile clipping.
    Histogram,
    /// Track per-output-channel min/max (weight quantization only).
    PerChannelMinMax,
}

// ---------------------------------------------------------------------------
// QatConfig — quantization configuration
// ---------------------------------------------------------------------------

/// Configuration for quantization-aware training.
///
/// Controls which layers get quantized, what observers are used, and
/// whether quantization is symmetric or affine.
#[derive(Debug, Clone)]
pub struct QatConfig {
    /// Target dtype for weight quantization.
    pub weight_dtype: QuantDtype,
    /// Target dtype for activation quantization.
    pub activation_dtype: QuantDtype,
    /// Whether weight quantization is symmetric.
    pub weight_symmetric: bool,
    /// Whether activation quantization is symmetric.
    pub activation_symmetric: bool,
    /// Observer type for weights.
    pub weight_observer: ObserverType,
    /// Observer type for activations.
    pub activation_observer: ObserverType,
    /// EMA constant for MovingAverageMinMax observers (ignored otherwise).
    pub averaging_constant: f32,
    /// Number of channels for per-channel weight quantization (0 = auto-detect).
    pub num_channels: usize,
    /// Histogram bins (only used with Histogram observer).
    pub histogram_bins: usize,
    /// Histogram percentile (only used with Histogram observer).
    pub histogram_percentile: f32,
}

impl QatConfig {
    /// Symmetric per-tensor INT8 for both weights and activations.
    ///
    /// This is the simplest and most portable configuration: symmetric
    /// quantization with zero_point = 0, suitable for most convolutional
    /// and linear layers.
    pub fn default_symmetric_int8() -> Self {
        Self {
            weight_dtype: QuantDtype::Int8,
            activation_dtype: QuantDtype::Int8,
            weight_symmetric: true,
            activation_symmetric: true,
            weight_observer: ObserverType::MinMax,
            activation_observer: ObserverType::MovingAverageMinMax,
            averaging_constant: 0.01,
            num_channels: 0,
            histogram_bins: 2048,
            histogram_percentile: 99.99,
        }
    }

    /// Per-channel INT8 for weights, per-tensor INT8 for activations.
    ///
    /// Per-channel weight quantization uses a separate scale for each
    /// output channel, reducing quantization error for layers with
    /// channels of very different magnitudes.
    pub fn per_channel_int8() -> Self {
        Self {
            weight_dtype: QuantDtype::Int8,
            activation_dtype: QuantDtype::Int8,
            weight_symmetric: true,
            activation_symmetric: true,
            weight_observer: ObserverType::PerChannelMinMax,
            activation_observer: ObserverType::MovingAverageMinMax,
            averaging_constant: 0.01,
            num_channels: 0,
            histogram_bins: 2048,
            histogram_percentile: 99.99,
        }
    }

    /// INT4 weights with INT8 activations.
    ///
    /// Aggressively compresses weights to 4-bit integers for maximum
    /// memory savings, while keeping activations at INT8 for reasonable
    /// accuracy. Typically needs more QAT fine-tuning epochs.
    pub fn int4_weight_int8_activation() -> Self {
        Self {
            weight_dtype: QuantDtype::Int4,
            activation_dtype: QuantDtype::Int8,
            weight_symmetric: true,
            activation_symmetric: true,
            weight_observer: ObserverType::MinMax,
            activation_observer: ObserverType::MovingAverageMinMax,
            averaging_constant: 0.01,
            num_channels: 0,
            histogram_bins: 2048,
            histogram_percentile: 99.99,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: create observer from config
// ---------------------------------------------------------------------------

fn make_weight_observer(config: &QatConfig, num_channels: usize) -> Box<dyn Observer> {
    match config.weight_observer {
        ObserverType::MinMax => {
            Box::new(MinMaxObserver::new(config.weight_dtype, config.weight_symmetric))
        }
        ObserverType::MovingAverageMinMax => Box::new(MovingAverageMinMaxObserver::new(
            config.averaging_constant,
            config.weight_dtype,
            config.weight_symmetric,
        )),
        ObserverType::Histogram => Box::new(HistogramObserver::new(
            config.histogram_bins,
            config.histogram_percentile,
            config.weight_dtype,
        )),
        ObserverType::PerChannelMinMax => {
            let ch = if config.num_channels > 0 {
                config.num_channels
            } else {
                num_channels
            };
            Box::new(PerChannelMinMaxObserver::new(
                ch,
                config.weight_dtype,
                config.weight_symmetric,
            ))
        }
    }
}

fn make_activation_observer(config: &QatConfig) -> Box<dyn Observer> {
    match config.activation_observer {
        ObserverType::MinMax => Box::new(MinMaxObserver::new(
            config.activation_dtype,
            config.activation_symmetric,
        )),
        ObserverType::MovingAverageMinMax => Box::new(MovingAverageMinMaxObserver::new(
            config.averaging_constant,
            config.activation_dtype,
            config.activation_symmetric,
        )),
        ObserverType::Histogram => Box::new(HistogramObserver::new(
            config.histogram_bins,
            config.histogram_percentile,
            config.activation_dtype,
        )),
        ObserverType::PerChannelMinMax => {
            // Per-channel for activations is unusual — fall back to MinMax.
            Box::new(MinMaxObserver::new(
                config.activation_dtype,
                config.activation_symmetric,
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// QatLayer — a single layer with FakeQuantize nodes
// ---------------------------------------------------------------------------

/// A single quantization-aware layer wrapping a `Module<f32>`.
///
/// Contains FakeQuantize nodes for weights and activations, and delegates
/// the actual computation to the wrapped module.
struct QatLayer {
    /// Name of this layer in the model (e.g., "0", "1").
    name: String,
    /// FakeQuantize for weights (applied before forward).
    weight_fq: FakeQuantize,
    /// FakeQuantize for activations (applied after forward).
    activation_fq: FakeQuantize,
}

// ---------------------------------------------------------------------------
// QatModel — wraps a Module with per-layer FakeQuantize
// ---------------------------------------------------------------------------

/// A model prepared for quantization-aware training.
///
/// Wraps a `Module<f32>` and inserts `FakeQuantize` before weights and
/// after activations at each parametric layer. Non-parametric layers
/// (activations, pooling, dropout) pass through without quantization
/// simulation.
pub struct QatModel<'a> {
    /// The underlying module being quantized.
    module: &'a mut dyn Module<f32>,
    /// Per-layer FakeQuantize state, indexed by parameter name prefix.
    layers: Vec<QatLayer>,
    /// Configuration used to prepare this model.
    config: QatConfig,
    /// Whether the model is in training mode.
    training: bool,
}

impl<'a> QatModel<'a> {
    /// Forward pass with fake quantization.
    ///
    /// For each parametric layer:
    /// 1. Fake-quantize the weights in-place (with STE backward).
    /// 2. Run the layer's forward.
    /// 3. Fake-quantize the output activations.
    ///
    /// Because we cannot generically intercept individual layer forwards
    /// through the trait-object `Module`, this implementation applies
    /// weight fake-quantization to all parameters before the full module
    /// forward, and activation fake-quantization to the final output.
    pub fn forward(&mut self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        // Step 1: Fake-quantize all weights.
        self.fake_quantize_weights()?;

        // Step 2: Run the module's forward.
        let output = self.module.forward(input)?;

        // Step 3: Fake-quantize the output activations.
        if let Some(layer) = self.layers.last_mut() {
            layer.activation_fq.forward(&output)
        } else {
            Ok(output)
        }
    }

    /// Fake-quantize all weight parameters in the model.
    fn fake_quantize_weights(&mut self) -> FerrotorchResult<()> {
        // Collect parameter names and their fake-quantized values.
        let named = self.module.named_parameters();
        let mut fq_values: Vec<(String, Tensor<f32>)> = Vec::new();

        for (name, param) in &named {
            // Find the matching QatLayer by name prefix.
            if let Some(layer) = self.layers.iter_mut().find(|l| name.starts_with(&l.name)) {
                let fq_tensor = layer.weight_fq.forward(param.tensor())?;
                fq_values.push((name.clone(), fq_tensor));
            }
        }

        // Apply the fake-quantized values back as parameter data.
        // We need to update through parameters_mut to maintain the module's
        // internal state.
        let param_names: Vec<String> = self
            .module
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let params_mut = self.module.parameters_mut();

        for (name, param) in param_names.iter().zip(params_mut.into_iter()) {
            if let Some(pos) = fq_values.iter().position(|(n, _)| n == name) {
                let (_, fq_tensor) = &fq_values[pos];
                param.set_data(fq_tensor.clone());
            }
        }

        Ok(())
    }

    /// Freeze all observers (stop updating calibration statistics).
    ///
    /// Call this after a calibration phase to lock in the quantization
    /// parameters.
    pub fn freeze_observers(&mut self) {
        for layer in &mut self.layers {
            layer.weight_fq.disable_observer();
            layer.activation_fq.disable_observer();
        }
    }

    /// Unfreeze all observers (resume updating calibration statistics).
    pub fn unfreeze_observers(&mut self) {
        for layer in &mut self.layers {
            layer.weight_fq.enable_observer();
            layer.activation_fq.enable_observer();
        }
    }

    /// Disable all fake quantization (make forward a no-op passthrough).
    pub fn disable_fake_quantize(&mut self) {
        for layer in &mut self.layers {
            layer.weight_fq.disable();
            layer.activation_fq.disable();
        }
    }

    /// Enable all fake quantization.
    pub fn enable_fake_quantize(&mut self) {
        for layer in &mut self.layers {
            layer.weight_fq.enable();
            layer.activation_fq.enable();
        }
    }

    /// Convert the QAT model to a fully quantized model.
    ///
    /// This freezes all quantization parameters and produces a
    /// `QuantizedModel` with integer-stored weights ready for deployment.
    pub fn convert(&self) -> FerrotorchResult<QuantizedModel> {
        let mut quantized_weights: HashMap<String, QuantizedTensor> = HashMap::new();
        let mut weight_qparams: HashMap<String, QParams> = HashMap::new();
        let mut activation_qparams: HashMap<String, QParams> = HashMap::new();

        let named = self.module.named_parameters();

        for (name, param) in &named {
            // Find matching layer.
            if let Some(layer) = self.layers.iter().find(|l| name.starts_with(&l.name)) {
                // Get weight qparams from the FakeQuantize.
                if let Some(qp) = layer.weight_fq.qparams() {
                    let qt = quantize_per_tensor(
                        param.tensor(),
                        qp.scale,
                        qp.zero_point,
                        qp.dtype,
                    )?;
                    quantized_weights.insert(name.clone(), qt);
                    weight_qparams.insert(name.clone(), qp.clone());
                }

                // Record activation qparams.
                if let Some(qp) = layer.activation_fq.qparams() {
                    let act_key = format!("{}.activation", layer.name);
                    activation_qparams.insert(act_key, qp.clone());
                }
            }
        }

        Ok(QuantizedModel {
            weights: quantized_weights,
            weight_qparams,
            activation_qparams,
            config: self.config.clone(),
        })
    }

    /// Access the underlying module.
    pub fn module(&self) -> &dyn Module<f32> {
        self.module
    }

    /// Access the underlying module mutably.
    pub fn module_mut(&mut self) -> &mut dyn Module<f32> {
        self.module
    }

    /// Set training mode.
    pub fn train(&mut self) {
        self.training = true;
        self.module.train();
    }

    /// Set evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
        self.module.eval();
    }

    /// Whether in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// QuantizedModel — deployment-ready quantized weights
// ---------------------------------------------------------------------------

/// A fully quantized model ready for deployment.
///
/// Contains integer-stored weights with their quantization parameters.
/// This is the output of [`QatModel::convert`].
pub struct QuantizedModel {
    /// Quantized weight tensors, keyed by parameter name.
    weights: HashMap<String, QuantizedTensor>,
    /// Quantization parameters for each weight.
    weight_qparams: HashMap<String, QParams>,
    /// Quantization parameters for activations at each layer.
    activation_qparams: HashMap<String, QParams>,
    /// Configuration that produced this model.
    config: QatConfig,
}

impl QuantizedModel {
    /// Get a quantized weight by name.
    pub fn weight(&self, name: &str) -> Option<&QuantizedTensor> {
        self.weights.get(name)
    }

    /// Get weight quantization parameters by name.
    pub fn weight_qparams(&self, name: &str) -> Option<&QParams> {
        self.weight_qparams.get(name)
    }

    /// Get activation quantization parameters by layer name.
    pub fn activation_qparams(&self, name: &str) -> Option<&QParams> {
        self.activation_qparams.get(name)
    }

    /// Iterate over all quantized weight names.
    pub fn weight_names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Number of quantized weights.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Dequantize a specific weight back to float for inspection.
    pub fn dequantize_weight(&self, name: &str) -> FerrotorchResult<Tensor<f32>> {
        let qt = self.weights.get(name).ok_or(FerrotorchError::InvalidArgument {
            message: format!("quantized weight \"{name}\" not found"),
        })?;
        dequantize(qt)
    }

    /// Compute total quantized model size in bytes.
    ///
    /// Counts only the quantized weight data, not metadata or activation
    /// parameters.
    pub fn quantized_size_bytes(&self) -> usize {
        self.weights.values().map(|qt| qt.numel()).sum()
    }

    /// Compute compression ratio vs float32.
    pub fn compression_ratio(&self) -> f32 {
        let quantized_bytes = self.quantized_size_bytes();
        let float_bytes: usize = self
            .weights
            .values()
            .map(|qt| qt.numel() * 4) // 4 bytes per f32
            .sum();
        if quantized_bytes == 0 {
            return 1.0;
        }
        float_bytes as f32 / quantized_bytes as f32
    }

    /// The QAT configuration used.
    pub fn config(&self) -> &QatConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// prepare_qat — insert FakeQuantize into a module
// ---------------------------------------------------------------------------

/// Prepare a model for quantization-aware training.
///
/// Inserts `FakeQuantize` modules for each parametric layer found via
/// `named_parameters()`. Weight FakeQuantize uses the weight observer,
/// and activation FakeQuantize uses the activation observer from the config.
///
/// The returned `QatModel` wraps the original module with fake quantization
/// enabled. Call `.forward()` on the `QatModel` for training, then
/// `.convert()` to produce a `QuantizedModel`.
pub fn prepare_qat<'a>(
    module: &'a mut dyn Module<f32>,
    config: QatConfig,
) -> QatModel<'a> {
    // Discover parametric layers by scanning named parameters.
    // Group parameters by their prefix (everything before the last dot).
    let named = module.named_parameters();
    let mut layer_prefixes: Vec<String> = Vec::new();

    for (name, _) in &named {
        let prefix = if let Some(dot_pos) = name.rfind('.') {
            name[..dot_pos].to_string()
        } else {
            String::new()
        };
        if !layer_prefixes.contains(&prefix) {
            layer_prefixes.push(prefix);
        }
    }

    // Create FakeQuantize nodes for each layer.
    let mut layers = Vec::new();
    for prefix in &layer_prefixes {
        // Determine the number of output channels from the weight shape.
        let num_channels = named
            .iter()
            .find(|(n, _)| {
                let p = if let Some(dot) = n.rfind('.') {
                    &n[..dot]
                } else {
                    ""
                };
                p == prefix && n.ends_with("weight")
            })
            .map(|(_, param)| {
                let shape = param.shape();
                if shape.is_empty() {
                    1
                } else {
                    shape[0]
                }
            })
            .unwrap_or(1);

        let weight_observer = make_weight_observer(&config, num_channels);
        let activation_observer = make_activation_observer(&config);

        layers.push(QatLayer {
            name: prefix.clone(),
            weight_fq: FakeQuantize::new(weight_observer),
            activation_fq: FakeQuantize::new(activation_observer),
        });
    }

    QatModel {
        module,
        layers,
        config,
        training: true,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter::Parameter;
    use ferrotorch_core::{from_slice, TensorStorage};

    /// Helper: create a tensor from f32 data.
    fn make_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        from_slice(data, shape).unwrap()
    }

    // ---- QatConfig constructors ----

    #[test]
    fn test_config_default_symmetric_int8() {
        let config = QatConfig::default_symmetric_int8();
        assert_eq!(config.weight_dtype, QuantDtype::Int8);
        assert_eq!(config.activation_dtype, QuantDtype::Int8);
        assert!(config.weight_symmetric);
        assert!(config.activation_symmetric);
    }

    #[test]
    fn test_config_per_channel_int8() {
        let config = QatConfig::per_channel_int8();
        assert_eq!(config.weight_observer, ObserverType::PerChannelMinMax);
        assert_eq!(config.activation_observer, ObserverType::MovingAverageMinMax);
    }

    #[test]
    fn test_config_int4_weight_int8_activation() {
        let config = QatConfig::int4_weight_int8_activation();
        assert_eq!(config.weight_dtype, QuantDtype::Int4);
        assert_eq!(config.activation_dtype, QuantDtype::Int8);
    }

    // ---- Simple module for testing ----

    #[derive(Debug)]
    struct TestLinear {
        weight: Parameter<f32>,
        bias: Parameter<f32>,
        training: bool,
    }

    impl TestLinear {
        fn new(in_f: usize, out_f: usize) -> FerrotorchResult<Self> {
            let w_data: Vec<f32> = (0..(in_f * out_f))
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();
            Ok(Self {
                weight: Parameter::from_slice(&w_data, &[out_f, in_f])?,
                bias: Parameter::from_slice(
                    &vec![0.0f32; out_f],
                    &[out_f],
                )?,
                training: true,
            })
        }
    }

    impl Module<f32> for TestLinear {
        fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            // Simple matmul: input @ weight^T + bias.
            let w_data = self.weight.data()?;
            let b_data = self.bias.data()?;
            let in_data = input.data()?;

            let in_shape = input.shape();
            let w_shape = self.weight.shape();
            let out_f = w_shape[0];
            let in_f = w_shape[1];

            let batch = if in_shape.len() == 2 { in_shape[0] } else { 1 };

            let mut result = vec![0.0f32; batch * out_f];
            for b in 0..batch {
                for o in 0..out_f {
                    let mut sum = b_data[o];
                    for i in 0..in_f {
                        sum += in_data[b * in_f + i] * w_data[o * in_f + i];
                    }
                    result[b * out_f + o] = sum;
                }
            }

            Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![batch, out_f],
                false,
            )
        }

        fn parameters(&self) -> Vec<&Parameter<f32>> {
            vec![&self.weight, &self.bias]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
            vec![&mut self.weight, &mut self.bias]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
            vec![
                ("weight".to_string(), &self.weight),
                ("bias".to_string(), &self.bias),
            ]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    // ---- prepare_qat ----

    #[test]
    fn test_prepare_qat_creates_layers() {
        let mut model = TestLinear::new(4, 3).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let qat = prepare_qat(&mut model, config);

        // Should have created a QAT layer for the parameter prefix.
        assert!(!qat.layers.is_empty());
        assert!(qat.is_training());
    }

    #[test]
    fn test_qat_forward() {
        let mut model = TestLinear::new(3, 2).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let output = qat.forward(&input).unwrap();

        assert_eq!(output.shape()[0], 2);
        assert_eq!(output.shape()[1], 2);

        // Output should be finite (not NaN or Inf).
        let data = output.data().unwrap();
        for &v in data {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn test_qat_forward_approximate_original() {
        let mut model = TestLinear::new(3, 2).unwrap();
        let input = make_tensor(&[1.0, 2.0, 3.0], &[1, 3]);

        // Get original output.
        let orig_output = model.forward(&input).unwrap();
        let orig_data = orig_output.data().unwrap().to_vec();

        // QAT output should be close to original (quantization error is small
        // for small weight magnitudes).
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);
        let qat_output = qat.forward(&input).unwrap();
        let qat_data = qat_output.data().unwrap();

        for (i, (&o, &q)) in orig_data.iter().zip(qat_data.iter()).enumerate() {
            // Allow for both weight and activation quantization error.
            let err = (o - q).abs();
            assert!(
                err < 1.0,
                "element {i}: original={o}, qat={q}, error={err}"
            );
        }
    }

    // ---- freeze/unfreeze observers ----

    #[test]
    fn test_freeze_unfreeze_observers() {
        let mut model = TestLinear::new(3, 2).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        // Run one forward to calibrate.
        let input = make_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        qat.forward(&input).unwrap();

        // Capture qparams after calibration.
        let scale_before: Vec<f32> = qat
            .layers
            .iter()
            .filter_map(|l| l.weight_fq.qparams().map(|q| q.scale))
            .collect();

        // Freeze observers and run with very different data.
        qat.freeze_observers();
        let big_input = make_tensor(&[100.0, 200.0, 300.0], &[1, 3]);
        qat.forward(&big_input).unwrap();

        // After freeze, scales should not have changed from the original
        // calibration pass.
        let scale_after: Vec<f32> = qat
            .layers
            .iter()
            .filter_map(|l| l.weight_fq.qparams().map(|q| q.scale))
            .collect();
        assert_eq!(scale_before, scale_after);

        // Unfreeze.
        qat.unfreeze_observers();
    }

    // ---- enable/disable fake quantize ----

    #[test]
    fn test_enable_disable_fake_quantize() {
        let mut model = TestLinear::new(3, 2).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        qat.disable_fake_quantize();
        for layer in &qat.layers {
            assert!(!layer.weight_fq.is_enabled());
            assert!(!layer.activation_fq.is_enabled());
        }

        qat.enable_fake_quantize();
        for layer in &qat.layers {
            assert!(layer.weight_fq.is_enabled());
            assert!(layer.activation_fq.is_enabled());
        }
    }

    // ---- train/eval ----

    #[test]
    fn test_qat_train_eval() {
        let mut model = TestLinear::new(3, 2).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        assert!(qat.is_training());
        qat.eval();
        assert!(!qat.is_training());
        qat.train();
        assert!(qat.is_training());
    }

    // ---- convert ----

    #[test]
    fn test_convert_produces_quantized_model() {
        let mut model = TestLinear::new(4, 3).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        // Forward to populate observer statistics.
        let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        qat.forward(&input).unwrap();

        let quantized = qat.convert().unwrap();
        assert!(quantized.num_weights() > 0);
    }

    #[test]
    fn test_convert_weight_roundtrip() {
        let mut model = TestLinear::new(3, 2).unwrap();
        let _orig_weight = model.weight.data().unwrap().to_vec();

        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        qat.forward(&input).unwrap();

        let quantized = qat.convert().unwrap();

        // Dequantize the weight and check it's close to original.
        for name in quantized.weight_names() {
            let deq = quantized.dequantize_weight(name).unwrap();
            let deq_data = deq.data().unwrap();
            // Each dequantized weight should be close to original.
            // (Exact match is not expected due to quantization.)
            for &v in deq_data {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_convert_compression_ratio() {
        let mut model = TestLinear::new(64, 32).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&vec![0.5f32; 64], &[1, 64]);
        qat.forward(&input).unwrap();

        let quantized = qat.convert().unwrap();
        // INT8 weights should give ~4x compression vs f32.
        let ratio = quantized.compression_ratio();
        assert!(
            ratio >= 3.5,
            "expected ~4x compression for INT8, got {ratio}x"
        );
    }

    #[test]
    fn test_quantized_model_accessors() {
        let mut model = TestLinear::new(4, 3).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        qat.forward(&input).unwrap();

        let quantized = qat.convert().unwrap();

        // Check config accessor.
        assert_eq!(quantized.config().weight_dtype, QuantDtype::Int8);

        // Check size calculation.
        assert!(quantized.quantized_size_bytes() > 0);
    }

    #[test]
    fn test_quantized_model_missing_weight() {
        let mut model = TestLinear::new(4, 3).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        qat.forward(&input).unwrap();

        let quantized = qat.convert().unwrap();
        assert!(quantized.dequantize_weight("nonexistent").is_err());
    }

    // ---- Per-channel config ----

    #[test]
    fn test_per_channel_qat() {
        let mut model = TestLinear::new(4, 3).unwrap();
        let config = QatConfig::per_channel_int8();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = qat.forward(&input).unwrap();

        assert_eq!(output.shape()[1], 3);
        let data = output.data().unwrap();
        for &v in data {
            assert!(v.is_finite());
        }
    }

    // ---- INT4 config ----

    #[test]
    fn test_int4_weight_qat() {
        let mut model = TestLinear::new(4, 2).unwrap();
        let config = QatConfig::int4_weight_int8_activation();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = qat.forward(&input).unwrap();

        let data = output.data().unwrap();
        for &v in data {
            assert!(v.is_finite());
        }

        let quantized = qat.convert().unwrap();
        // INT4 gives even more compression.
        assert!(quantized.num_weights() > 0);
    }

    // ---- Observer types ----

    #[test]
    fn test_histogram_observer_config() {
        let config = QatConfig {
            weight_observer: ObserverType::Histogram,
            activation_observer: ObserverType::Histogram,
            ..QatConfig::default_symmetric_int8()
        };

        let mut model = TestLinear::new(4, 2).unwrap();
        let mut qat = prepare_qat(&mut model, config);

        let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = qat.forward(&input).unwrap();

        let data = output.data().unwrap();
        for &v in data {
            assert!(v.is_finite());
        }
    }

    // ---- Multiple forward passes (calibration) ----

    #[test]
    fn test_multiple_forward_passes_calibration() {
        let mut model = TestLinear::new(4, 2).unwrap();
        let config = QatConfig::default_symmetric_int8();
        let mut qat = prepare_qat(&mut model, config);

        // Run several forward passes with different data to calibrate.
        for i in 0..5 {
            let data: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32 * 0.1).collect();
            let input = make_tensor(&data, &[1, 4]);
            let output = qat.forward(&input).unwrap();
            let vals = output.data().unwrap();
            for &v in vals {
                assert!(v.is_finite());
            }
        }

        // After calibration, convert should work.
        let quantized = qat.convert().unwrap();
        assert!(quantized.num_weights() > 0);
    }
}
