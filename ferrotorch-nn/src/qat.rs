//! Quantization-aware training (QAT) for ferrotorch nn modules.
//!
//! This module re-exports the core QAT types from `ferrotorch_core::quantize`
//! and provides nn-module-level integration for preparing models.

use std::collections::HashMap;

pub use ferrotorch_core::quantize::{QatLayer, QatModel, prepare_qat as core_prepare_qat};
use ferrotorch_core::{
    FerrotorchError, FerrotorchResult, QParams, QuantDtype, QuantizedTensor, Tensor, dequantize,
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
}

impl QatConfig {
    /// Symmetric per-tensor INT8 for both weights and activations.
    pub fn default_symmetric_int8() -> Self {
        Self {
            weight_dtype: QuantDtype::Int8,
            activation_dtype: QuantDtype::Int8,
            weight_symmetric: true,
            activation_symmetric: true,
            weight_observer: ObserverType::MinMax,
            activation_observer: ObserverType::MovingAverageMinMax,
        }
    }

    /// Per-channel INT8 for weights, per-tensor INT8 for activations.
    pub fn per_channel_int8() -> Self {
        Self {
            weight_dtype: QuantDtype::Int8,
            activation_dtype: QuantDtype::Int8,
            weight_symmetric: true,
            activation_symmetric: true,
            weight_observer: ObserverType::PerChannelMinMax,
            activation_observer: ObserverType::MovingAverageMinMax,
        }
    }

    /// INT4 weights with INT8 activations.
    pub fn int4_weight_int8_activation() -> Self {
        Self {
            weight_dtype: QuantDtype::Int4,
            activation_dtype: QuantDtype::Int8,
            weight_symmetric: true,
            activation_symmetric: true,
            weight_observer: ObserverType::MinMax,
            activation_observer: ObserverType::MovingAverageMinMax,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedModel — deployment-ready quantized weights
// ---------------------------------------------------------------------------

/// A fully quantized model ready for deployment.
///
/// Contains integer-stored weights with their quantization parameters.
pub struct QuantizedModel {
    /// Quantized weight tensors, keyed by parameter name.
    weights: HashMap<String, QuantizedTensor>,
    /// Quantization parameters for each weight.
    weight_qparams: HashMap<String, QParams>,
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
        let qt = self
            .weights
            .get(name)
            .ok_or(FerrotorchError::InvalidArgument {
                message: format!("quantized weight \"{name}\" not found"),
            })?;
        dequantize(qt)
    }

    /// Compute total quantized model size in bytes.
    pub fn quantized_size_bytes(&self) -> usize {
        self.weights.values().map(|qt| qt.numel()).sum()
    }

    /// Compute compression ratio vs float32.
    pub fn compression_ratio(&self) -> f32 {
        let quantized_bytes = self.quantized_size_bytes();
        let float_bytes: usize = self.weights.values().map(|qt| qt.numel() * 4).sum();
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
// prepare_qat — nn-level integration
// ---------------------------------------------------------------------------

/// Prepare a module for quantization-aware training.
///
/// Scans the module's named parameters and creates a `QatModel` with
/// FakeQuantize nodes for each parametric layer. Only parameters whose
/// name contains "weight" get weight FakeQuantize; bias parameters are
/// skipped.
pub fn prepare_qat(module: &dyn Module<f32>, config: QatConfig) -> QatModel {
    let named = module.named_parameters();
    let param_names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    core_prepare_qat(&param_names, config.weight_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_config_presets() {
        let c1 = QatConfig::default_symmetric_int8();
        assert_eq!(c1.weight_dtype, QuantDtype::Int8);
        assert!(c1.weight_symmetric);

        let c2 = QatConfig::per_channel_int8();
        assert_eq!(c2.weight_observer, ObserverType::PerChannelMinMax);

        let c3 = QatConfig::int4_weight_int8_activation();
        assert_eq!(c3.weight_dtype, QuantDtype::Int4);
        assert_eq!(c3.activation_dtype, QuantDtype::Int8);
    }

    #[test]
    fn test_quantized_model_empty() {
        let qm = QuantizedModel {
            weights: HashMap::new(),
            weight_qparams: HashMap::new(),
            config: QatConfig::default_symmetric_int8(),
        };
        assert_eq!(qm.num_weights(), 0);
        assert_eq!(qm.compression_ratio(), 1.0);
    }
}
