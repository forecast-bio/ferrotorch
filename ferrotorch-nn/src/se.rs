//! Squeeze-and-Excitation (SE) block — Hu et al. 2018, *Squeeze-and-Excitation
//! Networks*.
//!
//! Mirrors `torchvision.ops.misc.SqueezeExcitation`, including the
//! `named_children()` order and the use of 1×1 [`Conv2d`] (NOT [`Linear`])
//! for both the squeeze and excitation projections. Per-channel attention
//! via global average pool → 1×1 conv → activation → 1×1 conv →
//! scale_activation → broadcast multiply.
//!
//! ```text
//! x: [B, C, H, W]
//!     │
//!     ├─────────────────────────────────────┐
//!     ▼                                     │
//! avgpool([B,C,1,1]) → fc1([B,sq,1,1])     │
//!     → activation                          │
//!     → fc2([B,C,1,1])                      │
//!     → scale_activation                    │
//!     │                                     │
//!     └────────────── (broadcast *) ────────┘
//!                     ▼
//!                  [B, C, H, W]
//! ```
//!
//! Used by MobileNetV3 (with [`HardSigmoid`](crate::activation::HardSigmoid)
//! as `scale_activation`) and EfficientNet (with [`Sigmoid`] as
//! `scale_activation`). The default [`SqueezeExcitation::new`] constructor
//! ([`ReLU`] + [`Sigmoid`]) matches `torchvision.ops.SqueezeExcitation`'s
//! default. For mixed-precision or alternative activations, use
//! [`SqueezeExcitation::new_with_activations`].
//!
//! # Module trait surface
//!
//! - [`named_parameters`](Module::named_parameters): `fc1.weight`,
//!   `fc1.bias`, `fc2.weight`, `fc2.bias` — exactly the four keys
//!   produced by torchvision's `SqueezeExcitation`.
//! - [`named_children`](Module::named_children): `avgpool`, `fc1`,
//!   `activation`, `fc2`, `scale_activation` — same order torchvision
//!   exposes through `Sequential`-style submodule naming.
//!
//! # Differentiability
//!
//! Forward composes only [`Module::forward`]-shaped primitives that
//! already track gradients (`Conv2d`, `AdaptiveAvgPool2d`, `ReLU`,
//! `Sigmoid`, `HardSigmoid`, `mul`), so backward flows end-to-end.

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::activation::{ReLU, Sigmoid};
use crate::conv::Conv2d;
use crate::module::Module;
use crate::parameter::Parameter;
use crate::pooling::AdaptiveAvgPool2d;

/// Squeeze-and-Excitation block.
///
/// Channel-wise attention: pool → squeeze → excite → scale.
/// See module docs for the full diagram. The default constructor
/// ([`Self::new`]) uses ReLU as the inner activation and Sigmoid
/// as the scale activation, matching torchvision's
/// `SqueezeExcitation` default; [`Self::new_with_activations`] lets
/// callers swap either (e.g. SiLU + HardSigmoid for MobileNetV3).
pub struct SqueezeExcitation<T: Float> {
    /// Global-average-pool to `[B, C, 1, 1]`.
    avgpool: AdaptiveAvgPool2d,
    /// 1×1 convolution: `[B, C, 1, 1] → [B, sq, 1, 1]`.
    fc1: Conv2d<T>,
    /// Inner activation between fc1 and fc2 (default ReLU).
    activation: Box<dyn Module<T>>,
    /// 1×1 convolution: `[B, sq, 1, 1] → [B, C, 1, 1]`.
    fc2: Conv2d<T>,
    /// Output activation that gates the input (default Sigmoid).
    scale_activation: Box<dyn Module<T>>,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> std::fmt::Debug for SqueezeExcitation<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqueezeExcitation")
            .field("fc1", &self.fc1)
            .field("fc2", &self.fc2)
            .field("training", &self.training)
            .finish()
    }
}

impl<T: Float> SqueezeExcitation<T> {
    /// Create a new SE block with the default activations
    /// (ReLU squeeze, Sigmoid scale).
    ///
    /// `input_channels` is `C` in the diagram above; `squeeze_channels`
    /// is the bottleneck width (typically `C / r` for a reduction ratio
    /// `r`).
    ///
    /// # Errors
    /// Returns [`FerrotorchError::InvalidArgument`] if either channel
    /// count is zero.
    pub fn new(input_channels: usize, squeeze_channels: usize) -> FerrotorchResult<Self> {
        Self::new_with_activations(
            input_channels,
            squeeze_channels,
            Box::new(ReLU::new()),
            Box::new(Sigmoid::new()),
        )
    }

    /// Create a new SE block with caller-supplied activations.
    ///
    /// MobileNetV3-Small uses ReLU + HardSigmoid; EfficientNet uses
    /// SiLU + Sigmoid. Both are constructible from this entry point.
    ///
    /// # Errors
    /// Returns [`FerrotorchError::InvalidArgument`] if either channel
    /// count is zero. Errors from [`Conv2d::new`] (negative shape,
    /// allocation failure) bubble up.
    pub fn new_with_activations(
        input_channels: usize,
        squeeze_channels: usize,
        activation: Box<dyn Module<T>>,
        scale_activation: Box<dyn Module<T>>,
    ) -> FerrotorchResult<Self> {
        if input_channels == 0 || squeeze_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "SqueezeExcitation: input_channels and squeeze_channels must be > 0 \
                     (got input_channels={input_channels}, squeeze_channels={squeeze_channels})"
                ),
            });
        }

        // 1×1 convolutions with bias — torchvision's `SqueezeExcitation`
        // uses bias=True for both fc1 and fc2 (they appear unset, which
        // defaults to True in `nn.Conv2d`).
        let fc1 = Conv2d::new(
            input_channels,
            squeeze_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            true,
        )?;
        let fc2 = Conv2d::new(
            squeeze_channels,
            input_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            true,
        )?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));

        Ok(Self {
            avgpool,
            fc1,
            activation,
            fc2,
            scale_activation,
            training: true,
        })
    }

    /// Forward pass.
    ///
    /// `input` must be 4-D `[B, C, H, W]` with `C == input_channels`.
    pub fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // 1) Squeeze: global average pool to [B, C, 1, 1].
        let scale = Module::<T>::forward(&self.avgpool, input)?;
        // 2) fc1 → activation → fc2.
        let scale = self.fc1.forward(&scale)?;
        let scale = self.activation.forward(&scale)?;
        let scale = self.fc2.forward(&scale)?;
        // 3) scale_activation gates the [B, C, 1, 1] tensor.
        let scale = self.scale_activation.forward(&scale)?;
        // 4) Broadcast multiply: [B, C, H, W] * [B, C, 1, 1].
        mul(input, &scale)
    }
}

impl<T: Float> Module<T> for SqueezeExcitation<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.fc1.parameters());
        p.extend(self.fc2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.fc1.parameters_mut());
        p.extend(self.fc2.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.fc1.named_parameters() {
            p.push((format!("fc1.{n}"), param));
        }
        for (n, param) in self.fc2.named_parameters() {
            p.push((format!("fc2.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.avgpool,
            &self.fc1,
            self.activation.as_ref(),
            &self.fc2,
            self.scale_activation.as_ref(),
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("avgpool".to_string(), &self.avgpool as &dyn Module<T>),
            ("fc1".to_string(), &self.fc1),
            ("activation".to_string(), self.activation.as_ref()),
            ("fc2".to_string(), &self.fc2),
            (
                "scale_activation".to_string(),
                self.scale_activation.as_ref(),
            ),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.activation.train();
        self.scale_activation.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.activation.eval();
        self.scale_activation.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{HardSigmoid, SiLU};
    use ferrotorch_core::storage::TensorStorage;

    fn cpu_tensor_4d(data: Vec<f32>, shape: [usize; 4]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn se_construction_smoke() {
        let se = SqueezeExcitation::<f32>::new(16, 4).expect("SE construction");
        assert_eq!(se.fc1.parameters().len(), 2); // weight + bias
        assert_eq!(se.fc2.parameters().len(), 2);
    }

    /// Evidence #6: named_parameters returns exactly fc1.{weight,bias},
    /// fc2.{weight,bias} — matching torchvision's SqueezeExcitation
    /// state_dict key set.
    #[test]
    fn se_named_parameters_match_torchvision() {
        let se = SqueezeExcitation::<f32>::new(16, 4).unwrap();
        let names: Vec<String> = se.named_parameters().into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            vec![
                "fc1.weight".to_string(),
                "fc1.bias".to_string(),
                "fc2.weight".to_string(),
                "fc2.bias".to_string(),
            ]
        );
    }

    /// Evidence #7: named_children order matches torchvision's
    /// `(avgpool, fc1, activation, fc2, scale_activation)`.
    #[test]
    fn se_named_children_match_torchvision_order() {
        let se = SqueezeExcitation::<f32>::new(16, 4).unwrap();
        let names: Vec<String> = se.named_children().into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            vec![
                "avgpool".to_string(),
                "fc1".to_string(),
                "activation".to_string(),
                "fc2".to_string(),
                "scale_activation".to_string(),
            ]
        );
    }

    /// Evidence #4: SE primitive forward equals manually-composed
    /// AdaptiveAvgPool2d + Conv2d(1×1) + ReLU + Conv2d(1×1) + Sigmoid +
    /// broadcast multiply.
    #[test]
    fn se_forward_matches_manual_composition() {
        // Build SE block then a parallel manual pipeline that shares its
        // weights. Forward both on a deterministic input and check
        // bitwise (or near-bitwise) equality.
        let mut se = SqueezeExcitation::<f32>::new(8, 2).unwrap();

        // Replace fc1, fc2 weights with deterministic small values so the
        // intermediate magnitudes stay finite.
        let fc1_weight = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05_f32; 2 * 8]),
            vec![2, 8, 1, 1],
            false,
        )
        .unwrap();
        let fc1_bias =
            Tensor::from_storage(TensorStorage::cpu(vec![0.01_f32; 2]), vec![2], false).unwrap();
        let fc2_weight = Tensor::from_storage(
            TensorStorage::cpu(vec![0.07_f32; 8 * 2]),
            vec![8, 2, 1, 1],
            false,
        )
        .unwrap();
        let fc2_bias =
            Tensor::from_storage(TensorStorage::cpu(vec![0.02_f32; 8]), vec![8], false).unwrap();

        se.fc1
            .set_weight(Parameter::new(fc1_weight.clone()))
            .unwrap();
        // Conv2d::set_weight only validates shape; bias is inaccessible
        // through the public API. Re-build fc1/fc2 from_parts to inject
        // bias deterministically.
        let new_fc1 =
            Conv2d::from_parts(fc1_weight, Some(fc1_bias.clone()), (1, 1), (0, 0)).unwrap();
        let new_fc2 =
            Conv2d::from_parts(fc2_weight, Some(fc2_bias.clone()), (1, 1), (0, 0)).unwrap();
        se.fc1 = new_fc1;
        se.fc2 = new_fc2;

        // 1×8×4×4 input, deterministic.
        let n = /* B*C*H*W = 1*8*4*4 */ 8 * 4 * 4;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let x = cpu_tensor_4d(data.clone(), [1, 8, 4, 4]);

        let out_se = se.forward(&x).unwrap();

        // Manual pipeline: AdaptiveAvgPool2d → Conv2d(fc1) → ReLU →
        // Conv2d(fc2) → Sigmoid → mul.
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let m_relu = ReLU::new();
        let m_sig = Sigmoid::new();
        let p = Module::<f32>::forward(&pool, &x).unwrap();
        let p = se.fc1.forward(&p).unwrap();
        let p = m_relu.forward(&p).unwrap();
        let p = se.fc2.forward(&p).unwrap();
        let p = m_sig.forward(&p).unwrap();
        let manual = mul(&x, &p).unwrap();

        let a = out_se.data().unwrap();
        let m = manual.data().unwrap();
        assert_eq!(a.len(), m.len());
        for i in 0..a.len() {
            assert!(
                (a[i] - m[i]).abs() < 1e-6,
                "SE primitive vs manual mismatch at {i}: se={} manual={}",
                a[i],
                m[i]
            );
        }
    }

    /// Probe-before-fix (Evidence #3): hand-computed reference for a
    /// trivial 1×4×8×8 input where every element is 1.0, with both fc
    /// weights set to 0 and biases tuned so the gate is 0.5 → output
    /// is 0.5 * input = 0.5 everywhere.
    #[test]
    fn se_probe_handcomputed_reference() {
        // Inputs all 1.0, fc1 weights/bias = 0, fc2 weights/bias = 0.
        // After avgpool: [1,4,1,1] all 1.0.
        // After fc1: [1,2,1,1] all 0.0.
        // After ReLU: [1,2,1,1] all 0.0.
        // After fc2: [1,4,1,1] all 0.0.
        // After Sigmoid: [1,4,1,1] all 0.5.
        // Final: input * 0.5 = 0.5 everywhere.
        let mut se = SqueezeExcitation::<f32>::new(4, 2).unwrap();
        let fc1_weight = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0_f32; 2 * 4]),
            vec![2, 4, 1, 1],
            false,
        )
        .unwrap();
        let fc1_bias =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0_f32; 2]), vec![2], false).unwrap();
        let fc2_weight = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0_f32; 4 * 2]),
            vec![4, 2, 1, 1],
            false,
        )
        .unwrap();
        let fc2_bias =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0_f32; 4]), vec![4], false).unwrap();
        se.fc1 = Conv2d::from_parts(fc1_weight, Some(fc1_bias), (1, 1), (0, 0)).unwrap();
        se.fc2 = Conv2d::from_parts(fc2_weight, Some(fc2_bias), (1, 1), (0, 0)).unwrap();

        let n = /* B*C*H*W = 1*4*8*8 */ 4 * 8 * 8;
        let x = cpu_tensor_4d(vec![1.0_f32; n], [1, 4, 8, 8]);
        let out = se.forward(&x).unwrap();
        let data = out.data().unwrap();
        for &v in data.iter() {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "expected gate output 0.5 everywhere, got {v}"
            );
        }
    }

    /// Evidence #5: backward by finite differences vs analytic gradient
    /// (small input; tolerance 1e-2). Confirms forward is differentiable
    /// end-to-end.
    #[test]
    fn se_backward_finite_differences() {
        use ferrotorch_core::grad_fns::reduction::sum;
        // Build SE block on a small input.
        let se = SqueezeExcitation::<f32>::new(4, 2).unwrap();

        let n = /* B*C*H*W = 1*4*4*4 */ 4 * 4 * 4;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin()).collect();
        let x =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![1, 4, 4, 4], true).unwrap();

        let out = se.forward(&x).unwrap();
        let loss = sum(&out).unwrap();
        loss.backward().unwrap();
        let grad = x.grad().unwrap().expect("x should carry grad");

        // FD on a few elements.
        let analytic = grad.data().unwrap().to_vec();
        let h = 1e-3_f32;

        for &i in &[0_usize, 7, 25, 50, n - 1] {
            let mut p = data.clone();
            p[i] += h;
            let xp = Tensor::from_storage(TensorStorage::cpu(p), vec![1, 4, 4, 4], false).unwrap();
            let mut m = data.clone();
            m[i] -= h;
            let xm = Tensor::from_storage(TensorStorage::cpu(m), vec![1, 4, 4, 4], false).unwrap();
            let lp: f32 = se.forward(&xp).unwrap().data().unwrap().iter().sum();
            let lm: f32 = se.forward(&xm).unwrap().data().unwrap().iter().sum();
            let fd = (lp - lm) / (2.0 * h);
            assert!(
                (analytic[i] - fd).abs() < 1e-2,
                "SE backward FD mismatch at {i}: analytic={} fd={}",
                analytic[i],
                fd
            );
        }
    }

    #[test]
    fn se_with_hardsigmoid_scale_smoke() {
        // V3-style: ReLU + HardSigmoid.
        let se: SqueezeExcitation<f32> = SqueezeExcitation::new_with_activations(
            8,
            2,
            Box::new(ReLU::new()),
            Box::new(HardSigmoid::new()),
        )
        .unwrap();
        let x = cpu_tensor_4d(vec![0.1_f32; 8 * 6 * 6], [1, 8, 6, 6]);
        let out = se.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 6, 6]);
    }

    #[test]
    fn se_with_silu_sigmoid_smoke() {
        // EfficientNet-style: SiLU + Sigmoid.
        let se: SqueezeExcitation<f32> = SqueezeExcitation::new_with_activations(
            16,
            4,
            Box::new(SiLU::new()),
            Box::new(Sigmoid::new()),
        )
        .unwrap();
        let x = cpu_tensor_4d(vec![0.05_f32; 16 * 4 * 4], [1, 16, 4, 4]);
        let out = se.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 16, 4, 4]);
    }

    /// Validate that SE is `Send + Sync` so it can compose into any
    /// model whose `Module` bound requires both. (`Box<dyn Module<T>>`
    /// inside the struct must propagate these bounds.)
    #[test]
    fn se_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SqueezeExcitation<f32>>();
    }
}
