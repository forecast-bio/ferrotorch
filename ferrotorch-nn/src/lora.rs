//! Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
//!
//! Instead of fine-tuning all weights of a pretrained model, LoRA freezes
//! the original weights and injects a trainable low-rank decomposition:
//!
//! ```text
//! W' = W + (alpha / r) * B @ A
//! ```
//!
//! where `A` is `[r, in_features]` and `B` is `[out_features, r]`. Only `A`
//! and `B` are trainable — the original `W` stays frozen. This dramatically
//! reduces the number of trainable parameters while preserving model quality.
//!
//! # References
//!
//! Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.

use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::linalg::mm_differentiable;
use ferrotorch_core::grad_fns::shape::transpose_2d;
use ferrotorch_core::{scalar, Float, FerrotorchError, FerrotorchResult, Tensor};

use crate::dropout::Dropout;
use crate::init;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Low-Rank Adaptation wrapper for a [`Linear`] layer.
///
/// Freezes the original weight and adds a trainable low-rank decomposition.
/// The forward pass computes:
///
/// ```text
/// y = x @ W^T + x @ (B @ A)^T * (alpha / r) + bias
/// ```
///
/// Only `lora_a` and `lora_b` appear in [`parameters()`](Module::parameters),
/// so optimizers only update the low-rank matrices. The base layer's weight
/// and bias are excluded from the parameter list (frozen).
///
/// # Initialization
///
/// - **A**: `N(0, 1/sqrt(r))` — Kaiming-style for the rank dimension.
/// - **B**: Zeros — so the LoRA contribution starts at zero and training
///   begins from the pretrained checkpoint.
///
/// # Merging
///
/// After fine-tuning, call [`merge()`](LoRALinear::merge) to fold the LoRA
/// weights into the base layer. This eliminates the runtime overhead of the
/// extra matmuls, producing a standard `Linear` layer for inference.
///
/// # Examples
///
/// ```ignore
/// let base = Linear::<f32>::new(768, 768, true)?;
/// let lora = LoRALinear::new(base, 8, 1.0, 0.0)?;
/// let output = lora.forward(&input)?;   // only lora_a, lora_b are trainable
/// ```
#[derive(Debug)]
pub struct LoRALinear<T: Float> {
    /// Original frozen linear layer (not included in `parameters()`).
    base: Linear<T>,
    /// Low-rank A matrix: `[r, in_features]`, trainable.
    lora_a: Parameter<T>,
    /// Low-rank B matrix: `[out_features, r]`, trainable.
    lora_b: Parameter<T>,
    /// Scaling factor (numerator of `alpha / r`).
    alpha: f64,
    /// Rank of the low-rank decomposition.
    rank: usize,
    /// Optional dropout on the LoRA input path.
    dropout: Option<Dropout<T>>,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> LoRALinear<T> {
    /// Create a LoRA wrapper around an existing `Linear` layer.
    ///
    /// # Arguments
    ///
    /// - `base` — The pretrained linear layer to adapt. Its parameters are
    ///   frozen (excluded from `parameters()`).
    /// - `rank` — Rank of the low-rank decomposition. Typical values: 1–64.
    /// - `alpha` — Scaling factor. The LoRA contribution is scaled by
    ///   `alpha / rank`. Common choice: `alpha == rank` (scale = 1).
    /// - `dropout_p` — Dropout probability on the LoRA input path. Set to
    ///   `0.0` to disable.
    ///
    /// # Errors
    ///
    /// Returns an error if `rank` is zero, if `dropout_p` is invalid, or if
    /// parameter allocation fails.
    pub fn new(
        base: Linear<T>,
        rank: usize,
        alpha: f64,
        dropout_p: f64,
    ) -> FerrotorchResult<Self> {
        if rank == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LoRALinear: rank must be > 0".into(),
            });
        }

        let in_features = base.in_features();
        let out_features = base.out_features();

        // A initialized from N(0, 1/sqrt(r)) — so the initial LoRA output
        // has variance independent of rank.
        let mut lora_a = Parameter::zeros(&[rank, in_features])?;
        init::normal(&mut lora_a, 0.0, 1.0 / (rank as f64).sqrt())?;

        // B initialized to zeros — LoRA contribution starts at zero.
        let lora_b = Parameter::zeros(&[out_features, rank])?;

        let dropout = if dropout_p > 0.0 {
            Some(Dropout::new(dropout_p)?)
        } else {
            None
        };

        Ok(Self {
            base,
            lora_a,
            lora_b,
            alpha,
            rank,
            dropout,
            training: true,
        })
    }

    /// Merge LoRA weights into the base layer for inference efficiency.
    ///
    /// Computes `W_merged = W + (alpha/r) * B @ A` and replaces the base
    /// weight. After merging, the forward pass is a single matmul with no
    /// overhead. The LoRA matrices are reset to their initial state (A
    /// re-initialized, B zeroed) so that additional fine-tuning can continue
    /// from the merged checkpoint if desired.
    pub fn merge(&mut self) -> FerrotorchResult<()> {
        let scale = T::from(self.alpha / self.rank as f64).unwrap();

        // B @ A: [out_features, r] @ [r, in_features] = [out_features, in_features]
        let b_data = self.lora_b.data()?;
        let a_data = self.lora_a.data()?;
        let out_features = self.base.out_features();
        let in_features = self.base.in_features();
        let r = self.rank;

        let zero = <T as num_traits::Zero>::zero();
        let mut ba = vec![zero; out_features * in_features];
        for i in 0..out_features {
            for j in 0..in_features {
                let mut sum = zero;
                for k in 0..r {
                    sum = sum + b_data[i * r + k] * a_data[k * in_features + j];
                }
                ba[i * in_features + j] = sum;
            }
        }

        // W_merged = W + scale * B @ A
        let w_data = self.base.weight.data()?;
        let merged: Vec<T> = w_data
            .iter()
            .zip(ba.iter())
            .map(|(&w, &d)| w + scale * d)
            .collect();

        self.base.weight = Parameter::from_slice(&merged, &[out_features, in_features])?;

        // Reset LoRA matrices so the module can be fine-tuned again.
        self.lora_a = Parameter::zeros(&[r, in_features])?;
        init::normal(&mut self.lora_a, 0.0, 1.0 / (r as f64).sqrt())?;
        self.lora_b = Parameter::zeros(&[out_features, r])?;

        Ok(())
    }

    /// The effective rank of the adaptation.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The scaling factor alpha.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Borrow the underlying base linear layer.
    #[inline]
    pub fn base(&self) -> &Linear<T> {
        &self.base
    }

    /// Consume the LoRA wrapper and return the base linear layer.
    ///
    /// Call [`merge()`](LoRALinear::merge) first if you want the LoRA
    /// weights folded into the base.
    pub fn into_base(self) -> Linear<T> {
        self.base
    }
}

impl<T: Float> Module<T> for LoRALinear<T> {
    /// Forward pass: base linear output plus scaled low-rank adaptation.
    ///
    /// ```text
    /// y = base.forward(x) + (x @ A^T @ B^T) * (alpha / r)
    /// ```
    ///
    /// When dropout is configured and the module is in training mode,
    /// dropout is applied to the input on the LoRA path only (the base
    /// path is unaffected).
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Base forward (frozen weights — not in parameters()).
        let base_out = self.base.forward(input)?;

        // LoRA path: optionally apply dropout to input.
        let lora_input = if let Some(ref dropout) = self.dropout {
            if self.training {
                dropout.forward(input)?
            } else {
                input.clone()
            }
        } else {
            input.clone()
        };

        // lora_out = input @ A^T @ B^T
        // A^T: [in_features, r]
        let a_t = transpose_2d(self.lora_a.tensor())?;
        // xa: [batch, r]
        let xa = mm_differentiable(&lora_input, &a_t)?;
        // B^T: [r, out_features]
        let b_t = transpose_2d(self.lora_b.tensor())?;
        // lora_out: [batch, out_features]
        let lora_out = mm_differentiable(&xa, &b_t)?;

        // Scale by alpha / r.
        let scale_val = T::from(self.alpha / self.rank as f64).unwrap();
        let scale_tensor = scalar(scale_val)?;
        let scaled = mul(&lora_out, &scale_tensor)?;

        // Add to base output.
        add(&base_out, &scaled)
    }

    /// Returns only the LoRA parameters (A and B). The base layer's
    /// parameters are frozen and excluded.
    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.lora_a, &self.lora_b]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![
            ("lora_a".to_string(), &self.lora_a),
            ("lora_b".to_string(), &self.lora_b),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.base.train();
        if let Some(ref mut d) = self.dropout {
            d.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        self.base.eval();
        if let Some(ref mut d) = self.dropout {
            d.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<T: Float> std::fmt::Display for LoRALinear<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoRALinear(in_features={}, out_features={}, rank={}, alpha={}, bias={}, dropout={})",
            self.base.in_features(),
            self.base.out_features(),
            self.rank,
            self.alpha,
            self.base.bias.is_some(),
            self.dropout.is_some(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a leaf tensor with given data and shape.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), requires_grad)
            .unwrap()
    }

    /// Assert two float slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_construction() {
        let base = Linear::<f32>::new(10, 5, true).unwrap();
        let lora = LoRALinear::new(base, 4, 1.0, 0.0).unwrap();
        assert_eq!(lora.rank(), 4);
        assert_eq!(lora.alpha(), 1.0);
        assert_eq!(lora.lora_a.shape(), &[4, 10]);
        assert_eq!(lora.lora_b.shape(), &[5, 4]);
    }

    #[test]
    fn test_construction_zero_rank_rejected() {
        let base = Linear::<f32>::new(10, 5, true).unwrap();
        assert!(LoRALinear::new(base, 0, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_construction_with_dropout() {
        let base = Linear::<f32>::new(10, 5, true).unwrap();
        let lora = LoRALinear::new(base, 4, 1.0, 0.1).unwrap();
        assert!(lora.dropout.is_some());
    }

    #[test]
    fn test_construction_invalid_dropout_rejected() {
        let base = Linear::<f32>::new(10, 5, true).unwrap();
        assert!(LoRALinear::new(base, 4, 1.0, 1.5).is_err());
    }

    // -----------------------------------------------------------------------
    // Forward shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_shape() {
        let base = Linear::<f32>::new(8, 4, true).unwrap();
        let lora = LoRALinear::new(base, 2, 1.0, 0.0).unwrap();
        let input = leaf(&[0.0; 24], &[3, 8], false);
        let output = lora.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_forward_shape_no_bias() {
        let base = Linear::<f32>::new(6, 3, false).unwrap();
        let lora = LoRALinear::new(base, 2, 1.0, 0.0).unwrap();
        let input = leaf(&[0.0; 12], &[2, 6], false);
        let output = lora.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }

    // -----------------------------------------------------------------------
    // Parameters — only LoRA A and B, not base
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameters_only_lora() {
        let base = Linear::<f32>::new(10, 5, true).unwrap();
        let lora = LoRALinear::new(base, 4, 1.0, 0.0).unwrap();
        let params = lora.parameters();
        // Only lora_a and lora_b — NOT base weight/bias.
        assert_eq!(params.len(), 2);
        // lora_a: 4 * 10 = 40, lora_b: 5 * 4 = 20
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 60);
    }

    #[test]
    fn test_named_parameters_keys() {
        let base = Linear::<f32>::new(10, 5, true).unwrap();
        let lora = LoRALinear::new(base, 4, 1.0, 0.0).unwrap();
        let named = lora.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "lora_a");
        assert_eq!(named[1].0, "lora_b");
    }

    // -----------------------------------------------------------------------
    // Zero-initialized B means output matches base
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_b_matches_base_output() {
        // Since B is initialized to zeros, the LoRA contribution is zero.
        // The LoRA output should exactly match the base Linear output.
        let mut base = Linear::<f32>::new(3, 2, true).unwrap();
        base.weight =
            Parameter::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();
        *base.bias.as_mut().unwrap() =
            Parameter::from_slice(&[10.0, 20.0], &[2]).unwrap();

        // Compute base output for reference.
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let base_out = base.forward(&input).unwrap();
        let base_data = base_out.data().unwrap().to_vec();

        // Wrap in LoRA with rank=1. B is zeros, so LoRA contribution is zero.
        let lora = LoRALinear::new(base, 1, 1.0, 0.0).unwrap();
        let input2 = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let lora_out = lora.forward(&input2).unwrap();

        assert_eq!(lora_out.shape(), &[2, 2]);
        assert_close(lora_out.data().unwrap(), &base_data, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Different ranks
    // -----------------------------------------------------------------------

    #[test]
    fn test_rank_1() {
        let base = Linear::<f32>::new(8, 4, true).unwrap();
        let lora = LoRALinear::new(base, 1, 1.0, 0.0).unwrap();
        assert_eq!(lora.rank(), 1);
        assert_eq!(lora.lora_a.shape(), &[1, 8]);
        assert_eq!(lora.lora_b.shape(), &[4, 1]);
        let input = leaf(&[0.0; 16], &[2, 8], false);
        let output = lora.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn test_rank_4() {
        let base = Linear::<f32>::new(16, 8, false).unwrap();
        let lora = LoRALinear::new(base, 4, 2.0, 0.0).unwrap();
        assert_eq!(lora.rank(), 4);
        assert_eq!(lora.lora_a.shape(), &[4, 16]);
        assert_eq!(lora.lora_b.shape(), &[8, 4]);
        let input = leaf(&[0.0; 32], &[2, 16], false);
        let output = lora.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_rank_16() {
        let base = Linear::<f32>::new(64, 32, true).unwrap();
        let lora = LoRALinear::new(base, 16, 8.0, 0.0).unwrap();
        assert_eq!(lora.rank(), 16);
        assert_eq!(lora.lora_a.shape(), &[16, 64]);
        assert_eq!(lora.lora_b.shape(), &[32, 16]);
        let input = leaf(&[0.0; 128], &[2, 64], false);
        let output = lora.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 32]);
    }

    // -----------------------------------------------------------------------
    // Merge produces equivalent output
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_produces_same_output() {
        // Create a base layer with known weights.
        let mut base = Linear::<f32>::new(4, 3, true).unwrap();
        base.weight = Parameter::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 4],
        )
        .unwrap();
        *base.bias.as_mut().unwrap() =
            Parameter::from_slice(&[0.1, 0.2, 0.3], &[3]).unwrap();

        let mut lora = LoRALinear::new(base, 2, 1.0, 0.0).unwrap();

        // Set known LoRA weights so the contribution is non-zero.
        lora.lora_a = Parameter::from_slice(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            &[2, 4],
        )
        .unwrap();
        lora.lora_b = Parameter::from_slice(
            &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
            &[3, 2],
        )
        .unwrap();

        // Compute output before merge.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4], false);
        let pre_merge_out = lora.forward(&input).unwrap();
        let pre_data = pre_merge_out.data().unwrap().to_vec();

        // Merge and compute output from the base layer directly.
        lora.merge().unwrap();
        let merged_base = &lora.base;
        let input2 = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4], false);
        let post_merge_out = merged_base.forward(&input2).unwrap();

        assert_close(post_merge_out.data().unwrap(), &pre_data, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Forward correctness with known weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_correctness_known_weights() {
        // base: W = [[1, 0], [0, 1]], bias = [0, 0]  (identity, 2->2)
        let mut base = Linear::<f32>::new(2, 2, true).unwrap();
        base.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        *base.bias.as_mut().unwrap() =
            Parameter::from_slice(&[0.0, 0.0], &[2]).unwrap();

        let mut lora = LoRALinear::new(base, 1, 2.0, 0.0).unwrap();

        // A = [[1, 0]]  (rank=1, in=2)
        // B = [[1], [0]] (out=2, rank=1)
        lora.lora_a = Parameter::from_slice(&[1.0, 0.0], &[1, 2]).unwrap();
        lora.lora_b = Parameter::from_slice(&[1.0, 0.0], &[2, 1]).unwrap();

        // input = [[1, 2]]
        let input = leaf(&[1.0, 2.0], &[1, 2], false);
        let output = lora.forward(&input).unwrap();

        // base_out = [1, 2]  (identity)
        // LoRA: x @ A^T = [1,2] @ [[1],[0]] = [1]
        //       [1] @ B^T = [1] @ [[1, 0]] = [1, 0]
        //       scaled = [1, 0] * (2.0 / 1) = [2, 0]
        // total = [1+2, 2+0] = [3, 2]
        assert_eq!(output.shape(), &[1, 2]);
        assert_close(output.data().unwrap(), &[3.0, 2.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Train / Eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_eval() {
        let base = Linear::<f32>::new(4, 3, true).unwrap();
        let mut lora = LoRALinear::new(base, 2, 1.0, 0.1).unwrap();
        assert!(lora.is_training());
        lora.eval();
        assert!(!lora.is_training());
        lora.train();
        assert!(lora.is_training());
    }

    // -----------------------------------------------------------------------
    // State dict
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_dict_keys() {
        let base = Linear::<f32>::new(8, 4, true).unwrap();
        let lora = LoRALinear::new(base, 2, 1.0, 0.0).unwrap();
        let sd = lora.state_dict();
        assert!(sd.contains_key("lora_a"));
        assert!(sd.contains_key("lora_b"));
        assert!(!sd.contains_key("weight"));
        assert!(!sd.contains_key("bias"));
        assert_eq!(sd["lora_a"].shape(), &[2, 8]);
        assert_eq!(sd["lora_b"].shape(), &[4, 2]);
    }

    #[test]
    fn test_state_dict_roundtrip() {
        let base = Linear::<f32>::new(6, 3, true).unwrap();
        let lora = LoRALinear::new(base, 2, 1.0, 0.0).unwrap();
        let sd = lora.state_dict();

        let base2 = Linear::<f32>::new(6, 3, true).unwrap();
        let mut lora2 = LoRALinear::new(base2, 2, 1.0, 0.0).unwrap();
        lora2.load_state_dict(&sd, true).unwrap();

        assert_close(
            lora2.lora_a.data().unwrap(),
            lora.lora_a.data().unwrap(),
            1e-7,
        );
        assert_close(
            lora2.lora_b.data().unwrap(),
            lora.lora_b.data().unwrap(),
            1e-7,
        );
    }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let base = Linear::<f32>::new(10, 5, true).unwrap();
        let lora = LoRALinear::new(base, 4, 2.0, 0.0).unwrap();
        let s = format!("{lora}");
        assert_eq!(
            s,
            "LoRALinear(in_features=10, out_features=5, rank=4, alpha=2, bias=true, dropout=false)"
        );
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LoRALinear<f32>>();
        assert_send_sync::<LoRALinear<f64>>();
    }
}
