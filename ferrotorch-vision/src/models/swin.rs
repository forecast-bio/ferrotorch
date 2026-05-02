//! Swin Transformer Tiny architecture.
//!
//! Implements a simplified Swin Transformer Tiny from "Swin Transformer:
//! Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021).
//!
//! **Simplification**: This uses standard (global) multi-head attention from
//! `ferrotorch_nn` instead of shifted-window attention. The result is a
//! hierarchical ViT with the same parameter count and stage structure as
//! Swin-T, but without windowed attention. This is sufficient to prove the
//! architecture and matches the expected ~29M parameters.
//!
//! Architecture:
//! - Patch partition: `Conv2d(3, 96, kernel=4, stride=4)`
//! - 4 stages with `[2, 2, 6, 2]` transformer blocks
//! - Channel dimensions: `[96, 192, 384, 768]`
//! - Each block: `LayerNorm -> MultiheadAttention -> residual -> LayerNorm -> MLP -> residual`
//! - Downsampling between stages: `Conv2d(C, 2*C, kernel=2, stride=2)`
//! - Head: adaptive average pool -> `LayerNorm(768)` -> `Linear(768, num_classes)`

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage};

use ferrotorch_nn::activation::GELU;
use ferrotorch_nn::attention::MultiheadAttention;
use ferrotorch_nn::conv::Conv2d;
use ferrotorch_nn::linear::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::LayerNorm;
use ferrotorch_nn::parameter::Parameter;

// ===========================================================================
// SwinBlock
// ===========================================================================

/// A single transformer block used within a Swin stage.
///
/// ```text
/// x -> LayerNorm -> MultiheadAttention -> (+x) -> LayerNorm -> MLP -> (+) -> out
///      (norm1)        (attn)               |       (norm2)    (mlp)    |
///      ^                                   |       ^                   |
///      +--- residual connection -----------+       +--- residual ------+
/// ```
///
/// This uses global (non-windowed) attention as a simplification.
pub struct SwinBlock<T: Float> {
    norm1: LayerNorm<T>,
    attn: MultiheadAttention<T>,
    norm2: LayerNorm<T>,
    mlp_fc1: Linear<T>,
    mlp_fc2: Linear<T>,
    gelu: GELU,
    training: bool,
}

impl<T: Float> SwinBlock<T> {
    /// Create a new Swin transformer block.
    ///
    /// * `dim` -- hidden dimension for this stage.
    /// * `num_heads` -- number of attention heads.
    /// * `mlp_dim` -- intermediate dimension of the MLP (typically `4 * dim`).
    pub fn new(dim: usize, num_heads: usize, mlp_dim: usize) -> FerrotorchResult<Self> {
        let norm1 = LayerNorm::new(vec![dim], 1e-6, true)?;
        let attn = MultiheadAttention::new(dim, num_heads, true)?;
        let norm2 = LayerNorm::new(vec![dim], 1e-6, true)?;
        let mlp_fc1 = Linear::new(dim, mlp_dim, true)?;
        let mlp_fc2 = Linear::new(mlp_dim, dim, true)?;
        let gelu = GELU::new();

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp_fc1,
            mlp_fc2,
            gelu,
            training: true,
        })
    }

    /// Forward pass for the MLP sub-block on a single batch slice.
    ///
    /// Input: `[seq_len, dim]` (2-D)
    /// Output: `[seq_len, dim]` (2-D)
    fn mlp_forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.mlp_fc1.forward(input)?;
        let x = self.gelu.forward(&x)?;
        self.mlp_fc2.forward(&x)
    }
}

impl<T: Float> Module<T> for SwinBlock<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // input: [B, seq_len, dim]
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];
        let device = input.device();

        // --- Sub-block 1: LayerNorm -> MultiheadAttention -> residual ---
        let normed1 = self.norm1.forward(input)?;
        let attn_out = self.attn.forward(&normed1)?;
        let x = add(input, &attn_out)?;

        // --- Sub-block 2: LayerNorm -> MLP -> residual ---
        let normed2 = self.norm2.forward(&x)?;

        // MLP expects 2-D [seq_len, dim] — process each batch element.
        let normed2_data = normed2.data_vec()?;
        let slice_size = seq_len * dim;
        let mut mlp_out_data = Vec::with_capacity(batch * slice_size);

        for b in 0..batch {
            let start = b * slice_size;
            let end = start + slice_size;
            let slice_data = normed2_data[start..end].to_vec();
            let slice_tensor = Tensor::from_storage(
                TensorStorage::cpu(slice_data),
                vec![seq_len, dim],
                normed2.requires_grad(),
            )?
            .to(device)?;
            let mlp_result = self.mlp_forward(&slice_tensor)?;
            let result_data = mlp_result.data_vec()?;
            mlp_out_data.extend_from_slice(&result_data);
        }

        let mlp_out = Tensor::from_storage(
            TensorStorage::cpu(mlp_out_data),
            vec![batch, seq_len, dim],
            normed2.requires_grad(),
        )?
        .to(device)?;

        add(&x, &mlp_out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.norm1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.norm1.parameters_mut());
        params.extend(self.attn.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.mlp_fc1.parameters_mut());
        params.extend(self.mlp_fc2.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.norm1.named_parameters() {
            params.push((format!("norm1.{name}"), p));
        }
        for (name, p) in self.attn.named_parameters() {
            params.push((format!("attn.{name}"), p));
        }
        for (name, p) in self.norm2.named_parameters() {
            params.push((format!("norm2.{name}"), p));
        }
        for (name, p) in self.mlp_fc1.named_parameters() {
            params.push((format!("mlp.fc1.{name}"), p));
        }
        for (name, p) in self.mlp_fc2.named_parameters() {
            params.push((format!("mlp.fc2.{name}"), p));
        }
        params
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

// ===========================================================================
// SwinStage
// ===========================================================================

/// A single stage of the Swin Transformer.
///
/// Each stage consists of multiple [`SwinBlock`]s followed by an optional
/// downsampling layer. The final stage (stage 4) has no downsampling.
///
/// Downsampling is implemented as `Conv2d(dim, 2*dim, kernel=2, stride=2)`
/// which halves spatial resolution and doubles channels — analogous to
/// the patch-merging layer in the original Swin paper.
struct SwinStage<T: Float> {
    blocks: Vec<SwinBlock<T>>,
    downsample: Option<Conv2d<T>>,
}

impl<T: Float> SwinStage<T> {
    /// Create a new Swin stage.
    ///
    /// * `dim` -- channel dimension for this stage.
    /// * `depth` -- number of transformer blocks.
    /// * `num_heads` -- number of attention heads per block.
    /// * `mlp_ratio` -- ratio of MLP hidden dim to `dim`.
    /// * `downsample` -- whether to add a downsampling layer after blocks.
    fn new(
        dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: usize,
        downsample: bool,
    ) -> FerrotorchResult<Self> {
        let mlp_dim = dim * mlp_ratio;

        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(SwinBlock::new(dim, num_heads, mlp_dim)?);
        }

        let downsample_layer = if downsample {
            Some(Conv2d::new(dim, dim * 2, (2, 2), (2, 2), (0, 0), false)?)
        } else {
            None
        };

        Ok(Self {
            blocks,
            downsample: downsample_layer,
        })
    }

    /// Forward pass.
    ///
    /// Input: `[B, H*W, C]` (3-D sequence)
    /// Output: `[B, H'*W', C']` where if downsampled, `H'=H/2`, `W'=W/2`, `C'=2*C`.
    ///
    /// The `spatial_h` and `spatial_w` arguments provide the spatial dimensions
    /// so we can reshape for the downsampling convolution.
    fn stage_forward(
        &self,
        input: &Tensor<T>,
        spatial_h: usize,
        spatial_w: usize,
    ) -> FerrotorchResult<Tensor<T>> {
        let device = input.device();

        // Run transformer blocks on [B, H*W, C]
        let mut x = self.blocks[0].forward(input)?;
        for block in &self.blocks[1..] {
            x = block.forward(&x)?;
        }

        // Downsample if needed: reshape to [B, C, H, W], apply Conv2d, reshape back.
        if let Some(ref ds) = self.downsample {
            let shape = x.shape();
            let batch = shape[0];
            let dim = shape[2];

            // Transpose [B, H*W, C] -> [B, C, H, W]
            let x_data = x.data_vec()?;
            let num_tokens = spatial_h * spatial_w;
            let mut transposed = vec![<T as num_traits::Zero>::zero(); batch * dim * num_tokens];

            for b in 0..batch {
                for t in 0..num_tokens {
                    for c in 0..dim {
                        // src: [b, t, c] in [B, H*W, C]
                        let src = b * num_tokens * dim + t * dim + c;
                        // dst: [b, c, t] in [B, C, H*W]
                        let dst = b * dim * num_tokens + c * num_tokens + t;
                        transposed[dst] = x_data[src];
                    }
                }
            }

            let x_4d = Tensor::from_storage(
                TensorStorage::cpu(transposed),
                vec![batch, dim, spatial_h, spatial_w],
                x.requires_grad(),
            )?
            .to(device)?;

            // Conv2d downsample: [B, C, H, W] -> [B, 2*C, H/2, W/2]
            let ds_out = ds.forward(&x_4d)?;
            let ds_shape = ds_out.shape();
            let new_dim = ds_shape[1];
            let new_h = ds_shape[2];
            let new_w = ds_shape[3];
            let new_tokens = new_h * new_w;

            // Transpose [B, 2*C, H/2, W/2] -> [B, (H/2)*(W/2), 2*C]
            let ds_data = ds_out.data_vec()?;
            let mut out = vec![<T as num_traits::Zero>::zero(); batch * new_tokens * new_dim];

            for b in 0..batch {
                for c in 0..new_dim {
                    for t in 0..new_tokens {
                        let src = b * new_dim * new_tokens + c * new_tokens + t;
                        let dst = b * new_tokens * new_dim + t * new_dim + c;
                        out[dst] = ds_data[src];
                    }
                }
            }

            Tensor::from_storage(
                TensorStorage::cpu(out),
                vec![batch, new_tokens, new_dim],
                x.requires_grad(),
            )?
            .to(device)
        } else {
            Ok(x)
        }
    }
}

// ===========================================================================
// SwinTransformer
// ===========================================================================

/// Swin Transformer (simplified with global attention).
///
/// This is a hierarchical vision transformer with the same structure and
/// parameter count as Swin-T, but using global multi-head attention instead
/// of shifted-window attention.
///
/// Architecture:
/// 1. **Patch partition**: `Conv2d(3, 96, 4, 4)` — splits 224x224 into 56x56 patches.
/// 2. **Stage 1**: 2 blocks at dim=96, then downsample to 28x28, dim=192.
/// 3. **Stage 2**: 2 blocks at dim=192, then downsample to 14x14, dim=384.
/// 4. **Stage 3**: 6 blocks at dim=384, then downsample to 7x7, dim=768.
/// 5. **Stage 4**: 2 blocks at dim=768 (no downsample).
/// 6. **Head**: Adaptive average pool over spatial dims, LayerNorm, Linear.
pub struct SwinTransformer<T: Float> {
    patch_embed: Conv2d<T>,
    stages: Vec<SwinStage<T>>,
    norm: LayerNorm<T>,
    head: Linear<T>,
    final_dim: usize,
    training: bool,
}

/// Configuration for the Swin Transformer.
struct SwinConfig {
    patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
    depths: Vec<usize>,
    num_heads: Vec<usize>,
    mlp_ratio: usize,
    num_classes: usize,
}

impl<T: Float> SwinTransformer<T> {
    /// Create a new Swin Transformer from a configuration.
    fn from_config(cfg: SwinConfig) -> FerrotorchResult<Self> {
        let patch_embed = Conv2d::new(
            cfg.in_channels,
            cfg.embed_dim,
            (cfg.patch_size, cfg.patch_size),
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            false,
        )?;

        let num_stages = cfg.depths.len();
        let mut stages = Vec::with_capacity(num_stages);
        let mut dim = cfg.embed_dim;

        for i in 0..num_stages {
            let has_downsample = i < num_stages - 1;
            stages.push(SwinStage::new(
                dim,
                cfg.depths[i],
                cfg.num_heads[i],
                cfg.mlp_ratio,
                has_downsample,
            )?);
            if has_downsample {
                dim *= 2;
            }
        }

        let norm = LayerNorm::new(vec![dim], 1e-6, true)?;
        let head = Linear::new(dim, cfg.num_classes, true)?;

        Ok(Self {
            patch_embed,
            stages,
            norm,
            head,
            final_dim: dim,
            training: true,
        })
    }

    /// Total number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<T: Float> Module<T> for SwinTransformer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        let batch = shape[0];
        let device = input.device();

        // 1. Patch partition: [B, 3, H, W] -> [B, embed_dim, H/4, W/4]
        let x = self.patch_embed.forward(input)?;
        let x_shape = x.shape();
        let embed_dim = x_shape[1];
        let mut spatial_h = x_shape[2];
        let mut spatial_w = x_shape[3];
        let num_tokens = spatial_h * spatial_w;

        // Transpose [B, C, H, W] -> [B, H*W, C]
        let x_data = x.data_vec()?;
        let mut seq_data = vec![<T as num_traits::Zero>::zero(); batch * num_tokens * embed_dim];

        for b in 0..batch {
            for c in 0..embed_dim {
                for t in 0..num_tokens {
                    let src = b * embed_dim * num_tokens + c * num_tokens + t;
                    let dst = b * num_tokens * embed_dim + t * embed_dim + c;
                    seq_data[dst] = x_data[src];
                }
            }
        }

        let mut x = Tensor::from_storage(
            TensorStorage::cpu(seq_data),
            vec![batch, num_tokens, embed_dim],
            input.requires_grad(),
        )?
        .to(device)?;

        // 2. Process through stages.
        for stage in &self.stages {
            x = stage.stage_forward(&x, spatial_h, spatial_w)?;

            if stage.downsample.is_some() {
                spatial_h /= 2;
                spatial_w /= 2;
            }
        }

        // 3. Adaptive average pool: mean over the spatial (token) dimension.
        //    x: [B, num_tokens_final, final_dim] -> [B, final_dim]
        let x_data = x.data_vec()?;
        let final_tokens = spatial_h * spatial_w;
        let mut pooled = vec![<T as num_traits::Zero>::zero(); batch * self.final_dim];

        let inv_tokens = T::from(1.0).unwrap() / T::from(final_tokens).unwrap();
        for b in 0..batch {
            for c in 0..self.final_dim {
                let mut sum = <T as num_traits::Zero>::zero();
                for t in 0..final_tokens {
                    let idx = b * final_tokens * self.final_dim + t * self.final_dim + c;
                    sum += x_data[idx];
                }
                pooled[b * self.final_dim + c] = sum * inv_tokens;
            }
        }

        let x = Tensor::from_storage(
            TensorStorage::cpu(pooled),
            vec![batch, self.final_dim],
            input.requires_grad(),
        )?
        .to(device)?;

        // 4. LayerNorm on the pooled features.
        //    LayerNorm expects [*, normalized_shape], so [B, final_dim] works directly.
        let x = self.norm.forward(&x)?;

        // 5. Classification head.
        self.head.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters());
        for stage in &self.stages {
            for block in &stage.blocks {
                params.extend(block.parameters());
            }
            if let Some(ref ds) = stage.downsample {
                params.extend(ds.parameters());
            }
        }
        params.extend(self.norm.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters_mut());
        for stage in &mut self.stages {
            for block in &mut stage.blocks {
                params.extend(block.parameters_mut());
            }
            if let Some(ref mut ds) = stage.downsample {
                params.extend(ds.parameters_mut());
            }
        }
        params.extend(self.norm.parameters_mut());
        params.extend(self.head.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();

        for (name, p) in self.patch_embed.named_parameters() {
            params.push((format!("patch_embed.{name}"), p));
        }

        for (i, stage) in self.stages.iter().enumerate() {
            for (j, block) in stage.blocks.iter().enumerate() {
                for (name, p) in block.named_parameters() {
                    params.push((format!("stages.{i}.blocks.{j}.{name}"), p));
                }
            }
            if let Some(ref ds) = stage.downsample {
                for (name, p) in ds.named_parameters() {
                    params.push((format!("stages.{i}.downsample.{name}"), p));
                }
            }
        }

        for (name, p) in self.norm.named_parameters() {
            params.push((format!("norm.{name}"), p));
        }
        for (name, p) in self.head.named_parameters() {
            params.push((format!("head.{name}"), p));
        }
        params
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

// ===========================================================================
// IntermediateFeatures — CL-499
// ===========================================================================

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for SwinTransformer<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();
        let shape = input.shape();
        let batch = shape[0];
        let device = input.device();

        // 1. Patch partition.
        let x = self.patch_embed.forward(input)?;
        out.insert("patch_embed".to_string(), x.clone());

        let x_shape = x.shape();
        let embed_dim = x_shape[1];
        let mut spatial_h = x_shape[2];
        let mut spatial_w = x_shape[3];
        let num_tokens = spatial_h * spatial_w;

        // Transpose [B, C, H, W] -> [B, H*W, C].
        let x_data = x.data_vec()?;
        let mut seq_data = vec![<T as num_traits::Zero>::zero(); batch * num_tokens * embed_dim];
        for b in 0..batch {
            for c in 0..embed_dim {
                for t in 0..num_tokens {
                    let src = b * embed_dim * num_tokens + c * num_tokens + t;
                    let dst = b * num_tokens * embed_dim + t * embed_dim + c;
                    seq_data[dst] = x_data[src];
                }
            }
        }
        let mut x = Tensor::from_storage(
            TensorStorage::cpu(seq_data),
            vec![batch, num_tokens, embed_dim],
            input.requires_grad(),
        )?
        .to(device)?;

        // 2. Stages.
        for (i, stage) in self.stages.iter().enumerate() {
            x = stage.stage_forward(&x, spatial_h, spatial_w)?;
            out.insert(format!("stage{i}"), x.clone());
            if stage.downsample.is_some() {
                spatial_h /= 2;
                spatial_w /= 2;
            }
        }

        // 3. Global average pool over the final spatial tokens.
        let x_data = x.data_vec()?;
        let final_tokens = spatial_h * spatial_w;
        let mut pooled = vec![<T as num_traits::Zero>::zero(); batch * self.final_dim];
        let inv_tokens = T::from(1.0).unwrap() / T::from(final_tokens).unwrap();
        for b in 0..batch {
            for c in 0..self.final_dim {
                let mut sum = <T as num_traits::Zero>::zero();
                for t in 0..final_tokens {
                    let idx = b * final_tokens * self.final_dim + t * self.final_dim + c;
                    sum += x_data[idx];
                }
                pooled[b * self.final_dim + c] = sum * inv_tokens;
            }
        }
        let x = Tensor::from_storage(
            TensorStorage::cpu(pooled),
            vec![batch, self.final_dim],
            input.requires_grad(),
        )?
        .to(device)?;
        out.insert("avgpool".to_string(), x.clone());

        let x = self.norm.forward(&x)?;
        out.insert("norm".to_string(), x.clone());

        let logits = self.head.forward(&x)?;
        out.insert("head".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        let mut names = vec!["patch_embed".to_string()];
        for i in 0..self.stages.len() {
            names.push(format!("stage{i}"));
        }
        names.push("avgpool".to_string());
        names.push("norm".to_string());
        names.push("head".to_string());
        names
    }
}

/// Construct a Swin Transformer Tiny model.
///
/// Architecture:
/// - Patch size: 4x4
/// - Embedding dimension: 96
/// - Depths: `[2, 2, 6, 2]`
/// - Heads: `[3, 6, 12, 24]`
/// - MLP ratio: 4
/// - Image size: 224x224
///
/// Total parameters: ~29M (for 1000 classes).
pub fn swin_tiny<T: Float>(num_classes: usize) -> FerrotorchResult<SwinTransformer<T>> {
    SwinTransformer::from_config(SwinConfig {
        patch_size: 4,
        in_channels: 3,
        embed_dim: 96,
        depths: vec![2, 2, 6, 2],
        num_heads: vec![3, 6, 12, 24],
        mlp_ratio: 4,
        num_classes,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{TensorStorage, no_grad};

    /// Create a 4-D tensor from flat data.
    fn leaf_4d(data: &[f32], shape: [usize; 4], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // SwinBlock tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swin_block_output_shape() {
        let block = SwinBlock::<f32>::new(96, 3, 384).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 16 * 96]),
            vec![1, 16, 96],
            false,
        )
        .unwrap();
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 16, 96]);
    }

    #[test]
    fn test_swin_block_parameter_count() {
        let dim = 96;
        let num_heads = 3;
        let mlp_dim = 384;
        let block = SwinBlock::<f32>::new(dim, num_heads, mlp_dim).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();

        // LayerNorm x2: 2 * 2 * dim = 384
        let ln_params = 2 * 2 * dim;
        // MHA: 4 * (dim*dim + dim) = 4 * (9216 + 96) = 37248
        let mha_params = 4 * (dim * dim + dim);
        // MLP: dim*mlp_dim + mlp_dim + mlp_dim*dim + dim = 96*384 + 384 + 384*96 + 96 = 74112
        let mlp_params = dim * mlp_dim + mlp_dim + mlp_dim * dim + dim;

        let expected = ln_params + mha_params + mlp_params;
        assert_eq!(count, expected);
    }

    // -----------------------------------------------------------------------
    // SwinTransformer (Swin-T) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swin_tiny_output_shape() {
        let model = swin_tiny::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_swin_tiny_param_count() {
        let model = swin_tiny::<f32>(1000).unwrap();
        let total = model.num_parameters();

        // Swin-T has ~29M parameters.
        //
        // Breakdown:
        //   patch_embed Conv2d(3, 96, 4, 4): 96*3*4*4 = 4,608
        //
        //   Stage 1: 2 blocks @ dim=96, heads=3, mlp=384
        //     Per block: 4*96 + 4*(96*96+96) + 2*96*384 + 384 + 96 = 111,840
        //     Stage total: 2 * 111,840 = 223,680
        //   Downsample 1: Conv2d(96, 192, 2, 2): 192*96*2*2 = 73,728
        //
        //   Stage 2: 2 blocks @ dim=192, heads=6, mlp=768
        //     Per block: 444,864
        //     Stage total: 2 * 444,864 = 889,728
        //   Downsample 2: Conv2d(192, 384, 2, 2): 384*192*2*2 = 589,824
        //
        //   Stage 3: 6 blocks @ dim=384, heads=12, mlp=1536
        //     Per block: 1,774,464
        //     Stage total: 6 * 1,774,464 = 10,646,784
        //   Downsample 3: Conv2d(384, 768, 2, 2): 768*384*2*2 = 2,359,296
        //
        //   Stage 4: 2 blocks @ dim=768, heads=24, mlp=3072
        //     Per block: 7,087,872
        //     Stage total: 2 * 7,087,872 = 14,175,744
        //
        //   norm LayerNorm(768): 2*768 = 1,536
        //   head Linear(768, 1000): 768*1000 + 1000 = 769,000
        //
        //   Grand total: 29,733,928
        assert!(
            total > 28_000_000,
            "Swin-T should have >28M params, got {total}"
        );
        assert!(
            total < 31_000_000,
            "Swin-T should have <31M params, got {total}"
        );
    }

    #[test]
    fn test_swin_tiny_named_parameters_prefixes() {
        let model = swin_tiny::<f32>(1000).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("patch_embed.")));
        assert!(names.iter().any(|n| n.starts_with("stages.0.blocks.0.")));
        assert!(names.iter().any(|n| n.starts_with("stages.0.downsample.")));
        assert!(names.iter().any(|n| n.starts_with("stages.2.blocks.5.")));
        assert!(names.iter().any(|n| n.starts_with("stages.3.blocks.1.")));
        // Stage 3 (the last) should NOT have downsample.
        assert!(!names.iter().any(|n| n.starts_with("stages.3.downsample.")));
        assert!(names.iter().any(|n| n.starts_with("norm.")));
        assert!(names.iter().any(|n| n.starts_with("head.")));
    }

    #[test]
    fn test_swin_tiny_train_eval() {
        let mut model = swin_tiny::<f32>(1000).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }

    #[test]
    fn test_swin_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SwinTransformer<f32>>();
        assert_send_sync::<SwinBlock<f32>>();
    }

    // -----------------------------------------------------------------------
    // Small Swin forward pass (fast test)
    // -----------------------------------------------------------------------

    #[test]
    fn test_small_swin_forward() {
        // A tiny Swin for fast testing: 32x32 image, patch_size=4 -> 8x8=64 tokens.
        // Depths [1, 1, 1, 1], minimal heads.
        let model = SwinTransformer::<f32>::from_config(SwinConfig {
            patch_size: 4,
            in_channels: 3,
            embed_dim: 32,
            depths: vec![1, 1, 1, 1],
            num_heads: vec![1, 2, 4, 8],
            mlp_ratio: 4,
            num_classes: 10,
        })
        .unwrap();

        let input = leaf_4d(&vec![0.1; 2 * 3 * 32 * 32], [2, 3, 32, 32], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);

        // Verify output is finite.
        let data = output.data().unwrap();
        for &v in data.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn test_swin_tiny_custom_classes() {
        let model = SwinTransformer::<f32>::from_config(SwinConfig {
            patch_size: 4,
            in_channels: 3,
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            mlp_ratio: 4,
            num_classes: 10,
        })
        .unwrap();

        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 10]);
    }
}
