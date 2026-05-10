//! Vision Transformer (ViT) architectures.
//!
//! Implements the ViT-B/16 model from "An Image is Worth 16x16 Words:
//! Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).
//!
//! The architecture splits an image into fixed-size patches via a
//! `Conv2d(3, embed_dim, kernel=patch_size, stride=patch_size)`, prepends a
//! learnable CLS token, adds positional embeddings, and processes the sequence
//! through a stack of Transformer encoder blocks. The CLS token output is
//! used for classification.
//!
//! All operations use differentiable primitives from `ferrotorch_core`, so
//! autograd handles the backward pass automatically.

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::{cat, expand};
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::activation::GELU;
use ferrotorch_nn::attention::MultiheadAttention;
use ferrotorch_nn::conv::Conv2d;
use ferrotorch_nn::linear::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::LayerNorm;
use ferrotorch_nn::parameter::Parameter;

// ===========================================================================
// PatchEmbed
// ===========================================================================

/// Patch embedding layer.
///
/// Splits a 4-D image `[B, 3, H, W]` into non-overlapping patches using a
/// convolution with `kernel_size = stride = patch_size`, then reshapes the
/// output to `[B, num_patches, embed_dim]`.
///
/// For a 224x224 image with `patch_size=16`:
///   - `num_patches = (224/16)^2 = 196`
///   - Each patch becomes a `embed_dim`-dimensional vector.
pub struct PatchEmbed<T: Float> {
    proj: Conv2d<T>,
    patch_size: usize,
    embed_dim: usize,
    training: bool,
}

impl<T: Float> PatchEmbed<T> {
    /// Create a new patch embedding layer.
    ///
    /// * `in_channels` -- number of input channels (typically 3 for RGB).
    /// * `embed_dim` -- output embedding dimension per patch.
    /// * `patch_size` -- spatial size of each patch (both height and width).
    pub fn new(in_channels: usize, embed_dim: usize, patch_size: usize) -> FerrotorchResult<Self> {
        // bias=true matches torchvision `vit_b_16`'s `conv_proj` (which uses
        // the default `bias=True` of `nn.Conv2d`). Phase 4 (#999/#1001)
        // surfaced the prior bias=false as a structural divergence — the
        // strict torchvision-→-ferrotorch loader rejected `conv_proj.bias`
        // as "no mapping to a ferrotorch parameter". Switching to bias=true
        // adds `embed_dim` learnable scalars (768 for ViT-B/16 — negligible
        // relative to the ~86M total) and makes ferrotorch's PatchEmbed
        // numerically faithful to the reference. The test-side remap
        // (`remap_torchvision_to_ferrotorch_vit_keys`) translates
        // `conv_proj.bias` → `patch_embed.proj.bias` accordingly.
        let proj = Conv2d::new(
            in_channels,
            embed_dim,
            (patch_size, patch_size),
            (patch_size, patch_size),
            (0, 0),
            true,
        )?;

        Ok(Self {
            proj,
            patch_size,
            embed_dim,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for PatchEmbed<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // input: [B, C, H, W]
        let shape = input.shape();
        let batch = shape[0];
        let h = shape[2];
        let w = shape[3];
        let num_patches = (h / self.patch_size) * (w / self.patch_size);

        // Conv2d: [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        let x = self.proj.forward(input)?;

        // Migrate to autograd-correct primitive chain (#996, closes #986/#987):
        // [B, embed_dim, grid_h, grid_w] -> view([B, embed_dim, num_patches])
        // -> permute(0, 2, 1) -> [B, num_patches, embed_dim] -> contiguous().
        // The probe at tests/probe_permute_migration.rs proves this matches the
        // previous manual data_vec()+indexed-loop output element-for-element.
        // Stays on `input.device()` end-to-end — no CPU pull-in-the-middle.
        x.view(&[batch as i64, self.embed_dim as i64, num_patches as i64])?
            .permute(&[0, 2, 1])?
            .contiguous()
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.proj.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.proj.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.proj
            .named_parameters()
            .into_iter()
            .map(|(name, p)| (format!("proj.{name}"), p))
            .collect()
    }

    // Phase 4 (#995): mirror `named_parameters` so a Phase 2-style
    // walk reaches the inner Conv2d at path `proj`.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.proj]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![("proj".to_string(), &self.proj)]
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
// TransformerBlock
// ===========================================================================

/// A single Transformer encoder block.
///
/// ```text
/// x -> LayerNorm -> MultiheadAttention -> (+x) -> LayerNorm -> MLP -> (+) -> out
///      (norm1)        (attn)               |       (norm2)    (mlp)    |
///      ^                                   |       ^                   |
///      +--- residual connection -----------+       +--- residual ------+
/// ```
///
/// The MLP is: `Linear(embed_dim, mlp_dim) -> GELU -> Linear(mlp_dim, embed_dim)`.
pub struct TransformerBlock<T: Float> {
    norm1: LayerNorm<T>,
    attn: MultiheadAttention<T>,
    norm2: LayerNorm<T>,
    mlp_fc1: Linear<T>,
    mlp_fc2: Linear<T>,
    gelu: GELU,
    training: bool,
}

impl<T: Float> TransformerBlock<T> {
    /// Create a new transformer encoder block.
    ///
    /// * `embed_dim` -- hidden dimension / embedding dimension.
    /// * `num_heads` -- number of attention heads.
    /// * `mlp_dim` -- intermediate dimension of the MLP (typically `4 * embed_dim`).
    pub fn new(embed_dim: usize, num_heads: usize, mlp_dim: usize) -> FerrotorchResult<Self> {
        let norm1 = LayerNorm::new(vec![embed_dim], 1e-6, true)?;
        let attn = MultiheadAttention::new(embed_dim, num_heads, true)?;
        let norm2 = LayerNorm::new(vec![embed_dim], 1e-6, true)?;
        let mlp_fc1 = Linear::new(embed_dim, mlp_dim, true)?;
        let mlp_fc2 = Linear::new(mlp_dim, embed_dim, true)?;
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

    /// Forward pass for the MLP sub-block.
    ///
    /// Linear accepts arbitrary leading dims (it flattens to 2-D internally
    /// and reshapes back), so this is rank-polymorphic. Pass 2B (#1000)
    /// migrated callers from per-batch slicing to a single 3-D dispatch.
    ///
    /// Input: `[*, embed_dim]` — typically `[B, seq_len, embed_dim]`
    /// Output: same leading dims as input, last dim `embed_dim`.
    fn mlp_forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.mlp_fc1.forward(input)?;
        let x = self.gelu.forward(&x)?;
        self.mlp_fc2.forward(&x)
    }
}

impl<T: Float> Module<T> for TransformerBlock<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // input: [B, seq_len, embed_dim]

        // --- Sub-block 1: LayerNorm -> MultiheadAttention -> residual add ---
        let normed1 = self.norm1.forward(input)?;
        let attn_out = self.attn.forward(&normed1)?;
        let x = add(input, &attn_out)?;

        // --- Sub-block 2: LayerNorm -> MLP -> residual add ---
        // Pass 2B (#1000): Linear accepts arbitrary leading dims, so the MLP
        // takes the 3-D `[B, seq_len, embed_dim]` tensor directly. The prior
        // implementation pulled `normed2` to host RAM, sliced per batch,
        // re-uploaded to device, ran 2-D MLP, then concatenated — a
        // round-trip per block. The 3-D dispatch stays on `input.device()`
        // end-to-end and preserves grad_fn through `mm_differentiable`.
        let normed2 = self.norm2.forward(&x)?;
        let mlp_out = self.mlp_forward(&normed2)?;

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

    // Phase 4 (#995): mirror the `named_parameters` paths above so a
    // Phase 2-style walk reaches every sub-module under each block —
    // `norm1` / `attn` / `norm2` / `mlp.fc1` / `mlp.fc2`.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.norm1,
            &self.attn,
            &self.norm2,
            &self.mlp_fc1,
            &self.mlp_fc2,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("norm1".to_string(), &self.norm1),
            ("attn".to_string(), &self.attn),
            ("norm2".to_string(), &self.norm2),
            ("mlp.fc1".to_string(), &self.mlp_fc1),
            ("mlp.fc2".to_string(), &self.mlp_fc2),
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

// ===========================================================================
// VisionTransformer
// ===========================================================================

/// Vision Transformer (ViT) model.
///
/// The full architecture:
///
/// 1. **Patch embedding**: `Conv2d(3, embed_dim, patch_size, patch_size)` splits
///    the image into non-overlapping patches.
/// 2. **CLS token**: A learnable `[1, 1, embed_dim]` token prepended to the
///    patch sequence.
/// 3. **Position embedding**: A learnable `[1, num_patches+1, embed_dim]` tensor
///    added to the sequence.
/// 4. **Transformer encoder**: `depth` transformer blocks.
/// 5. **Classification head**: `LayerNorm -> extract CLS -> Linear(embed_dim, num_classes)`.
pub struct VisionTransformer<T: Float> {
    patch_embed: PatchEmbed<T>,
    cls_token: Parameter<T>,
    pos_embed: Parameter<T>,
    blocks: Vec<TransformerBlock<T>>,
    norm: LayerNorm<T>,
    head: Linear<T>,
    embed_dim: usize,
    /// Number of patches the input image is split into.
    ///
    /// Recorded at construction so the constructor can size `pos_embed` to
    /// `[1, num_patches+1, embed_dim]`. After Pass 2B (#1000) the forward
    /// path no longer reads this directly — `cat`/`expand` derive the
    /// sequence length from the patch tensor itself — but it remains part
    /// of the model's documented architecture for inspection/debugging.
    #[allow(dead_code)]
    num_patches: usize,
    training: bool,
}

impl<T: Float> VisionTransformer<T> {
    /// Create a new Vision Transformer.
    ///
    /// * `image_size` -- expected input spatial size (assumes square images).
    /// * `patch_size` -- spatial size of each patch.
    /// * `in_channels` -- number of input channels (typically 3 for RGB).
    /// * `num_classes` -- number of output classes.
    /// * `embed_dim` -- hidden dimension.
    /// * `depth` -- number of transformer encoder blocks.
    /// * `num_heads` -- number of attention heads per block.
    /// * `mlp_ratio` -- ratio of MLP hidden dim to embed_dim.
    // justification: ViT constructor mirrors the original Dosovitskiy et al.
    // hyperparameter set; bundling into a config struct hides the canonical
    // [image_size, patch_size, in_ch, num_classes, embed_dim, depth,
    // num_heads, mlp_ratio] ordering shared with timm/torchvision call sites.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        num_classes: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: usize,
    ) -> FerrotorchResult<Self> {
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let mlp_dim = embed_dim * mlp_ratio;

        let patch_embed = PatchEmbed::new(in_channels, embed_dim, patch_size)?;
        let cls_token = Parameter::zeros(&[1, 1, embed_dim])?;
        let pos_embed = Parameter::zeros(&[1, num_patches + 1, embed_dim])?;

        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(TransformerBlock::new(embed_dim, num_heads, mlp_dim)?);
        }

        let norm = LayerNorm::new(vec![embed_dim], 1e-6, true)?;
        let head = Linear::new(embed_dim, num_classes, true)?;

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
            embed_dim,
            num_patches,
            training: true,
        })
    }

    /// Total number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<T: Float> Module<T> for VisionTransformer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let batch = input.shape()[0];

        // 1. Patch embedding: [B, 3, H, W] -> [B, num_patches, embed_dim]
        let x = self.patch_embed.forward(input)?;

        // 2. Prepend CLS token: [B, num_patches, embed_dim] -> [B, num_patches+1, embed_dim]
        // Pass 2B (#1000): broadcast cls_token from `[1, 1, embed_dim]` to
        // `[B, 1, embed_dim]` via `expand`, then `cat` along axis=1. Stays
        // on `input.device()` end-to-end and preserves grad_fn so the
        // `cls_token` Parameter accumulates gradient through CatBackward.
        let cls_expanded = expand(self.cls_token.tensor(), &[batch, 1, self.embed_dim])?;
        let x = cat(&[cls_expanded, x], 1)?;

        // 3. Add position embedding: [B, seq_len, embed_dim] + [1, seq_len, embed_dim]
        // Pass 2B (#1000): `add` broadcasts the leading batch dim natively
        // (BroadcastAddBackward reduces along it for pos_embed.grad). No
        // host detour, autograd-correct.
        let x = add(&x, self.pos_embed.tensor())?;

        // 4. Transformer encoder blocks.
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // 5. Final LayerNorm.
        let x = self.norm.forward(&x)?;

        // 6. Extract CLS token: take [:, 0, :] -> [B, embed_dim]
        // Pass 2B (#1000): `narrow(dim=1, start=0, length=1)` → `squeeze(1)`
        // is the device-resident analog of the prior `data_vec()`+slice
        // loop. NarrowBackward fills the unsliced positions with zeros on
        // backward, matching torchvision's slice semantics.
        let cls_features = x.narrow(1, 0, 1)?.squeeze_t(1)?;

        // 7. Classification head: Linear(embed_dim, num_classes)
        self.head.forward(&cls_features)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters());
        params.push(&self.cls_token);
        params.push(&self.pos_embed);
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.norm.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters_mut());
        params.push(&mut self.cls_token);
        params.push(&mut self.pos_embed);
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
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
        params.push(("cls_token".to_string(), &self.cls_token));
        params.push(("pos_embed".to_string(), &self.pos_embed));

        for (i, block) in self.blocks.iter().enumerate() {
            for (name, p) in block.named_parameters() {
                params.push((format!("blocks.{i}.{name}"), p));
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

    // Phase 4 (#995): expose patch_embed + every TransformerBlock + the
    // final norm/head so the BN-buffer-style loader can walk into each
    // block's `norm1` / `attn` / `mlp.fc1` paths. ViT has no BN buffers
    // so the override is a no-op for the resnet50 fixture path, but the
    // ViT remap test (#999) uses `named_descendants_dyn` to locate
    // sub-modules indirectly via `parameters_mut`-keyed traversal.
    // Following torchvision-shaped paths: `patch_embed` / `blocks.<i>` /
    // `norm` / `head` (matches `named_parameters` above).
    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![&self.patch_embed];
        for block in &self.blocks {
            out.push(block);
        }
        out.push(&self.norm);
        out.push(&self.head);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> =
            vec![("patch_embed".to_string(), &self.patch_embed)];
        for (i, block) in self.blocks.iter().enumerate() {
            out.push((format!("blocks.{i}"), block));
        }
        out.push(("norm".to_string(), &self.norm));
        out.push(("head".to_string(), &self.head));
        out
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

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for VisionTransformer<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();
        let batch = input.shape()[0];

        // 1. Patch embedding.
        let x = self.patch_embed.forward(input)?;
        out.insert("patch_embed".to_string(), x.clone());

        // 2. Prepend CLS token. (Pass 2B #1000 — see `forward` for rationale.)
        let cls_expanded = expand(self.cls_token.tensor(), &[batch, 1, self.embed_dim])?;
        let x = cat(&[cls_expanded, x], 1)?;

        // 3. Add position embedding (broadcasts the leading batch dim).
        let mut x = add(&x, self.pos_embed.tensor())?;
        out.insert("embedded".to_string(), x.clone());

        // 4. Transformer encoder blocks.
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            out.insert(format!("block{i}"), x.clone());
        }

        // 5. Final LayerNorm.
        let x = self.norm.forward(&x)?;
        out.insert("norm".to_string(), x.clone());

        // 6. Extract CLS token: [:, 0, :] -> [B, embed_dim].
        let cls_features = x.narrow(1, 0, 1)?.squeeze_t(1)?;

        // 7. Classification head.
        let logits = self.head.forward(&cls_features)?;
        out.insert("head".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        let mut names = vec!["patch_embed".to_string(), "embedded".to_string()];
        for i in 0..self.blocks.len() {
            names.push(format!("block{i}"));
        }
        names.push("norm".to_string());
        names.push("head".to_string());
        names
    }
}

/// Construct a ViT-B/16 model.
///
/// Architecture:
/// - Patch size: 16x16
/// - Embedding dimension: 768
/// - Depth: 12 transformer blocks
/// - Heads: 12 attention heads
/// - MLP ratio: 4 (MLP hidden dim = 3072)
/// - Image size: 224x224
///
/// Total parameters: ~86M (for 1000 classes).
pub fn vit_b_16<T: Float>(num_classes: usize) -> FerrotorchResult<VisionTransformer<T>> {
    VisionTransformer::new(
        224, // image_size
        16,  // patch_size
        3,   // in_channels
        num_classes,
        768, // embed_dim
        12,  // depth
        12,  // num_heads
        4,   // mlp_ratio
    )
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
    // PatchEmbed tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_patch_embed_output_shape() {
        let pe = PatchEmbed::<f32>::new(3, 768, 16).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| pe.forward(&input).unwrap());
        // 224/16 = 14, 14*14 = 196 patches
        assert_eq!(output.shape(), &[1, 196, 768]);
    }

    #[test]
    fn test_patch_embed_batch_2() {
        let pe = PatchEmbed::<f32>::new(3, 64, 16).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 32 * 32], [2, 3, 32, 32], false);
        let output = no_grad(|| pe.forward(&input).unwrap());
        // 32/16 = 2, 2*2 = 4 patches
        assert_eq!(output.shape(), &[2, 4, 64]);
    }

    #[test]
    fn test_patch_embed_parameter_count() {
        let pe = PatchEmbed::<f32>::new(3, 768, 16).unwrap();
        let count: usize = pe.parameters().iter().map(|p| p.numel()).sum();
        // Conv2d(3, 768, 16, 16) weight: 768 * 3 * 16 * 16 = 589824
        // plus Conv2d bias: 768 (Phase 4 #999/#1001 — bias=true now
        // matches torchvision conv_proj's default).
        // Total: 589824 + 768 = 590592.
        assert_eq!(count, 768 * 3 * 16 * 16 + 768);
    }

    // -----------------------------------------------------------------------
    // TransformerBlock tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_transformer_block_output_shape() {
        let block = TransformerBlock::<f32>::new(64, 4, 256).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 10 * 64]),
            vec![1, 10, 64],
            false,
        )
        .unwrap();
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 10, 64]);
    }

    #[test]
    fn test_transformer_block_parameter_count() {
        let embed_dim = 64;
        let num_heads = 4;
        let mlp_dim = 256;
        let block = TransformerBlock::<f32>::new(embed_dim, num_heads, mlp_dim).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();

        // LayerNorm: 2 * (weight + bias) = 2 * 2 * embed_dim = 256
        let ln_params = 2 * 2 * embed_dim;
        // MHA: 4 * (embed_dim * embed_dim + embed_dim) = 4 * (4096 + 64) = 16640
        let mha_params = 4 * (embed_dim * embed_dim + embed_dim);
        // MLP: fc1 weight + bias + fc2 weight + bias
        //    = embed_dim * mlp_dim + mlp_dim + mlp_dim * embed_dim + embed_dim
        //    = 64*256 + 256 + 256*64 + 64 = 16384 + 256 + 16384 + 64 = 33088
        let mlp_params = embed_dim * mlp_dim + mlp_dim + mlp_dim * embed_dim + embed_dim;

        let expected = ln_params + mha_params + mlp_params;
        assert_eq!(count, expected);
    }

    // -----------------------------------------------------------------------
    // VisionTransformer (ViT-B/16) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vit_b16_output_shape() {
        let model = vit_b_16::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_vit_b16_param_count() {
        let model = vit_b_16::<f32>(1000).unwrap();
        let total = model.num_parameters();
        // ViT-B/16 has ~86M parameters.
        // Exact breakdown:
        //   patch_embed: 768 * 3 * 16 * 16 = 589,824
        //   cls_token: 768
        //   pos_embed: 197 * 768 = 151,296
        //   12 blocks, each:
        //     2 * LayerNorm(768): 2 * 2 * 768 = 3,072
        //     MHA(768, 12, bias): 4 * (768*768 + 768) = 2,362,368
        //     MLP(768->3072->768, bias): 768*3072 + 3072 + 3072*768 + 768 = 4,722,432
        //     block total: 3,072 + 2,362,368 + 4,722,432 = 7,087,872
        //   12 blocks total: 85,054,464
        //   final LayerNorm(768): 1,536
        //   head Linear(768, 1000, bias): 768*1000 + 1000 = 769,000
        //   Grand total: 589,824 + 768 + 151,296 + 85,054,464 + 1,536 + 769,000 = 86,566,888
        assert!(
            total > 80_000_000,
            "ViT-B/16 should have >80M params, got {total}"
        );
        assert!(
            total < 90_000_000,
            "ViT-B/16 should have <90M params, got {total}"
        );
    }

    #[test]
    fn test_vit_cls_token_shape() {
        let model = vit_b_16::<f32>(1000).unwrap();
        assert_eq!(model.cls_token.shape(), &[1, 1, 768]);
    }

    #[test]
    fn test_vit_pos_embed_shape() {
        let model = vit_b_16::<f32>(1000).unwrap();
        // 196 patches + 1 CLS = 197
        assert_eq!(model.pos_embed.shape(), &[1, 197, 768]);
    }

    #[test]
    fn test_vit_named_parameters_prefixes() {
        let model = vit_b_16::<f32>(1000).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("patch_embed.")));
        assert!(names.contains(&"cls_token"));
        assert!(names.contains(&"pos_embed"));
        assert!(names.iter().any(|n| n.starts_with("blocks.0.")));
        assert!(names.iter().any(|n| n.starts_with("blocks.11.")));
        assert!(names.iter().any(|n| n.starts_with("norm.")));
        assert!(names.iter().any(|n| n.starts_with("head.")));
    }

    #[test]
    fn test_vit_custom_classes() {
        let model = VisionTransformer::<f32>::new(224, 16, 3, 10, 768, 12, 12, 4).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 10]);
    }

    #[test]
    fn test_vit_train_eval() {
        let mut model = vit_b_16::<f32>(1000).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }

    #[test]
    fn test_vit_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VisionTransformer<f32>>();
        assert_send_sync::<PatchEmbed<f32>>();
        assert_send_sync::<TransformerBlock<f32>>();
    }

    // -----------------------------------------------------------------------
    // Small ViT forward pass (fast test)
    // -----------------------------------------------------------------------

    #[test]
    fn test_small_vit_forward() {
        // A tiny ViT for fast testing.
        let model = VisionTransformer::<f32>::new(
            32, // image_size
            16, // patch_size -> 2x2 = 4 patches
            3,  // in_channels
            10, // num_classes
            64, // embed_dim
            2,  // depth (just 2 blocks)
            4,  // num_heads
            4,  // mlp_ratio -> mlp_dim = 256
        )
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

    // -----------------------------------------------------------------------
    // Pass 2B (#1000) autograd preservation probe
    //
    // After the migration the cls_token concat path is `expand` + `cat` (no
    // `data_vec()` round-trip). Both ops have grad_fns, so the cls_token
    // Parameter must still receive a non-zero gradient on backward.
    // -----------------------------------------------------------------------

    #[test]
    fn test_vit_cls_token_grad_flows_through_cat() {
        use ferrotorch_core::grad_fns::reduction::sum;

        let model = VisionTransformer::<f32>::new(
            32, // image_size
            16, // patch_size
            3,  // in_channels
            10, // num_classes
            64, // embed_dim
            1,  // depth (one block)
            4,  // num_heads
            4,  // mlp_ratio
        )
        .unwrap();

        // Input: requires_grad=true so autograd graph is constructed
        // (cls_token is itself a Parameter so always has requires_grad=true).
        let input = leaf_4d(&vec![0.1; 3 * 32 * 32], [1, 3, 32, 32], true);
        let output = model.forward(&input).unwrap();
        let loss = sum(&output).unwrap();
        loss.backward().unwrap();

        // cls_token must have a non-None gradient (was reachable through
        // ExpandBackward -> CatBackward in the graph).
        let cls_grad = model
            .cls_token
            .grad()
            .expect("cls_token grad accessor")
            .expect("cls_token must have a gradient after backward");
        assert_eq!(cls_grad.shape(), model.cls_token.shape());

        // And at least one element of the gradient must be non-zero — a
        // zero-only grad would be indistinguishable from the path being
        // detached from autograd.
        let g = cls_grad.data().unwrap();
        let any_nonzero = g.iter().any(|x| x.abs() > 0.0);
        assert!(
            any_nonzero,
            "cls_token gradient was all zeros — autograd path appears detached"
        );
    }
}
