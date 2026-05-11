//! Configuration for the Stable-Diffusion UNet2DConditionModel.
//!
//! Mirrors the public surface of
//! `diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.config`
//! for the fields the UNet's forward pass consumes. Encoder-side and
//! training-only fields are not stored.

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// Frozen config for the SD-1.5 UNet2DConditionModel.
///
/// Default = the runwayml/stable-diffusion-v1-5 unet/config.json values.
#[derive(Debug, Clone)]
pub struct UNet2DConditionConfig {
    /// Input latent channels (4).
    pub in_channels: usize,
    /// Output channels (4).
    pub out_channels: usize,
    /// Per-resolution channel counts, encoder order. SD-1.5: `[320, 640, 1280, 1280]`.
    pub block_out_channels: Vec<usize>,
    /// Resnets per resolution (SD-1.5: 2).
    pub layers_per_block: usize,
    /// Attention head dim — diffusers `attention_head_dim`. Either a
    /// single int (per-block num_heads = channels / dim_head) or a list.
    /// SD-1.5: `8`.
    pub attention_head_dim: usize,
    /// Cross-attention embedding dim (SD-1.5: 768).
    pub cross_attention_dim: usize,
    /// GroupNorm group count (SD-1.5: 32).
    pub norm_num_groups: usize,
    /// Encoder spatial size (SD-1.5: 64).
    pub sample_size: usize,
    /// Whether the sinusoidal timestep encoding flips the cos/sin order
    /// (SD-1.5: true).
    pub flip_sin_to_cos: bool,
    /// `downscale_freq_shift` of the sinusoidal encoding (SD-1.5: 0).
    pub freq_shift: f64,
    /// Per-block `transformer_layers_per_block`. SD-1.5: 1 everywhere
    /// (default when absent from config.json).
    pub transformer_layers_per_block: usize,
    /// Which down-blocks have cross-attention. SD-1.5:
    /// `[CrossAttnDownBlock2D, CrossAttnDownBlock2D,
    ///   CrossAttnDownBlock2D, DownBlock2D]` → `[true, true, true, false]`.
    pub down_block_has_attn: Vec<bool>,
    /// Which up-blocks have cross-attention. SD-1.5:
    /// `[UpBlock2D, CrossAttnUpBlock2D, CrossAttnUpBlock2D,
    ///   CrossAttnUpBlock2D]` → `[false, true, true, true]`.
    pub up_block_has_attn: Vec<bool>,
}

impl Default for UNet2DConditionConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            attention_head_dim: 8,
            cross_attention_dim: 768,
            norm_num_groups: 32,
            sample_size: 64,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            transformer_layers_per_block: 1,
            down_block_has_attn: vec![true, true, true, false],
            up_block_has_attn: vec![false, true, true, true],
        }
    }
}

impl UNet2DConditionConfig {
    /// SD-1.5 UNet config (alias for `Default::default()`).
    pub fn sd_v1_5() -> Self {
        Self::default()
    }

    /// Validate field bounds and shape compatibility.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for any out-of-bounds
    /// or arithmetic-incompatible field.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.block_out_channels.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "UNet2DConditionConfig: block_out_channels must be non-empty".into(),
            });
        }
        if self.norm_num_groups == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "UNet2DConditionConfig: norm_num_groups must be > 0".into(),
            });
        }
        for &c in &self.block_out_channels {
            if c == 0 || c % self.norm_num_groups != 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "UNet2DConditionConfig: block_out_channels entry {c} must be > 0 and \
                         divisible by norm_num_groups={}",
                        self.norm_num_groups
                    ),
                });
            }
            if c % self.attention_head_dim != 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "UNet2DConditionConfig: block_out_channels entry {c} must be divisible \
                         by attention_head_dim={}",
                        self.attention_head_dim
                    ),
                });
            }
        }
        if self.in_channels == 0
            || self.out_channels == 0
            || self.layers_per_block == 0
            || self.cross_attention_dim == 0
            || self.attention_head_dim == 0
            || self.sample_size == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "UNet2DConditionConfig: a positive-only field was zero".into(),
            });
        }
        if self.down_block_has_attn.len() != self.block_out_channels.len()
            || self.up_block_has_attn.len() != self.block_out_channels.len()
        {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "UNet2DConditionConfig: down/up_block_has_attn must have the same length as \
                     block_out_channels ({} blocks), got down={} up={}",
                    self.block_out_channels.len(),
                    self.down_block_has_attn.len(),
                    self.up_block_has_attn.len(),
                ),
            });
        }
        Ok(())
    }

    /// Time embedding dim (`block_out_channels[0] * 4`). For SD-1.5 this
    /// is `1280`.
    pub fn time_embed_dim(&self) -> usize {
        self.block_out_channels[0] * 4
    }

    /// Number of (down|up) blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_out_channels.len()
    }

    /// Parse a `unet/config.json` document into a `UNet2DConditionConfig`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] on malformed JSON or
    /// any wrong-type field.
    pub fn from_json_str(s: &str) -> FerrotorchResult<Self> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| {
            FerrotorchError::InvalidArgument {
                message: format!("UNet2DConditionConfig::from_json_str: bad JSON: {e}"),
            }
        })?;
        let mut cfg = Self::default();
        if let Some(x) = v.get("in_channels").and_then(serde_json::Value::as_u64) {
            cfg.in_channels = x as usize;
        }
        if let Some(x) = v.get("out_channels").and_then(serde_json::Value::as_u64) {
            cfg.out_channels = x as usize;
        }
        if let Some(arr) = v.get("block_out_channels").and_then(serde_json::Value::as_array) {
            let mut out = Vec::with_capacity(arr.len());
            for e in arr {
                let n = e.as_u64().ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "UNet2DConditionConfig::from_json_str: block_out_channels entry \
                         must be a non-negative integer, got {e}"
                    ),
                })?;
                out.push(n as usize);
            }
            cfg.block_out_channels = out;
        }
        if let Some(x) = v.get("layers_per_block").and_then(serde_json::Value::as_u64) {
            cfg.layers_per_block = x as usize;
        }
        if let Some(x) = v.get("attention_head_dim").and_then(serde_json::Value::as_u64) {
            cfg.attention_head_dim = x as usize;
        }
        if let Some(x) = v.get("cross_attention_dim").and_then(serde_json::Value::as_u64) {
            cfg.cross_attention_dim = x as usize;
        }
        if let Some(x) = v.get("norm_num_groups").and_then(serde_json::Value::as_u64) {
            cfg.norm_num_groups = x as usize;
        }
        if let Some(x) = v.get("sample_size").and_then(serde_json::Value::as_u64) {
            cfg.sample_size = x as usize;
        }
        if let Some(x) = v.get("flip_sin_to_cos").and_then(serde_json::Value::as_bool) {
            cfg.flip_sin_to_cos = x;
        }
        if let Some(x) = v.get("freq_shift").and_then(serde_json::Value::as_f64) {
            cfg.freq_shift = x;
        }
        if let Some(x) = v
            .get("transformer_layers_per_block")
            .and_then(serde_json::Value::as_u64)
        {
            cfg.transformer_layers_per_block = x as usize;
        }
        // Down/up block types: derive has_attn from CrossAttn* prefix.
        if let Some(arr) = v.get("down_block_types").and_then(serde_json::Value::as_array) {
            let mut out = Vec::with_capacity(arr.len());
            for e in arr {
                let s = e.as_str().ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "UNet2DConditionConfig::from_json_str: down_block_types entry must \
                         be a string, got {e}"
                    ),
                })?;
                out.push(match s {
                    "CrossAttnDownBlock2D" => true,
                    "DownBlock2D" => false,
                    other => {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "UNet2DConditionConfig::from_json_str: unsupported down_block_type \
                                 '{other}'. Only CrossAttnDownBlock2D / DownBlock2D are recognised."
                            ),
                        });
                    }
                });
            }
            cfg.down_block_has_attn = out;
        }
        if let Some(arr) = v.get("up_block_types").and_then(serde_json::Value::as_array) {
            let mut out = Vec::with_capacity(arr.len());
            for e in arr {
                let s = e.as_str().ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "UNet2DConditionConfig::from_json_str: up_block_types entry must \
                         be a string, got {e}"
                    ),
                })?;
                out.push(match s {
                    "CrossAttnUpBlock2D" => true,
                    "UpBlock2D" => false,
                    other => {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "UNet2DConditionConfig::from_json_str: unsupported up_block_type \
                                 '{other}'. Only CrossAttnUpBlock2D / UpBlock2D are recognised."
                            ),
                        });
                    }
                });
            }
            cfg.up_block_has_attn = out;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a `unet/config.json` file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for I/O or parse
    /// failures.
    pub fn from_file(path: &std::path::Path) -> FerrotorchResult<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "UNet2DConditionConfig::from_file: failed to read {}: {e}",
                path.display()
            ),
        })?;
        Self::from_json_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_sd_v1_5() {
        let c = UNet2DConditionConfig::default();
        assert_eq!(c.block_out_channels, vec![320, 640, 1280, 1280]);
        assert_eq!(c.layers_per_block, 2);
        assert_eq!(c.attention_head_dim, 8);
        assert_eq!(c.cross_attention_dim, 768);
        assert_eq!(c.norm_num_groups, 32);
        assert_eq!(c.sample_size, 64);
        assert_eq!(c.time_embed_dim(), 1280);
        assert_eq!(c.down_block_has_attn, vec![true, true, true, false]);
        assert_eq!(c.up_block_has_attn, vec![false, true, true, true]);
        c.validate().unwrap();
    }

    #[test]
    fn from_json_parses_block_types() {
        let json = r#"{
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": [320, 640, 1280, 1280],
            "layers_per_block": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 768,
            "norm_num_groups": 32,
            "sample_size": 64,
            "flip_sin_to_cos": true,
            "freq_shift": 0,
            "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                 "CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D",
                                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]
        }"#;
        let c = UNet2DConditionConfig::from_json_str(json).unwrap();
        assert_eq!(c.down_block_has_attn, vec![true, true, true, false]);
        assert_eq!(c.up_block_has_attn, vec![false, true, true, true]);
    }
}
