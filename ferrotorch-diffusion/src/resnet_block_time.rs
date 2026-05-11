//! `ResnetBlock2DTime` — the time-conditioned variant of the
//! `ResnetBlock2D` used by the SD UNet.
//!
//! Identical to `diffusers.models.resnet.ResnetBlock2D` configured with
//! `temb_channels = 1280` and the default `time_embedding_norm = "default"`
//! (the SD-1.5 setting):
//!
//! ```text
//! h = silu(norm1(x)); h = conv1(h)
//! t = silu(temb); t = time_emb_proj(t).view(B, out, 1, 1)
//! h = h + t
//! h = silu(norm2(h)); h = conv2(h)
//! r = x if in==out else conv_shortcut(x)
//! out = h + r       (output_scale_factor = 1.0)
//! ```
//!
//! State-dict layout (matches diffusers):
//!
//! ```text
//! norm1.{weight,bias}
//! conv1.{weight,bias}
//! time_emb_proj.{weight,bias}    [out_channels, temb_channels]
//! norm2.{weight,bias}
//! conv2.{weight,bias}
//! conv_shortcut.{weight,bias}    (iff in_channels != out_channels)
//! ```

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, GroupNorm, Linear, SiLU};

/// Time-conditioned residual block.
#[derive(Debug)]
pub struct ResnetBlock2DTime<T: Float> {
    /// First GroupNorm.
    pub norm1: GroupNorm<T>,
    /// First Conv2d.
    pub conv1: Conv2d<T>,
    /// Linear over the time embedding (`temb_channels -> out_channels`).
    pub time_emb_proj: Linear<T>,
    /// Second GroupNorm.
    pub norm2: GroupNorm<T>,
    /// Second Conv2d.
    pub conv2: Conv2d<T>,
    /// Optional 1x1 shortcut.
    pub conv_shortcut: Option<Conv2d<T>>,
    activation: SiLU,
    in_channels: usize,
    out_channels: usize,
    training: bool,
}

impl<T: Float> ResnetBlock2DTime<T> {
    /// Build a randomly-initialized time-conditioned resnet block.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad channel/group
    /// config.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        norm_num_groups: usize,
        eps: f64,
    ) -> FerrotorchResult<Self> {
        let norm1 = GroupNorm::<T>::new(norm_num_groups, in_channels, eps, true)?;
        let conv1 = Conv2d::<T>::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), true)?;
        let time_emb_proj = Linear::<T>::new(temb_channels, out_channels, true)?;
        let norm2 = GroupNorm::<T>::new(norm_num_groups, out_channels, eps, true)?;
        let conv2 = Conv2d::<T>::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv_shortcut = if in_channels == out_channels {
            None
        } else {
            Some(Conv2d::<T>::new(
                in_channels,
                out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            conv_shortcut,
            activation: SiLU::new(),
            in_channels,
            out_channels,
            training: false,
        })
    }

    /// Forward with the time embedding `temb` (shape `[B, temb_channels]`).
    ///
    /// `x` has shape `[B, in_channels, H, W]`; output is
    /// `[B, out_channels, H, W]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] for bad input ranks.
    pub fn forward_t(&self, x: &Tensor<T>, temb: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if x.ndim() != 4 || x.shape()[1] != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ResnetBlock2DTime: expected x [B, {}, H, W], got {:?}",
                    self.in_channels,
                    x.shape()
                ),
            });
        }
        if temb.ndim() != 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ResnetBlock2DTime: expected temb [B, temb_channels], got {:?}",
                    temb.shape()
                ),
            });
        }
        let b = x.shape()[0];
        // h = silu(norm1(x)); h = conv1(h)
        let mut h = self.norm1.forward(x)?;
        h = self.activation.forward(&h)?;
        h = self.conv1.forward(&h)?;
        // Time bias: silu(temb) -> Linear -> [B, out_channels, 1, 1]
        let temb_silu = self.activation.forward(temb)?;
        let temb_proj = self.time_emb_proj.forward(&temb_silu)?;
        let temb_4d = temb_proj.reshape_t(&[
            b as isize,
            self.out_channels as isize,
            1,
            1,
        ])?;
        h = ferrotorch_core::grad_fns::arithmetic::add(&h, &temb_4d)?;
        // h = silu(norm2(h)); h = conv2(h)
        h = self.norm2.forward(&h)?;
        h = self.activation.forward(&h)?;
        h = self.conv2.forward(&h)?;
        // Residual.
        let res = if let Some(sc) = &self.conv_shortcut {
            sc.forward(x)?
        } else {
            x.clone()
        };
        ferrotorch_core::grad_fns::arithmetic::add(&h, &res)
    }
}

impl<T: Float> Module<T> for ResnetBlock2DTime<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "ResnetBlock2DTime::forward: time-conditioned block requires \
                      a time embedding — call forward_t instead"
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.norm1.parameters());
        o.extend(self.conv1.parameters());
        o.extend(self.time_emb_proj.parameters());
        o.extend(self.norm2.parameters());
        o.extend(self.conv2.parameters());
        if let Some(sc) = &self.conv_shortcut {
            o.extend(sc.parameters());
        }
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.norm1.parameters_mut());
        o.extend(self.conv1.parameters_mut());
        o.extend(self.time_emb_proj.parameters_mut());
        o.extend(self.norm2.parameters_mut());
        o.extend(self.conv2.parameters_mut());
        if let Some(sc) = self.conv_shortcut.as_mut() {
            o.extend(sc.parameters_mut());
        }
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (n, p) in self.norm1.named_parameters() {
            o.push((format!("norm1.{n}"), p));
        }
        for (n, p) in self.conv1.named_parameters() {
            o.push((format!("conv1.{n}"), p));
        }
        for (n, p) in self.time_emb_proj.named_parameters() {
            o.push((format!("time_emb_proj.{n}"), p));
        }
        for (n, p) in self.norm2.named_parameters() {
            o.push((format!("norm2.{n}"), p));
        }
        for (n, p) in self.conv2.named_parameters() {
            o.push((format!("conv2.{n}"), p));
        }
        if let Some(sc) = &self.conv_shortcut {
            for (n, p) in sc.named_parameters() {
                o.push((format!("conv_shortcut.{n}"), p));
            }
        }
        o
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
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                let ok = k.starts_with("norm1.")
                    || k.starts_with("conv1.")
                    || k.starts_with("time_emb_proj.")
                    || k.starts_with("norm2.")
                    || k.starts_with("conv2.")
                    || k.starts_with("conv_shortcut.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in ResnetBlock2DTime state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        self.norm1.load_state_dict(&extract("norm1"), strict)?;
        self.conv1.load_state_dict(&extract("conv1"), strict)?;
        self.time_emb_proj
            .load_state_dict(&extract("time_emb_proj"), strict)?;
        self.norm2.load_state_dict(&extract("norm2"), strict)?;
        self.conv2.load_state_dict(&extract("conv2"), strict)?;
        if let Some(sc) = self.conv_shortcut.as_mut() {
            sc.load_state_dict(&extract("conv_shortcut"), strict)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    #[test]
    fn resnet_time_shape_same_channels() {
        let r = ResnetBlock2DTime::<f32>::new(16, 16, 32, 4, 1e-5).unwrap();
        assert!(r.conv_shortcut.is_none());
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 16 * 4 * 4]),
            vec![1, 16, 4, 4],
            false,
        )
        .unwrap();
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 32]),
            vec![1, 32],
            false,
        )
        .unwrap();
        let y = r.forward_t(&x, &t).unwrap();
        assert_eq!(y.shape(), &[1, 16, 4, 4]);
    }

    #[test]
    fn resnet_time_shape_change_channels() {
        let r = ResnetBlock2DTime::<f32>::new(16, 32, 32, 4, 1e-5).unwrap();
        assert!(r.conv_shortcut.is_some());
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 16 * 4 * 4]),
            vec![1, 16, 4, 4],
            false,
        )
        .unwrap();
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 32]),
            vec![1, 32],
            false,
        )
        .unwrap();
        let y = r.forward_t(&x, &t).unwrap();
        assert_eq!(y.shape(), &[1, 32, 4, 4]);
    }

    #[test]
    fn resnet_time_named_parameters() {
        let r = ResnetBlock2DTime::<f32>::new(16, 32, 32, 4, 1e-5).unwrap();
        let names: Vec<String> = r.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "norm1.weight",
            "conv1.weight",
            "time_emb_proj.weight",
            "time_emb_proj.bias",
            "norm2.weight",
            "conv2.weight",
            "conv_shortcut.weight",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }
}
