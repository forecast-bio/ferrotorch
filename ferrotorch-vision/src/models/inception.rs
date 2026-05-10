//! Inception v3 architecture — full torchvision parity (Phase 10, #993, #1012).
//!
//! Follows Szegedy et al. 2016, "Rethinking the Inception Architecture for
//! Computer Vision," and matches torchvision's
//! `torchvision.models.inception_v3(aux_logits=False)` parameter layout
//! exactly.
//!
//! Layer dictionary (torchvision-exact field names):
//! ```text
//!   Conv2d_1a_3x3 :  3 → 32, k=3, s=2          (BasicConv2d)
//!   Conv2d_2a_3x3 : 32 → 32, k=3
//!   Conv2d_2b_3x3 : 32 → 64, k=3, padding=1
//!   maxpool1      : MaxPool2d(3, 2)
//!   Conv2d_3b_1x1 : 64 → 80, k=1
//!   Conv2d_4a_3x3 : 80 → 192, k=3
//!   maxpool2      : MaxPool2d(3, 2)
//!   Mixed_5b      : InceptionA(192, pool_features=32)
//!   Mixed_5c      : InceptionA(256, pool_features=64)
//!   Mixed_5d      : InceptionA(288, pool_features=64)
//!   Mixed_6a      : InceptionB(288)            — reduction
//!   Mixed_6b      : InceptionC(768, channels_7x7=128)
//!   Mixed_6c      : InceptionC(768, channels_7x7=160)
//!   Mixed_6d      : InceptionC(768, channels_7x7=160)
//!   Mixed_6e      : InceptionC(768, channels_7x7=192)
//!   Mixed_7a      : InceptionD(768)            — reduction
//!   Mixed_7b      : InceptionE(1280)
//!   Mixed_7c      : InceptionE(2048)
//!   avgpool       : AdaptiveAvgPool2d((1, 1))
//!   dropout       : Dropout(p=0.5)             — identity in eval
//!   fc            : Linear(2048, num_classes)
//! ```
//!
//! `BasicConv2d` is a `Conv2d(bias=false) + BatchNorm2d(eps=1e-3) + ReLU`
//! triple. Children render as `conv` + `bn`, matching torchvision's
//! `BasicConv2d` submodule and producing parameter keys
//! `<prefix>.conv.weight`, `<prefix>.bn.weight`, `<prefix>.bn.bias`. BN's
//! running buffers (`running_mean`, `running_var`) are exposed by the
//! shared `BatchNorm2d` `as_any` downcast hook used by the value-parity
//! loader (#984 / #995).
//!
//! Notes on torchvision-exact corner cases:
//! * Conv `bias = false` (BN follows everywhere). Failure mode #33.
//! * BatchNorm2d `eps = 1e-3` (NOT the 1e-5 default). Failure mode #32.
//! * `branch_pool` uses `F.avg_pool2d(..., padding=1)` whose default is
//!   `count_include_pad=True` — ferrotorch's [`AvgPool2d`] is
//!   hardcoded to that semantics, so the divisor matches.
//! * InceptionE's `branch3x3_2a/2b` and `branch3x3dbl_3a/3b` run on the
//!   SAME upstream tensor and concat their outputs, matching
//!   torchvision's parallel-branch-with-fan-out shape (failure mode #34).
//! * `aux_logits = False`: no AuxLogits submodule, no auxiliary head
//!   (matches `inception_v3(aux_logits=False)` reference).

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::shape::{cat, reshape};
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::dropout::Dropout;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::{AdaptiveAvgPool2d, AvgPool2d, MaxPool2d};
use ferrotorch_nn::{Conv2d, Linear};

// ===========================================================================
// BasicConv2d — torchvision `BasicConv2d` parity
// ===========================================================================
//
// `BasicConv2d` (torchvision.models.inception):
//   self.conv = nn.Conv2d(in, out, bias=False, **kw)
//   self.bn   = nn.BatchNorm2d(out, eps=0.001)
//   forward   = relu(bn(conv(x)))
//
// Children render as `conv` and `bn` to match torchvision's parameter keys
// `<path>.conv.weight`, `<path>.bn.weight`, `<path>.bn.bias`. BN running
// buffers expose `<path>.bn.running_mean` / `running_var`.

/// Inception-V3's per-conv building block: `Conv2d(bias=false) +
/// BatchNorm2d(eps=1e-3) + ReLU`.
///
/// Matches `torchvision.models.inception.BasicConv2d` exactly. Used as
/// the single conv-stage primitive throughout the stem and all 11 Mixed
/// modules.
pub struct BasicConv2d<T: Float> {
    conv: Conv2d<T>,
    bn: BatchNorm2d<T>,
    training: bool,
}

/// Eps used by every `BatchNorm2d` in Inception-V3. torchvision pins this
/// to `0.001`; the 1e-5 default would change BN output by ~1 ulp per
/// element on the value-parity logits and is failure mode #32.
const INCEPTION_BN_EPS: f64 = 1e-3;

/// Default torchvision BN momentum (`0.1`).
const INCEPTION_BN_MOM: f64 = 0.1;

impl<T: Float> BasicConv2d<T> {
    /// Build a `BasicConv2d` with arbitrary kernel / stride / padding.
    ///
    /// `bias` is hardcoded to `false` — torchvision's `BasicConv2d`
    /// always passes `bias=False` to `nn.Conv2d` because a `BatchNorm2d`
    /// follows immediately. Re-enabling bias is failure mode #33.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> FerrotorchResult<Self> {
        let conv = Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            false, // bias=false (BN follows)
        )?;
        let bn = BatchNorm2d::new(out_channels, INCEPTION_BN_EPS, INCEPTION_BN_MOM, true)?;
        Ok(Self {
            conv,
            bn,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for BasicConv2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv.forward(input)?;
        let x = Module::<T>::forward(&self.bn, &x)?;
        relu(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv.parameters());
        p.extend(self.bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv.parameters_mut());
        p.extend(self.bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.conv.named_parameters() {
            p.push((format!("conv.{n}"), param));
        }
        for (n, param) in self.bn.named_parameters() {
            p.push((format!("bn.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.conv, &self.bn]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("conv".to_string(), &self.conv as &dyn Module<T>),
            ("bn".to_string(), &self.bn),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.conv.train();
        self.bn.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.conv.eval();
        self.bn.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Branch-name helpers
// ===========================================================================
//
// Each Mixed module's `named_parameters` is built mechanically by walking
// its sub-`BasicConv2d` blocks and prefixing their (already-`conv.X`/
// `bn.X`) keys with the torchvision branch name. The helpers below keep
// the per-branch boilerplate from drowning out the architectural shape.

fn extend_named<'a, T: Float>(
    out: &mut Vec<(String, &'a Parameter<T>)>,
    prefix: &str,
    block: &'a BasicConv2d<T>,
) {
    for (n, p) in block.named_parameters() {
        out.push((format!("{prefix}.{n}"), p));
    }
}

// ===========================================================================
// InceptionA — Mixed_5b/c/d
// ===========================================================================
//
// torchvision (paraphrased — sub-branch names quoted exactly):
//   branch1x1            : BasicConv2d(in, 64, k=1)
//   branch5x5_1          : BasicConv2d(in, 48, k=1)
//   branch5x5_2          : BasicConv2d(48, 64, k=5, padding=2)
//   branch3x3dbl_1       : BasicConv2d(in, 64, k=1)
//   branch3x3dbl_2       : BasicConv2d(64, 96, k=3, padding=1)
//   branch3x3dbl_3       : BasicConv2d(96, 96, k=3, padding=1)
//   branch_pool          : BasicConv2d(in, pool_features, k=1)
//   forward (concat-1)   : [branch1x1, branch5x5, branch3x3dbl,
//                           branch_pool(F.avg_pool2d(x, k=3, s=1, pad=1))]
// out_channels = 64 + 64 + 96 + pool_features.

/// `Mixed_5b/5c/5d` block — torchvision `InceptionA`.
pub struct InceptionA<T: Float> {
    branch1x1: BasicConv2d<T>,
    branch5x5_1: BasicConv2d<T>,
    branch5x5_2: BasicConv2d<T>,
    branch3x3dbl_1: BasicConv2d<T>,
    branch3x3dbl_2: BasicConv2d<T>,
    branch3x3dbl_3: BasicConv2d<T>,
    branch_pool: BasicConv2d<T>,
    avg_pool: AvgPool2d,
    training: bool,
}

impl<T: Float> InceptionA<T> {
    pub fn new(in_channels: usize, pool_features: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            branch1x1: BasicConv2d::new(in_channels, 64, (1, 1), (1, 1), (0, 0))?,
            branch5x5_1: BasicConv2d::new(in_channels, 48, (1, 1), (1, 1), (0, 0))?,
            branch5x5_2: BasicConv2d::new(48, 64, (5, 5), (1, 1), (2, 2))?,
            branch3x3dbl_1: BasicConv2d::new(in_channels, 64, (1, 1), (1, 1), (0, 0))?,
            branch3x3dbl_2: BasicConv2d::new(64, 96, (3, 3), (1, 1), (1, 1))?,
            branch3x3dbl_3: BasicConv2d::new(96, 96, (3, 3), (1, 1), (1, 1))?,
            branch_pool: BasicConv2d::new(in_channels, pool_features, (1, 1), (1, 1), (0, 0))?,
            avg_pool: AvgPool2d::new([3, 3], [1, 1], [1, 1]),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for InceptionA<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let b1 = self.branch1x1.forward(x)?;

        let b5 = self.branch5x5_1.forward(x)?;
        let b5 = self.branch5x5_2.forward(&b5)?;

        let b3 = self.branch3x3dbl_1.forward(x)?;
        let b3 = self.branch3x3dbl_2.forward(&b3)?;
        let b3 = self.branch3x3dbl_3.forward(&b3)?;

        let bp = Module::<T>::forward(&self.avg_pool, x)?;
        let bp = self.branch_pool.forward(&bp)?;

        cat(&[b1, b5, b3, bp], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters());
        p.extend(self.branch5x5_1.parameters());
        p.extend(self.branch5x5_2.parameters());
        p.extend(self.branch3x3dbl_1.parameters());
        p.extend(self.branch3x3dbl_2.parameters());
        p.extend(self.branch3x3dbl_3.parameters());
        p.extend(self.branch_pool.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters_mut());
        p.extend(self.branch5x5_1.parameters_mut());
        p.extend(self.branch5x5_2.parameters_mut());
        p.extend(self.branch3x3dbl_1.parameters_mut());
        p.extend(self.branch3x3dbl_2.parameters_mut());
        p.extend(self.branch3x3dbl_3.parameters_mut());
        p.extend(self.branch_pool.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        extend_named(&mut p, "branch1x1", &self.branch1x1);
        extend_named(&mut p, "branch5x5_1", &self.branch5x5_1);
        extend_named(&mut p, "branch5x5_2", &self.branch5x5_2);
        extend_named(&mut p, "branch3x3dbl_1", &self.branch3x3dbl_1);
        extend_named(&mut p, "branch3x3dbl_2", &self.branch3x3dbl_2);
        extend_named(&mut p, "branch3x3dbl_3", &self.branch3x3dbl_3);
        extend_named(&mut p, "branch_pool", &self.branch_pool);
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.branch1x1,
            &self.branch5x5_1,
            &self.branch5x5_2,
            &self.branch3x3dbl_1,
            &self.branch3x3dbl_2,
            &self.branch3x3dbl_3,
            &self.branch_pool,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("branch1x1".to_string(), &self.branch1x1 as &dyn Module<T>),
            ("branch5x5_1".to_string(), &self.branch5x5_1),
            ("branch5x5_2".to_string(), &self.branch5x5_2),
            ("branch3x3dbl_1".to_string(), &self.branch3x3dbl_1),
            ("branch3x3dbl_2".to_string(), &self.branch3x3dbl_2),
            ("branch3x3dbl_3".to_string(), &self.branch3x3dbl_3),
            ("branch_pool".to_string(), &self.branch_pool),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.branch1x1.train();
        self.branch5x5_1.train();
        self.branch5x5_2.train();
        self.branch3x3dbl_1.train();
        self.branch3x3dbl_2.train();
        self.branch3x3dbl_3.train();
        self.branch_pool.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.branch1x1.eval();
        self.branch5x5_1.eval();
        self.branch5x5_2.eval();
        self.branch3x3dbl_1.eval();
        self.branch3x3dbl_2.eval();
        self.branch3x3dbl_3.eval();
        self.branch_pool.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// InceptionB — Mixed_6a (grid reduction, no pool conv)
// ===========================================================================
//
// torchvision:
//   branch3x3            : BasicConv2d(in, 384, k=3, s=2)
//   branch3x3dbl_1       : BasicConv2d(in, 64, k=1)
//   branch3x3dbl_2       : BasicConv2d(64, 96, k=3, padding=1)
//   branch3x3dbl_3       : BasicConv2d(96, 96, k=3, s=2)
//   forward (concat-1)   : [branch3x3, branch3x3dbl,
//                           F.max_pool2d(x, k=3, s=2)]
// out_channels = in + 384 + 96  (the maxpool branch passes `in` through
// unchanged with 1×1 conv-equivalent of identity on channels).
//
// IMPORTANT: there is NO `branch_pool` BasicConv2d in InceptionB — the
// pool path is a bare maxpool with no projection. Reference:
// `InceptionB._forward` in torchvision/models/inception.py.

/// `Mixed_6a` block — torchvision `InceptionB` (reduction).
pub struct InceptionB<T: Float> {
    branch3x3: BasicConv2d<T>,
    branch3x3dbl_1: BasicConv2d<T>,
    branch3x3dbl_2: BasicConv2d<T>,
    branch3x3dbl_3: BasicConv2d<T>,
    max_pool: MaxPool2d,
    training: bool,
}

impl<T: Float> InceptionB<T> {
    pub fn new(in_channels: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            branch3x3: BasicConv2d::new(in_channels, 384, (3, 3), (2, 2), (0, 0))?,
            branch3x3dbl_1: BasicConv2d::new(in_channels, 64, (1, 1), (1, 1), (0, 0))?,
            branch3x3dbl_2: BasicConv2d::new(64, 96, (3, 3), (1, 1), (1, 1))?,
            branch3x3dbl_3: BasicConv2d::new(96, 96, (3, 3), (2, 2), (0, 0))?,
            max_pool: MaxPool2d::new([3, 3], [2, 2], [0, 0]),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for InceptionB<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let b3 = self.branch3x3.forward(x)?;

        let bd = self.branch3x3dbl_1.forward(x)?;
        let bd = self.branch3x3dbl_2.forward(&bd)?;
        let bd = self.branch3x3dbl_3.forward(&bd)?;

        let bp = Module::<T>::forward(&self.max_pool, x)?;

        cat(&[b3, bd, bp], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch3x3.parameters());
        p.extend(self.branch3x3dbl_1.parameters());
        p.extend(self.branch3x3dbl_2.parameters());
        p.extend(self.branch3x3dbl_3.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch3x3.parameters_mut());
        p.extend(self.branch3x3dbl_1.parameters_mut());
        p.extend(self.branch3x3dbl_2.parameters_mut());
        p.extend(self.branch3x3dbl_3.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        extend_named(&mut p, "branch3x3", &self.branch3x3);
        extend_named(&mut p, "branch3x3dbl_1", &self.branch3x3dbl_1);
        extend_named(&mut p, "branch3x3dbl_2", &self.branch3x3dbl_2);
        extend_named(&mut p, "branch3x3dbl_3", &self.branch3x3dbl_3);
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.branch3x3,
            &self.branch3x3dbl_1,
            &self.branch3x3dbl_2,
            &self.branch3x3dbl_3,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("branch3x3".to_string(), &self.branch3x3 as &dyn Module<T>),
            ("branch3x3dbl_1".to_string(), &self.branch3x3dbl_1),
            ("branch3x3dbl_2".to_string(), &self.branch3x3dbl_2),
            ("branch3x3dbl_3".to_string(), &self.branch3x3dbl_3),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.branch3x3.train();
        self.branch3x3dbl_1.train();
        self.branch3x3dbl_2.train();
        self.branch3x3dbl_3.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.branch3x3.eval();
        self.branch3x3dbl_1.eval();
        self.branch3x3dbl_2.eval();
        self.branch3x3dbl_3.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// InceptionC — Mixed_6b/c/d/e (factorized 7×7)
// ===========================================================================
//
// torchvision:
//   branch1x1            : BasicConv2d(in, 192, k=1)
//   branch7x7_1          : BasicConv2d(in, c7, k=1)
//   branch7x7_2          : BasicConv2d(c7, c7, k=(1,7), padding=(0,3))
//   branch7x7_3          : BasicConv2d(c7, 192, k=(7,1), padding=(3,0))
//   branch7x7dbl_1       : BasicConv2d(in, c7, k=1)
//   branch7x7dbl_2       : BasicConv2d(c7, c7, k=(7,1), padding=(3,0))
//   branch7x7dbl_3       : BasicConv2d(c7, c7, k=(1,7), padding=(0,3))
//   branch7x7dbl_4       : BasicConv2d(c7, c7, k=(7,1), padding=(3,0))
//   branch7x7dbl_5       : BasicConv2d(c7, 192, k=(1,7), padding=(0,3))
//   branch_pool          : BasicConv2d(in, 192, k=1)
//   forward (concat-1)   : [branch1x1, branch7x7, branch7x7dbl,
//                           branch_pool(F.avg_pool2d(x, k=3, s=1, pad=1))]
// out_channels = 192 + 192 + 192 + 192 = 768.

/// `Mixed_6b/6c/6d/6e` block — torchvision `InceptionC`.
pub struct InceptionC<T: Float> {
    branch1x1: BasicConv2d<T>,
    branch7x7_1: BasicConv2d<T>,
    branch7x7_2: BasicConv2d<T>,
    branch7x7_3: BasicConv2d<T>,
    branch7x7dbl_1: BasicConv2d<T>,
    branch7x7dbl_2: BasicConv2d<T>,
    branch7x7dbl_3: BasicConv2d<T>,
    branch7x7dbl_4: BasicConv2d<T>,
    branch7x7dbl_5: BasicConv2d<T>,
    branch_pool: BasicConv2d<T>,
    avg_pool: AvgPool2d,
    training: bool,
}

impl<T: Float> InceptionC<T> {
    pub fn new(in_channels: usize, channels_7x7: usize) -> FerrotorchResult<Self> {
        let c7 = channels_7x7;
        Ok(Self {
            branch1x1: BasicConv2d::new(in_channels, 192, (1, 1), (1, 1), (0, 0))?,
            branch7x7_1: BasicConv2d::new(in_channels, c7, (1, 1), (1, 1), (0, 0))?,
            branch7x7_2: BasicConv2d::new(c7, c7, (1, 7), (1, 1), (0, 3))?,
            branch7x7_3: BasicConv2d::new(c7, 192, (7, 1), (1, 1), (3, 0))?,
            branch7x7dbl_1: BasicConv2d::new(in_channels, c7, (1, 1), (1, 1), (0, 0))?,
            branch7x7dbl_2: BasicConv2d::new(c7, c7, (7, 1), (1, 1), (3, 0))?,
            branch7x7dbl_3: BasicConv2d::new(c7, c7, (1, 7), (1, 1), (0, 3))?,
            branch7x7dbl_4: BasicConv2d::new(c7, c7, (7, 1), (1, 1), (3, 0))?,
            branch7x7dbl_5: BasicConv2d::new(c7, 192, (1, 7), (1, 1), (0, 3))?,
            branch_pool: BasicConv2d::new(in_channels, 192, (1, 1), (1, 1), (0, 0))?,
            avg_pool: AvgPool2d::new([3, 3], [1, 1], [1, 1]),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for InceptionC<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let b1 = self.branch1x1.forward(x)?;

        let b7 = self.branch7x7_1.forward(x)?;
        let b7 = self.branch7x7_2.forward(&b7)?;
        let b7 = self.branch7x7_3.forward(&b7)?;

        let bd = self.branch7x7dbl_1.forward(x)?;
        let bd = self.branch7x7dbl_2.forward(&bd)?;
        let bd = self.branch7x7dbl_3.forward(&bd)?;
        let bd = self.branch7x7dbl_4.forward(&bd)?;
        let bd = self.branch7x7dbl_5.forward(&bd)?;

        let bp = Module::<T>::forward(&self.avg_pool, x)?;
        let bp = self.branch_pool.forward(&bp)?;

        cat(&[b1, b7, bd, bp], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters());
        p.extend(self.branch7x7_1.parameters());
        p.extend(self.branch7x7_2.parameters());
        p.extend(self.branch7x7_3.parameters());
        p.extend(self.branch7x7dbl_1.parameters());
        p.extend(self.branch7x7dbl_2.parameters());
        p.extend(self.branch7x7dbl_3.parameters());
        p.extend(self.branch7x7dbl_4.parameters());
        p.extend(self.branch7x7dbl_5.parameters());
        p.extend(self.branch_pool.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters_mut());
        p.extend(self.branch7x7_1.parameters_mut());
        p.extend(self.branch7x7_2.parameters_mut());
        p.extend(self.branch7x7_3.parameters_mut());
        p.extend(self.branch7x7dbl_1.parameters_mut());
        p.extend(self.branch7x7dbl_2.parameters_mut());
        p.extend(self.branch7x7dbl_3.parameters_mut());
        p.extend(self.branch7x7dbl_4.parameters_mut());
        p.extend(self.branch7x7dbl_5.parameters_mut());
        p.extend(self.branch_pool.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        extend_named(&mut p, "branch1x1", &self.branch1x1);
        extend_named(&mut p, "branch7x7_1", &self.branch7x7_1);
        extend_named(&mut p, "branch7x7_2", &self.branch7x7_2);
        extend_named(&mut p, "branch7x7_3", &self.branch7x7_3);
        extend_named(&mut p, "branch7x7dbl_1", &self.branch7x7dbl_1);
        extend_named(&mut p, "branch7x7dbl_2", &self.branch7x7dbl_2);
        extend_named(&mut p, "branch7x7dbl_3", &self.branch7x7dbl_3);
        extend_named(&mut p, "branch7x7dbl_4", &self.branch7x7dbl_4);
        extend_named(&mut p, "branch7x7dbl_5", &self.branch7x7dbl_5);
        extend_named(&mut p, "branch_pool", &self.branch_pool);
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.branch1x1,
            &self.branch7x7_1,
            &self.branch7x7_2,
            &self.branch7x7_3,
            &self.branch7x7dbl_1,
            &self.branch7x7dbl_2,
            &self.branch7x7dbl_3,
            &self.branch7x7dbl_4,
            &self.branch7x7dbl_5,
            &self.branch_pool,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("branch1x1".to_string(), &self.branch1x1 as &dyn Module<T>),
            ("branch7x7_1".to_string(), &self.branch7x7_1),
            ("branch7x7_2".to_string(), &self.branch7x7_2),
            ("branch7x7_3".to_string(), &self.branch7x7_3),
            ("branch7x7dbl_1".to_string(), &self.branch7x7dbl_1),
            ("branch7x7dbl_2".to_string(), &self.branch7x7dbl_2),
            ("branch7x7dbl_3".to_string(), &self.branch7x7dbl_3),
            ("branch7x7dbl_4".to_string(), &self.branch7x7dbl_4),
            ("branch7x7dbl_5".to_string(), &self.branch7x7dbl_5),
            ("branch_pool".to_string(), &self.branch_pool),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.branch1x1.train();
        self.branch7x7_1.train();
        self.branch7x7_2.train();
        self.branch7x7_3.train();
        self.branch7x7dbl_1.train();
        self.branch7x7dbl_2.train();
        self.branch7x7dbl_3.train();
        self.branch7x7dbl_4.train();
        self.branch7x7dbl_5.train();
        self.branch_pool.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.branch1x1.eval();
        self.branch7x7_1.eval();
        self.branch7x7_2.eval();
        self.branch7x7_3.eval();
        self.branch7x7dbl_1.eval();
        self.branch7x7dbl_2.eval();
        self.branch7x7dbl_3.eval();
        self.branch7x7dbl_4.eval();
        self.branch7x7dbl_5.eval();
        self.branch_pool.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// InceptionD — Mixed_7a (reduction, factorized 7×7×3)
// ===========================================================================
//
// torchvision:
//   branch3x3_1          : BasicConv2d(in, 192, k=1)
//   branch3x3_2          : BasicConv2d(192, 320, k=3, s=2)
//   branch7x7x3_1        : BasicConv2d(in, 192, k=1)
//   branch7x7x3_2        : BasicConv2d(192, 192, k=(1,7), padding=(0,3))
//   branch7x7x3_3        : BasicConv2d(192, 192, k=(7,1), padding=(3,0))
//   branch7x7x3_4        : BasicConv2d(192, 192, k=3, s=2)
//   forward (concat-1)   : [branch3x3, branch7x7x3,
//                           F.max_pool2d(x, k=3, s=2)]
// out_channels = 320 + 192 + in.

/// `Mixed_7a` block — torchvision `InceptionD` (reduction).
pub struct InceptionD<T: Float> {
    branch3x3_1: BasicConv2d<T>,
    branch3x3_2: BasicConv2d<T>,
    branch7x7x3_1: BasicConv2d<T>,
    branch7x7x3_2: BasicConv2d<T>,
    branch7x7x3_3: BasicConv2d<T>,
    branch7x7x3_4: BasicConv2d<T>,
    max_pool: MaxPool2d,
    training: bool,
}

impl<T: Float> InceptionD<T> {
    pub fn new(in_channels: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            branch3x3_1: BasicConv2d::new(in_channels, 192, (1, 1), (1, 1), (0, 0))?,
            branch3x3_2: BasicConv2d::new(192, 320, (3, 3), (2, 2), (0, 0))?,
            branch7x7x3_1: BasicConv2d::new(in_channels, 192, (1, 1), (1, 1), (0, 0))?,
            branch7x7x3_2: BasicConv2d::new(192, 192, (1, 7), (1, 1), (0, 3))?,
            branch7x7x3_3: BasicConv2d::new(192, 192, (7, 1), (1, 1), (3, 0))?,
            branch7x7x3_4: BasicConv2d::new(192, 192, (3, 3), (2, 2), (0, 0))?,
            max_pool: MaxPool2d::new([3, 3], [2, 2], [0, 0]),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for InceptionD<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let b3 = self.branch3x3_1.forward(x)?;
        let b3 = self.branch3x3_2.forward(&b3)?;

        let b7 = self.branch7x7x3_1.forward(x)?;
        let b7 = self.branch7x7x3_2.forward(&b7)?;
        let b7 = self.branch7x7x3_3.forward(&b7)?;
        let b7 = self.branch7x7x3_4.forward(&b7)?;

        let bp = Module::<T>::forward(&self.max_pool, x)?;

        cat(&[b3, b7, bp], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch3x3_1.parameters());
        p.extend(self.branch3x3_2.parameters());
        p.extend(self.branch7x7x3_1.parameters());
        p.extend(self.branch7x7x3_2.parameters());
        p.extend(self.branch7x7x3_3.parameters());
        p.extend(self.branch7x7x3_4.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch3x3_1.parameters_mut());
        p.extend(self.branch3x3_2.parameters_mut());
        p.extend(self.branch7x7x3_1.parameters_mut());
        p.extend(self.branch7x7x3_2.parameters_mut());
        p.extend(self.branch7x7x3_3.parameters_mut());
        p.extend(self.branch7x7x3_4.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        extend_named(&mut p, "branch3x3_1", &self.branch3x3_1);
        extend_named(&mut p, "branch3x3_2", &self.branch3x3_2);
        extend_named(&mut p, "branch7x7x3_1", &self.branch7x7x3_1);
        extend_named(&mut p, "branch7x7x3_2", &self.branch7x7x3_2);
        extend_named(&mut p, "branch7x7x3_3", &self.branch7x7x3_3);
        extend_named(&mut p, "branch7x7x3_4", &self.branch7x7x3_4);
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.branch3x3_1,
            &self.branch3x3_2,
            &self.branch7x7x3_1,
            &self.branch7x7x3_2,
            &self.branch7x7x3_3,
            &self.branch7x7x3_4,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            (
                "branch3x3_1".to_string(),
                &self.branch3x3_1 as &dyn Module<T>,
            ),
            ("branch3x3_2".to_string(), &self.branch3x3_2),
            ("branch7x7x3_1".to_string(), &self.branch7x7x3_1),
            ("branch7x7x3_2".to_string(), &self.branch7x7x3_2),
            ("branch7x7x3_3".to_string(), &self.branch7x7x3_3),
            ("branch7x7x3_4".to_string(), &self.branch7x7x3_4),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.branch3x3_1.train();
        self.branch3x3_2.train();
        self.branch7x7x3_1.train();
        self.branch7x7x3_2.train();
        self.branch7x7x3_3.train();
        self.branch7x7x3_4.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.branch3x3_1.eval();
        self.branch3x3_2.eval();
        self.branch7x7x3_1.eval();
        self.branch7x7x3_2.eval();
        self.branch7x7x3_3.eval();
        self.branch7x7x3_4.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// InceptionE — Mixed_7b/c (parallel 1×3 / 3×1 sub-branches)
// ===========================================================================
//
// torchvision (failure mode #34: 2a/2b run in PARALLEL on the same input
// then concat — they DO NOT chain):
//   branch1x1            : BasicConv2d(in, 320, k=1)
//   branch3x3_1          : BasicConv2d(in, 384, k=1)
//   branch3x3_2a         : BasicConv2d(384, 384, k=(1,3), padding=(0,1))
//   branch3x3_2b         : BasicConv2d(384, 384, k=(3,1), padding=(1,0))
//   branch3x3dbl_1       : BasicConv2d(in, 448, k=1)
//   branch3x3dbl_2       : BasicConv2d(448, 384, k=3, padding=1)
//   branch3x3dbl_3a      : BasicConv2d(384, 384, k=(1,3), padding=(0,1))
//   branch3x3dbl_3b      : BasicConv2d(384, 384, k=(3,1), padding=(1,0))
//   branch_pool          : BasicConv2d(in, 192, k=1)
//   forward:
//     branch3x3 = branch3x3_1(x); branch3x3 = cat([2a(b), 2b(b)], 1)
//     branch3x3dbl = branch3x3dbl_1(x); = branch3x3dbl_2(...);
//                  = cat([3a(b), 3b(b)], 1)
//     branch_pool = branch_pool(F.avg_pool2d(x, k=3, s=1, pad=1))
//     return cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)
// out_channels = 320 + (384+384) + (384+384) + 192 = 2048.

/// `Mixed_7b/7c` block — torchvision `InceptionE`.
pub struct InceptionE<T: Float> {
    branch1x1: BasicConv2d<T>,
    branch3x3_1: BasicConv2d<T>,
    branch3x3_2a: BasicConv2d<T>,
    branch3x3_2b: BasicConv2d<T>,
    branch3x3dbl_1: BasicConv2d<T>,
    branch3x3dbl_2: BasicConv2d<T>,
    branch3x3dbl_3a: BasicConv2d<T>,
    branch3x3dbl_3b: BasicConv2d<T>,
    branch_pool: BasicConv2d<T>,
    avg_pool: AvgPool2d,
    training: bool,
}

impl<T: Float> InceptionE<T> {
    pub fn new(in_channels: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            branch1x1: BasicConv2d::new(in_channels, 320, (1, 1), (1, 1), (0, 0))?,
            branch3x3_1: BasicConv2d::new(in_channels, 384, (1, 1), (1, 1), (0, 0))?,
            branch3x3_2a: BasicConv2d::new(384, 384, (1, 3), (1, 1), (0, 1))?,
            branch3x3_2b: BasicConv2d::new(384, 384, (3, 1), (1, 1), (1, 0))?,
            branch3x3dbl_1: BasicConv2d::new(in_channels, 448, (1, 1), (1, 1), (0, 0))?,
            branch3x3dbl_2: BasicConv2d::new(448, 384, (3, 3), (1, 1), (1, 1))?,
            branch3x3dbl_3a: BasicConv2d::new(384, 384, (1, 3), (1, 1), (0, 1))?,
            branch3x3dbl_3b: BasicConv2d::new(384, 384, (3, 1), (1, 1), (1, 0))?,
            branch_pool: BasicConv2d::new(in_channels, 192, (1, 1), (1, 1), (0, 0))?,
            avg_pool: AvgPool2d::new([3, 3], [1, 1], [1, 1]),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for InceptionE<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let b1 = self.branch1x1.forward(x)?;

        // Parallel-branch concat — failure mode #34. 2a/2b BOTH consume
        // the output of branch3x3_1 and run side-by-side; their results
        // are concatenated along the channel axis.
        let b3 = self.branch3x3_1.forward(x)?;
        let b3a = self.branch3x3_2a.forward(&b3)?;
        let b3b = self.branch3x3_2b.forward(&b3)?;
        let b3 = cat(&[b3a, b3b], 1)?;

        // Same parallel pattern for the double-3×3 branch's 3a/3b.
        let bd = self.branch3x3dbl_1.forward(x)?;
        let bd = self.branch3x3dbl_2.forward(&bd)?;
        let bd_a = self.branch3x3dbl_3a.forward(&bd)?;
        let bd_b = self.branch3x3dbl_3b.forward(&bd)?;
        let bd = cat(&[bd_a, bd_b], 1)?;

        let bp = Module::<T>::forward(&self.avg_pool, x)?;
        let bp = self.branch_pool.forward(&bp)?;

        cat(&[b1, b3, bd, bp], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters());
        p.extend(self.branch3x3_1.parameters());
        p.extend(self.branch3x3_2a.parameters());
        p.extend(self.branch3x3_2b.parameters());
        p.extend(self.branch3x3dbl_1.parameters());
        p.extend(self.branch3x3dbl_2.parameters());
        p.extend(self.branch3x3dbl_3a.parameters());
        p.extend(self.branch3x3dbl_3b.parameters());
        p.extend(self.branch_pool.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters_mut());
        p.extend(self.branch3x3_1.parameters_mut());
        p.extend(self.branch3x3_2a.parameters_mut());
        p.extend(self.branch3x3_2b.parameters_mut());
        p.extend(self.branch3x3dbl_1.parameters_mut());
        p.extend(self.branch3x3dbl_2.parameters_mut());
        p.extend(self.branch3x3dbl_3a.parameters_mut());
        p.extend(self.branch3x3dbl_3b.parameters_mut());
        p.extend(self.branch_pool.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        extend_named(&mut p, "branch1x1", &self.branch1x1);
        extend_named(&mut p, "branch3x3_1", &self.branch3x3_1);
        extend_named(&mut p, "branch3x3_2a", &self.branch3x3_2a);
        extend_named(&mut p, "branch3x3_2b", &self.branch3x3_2b);
        extend_named(&mut p, "branch3x3dbl_1", &self.branch3x3dbl_1);
        extend_named(&mut p, "branch3x3dbl_2", &self.branch3x3dbl_2);
        extend_named(&mut p, "branch3x3dbl_3a", &self.branch3x3dbl_3a);
        extend_named(&mut p, "branch3x3dbl_3b", &self.branch3x3dbl_3b);
        extend_named(&mut p, "branch_pool", &self.branch_pool);
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.branch1x1,
            &self.branch3x3_1,
            &self.branch3x3_2a,
            &self.branch3x3_2b,
            &self.branch3x3dbl_1,
            &self.branch3x3dbl_2,
            &self.branch3x3dbl_3a,
            &self.branch3x3dbl_3b,
            &self.branch_pool,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("branch1x1".to_string(), &self.branch1x1 as &dyn Module<T>),
            ("branch3x3_1".to_string(), &self.branch3x3_1),
            ("branch3x3_2a".to_string(), &self.branch3x3_2a),
            ("branch3x3_2b".to_string(), &self.branch3x3_2b),
            ("branch3x3dbl_1".to_string(), &self.branch3x3dbl_1),
            ("branch3x3dbl_2".to_string(), &self.branch3x3dbl_2),
            ("branch3x3dbl_3a".to_string(), &self.branch3x3dbl_3a),
            ("branch3x3dbl_3b".to_string(), &self.branch3x3dbl_3b),
            ("branch_pool".to_string(), &self.branch_pool),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.branch1x1.train();
        self.branch3x3_1.train();
        self.branch3x3_2a.train();
        self.branch3x3_2b.train();
        self.branch3x3dbl_1.train();
        self.branch3x3dbl_2.train();
        self.branch3x3dbl_3a.train();
        self.branch3x3dbl_3b.train();
        self.branch_pool.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.branch1x1.eval();
        self.branch3x3_1.eval();
        self.branch3x3_2a.eval();
        self.branch3x3_2b.eval();
        self.branch3x3dbl_1.eval();
        self.branch3x3dbl_2.eval();
        self.branch3x3dbl_3a.eval();
        self.branch3x3dbl_3b.eval();
        self.branch_pool.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// InceptionV3 — top-level model
// ===========================================================================

/// Inception-V3 with `aux_logits=False`, matching torchvision exactly.
///
/// ```text
/// stem -> 11 Mixed -> avgpool -> dropout -> fc
/// ```
///
/// Default input is `1×3×299×299` (see torchvision design intent;
/// `AdaptiveAvgPool2d((1, 1))` lets smaller inputs work but value-parity
/// is only guaranteed at 299×299 with the trained weights).
// `non_snake_case` is allowed here because the field names mirror
// torchvision's `Conv2d_1a_3x3` / `Mixed_5b` / ... attribute names
// directly. `named_parameters` and `named_children` use these strings
// as the path components seen by the value-parity loader and any
// state-dict consumer, so renaming to snake_case would force a manual
// re-mapping table to recover the same external names. Since the entire
// point of this rebuild is `inception_v3(aux_logits=False)` parity,
// preserving torchvision identifiers is correct.
#[allow(non_snake_case)]
pub struct InceptionV3<T: Float> {
    Conv2d_1a_3x3: BasicConv2d<T>,
    Conv2d_2a_3x3: BasicConv2d<T>,
    Conv2d_2b_3x3: BasicConv2d<T>,
    maxpool1: MaxPool2d,
    Conv2d_3b_1x1: BasicConv2d<T>,
    Conv2d_4a_3x3: BasicConv2d<T>,
    maxpool2: MaxPool2d,
    Mixed_5b: InceptionA<T>,
    Mixed_5c: InceptionA<T>,
    Mixed_5d: InceptionA<T>,
    Mixed_6a: InceptionB<T>,
    Mixed_6b: InceptionC<T>,
    Mixed_6c: InceptionC<T>,
    Mixed_6d: InceptionC<T>,
    Mixed_6e: InceptionC<T>,
    Mixed_7a: InceptionD<T>,
    Mixed_7b: InceptionE<T>,
    Mixed_7c: InceptionE<T>,
    avgpool: AdaptiveAvgPool2d,
    dropout: Dropout<T>,
    fc: Linear<T>,
    training: bool,
}

impl<T: Float> InceptionV3<T> {
    /// Construct Inception-V3 (`aux_logits=False`) with `num_classes`
    /// output logits.
    ///
    /// All `BasicConv2d` blocks use `bias=false`, `BatchNorm2d` with
    /// `eps=1e-3`. The classifier head is `Dropout(p=0.5) → Linear(2048,
    /// num_classes)` matching torchvision's `inception_v3`.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            Conv2d_1a_3x3: BasicConv2d::new(3, 32, (3, 3), (2, 2), (0, 0))?,
            Conv2d_2a_3x3: BasicConv2d::new(32, 32, (3, 3), (1, 1), (0, 0))?,
            Conv2d_2b_3x3: BasicConv2d::new(32, 64, (3, 3), (1, 1), (1, 1))?,
            maxpool1: MaxPool2d::new([3, 3], [2, 2], [0, 0]),
            Conv2d_3b_1x1: BasicConv2d::new(64, 80, (1, 1), (1, 1), (0, 0))?,
            Conv2d_4a_3x3: BasicConv2d::new(80, 192, (3, 3), (1, 1), (0, 0))?,
            maxpool2: MaxPool2d::new([3, 3], [2, 2], [0, 0]),
            Mixed_5b: InceptionA::new(192, 32)?,
            Mixed_5c: InceptionA::new(256, 64)?,
            Mixed_5d: InceptionA::new(288, 64)?,
            Mixed_6a: InceptionB::new(288)?,
            Mixed_6b: InceptionC::new(768, 128)?,
            Mixed_6c: InceptionC::new(768, 160)?,
            Mixed_6d: InceptionC::new(768, 160)?,
            Mixed_6e: InceptionC::new(768, 192)?,
            Mixed_7a: InceptionD::new(768)?,
            Mixed_7b: InceptionE::new(1280)?,
            Mixed_7c: InceptionE::new(2048)?,
            avgpool: AdaptiveAvgPool2d::new((1, 1)),
            dropout: Dropout::new(0.5)?,
            fc: Linear::new(2048, num_classes, true)?,
            training: true,
        })
    }

    /// Number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<T: Float> Module<T> for InceptionV3<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.Conv2d_1a_3x3.forward(input)?;
        let x = self.Conv2d_2a_3x3.forward(&x)?;
        let x = self.Conv2d_2b_3x3.forward(&x)?;
        let x = Module::<T>::forward(&self.maxpool1, &x)?;
        let x = self.Conv2d_3b_1x1.forward(&x)?;
        let x = self.Conv2d_4a_3x3.forward(&x)?;
        let x = Module::<T>::forward(&self.maxpool2, &x)?;
        let x = self.Mixed_5b.forward(&x)?;
        let x = self.Mixed_5c.forward(&x)?;
        let x = self.Mixed_5d.forward(&x)?;
        let x = self.Mixed_6a.forward(&x)?;
        let x = self.Mixed_6b.forward(&x)?;
        let x = self.Mixed_6c.forward(&x)?;
        let x = self.Mixed_6d.forward(&x)?;
        let x = self.Mixed_6e.forward(&x)?;
        let x = self.Mixed_7a.forward(&x)?;
        let x = self.Mixed_7b.forward(&x)?;
        let x = self.Mixed_7c.forward(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let x = Module::<T>::forward(&self.dropout, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.Conv2d_1a_3x3.parameters());
        p.extend(self.Conv2d_2a_3x3.parameters());
        p.extend(self.Conv2d_2b_3x3.parameters());
        p.extend(self.Conv2d_3b_1x1.parameters());
        p.extend(self.Conv2d_4a_3x3.parameters());
        p.extend(self.Mixed_5b.parameters());
        p.extend(self.Mixed_5c.parameters());
        p.extend(self.Mixed_5d.parameters());
        p.extend(self.Mixed_6a.parameters());
        p.extend(self.Mixed_6b.parameters());
        p.extend(self.Mixed_6c.parameters());
        p.extend(self.Mixed_6d.parameters());
        p.extend(self.Mixed_6e.parameters());
        p.extend(self.Mixed_7a.parameters());
        p.extend(self.Mixed_7b.parameters());
        p.extend(self.Mixed_7c.parameters());
        p.extend(self.fc.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.Conv2d_1a_3x3.parameters_mut());
        p.extend(self.Conv2d_2a_3x3.parameters_mut());
        p.extend(self.Conv2d_2b_3x3.parameters_mut());
        p.extend(self.Conv2d_3b_1x1.parameters_mut());
        p.extend(self.Conv2d_4a_3x3.parameters_mut());
        p.extend(self.Mixed_5b.parameters_mut());
        p.extend(self.Mixed_5c.parameters_mut());
        p.extend(self.Mixed_5d.parameters_mut());
        p.extend(self.Mixed_6a.parameters_mut());
        p.extend(self.Mixed_6b.parameters_mut());
        p.extend(self.Mixed_6c.parameters_mut());
        p.extend(self.Mixed_6d.parameters_mut());
        p.extend(self.Mixed_6e.parameters_mut());
        p.extend(self.Mixed_7a.parameters_mut());
        p.extend(self.Mixed_7b.parameters_mut());
        p.extend(self.Mixed_7c.parameters_mut());
        p.extend(self.fc.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.Conv2d_1a_3x3.named_parameters() {
            p.push((format!("Conv2d_1a_3x3.{n}"), param));
        }
        for (n, param) in self.Conv2d_2a_3x3.named_parameters() {
            p.push((format!("Conv2d_2a_3x3.{n}"), param));
        }
        for (n, param) in self.Conv2d_2b_3x3.named_parameters() {
            p.push((format!("Conv2d_2b_3x3.{n}"), param));
        }
        for (n, param) in self.Conv2d_3b_1x1.named_parameters() {
            p.push((format!("Conv2d_3b_1x1.{n}"), param));
        }
        for (n, param) in self.Conv2d_4a_3x3.named_parameters() {
            p.push((format!("Conv2d_4a_3x3.{n}"), param));
        }
        for (n, param) in self.Mixed_5b.named_parameters() {
            p.push((format!("Mixed_5b.{n}"), param));
        }
        for (n, param) in self.Mixed_5c.named_parameters() {
            p.push((format!("Mixed_5c.{n}"), param));
        }
        for (n, param) in self.Mixed_5d.named_parameters() {
            p.push((format!("Mixed_5d.{n}"), param));
        }
        for (n, param) in self.Mixed_6a.named_parameters() {
            p.push((format!("Mixed_6a.{n}"), param));
        }
        for (n, param) in self.Mixed_6b.named_parameters() {
            p.push((format!("Mixed_6b.{n}"), param));
        }
        for (n, param) in self.Mixed_6c.named_parameters() {
            p.push((format!("Mixed_6c.{n}"), param));
        }
        for (n, param) in self.Mixed_6d.named_parameters() {
            p.push((format!("Mixed_6d.{n}"), param));
        }
        for (n, param) in self.Mixed_6e.named_parameters() {
            p.push((format!("Mixed_6e.{n}"), param));
        }
        for (n, param) in self.Mixed_7a.named_parameters() {
            p.push((format!("Mixed_7a.{n}"), param));
        }
        for (n, param) in self.Mixed_7b.named_parameters() {
            p.push((format!("Mixed_7b.{n}"), param));
        }
        for (n, param) in self.Mixed_7c.named_parameters() {
            p.push((format!("Mixed_7c.{n}"), param));
        }
        for (n, param) in self.fc.named_parameters() {
            p.push((format!("fc.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.Conv2d_1a_3x3,
            &self.Conv2d_2a_3x3,
            &self.Conv2d_2b_3x3,
            &self.maxpool1,
            &self.Conv2d_3b_1x1,
            &self.Conv2d_4a_3x3,
            &self.maxpool2,
            &self.Mixed_5b,
            &self.Mixed_5c,
            &self.Mixed_5d,
            &self.Mixed_6a,
            &self.Mixed_6b,
            &self.Mixed_6c,
            &self.Mixed_6d,
            &self.Mixed_6e,
            &self.Mixed_7a,
            &self.Mixed_7b,
            &self.Mixed_7c,
            &self.avgpool,
            &self.dropout,
            &self.fc,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            (
                "Conv2d_1a_3x3".to_string(),
                &self.Conv2d_1a_3x3 as &dyn Module<T>,
            ),
            ("Conv2d_2a_3x3".to_string(), &self.Conv2d_2a_3x3),
            ("Conv2d_2b_3x3".to_string(), &self.Conv2d_2b_3x3),
            ("maxpool1".to_string(), &self.maxpool1),
            ("Conv2d_3b_1x1".to_string(), &self.Conv2d_3b_1x1),
            ("Conv2d_4a_3x3".to_string(), &self.Conv2d_4a_3x3),
            ("maxpool2".to_string(), &self.maxpool2),
            ("Mixed_5b".to_string(), &self.Mixed_5b),
            ("Mixed_5c".to_string(), &self.Mixed_5c),
            ("Mixed_5d".to_string(), &self.Mixed_5d),
            ("Mixed_6a".to_string(), &self.Mixed_6a),
            ("Mixed_6b".to_string(), &self.Mixed_6b),
            ("Mixed_6c".to_string(), &self.Mixed_6c),
            ("Mixed_6d".to_string(), &self.Mixed_6d),
            ("Mixed_6e".to_string(), &self.Mixed_6e),
            ("Mixed_7a".to_string(), &self.Mixed_7a),
            ("Mixed_7b".to_string(), &self.Mixed_7b),
            ("Mixed_7c".to_string(), &self.Mixed_7c),
            ("avgpool".to_string(), &self.avgpool),
            ("dropout".to_string(), &self.dropout),
            ("fc".to_string(), &self.fc),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.Conv2d_1a_3x3.train();
        self.Conv2d_2a_3x3.train();
        self.Conv2d_2b_3x3.train();
        self.Conv2d_3b_1x1.train();
        self.Conv2d_4a_3x3.train();
        self.Mixed_5b.train();
        self.Mixed_5c.train();
        self.Mixed_5d.train();
        self.Mixed_6a.train();
        self.Mixed_6b.train();
        self.Mixed_6c.train();
        self.Mixed_6d.train();
        self.Mixed_6e.train();
        self.Mixed_7a.train();
        self.Mixed_7b.train();
        self.Mixed_7c.train();
        self.dropout.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.Conv2d_1a_3x3.eval();
        self.Conv2d_2a_3x3.eval();
        self.Conv2d_2b_3x3.eval();
        self.Conv2d_3b_1x1.eval();
        self.Conv2d_4a_3x3.eval();
        self.Mixed_5b.eval();
        self.Mixed_5c.eval();
        self.Mixed_5d.eval();
        self.Mixed_6a.eval();
        self.Mixed_6b.eval();
        self.Mixed_6c.eval();
        self.Mixed_6d.eval();
        self.Mixed_6e.eval();
        self.Mixed_7a.eval();
        self.Mixed_7b.eval();
        self.Mixed_7c.eval();
        self.dropout.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// Convenience constructor for Inception-V3 (`aux_logits=False`).
pub fn inception_v3<T: Float>(num_classes: usize) -> FerrotorchResult<InceptionV3<T>> {
    InceptionV3::new(num_classes)
}

// ---------------------------------------------------------------------------
// IntermediateFeatures — CL-499
// ---------------------------------------------------------------------------

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for InceptionV3<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        let x = self.Conv2d_1a_3x3.forward(input)?;
        out.insert("Conv2d_1a_3x3".to_string(), x.clone());
        let x = self.Conv2d_2a_3x3.forward(&x)?;
        out.insert("Conv2d_2a_3x3".to_string(), x.clone());
        let x = self.Conv2d_2b_3x3.forward(&x)?;
        out.insert("Conv2d_2b_3x3".to_string(), x.clone());
        let x = Module::<T>::forward(&self.maxpool1, &x)?;
        let x = self.Conv2d_3b_1x1.forward(&x)?;
        out.insert("Conv2d_3b_1x1".to_string(), x.clone());
        let x = self.Conv2d_4a_3x3.forward(&x)?;
        out.insert("Conv2d_4a_3x3".to_string(), x.clone());
        let x = Module::<T>::forward(&self.maxpool2, &x)?;

        let x = self.Mixed_5b.forward(&x)?;
        out.insert("Mixed_5b".to_string(), x.clone());
        let x = self.Mixed_5c.forward(&x)?;
        out.insert("Mixed_5c".to_string(), x.clone());
        let x = self.Mixed_5d.forward(&x)?;
        out.insert("Mixed_5d".to_string(), x.clone());
        let x = self.Mixed_6a.forward(&x)?;
        out.insert("Mixed_6a".to_string(), x.clone());
        let x = self.Mixed_6b.forward(&x)?;
        out.insert("Mixed_6b".to_string(), x.clone());
        let x = self.Mixed_6c.forward(&x)?;
        out.insert("Mixed_6c".to_string(), x.clone());
        let x = self.Mixed_6d.forward(&x)?;
        out.insert("Mixed_6d".to_string(), x.clone());
        let x = self.Mixed_6e.forward(&x)?;
        out.insert("Mixed_6e".to_string(), x.clone());
        let x = self.Mixed_7a.forward(&x)?;
        out.insert("Mixed_7a".to_string(), x.clone());
        let x = self.Mixed_7b.forward(&x)?;
        out.insert("Mixed_7b".to_string(), x.clone());
        let x = self.Mixed_7c.forward(&x)?;
        out.insert("Mixed_7c".to_string(), x.clone());

        let x = Module::<T>::forward(&self.avgpool, &x)?;
        out.insert("avgpool".to_string(), x.clone());
        let x = Module::<T>::forward(&self.dropout, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        let logits = self.fc.forward(&x)?;
        out.insert("fc".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        vec![
            "Conv2d_1a_3x3".to_string(),
            "Conv2d_2a_3x3".to_string(),
            "Conv2d_2b_3x3".to_string(),
            "Conv2d_3b_1x1".to_string(),
            "Conv2d_4a_3x3".to_string(),
            "Mixed_5b".to_string(),
            "Mixed_5c".to_string(),
            "Mixed_5d".to_string(),
            "Mixed_6a".to_string(),
            "Mixed_6b".to_string(),
            "Mixed_6c".to_string(),
            "Mixed_6d".to_string(),
            "Mixed_6e".to_string(),
            "Mixed_7a".to_string(),
            "Mixed_7b".to_string(),
            "Mixed_7c".to_string(),
            "avgpool".to_string(),
            "fc".to_string(),
        ]
    }
}

// ===========================================================================
// Tests — per-module forward shape (Phase 10 Level 2 probe)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;

    fn dummy_image(batch: usize, ch: usize, h: usize, w: usize) -> Tensor<f32> {
        let numel = batch * ch * h * w;
        let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 1e-3).collect();
        Tensor::from_storage(TensorStorage::cpu(data), vec![batch, ch, h, w], false).unwrap()
    }

    // ── BasicConv2d ──────────────────────────────────────────────────────

    #[test]
    fn basic_conv2d_forward_shape() {
        let bc: BasicConv2d<f32> = BasicConv2d::new(3, 16, (3, 3), (2, 2), (0, 0)).unwrap();
        let x = dummy_image(1, 3, 16, 16);
        let y = bc.forward(&x).unwrap();
        // (16 - 3) / 2 + 1 = 7
        assert_eq!(y.shape(), &[1, 16, 7, 7]);

        // named_parameters surface conv.weight, bn.weight, bn.bias
        let keys: Vec<String> = bc.named_parameters().into_iter().map(|(n, _)| n).collect();
        assert!(keys.contains(&"conv.weight".to_string()));
        assert!(keys.contains(&"bn.weight".to_string()));
        assert!(keys.contains(&"bn.bias".to_string()));
        // Conv has bias=false, so no conv.bias key.
        assert!(!keys.contains(&"conv.bias".to_string()));
    }

    // ── InceptionA ───────────────────────────────────────────────────────

    #[test]
    fn inception_a_forward_shape_and_keys() {
        // Mixed_5b: in=192, pool_features=32 → out = 64+64+96+32 = 256.
        let block: InceptionA<f32> = InceptionA::new(192, 32).unwrap();
        let x = dummy_image(1, 192, 35, 35);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 256, 35, 35]);

        let keys: Vec<String> = block
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for sub in [
            "branch1x1",
            "branch5x5_1",
            "branch5x5_2",
            "branch3x3dbl_1",
            "branch3x3dbl_2",
            "branch3x3dbl_3",
            "branch_pool",
        ] {
            assert!(
                keys.contains(&format!("{sub}.conv.weight")),
                "missing {sub}.conv.weight"
            );
            assert!(
                keys.contains(&format!("{sub}.bn.weight")),
                "missing {sub}.bn.weight"
            );
        }
    }

    // ── InceptionB ───────────────────────────────────────────────────────

    #[test]
    fn inception_b_forward_shape_and_keys() {
        // Mixed_6a: in=288 → out = 384 + 96 + 288 = 768. Spatial: 35→17.
        let block: InceptionB<f32> = InceptionB::new(288).unwrap();
        let x = dummy_image(1, 288, 35, 35);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 768, 17, 17]);

        let keys: Vec<String> = block
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for sub in [
            "branch3x3",
            "branch3x3dbl_1",
            "branch3x3dbl_2",
            "branch3x3dbl_3",
        ] {
            assert!(
                keys.contains(&format!("{sub}.conv.weight")),
                "missing {sub}.conv.weight"
            );
        }
        // No branch_pool conv in InceptionB.
        assert!(!keys.iter().any(|k| k.starts_with("branch_pool.")));
    }

    // ── InceptionC ───────────────────────────────────────────────────────

    #[test]
    fn inception_c_forward_shape_and_keys() {
        // Mixed_6b: in=768, c7=128 → out = 192*4 = 768.
        let block: InceptionC<f32> = InceptionC::new(768, 128).unwrap();
        let x = dummy_image(1, 768, 17, 17);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 768, 17, 17]);

        let keys: Vec<String> = block
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for sub in [
            "branch1x1",
            "branch7x7_1",
            "branch7x7_2",
            "branch7x7_3",
            "branch7x7dbl_1",
            "branch7x7dbl_2",
            "branch7x7dbl_3",
            "branch7x7dbl_4",
            "branch7x7dbl_5",
            "branch_pool",
        ] {
            assert!(
                keys.contains(&format!("{sub}.conv.weight")),
                "missing {sub}.conv.weight"
            );
        }
    }

    // ── InceptionD ───────────────────────────────────────────────────────

    #[test]
    fn inception_d_forward_shape_and_keys() {
        // Mixed_7a: in=768 → out = 320 + 192 + 768 = 1280. Spatial: 17→8.
        let block: InceptionD<f32> = InceptionD::new(768).unwrap();
        let x = dummy_image(1, 768, 17, 17);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1280, 8, 8]);

        let keys: Vec<String> = block
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for sub in [
            "branch3x3_1",
            "branch3x3_2",
            "branch7x7x3_1",
            "branch7x7x3_2",
            "branch7x7x3_3",
            "branch7x7x3_4",
        ] {
            assert!(
                keys.contains(&format!("{sub}.conv.weight")),
                "missing {sub}.conv.weight"
            );
        }
    }

    // ── InceptionE (parallel branches) ──────────────────────────────────

    #[test]
    fn inception_e_forward_shape_and_parallel_concat() {
        // Mixed_7b: in=1280 → out = 320 + (384+384) + (384+384) + 192 = 2048.
        let block: InceptionE<f32> = InceptionE::new(1280).unwrap();
        let x = dummy_image(1, 1280, 8, 8);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2048, 8, 8]);

        let keys: Vec<String> = block
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for sub in [
            "branch1x1",
            "branch3x3_1",
            "branch3x3_2a",
            "branch3x3_2b",
            "branch3x3dbl_1",
            "branch3x3dbl_2",
            "branch3x3dbl_3a",
            "branch3x3dbl_3b",
            "branch_pool",
        ] {
            assert!(
                keys.contains(&format!("{sub}.conv.weight")),
                "missing {sub}.conv.weight"
            );
        }
    }

    // ── InceptionV3 — top-level ─────────────────────────────────────────

    #[test]
    fn inception_v3_output_shape_299() {
        // 299×299 — torchvision canonical input.
        let model: InceptionV3<f32> = inception_v3(10).unwrap();
        let x = dummy_image(1, 3, 299, 299);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 10]);
    }

    #[test]
    fn inception_v3_custom_classes_299() {
        let model: InceptionV3<f32> = inception_v3(3).unwrap();
        let x = dummy_image(1, 3, 299, 299);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
    }

    #[test]
    fn inception_v3_param_count_matches_torchvision() {
        // Reference (independently verified):
        //   sum(p.numel() for p in tvm.inception_v3(
        //     weights=None, aux_logits=False, init_weights=True).parameters())
        //   == 23_834_568
        // Exact equality — failure mode #26 (block-config-translation-error)
        // would surface as ANY divergence from the canonical count.
        let model: InceptionV3<f32> = inception_v3(1000).unwrap();
        let n = model.num_parameters();
        assert_eq!(
            n, 23_834_568,
            "InceptionV3 param count {n} != torchvision reference 23,834,568",
        );
    }

    #[test]
    fn inception_v3_named_parameters_top_level_prefixes() {
        let model: InceptionV3<f32> = inception_v3(10).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for prefix in [
            "Conv2d_1a_3x3.",
            "Conv2d_2a_3x3.",
            "Conv2d_2b_3x3.",
            "Conv2d_3b_1x1.",
            "Conv2d_4a_3x3.",
            "Mixed_5b.",
            "Mixed_5c.",
            "Mixed_5d.",
            "Mixed_6a.",
            "Mixed_6b.",
            "Mixed_6c.",
            "Mixed_6d.",
            "Mixed_6e.",
            "Mixed_7a.",
            "Mixed_7b.",
            "Mixed_7c.",
            "fc.",
        ] {
            assert!(
                names.iter().any(|n| n.starts_with(prefix)),
                "missing parameter prefix {prefix:?}",
            );
        }
        // BasicConv2d sub-parameters surface as `<prefix>.conv.weight` and
        // `<prefix>.bn.{weight,bias}` — torchvision parity.
        assert!(names.contains(&"Conv2d_1a_3x3.conv.weight".to_string()));
        assert!(names.contains(&"Conv2d_1a_3x3.bn.weight".to_string()));
        assert!(names.contains(&"Conv2d_1a_3x3.bn.bias".to_string()));
        assert!(!names.contains(&"Conv2d_1a_3x3.conv.bias".to_string()));
    }

    #[test]
    fn inception_v3_train_eval() {
        let mut model: InceptionV3<f32> = inception_v3(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }
}
