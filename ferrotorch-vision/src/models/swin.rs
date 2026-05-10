//! Swin Transformer Tiny — torchvision-parity implementation (Phase 11 #998).
//!
//! Implements `torchvision.models.swin_t` end-to-end with shifted-window
//! self-attention. The named-parameter schema matches torchvision exactly so
//! the strict value-parity loader can ingest a torchvision swin_t state dict
//! without a per-key remap.
//!
//! Architecture (closes #998):
//!
//! ```text
//! features.0:               # PatchEmbed
//!   features.0.0  Conv2d(3, 96, k=4, s=4, bias=true)
//!   features.0.1  Permute([0, 2, 3, 1])     (no params)
//!   features.0.2  LayerNorm(96)
//!
//! features.{1,3,5,7}:       # 4 stage Sequential[SwinBlock]
//!   features.<i>.<j>.norm1                LayerNorm(dim)
//!   features.<i>.<j>.attn.qkv             Linear(dim, 3*dim, bias=true)
//!   features.<i>.<j>.attn.proj            Linear(dim, dim, bias=true)
//!   features.<i>.<j>.attn.relative_position_bias_table  Parameter[(2*ws-1)^2, num_heads]
//!   features.<i>.<j>.norm2                LayerNorm(dim)
//!   features.<i>.<j>.mlp.0                Linear(dim, 4*dim, bias=true)
//!   features.<i>.<j>.mlp.3                Linear(4*dim, dim, bias=true)
//!
//! features.{2,4,6}:         # PatchMerging
//!   features.<i>.norm                     LayerNorm(4*dim)
//!   features.<i>.reduction                Linear(4*dim, 2*dim, bias=false)
//!
//! norm:                    LayerNorm(8 * embed_dim)
//! head:                    Linear(8 * embed_dim, num_classes)
//! ```
//!
//! Stage configuration for swin_t:
//! - patch_size = 4
//! - embed_dim  = 96
//! - depths     = [2, 2, 6, 2]
//! - num_heads  = [3, 6, 12, 24]
//! - window_size = 7
//! - mlp_ratio  = 4
//!
//! Per-block shift_size alternates `0` and `floor(window_size/2) = 3`.
//! When the spatial dim is `<= window_size` the runtime guard forces
//! `shift_size = 0` (matches torchvision's `if window_size[0] >= pad_H:
//! shift_size[0] = 0` in `shifted_window_attention`). For the swin_t
//! pipeline (224 → 56 → 28 → 14 → 7) only the final 7×7 stage triggers
//! this; earlier stages always shift on odd blocks.
//!
//! ## Eval-mode parity vs. training
//!
//! The cyclic-shift primitive `ferrotorch_core::ops::tensor_ops::roll`
//! (line 167 of tensor_ops.rs) is implemented as a CPU `data_vec()` →
//! reconstruct cycle that returns `requires_grad = false`. It does NOT
//! propagate gradients. This implementation is therefore correct under
//! `no_grad` (eval-mode) — which is what the value-parity test exercises
//! — but a full `roll` autograd backward is required before training a
//! Swin-T model on real data. That backward is tracked separately
//! (#1014); see `ShiftedWindowAttention::forward` for the load-bearing
//! call site.

use ferrotorch_core::creation::zeros;
use ferrotorch_core::grad_fns::activation::softmax;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::linalg::matmul_differentiable;
use ferrotorch_core::grad_fns::reduction::mean_dim;
use ferrotorch_core::grad_fns::shape::cat;
use ferrotorch_core::ops::tensor_ops::roll;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage};

use ferrotorch_nn::activation::GELU;
use ferrotorch_nn::conv::Conv2d;
use ferrotorch_nn::linear::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::LayerNorm;
use ferrotorch_nn::parameter::Parameter;

// ===========================================================================
// PatchEmbed — features.0
// ===========================================================================
//
// Three named children (`0`, `1`, `2`) mirroring torchvision's
// `nn.Sequential(Conv2d, Permute, LayerNorm)`. The Permute child is a
// pure shape op with no parameters — keeping it here purely so the
// `features.0.<i>` numbering matches torchvision's state-dict keys
// (the Permute slot is index 1; LayerNorm is index 2). Forward applies
// Conv2d then permute(0,2,3,1) then LayerNorm to land in [B,H,W,C]
// layout for downstream window attention.

struct PatchEmbed<T: Float> {
    conv: Conv2d<T>,
    norm: LayerNorm<T>,
    training: bool,
}

impl<T: Float> PatchEmbed<T> {
    fn new(in_channels: usize, embed_dim: usize, patch_size: usize) -> FerrotorchResult<Self> {
        let conv = Conv2d::new(
            in_channels,
            embed_dim,
            (patch_size, patch_size),
            (patch_size, patch_size),
            (0, 0),
            true, // torchvision swin_t Conv2d carries bias=True
        )?;
        let norm = LayerNorm::new(vec![embed_dim], 1e-5, true)?;
        Ok(Self {
            conv,
            norm,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for PatchEmbed<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // input: [B, 3, H, W]
        let x = self.conv.forward(input)?;
        // [B, C, H', W'] -> [B, H', W', C] via permute().contiguous()
        let x = x.permute(&[0, 2, 3, 1])?.contiguous()?;
        // LayerNorm normalizes over the last dim (C).
        self.norm.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.conv.parameters();
        p.extend(self.norm.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.conv.parameters_mut();
        p.extend(self.norm.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // Match torchvision's nn.Sequential(Conv2d, Permute, LayerNorm)
        // → Conv2d at index 0, LayerNorm at index 2 (Permute=1 has no params).
        let mut out = Vec::new();
        for (n, p) in self.conv.named_parameters() {
            out.push((format!("0.{n}"), p));
        }
        for (n, p) in self.norm.named_parameters() {
            out.push((format!("2.{n}"), p));
        }
        out
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.conv, &self.norm]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("0".to_string(), &self.conv as &dyn Module<T>),
            ("2".to_string(), &self.norm as &dyn Module<T>),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.conv.train();
        self.norm.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv.eval();
        self.norm.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// ShiftedWindowAttention — Phase 11 #998 core primitive
// ===========================================================================
//
// Window-based multi-head self-attention with cyclic-shift support and
// learned relative-position bias. Mirrors
// `torchvision.models.swin_transformer.ShiftedWindowAttention` exactly
// in both parameter layout and forward arithmetic.
//
// Parameter set (matches torchvision keys):
// - `qkv.{weight,bias}`           : Linear(dim, 3*dim, bias=true)
// - `proj.{weight,bias}`          : Linear(dim, dim, bias=true)
// - `relative_position_bias_table`: Parameter[(2*ws-1)^2, num_heads]
//
// `relative_position_index` is a precomputed Vec<i64> field. It is
// EXPLICITLY NOT a Parameter (per the Phase 11 pre-flight contract,
// failure mode #36 "relative_position_index-as-Parameter") and NOT
// surfaced via `named_parameters` — the fixture descriptor lists it
// under `skipped_int_buffer_keys` because torchvision serializes it
// as an int64 buffer that the safetensors payload deliberately omits.
struct ShiftedWindowAttention<T: Float> {
    qkv: Linear<T>,
    proj: Linear<T>,
    relative_position_bias_table: Parameter<T>,
    /// Precomputed integer indices of shape [ws*ws * ws*ws], values in
    /// `0..(2*ws-1)*(2*ws-1)`. NOT a Parameter — see #36.
    relative_position_index: Vec<i64>,
    window_size: usize,
    shift_size: usize,
    num_heads: usize,
    training: bool,
}

impl<T: Float> ShiftedWindowAttention<T> {
    fn new(
        dim: usize,
        window_size: usize,
        shift_size: usize,
        num_heads: usize,
    ) -> FerrotorchResult<Self> {
        let qkv = Linear::new(dim, 3 * dim, true)?;
        let proj = Linear::new(dim, dim, true)?;
        let table_rows = (2 * window_size - 1) * (2 * window_size - 1);
        let relative_position_bias_table = Parameter::zeros(&[table_rows, num_heads])?;
        let relative_position_index = compute_relative_position_index(window_size);
        Ok(Self {
            qkv,
            proj,
            relative_position_bias_table,
            relative_position_index,
            window_size,
            shift_size,
            num_heads,
            training: true,
        })
    }

    /// Build the per-pair relative-position bias tensor of shape
    /// `[1, num_heads, N, N]` where `N = ws*ws`.
    ///
    /// Mirrors torchvision's `_get_relative_position_bias`:
    ///   bias_table[index].view(N, N, num_heads).permute(2, 0, 1)
    ///
    /// We perform the integer-index gather on CPU because:
    /// 1. `relative_position_index` is NOT a Parameter — there are no
    ///    grad implications from the gather itself.
    /// 2. The gather is over a tiny `[(2*ws-1)^2, num_heads]` table
    ///    (largest is `[169, 24] = 16 KB f32`), so a `data_vec()` pull
    ///    here does not touch any large activation tensor — failure
    ///    mode #15 (CPU-pull-disguised-as-device-op for HOT-PATH
    ///    activations) does NOT apply to this small bias gather.
    /// 3. The output of this helper is a fresh tensor that downstream
    ///    `add` consumes with full broadcast semantics; the eval-mode
    ///    value path is identical to torchvision's bit-for-bit.
    ///
    /// For training-time gradient flow into the bias TABLE itself, a
    /// future migration would route through a differentiable gather
    /// (e.g. `index_select` over the flattened table); tracked under
    /// the same #1014 follow-up that covers `roll` backward. Eval-mode
    /// parity does not require it.
    fn build_relative_position_bias(&self) -> FerrotorchResult<Tensor<T>> {
        let ws = self.window_size;
        let n = ws * ws;
        let nh = self.num_heads;
        let table_data = self.relative_position_bias_table.data_vec()?;
        let mut gathered = Vec::with_capacity(n * n * nh);
        for &idx_i64 in &self.relative_position_index {
            let row_start = (idx_i64 as usize) * nh;
            gathered.extend_from_slice(&table_data[row_start..row_start + nh]);
        }
        // gathered is laid out as [N*N, num_heads]. View as [N, N, num_heads]
        // then permute to [num_heads, N, N], then unsqueeze leading 1 →
        // [1, num_heads, N, N] for batched broadcast over [B*nW, num_heads, N, N].
        let bias = Tensor::from_storage(
            TensorStorage::cpu(gathered),
            vec![n, n, nh],
            false, // not part of the autograd graph for eval-mode
        )?
        .to(self.relative_position_bias_table.tensor().device())?
        .permute(&[2, 0, 1])?
        .contiguous()?
        .view(&[1, nh as i64, n as i64, n as i64])?;
        Ok(bias)
    }
}

impl<T: Float> Module<T> for ShiftedWindowAttention<T> {
    /// Forward.
    ///
    /// Input layout: `[B, H, W, C]`.
    /// Output layout: `[B, H, W, C]`.
    ///
    /// The implementation mirrors torchvision's
    /// `shifted_window_attention(...)` step-for-step:
    ///   1. zero-pad on bottom/right so `pad_H % ws == 0` and `pad_W % ws == 0`
    ///   2. cyclic shift via `roll` if `shift_size > 0`
    ///   3. window partition via view().permute().contiguous().view()
    ///      — NEVER via data_vec() (failure mode #15)
    ///   4. fused QKV, head split, scale, scores
    ///   5. add relative_position_bias broadcast
    ///   6. add attention mask for shifted blocks (precomputed in
    ///      `build_attn_mask`)
    ///   7. softmax + matmul with V
    ///   8. window reverse via the inverse permute chain
    ///   9. reverse cyclic shift
    ///  10. unpad to original H, W
    ///
    /// For swin_t @ 224×224 every spatial dim is divisible by ws=7
    /// (56, 28, 14, 7), so step (1) and (10) are no-ops on the
    /// reference path; the padding code runs only when callers feed
    /// non-multiple-of-7 spatials (e.g. the small-input shape tests).
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        let batch = shape[0];
        let orig_h = shape[1];
        let orig_w = shape[2];
        let channels = shape[3];
        let ws = self.window_size;
        let nh = self.num_heads;
        let head_dim = channels / nh;

        // 1. Pad to a multiple of ws on H and W (torchvision F.pad).
        let pad_b = (ws - orig_h % ws) % ws;
        let pad_r = (ws - orig_w % ws) % ws;
        let padded = if pad_b == 0 && pad_r == 0 {
            input.clone()
        } else {
            pad_bhwc_zero(input, pad_b, pad_r)?
        };
        let pad_shape = padded.shape();
        let height = pad_shape[1];
        let width = pad_shape[2];

        let n_h = height / ws;
        let n_w = width / ws;
        let num_windows = n_h * n_w;

        // Mirrors torchvision: "if window_size[0] >= pad_H: shift_size[0] = 0".
        // In swin_t this applies to the 7×7 final stage.
        let effective_shift = if ws >= height || ws >= width {
            0
        } else {
            self.shift_size
        };

        // 2. Cyclic shift. `roll` shifts +shift along the dimension; the
        //    forward shift is by `-shift_size` per torchvision, so we
        //    pass a negative shift on dims 1 (H) and 2 (W).
        let shifted = if effective_shift > 0 {
            let s = effective_shift as i64;
            let r1 = roll(&padded, -s, 1)?;
            roll(&r1, -s, 2)?
        } else {
            padded
        };

        // 3. Window partition via the autograd-correct primitive chain
        //    (failure mode #15: NEVER via data_vec()):
        //    [B, H, W, C]
        //      .view([B, n_h, ws, n_w, ws, C])
        //      .permute([0, 1, 3, 2, 4, 5])           # group window dims
        //      .contiguous()
        //      .view([B*num_windows, ws*ws, C])
        let n = ws * ws;
        let x_windows = shifted
            .view(&[
                batch as i64,
                n_h as i64,
                ws as i64,
                n_w as i64,
                ws as i64,
                channels as i64,
            ])?
            .permute(&[0, 1, 3, 2, 4, 5])?
            .contiguous()?
            .view(&[(batch * num_windows) as i64, n as i64, channels as i64])?;

        // 4. Fused QKV projection. `qkv` is Linear(dim, 3*dim).
        //    Output: [B*nW, N, 3*C]. Reshape to [B*nW, N, 3, num_heads, head_dim]
        //    then permute to [3, B*nW, num_heads, N, head_dim].
        let qkv = self.qkv.forward(&x_windows)?;
        let qkv = qkv
            .view(&[
                (batch * num_windows) as i64,
                n as i64,
                3,
                nh as i64,
                head_dim as i64,
            ])?
            .permute(&[2, 0, 3, 1, 4])?
            .contiguous()?;

        // Split across the leading dim using narrow → contiguous.
        // Each slice has shape [B*nW, num_heads, N, head_dim].
        let q = qkv.narrow(0, 0, 1)?.contiguous()?.view(&[
            (batch * num_windows) as i64,
            nh as i64,
            n as i64,
            head_dim as i64,
        ])?;
        let k = qkv.narrow(0, 1, 1)?.contiguous()?.view(&[
            (batch * num_windows) as i64,
            nh as i64,
            n as i64,
            head_dim as i64,
        ])?;
        let v = qkv.narrow(0, 2, 1)?.contiguous()?.view(&[
            (batch * num_windows) as i64,
            nh as i64,
            n as i64,
            head_dim as i64,
        ])?;

        // q ← q * (1/sqrt(head_dim))   (torchvision: `q = q * (C // num_heads) ** -0.5`)
        let scale_val = T::from(1.0 / (head_dim as f64).sqrt()).unwrap();
        let scale_tensor = Tensor::from_storage(
            TensorStorage::on_device(vec![scale_val], q.device())?,
            vec![1],
            false,
        )?;
        let q_scaled = mul(&q, &scale_tensor)?;

        // attn = q @ k^T  → [B*nW, num_heads, N, N]
        let k_t = k.permute(&[0, 1, 3, 2])?.contiguous()?;
        let attn = matmul_differentiable(&q_scaled, &k_t)?;

        // 5. Add relative position bias.
        //    bias is [1, num_heads, N, N]; broadcasts over [B*nW, num_heads, N, N].
        let bias = self.build_relative_position_bias()?;
        let attn = add(&attn, &bias)?;

        // 6. For shifted blocks, add the attention mask. The mask is built
        //    fresh each forward call because it depends on (H, W, ws,
        //    shift_size) — all of which are runtime values.
        let attn = if effective_shift > 0 {
            let mask = build_attn_mask::<T>(height, width, ws, effective_shift, attn.device())?;
            // mask shape: [num_windows, N, N]. Reshape attn to
            //   [B, num_windows, num_heads, N, N], add mask broadcast across
            //   batch+heads via [1, num_windows, 1, N, N], reshape back to
            //   [B*num_windows, num_heads, N, N].
            let attn_5d = attn.view(&[
                batch as i64,
                num_windows as i64,
                nh as i64,
                n as i64,
                n as i64,
            ])?;
            let mask_5d = mask.view(&[1, num_windows as i64, 1, n as i64, n as i64])?;
            let attn_masked = add(&attn_5d, &mask_5d)?;
            attn_masked.contiguous()?.view(&[
                (batch * num_windows) as i64,
                nh as i64,
                n as i64,
                n as i64,
            ])?
        } else {
            attn
        };

        // 7. softmax over last dim, then matmul with V.
        let attn = softmax(&attn)?;
        // attn @ v  → [B*nW, num_heads, N, head_dim]
        let context = matmul_differentiable(&attn, &v)?;

        // Reshape to [B*nW, N, C] via permute(0, 2, 1, 3) + contiguous + view.
        let context = context.permute(&[0, 2, 1, 3])?.contiguous()?.view(&[
            (batch * num_windows) as i64,
            n as i64,
            channels as i64,
        ])?;

        // 8. Output projection.
        let context = self.proj.forward(&context)?;

        // 9. Reverse window partition:
        //    [B*nW, N, C]
        //      .view([B, n_h, n_w, ws, ws, C])
        //      .permute([0, 1, 3, 2, 4, 5])
        //      .contiguous()
        //      .view([B, H, W, C])
        let out = context
            .view(&[
                batch as i64,
                n_h as i64,
                n_w as i64,
                ws as i64,
                ws as i64,
                channels as i64,
            ])?
            .permute(&[0, 1, 3, 2, 4, 5])?
            .contiguous()?
            .view(&[batch as i64, height as i64, width as i64, channels as i64])?;

        // 10. Reverse cyclic shift.
        let out = if effective_shift > 0 {
            let s = effective_shift as i64;
            let r1 = roll(&out, s, 1)?;
            roll(&r1, s, 2)?
        } else {
            out
        };

        // 11. Unpad to original H, W (no-op when pad_b == pad_r == 0).
        let out = if pad_b == 0 && pad_r == 0 {
            out
        } else {
            unpad_bhwc(&out, orig_h, orig_w)?
        };

        Ok(out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.qkv.parameters();
        p.extend(self.proj.parameters());
        p.push(&self.relative_position_bias_table);
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.qkv.parameters_mut();
        p.extend(self.proj.parameters_mut());
        p.push(&mut self.relative_position_bias_table);
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        // NOTE: torchvision sorts state_dict alphabetically per
        // `state_dict()` semantics; the strict loader walks both sides,
        // so the order here is informational, not contract-bearing.
        for (n, p) in self.qkv.named_parameters() {
            out.push((format!("qkv.{n}"), p));
        }
        for (n, p) in self.proj.named_parameters() {
            out.push((format!("proj.{n}"), p));
        }
        out.push((
            "relative_position_bias_table".to_string(),
            &self.relative_position_bias_table,
        ));
        // INTENTIONALLY OMITTED: relative_position_index. It is a
        // non-trainable integer lookup table. See failure mode #36.
        out
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.qkv, &self.proj]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("qkv".to_string(), &self.qkv as &dyn Module<T>),
            ("proj".to_string(), &self.proj as &dyn Module<T>),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.qkv.train();
        self.proj.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.qkv.eval();
        self.proj.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

/// Zero-pad a `[B, H, W, C]` tensor on the bottom (`pad_b` rows) and the
/// right (`pad_r` columns).
///
/// Mirrors `F.pad(input, (0, 0, 0, pad_r, 0, pad_b), value=0)` from
/// torchvision's `shifted_window_attention`. Goes through CPU because
/// ferrotorch lacks a generic spatial-pad primitive; the swin_t parity
/// path at 224×224 never invokes it (every stage's spatial dim is
/// divisible by ws=7), so this is exercised only by the small-input
/// shape tests where padding actually fires. Tracked alongside the
/// other Phase-11 pure-CPU helpers under the same #1014 follow-up.
/// Pad a `[B, H, W, C]` tensor with zeros on the bottom (`pad_b` rows) and
/// right (`pad_r` columns).
///
/// Pass 2B (#1000): rebuilt as two `cat`s of small zero tensors instead
/// of a `data_vec()` round trip + indexed write back. Stays on
/// `input.device()` end-to-end; on GPU the cats hit `strided_cat`'s
/// device-resident fast path (see `grad_fns::shape::cat`).
fn pad_bhwc_zero<T: Float>(
    input: &Tensor<T>,
    pad_b: usize,
    pad_r: usize,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let b = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];
    if pad_b == 0 && pad_r == 0 {
        return Ok(input.clone());
    }
    let device = input.device();

    // Pad bottom: cat(input, zeros[B, pad_b, W, C], axis=1).
    let with_bottom = if pad_b > 0 {
        let zb = zeros::<T>(&[b, pad_b, w, c])?.to(device)?;
        cat(&[input.clone(), zb], 1)?
    } else {
        input.clone()
    };

    // Pad right: cat(with_bottom, zeros[B, H+pad_b, pad_r, C], axis=2).
    if pad_r > 0 {
        let zr = zeros::<T>(&[b, h + pad_b, pad_r, c])?.to(device)?;
        cat(&[with_bottom, zr], 2)
    } else {
        Ok(with_bottom)
    }
}

/// Inverse of [`pad_bhwc_zero`]: drop the bottom and right padding rows
/// to restore an input's original `[B, orig_h, orig_w, C]` shape.
///
/// Pass 2B (#1000): rebuilt as two `narrow` views (zero-copy) instead of
/// a `data_vec()` + per-element copy. NarrowBackward zero-fills the
/// dropped rows on backward, matching the slice semantics.
fn unpad_bhwc<T: Float>(
    input: &Tensor<T>,
    orig_h: usize,
    orig_w: usize,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let h = shape[1];
    let w = shape[2];
    if h == orig_h && w == orig_w {
        return Ok(input.clone());
    }
    let mut x = input.clone();
    if h != orig_h {
        x = x.narrow(1, 0, orig_h)?;
    }
    if w != orig_w {
        x = x.narrow(2, 0, orig_w)?;
    }
    // narrow returns a non-contiguous view; subsequent cats / reshapes
    // that follow PatchMerging require contiguous memory.
    x.contiguous()
}

/// Compute torchvision's `relative_position_index` for a square window.
///
/// Output: `Vec<i64>` of length `ws*ws*ws*ws`. Each entry is in the
/// range `0..(2*ws-1)*(2*ws-1)` and indexes into the per-pair
/// relative-position bias table. Faithfully reproduces the meshgrid
/// → flatten → pairwise diff → axis-shift → row-major-flatten chain
/// from torchvision (`define_relative_position_index`).
fn compute_relative_position_index(window_size: usize) -> Vec<i64> {
    let ws = window_size as i64;
    let n = (ws * ws) as usize;
    // coords[i] = (h_i, w_i) for token i in row-major order.
    let mut coords: Vec<(i64, i64)> = Vec::with_capacity(n);
    for h in 0..ws {
        for w in 0..ws {
            coords.push((h, w));
        }
    }
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let dh = coords[i].0 - coords[j].0 + (ws - 1);
            let dw = coords[i].1 - coords[j].1 + (ws - 1);
            // torchvision: relative_coords[..., 0] *= 2*ws-1 then sum(-1).
            let idx = dh * (2 * ws - 1) + dw;
            out.push(idx);
        }
    }
    out
}

/// Build the additive attention mask for a shifted-window block.
///
/// The mask has shape `[num_windows, N, N]` where `N = ws*ws`. Pairs
/// of tokens that originated in the same pre-shift region get 0;
/// pairs that crossed a region boundary get `-100.0` (matching
/// torchvision's `masked_fill(... != 0, -100.0)`).
///
/// Mirrors torchvision's `shifted_window_attention` mask construction
/// exactly: build a [pad_H, pad_W] count grid where each of 9 regions
/// (3 H-slices × 3 W-slices) gets a unique integer id, partition into
/// windows, then `mask[..,i,j] = (region[i] == region[j]) ? 0 : -100`.
fn build_attn_mask<T: Float>(
    height: usize,
    width: usize,
    window_size: usize,
    shift_size: usize,
    device: ferrotorch_core::Device,
) -> FerrotorchResult<Tensor<T>> {
    let ws = window_size;
    let n = ws * ws;
    let n_h = height / ws;
    let n_w = width / ws;
    let num_windows = n_h * n_w;

    // 1. Build the [H, W] region-id grid.
    //    h_slices = [(0, H-ws), (H-ws, H-shift), (H-shift, H)]
    //    w_slices analogously. There are 3*3 = 9 regions; each window
    //    in the post-roll layout sees a subset of these.
    let h_slices: [(usize, usize); 3] = [
        (0, height.saturating_sub(ws)),
        (height.saturating_sub(ws), height - shift_size),
        (height - shift_size, height),
    ];
    let w_slices: [(usize, usize); 3] = [
        (0, width.saturating_sub(ws)),
        (width.saturating_sub(ws), width - shift_size),
        (width - shift_size, width),
    ];

    let mut region: Vec<i64> = vec![0; height * width];
    let mut count: i64 = 0;
    for (hs, he) in h_slices.iter() {
        for (ws_lo, ws_hi) in w_slices.iter() {
            for h in *hs..*he {
                for w in *ws_lo..*ws_hi {
                    region[h * width + w] = count;
                }
            }
            count += 1;
        }
    }

    // 2. Partition into windows: [n_h, ws, n_w, ws] → permute to
    //    [n_h, n_w, ws, ws] → flatten to [num_windows, ws*ws].
    let mut windowed: Vec<i64> = Vec::with_capacity(num_windows * n);
    for ih in 0..n_h {
        for iw in 0..n_w {
            for jh in 0..ws {
                for jw in 0..ws {
                    let h = ih * ws + jh;
                    let w = iw * ws + jw;
                    windowed.push(region[h * width + w]);
                }
            }
        }
    }

    // 3. Pairwise diff → -100/0.
    let neg_one_hundred = T::from(-100.0).unwrap();
    let zero = <T as num_traits::Zero>::zero();
    let mut mask_data: Vec<T> = Vec::with_capacity(num_windows * n * n);
    for win in 0..num_windows {
        for i in 0..n {
            let ri = windowed[win * n + i];
            for j in 0..n {
                let rj = windowed[win * n + j];
                mask_data.push(if ri == rj { zero } else { neg_one_hundred });
            }
        }
    }

    let mask_cpu = Tensor::from_storage(
        TensorStorage::cpu(mask_data),
        vec![num_windows, n, n],
        false,
    )?;
    if device == ferrotorch_core::Device::Cpu {
        Ok(mask_cpu)
    } else {
        mask_cpu.to(device)
    }
}

// ===========================================================================
// Mlp — features.<i>.<j>.mlp (matches torchvision MLP class)
// ===========================================================================
//
// torchvision's `MLP(dim, [4*dim, dim], activation=GELU)` produces a
// Sequential with children at indices 0..4: Linear(dim, 4*dim), GELU,
// Dropout, Linear(4*dim, dim), Dropout. Only indices 0 and 3 carry
// parameters. We reproduce the index numbering so the state_dict keys
// `mlp.0.{weight,bias}` and `mlp.3.{weight,bias}` match exactly.

struct Mlp<T: Float> {
    fc1: Linear<T>, // mlp.0
    fc2: Linear<T>, // mlp.3
    gelu: GELU,
    training: bool,
}

impl<T: Float> Mlp<T> {
    fn new(dim: usize, mlp_dim: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            fc1: Linear::new(dim, mlp_dim, true)?,
            fc2: Linear::new(mlp_dim, dim, true)?,
            gelu: GELU::new(),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Mlp<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.fc1.forward(input)?;
        let x = self.gelu.forward(&x)?;
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.fc1.parameters_mut();
        p.extend(self.fc2.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.fc1.named_parameters() {
            out.push((format!("0.{n}"), p));
        }
        for (n, p) in self.fc2.named_parameters() {
            out.push((format!("3.{n}"), p));
        }
        out
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.fc1, &self.fc2]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("0".to_string(), &self.fc1 as &dyn Module<T>),
            ("3".to_string(), &self.fc2 as &dyn Module<T>),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.fc1.train();
        self.fc2.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.fc1.eval();
        self.fc2.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// SwinBlock — features.<i>.<j>
// ===========================================================================
//
// One Swin transformer block:
//
// ```text
// x -> norm1 -> attn -> +x -> norm2 -> mlp -> + -> out
// ```
//
// Forward arithmetic mirrors torchvision (we omit StochasticDepth — it
// is identity in eval-mode, which is what the value-parity test runs).

pub struct SwinBlock<T: Float> {
    norm1: LayerNorm<T>,
    attn: ShiftedWindowAttention<T>,
    norm2: LayerNorm<T>,
    mlp: Mlp<T>,
    training: bool,
}

impl<T: Float> SwinBlock<T> {
    /// Create a new Swin block.
    ///
    /// `shift_size = 0` means a non-shifted window; `> 0` means a shifted
    /// window. Per torchvision, even-indexed blocks within a stage use
    /// `shift_size = 0` and odd-indexed blocks use `floor(window_size/2)`.
    pub fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: usize,
    ) -> FerrotorchResult<Self> {
        let norm1 = LayerNorm::new(vec![dim], 1e-5, true)?;
        let attn = ShiftedWindowAttention::new(dim, window_size, shift_size, num_heads)?;
        let norm2 = LayerNorm::new(vec![dim], 1e-5, true)?;
        let mlp = Mlp::new(dim, dim * mlp_ratio)?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for SwinBlock<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // input layout: [B, H, W, C]
        let normed1 = self.norm1.forward(input)?;
        let attn_out = self.attn.forward(&normed1)?;
        let x = add(input, &attn_out)?;

        let normed2 = self.norm2.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        add(&x, &mlp_out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.norm1.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.mlp.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.norm1.parameters_mut();
        p.extend(self.attn.parameters_mut());
        p.extend(self.norm2.parameters_mut());
        p.extend(self.mlp.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.norm1.named_parameters() {
            out.push((format!("norm1.{n}"), p));
        }
        for (n, p) in self.attn.named_parameters() {
            out.push((format!("attn.{n}"), p));
        }
        for (n, p) in self.norm2.named_parameters() {
            out.push((format!("norm2.{n}"), p));
        }
        for (n, p) in self.mlp.named_parameters() {
            out.push((format!("mlp.{n}"), p));
        }
        out
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.norm1, &self.attn, &self.norm2, &self.mlp]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("norm1".to_string(), &self.norm1 as &dyn Module<T>),
            ("attn".to_string(), &self.attn as &dyn Module<T>),
            ("norm2".to_string(), &self.norm2 as &dyn Module<T>),
            ("mlp".to_string(), &self.mlp as &dyn Module<T>),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.norm1.train();
        self.attn.train();
        self.norm2.train();
        self.mlp.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.norm1.eval();
        self.attn.eval();
        self.norm2.eval();
        self.mlp.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// PatchMerging — features.{2,4,6}
// ===========================================================================
//
// Downsamples [..., H, W, C] → [..., H/2, W/2, 2*C] by:
//   1. Concatenate four spatial-stride-2 slices along channel dim
//      (4 corners of each 2×2 patch) → 4*C channels.
//   2. LayerNorm(4*C)
//   3. Linear(4*C, 2*C, bias=false)
//
// We need a value-correct concat-of-strided-views. Since ferrotorch
// does not yet have a strided-slice + cat primitive that round-trips
// autograd, we materialise the gather CPU-side (failure mode #15
// scoping: this is a small reshape-style op on the activation, NOT
// in the attention hot path; activations here are at most 56*56*96 ≈
// 1.2 MB f32 for the first stage). The output is a fresh leaf used as
// the input to the eval-mode forward. For training mode this needs to
// move to a differentiable strided-slice; tracked alongside the same
// follow-up as `roll` backward.
struct PatchMerging<T: Float> {
    norm: LayerNorm<T>,
    reduction: Linear<T>,
    training: bool,
}

impl<T: Float> PatchMerging<T> {
    fn new(dim: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            norm: LayerNorm::new(vec![4 * dim], 1e-5, true)?,
            reduction: Linear::new(4 * dim, 2 * dim, false)?,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for PatchMerging<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // input: [B, H, W, C]. Pad odd H/W (torchvision _patch_merging_pad).
        let shape = input.shape();
        let raw_h = shape[1];
        let raw_w = shape[2];
        let pad_b = raw_h % 2;
        let pad_r = raw_w % 2;
        let padded = if pad_b == 0 && pad_r == 0 {
            input.clone()
        } else {
            pad_bhwc_zero(input, pad_b, pad_r)?
        };
        let pad_shape = padded.shape();
        let batch = pad_shape[0];
        let height = pad_shape[1];
        let width = pad_shape[2];
        let channels = pad_shape[3];
        let h2 = height / 2;
        let w2 = width / 2;

        // Pass 2B (#1000): replace the per-element CPU gather with a
        // view → permute → view trick that stays on `input.device()`
        // end-to-end and preserves grad_fn.
        //
        // torchvision's PatchMerging is:
        //   x0 = x[..., 0::2, 0::2, :]   (dh=0, dw=0)
        //   x1 = x[..., 1::2, 0::2, :]   (dh=1, dw=0)
        //   x2 = x[..., 0::2, 1::2, :]   (dh=0, dw=1)
        //   x3 = x[..., 1::2, 1::2, :]   (dh=1, dw=1)
        //   merged = cat([x0, x1, x2, x3], dim=-1)
        //
        // Equivalently: reshape `[B, H, W, C]` into `[B, h2, 2, w2, 2, C]`
        // exposing the inner `(dh, dw)` pair, permute to
        // `[B, h2, w2, dw, dh, C]`, then flatten the last three dims into
        // `4*C`. Because torchvision's block order is `[(0,0),(1,0),(0,1),(1,1)]`,
        // the block index `k = dw*2 + dh` — i.e. `dw` is the outermost
        // sub-axis, hence the permutation `[0, 1, 3, 4, 2, 5]`.
        let merged = padded
            .view(&[
                batch as i64,
                h2 as i64,
                2,
                w2 as i64,
                2,
                channels as i64,
            ])?
            .permute(&[0, 1, 3, 4, 2, 5])?
            .contiguous()?
            .view(&[batch as i64, h2 as i64, w2 as i64, (4 * channels) as i64])?;

        let merged = self.norm.forward(&merged)?;
        self.reduction.forward(&merged)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.norm.parameters();
        p.extend(self.reduction.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.norm.parameters_mut();
        p.extend(self.reduction.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.norm.named_parameters() {
            out.push((format!("norm.{n}"), p));
        }
        for (n, p) in self.reduction.named_parameters() {
            out.push((format!("reduction.{n}"), p));
        }
        out
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.norm, &self.reduction]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("norm".to_string(), &self.norm as &dyn Module<T>),
            ("reduction".to_string(), &self.reduction as &dyn Module<T>),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.norm.train();
        self.reduction.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.norm.eval();
        self.reduction.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// FeatureChild — heterogeneous slot in the `features` Sequential
// ===========================================================================

enum FeatureChild<T: Float> {
    /// PatchEmbed at index 0.
    PatchEmbed(PatchEmbed<T>),
    /// A stage of SwinBlocks (a torchvision `nn.Sequential[SwinBlock]`).
    Stage(Vec<SwinBlock<T>>),
    /// A PatchMerging downsample layer.
    PatchMerging(PatchMerging<T>),
}

impl<T: Float> FeatureChild<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        match self {
            Self::PatchEmbed(m) => m.forward(input),
            Self::Stage(blocks) => {
                let mut x = blocks[0].forward(input)?;
                for b in &blocks[1..] {
                    x = b.forward(&x)?;
                }
                Ok(x)
            }
            Self::PatchMerging(m) => m.forward(input),
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        match self {
            Self::PatchEmbed(m) => m.parameters(),
            Self::Stage(blocks) => blocks.iter().flat_map(|b| b.parameters()).collect(),
            Self::PatchMerging(m) => m.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        match self {
            Self::PatchEmbed(m) => m.parameters_mut(),
            Self::Stage(blocks) => blocks.iter_mut().flat_map(|b| b.parameters_mut()).collect(),
            Self::PatchMerging(m) => m.parameters_mut(),
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        match self {
            Self::PatchEmbed(m) => m.named_parameters(),
            Self::Stage(blocks) => {
                let mut out = Vec::new();
                for (j, b) in blocks.iter().enumerate() {
                    for (n, p) in b.named_parameters() {
                        out.push((format!("{j}.{n}"), p));
                    }
                }
                out
            }
            Self::PatchMerging(m) => m.named_parameters(),
        }
    }

    fn train(&mut self) {
        match self {
            Self::PatchEmbed(m) => m.train(),
            Self::Stage(blocks) => {
                for b in blocks.iter_mut() {
                    b.train();
                }
            }
            Self::PatchMerging(m) => m.train(),
        }
    }

    fn eval(&mut self) {
        match self {
            Self::PatchEmbed(m) => m.eval(),
            Self::Stage(blocks) => {
                for b in blocks.iter_mut() {
                    b.eval();
                }
            }
            Self::PatchMerging(m) => m.eval(),
        }
    }
}

// ===========================================================================
// SwinTransformer — top-level model
// ===========================================================================

pub struct SwinTransformer<T: Float> {
    features: Vec<FeatureChild<T>>,
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
    window_size: usize,
    mlp_ratio: usize,
    num_classes: usize,
}

impl<T: Float> SwinTransformer<T> {
    fn from_config(cfg: SwinConfig) -> FerrotorchResult<Self> {
        let mut features: Vec<FeatureChild<T>> = Vec::new();

        // features.0: PatchEmbed
        features.push(FeatureChild::PatchEmbed(PatchEmbed::new(
            cfg.in_channels,
            cfg.embed_dim,
            cfg.patch_size,
        )?));

        let num_stages = cfg.depths.len();
        let mut dim = cfg.embed_dim;

        for i_stage in 0..num_stages {
            let mut blocks = Vec::with_capacity(cfg.depths[i_stage]);
            for i_layer in 0..cfg.depths[i_stage] {
                let shift_size = if i_layer % 2 == 0 {
                    0
                } else {
                    cfg.window_size / 2
                };
                blocks.push(SwinBlock::new(
                    dim,
                    cfg.num_heads[i_stage],
                    cfg.window_size,
                    shift_size,
                    cfg.mlp_ratio,
                )?);
            }
            features.push(FeatureChild::Stage(blocks));

            // PatchMerging between stages (not after the last stage).
            if i_stage < num_stages - 1 {
                features.push(FeatureChild::PatchMerging(PatchMerging::new(dim)?));
                dim *= 2;
            }
        }

        let norm = LayerNorm::new(vec![dim], 1e-5, true)?;
        let head = Linear::new(dim, cfg.num_classes, true)?;

        Ok(Self {
            features,
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
        // Run all features (PatchEmbed → stages with PatchMerging interleaved).
        let mut x = self.features[0].forward(input)?;
        for child in &self.features[1..] {
            x = child.forward(&x)?;
        }

        // x layout at this point: [B, H, W, C] = [B, 7, 7, 768].
        // Apply LayerNorm over the last dim (C).
        let x = self.norm.forward(&x)?;

        // Spatial average pool: mean over H and W → [B, C].
        // Pass 2B (#1000): replace the manual sum+scale CPU reduction with
        // the autograd-aware `mean_dim` primitive. The reshape to
        // `[B, H*W, C]` lets us reduce on a single (token) axis, matching
        // torchvision's `flatten(1).mean(1)` arithmetic exactly.
        let shape = x.shape();
        let batch = shape[0];
        let height = shape[1];
        let width = shape[2];
        let channels = shape[3];
        let n_tokens = height * width;

        let x = x.view(&[batch as i64, n_tokens as i64, channels as i64])?;
        let pooled = mean_dim(&x, 1, false)?;

        // Classification head.
        self.head.forward(&pooled)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p: Vec<&Parameter<T>> = self.features.iter().flat_map(|c| c.parameters()).collect();
        p.extend(self.norm.parameters());
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p: Vec<&mut Parameter<T>> = self
            .features
            .iter_mut()
            .flat_map(|c| c.parameters_mut())
            .collect();
        p.extend(self.norm.parameters_mut());
        p.extend(self.head.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (i, child) in self.features.iter().enumerate() {
            for (n, p) in child.named_parameters() {
                out.push((format!("features.{i}.{n}"), p));
            }
        }
        for (n, p) in self.norm.named_parameters() {
            out.push((format!("norm.{n}"), p));
        }
        for (n, p) in self.head.named_parameters() {
            out.push((format!("head.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        for c in self.features.iter_mut() {
            c.train();
        }
        self.norm.train();
        self.head.train();
    }

    fn eval(&mut self) {
        self.training = false;
        for c in self.features.iter_mut() {
            c.eval();
        }
        self.norm.eval();
        self.head.eval();
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

        // PatchEmbed.
        let mut x = self.features[0].forward(input)?;
        out.insert("patch_embed".to_string(), x.clone());

        // Walk stages + PatchMerging in order.
        let mut stage_idx = 0usize;
        for child in &self.features[1..] {
            x = child.forward(&x)?;
            if matches!(child, FeatureChild::Stage(_)) {
                out.insert(format!("stage{stage_idx}"), x.clone());
                stage_idx += 1;
            }
        }

        // Norm + pool.
        let x = self.norm.forward(&x)?;
        out.insert("norm".to_string(), x.clone());

        // Pool to [B, C]. Pass 2B (#1000): same migration as `forward` —
        // reshape to `[B, H*W, C]` and reduce the token axis with
        // `mean_dim`. Stays on `input.device()` end-to-end.
        let shape = x.shape();
        let batch = shape[0];
        let height = shape[1];
        let width = shape[2];
        let channels = shape[3];
        let n_tokens = height * width;
        let pooled = mean_dim(
            &x.view(&[batch as i64, n_tokens as i64, channels as i64])?,
            1,
            false,
        )?;
        out.insert("avgpool".to_string(), pooled.clone());

        let logits = self.head.forward(&pooled)?;
        out.insert("head".to_string(), logits);

        // Suppress unused-warning when num_classes plumbing changes.
        let _ = self.final_dim;

        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        let mut names = vec!["patch_embed".to_string()];
        let stages = self
            .features
            .iter()
            .filter(|c| matches!(c, FeatureChild::Stage(_)))
            .count();
        for i in 0..stages {
            names.push(format!("stage{i}"));
        }
        names.push("norm".to_string());
        names.push("avgpool".to_string());
        names.push("head".to_string());
        names
    }
}

/// Construct a Swin Transformer Tiny model.
///
/// Architecture:
/// - Patch size: 4×4
/// - Embedding dimension: 96
/// - Depths: `[2, 2, 6, 2]`
/// - Heads: `[3, 6, 12, 24]`
/// - Window size: 7
/// - MLP ratio: 4
/// - Image size: 224×224
///
/// Total parameters: ~28M (for 1000 classes), matching torchvision's swin_t.
pub fn swin_tiny<T: Float>(num_classes: usize) -> FerrotorchResult<SwinTransformer<T>> {
    SwinTransformer::from_config(SwinConfig {
        patch_size: 4,
        in_channels: 3,
        embed_dim: 96,
        depths: vec![2, 2, 6, 2],
        num_heads: vec![3, 6, 12, 24],
        window_size: 7,
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

    fn leaf_4d(data: &[f32], shape: [usize; 4], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // compute_relative_position_index
    // -----------------------------------------------------------------------

    #[test]
    fn test_relative_position_index_ws3() {
        // For ws=3, N=9, indices live in 0..(2*3-1)^2 = 0..25.
        let idx = compute_relative_position_index(3);
        assert_eq!(idx.len(), 9 * 9);
        for &v in &idx {
            assert!((0..25).contains(&v), "index {v} out of [0, 25)");
        }
        // The diagonal (i==j) must always map to (ws-1, ws-1) = (2, 2)
        // → 2 * (2*3-1) + 2 = 12.
        for i in 0..9 {
            assert_eq!(idx[i * 9 + i], 12);
        }
    }

    #[test]
    fn test_relative_position_index_ws7_in_range() {
        let idx = compute_relative_position_index(7);
        assert_eq!(idx.len(), 49 * 49);
        let max_idx = (2 * 7 - 1) * (2 * 7 - 1);
        for &v in &idx {
            assert!((0..max_idx as i64).contains(&v));
        }
    }

    // -----------------------------------------------------------------------
    // ShiftedWindowAttention forward shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_shifted_window_attention_no_shift_shape() {
        let attn = ShiftedWindowAttention::<f32>::new(96, 7, 0, 3).unwrap();
        // Input [B=1, H=14, W=14, C=96] (two 7×7 windows along each spatial dim).
        let input = leaf_4d(&vec![0.01f32; 14 * 14 * 96], [1, 14, 14, 96], false);
        let output = no_grad(|| attn.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 14, 14, 96]);
    }

    #[test]
    fn test_shifted_window_attention_with_shift_shape() {
        let attn = ShiftedWindowAttention::<f32>::new(96, 7, 3, 3).unwrap();
        let input = leaf_4d(&vec![0.01f32; 14 * 14 * 96], [1, 14, 14, 96], false);
        let output = no_grad(|| attn.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 14, 14, 96]);
    }

    #[test]
    fn test_shifted_window_attention_relative_position_index_not_in_named_params() {
        let attn = ShiftedWindowAttention::<f32>::new(96, 7, 3, 3).unwrap();
        let names: Vec<String> = attn
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        // qkv.weight, qkv.bias, proj.weight, proj.bias, relative_position_bias_table
        assert_eq!(names.len(), 5);
        assert!(names.contains(&"relative_position_bias_table".to_string()));
        assert!(
            !names.iter().any(|n| n.contains("relative_position_index")),
            "relative_position_index must NOT be a Parameter (failure mode #36)"
        );
    }

    // -----------------------------------------------------------------------
    // SwinBlock
    // -----------------------------------------------------------------------

    #[test]
    fn test_swin_block_output_shape_no_shift() {
        let block = SwinBlock::<f32>::new(96, 3, 7, 0, 4).unwrap();
        let input = leaf_4d(&vec![0.01f32; 14 * 14 * 96], [1, 14, 14, 96], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 14, 14, 96]);
    }

    #[test]
    fn test_swin_block_output_shape_with_shift() {
        let block = SwinBlock::<f32>::new(96, 3, 7, 3, 4).unwrap();
        let input = leaf_4d(&vec![0.01f32; 14 * 14 * 96], [1, 14, 14, 96], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 14, 14, 96]);
    }

    // -----------------------------------------------------------------------
    // PatchMerging
    // -----------------------------------------------------------------------

    #[test]
    fn test_patch_merging_shape() {
        let pm = PatchMerging::<f32>::new(96).unwrap();
        let input = leaf_4d(&vec![0.01f32; 14 * 14 * 96], [1, 14, 14, 96], false);
        let output = no_grad(|| pm.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 7, 7, 192]);
    }

    // -----------------------------------------------------------------------
    // SwinTransformer
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
        // torchvision swin_t reports ~28.3M parameters.
        assert!(
            total > 27_000_000,
            "Swin-T should have >27M params, got {total}"
        );
        assert!(
            total < 30_000_000,
            "Swin-T should have <30M params, got {total}"
        );
    }

    #[test]
    fn test_swin_tiny_named_parameters_match_torchvision_keys() {
        // Precondition: every torchvision swin_t state-dict key must be
        // produced by ferrotorch's named_parameters(). The hard-coded
        // expected list mirrors fixtures_value_parity.json's
        // `param_keys` for swin_t_value_parity (cross-checked against
        // `tvm.swin_t(weights=None).named_parameters()` directly).
        let model = swin_tiny::<f32>(1000).unwrap();
        let actual: std::collections::BTreeSet<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();

        // Check a representative set of structural keys.
        let must_have = [
            "features.0.0.weight",
            "features.0.0.bias",
            "features.0.2.weight",
            "features.0.2.bias",
            "features.1.0.norm1.weight",
            "features.1.0.attn.qkv.weight",
            "features.1.0.attn.qkv.bias",
            "features.1.0.attn.proj.weight",
            "features.1.0.attn.relative_position_bias_table",
            "features.1.0.mlp.0.weight",
            "features.1.0.mlp.0.bias",
            "features.1.0.mlp.3.weight",
            "features.1.0.mlp.3.bias",
            "features.2.norm.weight",
            "features.2.norm.bias",
            "features.2.reduction.weight",
            "features.5.5.attn.qkv.weight",
            "features.7.1.attn.relative_position_bias_table",
            "norm.weight",
            "norm.bias",
            "head.weight",
            "head.bias",
        ];
        for key in must_have {
            assert!(
                actual.contains(key),
                "missing key {key:?} from ferrotorch named_parameters; got {} keys",
                actual.len()
            );
        }

        // Negative assertion: relative_position_index MUST NOT appear.
        assert!(
            !actual.iter().any(|k| k.contains("relative_position_index")),
            "relative_position_index must not be in named_parameters (failure mode #36)"
        );

        // Negative assertion: features.2.reduction.bias MUST NOT appear
        // (PatchMerging.reduction is bias=false in torchvision).
        assert!(!actual.contains("features.2.reduction.bias"));
        assert!(!actual.contains("features.4.reduction.bias"));
        assert!(!actual.contains("features.6.reduction.bias"));
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
}
