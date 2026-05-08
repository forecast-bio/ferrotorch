//! Probe-before-fix for Phase 3 (#996) — verifies that
//! `Tensor::permute().contiguous()` reproduces the *exact* element ordering
//! produced by the manual `data_vec()` + indexed-loop pattern that Phase 3
//! migrates out of `vit.rs`, `convnext.rs`, and `swin.rs`.
//!
//! This file MUST run and pass BEFORE any model code is replaced. It is the
//! safety net for failure mode #18 (mass workaround propagation): if the
//! primitive does not match the manual loop on the patterns we're about to
//! migrate, the migration would silently rotate semantics — exactly what the
//! Phase 3 pre-flight forbids.
//!
//! Patterns probed (tracked under #996):
//!
//! - `vit_patch_embed_BCp_to_BpC`     — `[B, C, P] -> [B, P, C]`
//! - `convnext_nhwc_from_nchw`        — `[B, C, H, W] -> [B, H, W, C]`
//! - `convnext_nchw_from_nhwc`        — `[B, H, W, C] -> [B, C, H, W]`
//! - `swin_BCHW_to_BHWC_flat`         — `[B, C, H, W] -> [B, H*W, C]`
//! - `swin_BHWC_to_BCHW_flat`         — `[B, H*W, C] (logical [B,H,W,C]) -> [B, C, H, W]`
//!
//! Each probe reproduces the original manual indexed loop (copied verbatim
//! from the production model, not re-derived) and the migrated primitive
//! chain, then asserts byte-for-byte equality on a deterministic
//! arange-style input tensor.

use ferrotorch_core::{Tensor, TensorStorage};

/// Deterministic 1.0-based arange-style payload for a flat tensor of length
/// `n`. We avoid randomness so probe failures are reproducible without seeds
/// or fixture files.
fn arange_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) + 1.0).collect()
}

fn make_cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
        .expect("probe tensor construction must succeed")
}

/// Pattern 1 — ViT PatchEmbed.
///
/// Manual loop reproduces `vit.rs:91-104` verbatim (B=2, C=4, P=6):
///
/// ```text
/// src: [b, e, p] in [B, embed_dim, num_patches]
/// dst: [b, p, e] in [B, num_patches, embed_dim]
/// ```
///
/// Equivalent primitive chain: `permute(&[0, 2, 1]).contiguous()`.
#[test]
fn probe_vit_patch_embed_bcp_to_bpc() {
    let (b, c, p) = (2usize, 4usize, 6usize);
    let data = arange_f32(b * c * p);
    let input = make_cpu_tensor(data.clone(), vec![b, c, p]);

    // Manual indexed loop — copied from vit.rs:94-103.
    let x_data = input.data_vec().expect("data_vec");
    let mut manual_out = vec![0.0_f32; b * p * c];
    for bi in 0..b {
        for e in 0..c {
            for pi in 0..p {
                let src = bi * c * p + e * p + pi;
                let dst = bi * p * c + pi * c + e;
                manual_out[dst] = x_data[src];
            }
        }
    }

    // Primitive chain.
    let primitive = input
        .permute(&[0, 2, 1])
        .expect("permute(0,2,1)")
        .contiguous()
        .expect("contiguous after permute");
    assert_eq!(primitive.shape(), &[b, p, c]);
    let primitive_out = primitive.data_vec().expect("primitive data_vec");

    assert_eq!(
        manual_out.len(),
        primitive_out.len(),
        "probe vit_patch_embed: length mismatch"
    );
    for (i, (m, p_)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        assert_eq!(
            m, p_,
            "probe vit_patch_embed: element {i} differs (manual={m} primitive={p_})"
        );
    }
}

/// Pattern 2 — ConvNeXt `nhwc_from_nchw`.
///
/// Manual loop reproduces `convnext.rs:46-55` verbatim:
///
/// ```text
/// src = bi*C*H*W + ci*H*W + hi*W + wi   ([B, C, H, W])
/// dst = bi*H*W*C + hi*W*C + wi*C + ci   ([B, H, W, C])
/// ```
///
/// Equivalent primitive chain: `permute(&[0, 2, 3, 1]).contiguous()`.
#[test]
fn probe_convnext_nhwc_from_nchw() {
    let (b, c, h, w) = (2usize, 3usize, 4usize, 5usize);
    let data = arange_f32(b * c * h * w);
    let input = make_cpu_tensor(data.clone(), vec![b, c, h, w]);

    let in_data = input.data_vec().expect("data_vec");
    let total = b * c * h * w;
    let mut manual_out = vec![0.0_f32; total];
    for bi in 0..b {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = bi * c * h * w + ci * h * w + hi * w + wi;
                    let dst = bi * h * w * c + hi * w * c + wi * c + ci;
                    manual_out[dst] = in_data[src];
                }
            }
        }
    }

    let primitive = input
        .permute(&[0, 2, 3, 1])
        .expect("permute(0,2,3,1)")
        .contiguous()
        .expect("contiguous");
    assert_eq!(primitive.shape(), &[b, h, w, c]);
    let primitive_out = primitive.data_vec().expect("primitive data_vec");

    for (i, (m, p_)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        assert_eq!(
            m, p_,
            "probe convnext_nhwc_from_nchw: element {i} differs (manual={m} primitive={p_})"
        );
    }
}

/// Pattern 3 — ConvNeXt `nchw_from_nhwc`.
///
/// Manual loop reproduces `convnext.rs:75-84` verbatim:
///
/// ```text
/// src = bi*H*W*C + hi*W*C + wi*C + ci   ([B, H, W, C])
/// dst = bi*C*H*W + ci*H*W + hi*W + wi   ([B, C, H, W])
/// ```
///
/// Equivalent primitive chain: `permute(&[0, 3, 1, 2]).contiguous()`.
#[test]
fn probe_convnext_nchw_from_nhwc() {
    let (b, h, w, c) = (2usize, 4usize, 5usize, 3usize);
    let data = arange_f32(b * h * w * c);
    let input = make_cpu_tensor(data.clone(), vec![b, h, w, c]);

    let in_data = input.data_vec().expect("data_vec");
    let total = b * c * h * w;
    let mut manual_out = vec![0.0_f32; total];
    for bi in 0..b {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let src = bi * h * w * c + hi * w * c + wi * c + ci;
                    let dst = bi * c * h * w + ci * h * w + hi * w + wi;
                    manual_out[dst] = in_data[src];
                }
            }
        }
    }

    let primitive = input
        .permute(&[0, 3, 1, 2])
        .expect("permute(0,3,1,2)")
        .contiguous()
        .expect("contiguous");
    assert_eq!(primitive.shape(), &[b, c, h, w]);
    let primitive_out = primitive.data_vec().expect("primitive data_vec");

    for (i, (m, p_)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        assert_eq!(
            m, p_,
            "probe convnext_nchw_from_nhwc: element {i} differs (manual={m} primitive={p_})"
        );
    }
}

/// Pattern 4 — Swin `[B, C, H, W] -> [B, H*W, C]`.
///
/// Manual loop reproduces `swin.rs:432-440` (and the identical block at
/// `swin.rs:593-601`):
///
/// ```text
/// src = b*C*N + c*N + t              ([B, C, N=H*W])
/// dst = b*N*C + t*C + c              ([B, N, C])
/// ```
///
/// Equivalent primitive chain: `permute(&[0, 2, 3, 1]).contiguous().view([B, H*W, C])`.
#[test]
fn probe_swin_bchw_to_bhwc_flat() {
    let (b, c, h, w) = (2usize, 3usize, 4usize, 5usize);
    let n = h * w;
    let data = arange_f32(b * c * n);
    let input = make_cpu_tensor(data.clone(), vec![b, c, h, w]);

    let in_data = input.data_vec().expect("data_vec");
    let mut manual_out = vec![0.0_f32; b * n * c];
    for bi in 0..b {
        for ci in 0..c {
            for ti in 0..n {
                let src = bi * c * n + ci * n + ti;
                let dst = bi * n * c + ti * c + ci;
                manual_out[dst] = in_data[src];
            }
        }
    }

    let primitive = input
        .permute(&[0, 2, 3, 1])
        .expect("permute(0,2,3,1)")
        .contiguous()
        .expect("contiguous")
        .view(&[b as i64, n as i64, c as i64])
        .expect("view to [B, H*W, C]");
    assert_eq!(primitive.shape(), &[b, n, c]);
    let primitive_out = primitive.data_vec().expect("primitive data_vec");

    for (i, (m, p_)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        assert_eq!(
            m, p_,
            "probe swin_bchw_to_bhwc_flat: element {i} differs (manual={m} primitive={p_})"
        );
    }
}

/// Pattern 5 — Swin `[B, H*W, C] -> [B, C, H, W]`.
///
/// Manual loop reproduces `swin.rs:275-285`:
///
/// ```text
/// src = b*N*C + t*C + c              ([B, N=H*W, C])
/// dst = b*C*N + c*N + t              ([B, C, N], reshaped to [B, C, H, W])
/// ```
///
/// And the identical sibling at `swin.rs:306-313` for the post-downsample
/// `[B, C, H', W'] -> [B, H'*W', C]` direction (covered by pattern 4).
///
/// Equivalent primitive chain: `view([B, H, W, C]).permute(0,3,1,2).contiguous()`.
#[test]
fn probe_swin_bhwc_to_bchw_flat() {
    let (b, h, w, c) = (2usize, 4usize, 5usize, 3usize);
    let n = h * w;
    let data = arange_f32(b * n * c);
    let input = make_cpu_tensor(data.clone(), vec![b, n, c]);

    let in_data = input.data_vec().expect("data_vec");
    let mut manual_out = vec![0.0_f32; b * c * n];
    for bi in 0..b {
        for ti in 0..n {
            for ci in 0..c {
                let src = bi * n * c + ti * c + ci;
                let dst = bi * c * n + ci * n + ti;
                manual_out[dst] = in_data[src];
            }
        }
    }

    let primitive = input
        .view(&[b as i64, h as i64, w as i64, c as i64])
        .expect("view to [B, H, W, C]")
        .permute(&[0, 3, 1, 2])
        .expect("permute(0,3,1,2)")
        .contiguous()
        .expect("contiguous");
    assert_eq!(primitive.shape(), &[b, c, h, w]);
    let primitive_out = primitive.data_vec().expect("primitive data_vec");

    for (i, (m, p_)) in manual_out.iter().zip(primitive_out.iter()).enumerate() {
        assert_eq!(
            m, p_,
            "probe swin_bhwc_to_bchw_flat: element {i} differs (manual={m} primitive={p_})"
        );
    }
}
