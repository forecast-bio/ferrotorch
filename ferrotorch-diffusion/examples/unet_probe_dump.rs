//! Per-stage probe for the SD-1.5 UNet (#1151 redirect).
//!
//! Companion to `scripts/probe_unet_stages_1151.py`. Loads the pinned
//! `ferrotorch/sd-v1-5-unet` mirror, runs the ferrotorch UNet forward
//! pass on the frozen parity probe, and dumps each interesting
//! intermediate tensor in the same `[u32 ndim][u32 × ndim shape][f32]`
//! format used by the rest of the harness. The compare step is a
//! separate small script.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_diffusion::{load_unet, UNet2DConditionConfig};
use ferrotorch_hub::{hf_download_model, HubCache};
use ferrotorch_nn::module::Module;

fn read_dump_f32(path: &Path) -> FerrotorchResult<(Vec<usize>, Vec<f32>)> {
    let mut f = File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("open {}: {e}", path.display()),
    })?;
    let mut header4 = [0u8; 4];
    f.read_exact(&mut header4).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("read header from {}: {e}", path.display()),
    })?;
    let ndim = u32::from_le_bytes(header4) as usize;
    let mut shape = vec![0usize; ndim];
    for entry in &mut shape {
        f.read_exact(&mut header4).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("read shape from {}: {e}", path.display()),
        })?;
        *entry = u32::from_le_bytes(header4) as usize;
    }
    let count: usize = shape.iter().product();
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("read data from {}: {e}", path.display()),
    })?;
    let mut data = Vec::with_capacity(count);
    for chunk in buf.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, data))
}

fn write_dump(path: &Path, t: &Tensor<f32>) -> FerrotorchResult<()> {
    let shape = t.shape();
    let data = t.data()?;
    let mut f = File::create(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("create {}: {e}", path.display()),
    })?;
    f.write_all(&(shape.len() as u32).to_le_bytes())
        .map_err(|e| FerrotorchError::InvalidArgument { message: e.to_string() })?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())
            .map_err(|e| FerrotorchError::InvalidArgument { message: e.to_string() })?;
    }
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &v in data.iter() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
        .map_err(|e| FerrotorchError::InvalidArgument { message: e.to_string() })
}

fn norm(d: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &v in d {
        s += (v as f64) * (v as f64);
    }
    s.sqrt()
}
fn max_abs(d: &[f32]) -> f64 {
    d.iter().fold(0.0f32, |a, &v| a.max(v.abs())) as f64
}

fn report(name: &str, t: &Tensor<f32>) -> FerrotorchResult<()> {
    let d = t.data()?;
    eprintln!(
        "  {name}: shape={:?}  norm={:.4}  max_abs={:.4}",
        t.shape(),
        norm(d),
        max_abs(d)
    );
    Ok(())
}

fn main() -> FerrotorchResult<()> {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/ferrotorch_probe_1151_rust"));
    std::fs::create_dir_all(&out_dir).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("create {}: {e}", out_dir.display()),
    })?;

    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model("ferrotorch/sd-v1-5-unet", "main", &cache)?;

    // Load config + weights + parity probe.
    let cfg = UNet2DConditionConfig::from_file(&repo_dir.join("config.json"))?;
    let (lat_shape, lat_data) =
        read_dump_f32(&repo_dir.join("_value_parity_noisy_latent.bin"))?;
    let (ts_shape, ts_data) = read_dump_f32(&repo_dir.join("_value_parity_timestep.bin"))?;
    let (text_shape, text_data) =
        read_dump_f32(&repo_dir.join("_value_parity_text_embedding.bin"))?;

    let latent = Tensor::from_storage(TensorStorage::cpu(lat_data), lat_shape, false)?;
    let timestep = Tensor::from_storage(TensorStorage::cpu(ts_data), ts_shape, false)?;
    let text = Tensor::from_storage(TensorStorage::cpu(text_data), text_shape, false)?;

    let (unet, _drop) =
        load_unet::<f32>(&repo_dir.join("model.safetensors"), cfg, /* strict = */ false)?;

    // -- Stage 1: time embedding. ---------------------------------------
    let t_proj = unet.time_proj.forward_t(&timestep)?;
    let temb = unet.time_embedding.forward(&t_proj)?;
    report("01_time_emb", &temb)?;
    write_dump(&out_dir.join("01_time_emb.bin"), &temb)?;

    // -- Stage 2: conv_in. ----------------------------------------------
    let mut h = unet.conv_in.forward(&latent)?;
    report("02_conv_in", &h)?;
    write_dump(&out_dir.join("02_conv_in.bin"), &h)?;

    // -- Stages 3-6: down blocks. ---------------------------------------
    let mut skips: Vec<Tensor<f32>> = Vec::new();
    skips.push(h.clone());
    for (i, db) in unet.down_blocks.iter().enumerate() {
        let (out, mut block_skips) = match db {
            ferrotorch_diffusion::AnyDownBlock::CrossAttn(b) => {
                // Fine-grained dump for the first down-block to localize
                // the transformer-side bug surfaced by #1151.
                if i == 0 {
                    let mut h2 = h.clone();
                    let mut local_skips = Vec::new();
                    for (j, (r, a)) in b.resnets.iter().zip(b.attentions.iter()).enumerate() {
                        h2 = r.forward_t(&h2, &temb)?;
                        let nm = format!("03a_db0_resnet{j}_out");
                        report(&nm, &h2)?;
                        write_dump(&out_dir.join(format!("{nm}.bin")), &h2)?;
                        // Deep-dive on the first transformer to localize
                        // the bug surfaced by the harness — inline-reproduce
                        // Transformer2DModel::forward_xattn sub-steps.
                        if j == 0 {
                            use ferrotorch_core::grad_fns::arithmetic::add as tadd;
                            let bdim = h2.shape()[0];
                            let cdim = h2.shape()[1];
                            let hh = h2.shape()[2];
                            let ww = h2.shape()[3];
                            let hw = hh * ww;
                            let residual = h2.clone();
                            // norm + proj_in
                            let mut s = a.norm.forward(&h2)?;
                            report("03b_t2d_norm_out", &s)?;
                            write_dump(&out_dir.join("03b_t2d_norm_out.bin"), &s)?;
                            s = a.proj_in.forward(&s)?;
                            report("03b_t2d_proj_in_out", &s)?;
                            write_dump(&out_dir.join("03b_t2d_proj_in_out.bin"), &s)?;
                            // [B, inner, HW] -> [B, HW, inner]
                            let inner = s.shape()[1];
                            let mut seq = s
                                .reshape_t(&[bdim as isize, inner as isize, hw as isize])?
                                .transpose(1, 2)?
                                .contiguous()?;
                            report("03b_t2d_seq_in", &seq)?;
                            write_dump(&out_dir.join("03b_t2d_seq_in.bin"), &seq)?;
                            // Inline BasicTransformerBlock[0]
                            let blk = &a.transformer_blocks[0];
                            let n1 = blk.norm1.forward(&seq)?;
                            report("03b_t2d_b0_norm1_out", &n1)?;
                            write_dump(&out_dir.join("03b_t2d_b0_norm1_out.bin"), &n1)?;
                            // Inline-reproduce Attention::forward_xattn
                            // (self-attn variant) sub-steps to localize the
                            // bug surfaced by `03b_t2d_b0_attn1_out`.
                            {
                                let attn = &blk.attn1;
                                let bb = n1.shape()[0];
                                let nn = n1.shape()[1];
                                let qraw = attn.to_q.forward(&n1)?;
                                report("03c_attn1_q_proj", &qraw)?;
                                write_dump(&out_dir.join("03c_attn1_q_proj.bin"), &qraw)?;
                                let kraw = attn.to_k.forward(&n1)?;
                                report("03c_attn1_k_proj", &kraw)?;
                                write_dump(&out_dir.join("03c_attn1_k_proj.bin"), &kraw)?;
                                let vraw = attn.to_v.forward(&n1)?;
                                report("03c_attn1_v_proj", &vraw)?;
                                write_dump(&out_dir.join("03c_attn1_v_proj.bin"), &vraw)?;
                                let hh = attn.heads;
                                let dd = attn.dim_head;
                                let q = qraw
                                    .reshape_t(&[bb as isize, nn as isize, hh as isize, dd as isize])?
                                    .transpose(1, 2)?
                                    .contiguous()?
                                    .reshape_t(&[(bb * hh) as isize, nn as isize, dd as isize])?;
                                let k = kraw
                                    .reshape_t(&[bb as isize, nn as isize, hh as isize, dd as isize])?
                                    .transpose(1, 2)?
                                    .contiguous()?
                                    .reshape_t(&[(bb * hh) as isize, nn as isize, dd as isize])?;
                                let v = vraw
                                    .reshape_t(&[bb as isize, nn as isize, hh as isize, dd as isize])?
                                    .transpose(1, 2)?
                                    .contiguous()?
                                    .reshape_t(&[(bb * hh) as isize, nn as isize, dd as isize])?;
                                let k_t = k.transpose(1, 2)?.contiguous()?;
                                let scores = q.bmm(&k_t)?;
                                let scale_v = (dd as f64).sqrt().recip();
                                let scale_t = ferrotorch_core::scalar::<f32>(scale_v as f32)?;
                                let scores_scaled =
                                    ferrotorch_core::grad_fns::arithmetic::mul(&scores, &scale_t)?;
                                let probs = scores_scaled.softmax()?;
                                let attended = probs.bmm(&v)?;
                                let merged = attended
                                    .reshape_t(&[bb as isize, hh as isize, nn as isize, dd as isize])?
                                    .transpose(1, 2)?
                                    .contiguous()?
                                    .reshape_t(&[bb as isize, nn as isize, (hh * dd) as isize])?;
                                report("03c_attn1_merged", &merged)?;
                                write_dump(&out_dir.join("03c_attn1_merged.bin"), &merged)?;
                                let out = attn.to_out_0.forward(&merged)?;
                                report("03c_attn1_to_out", &out)?;
                                write_dump(&out_dir.join("03c_attn1_to_out.bin"), &out)?;
                            }
                            let sa = blk.attn1.forward_xattn(&n1, None)?;
                            report("03b_t2d_b0_attn1_out", &sa)?;
                            write_dump(&out_dir.join("03b_t2d_b0_attn1_out.bin"), &sa)?;
                            seq = tadd(&sa, &seq)?;
                            report("03b_t2d_b0_after_sa", &seq)?;
                            write_dump(&out_dir.join("03b_t2d_b0_after_sa.bin"), &seq)?;
                            let n2 = blk.norm2.forward(&seq)?;
                            report("03b_t2d_b0_norm2_out", &n2)?;
                            write_dump(&out_dir.join("03b_t2d_b0_norm2_out.bin"), &n2)?;
                            let ca = blk.attn2.forward_xattn(&n2, Some(&text))?;
                            report("03b_t2d_b0_attn2_out", &ca)?;
                            write_dump(&out_dir.join("03b_t2d_b0_attn2_out.bin"), &ca)?;
                            seq = tadd(&ca, &seq)?;
                            report("03b_t2d_b0_after_ca", &seq)?;
                            write_dump(&out_dir.join("03b_t2d_b0_after_ca.bin"), &seq)?;
                            let n3 = blk.norm3.forward(&seq)?;
                            report("03b_t2d_b0_norm3_out", &n3)?;
                            write_dump(&out_dir.join("03b_t2d_b0_norm3_out.bin"), &n3)?;
                            let ff = blk.ff.forward(&n3)?;
                            report("03b_t2d_b0_ff_out", &ff)?;
                            write_dump(&out_dir.join("03b_t2d_b0_ff_out.bin"), &ff)?;
                            seq = tadd(&ff, &seq)?;
                            report("03b_t2d_b0_block_out", &seq)?;
                            write_dump(&out_dir.join("03b_t2d_b0_block_out.bin"), &seq)?;
                            // Reshape back and proj_out + residual.
                            let back = seq
                                .transpose(1, 2)?
                                .reshape_t(&[bdim as isize, inner as isize, hh as isize, ww as isize])?
                                .contiguous()?;
                            let po = a.proj_out.forward(&back)?;
                            report("03b_t2d_proj_out", &po)?;
                            write_dump(&out_dir.join("03b_t2d_proj_out.bin"), &po)?;
                            let final_out = tadd(&po, &residual)?;
                            report("03b_t2d_final_out", &final_out)?;
                            write_dump(&out_dir.join("03b_t2d_final_out.bin"), &final_out)?;
                            let _ = cdim;
                        }
                        h2 = a.forward_xattn(&h2, &text)?;
                        let nm = format!("03a_db0_attn{j}_out");
                        report(&nm, &h2)?;
                        write_dump(&out_dir.join(format!("{nm}.bin")), &h2)?;
                        local_skips.push(h2.clone());
                    }
                    if let Some(d) = &b.downsamplers_0 {
                        h2 = d.forward(&h2)?;
                        let nm = "03a_db0_downsample_out";
                        report(nm, &h2)?;
                        write_dump(&out_dir.join(format!("{nm}.bin")), &h2)?;
                        local_skips.push(h2.clone());
                    }
                    (h2, local_skips)
                } else {
                    b.forward_t(&h, &temb, &text)?
                }
            }
            ferrotorch_diffusion::AnyDownBlock::Plain(b) => b.forward_t(&h, &temb)?,
        };
        h = out;
        let name = format!("{:02}_down_block_{}_out", 3 + i, i);
        report(&name, &h)?;
        write_dump(&out_dir.join(format!("{name}.bin")), &h)?;
        skips.append(&mut block_skips);
    }

    // -- Stage 7: mid block. --------------------------------------------
    h = unet.mid_block.forward_t(&h, &temb, &text)?;
    report("07_mid_out", &h)?;
    write_dump(&out_dir.join("07_mid_out.bin"), &h)?;

    // -- Stages 8-11: up blocks. ----------------------------------------
    for (i, ub) in unet.up_blocks.iter().enumerate() {
        let n = match ub {
            ferrotorch_diffusion::AnyUpBlock::CrossAttn(b) => b.resnets.len(),
            ferrotorch_diffusion::AnyUpBlock::Plain(b) => b.resnets.len(),
        };
        let split_at = skips.len() - n;
        let popped = skips.split_off(split_at);
        let popped_rev: Vec<Tensor<f32>> = popped.into_iter().rev().collect();
        h = match ub {
            ferrotorch_diffusion::AnyUpBlock::CrossAttn(b) => {
                b.forward_t(&h, &popped_rev, &temb, &text)?
            }
            ferrotorch_diffusion::AnyUpBlock::Plain(b) => b.forward_t(&h, &popped_rev, &temb)?,
        };
        let name = format!("{:02}_up_block_{}_out", 8 + i, i);
        report(&name, &h)?;
        write_dump(&out_dir.join(format!("{name}.bin")), &h)?;
    }

    // -- Stage 12: conv_norm_out (no SiLU). -----------------------------
    h = unet.conv_norm_out.forward(&h)?;
    report("12_conv_norm_out", &h)?;
    write_dump(&out_dir.join("12_conv_norm_out.bin"), &h)?;

    // -- Stage 13: final output (after SiLU + conv_out). ----------------
    h = unet.conv_act.forward(&h)?;
    h = unet.conv_out.forward(&h)?;
    report("13_predicted_noise", &h)?;
    write_dump(&out_dir.join("13_predicted_noise.bin"), &h)?;

    eprintln!("[unet_probe_dump] wrote intermediates to {}", out_dir.display());
    Ok(())
}
