//! #1141 round-4 per-stage RPN probe.
//!
//! Dumps the intermediate tensors from the FasterRcnn pipeline (FPN
//! features, RPN cls_logits, RPN bbox_deltas, anchors, decoded proposals,
//! final post-NMS proposals) for a single image so a companion Python
//! probe can diff them against torchvision stage-by-stage.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-vision --release \
//!     --example probe_rpn_stages_1141 -- \
//!     --image /tmp/ferrotorch_verify_images/coco_000000087038.jpg \
//!     --out /tmp/ferrotorch_probe_1141_rust.safetensors
//! ```
//!
//! Tensors written (safetensors keys):
//! - `input`                : [1, 3, H, W]
//! - `fpn_p{2..6}`          : [1, 256, Hi, Wi]
//! - `cls_p{2..6}`          : [1, A,   Hi, Wi]   (raw objectness logits, pre-sigmoid)
//! - `bbox_p{2..6}`         : [1, A*4, Hi, Wi]   (raw bbox deltas)
//! - `anchors_p{2..6}`      : [Hi*Wi*A, 4]       (xyxy in image pixel coords)
//! - `decoded_p{2..6}`      : [Hi*Wi*A, 4]       (anchors + bbox_deltas, BoxCoder weights=(1,1,1,1))
//! - `proposals_post_nms`   : [N_final, 4]
//! - `proposals_scores_post_nms` : [N_final]

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{InterpolateMode, interpolate};
use std::collections::HashMap;

use ferrotorch_nn::StateDict;
use ferrotorch_serialize::save_safetensors;
use ferrotorch_vision::io::read_image_as_tensor;
use ferrotorch_vision::models::FasterRcnn;
use ferrotorch_vision::models::detection::anchor_utils::decode_boxes;
use ferrotorch_vision::models::detection::fasterrcnn_resnet50_fpn;
use ferrotorch_vision::models::detection::rpn::RpnConfig;

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let args: Vec<String> = env::args().collect();
    let mut image: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                image = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--image needs a value")?,
                ));
                i += 2;
            }
            "--out" => {
                output = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--out needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok((image.ok_or("--image required")?, output.ok_or("--out required")?))
}

fn bilinear_resize_chw_to_bchw(
    chw: &Tensor<f32>,
    out_h: usize,
    out_w: usize,
) -> FerrotorchResult<Tensor<f32>> {
    let shape = chw.shape().to_vec();
    let data = chw.data_vec()?;
    let bchw = Tensor::from_storage(
        TensorStorage::cpu(data),
        vec![1, shape[0], shape[1], shape[2]],
        false,
    )?;
    interpolate(&bchw, Some([out_h, out_w]), None, InterpolateMode::Bilinear, false)
}

fn normalize_bchw(
    bchw: &Tensor<f32>,
    mean: [f32; 3],
    std: [f32; 3],
) -> FerrotorchResult<Tensor<f32>> {
    let shape = bchw.shape().to_vec();
    let h = shape[2];
    let w = shape[3];
    let mut data = bchw.data_vec()?;
    let plane = h * w;
    for ci in 0..3 {
        let base = ci * plane;
        let m = mean[ci];
        let s = std[ci];
        for i in 0..plane {
            data[base + i] = (data[base + i] - m) / s;
        }
    }
    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
}

/// Same preprocessing as `examples/inference_dump.rs` fasterrcnn branch.
fn preprocess(raw: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let shape = raw.shape().to_vec();
    let h_in = shape[1];
    let w_in = shape[2];
    let min_size = 800.0_f64;
    let max_size = 1333.0_f64;
    let h = h_in as f64;
    let w = w_in as f64;
    let s_min = min_size / h.min(w);
    let s_max = max_size / h.max(w);
    let scale = s_min.min(s_max);
    let out_h = (h * scale).round() as usize;
    let out_w = (w * scale).round() as usize;
    let resized = bilinear_resize_chw_to_bchw(&raw, out_h, out_w)?;
    let normed = normalize_bchw(&resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])?;
    let stride: usize = 32;
    let pad_h = out_h.div_ceil(stride) * stride;
    let pad_w = out_w.div_ceil(stride) * stride;
    if pad_h == out_h && pad_w == out_w {
        return Ok(normed);
    }
    let nd = normed.data_vec()?;
    let c = 3;
    let mut padded = vec![0.0_f32; c * pad_h * pad_w];
    for ci in 0..c {
        for r in 0..out_h {
            let src_base = (ci * out_h + r) * out_w;
            let dst_base = (ci * pad_h + r) * pad_w;
            padded[dst_base..dst_base + out_w]
                .copy_from_slice(&nd[src_base..src_base + out_w]);
        }
    }
    Tensor::from_storage(
        TensorStorage::cpu(padded),
        vec![1, c, pad_h, pad_w],
        false,
    )
}

/// Build a fasterrcnn and load pretrained weights with the same logic
/// the registry uses (so the probe sees the actual production model).
fn build_model() -> FerrotorchResult<FasterRcnn<f32>> {
    let mut model = fasterrcnn_resnet50_fpn::<f32>(91)?;
    let info = ferrotorch_hub::registry::get_model_info("fasterrcnn_resnet50_fpn")
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "no hub entry for fasterrcnn_resnet50_fpn".into(),
        })?;
    let cache = ferrotorch_hub::cache::HubCache::with_default_dir();
    let path = ferrotorch_hub::download::download_weights(info, &cache)?;
    let state_dict = match info.format {
        ferrotorch_hub::registry::WeightsFormat::SafeTensors => {
            ferrotorch_serialize::load_safetensors::<f32>(&path)?
        }
        ferrotorch_hub::registry::WeightsFormat::FerrotorchStateDict => {
            ferrotorch_serialize::load_state_dict::<f32>(&path)?
        }
    };
    model.load_state_dict(&state_dict, false)?;
    ferrotorch_vision::models::bn_buffer_loader::apply_bn_buffers_from_state_dict(
        &model as &dyn Module<f32>,
        &state_dict,
    )?;
    model.eval();
    Ok(model)
}

fn main() -> Result<(), String> {
    let (img_path, out_path) = parse_args()?;
    eprintln!("[probe_rpn_stages_1141] image={img_path:?} out={out_path:?}");

    let raw = read_image_as_tensor::<f32>(&img_path)
        .map_err(|e| format!("read_image_as_tensor: {e}"))?;
    let input = preprocess(raw).map_err(|e| format!("preprocess: {e}"))?;
    let img_h = input.shape()[2];
    let img_w = input.shape()[3];
    eprintln!("[probe_rpn_stages_1141] padded shape: {:?}", input.shape());

    let model = build_model().map_err(|e| format!("build_model: {e}"))?;

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert("input".into(), input.clone());

    // ---- Backbone ----
    let backbone_feats = model
        .forward_backbone(&input)
        .map_err(|e| format!("backbone: {e}"))?;
    for (k, v) in &backbone_feats {
        sd.insert(format!("backbone_{k}"), v.clone());
    }

    // ---- FPN ----
    let fpn_feats = model
        .forward_fpn(&backbone_feats)
        .map_err(|e| format!("fpn: {e}"))?;
    let level_keys = ["p2", "p3", "p4", "p5", "p6"];
    for &k in &level_keys {
        sd.insert(format!("fpn_{k}"), fpn_feats[k].clone());
    }

    // ---- RPN head per level ----
    let rpn = model.rpn();
    let mut all_cls: Vec<Tensor<f32>> = Vec::new();
    let mut all_box: Vec<Tensor<f32>> = Vec::new();
    let mut fm_sizes: Vec<(usize, usize)> = Vec::new();
    for &k in &level_keys {
        let feat = &fpn_feats[k];
        let (cls, bbox) = rpn
            .head
            .forward_level(feat)
            .map_err(|e| format!("rpn head {k}: {e}"))?;
        fm_sizes.push((feat.shape()[2], feat.shape()[3]));
        sd.insert(format!("cls_{k}"), cls.clone());
        sd.insert(format!("bbox_{k}"), bbox.clone());
        all_cls.push(cls);
        all_box.push(bbox);
    }

    // ---- Anchors per level ----
    // Use the model's anchor generator (whatever stride/aspect/size config
    // the production code currently has). We dump per-level slices so the
    // python side can compare level-by-level.
    let anchor_gen = &rpn.anchor_gen;
    let all_anchors = anchor_gen
        .generate_anchors_for_image::<f32>(&fm_sizes, (img_h, img_w))
        .map_err(|e| format!("generate_anchors_for_image: {e}"))?;
    let anc_data = all_anchors.data_vec().map_err(|e| format!("anc data: {e}"))?;
    let mut offset = 0usize;
    for (i, &k) in level_keys.iter().enumerate() {
        let n = fm_sizes[i].0 * fm_sizes[i].1 * anchor_gen.num_anchors_per_location(i);
        let slice: Vec<f32> = anc_data[offset * 4..(offset + n) * 4].to_vec();
        let t = Tensor::from_storage(TensorStorage::cpu(slice), vec![n, 4], false)
            .map_err(|e| format!("anchor slice: {e}"))?;
        sd.insert(format!("anchors_{k}"), t);
        offset += n;
    }

    // ---- Decoded per-level (anchors + bbox_deltas, weights=1,1,1,1) ----
    let mut offset = 0usize;
    for (i, &k) in level_keys.iter().enumerate() {
        let (fh, fw) = fm_sizes[i];
        let a = anchor_gen.num_anchors_per_location(i);
        let n = fh * fw * a;
        // Build a [n, 4] deltas tensor in matching anchor-flatten order:
        // for fh,fw,ai (row-major), pull bbox channels [ai*4..ai*4+4] at (fh,fw).
        let bbox = &all_box[i];
        let bd = bbox.data_vec().map_err(|e| format!("bbox vec: {e}"))?;
        let mut flat_d: Vec<f32> = Vec::with_capacity(n * 4);
        for fy in 0..fh {
            for fx in 0..fw {
                for ai in 0..a {
                    for d in 0..4 {
                        let didx = (ai * 4 + d) * fh * fw + fy * fw + fx;
                        flat_d.push(bd[didx]);
                    }
                }
            }
        }
        let deltas_t =
            Tensor::from_storage(TensorStorage::cpu(flat_d), vec![n, 4], false)
                .map_err(|e| format!("deltas tensor: {e}"))?;
        // Slice the anchors for this level.
        let anc_slice: Vec<f32> = anc_data[offset * 4..(offset + n) * 4].to_vec();
        let anc_t = Tensor::from_storage(TensorStorage::cpu(anc_slice), vec![n, 4], false)
            .map_err(|e| format!("anc tensor: {e}"))?;
        let decoded = decode_boxes::<f32>(&anc_t, &deltas_t, (1.0, 1.0, 1.0, 1.0))
            .map_err(|e| format!("decode: {e}"))?;
        sd.insert(format!("decoded_{k}"), decoded);
        offset += n;
    }

    // ---- Final post-NMS proposals ----
    // Run the production RPN end-to-end so we can compare proposal sets.
    let fpn_refs: Vec<&Tensor<f32>> = level_keys.iter().map(|&k| &fpn_feats[k]).collect();
    let cfg = RpnConfig::default_eval([img_h, img_w]);
    let proposals = rpn
        .forward(&fpn_refs, &cfg)
        .map_err(|e| format!("rpn forward: {e}"))?;
    sd.insert("proposals_post_nms".into(), proposals);

    save_safetensors::<f32>(&sd, &out_path)
        .map_err(|e| format!("save_safetensors: {e}"))?;

    // Also write a tiny JSON sidecar with image size / level shapes / counts.
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"img_h\": {img_h},\n"));
    json.push_str(&format!("  \"img_w\": {img_w},\n"));
    json.push_str("  \"levels\": [\n");
    for (i, &k) in level_keys.iter().enumerate() {
        let (fh, fw) = fm_sizes[i];
        let a = anchor_gen.num_anchors_per_location(i);
        let sep = if i + 1 < level_keys.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{\"key\": \"{k}\", \"h\": {fh}, \"w\": {fw}, \"a\": {a}}}{sep}\n"
        ));
    }
    json.push_str("  ]\n");
    json.push_str("}\n");
    let mut jp = out_path.clone();
    jp.set_extension("json");
    let mut f = File::create(&jp).map_err(|e| format!("json create: {e}"))?;
    f.write_all(json.as_bytes()).map_err(|e| format!("json write: {e}"))?;

    eprintln!("[probe_rpn_stages_1141] dumped tensors to {out_path:?}");
    eprintln!("[probe_rpn_stages_1141] dumped metadata to {jp:?}");
    Ok(())
}
