//! Per-stage probe for the #1146 LRASPP-MobileNetV3 parity diagnostic.
//!
//! Builds the pretrained `lraspp_mobilenet_v3_large` model, forwards a
//! synthetic input through stem → block0 → ... → block14 → head, and
//! dumps each intermediate to disk as a raw flat f32 binary
//! (`[u32 ndim][u32 × ndim shape][f32 × numel data]`). A Python script
//! then loads each dump, runs the matching torchvision sub-module on
//! the same input, and reports per-stage max-abs / max-rel drift —
//! which pinpoints WHICH stage starts to diverge.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_hub::{
    cache::HubCache,
    download::download_weights,
    registry::{WeightsFormat, get_model_info},
};
use ferrotorch_nn::Module;
use ferrotorch_serialize::{load_safetensors, load_state_dict};
use ferrotorch_vision::models::bn_buffer_loader::apply_bn_buffers_from_state_dict;
use ferrotorch_vision::models::segmentation::lraspp_mobilenet_v3_large;

fn dump_tensor(t: &Tensor<f32>, path: &PathBuf) {
    let shape = t.shape().to_vec();
    let data = t.data_vec().unwrap();
    let mut f = File::create(path).unwrap();
    let ndim = shape.len() as u32;
    f.write_all(&ndim.to_le_bytes()).unwrap();
    for d in &shape {
        f.write_all(&(*d as u32).to_le_bytes()).unwrap();
    }
    for v in &data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/lraspp_probe".to_string());
    std::fs::create_dir_all(&out_dir).unwrap();

    // Build model + load pretrained weights.
    let mut model = lraspp_mobilenet_v3_large::<f32>(21).expect("build");
    let info = get_model_info("lraspp_mobilenet_v3_large").expect("hub registry");
    let cache = HubCache::with_default_dir();
    let path = download_weights(info, &cache).expect("download");
    let state_dict = match info.format {
        WeightsFormat::SafeTensors => load_safetensors::<f32>(&path).expect("safetensors"),
        WeightsFormat::FerrotorchStateDict => load_state_dict::<f32>(&path).expect("state"),
    };
    model.load_state_dict(&state_dict, false).expect("load_state_dict");
    apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state_dict)
        .expect("apply_bn_buffers");
    model.eval();
    eprintln!("[probe] model loaded; running stages...");

    // Fixed seed-ish synthetic input — same recipe Python side will use.
    let b: usize = 1;
    let c: usize = 3;
    let h: usize = 520;
    let w: usize = 520;
    let numel = b * c * h * w;
    let data: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.0001).sin() * 0.5)
        .collect();
    let input = Tensor::from_storage(TensorStorage::cpu(data), vec![b, c, h, w], false).unwrap();

    dump_tensor(&input, &PathBuf::from(format!("{out_dir}/input.bin")));

    // Per-block diagnostic dumps to localize divergence by stage.
    let staged = model
        .backbone_forward_with_block_dumps(&input)
        .expect("backbone_forward_with_block_dumps");
    dump_tensor(&staged.stem, &PathBuf::from(format!("{out_dir}/stem.bin")));
    for (i, b) in staged.blocks.iter().enumerate() {
        dump_tensor(b, &PathBuf::from(format!("{out_dir}/block{i:02}.bin")));
    }
    dump_tensor(&staged.head_conv, &PathBuf::from(format!("{out_dir}/head_conv.bin")));

    // Run backbone explicitly to get (low, high).
    let (low, high) = model
        .backbone_forward_low_high(&input)
        .expect("backbone_forward_low_high");
    dump_tensor(&low, &PathBuf::from(format!("{out_dir}/low.bin")));
    dump_tensor(&high, &PathBuf::from(format!("{out_dir}/high.bin")));

    // Full output.
    let logits = model.forward(&input).expect("forward");
    dump_tensor(&logits, &PathBuf::from(format!("{out_dir}/logits.bin")));

    eprintln!(
        "[probe] dumped input, low [{:?}], high [{:?}], logits [{:?}] to {}",
        low.shape(),
        high.shape(),
        logits.shape(),
        out_dir
    );
}
