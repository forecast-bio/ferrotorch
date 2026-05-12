//! Format-parity dump binary for the ferrotorch-serialize real-artifact
//! harness (Phase G.3, #1169).
//!
//! Companion to `scripts/verify_serialize_inference.py` and
//! `scripts/pin_pretrained_serialize_fixtures.py`. Given a target
//! (`pth` / `safetensors` / `gguf` / `onnx`), a fixture root, and an
//! output dir, this example exercises one of ferrotorch-serialize's
//! four loader/exporter paths and dumps tensors in the same
//! `[u32 ndim][u32 shape...][f32 bytes]` little-endian format the
//! Python pin script uses, so the verifier can do byte-exact (or
//! tolerance-bounded) comparison.
//!
//! Targets:
//!
//! * `pth` — `ferrotorch_serialize::load_pytorch_state_dict::<f32>`
//!   on `<fixture_dir>/resnet18-f37072fd.pth`, dumping each tensor to
//!   `<output_dir>/<key>.bin`.
//! * `safetensors` — `ferrotorch_serialize::load_safetensors::<f32>`
//!   on `<fixture_dir>/resnet18.safetensors`.
//! * `gguf` — `ferrotorch_serialize::load_gguf` +
//!   `dequantize_gguf_tensor` for the names listed in
//!   `<fixture_dir>/sampled_tensor_names.json`.
//! * `onnx` — builds the tiny `Linear(4 -> 8) + ReLU + Linear(8 -> 2)`
//!   MLP from the fixed-seed weights in `<fixture_dir>/mlp_weights.bin`,
//!   runs ferrotorch forward on each `input_*.bin`, and exports the
//!   graph to `<output_dir>/mlp.onnx` via `export_ir_graph_to_onnx`
//!   (manual IR build with `add_constant` for weights — see the inline
//!   note in `dump_onnx` for why the trace-based `export_onnx` path
//!   is not usable for an `nn::Module` today). Dumps
//!   `<output_dir>/ferrotorch_forward_<name>.bin` for the three
//!   inputs so the verifier can compare rust-side ferrotorch vs
//!   rust-emitted-onnx (loaded via Python `onnxruntime`) and vs the
//!   torch reference shipped on the HF mirror.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-serialize --release \
//!   --example serialize_parity_dump -- \
//!     --target pth \
//!     --fixture-dir /tmp/.../resnet18-pth \
//!     --output-dir  /tmp/.../rust_out/pth
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::creation::from_slice;
use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_jit::graph::{IrGraph, IrOpKind};
use ferrotorch_nn::{Linear, Module};
use ferrotorch_serialize::{
    OnnxExportConfig, dequantize_gguf_tensor, export_ir_graph_to_onnx, load_gguf,
    load_pytorch_state_dict, load_safetensors,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    target: String,
    fixture_dir: PathBuf,
    output_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut target: Option<String> = None;
    let mut fixture_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--target" => {
                target = Some(argv.get(i + 1).ok_or("--target needs a value")?.clone());
                i += 2;
            }
            "--fixture-dir" => {
                fixture_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--fixture-dir needs a value")?,
                ));
                i += 2;
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output-dir needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        target: target.ok_or("--target is required")?,
        fixture_dir: fixture_dir.ok_or("--fixture-dir is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
    })
}

// ---------------------------------------------------------------------------
// Single-tensor binary format (mirrors the Python pin script).
// ---------------------------------------------------------------------------

/// Read a single-tensor `[u32 ndim][u32 * ndim shape][f32 * prod(shape)]`
/// little-endian file into `(shape, flat row-major data)`.
fn read_single_tensor_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)
        .map_err(|e| format!("read ndim from {}: {e}", path.display()))?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for di in 0..ndim {
        f.read_exact(&mut buf4)
            .map_err(|e| format!("read shape[{di}] from {}: {e}", path.display()))?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }
    let numel: usize = shape.iter().product();
    let mut data_bytes = vec![0u8; numel * 4];
    f.read_exact(&mut data_bytes)
        .map_err(|e| format!("read data from {}: {e}", path.display()))?;
    let mut data = Vec::with_capacity(numel);
    for chunk in data_bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, data))
}

fn write_single_tensor_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())?;
    }
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)?;
    Ok(())
}

fn write_state_dict_per_tensor(
    state: &HashMap<String, Tensor<f32>>,
    output_dir: &Path,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|e| format!("mkdir {}: {e}", output_dir.display()))?;
    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();
    for k in keys {
        let t = &state[k];
        // Ensure contiguous + CPU before pulling raw data.
        let t_cpu = t
            .contiguous()
            .map_err(|e| format!("contiguous({k}): {e:?}"))?;
        let data = t_cpu
            .data()
            .map_err(|e| format!("data({k}): {e:?}"))?
            .to_vec();
        let shape = t_cpu.shape().to_vec();
        write_single_tensor_f32(&output_dir.join(format!("{k}.bin")), &shape, &data)
            .map_err(|e| format!("write {k}.bin: {e}"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Target A — .pth
// ---------------------------------------------------------------------------

fn dump_pth(fixture_dir: &Path, output_dir: &Path) -> Result<(), String> {
    let pth = fixture_dir.join("resnet18-f37072fd.pth");
    println!("[pth] loading {}", pth.display());
    let state = load_pytorch_state_dict::<f32>(&pth)
        .map_err(|e| format!("load_pytorch_state_dict({}): {e:?}", pth.display()))?;
    println!("[pth] {} tensors loaded", state.len());
    write_state_dict_per_tensor(&state, output_dir)?;
    println!("[pth] dumped to {}", output_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Target B — SafeTensors round-trip
// ---------------------------------------------------------------------------

fn dump_safetensors(fixture_dir: &Path, output_dir: &Path) -> Result<(), String> {
    let st = fixture_dir.join("resnet18.safetensors");
    println!("[safetensors] loading {}", st.display());
    let state = load_safetensors::<f32>(&st)
        .map_err(|e| format!("load_safetensors({}): {e:?}", st.display()))?;
    println!("[safetensors] {} tensors loaded", state.len());
    write_state_dict_per_tensor(&state, output_dir)?;
    println!("[safetensors] dumped to {}", output_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Target C — GGUF
// ---------------------------------------------------------------------------

fn dump_gguf(fixture_dir: &Path, output_dir: &Path) -> Result<(), String> {
    let gguf = fixture_dir.join("SmolLM2-135M-Instruct-Q8_0.gguf");
    println!("[gguf] loading {}", gguf.display());
    let file = load_gguf(&gguf).map_err(|e| format!("load_gguf({}): {e:?}", gguf.display()))?;
    println!("[gguf] {} tensors in file", file.tensors.len());

    let names_path = fixture_dir.join("sampled_tensor_names.json");
    let names_body = fs::read(&names_path)
        .map_err(|e| format!("read {}: {e}", names_path.display()))?;
    let names: Vec<String> = serde_json::from_slice(&names_body)
        .map_err(|e| format!("parse {}: {e}", names_path.display()))?;
    println!("[gguf] dequantizing {} sampled tensors", names.len());

    fs::create_dir_all(output_dir).map_err(|e| format!("mkdir {}: {e}", output_dir.display()))?;
    for name in &names {
        let tensor = dequantize_gguf_tensor(&file, name)
            .map_err(|e| format!("dequantize {name}: {e:?}"))?;
        let t_cpu = tensor
            .contiguous()
            .map_err(|e| format!("contiguous({name}): {e:?}"))?;
        let data = t_cpu
            .data()
            .map_err(|e| format!("data({name}): {e:?}"))?
            .to_vec();
        let shape = t_cpu.shape().to_vec();
        write_single_tensor_f32(&output_dir.join(format!("{name}.bin")), &shape, &data)
            .map_err(|e| format!("write {name}.bin: {e}"))?;
        println!("[gguf]   {name}: shape={shape:?}");
    }
    println!("[gguf] dumped to {}", output_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Target D — ONNX export
// ---------------------------------------------------------------------------

/// Four tensors in the order `(fc1.weight, fc1.bias, fc2.weight, fc2.bias)`,
/// returned by [`read_mlp_weights`]. The named alias keeps the function
/// signature readable for clippy's `type_complexity` lint.
type MlpWeights = (Tensor<f32>, Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// Read the rust-side MLP weights from the python pin's
/// `mlp_weights.bin`. Layout, in order: `fc1.weight [hidden, in]`,
/// `fc1.bias [hidden]`, `fc2.weight [out, hidden]`, `fc2.bias [out]`,
/// each preceded by `[u32 ndim][u32 shape...]`.
fn read_mlp_weights(path: &Path) -> Result<MlpWeights, String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut read_one = |label: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)
            .map_err(|e| format!("[{label}] read ndim: {e}"))?;
        let ndim = u32::from_le_bytes(buf4) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for di in 0..ndim {
            f.read_exact(&mut buf4)
                .map_err(|e| format!("[{label}] read shape[{di}]: {e}"))?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }
        let numel: usize = shape.iter().product();
        let mut data_bytes = vec![0u8; numel * 4];
        f.read_exact(&mut data_bytes)
            .map_err(|e| format!("[{label}] read data: {e}"))?;
        let mut data = Vec::with_capacity(numel);
        for chunk in data_bytes.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok((shape, data))
    };
    let (fc1_w_shape, fc1_w_data) = read_one("fc1.weight")?;
    let (fc1_b_shape, fc1_b_data) = read_one("fc1.bias")?;
    let (fc2_w_shape, fc2_w_data) = read_one("fc2.weight")?;
    let (fc2_b_shape, fc2_b_data) = read_one("fc2.bias")?;
    let fc1_w = from_slice(&fc1_w_data, &fc1_w_shape)
        .map_err(|e| format!("fc1.weight tensor: {e:?}"))?;
    let fc1_b = from_slice(&fc1_b_data, &fc1_b_shape)
        .map_err(|e| format!("fc1.bias tensor: {e:?}"))?;
    let fc2_w = from_slice(&fc2_w_data, &fc2_w_shape)
        .map_err(|e| format!("fc2.weight tensor: {e:?}"))?;
    let fc2_b = from_slice(&fc2_b_data, &fc2_b_shape)
        .map_err(|e| format!("fc2.bias tensor: {e:?}"))?;
    Ok((fc1_w, fc1_b, fc2_w, fc2_b))
}

fn dump_onnx(fixture_dir: &Path, output_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|e| format!("mkdir {}: {e}", output_dir.display()))?;
    let weights = read_mlp_weights(&fixture_dir.join("mlp_weights.bin"))?;
    let (fc1_w, fc1_b, fc2_w, fc2_b) = weights;
    println!(
        "[onnx] loaded weights: fc1.weight={:?} fc1.bias={:?} fc2.weight={:?} fc2.bias={:?}",
        fc1_w.shape(),
        fc1_b.shape(),
        fc2_w.shape(),
        fc2_b.shape()
    );

    // Build ferrotorch Linear layers and overwrite their parameters
    // with the torch-pinned weights.
    let in_features = fc1_w.shape()[1];
    let hidden_features = fc1_w.shape()[0];
    let out_features = fc2_w.shape()[0];
    assert_eq!(fc2_w.shape()[1], hidden_features, "shape mismatch fc2");
    let mut fc1: Linear<f32> = Linear::new(in_features, hidden_features, true)
        .map_err(|e| format!("Linear::new(fc1): {e:?}"))?;
    let mut fc2: Linear<f32> = Linear::new(hidden_features, out_features, true)
        .map_err(|e| format!("Linear::new(fc2): {e:?}"))?;
    fc1.weight.set_data(fc1_w);
    fc1.bias
        .as_mut()
        .expect("fc1 bias=true")
        .set_data(fc1_b);
    fc2.weight.set_data(fc2_w);
    fc2.bias
        .as_mut()
        .expect("fc2 bias=true")
        .set_data(fc2_b);

    // Forward the three fixed inputs through the ferrotorch MLP.
    for name in &["zeros", "ones", "random"] {
        let (shape, data) =
            read_single_tensor_f32(&fixture_dir.join(format!("input_{name}.bin")))?;
        let x = from_slice(&data, &shape)
            .map_err(|e| format!("input_{name} tensor: {e:?}"))?;
        let h = fc1.forward(&x).map_err(|e| format!("fc1.forward: {e:?}"))?;
        let r = relu(&h).map_err(|e| format!("relu: {e:?}"))?;
        let y = fc2.forward(&r).map_err(|e| format!("fc2.forward: {e:?}"))?;
        let y_cpu = y.contiguous().map_err(|e| format!("contiguous(y): {e:?}"))?;
        let y_data = y_cpu
            .data()
            .map_err(|e| format!("data(y): {e:?}"))?
            .to_vec();
        let y_shape = y_cpu.shape().to_vec();
        write_single_tensor_f32(
            &output_dir.join(format!("ferrotorch_forward_{name}.bin")),
            &y_shape,
            &y_data,
        )
        .map_err(|e| format!("write ferrotorch_forward_{name}.bin: {e}"))?;
        println!(
            "[onnx]   ferrotorch forward[{name}]: y.shape={y_shape:?} y_norm={:.4}",
            y_data.iter().map(|v| v * v).sum::<f32>().sqrt()
        );
    }

    // Export the model to ONNX. We construct the `IrGraph` by hand
    // here — using `export_onnx(trace_fn, ...)` would route the
    // weights through the JIT tracer, which emits every leaf tensor
    // (including `nn::Parameter`s) as a graph **input** rather than
    // a Constant initializer. Modern ONNX Runtime (IR_VERSION>=4)
    // refuses to run such a graph because the weight inputs have no
    // feeds at inference time. There is no ferrotorch API today that
    // converts a `Module`'s parameters into ONNX initializers via
    // tracing — `export_from_program` shares the same IR-graph
    // representation and inherits the gap. The manual IR build
    // sidesteps it: `add_constant` produces `IrOpKind::Constant`
    // nodes, which the ONNX exporter (the constant arm in
    // `ir_graph_to_onnx`) emits as initializer `TensorProto`s.
    // This is the correct shape for the exporter; until ferrotorch
    // grows a `Module::to_onnx_program(...)` style API, real users
    // exporting a `nn::Module` will have to mirror this manual build.
    let mut graph = IrGraph::new();
    let x = graph.add_input(vec![1, in_features]);

    // Linear-1 weights + bias as constants.
    let fc1_w_t = fc1.weight.tensor();
    let fc1_w_data: Vec<f64> = fc1_w_t
        .contiguous()
        .map_err(|e| format!("fc1.weight contig: {e:?}"))?
        .data()
        .map_err(|e| format!("fc1.weight data: {e:?}"))?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let fc1_w_val = graph.add_constant(fc1_w_data, vec![hidden_features, in_features]);
    let fc1_b_t = fc1.bias.as_ref().unwrap().tensor();
    let fc1_b_data: Vec<f64> = fc1_b_t
        .contiguous()
        .map_err(|e| format!("fc1.bias contig: {e:?}"))?
        .data()
        .map_err(|e| format!("fc1.bias data: {e:?}"))?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let fc1_b_val = graph.add_constant(fc1_b_data, vec![hidden_features]);

    // fc1.forward(x) → Linear node (Gemm) → relu → fc2.forward → output.
    let (_, h_outs) = graph.add_node(
        IrOpKind::Linear,
        vec![x, fc1_w_val, fc1_b_val],
        vec![vec![1, hidden_features]],
    );
    let (_, r_outs) = graph.add_node(IrOpKind::Relu, vec![h_outs[0]], vec![vec![1, hidden_features]]);

    let fc2_w_t = fc2.weight.tensor();
    let fc2_w_data: Vec<f64> = fc2_w_t
        .contiguous()
        .map_err(|e| format!("fc2.weight contig: {e:?}"))?
        .data()
        .map_err(|e| format!("fc2.weight data: {e:?}"))?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let fc2_w_val = graph.add_constant(fc2_w_data, vec![out_features, hidden_features]);
    let fc2_b_t = fc2.bias.as_ref().unwrap().tensor();
    let fc2_b_data: Vec<f64> = fc2_b_t
        .contiguous()
        .map_err(|e| format!("fc2.bias contig: {e:?}"))?
        .data()
        .map_err(|e| format!("fc2.bias data: {e:?}"))?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let fc2_b_val = graph.add_constant(fc2_b_data, vec![out_features]);

    let (_, y_outs) = graph.add_node(
        IrOpKind::Linear,
        vec![r_outs[0], fc2_w_val, fc2_b_val],
        vec![vec![1, out_features]],
    );
    graph.set_outputs(vec![y_outs[0]]);

    let onnx_path = output_dir.join("mlp.onnx");
    let mut config = OnnxExportConfig::default();
    config.opset_version = 17;
    config.model_name = "ferrotorch_serialize_parity_mlp".to_string();
    export_ir_graph_to_onnx(&graph, &onnx_path, config, 1 /* ONNX_FLOAT */)
        .map_err(|e| format!("export_ir_graph_to_onnx: {e:?}"))?;
    println!(
        "[onnx] wrote {} ({} bytes)",
        onnx_path.display(),
        fs::metadata(&onnx_path)
            .map(|m| m.len())
            .unwrap_or_default()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args = parse_args()?;
    fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("mkdir {}: {e}", args.output_dir.display()))?;
    match args.target.as_str() {
        "pth" => dump_pth(&args.fixture_dir, &args.output_dir),
        "safetensors" => dump_safetensors(&args.fixture_dir, &args.output_dir),
        "gguf" => dump_gguf(&args.fixture_dir, &args.output_dir),
        "onnx" => dump_onnx(&args.fixture_dir, &args.output_dir),
        other => Err(format!(
            "unknown --target {other:?}; expected one of pth|safetensors|gguf|onnx"
        )),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("serialize_parity_dump FAILED: {e}");
        std::process::exit(1);
    }
}
