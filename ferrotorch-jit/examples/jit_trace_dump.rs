//! JIT trace + AoT-compile dump binary for the ferrotorch-jit
//! real-artifact parity harness (Phase G.4, #1170).
//!
//! Companion to `scripts/verify_jit_inference.py` and
//! `scripts/pin_pretrained_jit_fixtures.py`. The pin script ships a
//! deterministic 2-layer MLP
//! (`Linear(8 -> 16, bias=True) -> ReLU -> Linear(16 -> 4, bias=True)`)
//! plus `_value_parity_{input,output}.bin` reference fixtures to the
//! HF mirror `ferrotorch/jit-trace-parity-v1`. This example:
//!
//!   1. Resolves the pinned model + fixture directory (either passed
//!      via `--fixture-dir`, or downloaded through
//!      [`ferrotorch_hub::load_pretrained`] which lands the
//!      safetensors in the local cache; the harness pre-downloads the
//!      parity `.bin`s alongside).
//!   2. Loads the four flat `state_dict` keys
//!      (`l1_weight`, `l1_bias`, `l2_weight`, `l2_bias`) into a
//!      [`Mlp`] holding two [`Linear`] layers.
//!   3. Reads the `[N, in_features]` input batch from
//!      `_value_parity_input.bin`.
//!   4. **Eager** forward: runs the MLP's standard
//!      [`Module::forward`] (uses `linear_fused` + `relu` + `linear_fused`)
//!      and dumps the `[N, out_features]` output to `eager.bin`.
//!   5. **Traced** forward: calls [`ferrotorch_jit::trace`] over the
//!      same closure (with the inputs cloned + `requires_grad`
//!      enabled so the autograd graph builds) and re-executes the IR
//!      via [`TracedModule::forward_multi`] (the tracer pulls weights
//!      in as additional leaf inputs — we pass `[input, l1.weight,
//!      l1.bias, l2.weight, l2.bias]`). The result is dumped to
//!      `traced.bin`.
//!   6. **Compiled** forward: calls [`ferrotorch_jit::compile`] (which
//!      traces + runs `constant_fold` / `dce` / `operator_fusion` /
//!      `memory_planning`) on the same closure, then executes and
//!      dumps `compiled.bin`.
//!
//! Output binary format (matches the python pin script's `dump_f32`):
//!
//!   `[u32 ndim][u32 × ndim shape][f32 le data]`
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-jit --release --example jit_trace_dump -- \
//!     --model jit-trace-parity-v1 \
//!     --fixture-dir /tmp/.../fixtures \
//!     --output-dir /tmp/jit_dump
//! ```
//!
//! `--fixture-dir` must contain `_value_parity_input.bin` and
//! `_value_parity_output.bin`. When omitted, the example falls back
//! to reading from the same directory as the cached safetensors
//! (sibling files) — works when the python harness pre-downloaded
//! the whole repo to the HF cache.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{from_slice, FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_hub::load_pretrained;
use ferrotorch_jit::{compile, trace, TracedModule};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::Linear;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    model: String,
    fixture_dir: Option<PathBuf>,
    output_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut fixture_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv.get(i + 1).ok_or("--model needs a value")?.clone());
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
        model: model.ok_or("--model is required")?,
        fixture_dir,
        output_dir: output_dir.ok_or("--output-dir is required")?,
    })
}

// ---------------------------------------------------------------------------
// Binary I/O — single-tensor `[u32 ndim][u32 × ndim shape][f32 data]`
// (matches the python pin script's `dump_f32`).
// ---------------------------------------------------------------------------

fn read_f32_tensor(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
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

fn write_f32_tensor(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(&(shape.len() as u32).to_le_bytes())
        .map_err(|e| format!("write ndim to {}: {e}", path.display()))?;
    for d in shape {
        f.write_all(&(*d as u32).to_le_bytes())
            .map_err(|e| format!("write shape to {}: {e}", path.display()))?;
    }
    let mut buf: Vec<u8> = Vec::with_capacity(data.len() * 4);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
        .map_err(|e| format!("write data to {}: {e}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// MLP — the rust mirror of the python pin's `class MLP(nn.Module)`. We
// hand-roll the struct instead of using `nn::Sequential` so the weight
// keys load 1:1 from the flat pinned state_dict (`l1_weight`, `l1_bias`,
// `l2_weight`, `l2_bias`).
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Mlp {
    l1: Linear<f32>,
    l2: Linear<f32>,
}

impl Mlp {
    fn new(in_features: usize, hidden: usize, out_features: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            l1: Linear::<f32>::new(in_features, hidden, true)?,
            l2: Linear::<f32>::new(hidden, out_features, true)?,
        })
    }

    fn load_pin(&mut self, state: &std::collections::HashMap<String, Tensor<f32>>) -> Result<(), String> {
        let l1_w = state
            .get("l1_weight")
            .ok_or("pin state_dict missing key `l1_weight`")?;
        let l1_b = state
            .get("l1_bias")
            .ok_or("pin state_dict missing key `l1_bias`")?;
        let l2_w = state
            .get("l2_weight")
            .ok_or("pin state_dict missing key `l2_weight`")?;
        let l2_b = state
            .get("l2_bias")
            .ok_or("pin state_dict missing key `l2_bias`")?;
        let to_data = |t: &Tensor<f32>| -> Result<Vec<f32>, String> {
            t.data()
                .map(<[f32]>::to_vec)
                .map_err(|e| format!("pin tensor data(): {e}"))
        };
        let l1_w_shape = l1_w.shape().to_vec();
        let l1_b_shape = l1_b.shape().to_vec();
        let l2_w_shape = l2_w.shape().to_vec();
        let l2_b_shape = l2_b.shape().to_vec();
        let l1_w_data = to_data(l1_w)?;
        let l1_b_data = to_data(l1_b)?;
        let l2_w_data = to_data(l2_w)?;
        let l2_b_data = to_data(l2_b)?;
        self.l1.weight =
            Parameter::from_slice(&l1_w_data, &l1_w_shape).map_err(|e| format!("l1.weight: {e}"))?;
        self.l1.bias = Some(
            Parameter::from_slice(&l1_b_data, &l1_b_shape).map_err(|e| format!("l1.bias: {e}"))?,
        );
        self.l2.weight =
            Parameter::from_slice(&l2_w_data, &l2_w_shape).map_err(|e| format!("l2.weight: {e}"))?;
        self.l2.bias = Some(
            Parameter::from_slice(&l2_b_data, &l2_b_shape).map_err(|e| format!("l2.bias: {e}"))?,
        );
        Ok(())
    }

    fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let h = self.l1.forward(input)?;
        let h = ferrotorch_core::grad_fns::activation::relu(&h)?;
        self.l2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Returns a tuple `(state_dict, fixture_dir)` for the pinned model.
/// `fixture_dir` is whatever the user passed via `--fixture-dir`; the
/// caller falls back to other locations if it's `None`.
fn load_model_state(
    model: &str,
) -> Result<std::collections::HashMap<String, Tensor<f32>>, String> {
    load_pretrained::<f32>(model).map_err(|e| match e {
        FerrotorchError::InvalidArgument { message } => format!("hub load_pretrained: {message}"),
        other => format!("hub load_pretrained: {other}"),
    })
}

fn pick_fixture(args: &Args, name: &str) -> Result<PathBuf, String> {
    if let Some(dir) = &args.fixture_dir {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
        return Err(format!(
            "fixture {name} not found in --fixture-dir {}",
            dir.display()
        ));
    }
    // Fallback: look in the HF hub cache alongside the safetensors.
    // `default_cache_dir()` is `~/.cache/ferrotorch_hub`; the
    // `load_pretrained` path is `<cache>/<model_name>.safetensors`,
    // but the harness pre-downloads the parity .bins next to the
    // safetensors via `hf_hub_download`. We do not try to walk the
    // HF cache layout here — the harness always passes `--fixture-dir`.
    Err(format!(
        "fixture {name} not found and --fixture-dir was not passed; the harness must pre-download \
         `_value_parity_input.bin` and `_value_parity_output.bin` and point --fixture-dir at the \
         directory containing them"
    ))
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("mkdir {}: {e}", args.output_dir.display()))?;

    eprintln!("[jit_trace_dump] model = {}", args.model);
    eprintln!("[jit_trace_dump] output-dir = {}", args.output_dir.display());

    // ---- Load pinned state + fixtures. -----------------------------------
    let state = load_model_state(&args.model)?;
    eprintln!("[jit_trace_dump] loaded state_dict keys: {:?}", {
        let mut k: Vec<_> = state.keys().cloned().collect();
        k.sort();
        k
    });

    let input_path = pick_fixture(&args, "_value_parity_input.bin")?;
    let (input_shape, input_data) = read_f32_tensor(&input_path)?;
    eprintln!(
        "[jit_trace_dump] loaded input fixture: shape={input_shape:?} numel={}",
        input_data.len()
    );

    // The pin script's `class MLP` is fixed at in=8 / hidden=16 / out=4.
    // We re-derive them from the pinned state_dict shapes so a future
    // architecture change in the pin script is surfaced as a shape
    // mismatch error rather than a silent wrong-shape forward.
    let l1_w_shape = state
        .get("l1_weight")
        .ok_or("state_dict missing `l1_weight`")?
        .shape()
        .to_vec();
    if l1_w_shape.len() != 2 {
        return Err(format!(
            "expected l1_weight ndim==2 (`[out, in]`), got {l1_w_shape:?}"
        ));
    }
    let hidden = l1_w_shape[0];
    let in_features = l1_w_shape[1];

    let l2_w_shape = state
        .get("l2_weight")
        .ok_or("state_dict missing `l2_weight`")?
        .shape()
        .to_vec();
    if l2_w_shape.len() != 2 {
        return Err(format!(
            "expected l2_weight ndim==2 (`[out, in]`), got {l2_w_shape:?}"
        ));
    }
    if l2_w_shape[1] != hidden {
        return Err(format!(
            "shape mismatch: l1 hidden={hidden} but l2 expects hidden={}",
            l2_w_shape[1]
        ));
    }
    let out_features = l2_w_shape[0];

    eprintln!(
        "[jit_trace_dump] architecture: Linear({in_features}->{hidden}) -> ReLU -> \
         Linear({hidden}->{out_features})",
    );

    let mut mlp = Mlp::new(in_features, hidden, out_features)
        .map_err(|e| format!("Mlp::new: {e}"))?;
    mlp.load_pin(&state)?;

    // ---- Build the input tensor. ------------------------------------------
    let input = from_slice::<f32>(&input_data, &input_shape).map_err(|e| format!("from_slice input: {e}"))?;

    // ---- 1) Eager forward. ------------------------------------------------
    let eager_out = mlp.forward(&input).map_err(|e| format!("eager forward: {e}"))?;
    let eager_shape = eager_out.shape().to_vec();
    let eager_vec = eager_out
        .data()
        .map_err(|e| format!("eager data(): {e}"))?
        .to_vec();
    eprintln!(
        "[jit_trace_dump] eager output shape={eager_shape:?} sample={:?}",
        &eager_vec[..eager_vec.len().min(4)]
    );
    write_f32_tensor(&args.output_dir.join("eager.bin"), &eager_shape, &eager_vec)?;

    // ---- Build a `requires_grad=true` input clone for tracing.  ----------
    // The tracer's BFS walks `output.grad_fn().inputs()`, so the input
    // tensor must participate in the autograd graph. We clone the
    // input data and call `.requires_grad_(true)`.
    let input_grad = from_slice::<f32>(&input_data, &input_shape)
        .map_err(|e| format!("from_slice traced input: {e}"))?
        .requires_grad_(true);

    // ---- 2) Traced forward. ----------------------------------------------
    // The tracer captures `linear_fused` + `relu` + `linear_fused` into
    // an IR graph; weight + bias parameters become additional leaf
    // inputs (the tracer cannot inline `Parameter<T>` as constants
    // without an explicit `constant_folding` pass — that's what
    // `compile()` adds).
    //
    // We pass the closure references to `mlp.l1/l2` so the autograd
    // graph picks up the parameter tensors as `requires_grad` leaves.
    let traced_graph = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            let h = mlp.l1.forward(&inputs[0])?;
            let h = ferrotorch_core::grad_fns::activation::relu(&h)?;
            mlp.l2.forward(&h)
        },
        std::slice::from_ref(&input_grad),
    )
    .map_err(|e| format!("trace: {e}"))?;

    let traced_module = TracedModule::<f32>::new(traced_graph);
    eprintln!(
        "[jit_trace_dump] traced graph: input_count={} output_shape={:?} nodes={}",
        traced_module.input_count(),
        traced_module.output_shape(),
        traced_module.graph().node_count()
    );

    // The tracer pulls Parameter leaves in as extra inputs. We build
    // the full inputs vec in BFS-leaf order: the public input first,
    // then whatever extra leaves the graph captured. Concretely the
    // tracer's BFS for our two-linear-then-add chain typically
    // surfaces `[input, l2.weight, l2.bias, l1.weight, l1.bias]` —
    // we discover the right ordering by reading `graph.input_values`.
    let traced_inputs = build_traced_inputs(&traced_module, &input, &mlp)?;
    let traced_out = traced_module
        .forward_multi(&traced_inputs)
        .map_err(|e| format!("traced forward: {e}"))?;
    let traced_shape = traced_out.shape().to_vec();
    let traced_vec = traced_out
        .data()
        .map_err(|e| format!("traced data(): {e}"))?
        .to_vec();
    eprintln!(
        "[jit_trace_dump] traced output shape={traced_shape:?} sample={:?}",
        &traced_vec[..traced_vec.len().min(4)]
    );
    write_f32_tensor(&args.output_dir.join("traced.bin"), &traced_shape, &traced_vec)?;

    // ---- 3) Compiled forward. --------------------------------------------
    // `compile()` re-traces + applies constant_fold + dce + fusion +
    // memory_planning. Constant folding inlines the Parameter leaves,
    // which usually collapses input_count back down to 1.
    let input_grad2 = from_slice::<f32>(&input_data, &input_shape)
        .map_err(|e| format!("from_slice compiled input: {e}"))?
        .requires_grad_(true);
    let compiled = compile(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            let h = mlp.l1.forward(&inputs[0])?;
            let h = ferrotorch_core::grad_fns::activation::relu(&h)?;
            mlp.l2.forward(&h)
        },
        std::slice::from_ref(&input_grad2),
        None,
    )
    .map_err(|e| format!("compile: {e}"))?;
    eprintln!(
        "[jit_trace_dump] compiled graph: input_count={} output_shape={:?} nodes={}",
        compiled.input_count(),
        compiled.output_shape(),
        compiled.graph().node_count()
    );

    let compiled_inputs = build_traced_inputs(&compiled, &input, &mlp)?;
    let compiled_out = compiled
        .forward_multi(&compiled_inputs)
        .map_err(|e| format!("compiled forward: {e}"))?;
    let compiled_shape = compiled_out.shape().to_vec();
    let compiled_vec = compiled_out
        .data()
        .map_err(|e| format!("compiled data(): {e}"))?
        .to_vec();
    eprintln!(
        "[jit_trace_dump] compiled output shape={compiled_shape:?} sample={:?}",
        &compiled_vec[..compiled_vec.len().min(4)]
    );
    write_f32_tensor(
        &args.output_dir.join("compiled.bin"),
        &compiled_shape,
        &compiled_vec,
    )?;

    eprintln!("[jit_trace_dump] done. Wrote eager.bin / traced.bin / compiled.bin");
    Ok(())
}

/// Build the `inputs` slice the traced/compiled IR graph expects.
///
/// The interpreter's contract is `inputs.len() == graph.input_values.len()`.
/// `compile()`'s constant-folding pass inlines Parameter leaves and
/// usually collapses `input_count` to 1; the pure `trace()` path may
/// leave them as extra leaves. We support both: pass the public input
/// first, then — only if `input_count > 1` — also pass every Parameter
/// the MLP owns, in a fixed order. If `input_count` still doesn't
/// match, we surface a clear error rather than guess.
fn build_traced_inputs(
    module: &TracedModule<f32>,
    public_input: &Tensor<f32>,
    mlp: &Mlp,
) -> Result<Vec<Tensor<f32>>, String> {
    let want = module.input_count();
    if want == 1 {
        return Ok(vec![public_input.clone()]);
    }

    // Build a candidate pool of Parameter leaves. The tracer's BFS
    // order is deterministic but depends on how `linear_fused` wires
    // its weight/bias inputs; rather than reverse-engineering that,
    // we use the IrGraph's `input_values` shape metadata to match
    // each leaf to the right Parameter.
    let graph = module.graph();
    let mut leaves: Vec<Tensor<f32>> = Vec::with_capacity(want);
    leaves.push(public_input.clone());
    for &val_id in graph.input_values.iter().skip(1) {
        let shape = graph
            .values
            .iter()
            .find(|v| v.id == val_id)
            .map(|v| v.shape.clone())
            .ok_or_else(|| format!("IR input value {val_id:?} has no metadata"))?;
        let candidates: [(&'static str, &Tensor<f32>); 4] = [
            ("l1.weight", mlp.l1.weight.tensor()),
            ("l1.bias", mlp.l1.bias.as_ref().expect("bias=true").tensor()),
            ("l2.weight", mlp.l2.weight.tensor()),
            ("l2.bias", mlp.l2.bias.as_ref().expect("bias=true").tensor()),
        ];
        let pick = candidates
            .iter()
            .find(|(_, t)| t.shape() == shape.as_slice())
            .ok_or_else(|| {
                format!(
                    "could not match IR input shape {shape:?} to any MLP parameter; \
                     candidate shapes: l1.weight={:?} l1.bias={:?} l2.weight={:?} l2.bias={:?}",
                    candidates[0].1.shape(),
                    candidates[1].1.shape(),
                    candidates[2].1.shape(),
                    candidates[3].1.shape(),
                )
            })?;
        eprintln!(
            "[jit_trace_dump]   IR extra leaf {val_id:?} shape={shape:?} -> {}",
            pick.0
        );
        leaves.push(pick.1.clone());
    }
    if leaves.len() != want {
        return Err(format!(
            "build_traced_inputs: built {} leaves but graph expects {want}",
            leaves.len()
        ));
    }
    Ok(leaves)
}
