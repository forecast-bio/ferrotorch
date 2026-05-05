//! End-to-end GPU training integration test (closes issue #733).
//!
//! Counterpart to `cpu_training.rs`: a small MLP trained on synthetic
//! regression data, with every parameter, every activation, and every
//! gradient living on `Device::Cuda(0)` for the entire training loop.
//!
//! What this test enforces beyond a compile-only smoke check:
//!
//! 1. **Loss decreases.** The initial loss is recorded; after N steps
//!    the final loss must drop below 0.9 of the initial value. A
//!    no-op loop (forward only, no parameter update) cannot satisfy
//!    this.
//! 2. **Gradients live on CUDA after `backward()`.** Every parameter
//!    whose `.grad()` is `Some(_)` must have device == `Cuda(0)`.
//!    Catches the autograd "CPU detour" anti-pattern where backward
//!    silently moves through host RAM.
//! 3. **Parameters stay on CUDA after `optimizer.step()`.** Every
//!    parameter must remain device-resident across the optimizer
//!    update. Catches the optimizer "CPU detour" anti-pattern where
//!    `step()` materializes parameters on CPU and replaces them.
//!
//! Runtime: this file is feature-gated on `gpu`. Without that feature
//! it produces no compilation output (cf. `#![cfg(feature = "gpu")]`).
//! Run with `cargo test -p ferrotorch --test gpu_training --features gpu`.

#![cfg(feature = "gpu")]

use ferrotorch::gpu::init_cuda_backend;
use ferrotorch_core::grad_fns::{activation, arithmetic, linalg, reduction, transcendental};
use ferrotorch_core::{
    Device, FerrotorchResult, Tensor, chunk_t, from_vec, permute_t, rand, randn, scalar,
};
use ferrotorch_nn::{
    AvgPool2d, Conv2d, Embedding, LSTM, LayerNorm, Linear, Module, Parameter, ReLU,
    RotaryPositionEmbedding, Sequential,
};
use ferrotorch_optim::{Adam, AdamConfig, AdamW, AdamWConfig, Optimizer, Sgd, SgdConfig};

/// Build a deterministic synthetic regression dataset: 32 samples,
/// 4 features each, target is a fixed linear combination of the
/// features so the optimizer has a real signal to track.
fn synth_dataset() -> FerrotorchResult<(Tensor<f32>, Tensor<f32>)> {
    let n_samples = 32usize;
    let n_features = 4usize;

    let x_data: Vec<f32> = (0..n_samples * n_features)
        .map(|i| {
            // Spread values in [-1.0, 1.0] deterministically.
            let f = (i as f32) * 0.07;
            f.sin()
        })
        .collect();
    let x = from_vec(x_data, &[n_samples, n_features])?;

    // Targets: fixed linear combination + small offset, so the MLP
    // must actually learn weights — a zero-weight init produces a
    // non-trivial MSE at step 0.
    let weights = [0.5_f32, -0.3, 0.7, -0.2];
    let bias = 0.1_f32;
    let mut y_data = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let mut acc = bias;
        for (f, &w) in weights.iter().enumerate() {
            acc += w * (((s * n_features + f) as f32) * 0.07).sin();
        }
        y_data.push(acc);
    }
    let y = from_vec(y_data, &[n_samples, 1])?;

    Ok((x, y))
}

#[test]
fn gpu_mlp_training_smoke() {
    // Initialize the CUDA backend exactly once. If CUDA is not
    // available on this machine the test fails loudly here — there
    // is no silent CPU fallback, per `rust-gpu-discipline` §3.
    init_cuda_backend().expect("CUDA backend must initialize for the GPU training test");

    let device = Device::Cuda(0);

    // -------------------------------------------------------------
    // Model: tiny MLP, 4 -> 16 -> 1.
    // Built on CPU then moved to CUDA via `Module::to_device`. After
    // the move every parameter's underlying storage lives in VRAM.
    // -------------------------------------------------------------
    let mut model: Sequential<f32> = Sequential::new(vec![
        Box::new(Linear::<f32>::new(4, 16, true).expect("Linear(4,16)")),
        Box::new(ReLU::default()),
        Box::new(Linear::<f32>::new(16, 1, true).expect("Linear(16,1)")),
    ]);
    model
        .to_device(device)
        .expect("model.to_device(Cuda(0)) must succeed");

    // Sanity: every parameter is device-resident before training begins.
    for (name, p) in model.named_parameters() {
        assert_eq!(
            p.tensor().device(),
            device,
            "parameter {name} not on Cuda(0) after to_device(); got {:?}",
            p.tensor().device(),
        );
    }

    // -------------------------------------------------------------
    // Optimizer.
    // The optimizer takes ownership of cloned `Parameter`s. Cloning
    // is Arc-shared, so the optimizer's view of each parameter still
    // points at the same CUDA buffer as the model's view.
    // SGD is intentional here — exercising every kernel needed for
    // a parameter update without depending on Adam's moment buffers.
    // -------------------------------------------------------------
    let opt_params = model.parameters().into_iter().cloned().collect();
    let mut optimizer = Sgd::new(opt_params, SgdConfig::new(0.05));

    // -------------------------------------------------------------
    // Data: pinned to GPU once, reused across all steps. Mirroring
    // the CPU test which holds the batch fixed across steps so loss
    // is guaranteed to decrease for a sane optimizer.
    // -------------------------------------------------------------
    let (x_cpu, y_cpu) = synth_dataset().expect("synthetic dataset must build");
    let x = x_cpu.to(device).expect("x.to(Cuda(0))");
    let y = y_cpu.to(device).expect("y.to(Cuda(0))");
    assert_eq!(x.device(), device);
    assert_eq!(y.device(), device);

    // We compute MSE manually via the autograd-aware primitives
    // (`sub` + `mul` + `mean`) rather than `nn::MSELoss::forward`.
    //
    // Why: `nn::MSELoss::forward` calls into `binary_map` and then
    // unconditionally rewraps the output in `TensorStorage::cpu(...)`
    // (loss.rs:79-93). On a CUDA tensor `binary_map` further calls
    // `tensor.data()`, which returns `GpuTensorNotAccessible`. Both
    // sites are real bugs surfaced by this test; see the report's
    // "Findings" section for follow-ups.
    //
    // The transformer test in `cpu_training.rs` (Experiment 2) uses
    // the same manual MSE pattern, so we are matching an existing
    // workspace convention rather than inventing one.

    // -------------------------------------------------------------
    // Training loop.
    // -------------------------------------------------------------
    let n_steps = 30usize;
    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;

    for step in 0..n_steps {
        optimizer.zero_grad().expect("zero_grad");

        // Forward.
        let output = model.forward(&x).expect("model.forward on GPU");
        assert_eq!(
            output.device(),
            device,
            "output of GPU forward landed on {:?} at step {step}",
            output.device(),
        );

        // Manual MSE: mean((output - y)^2). Each call dispatches on
        // the input tensor's device, so the entire chain stays on GPU.
        let diff = arithmetic::sub(&output, &y).expect("sub on GPU");
        assert_eq!(
            diff.device(),
            device,
            "diff landed on {:?} at step {step}",
            diff.device(),
        );
        let sq = arithmetic::mul(&diff, &diff).expect("square on GPU");
        let loss = reduction::mean(&sq).expect("mean on GPU");
        assert_eq!(
            loss.device(),
            device,
            "loss tensor landed on {:?} at step {step}",
            loss.device(),
        );

        // Read the scalar loss (transparent D2H readback). We do this
        // for the assertion only — the entire forward + backward
        // chain ran on GPU.
        let loss_val = loss
            .data_vec()
            .expect("loss.data_vec read-back")
            .first()
            .copied()
            .expect("loss is a scalar tensor");
        assert!(
            loss_val.is_finite(),
            "non-finite loss {loss_val} at step {step}",
        );
        if first_loss.is_none() {
            first_loss = Some(loss_val);
        }
        last_loss = loss_val;

        // Backward.
        loss.backward().expect("backward on GPU graph");

        // ---- Assertion (b): gradients live on CUDA after backward.
        let model_params = model.parameters();
        for (i, p) in model_params.iter().enumerate() {
            if let Some(g) = p.grad().expect("read grad") {
                assert_eq!(
                    g.device(),
                    device,
                    "param[{i}] grad landed on {:?} at step {step} \
                     (autograd CPU detour suspected)",
                    g.device(),
                );
            }
        }

        // Sync gradients from model parameters into the optimizer's
        // cloned parameters (mirrors the CPU test pattern).
        let opt_params_view = optimizer.param_groups()[0].params();
        for (mp, op) in model_params.iter().zip(opt_params_view.iter()) {
            if let Some(g) = mp.grad().expect("read grad for sync") {
                op.set_grad(Some(g)).expect("set_grad on opt param");
            }
        }

        optimizer.step().expect("optimizer.step on GPU params");

        // ---- Assertion (c): parameters stay on CUDA after step().
        for (i, p) in optimizer.param_groups()[0].params().iter().enumerate() {
            assert_eq!(
                p.device(),
                device,
                "optimizer param[{i}] migrated off device to {:?} at step {step} \
                 (optimizer CPU detour suspected)",
                p.device(),
            );
        }
        for (i, p) in model.parameters().iter().enumerate() {
            assert_eq!(
                p.tensor().device(),
                device,
                "model param[{i}] migrated off device to {:?} at step {step}",
                p.tensor().device(),
            );
        }
    }

    let initial = first_loss.expect("at least one step ran");
    eprintln!("gpu_mlp_training_smoke: initial loss = {initial:.6}, final loss = {last_loss:.6}");

    // ---- Assertion (a): loss decreased.
    // 0.9 * initial gives noise headroom but cleanly fails a no-op
    // loop where the loss stays roughly constant.
    assert!(
        last_loss < 0.9 * initial,
        "MLP GPU: loss should decrease meaningfully. \
         initial={initial:.6}, final={last_loss:.6}, threshold={:.6}",
        0.9 * initial,
    );
}

// =====================================================================
// Shared helpers for the GPU training experiments below.
// =====================================================================

/// Assert that every parameter in `params` is on `device`.
///
/// Failure messages name the parameter index and the offending device, so a
/// stray CPU-resident tensor is identified immediately. Mirrors the inline
/// asserts in `gpu_mlp_training_smoke` and is shared because the post-step
/// assertion is identical across every experiment.
fn assert_all_on_device(label: &str, params: &[&Parameter<f32>], device: Device, step: usize) {
    for (i, p) in params.iter().enumerate() {
        assert_eq!(
            p.tensor().device(),
            device,
            "[{label}] param[{i}] migrated off device to {:?} at step {step} \
             (CPU detour suspected)",
            p.tensor().device(),
        );
    }
}

/// Assert that every populated gradient on `params` is on `device`.
fn assert_grads_on_device(label: &str, params: &[&Parameter<f32>], device: Device, step: usize) {
    for (i, p) in params.iter().enumerate() {
        if let Some(g) = p.grad().expect("read grad") {
            assert_eq!(
                g.device(),
                device,
                "[{label}] param[{i}] grad landed on {:?} at step {step} \
                 (autograd CPU detour suspected)",
                g.device(),
            );
        }
    }
}

/// Sync gradients from `model_params` into `optimizer`'s cloned parameters.
///
/// The optimizer holds Arc-shared clones of each `Parameter`, so a gradient
/// set on a model param is invisible to the optimizer until copied across.
/// This mirrors the pattern in every test in this file.
fn sync_grads(model_params: &[&Parameter<f32>], opt_params: &[Parameter<f32>]) {
    for (mp, op) in model_params.iter().zip(opt_params.iter()) {
        if let Some(g) = mp.grad().expect("read grad for sync") {
            op.set_grad(Some(g)).expect("set_grad on opt param");
        }
    }
}

/// Manual GPU-resident MSE: `mean((a - b)^2)`.
///
/// Avoids `nn::MSELoss::forward`, which unconditionally rebuilds the loss
/// tensor in CPU storage (loss.rs:79-93) and would defeat the whole point
/// of an end-to-end-on-GPU test. Each call dispatches on the input tensor's
/// device, so the entire chain stays on GPU.
fn mse_gpu(a: &Tensor<f32>, b: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let diff = arithmetic::sub(a, b)?;
    let sq = arithmetic::mul(&diff, &diff)?;
    reduction::mean(&sq)
}

/// Manual GPU-resident negative-log-likelihood for class-index targets.
///
/// Computes `-mean(log(softmax(logits)) * one_hot(targets))`, scaled so the
/// result equals the `CrossEntropyLoss` reduction-mean form. We can't use
/// `nn::CrossEntropyLoss::forward` for the same reason we can't use
/// `MSELoss::forward`: it materializes intermediate tensors via
/// `TensorStorage::cpu(...)` (loss.rs:269+), forcing the loss back to CPU.
///
/// `logits` is `[B, C]` on `device`; `target_indices_f32` is `[B]` on
/// `device`, holding integer class IDs encoded as f32 (matching the CPU
/// tests in `cpu_training.rs`). The `num_classes` argument is the C
/// dimension and is used to build a one-hot matrix on the host before
/// transfer.
fn ce_loss_gpu(
    logits: &Tensor<f32>,
    target_indices_f32: &Tensor<f32>,
    num_classes: usize,
    device: Device,
) -> FerrotorchResult<Tensor<f32>> {
    // Read targets back to CPU once to build the one-hot mask. The
    // backward chain still flows through `logits` (which lives on GPU),
    // so this readback does not break autograd device-residency.
    let tgt_cpu = target_indices_f32.data_vec()?;
    let batch = tgt_cpu.len();
    let mut one_hot = vec![0.0f32; batch * num_classes];
    for (b, &t) in tgt_cpu.iter().enumerate() {
        let idx = t as usize;
        assert!(
            idx < num_classes,
            "ce_loss_gpu: target index {idx} >= num_classes {num_classes}",
        );
        one_hot[b * num_classes + idx] = 1.0;
    }
    let one_hot_cpu = from_vec(one_hot, &[batch, num_classes])?;
    let one_hot_gpu = one_hot_cpu.to(device)?;

    // log_softmax = log(softmax(logits)). softmax + log both have GPU
    // dispatch; we lean on grad_fns to keep autograd intact.
    let probs = activation::softmax(logits)?;
    let log_probs = transcendental::log(&probs)?;
    let masked = arithmetic::mul(&log_probs, &one_hot_gpu)?;

    // sum over classes then mean over batch === mean over (B*C) * C
    // because one_hot has exactly one 1 per row. Use mean(masked) * C
    // to match the standard CE reduction-mean.
    //
    // `scalar::<f32>(...)` constructs CPU-resident tensors; we must move
    // them to `device` before mixing with the GPU log_probs/masked
    // chain or arithmetic::mul errors with `DeviceMismatch`.
    let neg_one = scalar::<f32>(-1.0)?.to(device)?;
    let class_scale = scalar::<f32>(num_classes as f32)?.to(device)?;
    let m = reduction::mean(&masked)?;
    let scaled = arithmetic::mul(&m, &class_scale)?;
    arithmetic::mul(&scaled, &neg_one)
}

// =====================================================================
// Experiment 2: Small Transformer (attention + MLP) on GPU
// =====================================================================
//
// Mirrors `cpu_training.rs::test_transformer_training_cpu`: a single QKV
// projection + scaled-dot-product attention + output projection + 2-layer
// MLP + LayerNorm residual. All weights, activations, and gradients live
// on `Device::Cuda(0)` for the entire training loop.
#[test]
fn gpu_transformer_training_smoke() {
    init_cuda_backend()
        .expect("CUDA backend must initialize for the GPU transformer training test");
    let device = Device::Cuda(0);

    let d_model = 64;
    let n_heads = 4;
    let head_dim = d_model / n_heads;
    let batch = 2;
    let seq = 8;

    let mut qkv_proj = Linear::<f32>::new(d_model, 3 * d_model, false).expect("qkv_proj");
    let mut out_proj = Linear::<f32>::new(d_model, d_model, false).expect("out_proj");
    let mut mlp_up = Linear::<f32>::new(d_model, 4 * d_model, true).expect("mlp_up");
    let mut mlp_down = Linear::<f32>::new(4 * d_model, d_model, true).expect("mlp_down");
    let mut ln = LayerNorm::<f32>::new(vec![d_model], 1e-5, true).expect("LayerNorm");

    for m in [
        &mut qkv_proj as &mut dyn Module<f32>,
        &mut out_proj,
        &mut mlp_up,
        &mut mlp_down,
        &mut ln,
    ] {
        m.to_device(device).expect("module.to_device(Cuda(0))");
    }

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(qkv_proj.parameters().into_iter().cloned());
    all_params.extend(out_proj.parameters().into_iter().cloned());
    all_params.extend(mlp_up.parameters().into_iter().cloned());
    all_params.extend(mlp_down.parameters().into_iter().cloned());
    all_params.extend(ln.parameters().into_iter().cloned());

    let mut adamw_cfg = AdamWConfig::default();
    adamw_cfg.lr = 1e-3;
    let mut optimizer = AdamW::new(all_params, adamw_cfg);

    // Fix the batch across steps so loss is guaranteed to decrease for a
    // sane optimizer (see cpu_training.rs comment of the same shape).
    let x = randn::<f32>(&[batch, seq, d_model])
        .expect("x")
        .to(device)
        .expect("x.to(Cuda)");
    let target = randn::<f32>(&[batch, seq, d_model])
        .expect("target")
        .to(device)
        .expect("target.to(Cuda)");

    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let n_steps = 10usize;

    for step in 0..n_steps {
        optimizer.zero_grad().expect("zero_grad");

        let qkv = qkv_proj.forward(&x).expect("qkv_proj.forward");
        let qkv_chunks = chunk_t(&qkv, 3, 2).expect("chunk_t");
        let q = qkv_chunks[0]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .expect("q.view");
        let k = qkv_chunks[1]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .expect("k.view");
        let v = qkv_chunks[2]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .expect("v.view");

        let k_t = permute_t(&k, &[0, 2, 1]).expect("permute_t(k)");
        let scores = linalg::bmm_differentiable(&q, &k_t).expect("bmm Q@K^T");
        let scale = scalar::<f32>((head_dim as f32).sqrt())
            .expect("scalar")
            .to(device)
            .expect("scale.to(Cuda)");
        let scores = arithmetic::div(&scores, &scale).expect("scores/scale");
        let attn = activation::softmax(&scores).expect("softmax on GPU");
        let attn_out = linalg::bmm_differentiable(&attn, &v).expect("bmm attn@V");

        let attn_out = attn_out
            .view(&[batch as i64, seq as i64, d_model as i64])
            .expect("attn_out.view");
        let attn_out = out_proj.forward(&attn_out).expect("out_proj.forward");

        let h = arithmetic::add(&x, &attn_out).expect("residual add");
        let h = ln.forward(&h).expect("LayerNorm.forward");
        let up = mlp_up.forward(&h).expect("mlp_up");
        let activated = up.relu().expect("relu");
        let down = mlp_down.forward(&activated).expect("mlp_down");
        let output = arithmetic::add(&h, &down).expect("mlp residual add");

        assert_eq!(output.device(), device, "transformer output off-device");

        let loss = mse_gpu(&output, &target).expect("MSE loss");
        assert_eq!(loss.device(), device, "loss off-device");

        let loss_val = loss
            .data_vec()
            .expect("loss readback")
            .first()
            .copied()
            .expect("scalar loss");
        assert!(loss_val.is_finite(), "non-finite loss at step {step}");
        if first_loss.is_none() {
            first_loss = Some(loss_val);
        }
        last_loss = loss_val;

        loss.backward().expect("backward");

        let mut model_params: Vec<&Parameter<f32>> = Vec::new();
        model_params.extend(qkv_proj.parameters());
        model_params.extend(out_proj.parameters());
        model_params.extend(mlp_up.parameters());
        model_params.extend(mlp_down.parameters());
        model_params.extend(ln.parameters());

        assert_grads_on_device("transformer", &model_params, device, step);
        sync_grads(&model_params, optimizer.param_groups()[0].params());

        optimizer.step().expect("optimizer.step");

        assert_all_on_device("transformer", &model_params, device, step);
    }

    let initial = first_loss.expect("at least one step ran");
    eprintln!(
        "gpu_transformer_training_smoke: initial loss = {initial:.6}, final loss = {last_loss:.6}"
    );
    assert!(
        last_loss < 0.9 * initial,
        "Transformer GPU: loss should decrease meaningfully. \
         initial={initial:.6}, final={last_loss:.6}, threshold={:.6}",
        0.9 * initial,
    );
}

// =====================================================================
// Experiment 3: CNN on synthetic CIFAR-like data on GPU
// =====================================================================
//
// Mirrors `cpu_training.rs::test_cnn_training_cpu`: Conv2d -> ReLU ->
// AvgPool2d -> Linear, with manual NLL loss on integer-class targets.
#[test]
fn gpu_cnn_training_smoke() {
    init_cuda_backend().expect("CUDA backend must initialize for the GPU CNN training test");
    let device = Device::Cuda(0);

    let mut conv = Conv2d::<f32>::new(3, 8, (3, 3), (1, 1), (1, 1), true).expect("Conv2d");
    let pool = AvgPool2d::new([4, 4], [4, 4], [0, 0]);
    let mut linear = Linear::<f32>::new(8 * 8 * 8, 10, true).expect("Linear");

    conv.to_device(device).expect("conv.to_device");
    linear.to_device(device).expect("linear.to_device");

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(conv.parameters().into_iter().cloned());
    all_params.extend(linear.parameters().into_iter().cloned());

    let mut optimizer = Adam::new(all_params, AdamConfig::default());

    // Fix input/target across steps for a reliable signal.
    let input = randn::<f32>(&[4, 3, 32, 32])
        .expect("input")
        .to(device)
        .expect("input.to(Cuda)");
    let target = from_vec((0..4).map(|i| (i % 10) as f32).collect(), &[4])
        .expect("target")
        .to(device)
        .expect("target.to(Cuda)");

    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let n_steps = 10usize;

    for step in 0..n_steps {
        optimizer.zero_grad().expect("zero_grad");

        let h = conv.forward(&input).expect("conv");
        let h = h.relu().expect("relu");
        let h = pool.forward(&h).expect("pool");
        let h = h.view(&[4, -1]).expect("flatten");
        let output = linear.forward(&h).expect("linear");

        assert_eq!(output.device(), device, "output off-device");

        let loss = ce_loss_gpu(&output, &target, 10, device).expect("CE loss");
        assert_eq!(loss.device(), device, "loss off-device");

        let loss_val = loss
            .data_vec()
            .expect("loss readback")
            .first()
            .copied()
            .expect("scalar loss");
        assert!(loss_val.is_finite(), "non-finite loss at step {step}");
        if first_loss.is_none() {
            first_loss = Some(loss_val);
        }
        last_loss = loss_val;

        loss.backward().expect("backward");

        let mut model_params: Vec<&Parameter<f32>> = Vec::new();
        model_params.extend(conv.parameters());
        model_params.extend(linear.parameters());
        assert_grads_on_device("cnn", &model_params, device, step);
        sync_grads(&model_params, optimizer.param_groups()[0].params());
        optimizer.step().expect("optimizer.step");
        assert_all_on_device("cnn", &model_params, device, step);
    }

    let initial = first_loss.expect("at least one step ran");
    eprintln!("gpu_cnn_training_smoke: initial loss = {initial:.6}, final loss = {last_loss:.6}");
    assert!(
        last_loss < 0.9 * initial,
        "CNN GPU: loss should decrease meaningfully. \
         initial={initial:.6}, final={last_loss:.6}, threshold={:.6}",
        0.9 * initial,
    );
}

// =====================================================================
// Experiment 4: LSTM end-to-end training on GPU
// =====================================================================
//
// Mirrors `gpu_transformer_training_smoke` structurally: every parameter,
// activation, and gradient is on `Device::Cuda(0)` for the entire loop.
//
// Closes the sentinel that previously documented `forward_with_state`
// returning `GpuTensorNotAccessible` on CUDA-resident weights. The fix
// (#750) replaced the host-detour `data_vec()` / `TensorStorage::cpu`
// path inside `ferrotorch-nn::rnn` with device-aware ops (`narrow`,
// `squeeze_t`, `chunk`, `expand`, `cat`), so the gate math now runs on
// whichever device the inputs and weights live on.
//
// What this test enforces:
//   1. Loss decreases meaningfully (final < 0.9 × initial).
//   2. Gradients land on Cuda(0) after `backward()`.
//   3. Parameters stay on Cuda(0) after `optimizer.step()`.
#[test]
fn gpu_lstm_training_smoke() {
    init_cuda_backend().expect("CUDA backend must initialize for the GPU LSTM training test");
    let device = Device::Cuda(0);

    let embed_dim = 32;
    let hidden_size = 32;
    let batch = 2;
    let seq = 8;

    let mut lstm = LSTM::<f32>::new(embed_dim, hidden_size, 1).expect("LSTM");
    let mut proj = Linear::<f32>::new(hidden_size, embed_dim, true).expect("proj");

    for m in [&mut lstm as &mut dyn Module<f32>, &mut proj] {
        m.to_device(device).expect("module.to_device(Cuda(0))");
    }

    // Sanity: every parameter is device-resident before training begins.
    for p in lstm.parameters() {
        assert_eq!(
            p.tensor().device(),
            device,
            "LSTM parameter not on Cuda(0) after to_device",
        );
    }
    for p in proj.parameters() {
        assert_eq!(
            p.tensor().device(),
            device,
            "Linear parameter not on Cuda(0) after to_device",
        );
    }

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(lstm.parameters().into_iter().cloned());
    all_params.extend(proj.parameters().into_iter().cloned());

    let mut adamw_cfg = AdamWConfig::default();
    adamw_cfg.lr = 1e-2;
    let mut optimizer = AdamW::new(all_params, adamw_cfg);

    // Fix the batch across steps so loss is guaranteed to decrease for a
    // sane optimizer (mirrors gpu_transformer_training_smoke).
    let x = randn::<f32>(&[batch, seq, embed_dim])
        .expect("x")
        .to(device)
        .expect("x.to(Cuda)");
    let target = randn::<f32>(&[batch, seq, embed_dim])
        .expect("target")
        .to(device)
        .expect("target.to(Cuda)");

    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let n_steps = 10usize;

    for step in 0..n_steps {
        optimizer.zero_grad().expect("zero_grad");

        let (lstm_out, _state) = lstm
            .forward_with_state(&x, None)
            .expect("LSTM forward on Cuda");
        assert_eq!(lstm_out.device(), device, "LSTM output off-device");

        let output = proj.forward(&lstm_out).expect("proj.forward");
        assert_eq!(output.device(), device, "Linear output off-device");

        let loss = mse_gpu(&output, &target).expect("MSE loss");
        assert_eq!(loss.device(), device, "loss off-device");

        let loss_val = loss
            .data_vec()
            .expect("loss readback")
            .first()
            .copied()
            .expect("scalar loss");
        assert!(loss_val.is_finite(), "non-finite loss at step {step}");
        if first_loss.is_none() {
            first_loss = Some(loss_val);
        }
        last_loss = loss_val;

        loss.backward().expect("backward");

        let mut model_params: Vec<&Parameter<f32>> = Vec::new();
        model_params.extend(lstm.parameters());
        model_params.extend(proj.parameters());

        assert_grads_on_device("lstm", &model_params, device, step);
        sync_grads(&model_params, optimizer.param_groups()[0].params());

        optimizer.step().expect("optimizer.step");

        assert_all_on_device("lstm", &model_params, device, step);
    }

    let initial = first_loss.expect("at least one step ran");
    eprintln!("gpu_lstm_training_smoke: initial loss = {initial:.6}, final loss = {last_loss:.6}");
    assert!(
        last_loss < 0.9 * initial,
        "LSTM GPU: loss should decrease meaningfully. \
         initial={initial:.6}, final={last_loss:.6}, threshold={:.6}",
        0.9 * initial,
    );
}

// =====================================================================
// Experiment 5: VAE with reparameterization on GPU
// =====================================================================
//
// Mirrors `cpu_training.rs::test_vae_training_cpu`: encoder -> mu/logvar
// -> reparameterize -> decoder -> sigmoid -> reconstruction MSE + KL.
#[test]
fn gpu_vae_training_smoke() {
    init_cuda_backend().expect("CUDA backend must initialize for the GPU VAE training test");
    let device = Device::Cuda(0);

    let input_dim = 28 * 28;
    let hidden_dim = 128;
    let latent_dim = 16;

    let mut enc_fc = Linear::<f32>::new(input_dim, hidden_dim, true).expect("enc_fc");
    let mut enc_mu = Linear::<f32>::new(hidden_dim, latent_dim, true).expect("enc_mu");
    let mut enc_logvar = Linear::<f32>::new(hidden_dim, latent_dim, true).expect("enc_logvar");
    let mut dec_fc1 = Linear::<f32>::new(latent_dim, hidden_dim, true).expect("dec_fc1");
    let mut dec_fc2 = Linear::<f32>::new(hidden_dim, input_dim, true).expect("dec_fc2");

    for m in [
        &mut enc_fc as &mut dyn Module<f32>,
        &mut enc_mu,
        &mut enc_logvar,
        &mut dec_fc1,
        &mut dec_fc2,
    ] {
        m.to_device(device).expect("module.to_device(Cuda)");
    }

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(enc_fc.parameters().into_iter().cloned());
    all_params.extend(enc_mu.parameters().into_iter().cloned());
    all_params.extend(enc_logvar.parameters().into_iter().cloned());
    all_params.extend(dec_fc1.parameters().into_iter().cloned());
    all_params.extend(dec_fc2.parameters().into_iter().cloned());

    let mut adam_cfg = AdamConfig::default();
    adam_cfg.lr = 1e-3;
    let mut optimizer = Adam::new(all_params, adam_cfg);

    // Fix the input/eps across steps so the loss signal is monotone for a
    // sane optimizer; matches the gpu_mlp_training_smoke pattern.
    let input = rand::<f32>(&[8, input_dim])
        .expect("input")
        .to(device)
        .expect("input.to(Cuda)");
    let eps = randn::<f32>(&[8, latent_dim])
        .expect("eps")
        .to(device)
        .expect("eps.to(Cuda)");

    // Constants that need to live on-device for arithmetic dispatch to
    // stay GPU-resident (otherwise mixing CPU+GPU operands errors out).
    let half = scalar::<f32>(0.5)
        .expect("half")
        .to(device)
        .expect("half.to");
    let one = scalar::<f32>(1.0).expect("one").to(device).expect("one.to");
    let neg_half = scalar::<f32>(-0.5)
        .expect("neg_half")
        .to(device)
        .expect("neg_half.to");

    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let n_steps = 10usize;

    for step in 0..n_steps {
        optimizer.zero_grad().expect("zero_grad");

        // Encode
        let h = enc_fc
            .forward(&input)
            .expect("enc_fc")
            .relu()
            .expect("relu");
        let mu = enc_mu.forward(&h).expect("enc_mu");
        let logvar = enc_logvar.forward(&h).expect("enc_logvar");

        // Reparameterize: z = mu + eps * exp(0.5 * logvar)
        let half_logvar = arithmetic::mul(&logvar, &half).expect("half * logvar");
        let std = transcendental::exp(&half_logvar).expect("exp on GPU");
        let z = arithmetic::add(&mu, &arithmetic::mul(&eps, &std).expect("eps * std"))
            .expect("mu + eps*std");

        // Decode
        let h_dec = dec_fc1.forward(&z).expect("dec_fc1").relu().expect("relu");
        let recon = dec_fc2.forward(&h_dec).expect("dec_fc2");
        let recon = activation::sigmoid(&recon).expect("sigmoid on GPU");
        assert_eq!(recon.device(), device, "recon off-device");

        let recon_loss = mse_gpu(&recon, &input).expect("recon MSE");

        // KL divergence: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        let mu_sq = arithmetic::mul(&mu, &mu).expect("mu^2");
        let exp_logvar = transcendental::exp(&logvar).expect("exp(logvar)");
        let kl_inner = arithmetic::sub(
            &arithmetic::sub(&arithmetic::add(&one, &logvar).expect("1 + logvar"), &mu_sq)
                .expect("(1+logvar) - mu^2"),
            &exp_logvar,
        )
        .expect("(1+logvar - mu^2) - exp(logvar)");
        let kl_loss = arithmetic::mul(&reduction::mean(&kl_inner).expect("KL mean"), &neg_half)
            .expect("-0.5 * mean(...)");

        let loss = arithmetic::add(&recon_loss, &kl_loss).expect("recon + KL");
        assert_eq!(loss.device(), device, "loss off-device");

        let loss_val = loss
            .data_vec()
            .expect("loss readback")
            .first()
            .copied()
            .expect("scalar loss");
        assert!(loss_val.is_finite(), "non-finite loss at step {step}");
        if first_loss.is_none() {
            first_loss = Some(loss_val);
        }
        last_loss = loss_val;

        loss.backward().expect("backward");

        let mut model_params: Vec<&Parameter<f32>> = Vec::new();
        model_params.extend(enc_fc.parameters());
        model_params.extend(enc_mu.parameters());
        model_params.extend(enc_logvar.parameters());
        model_params.extend(dec_fc1.parameters());
        model_params.extend(dec_fc2.parameters());

        assert_grads_on_device("vae", &model_params, device, step);
        sync_grads(&model_params, optimizer.param_groups()[0].params());
        optimizer.step().expect("optimizer.step");
        assert_all_on_device("vae", &model_params, device, step);
    }

    let initial = first_loss.expect("at least one step ran");
    eprintln!("gpu_vae_training_smoke: initial loss = {initial:.6}, final loss = {last_loss:.6}");
    assert!(
        last_loss < 0.9 * initial,
        "VAE GPU: loss should decrease meaningfully. \
         initial={initial:.6}, final={last_loss:.6}, threshold={:.6}",
        0.9 * initial,
    );
}

// =====================================================================
// Experiment 6: Pythia-70M architecture (GPT-NeoX) on GPU
// =====================================================================
//
// Mirrors `cpu_training.rs::test_pythia_architecture_cpu`. Reduced
// dimensions (2 layers, 4 heads, d=64, vocab=256, seq=16, batch=2) match
// the CPU test exactly to keep VRAM footprint small — the spec
// explicitly calls out using the same smaller config.

/// GPT-NeoX block with parallel Attn + MLP, single LayerNorm, RoPE on Q/K.
struct NeoXBlockGpu {
    ln: LayerNorm<f32>,
    qkv: Linear<f32>,
    out_proj: Linear<f32>,
    mlp_up: Linear<f32>,
    mlp_down: Linear<f32>,
    rope: RotaryPositionEmbedding<f32>,
    n_heads: usize,
    head_dim: usize,
}

impl NeoXBlockGpu {
    fn new(
        d_model: usize,
        n_heads: usize,
        max_seq: usize,
        device: Device,
    ) -> FerrotorchResult<Self> {
        let head_dim = d_model / n_heads;
        let mut ln = LayerNorm::<f32>::new(vec![d_model], 1e-5, true)?;
        let mut qkv = Linear::<f32>::new(d_model, 3 * d_model, true)?;
        let mut out_proj = Linear::<f32>::new(d_model, d_model, true)?;
        let mut mlp_up = Linear::<f32>::new(d_model, 4 * d_model, true)?;
        let mut mlp_down = Linear::<f32>::new(4 * d_model, d_model, true)?;
        // RoPE is not a `Module` and has no `to_device`. Its `apply()`
        // does an internal host readback (transformer.rs:533-534, plus a
        // closing `to(device)` at line 615-619), so it functions on a
        // GPU input but bounces through host RAM each call. This is a
        // real `rust-gpu-discipline` §3 detour and should be replaced
        // by a sin/cos/rotation kernel — tracked separately because it
        // is a multi-kernel design, exceeding this dispatch's scope.
        // See the Section B report for the Step 4 escalation.
        let rope = RotaryPositionEmbedding::<f32>::new(head_dim, max_seq, 10000.0)?;
        ln.to_device(device)?;
        qkv.to_device(device)?;
        out_proj.to_device(device)?;
        mlp_up.to_device(device)?;
        mlp_down.to_device(device)?;
        Ok(Self {
            ln,
            qkv,
            out_proj,
            mlp_up,
            mlp_down,
            rope,
            n_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor<f32>, mask: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let d_model = x.shape()[2];

        let normed = self.ln.forward(x)?;

        // Attention path
        let qkv = self.qkv.forward(&normed)?;
        let qkv_chunks = chunk_t(&qkv, 3, 2)?;
        let q = qkv_chunks[0].view(&[
            batch as i64 * self.n_heads as i64,
            seq as i64,
            self.head_dim as i64,
        ])?;
        let k = qkv_chunks[1].view(&[
            batch as i64 * self.n_heads as i64,
            seq as i64,
            self.head_dim as i64,
        ])?;
        let v = qkv_chunks[2].view(&[
            batch as i64 * self.n_heads as i64,
            seq as i64,
            self.head_dim as i64,
        ])?;

        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;

        let k_t = permute_t(&k, &[0, 2, 1])?;
        let scores = linalg::bmm_differentiable(&q, &k_t)?;
        let scale = scalar::<f32>((self.head_dim as f32).sqrt())?.to(x.device())?;
        let scores = arithmetic::div(&scores, &scale)?;
        let scores = arithmetic::add(&scores, mask)?;

        let attn = activation::softmax(&scores)?;
        let attn_out = linalg::bmm_differentiable(&attn, &v)?;
        let attn_out = attn_out.view(&[batch as i64, seq as i64, d_model as i64])?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // Parallel MLP path (uses the same `normed`)
        let mlp_h = self.mlp_up.forward(&normed)?;
        let mlp_h = mlp_h.relu()?;
        let mlp_out = self.mlp_down.forward(&mlp_h)?;

        // Parallel residual: x + attn + mlp
        let h = arithmetic::add(x, &attn_out)?;
        arithmetic::add(&h, &mlp_out)
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        let mut p = Vec::new();
        p.extend(self.ln.parameters());
        p.extend(self.qkv.parameters());
        p.extend(self.out_proj.parameters());
        p.extend(self.mlp_up.parameters());
        p.extend(self.mlp_down.parameters());
        p
    }
}

#[test]
fn gpu_pythia_training_smoke() {
    init_cuda_backend().expect("CUDA backend must initialize for the GPU Pythia training test");
    let device = Device::Cuda(0);

    let d_model = 64;
    let n_heads = 4;
    let n_layers = 2;
    let vocab_size = 256;
    let max_seq = 32;
    let batch = 2;
    let seq = 16;

    let mut token_emb = Embedding::<f32>::new(vocab_size, d_model, None).expect("Embedding");
    token_emb
        .to_device(device)
        .expect("token_emb.to_device(Cuda)");

    let blocks: Vec<NeoXBlockGpu> = (0..n_layers)
        .map(|_| NeoXBlockGpu::new(d_model, n_heads, max_seq, device).expect("NeoXBlockGpu"))
        .collect();

    let mut final_ln = LayerNorm::<f32>::new(vec![d_model], 1e-5, true).expect("final_ln");
    final_ln
        .to_device(device)
        .expect("final_ln.to_device(Cuda)");
    let mut lm_head = Linear::<f32>::new(d_model, vocab_size, false).expect("lm_head");
    lm_head.to_device(device).expect("lm_head.to_device(Cuda)");

    // Collect all parameters
    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(token_emb.parameters().into_iter().cloned());
    for block in &blocks {
        all_params.extend(block.parameters().into_iter().cloned());
    }
    all_params.extend(final_ln.parameters().into_iter().cloned());
    all_params.extend(lm_head.parameters().into_iter().cloned());

    let param_count: usize = all_params.iter().map(|p| p.tensor().numel()).sum();
    eprintln!("gpu_pythia_training_smoke: parameter count = {param_count}");

    let mut adamw_cfg = AdamWConfig::default();
    adamw_cfg.lr = 1e-3;
    adamw_cfg.betas = (0.9, 0.95);
    adamw_cfg.weight_decay = 0.01;
    let mut optimizer = AdamW::new(all_params, adamw_cfg);

    // Build the causal mask on host once and ship to GPU.
    let mask_data: Vec<f32> = (0..seq * seq)
        .map(|i| {
            let row = i / seq;
            let col = i % seq;
            if col <= row { 0.0 } else { -1e9 }
        })
        .collect();
    let mask = from_vec(mask_data, &[1, seq, seq])
        .expect("mask")
        .to(device)
        .expect("mask.to(Cuda)");

    // Fix tokens/targets across steps.
    let token_ids: Vec<f32> = (0..batch * seq).map(|i| (i % vocab_size) as f32).collect();
    let target_ids: Vec<f32> = (0..batch * seq)
        .map(|i| ((i + 1) % vocab_size) as f32)
        .collect();
    let token_tensor = from_vec(token_ids, &[batch * seq])
        .expect("token tensor")
        .to(device)
        .expect("tokens.to(Cuda)");
    let targets = from_vec(target_ids, &[batch * seq])
        .expect("targets")
        .to(device)
        .expect("targets.to(Cuda)");

    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let n_steps = 10usize;

    for step in 0..n_steps {
        optimizer.zero_grad().expect("zero_grad");

        // Single batched embedding lookup, then reshape to [B, S, D].
        let embedded_flat = token_emb.forward(&token_tensor).expect("embedding");
        let mut h = embedded_flat
            .view(&[batch as i64, seq as i64, d_model as i64])
            .expect("embedded.view");
        assert_eq!(h.device(), device, "embedded off-device");

        for block in &blocks {
            h = block.forward(&h, &mask).expect("block.forward");
        }

        let h = final_ln.forward(&h).expect("final_ln");
        let logits = lm_head.forward(&h).expect("lm_head");
        let logits_flat = logits
            .view(&[(batch * seq) as i64, vocab_size as i64])
            .expect("logits.view");

        let loss = ce_loss_gpu(&logits_flat, &targets, vocab_size, device).expect("CE loss");
        assert_eq!(loss.device(), device, "loss off-device");

        let loss_val = loss
            .data_vec()
            .expect("loss readback")
            .first()
            .copied()
            .expect("scalar loss");
        assert!(loss_val.is_finite(), "non-finite loss at step {step}");
        if first_loss.is_none() {
            first_loss = Some(loss_val);
            eprintln!("gpu_pythia_training_smoke: step 0 loss = {loss_val:.4}");
        }
        last_loss = loss_val;

        loss.backward().expect("backward");

        let mut model_params: Vec<&Parameter<f32>> = Vec::new();
        model_params.extend(token_emb.parameters());
        for block in &blocks {
            model_params.extend(block.parameters());
        }
        model_params.extend(final_ln.parameters());
        model_params.extend(lm_head.parameters());

        assert_grads_on_device("pythia", &model_params, device, step);
        sync_grads(&model_params, optimizer.param_groups()[0].params());
        optimizer.step().expect("optimizer.step");
        assert_all_on_device("pythia", &model_params, device, step);
    }

    let initial = first_loss.expect("at least one step ran");
    eprintln!(
        "gpu_pythia_training_smoke: initial loss = {initial:.4}, final loss = {last_loss:.4}"
    );
    assert!(
        last_loss < 0.9 * initial,
        "Pythia GPU: loss should decrease meaningfully. \
         initial={initial:.4}, final={last_loss:.4}, threshold={:.4}",
        0.9 * initial,
    );
}
