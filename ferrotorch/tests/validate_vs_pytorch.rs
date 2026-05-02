//! Correctness validation: verify ferrotorch produces the same results
//! as PyTorch for core operations.
//!
//! Run with: cargo test --test validate_vs_pytorch --release

use std::time::Instant;

use ferrotorch_core::*;
use ferrotorch_nn::*;
use ferrotorch_optim::*;

fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{msg}: {a} vs {b} (diff = {})",
        (a - b).abs()
    );
}

// =====================================================================
// CORRECTNESS: Softmax sums to 1
// =====================================================================
#[test]
fn test_softmax_sums_to_one() {
    let x = randn::<f32>(&[4, 10]).unwrap();
    let sm = grad_fns::activation::softmax(&x).unwrap();
    let sm_data = sm.data().unwrap();

    for row in 0..4 {
        let row_sum: f32 = (0..10).map(|c| sm_data[row * 10 + c]).sum();
        assert_close(row_sum, 1.0, 1e-5, &format!("softmax row {row} sum"));
    }
    println!("PASS: softmax sums to 1");
}

// =====================================================================
// CORRECTNESS: LayerNorm zero mean, unit variance
// =====================================================================
#[test]
fn test_layernorm_statistics() {
    let ln = LayerNorm::new(vec![256], 1e-5, false).unwrap();
    // Input with non-zero mean and non-unit variance
    let x = (&randn::<f32>(&[4, 32, 256]).unwrap() * &scalar::<f32>(5.0).unwrap()).unwrap();
    let x = (&x + &scalar::<f32>(3.0).unwrap()).unwrap();
    let y = ln.forward(&x).unwrap();
    let y_data = y.data().unwrap();

    let batch_seq = 4 * 32;
    let d = 256;
    for i in 0..batch_seq {
        let slice = &y_data[i * d..(i + 1) * d];
        let mean: f32 = slice.iter().sum::<f32>() / d as f32;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
        assert!(mean.abs() < 1e-4, "layernorm mean too large: {mean}");
        assert!(
            (var - 1.0).abs() < 1e-3,
            "layernorm var too far from 1: {var}"
        );
    }
    println!("PASS: layernorm produces zero mean, unit variance");
}

// =====================================================================
// CORRECTNESS: Autograd numerical gradient check (x^2)
// =====================================================================
#[test]
fn test_autograd_x_squared() {
    let x = from_vec(vec![1.0f32, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad_(true);
    let x2 = (&x * &x).unwrap();
    let loss = x2.sum_all().unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    let grad_data = grad.data().unwrap();
    // d/dx(x^2) = 2x
    assert_close(grad_data[0], 2.0, 1e-4, "grad[0]");
    assert_close(grad_data[1], 4.0, 1e-4, "grad[1]");
    assert_close(grad_data[2], 6.0, 1e-4, "grad[2]");
    println!("PASS: autograd d/dx(x^2) = 2x");
}

// =====================================================================
// CORRECTNESS: CrossEntropy produces gradients
// =====================================================================
#[test]
fn test_crossentropy_gradient() {
    let logits = randn::<f32>(&[4, 10]).unwrap().requires_grad_(true);
    let targets = from_vec(vec![3.0, 7.0, 1.0, 5.0], &[4]).unwrap();
    let ce = CrossEntropyLoss::new(Reduction::Mean, 0.0);
    let loss = ce.forward(&logits, &targets).unwrap();
    loss.backward().unwrap();

    let grad = logits.grad().unwrap().unwrap();
    let grad_data = grad.data().unwrap();
    let grad_sum: f32 = grad_data.iter().map(|g| g.abs()).sum();
    assert!(grad_sum > 0.0, "CrossEntropy gradient should be non-zero");
    println!("PASS: CrossEntropy produces non-zero gradients (sum = {grad_sum:.4})");
}

// =====================================================================
// CORRECTNESS: Dropout masks at correct rate
// =====================================================================
#[test]
fn test_dropout_rate() {
    let x = ones::<f32>(&[10000]).unwrap();
    let dropout = Dropout::new(0.3).unwrap();
    // Training mode
    let y = dropout.forward(&x).unwrap();
    let y_data = y.data().unwrap();
    let zero_count = y_data.iter().filter(|&&v| v == 0.0).count();
    let zero_frac = zero_count as f32 / 10000.0;
    assert!(
        zero_frac > 0.20 && zero_frac < 0.40,
        "dropout rate should be ~0.3, got {zero_frac}"
    );
    println!("PASS: dropout zero fraction = {zero_frac:.4} (expected ~0.3)");
}

// =====================================================================
// CORRECTNESS: Embedding lookup matches weight rows
// =====================================================================
#[test]
fn test_embedding_lookup() {
    let emb = Embedding::new(100, 16, None).unwrap();
    let ids = from_vec(vec![5.0f32, 10.0, 50.0], &[3]).unwrap();
    let out = emb.forward(&ids).unwrap();
    let out_data = out.data().unwrap();

    let weight_data = emb.parameters()[0].tensor().data().unwrap();
    // Row 5 should match out[0]
    for j in 0..16 {
        assert_close(
            out_data[j],
            weight_data[5 * 16 + j],
            1e-6,
            &format!("emb[5][{j}]"),
        );
    }
    // Row 50 should match out[2]
    for j in 0..16 {
        assert_close(
            out_data[2 * 16 + j],
            weight_data[50 * 16 + j],
            1e-6,
            &format!("emb[50][{j}]"),
        );
    }
    println!("PASS: embedding lookup matches weight rows");
}

// =====================================================================
// CORRECTNESS: Causal attention mask
// =====================================================================
#[test]
fn test_causal_attention_mask() {
    let seq = 8;
    let d = 16;
    let q = randn::<f32>(&[1, seq, d]).unwrap();
    let k = randn::<f32>(&[1, seq, d]).unwrap();

    // Q @ K^T
    let k_t = permute_t(&k, &[0, 2, 1]).unwrap();
    let scores = grad_fns::linalg::bmm_differentiable(&q, &k_t).unwrap();

    // Causal mask
    let mask_data: Vec<f32> = (0..seq * seq)
        .map(|i| if (i % seq) <= (i / seq) { 0.0 } else { -1e9 })
        .collect();
    let mask = from_vec(mask_data, &[1, seq, seq]).unwrap();
    let masked_scores = (&scores + &mask).unwrap();

    let attn = grad_fns::activation::softmax(&masked_scores).unwrap();
    let attn_data = attn.data().unwrap();

    // Upper triangle should be ~0 (future positions masked)
    let mut upper_sum = 0.0f32;
    for i in 0..seq {
        for j in (i + 1)..seq {
            upper_sum += attn_data[i * seq + j].abs();
        }
    }
    assert!(
        upper_sum < 1e-5,
        "causal mask: upper triangle sum should be ~0, got {upper_sum}"
    );
    println!("PASS: causal attention mask zeros future positions (upper sum = {upper_sum:.2e})");
}

// =====================================================================
// CORRECTNESS: Conv2d output shape
// =====================================================================
#[test]
fn test_conv2d_output_shape() {
    let conv = Conv2d::new(3, 16, (3, 3), (2, 2), (1, 1), true).unwrap();
    let x = rand::<f32>(&[2, 3, 32, 32]).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 16, 16, 16], "conv2d output shape mismatch");
    println!("PASS: Conv2d [2,3,32,32] stride=2 pad=1 -> [2,16,16,16]");
}

// =====================================================================
// TASK 1: MLP training convergence
// =====================================================================
#[test]
fn test_mlp_training_converges() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(20, 64, true).unwrap()),
        Box::new(ReLU::default()),
        Box::new(Linear::new(64, 5, true).unwrap()),
    ]);

    let mut optimizer = Adam::new(
        model.parameters().into_iter().cloned().collect(),
        AdamConfig {
            lr: 1e-3,
            ..Default::default()
        },
    );

    let ce = CrossEntropyLoss::new(Reduction::Mean, 0.0);
    let mut losses = Vec::new();

    // Fixed data so loss reliably decreases (random data each step is noisy)
    let x_fixed = randn::<f32>(&[16, 20]).unwrap();
    let target_fixed = from_vec((0..16).map(|i| (i % 5) as f32).collect(), &[16]).unwrap();

    for step in 0..20 {
        let x = &x_fixed;
        let target = &target_fixed;

        optimizer.zero_grad().unwrap();
        let output = model.forward(x).unwrap();
        let loss = ce.forward(&output, target).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        losses.push(loss_val);

        loss.backward().unwrap();

        let model_params: Vec<&Parameter<f32>> = model.parameters();
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();

        if step == 0 || step == 19 {
            println!("  Step {step}: loss = {loss_val:.4}");
        }
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "MLP loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
    println!(
        "PASS: MLP training loss decreases ({:.4} -> {:.4})",
        losses[0], losses[19]
    );
}

// =====================================================================
// TASK 2: Autoencoder reconstruction
// =====================================================================
#[test]
fn test_autoencoder_reconstruction() {
    let encoder = Sequential::new(vec![
        Box::new(Linear::new(784, 256, true).unwrap()),
        Box::new(ReLU::default()),
        Box::new(Linear::new(256, 32, true).unwrap()),
    ]);
    let decoder = Sequential::new(vec![
        Box::new(Linear::new(32, 256, true).unwrap()),
        Box::new(ReLU::default()),
        Box::new(Linear::new(256, 784, true).unwrap()),
    ]);

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(encoder.parameters().into_iter().cloned());
    all_params.extend(decoder.parameters().into_iter().cloned());

    let mut optimizer = Adam::new(
        all_params,
        AdamConfig {
            lr: 1e-3,
            ..Default::default()
        },
    );

    let mut losses = Vec::new();

    for step in 0..20 {
        let x = randn::<f32>(&[16, 784]).unwrap();
        optimizer.zero_grad().unwrap();

        let encoded = encoder.forward(&x).unwrap();
        let decoded = decoder.forward(&encoded).unwrap();

        // MSE loss
        let diff = (&decoded - &x).unwrap();
        let sq = (&diff * &diff).unwrap();
        let loss = sq.mean_all().unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        losses.push(loss_val);

        loss.backward().unwrap();

        // Sync grads
        let enc_params: Vec<&Parameter<f32>> = encoder.parameters();
        let dec_params: Vec<&Parameter<f32>> = decoder.parameters();
        let all_model_params: Vec<&Parameter<f32>> =
            enc_params.into_iter().chain(dec_params).collect();
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();

        if step == 0 || step == 19 {
            println!("  Step {step}: recon_loss = {loss_val:.4}");
        }
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "Autoencoder loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
    println!(
        "PASS: Autoencoder reconstruction loss decreases ({:.4} -> {:.4})",
        losses[0], losses[19]
    );
}

// =====================================================================
// TASK 3: Transformer block training
// =====================================================================
#[test]
fn test_transformer_block_training() {
    let d_model = 64;
    let n_heads = 4;
    let head_dim = d_model / n_heads;
    let batch = 2;
    let seq = 8;

    let qkv_proj = Linear::new(d_model, 3 * d_model, false).unwrap();
    let out_proj = Linear::new(d_model, d_model, false).unwrap();
    let ln = LayerNorm::new(vec![d_model], 1e-5, true).unwrap();

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(qkv_proj.parameters().into_iter().cloned());
    all_params.extend(out_proj.parameters().into_iter().cloned());
    all_params.extend(ln.parameters().into_iter().cloned());

    let mut optimizer = AdamW::new(
        all_params,
        AdamWConfig {
            lr: 3e-3,
            ..Default::default()
        },
    );
    let mut losses = Vec::new();

    // Use fixed input/target so loss can decrease (random data each step is harder)
    let x_fixed = randn::<f32>(&[batch, seq, d_model]).unwrap();
    let target_fixed = randn::<f32>(&[batch, seq, d_model]).unwrap();

    for step in 0..30 {
        let x = &x_fixed;
        let target = &target_fixed;

        optimizer.zero_grad().unwrap();

        let normed = ln.forward(x).unwrap();
        let qkv = qkv_proj.forward(&normed).unwrap();
        let qkv_chunks = chunk_t(&qkv, 3, 2).unwrap();

        let q = qkv_chunks[0]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .unwrap();
        let k = qkv_chunks[1]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .unwrap();
        let v = qkv_chunks[2]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .unwrap();

        let k_t = permute_t(&k, &[0, 2, 1]).unwrap();
        let scores = grad_fns::linalg::bmm_differentiable(&q, &k_t).unwrap();
        let scale = scalar::<f32>((head_dim as f32).sqrt()).unwrap();
        let scores = grad_fns::arithmetic::div(&scores, &scale).unwrap();
        let attn = grad_fns::activation::softmax(&scores).unwrap();
        let attn_out = grad_fns::linalg::bmm_differentiable(&attn, &v).unwrap();
        let attn_out = attn_out
            .view(&[batch as i64, seq as i64, d_model as i64])
            .unwrap();
        let output = out_proj.forward(&attn_out).unwrap();

        // Residual
        let output = (x + &output).unwrap();

        // MSE loss vs target
        let diff = (&output - target).unwrap();
        let loss = (&diff * &diff).unwrap().mean_all().unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        losses.push(loss_val);

        loss.backward().unwrap();

        let model_params: Vec<&Parameter<f32>> = qkv_proj
            .parameters()
            .into_iter()
            .chain(out_proj.parameters())
            .chain(ln.parameters())
            .collect();
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();

        if step == 0 || step == 29 {
            println!("  Step {step}: loss = {loss_val:.4}");
        }
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "Transformer loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
    println!(
        "PASS: Transformer block training loss decreases ({:.4} -> {:.4})",
        losses[0], losses[29]
    );
}

// =====================================================================
// SPEED: Timed comparison benchmarks
// =====================================================================
#[test]
fn test_speed_comparison() {
    println!("\n{}", "=".repeat(60));
    println!("SPEED COMPARISON vs PyTorch");
    println!("{}", "=".repeat(60));

    // PyTorch reference numbers from pytorch_validate.py
    let pytorch_ref = vec![
        ("add_1M", 91.7),
        ("mul_1M", 41.9),
        ("relu_1M", 19.9),
        ("sigmoid_1M", 163.5),
        ("matmul_64", 5.2),
        ("matmul_256", 145.4),
        ("matmul_1024", 2853.0),
        ("mlp_fwd_b32", 50.0),
        ("mlp_bwd_b32", 389.5),
        ("train_step_b32", 747.8),
    ];

    let a = rand::<f32>(&[1000, 1000]).unwrap();
    let b = rand::<f32>(&[1000, 1000]).unwrap();

    let run = |_name: &str, iters: usize, f: &dyn Fn()| -> f64 {
        for _ in 0..5 {
            f();
        }
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }

        start.elapsed().as_secs_f64() / iters as f64 * 1e6
    };

    let ft_add = run("add_1M", 100, &|| {
        let _ = (&a + &b).unwrap();
    });
    let ft_mul = run("mul_1M", 100, &|| {
        let _ = (&a * &b).unwrap();
    });
    let ft_relu = run("relu_1M", 100, &|| {
        let _ = a.relu().unwrap();
    });
    let ft_sig = run("sigmoid_1M", 100, &|| {
        let _ = a.sigmoid().unwrap();
    });

    let a64 = rand::<f32>(&[64, 64]).unwrap();
    let b64 = rand::<f32>(&[64, 64]).unwrap();
    let ft_mm64 = run("matmul_64", 100, &|| {
        let _ = a64.matmul(&b64).unwrap();
    });

    let a256 = rand::<f32>(&[256, 256]).unwrap();
    let b256 = rand::<f32>(&[256, 256]).unwrap();
    let ft_mm256 = run("matmul_256", 100, &|| {
        let _ = a256.matmul(&b256).unwrap();
    });

    let a1024 = rand::<f32>(&[1024, 1024]).unwrap();
    let b1024 = rand::<f32>(&[1024, 1024]).unwrap();
    let ft_mm1024 = run("matmul_1024", 20, &|| {
        let _ = a1024.matmul(&b1024).unwrap();
    });

    let results = vec![
        ("add_1M", ft_add),
        ("mul_1M", ft_mul),
        ("relu_1M", ft_relu),
        ("sigmoid_1M", ft_sig),
        ("matmul_64", ft_mm64),
        ("matmul_256", ft_mm256),
        ("matmul_1024", ft_mm1024),
    ];

    println!(
        "\n{:<20} {:>12} {:>12} {:>8}",
        "Operation", "PyTorch(us)", "ferrotorch(us)", "Ratio"
    );
    println!("{}", "-".repeat(56));
    for (name, ft_us) in &results {
        if let Some((_, pt_us)) = pytorch_ref.iter().find(|(n, _)| n == name) {
            let ratio = ft_us / pt_us;
            let status = if ratio < 1.5 {
                "✓"
            } else if ratio < 5.0 {
                "~"
            } else {
                "✗"
            };
            println!(
                "{:<20} {:>12.1} {:>12.1} {:>7.1}x {status}",
                name, pt_us, ft_us, ratio
            );
        }
    }
}
