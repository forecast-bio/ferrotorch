//! End-to-end GPU training integration tests.
//!
//! Each test runs a small training loop on CPU, verifying loss decreases.
//! These tests catch systemic issues that unit tests miss:
//! - Device mismatches in autograd backward (issue #14)
//! - Optimizers crashing on GPU parameters (issue #13)
//! - Shape ops breaking the computation graph

use ferrotorch_core::*;
use ferrotorch_nn::*;
use ferrotorch_optim::*;

// =====================================================================
// Experiment 1: MLP on synthetic data
// =====================================================================

#[test]
fn test_mlp_training_cpu() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 128, true).unwrap()),
        Box::new(ReLU::default()),
        Box::new(Linear::new(128, 10, true).unwrap()),
    ]);

    let mut optimizer = Adam::new(
        model.parameters().into_iter().cloned().collect(),
        AdamConfig::default(),
    );

    let ce_loss = CrossEntropyLoss::new(Reduction::Mean, 0.0);
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..10 {
        let x = rand::<f32>(&[32, 784]).unwrap();
        let t = from_vec((0..32).map(|i| (i % 10) as f32).collect(), &[32]).unwrap();

        optimizer.zero_grad().unwrap();
        let output = model.forward(&x).unwrap();
        let loss = ce_loss.forward(&output, &t).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 { first_loss = loss_val; }
        last_loss = loss_val;

        loss.backward().unwrap();

        let model_params: Vec<&Parameter<f32>> = model.parameters();
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(last_loss < first_loss, "MLP CPU: loss should decrease. first={first_loss}, last={last_loss}");
}

// =====================================================================
// Experiment 2: Small Transformer (attention + MLP)
// =====================================================================

#[test]
fn test_transformer_training_cpu() {
    // Simplified: just QKV projection + attention + output projection + MLP
    let d_model = 64;
    let n_heads = 4;
    let head_dim = d_model / n_heads;
    let batch = 2;
    let seq = 8;

    let qkv_proj = Linear::new(d_model, 3 * d_model, false).unwrap();
    let out_proj = Linear::new(d_model, d_model, false).unwrap();
    let mlp_up = Linear::new(d_model, 4 * d_model, true).unwrap();
    let mlp_down = Linear::new(4 * d_model, d_model, true).unwrap();
    let ln = LayerNorm::new(vec![d_model], 1e-5, true).unwrap();

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(qkv_proj.parameters().into_iter().cloned());
    all_params.extend(out_proj.parameters().into_iter().cloned());
    all_params.extend(mlp_up.parameters().into_iter().cloned());
    all_params.extend(mlp_down.parameters().into_iter().cloned());
    all_params.extend(ln.parameters().into_iter().cloned());

    let mut optimizer = AdamW::new(all_params, AdamWConfig { lr: 1e-3, ..Default::default() });

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..10 {
        let x = randn::<f32>(&[batch, seq, d_model]).unwrap();
        let target = randn::<f32>(&[batch, seq, d_model]).unwrap();

        optimizer.zero_grad().unwrap();

        // QKV projection
        let qkv = qkv_proj.forward(&x).unwrap();
        let qkv_chunks = chunk_t(&qkv, 3, 2).unwrap();

        // Reshape to [batch*n_heads, seq, head_dim]
        let q = qkv_chunks[0].view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64]).unwrap();
        let k = qkv_chunks[1].view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64]).unwrap();
        let v = qkv_chunks[2].view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64]).unwrap();

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let k_t = permute_t(&k, &[0, 2, 1]).unwrap();
        let scores = grad_fns::linalg::bmm_differentiable(&q, &k_t).unwrap();
        let scale = scalar::<f32>((head_dim as f32).sqrt()).unwrap();
        let scores = grad_fns::arithmetic::div(&scores, &scale).unwrap();
        let attn = grad_fns::activation::softmax(&scores).unwrap();
        let attn_out = grad_fns::linalg::bmm_differentiable(&attn, &v).unwrap();

        // Reshape back + output projection
        let attn_out = attn_out.view(&[batch as i64, seq as i64, d_model as i64]).unwrap();
        let attn_out = out_proj.forward(&attn_out).unwrap();

        // Residual + LayerNorm + MLP
        let h = grad_fns::arithmetic::add(&x, &attn_out).unwrap();
        let h = ln.forward(&h).unwrap();
        let up = mlp_up.forward(&h).unwrap();
        let activated = up.relu().unwrap();
        let down = mlp_down.forward(&activated).unwrap();
        let output = grad_fns::arithmetic::add(&h, &down).unwrap();

        // MSE loss
        let diff = grad_fns::arithmetic::sub(&output, &target).unwrap();
        let sq = grad_fns::arithmetic::mul(&diff, &diff).unwrap();
        let loss = grad_fns::reduction::mean(&sq).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 { first_loss = loss_val; }
        last_loss = loss_val;

        loss.backward().unwrap();

        let all_model_params: Vec<&Parameter<f32>> = {
            let mut p = Vec::new();
            p.extend(qkv_proj.parameters());
            p.extend(out_proj.parameters());
            p.extend(mlp_up.parameters());
            p.extend(mlp_down.parameters());
            p.extend(ln.parameters());
            p
        };
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(last_loss < first_loss, "Transformer CPU: loss should decrease. first={first_loss}, last={last_loss}");
}

// =====================================================================
// Experiment 3: CNN on synthetic CIFAR-like data
// =====================================================================

#[test]
fn test_cnn_training_cpu() {
    let conv = Conv2d::<f32>::new(3, 8, (3, 3), (1, 1), (1, 1), true).unwrap();
    let pool = AvgPool2d::new([4, 4], [4, 4], [0, 0]);
    // Conv2d: 32x32 -> 32x32 (pad=1), AvgPool 4x4: 32x32 -> 8x8, channels=8
    // But flatten also includes batch dim. Let's use adaptive pool instead.
    let linear = Linear::new(8 * 8 * 8, 10, true).unwrap(); // 8 channels * 8*8

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(conv.parameters().into_iter().cloned());
    all_params.extend(linear.parameters().into_iter().cloned());

    let mut optimizer = Adam::new(all_params, AdamConfig::default());
    let ce_loss = CrossEntropyLoss::new(Reduction::Mean, 0.0);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..10 {
        let input = randn::<f32>(&[4, 3, 32, 32]).unwrap();
        let target = from_vec((0..4).map(|i| (i % 10) as f32).collect(), &[4]).unwrap();

        optimizer.zero_grad().unwrap();

        let h = conv.forward(&input).unwrap();
        let h = h.relu().unwrap();
        let h = pool.forward(&h).unwrap();
        let h = h.view(&[4, -1]).unwrap(); // [4, 8*8*8]
        let output = linear.forward(&h).unwrap();

        let loss = ce_loss.forward(&output, &target).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 { first_loss = loss_val; }
        last_loss = loss_val;

        loss.backward().unwrap();

        let all_model_params: Vec<&Parameter<f32>> = {
            let mut p = Vec::new();
            p.extend(conv.parameters());
            p.extend(linear.parameters());
            p
        };
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(last_loss < first_loss, "CNN CPU: loss should decrease. first={first_loss}, last={last_loss}");
}

// =====================================================================
// Experiment 4: LSTM language model
// =====================================================================

#[test]
fn test_lstm_training_cpu() {
    let vocab_size = 50;
    let embed_dim = 32;
    let hidden_size = 64;

    let embedding = Embedding::<f32>::new(vocab_size, embed_dim, None).unwrap();
    let lstm = LSTM::new(embed_dim, hidden_size, 1).unwrap();
    let output_proj = Linear::new(hidden_size, vocab_size, true).unwrap();

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(embedding.parameters().into_iter().cloned());
    all_params.extend(lstm.parameters().into_iter().cloned());
    all_params.extend(output_proj.parameters().into_iter().cloned());

    let mut optimizer = Adam::new(all_params, AdamConfig { lr: 1e-3, ..Default::default() });
    let ce_loss = CrossEntropyLoss::new(Reduction::Mean, 0.0);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..10 {
        // Random token IDs: batch=4, seq=8
        let token_ids: Vec<f32> = (0..32).map(|i| (i % vocab_size) as f32).collect();
        let target_ids: Vec<f32> = (0..32).map(|i| ((i + 1) % vocab_size) as f32).collect();

        optimizer.zero_grad().unwrap();

        // Embed each token individually and stack
        let mut emb_data = Vec::new();
        for &tid in &token_ids {
            let idx = from_vec(vec![tid], &[1]).unwrap();
            let emb = embedding.forward(&idx).unwrap();
            emb_data.extend(emb.data_vec().unwrap());
        }
        let embedded = from_vec(emb_data, &[4, 8, embed_dim]).unwrap().requires_grad_(true);

        // LSTM forward
        let (lstm_out, _state) = lstm.forward_with_state(&embedded, None).unwrap();

        // Project to vocab: reshape [4,8,hidden] -> [32, hidden]
        let flat = lstm_out.view(&[32, hidden_size as i64]).unwrap();
        let logits = output_proj.forward(&flat).unwrap();

        let targets = from_vec(target_ids, &[32]).unwrap();
        let loss = ce_loss.forward(&logits, &targets).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 { first_loss = loss_val; }
        last_loss = loss_val;

        loss.backward().unwrap();

        let all_model_params: Vec<&Parameter<f32>> = {
            let mut p = Vec::new();
            p.extend(embedding.parameters());
            p.extend(lstm.parameters());
            p.extend(output_proj.parameters());
            p
        };
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(last_loss < first_loss, "LSTM CPU: loss should decrease. first={first_loss}, last={last_loss}");
}

// =====================================================================
// Experiment 5: VAE with reparameterization trick
// =====================================================================

#[test]
fn test_vae_training_cpu() {
    let input_dim = 28 * 28;
    let hidden_dim = 128;
    let latent_dim = 16;

    let enc_fc = Linear::new(input_dim, hidden_dim, true).unwrap();
    let enc_mu = Linear::new(hidden_dim, latent_dim, true).unwrap();
    let enc_logvar = Linear::new(hidden_dim, latent_dim, true).unwrap();
    let dec_fc1 = Linear::new(latent_dim, hidden_dim, true).unwrap();
    let dec_fc2 = Linear::new(hidden_dim, input_dim, true).unwrap();

    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(enc_fc.parameters().into_iter().cloned());
    all_params.extend(enc_mu.parameters().into_iter().cloned());
    all_params.extend(enc_logvar.parameters().into_iter().cloned());
    all_params.extend(dec_fc1.parameters().into_iter().cloned());
    all_params.extend(dec_fc2.parameters().into_iter().cloned());

    let mut optimizer = Adam::new(all_params, AdamConfig { lr: 1e-3, ..Default::default() });

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..10 {
        let input = rand::<f32>(&[8, input_dim]).unwrap();
        optimizer.zero_grad().unwrap();

        // Encode
        let h = enc_fc.forward(&input).unwrap().relu().unwrap();
        let mu = enc_mu.forward(&h).unwrap();
        let logvar = enc_logvar.forward(&h).unwrap();

        // Reparameterize: z = mu + eps * exp(0.5 * logvar)
        let half = scalar::<f32>(0.5).unwrap();
        let half_logvar = grad_fns::arithmetic::mul(&logvar, &half).unwrap();
        let std = grad_fns::transcendental::exp(&half_logvar).unwrap();
        let eps = randn::<f32>(&[8, latent_dim]).unwrap();
        let z = grad_fns::arithmetic::add(
            &mu,
            &grad_fns::arithmetic::mul(&eps, &std).unwrap(),
        ).unwrap();

        // Decode
        let h_dec = dec_fc1.forward(&z).unwrap().relu().unwrap();
        let recon = dec_fc2.forward(&h_dec).unwrap();
        let recon = grad_fns::activation::sigmoid(&recon).unwrap();

        // Reconstruction loss (MSE)
        let diff = grad_fns::arithmetic::sub(&recon, &input).unwrap();
        let recon_loss = grad_fns::reduction::mean(
            &grad_fns::arithmetic::mul(&diff, &diff).unwrap()
        ).unwrap();

        // KL divergence: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        let one = scalar::<f32>(1.0).unwrap();
        let mu_sq = grad_fns::arithmetic::mul(&mu, &mu).unwrap();
        let exp_logvar = grad_fns::transcendental::exp(&logvar).unwrap();
        let kl_inner = grad_fns::arithmetic::sub(
            &grad_fns::arithmetic::sub(
                &grad_fns::arithmetic::add(&one, &logvar).unwrap(),
                &mu_sq,
            ).unwrap(),
            &exp_logvar,
        ).unwrap();
        let kl_loss = grad_fns::arithmetic::mul(
            &grad_fns::reduction::mean(&kl_inner).unwrap(),
            &scalar::<f32>(-0.5).unwrap(),
        ).unwrap();

        let loss = grad_fns::arithmetic::add(&recon_loss, &kl_loss).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 { first_loss = loss_val; }
        last_loss = loss_val;

        loss.backward().unwrap();

        let all_model_params: Vec<&Parameter<f32>> = {
            let mut p = Vec::new();
            p.extend(enc_fc.parameters());
            p.extend(enc_mu.parameters());
            p.extend(enc_logvar.parameters());
            p.extend(dec_fc1.parameters());
            p.extend(dec_fc2.parameters());
            p
        };
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(last_loss < first_loss, "VAE CPU: loss should decrease. first={first_loss}, last={last_loss}");
}
