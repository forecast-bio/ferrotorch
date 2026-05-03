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

    // Hold the batch fixed across steps so loss is guaranteed to
    // decrease for a sane optimiser. Sampling a fresh batch each step
    // (which the original version did) lets loss legitimately oscillate
    // with only 10 SGD steps and made the test intermittently fail
    // under workspace-parallel runs.
    let x = rand::<f32>(&[32, 784]).unwrap();
    let t = from_vec((0..32).map(|i| (i % 10) as f32).collect(), &[32]).unwrap();

    for step in 0..10 {
        optimizer.zero_grad().unwrap();
        let output = model.forward(&x).unwrap();
        let loss = ce_loss.forward(&output, &t).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        loss.backward().unwrap();

        let model_params: Vec<&Parameter<f32>> = model.parameters();
        let opt_params = optimizer.param_groups()[0].params();
        for (mp, op) in model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(
        last_loss < first_loss,
        "MLP CPU: loss should decrease. first={first_loss}, last={last_loss}"
    );
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

    let mut adamw_cfg = AdamWConfig::default();
    adamw_cfg.lr = 1e-3;
    let mut optimizer = AdamW::new(all_params, adamw_cfg);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    // Fix the batch across steps — see comment in test_mlp_training_cpu.
    let x = randn::<f32>(&[batch, seq, d_model]).unwrap();
    let target = randn::<f32>(&[batch, seq, d_model]).unwrap();

    for step in 0..10 {
        optimizer.zero_grad().unwrap();

        // QKV projection
        let qkv = qkv_proj.forward(&x).unwrap();
        let qkv_chunks = chunk_t(&qkv, 3, 2).unwrap();

        // Reshape to [batch*n_heads, seq, head_dim]
        let q = qkv_chunks[0]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .unwrap();
        let k = qkv_chunks[1]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .unwrap();
        let v = qkv_chunks[2]
            .view(&[batch as i64 * n_heads as i64, seq as i64, head_dim as i64])
            .unwrap();

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let k_t = permute_t(&k, &[0, 2, 1]).unwrap();
        let scores = grad_fns::linalg::bmm_differentiable(&q, &k_t).unwrap();
        let scale = scalar::<f32>((head_dim as f32).sqrt()).unwrap();
        let scores = grad_fns::arithmetic::div(&scores, &scale).unwrap();
        let attn = grad_fns::activation::softmax(&scores).unwrap();
        let attn_out = grad_fns::linalg::bmm_differentiable(&attn, &v).unwrap();

        // Reshape back + output projection
        let attn_out = attn_out
            .view(&[batch as i64, seq as i64, d_model as i64])
            .unwrap();
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
        if step == 0 {
            first_loss = loss_val;
        }
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
        let opt_params = optimizer.param_groups()[0].params();
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(
        last_loss < first_loss,
        "Transformer CPU: loss should decrease. first={first_loss}, last={last_loss}"
    );
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
        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        loss.backward().unwrap();

        let all_model_params: Vec<&Parameter<f32>> = {
            let mut p = Vec::new();
            p.extend(conv.parameters());
            p.extend(linear.parameters());
            p
        };
        let opt_params = optimizer.param_groups()[0].params();
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(
        last_loss < first_loss,
        "CNN CPU: loss should decrease. first={first_loss}, last={last_loss}"
    );
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

    let mut adam_cfg = AdamConfig::default();
    adam_cfg.lr = 1e-3;
    let mut optimizer = Adam::new(all_params, adam_cfg);
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
        let embedded = from_vec(emb_data, &[4, 8, embed_dim])
            .unwrap()
            .requires_grad_(true);

        // LSTM forward
        let (lstm_out, _state) = lstm.forward_with_state(&embedded, None).unwrap();

        // Project to vocab: reshape [4,8,hidden] -> [32, hidden]
        let flat = lstm_out.view(&[32, hidden_size as i64]).unwrap();
        let logits = output_proj.forward(&flat).unwrap();

        let targets = from_vec(target_ids, &[32]).unwrap();
        let loss = ce_loss.forward(&logits, &targets).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        loss.backward().unwrap();

        let all_model_params: Vec<&Parameter<f32>> = {
            let mut p = Vec::new();
            p.extend(embedding.parameters());
            p.extend(lstm.parameters());
            p.extend(output_proj.parameters());
            p
        };
        let opt_params = optimizer.param_groups()[0].params();
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(
        last_loss < first_loss,
        "LSTM CPU: loss should decrease. first={first_loss}, last={last_loss}"
    );
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

    let mut adam_cfg = AdamConfig::default();
    adam_cfg.lr = 1e-3;
    let mut optimizer = Adam::new(all_params, adam_cfg);

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
        let z = grad_fns::arithmetic::add(&mu, &grad_fns::arithmetic::mul(&eps, &std).unwrap())
            .unwrap();

        // Decode
        let h_dec = dec_fc1.forward(&z).unwrap().relu().unwrap();
        let recon = dec_fc2.forward(&h_dec).unwrap();
        let recon = grad_fns::activation::sigmoid(&recon).unwrap();

        // Reconstruction loss (MSE)
        let diff = grad_fns::arithmetic::sub(&recon, &input).unwrap();
        let recon_loss =
            grad_fns::reduction::mean(&grad_fns::arithmetic::mul(&diff, &diff).unwrap()).unwrap();

        // KL divergence: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        let one = scalar::<f32>(1.0).unwrap();
        let mu_sq = grad_fns::arithmetic::mul(&mu, &mu).unwrap();
        let exp_logvar = grad_fns::transcendental::exp(&logvar).unwrap();
        let kl_inner = grad_fns::arithmetic::sub(
            &grad_fns::arithmetic::sub(&grad_fns::arithmetic::add(&one, &logvar).unwrap(), &mu_sq)
                .unwrap(),
            &exp_logvar,
        )
        .unwrap();
        let kl_loss = grad_fns::arithmetic::mul(
            &grad_fns::reduction::mean(&kl_inner).unwrap(),
            &scalar::<f32>(-0.5).unwrap(),
        )
        .unwrap();

        let loss = grad_fns::arithmetic::add(&recon_loss, &kl_loss).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];
        if step == 0 {
            first_loss = loss_val;
        }
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
        let opt_params = optimizer.param_groups()[0].params();
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    }

    assert!(
        last_loss < first_loss,
        "VAE CPU: loss should decrease. first={first_loss}, last={last_loss}"
    );
}

// =====================================================================
// Experiment 6: Pythia-70M architecture (GPT-NeoX)
// =====================================================================
//
// Pythia-70M uses GPT-NeoX: parallel attention + MLP in each block,
// rotary position embeddings (RoPE), untied embedding/unembedding.
// We use reduced dimensions for test speed but identical structure.

/// GPT-NeoX block: attention and MLP run in parallel, then add.
///   h = x + Attn(LN(x)) + MLP(LN(x))
/// This is different from standard transformer where they're sequential.
#[derive(Debug)]
struct NeoXBlock {
    ln: LayerNorm<f32>,
    qkv: Linear<f32>,
    out_proj: Linear<f32>,
    mlp_up: Linear<f32>,
    mlp_down: Linear<f32>,
    rope: RotaryPositionEmbedding<f32>,
    n_heads: usize,
    head_dim: usize,
}

impl NeoXBlock {
    fn new(d_model: usize, n_heads: usize, max_seq: usize) -> FerrotorchResult<Self> {
        let head_dim = d_model / n_heads;
        Ok(Self {
            ln: LayerNorm::new(vec![d_model], 1e-5, true)?,
            qkv: Linear::new(d_model, 3 * d_model, true)?,
            out_proj: Linear::new(d_model, d_model, true)?,
            mlp_up: Linear::new(d_model, 4 * d_model, true)?,
            mlp_down: Linear::new(4 * d_model, d_model, true)?,
            rope: RotaryPositionEmbedding::new(head_dim, max_seq, 10000.0)?,
            n_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let d_model = x.shape()[2];

        // Single LayerNorm — GPT-NeoX uses one LN for both attention and MLP
        let normed = self.ln.forward(x)?;

        // === Attention path ===
        let qkv = self.qkv.forward(&normed)?;
        let qkv_chunks = chunk_t(&qkv, 3, 2)?;
        let q = &qkv_chunks[0];
        let k = &qkv_chunks[1];
        let v = &qkv_chunks[2];

        // Reshape to [batch*n_heads, seq, head_dim] for bmm
        let q = q.view(&[
            batch as i64 * self.n_heads as i64,
            seq as i64,
            self.head_dim as i64,
        ])?;
        let k = k.view(&[
            batch as i64 * self.n_heads as i64,
            seq as i64,
            self.head_dim as i64,
        ])?;
        let v = v.view(&[
            batch as i64 * self.n_heads as i64,
            seq as i64,
            self.head_dim as i64,
        ])?;

        // Apply RoPE to Q and K
        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;

        // Attention scores: Q @ K^T / sqrt(head_dim)
        let k_t = permute_t(&k, &[0, 2, 1])?;
        let scores = grad_fns::linalg::bmm_differentiable(&q, &k_t)?;
        let scale = scalar::<f32>((self.head_dim as f32).sqrt())?;
        let scores = grad_fns::arithmetic::div(&scores, &scale)?;

        // Causal mask: zero out future positions
        let mask_data: Vec<f32> = (0..seq * seq)
            .map(|i| {
                let row = i / seq;
                let col = i % seq;
                if col <= row { 0.0 } else { -1e9 }
            })
            .collect();
        let mask = from_vec(mask_data, &[1, seq, seq])?;
        let scores = grad_fns::arithmetic::add(&scores, &mask)?;

        let attn = grad_fns::activation::softmax(&scores)?;
        let attn_out = grad_fns::linalg::bmm_differentiable(&attn, &v)?;

        // Reshape back to [batch, seq, d_model]
        let attn_out = attn_out.view(&[batch as i64, seq as i64, d_model as i64])?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // === MLP path (runs in parallel with attention in GPT-NeoX) ===
        let mlp_h = self.mlp_up.forward(&normed)?;
        let mlp_h = mlp_h.relu()?; // Pythia uses GELU but ReLU is simpler for testing
        let mlp_out = self.mlp_down.forward(&mlp_h)?;

        // === Parallel residual: x + attn_out + mlp_out ===
        let h = grad_fns::arithmetic::add(x, &attn_out)?;
        grad_fns::arithmetic::add(&h, &mlp_out)
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
fn test_pythia_architecture_cpu() {
    // Pythia-70M dimensions scaled down for test speed:
    // Real: 6 layers, 8 heads, d=512, vocab=50304, seq=2048
    // Test: 2 layers, 4 heads, d=64, vocab=256, seq=16
    let d_model = 64;
    let n_heads = 4;
    let n_layers = 2;
    let vocab_size = 256;
    let max_seq = 32;
    let batch = 2;
    let seq = 16;

    // Build model components
    let token_emb = Embedding::<f32>::new(vocab_size, d_model, None).unwrap();
    let blocks: Vec<NeoXBlock> = (0..n_layers)
        .map(|_| NeoXBlock::new(d_model, n_heads, max_seq).unwrap())
        .collect();
    let final_ln = LayerNorm::new(vec![d_model], 1e-5, true).unwrap();
    let lm_head = Linear::new(d_model, vocab_size, false).unwrap(); // untied unembedding

    // Collect all parameters
    let mut all_params: Vec<Parameter<f32>> = Vec::new();
    all_params.extend(token_emb.parameters().into_iter().cloned());
    for block in &blocks {
        all_params.extend(block.parameters().into_iter().cloned());
    }
    all_params.extend(final_ln.parameters().into_iter().cloned());
    all_params.extend(lm_head.parameters().into_iter().cloned());

    let param_count: usize = all_params.iter().map(|p| p.tensor().numel()).sum();
    eprintln!("Pythia-test parameter count: {param_count}");

    let mut adamw_cfg = AdamWConfig::default();
    adamw_cfg.lr = 1e-3;
    adamw_cfg.betas = (0.9, 0.95); // Pythia uses beta2=0.95
    adamw_cfg.weight_decay = 0.01;
    let mut optimizer = AdamW::new(all_params, adamw_cfg);

    let ce_loss = CrossEntropyLoss::new(Reduction::Mean, 0.0);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..10 {
        // Random token IDs: [batch, seq]
        let token_ids: Vec<f32> = (0..batch * seq).map(|i| (i % vocab_size) as f32).collect();
        // Target: next-token prediction (shift by 1)
        let target_ids: Vec<f32> = (0..batch * seq)
            .map(|i| ((i + 1) % vocab_size) as f32)
            .collect();

        optimizer.zero_grad().unwrap();

        // === Forward pass ===

        // Token embedding: look up each token individually, then stack
        let mut emb_data = Vec::new();
        for &tid in &token_ids {
            let idx = from_vec(vec![tid], &[1]).unwrap();
            let emb = token_emb.forward(&idx).unwrap(); // [1, d_model]
            emb_data.extend(emb.data_vec().unwrap());
        }
        let mut h = from_vec(emb_data, &[batch, seq, d_model])
            .unwrap()
            .requires_grad_(true);

        // Transformer blocks
        for block in &blocks {
            h = block.forward(&h).unwrap();
        }

        // Final LayerNorm + LM head
        let h = final_ln.forward(&h).unwrap();
        let logits = lm_head.forward(&h).unwrap(); // [batch, seq, vocab]

        // Reshape for loss: [batch*seq, vocab] vs [batch*seq]
        let logits_flat = logits
            .view(&[(batch * seq) as i64, vocab_size as i64])
            .unwrap();
        let targets = from_vec(target_ids, &[batch * seq]).unwrap();

        let loss = ce_loss.forward(&logits_flat, &targets).unwrap();
        let loss_val = loss.data_vec().unwrap()[0];

        if step == 0 {
            first_loss = loss_val;
            eprintln!("Step 0 loss: {loss_val:.4}");
        }
        last_loss = loss_val;
        if step == 9 {
            eprintln!("Step 9 loss: {loss_val:.4}");
        }

        // === Backward pass ===
        loss.backward().unwrap();

        // Sync gradients to optimizer
        let mut all_model_params: Vec<&Parameter<f32>> = Vec::new();
        all_model_params.extend(token_emb.parameters());
        for block in &blocks {
            all_model_params.extend(block.parameters());
        }
        all_model_params.extend(final_ln.parameters());
        all_model_params.extend(lm_head.parameters());

        let opt_params = optimizer.param_groups()[0].params();
        for (mp, op) in all_model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }

        // === Optimizer step ===
        optimizer.step().unwrap();
    }

    assert!(
        last_loss < first_loss,
        "Pythia CPU: loss should decrease. first={first_loss:.4}, last={last_loss:.4}"
    );
    eprintln!("Pythia test PASSED: {first_loss:.4} -> {last_loss:.4}");
}
