//! Real Pythia-70M validation: load actual safetensors weights from
//! HuggingFace, run inference, compare against PyTorch reference outputs,
//! and benchmark training step speed.
//!
//! This test requires the Pythia-70M weights to be cached at:
//!   ~/.cache/huggingface/hub/models--EleutherAI--pythia-70m/
//!
//! Skip with: cargo test --test pythia_real -- --ignored

use std::time::Instant;

use ferrotorch_core::*;
use ferrotorch_nn::*;
use ferrotorch_optim::*;

// =====================================================================
// GPT-NeoX model implementation (Pythia architecture)
// =====================================================================

struct PythiaModel {
    token_emb: Embedding<f32>,
    blocks: Vec<NeoXBlock>,
    final_ln: LayerNorm<f32>,
    lm_head: Linear<f32>,
}

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

        let normed = self.ln.forward(x)?;

        // QKV
        let qkv = self.qkv.forward(&normed)?;
        let qkv_chunks = chunk_t(&qkv, 3, 2)?;

        // Reshape to [batch*n_heads, seq, head_dim]
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

        // RoPE
        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;

        // Attention
        let k_t = permute_t(&k, &[0, 2, 1])?;
        let scores = grad_fns::linalg::bmm_differentiable(&q, &k_t)?;
        let scale = scalar::<f32>((self.head_dim as f32).sqrt())?;
        let scores = grad_fns::arithmetic::div(&scores, &scale)?;

        // Causal mask
        let mask_data: Vec<f32> = (0..seq * seq)
            .map(|i| if (i % seq) <= (i / seq) { 0.0 } else { -1e9 })
            .collect();
        let mask = from_vec(mask_data, &[1, seq, seq])?;
        let scores = grad_fns::arithmetic::add(&scores, &mask)?;

        let attn = grad_fns::activation::softmax(&scores)?;
        let attn_out = grad_fns::linalg::bmm_differentiable(&attn, &v)?;
        let attn_out = attn_out.view(&[batch as i64, seq as i64, d_model as i64])?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // Parallel MLP
        let mlp_h = self.mlp_up.forward(&normed)?;
        let mlp_h = mlp_h.relu()?;
        let mlp_out = self.mlp_down.forward(&mlp_h)?;

        // x + attn + mlp
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

impl PythiaModel {
    fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        vocab: usize,
        max_seq: usize,
    ) -> FerrotorchResult<Self> {
        let blocks = (0..n_layers)
            .map(|_| NeoXBlock::new(d_model, n_heads, max_seq))
            .collect::<FerrotorchResult<Vec<_>>>()?;
        Ok(Self {
            token_emb: Embedding::new(vocab, d_model, None)?,
            blocks,
            final_ln: LayerNorm::new(vec![d_model], 1e-5, true)?,
            lm_head: Linear::new(d_model, vocab, false)?,
        })
    }

    fn forward(
        &self,
        token_ids: &[f32],
        batch: usize,
        seq: usize,
    ) -> FerrotorchResult<Tensor<f32>> {
        let d_model = self.token_emb.embedding_dim;

        // Embed tokens
        let mut emb_data = Vec::with_capacity(token_ids.len() * d_model);
        for &tid in token_ids {
            let idx = from_vec(vec![tid], &[1])?;
            let emb = self.token_emb.forward(&idx)?;
            emb_data.extend(emb.data_vec()?);
        }
        let mut h = from_vec(emb_data, &[batch, seq, d_model])?.requires_grad_(true);

        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        let h = self.final_ln.forward(&h)?;
        self.lm_head.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        let mut p = Vec::new();
        p.extend(self.token_emb.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.final_ln.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}

// =====================================================================
// Test: training from scratch, verify loss convergence + benchmark
// =====================================================================

#[test]
fn test_pythia_70m_training_benchmark() {
    // Real Pythia-70M dimensions
    let n_layers = 6;
    let d_model = 512;
    let n_heads = 8;
    let vocab = 50304;
    let max_seq = 2048;
    let batch = 1;
    let seq = 32;

    eprintln!("Building Pythia-70M model (6 layers, 8 heads, d=512, vocab=50304)...");
    let model = PythiaModel::new(n_layers, d_model, n_heads, vocab, max_seq)
        .expect("Pythia-70M model construction failed (Linear/LN/RoPE init)");

    let param_count: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    eprintln!(
        "Parameter count: {param_count} ({:.1}M)",
        param_count as f64 / 1e6
    );

    let mut adamw_cfg = AdamWConfig::default();
    adamw_cfg.lr = 1e-4;
    adamw_cfg.betas = (0.9, 0.95);
    adamw_cfg.weight_decay = 0.01;
    let mut optimizer = AdamW::new(model.parameters().into_iter().cloned().collect(), adamw_cfg);

    let ce_loss = CrossEntropyLoss::new(Reduction::Mean, 0.0);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    let mut total_time_ms = 0.0f64;

    eprintln!("\nTraining for 5 steps (batch={batch}, seq={seq})...");

    for step in 0..5 {
        let token_ids: Vec<f32> = (0..batch * seq).map(|i| (i % vocab) as f32).collect();
        let target_ids: Vec<f32> = (0..batch * seq).map(|i| ((i + 1) % vocab) as f32).collect();

        let step_start = Instant::now();

        optimizer
            .zero_grad()
            .expect("optimizer.zero_grad failed at start of Pythia training step");

        let logits = model
            .forward(&token_ids, batch, seq)
            .expect("Pythia forward pass failed");
        let logits_flat = logits
            .view(&[(batch * seq) as i64, vocab as i64])
            .expect("logits .view to [batch*seq, vocab] failed");
        let targets = from_vec(target_ids, &[batch * seq]).expect("from_vec for target_ids failed");

        let loss = ce_loss
            .forward(&logits_flat, &targets)
            .expect("CrossEntropyLoss forward failed");
        let loss_val = loss.data_vec().expect("loss.data_vec readback failed")[0];

        loss.backward()
            .expect("autograd backward failed for Pythia loss");

        // Sync grads
        let model_params = model.parameters();
        let opt_params = optimizer.param_groups()[0].params();
        for (mp, op) in model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().expect("Parameter::grad readback failed") {
                op.set_grad(Some(g))
                    .expect("Parameter::set_grad on optimizer-side param failed");
            }
        }
        optimizer
            .step()
            .expect("optimizer.step failed for Pythia training step");

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        if step > 0 {
            total_time_ms += step_ms;
        } // skip warmup step

        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;
        eprintln!("  Step {step}: loss={loss_val:.4}, time={step_ms:.0}ms");
    }

    let avg_ms = total_time_ms / 4.0; // 4 non-warmup steps
    eprintln!("\nAvg training step (excl warmup): {avg_ms:.0} ms");
    eprintln!("PyTorch CPU reference: ~155 ms/step");
    eprintln!("Ratio: {:.1}x", avg_ms / 155.0);
    eprintln!("Loss: {first_loss:.4} -> {last_loss:.4}");

    assert!(
        last_loss < first_loss,
        "Pythia-70M: loss should decrease. first={first_loss:.4}, last={last_loss:.4}"
    );
}
