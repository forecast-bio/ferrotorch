//! End-to-end MNIST training example.
//!
//! Demonstrates the full ferrotorch stack working together:
//! - **Model**: 3-layer MLP (784 -> 128 -> 64 -> 10) with ReLU activations
//! - **Dataset**: Synthetic MNIST from `ferrotorch-vision`
//! - **DataLoader**: Batched iteration with shuffle
//! - **Optimizer**: Adam with default hyperparameters
//! - **Loss**: CrossEntropyLoss (log-softmax + NLL)
//! - **Training loop**: Epoch/batch loss reporting with full parameter sync
//!
//! Run with:
//! ```sh
//! cargo run --example train_mnist -p ferrotorch
//! ```

use std::sync::Arc;

use ferrotorch_core::{FerrotorchResult, from_vec};
use ferrotorch_data::DataLoader;
use ferrotorch_nn::{CrossEntropyLoss, Linear, Module, Parameter, ReLU, Reduction, Sequential};
use ferrotorch_optim::{Adam, AdamConfig, Optimizer};
use ferrotorch_vision::{Mnist, Split};

fn main() -> FerrotorchResult<()> {
    println!("ferrotorch MNIST training example");
    println!("==================================\n");

    // ── 1. Build model ─────────────────────────────────────────────
    //
    // A 3-layer MLP: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10
    let layer1 = Linear::<f32>::new(784, 128, true)?;
    let relu1 = ReLU::default();
    let layer2 = Linear::<f32>::new(128, 64, true)?;
    let relu2 = ReLU::default();
    let layer3 = Linear::<f32>::new(64, 10, true)?;

    let mut model = Sequential::new(vec![
        Box::new(layer1),
        Box::new(relu1),
        Box::new(layer2),
        Box::new(relu2),
        Box::new(layer3),
    ]);

    let num_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("Model: 3-layer MLP (784 -> 128 -> 64 -> 10)");
    println!(
        "Parameters: {} ({} total elements)",
        model.parameters().len(),
        num_params
    );
    println!();

    // ── 2. Create synthetic dataset and data loader ────────────────
    //
    // Using synthetic data (random images + labels) so this example
    // runs without downloading real MNIST files.
    let num_samples = 1000;
    let batch_size = 32;
    let train_dataset = Mnist::<f32>::synthetic(Split::Train, num_samples)?;
    let train_loader = DataLoader::new(Arc::new(train_dataset), batch_size)?
        .shuffle(true)
        .seed(42);

    println!("Dataset: Synthetic MNIST, {} samples", num_samples);
    println!(
        "Batch size: {}, {} batches/epoch\n",
        batch_size,
        train_loader.len()
    );

    // ── 3. Create optimizer ────────────────────────────────────────
    //
    // Adam with default lr=0.001, betas=(0.9, 0.999).
    // The optimizer takes ownership of cloned parameters.
    let mut optimizer = Adam::new(
        model.parameters().into_iter().cloned().collect(),
        AdamConfig::default(),
    );

    // ── 4. Loss function ───────────────────────────────────────────
    let loss_fn = CrossEntropyLoss::new(Reduction::Mean, 0.0);

    // ── 5. Training loop ───────────────────────────────────────────
    let num_epochs = 5;

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0u32;

        for batch_result in train_loader.iter(epoch) {
            let batch = batch_result?;
            let current_batch_size = batch.len();
            if current_batch_size == 0 {
                continue;
            }

            // -- Collate: stack individual samples into batch tensors --
            //
            // Flatten images from [1, 28, 28] to [784] and stack into
            // a single [B, 784] input tensor. Collect labels into [B].
            let mut input_data = Vec::with_capacity(current_batch_size * 784);
            let mut target_data = Vec::with_capacity(current_batch_size);

            for sample in &batch {
                let img_data = sample.image.data()?;
                input_data.extend_from_slice(img_data);
                target_data.push(sample.label as f32);
            }

            // Input does not need gradients (leaf data).
            let input = from_vec(input_data, &[current_batch_size, 784])?;
            // CrossEntropyLoss expects targets as float-encoded class indices.
            let target = from_vec(target_data, &[current_batch_size])?;

            // -- Forward pass --
            let logits = model.forward(&input)?;
            let loss = loss_fn.forward(&logits, &target)?;

            let loss_val = loss.item()?;
            epoch_loss += loss_val;
            num_batches += 1;

            // -- Backward pass --
            optimizer.zero_grad()?;
            loss.backward()?;

            // -- Sync gradients from model to optimizer --
            //
            // The optimizer holds its own clones of the parameters (not
            // references to the model's parameters). After backward()
            // accumulates gradients on the model's parameters, we must
            // copy those gradients to the optimizer's parameter copies
            // so that optimizer.step() can read them.
            //
            // A future API revision will use Arc-shared parameters to
            // avoid this manual synchronization.
            sync_grads_to_optimizer(&model, &mut optimizer)?;

            // -- Optimizer step --
            optimizer.step()?;

            // -- Sync updated weights back to model --
            //
            // After step(), the optimizer has new parameter tensors with
            // updated values. Copy them back into the model so the next
            // forward pass uses the updated weights.
            sync_params_from_optimizer(&mut model, &optimizer)?;
        }

        let avg_loss = if num_batches > 0 {
            epoch_loss / num_batches as f32
        } else {
            0.0
        };
        println!(
            "Epoch {}/{}: avg_loss = {:.4}  ({} batches)",
            epoch + 1,
            num_epochs,
            avg_loss,
            num_batches,
        );
    }

    println!("\nTraining complete!");
    println!(
        "Note: Loss may not decrease significantly because the dataset is synthetic \
         (random images with random labels). With real MNIST data, the loss would \
         converge to near zero."
    );

    Ok(())
}

/// Copy gradients from the model's parameters to the optimizer's parameter copies.
///
/// This is necessary because `Adam::new` clones the parameters, so the model
/// and optimizer hold separate `Parameter` instances. After `loss.backward()`
/// writes gradients to the model's parameters, we forward them to the
/// optimizer so `step()` can use them.
fn sync_grads_to_optimizer(
    model: &Sequential<f32>,
    optimizer: &mut Adam<f32>,
) -> FerrotorchResult<()> {
    let model_params: Vec<&Parameter<f32>> = model.parameters();
    let opt_params = optimizer.param_groups()[0].params();

    for (mp, op) in model_params.iter().zip(opt_params.iter()) {
        if let Some(grad) = mp.grad()? {
            op.tensor().set_grad(Some(grad))?;
        }
    }

    Ok(())
}

/// Copy updated parameter data from the optimizer back into the model.
///
/// After `optimizer.step()`, the optimizer holds fresh `Parameter` instances
/// wrapping tensors with updated weight values. We use `parameters_mut()` to
/// get mutable access to the model's parameters and replace their underlying
/// tensor data with the optimizer's updated values via `Parameter::set_data`.
fn sync_params_from_optimizer(
    model: &mut Sequential<f32>,
    optimizer: &Adam<f32>,
) -> FerrotorchResult<()> {
    let opt_params = optimizer.param_groups()[0].params();

    // Collect updated tensors first to avoid borrow overlap.
    let updated_tensors: Vec<_> = opt_params.iter().map(|p| p.tensor().clone()).collect();

    let model_params = model.parameters_mut();
    for (mp, updated) in model_params.into_iter().zip(updated_tensors) {
        mp.set_data(updated);
    }

    Ok(())
}
