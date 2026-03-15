#!/usr/bin/env -S cargo +nightly -Zscript
//! ferrotorch performance benchmarks — run with: cargo run --release --example ferrotorch_bench
//! Or compile manually: rustc -O benchmarks/ferrotorch_bench.rs ...
//!
//! This is meant to be run as a standalone binary via the ferrotorch crate.

use std::time::Instant;

use ferrotorch_core::*;
use ferrotorch_nn::*;
use ferrotorch_optim::*;

fn bench<F>(name: &str, warmup: usize, iters: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64 * 1e6; // microseconds
    println!("  {name}: {elapsed:.1} us");
    elapsed
}

fn main() -> FerrotorchResult<()> {
    println!("ferrotorch benchmarks");
    println!();

    // ========== CPU Benchmarks ==========
    println!("{}", "=".repeat(60));
    println!("CPU BENCHMARKS");
    println!("{}", "=".repeat(60));

    // 1. Tensor creation
    println!("\n--- Tensor Creation ---");
    bench("zeros [1000,1000]", 5, 100, || {
        let _ = zeros::<f32>(&[1000, 1000]).unwrap();
    });
    bench("rand [1000,1000]", 5, 100, || {
        let _ = rand::<f32>(&[1000, 1000]).unwrap();
    });

    // 2. Elementwise ops
    println!("\n--- Elementwise Ops ---");
    let a = rand::<f32>(&[1000, 1000])?;
    let b = rand::<f32>(&[1000, 1000])?;
    bench("add [1000,1000]", 5, 100, || {
        let _ = (&a + &b).unwrap();
    });
    bench("mul [1000,1000]", 5, 100, || {
        let _ = (&a * &b).unwrap();
    });
    bench("relu [1000,1000]", 5, 100, || {
        let _ = a.relu().unwrap();
    });
    bench("sigmoid [1000,1000]", 5, 100, || {
        let _ = a.sigmoid().unwrap();
    });

    // 3. Matrix multiply
    println!("\n--- Matrix Multiply ---");
    for &size in &[64usize, 256, 1024] {
        let a = rand::<f32>(&[size, size])?;
        let b = rand::<f32>(&[size, size])?;
        let iters = if size <= 256 { 100 } else { 20 };
        bench(&format!("matmul [{size},{size}]"), 5, iters, || {
            let _ = a.matmul(&b).unwrap();
        });
    }

    // 4. Forward pass (MLP)
    println!("\n--- Forward Pass (MLP 784->256->10) ---");
    let mlp = Sequential::new(vec![
        Box::new(Linear::new(784, 256, true)?),
        Box::new(ReLU::default()),
        Box::new(Linear::new(256, 10, true)?),
    ]);
    let x_mlp = rand::<f32>(&[32, 784])?;
    bench("MLP forward B=32", 5, 100, || {
        let _ = mlp.forward(&x_mlp).unwrap();
    });

    // 5. Backward pass
    println!("\n--- Backward Pass (MLP 784->256->10) ---");
    bench("MLP backward B=32", 5, 50, || {
        let x = rand::<f32>(&[32, 784]).unwrap().requires_grad_(true);
        let out = mlp.forward(&x).unwrap();
        let loss = out.sum_all().unwrap();
        loss.backward().unwrap();
    });

    // 6. Full training step
    println!("\n--- Full Training Step (MLP + Adam) ---");
    let mut optimizer = Adam::new(
        mlp.parameters().into_iter().cloned().collect(),
        AdamConfig::default(),
    );
    let loss_fn = CrossEntropyLoss::new(Reduction::Mean, 0.0);
    bench("training step B=32", 3, 50, || {
        let x = rand::<f32>(&[32, 784]).unwrap();
        let target = from_vec(
            (0..32).map(|i| (i % 10) as f32).collect(),
            &[32],
        ).unwrap();
        optimizer.zero_grad().unwrap();
        let out = mlp.forward(&x).unwrap();
        let loss = loss_fn.forward(&out, &target).unwrap();
        loss.backward().unwrap();

        // Sync grads to optimizer
        let model_params: Vec<&Parameter<f32>> = mlp.parameters();
        let opt_params = &optimizer.param_groups()[0].params;
        for (mp, op) in model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        optimizer.step().unwrap();
    });

    println!("\n{}", "=".repeat(60));
    println!("Done.");
    Ok(())
}
