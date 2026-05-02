#!/usr/bin/env -S cargo +nightly -Zscript
//! ferrotorch performance benchmarks — run with: cargo run --release --example ferrotorch_bench
//! Or compile manually: rustc -O benchmarks/ferrotorch_bench.rs ...
//!
//! This is meant to be run as a standalone binary via the ferrotorch crate.

use std::hint::black_box;
use std::time::Instant;

use ferrotorch_core::*;
use ferrotorch_nn::*;
use ferrotorch_optim::*;

fn bench<R, F>(name: &str, warmup: usize, iters: usize, mut f: F) -> f64
where
    F: FnMut() -> R,
{
    for _ in 0..warmup {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
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
        zeros::<f32>(&[1000, 1000]).unwrap()
    });
    bench("rand [1000,1000]", 5, 100, || {
        rand::<f32>(&[1000, 1000]).unwrap()
    });

    // 2. Elementwise ops
    println!("\n--- Elementwise Ops ---");
    let a = rand::<f32>(&[1000, 1000])?;
    let b = rand::<f32>(&[1000, 1000])?;
    bench("add [1000,1000]", 5, 100, || (&a + &b).unwrap());
    bench("mul [1000,1000]", 5, 100, || (&a * &b).unwrap());
    bench("relu [1000,1000]", 5, 100, || a.relu().unwrap());
    bench("sigmoid [1000,1000]", 5, 100, || a.sigmoid().unwrap());

    // 3. Matrix multiply
    println!("\n--- Matrix Multiply ---");
    for &size in &[64usize, 256, 1024] {
        let a = rand::<f32>(&[size, size])?;
        let b = rand::<f32>(&[size, size])?;
        let iters = if size <= 256 { 100 } else { 20 };
        bench(&format!("matmul [{size},{size}]"), 5, iters, || {
            a.matmul(&b).unwrap()
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
    bench("MLP forward B=32", 5, 100, || mlp.forward(&x_mlp).unwrap());

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
        let target = from_vec((0..32).map(|i| (i % 10) as f32).collect(), &[32]).unwrap();
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

    // ========== New Benchmarks (Waves 1-5) ==========

    // 7. Transcendental ops
    println!("\n--- Transcendental Ops ---");
    let t = rand::<f32>(&[1000, 1000])?;
    bench("exp [1000,1000]", 5, 100, || exp(&t).unwrap());
    // log needs positive input
    let t_pos = (&t + &scalar::<f32>(1.0)?)?;
    bench("log [1000,1000]", 5, 100, || log(&t_pos).unwrap());
    bench("sin [1000,1000]", 5, 100, || sin(&t).unwrap());
    bench("cos [1000,1000]", 5, 100, || cos(&t).unwrap());
    bench("tanh [1000,1000]", 5, 100, || tanh(&t).unwrap());

    // 8. Reduction ops with axis
    println!("\n--- Reduction Ops (with axis) ---");
    let r = rand::<f32>(&[1000, 1000])?;
    bench("sum_all [1000,1000]", 5, 100, || r.sum_all().unwrap());
    bench("sum dim=0 [1000,1000]", 5, 100, || {
        sum_dim(&r, 0, false).unwrap()
    });
    bench("mean dim=1 [1000,1000]", 5, 100, || {
        mean_dim(&r, 1, false).unwrap()
    });

    // 9. Tensor manipulation ops
    println!("\n--- Tensor Manipulation ---");
    let m = rand::<f32>(&[1000, 1000])?;
    bench("permute [1000,1000]", 5, 100, || {
        permute_t(&m, &[1, 0]).unwrap().contiguous().unwrap()
    });
    bench("chunk [1000,1000] into 4", 5, 100, || {
        chunk_t(&m, 4, 0).unwrap()
    });
    let chunks: Vec<Tensor<f32>> = chunk_t(&m, 4, 0)?;
    bench("cat [4x 250,1000]", 5, 100, || cat(&chunks, 0).unwrap());

    // 10. GRU forward pass
    println!("\n--- GRU Forward ---");
    let gru = GRU::new(128, 256)?;
    let x_gru = rand::<f32>(&[16, 32, 128])?;
    bench("GRU forward (128->256, seq=32, B=16)", 5, 50, || {
        gru.forward(&x_gru, None).unwrap()
    });

    // 11. Larger MLP (784->512->256->10)
    println!("\n--- Larger MLP (784->512->256->10, B=128) ---");
    let mlp_large = Sequential::new(vec![
        Box::new(Linear::new(784, 512, true)?),
        Box::new(ReLU::default()),
        Box::new(Linear::new(512, 256, true)?),
        Box::new(ReLU::default()),
        Box::new(Linear::new(256, 10, true)?),
    ]);
    let x_large = rand::<f32>(&[128, 784])?;
    bench("MLP forward B=128 (784->512->256->10)", 5, 50, || {
        mlp_large.forward(&x_large).unwrap()
    });
    bench("MLP backward B=128", 3, 30, || {
        let x = rand::<f32>(&[128, 784]).unwrap().requires_grad_(true);
        let out = mlp_large.forward(&x).unwrap();
        let loss = out.sum_all().unwrap();
        loss.backward().unwrap();
    });
    let mut opt_large = Adam::new(
        mlp_large.parameters().into_iter().cloned().collect(),
        AdamConfig::default(),
    );
    let loss_fn_large = CrossEntropyLoss::new(Reduction::Mean, 0.0);
    bench("training step B=128", 3, 30, || {
        let x = rand::<f32>(&[128, 784]).unwrap();
        let target = from_vec((0..128).map(|i| (i % 10) as f32).collect(), &[128]).unwrap();
        opt_large.zero_grad().unwrap();
        let out = mlp_large.forward(&x).unwrap();
        let loss = loss_fn_large.forward(&out, &target).unwrap();
        loss.backward().unwrap();

        let model_params: Vec<&Parameter<f32>> = mlp_large.parameters();
        let opt_params = &opt_large.param_groups()[0].params;
        for (mp, op) in model_params.iter().zip(opt_params.iter()) {
            if let Some(g) = mp.grad().unwrap() {
                op.set_grad(Some(g)).unwrap();
            }
        }
        opt_large.step().unwrap();
    });

    // 12. Conv2d forward
    println!("\n--- Conv2d Forward ---");
    let conv = Conv2d::<f32>::new(3, 16, (3, 3), (1, 1), (0, 0), true)?;
    let x_conv = rand::<f32>(&[32, 3, 32, 32])?;
    bench("Conv2d forward [32,3,32,32]->[32,16,30,30]", 5, 50, || {
        conv.forward(&x_conv).unwrap()
    });

    // 13. Broadcast operations
    println!("\n--- Broadcast Ops ---");
    let a_bc = rand::<f32>(&[1000, 1])?;
    let b_bc = rand::<f32>(&[1, 1000])?;
    bench("broadcast add [1000,1]+[1,1000]", 5, 100, || {
        (&a_bc + &b_bc).unwrap()
    });
    let a_bc3 = rand::<f32>(&[64, 1, 256])?;
    let b_bc3 = rand::<f32>(&[1, 128, 1])?;
    bench("broadcast mul [64,1,256]*[1,128,1]", 5, 100, || {
        (&a_bc3 * &b_bc3).unwrap()
    });

    // 14. Creation ops
    println!("\n--- Creation Ops (like) ---");
    let tpl = rand::<f32>(&[1000, 1000])?;
    bench("zeros_like [1000,1000]", 5, 100, || {
        zeros_like(&tpl).unwrap()
    });
    bench("randn_like [1000,1000]", 5, 100, || {
        randn_like(&tpl).unwrap()
    });

    // ========== GPU Benchmarks ==========
    println!("\n{}", "=".repeat(60));
    println!("GPU BENCHMARKS");
    println!("{}", "=".repeat(60));

    if ferrotorch_core::gpu_dispatch::gpu_backend().is_none() {
        // Try to initialize the GPU backend.
        #[cfg(feature = "gpu")]
        {
            let _ = ferrotorch_gpu::init_cuda_backend();
        }
    }

    if ferrotorch_core::gpu_dispatch::gpu_backend().is_some() {
        // Tensor creation on GPU
        println!("\n--- GPU Tensor Creation ---");
        bench("GPU zeros [1000,1000]", 5, 100, || {
            let t = zeros::<f32>(&[1000, 1000]).unwrap();
            t.cuda().unwrap()
        });
        bench("GPU rand [1000,1000]", 5, 100, || {
            let t = rand::<f32>(&[1000, 1000]).unwrap();
            t.cuda().unwrap()
        });

        // Elementwise on GPU
        println!("\n--- GPU Elementwise Ops ---");
        let ga = rand::<f32>(&[1000, 1000])?.cuda()?;
        let gb = rand::<f32>(&[1000, 1000])?.cuda()?;
        bench("GPU add [1000,1000]", 10, 100, || (&ga + &gb).unwrap());
        bench("GPU mul [1000,1000]", 10, 100, || (&ga * &gb).unwrap());
        bench("GPU sub [1000,1000]", 10, 100, || (&ga - &gb).unwrap());
        bench("GPU div [1000,1000]", 10, 100, || (&ga / &gb).unwrap());

        // Unary on GPU
        println!("\n--- GPU Unary Ops ---");
        let gt = rand::<f32>(&[1000, 1000])?.cuda()?;
        bench("GPU relu [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::activation::relu(&gt).unwrap()
        });
        bench("GPU sigmoid [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::activation::sigmoid(&gt).unwrap()
        });
        bench("GPU tanh [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::activation::tanh(&gt).unwrap()
        });
        bench("GPU exp [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::transcendental::exp(&gt).unwrap()
        });
        bench("GPU log [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::transcendental::log(&gt).unwrap()
        });
        bench("GPU neg [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::arithmetic::neg(&gt).unwrap()
        });

        // Matmul on GPU — use the GPU backend directly
        println!("\n--- GPU Matrix Multiply ---");
        for &size in &[64usize, 256, 1024, 4096] {
            let ga = rand::<f32>(&[size, size])?.cuda()?;
            let gb = rand::<f32>(&[size, size])?.cuda()?;
            let iters = if size >= 4096 { 20 } else { 100 };
            let backend = ferrotorch_core::gpu_dispatch::gpu_backend().unwrap();
            bench(&format!("GPU matmul [{size},{size}]"), 10, iters, || {
                backend
                    .matmul_f32(
                        ga.gpu_handle().unwrap(),
                        gb.gpu_handle().unwrap(),
                        size,
                        size,
                        size,
                    )
                    .unwrap()
            });
        }

        // Reduction on GPU
        println!("\n--- GPU Reduction Ops ---");
        let gr = rand::<f32>(&[1000, 1000])?.cuda()?;
        bench("GPU sum_all [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::reduction::sum(&gr).unwrap()
        });
        bench("GPU sum dim=0 [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::reduction::sum_dim(&gr, 0, false).unwrap()
        });
        bench("GPU mean [1000,1000]", 10, 100, || {
            ferrotorch_core::grad_fns::reduction::mean(&gr).unwrap()
        });

        // Forward/backward on GPU
        println!("\n--- GPU Forward/Backward (MLP 784->256->10) ---");
        let mut mlp_gpu = Sequential::new(vec![
            Box::new(Linear::<f32>::new(784, 256, true)?),
            Box::new(ReLU::new()),
            Box::new(Linear::<f32>::new(256, 10, true)?),
        ]);
        mlp_gpu.to_device(ferrotorch_core::device::Device::Cuda(0))?;

        let x_gpu = rand::<f32>(&[32, 784])?.cuda()?;
        bench("GPU MLP forward B=32", 10, 100, || {
            mlp_gpu.forward(&x_gpu).unwrap()
        });

        // Host<->Device transfer
        println!("\n--- Host <-> Device Transfer ---");
        let h = rand::<f32>(&[1000, 1000])?;
        let d = h.cuda()?;
        bench("CPU->GPU [1000,1000]", 5, 100, || h.cuda().unwrap());
        bench("GPU->CPU [1000,1000]", 5, 100, || d.cpu().unwrap());

        // Softmax/LayerNorm on GPU
        println!("\n--- GPU Normalization ---");
        let gn = rand::<f32>(&[64, 256])?.cuda()?;
        bench("GPU softmax [64,256]", 10, 100, || {
            ferrotorch_core::grad_fns::activation::softmax(&gn).unwrap()
        });
    } else {
        println!("\n  (no GPU backend available — skipping GPU benchmarks)");
    }

    println!("\n{}", "=".repeat(60));
    println!("Done.");
    Ok(())
}
