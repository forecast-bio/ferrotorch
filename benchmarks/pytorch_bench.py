#!/usr/bin/env python3
"""PyTorch performance benchmarks for comparison with ferrotorch."""

import time
import torch
import torch.nn as nn

def bench(name, fn, warmup=5, iters=100):
    """Run a benchmark and return average time in microseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1e6  # microseconds
    print(f"  {name}: {elapsed:.1f} us")
    return elapsed

def main():
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

    print(f"PyTorch {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if device_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {}

    # ========== CPU Benchmarks ==========
    print("=" * 60)
    print("CPU BENCHMARKS")
    print("=" * 60)

    # 1. Tensor creation
    print("\n--- Tensor Creation ---")
    results["cpu_zeros_1000x1000"] = bench("zeros [1000,1000]", lambda: torch.zeros(1000, 1000))
    results["cpu_rand_1000x1000"] = bench("rand [1000,1000]", lambda: torch.rand(1000, 1000))

    # 2. Elementwise ops
    print("\n--- Elementwise Ops ---")
    a = torch.rand(1000, 1000)
    b = torch.rand(1000, 1000)
    results["cpu_add_1000x1000"] = bench("add [1000,1000]", lambda: a + b)
    results["cpu_mul_1000x1000"] = bench("mul [1000,1000]", lambda: a * b)
    results["cpu_relu_1000x1000"] = bench("relu [1000,1000]", lambda: torch.relu(a))
    results["cpu_sigmoid_1000x1000"] = bench("sigmoid [1000,1000]", lambda: torch.sigmoid(a))

    # 3. Matrix multiply
    print("\n--- Matrix Multiply ---")
    for size in [64, 256, 1024]:
        a = torch.rand(size, size)
        b = torch.rand(size, size)
        iters = 100 if size <= 256 else 20
        results[f"cpu_matmul_{size}x{size}"] = bench(f"matmul [{size},{size}]", lambda: a @ b, iters=iters)

    # 4. Forward pass (MLP)
    print("\n--- Forward Pass (MLP 784->256->10) ---")
    mlp = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    x_mlp = torch.rand(32, 784)
    results["cpu_mlp_fwd_b32"] = bench("MLP forward B=32", lambda: mlp(x_mlp))

    # 5. Backward pass
    print("\n--- Backward Pass (MLP 784->256->10) ---")
    def mlp_backward():
        x = torch.rand(32, 784, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
    results["cpu_mlp_bwd_b32"] = bench("MLP backward B=32", mlp_backward, iters=50)

    # 6. Full training step
    print("\n--- Full Training Step (MLP + Adam) ---")
    optimizer = torch.optim.Adam(mlp.parameters())
    loss_fn = nn.CrossEntropyLoss()
    def training_step():
        x = torch.rand(32, 784)
        target = torch.randint(0, 10, (32,))
        optimizer.zero_grad()
        out = mlp(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    results["cpu_train_step_b32"] = bench("training step B=32", training_step, iters=50)

    # ========== GPU Benchmarks ==========
    if device_gpu:
        print("\n" + "=" * 60)
        print("GPU BENCHMARKS")
        print("=" * 60)

        # 1. Tensor creation
        print("\n--- Tensor Creation ---")
        results["gpu_zeros_1000x1000"] = bench("zeros [1000,1000]", lambda: torch.zeros(1000, 1000, device="cuda"))
        results["gpu_rand_1000x1000"] = bench("rand [1000,1000]", lambda: torch.rand(1000, 1000, device="cuda"))

        # 2. Elementwise
        print("\n--- Elementwise Ops ---")
        a_g = torch.rand(1000, 1000, device="cuda")
        b_g = torch.rand(1000, 1000, device="cuda")
        results["gpu_add_1000x1000"] = bench("add [1000,1000]", lambda: a_g + b_g)
        results["gpu_mul_1000x1000"] = bench("mul [1000,1000]", lambda: a_g * b_g)
        results["gpu_relu_1000x1000"] = bench("relu [1000,1000]", lambda: torch.relu(a_g))

        # 3. Matmul
        print("\n--- Matrix Multiply ---")
        for size in [64, 256, 1024, 4096]:
            a_g = torch.rand(size, size, device="cuda")
            b_g = torch.rand(size, size, device="cuda")
            results[f"gpu_matmul_{size}x{size}"] = bench(f"matmul [{size},{size}]", lambda: a_g @ b_g)

        # 4. Transfer
        print("\n--- Host <-> Device Transfer ---")
        t_cpu = torch.rand(1000, 1000)
        t_gpu = t_cpu.cuda()
        results["gpu_h2d_1000x1000"] = bench("CPU->GPU [1000,1000]", lambda: t_cpu.cuda())
        results["gpu_d2h_1000x1000"] = bench("GPU->CPU [1000,1000]", lambda: t_gpu.cpu())

        # 5. Forward/backward
        print("\n--- Forward/Backward (MLP on GPU) ---")
        mlp_g = mlp.cuda()
        x_g = torch.rand(32, 784, device="cuda")
        results["gpu_mlp_fwd_b32"] = bench("MLP forward B=32", lambda: mlp_g(x_g))

        def mlp_backward_gpu():
            x = torch.rand(32, 784, device="cuda", requires_grad=True)
            out = mlp_g(x)
            loss = out.sum()
            loss.backward()
        results["gpu_mlp_bwd_b32"] = bench("MLP backward B=32", mlp_backward_gpu, iters=50)

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY (all times in microseconds)")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.1f} us")

if __name__ == "__main__":
    main()
