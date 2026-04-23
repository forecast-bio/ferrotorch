# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-04-08

### Fixed
- reduce_grad_to_shape underflow when grad_ndim < target_ndim (#498)
- Fix permute benchmark to include contiguous copy for apples-to-apples PyTorch comparison (#409)
- Fix unused import warnings in ferrotorch-optim (#194)
- Fix dead code warnings properly — wire unused is_f64 and unwrap_buffer_f64_mut into dispatch paths (#493)
- Fix dead code properly — wire is_f64 into remaining dispatch paths, restore binary_map import, remove all allow(dead_code) suppressions (#494)
- Fix div() broadcast path panic on GPU tensors (#407)
- Fix LinearFusedBackward GPU path returning wrong gradient count when bias is None (#411)
- Fix incomplete cache miss tracking in cpu_pool thread pool (#471)
- Fix unsigned hub commits blocking crosslink sync (#413)
- T1.1: GPU gradient accumulation — use backend.add_f32 instead of CPU roundtrip (#259)
- Add GPU dispatch for div, exp, log, sqrt, pow, abs elementwise ops (#218)
- Fix into_storage_and_shape panic on shared GPU tensors (#216)
- QA review: audit all Tier 3 changes for bugs, lazy shortcuts, and correctness issues (#286)
- QA review: audit all Tier 2 changes for bugs, lazy shortcuts, and correctness issues (#274)
- QA review: audit all Tier 1 changes for lazy shortcuts, bugs, and correctness issues (#266)
- Fix shape ops breaking GPU autograd — upload to device before from_operation, not after (#247)
- Fix Conv2d, Conv1d, and all pooling layers to work on GPU (#221)
- Fix GPU backend f64 path — hardcoded f32 in cpu_to_gpu, gpu_to_cpu, clone_buffer (#238)
- Fix batched matmul and broadcast matmul backward crash on GPU (#228)
- Fix einsum and einops crash on GPU tensors (#226)
- Fix checkpoint, higher_order grad, grad_penalty, fixed_point crash on GPU (#227)
- Fix all probability distributions crash on GPU tensors (#234)
- Fix vision models GPU crashes — ViT, Swin, ConvNeXt, UNet (#235)
- Fix Dropout2d and functional dropout crash on GPU (#240)
- Fix LSTM autograd graph severed by from_storage leaf creation (#222)
- Fix FlashAttention, RoPE, KVCache, SwiGLU crash on GPU (#229)
- Fix FFT operations crash on GPU tensors (#230)
- Fix JIT fusion engine crash on GPU tensors (#237)
- Fix permute, split, chunk, where crash on GPU tensors (#225)
- Fix distributed collective ops and DDP crash on GPU tensors (#236)
- Fix serialization save crash on GPU tensors (#233)
- Fix KFAC optimizer and GradScaler GPU crashes (#232)
- Fix all loss functions to work on GPU tensors (#220)
- Fix gradient clipping utilities crash on GPU tensors (#224)
- Fix in-place ops crash on GPU — add_scalar_, mul_scalar_, fill_, clamp_ (#223)
- Fix backward device restore in GroupNorm, RMSNorm, Softplus, reduce_grad (#231)
- Fix tanh, silu, softplus returning CPU tensors when GPU input has grad (#219)
- Fix into_storage_and_shape panic on shared GPU tensors (#217)
- Fix optimizer step on GPU parameters — add in-place GPU write path (#215)
- Fix Tensor::to() to preserve autograd graph across device transfers (#214)
- Fix Embedding backward to produce weight gradients on GPU (#213)
- Fix bmm_differentiable GPU crash from .data() on GPU tensors (#212)
- Fix view/reshape on GPU tensors dropping requires_grad and breaking autograd graph (#211)
- Fix index_select and masked_fill to use proper GPU kernels instead of CPU fallback (#210)
- Update rustls-webpki to 0.103.10 (#204)
- Fix PTX register name collision (`%tid` → `%r_tid`) — all elementwise kernels were silently falling back to CPU due to `CUDA_ERROR_INVALID_PTX`
- Fix softmax PTX: wrong hex prefix for float literals (`0xff` → `0f`), undeclared shared memory registers (`%saddr`, `%sbase`)
- Fix CUDA graph capture on legacy default stream — fork non-blocking stream via `GpuDevice::fork_for_capture()`

### Added
- Sharded safetensors loader (`model.safetensors.index.json` + multi-file) (#507). New `load_safetensors_sharded(index_path)` in `ferrotorch-serialize` parses the HF index, groups tensor names by shard, loads each shard sequentially, and assembles a single `StateDict<T>`. Shards are loaded in filename-sorted order for determinism; tensors present in a shard but not listed in the index are skipped; index entries pointing at absent tensors produce a clear error. `load_safetensors_auto(path)` dispatches on the filename suffix (`.index.json` → sharded, else single-file). Public `SafeTensorsIndex` struct with `from_file`, `shard_files()`, `group_by_shard()`. The bf16/f16 → f32 upcast path is now shared between single-file and sharded loaders via a new `decode_view` helper. 7 new unit tests (sharded merge, index-authoritative skip, missing tensor in shard, missing shard file, malformed JSON, index accessors, auto dispatch) plus one `#[ignore]`-gated end-to-end test that loads the real Meta-Llama-3-8B checkpoint (4 shards, ~16GB bf16) and asserts all 291 tensors are present with correct shapes (including the `[1024, 4096]` K/V projections that confirm GQA).
- HuggingFace `config.json` parser (#508). New `ferrotorch-hub::hf_config::HfTransformerConfig` serde struct parses the flat config.json shape emitted by HF transformers (decoder-only models: Llama, Mistral, Gemma, Qwen, Falcon, …). Fields cover the Llama 3 superset with safe defaults for optional ones (`rms_norm_eps` default 1e-6, `rope_theta` default 10000.0, `hidden_act` default "silu"). Derived accessors: `num_key_value_heads()` (falls back to `num_attention_heads` when absent), `head_dim()`, `is_gqa()`. `validate()` enforces positive counts, `hidden_size % num_attention_heads == 0`, `num_attention_heads % num_kv_heads == 0`, and known activation name. `from_json_str` and `from_file` entry points; unknown fields are silently ignored so configs from newer HF versions still parse. `serde`/`serde_json` moved out of the `http` feature so the parser is available on no-network builds. 14 tests including parity against the exact Llama 3 8B `config.json`.
- `KVCache` supports `n_kv_heads != num_heads` for GQA (#506). KVCache now tracks `CacheDims {batch, num_kv_heads, head_dim}` — inferred on first update or pre-declared via `KVCache::with_dims(max_seq_len, batch, num_kv_heads, head_dim)`. Every subsequent update is validated against the pinned dims, rejecting mismatches immediately. Cache stays at KV-head granularity (~1/4 size for Llama 3 8B vs. storing at Q-head granularity); `repeat_kv` happens at read time inside attention. Public `num_kv_heads()`, `head_dim()`, `batch_size()` getters. `reset()` preserves pinned dims. 10 new tests cover Llama 3 shape, pre-declaration, first/subsequent-update mismatch rejection, reset-preserves-dims, and a position-by-position prefill-then-decode correctness check across [B=1, H=8, S=5, D=16].
- Grouped-Query Attention in `MultiheadAttention` (#505). Adds `MultiheadAttention::with_gqa(embed_dim, num_heads, num_kv_heads, bias)`; `new()` forwards to it with `num_kv_heads = num_heads`, preserving classical MHA. K/V projections are sized `[num_kv_heads * head_dim, embed_dim]` (1/4 the weights for Llama 3 8B's 32:8 ratio) and a new `repeat_kv` helper broadcasts KV heads up to Q-head count before the attention matmul. `group_size == 1` is a fast no-op clone so MHA pays nothing. Public `num_kv_heads()` and `is_gqa()` getters. 12 new tests cover Llama 3 8B layout, k/v_proj shapes, divisibility / zero-kv error paths, MHA equivalence when kv==q, `repeat_kv` head-copy correctness, and forward paths (standard, autoregressive single-token, causal-masked).
- Intel XPU backend (#452). Adds a new `Device::Xpu(usize)` variant to `ferrotorch-core` (with `is_xpu()` query, `Display` "xpu:N", and `Tensor::to(Device::Xpu(_))` zero-copy retag paths for CPU↔XPU plus round-trip-via-CPU XPU↔CUDA / cross-XPU-ordinal). Introduces a new `ferrotorch-xpu` crate that wraps a `ferrotorch_cubecl::CubeRuntime` configured for the wgpu backend (which targets Intel Arc / Data Center GPU Max via Vulkan). `XpuDevice::new(0)` initialises the runtime and 15 ops (`xpu_add`/`sub`/`mul`/`div`/`matmul`, `xpu_neg`/`abs`/`relu`/`exp`/`ln`/`sqrt`/`sin`/`cos`/`tanh`/`sigmoid`) upload XPU-tagged tensors through the cubecl `portable_*` API, run a real `#[cube]` kernel on the GPU, and return the result tagged back as XPU storage. Each op validates that both inputs live on the matching XPU ordinal and rejects mismatched device tensors with `DeviceMismatch`. 9 new tests pass on the wgpu/Vulkan path including matmul, transcendentals, and device-mismatch error paths.
- CUDA graph maturity pass (#454). Expands `ferrotorch-gpu::graph` from the bare begin/end/launch surface to a PyTorch-parity API: `CaptureMode::{Global,ThreadLocal,Relaxed}` typed wrapper over `CUstreamCaptureMode` with a default matching PyTorch's `thread_local`; `CaptureStatus::{None,Active,Invalidated}` typed wrapper with `is_capturing()`/`is_invalidated()` queries; `capture_status(stream)` / `is_stream_capturing(stream)` mirroring `torch.cuda.is_current_stream_capturing`; `begin_capture_with_mode` for Global/Relaxed capture; `CapturedGraph::upload()` (requires cudarc 0.19.4, bumped the workspace lock) for one-time pre-upload of the exec into device memory so the first `launch()` doesn't pay setup cost, plus `is_uploaded()` caching; `CapturedGraph::num_replays()` atomic counter bumped on every successful launch; `CapturedGraph::pool()` accessor; `GraphCaptureGuard` RAII type with `begin`/`begin_with_mode`/`begin_with_pool`/`finish` that auto-ends capture on drop so a mid-capture error can't leave the stream stuck; a process-wide graph pool handle registry (`graph_pool_handle()` / `capture_pool_for_handle(h)` / `release_graph_pool_handle(h)`) mirroring `torch.cuda.graph_pool_handle()` so multiple graphs can share the same buffer-keeping `CapturePool`; and a `make_graphed_callable(stream, mode, f)` helper that captures a closure into a replayable graph (PyTorch's `make_graphed_callables` for the single-callable case) with automatic guard cleanup on error. 9 new tests pass on the simulated-test path; CUDA-feature parity path compiles clean on cudarc 0.19.4.
- Add ShardGradOp, NoShard, HybridShard strategies and backward prefetch to FSDP (#327)
- FSDP HybridShard strategy + subgroup API (#327). Adds `ferrotorch_distributed::SubBackend`, a `Backend` trait impl that wraps a parent `Backend` + a list of member global ranks and maps local subgroup indices to global ranks on every `send`/`recv`/`barrier`. Because `SubBackend` implements `Backend`, the existing `all_gather` / `reduce_scatter` / `allreduce` collectives run on a subgroup without any changes. The new `ShardingStrategy::HybridShard { intra_node_size }` builds two `SubBackend`s in `FSDP::new_with_strategy`: an intra-node group (contiguous blocks of `intra_node_size` ranks) for FullShard-style parameter sharding, and an inter-node group (every `intra_node_size`-th rank) for DDP-style gradient replication. `forward` all-gathers shards within the node only; `sync_gradients` does an intra-node reduce_scatter + inter-node allreduce to give every replica of each intra-rank the same shard gradient. 4 new HybridShard FSDP tests and 6 SubBackend tests on the simulated backend verify intra-only sharding, intra-group gradient reduction, and inter-node averaging across a 4-rank (2 nodes × 2 local) topology with divergent per-node gradients.
- Add ferrotorch-gpt2 standalone project — GPT-2 124M implementation (#193)
- Add foreach (on-device) update path to remaining optimizers (#497)
- Expanded foreach (on-device) update path to the common optimizers in `ferrotorch-optim` (#497). Beyond the existing `sgd` and `adamw`, this adds a `foreach: bool` config field and a `step_foreach`/`update_foreach` method to: `adam` (including AMSGrad), `adagrad`, `rmsprop` (including centered + momentum), `adamax`, `nadam`, `radam` (including the rho_t > 5 rectified branch), `adadelta`, `asgd`, `ema::ExponentialMovingAverage`, and `swa::AveragedModel` (SWA + EMA strategies). Each foreach path keeps per-parameter state as `Tensor<T>` on the parameter's native device and computes the update via runtime-generic tensor ops, eliminating the per-step CPU↔GPU round-trip the legacy `Vec<f64>` path incurred. A new shared `ferrotorch_optim::foreach_utils` module provides `elemwise_max(a, b, device)` (computed as `0.5 * (a + b + |a - b|)` using the existing abs grad_fn) for optimizers that need on-device elementwise max (AMSGrad, Adamax infinity norm). Each new foreach path is backed by parity tests comparing legacy vs foreach output across hyperparameter variants. 27 new foreach parity tests total; remaining out-of-scope optimizers (rprop sign-based, adafactor factored shapes, muon Newton-Schulz orthogonalization, lbfgs line search, sparse_adam sparse semantics, natural_gradient K-FAC, grad_scaler/grad_accumulator utilities) are explicitly documented as not applicable to the foreach pattern.
- Expanded `ferrotorch-cubecl` op coverage: added real CubeCL `#[cube]` kernels for `div`, `neg`, `abs`, `exp`, `ln`, `sqrt`, `sin`, `cos`, `tanh`, and `sigmoid` alongside the existing add/sub/mul/relu/matmul, each wired through `run_*` launchers (now macro-generated for brevity) and `portable_*` entry points on `CubeRuntime`. Adds 13 new GPU tests on wgpu including an `exp`→`ln` round-trip identity check, a 1024-element sigmoid numerical-stability check across multiple cubes, and shape-validation for `portable_div`. (#453)
- CubeCL actual GPU kernels — currently all ops fall back to CPU despite abstraction (#398)
- Real CubeCL GPU kernels in `ferrotorch-cubecl`: the crate now dispatches `#[cube]` kernels (`kernel_add`, `kernel_sub`, `kernel_mul`, `kernel_relu`, `kernel_matmul_naive`) through a real `ComputeClient` per backend. `CubeRuntime::new` constructs the matching `CubeClient::{Wgpu,Cuda,Rocm}` enum variant from `cubecl-wgpu` / `cubecl-cuda` / `cubecl-hip`, and `portable_add/sub/mul/relu/matmul` upload inputs, launch the kernel via runtime-generic `run_*` helpers, and read results back. No more CPU fallback — without a backend feature `CubeRuntime::new` returns `DeviceUnavailable`. Verified end-to-end against wgpu with 12 new GPU tests including `portable_matmul_square_8x8` and `portable_add_large_shape` (1024 elements across multiple cubes). (#398)
- T4.3 Inductor-style codegen (Triton/C++ backends) (#290)
- Inductor CpuC JIT execution: `InductorBackend::compile` with `InductorTarget::CpuC` now really compiles generated C source into a shared library via the system C compiler (`cc`/`gcc`/`clang`, or `$CC`), loads it with `libloading`, and dispatches to the native kernel on every `execute` call — replacing the previous interpreter fallback. Single-fusion-group elementwise graphs (including ones with constant inputs) are fully JIT'd; mixed graphs with reductions/matmul/etc. still fall back to the interpreter. A global hash-keyed compile cache returns the same `Arc<JitCompiledKernel>` for byte-identical source, so repeated compiles are O(1). (#290)
- Semi-structured 2:4 sparsity: `SemiStructuredSparseTensor<T>` with compressed values + 4-bit-per-group mask storage, `compress`/`decompress` round-trip, deterministic tie-breaking, and a `sparse_matmul_24(a, b)` reference implementation. Matches the NVIDIA Sparse Tensor Core format for Ampere+ hardware. (#292)
- Differentiable QAT: `fake_quantize_differentiable(tensor, scale, zero_point, qmin, qmax)` integrates fake-quantization into the autograd engine with clipped straight-through-estimator backward, so models can train end-to-end through simulated quantization noise (#293)
- Dispatch key system: `DispatchKey` enum (Cpu/Cuda/Meta/Sparse/Quantized/Nested/Autocast/Autograd/Vmap/Profiler/Tracer), `DispatchKeySet` bitmask with priority iteration, and a `Dispatcher<T>` kernel registration table with `call` (priority resolution + redispatch) and `call_direct` (bypass for testing). Enables composable sparse/quantized/autograd/tracer layers. (#397)
- PackedNestedTensor: flat packed storage + offsets layout for nested/jagged tensors with elementwise map/add/sub/mul/div, per-component sum/mean reductions, to_padded/from_padded conversion, and roundtrip with the existing list-of-tensors NestedTensor (#291)
- AOT autograd: `decompose_forward_backward` now emits real backward IR nodes for Add/Sub/Mul/Neg/Relu/Sum/Mean (replacing the previous no-op pass-through), with grad accumulation, deterministic saved-tensor ordering, and zero-constant fallback for unused inputs (#289)
- channels_last memory format on CUDA: `Tensor::contiguous_in(MemoryFormat::ChannelsLast{,3d})` now dispatches to `gpu_strided_copy` with permuted shape+stride parameters, keeping the conversion entirely on-device instead of round-tripping through CPU memory (#455)
- CUDA graph allocator pool: `CapturePool::record_buffer` registers GPU buffers for lifetime extension and `end_capture_with_pool` produces a `CapturedGraph` that holds an `Arc<CapturePool>`, keeping recorded buffers alive across replays (#278)
- Profiler CUDA event timing: `CudaKernelScope` records start/end CUDA events around a region and `Profiler::flush_cuda_kernels()` synchronizes and converts them to `ProfileEvent`s with real GPU-measured `cuEventElapsedTime` durations, replacing the CPU wall-clock fallback for async kernels (#380)
- CUDA stream priority levels: `StreamPriority::{High,Normal,Low}`, `get_stream_priority_range`, `new_stream_with_priority`, and `StreamPool::get_priority_stream` for round-robin priority pools per device (#322)
- GPU `strided_copy` primitive: PTX kernel and backend dispatch for N-d strided→contiguous gather entirely on-device, wired into `Tensor::contiguous()` so non-contiguous CUDA tensors no longer roundtrip through CPU (#496)
- torch.export-style runtime guards: `ExportedProgram::check_inputs` and `run_with_guards` validate runtime inputs against `input_specs` (static dim match, dynamic dim range) before graph execution (#461)
- Profiler auto-instrumentation extended to div/neg/pow/sqrt/abs, exp/log/sin/cos, mean/prod/sum_dim/mean_dim, and relu/sigmoid/tanh/gelu/silu/log_softmax — trace output now covers the core tensor op surface area (#501)
- Add Claude skill with ferrotorch API usage hints (#192)
- Distributed collectives: `all_to_all`, `all_to_all_single_uneven`, and `reduce_scatter_tensor` matching PyTorch's dist API (#460)
- IntermediateFeatures impls for all vision models: VGG, ViT, EfficientNet, ConvNeXt, Swin, U-Net, YOLO, MobileNetV2/V3, DenseNet-121, Inception v3. Trait now returns `Vec<String>` so architectures with variable block counts can expose dynamic per-block node names (#499)
- ExportedProgram binary save/load roundtrip (`.ftep` format) preserving graph, state_dict, input_shapes, input_specs, and output_shape (#296)
- Vision models: MobileNetV2, MobileNetV3-Small, DenseNet-121, Inception v3, each registered in both `ferrotorch_vision::models::REGISTRY` and `ferrotorch_hub::registry` (#436)
- Symbolic shapes for export: `DimSpec` (Static/Dynamic), `InputSpec`, `export_with_dynamic_shapes`, and automatic forwarding of dynamic axes from `ExportedProgram.input_specs` into ONNX `dim_param` output (#396)
- ONNX exporter: decompose Silu (Sigmoid+Mul) and Gelu (Div+Erf+Add+Mul+Mul via erf formula) into standard ONNX ops, re-enable `export_from_program` on the current ExportedProgram API (#375)
- DataLoader cross-batch worker pipeline: `WorkerMode::CrossBatch` spawns `num_workers` dedicated threads each producing independent batches, with a reorder buffer to preserve sampler order (#377)
- JIT kernel autotuning: `Autotuner` benchmarks candidate codegen backends/configs and caches the winner keyed by graph fingerprint + input shapes (#369)
- JIT symbolic shapes with guards: `SymbolicTracedModule`, `ShapeSignature`, and `compile_symbolic` for dynamic batch sizes with runtime validation and reshape patching (#367)
- FSDP backward prefetch: `prefetch_forward_params()` + async all-gather handles for overlapping collectives with compute (#373)
- FSDP SHARD_GRAD_OP (ZeRO-2) and NoShard (ZeRO-0/DDP) sharding strategies, with `broadcast_updated_params` re-sync hook (#372)
- Expand KL divergence registry — cross-family pairs (Normal-Laplace, Gamma-Exponential, etc.) (#365)
- Optimizer differentiable mode — autograd through optimizer step for meta-learning (#389)
- Profiler TensorBoard export integration (#381)
- Hub dynamic model discovery from HuggingFace API (#383)
- Add profiler scheduling, memory categories, FLOPS estimation, and stack traces (#333)
- WU-08: vmap composability — multi-arg/output, matmul batching rule, per_sample_grad (#362)
- Fix vmap composability and add batching rules for matmul, elementwise ops (#312)
- Meta device: propagate through arithmetic and linalg ops for full shape inference (#500)
- JIT multi-output graph support (currently single output only) (#368)
- Profiler auto-instrumentation of tensor ops (currently manual record() calls only) (#379)
- Meta device — dry-run tensor allocation for shape inference without data (#395)
- Vision feature extraction — return intermediate layer outputs, not just final (#384)
- Vision pretrained weight auto-download (registry returns false for all models) (#385)
- Hub auto-download with reqwest HTTP client + SHA-256 verification (#382)
- Add SyncBatchNorm — distributed batch normalization across GPUs (#392)
- DataLoader pin_memory for async CPU→GPU transfer (#378)
- Optimizer foreach/fused kernel modes for batched parameter updates (#388)
- Gradient checkpointing multi-tensor input and autocast state preservation (#400)
- Integrate hierarchical-llm-rust as ferrotorch consumer test (#410)
- Add PyTorch export writer, expand dtype support, expand ONNX op coverage (#328)
- WU-07 remainder: integrate anomaly detection into backward engine (check NaN/Inf) (#361)
- Wire autocast context to classify operations via autocast_ops module (#161)
- Expand state_dict dtype support — f16, bf16, i8, i16, i32, i64, u8 (#376)
- Add Compose, RandomHorizontalFlip, RandomCrop, ColorJitter, and augmentation pipeline (#332)
- Add LazyLinear, LazyConv1d/2d — auto-infer input dimensions at first forward (#393)
- Add ChannelShuffle module for ShuffleNet architectures (#394)
- WU-03: nanmean, nansum, logsumexp, erfinv, polygamma, lgamma, digamma (#360)
- Inference mode — faster than no_grad, no view tracking overhead (#356)
- Add special mathematical functions module (torch.special equivalent) to ferrotorch-core (#159)
- Implement LoRA (Low-Rank Adaptation) module in ferrotorch-nn (#175)
- Add nanmean, nansum, logsumexp, erfinv, polygamma, lgamma, digamma (#307)
- Add trunc_normal, orthogonal, sparse init and weight/spectral norm (#318)
- T3.6: GPU profiler CUDA event timing (#280)
- T3.7: Missing nn modules — Identity, Flatten, L1Loss, NLLLoss, BatchNorm1d (#281)
- Add relaxed distributions — RelaxedBernoulli, RelaxedOneHotCategorical (Gumbel-Softmax) (#364)
- Add ExponentialMovingAverage, AveragedModel, SWALR, and maximize flag (#321)
- Add LambdaLR, MultiStepLR, ExponentialLR, CosineWarmRestarts, CyclicLR, OneCycleLR (#320)
- Add Beta, Gamma, Exponential, Laplace, Poisson, LogNormal, StudentT, Gumbel, HalfNormal distributions (#329)
- Add Transform trait, Constraint trait, TransformedDistribution, and KL divergence registry (#330)
- Add MultivariateNormal, Dirichlet, Multinomial, Independent, MixtureSameFamily (#331)
- Add Poisson, LogNormal, StudentT, Gumbel, HalfNormal, Weibull, Cauchy, Chi2 distributions (#363)
- Adafactor optimizer — factorized adaptive learning rate for memory efficiency (#386)
- SparseAdam optimizer — Adam variant for sparse gradients (embedding tables) (#387)
- Add CosineSimilarity and PairwiseDistance distance modules (#402)
- Add Dropout1d, Dropout3d, AlphaDropout variants (#403)
- Expand benchmarks suite for new wave 1-5 features (#188)
- Verify ferrotorch umbrella crate links all sub-crates + update README (#190)
- T1.6: Lower CPU parallel threshold from 2M to 32K elements (#264)
- T3.9: FSDP — parameter sharding with all-gather and reduce-scatter (#283)
- T4.1 Higher-order ops (cond, scan, flex_attention) (#288)
- T4.7 Distributed checkpointing (#294)
- T4.8 RPC and pipeline parallelism (#295)
- T4.10 CUDA RNG state management (#297)
- WU-01 remainder: wire gather/scatter/scatter_add/where_cond into grad_fns properly (#359)
- Add pack_padded_sequence and pad_packed_sequence to ferrotorch-nn rnn_utils (#157)
- Create ferrotorch-hub crate for downloading and caching pretrained model weights (#162)
- Create ferrotorch-profiler crate for operation profiling (#150)
- Add gradient penalty utilities (gradient_penalty, grad_norm, jvp, vjp) to ferrotorch-core autograd (#166)
- Bump all crate versions to 0.1.2 (#191)
- Add tensor gradient hooks and anomaly detection mode (#311)
- Add ReLU6, Hardtanh, LogSigmoid, Softmin, Threshold, shrinkage, and RReLU activations (#313)
- Add BCELoss, TripletMarginLoss, CTCLoss, and other missing losses (#316)
- Add WeightedRandomSampler, composite datasets, default_collate, worker_init_fn (#325)
- T3.8: GPU linalg via cuSOLVER — SVD, Cholesky, LU, QR on GPU (#282)
- NCCL native GPU collective backend (currently CPU-fallback TCP only) (#374)
- Add EmbeddingBag — efficient bag-of-embeddings with sum/mean/max reduction (#390)
- Add RNNCell, LSTMCell, GRUCell — single-timestep manual loop control (#391)
- Add Transformer and TransformerEncoder/Decoder container modules (#401)
- WU-20: GPU kernel expansion — GroupNorm, BatchNorm backward, MaxPool2d, AvgPool2d GPU kernels (#358)
- T3.5: DataLoader prefetch pipeline + pin_memory (#279)
- Multi-threaded backward engine with per-device worker threads and priority queue (#354)
- JIT pattern fusion — fuse_attention, fuse_linear, fuse_conv_bn (3-5x transformer speedup) (#366)
- DDP communication/computation overlap — allreduce during backward (#371)
- DDP gradient bucketing — group params into 25MB buckets for async allreduce (#370)
- SavedVariable hooks — pack/unpack for memory offloading in autograd (#355)
- T3.10: MultiheadAttention batched matmul — eliminate serial batch/head loops (#284)
- T3.2: Checkpoint RNG preservation + multi-tensor input (#276)
- T3.1: JIT multi-input fusion + GPU kernel execution (#275)
- T3.3: GradScaler GPU kernel for fused unscale+inf-check (#277)
- WU-04: In-place tensor ops (add_, mul_, sub_, div_, zero_, fill_) with version counter (#357)
- T2.2: fp16/bf16 kernels and cublasGemmEx for Tensor Cores (#268)
- T2.1: CUDA stream pool with thread-local current stream and events (#267)
- T1.3: Zero-copy stride-based views for transpose/permute/slice (#261)
- T1.2: GPU-resident optimizer state — store exp_avg/exp_avg_sq as GPU tensors (#260)
- Perf Phase 5B: Wire backward GPU kernels — eliminate all CPU roundtrips in backward passes (#255)
- Perf Phase 3C: Fused SIMD sigmoid, sin, cos kernels — eliminate intermediate allocations (#254)
- Perf Phase 3B: Wire fast_sigmoid and fast_tanh into activation forward paths (#253)
- Perf Phase 5: Wire GPU kernels into grad_fns to eliminate CPU roundtrips (#252)
- Perf Phase 4B: Add backward and reduction GPU kernels (#251)
- Perf Phase 4: Add missing GPU kernels via CubeCL — div, exp, log, sqrt, sigmoid, tanh, axis reductions (#250)
- Add Pythia-70M architecture integration test for GPU training validation (#249)
- Add GPU integration test suite — 5 end-to-end training experiments (#248)
- Add CPU tensor buffer pool to eliminate allocation overhead in elementwise ops (#246)
- Perf Phase 3: Rewrite elementwise kernels with pulp SIMD + rayon parallelism (#245)
- Perf Phase 2: Migrate CPU matmul from matrixmultiply to faer GEMM (#242)
- Perf Phase 1: Add data_ref() zero-copy CPU path to eliminate data_vec() copies (#241)
- Perf Phase 8: Add .cargo/config.toml with target-cpu=native (#244)
- Perf Phase 7: Switch global allocator to mimalloc (#243)
- Add GELU approximation modes matching PyTorch (none, tanh) plus existing sigmoid (#205)
- Add GELU approximation modes matching PyTorch (none, tanh) plus existing sigmoid (#205)
- Add GELU approximation modes matching PyTorch (none, tanh) plus existing sigmoid (#205)
- Update differentiable matmul wrappers and backward passes for broadcast (#203)
- Update nn::Linear to accept arbitrary-rank inputs (#201)
- Add batched broadcast matmul for arbitrary-rank tensors (#200)
- **GPU buffer pool** (`pool.rs`): caching allocator that reuses freed `CudaSlice`s by element count, eliminating `cuMemAllocAsync`/`cuMemFreeAsync` per op
- **CUDA graph capture** (`graph.rs`): `DeviceScalar<T>`, `CapturedGraph`, `begin_capture`/`end_capture` API for replaying entire decode passes as a single driver call
- **`_into` kernel variants**: non-allocating versions of all decode-path kernels (add, mul, scale, gelu, layernorm, softmax, permute, embed_lookup, matmul, bmm, slice_read) for pre-allocated output buffers
- **Indirect-parameter PTX kernels**: `slice_write_indirect` and `causal_mask_indirect` read variable parameters (pos, total_len) from device memory for CUDA graph compatibility
- **`scale_f32` PTX kernel**: scalar multiply (`out[i] = a[i] * scalar`) exposed via `GpuBackend::scale_f32()`
- **`GpuBackend::as_any()`**: downcast trait method for backend-specific access
- **`GpuBufferHandle::into_inner()`**: consume handle and extract concrete type
- **`GpuDevice::fork_for_capture()`**: create non-blocking stream for CUDA graph capture
- **`get_cuda_device()`**: retrieve the shared `GpuDevice` from the registered backend
- **`precompile_decode_kernels()`**: pre-compile all decode-path PTX modules before graph capture
- **`CudaBuffer` pool-aware Drop**: returns `CudaSlice` to pool via function pointer dispatch (f32/f64)
- **`GpuError::PtxCompileFailed`** variant for explicit PTX compilation failure reporting
- M≤4 cuBLAS bypass: route vector-matrix multiplies through PTX `small_matmul` kernel instead of cuBLAS SGEMM

### Changed
- HF config.json parser (#508)
- Clean up pre-existing clippy errors (approx_constant, erasing_op) (#517)
- Bump crate versions to 0.3.0 (#503)
- Add XPU backend for Intel GPUs (#452)
- Improve CUDA Graph support to match PyTorch maturity (#454)
- Expand ferrotorch-cubecl with full op coverage (#453)
- Complete Tier 4 gap analysis sections (#287)
- 292 (#502)
- Add vision transforms: GaussianNoise, ElasticTransform, TrivialAugmentWide (#458)
- Implement LazyLinear and LazyConv variants (deferred shape inference) (#445)
- Add missing crate READMEs and update workspace README for crates.io publishing (#177)
- Add relaxed distributions: RelaxedBernoulli, RelaxedOneHotCategorical, OneHotCategorical (#430)
- Add missing distributions: LowRankMultivariateNormal, MixtureSameFamily, Independent (#429)
- Add GPU path for flex_attention (currently downloads Q/K/V to CPU) (#483)
- Add GPU path for einops rearrange/repeat/reduce (currently CPU roundtrip) (#484)
- Add vision transforms: RandomErasing, AutoAugment, RandAugment, AugMix (#437)
- Add missing distributions: Kumaraswamy, LKJCholesky, LogisticNormal, Wishart (#428)
- Add missing distributions: Weibull, VonMises, Gumbel, Pareto, GeneralizedPareto (#426)
- Implement CircularPad1d, CircularPad2d, CircularPad3d (#446)
- Implement PairwiseDistance and CosineSimilarity as nn.Module (#448)
- Implement Unflatten and ChannelShuffle modules (#449)
- Implement ParameterList and ParameterDict containers (#447)
- Implement ReLU6 and Softmax2d activation modules (#450)
- randn_like is 6x slower than PyTorch — needs optimized Box-Muller or Ziggurat RNG (#346)
- Implement missing tensor ops: stft, istft, triu, tril, diag, diagflat, roll, cdist (#442)
- Implement missing tensor ops: histc, histogram, meshgrid, multinomial (#441)
- Implement missing tensor ops: searchsorted, bucketize, unique, unique_consecutive (#440)
- Implement SparseAdam and Adafactor optimizers (#438)
- Add missing distributions: HalfNormal, HalfCauchy, InverseGamma, ContinuousBernoulli (#427)
- Add missing distributions: StudentT, LogNormal, Chi2, FisherSnedecor (#425)
- Add missing distributions: Poisson, Binomial, Geometric, NegativeBinomial (#424)
- Implement LocalResponseNorm and CrossMapLRN2d (#435)
- Implement missing pooling: FractionalMaxPool2d/3d, LPPool1d/2d/3d, MaxUnpool1d/3d (#432)
- Implement missing dropout variants: Dropout1d, Dropout3d, AlphaDropout, FeatureAlphaDropout (#433)
- Implement BatchNorm3d and SyncBatchNorm (#434)
- Implement missing loss functions: GaussianNLLLoss, SoftMarginLoss, NLLLoss2d, AdaptiveLogSoftmaxWithLoss (#431)
- Implement missing LR schedulers: MultiplicativeLR, ChainedScheduler, SWALR (#439)
- Add missing distributions: Poisson, Binomial, Geometric, NegativeBinomial (#424)
- Implement LocalResponseNorm and CrossMapLRN2d (#435)
- Implement missing pooling: FractionalMaxPool2d/3d, LPPool1d/2d/3d, MaxUnpool1d/3d (#432)
- Implement missing dropout variants: Dropout1d, Dropout3d, AlphaDropout, FeatureAlphaDropout (#433)
- Implement BatchNorm3d and SyncBatchNorm (#434)
- Implement missing loss functions: GaussianNLLLoss, SoftMarginLoss, NLLLoss2d, AdaptiveLogSoftmaxWithLoss (#431)
- Implement missing LR schedulers: MultiplicativeLR, ChainedScheduler, SWALR (#439)
- Add comprehensive PyTorch vs ferrotorch gap analysis report (#301)
- Add comprehensive PyTorch vs ferrotorch gap analysis report (#302)
- Add GPU benchmarks + numpy comparison to ferrotorch_bench (#495)
- Implement WeightedRandomSampler and BatchSampler (#423)
- Implement TensorDataset, ConcatDataset, Subset dataset types (#422)
- Implement Transformer, TransformerEncoder, TransformerDecoder wrapper modules (#420)
- Add gradcheck/gradgradcheck numerical gradient verification utilities (#444)
- Implement where (conditional select), complex number tensor support (#443)
- Implement RNN (vanilla), RNNCell, LSTMCell, GRUCell (#419)
- Implement Conv3d, ConvTranspose1d, ConvTranspose3d (#418)
- Implement EmbeddingBag module (#421)
- Wire ferray-ufunc SIMD kernels into CPU elementwise ops (#416)
- Wire ferray-linalg (faer) into CPU matmul path (#415)
- Add ptx_kernel! macro to eliminate f32/f64 PTX duplication (#489)
- Replace raw_device_ptr usize hack with proper typed NcclOps trait (#492)
- NCCL backend: add hybrid TCP+NCCL backend for P2P fallback (#491)
- Fix f64 transcendental PTX kernels — replace f32 downcast with proper f64 precision (#488)
- NCCL backend: add dedicated NCCL stream for async communication overlap (#490)
- Add NCCL backend for distributed GPU collective operations (#417)
- Add f64 GPU kernel variants for all existing f32 PTX kernels (#487)
- Eliminate Dropout2d GPU→CPU→GPU mask roundtrip (#482)
- Add GPU kernels for indexing ops (gather, scatter, scatter_add) — currently documented CPU-only (#479)
- Replace silent CPU fallbacks with hard errors for GPU tensors missing kernel coverage (#485)
- Bump all crate versions to 0.2.0 (#408)
- Bump all crate versions to 0.2.1 (#412)
- Add GPU kernels for cumulative ops (cumsum, cumprod, cummax, cummin, logcumsumexp) (#478)
- Eliminate CPU roundtrip in unary_map for GPU tensors (foundation function for many ops) (#480)
- Add GPU backward kernels for SiLU, ELU, Mish, LogSoftmax (no GPU path, force CPU roundtrip) (#477)
- Add GPU forward kernel for log_softmax (currently explicit .cpu() download) (#476)
- Add GPU forward kernels for SiLU, ELU, Mish activations (currently use unary_map CPU roundtrip) (#475)
- Eliminate GPU→CPU roundtrips in norm.rs forward/backward (LayerNorm, GroupNorm, RMSNorm, BatchNorm) (#474)
- Eliminate GPU→CPU roundtrips in all loss function backward passes (13+ losses, 42 .cpu() calls) (#473)
- Add GPU backward kernel for GELU Tanh approximation mode (#465)
- Add GPU forward kernels for GELU Tanh and erf approximation modes (#469)
- GPU Conv2d backward pass — forward-only GPU kernel, backward falls to CPU (#349)
- Fused GRU/LSTM kernels — GRU forward is 10.7x slower than PyTorch (gate-level fusion needed) (#345)
- Optimized Conv2d — im2col is 7.4x slower than PyTorch, needs cache-friendly tiling (#344)
- GPU vectorized loads — ld.global.f32 (32-bit) should be ld.global.v4.f32 (128-bit) (#351)
- Lower CPU parallel threshold from 2M to 32K elements to match PyTorch grain size (#343)
- SLEEF vectorized transcendentals — exp/sin/cos are 22-33x slower than PyTorch (scalar libm) (#341)
- TensorIterator abstraction for broadcasting — broadcast ops are 166-185x slower than PyTorch (#339)
- Vectorized reduction kernels — sum/mean/max/min along axis are 500-670x slower than PyTorch (#338)
- `CudaBuffer<T>.data` is now `Option<CudaSlice<T>>` with custom Drop for pool integration
- `alloc_zeros_f32` / `alloc_zeros_f64` check pool before allocating from CUDA driver
- All kernel output allocations in `kernels.rs` and `blas.rs` use pool-aware `alloc_zeros_f32`/`alloc_zeros_f64`

### Performance
- **GPT-2 124M decode: 3.5 tok/s → 100 tok/s (29x) on WSL2/RTX 3090**
  - PTX bug fixes alone: 3.5 → 90 tok/s (elementwise ops moved from CPU fallback to GPU)
  - CUDA graph capture: 90 → 100 tok/s (300 kernel launches collapsed to 1 graph replay)

## [0.1.0] - 2026-03-15

### Fixed
- Fix PTX kernel recompilation on every call — add module cache (#178)
- Fix flaky backend_impl OnceLock test ordering (#173)
- Fix FlashAttention GPU PTX register name collision (#168)
- Rewrite GPU conv2d as pure GPU — im2col PTX kernel, no CPU roundtrip (#163)
- Fix flaky watchdog timing test (#144)
- Wire up ferray-ufunc SIMD kernels for elementwise ops (#141)
- Wire up ferray-linalg for CPU matmul and fix crates.io dependency versions (#140)

### Added
- Update README and prep all crates for crates.io with per-crate READMEs (#176)
- Add einops, LoRA, fixed-point derivatives, and natural gradient (#174)
- Commit unified device-aware Tensor Steps 2-4 (#172)
- Implement unified device-aware Tensor — Step 1: core infrastructure (#170)
- Design unified device-aware Tensor architecture (#169)
- Phase 10 Wave 4: FlashAttention GPU, gradient penalty, PagedAttention, GGUF (#165)
- Phase 10 Wave 3: higher-order grads, FlashAttention, autocast wiring, model hub (#160)
- Phase 10 Wave 2: linalg, vmap, pack_padded_seq, special functions, TensorBoard (#154)
- Add per-crate READMEs and prep new crates for publishing (#153)
- Phase 10 Wave 1: einsum, hooks, distributions, profiler, sparse, FFT (#149)
- Design document for remaining PyTorch feature parity (#148)
- Update README with pre-OOM hooks, ONNX export, and latest features (#147)
- Add pre-OOM hooks system and ONNX export (#145)
- Add GPU memory reservation, OOM recovery, and graceful pause-on-pressure (#142)
- Add performance benchmarks vs PyTorch (#139)
- Add comprehensive README and prepare crates for crates.io publishing (#138)
- Phase 9: CubeCL + AMD GPU + Training + LLM + Quantization (#136)
- Update CHANGELOG, add licenses, analyze Burn, plan AMD GPU support (#135)

Initial release of ferrotorch — a complete deep learning framework in pure Rust, built on ferray.

### Core Engine (ferrotorch-core)

- **Tensor type** with dynamic shapes, Arc-based identity sharing, and Mutex-guarded gradients (Send + Sync)
- **Reverse-mode autograd** with Kahn's topological sort, gradient accumulation, and computation graph
- **30+ differentiable operations**: arithmetic (add, sub, mul, div, neg, pow, sqrt, abs), reductions (sum, mean, prod), linalg (matmul, mm, mv, dot, bmm, transpose), activations (relu, sigmoid, tanh, gelu, silu, softmax, log_softmax), shape (reshape, flatten, squeeze, unsqueeze, cat, expand), indexing (index_select, masked_fill), comparison (where_)
- **Operator overloading**: `&a + &b`, `a * b`, `-a` with all ownership combinations
- **Method-style API**: `tensor.matmul(&other)`, `tensor.relu()`, `tensor.sum_all()`
- **In-place operations**: `zero_()`, `fill_()`, `add_scalar_()`, `mul_scalar_()`, `clamp_()` with autograd safety guards
- **Display formatting** matching PyTorch style with grad_fn names
- **Gradient checkpointing** for memory-efficient deep networks
- **no_grad() and autocast()** context managers
- **Tensor creation**: zeros, ones, full, rand, randn, eye, arange, linspace, from_slice, scalar

### Neural Network Modules (ferrotorch-nn)

- **Module trait** with forward, parameters, named_parameters, train/eval, state_dict, load_state_dict (strict mode)
- **`#[derive(Module)]` proc macro** with `#[param]`, `#[submodule]`, `#[skip]` attributes
- **Layers**: Linear, Conv1d, Conv2d, ConvTranspose2d, BatchNorm2d, LayerNorm, GroupNorm, RMSNorm, Dropout, Dropout2d, Embedding, MultiheadAttention, LSTM, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- **Activations**: ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyReLU, ELU, Mish
- **Containers**: Sequential, ModuleList, ModuleDict (insertion-order preserving)
- **Loss functions**: CrossEntropyLoss (label smoothing), MSELoss, BCEWithLogitsLoss, HuberLoss
- **Weight initialization**: xavier_uniform/normal, kaiming_uniform/normal, uniform, normal, zeros, ones
- **Functional API**: `functional::linear`, `functional::relu`, `functional::dropout`, `functional::cross_entropy`
- **Gradient clipping**: `clip_grad_norm_()`, `clip_grad_value_()`

### Optimizers (ferrotorch-optim)

- **6 optimizers**: SGD (momentum, Nesterov, weight decay), Adam (AMSGrad), AdamW (decoupled weight decay), RMSprop (centered mode), Adagrad (LR decay), L-BFGS (two-loop recursion)
- **Parameter groups** with per-group hyperparameters
- **5 LR schedulers**: StepLR, CosineAnnealingLR, LinearWarmup, ReduceLROnPlateau, SequentialLr
- **GradScaler** for mixed-precision training with dynamic loss scaling

### Data Loading (ferrotorch-data)

- **Dataset and IterableDataset traits** with VecDataset and MappedDataset
- **DataLoader** with batching, shuffling, drop_last, seeded reproducibility
- **Samplers**: SequentialSampler, RandomSampler (deterministic Fisher-Yates)
- **Transforms**: Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip

### Vision (ferrotorch-vision)

- **8 model architectures**: ResNet-18/34/50, VGG-11/16, ViT-B/16, EfficientNet-B0, ConvNeXt-Tiny, Swin Transformer Tiny, U-Net, YOLO
- **Model registry**: `list_models()`, `get_model()`, `register_model()`
- **Datasets**: MNIST (real IDX parsing + synthetic), CIFAR-10/100 (synthetic)
- **Image transforms**: Resize, CenterCrop, VisionToTensor, VisionNormalize
- **Image I/O**: read_image, write_image, read_image_as_tensor (PNG/JPEG)

### JIT Compiler (ferrotorch-jit)

- **Tracing**: captures autograd graphs into static IR
- **IR graph** with 27+ operation kinds and binary serialization
- **4 optimization passes**: constant folding, dead code elimination, operator fusion, memory planning
- **compile() API** (torch.compile equivalent)
- **Codegen backends**: InterpreterBackend, NativeBackend
- **Graph break handling**: SegmentedModule with compiled + eager segments

### GPU Backend (ferrotorch-gpu)

- **CUDA via cudarc 0.19** (pure Rust, no C FFI)
- **cuBLAS matmul**: 81.8x speedup on RTX 3090
- **GPU Conv2d**: im2col + cuBLAS GEMM
- **PTX kernels**: add, sub, mul, neg, relu
- **Caching allocator** with memory tracking

### Serialization (ferrotorch-serialize)

- **SafeTensors** via the real `safetensors` crate (HuggingFace compatible)
- **PyTorch .pt import** with custom pickle parser (pure Rust)
- **Training checkpoints** (model + optimizer + epoch)

### Distributed (ferrotorch-distributed)

- **TCP backend** with allreduce, broadcast, barrier
- **DDP wrapper** with gradient synchronization
- **GPU-aware collectives** via transfer transport
