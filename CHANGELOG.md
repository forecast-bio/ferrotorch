# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-04-08

### Added
- GPU `prod` (forward) via a native `reduce_prod` PTX kernel (#524, additional slice). New `gpu_reduce_prod` (f32) / `gpu_reduce_prod_f64` in `ferrotorch-gpu/src/kernels.rs` mirror `gpu_reduce_sum` but with `1.0` identity and `mul.f32` combiner. f64 reuses the kernel via the existing `ptx_f32_to_f64` rewriter. New `prod_f32` / `prod_f64` trait methods on `GpuBackend`. `reduction::prod` routes f32/f64 CUDA inputs to the new kernel — closes the previous `NotImplementedOnCuda { op: "prod" }` gap. 4 new GPU integration tests on RTX 3090 (f32 vs CPU on small input, f64 closed-form 3.0, GPU-output device contract, 10K-unit-factor stress that exercises the multi-pass tree reduction). The existing `ProdBackward` already had a CPU-only path; its GPU dispatch is wired alongside (#524)
- GPU `sum_dim` backward via a dedicated `repeat_along_dim` PTX kernel (#524, additional slice). New `gpu_repeat_along_dim` (f32) / `gpu_repeat_along_dim_f64` in `ferrotorch-gpu/src/kernels.rs` expand a `[outer, inner]` source into `[outer, repeat_count, inner]` by replicating the gradient along an inserted middle axis — exactly the broadcast `sum_dim`'s VJP needs to map a reduced gradient back to the input shape. New `repeat_along_dim_f32` / `repeat_along_dim_f64` trait methods on `GpuBackend`. `SumDimBackward::backward` routes through these for f32/f64 CUDA tensors, eliminating another `NotImplementedOnCuda` gap. 3 new GPU integration tests on RTX 3090 (forward sum_dim → backward propagates 1s, keepdim variant, GPU vs CPU equivalence on a non-trivial 2×3×4 tensor) (#524)
- GPU 2-D FFT via `cufftPlan2d` (#634, partial close — fft2/ifft2 landed). New `gpu_fft2_c2c_f32` / `gpu_fft2_c2c_f64` in `ferrotorch-gpu/src/cufft.rs` wrap `cudarc::cufft::CudaFft::plan_2d` for unbatched 2-D complex-to-complex transforms. Layout matches the 1-D ops: `[h, w, 2]` interleaved real/imag f32 (or f64). `inverse=true` divides by `h*w` to match torch / numpy. New trait methods `fft2_c2c_f32` / `fft2_c2c_f64` on `GpuBackend`. `core::fft::fft2` and `ifft2` route through these directly when input is f32/f64 CUDA with no leading batch dims, replacing the previous "two 1-D passes + inner transpose" path with a single cuFFT plan execution. 5 new GPU integration tests on RTX 3090 (4×4 f32 vs CPU, fft2/ifft2 round-trip f32, f64 round-trip, 8×8 cosine/sine pattern matches CPU, GPU-output device contract). Remaining cuFFT tail (hfft/ihfft 1-D Hermitian + fftn/ifftn 3-D / arbitrary-rank) splits to follow-up #636 (#634)
- Device-resident GPU `solve` and `cholesky` (#632, partial close — covers section B's solve + cholesky). New `gpu_solve_f32_dev` / `gpu_solve_f64_dev` (cuSOLVER getrf + getrs on a column-major device-side copy) and `gpu_cholesky_f32_dev` / `gpu_cholesky_f64_dev` (cuSOLVER potrf via `memcpy_dtod` clone since SPD matrices are symmetric so row-major == column-major). Both eliminate the previous `gpu_to_cpu → cusolver_on_host_buffers → cpu_to_gpu` sandwich the `Backend::solve_*` and `Backend::cholesky_*` impls used. The on-device row→column-major transpose uses the existing `gpu_transpose_2d` / `gpu_transpose_2d_f64` kernels. 5 new GPU integration tests for solve_dev (2×2 closed-form recovery, diagonal system, f64 4×4, multi-column B with identity, singular-matrix → error). Cholesky correctness covered by the existing test suite (which now runs through the device-resident path). The remaining cuSOLVER bounce-drops — svd and qr — split to follow-up since they're the trickiest cases (SVD has 3 outputs, QR needs Householder reflector unpacking) (#632)
- GPU-resident `lstsq_solve` via cuSOLVER `cusolverDnSSgels` / `cusolverDnDDgels` (#630). New `gpu_lstsq_f32` / `gpu_lstsq_f64` in `ferrotorch-gpu/src/cusolver.rs` accept `&CudaBuffer<T>` for both A and B, run the row→column-major transpose on device, dispatch to the iterative-refinement gels (cudarc only exposes the SS/DD variants), and transpose the solution back — staying in VRAM end-to-end. `GpuBackend::lstsq_f32` / `lstsq_f64` trait methods. New public `linalg::lstsq_solve(&Tensor<T>, &Tensor<T>) -> Tensor<T>` mirroring `torch.linalg.lstsq`'s solution output (just X — the full 4-tuple `linalg::lstsq` with residuals/rank/singular-values stays CPU). Accepts both 1-D `b: [M]` and 2-D `b: [M, K]`. 6 new GPU integration tests on RTX 3090 (square exact-solve, 3×2 over-determined fit with known true x, f64 square, 1-D b round-trip, multi-column B with identity A, shape-mismatch reject) (#630)
- Memory-mapped pytorch-pickle loader (#629): `load_pytorch_state_dict_mmap(path)` mirrors the existing `load_pytorch_state_dict` but feeds the ZIP archive a `Cursor<&[u8]>` over an `mmap2::Mmap` instead of an open `File`. Internal helpers (`find_pkl_name` / `find_data_prefix` / `read_zip_entry`) made generic over `R: Read + Seek` so both code paths funnel through a single `load_pytorch_state_dict_inner<T, R>`. The mmap is dropped before return; tensor data is copied into owned `Tensor<T>` buffers, so no file-lifetime invariants leak. Same trade-off as the GGUF / safetensors mmap variants: the OS page cache holds raw archive bytes instead of `std::fs::read`-ing them into a heap `Vec<u8>` up front. The pickle parser still allocates internally as it walks the bytecode, but the outer ZIP reader is now lazy. 2 new tests (mmap output equals read-path output for the same fixture, missing-file rejection) (#629)
- GPU `clamp` backward kernel (#524, partial close): native PTX `clamp_backward_kernel` + host launchers `gpu_clamp_backward` (f32) / `gpu_clamp_backward_f64` in `ferrotorch-gpu/src/kernels.rs`. Single-pass elementwise: `out[i] = grad[i]` when `min <= x[i] <= max`, else `0`. Closes the previous `NotImplementedOnCuda { op: "ClampBackward" }` gap in `transcendental.rs`. New `clamp_backward_f32` / `clamp_backward_f64` trait methods on `GpuBackend`. `ClampBackward::backward` routes f32/f64 CUDA inputs to the new kernel; non-f32/f64 dtypes fall through to the existing CPU walk. 4 new GPU integration tests on RTX 3090 (f32 vs CPU equivalence, f32 grad-chain propagation, 10K-element stress, f64 kernel direct-call correctness). #524's full scope — the remaining ~38 backward kernels across reduction/indexing/cumulative/etc. — continues as broader follow-up work.
- GPU pad/truncate path for `fft` / `ifft` when `n != input_n` (#605, partial close — covers section C of the issue). New PTX `pad_truncate_kernel` + host-side `gpu_pad_truncate_complex_f32` / `gpu_pad_truncate_complex_f64` in `ferrotorch-gpu/src/kernels.rs` build a `[batch, dst_n, 2]` output from a `[batch, src_n, 2]` input — zero-padding when `dst_n > src_n`, truncating when `dst_n < src_n`. Each thread writes one complex pair; `selp.f32`-equivalent branch via predicate to either copy from `src` or write zeros. f64 reuses the kernel via the existing `ptx_f32_to_f64` rewriter (added `%off_src` / `%off_dst` byte-stride patches to the rewrite table, plus a 4→8 offset rewrite for the imaginary lane). New trait methods `pad_truncate_complex_f32` / `pad_truncate_complex_f64` on `GpuBackend`. `core::fft::fft` / `ifft` updated to invoke the kernel directly when their `n` argument differs from the input length, replacing the previous fall-through-to-CPU branch — the GPU FFT path now stays on device end-to-end for any `n`. 5 new GPU integration tests on RTX 3090 (f32 pad, f32 truncate, f64 pad, ifft pad, batched pad). The remaining cuFFT tail — hfft/ihfft + fft2/ifft2 + fftn/ifftn — splits to follow-up #634 (#605)
- HQQ (Half-Quadratic Quantization, Mobius Labs) weight unpacker (#613): new `ferrotorch_llama::quant_loaders::HqqWeights` struct + `dequantize_hqq(&packed) -> Vec<f32>` for 1/2/3/4/8-bit packed integer weights with per-row f32 `scale` / `zero` (HF transformers ships these as f16 — caller casts on load). Five bit-width-specific unpackers handle the packing strategies: 8-bit one byte per int, 4-bit two nibbles per byte (low first), 2-bit four ints per byte (low 2 bits first), 1-bit eight LSB-first, 3-bit tightly packed 8 ints per 3 bytes. Output is `[out_features, in_features]` row-major, ready for `Linear.weight`. 9 new tests covering each bit-width's pack→unpack round-trip with known patterns, per-row scale/zero application, invalid-bits / short-buffer / scale-length-mismatch rejections (#613)
- GPU-resident `lu_factor` via cuSOLVER `getrf` (#604, partial close — covers section A's first sub-op). New `gpu_lu_factor_f32` / `gpu_lu_factor_f64` in `ferrotorch-gpu/src/cusolver.rs` accept a device buffer (`&CudaBuffer<T>`), do the row→column-major transpose on device via the existing `gpu_transpose_2d` / `gpu_transpose_2d_f64` kernel, run cuSOLVER `getrf`, then transpose the LU factors back to row-major — staying in VRAM end-to-end. The pivot vector is small (O(n) ints) so it's downloaded as `Vec<i32>` rather than inventing a typed-int GPU handle. `GpuBackend` trait gains `lu_factor_f32` / `lu_factor_f64` returning `(GpuBufferHandle, Vec<i32>)`. New public `linalg::lu_factor(&Tensor<T>) -> (Tensor<T>, Vec<i32>)` mirrors `torch.linalg.lu_factor` — packed `LU` (strict lower = L, upper = U) plus 1-based swap-sequence pivots in cuSOLVER / LAPACK convention. CPU path packs `ferray-linalg::lu`'s P/L/U output and converts the permutation matrix to the swap-sequence form so CPU and GPU paths return interchangeable output. `Tensor::lu_factor()` method exposes the op. 5 new GPU integration tests on RTX 3090 (3×3 f32, 5×5 f64, 64×64 large-matrix reconstruction, CPU-vs-GPU equivalence via reconstruction, non-square reject) — each verifies `A == reconstruct(LU, ipiv)` element-wise. The remaining cuSOLVER tail — bounce-drop refactors of svd/cholesky/solve/qr (section B) and lstsq/eig (section A.2/A.3) — splits to follow-ups (#604)
- Native GPU `reduce_min` / `reduce_max` PTX kernels (#627). Six new f32+f64 PTX kernels in `ferrotorch-gpu/src/kernels.rs`: `reduce_min_kernel` / `reduce_max_kernel` (parallel two-pass tree reductions over a 1-D buffer with `min.f32` / `max.f32` PTX instructions, ±inf sentinels, 256-thread blocks × ≤1024 grid, cap-and-recurse for >256 partials), and the **fused** `masked_reduce_min_kernel` / `masked_reduce_max_kernel` that take `(data, mask_f, n)` and combine `mask_f != 0 ? data : ±inf` directly into the running accumulator via `selp.f32` — eliminating the `mul + add + reduce` chain and the host-side sentinel construction the unfused path required. f64 versions reuse the kernels via the existing `ptx_f32_to_f64` rewriter (added `selp.f32` → `selp.f64` to the rewrite table). Trait surface on `GpuBackend`: `min_f32` / `max_f32` / `min_f64` / `max_f64` (1-buffer reductions) and `masked_min_f32` / `masked_max_f32` / `masked_min_f64` / `masked_max_f64` (2-buffer fused). `Tensor<T>` gains public `amin()` / `amax()` mirroring `torch.amin` / `torch.amax` — full-tensor global min/max on CUDA f32/f64 routes to the native kernel; CPU and other dtypes walk the buffer. Backward (`AminBackward` / `AmaxBackward`) routes the upstream grad to every input position equal to the extremum (subgradient at ties, matches torch). `masked_min` / `masked_max` GPU lowering rewritten to call `masked_min_f32` directly — single kernel launch, only the float mask is uploaded (which we needed anyway for the indicator role). 15 new GPU integration tests on RTX 3090 covering f32/f64 round-trip vs CPU, 4096-element masked stress, 100K-element multi-pass `amin/amax` stress, all-masked → NaN, and GPU-output device contract (#627)
- `DTensor` distributed tensor over a `DeviceMesh` (#611): new `ferrotorch_distributed::dtensor` module with the placement spec callers expect from `torch.distributed.tensor`. `Placement` enum carries `Replicate` / `Shard(dim)` / `Partial(reduce_op)` per mesh dim. `DTensor<T>` wraps a per-rank `local_tensor` plus `placements: Vec<Placement>` (one per mesh dim) plus the logical `global_shape`. Constructors: `from_local(local, mesh, placements, global_shape)` (general) and `from_local_replicated(local, mesh)` (every rank holds the same tensor). API surface: `to_local()` / `shape()` / `placements()` / `mesh()` / `numel()` accessors, `redistribute(target_placements)` for placement transitions. Validation rejects placement-count mismatch, OOB shard dims on construct, OOB shard dims on redistribute target. Cross-rank collective dispatch for the actual `Sharded → Replicated` (all_gather), `Partial → Replicated` (all_reduce), `Sharded(d) → Sharded(e)` (all_to_all) transitions delegates to the existing `crate::collective::*` ops — DTensor is the placement-tracking layer on top. 8 new tests covering predicates, replicated-from-local, count-mismatch / OOB-dim rejects, redistribute updates, numel uses global shape (#611)
- Distributed backend skeletons for Gloo / MPI / UCC (#459): three new modules `ferrotorch_distributed::gloo_backend` / `::mpi_backend` / `::ucc_backend`, each with an `is_*_available()` runtime predicate and a `*Backend` struct that implements the `Backend` trait by erroring with the new `DistributedError::BackendUnavailable { backend }` variant on non-feature builds. Default off via three new cargo features (`gloo-backend` / `mpi-backend` / `ucc-backend`). Mirrors the MPS skeleton pattern from #451 — establishes the public API contract so callers can write `GlooBackend::new(rank, world_size)` paths now; the actual FFI binding layers (gloo-sys, the `mpi` Rust crate, UCC C bindings) need a CI runner with the corresponding C library and are tracked as separate follow-ups under the same issue label. 4 new tests confirming the unavailable path works on the workspace's default Linux build (#459)
- Memory-mapped GGUF loader (#609): new `load_gguf_mmap(path)` and `load_gguf_state_dict_mmap(path)` mirror `load_safetensors_mmap` from #587. `mmap2::Mmap` replaces the up-front `std::fs::read` so peak RSS at parse time goes from ~2× file size (file-bytes Vec + parsed-data Vec) down to ~1× (mmap region + parsed-data Vec, with the mmap region only paged in on demand and dropped before return). Bigger savings on GGUF than safetensors because GGUF often holds quantized blocks that expand 2-8× during dequantization, so the input file's 1× footprint is a non-trivial fraction of total memory. 2 new tests (mmap output equals read-path output for both GgufFile and StateDict, missing-file path returns InvalidArgument). Pickle mmap split out to follow-up #629 (the pickle parser needs `Read` rather than `&[u8]` so it's heavier surgery) (#609)
- `ferrotorch-llama` beam search (#612): new `generation::beam_search(model, prompt, &BeamSearchConfig)` returns the top `num_beams` continuations sorted by length-normalised score, best first. `BeamSearchConfig` carries `num_beams` / `max_new_tokens` / `length_penalty` / `eos_token_ids`. Each step expands every live beam to its top-`num_beams` softmax continuations, then keeps the global top `num_beams` across (beam × vocab). Beams that produce an EOS token finalise and pass through subsequent rounds at their final score. Length-normalisation uses `cum_log_prob / L^length_penalty`. 4 new tests covering defaults, input validation (empty prompt, num_beams=0, length_penalty=0), the `num_beams` × `max_new_tokens` shape contract, and EOS-on-every-token early stopping. (Speculative decoding split out for a future pass — needs a draft+target pair which is a heavier infrastructure change.) (#612)
- `ferrotorch-hub` HF auth + sharded download (#509): new `auth.rs` module exports `hf_token()` (resolves `HF_TOKEN` env var → `$HF_HOME/token` → `$HOME/.cache/huggingface/token`) and `with_auth(req)` which decorates a `ureq::Request` with `Authorization: Bearer <token>` when a token is found, no-op otherwise. Wired through every existing HTTP call site (`download::download_and_verify`, `discovery::search_models`, `discovery::get_model`) so gated repos like `meta-llama/Meta-Llama-3-8B` work end-to-end without per-call boilerplate. New `download::hf_download_model(repo, revision, cache)` discovers shards via `model.safetensors.index.json`'s `weight_map`, downloads `config.json` + every shard + best-effort tokenizer files into the cache. `HubCache::store` upgraded to `mkdir -p` the parent of nested keys so the `{repo}/{filename}` layout works. 4 new auth-resolution tests (env precedence, empty-env-falls-through, no-op without token) (#509)
- Closing already-shipped issues (#512, #513, #511, #519, #515) — these features landed earlier (LlamaForCausalLM model composition, end-to-end smoke example, bf16 hybrid GPU weight storage, GPU forward via `CudaSlice<u16>` in VRAM, RoPE NTK-aware scaling) but the issues weren't marked closed at the time. The earlier `### Added` entries covering each one already document the work; this is just an issue-tracker hygiene sweep.
- `ferrotorch-distributions` properties tail (#608): native `cdf` / `icdf` / `mean` / `mode` / `variance` for `Weibull` (closed-form via `lgamma` for the gamma-function moments) and `Kumaraswamy` (`mean = b·B(1+1/a, b)`, `mode = ((a-1)/(a·b-1))^(1/a)`, plus the closed-form CDF/ICDF that already characterises the distribution). `Uniform` gains `mode` (midpoint convention, matches torch) and `stddev` (`(high − low) / √12`). `Bernoulli` gains `icdf` (step at `1 − p`, generalized inverse of the discrete CDF). 13 new tests including identity checks under the uniform-case parameters and CDF/ICDF round-trips for each new pair (#608)
- `ferrotorch_vision::ops` tail: `generalized_box_iou` / `distance_box_iou` / `complete_box_iou` (pairwise GIoU/DIoU/CIoU between two `[N, 4]` xyxy box sets, computed in one pass alongside the union/center-distance/aspect-ratio terms), plus `roi_align(input, boxes, output_size, spatial_scale, sampling_ratio)` (bilinear-sampled per-RoI feature extraction with `aligned=true` semantics) and `roi_pool(input, boxes, output_size, spatial_scale)` (integer-rounded max-pool variant). `boxes` follows torchvision's `[K, 5]` `(batch_idx, x1, y1, x2, y2)` convention; output is `[K, C, out_h, out_w]`. 9 new tests including disjoint-box GIoU = -7/9 closed form, identical-box DIoU/CIoU = 1, and roi_pool exactly recovering input pixels at unit scale (#610)
- `ferrotorch-ml::metrics` ranking + clustering tail: `dcg_score` / `ndcg_score` (single-query relevance ranking), `coverage_error` / `label_ranking_average_precision_score` / `label_ranking_loss` (multilabel ranking with 2-D `[N, K]` indicator inputs). Clustering: `adjusted_rand_score` / `adjusted_mutual_info_score` / `normalized_mutual_info_score` (arithmetic-mean normalisation, sklearn default) / `homogeneity_score` / `completeness_score` / `v_measure_score` / `fowlkes_mallows_score` (label-pair metrics over `Array1<isize>` so sklearn's `-1` noise sentinel survives the round-trip), plus `silhouette_score` / `davies_bouldin_score` / `calinski_harabasz_score` (data-matrix metrics over `[N, D]` features + cluster assignments). Two new private adapters: `tensor_to_array2_f64` / `tensor_to_array2_usize` for the 2-D ranking inputs and `tensor_to_array1_isize` for clustering labels. 11 new tests including known-value perfect-clustering checks for ARI/NMI/V-measure/Fowlkes-Mallows and a well-separated 2-D silhouette ≈1.0 sanity (#617)
- Hook registration on the `Module` trait: new provided methods `with_forward_hook` / `with_forward_pre_hook` / `with_backward_hook` consume `self`, wrap it in a `HookedModule`, register the supplied hook, and return `(HookedModule<Self, T>, HookHandle)`. Mirrors `torch.nn.Module.register_*_hook` ergonomically — callers no longer have to write `HookedModule::new(m).register_*_hook(...)` manually. Trait-method names use the `with_*` prefix to avoid clashing with `HookedModule`'s inherent `register_*_hook` methods (which take `&self` and append to an already-wrapped instance — both surfaces compose). Gated on `Self: Sized` so the trait stays dyn-compatible. 3 new tests covering forward, forward_pre, backward (#606)
- `ComplexTensor` op integration: `matmul(self, other)` for 2-D complex GEMM via four real `mm` calls (re = ac − bd, im = ad + bc — keeps the implementation entirely in the existing real GEMM surface, no new kernels needed). Plus `fft` / `ifft` / `fft2` / `ifft2` bridges to the existing interleaved `crate::fft::*` ops via `to_interleaved` / `from_interleaved` round-trip — so `torch.fft.fft(complex_tensor)` shape parity lands without duplicating the FFT path. 5 new tests (matmul known value, real-only sanity, shape-mismatch reject, fft/ifft 1-D round-trip, fft2/ifft2 4×4 round-trip) (#624)
- `nn::Embedding` sparse-grad integration: new `with_sparse(true)` builder + `sparse_grad()` method that materializes a coalesced `SparseGrad<T>` from the dense weight gradient, keyed on the unique row indices touched by the most recent forward pass. The dense grad is unchanged — sparse_grad is a compact view callers can feed into `SparseGrad::apply_sgd` or `optim::SparseAdam` to skip zero rows. Indices are deduped + sorted on cache, so callers don't have to coalesce. Mirrors `torch.nn.Embedding(sparse=True)` → `optim.SparseAdam` flow. Forward path zero-overhead when sparse mode is off. 4 new tests including end-to-end `forward → set grad → sparse_grad → apply_sgd` verifying only touched rows update (#623)
- `masked_min` / `masked_max` accept GPU inputs without erroring — the data is downloaded, reduced on host (skipping masked entries), and the resulting 0-d tensor is uploaded back to the input device so the GPU-input → GPU-output contract matches `masked_sum_gpu`. The "real" inf-fill + on-device reduce path waits on native `gpu_reduce_min` / `gpu_reduce_max` PTX kernels (filed as #627) since the workspace doesn't yet have GPU min/max reduction kernels. 2 new tests covering the visible-elements walk and the all-masked → NaN edge case (#616)
- Native fused `prelu` / `glu` GradFns in `ferrotorch_core::grad_fns::activation` — replaces the previous decomposed implementations (PReLU was `(1-α)·relu(x) + α·x` over three GradFn nodes, GLU was `split + sigmoid + mul` over three nodes). Now each is a single forward + single backward node: `PReluBackward` routes grad to both `x` and `alpha` in one pass; `GluBackward` caches the sigmoid for the cheaper VJP and concatenates both halves directly. The `nn::PReLU` module and `nn::functional::glu` / new `nn::functional::prelu(input, alpha)` thin-shim through to the fused ops. 7 new tests covering forward, backward, shape errors, and 2-D `glu` over an inner dim (#614)
- `nn::functional` parity tail: stateless forwarders for `conv1d` / `conv2d` / `conv3d` / `conv_transpose1d` / `conv_transpose2d` / `conv_transpose3d` (each accepts `&Tensor` weight + optional bias and runs the existing im2col + matmul path under the hood, including the GPU fast path and CPU autograd graph). Plus a stateless `embedding(input, weight, padding_idx)` and a `scaled_dot_product_attention(q, k, v, is_causal)` wrapper over `flash_attention`. Pool fns (`max_pool*` / `avg_pool*` / `adaptive_*` / `lp_pool*`) and pad fns (`pad1d` / `pad2d` / `pad3d`) re-exported from their existing modules so callers can write `nn::functional::max_pool2d(..)` instead of `nn::pooling::max_pool2d(..)`. Each conv module now carries a `from_parts(weight, bias, stride, padding[, output_padding])` constructor for the functional layer. 6 new tests (#607)
- IntTensor / BoolTensor wired into existing op surface: `masked_fill_bt(&Tensor, &BoolTensor, value)`, `where_bt(&BoolTensor, &Tensor, &Tensor)`, `index_select_1d_it(&Tensor, &IntTensor)` accept the new types directly. Comparison helpers on `BoolTensor`: `gt` / `lt` / `ge` / `le` / `eq_t` / `ne` produce a mask from two float tensors. All thin wrappers over the existing `&[bool]` / `&[usize]` paths so behavior is unchanged. 12 new tests across indexing/comparison/bool_tensor (#615)
- `Device::Mps(usize)` variant for Apple Silicon, plus a new `ferrotorch-mps` crate skeleton with `is_mps_available()` / `MpsDevice::new(ordinal)` / `device_count` / `init_mps_backend`. Platform-gated: returns `DeviceUnavailable` on non-Apple builds and when the `metal-backend` feature is off (default). Display formatting (`mps:0`) and `Device::is_mps` predicate land alongside. Establishes the API contract so callers can write `Device::Mps(_)` paths now; the MSL kernel layer for each `GpuBackend` trait method is split out to #626 (needs a macOS CI runner). 5 new mps tests + downstream Device-match call sites updated. (#451)
- `TracedModule::save` / `load` / `to_bytes` / `from_bytes` in `ferrotorch-jit` — round-trip a traced graph through disk or an in-memory byte buffer using the existing `IrGraph::serialize` / `deserialize`. Mirrors `torch.jit.save` / `torch.jit.load` for the trace path. 3 new tests including a save → load → execute roundtrip on disk (#620)
- New `ferrotorch-jit-script` proc-macro crate — `#[script]` attribute that compiles a tensor function into a `TracedModule` at call time. Mirrors `torch.jit.script`: annotate `fn my_fn(a: Tensor<f32>, b: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> { ... }` and the function returns `FerrotorchResult<TracedModule<f32>>` instead. Implementation shim over `ferrotorch_jit::trace` so op coverage stays in lockstep with the trace path. Auto-detects scalar `T` from the return type (Tensor<T>, FerrotorchResult<Tensor<T>>, or Result<Tensor<T>, _>). 3 integration tests covering single-binding bodies, three-arg add chain, save→load roundtrip on a scripted module (#620)
- `ComplexTensor<T>` in `ferrotorch-core` — first-class complex tensors stored as structure-of-arrays (parallel `Arc<Vec<T>>` real and imaginary buffers). Construction via `from_re_im` / `from_real` / `zeros` / `scalar`; conversions `from_interleaved` / `to_interleaved` bridge to the array-of-structures `[..., 2]` layout used by the existing FFT surface. Pointwise `add` / `sub` / `mul` (full complex multiplication), `conj`, `abs` (modulus → real Tensor), `angle` (atan2 → real Tensor), `reshape`. Standalone — leaves `Tensor<T: Float>` untouched. 17 new tests (#618)
- `CscTensor<T>` (Compressed Sparse Column) in `ferrotorch-core::sparse` joins the existing CSR / COO / SparseTensor / SemiStructuredSparseTensor surface; `from_csr` / `to_csr` / `to_dense` round-trips. `SparseGrad<T>` for sparse-gradient optimizer steps: holds `(indices, values, slab_shape)` triple, supports `coalesce` (sum duplicate indices) and `apply_sgd(param, lr)` (update only affected rows of an embedding-shape parameter). Mirrors torch's `nn.Embedding(sparse=True)` → `optim.SparseAdam` flow. 9 new tests (#619)
- `NamedTensor<T>` in `ferrotorch-core` — dim-name annotations on top of `Tensor<T>` (mirrors torch's experimental named-tensor surface). Construction via `new(t, names)` or `refined(t, &["batch", "seq", "feat"])` (empty string = anonymous dim); `align_to(target_names)` permutes dims to match a name ordering via the existing `permute_t`; `rename(mapping)` replaces selected names; `dim_index` / `size_of` lookups; rejects duplicate non-None names. Advisory annotation that doesn't intercept ops — used to avoid "did I get the batch dim right?" bugs at op boundaries. 12 new tests (#621)
- Lazy modules: `LazyBatchNorm{1,2,3}d`, `LazyInstanceNorm{1,2,3}d`, `LazyConvTranspose{1,2,3}d` join the existing `LazyLinear` / `LazyConv{1,2,3}d`. Each defers `num_features` (or `in_channels`) discovery to the first forward call via `OnceLock`, then materializes the corresponding eager module and forwards to it. `is_initialized()` / `materialize(...)` accessors mirror the LazyLinear pattern. 11 new tests (#622)
- `ferrotorch-ml::metrics` 2-D probability scoring metrics: `brier_score_loss` (binary, mean squared error of predicted probabilities), `d2_brier_score` (skill score relative to the null model), `top_k_accuracy_score` (true label is in top-K predicted classes; `[N, n_classes]` 2-D scores), `zero_one_loss` (with `normalize` flag for fraction-vs-count), `average_precision_score` (area under precision-recall curve). 9 new tests including known-value Brier checks and top-1/top-K parity vs argmax (#599)

### Fixed
- Restore fast_log_f32; wire into vlog_f32 to fix +inf/NaN bug (#641)
- Fix autograd cond/scan backward; wire fast_log_f32 into log_softmax CPU; delete dead BroadcastScalarBackward.numel (#640)
- ferrotorch umbrella: expose jit-script/tokenize/mps/xpu/llama/ml features to match README (#639)
- Flaky `test_cuda_rng_fork_join` (and `test_cuda_rng_next_seed`) under workspace-parallel: both touch the process-global `cuda_rng::RNG_STATE` mutex and raced against each other when scheduled on different threads. Serialized via a static test-local lock matching the pattern from #602 (#599)

### Added
- `ferrotorch-ml::metrics` classification expansion: `precision_score`, `recall_score`, `f1_score` (with `Average::{Binary, Macro, Micro, Weighted}`), `roc_auc_score`, `log_loss`, `confusion_matrix`, `hamming_loss`, `balanced_accuracy_score` (with `adjusted` flag), `matthews_corrcoef`, `cohen_kappa_score`. All thin wrappers over `ferrolearn-metrics` 0.3.0 with the same Tensor-input adapter as the existing regression metrics. 10 new tests covering binary precision/recall/F1, confusion matrix shape, log_loss closed-form, perfect-separation ROC-AUC. Ranking + clustering metrics split out to #617 (#598)
- GPU `masked_sum` / `masked_mean` via `mul(data, mask_as_float) → reduce_sum` lowering. Both f32 and f64 paths use the existing `GpuBackend` `mul_*` and `sum_*` trait methods, no new kernels required. The all-masked case for `masked_mean` short-circuits to NaN on host so we don't pay a GPU round-trip for trivial inputs. `MaskedTensor::new` now accepts CUDA tensors. `masked_min` / `masked_max` GPU paths split out to #616 (need new `reduce_min` / `reduce_max` PTX kernels). 5 new GPU integration tests (#597)
- New `IntTensor<I>` (`I = i32 | i64` via `IntElement` trait) and `BoolTensor` types in `ferrotorch-core`. CPU-resident, contiguous, `Arc`-shared storage for cheap clones. `IntTensor` provides `from_vec` / `from_slice` / `zeros` / `arange` / `scalar` / `cast` / `reshape`; `BoolTensor` provides the same plus `from_predicate(&Tensor<T>, |x| ...)`, pointwise `not` / `and` / `or` / `xor`, `count_true` / `any` / `all`, and `to_float<T>` (true → 1.0, false → 0.0). Standalone types — they don't touch the existing `Tensor<T: Float>` surface; threading them through indexing / mask ops is split out to #615. 22 new tests (12 int, 10 bool) (#596)
- `Learner::with_grad_scaler` wires `ferrotorch_optim::GradScaler` into the training loop. When set, `fit` scales the loss before backward, calls `scaler.step` (which unscales gradients, skips the optimizer step on inf/NaN, and returns whether the step was applied), then `scaler.update` to dynamically tune the scale. New `skipped_steps()` accessor surfaces how many steps the AMP path discarded — a useful health indicator for mixed-precision training. `skipped_steps` resets at the start of every `fit` call. 3 new tests (#595)
- Native fused GradFns for the activation tail in `ferrotorch_core::grad_fns::activation`: `leaky_relu`, `hardtanh` / `hardtanh_with`, `relu6` (alias of hardtanh restricted to [0, 6]), `hardsigmoid`, `hardswish`, `selu` (canonical α/scale constants), `softsign`. Each comes with a closed-form `*Backward` `GradFn` rather than going through the composition chain, so the backward pass stays at O(1) extra ops per activation instead of O(k). `mish` / `softplus` / `elu` were already native; `prelu` / `glu` deferred to #614 (need multi-input or split-axis GradFn). 14 new tests including numerical-vs-analytic gradient checks (#594)
- `ferrotorch-llama` quantization loaders in new `quant_loaders.rs`: GPTQ q4 unpacker (group-wise asymmetric int4 with per-group scales/zeros, packed 8-per-i32; supports `act_order=False` and `g_idx`-based reordering for `act_order=True`) and AWQ q4 unpacker (handles the AWQ-specific channel-shuffle pack order `[0, 4, 1, 5, 2, 6, 3, 7]`). Both produce row-major `[out_features, in_features]` f32 weight matrices ready for the standard state-dict path. 9 new tests including a closed-form identity check, multi-group scale isolation, and an explicit AWQ shuffle-order verification. HQQ + real-checkpoint integration tests split out to #613 (#593)
- `ferrotorch-llama` generation API in new `generation.rs`: greedy / temperature / top-k / top-p (nucleus) sampling, repetition penalty (Keskar et al. 2019), per-token streaming callback (`generate_with_streamer` accepts `&mut dyn FnMut(u32) -> bool` returning `false` to stop early), seedable xorshift PRNG for reproducible draws, EOS-token list. `GenerationConfig` with helpers for greedy / sampling / nucleus presets. Sampling primitives (`apply_temperature`, `top_k_filter`, `top_p_filter`, `apply_repetition_penalty`, `argmax`, `sample_softmax`) re-exported so callers can roll their own loops. 12 new tests covering filter correctness, distribution shape, repetition penalty signs, edge cases. Beam search + speculative decoding split out to #612 (#592)
- `ferrotorch-distributed` point-to-point + mesh additions: tensor-level `send` / `recv` / `recv_with_timeout` / `recv_into` / `recv_into_with_timeout` / `sendrecv` (atomic two-party exchange with rank-ordered deadlock avoidance) in new `p2p.rs`. New `DeviceMesh` n-D rank layout with `coords` / `rank_of` / `ranks_along_dim` / `groups_along_dim` / `dim_index` for organizing data-parallel × tensor-parallel × pipeline-parallel topologies. The `all_to_all` collective was already in place. 15 new tests (5 p2p, 10 mesh) covering rank-mismatch errors, multi-dim coord roundtrips, and per-axis sub-group enumeration. Full DTensor sharded-tensor abstraction split out to #611 (#591)
- `ferrotorch-vision::ops` module — detection / segmentation operators: `box_convert` (xyxy / xywh / cxcywh round-trips), `box_iou` (pairwise N×M), `box_area`, `clip_boxes_to_image`, `remove_small_boxes`, `nms` (greedy non-max suppression sorted by descending score), `batched_nms` (per-class via the torchvision shift-by-class trick), `sigmoid_focal_loss` (numerically stable via softplus + composable autograd), `focal_loss` (CE-style on probabilities). 20 new tests covering box-format roundtrips, IoU corner cases, NMS behavior, and focal-loss invariants. RoI ops + GIoU/DIoU/CIoU split out to #610 (#590)
- `ImageFolder` and `DatasetFolder` for `ferrotorch-vision::datasets`. Class-per-subdirectory layout with alphabetical class ordering, case-insensitive extension filtering (default: `jpg`/`jpeg`/`png`/`ppm`/`bmp`/`pgm`/`tif`/`tiff`/`webp`), dotfile/dot-dir skip, and an optional `is_valid_file` predicate. `DatasetFolder<S, F>` is generic over the loader closure so any custom file format works (e.g. audio, binary). Both implement `ferrotorch_data::Dataset`. 11 new tests including a real-PNG decode round-trip (#589)

### Fixed
- Flaky `test_relaxed_one_hot_sample_shape_and_simplex`: strict `(0, 1)` bounds occasionally failed at temperature 0.5 due to softmax underflow / saturation; relaxed to the closed simplex `[0, 1]` which is the mathematically correct support (#589)

### Added
- Chat-template rendering in `ferrotorch-tokenize`: minijinja-backed `apply_chat_template` (Jinja2 over `messages` → string), `apply_chat_template_to_ids` (template + tokenize in one call), `ChatMessage` struct with flattened extra fields for `name`/`tool_calls`/etc., and `load_chat_template` (extracts `chat_template` from a `tokenizer_config.json`, handles both string and array-of-templates forms). Custom `raise_exception` and `strftime_now` Jinja helpers cover the templates used by Mistral and Llama 3.1. 10 new tests (#588)
- Memory-mapped safetensors loaders: `load_safetensors_mmap` (single file) and `load_safetensors_sharded_mmap` (sharded). memmap2-backed; halves peak RSS during decode at LLM scale by letting the OS page cache hold raw tensor bytes instead of `std::fs::read`-ing them into a heap `Vec<u8>`. Decoded `Tensor<T>` buffers remain owned (mmap dropped before return), so callers don't inherit any file-lifetime invariants. 3 new tests including a file-overwrite-after-load check for owned-data semantics. Added `memmap2 = "0.9"` dep to `ferrotorch-serialize`. GGUF / pytorch-pickle mmap split out to #609 (#587)
- Sharded safetensors loader extensions for HF transformer checkpoints (`model.safetensors.index.json`): `load_safetensors_sharded_with_progress` (per-shard progress callback exposing `ShardProgress { shard_index, shard_count, shard_file, tensors_in_shard, tensors_loaded_so_far, total_tensors }`) and `load_safetensors_sharded_filtered` (predicate-driven selective load — skips shards with no matches). The base `load_safetensors_sharded` / `load_safetensors_auto` already shipped earlier; these add the progress UX and selective-load patterns required by inference servers and adapter / LoRA training. 3 new tests (#586)
- `ferrotorch-distributions` property expansion: extended the `Distribution<T>` trait with `cdf`, `icdf`, `mean`, `mode`, `variance`, `stddev` (default impls return `InvalidArgument`; `stddev` defaults to `sqrt(variance)`). Implemented on 12 distributions: Normal (full incl. cdf via erf, icdf via erfinv), Uniform (full), Exponential (full), Laplace (full), Cauchy (cdf/icdf; mean=NaN, var=∞ per torch convention), Bernoulli (cdf, mean, mode, variance), Beta / Gamma / Poisson / LogNormal / HalfNormal / Gumbel (mean / mode / variance, plus cdf/icdf where closed-form exists). 30+ new tests across the affected files (#585)
- `ferrotorch-nn::functional` parity expansion: 29 new fns covering activations (`hardtanh`/`hardtanh_with`, `relu6`, `hardsigmoid`, `hardswish`, `log_sigmoid`, `softmin`, `softsign`, `tanhshrink`, `selu`, `softplus`/`softplus_with`, `elu`/`elu_with`, `mish`, `glu`), losses (`l1_loss`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `kl_div`), distance / normalization (`normalize`, `cosine_similarity`, `pairwise_distance`), and `one_hot` utility. All composable from existing differentiable primitives so autograd works automatically. 29 new unit tests (#584)
- `Module` trait additions: new `Buffer<T>` type for non-trainable persistent state; trait surface for `buffers` / `named_buffers`, `children` / `named_children`, `modules` / `named_modules` (with `descendants_dyn` / `named_descendants_dyn` object-safe variants), `zero_grad`, `requires_grad_`, `apply_to_parameters`. `state_dict` / `load_state_dict` / `to_device` extended to include buffers. All new methods have default implementations preserving backward compatibility. 12 new module tests + 5 new buffer tests (#583)
- Upstream (ferray-fft 0.3.2): N-D Hermitian FFTs `hfftn` / `ihfftn` / `hfft2` / `ihfft2`. Built on existing `rfftn` / `irfftn` with conjugation and norm-swap; defaults to last-two axes for the 2-D variants. 5 new tests (#582)
- Linalg tail: `solve_triangular`, `ldl_factor` / `ldl_solve`, `householder_product`, `matrix_exp` (Padé(13) with scaling-and-squaring), and `cholesky_ex` / `inv_ex` / `solve_ex` non-throwing variants. CPU implementations in pure Rust at f64 internally. 18 unit tests covering forward / back-substitution, transpose, unit-diagonal, LDL reconstruction, matrix-exp identity / diagonal / rotation, and `_ex` info codes (#581)
- Differentiable wrappers + backward fns for `fftn` / `ifftn` / `rfftn` / `irfftn` / `hfft` / `ihfft` (`grad_fns::fft`). Six new `*_differentiable` entry points and matching `*Backward` `GradFn` impls; mirrors `torch.fft` autograd semantics (#580)
- GPU cuFFT dispatch for `linalg::fft` / `ifft` / `rfft` / `irfft` — fully GPU-resident: takes `&CudaBuffer<T>`, plans via `cufftPlan1d`, executes C2C/R2C/C2R, returns `CudaBuffer<T>`. f32 + f64. Inverse paths apply `1/n` normalization to match torch / numpy. 8 integration tests covering forward/inverse roundtrips, batched, f32/f64 (#579)
- Portable cubecl GPU kernels for orthogonal polynomial families (Chebyshev T/U/V/W, Hermite H/He, Laguerre L, Legendre P) via `#[cube]` three-term recurrences. Public `portable_<family>(x, n, rt)` API in `ferrotorch-cubecl::ops`; runs on wgpu/cuda/rocm. 11 integration tests (#577)
- GPU dispatch for `linalg::matrix_norm` (Frobenius) — composes `mul → reduce_sum → sqrt`, fully GPU-resident, returns 0-d device tensor (#576)
- GPU cuSOLVER `eigh` / `eigvalsh` (f32, f64) — fully GPU-resident: takes `&CudaBuffer<T>`, clones via `memcpy_dtod`, exploits symmetric matrix layout (no input transpose), uses on-GPU `gpu_transpose_2d` for output eigenvectors. No host bounces (#575)

### Fixed
- Fix `ptx_f32_to_f64` missing `%off_in`/`%off_out` byte-shift rewrites — caused `gpu_transpose_2d_f64` (and any other gather/scatter f64 kernel) to issue f32-stride loads against an f64 buffer, yielding `CUDA_ERROR_MISALIGNED_ADDRESS`. Surfaced while wiring GPU eigh f64 (#575)
- Fix CUDA stream-capture race in test_gpu_graph_pool (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED under workspace-parallel) (#602)
- Fix flaky test_dirichlet_concentrated: per-sample 0.1 tolerance was ~3.7sigma (0.4% false-fail rate) (#603)
- Fix two flaky tests: cpu_pool global-counter race + gpu_training fresh-batch oscillation (#601)
- cubecl-wgpu tests panic in headless / no-GPU envs (RecvError); gate with availability check (#573)
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

### Fixed
- `DropoutBackward` now stays GPU-native (#525). Previously the backward stored its mask as `Vec<T>` on the host and bailed with `NotImplementedOnCuda` when `grad_output.is_cuda()`. Store the mask as a `Tensor<T>` on the same device as the forward input — for the GPU path the Philox-derived mask is uploaded once during forward, so backward reduces to a single `mul(grad_output, mask_tensor)` that composes with the rest of the f32 GPU autograd chain. 38 dropout lib tests + 1009 ferrotorch-core tests still pass.
- `Tensor::update_storage` leaked the previous `TensorStorage` on every call (#524). The function used `std::ptr::write`, which overwrites the target as uninitialised memory and skips the old value's destructor. AdamW's foreach path calls `update_storage` once per parameter per step, so every training step on GPU leaked the replaced weight storage — including the `GpuBufferHandle` → `CudaBuffer` → pooled `CudaSlice` underneath. Replacing with `std::ptr::replace` + explicit `drop(old)` restores correct single-ownership semantics. Observed effect on `train_hidden_predictor_8b`: GPU VRAM grew ~750 MB/s and OOM'd a 24 GB card before epoch 2 finished; after the fix memory stays flat at ~1 GB across 6 epochs. 1009 `ferrotorch-core` + 322 `ferrotorch-optim` tests still pass.

### Added
- Add JSON-schema constrained-decoding logits processor (ferrotorch-llama::grammar::JsonSchemaProcessor) (#554)
- Add Llama-3.3-70B-Instruct support to ferrotorch-llama (architecture + weight loader + quantization) (#553)
- GPU backward for `abs`, plus on-device `fill` primitive used by `sum` / `mean` backward (#524). Every missing GPU grad that blocked pure-GPU training got a real kernel instead of a CPU round-trip. `abs_backward`: new PTX kernel `abs_backward_kernel` computing `out[i] = input[i] > 0 ? grad[i] : (input[i] < 0 ? -grad[i] : 0)` via `setp.lt/.gt.f32` + `selp.f32` — wired through `GpuBackend::abs_backward_f32` / `_f64` and consumed by `AbsBackward::backward`, mirroring the existing `ReluBackward` / `SigmoidBackward` dispatch. `fill_f32`: new PTX kernel `fill_f32_kernel` that broadcasts a single `.f32` kernel arg to `n` elements of a freshly-allocated `CudaBuffer<f32>`, exposed as `gpu_fill_f32(n, scalar, device)` and `GpuBackend::fill_f32`. `SumBackward` and `MeanBackward` now dispatch to `fill_f32`/`_f64` when the input lives on CUDA, eliminating the legacy `vec![go; numel].to(device)` CPU→GPU upload per backward step (on BCE+Linear at batch 128 / 1920 blocks that upload was ~1 MB per call, ~500 MB per epoch). Combined with the `update_storage` leak fix and the pre-existing `relu`/`sigmoid`/`exp`/`log`/`mul`/`add`/`sub`/`neg` GPU backward paths, the full BCE-with-logits + 2-layer MLP autograd chain now stays on-device end-to-end.
- Native bf16 GPU kernels via nvrtc (#519, in progress). Foundational set for Llama 3 8B on-device inference: `matmul_bf16_bf16` (cublasGemmEx with `CUDA_R_16BF` operands and `CUBLAS_COMPUTE_32F` — f32-accumulator tensor cores), `mul_bf16`, `add_bf16`, `silu_bf16`, `embedding_gather_bf16`, `rmsnorm_bf16`, `softmax_bf16`, `rope_half_bf16`. All non-matmul kernels are CUDA C++ source strings compiled at runtime via `cudarc::nvrtc`, cached per kernel name through a new `get_or_compile_cuda` helper. Storage is `CudaSlice<u16>` (bf16 bit layout); compute is f32 using `__nv_bfloat16` / `__bfloat162float` / `__float2bfloat16` intrinsics from `<cuda_bf16.h>` — matches PyTorch's bf16 GPU architecture. 9 new tests on an RTX 3090 cover: matmul correctness (2x3@3x2, identity, 512x512 finite), elementwise mul/add/silu value parity, embedding row lookup, RMSNorm vs f32 ground truth, softmax rows summing to 1 + ground-truth comparison, RoPE identity at pos=0.
- Mixed-precision kernels: bf16 storage × f32 accumulators (#518). Four hot kernels now detect `T = half::bf16` at runtime and route through an f32-accumulator variant so cascading precision loss doesn't collapse `Tensor<bf16>` forward passes: `mm_raw` / `mm_raw_bt` / `mm_raw_at` direct-loop small-matrix paths (under the 128-element threshold, used by attention head matmuls); `standard_attention` (Q@K^T dot, softmax sum_exp, Attn@V dot all moved to f32 end-to-end for bf16); `RMSNorm` CPU path `mean(x^2)` — summing 4096 bf16 squared values in bf16 was the single biggest killer; `softmax` CPU path's `sum_exp` accumulator and normalization. Large matmuls (max_dim > 128) were already promoting to f64 via the existing faer fallback, so no change there. Concrete effect on the Llama 3 8B end-to-end smoke: `"The meaning of life is"` previously produced the nonsense single-token continuation `"don"` and now decodes through a multi-token greedy loop to `"The meaning of life is a free-line"` — coherent English, each argmax a plausible next token, model stable across the 5-token horizon without NaN / collapse. 2,110 existing lib tests still pass (`ferrotorch-core` 1009, `ferrotorch-llama` 11, `ferrotorch-nn` 1090).
- Llama 3 8B end-to-end smoke example (#513). New `ferrotorch-llama/examples/llama3_8b.rs` ties the whole stack together: resolves the cached `meta-llama/Meta-Llama-3-8B` snapshot, parses `config.json` via `HfTransformerConfig::from_file`, loads `tokenizer.json` via `ferrotorch-tokenize`, loads all 4 safetensors shards into a `StateDict<bf16>` via `load_safetensors_sharded`, constructs a `LlamaForCausalLM<bf16>`, `load_hf_state_dict()` (strict), tokenizes a prompt ("The meaning of life is" by default, overridable via CLI arg), runs a single prefill pass, picks the argmax next token, and prints the continuation. Verified end-to-end: 291 tensors load in 24s; 6-token prefill through all 32 decoder layers completes in 41s and produces a valid vocabulary-range token id that round-trips through the tokenizer back to text. Output coherence is limited by bf16 being the accumulator type in the current generic kernels — mixed-precision (bf16 weights × f32 accumulators) is required for production-quality generation and is the next follow-up.
- `ferrotorch-llama` crate: LlamaConfig + LlamaForCausalLM composition (#512). New workspace crate builds the Llama 3 decoder stack from ferrotorch primitives: `LlamaConfig` (with `from_hf(HfTransformerConfig)` and the canonical `llama3_8b()` factory), `LlamaAttention` (Linear Q/K/V/O + GQA via `repeat_kv` + RoPE `HalfRotation`), `LlamaMLP` (SwiGLU with separate `gate_proj`/`up_proj`/`down_proj` Linear layers for HF weight-name parity), `LlamaDecoderLayer` (pre-norm: RMSNorm → self_attn → residual → RMSNorm → mlp → residual), `LlamaModel` (embedding + N × decoder + final RMSNorm), `LlamaForCausalLM` (model + `lm_head: Linear`). `forward_from_ids(&[u32])` handles the token-id → embedding → decoder → logits path end-to-end. `load_hf_state_dict(..., tie_word_embeddings)` maps HuggingFace names 1:1 onto our parameter paths (no renames needed) and copies `model.embed_tokens.weight` → `lm_head.weight` when the config asks for tied embeddings. Also exposes `repeat_kv`, `reshape_to_heads`, `transpose_heads_to_2d` publicly from `ferrotorch-nn` so the Llama crate can compose them. Current scope: single-batch (`batch=1`) inference; multi-batch needs 4-D reshape helpers and will follow. 11 tests cover config parsing + validation, parameter count, HF-layout named_parameters, forward_from_ids shape, full state_dict round-trip producing identical logits, strict-mode unknown-key rejection, and tied-embedding auto-copy of lm_head.
- NTK-aware / Linear PI / YARN scaling for `RotaryPositionEmbedding` (#515). New `RoPEScaling` enum with variants `None` (default, preserves classical RoFormer), `Linear { factor }` (Chen et al. 2023 positional interpolation), `NtkAware { factor, original_max_pos_embeddings }` (bloc97's NTK-aware scaling, `base' = base * factor^(dim / (dim - 2))`), and `Yarn { factor, original_max_pos_embeddings, beta_fast, beta_slow }` (Peng et al. 2023 piecewise mix between extrapolation and interpolation, with `RoPEScaling::yarn_default` helper supplying the paper's defaults beta_fast=32, beta_slow=1). New constructor `RotaryPositionEmbedding::with_scaling`; `new`/`with_convention` default to `RoPEScaling::None`. Invalid factors rejected. 8 new tests cover default-is-None, none-is-identity, linear halves frequencies, NTK preserves `inv_freq[0]` exactly and stretches `inv_freq[dim/2-1]` by 1/factor, YARN extrapolates high-freq / interpolates low-freq, construction, factor validation. 1090 ferrotorch-nn lib tests pass.
- `ferrotorch-tokenize` crate — HuggingFace `tokenizers` wrapper (#510). New workspace crate wraps `tokenizers = "0.22"` so Llama 3's 128,256-vocab BPE (and any HF `tokenizer.json`) can be loaded without reimplementing BPE. Public API: `load_tokenizer(path)`, `encode(&tok, text, add_special_tokens)`, `encode_batch(&tok, texts, add_special_tokens)`, `decode(&tok, ids, skip_special_tokens)`, `vocab_size`, `token_to_id`, `id_to_token`. Re-exports `tokenizers::Tokenizer` for advanced features (chat templates, truncation, added tokens) accessible without dropping down to the upstream crate directly. 2 unit tests cover malformed / missing files; one `#[ignore]`-gated integration test verifies the real Meta-Llama-3-8B tokenizer loads (vocab=128256), special tokens resolve (`<|begin_of_text|>` → 128000, `<|end_of_text|>` → 128001), BOS prepends correctly on `encode(..., true)`, and round-trip encode→decode preserves text.
- Native `bf16` `Float` impl (Plan A) (#514). Enables `half`'s `num-traits` + `bytemuck` features in the workspace so `half::bf16` implements `num_traits::Float` and `bytemuck::Pod`. Adds `impl Float for half::bf16 {}` in `ferrotorch-core::dtype`, making `bf16` a first-class tensor element type alongside `f32`/`f64`. `Tensor<half::bf16>` now compiles, constructs, and runs the generic `T: Float` op surface (confirmed end-to-end for `from_slice` + `add_t`). `ferray-core`'s existing `Element` impl for `bf16` (gated on the `bf16` feature already enabled in the workspace dep) satisfies the `Element` bound. 7 new tests in `dtype::tests` cover trait bounds, `DType::BF16` tag, zero/one round-trip, `num_traits::Float` composition (addition, sqrt), `AddAssign`, `Tensor<bf16>` construction from a bf16 slice, and `Tensor<bf16>::add_t` yielding byte-correct output. Numerical-sensitivity caveat: the generic kernels accumulate in `T`, so bf16 matmul/softmax/norms should upcast to `f32` for the accumulator when accuracy matters — a follow-up will add mixed-precision kernel specializations.
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
- tensor_bridge.rs cpu_fallback else arms (add/sub/mul/neg/relu): silent CPU round-trip per §3 (#706)
- Restore ops/higher_order.rs and migrate cond/scan canonical impl off cond_scan.rs (#642)
- Copy crosslink-porting skills into ~/.claude/skills (#638)
- Closed audit / tracking issues whose deliverables already shipped: #562 (`docs/audit/01-15-*.md` cover the crate-by-crate gap analysis), #563 (`docs/audit/16-ferray-ferrolearn-integration.md` covers the integration audit), #572 (umbrella tracking #564-#571 — every phased subitem closed).
- Closed previously-shipped issues #414 (gap-analysis result already in tracker comments) and #504 (Adagrad fully implemented in `ferrotorch-optim/src/adagrad.rs` with foreach on-device path, weight_decay, lr_decay, initial_accumulator_value, eps, maximize, plus 24 tests; was already re-exported from lib.rs).
- GPU strided_scatter kernel for as_strided_scatter (CPU-only today) (#574)
- Relax ferrotorch-ml CPU gate: auto-convert GPU tensors to CPU silently (matches torch .cpu().item() flow) (#600)
- Phase 3: create ferrotorch-ml workspace crate bridging Tensor <-> ferrolearn (sklearn metrics, preprocessing, CV, decomposition) (#571)
- Phase 2B: feature-gated Arrow/Polars data interop via ferray-numpy-interop (#570)
- Phase 2A: ferrotorch-core::masked module via ferray-ma (torch.masked parity) (#569)
- Phase 1C extension: integrate 9 new ferray-window 0.3.1 windows into ferrotorch-core::signal::windows (#578)
- Phase 1D: expand ferrotorch-core::special with ferray-polynomial orthogonal polynomial families (#567)
- Phase 1C: add ferrotorch-core::signal::windows wrapping ferray-window (#566)
- Phase 1B: wire ferray-linalg eig/lstsq/lu/etc into ferrotorch-core::linalg + cuSOLVER GPU dispatch (#565)
- Phase 1E: Tensor::as_strided + stride-tricks via ferray-stride-tricks (#568)
- Phase 1A: wire ferray-fft into ferrotorch-core::fft (close 10-fn fft gap) (#564)
- Crate-by-crate audit: ferrotorch vs pytorch parity gap analysis (#561)
- GPU constrained-decoding token-mask compute (qnd-trial-intel-rs#6 stages 1–5):
  - Stage 1: CubeCL `kernel_compute_token_mask_dfa` — one thread per vocab entry walks a schema-derived DFA; new `ferrotorch-cubecl/src/grammar.rs` with 3 cuda_tests on RTX 3090. (commit 0c68a804)
  - Stage 2: `JsonSchemaProcessor` ↔ kernel bridge in `ferrotorch-llama/src/grammar/gpu_dispatch.rs`; supports `Schema::Boolean`. (commit 03a64ff5)
  - Stage 3: extends bridge to `Schema::Null`, `Schema::Integer`, `Schema::Number`, `Schema::String`. (commit 54ecb738)
  - Stage 4: extends bridge to `Schema::StringEnum` (prefix trie) and `Schema::Nullable(_)` (inner-DFA + null-branch merge with class-splitting). (commit 774e5b10)
  - Stage 5: multi-frame scalar dispatch with parent terminators (`,`, `}`, `]` baked into completion states) + `Phase::ObjectKey` prefix trie over unseen properties. (commit 207fd0e3)
  - Total: 31 cuda_tests, all byte-equal to CPU `compute_mask` on RTX 3090, no `#[ignore]`, no CPU fallback inside test bodies. Object/Array structural phases intentionally remain CPU.
- AC-18: scale JsonSchemaProcessor sampled-completion coverage to 10000 per schema (#558)
- Move ferrotorch-paged notes/memories to private repo (#557)
- Extract ferrotorch-paged to private repo at /home/doll/ferrotorch-paged (#556)
- Bump crate version, commit codebase changes, gitignore ferrotorch-paged (#555)
- pin_hot_blocks must stop when VRAM budget is exhausted, not panic (#543)
- Sharded safetensors loader (index.json + multi-file) (#507)
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
