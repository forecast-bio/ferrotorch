# Workspace audit — rust-quality + rust-gpu-discipline

**Date**: 2026-05-02
**Auditor**: y4rr (orchestrator) + 22× sonnet 4.6 subagents (one per crate)
**Tracking**: crosslink issue #648
**Skill set applied**: `rust-quality` (universal Rust discipline) + `rust-gpu-discipline` (GPU-specific anti-patterns)
**Skill activation result**: every subagent successfully loaded both skills via the `Skill` tool — empirical confirmation that general-purpose subagents on Sonnet 4.6 can both see the available-skills list and invoke them.

## Methodology

For each of the 22 workspace crates, a separate Sonnet 4.6 subagent:

1. Loaded both `rust-quality` and `rust-gpu-discipline` skills via the `Skill` tool.
2. Read `Cargo.toml` and `lib.rs` (or `main.rs`).
3. Listed `src/` and grepped for the canonical forbidden-pattern checklists from both skills (`unsafe {`, `// SAFETY:`, `.unwrap()`, `.expect(`, `panic!`, `todo!`, `unimplemented!`, `dbg!(`, `println!`, `eprintln!`, `&String`/`&Vec<T>`/`&PathBuf` parameters, `#[allow]` at crate root, `pub` fields, `// CPU fallback`, `Err(_)` swallowing kernel errors, `.cpu()`/`.to_vec()` in supposed-GPU code, `#[ignore]` on GPU tests, etc.).
4. Read 1–2 specific files where findings clustered.
5. Reported findings in a fixed structured format (vital stats / severity-ordered findings / honest report) capped at ≤600 words.

**No subagent ran `cargo build`/`clippy`/`test`/`miri`/runtime checks.** All findings are static — verified by source reads and grep. Where a finding's severity depends on runtime evidence the subagent could not gather, the report says so.

## Executive summary

ferrotorch is a serious workspace — 22 crates, ~256k LoC, real test coverage in most crates, structured error types, RAII-style GPU resource management — with **two systemic gaps** that dominate the audit and a handful of high-severity local bugs.

The systemic gaps are:

1. **Zero lint discipline anywhere.** Not a single crate has a `[lints]` table; the workspace `Cargo.toml` has no `[workspace.lints]`. Twenty-two crates with ~514 `unsafe` blocks, 256k LoC of public API, and no `clippy::all` / `missing_docs` / `unsafe_code` posture enforced in CI. Every other finding in this audit would be caught earlier with a lint floor in place.

2. **The "appears to be GPU code, secretly host-readback" pattern recurs in 7 crates.** ferrotorch-gpu (f64 ops always CPU; 65+ "CPU fallback" sites in `kernels.rs`), ferrotorch-cubecl (every elementwise op reads back), ferrotorch-xpu (every op reads back), ferrotorch-distributions (every distribution's sample/log_prob/entropy is CPU), ferrotorch-jit (`InductorTarget::GpuCuda` falls through to a CPU interpreter), ferrotorch-distributed's `gpu_collective.rs` is a documented CPU round-trip per op, and ferrotorch-llama has no GPU smoke tests. The `rust-gpu-discipline` skill exists precisely because this pattern is easy to introduce and hard to spot — and it has been introduced systematically.

The high-severity local bugs are: a path-traversal vulnerability in ferrotorch-hub via server-controlled shard filenames; a pre-auth DoS in ferrotorch-serialize's GGUF parser via attacker-controlled `Vec::with_capacity`; a "lying success stub" in ferrotorch-mps's `init_mps_backend`; a fake-GPU test file in the umbrella ferrotorch crate; `Clone::clone` impls that panic in ferrotorch-core (TensorStorage) and ferrotorch-gpu (GpuDevice); raw-pointer aliasing tricks in ferrotorch-optim/adam.rs without rigorous justification; and silent fallback-to-CPU on PTX-compile errors in ferrotorch-gpu that swallow `Err(_)` and discard the existing `GpuError::PtxCompileFailed` variant the codebase already supports.

The bones are solid. The discipline gaps are mostly mechanical fixes (a pre-commit clippy gate, a workspace `[workspace.lints]` table, a single-pass `// SAFETY:` annotation pass, a `cast::<T>()` helper crate-wide). The architectural decisions about CPU-fallback policy require a real design conversation — they're inconsistent across crates (some return errors, some `eprintln!` and degrade, some silently swallow), and that inconsistency is itself the bug.

---

## Cross-cutting findings

The workspace patterns below appear in 5+ crates each. The single-crate findings live in the per-crate appendix.

### 1. No lint configuration anywhere — universal

**Crates affected**: all 22, plus the workspace root.

**Evidence**: every subagent reported "Crate-level lint config: absent — no `[lints]` in `Cargo.toml`, no `#![warn/deny(...)]` in `lib.rs`". The workspace `Cargo.toml` has no `[workspace.lints]` table either, so even an `[lints] workspace = true` in a crate would inherit nothing.

**Why it matters**: the rust-quality skill calls for at minimum `#![warn(clippy::all, clippy::pedantic)]`, `#![deny(unsafe_code)]` (override per-module with `#[allow(unsafe_code)]` + a comment), `#![deny(rust_2018_idioms)]`, `#![deny(missing_debug_implementations)]`, and for libraries `#![deny(missing_docs)]`. Without any of this, every other finding in this audit would have been caught in CI rather than discovered by reading source.

**Action** (workspace-wide):

```toml
# workspace Cargo.toml
[workspace.lints.rust]
unsafe_code = "deny"        # override per-module where needed
missing_docs = "warn"
missing_debug_implementations = "warn"
rust_2018_idioms = "deny"

[workspace.lints.clippy]
all = "warn"
pedantic = "warn"
unwrap_used = "warn"
expect_used = "warn"
panic = "warn"
todo = "warn"
unimplemented = "warn"
dbg_macro = "warn"
print_stdout = "warn"
clone_on_ref_ptr = "warn"
```

Then in every crate's `Cargo.toml`:

```toml
[lints]
workspace = true
```

Expect a flood of warnings on first run; resolve in a series of crate-scoped PRs.

### 2. SAFETY-comment coverage is ~18% workspace-wide

**Aggregate count**: roughly **93 `// SAFETY:` comments** for **~514 `unsafe {` blocks** across the workspace.

**Worst offenders**:

| Crate | `unsafe {` | `// SAFETY:` | Coverage |
|---|---:|---:|---:|
| ferrotorch-cubecl | 28 | 0 | 0% |
| ferrotorch-core | 103 | 14 | ~14% |
| ferrotorch-gpu | 268 | 42 | 16% |
| ferrotorch-serialize | 16 | 4 | 25% |
| ferrotorch-distributed | 35 | 10 | ~29% |
| ferrotorch-optim | 61 | 22 | 36% |
| ferrotorch-jit | 3 | 1 | ~33% |
| ferrotorch-nn | 2 | 0 | 0% |
| ferrotorch-profiler | 8 (test only) | 1 | ~12% |
| ferrotorch-hub | 6 (test only) | 2 | 33% |

**Why it matters**: every `unsafe {}` block is asserting an invariant. Without a written `// SAFETY:` comment, miri can't help, reviewers can't tell mechanical kernel-launches apart from genuine pointer tricks, and the codebase loses the audit trail that would justify its `unsafe`-heavy posture.

**Hot zones**:

- ferrotorch-gpu/`bf16.rs` (15 blocks, 0 SAFETY)
- ferrotorch-gpu/`tensor_bridge.rs` (~30 blocks of `transmute_buffer_ref::<T, f32>(...)` for dtype dispatch — generic transmutes that genuinely need the invariant written)
- ferrotorch-gpu/`kernels.rs` (~150 kernel-launch unsafe blocks, 4 SAFETY)
- ferrotorch-cubecl entire crate (zero coverage)
- ferrotorch-core/`storage.rs`, `tensor.rs`, `ops/linalg.rs` (TypeId-gated transmutes)
- ferrotorch-distributed/`nccl_backend.rs` (dlopen/dlsym via `transmute` with no ABI justification)
- ferrotorch-optim/`adam.rs:356` (`Arc::as_ptr(...) as *mut T` then `&mut *p` — aliasing-rule violation if any other Arc clone is alive)

**Action**: dedicated annotation-pass PR per crate. Use `cusolver.rs` (the one ferrotorch-gpu module that already does this consistently) as the template. Most blocks resolve to one-line comments like `// SAFETY: kernel arity matches signature; CudaBuffer-owned slices outlive the launch`.

### 3. Systemic `T::from(x).unwrap()` / `.to_f64().unwrap()` panic vectors

**Affected crates** (with site counts where reported):

- ferrotorch-distributions: ~350 sites in `special_fns.rs` and per-distribution files
- ferrotorch-optim: ~123 sites across optimizers
- ferrotorch-vision: 60+ sites in `ops.rs`, `transforms/*`
- ferrotorch-core: ~14 sites in `fft.rs` plus scattered
- ferrotorch-llama: model.rs, mlp.rs, generation.rs (hot path including beam_search)
- ferrotorch-train: grad_utils.rs (every batch in `fit`/`evaluate`), callback.rs (EmaCallback)
- ferrotorch-nn: utils.rs (clip_grad_norm/value)
- ferrotorch-ml: metrics.rs, adapter.rs
- ferrotorch-data: transforms.rs (Normalize::new constructor)
- ferrotorch-serialize: state_dict.rs, checkpoint.rs (untrusted-input parsing!)
- ferrotorch-jit: interpreter.rs (constant evaluation)

**The pattern**:

```rust
let scaled = T::from(0.5).unwrap();           // panics if T can't represent 0.5
let f = v.to_f64().unwrap();                   // panics if v can't convert
```

`num_traits::NumCast::from` and `ToPrimitive::to_f64` both return `Option<T>`. The `.unwrap()` is structurally a panic on conversion failure. In practice `T: Float` is `f32` or `f64` so the conversions usually succeed, but: (a) `bf16` is in the workspace deps and `Float` covers it, (b) NaN/Inf on some conversions return `None`, (c) the bound doesn't *guarantee* infallibility, so the codebase has hundreds of latent panic sites.

**Action**: introduce a workspace-shared helper, e.g. in ferrotorch-core:

```rust
pub fn cast<T: NumCast>(v: f64, what: &'static str) -> FerrotorchResult<T> {
    T::from(v).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("cannot cast {v} to target type ({what})"),
    })
}
```

Then a workspace-wide grep-and-replace turns every `T::from(x).unwrap()` into `cast(x, "context")?`. Equivalent for `.to_f64()`. This is a one-PR fix that eliminates hundreds of panic sites at once.

### 4. Silent CPU host-readback under GPU-flavored names

The rust-gpu-discipline skill enumerates 10 forbidden patterns. The audit found pattern matches in 7 crates:

**Anti-pattern #1 (fake-GPU function)** + **#5 (dispatch-only dispatch)** + **#7 (synchronous host-readback)**:

| Crate | Site | What happens |
|---|---|---|
| ferrotorch-gpu | `kernels.rs` (65+ sites incl. lines 14309, 17270, 17711, 17924, 18008, 18344) | PTX compile error → `Err(_) =>` → `gpu_to_cpu` → CPU compute → `cpu_to_gpu` round-trip. `gpu_embed_lookup` swallows silently; LayerNorm/RMSNorm `eprintln!` and proceed. |
| ferrotorch-gpu | All `*_f64` ops (`gpu_add_f64`, `gpu_sub_f64`, `gpu_mul_f64`, `gpu_relu_f64`, `gpu_gelu_f64`, …) | Always falls back to CPU because no f64 PTX kernels exist. No rustdoc warning to caller. |
| ferrotorch-cubecl | `kernels.rs:408,432,502,574` (every `run_unary`, `run_binary`, `run_matmul`) | Every op calls `client.read_one(out_handle)` (implicit sync + DMA back) before returning. Chained ops thus round-trip per op. `quant.rs`/`grammar.rs` correctly return raw `Handle`s; the elementwise path does not. |
| ferrotorch-xpu | `lib.rs:147,161` (every `xpu_binary!`/`xpu_unary!` macro expansion) | Each op calls `result_on_cubecl.data_vec()?` (device→host) then re-tags as `Device::Xpu`. Three round-trips for `xpu_add → xpu_relu → xpu_exp`. |
| ferrotorch-distributions | All sample/rsample/log_prob/entropy paths in `normal.rs`, `exponential.rs`, `multivariate_normal.rs`, etc. | `data_vec()` (device→host) → CPU compute → `TensorStorage::cpu(...)` → `.to(device)`. No GPU kernel path. The `is_cuda()` branches only re-upload after computing on host. |
| ferrotorch-jit | `codegen.rs:871-886` | `InductorTarget::GpuCuda` and `GpuPtx` generate source strings, discard them via `let _sources`, then fall through to `InterpreterBackend.compile(graph)` which runs on CPU. The user thinks they got a GPU kernel; they got a CPU interpreter. |
| ferrotorch-distributed | `gpu_collective.rs:1-13` | Module docstring openly admits the GPU→CPU→collective→CPU→GPU round-trip per call. Honest, but never surfaced through `tracing::warn!` so callers are silently surprised by the cost. |
| ferrotorch-nn | `utils.rs:44-76, 108-128` (`clip_grad_norm_`, `clip_grad_value_`) | Every call on GPU parameters: `g.data_vec()` → CPU compute → `TensorStorage::cpu(...)` → `.to(device)`. Public methods, no doc warning. |
| ferrotorch-core | `gpu_dispatch.rs:1608-1613` (`has_inf_nan_f32` default impl) | Default trait method does `gpu_to_cpu(a)?` and scans on CPU. Any backend that forgets to override it does host-readback on every NaN check. |

**Action — architectural decision required**: the workspace currently has three different policies for "GPU op can't run":

1. ferrotorch-gpu/bf16.rs returns `Err(GpuError::PtxCompileFailed)` (the right pattern; `eprintln!` aside).
2. ferrotorch-gpu/kernels.rs swallows `Err(_)` and silently does CPU.
3. ferrotorch-cubecl, ferrotorch-xpu, ferrotorch-distributions: always do CPU and return without indicating it.

**Recommended unified policy**: when a kernel can't run, return `Err(...PtxCompileFailed | Unsupported | DeviceUnavailable)` with the kernel name. Caller can catch and decide whether to fall back. Add a workspace-level `#[non_exhaustive]` enum capability flag (`DeviceCapabilities`) that callers can query *before* dispatching, so the failure mode is "I knew I couldn't do f64 on GPU, so I asked for CPU explicitly." The current "silent degradation" model is the worst of both worlds.

The `GpuError::PtxCompileFailed { kernel: &'static str }` variant **already exists in ferrotorch-gpu** — kernels.rs just chooses not to use it.

### 5. Pervasive `pub` fields on library structs

Every crate has this pattern. Worst offenders:

- ferrotorch-jit: every IR type (`IrGraph`, `IrNode`, `IrValue`, `AotGraphPair`, `FusionGroup`, `CompileConfig`) — downstream can mutate `IrGraph::nodes` directly, bypassing `next_value_id`/`next_node_id` invariants.
- ferrotorch-nn: nearly every loss struct (`reduction`, `delta`, `margin`, `label_smoothing`, `blank`, `eps`, …). 23+ fields enumerated by the subagent.
- ferrotorch-optim: every `*Config` struct (`AdamWConfig`, `AdamConfig`, `KfacConfig`, `MuonConfig`, …) plus `ParamGroup.params: pub Vec<Parameter<T>>`.
- ferrotorch-distributed: `ShardMetadata`, `TensorShardSpec`, `DistributedCheckpoint`, `SyncBatchNorm2d`.
- ferrotorch-llama: `GptqQ4`, `AwqQ4`, `HqqWeights`.
- ferrotorch-vision, ferrotorch-data, ferrotorch-hub, ferrotorch-serialize, ferrotorch-tokenize, ferrotorch-core (`GpuRngState`), ferrotorch-train (`EpochResult`, `EvalResult`, `TrainingHistory`), ferrotorch-cubecl (`DfaMaskInputs`).

**Why it matters**: every `pub` field is part of the semver-stable surface. Renaming, retyping, or replacing a field with computed-on-access logic is a major version bump. Adding a field breaks struct-literal construction in downstream code.

**Action — three-tier fix**:

1. **Pure data-bag types** (e.g., `EpochResult`, `MemoryStats`, `ChatMessage`): add `#[non_exhaustive]` so downstream can't construct via struct literal but can read fields. Cheapest fix.
2. **Configuration types** (`*Config`): convert to private fields + builder, OR `#[non_exhaustive]` + `Default + struct-update` pattern. The crate already provides `Default` impls in many places.
3. **State-bearing types** (`IrGraph`, `ParamGroup`, `GpuRngState`): make fields private and expose accessors. These have invariants that `pub` fields actively violate.

### 6. `thiserror` is in the workspace but multiple crates hand-roll `Display + Error`

**Workspace `Cargo.toml`:81** declares `thiserror = "2.0"`.

**Hand-rolled instead**:

- ferrotorch-gpu/`error.rs` (`GpuError`): manual 120-line `fmt::Display` match + `Error::source` impl + per-variant `From` impls. ~190 lines that `#[derive(thiserror::Error)]` would collapse to ~80.
- ferrotorch-llama/`grammar/{schema,state,json_schema}.rs`: three error enums each with manual `Display + Error`.

**Action**: add `thiserror.workspace = true` to each crate's `[dependencies]` and convert. Saves boilerplate, removes drift risk between `Display` text and variant data, and the `#[from]` and `#[source]` attributes are more discoverable than hand-written `From` impls.

### 7. Public types missing `Debug`

The rust-quality rule: every public type should impl `Debug`. The `#![deny(missing_debug_implementations)]` lint catches this once enabled.

- ferrotorch-data: `DataLoader<D>`, `BatchIter`, `DataLoaderIter`, `PrefetchIter`, `MultiWorkerIter`, `CollatedIter`, all 5 `Transform` impls (`Compose`, `Normalize`, `ToTensor`, `RandomHorizontalFlip`, `RandomCrop`).
- ferrotorch-distributions: every public distribution (`Normal`, `Beta`, `Bernoulli`, `Categorical`, `Independent`, `MixtureSameFamily`, `StudentT`, `Gamma`, `Laplace`, `Gumbel`, `HalfNormal`, `Cauchy`, `Uniform`, …).
- ferrotorch-vision: `RawImage`, `Cifar10<T>`, `Cifar100<T>`.
- ferrotorch-profiler: `Profiler`, `ProfileReport`.
- ferrotorch-llama: `LlamaGpuLayer`, `LlamaGpuInferencer` (blocked by `cudarc::CudaSlice` not impl `Debug` — needs manual impl), `PackedVocab`.
- ferrotorch-hub: `HubCache`.
- ferrotorch-mps: `MpsDevice` is missing `PartialEq`/`Eq`/`Hash`.

**Action**: derive `#[derive(Debug)]` where possible; manual impl where blocked by FFI types (cudarc, libloading, Box<dyn Fn>). Driven by enabling `missing_debug_implementations` lint.

---

## Workspace severity tally

| Crate | 🔴 | 🟡 | 🟢 |
|---|---:|---:|---:|
| ferrotorch | 2 | 4 | 2 |
| ferrotorch-core | 4 | 6 | 2 |
| ferrotorch-nn | 3 | 6 | 2 |
| ferrotorch-nn-derive | 2 | 4 | 2 |
| ferrotorch-optim | 5 | 5 | 2 |
| ferrotorch-data | 4 | 5 | 4 |
| ferrotorch-train | 5 | 5 | 2 |
| ferrotorch-vision | 4 | 5 | 2 |
| ferrotorch-jit | 5 | 4 | 2 |
| ferrotorch-jit-script | 2 | 5 | 2 |
| ferrotorch-serialize | 3 | 5 | 2 |
| ferrotorch-gpu | 3 | 7 | 2 |
| ferrotorch-cubecl | 2 | 4 | 2 |
| ferrotorch-xpu | 3 | 5 | 2 |
| ferrotorch-distributed | 5 | 5 | 2 |
| ferrotorch-distributions | 4 | 5 | 2 |
| ferrotorch-profiler | 3 | 8 | 2 |
| ferrotorch-hub | 3 | 6 | 2 |
| ferrotorch-tokenize | 1 | 6 | 2 |
| ferrotorch-llama | 4 | 5 | 2 |
| ferrotorch-ml | 1 | 4 | 2 |
| ferrotorch-mps | 3 | 4 | 1 |
| **Totals** | **75** | **117** | **47** |

---

## Workspace-level action plan (ranked)

These actions in this order would close the largest portion of the audit's findings with the smallest effort:

1. **Add `[workspace.lints]` to root `Cargo.toml` + `[lints] workspace = true` to every crate.** One PR. Adds the safety net. Then a series of small per-crate PRs to clear the resulting warnings. Closes finding #1 directly and provides CI gating for findings #5, #6, #7 going forward.

2. **Add a `cast<T: NumCast>(v: f64) -> FerrotorchResult<T>` helper to ferrotorch-core and grep-replace `T::from(x).unwrap()` workspace-wide.** One PR. Eliminates ~600+ panic sites in one shot.

3. **Fix the path-traversal in ferrotorch-hub.** Sanitize `relative` shard filenames against `..`, leading `/`, and null bytes before building cache paths. (Security bug — should be done before the lint sweep, but it's a small, isolated change.)

4. **Fix the GGUF parser DoS in ferrotorch-serialize.** Cap `metadata_kv_count` and `tensor_count` against remaining buffer length before pre-allocating. Same urgency as #3.

5. **Decide and document a workspace-wide CPU-fallback policy.** Currently inconsistent across ferrotorch-gpu (silent), ferrotorch-cubecl (silent), ferrotorch-xpu (silent), ferrotorch-distributions (silent), ferrotorch-jit (silent fall-through to interpreter), ferrotorch-distributed/gpu_collective (documented but not surfaced). Recommend: **return `Err(...PtxCompileFailed | Unsupported)` always**, and add a `DeviceCapabilities` query API for callers that want to opt into a fallback. Then audit each silent fallback site.

6. **`Clone::clone` panic fixes.**
   - ferrotorch-core/`storage.rs:304,307` — `Clone for TensorStorage` calls `panic!` on GPU clone; introduce `try_clone() -> Result` and either remove `Clone` or use a sentinel.
   - ferrotorch-gpu/`device.rs:99` — `Clone for GpuDevice` `.expect("CudaBlas::new failed")`; wrap `CudaBlas` in `Arc<CudaBlas>` so `clone()` becomes `Arc::clone`.

7. **`init_mps_backend` honesty fix in ferrotorch-mps.** Currently returns `Ok(())` while registering nothing (textbook anti-pattern #5). Either make it `Err(DeviceUnavailable)` or actually wire the backend.

8. **Fix the fake-GPU test file in ferrotorch.** `tests/gpu_training.rs` has 6 tests, all named `*_cpu`, none constructing a GPU device. Either rename to `cpu_training.rs` or rewrite to actually use `Device::Cuda`.

9. **Fix the silent CPU fall-through in ferrotorch-jit's GPU compile targets.** Either implement real GPU compilation via cudarc or return `JitError::Unimplemented` for `InductorTarget::GpuCuda`/`GpuPtx`.

10. **`SAFETY:` annotation pass per backend crate.** Worst-first: ferrotorch-cubecl (0 → required), ferrotorch-gpu/bf16.rs, ferrotorch-gpu/kernels.rs, ferrotorch-distributed/nccl_*.rs, ferrotorch-core unsafe transmute clusters, ferrotorch-optim/adam.rs (especially the `Arc::as_ptr` aliasing dance which warrants more than a one-line comment).

11. **Migrate hand-rolled errors to `thiserror`.** ferrotorch-gpu/error.rs (~190 lines → ~80), ferrotorch-llama/grammar/{schema,state,json_schema}.rs (3 enums).

12. **`#[derive(Debug)]` sweep and `#[non_exhaustive]` on data-bag structs.** Once `missing_debug_implementations` is warning, this is mechanical.

13. **`.unwrap()` removal from doctests, library-code `eprintln!`/`println!` → `tracing::warn!`, miscellaneous.** Cleanup that the lint floor will surface naturally.

---

## Per-crate appendix

### Crate: ferrotorch

#### Vital stats
- **Kind**: lib (umbrella re-export crate)
- **.rs files**: 1 (src/lib.rs only; all logic lives in sub-crates)
- **Approx LoC**: 127 (lib.rs) + ~1,400 across 4 test files + 2 example files
- **`unsafe {` blocks**: 0 (in this crate's src/)
- **`// SAFETY:` comments**: 0 (none needed — no unsafe)
- **Tests**: yes (4 integration test files: gpu_training.rs 6 tests, validate_vs_pytorch.rs 24 tests, pythia_real.rs 2 tests, public_surface.rs 16 tests)
- **Crate-level lint config**: absent — no `[lints]` in ferrotorch/Cargo.toml, no `[lints]` in workspace Cargo.toml, no `#![warn/deny(...)]` in lib.rs
- **Touches GPU**: yes (feature-gated re-exports of ferrotorch-gpu, ferrotorch-cubecl, ferrotorch-mps, ferrotorch-xpu)
- **Edition / MSRV**: edition = "2024", rust-version = "1.85" (workspace-inherited)

#### Findings

🔴 **`gpu_training.rs` is a fake-GPU test file** — `tests/gpu_training.rs:1–693` — The file is named `gpu_training.rs`, registered as `[[test]] name = "gpu_training"`, and its module doc says "End-to-end GPU training integration tests." Every single one of its 6 test functions is named `test_*_cpu` and constructs tensors with no device specification (CPU default). There is zero use of `Device::Cuda`, `.cuda()`, `cuda_device`, `#[cfg(feature = "gpu")]`, or any GPU handle anywhere in the file. This is the "fake-GPU function" anti-pattern from rust-gpu-discipline §2 (#1 and #5) applied at the test-file level: a file that promises GPU coverage but delivers none. — **Action**: either rename the file to `cpu_training.rs` and add real `#[cfg(feature = "gpu")]` GPU tests, or gate every test in it with `#[cfg(feature = "gpu")]` and add GPU device construction.

🔴 **No crate-level lint config anywhere** — `ferrotorch/Cargo.toml` and workspace `Cargo.toml` — Neither `[lints]` nor `#![warn/deny(...)]` appear at the workspace level or in this crate. Without this, all clippy pedantic lints, `unwrap_used`, `missing_docs`, and the correctness group are silently un-enforced in CI. — **Action**: add a `[lints.rust]` + `[lints.clippy]` table to workspace `Cargo.toml` and have `ferrotorch/Cargo.toml` inherit it with `[lints] workspace = true`.

🟡 **`validate_vs_pytorch.rs` uses `println!` as test output** — `tests/validate_vs_pytorch.rs:33,60,81,…` (24 occurrences) — Test files use `println!("PASS: …")` to report sub-assertions. Rust's test harness captures stdout; these prints are invisible unless `-- --nocapture` is passed and do not constitute actual assertions. The pattern also looks like leftover manual-test-script style. — **Action**: replace informational `println!` with proper `assert!` / `assert_eq!` with descriptive messages, or use `eprintln!` consistently (the pythia test does this correctly).

🟡 **`public_surface.rs` top-level `#[allow(unused_imports)]`** — `tests/public_surface.rs:12` — The file-level `#![allow(unused_imports)]` is broad. The comment explains the pattern (`use … as _` is intentionally unused), which is good, but the blanket allow covers the whole file rather than being scoped to each `use` site. Note: `as _` already suppresses the warning, making the file-level allow redundant. — **Action**: remove the `#![allow(unused_imports)]` banner.

🟡 **`pythia_real.rs` uses `.unwrap()` prolifically in integration test code** — `tests/pythia_real.rs:196,228–247` — While `tests/` is an allowed zone for `.unwrap()`, the pythia test is heavyweight (builds a 70 M-parameter model, trains 5 steps). Panics from `.unwrap()` produce opaque messages with no context. — **Action**: replace bare `.unwrap()` with `.expect("descriptive context")`.

🟡 **Doctests use `.unwrap()` (bad teaching pattern)** — While lib.rs itself has no doctests, the public surface of ferrotorch is the primary user-facing crate. — **Action**: add at least one `# Examples` doctest per public module in lib.rs using `?` rather than bare `.unwrap()`.

🟢 **`ferrotorch_bench.rs` example silently skips GPU benchmarks without an error** — `examples/ferrotorch_bench.rs:347` — The GPU bench section prints `"(no GPU backend available — skipping GPU benchmarks)"` but exits normally with `Ok(())`. — **Action**: consider using `eprintln!` and/or returning a non-zero exit code.

🟢 **No `[[test]] path` declaration for `validate_vs_pytorch.rs`** — `ferrotorch/Cargo.toml` — Inconsistent with the explicit style chosen for the other two declared tests. — **Action**: either declare all four as `[[test]]` entries or remove the explicit entries.

#### Honest report
- **Did**: read `ferrotorch/Cargo.toml`, `ferrotorch/src/lib.rs`, workspace `Cargo.toml`; read all 4 test files (full content); read both example files (via grep output); grepped for `unwrap`/`expect`, `todo!/unimplemented!/dbg!/panic!/println!/eprintln!`, `unsafe {`, `// SAFETY:`, `#[allow]`, `&String`/`&Vec`/`&PathBuf`, `#[ignore]`/GPU early-exit patterns, `cuda_available`/`Device::Cuda`/`.cuda()` in gpu_training.rs, lint config in all Cargo.toml files.
- **Did NOT do**: cargo build, cargo clippy, cargo test, miri, runtime execution.
- **Confidence**: high — the crate is a thin re-export layer (127 lines of lib.rs). The most significant finding (fake-GPU test file) is a factual observation from the file contents.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-core

#### Vital stats
- **Kind**: lib
- **.rs files**: 70
- **Approx LoC**: ~55,700
- **`unsafe {` blocks**: 103
- **`// SAFETY:` comments**: 14 (covers 13 blocks; **90 unsafe blocks have no SAFETY comment**)
- **Tests**: yes (64 `#[cfg(test)]` modules)
- **Crate-level lint config**: absent
- **Touches GPU**: yes — `gpu_dispatch.rs` defines the `GpuBackend` trait; `storage.rs` and `grad_fns/` call through to it; no `cudarc`/`cubecl` in this crate's direct deps (runtime is dynamically registered by `ferrotorch-gpu`)
- **Edition / MSRV**: edition 2024, rust-version 1.85

#### Findings

🔴 **`Clone` impl on `TensorStorage` calls `panic!`** — `src/storage.rs:304,307` — `impl Clone` contains `panic!("failed to clone GPU buffer")` and `panic!("no GPU backend registered")`. `Clone` is contractually infallible. Action: replace with a `try_clone() -> Result<Self>` method and either remove `Clone` or make the GPU arm use a sentinel/zeroed buffer.

🔴 **90 `unsafe {}` blocks missing `// SAFETY:` comments** — widespread across `storage.rs`, `tensor.rs`, `ops/linalg.rs`, `grad_fns/reduction.rs`, `inplace.rs`, `cpu_pool.rs`, `grad_fns/indexing.rs`, `autograd/saved_tensors.rs` — Many are type-punning transmutes (`*const [T] as *const [f32]`) where the invariant (TypeId equality) is local but not written. Action: annotate every block; the TypeId-gated casts in `ops/linalg.rs` are the most numerous cluster.

🔴 **`has_inf_nan_f32` default in `GpuBackend` silently downloads to host** — `src/gpu_dispatch.rs:1608-1613` — The default implementation calls `self.gpu_to_cpu(a)?`, converting the GPU buffer to a host `Vec<u8>` and scanning on CPU. Any backend that forgets to override it will silently do a full host-readback on every NaN check. This is anti-pattern #7 (synchronous host-readback). Action: remove the default body and require all backends to implement it.

🔴 **`as_slice` / `as_mut_slice` panic on GPU tensors** — `src/storage.rs:168,171,181,184` — `pub` methods that `panic!` rather than returning `Result`. Action: change to return `Option<&[T]>` or `Result<&[T], FerrotorchError>`.

🟡 **`eprintln!` in library code** — `src/quantize.rs:652,684` — `PerChannelMinMaxObserver::observe_with_shape` and `observe` both `eprintln!("WARNING: …")` before returning `Err`. The warning text is duplicated in the `Err` payload. Action: remove the `eprintln!` calls; the `Err` return is sufficient.

🟡 **`println!` in a public library method** — `src/methods.rs:296` — `pub fn print(&self)` calls `println!("{self}")`. Action: emit via `tracing::info!`.

🟡 **No crate-level lint configuration** — lib.rs has zero `#![warn/deny]` attributes. Given the crate's size (55 kLoC) and public surface, at minimum `#![deny(unsafe_code)]` with per-module overrides, `#![warn(missing_docs)]`, and `#![warn(clippy::all)]` should be present.

🟡 **`GpuRngState` has fully `pub` fields** — `src/gpu_dispatch.rs:25-31` — `counter`, `seed`, `offset`, `device` are all `pub`. For a checkpoint-restoration type meant to be opaque to callers this leaks internal layout. Action: make fields `pub(crate)` and add constructor / accessor methods.

🟡 **~93 public functions lack rustdoc** — checked mechanically across 784 `pub fn` items; approximately 93 have no `///` comment. Action: enable `#![warn(missing_docs)]` and address.

🟡 **`unwrap()` on numeric conversions in hot paths** — `src/fft.rs:53,54,177,178,337,354,355,454,572,601,676,693,694,706` and many more — `T::from(c.re).unwrap()`, `v.to_f64().unwrap()` etc. Will panic if the `num-traits` conversion returns `None`. Action: replace with `T::from(c.re).ok_or(FerrotorchError::…)?`.

🟢 **`#[allow(clippy::too_many_arguments)]` repeated 9 times** — `src/gpu_dispatch.rs:1576,1591,1989,2030,2049,2070,2089,2112,2127` — Each is item-level (correct placement) but the trait methods all take 7-9 arguments. Consider grouping related args into small parameter structs.

🟢 **nit: `Vec<&String>` locals in `einops.rs`** — `src/einops.rs:612,687,754` — `Vec<&str>` would be more idiomatic.

#### Honest report
- **Did**: listed all src files, read `Cargo.toml`, workspace `Cargo.toml`, `lib.rs`, `gpu_dispatch.rs` (full head + key sections), `storage.rs` (Clone impl, unsafe blocks), `ops/linalg.rs` (unsafe transmute clusters), `inplace.rs`, `creation.rs`, `fft.rs` (unwrap cluster), `grad_fns/reduction.rs`, `quantize.rs` (eprintln), `methods.rs` (println), `autograd/saved_tensors.rs`, `masked.rs`, `einops.rs`. Ran greps for all forbidden patterns. Ran a Python script to precisely count unsafe blocks vs SAFETY coverage and undocumented public functions.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo miri`, any file edits or writes.
- **Confidence**: high for the structural findings; medium for the `unwrap()` severity classification in `fft.rs` (some may be on numerically-safe paths but the invariant is not documented).
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-nn

#### Vital stats
- **Kind**: lib
- **.rs files**: 34
- **Approx LoC**: ~52,100
- **`unsafe {` blocks**: 2
- **`// SAFETY:` comments**: 0
- **Tests**: yes (33 `#[cfg(test)]` modules)
- **Crate-level lint config**: absent
- **Touches GPU**: indirectly — `ferrotorch-core` is the GPU dependency; `ferrotorch-nn` performs host-readback via `data_vec()` and `TensorStorage::cpu(...)` in production code paths; `embedding.rs` directly calls `gpu_backend()` / `cpu_to_gpu`
- **Edition / MSRV**: Edition 2024, MSRV 1.85

#### Findings

🔴 **Two `unsafe` blocks with zero `// SAFETY:` comments** — `embedding.rs:40` and `embedding.rs:521` — Both cast `*const T` to `*const u8` via `std::slice::from_raw_parts` with manually computed byte lengths; the correctness relies on `T` being `repr(C)`/no-padding, size calculation being exact (`data.len() * 4` hardcodes 4 for `f32` at line 40 but uses `size_of::<T>()` at line 521 — the inconsistency is itself a bug risk), and the input slice being valid for the entire computed range. Add `// SAFETY:` blocks naming the invariants.

🔴 **`clip_grad_norm_` and `clip_grad_value_` are silent CPU detours for GPU gradients** — `utils.rs:44–76` and `utils.rs:108–128` — Both call `g.data_vec()` (GPU→host readback per `tensor.rs:673–674`), compute the operation entirely on CPU (`TensorStorage::cpu(scaled)` at line 76, `TensorStorage::cpu(clamped)` at line 123), then push back to device with `.to(device)?`. Synchronous host-readback anti-pattern. Public functions, no `# Panics` or `# Errors`, GPU readback invisible to callers.

🔴 **`T::from(clip_coef).unwrap()` panics on `bf16`** — `utils.rs:68, 103, 104` — `Float` is implemented for `half::bf16`, and `num_traits::NumCast::from::<f64>` for `bf16` returns `None` when the value overflows bf16 range. Same for `v.to_f64().unwrap()` at lines 46 and 59. These are production code paths.

🟡 **No crate-level lint config** — A 52 kLoC library crate with no `#![warn(clippy::all)]`, no `#![deny(unsafe_code)]`, no `#![deny(missing_docs)]`. Any new `unsafe` block or missing doc silently ships.

🟡 **Eleven loss backward functions explicitly unimplemented on CUDA** — `loss.rs:313, 922, 1098, 1341, 1535, 1758, 1944, 2223, 2665, 2845, 2995, 3206` — Each returns `Err(FerrotorchError::NotImplementedOnCuda { op: "..." })`. Correct error-returning pattern (not a silent CPU detour). However, none of these `Result`-returning functions have a `# Errors` rustdoc section documenting this error condition.

🟡 **Pervasive `pub` fields on library structs** — `loss.rs:46, 163, 164, 376, 506, 507, 645, 775, 776, 1027, 1179, 1181, 1402, 1446, 1630–1632, 1853, 1854, 2027–2029, 2395–2397, 2548–2550, 2741, 2903` — Nearly every loss struct exposes its configuration fields as `pub`. This locks the ABI.

🟡 **`lazy_norm.rs:88, 176` — `.expect("inner initialized")` / `.expect("inner")` in production forward path** — These follow an explicit `materialize()` call and a `is_none()` guard, so they should be unreachable. However, the invariant is not written as a comment, and if `materialize()` fails, the panic fires in a user's model forward pass. Use `ok_or(FerrotorchError::UninitializedModule)` instead.

🟡 **`parameter_container.rs:167` — `impl Iterator<Item = &String>` instead of `impl Iterator<Item = &str>`** — `keys()` iterates over `HashMap<String, ...>` keys and exposes `&String`; callers receive a strictly less ergonomic type.

🟡 **Doctests in `container.rs:167, 312, 313` use `.unwrap()`** — Inside `///` doc examples shown to library users as canonical usage.

🟢 **`#[allow(clippy::too_many_arguments)]` repeated across 11 sites without comments** — `conv.rs:30, 95, 1845, 1926, 3249`, `padding.rs:104, 254, 406, 523, 639`, `upsample.rs:255, 284, 346, 500, 529, 587` — Item-scoped (correct), but no comments explaining why a builder/config struct isn't used.

🟢 **No `prelude` module** — `lib.rs` re-exports everything directly at the crate root.

#### Honest report
- **Did**: read `Cargo.toml`, `lib.rs`, `utils.rs`, `lazy_norm.rs`, `embedding.rs`, `loss.rs` (selectively), `paged_attention.rs`, `parameter_container.rs`; grepped patterns; read `tensor.rs:664` to confirm `data_vec()` readback behaviour; read `ferrotorch-core/src/dtype.rs` to confirm `Float` includes `bf16`.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, `miri`, runtime execution.
- **Confidence**: high for the issues flagged; medium on doc coverage completeness.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-nn-derive

#### Vital stats
- **Kind**: proc-macro (`[lib] proc-macro = true`)
- **.rs files**: 1 (`src/lib.rs`)
- **Approx LoC**: 348
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0 (moot — no unsafe)
- **Tests**: no — no `#[cfg(test)]` block, no `tests/` directory, no trybuild harness
- **Crate-level lint config**: absent
- **Touches GPU**: no (correct for a proc-macro)
- **Edition / MSRV**: edition 2024, rust-version 1.85

#### Findings

🔴 **`.expect()` in non-test, non-`main` code** — `src/lib.rs:108` — `field.ident.clone().expect("named fields always have idents")`. Panicking produces an ICE-style diagnostic instead of a clean `compile_error!`. If the assumption breaks (future `syn` version, edge case of grammar), the user sees a compiler panic. Action: replace with `field.ident.as_ref().ok_or_else(|| syn::Error::new_spanned(field, "expected named field"))?.clone()`.

🔴 **No trybuild (or equivalent) integration tests** — `ferrotorch-nn-derive/` has no `tests/` directory, no `trybuild` dependency, and no compile-fail fixtures. For a proc-macro this is a serious gap. There is no way to verify that error-path messages don't regress. Action: add a `tests/` crate with `trybuild` covering at least the three documented error paths and one happy-path expansion check.

🟡 **No `[lints]` / `#![warn]` in the crate** — Add `[lints] workspace = true` after defining a `[workspace.lints]` table, or add explicit `#![…]` attrs.

🟡 **The crate-level doc example is `ignore`d without explanation** — `src/lib.rs:23` — The single example cannot be compiled as a doctest (depends on `ferrotorch_nn` types not in scope). Action: add a `# why this is ignored` note.

🟡 **Internal types (`FieldKind`, `ClassifiedField`) lack `Debug` impls** — During macro-development the inability to `eprintln!("{:?}")` slows iteration. Action: `#[derive(Debug)]`.

🟡 **`find_float_param` silently accepts a wrong ident from the where-clause path** — `src/lib.rs:326` — When the `where`-clause path has multiple segments (e.g., `<Self as Foo>::T`) the code takes `path.segments.first()`, which would be `Self`, not the float type parameter.

🟢 **`named_parameters` uses `.to_string()` on a string literal** — `src/lib.rs:205`: `#name_str.to_string()` where `name_str` is already a `&str` literal.

🟢 **`Vec::new()` without capacity in generated bodies** — Cannot be pre-sized statically. Document as a known limitation.

#### Honest report
- **Did**: read `Cargo.toml` (crate + workspace), read `src/lib.rs` in full (348 lines), ran targeted greps.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo doc`, miri, runtime execution.
- **Confidence**: high — the crate is a single 348-line file with no conditional compilation. Reduced confidence on the `find_float_param` where-clause logic.
- **Skill loading**: loaded via Skill tool (rust-gpu-discipline not applicable).

---

### Crate: ferrotorch-optim

#### Vital stats
- **Kind**: lib
- **.rs files**: 40 (24 in `src/`, 16 in `src/scheduler/`)
- **Approx LoC**: ~19,750
- **`unsafe {` blocks**: 61
- **`// SAFETY:` comments**: 22
- **Tests**: yes (38 `#[cfg(test)]` modules)
- **Crate-level lint config**: absent
- **Touches GPU**: yes — `adam.rs` and `grad_scaler.rs` import `ferrotorch_core::gpu_dispatch` and call `gpu_backend()`, `fused_adam_f32`, `gpu_to_cpu`
- **Edition / MSRV**: Edition 2024, MSRV 1.85

#### Findings

🔴 **39 `unsafe {}` blocks missing `// SAFETY:` comments** — `adamw.rs:250,387`, `adadelta.rs:199,292`, `asgd.rs:201,307`, `adamax.rs:191,285`, `radam.rs:223,358`, `nadam.rs:228,352`, `sgd.rs:243,348`, `swa.rs:316,326`, `sparse_adam.rs:140`, `muon.rs:299`, `rprop.rs:185`, `adafactor.rs:291`, `adagrad.rs:208,301,306`, `ema.rs:272,279,299`, `rmsprop.rs:256,390`, and more — 61 unsafe blocks but only 22 `// SAFETY:` comments. The majority of `param.tensor().update_data(...)` and `param_t.update_storage(...)` calls carry no written invariant justification.

🔴 **Silent CPU fallback when GPU alloc fails** — `adam.rs:380` — `// Fall through to CPU path if GPU alloc failed.` is a silent swallow: if `alloc_zeros` returns `None` the GPU state is left uninitialized (`Vec::new()`) and the code falls through to the CPU path with zero explanation to the caller.

🔴 **`unsafe` raw-pointer aliasing cast in `adam.rs`** — `adam.rs:356–358` — `Arc::as_ptr(...) as *mut TensorStorage<T>` followed by `&mut *storage_ptr` manufactures a mutable reference from a shared `Arc`; this violates Rust's aliasing rules if any other `Arc` clone exists at that moment. The existing `// SAFETY:` comment claims "exclusive access" but the invariant that no other `Arc` clone holds a live reference is not enforced at compile time.

🔴 **`slice::from_raw_parts` without `// SAFETY:`** — `adam.rs:523–532` — Two back-to-back `unsafe { std::slice::from_raw_parts(...) }` in the GPU state serialization path have no `// SAFETY:` comment.

🔴 **`.expect()` panics inside `or_insert_with` closures in production step loops** — `adam.rs:185–198`, `adamw.rs:190–196` — `zeros::<T>(shape).expect("zeros allocation").to(device).expect("zeros to device")` are called inside `HashMap::entry(...).or_insert_with(...)` on every first-step per-parameter initialization; an OOM or device error will panic the optimizer step rather than returning `Err`.

🟡 **123 `T::from(...).unwrap()` / `.to_f64().unwrap()` calls in production numeric paths** — spread across `adam.rs`, `adamw.rs`, `natural_gradient.rs`, `rprop.rs`, `nadam.rs`, `radam.rs`, etc.

🟡 **No crate-level lint config**.

🟡 **All `*Config` struct fields are `pub`** — `AdamWConfig`, `AdamConfig`, `AdadeltaConfig`, `KfacConfig`, `MuonConfig`, `AsgdConfig`, `AdamaxConfig`, and ~12 others. Action: `#[non_exhaustive]` or private fields with builder.

🟡 **`ParamGroup.params` is `pub Vec<Parameter<T>>`** — `optimizer.rs:13` — Callers can push/pop elements and bypass any invariants the optimizer relies on.

🟡 **`thiserror` is a workspace dep but no error types use it** — Re-uses `ferrotorch-core`'s error type, which is fine, but if this crate ever adds its own error variants they must use `#[derive(thiserror::Error)]`.

🟡 **`#[allow(dead_code)]` on struct fields without explanation** — `scheduler/cosine_warm_restarts.rs:38`, `scheduler/one_cycle_lr.rs:69`.

🟢 **`#[allow(clippy::too_many_arguments)]` on `lbfgs.rs:422`** — narrow, item-scoped, acceptable.

🟢 **No `prelude` module** — `lib.rs` re-exports ~30 items at crate root.

#### Honest report
- **Did**: read `Cargo.toml`, `lib.rs`, `optimizer.rs`, `adamw.rs` (full), `adam.rs` (selected), `natural_gradient.rs`, `lbfgs.rs`, `swa.rs`; ran greps.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, `miri`, runtime execution.
- **Confidence**: high for structural findings; medium for the aliasing-safety claim in `adam.rs:356`.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-train

#### Vital stats
- **Kind**: lib
- **.rs files**: 9
- **Approx LoC**: 3,662
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (8 modules)
- **Crate-level lint config**: absent
- **Touches GPU**: no direct GPU deps
- **Edition / MSRV**: edition = 2024, rust-version = 1.85

#### Findings

🔴 **`.unwrap()` on `num_traits::cast` in production gradient-clipping hot path** — `grad_utils.rs:79,91,103,163,164` — `val.to_f64().unwrap()` and `T::from(clip_coef).unwrap()` are called inside `clip_grad_norm_` and `clip_grad_value_`.

🔴 **`.unwrap()` on `to_f64()` inside the main training loop** — `learner.rs:253,397` — Both `fit()` and `evaluate()` call `loss.item()?.to_f64().unwrap()` in the per-batch hot path.

🔴 **`.unwrap()` on `to_f64()` in public API methods** — `callback.rs:300,313` — `EmaCallback::init_from_params` and `update_from_params` are `pub` methods that call `.to_f64().unwrap()` on every parameter value.

🔴 **`assert!` used as public input validation in library functions** — `grad_utils.rs:55,56,161` — `clip_grad_norm_` and `clip_grad_value_` use `assert!` to validate `max_norm >= 0.0`, `norm_type > 0.0`, and `clip_value >= 0.0`. Should return `Err(...)` not panic.

🔴 **Errors from `TensorBoardCallback::on_epoch_end`/`on_batch_end` are silently discarded** — `tensorboard.rs:430-443,451` — The `Callback` trait methods return `()`, so `add_scalar`, `flush`, etc. errors are swallowed with `let _ = ...`. A full disk or closed file handle is silent.

🟡 **`println!` in library code (`ProgressLogger`)** — `callback.rs:173,177,182,187,189,192` — Library crate; callers have no way to redirect or silence this output.

🟡 **No crate-level lint configuration**.

🟡 **`pub` fields on `EpochResult`, `EvalResult`, and `TrainingHistory`** — `history.rs:18-28,52-54,77` — Action: `#[non_exhaustive]`.

🟡 **`#[allow(dead_code)]` on `ProtobufWriter` without justification comment** — `tensorboard.rs:98`.

🟡 **Missing `# Errors` doc sections on public `Result`-returning fns** — `tensorboard.rs:307,335,368,414`; `learner.rs:186,370`.

🟡 **Doctest examples use `ignore` uniformly** — `lib.rs:22`, `grad_utils.rs:44`, etc.

🟢 **`TensorBoardCallback::writer` field is `pub(crate)`-accessible via test** — `tensorboard.rs:622`.

🟢 **`Learner::with_checkpointing` takes `PathBuf` by value** — `learner.rs:157` — Accepting `impl Into<PathBuf>` would be slightly more ergonomic.

#### Honest report
- **Did**: listed all 9 .rs files; read `Cargo.toml`, `lib.rs`, `grad_utils.rs`, `learner.rs`, `callback.rs`, `tensorboard.rs`, `amp.rs`, `history.rs`; grepped patterns; verified test module boundaries to triage production vs test code.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, miri.
- **Confidence**: high for what was flagged; medium on completeness — there could be additional API/doc gaps in `metric.rs` and `checkpoint.rs`.
- **Skill loading**: loaded via Skill tool (rust-gpu-discipline largely not applicable).

---

### Crate: ferrotorch-data

#### Vital stats
- **Kind**: lib
- **.rs files**: 7 under `src/`
- **Approx LoC**: 4,707
- **`unsafe {` blocks**: 0 (two `unsafe impl` blocks, no `unsafe { … }` expression blocks)
- **`// SAFETY:` comments**: 1
- **Tests**: yes (6 modules)
- **Crate-level lint config**: absent
- **Touches GPU**: minimal — `dataloader.rs` exposes a `ToDevice` trait and `Device` enum from `ferrotorch-core` for CPU→GPU transfer plumbing
- **Edition / MSRV**: workspace-inherited

#### Findings

🔴 **`Mutex::lock().unwrap()` and `Condvar::wait().unwrap()` in multi-worker runtime** — `dataloader.rs:894,910,976,1011,1016` — `std::sync::Mutex::lock()` returns `Err` on poisoned mutex (any panicking worker poisons the lock).

🔴 **`Normalize::new` calls `.unwrap()` inside a constructor on a cast that can fail** — `transforms.rs:100,104` — `<T as NumCast>::from(v)` returns `None` for out-of-range values.

🔴 **`shape.last().unwrap()` in `RandomHorizontalFlip::apply`** — `transforms.rs:251`.

🔴 **`iter_collated` panics via `.expect()` on missing collate_fn** — `dataloader.rs:357` — A public API method that panics on a misuse condition.

🟡 **No crate-level lint configuration**.

🟡 **`DataLoader<D>`, `BatchIter`, `DataLoaderIter`, `PrefetchIter`, `MultiWorkerIter`, `CollatedIter` lack `Debug`** — `dataloader.rs:146,454,498,576,786,1076`.

🟡 **All five `Transform` implementors (`Compose`, `Normalize`, `ToTensor`, `RandomHorizontalFlip`, `RandomCrop`) lack `Debug`** — `transforms.rs:40,78,160,180,281`.

🟡 **`unsafe impl Send for ToTensor` and `unsafe impl Sync for ToTensor` are unnecessary** — `transforms.rs:169–170` — `ToTensor` is a unit struct with zero fields. Action: remove both `unsafe impl` lines; the bounds are satisfied automatically.

🟡 **`WorkerInfo` has two `pub` fields** — `dataset.rs:51,53`.

🟡 **`DataLoader::new` and `RandomHorizontalFlip::new` use `assert!` / `assert_eq!` for input validation** — `dataloader.rs:172`, `transforms.rs:190–193`.

🟢 **`println!` only appears in a doctest comment, not live library code** — `dataloader.rs:142`.

🟢 **`collate.rs:51,56` — `.unwrap()` in helper function `t32`/`t64`** — Inside `#[cfg(test)] mod tests`.

🟢 **No `todo!()`, `unimplemented!()`, or `dbg!(` in non-test paths**.

🟢 **No `&String`/`&Vec<T>`/`&PathBuf` parameters**.

🟢 **GPU discipline is largely clean** — `interop.rs` explicitly rejects CUDA tensors with `FerrotorchError::NotImplementedOnCuda` rather than silently downloading.

#### Honest report
- **Did**: read all 7 `.rs` files; grepped extensively; read full source of `transforms.rs`, `collate.rs`, `interop.rs` critical sections, `dataloader.rs` struct layout + multi-worker runtime.
- **Did NOT do**: cargo build / clippy / test / miri.
- **Confidence**: high.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-vision

#### Vital stats
- **Kind**: lib
- **.rs files**: 39
- **Approx LoC**: ~15,382
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (34 modules)
- **Crate-level lint config**: absent
- **Touches GPU**: no
- **Edition / MSRV**: Edition 2024, MSRV 1.85

#### Findings

🔴 **`.unwrap()` on infallible `NumCast` conversions throughout production code** — `src/ops.rs:58,60,101,121…` (40+ sites), `src/io.rs:102,110,228`, `src/transforms/random_rotation.rs:66-69`, `src/transforms/trivial_augment_wide.rs:115-234`, `src/transforms/color_jitter.rs:102-167`, etc. — `<T as NumCast>::from(255.0).unwrap()` in library code on every non-test code path. >60 sites across the crate.

🔴 **`assert!` panics in public constructors (not `Result`)** — `src/transforms/random_rotation.rs:25`, `src/transforms/random_vertical_flip.rs:21`, `src/transforms/random_resized_crop.rs:33,39`, `src/transforms/random_gaussian_blur.rs:28,32`, `src/transforms/random_apply.rs:24,61`, `src/transforms/gaussian_noise.rs:31`, `src/transforms/trivial_augment_wide.rs:52`, `src/transforms/random_horizontal_flip.rs:18`, `src/transforms/elastic_transform.rs:38,42`, `src/transforms/color_jitter.rs:39,43,47,51`, `src/models/convnext.rs:332-333`, `src/models/unet.rs:117`.

🔴 **`.expect(…)` on `RwLock` in public library functions** — `src/models/registry.rs:271,283,291` — `list_models()`, `get_model()`, and `register_model()` are public library functions that call `.expect("model registry lock poisoned")`.

🔴 **`.expect(…)` in production `generate_synthetic` helper** — `src/datasets/cifar.rs:226`, `src/datasets/mnist.rs:101` — `generate_synthetic` is called from `pub fn synthetic(…)` which is a public constructor.

🟡 **No crate-level lint configuration**.

🟡 **Doctest uses `.unwrap()` — bad teaching pattern** — `src/io.rs:81`.

🟡 **`RawImage` lacks `#[derive(Debug)]`** — `src/io.rs:17`.

🟡 **`Cifar10<T>` and `Cifar100<T>` lack `#[derive(Debug)]`** — `src/datasets/cifar.rs:58,131`.

🟡 **Public struct fields on sample/dataset types** — `src/io.rs:19-25` (`RawImage`: `data`, `width`, `height`, `channels`), `src/datasets/cifar.rs:33,35` (`CifarSample`), `src/datasets/mnist.rs:31,33` (`MnistSample`), `src/datasets/folder.rs:31,33,161,163`.

🟡 **`#[allow(clippy::too_many_arguments)]` without justification comment** — `src/ops.rs:739`, `src/models/inception.rs:90`, `src/models/vit.rs:343`.

🟢 **`T::from(…).unwrap()` pattern could use a named helper** — Extract a `fn cast<T: Float>(v: f64) -> FerrotorchResult<T>`.

🟢 **No `prelude` module**.

#### Honest report
- **Did**: listed all 39 `.rs` source files, read `Cargo.toml` (crate and workspace), `src/lib.rs`, `src/io.rs`, `src/transforms/random_rotation.rs`, `src/models/registry.rs`, `src/datasets/cifar.rs` (partial). Ran greps using a Python script for accuracy.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, miri.
- **Confidence**: high. Partial confidence on whether `T: Float` in practice guarantees `NumCast::from` is infallible.
- **Skill loading**: loaded via Skill tool (rust-gpu-discipline rules not applicable).

---

### Crate: ferrotorch-jit

#### Vital stats
- **Kind**: lib
- **.rs files**: 21
- **Approx LoC**: 16,942
- **`unsafe {` blocks**: 3 (codegen_jit.rs:163, :348, :360)
- **`// SAFETY:` comments**: 1
- **Tests**: yes (19 modules)
- **Crate-level lint config**: absent
- **Touches GPU**: yes — emits CUDA C and PTX source strings (no cudarc/wgpu runtime dep; codegen only)
- **Edition / MSRV**: both via `workspace = true`

#### Findings

🔴 **GPU `InductorTarget::GpuCuda` / `GpuPtx` silently falls back to CPU interpreter** — `codegen.rs:871–886` — `InductorBackend::compile()` only attempts real JIT for `InductorTarget::CpuC`; both GPU targets generate source strings (discarded via `let _sources`), then fall through to `InterpreterBackend.compile(graph)`, which runs on CPU.

🔴 **`IrValue` doc comment claims dtype metadata that doesn't exist** — `graph.rs:106` — The struct is documented as "carrying shape/dtype metadata" but has no `dtype` field. The GPU codegen hard-codes `float` throughout; a f64 graph would silently produce wrong-typed C/PTX.

🔴 **`T::from(v).unwrap()` in production interpreter** — `interpreter.rs:142` — Converting f64 constants to `T` via `num_traits::cast::FromPrimitive::from`.

🔴 **`panic!` in production tracing and IR-building code** — `trace.rs:253`, `graph_break.rs:401`, `graph_break.rs:552` — All three are inside the non-test `build_ir_graph` / tracing codepath. They annotate themselves "BUG:" but are reachable via a malformed trace.

🔴 **`unsafe impl Send for JitCompiledKernel` / `unsafe impl Sync` lack `// SAFETY:` marker** — `codegen_jit.rs:98–99` — There is a prose comment two lines above ("Safety:" with capital S), but the convention required by rust-quality and auditing tools is `// SAFETY:` on the line immediately preceding each `unsafe` keyword.

🟡 **No crate-level lint config at all**.

🟡 **Dozens of public struct fields — no encapsulation** — `graph.rs`, `aot_autograd.rs`, `dag_fusion.rs`, `module.rs`, `optimize.rs`, `symbolic.rs`, `codegen.rs` — Core IR types (`IrGraph`, `IrNode`, `IrValue`, `AotGraphPair`, `FusionGroup`, `CompileConfig`) expose all fields as `pub`. Any downstream crate can mutate `IrGraph::nodes` directly, bypassing the invariants that `next_value_id` / `next_node_id` maintain.

🟡 **`eprintln!` in library test code used to skip tests** — `codegen_jit.rs:479,498,518,541`, `codegen.rs:1614`.

🟡 **GPU codegen tests only assert string contents — no compilation or execution** — `codegen_gpu.rs` test module — All 20+ GPU tests assert that the emitted CUDA/PTX strings contain expected substrings. None of them compile or run the kernels. Combined with the `InductorBackend::compile()` falling back to CPU, there is zero end-to-end GPU coverage.

🟡 **`#[allow(dead_code)]` on `pub(crate)` helpers without justification comment** — `fusion.rs:874,882`.

🟢 **Doctest in `interpreter.rs` uses `.unwrap()`** — `interpreter.rs:51–52` — Marked `ignore` so it won't fail CI, but still shapes user expectations.

🟢 **`DefaultHasher` used in compile cache** — `codegen_jit.rs:183` — `DefaultHasher` is explicitly not stable across Rust versions or processes.

#### Honest report
- **Did**: read all 21 source files, read `Cargo.toml`, grepped for all forbidden patterns, verified test module line boundaries.
- **Did NOT do**: cargo build, cargo clippy, cargo test, miri, runtime GPU verification.
- **Confidence**: high for structural findings; medium for the `unsafe impl` SAFETY-comment form.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-jit-script

#### Vital stats
- **Kind**: proc-macro lib (`[lib] proc-macro = true`)
- **.rs files**: 2 (src/lib.rs, tests/script_macro.rs)
- **Approx LoC**: 169 (lib) + 66 (tests) = 235
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (3 integration tests)
- **Crate-level lint config**: absent
- **Touches GPU**: no — deps are `syn`, `quote`, `proc-macro2` only
- **Edition / MSRV**: edition 2024, rust-version 1.85

#### Findings

🔴 **Silent scalar-type erasure for non-f32/f64 annotated functions** — `src/lib.rs:103` — `extract_tensor_param` returns `None` for any return type that doesn't syntactically match `Tensor<T>`, `FerrotorchResult<Tensor<T>>`, or `Result<Tensor<T>, _>`. The fallback is `unwrap_or_else(|| quote! { f32 })`, so a user writing `-> Tensor<f16>` or `-> Tensor<bf16>` silently gets a `TracedModule<f32>` wrapping their body.

🔴 **Unbounded recursion in `extract_tensor_param`** — `src/lib.rs:144–169` — The function calls itself recursively when it sees `FerrotorchResult` or `Result` wrapping an inner type. A pathological type like `Result<Result<Result<..., _>, _>, _>` will recurse until a stack overflow during macro expansion.

🟡 **No crate-level lint configuration**.

🟡 **Duplicate argument-walking loops** — `src/lib.rs:67–74` and `src/lib.rs:85–91`.

🟡 **`body_fn_inputs` clone is unused except to silence a borrow** — `src/lib.rs:75`.

🟡 **Doctest uses `ignore` without explaining why it won't compile as-is** — `src/lib.rs:40`.

🟡 **Private helper `extract_tensor_param` lacks any doc comment** — `src/lib.rs:143`.

🟢 **`inputs.clone()` (line 75) clones `Punctuated<FnArg, Comma>`** — Compile-time clone, not runtime.

🟢 **No `proptest` or `quickcheck` coverage for `extract_tensor_param`** — A property test would catch the unbounded-recursion case automatically.

#### Honest report
- **Did**: read `Cargo.toml`, read `src/lib.rs` (all 169 lines), read `tests/script_macro.rs` (all 66 lines), ran greps.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, miri.
- **Confidence**: high — small crate read in full.
- **Skill loading**: loaded via Skill tool (rust-gpu-discipline does not apply).

---

### Crate: ferrotorch-serialize

#### Vital stats
- **Kind**: lib
- **.rs files**: 8
- **Approx LoC**: 8,992
- **`unsafe {` blocks**: 16
- **`// SAFETY:` comments**: 4
- **Tests**: yes (7 modules, inline `#[cfg(test)]` only)
- **Crate-level lint config**: absent
- **Touches GPU**: no
- **Edition / MSRV**: edition 2024, rust-version 1.85

#### Findings

🔴 **12 `unsafe` blocks without `// SAFETY:` comments** — `checkpoint.rs:377`, `state_dict.rs:129`, `pytorch_export.rs:274`, `pytorch_import.rs:1029, 1050, 1079, 1082, 1107, 1110, 1131`, `safetensors_io.rs:91, 254`. The byte-slice cast blocks in checkpoint/state_dict/pytorch_export are copies of the same pattern; the `ptr::read` blocks in pytorch_import all perform `T: Copy` type punning for f32↔f64 "casts" where `T` is generic.

🔴 **`unwrap()` on untrusted-input closure inside `load_state_dict` / `load_checkpoint`** — `state_dict.rs:257-258, 261-262`, `checkpoint.rs:456-457, 460-461` — Inside a `.chunks_exact(elem_size).map(|chunk| { ... T::from(val).unwrap() })` closure that runs over bytes from a file. If a file header declares `elem_size == 4` but `T` is some exotic `Float` impl where `from(f32)` returns `None`, the parser panics instead of returning `Err`.

🔴 **Attacker-controlled `Vec::with_capacity` from raw u64 in GGUF parser** — `gguf.rs:312-313, 393-394, 397, 413, 417` — `metadata_kv_count` and `tensor_count` are read directly from the file as `u64 as usize` and then passed to `Vec::with_capacity` / `HashMap::with_capacity` without any upper bound check. A 6-byte GGUF stub with `metadata_kv_count = 0xFFFF_FFFF_FFFF_FFFF` will cause an immediate OOM. **Pre-auth DoS on any code path that calls `parse_gguf_bytes` with untrusted input.**

🟡 **No crate-level lint config**.

🟡 **`#[allow(dead_code)]` on live-ish code without justification** — `onnx_export.rs:97, 220, 226, 381` — Constants `ATTR_INTS` and `ATTR_TYPE_INTS` reserved ONNX wire constants. `ProtobufWriter` has unused methods.

🟡 **`pub` fields on data-transfer structs without API justification** — `TrainingCheckpoint` (`checkpoint.rs:38-42`), `GgufTensorInfo` (`gguf.rs:165-168`), `GgufFile` (`gguf.rs:175-179`), `SafeTensorsIndex` (`safetensors_io.rs:300-301, 308`), `ShardProgress` (`safetensors_io.rs:466-476`), `OnnxExportConfig` (`onnx_export.rs:43, 50`).

🟡 **`.unwrap()` inside helper functions in test-only context, but check production deserialization** — `state_dict.rs:373, 378` — Confirmed test-gated.

🟡 **`&String` used in internal iteration** — `safetensors_io.rs:113, 328, 374, 503, 595, 688`, `pytorch_export.rs:221`, `state_dict.rs:65`, `checkpoint.rs:179, 185, 338`. Not function-parameter hygiene issues, but `Vec<&str>` would be more idiomatic.

🟡 **No integration tests; all test modules are `#[cfg(test)]` inline** — proptest / quickcheck coverage for the parse/serialize round-trips would be valuable.

🟢 **`onnx_export.rs:1269`+ — `.unwrap()` calls in tests are fine** — Confirmed all inside `#[cfg(test)] mod tests` blocks.

🟢 **Docstring coverage is reasonable** — Public functions in `gguf.rs`, `safetensors_io.rs`, `checkpoint.rs`, and `state_dict.rs` all have `///` comments with `# Errors` sections.

#### Honest report
- **Did**: read all 8 `.rs` source files (focused reads on `gguf.rs`, `checkpoint.rs`, `state_dict.rs`, `pytorch_import.rs`, `safetensors_io.rs`); grepped for all forbidden patterns including overflow-prone arithmetic in untrusted-input paths.
- **Did NOT do**: cargo build, cargo clippy, cargo test, miri.
- **Confidence**: high for structural findings; medium for the `T::from(val).unwrap()` severity classification.
- **Skill loading**: loaded via Skill tool (gpu-discipline not materially applicable).

---

### Crate: ferrotorch-gpu

#### Vital stats
- **Kind**: lib
- **.rs files**: 21
- **Approx LoC**: 45,560
- **`unsafe {` blocks**: 268
- **`// SAFETY:` comments**: 42
- **Tests**: yes (16 modules, ~269 `#[test]` fns)
- **Crate-level lint config**: absent
- **Touches GPU**: yes (cudarc: driver, cublas, cusolver, cufft; PTX kernel launches)
- **Edition / MSRV**: edition = 2024, rust-version = 1.85

#### Findings

🔴 **226 unsafe blocks have no `// SAFETY:` comment** — `src/bf16.rs` (15 blocks), `src/conv.rs` (2), `src/rng.rs` (2), `src/backend_impl.rs` (11), others — Every kernel launch in `bf16.rs` is `unsafe { stream.launch_builder(&f)… }` with zero safety justification. The `transmute` in `stream.rs:232` and the `Vec::from_raw_parts` in `backend_impl.rs:256–276` do have comments. Ratio is 42 SAFETY annotations for 268 blocks (16%).

🔴 **f64 ops always take a silent CPU round-trip** — `src/kernels.rs:15221, 15242, 15263, 15284, 15295, 15306, …` — Every exported f64 elementwise function (`gpu_add_f64`, `gpu_sub_f64`, `gpu_mul_f64`, `gpu_relu_f64`, `gpu_gelu_f64`, etc.) attempts a PTX launch and falls back to `cpu_fallback_binary/unary_f64`, which does a full GPU→host download, CPU compute, host→GPU upload. There is no visible documentation in the function signatures or `# CPU fallback` rustdoc section warning callers this is a hot-path hazard.

🔴 **PTX compile errors are swallowed in `bf16.rs` with `eprintln!`** — `src/bf16.rs:370, 427, 465, 507, 616, 749, 961, 1204, 1410, 1604, 1653, 1777, 1895, 1995, 2177` — In the `launch_binary` / `launch_unary` helpers, a PTX compile failure prints to stderr and then returns `Err(GpuError::PtxCompileFailed)`. The error propagation is correct, but `eprintln!` in a library function is wrong (caller can't intercept or redirect it). Same pattern exists in `module_cache.rs:174`.

🟡 **No crate-level lint configuration**.

🟡 **Hand-rolled `Display + Error` impl when `thiserror` is in workspace** — `src/error.rs:1–191` — `GpuError` manually implements `fmt::Display` (120-line match) and `std::error::Error::source`. `thiserror = "2.0"` is declared in the workspace `Cargo.toml` but not added to this crate's `[dependencies]`.

🟡 **`PhiloxState` exposes all three fields as `pub`** — `src/rng.rs:62–70` — `counter`, `seed`, and `offset`. `offset` encodes an internal cursor (0..4) that the `PhiloxGenerator` manages; exposing it lets callers construct invalid states.

🟡 **`MemoryHook.callback` is a `pub Box<dyn Fn>` on a public struct** — `src/memory_guard.rs:96–111` — Public `Box<dyn Fn>` fields can't be cloned, equality-compared, or serialised.

🟡 **`MemoryStats` has public fields on a library struct** — `src/memory_guard.rs:218–232` — Action: `#[non_exhaustive]`.

🟡 **`rng.rs` mutex unlocked with `.unwrap()`** — `src/rng.rs:1175` — `CUDA_RNG_MANAGER.lock().unwrap()` is in production code.

🟡 **`lib.rs` doctest uses `.unwrap()`** — `src/lib.rs:19–20` — The quick-start doctest calls `.unwrap()` on both `GpuDevice::new(0)` and `cpu_to_gpu`.

🟢 **`#[allow(clippy::too_many_arguments)]` in production code without justification comment** — `src/kernels.rs:10896`.

🟢 **`module_cache.rs:174` timing `eprintln!` is inside a `#[test]` block** — Confirmed fine.

#### Honest report
- **Did**: listed all 21 source files; read `Cargo.toml`, `lib.rs`, `error.rs`, `stream.rs`, `rng.rs` (excerpts), `kernels.rs` (header + fallback sections + f64 dispatch), `bf16.rs` (launch helpers), `backend_impl.rs` (unsafe transmute section), `blas.rs` (eprintln context), `memory_guard.rs`, `tensor_bridge.rs`. Ran greps for all patterns.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `miri`.
- **Confidence**: high for structural/pattern findings; medium for the PhiloxState/MemoryHook API judgements (could be intentional design trade-offs).
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-cubecl

#### Vital stats
- **Kind**: lib
- **.rs files**: 6 (`lib.rs`, `runtime.rs`, `kernels.rs`, `ops.rs`, `quant.rs`, `grammar.rs`)
- **Approx LoC**: ~3,729
- **`unsafe {` blocks**: 28 (kernels.rs: 8, grammar.rs: 2, quant.rs: 18)
- **`// SAFETY:` comments**: 0
- **Tests**: yes (5 modules)
- **Crate-level lint config**: absent
- **Touches GPU**: yes — `cubecl`, `cubecl-cuda`, `cubecl-wgpu`, `cubecl-hip`
- **Cubecl API actually used**: `cubecl::prelude::*` (`#[cube]`, `Array`, `Float`, `ABSOLUTE_POS`, `CubeCount`, `CubeDim`, `ArrayArg`, `ComputeClient`, `Runtime`); `cubecl::server::Handle`; `launch_unchecked`; `CudaRuntime::client`, `WgpuRuntime::client`, `HipRuntime::client`
- **Edition / MSRV**: both inherited from workspace

#### Findings

🔴 **All 28 `unsafe {}` blocks have no `// SAFETY:` comment** — `kernels.rs:404-406,427-429,444,455,498-499,514,560-572`, `grammar.rs:146-162,168`, `quant.rs:530-548,567-586,602-618,635-658,678-700,720-726,1280-1294` — Every block calls `ArrayArg::from_raw_parts` or `std::slice::from_raw_parts`; the invariants (handle capacity matches `n`, pointer alignment, slice length accounting for `size_of`) are non-trivial and nowhere documented.

🔴 **Synchronous host readback on every elementwise op** — `kernels.rs:408,432,502,574` — `run_unary`, `run_binary`, `run_unary_with_n`, and `run_matmul` all call `client.read_one(out_handle)` (an implicit device sync + DMA copy back to host) before returning, meaning every `portable_add`, `portable_mul`, … round-trips through host RAM. Anti-pattern #7. The `quant.rs` and `grammar.rs` launchers correctly return the raw `Handle` to the caller; the elementwise ops do not.

🟡 **No crate-level lint configuration**.

🟡 **`elementwise_launch_dims` is copy-pasted verbatim three times** — `kernels.rs:375`, `quant.rs:504`, `grammar.rs:107`.

🟡 **`pub` fields on `DfaMaskInputs` with no per-field rustdoc** — `grammar.rs:119-126` — Eight public fields carry no `///` comments. The off-by-one on `vocab_offsets` (must be `vocab_size + 1` entries) is an easy misuse.

🟡 **`GgufBlockKind` variants use non-`UpperCamelCase` names** — `quant.rs:50-55` — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` violate RFC 430 casing.

🟡 **`ops.rs` doc-comment describes eager CPU round-trip as the API contract** — `ops.rs:6-8`.

🟢 **`println!` in lib-crate example snippet** — `lib.rs:27` — `println!("Using device: {:?}", rt.device());` lives inside a `no_run` doctest, so it does not execute at runtime.

🟢 **`expect("cubecl read_one failed")` in production run helpers** — `kernels.rs:408,432,502,574`.

#### Honest report
- **Did**: read all 6 `.rs` files (full text); grepped for all forbidden patterns; read `Cargo.toml` for both the crate and the workspace root.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, `miri`.
- **Confidence**: high for static patterns; medium for semantic correctness of the `#[cube]` kernel recurrences.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-xpu

#### Vital stats
- **Kind**: lib
- **.rs files**: 1 (`src/lib.rs`)
- **Approx LoC**: 489
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (2 modules — gated on `wgpu` feature, 8 + 1 tests)
- **Crate-level lint config**: absent
- **Touches GPU/XPU**: yes — delegates to `ferrotorch-cubecl`
- **Backend deps actually present**: `ferrotorch-cubecl`, `ferrotorch-core`; no direct `level-zero`, `hipBLAS`, or `oneapi` binding
- **Edition / MSRV**: edition 2024, MSRV 1.85

#### Findings

🔴 **Synchronous host-readback on every op (anti-pattern #7)** — `src/lib.rs:147,161` — Each `xpu_binary!` / `xpu_unary!` expansion calls `result_on_cubecl.data_vec()?` (device→host copy) immediately after the kernel completes, then `make_xpu_tensor` re-tags the host `Vec<f32>` as `Device::Xpu`. Chaining `xpu_add` → `xpu_relu` → `xpu_exp` incurs three complete round-trips through host RAM.

🔴 **`check_xpu_tensor` is monomorphic `f32`-only with no dtype dispatch** — `src/lib.rs:108-118` — All public ops accept only `Tensor<f32>`. The crate declares `num-traits` as a dependency but never uses it. f64, bf16, and half are silently unsupported with no API indication.

🔴 **`xpu_tensor_to_xpu_keeps_data_and_shape` test calls `.unwrap()` outside the `xpu()` guard** — `src/lib.rs:361-367`.

🟡 **Unused dependency `num-traits`** — `Cargo.toml:24`.

🟡 **No crate-level lint config**.

🟡 **`eprintln!` in library test helper** — `src/lib.rs:337-339`.

🟡 **No `# Errors` / `# Panics` sections in public op docs** — `src/lib.rs:172-257`.

🟡 **`xpu_matmul` only handles exact 2-D tensors (shape-bypass risk)** — `src/lib.rs:194-197`.

🟢 **`xpu_binary_stub!` / `xpu_unary_stub!` functions lack any rustdoc** — `src/lib.rs:265-316`.

🟢 **`XpuDevice` has no `Display` impl** — `src/lib.rs:49-101`.

#### Honest report
- **Did**: read `Cargo.toml`, read `src/lib.rs` in full (489 lines), read `ferrotorch-cubecl/src/ops.rs` (relevant sections), read `ferrotorch-cubecl/src/kernels.rs` (relevant sections), read `ferrotorch-core/src/tensor.rs:664-679`; grepped patterns.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `miri`.
- **Confidence**: high for structural and GPU-discipline findings; medium for dtype/shape-bypass.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-distributed

#### Vital stats
- **Kind**: lib
- **.rs files**: 22
- **Approx LoC**: ~10,877
- **`unsafe {` blocks**: 35
- **`// SAFETY:` comments**: 10
- **Tests**: yes (17 `#[cfg(test)]` modules)
- **Crate-level lint config**: absent
- **Touches GPU**: yes (`ferrotorch-gpu` optional dep; `gpu_collective.rs`, `nccl_backend.rs`, `nccl_collective.rs`, `nccl_sys.rs`)
- **Async/tokio**: no
- **Edition / MSRV**: edition 2024, rust-version 1.85

#### Findings

🔴 **25 `unsafe {}` blocks missing `// SAFETY:` comments** — `nccl_backend.rs:300,323,342,344,354,361,363,369,371,378,383,384,394,396,403,408,409`; `nccl_collective.rs:98,123,151,181`; `checkpoint.rs:129,242,257`; `pipeline.rs:257` — Every one dereferences raw pointers, calls `transmute`, or invokes dlsym-resolved function pointers without a written invariant proof. The `nccl_backend.rs` dlopen/dlsym chain is particularly dangerous: `transmute` of a raw symbol with no ABI or type verification comment.

🔴 **5 `.expect()` calls in production (non-test) code in `fsdp.rs`** — `fsdp.rs:229,454,529,699,703` — All five are `.expect("HybridShard: intra_node_group set…")` inside the `forward`/`sync_gradients` hot paths.

🔴 **`gpu_collective.rs` is a documented CPU-round-trip** — `gpu_collective.rs:1–13` — The module docstring explicitly describes: GPU → CPU → collective → CPU → GPU on every call. Anti-pattern #7. While the doc justifies it as a "portable alternative to NCCL," there's no `tracing::warn!` to surface the cost to callers.

🔴 **`collective.rs:105,378` — `.unwrap()` on `T::from(world_size)` in production**.

🔴 **`nccl_backend.rs` `Drop::drop` calls `unsafe` without `// SAFETY:`** — `nccl_backend.rs:323` — Inside `impl Drop for NcclBackend`, `comm_destroy(*comm)` is called without any comment establishing that `*comm` is still valid after lock acquisition.

🟡 **No crate-level lint config**.

🟡 **`pub` fields on library structs** — `checkpoint.rs:82,84,87,95,97,107,109`; `sync_batch_norm.rs:61-66`; `nccl_sys.rs:26`.

🟡 **`checkpoint.rs:149` uses `&String` parameter** — `let mut keys: Vec<&String> = tensors.keys().collect();`.

🟡 **`nccl_sys.rs:416` `eprintln!` in library test code**.

🟡 **`nccl_collective.rs` all four `unsafe` call sites lack `// SAFETY:`** — Functions `allreduce_raw`, `broadcast_raw`, `all_gather_raw`, `reduce_scatter_raw` are `pub unsafe fn`.

🟢 **`gloo_backend.rs`, `mpi_backend.rs`, `ucc_backend.rs` are skeleton modules** — Per `Cargo.toml` comment, intentional follow-ups (#459-gloo etc.).

🟢 **Quickstart doctest in `lib.rs` is `ignore`d** — `lib.rs:46`.

#### Honest report
- **Did**: listed all 22 `.rs` files; read `Cargo.toml`, `lib.rs`, `gpu_collective.rs` in full; grepped all patterns; read targeted line ranges in `fsdp.rs`, `collective.rs`, `checkpoint.rs`, `nccl_backend.rs`, `nccl_collective.rs`, `nccl_sys.rs`, `pipeline.rs`.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, `miri`.
- **Confidence**: high for structural/pattern findings; medium for `HybridShard` `.expect()` reachability.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-distributions

#### Vital stats
- **Kind**: lib
- **.rs files**: 32
- **Approx LoC**: 14,494
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (360 `#[test]` functions, all inline unit tests)
- **Crate-level lint config**: absent
- **Touches GPU**: yes — `is_cuda()` dispatch present in `normal.rs`, `exponential.rs`, `multivariate_normal.rs` and several others; no GPU deps
- **Edition / MSRV**: Edition 2024, MSRV 1.85

#### Findings

🔴 **`Distribution::stddev` default strips GPU device** — `src/lib.rs:202–211` — `self.variance()` may live on CUDA; the default `stddev` calls `data_vec()` (host readback), computes on CPU, and returns a CPU tensor with no `.to(device)` call.

🔴 **Pervasive silent CPU detour pattern in `sample`/`rsample`/`log_prob`/`entropy`** — `src/normal.rs:65–75`, `src/exponential.rs:47–66`, `src/multivariate_normal.rs:168–*`, and all remaining distributions — Every op calls `data_vec()` on potentially-CUDA tensors (host readback), computes entirely on CPU, builds a `TensorStorage::cpu(...)` result, then conditionally `.to(device)`. Anti-pattern #7. No `// CPU path:` justification comment is present on any of these paths.

🔴 **`Kumaraswamy::entropy` panics via `.unwrap()` in production lib code** — `src/kumaraswamy.rs:128,142` — `num_traits::ToPrimitive::to_f64(&b[i]).unwrap()` and `T::from(...).unwrap()` are called in the hot loop. Also duplicates `special_fns::digamma_scalar` (already present and correct) with a coarser approximation.

🔴 **`Kumaraswamy::entropy` uses a wrong/coarser digamma approximation** — `src/kumaraswamy.rs:127–141` — The asymptotic branch (`bf > 5.0`) uses `ln(b) + 1/(2b)`, omitting the standard Bernoulli-number correction terms. The crate already ships a high-quality Lanczos-based `digamma_scalar` in `special_fns.rs`. **Math-correctness defect.**

🟡 **No crate-level lint configuration**.

🟡 **`T::from(constant).unwrap()` pervasive in `special_fns.rs` and all distributions** — ~350 call sites in lib (non-test) code.

🟡 **All major distribution structs missing `Debug`** — e.g. `Normal`, `Beta`, `Bernoulli`, `Categorical`, `Independent`, `MixtureSameFamily`, `StudentT`, `Gamma`, `Laplace`, `Gumbel`, `HalfNormal`, `Cauchy`, `Uniform`, etc.

🟡 **No `prelude` module**.

🟡 **`rsample` for `Kumaraswamy`, `Pareto`, `Weibull` returns `InvalidArgument` with "not yet implemented"** — `src/kumaraswamy.rs:84`, `src/pareto.rs:81`, `src/weibull.rs:84` — These are real mathematical distributions with closed-form inverse CDFs.

🟢 **`Kumaraswamy::new` and `Kumaraswamy::a/b` are missing rustdoc**.

🟢 **`Distribution` trait missing `# Panics` section on `stddev`**.

#### Honest report
- **Did**: listed all 32 `.rs` files; read `Cargo.toml`, `lib.rs`, `special_fns.rs`, `normal.rs` (full), `exponential.rs` (first 160 lines), `independent.rs`, `kumaraswamy.rs`; grep-surveyed all files.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo fmt --check`, miri.
- **Confidence**: high for structural/pattern findings; medium for `stddev` device-stripping severity.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-profiler

#### Vital stats
- **Kind**: lib
- **.rs files**: 6 (lib.rs, cuda_timing.rs, event.rs, flops.rs, profiler.rs, report.rs, schedule.rs — 7 counting schedule)
- **Approx LoC**: 2,002 src + 908 integration tests = ~2,910 total
- **`unsafe {` blocks**: 8 (all in `#[cfg(test)]` code in `report.rs`)
- **`// SAFETY:` comments**: 1
- **Tests**: yes — 4 in `profiler.rs`, 7 in `flops.rs`, 9 in `schedule.rs`, 4 in `report.rs`; plus 2 integration test files
- **Crate-level lint config**: absent
- **Touches GPU**: yes (`cudarc` optional dep; CUDA event API)
- **Edition / MSRV**: workspace-inherited

#### Findings

🔴 **`eprintln!` in library production path** — `profiler.rs:325–329` — `warn_poisoned()` writes directly to stderr from a lib crate. Unconditional, no opt-out.

🔴 **`panic!` in `with_profiler` production path** — `profiler.rs:421` — `Arc::try_unwrap(profiler).unwrap_or_else(|_| panic!("profiler still has dangling references…"))` fires in non-test, non-main code if a user accidentally clones the `Arc`.

🔴 **Silent zero-duration event on CUDA elapsed-time failure** — `cuda_timing.rs:146–149` — `finalize()` calls `self.end.synchronize()` and discards the error with `let _ = ...`, then silently emits a `duration_us = 0` event when `elapsed_ms` returns `Err` or `0.0`. The caller cannot distinguish "GPU ran in <1 µs" from "event query failed".

🟡 **`Profiler` has no `Debug` impl** — `profiler.rs:46`.

🟡 **`ProfileReport` has no `Debug` or `Clone` impl** — `report.rs:56`.

🟡 **No crate-level lint config**.

🟡 **6 `unsafe` restore-blocks in tests lack `// SAFETY:` comments** — `report.rs:622, 625, 628, 636, 642, 643`.

🟡 **`with_profiler` missing `# Panics` doc section** — `profiler.rs:401`.

🟡 **`save_chrome_trace` and `save_tensorboard_trace` missing `# Errors` rustdoc sections** — `report.rs:431, 472`.

🟡 **`flops::estimate` missing `# Returns`/`# Examples` doc** — `flops.rs:20`.

🟡 **`PendingCudaScope` is `pub` but is an implementation detail** — `lib.rs:32`, `cuda_timing.rs:125`.

🟢 **`current_thread_id` uses `format!("{id:?}")` string-parsing as a thread-id hack** — `profiler.rs:429–436`.

🟢 **`ProfileSchedule` is not `Clone`** — `schedule.rs:41`.

#### Honest report
- **Did**: read all 7 source files in full; read Cargo.toml; grepped extensively; cross-referenced SAFETY comments against unsafe block positions; scanned integration test file names and line counts.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo doc`, miri.
- **Confidence**: high.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-hub

#### Vital stats
- **Kind**: lib
- **.rs files**: 7
- **Approx LoC**: 2,119
- **`unsafe {` blocks**: 6 (all inside `#[cfg(test)]` in `auth.rs`, calling `std::env::set_var` / `remove_var`)
- **`// SAFETY:` comments**: 2 (annotate comments above the unsafe blocks, not on the same line immediately preceding each `unsafe {`)
- **Tests**: yes (60 across 6 files)
- **Crate-level lint config**: absent
- **Touches GPU**: no
- **Async/tokio**: no — `ureq` is a blocking HTTP client
- **Network deps**: `ureq 2.12.1` with feature `"tls"`. Lockfile confirms it resolves to `rustls` — TLS posture is acceptable.
- **Edition / MSRV**: edition 2024, rust-version 1.85

#### Findings

🔴 **Path traversal via server-controlled shard filenames** — `download.rs:332-342` — In `hf_download_model`, shard filenames come from parsing `weight_map` in a server-returned `model.safetensors.index.json`. These strings go directly into `fetch_one(repo, revision, shard, cache)` → `cache_name = format!("{repo}/{relative}")` → `cache.store(&cache_name, …)` → `self.cache_dir.join(name)`. **No check is performed for `..` components, absolute paths, or null bytes.** A malicious or compromised HuggingFace response can write files anywhere writable by the process (e.g. `../../.bashrc`). Action: sanitize `relative` (and `repo`, `revision`) before building URLs and cache keys.

🔴 **`eprintln!` in library code** — `download.rs:121-126` — The placeholder-SHA256 warning is emitted via `eprintln!` directly to stderr from a library function.

🔴 **`unsafe` blocks without `// SAFETY:` comments** — `auth.rs:105,112,124,136` — Four `unsafe { std::env::set_var/remove_var }` calls in tests lack the required SAFETY justification comment.

🟡 **No `[lints]` table and no crate-level `#![warn/deny]`**.

🟡 **`HubCache` missing `Debug`** — `cache.rs:21`.

🟡 **All registry SHA-256 entries are all-zero placeholders** — `registry.rs:45-163` — Every model in the static registry uses `"000...000"` as the checksum. The download path detects this and skips integrity verification with a warning. Every download from the static registry is silently unverified.

🟡 **`load_pretrained` doctest uses `.unwrap()`** — `download.rs:185` / `lib.rs:18`.

🟡 **`ModelInfo` has all-public fields on a static-data struct** — `registry.rs:18-31`.

🟡 **`HfTransformerConfig` has all-public fields** — `hf_config.rs:42-113` — Twelve fields are `pub`.

🟡 **No `# Errors` doc section on several public `Result`-returning functions** — `cache.rs` — `HubCache::path_for_model`, `has`, `path`, `load`.

🟢 **`ureq "tls"` feature name is ambiguous but resolves correctly** — `Cargo.toml:29`.

🟢 **`default_cache_dir` falls back to empty string on missing `HOME`** — `cache.rs:14-17`.

🟢 **Hand-rolled hex encoding instead of using `hex` or `data-encoding` crate** — `download.rs:148-163`.

#### Honest report
- **Did**: read all 7 `.rs` files in full, read both `Cargo.toml` files, checked `Cargo.lock` for TLS resolution, grepped extensively for security-relevant patterns.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo doc`, miri.
- **Confidence**: high — small codebase, all files read in full. TLS posture conclusion required lockfile verification which was done.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-tokenize

#### Vital stats
- **Kind**: lib
- **.rs files**: 1 (`src/lib.rs`)
- **Approx LoC**: 493
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (14)
- **Crate-level lint config**: absent
- **Touches GPU**: no
- **Edition / MSRV**: edition 2024, rust-version 1.85

#### Findings

🔴 **Doctests use `.unwrap()` on fallible calls** — `lib.rs:14–16`.

🟡 **No `[lints]` table anywhere**.

🟡 **`encode_batch` allocates a `Vec<String>` unnecessarily** — `lib.rs:72` — `texts: &[&str]` is re-collected into `Vec<String>` via `.to_string()` before being passed to `tokenizers::Tokenizer::encode_batch`. The intermediate owned allocation is pure overhead.

🟡 **`ChatMessage` exposes all fields as `pub`** — `lib.rs:130,132,136`.

🟡 **`ChatMessage` missing `Deserialize`** — `lib.rs:127` — Derives `Serialize` but not `Deserialize`. The `extra` field's `#[serde(flatten)]` already suggests JSON interop is the design intent.

🟡 **`load_chat_template` and `load_tokenizer` map all I/O errors to `InvalidArgument`** — `lib.rs:43,252–256` — File-not-found, permission denied, and parse errors all collapsed.

🟡 **All public functions missing `# Errors` rustdoc section** — `lib.rs:36–113,165–278`.

🟢 **`strftime_now` stub is silently lossy** — `lib.rs:188–189`.

🟢 **`id_to_token` returns `Option<String>` where `Option<&str>` or `Cow<str>` might be possible** — `lib.rs:112`.

#### Honest report
- **Did**: read full `src/lib.rs` (493 lines), read `Cargo.toml` (crate + workspace), read `ferrotorch-core/src/error.rs`, grepped extensively, inspected `tokenizers` upstream API.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo doc`.
- **Confidence**: high.
- **Skill loading**: both loaded; GPU discipline does not apply.

---

### Crate: ferrotorch-llama

#### Vital stats
- **Kind**: lib (with 3 GPU-gated examples)
- **.rs files**: 15
- **Approx LoC**: ~8,046
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (8 modules)
- **Crate-level lint config**: absent
- **Touches GPU**: yes — `cuda` feature gates `ferrotorch-gpu`, `cudarc`, `ferrotorch-cubecl`, `cubecl`/`cubecl-cuda`; `gpu.rs` runs a full bf16 forward pass via PTX kernels; `grammar/gpu_dispatch.rs` dispatches DFA mask computation via CubeCL
- **Edition / MSRV**: edition 2024, MSRV 1.85

#### Findings

🔴 **`.unwrap()` on `T::from(i as f64)` in hot forward path** — `src/model.rs:189` and `src/mlp.rs:48` — In `model.rs` this runs inside `forward_from_ids` on every token; in `mlp.rs` it runs on every `FatRelu` activation call.

🔴 **`.to_f64().unwrap()` in generation loop** — `src/generation.rs:159` and `src/generation.rs:422` — Inside `generate_with_streamer` and `beam_search`.

🔴 **`.expect()` / `.unwrap()` in `JsonGrammar` state machine production code** — `src/grammar/state.rs:976, 1069, 1073, 1145, 1199` — Five panic sites in non-test code.

🔴 **GPU error types collapsed to `FerrotorchError::InvalidArgument` via `map_gpu_err` / `map_driver_err`** — `src/gpu.rs:721–730` — Both `GpuError` and `cudarc::driver::DriverError` are stringified into `InvalidArgument { message: format!("gpu error: {e}") }`. This destroys structured error information.

🟡 **No `[lints]` table — no crate-level or workspace-level lint discipline**.

🟡 **Hand-rolled `Display + Error` impls when `thiserror` is in the workspace** — `src/grammar/schema.rs:45–57`, `src/grammar/state.rs:47–62`, `src/grammar/json_schema.rs:62–72`.

🟡 **`LlamaGpuLayer` and `LlamaGpuInferencer` are `pub` structs with all fields `pub`, but no `Debug` derive** — `src/gpu.rs:79–101` — Blocked by `CudaSlice<u16>` not implementing `Debug`. Manual impl needed.

🟡 **`PackedVocab` (public struct in `gpu_dispatch`) has no `Debug` derive** — `src/grammar/gpu_dispatch.rs:709`.

🟡 **`generate` / `generate_with_streamer` lack `# Errors` and `# Panics` rustdoc sections** — `src/generation.rs:97,107`.

🟢 **No GPU tests in the crate** — `src/` has no `#[cfg(feature="cuda")]` test modules and no integration test exercising `LlamaGpuInferencer` or `compute_mask_gpu`. The GPU paths are tested only via the examples, which are not part of `cargo test`. Per gpu-discipline rule §4d, a test constructing a `GpuDevice`, running `forward_from_ids`, and asserting `result.len() == vocab_size` belongs in `#[cfg(all(test, feature="cuda"))]`.

🟢 **`GptqQ4`, `AwqQ4`, `HqqWeights` have all fields `pub` with no constructor**.

#### Honest report
- **Did**: Read all 15 `.rs` files; full reads of `lib.rs`, `gpu.rs`, `grammar/mod.rs`, `grammar/gpu_dispatch.rs`, `grammar/json_schema.rs`, `grammar/schema.rs`, `grammar/state.rs`, `model.rs`, `mlp.rs`, `generation.rs`, `quant_loaders.rs`. Ran greps for all forbidden patterns.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo doc`, `cargo fmt --check`, miri.
- **Confidence**: high for structural findings; medium for the `T::from().unwrap()` risk severity.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-ml

#### Vital stats
- **Kind**: lib
- **.rs files**: 4 (lib.rs, adapter.rs, metrics.rs, datasets.rs)
- **Approx LoC**: ~1,637
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (57 total)
- **Crate-level lint config**: absent
- **Touches GPU**: no (all deps are CPU-only ferrolearn + ndarray; GPU tensors are accepted as inputs but immediately materialised to CPU via `data_vec()`)
- **Edition / MSRV**: Edition 2024, MSRV 1.85

#### Findings

🔴 **`unwrap()` on `to_f64()` in lib code — silent panic on non-finite values** — `metrics.rs:149`, `212`, `322`, `375`; `adapter.rs:103` — `num_traits::ToPrimitive::to_f64()` returns `None` for values that don't fit the conversion.

🟡 **No crate-level lint configuration**.

🟡 **Module doc advertises `mean_absolute_percentage_error` but the function does not exist** — `metrics.rs:10` — `cargo doc` will emit a broken intra-doc link.

🟡 **Four `Result`-returning public functions in `adapter.rs` have no `# Errors` section** — `array1_to_tensor` (line 80), `array2_to_tensor` (line 87), `array1_usize_to_tensor` (line 102), `tensor_to_array1_usize` (line 113).

🟡 **No doctests on any public item** — zero `# Examples` code blocks across all 4 source files.

🟢 **`median_absolute_error` and `mean_absolute_percentage_error` tests are absent**.

🟢 **Transitive `T: num_traits::Float + Send + Sync + 'static` bound repeated across every metric and dataset function**.

#### Honest report
- **Did**: read all 4 `.rs` files in full; read `Cargo.toml`; grepped patterns.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `cargo doc`, miri.
- **Confidence**: high.
- **Skill loading**: loaded via Skill tool.

---

### Crate: ferrotorch-mps

#### Vital stats
- **Kind**: lib
- **.rs files**: 1 (`src/lib.rs`)
- **Approx LoC**: 162
- **`unsafe {` blocks**: 0
- **`// SAFETY:` comments**: 0
- **Tests**: yes (5)
- **Crate-level lint config**: absent
- **Touches GPU**: no (the `metal-backend` feature is declared but the `metal` crate is not in `[dependencies]` at all; the feature flag body is empty)
- **MSL/.metal kernel files present**: none
- **Edition / MSRV**: both `workspace = true`

#### Findings

🔴 **`init_mps_backend` is a lying success stub** — `src/lib.rs:95–107` — The function is documented as registering `MpsBackendImpl` with `gpu_dispatch::set_gpu_backend`, but its body does nothing at all after the availability guard — it returns `Ok(())` without touching the dispatch table. **Dispatch-only dispatch anti-pattern verbatim.** Callers see a successful init, then every subsequent op dispatch silently returns `DeviceUnavailable`. The comment in the body explicitly admits this ("Until that lands the init succeeds…").

🔴 **`device_count` hard-codes `1` with no Metal call behind the guard** — `src/lib.rs:78–86` — When `is_mps_available()` is `true` (macOS + `metal-backend` feature), this returns `1` via an inline comment `// Real Metal dispatch would call MTLCopyAllDevices()`. No `objc2-metal`/`metal` crate is wired, so even on a real macOS build the count is fabricated.

🔴 **`metal-backend` feature is declared but the `metal` (or `objc2-metal`) crate is not in `[dependencies]`** — `Cargo.toml:18` — Enabling `metal-backend` on macOS adds zero new dependencies and activates zero new code paths.

🟡 **No `[lints]` table and no `#![warn/deny]` crate attributes**.

🟡 **`device_count` is a free function with an ambiguous name in a glob re-export** — `src/lib.rs:78` — `ferrotorch::mps::device_count()` could collide with `ferrotorch_gpu::device_count()`.

🟡 **Tests are all conditionally no-op on macOS + metal-backend** — `src/lib.rs:113–161` — Every test body is wrapped in `if !is_mps_available() { … }`. When the feature is on and the platform is Apple, all five tests become empty passes.

🟡 **`MpsDevice` struct field `ordinal` is private but there is no `Display` impl** — `src/lib.rs:56–58` — Missing `PartialEq`/`Eq`/`Hash`.

🟢 **`is_mps_available` uses two `#[cfg]` blocks instead of `cfg_attr` or a single `cfg` expression** — `src/lib.rs:40–47`.

#### Honest report

- **Did**: read `Cargo.toml` and `src/lib.rs` in full; grepped extensively; checked workspace `Cargo.toml` for `[lints]`; examined `ferrotorch-core` for `Device::Mps` and `DeviceUnavailable` handling; checked the `ferrotorch` top-level crate for glob re-export scope; searched for `.metal`/`.msl`/`.air` kernel files.
- **Did NOT do**: `cargo build`, `cargo clippy`, `cargo test`, `miri`.
- **Confidence**: high for what is present; medium for downstream impact.
- **Skill loading**: both loaded via Skill tool.

---

## Audit footer

- **Audit ID**: crosslink #648
- **Subagent type**: general-purpose, model: sonnet 4.6, 22 parallel
- **Skill activation outcome**: 22/22 reported `Skill` tool worked and both skills loaded — empirical confirmation that general-purpose Sonnet 4.6 subagents can both see and invoke skills.
- **Cost note**: not measured; ~22 × ~50k token contexts ≈ 1.1M token of subagent compute.
- **Reproducibility**: each subagent's prompt is in this conversation's history; the same reviews can be re-run by re-dispatching with identical prompts.
