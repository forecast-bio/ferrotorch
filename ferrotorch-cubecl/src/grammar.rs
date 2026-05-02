//! GPU constrained-decoding token-mask computation.
//!
//! Companion to [`crate::quant::kernel_apply_token_mask`]. That kernel
//! *applies* a precomputed allow mask to a logits buffer; this module
//! *computes* the allow mask itself by walking each vocab entry through a
//! schema-derived DFA.
//!
//! Architectural rationale: with a Llama-3 vocab (128k entries × ~5 chars
//! each), per-token DFA walks are 600k char transitions per generation
//! step — embarrassingly parallel and dominant on the CPU side. Moving the
//! walk to GPU collapses the grammar processor's per-step cost from ~tens
//! of milliseconds (CPU) to ~hundreds of microseconds (GPU launch +
//! dispatch). One thread per vocab entry; each thread reads its token's
//! chars and walks the DFA in registers.
//!
//! ## DFA model
//!
//! - **States** are numbered `0..num_states`. The caller passes
//!   `start_state` and `reject_state` explicitly so any state numbering
//!   scheme works.
//! - **Char classes** collapse the 128-entry ASCII space into a small set
//!   (e.g., for Boolean: `t`, `r`, `u`, `e`, `f`, `a`, `l`, `s`, OTHER).
//!   Non-ASCII chars (codepoint ≥ 128) are always treated as REJECT.
//! - **Transitions** are flat `[num_states × num_classes]` row-major; the
//!   value at `[s * num_classes + class]` is the next state, or
//!   `reject_state`.
//!
//! Stage 1 supports any schema for which the host can produce a finite
//! DFA (e.g., `Schema::Boolean`, `Schema::Null`, fixed-shape literal
//! enums). Schemas with dynamic state — required-key tracking, partial
//! enum-value matching across an unbounded vocabulary — are out of scope
//! for this kernel and stay on the CPU loop in
//! `JsonSchemaProcessor::compute_mask`.

use cubecl::prelude::*;

/// One thread per vocab entry. Each thread walks its token's chars
/// through the DFA encoded by `transitions` + `char_classes`, then
/// writes `1` to `allow[tok]` iff the walk never lands on `reject_state`.
///
/// Scalar inputs (`num_classes`, `start_state`, `reject_state`,
/// `max_token_len`) are passed by value, matching the
/// `kernel_matmul_naive` idiom in `kernels.rs`. The kernel converts u32
/// scalars to `usize` for indexing — the same `as usize` pattern that
/// `kernel_matmul_naive` uses.
#[cube(launch_unchecked)]
pub fn kernel_compute_token_mask_dfa(
    transitions: &Array<u32>,
    char_classes: &Array<u32>,
    vocab_offsets: &Array<u32>,
    vocab_chars: &Array<u32>,
    allow: &mut Array<u32>,
    num_classes: u32,
    start_state: u32,
    reject_state: u32,
    max_token_len: u32,
) {
    let n_alw = allow.len();
    if ABSOLUTE_POS < n_alw {
        let num_classes_u = num_classes as usize;
        let max_token_len_u = max_token_len as usize;

        let tok = ABSOLUTE_POS;
        let tok_next = tok + 1usize;
        let start = vocab_offsets[tok] as usize;
        let end = vocab_offsets[tok_next] as usize;

        let mut state: u32 = start_state;
        let mut rejected: u32 = 0u32;

        // Empty tokens never advance the state. The CPU mirror
        // (`reference_walk` in tests) treats empty as reject for
        // consistency with the existing CPU `compute_mask`, which skips
        // empty tokens and leaves their slot at zero.
        if start == end {
            rejected = 1u32;
        }

        // Bounded loop. CubeCL doesn't have `break`, so we guard with
        // `rejected == 0` and the iteration count is fixed.
        for i in 0..max_token_len_u {
            let pos = start + i;
            if pos < end && rejected == 0u32 {
                let c = vocab_chars[pos];
                if c < 128u32 {
                    let class_u = char_classes[c as usize] as usize;
                    let idx = (state as usize) * num_classes_u + class_u;
                    state = transitions[idx];
                    if state == reject_state {
                        rejected = 1u32;
                    }
                } else {
                    rejected = 1u32;
                }
            }
        }

        if rejected == 0u32 {
            allow[tok] = 1u32;
        } else {
            allow[tok] = 0u32;
        }
    }
}

/// Launch dimensions: 256 threads per cube, one thread per output.
fn elementwise_launch_dims(n: u32) -> (CubeCount, CubeDim) {
    let units_per_cube: u32 = 256;
    let num_cubes = n.div_ceil(units_per_cube).max(1);
    (
        CubeCount::Static(num_cubes, 1, 1),
        CubeDim::new_1d(units_per_cube),
    )
}

/// Inputs the host has to assemble before calling
/// [`compute_token_mask_dfa_to_gpu`].
pub struct DfaMaskInputs<'a> {
    pub transitions: &'a [u32],
    pub char_classes: &'a [u32],
    pub vocab_offsets: &'a [u32],
    pub vocab_chars: &'a [u32],
    pub num_classes: u32,
    pub start_state: u32,
    pub reject_state: u32,
    pub max_token_len: u32,
}

/// Upload all inputs, dispatch [`kernel_compute_token_mask_dfa`], and
/// return the on-device allow-mask handle plus the vocab size. Caller
/// reads the handle back via `client.read_one`.
pub fn compute_token_mask_dfa_to_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    inputs: &DfaMaskInputs<'_>,
) -> (cubecl::server::Handle, usize) {
    let vocab_size = inputs.vocab_offsets.len() - 1;
    let bytes_u32 = std::mem::size_of::<u32>();

    let trans_handle = client.create_from_slice(u32_slice_as_bytes(inputs.transitions));
    let class_handle = client.create_from_slice(u32_slice_as_bytes(inputs.char_classes));
    let offsets_handle = client.create_from_slice(u32_slice_as_bytes(inputs.vocab_offsets));
    let chars_handle = client.create_from_slice(u32_slice_as_bytes(inputs.vocab_chars));
    let allow_handle = client.empty(vocab_size * bytes_u32);

    let (count, dim) = elementwise_launch_dims(vocab_size as u32);
    unsafe {
        kernel_compute_token_mask_dfa::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(trans_handle, inputs.transitions.len()),
            ArrayArg::from_raw_parts(class_handle, inputs.char_classes.len()),
            ArrayArg::from_raw_parts(offsets_handle, inputs.vocab_offsets.len()),
            ArrayArg::from_raw_parts(chars_handle, inputs.vocab_chars.len()),
            ArrayArg::from_raw_parts(allow_handle.clone(), vocab_size),
            inputs.num_classes,
            inputs.start_state,
            inputs.reject_state,
            inputs.max_token_len,
        );
    }
    (allow_handle, vocab_size)
}

fn u32_slice_as_bytes(s: &[u32]) -> &[u8] {
    // `&[u32]` is `align >= align_of::<u8>()` and contiguous, so this
    // reinterpret is safe. CubeCL's upload path accepts `&[u8]`.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
}

// ---------------------------------------------------------------------------
// CUDA runtime tests
// ---------------------------------------------------------------------------
//
// These tests construct a real `cubecl_cuda::CudaRuntime` client, dispatch
// `kernel_compute_token_mask_dfa` on the GPU, and compare the result to a
// pure-Rust reference walk. Gated on `--features cuda`; require a CUDA
// device at test time. There are NO `#[ignore]` markers and NO CPU
// fallbacks — if the test compiles in this configuration it MUST run on
// GPU, matching the discipline established by `quant.rs::cuda_tests`.

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use cubecl_cuda::{CudaDevice, CudaRuntime};

    fn cuda_client() -> ComputeClient<CudaRuntime> {
        let device = CudaDevice { index: 0 };
        CudaRuntime::client(&device)
    }

    fn read_u32(
        client: &ComputeClient<CudaRuntime>,
        handle: cubecl::server::Handle,
        n: usize,
    ) -> Vec<u32> {
        let bytes = client.read_one(handle).expect("CUDA read_one failed");
        assert_eq!(bytes.len(), n * std::mem::size_of::<u32>());
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out
    }

    /// Pure-Rust reference: walk each token through the DFA, return the
    /// allow mask. The kernel must produce byte-identical output.
    fn reference_walk(inputs: &DfaMaskInputs<'_>) -> Vec<u32> {
        let n = inputs.vocab_offsets.len() - 1;
        let mut out = vec![0u32; n];
        for (tok, slot) in out.iter_mut().enumerate().take(n) {
            let start = inputs.vocab_offsets[tok] as usize;
            let end = inputs.vocab_offsets[tok + 1] as usize;
            if start == end {
                continue;
            }
            let mut state = inputs.start_state;
            let mut rejected = false;
            for &c in &inputs.vocab_chars[start..end] {
                if c >= 128 {
                    rejected = true;
                    break;
                }
                let class = inputs.char_classes[c as usize];
                state = inputs.transitions[(state * inputs.num_classes + class) as usize];
                if state == inputs.reject_state {
                    rejected = true;
                    break;
                }
            }
            if !rejected {
                *slot = 1;
            }
        }
        out
    }

    /// A 2-state DFA that accepts any non-empty token consisting only of
    /// the letter 'a'. Verifies the kernel's char-class lookup, transition
    /// indexing, ASCII guard, and reject short-circuit on a hand-built
    /// input.
    #[test]
    fn dfa_kernel_matches_hand_built_walk() {
        let mut classes = vec![1u32; 128];
        classes[b'a' as usize] = 0;
        let num_classes = 2u32;
        // States: 0 = accept-state-so-far, 1 = REJECT.
        let transitions: Vec<u32> = vec![
            /* state 0, class 0 ('a') */ 0, /* state 0, class 1 (other) */ 1,
            /* state 1, class 0 */ 1, /* state 1, class 1 */ 1,
        ];

        let tokens: &[&str] = &[
            "a",   // accept
            "aa",  // accept
            "aaa", // accept
            "ab",  // reject
            "ba",  // reject
            "",    // empty -> reject by convention
        ];
        let mut vocab_offsets: Vec<u32> = vec![0];
        let mut vocab_chars: Vec<u32> = Vec::new();
        for tok in tokens {
            for c in tok.chars() {
                vocab_chars.push(c as u32);
            }
            vocab_offsets.push(vocab_chars.len() as u32);
        }

        let inputs = DfaMaskInputs {
            transitions: &transitions,
            char_classes: &classes,
            vocab_offsets: &vocab_offsets,
            vocab_chars: &vocab_chars,
            num_classes,
            start_state: 0,
            reject_state: 1,
            max_token_len: 8,
        };
        let expected = reference_walk(&inputs);
        assert_eq!(expected, vec![1, 1, 1, 0, 0, 0]);

        let client = cuda_client();
        let (handle, n) = compute_token_mask_dfa_to_gpu::<CudaRuntime>(&client, &inputs);
        let got = read_u32(&client, handle, n);
        assert_eq!(got, expected, "GPU mask must match CPU reference walk");
    }

    /// Stress: a DFA over ASCII digits that accepts any non-empty digit
    /// sequence. Exercises larger vocab + longer tokens.
    #[test]
    fn dfa_kernel_accepts_digit_sequences() {
        let mut classes = vec![1u32; 128];
        for d in b'0'..=b'9' {
            classes[d as usize] = 0;
        }
        let num_classes = 2u32;
        let transitions: Vec<u32> = vec![0, 1, 1, 1];

        let tokens: Vec<String> = (0..10)
            .map(|d| d.to_string())
            .chain((10..20).map(|n| n.to_string()))
            .chain((100..105).map(|n| n.to_string()))
            .chain(["1a", "2b", "3c", "4d", "x9"].iter().map(|s| s.to_string()))
            .chain(["".to_string(), "1234567".to_string()])
            .collect();

        let mut vocab_offsets: Vec<u32> = vec![0];
        let mut vocab_chars: Vec<u32> = Vec::new();
        for tok in &tokens {
            for c in tok.chars() {
                vocab_chars.push(c as u32);
            }
            vocab_offsets.push(vocab_chars.len() as u32);
        }

        let inputs = DfaMaskInputs {
            transitions: &transitions,
            char_classes: &classes,
            vocab_offsets: &vocab_offsets,
            vocab_chars: &vocab_chars,
            num_classes,
            start_state: 0,
            reject_state: 1,
            max_token_len: 8,
        };
        let expected = reference_walk(&inputs);

        let client = cuda_client();
        let (handle, n) = compute_token_mask_dfa_to_gpu::<CudaRuntime>(&client, &inputs);
        let got = read_u32(&client, handle, n);
        assert_eq!(got, expected, "digit-DFA GPU mask must match CPU reference");

        // Sanity: 10 single + 10 two-digit + 5 three-digit + 1 long-valid
        // = 26 accepts; 5 mixed-letter + 1 empty = 6 rejects.
        let accepted: u32 = got.iter().sum();
        assert_eq!(accepted, 26);
    }

    /// Non-ASCII codepoints must always reject. Verifies the `c < 128`
    /// guard in the kernel.
    #[test]
    fn dfa_kernel_rejects_non_ascii() {
        // DFA that would accept anything (single state, any class
        // self-loops). Without the ASCII guard the non-ASCII tokens would
        // pass; with it, they reject.
        let classes = vec![0u32; 128];
        let num_classes = 1u32;
        let transitions: Vec<u32> = vec![0];

        let tokens: Vec<String> = vec!["abc".to_string(), "héllo".to_string(), "x".to_string()];
        let mut vocab_offsets: Vec<u32> = vec![0];
        let mut vocab_chars: Vec<u32> = Vec::new();
        for tok in &tokens {
            for c in tok.chars() {
                vocab_chars.push(c as u32);
            }
            vocab_offsets.push(vocab_chars.len() as u32);
        }

        let inputs = DfaMaskInputs {
            transitions: &transitions,
            char_classes: &classes,
            vocab_offsets: &vocab_offsets,
            vocab_chars: &vocab_chars,
            num_classes,
            start_state: 0,
            reject_state: u32::MAX,
            max_token_len: 8,
        };
        let expected = reference_walk(&inputs);
        assert_eq!(expected, vec![1, 0, 1]);

        let client = cuda_client();
        let (handle, n) = compute_token_mask_dfa_to_gpu::<CudaRuntime>(&client, &inputs);
        let got = read_u32(&client, handle, n);
        assert_eq!(got, expected, "non-ASCII must reject on both CPU and GPU");
    }
}
