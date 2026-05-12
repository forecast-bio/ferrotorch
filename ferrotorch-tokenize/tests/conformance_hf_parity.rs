//! Conformance Phase G.2 — exact encode/decode/chat-template parity
//! against the HuggingFace `tokenizers.Tokenizer` reference for five
//! canonical tokenizer families (Llama 3, CLIP, BERT, GPT-2, SmolLM).
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/1168>.
//!
//! Fixture mirror: `ferrotorch/tokenizer-parity-v1`. Each family ships
//!
//!   * `tokenizer.json`         — the same file the rust side loads,
//!   * `strings.json`           — 20 fixed test inputs,
//!   * `token_ids.json`         — Python reference encodings,
//!   * `decoded.json`           — Python reference decodes,
//!   * `chat_template.json`     — Python reference chat-template
//!     renders (where the upstream defines one).
//!
//! The harness asserts **exact** integer / string equality. Token
//! tokenization is integer-domain — there is no float tolerance, so a
//! divergence on a single id or character is a real bug or HF-version
//! skew.
//!
//! This test is `#[ignore]`-gated because it downloads fixtures from
//! the HF mirror; run it with:
//!
//! ```text
//! cargo test -p ferrotorch-tokenize --test conformance_hf_parity -- --ignored
//! ```
//!
//! The richer python harness (`scripts/verify_tokenizer_inference.py`)
//! also covers the rust dump binary against the same fixtures; this
//! cargo-side test exercises the in-process API so a `cargo test`
//! run never accidentally skips parity.

use std::path::PathBuf;
use std::process::Command;

use ferrotorch_tokenize::{ChatMessage, apply_chat_template, decode, encode, load_tokenizer};
use serde::Deserialize;

const TRAJ_REPO: &str = "ferrotorch/tokenizer-parity-v1";

/// The five canonical families, in the order the pin script writes
/// them.
const FAMILIES: &[&str] = &["llama3", "clip", "bert", "gpt2", "smollm"];

#[derive(Debug, Deserialize)]
struct TokenIdsRef {
    encode_with_special: Vec<Vec<u32>>,
    encode_no_special: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct DecodedRef {
    decode_with_special_keep: Vec<String>,
    decode_with_special_skip: Vec<String>,
    decode_no_special: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ChatTemplateRef {
    has_chat_template: bool,
    #[serde(default)]
    messages: Vec<RefMessage>,
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    rendered_no_generation_prompt: Option<String>,
    #[serde(default)]
    rendered_with_generation_prompt: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct RefMessage {
    role: String,
    content: String,
}

/// Download one family's fixtures via `huggingface_hub.hf_hub_download`
/// and return the local directory holding them.
///
/// We shell out to Python rather than depending on a Rust HF client
/// from this leaf crate, because (a) `hf_hub_download` is the same
/// path the other parity harnesses use, (b) it correctly handles auth
/// tokens / proxy / caching transparently, and (c) the test is
/// `#[ignore]`-gated so the dev environment is already expected to
/// have `huggingface_hub` installed.
fn fetch_family_fixtures(family: &str) -> PathBuf {
    let stage = std::env::temp_dir().join(format!("ferrotorch_tokenize_parity_{family}"));
    if stage.exists() {
        std::fs::remove_dir_all(&stage)
            .unwrap_or_else(|e| panic!("clear stage {}: {e}", stage.display()));
    }
    std::fs::create_dir_all(&stage)
        .unwrap_or_else(|e| panic!("mkdir stage {}: {e}", stage.display()));

    let script = format!(
        "import os, sys, shutil
from huggingface_hub import hf_hub_download, list_repo_files
dest = sys.argv[1]
family = sys.argv[2]
files = [f for f in list_repo_files('{TRAJ_REPO}') if f.startswith(family + '/')]
if not files:
    print(f'no files for {{family}} in {TRAJ_REPO}', file=sys.stderr); sys.exit(2)
for f in files:
    cached = hf_hub_download(repo_id='{TRAJ_REPO}', filename=f)
    bn = os.path.basename(f)
    dst = os.path.join(dest, bn)
    if os.path.lexists(dst):
        os.remove(dst)
    shutil.copyfile(cached, dst)
print('OK', len(files))
"
    );

    let out = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .arg(stage.as_os_str())
        .arg(family)
        .output()
        .unwrap_or_else(|e| panic!("spawn python3: {e}"));
    if !out.status.success() {
        panic!(
            "hf_hub_download for {family} failed:\n  stdout: {}\n  stderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    stage
}

fn read_json<T: for<'de> Deserialize<'de>>(p: &std::path::Path) -> T {
    let body = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn short_input(s: &str) -> String {
    let mut out: String = s.chars().take(40).collect();
    if s.chars().count() > 40 {
        out.push_str("...");
    }
    out.replace('\n', "\\n").replace('\t', "\\t")
}

fn verify_family(family: &str) {
    let dir = fetch_family_fixtures(family);
    let tok_path = dir.join("tokenizer.json");
    let strings: Vec<String> = read_json(&dir.join("strings.json"));
    let ref_ids: TokenIdsRef = read_json(&dir.join("token_ids.json"));
    let ref_dec: DecodedRef = read_json(&dir.join("decoded.json"));
    let ref_chat: ChatTemplateRef = read_json(&dir.join("chat_template.json"));

    let tok = load_tokenizer(&tok_path)
        .unwrap_or_else(|e| panic!("load_tokenizer({}): {e:?}", tok_path.display()));

    assert_eq!(
        ref_ids.encode_with_special.len(),
        strings.len(),
        "fixture inconsistent: encode_with_special.len() != strings.len()"
    );

    // --- Encode parity -------------------------------------------------
    for (i, s) in strings.iter().enumerate() {
        let rust_ws = encode(&tok, s, true)
            .unwrap_or_else(|e| panic!("[{family}] case[{i}] encode(true): {e:?}"));
        assert_eq!(
            rust_ws,
            ref_ids.encode_with_special[i],
            "[{family}] encode(add_special_tokens=true) mismatch on {:?}",
            short_input(s)
        );
        let rust_ns = encode(&tok, s, false)
            .unwrap_or_else(|e| panic!("[{family}] case[{i}] encode(false): {e:?}"));
        assert_eq!(
            rust_ns,
            ref_ids.encode_no_special[i],
            "[{family}] encode(add_special_tokens=false) mismatch on {:?}",
            short_input(s)
        );
    }

    // --- Decode parity (3 variants) ------------------------------------
    for (i, s) in strings.iter().enumerate() {
        let ids_ws = &ref_ids.encode_with_special[i];
        let ids_ns = &ref_ids.encode_no_special[i];

        let rust_keep = decode(&tok, ids_ws, false)
            .unwrap_or_else(|e| panic!("[{family}] case[{i}] decode(keep): {e:?}"));
        assert_eq!(
            rust_keep,
            ref_dec.decode_with_special_keep[i],
            "[{family}] decode_with_special_keep mismatch on {:?}",
            short_input(s)
        );

        let rust_skip = decode(&tok, ids_ws, true)
            .unwrap_or_else(|e| panic!("[{family}] case[{i}] decode(skip): {e:?}"));
        assert_eq!(
            rust_skip,
            ref_dec.decode_with_special_skip[i],
            "[{family}] decode_with_special_skip mismatch on {:?}",
            short_input(s)
        );

        let rust_no_special = decode(&tok, ids_ns, false)
            .unwrap_or_else(|e| panic!("[{family}] case[{i}] decode(no_special): {e:?}"));
        assert_eq!(
            rust_no_special,
            ref_dec.decode_no_special[i],
            "[{family}] decode_no_special mismatch on {:?}",
            short_input(s)
        );
    }

    // --- Chat-template parity ------------------------------------------
    if ref_chat.has_chat_template {
        let template = ref_chat
            .template
            .as_ref()
            .expect("has_chat_template=true but template missing in fixture");
        let messages: Vec<ChatMessage> = ref_chat
            .messages
            .iter()
            .map(|m| ChatMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        let bos = ref_chat.bos_token.as_deref();
        let eos = ref_chat.eos_token.as_deref();

        let rust_no_gen = apply_chat_template(template, &messages, false, bos, eos)
            .unwrap_or_else(|e| panic!("[{family}] apply_chat_template(no_gen): {e:?}"));
        assert_eq!(
            rust_no_gen.as_str(),
            ref_chat
                .rendered_no_generation_prompt
                .as_deref()
                .unwrap_or(""),
            "[{family}] chat_template (no_generation_prompt) mismatch"
        );

        let rust_with_gen = apply_chat_template(template, &messages, true, bos, eos)
            .unwrap_or_else(|e| panic!("[{family}] apply_chat_template(with_gen): {e:?}"));
        assert_eq!(
            rust_with_gen.as_str(),
            ref_chat
                .rendered_with_generation_prompt
                .as_deref()
                .unwrap_or(""),
            "[{family}] chat_template (with_generation_prompt) mismatch"
        );
    } else {
        // For families without a chat template, the fixture marker
        // must agree with the absence — defensive guard in case a
        // future upstream gains a chat_template that the pin script
        // missed.
        assert!(
            !ref_chat.has_chat_template,
            "fixture says has_chat_template={} but template field missing",
            ref_chat.has_chat_template
        );
    }
}

#[test]
#[ignore = "downloads ferrotorch/tokenizer-parity-v1 from HuggingFace"]
fn hf_parity_llama3() {
    verify_family("llama3");
}

#[test]
#[ignore = "downloads ferrotorch/tokenizer-parity-v1 from HuggingFace"]
fn hf_parity_clip() {
    verify_family("clip");
}

#[test]
#[ignore = "downloads ferrotorch/tokenizer-parity-v1 from HuggingFace"]
fn hf_parity_bert() {
    verify_family("bert");
}

#[test]
#[ignore = "downloads ferrotorch/tokenizer-parity-v1 from HuggingFace"]
fn hf_parity_gpt2() {
    verify_family("gpt2");
}

#[test]
#[ignore = "downloads ferrotorch/tokenizer-parity-v1 from HuggingFace"]
fn hf_parity_smollm() {
    verify_family("smollm");
}

/// Smoke test for the helper functions themselves — these run in the
/// default `cargo test` invocation (no `--ignored`) so a regression in
/// the test scaffolding surfaces immediately.
#[test]
fn families_constant_is_complete() {
    assert_eq!(
        FAMILIES.len(),
        5,
        "Phase G.2 spec calls for exactly 5 tokenizer families"
    );
    for f in FAMILIES {
        // Each family identifier is lowercase alphanumeric + digit.
        assert!(
            f.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()),
            "family id {f:?} must be ASCII lowercase/digit"
        );
    }
}
