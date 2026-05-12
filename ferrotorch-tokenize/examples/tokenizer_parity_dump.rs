//! Tokenizer parity dump binary for the ferrotorch-tokenize real-
//! artifact harness (Phase G.2, #1168).
//!
//! Companion to `scripts/verify_tokenizer_inference.py` and
//! `scripts/pin_pretrained_tokenizer_fixtures.py`. Given a family
//! name, a path to the per-family fixture folder, and an output dir,
//! this example:
//!
//!   1. Loads the family's `tokenizer.json` via
//!      [`ferrotorch_tokenize::load_tokenizer`].
//!   2. Reads `strings.json` (the same 20-element corpus the python
//!      pin script used) from the fixture folder.
//!   3. Runs [`ferrotorch_tokenize::encode`] twice per string
//!      (`add_special_tokens` true and false) and dumps the resulting
//!      `Vec<u32>` per-string list to `token_ids.json`.
//!   4. Runs [`ferrotorch_tokenize::decode`] three times per string
//!      (with-special + skip false / skip true, no-special + skip
//!      false) and dumps the strings to `decoded.json`.
//!   5. Reads `chat_template.json` from the fixture folder. If the
//!      family has a chat template, renders the canonical conversation
//!      with [`ferrotorch_tokenize::apply_chat_template`] for
//!      `add_generation_prompt` false and true and writes the rendered
//!      strings to the rust-side `chat_template.json`. Otherwise
//!      writes `{ "has_chat_template": false }`.
//!
//! Files written to `--output-dir` (matching the python reference
//! layout exactly so the verifier can do basename diffs):
//!
//!   * `<family>/token_ids.json`
//!   * `<family>/decoded.json`
//!   * `<family>/chat_template.json`
//!
//! The verifier checks every list and string with **exact** equality;
//! there is no tolerance.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-tokenize --release \
//!   --example tokenizer_parity_dump -- \
//!     --family llama3 \
//!     --fixture-dir /tmp/.../llama3 \
//!     --output-dir  /tmp/.../rust_out/llama3
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use ferrotorch_tokenize::{ChatMessage, apply_chat_template, decode, encode, load_tokenizer};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    family: String,
    fixture_dir: PathBuf,
    output_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut family: Option<String> = None;
    let mut fixture_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--family" => {
                family = Some(argv.get(i + 1).ok_or("--family needs a value")?.clone());
                i += 2;
            }
            "--fixture-dir" => {
                fixture_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--fixture-dir needs a value")?,
                ));
                i += 2;
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output-dir needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        family: family.ok_or("--family is required")?,
        fixture_dir: fixture_dir.ok_or("--fixture-dir is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
    })
}

// ---------------------------------------------------------------------------
// On-disk JSON shapes (mirror the python pin script's layout exactly).
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct TokenIdsDump {
    encode_with_special: Vec<Vec<u32>>,
    encode_no_special: Vec<Vec<u32>>,
}

#[derive(Debug, Serialize)]
struct DecodedDump {
    decode_with_special_keep: Vec<String>,
    decode_with_special_skip: Vec<String>,
    decode_no_special: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RefChatBlock {
    has_chat_template: bool,
    #[serde(default)]
    messages: Vec<RefChatMessage>,
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    // The python reference renders we will compare against. We mirror
    // them in the rust-side block so the verifier can diff in-place
    // without re-loading the python file.
    #[serde(default)]
    rendered_no_generation_prompt: Option<String>,
    #[serde(default)]
    rendered_with_generation_prompt: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct RefChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatDump {
    has_chat_template: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    rendered_no_generation_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rendered_with_generation_prompt: Option<String>,
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

fn read_strings(path: &Path) -> Result<Vec<String>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice::<Vec<String>>(&bytes)
        .map_err(|e| format!("parse {} as Vec<String>: {e}", path.display()))
}

fn read_chat_block(path: &Path) -> Result<RefChatBlock, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice::<RefChatBlock>(&bytes)
        .map_err(|e| format!("parse {} as RefChatBlock: {e}", path.display()))
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let body = serde_json::to_string_pretty(value)
        .map_err(|e| format!("serialize {}: {e}", path.display()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), String> {
    let args = parse_args()?;

    // Resolve fixture inputs and rust outputs.
    let tok_path = args.fixture_dir.join("tokenizer.json");
    let strings_path = args.fixture_dir.join("strings.json");
    let chat_in_path = args.fixture_dir.join("chat_template.json");

    fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("mkdir {}: {e}", args.output_dir.display()))?;

    let token_ids_out = args.output_dir.join("token_ids.json");
    let decoded_out = args.output_dir.join("decoded.json");
    let chat_out = args.output_dir.join("chat_template.json");

    // --- Load tokenizer & corpus ----------------------------------------
    let tok = load_tokenizer(&tok_path)
        .map_err(|e| format!("load_tokenizer({}): {e:?}", tok_path.display()))?;
    let strings = read_strings(&strings_path)?;
    eprintln!(
        "family={} strings={} tokenizer={}",
        args.family,
        strings.len(),
        tok_path.display()
    );

    // --- Encode (with + without special tokens) -------------------------
    let mut encode_with_special: Vec<Vec<u32>> = Vec::with_capacity(strings.len());
    let mut encode_no_special: Vec<Vec<u32>> = Vec::with_capacity(strings.len());
    for s in &strings {
        let ids_ws = encode(&tok, s, true).map_err(|e| {
            format!(
                "encode(true) failed on {:?}: {e:?}",
                &s.chars().take(40).collect::<String>()
            )
        })?;
        let ids_ns = encode(&tok, s, false).map_err(|e| {
            format!(
                "encode(false) failed on {:?}: {e:?}",
                &s.chars().take(40).collect::<String>()
            )
        })?;
        encode_with_special.push(ids_ws);
        encode_no_special.push(ids_ns);
    }

    // --- Decode round-trips ---------------------------------------------
    let mut decode_with_special_keep: Vec<String> = Vec::with_capacity(strings.len());
    let mut decode_with_special_skip: Vec<String> = Vec::with_capacity(strings.len());
    let mut decode_no_special: Vec<String> = Vec::with_capacity(strings.len());
    for (ids_ws, ids_ns) in encode_with_special.iter().zip(encode_no_special.iter()) {
        decode_with_special_keep
            .push(decode(&tok, ids_ws, false).map_err(|e| format!("decode(skip=false): {e:?}"))?);
        decode_with_special_skip
            .push(decode(&tok, ids_ws, true).map_err(|e| format!("decode(skip=true): {e:?}"))?);
        decode_no_special
            .push(decode(&tok, ids_ns, false).map_err(|e| format!("decode(no-special): {e:?}"))?);
    }

    write_json_pretty(
        &token_ids_out,
        &TokenIdsDump {
            encode_with_special,
            encode_no_special,
        },
    )?;
    write_json_pretty(
        &decoded_out,
        &DecodedDump {
            decode_with_special_keep,
            decode_with_special_skip,
            decode_no_special,
        },
    )?;

    // --- Chat template (only if upstream defines one) -------------------
    let ref_chat = read_chat_block(&chat_in_path)?;
    let chat_dump = if ref_chat.has_chat_template {
        let template = ref_chat
            .template
            .as_ref()
            .ok_or("chat_template.json says has_chat_template=true but template is missing")?;
        let messages: Vec<ChatMessage> = ref_chat
            .messages
            .iter()
            .map(|m| ChatMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        let bos = ref_chat.bos_token.as_deref();
        let eos = ref_chat.eos_token.as_deref();

        let rendered_no_gen = apply_chat_template(template, &messages, false, bos, eos)
            .map_err(|e| format!("apply_chat_template(no_gen): {e:?}"))?;
        let rendered_with_gen = apply_chat_template(template, &messages, true, bos, eos)
            .map_err(|e| format!("apply_chat_template(with_gen): {e:?}"))?;
        eprintln!(
            "  chat: len(no_gen)={} len(with_gen)={}",
            rendered_no_gen.len(),
            rendered_with_gen.len()
        );

        // Bind the references to silence "unused" once we stop comparing here —
        // the verifier consumes the python reference directly.
        let _ = (
            &ref_chat.rendered_no_generation_prompt,
            &ref_chat.rendered_with_generation_prompt,
        );

        ChatDump {
            has_chat_template: true,
            rendered_no_generation_prompt: Some(rendered_no_gen),
            rendered_with_generation_prompt: Some(rendered_with_gen),
        }
    } else {
        ChatDump {
            has_chat_template: false,
            rendered_no_generation_prompt: None,
            rendered_with_generation_prompt: None,
        }
    };
    write_json_pretty(&chat_out, &chat_dump)?;

    // Single-line JSON verdict the python verifier can latch onto, in the
    // same shape used by other ferrotorch parity dumps.
    println!(
        "{{\"family\":\"{}\",\"n_strings\":{},\"has_chat_template\":{}}}",
        args.family,
        strings.len(),
        chat_dump.has_chat_template
    );
    Ok(())
}
