//! Text tokenization for ferrotorch models.
//!
//! This crate is a thin wrapper around the HuggingFace
//! [`tokenizers`] crate — the same library powering Python's
//! `transformers.AutoTokenizer` — with an API shaped for ferrotorch
//! idioms (`Vec<u32>` token ids, `FerrotorchResult` errors).
//!
//! # Quick start
//!
//! ```no_run
//! use ferrotorch_tokenize::{load_tokenizer, encode, decode};
//!
//! // Llama 3 ships a `tokenizer.json` alongside its weights.
//! let tok = load_tokenizer("/path/to/tokenizer.json")?;
//! let ids = encode(&tok, "Hello, world!", /* add_special_tokens = */ true)?;
//! let text = decode(&tok, &ids, /* skip_special_tokens = */ false)?;
//! # Ok::<(), ferrotorch_core::FerrotorchError>(())
//! ```
//!
//! # Scope
//!
//! The wrapper covers the path the Llama 3 8B PoC needs:
//! - Load a `tokenizer.json` file into a [`Tokenizer`].
//! - Encode / decode single strings and batches.
//! - Query vocab size and special-token ids.
//!
//! More advanced features (chat templates, truncation strategies,
//! added-token manipulation) are available by calling the re-exported
//! [`tokenizers`] API directly on the returned [`Tokenizer`].

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

pub use tokenizers::Tokenizer;

/// Load a tokenizer from a HuggingFace `tokenizer.json` file.
///
/// This accepts any format that `tokenizers::Tokenizer::from_file`
/// supports — which is the full HF tokenizer format including BPE,
/// WordPiece, Unigram, pre/post processors, and added tokens.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the file does not exist,
/// cannot be read, or is not a valid HuggingFace tokenizer JSON (parse
/// errors, unknown model types, etc.).
pub fn load_tokenizer(path: impl AsRef<Path>) -> FerrotorchResult<Tokenizer> {
    let path = path.as_ref();
    Tokenizer::from_file(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to load tokenizer {}: {e}", path.display()),
    })
}

/// Encode a single text into its token ids.
///
/// `add_special_tokens` controls whether BOS / EOS and other
/// template-defined special tokens are inserted (Llama 3 prepends
/// `<|begin_of_text|>` / `128000` when true).
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the tokenizer's underlying
/// model or post-processor rejects the input.
pub fn encode(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> FerrotorchResult<Vec<u32>> {
    let encoding = tokenizer.encode(text, add_special_tokens).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("tokenizer encode failed: {e}"),
        }
    })?;
    Ok(encoding.get_ids().to_vec())
}

/// Encode a batch of texts in parallel.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the underlying tokenizer
/// rejects any text in the batch.
pub fn encode_batch(
    tokenizer: &Tokenizer,
    texts: &[&str],
    add_special_tokens: bool,
) -> FerrotorchResult<Vec<Vec<u32>>> {
    // `tokenizers::Tokenizer::encode_batch` accepts any `E: Into<EncodeInput>`.
    // `&str` satisfies that chain via `InputSequence`, so we pass the slice
    // entries directly and avoid a `Vec<String>` intermediate allocation.
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), add_special_tokens)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("tokenizer encode_batch failed: {e}"),
        })?;
    Ok(encodings
        .into_iter()
        .map(|e| e.get_ids().to_vec())
        .collect())
}

/// Decode a sequence of token ids back to text.
///
/// `skip_special_tokens` drops BOS / EOS / pad tokens from the output.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the tokenizer's decoder
/// rejects the id sequence (e.g. out-of-range ids on some tokenizer types).
pub fn decode(
    tokenizer: &Tokenizer,
    ids: &[u32],
    skip_special_tokens: bool,
) -> FerrotorchResult<String> {
    tokenizer
        .decode(ids, skip_special_tokens)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("tokenizer decode failed: {e}"),
        })
}

/// Vocabulary size the tokenizer was trained with, including any
/// special / added tokens (Llama 3: 128_256).
pub fn vocab_size(tokenizer: &Tokenizer, with_added_tokens: bool) -> usize {
    tokenizer.get_vocab_size(with_added_tokens)
}

/// Resolve a token string to its numeric id, if present in the vocab
/// (including added/special tokens).
pub fn token_to_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}

/// Resolve a token id to its string form.
pub fn id_to_token(tokenizer: &Tokenizer, id: u32) -> Option<String> {
    tokenizer.id_to_token(id)
}

// ===========================================================================
// Chat-template rendering (#588)
// ===========================================================================

/// One message in a chat-completion conversation.
///
/// Mirrors the OpenAI / HuggingFace `messages` list structure that LLM
/// chat templates expect: `{ role, content }` plus optional structured
/// fields the template may reference (`name`, `tool_calls`, `tool_call_id`).
/// We model the optional fields as a free-form `serde_json::Value` map so
/// the renderer can pass any extra keys straight to Jinja.
///
/// Fields are intentionally `pub` to allow direct construction from parsed
/// JSON or deserialized data. The struct is `#[non_exhaustive]` so that
/// adding new fields (e.g. `tool_call_id`) is a non-breaking change;
/// use [`ChatMessage::new`] as the canonical constructor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ChatMessage {
    /// Conventional roles: `"system"`, `"user"`, `"assistant"`, `"tool"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
    /// Extra fields propagated to the Jinja template (e.g. `name`,
    /// `tool_calls`). Any JSON value is allowed.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

impl ChatMessage {
    /// Convenience constructor for the common `{ role, content }` case.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            extra: std::collections::BTreeMap::new(),
        }
    }
}

/// Apply a HuggingFace-style Jinja2 chat template to a list of messages.
///
/// Renders to the same string that Python's `tokenizer.apply_chat_template`
/// would produce. The template usually lives in `tokenizer_config.json`
/// under the key `chat_template` and references:
/// - `messages` — the list of `ChatMessage` records (passed through here)
/// - `add_generation_prompt` — bool, whether to append the assistant turn header
/// - `bos_token`, `eos_token` — special tokens (passed via the eponymous args)
///
/// Pass `bos_token` / `eos_token` as `None` if the template doesn't reference
/// them; otherwise pass the literal token text the template expects (e.g.
/// `"<|begin_of_text|>"` for Llama 3).
///
/// Use [`apply_chat_template_to_ids`] when you also want to tokenize the
/// rendered string in one call.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the template string is not
/// valid Jinja2, if a template variable is missing, or if the template calls
/// `raise_exception(msg)` (propagated as an error with `msg` in the message).
pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> FerrotorchResult<String> {
    let mut env = minijinja::Environment::new();
    // `raise_exception` is referenced by some HF templates (Mistral et al.).
    // Implement it as a no-arg helper that just panics with a message.
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    // `strftime_now` is referenced by some templates (Llama 3.1 system prompt).
    // Provide a stub that returns the empty string — callers wanting real
    // dates should preprocess the template.
    env.add_function(
        "strftime_now",
        |_fmt: String| -> Result<String, minijinja::Error> { Ok(String::new()) },
    );

    env.add_template("chat", template)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("invalid chat template: {e}"),
        })?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("chat template lookup failed: {e}"),
        })?;

    let context = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token.unwrap_or(""),
        eos_token => eos_token.unwrap_or(""),
    };
    tmpl.render(context)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("chat template render failed: {e}"),
        })
}

/// Apply a chat template and tokenize the result.
///
/// `add_special_tokens` is forwarded to [`encode`] — usually `false` here
/// because chat templates already embed the BOS / role headers literally.
///
/// Returns the rendered string and the encoded token ids so the caller can
/// log / inspect the prompt.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the template fails to
/// render (see [`apply_chat_template`]) or if the rendered string fails to
/// encode (see [`encode`]).
pub fn apply_chat_template_to_ids(
    tokenizer: &Tokenizer,
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    add_special_tokens: bool,
) -> FerrotorchResult<(String, Vec<u32>)> {
    let prompt = apply_chat_template(
        template,
        messages,
        add_generation_prompt,
        bos_token,
        eos_token,
    )?;
    let ids = encode(tokenizer, &prompt, add_special_tokens)?;
    Ok((prompt, ids))
}

/// Read the `chat_template` field out of a `tokenizer_config.json` file.
///
/// Returns `None` if the file exists but doesn't define `chat_template`,
/// or an error if the file can't be read or parsed. Some configs ship the
/// template as `chat_template: [{name, template}]` (multiple templates,
/// keyed by name); this loader returns the first one in that case to
/// match what `transformers.AutoTokenizer` does by default.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the file cannot be read
/// (not found, permission denied, I/O error), if the content is not valid
/// JSON, or if the `chat_template` field exists but is neither a string nor
/// an array of `{name, template}` objects.
pub fn load_chat_template(
    tokenizer_config_path: impl AsRef<Path>,
) -> FerrotorchResult<Option<String>> {
    let path = tokenizer_config_path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read tokenizer_config {}: {e}", path.display()),
    })?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse tokenizer_config {}: {e}", path.display()),
        })?;
    match value.get("chat_template") {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(serde_json::Value::Array(arr)) => {
            // Multi-template form: pick the first entry's `template` field.
            for entry in arr {
                if let Some(t) = entry.get("template").and_then(|v| v.as_str()) {
                    return Ok(Some(t.to_string()));
                }
            }
            Ok(None)
        }
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tokenizer_config {} has chat_template of unexpected type",
                path.display()
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the HF cache directory for a gated model.
    fn hf_cache_snapshot(repo_slug: &str) -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from)?;
        let base = home
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", repo_slug.replace('/', "--")))
            .join("snapshots");
        std::fs::read_dir(&base)
            .ok()?
            .next()?
            .ok()
            .map(|e| e.path())
    }

    #[test]
    fn loader_rejects_missing_file() {
        let r = load_tokenizer("/nonexistent/tokenizer.json");
        assert!(r.is_err());
    }

    #[test]
    fn loader_rejects_malformed_json() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_malformed.json");
        std::fs::write(&tmp, "{ not valid").unwrap();
        let r = load_tokenizer(&tmp);
        assert!(r.is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    /// End-to-end: load the real Llama 3 tokenizer.json from the HF
    /// cache and verify the basic surface works.
    /// Ignored by default so CI without the gated model skips it.
    #[test]
    #[ignore = "requires Meta-Llama-3-8B tokenizer.json in the HF cache"]
    fn llama3_tokenizer_loads_and_round_trips() {
        let snapshot = hf_cache_snapshot("meta-llama/Meta-Llama-3-8B")
            .expect("Meta-Llama-3-8B snapshot missing from HF cache");
        let tok_path = snapshot.join("tokenizer.json");
        let tok = load_tokenizer(&tok_path).unwrap();

        // Llama 3 vocab.
        assert_eq!(vocab_size(&tok, true), 128_256);

        // Special tokens the Llama 3 chat template uses.
        assert_eq!(token_to_id(&tok, "<|begin_of_text|>"), Some(128_000));
        assert_eq!(token_to_id(&tok, "<|end_of_text|>"), Some(128_001));

        // Encode with special tokens — BOS should prepend 128000.
        let ids = encode(&tok, "Hello, world!", true).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids[0], 128_000, "BOS not prepended: {ids:?}");

        // Without add_special_tokens, BOS is not prepended.
        let ids_bare = encode(&tok, "Hello, world!", false).unwrap();
        assert_ne!(ids_bare[0], 128_000);

        // Round-trip via decode.
        let text = decode(&tok, &ids_bare, false).unwrap();
        assert!(
            text.contains("Hello") && text.contains("world"),
            "decoded text unexpected: {text:?}"
        );

        // encode_batch returns one vec per input.
        let batch = encode_batch(&tok, &["hi", "bye"], false).unwrap();
        assert_eq!(batch.len(), 2);
        assert!(!batch[0].is_empty());
        assert!(!batch[1].is_empty());
    }

    // -----------------------------------------------------------------------
    // Chat-template rendering (#588)
    // -----------------------------------------------------------------------

    /// A simplified Llama-3-style template for testing — captures the same
    /// shape (per-message header + EOT, optional generation prompt) without
    /// the full template's quirks.
    const SIMPLE_LLAMA3_LIKE: &str = "{% for m in messages %}\
<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content | trim }}<|eot_id|>\
{% endfor %}\
{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";

    #[test]
    fn chat_template_renders_simple_two_turn() {
        let messages = vec![
            ChatMessage::new("user", "hi"),
            ChatMessage::new("assistant", "hello there"),
        ];
        let s = apply_chat_template(SIMPLE_LLAMA3_LIKE, &messages, false, None, None).unwrap();
        assert!(s.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(s.contains("hi<|eot_id|>"));
        assert!(s.contains("<|start_header_id|>assistant<|end_header_id|>"));
        assert!(s.contains("hello there<|eot_id|>"));
        // No generation prompt requested → string ends after the last EOT.
        assert!(s.ends_with("<|eot_id|>"));
    }

    #[test]
    fn chat_template_appends_generation_prompt_when_requested() {
        let messages = vec![ChatMessage::new("user", "hi")];
        let s = apply_chat_template(SIMPLE_LLAMA3_LIKE, &messages, true, None, None).unwrap();
        assert!(s.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn chat_template_trims_whitespace_in_content() {
        let messages = vec![ChatMessage::new("user", "   hi   ")];
        let s = apply_chat_template(SIMPLE_LLAMA3_LIKE, &messages, false, None, None).unwrap();
        // The | trim filter strips outer whitespace.
        assert!(s.contains("hi<|eot_id|>"));
        assert!(!s.contains("   hi"));
    }

    #[test]
    fn chat_template_passes_bos_token() {
        let template = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}";
        let messages = vec![ChatMessage::new("user", "hi")];
        let s = apply_chat_template(template, &messages, false, Some("<|begin_of_text|>"), None)
            .unwrap();
        assert_eq!(s, "<|begin_of_text|>hi");
    }

    #[test]
    fn chat_template_propagates_extra_fields() {
        let template = "{% for m in messages %}{{ m.role }}:{{ m.name }}{% endfor %}";
        let mut msg = ChatMessage::new("tool", "result");
        msg.extra.insert(
            "name".to_string(),
            serde_json::Value::String("my_tool".to_string()),
        );
        let s = apply_chat_template(template, &[msg], false, None, None).unwrap();
        assert_eq!(s, "tool:my_tool");
    }

    #[test]
    fn chat_template_rejects_invalid_template() {
        let messages = vec![ChatMessage::new("user", "hi")];
        // Unclosed braces.
        let s = apply_chat_template(
            "{% for m in messages %}{{ m.role",
            &messages,
            false,
            None,
            None,
        );
        assert!(s.is_err());
    }

    #[test]
    fn chat_template_raise_exception_function_propagates_error() {
        let template = "{% if messages | length == 0 %}\
{{ raise_exception(\"no messages\") }}\
{% else %}{{ messages[0].content }}{% endif %}";
        let empty: Vec<ChatMessage> = Vec::new();
        let err = apply_chat_template(template, &empty, false, None, None).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no messages") || msg.contains("invalid operation"),
            "expected raise_exception to surface: got {msg:?}"
        );
    }

    #[test]
    fn load_chat_template_extracts_string_field() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_chat_cfg.json");
        let body = serde_json::json!({
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
        });
        std::fs::write(&tmp, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        let t = load_chat_template(&tmp).unwrap().unwrap();
        assert!(t.contains("messages"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_chat_template_handles_array_form() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_chat_cfg_arr.json");
        let body = serde_json::json!({
            "chat_template": [
                {"name": "default", "template": "ARRAY_TEMPLATE"},
                {"name": "tool_use", "template": "OTHER"},
            ]
        });
        std::fs::write(&tmp, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        let t = load_chat_template(&tmp).unwrap().unwrap();
        assert_eq!(t, "ARRAY_TEMPLATE");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_chat_template_returns_none_when_missing() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_chat_cfg_no_field.json");
        std::fs::write(&tmp, serde_json::json!({"foo": "bar"}).to_string()).unwrap();
        let t = load_chat_template(&tmp).unwrap();
        assert!(t.is_none());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn chat_message_roundtrip_through_serde() {
        let mut msg = ChatMessage::new("assistant", "hello");
        msg.extra
            .insert("name".to_string(), serde_json::json!("alice"));
        let s = serde_json::to_string(&msg).unwrap();
        // Both top-level fields and the flattened extras must show up.
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"content\":\"hello\""));
        assert!(s.contains("\"name\":\"alice\""));
    }
}
