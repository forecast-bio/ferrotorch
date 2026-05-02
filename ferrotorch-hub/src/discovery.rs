//! Dynamic model discovery via the HuggingFace Hub API.
//!
//! The static registry in [`crate::registry`] is compiled into the
//! binary and only contains models ferrotorch has explicitly curated.
//! This module provides runtime discovery against the live HuggingFace
//! Hub so users can search, browse, and look up models that are not
//! in the static registry.
//!
//! All functions in this module require the `http` feature (enabled by
//! default) and make blocking HTTPS calls via `ureq`. They are
//! deliberately NOT called from any hot path — use them at model-load
//! time or from a CLI tool.
//!
//! # API reference
//!
//! The HuggingFace Hub API is documented at
//! <https://huggingface.co/docs/hub/api>. The relevant endpoints are:
//! - `GET /api/models` — list / search models
//! - `GET /api/models/{repo_id}` — detailed info for a single model
//!
//! # Example (conceptual; actual network calls not exercised by tests)
//!
//! ```rust,no_run
//! use ferrotorch_hub::discovery::{search_models, get_model, SearchQuery};
//!
//! // Search for ResNet-family image classifiers
//! let query = SearchQuery::new()
//!     .with_search("resnet")
//!     .with_limit(10);
//! let results = search_models(&query).unwrap();
//! for m in &results {
//!     println!("{}: {} downloads", m.model_id, m.downloads.unwrap_or(0));
//! }
//!
//! // Look up a specific model by id
//! let info = get_model("microsoft/resnet-50").unwrap();
//! println!("tags: {:?}", info.tags);
//! ```
//!
//! CL-383.

#[cfg(feature = "http")]
use ferrotorch_core::{FerrotorchError, FerrotorchResult};
#[cfg(feature = "http")]
use serde::{Deserialize, Serialize};

/// A summary of a model returned by the HuggingFace Hub search API.
///
/// The Hub API has many more fields per model; we deserialize only the
/// ones that are useful here. Unknown fields are ignored.
#[cfg(feature = "http")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HfModelSummary {
    /// The canonical model id (e.g. `"microsoft/resnet-50"`). On the
    /// Hub API this field is called `modelId` in some endpoints and
    /// `id` in others — we accept both via `#[serde(alias)]`.
    #[serde(rename = "modelId", alias = "id")]
    pub model_id: String,
    /// The model's owner namespace (org or user) — derived from
    /// `model_id` before the `/`, or `None` when there is no `/`.
    #[serde(default)]
    pub author: Option<String>,
    /// Download count in the last 30 days, when the Hub reports it.
    #[serde(default)]
    pub downloads: Option<u64>,
    /// Like count, when the Hub reports it.
    #[serde(default)]
    pub likes: Option<u64>,
    /// Tags attached to the model (e.g. `"image-classification"`,
    /// `"pytorch"`, `"safetensors"`).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Primary ML framework/library reported by the Hub
    /// (`"pytorch"`, `"transformers"`, `"safetensors"`, etc.).
    #[serde(default)]
    pub library_name: Option<String>,
    /// Pipeline tag (a coarse task label like `"image-classification"`
    /// or `"text-generation"`).
    #[serde(default)]
    pub pipeline_tag: Option<String>,
}

/// Detailed info for a single model, returned by
/// `GET /api/models/{repo_id}`. Superset of [`HfModelSummary`] with
/// per-file metadata.
#[cfg(feature = "http")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelInfo {
    #[serde(rename = "modelId", alias = "id")]
    pub model_id: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: Option<u64>,
    #[serde(default)]
    pub likes: Option<u64>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub library_name: Option<String>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    /// List of files in the model repo (names only, no contents).
    #[serde(default)]
    pub siblings: Vec<HfRepoFile>,
}

/// A single file entry in a HuggingFace model repo.
#[cfg(feature = "http")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfRepoFile {
    /// The file path within the repo (e.g. `"model.safetensors"`,
    /// `"config.json"`).
    pub rfilename: String,
}

/// Parameters for a model search query against the HuggingFace Hub.
///
/// All fields are optional — an empty query returns the Hub's default
/// listing (typically the most-downloaded models).
#[cfg(feature = "http")]
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// Free-text search term (matched against model id, description, etc.).
    pub search: Option<String>,
    /// Filter to a single pipeline tag (e.g. `"image-classification"`).
    pub pipeline_tag: Option<String>,
    /// Filter to models belonging to a specific library
    /// (e.g. `"transformers"`, `"diffusers"`).
    pub library: Option<String>,
    /// Maximum number of results to return. Default is whatever the
    /// Hub returns (usually 10–50 depending on endpoint).
    pub limit: Option<usize>,
    /// Sort order: `"downloads"` (most-downloaded first), `"likes"`,
    /// `"updated"`, `"lastModified"`.
    pub sort: Option<String>,
}

#[cfg(feature = "http")]
impl SearchQuery {
    /// Create an empty query (no filters).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the free-text search term.
    pub fn with_search(mut self, search: impl Into<String>) -> Self {
        self.search = Some(search.into());
        self
    }

    /// Filter to a specific pipeline tag (task label).
    pub fn with_pipeline_tag(mut self, tag: impl Into<String>) -> Self {
        self.pipeline_tag = Some(tag.into());
        self
    }

    /// Filter to a specific library.
    pub fn with_library(mut self, library: impl Into<String>) -> Self {
        self.library = Some(library.into());
        self
    }

    /// Limit the number of results returned.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the sort key. Must be one of `"downloads"`, `"likes"`,
    /// `"updated"`, or `"lastModified"`. Unknown keys are passed
    /// through to the Hub API, which will return its default order.
    pub fn with_sort(mut self, sort: impl Into<String>) -> Self {
        self.sort = Some(sort.into());
        self
    }

    /// Build the fully-encoded query string for the Hub API.
    ///
    /// Each field becomes a separate `?key=value` pair, URL-encoded
    /// via the tiny helper below (no extra dep). Returns the full
    /// path+query (without the scheme/host) so it can be appended to
    /// the base URL.
    pub(crate) fn to_query_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(s) = &self.search {
            parts.push(format!("search={}", url_encode(s)));
        }
        if let Some(t) = &self.pipeline_tag {
            parts.push(format!("pipeline_tag={}", url_encode(t)));
        }
        if let Some(lib) = &self.library {
            parts.push(format!("library={}", url_encode(lib)));
        }
        if let Some(l) = self.limit {
            parts.push(format!("limit={l}"));
        }
        if let Some(s) = &self.sort {
            parts.push(format!("sort={}", url_encode(s)));
        }
        if parts.is_empty() {
            "/api/models".to_string()
        } else {
            format!("/api/models?{}", parts.join("&"))
        }
    }
}

/// URL-encode a string for use as a query-parameter value.
///
/// Handles the small subset of characters the Hub API query string
/// values actually use (alphanumerics, `-`, `_`, `.`, `~` pass
/// through; everything else becomes `%XX`). This is a conservative
/// subset of RFC 3986 unreserved characters.
#[cfg(feature = "http")]
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

/// Search for models on the HuggingFace Hub. Returns a list of
/// summaries ordered by the Hub's default (or the sort key in the
/// query, if set).
///
/// The Hub endpoint is `GET https://huggingface.co/api/models` with
/// query parameters from [`SearchQuery::to_query_string`].
#[cfg(feature = "http")]
pub fn search_models(query: &SearchQuery) -> FerrotorchResult<Vec<HfModelSummary>> {
    const HUB_BASE: &str = "https://huggingface.co";
    let url = format!("{HUB_BASE}{}", query.to_query_string());
    let response = crate::auth::with_auth(ureq::get(&url))
        .call()
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ferrotorch-hub: HuggingFace Hub search failed ({url}): {e}"),
        })?;
    let summaries: Vec<HfModelSummary> =
        response
            .into_json()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("ferrotorch-hub: failed to parse Hub search response: {e}"),
            })?;
    Ok(populate_authors(summaries))
}

/// Fetch detailed info for a single model by its repo id (e.g.
/// `"microsoft/resnet-50"`).
///
/// Returns the full [`HfModelInfo`] including the file list. The Hub
/// endpoint is `GET https://huggingface.co/api/models/{repo_id}`.
#[cfg(feature = "http")]
pub fn get_model(repo_id: &str) -> FerrotorchResult<HfModelInfo> {
    if repo_id.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "get_model: repo_id must not be empty".into(),
        });
    }
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let response = crate::auth::with_auth(ureq::get(&url))
        .call()
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ferrotorch-hub: Hub model lookup failed ({url}): {e}"),
        })?;
    let mut info: HfModelInfo =
        response
            .into_json()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("ferrotorch-hub: failed to parse Hub model response: {e}"),
            })?;
    if info.author.is_none() {
        info.author = extract_author(&info.model_id);
    }
    Ok(info)
}

/// Populate the `author` field on each summary by parsing the leading
/// segment of the model id, for entries that did not have it set by
/// the Hub.
#[cfg(feature = "http")]
fn populate_authors(mut summaries: Vec<HfModelSummary>) -> Vec<HfModelSummary> {
    for s in &mut summaries {
        if s.author.is_none() {
            s.author = extract_author(&s.model_id);
        }
    }
    summaries
}

/// Extract the author (namespace) from a `namespace/model` id.
/// Returns `None` for top-level models that have no namespace.
#[cfg(feature = "http")]
fn extract_author(model_id: &str) -> Option<String> {
    model_id.split_once('/').map(|(ns, _)| ns.to_string())
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;

    #[test]
    fn test_search_query_empty_is_bare_endpoint() {
        let q = SearchQuery::new();
        assert_eq!(q.to_query_string(), "/api/models");
    }

    #[test]
    fn test_search_query_search_only() {
        let q = SearchQuery::new().with_search("resnet");
        assert_eq!(q.to_query_string(), "/api/models?search=resnet");
    }

    #[test]
    fn test_search_query_all_fields() {
        let q = SearchQuery::new()
            .with_search("resnet")
            .with_pipeline_tag("image-classification")
            .with_library("pytorch")
            .with_limit(25)
            .with_sort("downloads");
        let qs = q.to_query_string();
        assert!(qs.starts_with("/api/models?"));
        assert!(qs.contains("search=resnet"));
        assert!(qs.contains("pipeline_tag=image-classification"));
        assert!(qs.contains("library=pytorch"));
        assert!(qs.contains("limit=25"));
        assert!(qs.contains("sort=downloads"));
    }

    #[test]
    fn test_url_encode_alphanumeric_passthrough() {
        assert_eq!(url_encode("resnet50"), "resnet50");
        assert_eq!(url_encode("my-model.v1_beta"), "my-model.v1_beta");
    }

    #[test]
    fn test_url_encode_special_chars() {
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("a/b"), "a%2Fb");
        assert_eq!(url_encode("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn test_extract_author_namespaced() {
        assert_eq!(
            extract_author("microsoft/resnet-50"),
            Some("microsoft".to_string())
        );
        assert_eq!(
            extract_author("meta-llama/Llama-2-7b"),
            Some("meta-llama".to_string())
        );
    }

    #[test]
    fn test_extract_author_top_level() {
        assert_eq!(extract_author("bert-base-uncased"), None);
        assert_eq!(extract_author(""), None);
    }

    #[test]
    fn test_deserialize_model_summary_minimal() {
        // The Hub sometimes returns just {modelId: ...}. Everything
        // else should default.
        let json = r#"{"modelId": "microsoft/resnet-50"}"#;
        let m: HfModelSummary = serde_json::from_str(json).unwrap();
        assert_eq!(m.model_id, "microsoft/resnet-50");
        assert_eq!(m.downloads, None);
        assert_eq!(m.tags.len(), 0);
    }

    #[test]
    fn test_deserialize_model_summary_id_alias() {
        // Some endpoints return `id` instead of `modelId`.
        let json = r#"{"id": "bert-base-uncased"}"#;
        let m: HfModelSummary = serde_json::from_str(json).unwrap();
        assert_eq!(m.model_id, "bert-base-uncased");
    }

    #[test]
    fn test_deserialize_model_summary_full() {
        let json = r#"{
            "modelId": "microsoft/resnet-50",
            "downloads": 1234567,
            "likes": 42,
            "tags": ["image-classification", "pytorch", "safetensors"],
            "library_name": "transformers",
            "pipeline_tag": "image-classification"
        }"#;
        let m: HfModelSummary = serde_json::from_str(json).unwrap();
        assert_eq!(m.model_id, "microsoft/resnet-50");
        assert_eq!(m.downloads, Some(1234567));
        assert_eq!(m.likes, Some(42));
        assert_eq!(m.tags.len(), 3);
        assert!(m.tags.contains(&"pytorch".to_string()));
        assert_eq!(m.library_name, Some("transformers".to_string()));
        assert_eq!(m.pipeline_tag, Some("image-classification".to_string()));
    }

    #[test]
    fn test_deserialize_model_summary_unknown_fields_ignored() {
        // The Hub keeps adding fields; we must not fail on fields we
        // don't know about.
        let json = r#"{
            "modelId": "microsoft/resnet-50",
            "private": false,
            "sha": "abc123def456",
            "lastModified": "2024-01-01T00:00:00.000Z",
            "gated": false
        }"#;
        let m: HfModelSummary = serde_json::from_str(json).unwrap();
        assert_eq!(m.model_id, "microsoft/resnet-50");
    }

    #[test]
    fn test_deserialize_model_info_with_siblings() {
        let json = r#"{
            "modelId": "microsoft/resnet-50",
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "model.safetensors"},
                {"rfilename": "README.md"}
            ]
        }"#;
        let info: HfModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.model_id, "microsoft/resnet-50");
        assert_eq!(info.siblings.len(), 3);
        assert_eq!(info.siblings[0].rfilename, "config.json");
        assert_eq!(info.siblings[1].rfilename, "model.safetensors");
    }

    #[test]
    fn test_populate_authors_fills_missing() {
        let summaries = vec![
            HfModelSummary {
                model_id: "microsoft/resnet-50".into(),
                author: None,
                downloads: None,
                likes: None,
                tags: vec![],
                library_name: None,
                pipeline_tag: None,
            },
            HfModelSummary {
                model_id: "bert-base-uncased".into(),
                author: Some("already-set".into()),
                downloads: None,
                likes: None,
                tags: vec![],
                library_name: None,
                pipeline_tag: None,
            },
        ];
        let out = populate_authors(summaries);
        assert_eq!(out[0].author, Some("microsoft".to_string()));
        // Existing value is preserved.
        assert_eq!(out[1].author, Some("already-set".to_string()));
    }

    #[test]
    fn test_get_model_empty_repo_id_errors() {
        let result = get_model("");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("must not be empty"));
    }
}
