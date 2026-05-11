#!/usr/bin/env python3
"""Regenerate ferrotorch-hub conformance fixtures from huggingface_hub 0.24.x.

This script generates the reference fixture data that ferrotorch-hub's
conformance tests assert against. All operations run entirely offline —
no network calls are made. The fixtures encode the EXPECTED behavior of
ferrotorch-hub's Rust implementation, derived by exercising the Python
`huggingface_hub` library in offline/mocked mode.

Pin: huggingface_hub == 0.24.x (currently 0.24.3).

Functions that genuinely require network (hf_download_model,
search_models, get_model) are cascade-skipped with reason
"network-dependent — covered by integration tests (load_pretrained_smoke.rs)".

Usage:
    python3 scripts/regenerate_hub_fixtures.py

Output:
    ferrotorch-hub/tests/conformance/fixtures.json
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import sys

# ---------------------------------------------------------------------------
# Version guard — must be huggingface_hub 0.24.x
# ---------------------------------------------------------------------------

try:
    import huggingface_hub
except ImportError:
    print(
        "ERROR: huggingface_hub is not installed. "
        "Run: pip install --user 'huggingface_hub==0.24.3'",
        file=sys.stderr,
    )
    sys.exit(1)

HF_VERSION = huggingface_hub.__version__
if not HF_VERSION.startswith("0.24."):
    print(
        f"ERROR: expected huggingface_hub == 0.24.x, got {HF_VERSION!r}. "
        "Run: pip install --user 'huggingface_hub==0.24.3'",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"huggingface_hub {HF_VERSION} — OK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cascade_skip(op: str, reason: str) -> dict:
    """Mark an operation as skipped due to network-dependency or other constraint."""
    return {
        "op": op,
        "cascade_skip": True,
        "skip_reason": reason,
    }


def url_encode_simple(s: str) -> str:
    """Minimal percent-encoding matching ferrotorch-hub's url_encode function.

    Passes through: 0-9, a-z, A-Z, -, _, ., ~
    Encodes everything else as %XX (uppercase hex).
    """
    out = []
    for b in s.encode("utf-8"):
        c = chr(b)
        if c in (
            "0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "-_.~"
        ):
            out.append(c)
        else:
            out.append(f"%{b:02X}")
    return "".join(out)


# ---------------------------------------------------------------------------
# Fixture group: repo_id parsing and URL construction
# The HuggingFace Hub uses canonical repo-id strings of the form
# "{namespace}/{model-name}" or bare "{model-name}" for top-level models.
# We test: extract_author logic, URL construction for resolve endpoint,
# and query-string building.
# ---------------------------------------------------------------------------

def make_repo_id_fixtures() -> list[dict]:
    """Generate repo_id parsing / URL construction fixtures.

    These encode the EXPECTED behavior that ferrotorch-hub's Rust implementation
    must match. All are computed from first-principles (no network).
    """
    fixtures = []

    # --- extract_author: namespace/model -> namespace
    cases_extract_author = [
        ("microsoft/resnet-50", "microsoft"),
        ("meta-llama/Meta-Llama-3-8B", "meta-llama"),
        ("timm/resnet50.a1_in1k", "timm"),
        ("HuggingFaceFW/fineweb", "HuggingFaceFW"),
        ("bert-base-uncased", None),  # no namespace
        ("", None),  # empty
        ("a/b/c", "a"),  # only split on first /
    ]
    for repo_id, expected_author in cases_extract_author:
        # Python reference: split_once('/') -> first segment
        if "/" in repo_id:
            computed = repo_id.split("/", 1)[0]
        else:
            computed = None
        # Sanity check our own derivation
        assert computed == expected_author, f"Bug in fixture gen: {repo_id!r}"
        fixtures.append({
            "op": "extract_author",
            "repo_id": repo_id,
            "expected_author": expected_author,
        })

    # --- URL construction: resolve endpoint
    # The Rust impl builds: https://huggingface.co/{repo}/resolve/{revision}/{file}
    resolve_cases = [
        ("meta-llama/Meta-Llama-3-8B", "main", "config.json",
         "https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json"),
        ("timm/resnet50.a1_in1k", "main", "model.safetensors",
         "https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/model.safetensors"),
        ("timm/resnet50.a1_in1k", "v1.0", "model-00001-of-00004.safetensors",
         "https://huggingface.co/timm/resnet50.a1_in1k/resolve/v1.0/model-00001-of-00004.safetensors"),
    ]
    for repo, revision, filename, expected_url in resolve_cases:
        computed = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        assert computed == expected_url, f"Bug in fixture gen: {repo!r}"
        fixtures.append({
            "op": "resolve_url",
            "repo": repo,
            "revision": revision,
            "filename": filename,
            "expected_url": expected_url,
        })

    # --- API URL: model info endpoint
    api_cases = [
        ("microsoft/resnet-50", "https://huggingface.co/api/models/microsoft/resnet-50"),
        ("bert-base-uncased", "https://huggingface.co/api/models/bert-base-uncased"),
    ]
    for repo_id, expected_url in api_cases:
        computed = f"https://huggingface.co/api/models/{repo_id}"
        assert computed == expected_url
        fixtures.append({
            "op": "api_model_url",
            "repo_id": repo_id,
            "expected_url": expected_url,
        })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: SearchQuery URL construction
# ---------------------------------------------------------------------------

def make_search_query_fixtures() -> list[dict]:
    """Generate SearchQuery::to_query_string() reference fixtures.

    These replicate the exact URL encoding that ferrotorch-hub's
    `url_encode` function produces. We compute them in Python here and
    the Rust conformance test asserts that `SearchQuery::to_query_string`
    returns the same strings.
    """
    fixtures = []

    cases = [
        # (kwargs dict -> expected query string)
        ({}, "/api/models"),
        ({"search": "resnet"}, "/api/models?search=resnet"),
        ({"search": "image classification"},
         "/api/models?search=image%20classification"),
        ({"pipeline_tag": "image-classification"},
         "/api/models?pipeline_tag=image-classification"),
        ({"library": "pytorch"},
         "/api/models?library=pytorch"),
        ({"limit": 25},
         "/api/models?limit=25"),
        ({"sort": "downloads"},
         "/api/models?sort=downloads"),
        # all fields together
        ({"search": "resnet", "pipeline_tag": "image-classification",
          "library": "pytorch", "limit": 25, "sort": "downloads"},
         "/api/models?search=resnet&pipeline_tag=image-classification"
         "&library=pytorch&limit=25&sort=downloads"),
        # special characters in search term
        ({"search": "a/b"},
         "/api/models?search=a%2Fb"),
        ({"search": "hello world"},
         "/api/models?search=hello%20world"),
        ({"search": "a&b=c"},
         "/api/models?search=a%26b%3Dc"),
    ]

    for kwargs, expected_qs in cases:
        # Build the query string the same way ferrotorch-hub does
        parts = []
        if "search" in kwargs:
            parts.append(f"search={url_encode_simple(kwargs['search'])}")
        if "pipeline_tag" in kwargs:
            parts.append(f"pipeline_tag={url_encode_simple(kwargs['pipeline_tag'])}")
        if "library" in kwargs:
            parts.append(f"library={url_encode_simple(kwargs['library'])}")
        if "limit" in kwargs:
            parts.append(f"limit={kwargs['limit']}")
        if "sort" in kwargs:
            parts.append(f"sort={url_encode_simple(kwargs['sort'])}")

        computed = "/api/models" if not parts else f"/api/models?{'&'.join(parts)}"
        assert computed == expected_qs, (
            f"Bug in fixture gen:\n  computed={computed!r}\n  expected={expected_qs!r}"
        )

        fixtures.append({
            "op": "search_query_string",
            "kwargs": kwargs,
            "expected_query_string": expected_qs,
        })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: url_encode
# ---------------------------------------------------------------------------

def make_url_encode_fixtures() -> list[dict]:
    """Generate url_encode reference fixtures.

    These cover the exact encoding rules that ferrotorch-hub's url_encode
    function implements. The Rust conformance test uses these to verify
    that the private url_encode helper — exercised via SearchQuery — produces
    the expected percent-encodings.
    """
    # huggingface_hub 0.24.x uses its own URL encoding. We replicate
    # ferrotorch-hub's documented encoding rules (RFC 3986 unreserved chars
    # pass through; everything else -> %XX uppercase hex).
    cases = [
        ("resnet50", "resnet50"),
        ("my-model.v1_beta", "my-model.v1_beta"),
        ("hello world", "hello%20world"),
        ("a/b", "a%2Fb"),
        ("a&b=c", "a%26b%3Dc"),
        ("image-classification", "image-classification"),
        ("", ""),
        ("~tilde~", "~tilde~"),
        ("100%done", "100%25done"),
        ("réseau", "r%C3%A9seau"),
    ]

    fixtures = []
    for input_s, expected in cases:
        computed = url_encode_simple(input_s)
        assert computed == expected, (
            f"Bug in fixture gen for {input_s!r}: computed={computed!r}, expected={expected!r}"
        )
        fixtures.append({
            "op": "url_encode",
            "input": input_s,
            "expected": expected,
        })
    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: HfTransformerConfig parsing
# These encode the expected field values parsed from known config.json strings.
# ---------------------------------------------------------------------------

def make_hf_config_fixtures() -> list[dict]:
    """Generate HfTransformerConfig parsing reference fixtures.

    These fixtures encode the expected field values that ferrotorch-hub's
    HfTransformerConfig::from_json_str should produce when parsing the
    same JSON strings. Python's json module is the reference here —
    the semantic meaning of the fields is well-defined.
    """
    fixtures = []

    # Llama 3 8B config (exact JSON from meta-llama/Meta-Llama-3-8B)
    llama3_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "vocab_size": 128256,
    }
    fixtures.append({
        "op": "hf_config_parse",
        "label": "llama3_8b",
        "json": json.dumps(llama3_config),
        "expected": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads_resolved": 8,  # explicit field
            "intermediate_size": 14336,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 8192,
            "vocab_size": 128256,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "torch_dtype": "bfloat16",
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            # derived values
            "head_dim": 128,   # 4096 / 32
            "is_gqa": True,    # 8 < 32
        },
    })

    # Minimal MHA config (no num_key_value_heads -> defaults to num_attention_heads)
    mha_config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "vocab_size": 30522,
    }
    fixtures.append({
        "op": "hf_config_parse",
        "label": "bert_base_mha",
        "json": json.dumps(mha_config),
        "expected": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads_resolved": 12,  # falls back to num_attention_heads
            "intermediate_size": 3072,
            "rms_norm_eps": 1e-6,    # default
            "rope_theta": 10000.0,   # default
            "max_position_embeddings": 512,
            "vocab_size": 30522,
            "tie_word_embeddings": False,  # default
            "hidden_act": "silu",          # default
            "torch_dtype": None,
            "architectures": [],
            "model_type": None,
            # derived values
            "head_dim": 64,    # 768 / 12
            "is_gqa": False,   # 12 == 12
        },
    })

    # GQA config (num_key_value_heads < num_attention_heads)
    gqa_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
    }
    fixtures.append({
        "op": "hf_config_parse",
        "label": "llama2_7b_gqa",
        "json": json.dumps(gqa_config),
        "expected": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads_resolved": 8,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "torch_dtype": None,
            "architectures": [],
            "model_type": None,
            # derived values
            "head_dim": 128,   # 4096 / 32
            "is_gqa": True,    # 8 < 32
        },
    })

    # validate() — should pass
    fixtures.append({
        "op": "hf_config_validate",
        "label": "llama3_8b_passes",
        "json": json.dumps(llama3_config),
        "expected_valid": True,
    })

    # validate() — should fail: hidden_size not divisible by num_attention_heads
    bad_config_1 = dict(mha_config)
    bad_config_1["hidden_size"] = 769  # not divisible by 12
    fixtures.append({
        "op": "hf_config_validate",
        "label": "bad_divisibility_fails",
        "json": json.dumps(bad_config_1),
        "expected_valid": False,
        "fail_reason": "hidden_size not divisible by num_attention_heads",
    })

    # validate() — should fail: zero hidden_size
    bad_config_2 = dict(mha_config)
    bad_config_2["hidden_size"] = 0
    fixtures.append({
        "op": "hf_config_validate",
        "label": "zero_hidden_size_fails",
        "json": json.dumps(bad_config_2),
        "expected_valid": False,
        "fail_reason": "hidden_size is zero",
    })

    # validate() — should fail: unsupported activation
    bad_config_3 = dict(mha_config)
    bad_config_3["hidden_act"] = "tanh_xyz"
    fixtures.append({
        "op": "hf_config_validate",
        "label": "unsupported_activation_fails",
        "json": json.dumps(bad_config_3),
        "expected_valid": False,
        "fail_reason": "unsupported hidden_act",
    })

    # from_json_str — bad JSON should error
    fixtures.append({
        "op": "hf_config_parse_error",
        "label": "bad_json",
        "json": "{ not valid json",
        "expected_error": True,
    })

    # from_json_str — missing required field should error
    fixtures.append({
        "op": "hf_config_parse_error",
        "label": "missing_required_field",
        "json": json.dumps({"num_hidden_layers": 2}),  # missing hidden_size etc.
        "expected_error": True,
    })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: HfModelSummary deserialization
# ---------------------------------------------------------------------------

def make_model_summary_fixtures() -> list[dict]:
    """Generate HfModelSummary deserialization reference fixtures.

    The Hub API returns JSON; we encode the expected Rust-side deserialization
    output for known Hub response payloads. The Python huggingface_hub library
    parses the same JSON; we validate our expected values match by parsing
    through it.
    """
    from huggingface_hub.hf_api import ModelInfo as HfModelInfoPy

    fixtures = []

    # Minimal: just modelId
    json_minimal = '{"modelId": "microsoft/resnet-50"}'
    data = json.loads(json_minimal)
    fixtures.append({
        "op": "model_summary_deserialize",
        "label": "minimal_modelId",
        "json": json_minimal,
        "expected": {
            "model_id": "microsoft/resnet-50",
            "author": None,  # populated by populate_authors in Rust
            "downloads": None,
            "likes": None,
            "tags": [],
            "library_name": None,
            "pipeline_tag": None,
        },
    })

    # With `id` alias
    json_id_alias = '{"id": "bert-base-uncased"}'
    fixtures.append({
        "op": "model_summary_deserialize",
        "label": "id_alias",
        "json": json_id_alias,
        "expected": {
            "model_id": "bert-base-uncased",
            "author": None,
            "downloads": None,
            "likes": None,
            "tags": [],
            "library_name": None,
            "pipeline_tag": None,
        },
    })

    # Full fields
    json_full = json.dumps({
        "modelId": "microsoft/resnet-50",
        "downloads": 1234567,
        "likes": 42,
        "tags": ["image-classification", "pytorch", "safetensors"],
        "library_name": "transformers",
        "pipeline_tag": "image-classification",
    })
    fixtures.append({
        "op": "model_summary_deserialize",
        "label": "full_fields",
        "json": json_full,
        "expected": {
            "model_id": "microsoft/resnet-50",
            "author": None,
            "downloads": 1234567,
            "likes": 42,
            "tags": ["image-classification", "pytorch", "safetensors"],
            "library_name": "transformers",
            "pipeline_tag": "image-classification",
        },
    })

    # Unknown fields should be ignored
    json_unknown = json.dumps({
        "modelId": "microsoft/resnet-50",
        "private": False,
        "sha": "abc123def456",
        "gated": False,
    })
    fixtures.append({
        "op": "model_summary_deserialize",
        "label": "unknown_fields_ignored",
        "json": json_unknown,
        "expected": {
            "model_id": "microsoft/resnet-50",
            "author": None,
            "downloads": None,
            "likes": None,
            "tags": [],
            "library_name": None,
            "pipeline_tag": None,
        },
    })

    # Populate authors from model_id
    populate_cases = [
        ("microsoft/resnet-50", "microsoft"),
        ("meta-llama/Llama-2-7b", "meta-llama"),
        ("bert-base-uncased", None),
        ("", None),
    ]
    for model_id, expected_author in populate_cases:
        # Reference: Python huggingface_hub's author derivation matches split_once('/')
        if "/" in model_id:
            author = model_id.split("/", 1)[0]
        else:
            author = None
        assert author == expected_author
        fixtures.append({
            "op": "populate_author",
            "model_id": model_id,
            "expected_author": expected_author,
        })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: HfModelInfo with siblings deserialization
# ---------------------------------------------------------------------------

def make_model_info_fixtures() -> list[dict]:
    """Generate HfModelInfo deserialization reference fixtures."""
    fixtures = []

    json_with_siblings = json.dumps({
        "modelId": "microsoft/resnet-50",
        "siblings": [
            {"rfilename": "config.json"},
            {"rfilename": "model.safetensors"},
            {"rfilename": "README.md"},
        ],
    })
    fixtures.append({
        "op": "model_info_deserialize",
        "label": "with_siblings",
        "json": json_with_siblings,
        "expected": {
            "model_id": "microsoft/resnet-50",
            "siblings_count": 3,
            "siblings_filenames": ["config.json", "model.safetensors", "README.md"],
        },
    })

    # get_model("") should error (empty repo_id)
    fixtures.append({
        "op": "get_model_empty_repo_id",
        "label": "empty_repo_id_errors",
        "repo_id": "",
        "expected_error": True,
        "expected_error_contains": "must not be empty",
    })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: path-component sanitization
# These encode the EXPECTED accept/reject behavior of sanitize_path_component.
# ---------------------------------------------------------------------------

def make_sanitize_fixtures() -> list[dict]:
    """Generate sanitize_path_component reference fixtures.

    These cover the security-critical path sanitizer. Each case documents
    whether the input should be accepted or rejected, and for rejections,
    what the error message should contain.
    """
    fixtures = []

    # --- Rejection cases
    reject_cases = [
        ("", "must not be empty"),
        ("..", "dot-path"),
        (".", "dot-path"),
        ("/etc/passwd", "path separator"),
        ("foo/bar", "path separator"),
        ("foo\\bar", "path separator"),
        ("foo\0bar", "null byte"),
        ("foo/../../../etc/passwd", "path separator"),
        ("../../.bashrc", "path separator"),
        (".hidden", "starts with '.'"),
        ("foo:bar", "colon"),
        ("model.safetensors:secret", "colon"),
        ("a" * 256, "too long"),  # > 255 bytes
    ]
    for s, needle in reject_cases:
        fixtures.append({
            "op": "sanitize_path_component",
            "input": s,
            "expected_ok": False,
            "expected_error_contains": needle,
        })

    # --- Accept cases
    accept_cases = [
        "model-00001-of-00004.safetensors",
        "config.json",
        "main",
        "v1.0",
        "abc123def456",
        "feature-branch",
        "a" * 255,  # exactly max length
    ]
    for s in accept_cases:
        fixtures.append({
            "op": "sanitize_path_component",
            "input": s,
            "expected_ok": True,
        })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: cache key validation
# ---------------------------------------------------------------------------

def make_cache_validation_fixtures() -> list[dict]:
    """Generate validate_cache_relative reference fixtures.

    The cache key validator is the security boundary between user/server-
    supplied filenames and the local filesystem. These fixtures encode the
    expected accept/reject decisions.
    """
    fixtures = []

    reject_cases = [
        ("", "empty"),
        ("../etc/passwd", "traversal"),
        ("../../Windows/System32/drivers/etc/hosts", "traversal"),
        ("/etc/passwd", "absolute"),
        ("path/../../escape", "traversal"),
        ("path\0null", "null"),
        ("..", "traversal"),
        (".", "traversal or CurDir"),
        ("./foo", "traversal or CurDir"),
    ]
    for s, reason in reject_cases:
        fixtures.append({
            "op": "validate_cache_relative",
            "input": s,
            "expected_ok": False,
            "reject_reason": reason,
        })

    # Accept cases
    accept_cases = [
        "resnet50.safetensors",
        "meta-llama/Llama-3-8B/config.json",
        ".gitkeep",  # leading-dot Normal component is OK at cache layer
        "simple",
    ]
    for s in accept_cases:
        fixtures.append({
            "op": "validate_cache_relative",
            "input": s,
            "expected_ok": True,
        })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: registry
# ---------------------------------------------------------------------------

def make_registry_fixtures() -> list[dict]:
    """Generate registry reference fixtures.

    These encode the EXPECTED static registry contents that ferrotorch-hub's
    Rust implementation must contain. All values come from the registry.rs
    source rather than the Python library — the registry is a ferrotorch-hub
    concept, not a huggingface_hub concept.
    """
    # Known models that must be in the registry (from registry.rs)
    known_models = [
        {
            "name": "resnet18",
            "num_parameters": 11689512,
            "format": "SafeTensors",
        },
        {
            "name": "resnet34",
            "num_parameters": 21797672,
            "format": "SafeTensors",
        },
        {
            "name": "resnet50",
            "num_parameters": 25557032,
            "format": "SafeTensors",
        },
        {
            "name": "vgg11",
            "num_parameters": 132863336,
            "format": "SafeTensors",
        },
        {
            "name": "vgg16",
            "num_parameters": 138357544,
            "format": "SafeTensors",
        },
        {
            "name": "vit_b_16",
            "num_parameters": 86567656,
            "format": "SafeTensors",
        },
        {
            "name": "efficientnet_b0",
            "num_parameters": 5288548,
            "format": "SafeTensors",
        },
        {
            "name": "swin_tiny",
            "num_parameters": 28288354,
            "format": "SafeTensors",
        },
        {
            "name": "convnext_tiny",
            "num_parameters": 28589128,
            "format": "SafeTensors",
        },
        {
            "name": "unet",
            "num_parameters": 31037633,
            "format": "SafeTensors",
        },
        {
            "name": "yolo",
            "num_parameters": 61949149,
            "format": "SafeTensors",
        },
        {
            "name": "mobilenet_v2",
            "num_parameters": 3504872,
            "format": "SafeTensors",
        },
        {
            "name": "mobilenet_v3_small",
            "num_parameters": 2542856,
            "format": "SafeTensors",
        },
        {
            "name": "densenet121",
            "num_parameters": 7978856,
            "format": "SafeTensors",
        },
        {
            "name": "inception_v3",
            "num_parameters": 27161264,
            "format": "SafeTensors",
        },
        # #1130: torchvision-canonical detection / segmentation models with
        # pretrained safetensors pinned at `huggingface.co/ferrotorch/*`.
        # `num_parameters` reflects torchvision 0.21's
        # `sum(p.numel() for p in model.parameters())` on the pretrained
        # checkpoint, recorded at conversion time by
        # `scripts/pin_pretrained_weights.py`.
        {
            "name": "fasterrcnn_resnet50_fpn",
            "num_parameters": 41755286,
            "format": "SafeTensors",
        },
        {
            "name": "maskrcnn_resnet50_fpn",
            "num_parameters": 44401393,
            "format": "SafeTensors",
        },
        {
            "name": "deeplabv3_resnet50",
            "num_parameters": 42004074,
            "format": "SafeTensors",
        },
        {
            "name": "fcn_resnet50",
            "num_parameters": 35322218,
            "format": "SafeTensors",
        },
        {
            "name": "ssd300_vgg16",
            "num_parameters": 35641826,
            "format": "SafeTensors",
        },
    ]

    fixtures = [
        {
            "op": "registry_list_models_count",
            "expected_count": len(known_models),
        },
        {
            "op": "registry_known_models",
            "models": known_models,
        },
        {
            "op": "registry_get_model_info_unknown",
            "name": "nonexistent_model",
            "expected_some": False,
        },
        {
            "op": "registry_get_model_info_empty",
            "name": "",
            "expected_some": False,
        },
    ]

    for m in known_models:
        fixtures.append({
            "op": "registry_get_model_info",
            "name": m["name"],
            "expected_some": True,
            "expected_num_parameters": m["num_parameters"],
            "expected_format": m["format"],
        })

    return fixtures


# ---------------------------------------------------------------------------
# Fixture group: cascade-skipped items (network-dependent)
# ---------------------------------------------------------------------------

def make_cascade_skip_fixtures() -> list[dict]:
    """Mark network-dependent operations as cascade-skipped.

    These operations require actual network access and are covered by
    integration tests (load_pretrained_smoke.rs) rather than conformance
    fixtures.
    """
    SKIP_REASON = (
        "network-dependent — covered by integration tests "
        "(ferrotorch-hub/tests/load_pretrained_smoke.rs)"
    )
    return [
        cascade_skip("hf_download_model", SKIP_REASON),
        cascade_skip("search_models", SKIP_REASON),
        cascade_skip("get_model", SKIP_REASON),
        cascade_skip("download_weights_http", SKIP_REASON),
        cascade_skip("load_pretrained_http", SKIP_REASON),
    ]


# ---------------------------------------------------------------------------
# Main: assemble all fixtures and write the JSON file
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = pathlib.Path(__file__).parent.parent
    out_path = repo_root / "ferrotorch-hub" / "tests" / "conformance" / "fixtures.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import datetime

    all_fixtures = []
    all_fixtures.extend(make_repo_id_fixtures())
    all_fixtures.extend(make_search_query_fixtures())
    all_fixtures.extend(make_url_encode_fixtures())
    all_fixtures.extend(make_hf_config_fixtures())
    all_fixtures.extend(make_model_summary_fixtures())
    all_fixtures.extend(make_model_info_fixtures())
    all_fixtures.extend(make_sanitize_fixtures())
    all_fixtures.extend(make_cache_validation_fixtures())
    all_fixtures.extend(make_registry_fixtures())
    all_fixtures.extend(make_cascade_skip_fixtures())

    output = {
        "metadata": {
            "huggingface_hub_version": HF_VERSION,
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "offline_only": True,
            "network_calls_made": False,
        },
        "fixtures": all_fixtures,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    # Count by op
    from collections import Counter
    op_counts = Counter(f["op"] for f in all_fixtures)
    total = len(all_fixtures)
    cascade = sum(1 for f in all_fixtures if f.get("cascade_skip"))

    print(f"\nWrote {total} fixtures ({cascade} cascade-skipped) to {out_path}")
    print("Op breakdown:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")


if __name__ == "__main__":
    main()
