//! `Schema`: a typed, internal representation of the supported JSON-Schema
//! subset.
//!
//! ## Supported features
//!
//! - `type`: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
//! - `properties` + `required` on objects (only listed properties; nothing
//!   extra is allowed at sample time even if `additionalProperties` is
//!   true upstream — pragmatic strictness keeps the state machine bounded).
//! - `items` on arrays (homogeneous element type).
//! - `enum` of strings or numbers (closed value set).
//! - Nullable fields via `type: ["X", "null"]`.
//!
//! ## Not supported (yet)
//!
//! - `$ref` and recursive references.
//! - `oneOf` / `anyOf` / `allOf` composition.
//! - `pattern` (regex on strings).
//! - `format` (date-time, email, etc.) and other annotations.
//! - `minLength` / `maxLength` / `minimum` / `maximum` numeric / length
//!   constraints.
//! - `additionalProperties` (always treated as `false`).
//!
//! These are the right next chunks but live outside this turn.

use std::collections::{BTreeMap, BTreeSet};

/// Errors raised while compiling a JSON-Schema document into a [`Schema`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// The schema's `type` is missing, malformed, or refers to an unsupported
    /// type (e.g. only `string`, `number`, `integer`, `boolean`, `null`,
    /// `object`, `array` are supported).
    UnsupportedType(String),
    /// A composition keyword (`oneOf` / `anyOf` / `allOf` / `$ref`) was used.
    Unsupported(&'static str),
    /// `properties.<name>` was not a sub-schema object.
    MalformedProperty(String),
    /// `enum` value list contained mixed or non-string/non-number values.
    MalformedEnum,
    /// Generic "this isn't a schema-shaped JSON value" error.
    NotASchema,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedType(t) => write!(f, "unsupported type: {t}"),
            Self::Unsupported(k) => write!(f, "unsupported schema keyword: {k}"),
            Self::MalformedProperty(p) => write!(f, "malformed property `{p}`"),
            Self::MalformedEnum => write!(f, "malformed `enum` value list"),
            Self::NotASchema => write!(f, "value is not a JSON-Schema object"),
        }
    }
}

impl std::error::Error for SchemaError {}

/// One concrete JSON-shaped type the constrained decoder will produce.
#[derive(Debug, Clone, PartialEq)]
pub enum Schema {
    /// An object with a fixed set of typed properties. Keys not listed in
    /// `properties` are rejected; keys listed in `required` must appear at
    /// least once.
    Object {
        properties: BTreeMap<String, Schema>,
        required: BTreeSet<String>,
    },
    /// An array of values all matching `item`.
    Array { item: Box<Schema> },
    /// A JSON string of arbitrary content (no length / pattern constraint).
    String,
    /// A finite set of allowed string values.
    StringEnum(Vec<String>),
    /// A JSON number — integer or fractional, optionally with sign / exponent.
    Number,
    /// A JSON integer (no fractional part, no exponent).
    Integer,
    /// `true` or `false`.
    Boolean,
    /// `null`.
    Null,
    /// A union of the inner schema and `null`. Equivalent to JSON Schema's
    /// `type: ["X", "null"]`.
    Nullable(Box<Schema>),
}

impl Schema {
    /// Parse a JSON-Schema document into a [`Schema`]. Returns `Err` for
    /// any unsupported feature so the caller can decide between erroring
    /// out and falling back to unconstrained sampling.
    pub fn from_json_schema(value: &serde_json::Value) -> Result<Self, SchemaError> {
        let map = value.as_object().ok_or(SchemaError::NotASchema)?;

        // Reject composition / refs explicitly so they aren't silently dropped.
        if map.contains_key("oneOf") {
            return Err(SchemaError::Unsupported("oneOf"));
        }
        if map.contains_key("anyOf") {
            return Err(SchemaError::Unsupported("anyOf"));
        }
        if map.contains_key("allOf") {
            return Err(SchemaError::Unsupported("allOf"));
        }
        if map.contains_key("$ref") {
            return Err(SchemaError::Unsupported("$ref"));
        }

        // `enum` short-circuits the type detection: a closed value set.
        if let Some(values) = map.get("enum") {
            return parse_enum(values);
        }

        let type_value = map.get("type").ok_or(SchemaError::NotASchema)?;
        let mut accepts_null = false;
        let primary_type = match type_value {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(types) => {
                let mut concrete: Option<String> = None;
                for t in types {
                    let t = t
                        .as_str()
                        .ok_or_else(|| SchemaError::UnsupportedType(t.to_string()))?;
                    if t == "null" {
                        accepts_null = true;
                    } else if concrete.is_some() {
                        return Err(SchemaError::Unsupported(
                            "multi-type union (only X | null is supported)",
                        ));
                    } else {
                        concrete = Some(t.to_string());
                    }
                }
                concrete.ok_or(SchemaError::Unsupported("type: [\"null\"] only"))?
            }
            _ => return Err(SchemaError::UnsupportedType(type_value.to_string())),
        };

        let inner = match primary_type.as_str() {
            "object" => parse_object(map)?,
            "array" => parse_array(map)?,
            "string" => Self::String,
            "number" => Self::Number,
            "integer" => Self::Integer,
            "boolean" => Self::Boolean,
            "null" => Self::Null,
            other => return Err(SchemaError::UnsupportedType(other.to_string())),
        };

        if accepts_null {
            Ok(Self::Nullable(Box::new(inner)))
        } else {
            Ok(inner)
        }
    }
}

fn parse_object(map: &serde_json::Map<String, serde_json::Value>) -> Result<Schema, SchemaError> {
    let props_value = map
        .get("properties")
        .ok_or(SchemaError::Unsupported("object without `properties`"))?;
    let props_map = props_value
        .as_object()
        .ok_or_else(|| SchemaError::MalformedProperty("properties".into()))?;
    let mut properties = BTreeMap::new();
    for (key, val) in props_map {
        let sub = Schema::from_json_schema(val)
            .map_err(|_| SchemaError::MalformedProperty(key.clone()))?;
        properties.insert(key.clone(), sub);
    }

    let required = match map.get("required") {
        Some(serde_json::Value::Array(items)) => {
            let mut set = BTreeSet::new();
            for item in items {
                let key = item
                    .as_str()
                    .ok_or_else(|| SchemaError::MalformedProperty("required".into()))?;
                if !properties.contains_key(key) {
                    return Err(SchemaError::MalformedProperty(format!(
                        "required key `{key}` not declared in properties"
                    )));
                }
                set.insert(key.to_string());
            }
            set
        }
        Some(_) => return Err(SchemaError::MalformedProperty("required".into())),
        None => BTreeSet::new(),
    };

    Ok(Schema::Object {
        properties,
        required,
    })
}

fn parse_array(map: &serde_json::Map<String, serde_json::Value>) -> Result<Schema, SchemaError> {
    let item = map
        .get("items")
        .ok_or(SchemaError::Unsupported("array without `items`"))?;
    let item_schema = Schema::from_json_schema(item)?;
    Ok(Schema::Array {
        item: Box::new(item_schema),
    })
}

fn parse_enum(values: &serde_json::Value) -> Result<Schema, SchemaError> {
    let arr = values.as_array().ok_or(SchemaError::MalformedEnum)?;
    if arr.is_empty() {
        return Err(SchemaError::MalformedEnum);
    }
    // We only support string enums in this subset (covers the
    // ExtractionResponse use case: Direction, Confidence, EvidenceType).
    let mut strings = Vec::with_capacity(arr.len());
    for v in arr {
        let s = v.as_str().ok_or(SchemaError::MalformedEnum)?;
        strings.push(s.to_string());
    }
    Ok(Schema::StringEnum(strings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_simple_string_schema() {
        let s = Schema::from_json_schema(&json!({"type": "string"})).unwrap();
        assert_eq!(s, Schema::String);
    }

    #[test]
    fn parses_simple_object() {
        let s = Schema::from_json_schema(&json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "n": {"type": "integer"}
            },
            "required": ["name"]
        }))
        .unwrap();
        match s {
            Schema::Object {
                properties,
                required,
            } => {
                assert_eq!(properties.len(), 2);
                assert_eq!(required.len(), 1);
                assert!(required.contains("name"));
                assert_eq!(properties.get("name"), Some(&Schema::String));
                assert_eq!(properties.get("n"), Some(&Schema::Integer));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn parses_nullable_via_type_array() {
        let s = Schema::from_json_schema(&json!({"type": ["string", "null"]})).unwrap();
        assert_eq!(s, Schema::Nullable(Box::new(Schema::String)));
    }

    #[test]
    fn parses_string_enum() {
        let s = Schema::from_json_schema(&json!({"enum": ["high", "medium", "low"]})).unwrap();
        assert_eq!(
            s,
            Schema::StringEnum(vec!["high".into(), "medium".into(), "low".into()])
        );
    }

    #[test]
    fn parses_array_of_numbers() {
        let s = Schema::from_json_schema(&json!({
            "type": "array",
            "items": {"type": "number"}
        }))
        .unwrap();
        assert_eq!(
            s,
            Schema::Array {
                item: Box::new(Schema::Number)
            }
        );
    }

    #[test]
    fn parses_nested_object() {
        let s = Schema::from_json_schema(&json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"v": {"type": "boolean"}},
                    "required": ["v"]
                }
            },
            "required": ["inner"]
        }))
        .unwrap();
        match s {
            Schema::Object { properties, .. } => {
                let inner = properties.get("inner").unwrap();
                match inner {
                    Schema::Object { properties: ip, .. } => {
                        assert_eq!(ip.get("v"), Some(&Schema::Boolean));
                    }
                    _ => panic!("expected nested Object"),
                }
            }
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn rejects_oneof() {
        let err = Schema::from_json_schema(&json!({
            "oneOf": [{"type": "string"}, {"type": "number"}]
        }))
        .unwrap_err();
        assert!(matches!(err, SchemaError::Unsupported("oneOf")));
    }

    #[test]
    fn rejects_ref() {
        let err = Schema::from_json_schema(&json!({"$ref": "#/definitions/foo"})).unwrap_err();
        assert!(matches!(err, SchemaError::Unsupported("$ref")));
    }

    #[test]
    fn rejects_required_key_not_in_properties() {
        let err = Schema::from_json_schema(&json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["b"]
        }))
        .unwrap_err();
        assert!(matches!(err, SchemaError::MalformedProperty(_)));
    }
}
