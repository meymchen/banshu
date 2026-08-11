//! Tool definitions offered to the model, and which of them it may call.

/// A tool the model may call.
///
/// `parameters` is an opaque JSON Schema value — banshu does not dictate how
/// callers author schemas. Call [`Tool::validate_arguments`] before executing
/// a call to enforce the schema without coercing or mutating its parsed JSON.
/// A `schemars` convenience constructor may be offered later behind the
/// `schemars` feature.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// The tool name the model uses to invoke it.
    pub name: String,
    /// A description guiding when and how to use the tool.
    pub description: String,
    /// JSON Schema for the tool's arguments.
    pub parameters: serde_json::Value,
    /// Marks the schema as authored to strict-mode rules (every object closed
    /// with `additionalProperties: false`, every property declared), asking
    /// the provider for schema-constrained tool arguments.
    ///
    /// The marker reaches the wire only when the provider's compat declares
    /// strict tool schemas
    /// ([`OpenAiCompat::strict_tool_schemas`](crate::OpenAiCompat::strict_tool_schemas)
    /// /
    /// [`AnthropicCompat::strict_tool_schemas`](crate::AnthropicCompat::strict_tool_schemas));
    /// against a provider declaring no support the field is omitted entirely,
    /// and the tool works unconstrained.
    ///
    /// Skipped on serialization when `false`, so a context snapshot written
    /// before the marker existed is byte-identical to one written after.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
}

impl Tool {
    /// Validate parsed arguments against this tool's JSON Schema.
    ///
    /// Validation never coerces values or mutates `arguments`. On success the
    /// returned value is deeply equal to the caller's input. On failure,
    /// [`ToolValidationError`] identifies this tool, the failing value by RFC
    /// 6901 JSON Pointer, the violated JSON Schema keyword, and a readable
    /// reason. This accepts [`ToolCall::arguments`](crate::ToolCall::arguments)
    /// directly after a streamed call ends successfully.
    pub fn validate_arguments(
        &self,
        arguments: &serde_json::Value,
    ) -> Result<serde_json::Value, ToolValidationError> {
        let validator =
            jsonschema::validator_for(&self.parameters).map_err(|error| ToolValidationError {
                tool_name: self.name.clone(),
                path: String::new(),
                constraint: "schema".into(),
                reason: error.to_string(),
            })?;

        if let Err(error) = validator.validate(arguments) {
            let schema_path = error.schema_path().as_str();
            let constraint = schema_path
                .rsplit('/')
                .next()
                .filter(|segment| !segment.is_empty())
                .unwrap_or("schema")
                .replace("~1", "/")
                .replace("~0", "~");
            return Err(ToolValidationError {
                tool_name: self.name.clone(),
                path: error.instance_path().as_str().to_owned(),
                constraint,
                reason: error.to_string(),
            });
        }

        Ok(arguments.clone())
    }
}

/// A tool-argument validation failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("tool `{tool_name}` arguments at `{path}` violate `{constraint}`: {reason}")]
pub struct ToolValidationError {
    /// Name of the tool whose arguments failed validation.
    pub tool_name: String,
    /// RFC 6901 JSON Pointer to the failing value; empty means the root.
    pub path: String,
    /// JSON Schema keyword whose constraint was violated.
    pub constraint: String,
    /// Human-readable explanation of the validation failure.
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reports_a_nested_constraint_without_changing_the_arguments() {
        let tool = Tool {
            name: "lookup".into(),
            description: String::new(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "minLength": 2 }
                }
            }),
            strict: false,
        };
        let arguments = json!({ "query": "x" });
        let before = arguments.clone();

        let error = tool.validate_arguments(&arguments).unwrap_err();

        assert_eq!(error.path, "/query");
        assert_eq!(error.constraint, "minLength");
        assert_eq!(arguments, before);
    }
}

/// Which tool the model may or must call — one cross-protocol vocabulary, set
/// per request via [`StreamOptions::tool_choice`](crate::StreamOptions::tool_choice).
///
/// Whether a given choice can be honoured is provider-dependent: each provider
/// declares the choices its endpoint accepts
/// ([`OpenAiCompat::tool_choice`](crate::OpenAiCompat::tool_choice) /
/// [`AnthropicCompat::tool_choice`](crate::AnthropicCompat::tool_choice)), and
/// a choice it cannot express terminates in-band with
/// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before any
/// HTTP request — it is never silently remapped onto a choice the caller did
/// not ask for. The default, no choice supplied, sends no `tool_choice` field
/// at all: the provider's own default applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The provider default, made explicit: the model decides whether and
    /// which tool to call.
    Auto,
    /// Forbid tool calls even though tools are offered.
    None,
    /// Require at least one tool call, any tool.
    Required,
    /// Require one specific tool. The name goes on the wire exactly as given —
    /// never rewritten — and the request is refused before dispatch unless the
    /// provider declares it can express a named choice.
    Named(String),
}
