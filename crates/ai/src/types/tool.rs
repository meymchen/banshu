//! Tool definitions offered to the model, and which of them it may call.

/// A tool the model may call.
///
/// `parameters` is an opaque JSON Schema value — banshu does not dictate how
/// callers author schemas. A `schemars` convenience constructor may be offered
/// later behind the `schemars` feature.
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
