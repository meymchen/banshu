//! Tool definitions offered to the model.

/// A tool the model may call.
///
/// `parameters` is an opaque JSON Schema value — banshu does not dictate how
/// callers author schemas. A `schemars` convenience constructor may be offered
/// later behind the `schemars` feature.
#[derive(Debug, Clone)]
pub struct Tool {
    /// The tool name the model uses to invoke it.
    pub name: String,
    /// A description guiding when and how to use the tool.
    pub description: String,
    /// JSON Schema for the tool's arguments.
    pub parameters: serde_json::Value,
}
