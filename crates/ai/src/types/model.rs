//! The `Model` metadata type and its supporting enums.

use crate::auth::{ProviderHeaders, RedactedHeaders};

/// Which wire protocol a model speaks. Used for identification and to route
/// the model to the matching [`ProtocolAdapter`](crate::ProtocolAdapter).
///
/// Serializes to pi-ai's stable api id strings.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ApiKind {
    /// OpenAI-style `POST /chat/completions`.
    #[serde(rename = "openai-completions")]
    OpenAiCompletions,
    /// Anthropic-style `POST /v1/messages`.
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages,
}

/// An input modality a model accepts.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Modality {
    /// Text input.
    Text,
    /// Image input.
    Image,
}

/// Whether a model capability is known to work. `Unknown` is never presented
/// as supported: capability gating must match on [`Supported`](Self::Supported)
/// explicitly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CapabilitySupport {
    /// Confirmed supported by the metadata source.
    Supported,
    /// Confirmed unsupported by the metadata source.
    Unsupported,
    /// The metadata source says nothing; nothing is guessed.
    #[default]
    Unknown,
}

/// What a model can do, per capability. Every source that cannot attest a
/// capability leaves it [`CapabilitySupport::Unknown`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilities {
    /// Tool calling (function calling).
    pub tool_calling: CapabilitySupport,
}

/// Per-token cost rates in USD per million tokens.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelCost {
    /// Input (prompt) rate, $/1M tokens.
    pub input: f64,
    /// Output (completion) rate, $/1M tokens.
    pub output: f64,
    /// Cache-read rate, $/1M tokens.
    pub cache_read: f64,
    /// Cache-write rate, $/1M tokens.
    pub cache_write: f64,
}

/// Metadata describing a single model on a provider.
#[derive(Clone)]
pub struct Model {
    /// Provider-specific model id sent on the wire (e.g. `deepseek-chat`).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// The wire protocol this model speaks.
    pub api: ApiKind,
    /// Owning provider id.
    pub provider: String,
    /// Base URL of the API endpoint.
    pub base_url: String,
    /// Model-specific custom headers, applied above provider defaults.
    pub headers: ProviderHeaders,
    /// Whether the model supports reasoning / thinking.
    pub reasoning: bool,
    /// Accepted input modalities.
    pub input: Vec<Modality>,
    /// Capability attestations from the model's metadata source.
    pub capabilities: ModelCapabilities,
    /// Token cost rates.
    pub cost: ModelCost,
    /// Maximum context window in tokens.
    pub context_window: u32,
    /// Maximum output tokens per response.
    pub max_tokens: u32,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Model")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("api", &self.api)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("headers", &RedactedHeaders(&self.headers))
            .field("reasoning", &self.reasoning)
            .field("input", &self.input)
            .field("capabilities", &self.capabilities)
            .field("cost", &self.cost)
            .field("context_window", &self.context_window)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

impl Model {
    /// Build an OpenAI-completions model with sensible defaults. Chain
    /// [`with_base_url`](Self::with_base_url) and field updates to fill it in.
    pub fn openai_completions(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            api: ApiKind::OpenAiCompletions,
            provider: String::new(),
            base_url: String::new(),
            headers: ProviderHeaders::new(),
            reasoning: false,
            input: vec![Modality::Text],
            capabilities: ModelCapabilities::default(),
            cost: ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
        }
    }

    /// Build an Anthropic Messages model with sensible defaults.
    pub fn anthropic_messages(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            api: ApiKind::AnthropicMessages,
            provider: String::new(),
            base_url: String::new(),
            headers: ProviderHeaders::new(),
            reasoning: false,
            input: vec![Modality::Text],
            capabilities: ModelCapabilities::default(),
            cost: ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
        }
    }

    /// Set the endpoint base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}
