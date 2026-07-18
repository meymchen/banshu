//! Providers — mostly data: id, name, base URL, env-var auth, and a handle to
//! the wire protocol its models speak. Per-vendor constructors (DeepSeek, Z.AI,
//! …) live in submodules and delegate to the generic constructors here.

use std::sync::Arc;

use crate::api::anthropic_messages::AnthropicMessages;
use crate::api::openai_completions::OpenAiCompletions;
use crate::api::{ApiRequest, ChatApi};
use crate::http;
use crate::options::StreamOptions;
use crate::stream::MessageStream;
use crate::types::{ApiKind, Context, Model};

/// Request-side prompt-cache controls supported by an OpenAI-compatible
/// provider.
///
/// Cache usage is parsed for every provider regardless of this setting. This
/// only controls non-standard request fields or headers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiPromptCaching {
    /// The service manages caching automatically; send no cache extensions.
    #[default]
    Automatic,
    /// Send OpenAI's `prompt_cache_key` and optional 24-hour retention field.
    OpenAi,
    /// Send stable session-affinity headers when a session id is supplied.
    SessionAffinityHeaders,
}

/// Endpoint quirks declared by an OpenAI-compatible provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenAiCompat {
    /// Request-side prompt-cache controls.
    pub prompt_caching: OpenAiPromptCaching,
    /// Every replayed assistant message must carry a `reasoning_content`
    /// field (`""` when it produced no thinking) while a reasoning model is
    /// active. DeepSeek requires this.
    pub requires_reasoning_content_on_assistant_messages: bool,
}

/// Endpoint quirks declared by an Anthropic-compatible provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnthropicCompat {
    /// Replay signatureless thinking as `signature: ""` instead of
    /// downgrading it to a text block. Some compatible providers emit and
    /// accept empty signatures.
    pub allow_empty_signature: bool,
    /// Send `x-session-affinity` from the session id when caching is enabled,
    /// for providers that route prompt-cache hits by replica.
    pub send_session_affinity_headers: bool,
}

/// A configured provider: metadata + auth + a wire-protocol handle.
pub struct Provider {
    id: String,
    name: String,
    base_url: String,
    api_key_env: Vec<String>,
    api_kind: ApiKind,
    api: Arc<dyn ChatApi>,
    http: reqwest::Client,
    openai_compat: OpenAiCompat,
    anthropic_compat: AnthropicCompat,
}

impl Provider {
    /// Build a provider that speaks the OpenAI `chat/completions` protocol.
    ///
    /// `api_key_env` lists environment variables checked, in order, when no
    /// per-request key is supplied.
    pub fn openai_compatible(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            base_url: base_url.into(),
            api_key_env: api_key_env.into_iter().map(Into::into).collect(),
            api_kind: ApiKind::OpenAiCompletions,
            api: Arc::new(OpenAiCompletions),
            http: http::build_client(),
            openai_compat: OpenAiCompat::default(),
            anthropic_compat: AnthropicCompat::default(),
        }
    }

    /// Build a provider that speaks the Anthropic `/v1/messages` protocol.
    ///
    /// `api_key_env` lists environment variables checked, in order, when no
    /// per-request key is supplied.
    pub fn anthropic_compatible(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            base_url: base_url.into(),
            api_key_env: api_key_env.into_iter().map(Into::into).collect(),
            api_kind: ApiKind::AnthropicMessages,
            api: Arc::new(AnthropicMessages),
            http: http::build_client(),
            openai_compat: OpenAiCompat::default(),
            anthropic_compat: AnthropicCompat::default(),
        }
    }

    /// Configure the endpoint quirks of this OpenAI-compatible provider.
    pub fn with_openai_compat(mut self, compat: OpenAiCompat) -> Self {
        self.openai_compat = compat;
        self
    }

    /// Configure the endpoint quirks of this Anthropic-compatible provider.
    pub fn with_anthropic_compat(mut self, compat: AnthropicCompat) -> Self {
        self.anthropic_compat = compat;
        self
    }

    /// Configure the request-side prompt-cache controls accepted by this
    /// OpenAI-compatible provider.
    pub fn with_openai_prompt_caching(mut self, caching: OpenAiPromptCaching) -> Self {
        self.openai_compat.prompt_caching = caching;
        self
    }

    /// OpenAI — Chat Completions with explicit prompt-cache routing support.
    pub fn openai() -> Self {
        Self::openai_compatible(
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            ["OPENAI_API_KEY"],
        )
        .with_openai_prompt_caching(OpenAiPromptCaching::OpenAi)
    }

    /// DeepSeek — OpenAI-compatible, `DEEPSEEK_API_KEY`.
    pub fn deepseek() -> Self {
        Self::openai_compatible(
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com",
            ["DEEPSEEK_API_KEY"],
        )
        .with_openai_compat(OpenAiCompat {
            requires_reasoning_content_on_assistant_messages: true,
            ..OpenAiCompat::default()
        })
    }

    /// Z.AI (GLM coding plan) — OpenAI-compatible, `ZAI_API_KEY`.
    pub fn zai() -> Self {
        Self::openai_compatible(
            "zai",
            "Z.AI",
            "https://api.z.ai/api/coding/paas/v4",
            ["ZAI_API_KEY"],
        )
    }

    /// MiniMax — Anthropic-compatible, `MINIMAX_API_KEY`.
    pub fn minimax() -> Self {
        Self::anthropic_compatible(
            "minimax",
            "MiniMax",
            "https://api.minimax.io/anthropic",
            ["MINIMAX_API_KEY"],
        )
    }

    /// Moonshot AI — OpenAI-compatible, `MOONSHOT_API_KEY`.
    pub fn moonshot() -> Self {
        Self::openai_compatible(
            "moonshot",
            "Moonshot AI",
            "https://api.moonshot.ai/v1",
            ["MOONSHOT_API_KEY"],
        )
    }

    /// Kimi For Coding — Anthropic-compatible, `KIMI_API_KEY`.
    pub fn kimi() -> Self {
        Self::anthropic_compatible(
            "kimi",
            "Kimi For Coding",
            "https://api.kimi.com/coding",
            ["KIMI_API_KEY"],
        )
    }

    /// Xiaomi MiMo — OpenAI-compatible, `XIAOMI_API_KEY`.
    pub fn xiaomi() -> Self {
        Self::openai_compatible(
            "xiaomi",
            "Xiaomi MiMo",
            "https://api.xiaomimimo.com/v1",
            ["XIAOMI_API_KEY"],
        )
    }

    /// The provider id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The provider display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The provider base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The wire protocol this provider's models speak.
    pub fn api_kind(&self) -> ApiKind {
        self.api_kind
    }

    /// The provider's models, loaded from the bundled catalog and stamped with
    /// this provider's id, base URL, and wire protocol.
    pub fn models(&self) -> Vec<Model> {
        crate::models::catalog_models(&self.id, &self.base_url, self.api_kind)
    }

    /// Whether a key is resolvable from configured environment variables.
    pub fn has_env_api_key(&self) -> bool {
        self.env_api_key().is_some()
    }

    /// Stream a completion for `model`. Never fails synchronously — see
    /// [`MessageStream`].
    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> MessageStream {
        let api_key = options.api_key.clone().or_else(|| self.env_api_key());
        self.api.stream(ApiRequest {
            model,
            context,
            options,
            api_key,
            http: self.http.clone(),
            openai_compat: self.openai_compat,
            anthropic_compat: self.anthropic_compat,
        })
    }

    /// Resolve an API key from the configured environment variables, in order.
    fn env_api_key(&self) -> Option<String> {
        self.api_key_env
            .iter()
            .find_map(|name| std::env::var(name).ok())
    }
}
