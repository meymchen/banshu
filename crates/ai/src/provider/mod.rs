//! Providers: identity, models, auth, default headers, compat quirks, and the
//! protocol adapters their models speak. Per-vendor constructors (DeepSeek,
//! Z.AI, …) delegate to [`ProviderBuilder`]; custom providers — local servers,
//! mixed-protocol gateways, third-party protocols — are built through
//! [`Provider::builder`] directly.

mod builder;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub use builder::ProviderBuilder;

use crate::api::anthropic_messages::AnthropicMessages;
use crate::api::openai_completions::OpenAiCompletions;
use crate::api::{ProtocolAdapter, api_name};
use crate::auth::{Auth, AuthResolver, ProviderHeaders};
use crate::discovery::{RefreshEntry, RefreshOutcome};
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

/// The in-process overlay of dynamically discovered models, layered over the
/// bundled catalog by [`Provider::models`]. Refresh failures leave it
/// untouched; it is lost when the process exits.
#[derive(Default)]
struct Overlay {
    /// models.dev catalog-refresh entries (full metadata; override + append).
    refreshed: Vec<Model>,
    /// Probe-synthesized models (bare ids; append-only, zero-means-unknown).
    probed: Vec<Model>,
}

/// A configured provider: metadata + auth + the protocol adapters its models
/// speak, at most one per [`ApiKind`].
pub struct Provider {
    id: String,
    name: String,
    base_url: String,
    auth: Auth,
    /// The primary protocol: the first adapter registered at build time. Used
    /// for catalog stamping and the list-models probe of a mixed-protocol
    /// provider; request routing always goes by `Model.api`.
    api_kind: ApiKind,
    adapters: HashMap<ApiKind, Arc<dyn ProtocolAdapter>>,
    headers: ProviderHeaders,
    /// Caller-supplied models from the builder, listed ahead of the bundled
    /// catalog and discovery overlay.
    models: Vec<Model>,
    http: reqwest::Client,
    openai_compat: OpenAiCompat,
    anthropic_compat: AnthropicCompat,
    models_dev_id: Option<String>,
    overlay: RwLock<Overlay>,
}

impl Provider {
    /// Start building a custom provider; see [`ProviderBuilder`] for the
    /// invariants [`build`](ProviderBuilder::build) enforces.
    pub fn builder(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> ProviderBuilder {
        ProviderBuilder::new(id, name, base_url)
    }

    /// Build a provider that speaks the OpenAI `chat/completions` protocol.
    ///
    /// `api_key_env` lists environment variables checked, in order, when no
    /// per-request key is supplied.
    ///
    /// # Panics
    ///
    /// Panics when `id` is empty — a caller bug the fallible
    /// [`Provider::builder`] reports as [`Error::Config`](crate::Error)
    /// instead.
    pub fn openai_compatible(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self::builder(id, name, base_url)
            .auth(Auth::api_key_env(api_key_env))
            .adapter(Arc::new(OpenAiCompletions))
            .build()
            .expect("openai_compatible: valid by construction given a non-empty id")
    }

    /// Build a provider that speaks the Anthropic `/v1/messages` protocol.
    ///
    /// `api_key_env` lists environment variables checked, in order, when no
    /// per-request key is supplied.
    ///
    /// # Panics
    ///
    /// Panics when `id` is empty — see [`openai_compatible`](Self::openai_compatible).
    pub fn anthropic_compatible(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self::builder(id, name, base_url)
            .auth(Auth::api_key_env(api_key_env))
            .adapter(Arc::new(AnthropicMessages))
            .build()
            .expect("anthropic_compatible: valid by construction given a non-empty id")
    }

    /// Set the models.dev provider key used by the catalog-refresh layer of
    /// dynamic discovery. Vendor constructors set this; custom providers
    /// without one skip the models.dev layer entirely.
    pub fn with_models_dev_id(mut self, id: impl Into<String>) -> Self {
        self.models_dev_id = Some(id.into());
        self
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

    /// Replace how this provider resolves credentials. Use
    /// [`Auth::keyless`](crate::Auth::keyless) for a local server that needs no
    /// key, or [`Auth::custom`](crate::Auth::custom) for a bespoke resolver.
    /// The generic constructors default to
    /// [`Auth::api_key_env`](crate::Auth::api_key_env).
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
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
        .with_models_dev_id("openai")
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
        .with_models_dev_id("deepseek")
    }

    /// Z.AI (GLM coding plan) — OpenAI-compatible, `ZAI_API_KEY`.
    pub fn zai() -> Self {
        Self::openai_compatible(
            "zai",
            "Z.AI",
            "https://api.z.ai/api/coding/paas/v4",
            ["ZAI_API_KEY"],
        )
        .with_models_dev_id("zai")
    }

    /// MiniMax — Anthropic-compatible, `MINIMAX_API_KEY`.
    pub fn minimax() -> Self {
        Self::anthropic_compatible(
            "minimax",
            "MiniMax",
            "https://api.minimax.io/anthropic",
            ["MINIMAX_API_KEY"],
        )
        .with_models_dev_id("minimax")
    }

    /// Moonshot AI — OpenAI-compatible, `MOONSHOT_API_KEY`.
    pub fn moonshot() -> Self {
        Self::openai_compatible(
            "moonshot",
            "Moonshot AI",
            "https://api.moonshot.ai/v1",
            ["MOONSHOT_API_KEY"],
        )
        .with_models_dev_id("moonshotai")
    }

    /// Kimi For Coding — Anthropic-compatible, `KIMI_API_KEY`.
    pub fn kimi() -> Self {
        Self::anthropic_compatible(
            "kimi",
            "Kimi For Coding",
            "https://api.kimi.com/coding",
            ["KIMI_API_KEY"],
        )
        .with_models_dev_id("kimi-for-coding")
    }

    /// Xiaomi MiMo — OpenAI-compatible, `XIAOMI_API_KEY`.
    pub fn xiaomi() -> Self {
        Self::openai_compatible(
            "xiaomi",
            "Xiaomi MiMo",
            "https://api.xiaomimimo.com/v1",
            ["XIAOMI_API_KEY"],
        )
        .with_models_dev_id("xiaomi")
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

    /// The provider's primary wire protocol (the first adapter registered at
    /// build time). Request routing goes by `Model.api`, not this.
    pub fn api_kind(&self) -> ApiKind {
        self.api_kind
    }

    /// The provider's models: caller-supplied models first, then the bundled
    /// catalog, then anything dynamic discovery has found — models.dev entries
    /// override same-id entries and append new ones; probe-discovered models
    /// are append-only.
    pub fn models(&self) -> Vec<Model> {
        let mut merged = self.models.clone();
        for model in crate::models::catalog_models(&self.id, &self.base_url, self.api_kind) {
            if !merged.iter().any(|known| known.id == model.id) {
                merged.push(model);
            }
        }
        let overlay = self.overlay.read().expect("model overlay lock poisoned");
        for model in &overlay.refreshed {
            match merged.iter_mut().find(|m| m.id == model.id) {
                Some(slot) => *slot = model.clone(),
                None => merged.push(model.clone()),
            }
        }
        for model in &overlay.probed {
            if !merged.iter().any(|m| m.id == model.id) {
                merged.push(model.clone());
            }
        }
        merged
    }

    /// The models.dev provider key for the catalog-refresh layer, if any.
    pub(crate) fn models_dev_id(&self) -> Option<&str> {
        self.models_dev_id.as_deref()
    }

    /// Apply a fetched models.dev `api.json` to this provider's overlay.
    pub(crate) fn apply_models_dev(&self, data: &serde_json::Value) -> RefreshOutcome {
        let Some(key) = &self.models_dev_id else {
            return RefreshOutcome::Skipped;
        };
        match crate::models::dev::models_from_api_json(
            data,
            key,
            &self.id,
            &self.base_url,
            self.api_kind,
        ) {
            Some(models) => {
                self.overlay
                    .write()
                    .expect("model overlay lock poisoned")
                    .refreshed = models;
                RefreshOutcome::Refreshed
            }
            None => RefreshOutcome::Failed(format!("models.dev has no models for `{key}`")),
        }
    }

    /// Probe this provider's list-models endpoint, replacing the probed layer
    /// of the overlay with zero-means-unknown models for the returned ids.
    /// Skipped without an API key; 404/405/501 means the endpoint doesn't
    /// exist. Only ids no catalog layer knows ever surface from this layer.
    pub(crate) async fn probe_models(&self) -> RefreshOutcome {
        let Some(api_key) = self.env_api_key() else {
            return RefreshOutcome::Skipped;
        };
        let base = self.base_url.trim_end_matches('/');
        let request = match self.api_kind {
            ApiKind::OpenAiCompletions => {
                self.http.get(format!("{base}/models")).bearer_auth(api_key)
            }
            ApiKind::AnthropicMessages => self
                .http
                .get(format!("{base}/v1/models"))
                .header("x-api-key", api_key)
                .header(
                    "anthropic-version",
                    crate::api::anthropic_messages::ANTHROPIC_VERSION,
                ),
        };
        let response = match request
            .timeout(crate::discovery::DISCOVERY_TIMEOUT)
            .send()
            .await
        {
            Ok(response) => response,
            Err(err) => return RefreshOutcome::Failed(err.to_string()),
        };
        let status = response.status();
        if matches!(status.as_u16(), 404 | 405 | 501) {
            return RefreshOutcome::EndpointUnsupported;
        }
        if !status.is_success() {
            return RefreshOutcome::Failed(format!("list-models returned HTTP {status}"));
        }
        let listed: crate::discovery::ListModelsResponse = match response.json().await {
            Ok(listed) => listed,
            Err(err) => return RefreshOutcome::Failed(err.to_string()),
        };
        let probed = listed
            .data
            .into_iter()
            .map(|entry| Model {
                name: entry.display_name.unwrap_or_else(|| entry.id.clone()),
                id: entry.id,
                api: self.api_kind,
                provider: self.id.clone(),
                base_url: self.base_url.clone(),
                headers: Default::default(),
                reasoning: false,
                input: vec![crate::types::Modality::Text],
                // A bare id attests nothing: capabilities stay Unknown.
                capabilities: crate::types::ModelCapabilities::default(),
                cost: crate::types::ModelCost::default(),
                context_window: 0,
                max_tokens: 0,
            })
            .collect();
        self.overlay
            .write()
            .expect("model overlay lock poisoned")
            .probed = probed;
        RefreshOutcome::Refreshed
    }

    /// Refresh this provider's dynamic models without a registry: fetch
    /// models.dev when a models.dev id is configured, then probe the vendor
    /// list-models endpoint. Best-effort — failures are recorded in the
    /// returned entry and never disturb previously discovered models.
    pub async fn refresh_models(&self) -> RefreshEntry {
        self.refresh_models_from(crate::discovery::MODELS_DEV_URL)
            .await
    }

    /// [`refresh_models`](Self::refresh_models) against a specific models.dev
    /// catalog URL.
    pub async fn refresh_models_from(&self, catalog_url: &str) -> RefreshEntry {
        let catalog = if self.models_dev_id.is_some() {
            match crate::discovery::fetch_models_dev(&self.http, catalog_url).await {
                Ok(data) => self.apply_models_dev(&data),
                Err(err) => RefreshOutcome::Failed(err),
            }
        } else {
            RefreshOutcome::Skipped
        };
        self.refresh_entry(catalog).await
    }

    /// Assemble this provider's report entry from an already-decided catalog
    /// outcome, running the probe layer.
    pub(crate) async fn refresh_entry(&self, catalog: RefreshOutcome) -> RefreshEntry {
        RefreshEntry {
            provider: self.id.clone(),
            catalog,
            probe: self.probe_models().await,
        }
    }

    /// The provider's HTTP client, shared with discovery fetches.
    pub(crate) fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    /// Whether this provider looks usable without further configuration:
    /// keyless providers always are, an api-key-env provider is when one of its
    /// variables is set. A custom-resolver provider reports `false` here
    /// because its resolver can only be consulted asynchronously — use
    /// [`Models::available`](crate::Models::available), which is async and
    /// consults it, for gating.
    pub fn is_available(&self) -> bool {
        self.auth.is_available()
    }

    /// Async availability check behind
    /// [`Models::available`](crate::Models::available): consults the resolver,
    /// so a custom resolver gets its say. A resolver error reads as
    /// unavailable.
    pub(crate) async fn check_available(&self) -> bool {
        self.auth.check().await.unwrap_or(false)
    }

    /// Stream a completion for `model`, routed to the adapter registered for
    /// `model.api`. Never fails synchronously — a model whose protocol this
    /// provider has no adapter for yields an in-band error; see
    /// [`MessageStream`].
    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> MessageStream {
        match self.adapters.get(&model.api) {
            Some(adapter) => crate::api::drive(
                adapter,
                model,
                context,
                options,
                &self.auth,
                &self.headers,
                &self.http,
                self.openai_compat,
                self.anthropic_compat,
            ),
            None => MessageStream::immediate_error(
                &model.id,
                &self.id,
                &format!(
                    "provider `{}` has no adapter for the `{}` protocol",
                    self.id,
                    api_name(model.api),
                ),
            ),
        }
    }

    /// Best-effort synchronous key lookup from the configured environment
    /// variables. Only [`Auth::api_key_env`] resolves synchronously; keyless
    /// and custom resolvers report `None` here (availability gating and the
    /// list-models probe both treat that as "no key").
    fn env_api_key(&self) -> Option<String> {
        self.auth.env_api_key()
    }
}
