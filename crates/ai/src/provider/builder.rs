//! Validated construction of custom providers.
//!
//! [`ProviderBuilder`] is the seam for plugging in a third-party protocol or
//! assembling a provider the bundled constructors don't cover — a local
//! server, a mixed-protocol gateway, or a provider with caller-supplied
//! models. Every invariant is checked in [`build`](ProviderBuilder::build) and
//! reported as an [`Error::Config`]: construction never panics.

use std::collections::HashMap;
use std::sync::Arc;

use super::{AnthropicCompat, OpenAiCompat, Provider};
use crate::api::{ProtocolAdapter, api_name};
use crate::auth::{Auth, ProviderHeaders};
use crate::error::{Error, Result};
use crate::http;
use crate::types::{ApiKind, Model};

/// Builds a validated [`Provider`].
///
/// Required: a non-empty provider id (from
/// [`Provider::builder`](crate::Provider::builder)) and at least one
/// [`ProtocolAdapter`], at most one per [`ApiKind`]. Optional: auth (defaults
/// to [`Auth::keyless`]), provider-level default headers (empty by default),
/// caller-supplied models, endpoint quirks, and a models.dev id for catalog
/// refresh. Each registered model must belong to this provider id and speak a
/// protocol an adapter covers.
pub struct ProviderBuilder {
    id: String,
    name: String,
    base_url: String,
    auth: Auth,
    adapters: Vec<Arc<dyn ProtocolAdapter>>,
    headers: ProviderHeaders,
    models: Vec<Model>,
    openai_compat: OpenAiCompat,
    anthropic_compat: AnthropicCompat,
    models_dev_id: Option<String>,
}

impl ProviderBuilder {
    pub(crate) fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            base_url: base_url.into(),
            auth: Auth::keyless(),
            adapters: Vec::new(),
            headers: ProviderHeaders::new(),
            models: Vec::new(),
            openai_compat: OpenAiCompat::default(),
            anthropic_compat: AnthropicCompat::default(),
            models_dev_id: None,
        }
    }

    /// Register a protocol adapter. At most one adapter per [`ApiKind`] —
    /// duplicates fail the build.
    pub fn adapter(mut self, adapter: Arc<dyn ProtocolAdapter>) -> Self {
        self.adapters.push(adapter);
        self
    }

    /// How the provider resolves credentials. Defaults to [`Auth::keyless`];
    /// use [`Auth::api_key_env`] or [`Auth::custom`] for authenticated
    /// endpoints.
    pub fn auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// Provider-level default headers, applied to every request below the
    /// auth/request layers. Empty by default. (`None` values are currently
    /// no-ops; deletion semantics land with the headers-merge work.)
    pub fn headers(mut self, headers: ProviderHeaders) -> Self {
        self.headers = headers;
        self
    }

    /// Register one model the provider serves. Its `provider` must equal the
    /// provider id and its `api` must have a registered adapter.
    pub fn model(mut self, model: Model) -> Self {
        self.models.push(model);
        self
    }

    /// Register several models; see [`model`](Self::model).
    pub fn models(mut self, models: impl IntoIterator<Item = Model>) -> Self {
        self.models.extend(models);
        self
    }

    /// Configure the endpoint quirks of this OpenAI-compatible provider.
    pub fn openai_compat(mut self, compat: OpenAiCompat) -> Self {
        self.openai_compat = compat;
        self
    }

    /// Configure the endpoint quirks of this Anthropic-compatible provider.
    pub fn anthropic_compat(mut self, compat: AnthropicCompat) -> Self {
        self.anthropic_compat = compat;
        self
    }

    /// Set the models.dev provider key used by the catalog-refresh layer of
    /// dynamic discovery. Without one the provider skips that layer.
    pub fn models_dev_id(mut self, id: impl Into<String>) -> Self {
        self.models_dev_id = Some(id.into());
        self
    }

    /// Validate and build the provider.
    ///
    /// Fails with [`Error::Config`] when the id is empty, no adapter is
    /// registered, two adapters claim the same [`ApiKind`], or a registered
    /// model belongs to another provider id or speaks a protocol no adapter
    /// covers.
    pub fn build(self) -> Result<Provider> {
        if self.id.trim().is_empty() {
            return Err(Error::Config("provider id must not be empty".to_string()));
        }
        let Some(primary) = self.adapters.first().map(|adapter| adapter.kind()) else {
            return Err(Error::Config(format!(
                "provider `{}` needs at least one protocol adapter",
                self.id
            )));
        };
        let mut adapters: HashMap<ApiKind, Arc<dyn ProtocolAdapter>> = HashMap::new();
        for adapter in self.adapters {
            let kind = adapter.kind();
            if adapters.insert(kind, adapter).is_some() {
                return Err(Error::Config(format!(
                    "provider `{}` has more than one adapter for the `{}` protocol",
                    self.id,
                    api_name(kind),
                )));
            }
        }
        for model in &self.models {
            if model.provider != self.id {
                return Err(Error::Config(format!(
                    "model `{}` belongs to provider `{}`, not `{}`",
                    model.id, model.provider, self.id,
                )));
            }
            if !adapters.contains_key(&model.api) {
                return Err(Error::Config(format!(
                    "model `{}` speaks the `{}` protocol, which provider `{}` has no adapter for",
                    model.id,
                    api_name(model.api),
                    self.id,
                )));
            }
        }
        Ok(Provider {
            id: self.id,
            name: self.name,
            base_url: self.base_url,
            auth: self.auth,
            api_kind: primary,
            adapters,
            headers: self.headers,
            models: self.models,
            http: http::build_client(),
            openai_compat: self.openai_compat,
            anthropic_compat: self.anthropic_compat,
            models_dev_id: self.models_dev_id,
            overlay: std::sync::RwLock::default(),
        })
    }
}
