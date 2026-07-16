//! The `Models` registry — a collection of providers with lookup, auth-gated
//! availability, and dispatch. This is the ergonomic surface a consumer reaches
//! for: register providers once, then resolve and stream models by id without
//! caring which provider owns them.

use crate::options::StreamOptions;
use crate::provider::Provider;
use crate::stream::MessageStream;
use crate::types::{Context, Model};

/// A runtime collection of [`Provider`]s.
#[derive(Default)]
pub struct Models {
    providers: Vec<Provider>,
}

impl Models {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a provider (builder style).
    pub fn with_provider(mut self, provider: Provider) -> Self {
        self.providers.push(provider);
        self
    }

    /// Add a provider in place.
    pub fn register(&mut self, provider: Provider) {
        self.providers.push(provider);
    }

    /// All registered providers.
    pub fn providers(&self) -> &[Provider] {
        &self.providers
    }

    /// The registered provider with this id, if any.
    pub fn provider(&self, id: &str) -> Option<&Provider> {
        self.providers.iter().find(|p| p.id() == id)
    }

    /// Every model across all registered providers.
    pub fn models(&self) -> Vec<Model> {
        self.providers.iter().flat_map(Provider::models).collect()
    }

    /// Models whose provider has a resolvable API key (from options env vars).
    pub fn available(&self) -> Vec<Model> {
        self.providers
            .iter()
            .filter(|p| p.has_env_api_key())
            .flat_map(Provider::models)
            .collect()
    }

    /// Look up a single model by `(provider, id)`.
    pub fn get(&self, provider_id: &str, model_id: &str) -> Option<Model> {
        self.provider(provider_id)?
            .models()
            .into_iter()
            .find(|m| m.id == model_id)
    }

    /// Stream a completion, dispatching to the provider that owns `model`
    /// (matched on `model.provider`). An unknown provider yields an in-band
    /// error, keeping the non-failing stream contract.
    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> MessageStream {
        match self.provider(&model.provider) {
            Some(provider) => provider.stream(model, context, options),
            None => MessageStream::immediate_error(
                &model.id,
                &model.provider,
                &format!("no registered provider owns model `{}`", model.id),
            ),
        }
    }
}
