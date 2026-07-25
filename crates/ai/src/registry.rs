//! The `Models` registry — an id-keyed collection of providers with lookup,
//! auth-gated availability, and dispatch. This is the ergonomic surface a
//! consumer reaches for: register providers once, then resolve and stream
//! models by id without caring which provider owns them.

use crate::discovery::{self, RefreshOutcome, RefreshReport};
use crate::options::StreamOptions;
use crate::provider::Provider;
use crate::stream::MessageStream;
use crate::types::{AssistantMessage, Context, Model};

/// A runtime collection of [`Provider`]s, keyed by provider id.
#[derive(Default)]
pub struct Models {
    providers: Vec<Provider>,
}

impl Models {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a provider (builder style); see [`set_provider`](Self::set_provider).
    pub fn with_provider(mut self, provider: Provider) -> Self {
        self.set_provider(provider);
        self
    }

    /// Add a provider, replacing any existing provider with the same id.
    pub fn set_provider(&mut self, provider: Provider) {
        match self.providers.iter_mut().find(|p| p.id() == provider.id()) {
            Some(slot) => *slot = provider,
            None => self.providers.push(provider),
        }
    }

    /// Remove and return the provider with this id, if registered.
    pub fn remove_provider(&mut self, id: &str) -> Option<Provider> {
        let index = self.providers.iter().position(|p| p.id() == id)?;
        Some(self.providers.remove(index))
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

    /// Models whose provider is currently usable — a set env-var key, a
    /// keyless endpoint, or a custom resolver whose `check` passes. Async
    /// because a custom [`AuthResolver`](crate::AuthResolver) may need real
    /// I/O to answer; a resolver error reads as unavailable.
    pub async fn available(&self) -> Vec<Model> {
        let checks =
            futures_util::future::join_all(self.providers.iter().map(Provider::check_available))
                .await;
        self.providers
            .iter()
            .zip(checks)
            .filter(|(_, available)| *available)
            .flat_map(|(provider, _)| provider.models())
            .collect()
    }

    /// Look up a single model by `(provider, id)`.
    pub fn get(&self, provider_id: &str, model_id: &str) -> Option<Model> {
        self.provider(provider_id)?
            .models()
            .into_iter()
            .find(|m| m.id == model_id)
    }

    /// Refresh every provider's model overlay from models.dev and the vendor
    /// list-models endpoints. Best-effort: never fails, records per-provider
    /// outcomes in the report, and leaves existing overlays untouched on
    /// failure.
    pub async fn refresh(&self) -> RefreshReport {
        self.refresh_from(discovery::MODELS_DEV_URL).await
    }

    /// [`refresh`](Self::refresh) against a specific models.dev catalog URL.
    pub async fn refresh_from(&self, catalog_url: &str) -> RefreshReport {
        // One models.dev fetch shared by every provider that wants it.
        let catalog = match self.providers.iter().find(|p| p.models_dev_id().is_some()) {
            Some(provider) => {
                Some(discovery::fetch_models_dev(provider.http_client(), catalog_url).await)
            }
            None => None,
        };
        let entries = futures_util::future::join_all(self.providers.iter().map(|provider| {
            let catalog = &catalog;
            async move {
                let outcome = match catalog {
                    Some(Ok(data)) => provider.apply_models_dev(data),
                    Some(Err(err)) if provider.models_dev_id().is_some() => {
                        RefreshOutcome::Failed(err.clone())
                    }
                    _ => RefreshOutcome::Skipped,
                };
                provider.refresh_entry(outcome).await
            }
        }))
        .await;
        RefreshReport { entries }
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

    /// Stream a completion to the end and return the final message.
    ///
    /// Like [`stream`](Self::stream), failures are in-band: inspect
    /// `stop_reason`/`error_kind` on the returned message rather than
    /// expecting a `Result`.
    pub async fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessage {
        self.stream(model, context, options).finish().await
    }
}
