//! The `Models` registry — a collection of providers with lookup, auth-gated
//! availability, and dispatch. This is the ergonomic surface a consumer reaches
//! for: register providers once, then resolve and stream models by id without
//! caring which provider owns them.

use crate::discovery::{self, RefreshOutcome, RefreshReport};
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

    /// Models whose provider looks usable without further configuration — a set
    /// env-var key, or a keyless endpoint. Custom-resolver providers are
    /// excluded here (their resolver is only consultable asynchronously).
    pub fn available(&self) -> Vec<Model> {
        self.providers
            .iter()
            .filter(|p| p.is_available())
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
}
