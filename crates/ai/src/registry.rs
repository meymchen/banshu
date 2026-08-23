//! The `Models` registry — an id-keyed collection of providers with lookup,
//! auth-gated availability, and dispatch. This is the ergonomic surface a
//! consumer reaches for: register providers once, then resolve and stream
//! models by id without caring which provider owns them.

use std::sync::Arc;

use crate::discovery::{self, RefreshOutcome, RefreshReport};
use crate::models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry, RefreshOptions};
use crate::options::StreamOptions;
use crate::provider::Provider;
use crate::stream::MessageStream;
use crate::types::{AssistantMessage, Context, Model};

/// A runtime collection of [`Provider`]s, keyed by provider id.
pub struct Models {
    providers: Vec<Provider>,
    models_store: Arc<dyn ModelsStore>,
}

impl Default for Models {
    fn default() -> Self {
        Self {
            providers: Vec::new(),
            models_store: Arc::new(InMemoryModelsStore::new()),
        }
    }
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

    /// Replace the model-overlay store used by refresh operations.
    pub fn with_models_store(mut self, store: Arc<dyn ModelsStore>) -> Self {
        self.models_store = store;
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

    /// The subset of [`models`](Self::models) attested to support tool calling
    /// — the safe pool for an agent loop. Probe-discovered and caller-built
    /// models report [`CapabilitySupport::Unknown`](crate::CapabilitySupport)
    /// and are excluded; they remain explicitly selectable via
    /// [`models`](Self::models) and [`get`](Self::get).
    pub fn agent_models(&self) -> Vec<Model> {
        self.models()
            .into_iter()
            .filter(|model| model.capabilities.tool_calling == crate::CapabilitySupport::Supported)
            .collect()
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

    /// [`refresh`](Self::refresh) under an explicit restore/network policy.
    pub async fn refresh_with(&self, options: &RefreshOptions) -> RefreshReport {
        self.refresh_from_with(discovery::MODELS_DEV_URL, options)
            .await
    }

    /// [`refresh`](Self::refresh) against a specific models.dev catalog URL.
    pub async fn refresh_from(&self, catalog_url: &str) -> RefreshReport {
        self.refresh_from_with(catalog_url, &RefreshOptions::default())
            .await
    }

    /// Refresh against `catalog_url` under an explicit restore/network policy.
    pub async fn refresh_from_with(
        &self,
        catalog_url: &str,
        options: &RefreshOptions,
    ) -> RefreshReport {
        let mut stored = Vec::with_capacity(self.providers.len());
        for provider in &self.providers {
            let entry = match self.models_store.get(provider.id()).await {
                Ok(Some(entry)) => {
                    if entry.provider_id == provider.id() {
                        provider.restore_overlay(&entry);
                    } else {
                        tracing::warn!(
                            provider = provider.id(),
                            stored_provider = %entry.provider_id,
                            "ignored model overlay stored under the wrong provider id"
                        );
                    }
                    Some(entry)
                }
                Ok(None) => None,
                Err(error) => {
                    tracing::warn!(provider = provider.id(), %error, "model overlay restore failed");
                    None
                }
            };
            stored.push(entry);
        }
        let all_fresh = options.max_age.is_some_and(|max_age| {
            !stored.is_empty()
                && stored.iter().all(|entry| {
                    entry.as_ref().is_some_and(|entry| {
                        entry.checked_at.elapsed().unwrap_or_default() <= max_age
                    })
                })
        });
        if !options.allow_network || (!options.force && all_fresh) {
            return RefreshReport {
                entries: self
                    .providers
                    .iter()
                    .map(|provider| crate::RefreshEntry {
                        provider: provider.id().to_string(),
                        catalog: RefreshOutcome::Skipped,
                        probe: RefreshOutcome::Skipped,
                    })
                    .collect(),
            };
        }
        // One models.dev fetch shared by every provider that wants it.
        let catalog_entries: Vec<_> = self
            .providers
            .iter()
            .zip(&stored)
            .filter(|(provider, _)| provider.models_dev_id().is_some())
            .collect();
        let validator_entry = catalog_entries
            .iter()
            .filter_map(|(_, entry)| entry.as_ref())
            .filter(|entry| entry.etag.is_some() || entry.last_modified.is_some())
            .max_by_key(|entry| entry.checked_at);
        let prior_validators = validator_entry
            .map(|entry| discovery::Validators {
                etag: entry.etag.clone(),
                last_modified: entry.last_modified.clone(),
            })
            .unwrap_or_default();
        let validators_cover_every_provider = validator_entry.is_none_or(|selected| {
            catalog_entries.iter().all(|(_, entry)| {
                entry.as_ref().is_some_and(|entry| {
                    entry.etag == selected.etag && entry.last_modified == selected.last_modified
                })
            })
        });
        let catalog = match self.providers.iter().find(|p| p.models_dev_id().is_some()) {
            Some(provider) => {
                let conditional = discovery::fetch_models_dev_with(
                    provider.http_client(),
                    catalog_url,
                    prior_validators,
                    options.cancellation.as_ref(),
                )
                .await;
                match conditional {
                    Ok(discovery::CatalogResponse::NotModified(_))
                        if !validators_cover_every_provider =>
                    {
                        Some(
                            discovery::fetch_models_dev_with(
                                provider.http_client(),
                                catalog_url,
                                discovery::Validators::default(),
                                options.cancellation.as_ref(),
                            )
                            .await,
                        )
                    }
                    outcome => Some(outcome),
                }
            }
            None => None,
        };
        let entries = futures_util::future::join_all(self.providers.iter().zip(stored.iter()).map(|(provider, prior)| {
            let catalog = &catalog;
            let store = &self.models_store;
            let cancellation = options.cancellation.as_ref();
            async move {
                let outcome = match catalog {
                    Some(Ok(discovery::CatalogResponse::Modified(data, _))) => provider.apply_models_dev(data),
                    Some(Ok(discovery::CatalogResponse::NotModified(_))) if provider.models_dev_id().is_some() => RefreshOutcome::Refreshed,
                    Some(Err(err)) if provider.models_dev_id().is_some() => {
                        RefreshOutcome::Failed(err.clone())
                    }
                    _ => RefreshOutcome::Skipped,
                };
                let probe = provider.probe_models_with(cancellation).await;
                let successful = matches!(outcome, RefreshOutcome::Refreshed)
                    || matches!(probe, RefreshOutcome::Refreshed);
                if successful {
                    let validators = match catalog {
                        Some(Ok(discovery::CatalogResponse::Modified(_, validators)))
                        | Some(Ok(discovery::CatalogResponse::NotModified(validators)))
                            if provider.models_dev_id().is_some() => validators.clone(),
                        _ => discovery::Validators {
                            etag: prior.as_ref().and_then(|entry| entry.etag.clone()),
                            last_modified: prior.as_ref().and_then(|entry| entry.last_modified.clone()),
                        },
                    };
                    let (models, probed_model_ids) = provider.overlay_snapshot();
                    if let Err(error) = store.set(ModelsStoreEntry {
                        provider_id: provider.id().to_string(),
                        models,
                        probed_model_ids,
                        checked_at: std::time::SystemTime::now(),
                        etag: validators.etag,
                        last_modified: validators.last_modified,
                    }).await {
                        tracing::warn!(provider = provider.id(), %error, "model overlay persistence failed");
                    }
                }
                crate::RefreshEntry {
                    provider: provider.id().to_string(),
                    catalog: outcome,
                    probe,
                }
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

    /// Log in to a provider interactively, driving the user-facing half
    /// through `interaction`, and store the resulting credential. This is a
    /// plain call, not a message stream; cancellation and timeout live on the
    /// [`AuthInteraction`](crate::AuthInteraction).
    pub async fn login(
        &self,
        provider_id: &str,
        interaction: &crate::AuthInteraction,
    ) -> crate::Result<crate::OAuthCredential> {
        self.oauth_session(provider_id)?.login(interaction).await
    }

    /// Delete the provider's stored credential. Logging out of a provider
    /// that was never logged in to is not an error.
    pub async fn logout(&self, provider_id: &str) -> crate::Result<()> {
        self.oauth_session(provider_id)?.logout().await
    }

    /// Whether the provider holds a usable credential — a stored OAuth
    /// credential (expired ones count; request-time refresh renews them or
    /// fails loudly) or, for API-key providers, a resolvable key.
    pub async fn check_auth(&self, provider_id: &str) -> crate::Result<bool> {
        match self.provider(provider_id) {
            Some(provider) => Ok(provider.check_available().await),
            None => Err(crate::Error::Config(format!(
                "no registered provider `{provider_id}`"
            ))),
        }
    }

    /// Refresh the provider's stored OAuth credential now, outside any
    /// request. Concurrent callers — including in-flight requests — share the
    /// same single-flight refresh operation.
    pub async fn refresh_credential(
        &self,
        provider_id: &str,
    ) -> crate::Result<crate::OAuthCredential> {
        self.oauth_session(provider_id)?.refresh().await
    }

    /// The OAuth session of the named provider, or a config error when the
    /// provider is unknown or has no OAuth login configured.
    fn oauth_session(&self, provider_id: &str) -> crate::Result<crate::OAuthSession> {
        let provider = self.provider(provider_id).ok_or_else(|| {
            crate::Error::Config(format!("no registered provider `{provider_id}`"))
        })?;
        provider.oauth_session().ok_or_else(|| {
            crate::Error::Config(format!(
                "provider `{provider_id}` has no OAuth login configured"
            ))
        })
    }
}
