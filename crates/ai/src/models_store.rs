//! Application-injectable persistence for successful model-discovery overlays.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::{Model, Result};

/// A persisted provider overlay and the validators associated with its last
/// successful Catalog Refresh.
#[derive(Clone, Debug)]
pub struct ModelsStoreEntry {
    /// Provider id this entry belongs to.
    pub provider_id: String,
    /// The complete effective model set served by the provider.
    pub models: Vec<Model>,
    /// Ids in `models` contributed by Probe rather than Catalog Refresh.
    /// Keeping their source preserves Catalog ← Catalog Refresh ← Probe after
    /// a process restart.
    pub probed_model_ids: Vec<String>,
    /// When the remote sources last successfully validated this entry.
    pub checked_at: SystemTime,
    /// Catalog response ETag, when supplied.
    pub etag: Option<String>,
    /// Catalog response Last-Modified value, when supplied.
    pub last_modified: Option<String>,
}

/// Policy for restoring and optionally refreshing persisted model overlays.
#[derive(Clone, Debug)]
pub struct RefreshOptions {
    /// Whether discovery may contact models.dev and provider list-models endpoints.
    pub allow_network: bool,
    /// Ignore `max_age` and perform network work when networking is allowed.
    pub force: bool,
    /// Skip network work when every restored entry is no older than this age.
    /// `None` always checks the network.
    pub max_age: Option<Duration>,
    /// Cancels outstanding Catalog Refresh and Probe work.
    pub cancellation: Option<CancellationToken>,
}

impl Default for RefreshOptions {
    fn default() -> Self {
        Self {
            allow_network: true,
            force: false,
            max_age: None,
            cancellation: None,
        }
    }
}

/// Application-injectable storage for model-discovery overlays.
#[async_trait]
pub trait ModelsStore: Send + Sync {
    /// Return the last successful overlay for `provider_id`, if one exists.
    async fn get(&self, provider_id: &str) -> Result<Option<ModelsStoreEntry>>;

    /// Replace the stored overlay for its provider.
    async fn set(&self, entry: ModelsStoreEntry) -> Result<()>;
}

/// Process-local [`ModelsStore`] adapter, useful for tests and applications
/// that want refresh policy without durable storage.
#[derive(Default)]
pub struct InMemoryModelsStore {
    entries: Mutex<HashMap<String, ModelsStoreEntry>>,
}

impl InMemoryModelsStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ModelsStore for InMemoryModelsStore {
    async fn get(&self, provider_id: &str) -> Result<Option<ModelsStoreEntry>> {
        Ok(self
            .entries
            .lock()
            .expect("models store lock poisoned")
            .get(provider_id)
            .cloned())
    }

    async fn set(&self, entry: ModelsStoreEntry) -> Result<()> {
        self.entries
            .lock()
            .expect("models store lock poisoned")
            .insert(entry.provider_id.clone(), entry);
        Ok(())
    }
}
