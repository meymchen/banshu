//! Dynamic model discovery: a models.dev catalog refresh (full metadata,
//! overrides + appends) layered under per-provider `/models` probes
//! (append-only, bare ids). See `docs/adr/0001` for why the vendor endpoints
//! are not the primary source.

use std::time::Duration;

use serde_json::Value;

pub(crate) const MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// Per-request timeout for discovery calls; a refresh is best-effort and
/// should never hang a caller indefinitely.
pub(crate) const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(30);

/// Outcome of one discovery source for one provider during a refresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// Fetched and applied to the provider's model overlay.
    Refreshed,
    /// Not attempted: the provider has no models.dev id (catalog refresh) or
    /// no resolvable API key (probe).
    Skipped,
    /// The provider answered 404/405/501 — it has no list-models endpoint.
    EndpointUnsupported,
    /// The attempt failed; the existing overlay is untouched.
    Failed(String),
}

/// One provider's outcomes for both discovery sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshEntry {
    /// The provider id this entry reports on.
    pub provider: String,
    /// The models.dev catalog-refresh layer.
    pub catalog: RefreshOutcome,
    /// The vendor `/models` probe layer.
    pub probe: RefreshOutcome,
}

/// Best-effort report of a [`Models::refresh`](crate::Models::refresh).
/// Refreshing never fails as a whole: failures are recorded per provider and
/// never disturb previously discovered models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshReport {
    /// One entry per registered provider, in registration order.
    pub entries: Vec<RefreshEntry>,
}

/// A provider's list-models response: `{"data":[{"id":...}]}` in both the
/// OpenAI and Anthropic dialects (the latter adds `display_name`).
#[derive(serde::Deserialize)]
pub(crate) struct ListModelsResponse {
    pub data: Vec<ListedModel>,
}

#[derive(serde::Deserialize)]
pub(crate) struct ListedModel {
    pub id: String,
    pub display_name: Option<String>,
}

/// Fetch and parse a models.dev `api.json`.
pub(crate) async fn fetch_models_dev(http: &reqwest::Client, url: &str) -> Result<Value, String> {
    let response = http
        .get(url)
        .timeout(DISCOVERY_TIMEOUT)
        .send()
        .await
        .map_err(|err| err.to_string())?;
    if !response.status().is_success() {
        return Err(format!("models.dev returned HTTP {}", response.status()));
    }
    response.json().await.map_err(|err| err.to_string())
}
