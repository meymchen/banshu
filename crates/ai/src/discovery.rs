//! Dynamic model discovery: a models.dev catalog refresh (full metadata,
//! overrides + appends) layered under per-provider `/models` probes
//! (append-only, bare ids). See `docs/adr/0001` for why the vendor endpoints
//! are not the primary source.

use std::time::Duration;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

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

/// Validators carried by a models.dev response.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Validators {
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

/// A successful conditional Catalog Refresh.
pub(crate) enum CatalogResponse {
    Modified(Value, Validators),
    NotModified(Validators),
}

fn validators(response: &reqwest::Response) -> Validators {
    Validators {
        etag: response
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string),
        last_modified: response
            .headers()
            .get(reqwest::header::LAST_MODIFIED)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string),
    }
}

/// Conditionally fetch and parse models.dev, respecting cancellation.
pub(crate) async fn fetch_models_dev_with(
    http: &reqwest::Client,
    url: &str,
    prior: Validators,
    cancellation: Option<&CancellationToken>,
) -> Result<CatalogResponse, String> {
    let mut request = http.get(url).timeout(DISCOVERY_TIMEOUT);
    if let Some(etag) = &prior.etag {
        request = request.header(reqwest::header::IF_NONE_MATCH, etag);
    }
    if let Some(last_modified) = &prior.last_modified {
        request = request.header(reqwest::header::IF_MODIFIED_SINCE, last_modified);
    }
    let response = crate::cancel::race(cancellation, request.send())
        .await
        .map_err(|_| "cancelled".to_string())?
        .map_err(|err| err.to_string())?;
    let next_validators = validators(&response);
    if response.status() == reqwest::StatusCode::NOT_MODIFIED {
        return Ok(CatalogResponse::NotModified(Validators {
            etag: next_validators.etag.or(prior.etag),
            last_modified: next_validators.last_modified.or(prior.last_modified),
        }));
    }
    if !response.status().is_success() {
        return Err(format!("models.dev returned HTTP {}", response.status()));
    }
    let data = crate::cancel::race(cancellation, response.json())
        .await
        .map_err(|_| "cancelled".to_string())?
        .map_err(|err| err.to_string())?;
    Ok(CatalogResponse::Modified(data, next_validators))
}

/// Fetch and parse a models.dev `api.json`.
pub(crate) async fn fetch_models_dev(http: &reqwest::Client, url: &str) -> Result<Value, String> {
    match fetch_models_dev_with(http, url, Validators::default(), None).await? {
        CatalogResponse::Modified(data, _) => Ok(data),
        CatalogResponse::NotModified(_) => {
            Err("models.dev returned HTTP 304 without validators".into())
        }
    }
}
