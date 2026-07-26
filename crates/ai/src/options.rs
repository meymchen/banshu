//! Options controlling a single stream request.

use std::collections::BTreeMap;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::auth::{ProviderHeaders, RedactedHeaders, is_sensitive_header_name};

struct RedactedMetadata<'a>(&'a BTreeMap<String, serde_json::Value>);

impl std::fmt::Debug for RedactedMetadata<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = formatter.debug_map();
        for (name, value) in self.0 {
            if is_sensitive_header_name(name) {
                map.entry(name, &"[REDACTED]");
            } else {
                map.entry(name, &RedactedJson(value));
            }
        }
        map.finish()
    }
}

struct RedactedJson<'a>(&'a serde_json::Value);

impl std::fmt::Debug for RedactedJson<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            serde_json::Value::Array(values) => formatter
                .debug_list()
                .entries(values.iter().map(RedactedJson))
                .finish(),
            serde_json::Value::Object(values) => {
                let mut map = formatter.debug_map();
                for (name, value) in values {
                    if is_sensitive_header_name(name) {
                        map.entry(name, &"[REDACTED]");
                    } else {
                        map.entry(name, &RedactedJson(value));
                    }
                }
                map.finish()
            }
            value => std::fmt::Debug::fmt(value, formatter),
        }
    }
}

/// Requested lifetime for provider-managed prompt caches.
///
/// Providers that cache prompts automatically may ignore this option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheRetention {
    /// Do not send optional cache-routing or retention controls.
    Disabled,
    /// Use the provider's short-lived/default prompt cache.
    Short,
    /// Request the provider's extended prompt-cache lifetime when supported.
    Long,
}

/// Per-request knobs. All fields are optional; providers ignore what they
/// don't support.
#[derive(Clone, Default)]
pub struct StreamOptions {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// API key override. Takes precedence over provider env-var resolution.
    pub api_key: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
    /// Maximum client-side retry attempts.
    pub max_retries: Option<u32>,
    /// Prompt-cache retention preference. `None` uses the provider default.
    pub cache_retention: Option<CacheRetention>,
    /// Stable conversation identifier used by providers that support
    /// cache-routing keys or session-affinity headers.
    pub session_id: Option<String>,
    /// Request-level custom headers. The fixed priority is protocol defaults
    /// → provider defaults → [`Model::headers`](crate::Model::headers) →
    /// [`ResolvedAuth::headers`](crate::ResolvedAuth::headers) → these
    /// headers. Names compare case-insensitively, and `None` deletes a
    /// same-named lower-priority header. This highest-priority layer may
    /// deliberately override authentication headers.
    pub headers: ProviderHeaders,
    /// Caller-defined request metadata for adapters and diagnostics.
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// Cap on how long a server-requested `Retry-After` may ask the client to
    /// wait before the executor gives up and fails as `RateLimited` instead of
    /// sleeping. `None` uses the default of 60 seconds.
    pub max_retry_delay: Option<Duration>,
    /// Cancels the request when triggered. Covers the auth-resolver wait, HTTP
    /// connect and response-header wait, retry backoff sleeps, and SSE body
    /// reads. A cancelled stream terminates with `Error { reason: Aborted }`,
    /// preserving whatever content had already streamed; no further retries
    /// are attempted.
    pub cancellation: Option<CancellationToken>,
}

impl std::fmt::Debug for StreamOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamOptions")
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("cache_retention", &self.cache_retention)
            .field("session_id", &self.session_id)
            .field("headers", &RedactedHeaders(&self.headers))
            .field("metadata", &RedactedMetadata(&self.metadata))
            .field("max_retry_delay", &self.max_retry_delay)
            .field("cancellation", &self.cancellation)
            .finish()
    }
}
