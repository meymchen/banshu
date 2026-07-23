//! Options controlling a single stream request.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

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
#[derive(Debug, Clone, Default)]
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
