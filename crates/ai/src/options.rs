//! Options controlling a single stream request.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::auth::{ProviderHeaders, RedactedHeaders, is_sensitive_header_name};
use crate::observer::RequestObserver;
use crate::types::{ReasoningOptions, ToolChoice};

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
    /// Request the provider's extended prompt-cache lifetime. Emitted only
    /// when the provider attests a long-retention wire shape; against one
    /// that does not, an explicit request is refused in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request.
    Long,
}

/// Per-request knobs. All fields are optional; providers ignore what they
/// don't support.
#[derive(Clone, Default)]
pub struct StreamOptions {
    /// Sampling temperature.
    ///
    /// `Some(..)` is an explicit request. On the Anthropic Messages protocol
    /// it is checked before dispatch against the support the provider
    /// declares
    /// ([`AnthropicCompat::temperature`](crate::AnthropicCompat::temperature)):
    /// a provider that attests no temperature support — or none alongside an
    /// enabled reasoning request — refuses it in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request. It is never silently dropped to make a request
    /// succeed; an omitted temperature leaves the request shape untouched.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    ///
    /// An explicit value is never silently reduced: if it exceeds the context
    /// window remaining after
    /// [`Context::estimate_tokens`](crate::Context::estimate_tokens), the
    /// request terminates in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// HTTP. When omitted, dispatch uses the lower of the model's known
    /// maximum output and remaining context; a zero-valued model limit remains
    /// unknown rather than becoming zero capacity. An explicit value continues
    /// to override the model's advertised output maximum when context permits.
    pub max_tokens: Option<u32>,
    /// API key override. Takes precedence over provider env-var resolution.
    pub api_key: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
    /// Maximum client-side retry attempts.
    pub max_retries: Option<u32>,
    /// Prompt-cache retention preference. `None` uses the provider default.
    ///
    /// `Some(..)` is an explicit request. [`CacheRetention::Long`] is checked
    /// before dispatch against the retention the provider declares
    /// ([`OpenAiCompat::cache_retention`](crate::OpenAiCompat::cache_retention));
    /// a provider that attests no long-retention wire shape refuses the
    /// request in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request — it is never silently dropped onto the endpoint's
    /// normal cache behavior.
    pub cache_retention: Option<CacheRetention>,
    /// How much reasoning to ask the model for.
    ///
    /// `None` — the default — is *no override*: the request carries no
    /// reasoning field and its payload is byte-for-byte what it was before
    /// this option existed. `Some(..)` is an explicit request, checked before
    /// dispatch against the levels the model attests
    /// ([`Model::reasoning`](crate::Model::reasoning)) and the request shape
    /// the provider declares
    /// ([`OpenAiCompat::reasoning_format`](crate::OpenAiCompat::reasoning_format)
    /// /
    /// [`AnthropicCompat::reasoning_format`](crate::AnthropicCompat::reasoning_format)).
    /// A request neither can honour terminates in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request — it is never quietly clamped onto a level the caller
    /// did not ask for.
    ///
    /// Note that [`ReasoningOptions::effort`] `Off` is not the same as `None`:
    /// `Off` asks the provider to actively disable reasoning.
    pub reasoning: Option<ReasoningOptions>,
    /// Which tool the model may or must call.
    ///
    /// `None` — the default — is *no override*: the request carries no
    /// `tool_choice` field and the provider's own default applies.
    /// `Some(..)` is an explicit request, checked before dispatch against the
    /// choices the provider declares
    /// ([`OpenAiCompat::tool_choice`](crate::OpenAiCompat::tool_choice) /
    /// [`AnthropicCompat::tool_choice`](crate::AnthropicCompat::tool_choice)).
    /// A choice it cannot express terminates in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request — it is never silently remapped onto a choice the
    /// caller did not ask for.
    pub tool_choice: Option<ToolChoice>,
    /// Stable conversation identifier routed onto the session-affinity shape
    /// the provider declares
    /// ([`OpenAiCompat::session_affinity`](crate::OpenAiCompat::session_affinity))
    /// — a cache-routing request field or session-affinity headers. An
    /// endpoint with no declared affinity receives it nowhere.
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
    /// Extra sampling parameters merged into the top level of an
    /// OpenAI-compatible request body.
    ///
    /// This is the escape hatch for open-model sampling controls the crate
    /// does not model — `top_p`, `top_k`, `min_p`, repetition penalties,
    /// `seed`, `stop`, and any other key an OpenAI-compatible endpoint
    /// accepts. Values are arbitrary JSON (integer, float, boolean, string,
    /// array, object, or null) and are sent verbatim; an empty map — the
    /// default — adds nothing to the request.
    ///
    /// The map can never override a field the adapter owns: keys covering the
    /// model, messages, tools, stream controls, output budget, reasoning,
    /// tool choice, caching, metadata, and authentication are reserved, and a
    /// reserved key fails in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) —
    /// naming the offending key — before authentication or any HTTP request.
    /// The Anthropic Messages protocol ignores this map entirely.
    pub sampling: BTreeMap<String, serde_json::Value>,
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
    /// A read-only observer notified immediately before each send and when
    /// response headers arrive. Everything it sees is redacted — URL query,
    /// credentials, and sensitive header values never reach it — and it
    /// cannot rewrite the request; see [`RequestObserver`]. A panicking
    /// observer is caught and logged, never failing or duplicating the
    /// request it watches.
    ///
    /// Observations are reported by the built-in protocol adapters through
    /// the shared executor; a custom
    /// [`ProtocolAdapter`](crate::ProtocolAdapter) owns its own dispatch and
    /// does not report unless it chooses to.
    pub observer: Option<Arc<dyn RequestObserver>>,
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
            .field("reasoning", &self.reasoning)
            .field("tool_choice", &self.tool_choice)
            .field("session_id", &self.session_id)
            .field("headers", &RedactedHeaders(&self.headers))
            .field("metadata", &RedactedMetadata(&self.metadata))
            .field("sampling", &RedactedMetadata(&self.sampling))
            .field("max_retry_delay", &self.max_retry_delay)
            .field("cancellation", &self.cancellation)
            .field("observer", &self.observer.as_ref().map(|_| "Some(..)"))
            .finish()
    }
}
