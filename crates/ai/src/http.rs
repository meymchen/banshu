//! Shared HTTP client construction and pre-stream retry primitives.
//!
//! Centralizes client creation so every provider shares one connection pool
//! and TLS backend, plus the classified-failure/backoff building blocks
//! ([`send_once`] + [`retry_delay`]) [`crate::executor`] loops over. SSE
//! decoding lives in [`crate::sse`]; the executor wires the two together.

use std::time::Duration;

use serde_json::Value;

use crate::error::{ErrorKind, classify_status};
use crate::types::{Diagnostic, DiagnosticCode};

/// Retries attempted when `StreamOptions::max_retries` is unset. Matches the
/// Anthropic/OpenAI official SDK default.
pub(crate) const DEFAULT_MAX_RETRIES: u32 = 2;

/// Default cap on a server-requested `Retry-After` wait when
/// `StreamOptions::max_retry_delay` is unset.
pub(crate) const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

/// First-retry backoff ceiling; doubles per retry.
const BACKOFF_BASE: Duration = Duration::from_millis(500);
/// Computed backoff never exceeds this.
const BACKOFF_CAP: Duration = Duration::from_secs(8);

/// Non-2xx error bodies are read up to this many Unicode characters before
/// any parsing, redaction, or diagnostic-wrapping — a defensive cap on how
/// much of a provider's response we ever hold onto, independent of the
/// stricter per-diagnostic cap `Diagnostic::new` applies on top.
const MAX_ERROR_BODY_CHARS: usize = 4096;

/// A classified pre-stream failure from [`send_once`].
pub(crate) struct SendFailure {
    /// What went wrong, for retryability decisions and the terminal message.
    pub kind: ErrorKind,
    /// Short, human-readable summary safe for `AssistantMessage.error_message`
    /// — extracted JSON `message`/`code` when available, else a bare status
    /// fallback. Never the raw body; that's confined to `diagnostics`.
    pub detail: String,
    /// The raw (capped, redacted) error body, if any, for
    /// `AssistantMessage.diagnostics` — never concatenated into `detail`.
    pub diagnostics: Vec<Diagnostic>,
    /// Server-provided retry hint, if any.
    pub retry_after: Option<Duration>,
}

/// Send one attempt: a transport error or non-2xx status becomes a classified
/// [`SendFailure`] (consuming the error body for detail/diagnostics and
/// Retry-After).
pub(crate) async fn send_once(
    builder: reqwest::RequestBuilder,
) -> Result<reqwest::Response, SendFailure> {
    match builder.send().await {
        Ok(response) if response.status().is_success() => Ok(response),
        Ok(response) => {
            let status = response.status();
            let retry_after = retry_after(response.headers());
            let raw_body = response.text().await.unwrap_or_default();
            let body: String = raw_body.chars().take(MAX_ERROR_BODY_CHARS).collect();
            let parsed = parse_error_body(status.as_u16(), &body);
            Err(SendFailure {
                kind: classify_status(status.as_u16(), &body),
                detail: parsed.summary,
                diagnostics: parsed.diagnostic.into_iter().collect(),
                retry_after,
            })
        }
        Err(err) => Err(SendFailure {
            kind: ErrorKind::Transport,
            detail: format!("request failed: {err}"),
            diagnostics: Vec::new(),
            retry_after: None,
        }),
    }
}

/// A non-2xx body split into what's safe to surface directly and what must
/// stay confined to a diagnostic.
struct ErrorBody {
    summary: String,
    diagnostic: Option<Diagnostic>,
}

/// Build the human-readable summary and diagnostic for an already-capped
/// error body: prefer a JSON error object's `message`/`code`, falling back to
/// a bare status when the body isn't JSON or carries neither field. The raw
/// body itself only ever reaches the diagnostic, which independently redacts
/// secrets/base64 and re-caps at 1024 chars.
fn parse_error_body(status: u16, body: &str) -> ErrorBody {
    let summary = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| json_error_summary(&value))
        .map(|extracted| format!("HTTP {status}: {extracted}"))
        .unwrap_or_else(|| format!("HTTP {status}"));
    let diagnostic =
        (!body.is_empty()).then(|| Diagnostic::new(DiagnosticCode::ProviderError, body));
    ErrorBody {
        summary,
        diagnostic,
    }
}

/// Extract a clean `message (code)`-style summary from a JSON error object,
/// shaped either `{"error": {"message", "code"|"type"}}` (OpenAI/Anthropic)
/// or flat `{"message", "code"}`. `None` when neither field is present.
pub(crate) fn json_error_summary(value: &Value) -> Option<String> {
    let error = value.get("error").unwrap_or(value);
    let message = error.get("message").and_then(Value::as_str);
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| error.get("type").and_then(Value::as_str));
    match (message, code) {
        (Some(message), Some(code)) => Some(format!("{message} ({code})")),
        (Some(message), None) => Some(message.to_string()),
        (None, Some(code)) => Some(code.to_string()),
        (None, None) => None,
    }
}

/// Extract a retry hint: `retry-after-ms` (milliseconds) first, then
/// `retry-after` as either decimal seconds or an RFC 7231 HTTP-date. A date
/// already in the past yields a zero duration rather than `None`, so it's
/// still recognized as "the server gave an instruction" (falls through to
/// backoff in [`retry_delay`], same as an explicit zero).
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    if let Some(ms) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
    {
        return Some(Duration::from_millis(ms));
    }
    let raw = headers.get("retry-after")?.to_str().ok()?.trim();
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let target = httpdate::parse_http_date(raw).ok()?;
    Some(
        target
            .duration_since(std::time::SystemTime::now())
            .unwrap_or(Duration::ZERO),
    )
}

/// The outcome of deciding how long to wait before a retry.
pub(crate) enum RetryDecision {
    /// Sleep this long before re-sending.
    Sleep(Duration),
    /// The server's `Retry-After` asked for longer than the configured cap;
    /// the caller should fail as `RateLimited` instead of sleeping.
    ExceedsCap {
        /// What the server actually asked for.
        requested: Duration,
    },
}

/// Decide the wait before retry `attempt` (1-based): a server Retry-After
/// within `(0, max_retry_delay]` is used verbatim (it's a scheduling
/// instruction — no jitter); one beyond the cap is rejected via
/// `ExceedsCap` rather than silently downgraded to computed backoff, so the
/// caller can surface it as a terminal error instead of waiting anyway.
/// Otherwise: exponential backoff with full jitter,
/// `random(0..=500ms·2ⁿ⁻¹)` capped at 8s.
pub(crate) fn retry_delay(
    attempt: u32,
    retry_after: Option<Duration>,
    max_retry_delay: Duration,
) -> RetryDecision {
    if let Some(hint) = retry_after
        && hint > Duration::ZERO
    {
        return if hint <= max_retry_delay {
            RetryDecision::Sleep(hint)
        } else {
            RetryDecision::ExceedsCap { requested: hint }
        };
    }
    let ceiling = BACKOFF_BASE
        .saturating_mul(1u32 << (attempt.saturating_sub(1)).min(10))
        .min(BACKOFF_CAP);
    RetryDecision::Sleep(Duration::from_millis(fastrand::u64(
        0..=ceiling.as_millis() as u64,
    )))
}

/// Build the default shared HTTP client.
pub(crate) fn build_client() -> reqwest::Client {
    reqwest::Client::builder().build().unwrap_or_default()
}
