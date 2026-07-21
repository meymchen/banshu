//! Wire-protocol implementations.
//!
//! Each protocol implements [`ChatApi`]: given a resolved [`ApiRequest`], it
//! builds the provider payload, opens an SSE stream, and maps the response into
//! banshu's [`AssistantMessageEvent`](crate::AssistantMessageEvent)s.

pub mod anthropic_messages;
pub mod openai_completions;

use crate::options::StreamOptions;
use crate::provider::{AnthropicCompat, OpenAiCompat};
use crate::stream::{AssistantMessageEvent, MessageStream};
use crate::types::{
    AssistantMessage, Context, Cost, Diagnostic, DiagnosticCode, Model, ModelCost, StopReason,
    Usage,
};

/// A fully-resolved request handed to a [`ChatApi`] implementation.
///
/// The provider does auth resolution up front; the api layer only speaks the
/// wire protocol.
pub struct ApiRequest<'a> {
    /// The model to invoke (carries `base_url` and cost rates).
    pub model: &'a Model,
    /// The conversation context.
    pub context: &'a Context,
    /// Per-request options.
    pub options: &'a StreamOptions,
    /// Resolved API key, if any. `None` becomes an in-band error event.
    pub api_key: Option<String>,
    /// Shared HTTP client.
    pub http: reqwest::Client,
    /// Endpoint quirks declared by an OpenAI-compatible provider.
    pub openai_compat: OpenAiCompat,
    /// Endpoint quirks declared by an Anthropic-compatible provider.
    pub anthropic_compat: AnthropicCompat,
}

/// Mark `message` as failed and produce the terminal in-band `Error` event.
/// Shared by every protocol implementation.
pub(crate) fn fail(
    message: &mut AssistantMessage,
    kind: crate::ErrorKind,
    detail: &str,
) -> AssistantMessageEvent {
    message.stop_reason = StopReason::Error;
    message.error_message = Some(detail.to_string());
    message.error_kind = Some(kind);
    AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: message.clone(),
    }
}

/// [`fail`], plus attaching a diagnostic first — e.g. the raw (capped,
/// redacted) body behind a clean `detail` summary, or an in-stream protocol
/// violation's offending payload.
pub(crate) fn fail_with_diagnostic(
    message: &mut AssistantMessage,
    kind: crate::ErrorKind,
    detail: &str,
    diagnostic: Diagnostic,
) -> AssistantMessageEvent {
    message.diagnostics.push(diagnostic);
    fail(message, kind, detail)
}

/// Compute cost from token counts and per-million rates.
pub(crate) fn compute_cost(usage: &Usage, rates: &ModelCost) -> Cost {
    let per = |tokens: u64, rate: f64| tokens as f64 / 1_000_000.0 * rate;
    let input = per(usage.input, rates.input);
    let output = per(usage.output, rates.output);
    let cache_read = per(usage.cache_read, rates.cache_read);
    // 1h-TTL cache writes are billed at 2x the input rate, not the
    // (short-TTL) cache-write rate.
    let long_write = usage.cache_write_1h.unwrap_or(0);
    let short_write = usage.cache_write.saturating_sub(long_write);
    let cache_write = per(short_write, rates.cache_write) + per(long_write, rates.input * 2.0);
    Cost {
        input,
        output,
        cache_read,
        cache_write,
        total: input + output + cache_read + cache_write,
    }
}

/// Parse accumulated tool-call arguments; fall back to an empty object when the
/// fragments don't form valid JSON (e.g. an aborted stream).
pub(crate) fn parse_arguments(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| serde_json::json!({}))
}

/// Parse one SSE event's `data:` payload as JSON, or build the terminal
/// malformed-data detail/diagnostic a caller should `yield` via
/// [`fail_with_diagnostic`] before returning. Shared by both protocols —
/// each still checks for its own non-JSON sentinel (OpenAI's `[DONE]`) or
/// named `event:` field (Anthropic's `error`) before calling this, since
/// those are protocol-specific and don't belong in a shared parse step.
pub(crate) fn parse_sse_json(data: String) -> Result<serde_json::Value, (String, Diagnostic)> {
    serde_json::from_str(&data).map_err(|_| {
        (
            "malformed SSE data: not valid JSON".to_string(),
            Diagnostic::new(DiagnosticCode::ProtocolViolation, data),
        )
    })
}

/// A wire protocol that can stream a chat completion.
pub trait ChatApi: Send + Sync {
    /// Open a streamed completion. Per the contract, this never fails
    /// synchronously — request/transport failures are encoded as terminal
    /// `Error` events on the returned [`MessageStream`].
    fn stream(&self, request: ApiRequest<'_>) -> MessageStream;
}
