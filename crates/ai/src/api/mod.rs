//! Wire-protocol adapters — the public extension seam.
//!
//! Each protocol implements [`ProtocolAdapter`]: given a fully-resolved
//! [`PreparedRequest`], it translates between its own wire format and the
//! shared [`ProtocolEvent`] vocabulary. The provider layer drives the returned
//! [`ProtocolEventStream`] through the internal message assembler, which owns
//! block ordering, public `content_index` assignment, cancellation, and the
//! final [`AssistantMessage`].

pub mod anthropic_messages;
pub mod openai_completions;

mod assembler;
mod protocol_event;

use std::sync::Arc;

use futures_util::StreamExt;

pub use protocol_event::{ProtocolEvent, ProtocolEventStream};

use self::assembler::{MessageAssembler, is_terminal};
use crate::auth::{Auth, ProviderHeaders, ResolvedAuth};
use crate::cancel;
use crate::error::ErrorKind;
use crate::options::StreamOptions;
use crate::provider::{AnthropicCompat, OpenAiCompat};
use crate::stream::{AssistantMessageEvent, MessageStream};
use crate::types::{
    ApiKind, AssistantMessage, Context, Cost, Diagnostic, DiagnosticCode, Model, ModelCost,
    StopReason, Usage,
};

/// A fully-resolved request handed to a [`ProtocolAdapter`].
///
/// Fields are private and read through accessors only: external adapters can
/// depend on these values, but never on construction details, so the crate
/// stays free to add resolution inputs without a SemVer break. Auth is
/// resolved up front by the provider (an explicit
/// [`StreamOptions::api_key`] wins over the resolver); a resolution failure
/// never reaches the adapter — it terminates the stream in-band first.
pub struct PreparedRequest {
    model: Model,
    context: Context,
    options: StreamOptions,
    auth: ResolvedAuth,
    headers: ProviderHeaders,
    http: reqwest::Client,
    openai_compat: OpenAiCompat,
    anthropic_compat: AnthropicCompat,
}

impl PreparedRequest {
    /// The model to invoke (carries `base_url` and cost rates).
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The conversation context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Per-request options.
    pub fn options(&self) -> &StreamOptions {
        &self.options
    }

    /// The credentials and endpoint override resolved for this request.
    pub fn auth(&self) -> &ResolvedAuth {
        &self.auth
    }

    /// Provider-level default headers, applied below [`ResolvedAuth::headers`]
    /// in the priority chain. (`None` values are currently no-ops; deletion
    /// semantics land with the headers-merge work.)
    pub fn headers(&self) -> &ProviderHeaders {
        &self.headers
    }

    /// The provider's shared HTTP client (one connection pool per provider).
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    /// Endpoint quirks declared by an OpenAI-compatible provider.
    pub fn openai_compat(&self) -> OpenAiCompat {
        self.openai_compat
    }

    /// Endpoint quirks declared by an Anthropic-compatible provider.
    pub fn anthropic_compat(&self) -> AnthropicCompat {
        self.anthropic_compat
    }
}

/// A wire protocol that can stream a chat completion — the seam a third-party
/// protocol plugs into without touching crate internals.
///
/// An adapter only translates between its wire format and [`ProtocolEvent`]s:
/// it never builds [`AssistantMessageEvent`]s or the final message itself, and
/// it never fails synchronously — failures are terminal
/// [`ProtocolEvent::Failure`] events on the returned stream. Register adapters
/// on a provider via [`ProviderBuilder`](crate::ProviderBuilder).
pub trait ProtocolAdapter: Send + Sync {
    /// The wire protocol this adapter speaks. A provider holds at most one
    /// adapter per [`ApiKind`] and routes each model by its `Model.api`.
    fn kind(&self) -> ApiKind;

    /// Open a streamed completion for a fully-resolved request. The stream
    /// must end after a [`ProtocolEvent::Stop`] or [`ProtocolEvent::Failure`];
    /// ending without either is reported as an [`ErrorKind::Protocol`] error.
    fn stream(&self, request: PreparedRequest) -> ProtocolEventStream;
}

/// The stable string id of a wire protocol, matching its serde
/// representation — used for `AssistantMessage.api` and diagnostics.
pub(crate) fn api_name(kind: ApiKind) -> &'static str {
    match kind {
        ApiKind::OpenAiCompletions => "openai-completions",
        ApiKind::AnthropicMessages => "anthropic-messages",
    }
}

/// Attach one [`ProviderHeaders`] layer to a request, skipping `None` values.
///
/// Layers are applied in priority order (protocol defaults → provider
/// defaults → auth headers), but reqwest *appends* same-named headers rather
/// than overriding, so a layer colliding with an earlier one currently sends
/// both values. The case-insensitive override/delete merge (PRD v0.3 §5.5)
/// replaces this.
pub(crate) fn apply_headers(
    mut builder: reqwest::RequestBuilder,
    headers: &ProviderHeaders,
) -> reqwest::RequestBuilder {
    for (name, value) in headers {
        if let Some(value) = value {
            builder = builder.header(name, value);
        }
    }
    builder
}

/// Drive a [`ProtocolAdapter`] end to end: emit `Start`, resolve auth in-band
/// (cancellable), hand the adapter a [`PreparedRequest`], and fold its
/// [`ProtocolEvent`]s through the shared assembler into the public event
/// stream.
///
/// This is the single place every stream gets its guarantees: one `Start`,
/// exactly one terminal `Done`/`Error`, the caller's cancellation token raced
/// against every adapter await point (resolver, connect, backoff, SSE read),
/// and an [`ErrorKind::Protocol`] failure when an adapter's stream ends
/// without a formal `Stop`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn drive(
    adapter: &Arc<dyn ProtocolAdapter>,
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    auth: &Auth,
    headers: &ProviderHeaders,
    http_client: &reqwest::Client,
    openai_compat: OpenAiCompat,
    anthropic_compat: AnthropicCompat,
) -> MessageStream {
    let adapter = adapter.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();
    let auth = auth.clone();
    let headers = headers.clone();
    let http_client = http_client.clone();

    let stream = async_stream::stream! {
        let mut assembler = MessageAssembler::new(AssistantMessage::streaming(
            &model.id,
            &model.provider,
            api_name(adapter.kind()),
        ));
        yield AssistantMessageEvent::Start;

        let cancellation = options.cancellation.clone();
        let resolved = match cancel::race(
            cancellation.as_ref(),
            crate::auth::resolve_for_request(&auth, options.api_key.clone()),
        )
        .await
        {
            Ok(Ok(resolved)) => resolved,
            Ok(Err(err)) => {
                yield assembler.fail(ErrorKind::Auth, err.to_string(), Vec::new());
                return;
            }
            Err(cancel::Aborted) => {
                yield assembler.abort("request was cancelled");
                return;
            }
        };

        let prepared = PreparedRequest {
            model,
            context,
            options,
            auth: resolved,
            headers,
            http: http_client,
            openai_compat,
            anthropic_compat,
        };
        let events = adapter.stream(prepared);
        let mut events = std::pin::pin!(events);
        let mut stop_reason: Option<StopReason> = None;
        loop {
            let next = match cancel::race(cancellation.as_ref(), events.next()).await {
                Ok(next) => next,
                Err(cancel::Aborted) => {
                    yield assembler.abort("request was cancelled");
                    return;
                }
            };
            let Some(event) = next else { break };
            if let ProtocolEvent::Stop(reason) = &event {
                stop_reason = Some(*reason);
            }
            // Keep applying after `Stop`: content events that illegally
            // follow it must reach the assembler, which converts them into
            // the terminal `ErrorKind::Protocol` the contract requires.
            if let Some(public) = assembler.apply(event) {
                let terminal = is_terminal(&public);
                yield public;
                if terminal {
                    return;
                }
            }
        }
        let Some(reason) = stop_reason else {
            yield assembler.fail(
                ErrorKind::Protocol,
                "protocol adapter stream ended without a Stop or Failure event",
                Vec::new(),
            );
            return;
        };
        yield AssistantMessageEvent::Done {
            reason,
            message: assembler.into_message(),
        };
    };

    MessageStream::new(stream)
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
/// malformed-data detail/diagnostic a caller should yield as a
/// [`ProtocolEvent::Failure`]. Shared by both protocols — each still checks
/// for its own non-JSON sentinel (OpenAI's `[DONE]`) or named `event:` field
/// (Anthropic's `error`) before calling this, since those are
/// protocol-specific and don't belong in a shared parse step.
pub(crate) fn parse_sse_json(data: String) -> Result<serde_json::Value, (String, Diagnostic)> {
    serde_json::from_str(&data).map_err(|_| {
        (
            "malformed SSE data: not valid JSON".to_string(),
            Diagnostic::new(DiagnosticCode::ProtocolViolation, data),
        )
    })
}
