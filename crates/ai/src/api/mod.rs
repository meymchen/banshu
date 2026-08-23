//! Wire-protocol adapters — the public extension seam.
//!
//! Each protocol implements [`ProtocolAdapter`]: given a fully-resolved
//! [`PreparedRequest`], it translates between its own wire format and the
//! shared [`ProtocolEvent`] vocabulary. The provider layer drives the returned
//! [`ProtocolEventStream`] through the internal message assembler, which owns
//! block ordering, public `content_index` assignment, cancellation, and the
//! final [`AssistantMessage`].
//!
//! Cross-model rules are not an adapter's job: the context on a
//! [`PreparedRequest`] has already been through one normalization pass for the
//! target model, so an adapter translates what it is given verbatim.

pub mod anthropic_messages;
pub mod openai_completions;

mod assembler;
mod normalize;
mod output_budget;
mod protocol_event;
mod reasoning;
mod tool_choice;

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
    header_layers: HeaderLayers,
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

    /// The conversation context, already normalized for
    /// [`model()`](Self::model).
    ///
    /// Cross-model rules — image downgrade, reasoning downgrade, tool-call id
    /// rewrite, more as the crate grows — have already been applied to this
    /// copy, so an adapter only translates it to its own wire shape. The
    /// caller's own `Context` is untouched.
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

    /// Effective headers above the protocol-default layer, after provider,
    /// model, generated-auth, resolved-auth, and request layers have been
    /// merged case-insensitively.
    pub fn headers(&self) -> &ProviderHeaders {
        &self.headers
    }

    /// Merge this request's fixed header chain over adapter-supplied protocol
    /// defaults.
    ///
    /// This is the header seam for third-party adapters: the result contains
    /// no case-insensitive duplicates or `None` tombstones and is ready to
    /// attach to the final HTTP request.
    pub fn headers_with_protocol_defaults(
        &self,
        protocol_defaults: &ProviderHeaders,
    ) -> ProviderHeaders {
        self.header_layers.merge(protocol_defaults)
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

struct HeaderLayers {
    provider: ProviderHeaders,
    model: ProviderHeaders,
    generated_auth: ProviderHeaders,
    resolved_auth: ProviderHeaders,
    request: ProviderHeaders,
}

impl HeaderLayers {
    fn merge(&self, protocol_defaults: &ProviderHeaders) -> ProviderHeaders {
        crate::auth::merge_header_layers([
            protocol_defaults,
            &self.provider,
            &self.model,
            &self.generated_auth,
            &self.resolved_auth,
            &self.request,
        ])
    }
}

fn generated_auth_headers(kind: ApiKind, api_key: Option<&str>) -> ProviderHeaders {
    let Some(api_key) = api_key else {
        return ProviderHeaders::new();
    };
    let (name, value) = match kind {
        ApiKind::OpenAiCompletions => ("Authorization", format!("Bearer {api_key}")),
        ApiKind::AnthropicMessages => ("x-api-key", api_key.to_string()),
    };
    ProviderHeaders::from([(name.to_string(), Some(value))])
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

/// Attach an already-merged [`ProviderHeaders`] map to a request.
///
/// The merge step has removed tombstones and case-insensitive duplicates, so
/// this writes each effective header exactly once.
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

/// Drive a [`ProtocolAdapter`] end to end: emit `Start`, normalize the context
/// for the target model, resolve auth in-band (cancellable), hand the adapter a
/// [`PreparedRequest`], and fold its [`ProtocolEvent`]s through the shared
/// assembler into the public event stream.
///
/// This is the single place every stream gets its guarantees: one `Start`,
/// exactly one terminal `Done`/`Error`, one normalization pass over a copy of
/// the caller's context, the caller's cancellation token raced against every
/// adapter await point (resolver, connect, backoff, SSE read),
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
    let mut options = options.clone();
    let auth = auth.clone();
    let provider_headers = headers.clone();
    let http_client = http_client.clone();

    let stream = async_stream::stream! {
        let mut assembler = MessageAssembler::new(AssistantMessage::streaming(
            &model.id,
            &model.provider,
            api_name(adapter.kind()),
        ));
        yield AssistantMessageEvent::Start;

        // Resolve one shared output cap before either protocol sees the
        // request. Explicit caps are caller intent and fail instead of being
        // reduced; an omitted cap uses whatever model limits are actually
        // known, leaving zero-means-unknown metadata unknown.
        options.max_tokens = match output_budget::resolve(&model, &context, &options) {
            Ok(max_tokens) => max_tokens,
            Err(detail) => {
                yield assembler.fail(ErrorKind::InvalidRequest, detail, Vec::new());
                return;
            }
        };

        // The reasoning and tool-choice preflights read only the options, the
        // model's attested capability, and the provider's declared request
        // shape, so they run first: a request nothing can honour fails before
        // any work is done on its behalf.
        if let Err(detail) =
            reasoning::validate(&model, &options, openai_compat, anthropic_compat)
        {
            yield assembler.fail(ErrorKind::InvalidRequest, detail, Vec::new());
            return;
        }
        if let Err(detail) =
            tool_choice::validate(&model, &options, openai_compat, anthropic_compat)
        {
            yield assembler.fail(ErrorKind::InvalidRequest, detail, Vec::new());
            return;
        }

        // The one normalization pass: cross-model rules are resolved here,
        // once, so the adapter below sees a context it can translate
        // verbatim. A modality violation fails in-band before auth
        // resolution or any HTTP request.
        let normalized = match normalize::normalize(&model, &context) {
            Ok(normalized) => normalized,
            Err(detail) => {
                yield assembler.fail(ErrorKind::InvalidRequest, detail, Vec::new());
                return;
            }
        };
        let context = normalized.context;
        for diagnostic in normalized.diagnostics {
            let public = assembler.apply(ProtocolEvent::Diagnostic(diagnostic));
            debug_assert!(public.is_none(), "a diagnostic emits no public event");
        }

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

        let header_layers = HeaderLayers {
            provider: provider_headers,
            model: model.headers.clone(),
            generated_auth: generated_auth_headers(adapter.kind(), resolved.api_key.as_deref()),
            resolved_auth: resolved.headers.clone(),
            request: options.headers.clone(),
        };
        let headers = header_layers.merge(&ProviderHeaders::new());
        let prepared = PreparedRequest {
            model,
            context,
            options,
            auth: resolved,
            header_layers,
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

/// Compute cost from token counts and per-million rates. When the model
/// carries context tiers, total input usage (input + cache read + cache
/// write) selects one request-wide rate set: the highest tier threshold the
/// usage strictly exceeds, else the base rates.
pub(crate) fn compute_cost(usage: &Usage, cost: &ModelCost) -> Cost {
    let input_tokens = usage.input + usage.cache_read + usage.cache_write;
    let rates = cost.rates_for_input(input_tokens);
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
