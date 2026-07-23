//! The shared request/retry/SSE executor both protocol paths route through.
//!
//! An adapter supplies a rebuildable request factory (called fresh for every
//! attempt, so request bodies never need to be cloneable) and drives the
//! returned stream: [`ExecutorEvent::Retry`] to report progress,
//! [`ExecutorEvent::Event`] for each decoded SSE event, [`ExecutorEvent::Eof`]
//! when the body closes normally, [`ExecutorEvent::Failed`] as the sole
//! terminal failure shape covering pre-stream retryable-budget exhaustion, a
//! Retry-After beyond the configured cap, a mid-stream transport drop, and an
//! oversized SSE event, and [`ExecutorEvent::Aborted`] when the caller's
//! [`CancellationToken`] fires during connect, retry backoff, or an SSE read.

use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::cancel;
use crate::error::ErrorKind;
use crate::http::{self, RetryDecision};
use crate::sse::{SseDecoder, SseError, SseEvent};
use crate::types::{Diagnostic, DiagnosticCode};

/// One step of executing a request.
pub(crate) enum ExecutorEvent {
    /// A pre-stream attempt failed and will be retried after `delay`.
    Retry {
        /// Which retry this is (1-based).
        attempt: u32,
        /// Total attempts the budget allows (initial request + retries).
        max_attempts: u32,
        /// How long the executor will sleep before re-sending.
        delay: Duration,
        /// Classification of the failure that triggered the retry.
        kind: ErrorKind,
    },
    /// The response was established: headers are in, before any SSE event.
    Established {
        /// A provider-supplied request id, if a recognized header carried one.
        request_id: Option<String>,
    },
    /// One decoded SSE event, ready for protocol-specific parsing.
    Event(SseEvent),
    /// The connection closed normally (EOF) after zero or more events.
    Eof,
    /// A terminal failure: retry budget exhausted, a Retry-After beyond the
    /// cap, a mid-stream transport drop, or an oversized SSE event.
    Failed {
        /// Classification for `AssistantMessage.error_kind`.
        kind: ErrorKind,
        /// Human-readable, secret-free summary for `error_message`.
        message: String,
        /// Bounded, redacted detail for `AssistantMessage.diagnostics`.
        diagnostics: Vec<Diagnostic>,
    },
    /// The caller's [`CancellationToken`] fired before the request finished.
    /// No further retries follow this event.
    Aborted,
}

/// Execute one request with bounded pre-stream retry, then decode its SSE
/// body — the single place both protocol adapters get a response from.
///
/// `factory` rebuilds the request from scratch for every attempt (headers,
/// body, timeout — everything), rather than `RequestBuilder::try_clone`.
pub(crate) fn execute(
    factory: impl Fn() -> reqwest::RequestBuilder + Send + 'static,
    max_retries: u32,
    max_retry_delay: Option<Duration>,
    cancellation: Option<CancellationToken>,
) -> impl Stream<Item = ExecutorEvent> {
    async_stream::stream! {
        let max_retry_delay = max_retry_delay.unwrap_or(http::DEFAULT_MAX_RETRY_DELAY);
        let token = cancellation.as_ref();
        let mut attempt: u32 = 0;
        let response = loop {
            let attempt_result = match cancel::race(token, http::send_once(factory())).await {
                Ok(result) => result,
                Err(cancel::Aborted) => {
                    yield ExecutorEvent::Aborted;
                    return;
                }
            };
            match attempt_result {
                Ok(response) => break response,
                Err(failure) if failure.kind.is_retryable() && attempt < max_retries => {
                    attempt += 1;
                    match http::retry_delay(attempt, failure.retry_after, max_retry_delay) {
                        RetryDecision::Sleep(delay) => {
                            yield ExecutorEvent::Retry {
                                attempt,
                                max_attempts: max_retries + 1,
                                delay,
                                kind: failure.kind,
                            };
                            if cancel::race(token, tokio::time::sleep(delay)).await.is_err() {
                                yield ExecutorEvent::Aborted;
                                return;
                            }
                        }
                        RetryDecision::ExceedsCap { requested } => {
                            yield ExecutorEvent::Failed {
                                kind: ErrorKind::RateLimited,
                                message: format!(
                                    "provider requested a {:.0}s retry delay, exceeding the {:.0}s cap",
                                    requested.as_secs_f64(),
                                    max_retry_delay.as_secs_f64(),
                                ),
                                diagnostics: vec![Diagnostic::new(
                                    DiagnosticCode::ProviderError,
                                    format!("Retry-After requested {:.0}s", requested.as_secs_f64()),
                                )],
                            };
                            return;
                        }
                    }
                }
                Err(failure) => {
                    yield ExecutorEvent::Failed {
                        kind: failure.kind,
                        message: failure.detail,
                        diagnostics: failure.diagnostics,
                    };
                    return;
                }
            }
        };

        yield ExecutorEvent::Established {
            request_id: extract_request_id(response.headers()),
        };

        let mut decoder = SseDecoder::new();
        let mut body = response.bytes_stream();
        loop {
            let next = match cancel::race(token, body.next()).await {
                Ok(next) => next,
                Err(cancel::Aborted) => {
                    yield ExecutorEvent::Aborted;
                    return;
                }
            };
            match next {
                Some(Ok(chunk)) => match decoder.push(&chunk) {
                    Ok(events) => {
                        for event in events {
                            yield ExecutorEvent::Event(event);
                        }
                    }
                    Err(err) => {
                        yield oversized_event_failure(err);
                        return;
                    }
                },
                Some(Err(err)) => {
                    yield ExecutorEvent::Failed {
                        kind: ErrorKind::StreamInterrupted,
                        message: format!("stream error: {err}"),
                        diagnostics: Vec::new(),
                    };
                    return;
                }
                None => {
                    match decoder.finish() {
                        Ok(events) => {
                            for event in events {
                                yield ExecutorEvent::Event(event);
                            }
                        }
                        Err(err) => {
                            yield oversized_event_failure(err);
                            return;
                        }
                    }
                    yield ExecutorEvent::Eof;
                    return;
                }
            }
        }
    }
}

/// The `data:` cap is fixed at 8 MiB (see [`crate::sse`]) and unconfigurable,
/// so its only variant maps to one terminal shape: a wire-level protocol
/// violation, never retried.
fn oversized_event_failure(err: SseError) -> ExecutorEvent {
    let SseError::EventTooLarge { limit } = err;
    ExecutorEvent::Failed {
        kind: ErrorKind::Protocol,
        message: format!("SSE event data exceeded {limit} bytes"),
        diagnostics: Vec::new(),
    }
}

/// Best-effort provider request id from common header names.
fn extract_request_id(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .or_else(|| headers.get("request-id"))
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}
