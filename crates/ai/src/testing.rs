//! Deterministic, network-free fixtures for downstream streaming tests.
//!
//! [`FauxProvider`] drives the same public [`crate::MessageStream`]
//! contract as a real provider while taking its output from a [`FauxScript`].
//! It never opens a network connection and never resolves provider credentials.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use crate::{
    ApiKind, Context, ErrorKind, MessageStream, Model, PreparedRequest, ProtocolAdapter,
    ProtocolEvent, ProtocolEventStream, Provider, StopReason, StreamOptions, Usage,
};

/// One caller-visible response item in a [`FauxScript`].
///
/// Events describe content rather than wire-protocol frames: the faux adapter
/// assigns its own internal block ids and emits the corresponding start,
/// delta, and end events through the normal streaming assembler.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum FauxEvent {
    /// One complete text content block.
    Text(String),
    /// Replace the response's deterministic token usage.
    Usage(Usage),
    /// Wait before the next scripted event.
    Delay(Duration),
    /// One complete thinking content block, including replay metadata.
    Thinking {
        /// Reasoning text.
        thinking: String,
        /// Opaque signature associated with the thinking block.
        signature: Option<String>,
        /// Whether the provider treated the reasoning text as redacted.
        redacted: bool,
    },
    /// One complete tool-call block with raw JSON arguments.
    ToolCall {
        /// Provider-assigned call id.
        id: String,
        /// Tool name.
        name: String,
        /// Verbatim JSON arguments streamed for the call.
        arguments: String,
    },
}

/// A successful terminal reason a faux response may report.
///
/// Error and aborted terminations are deliberately absent: errors use
/// [`FauxScript::in_band_failure`], while aborts come from cancelling the
/// stream. This keeps faux streams inside the same public contract as real
/// providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FauxStopReason {
    /// Natural end of turn.
    Stop,
    /// The response reached its output-token limit.
    Length,
    /// The response ended to request one or more tool calls.
    ToolUse,
    /// The provider supplied an unrecognized successful reason.
    Unknown,
}

impl From<FauxStopReason> for StopReason {
    fn from(reason: FauxStopReason) -> Self {
        match reason {
            FauxStopReason::Stop => Self::Stop,
            FauxStopReason::Length => Self::Length,
            FauxStopReason::ToolUse => Self::ToolUse,
            FauxStopReason::Unknown => Self::Unknown,
        }
    }
}

impl FauxEvent {
    /// Script one complete text content block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Script the final token usage reported on the assembled message.
    pub fn usage(usage: Usage) -> Self {
        Self::Usage(usage)
    }

    /// Delay the next event by `duration`.
    ///
    /// The wait uses Tokio time, so tests can combine it with paused time or
    /// cancel it through [`StreamOptions::cancellation`].
    pub fn delay(duration: Duration) -> Self {
        Self::Delay(duration)
    }

    /// Script one complete thinking block with an optional opaque signature.
    pub fn thinking(
        thinking: impl Into<String>,
        signature: Option<impl Into<String>>,
        redacted: bool,
    ) -> Self {
        Self::Thinking {
            thinking: thinking.into(),
            signature: signature.map(Into::into),
            redacted,
        }
    }

    /// Script one complete tool call from verbatim JSON `arguments`.
    ///
    /// The normal stream assembler parses the arguments, so malformed input
    /// exercises the same in-band protocol error as a real adapter.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

/// One setup attempt in a [`FauxScript`].
///
/// A success streams its response events. A failure happens before content
/// starts; retryable kinds proceed to the next scripted attempt when the
/// request's retry budget permits it.
#[derive(Debug, Clone)]
pub struct FauxAttempt {
    outcome: AttemptOutcome,
}

#[derive(Debug, Clone)]
enum AttemptOutcome {
    Response {
        events: Vec<FauxEvent>,
        terminal: ResponseTerminal,
    },
    Failure {
        kind: ErrorKind,
        message: String,
        retry_delay: Duration,
    },
}

#[derive(Debug, Clone)]
enum ResponseTerminal {
    Success { stop_reason: FauxStopReason },
    Failure { kind: ErrorKind, message: String },
}

impl FauxAttempt {
    /// Script an attempt that establishes a response and streams `events`.
    pub fn success(
        events: impl IntoIterator<Item = FauxEvent>,
        stop_reason: FauxStopReason,
    ) -> Self {
        Self {
            outcome: AttemptOutcome::Response {
                events: events.into_iter().collect(),
                terminal: ResponseTerminal::Success { stop_reason },
            },
        }
    }

    /// Script a response that fails in-band after streaming `events`.
    ///
    /// The terminal message preserves all content emitted before the error.
    pub fn in_band_failure(
        events: impl IntoIterator<Item = FauxEvent>,
        kind: ErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            outcome: AttemptOutcome::Response {
                events: events.into_iter().collect(),
                terminal: ResponseTerminal::Failure {
                    kind,
                    message: message.into(),
                },
            },
        }
    }

    /// Script a setup failure before any response content starts.
    ///
    /// When `kind` is retryable, `retry_delay` is reported on the public
    /// retry event and awaited before the next scripted attempt.
    pub fn failure(kind: ErrorKind, message: impl Into<String>, retry_delay: Duration) -> Self {
        Self {
            outcome: AttemptOutcome::Failure {
                kind,
                message: message.into(),
                retry_delay,
            },
        }
    }
}

/// A deterministic sequence of setup attempts consumed for each stream.
#[derive(Debug, Clone)]
pub struct FauxScript {
    attempts: Vec<FauxAttempt>,
}

impl FauxScript {
    /// Build a successful response from `events` and a final `stop_reason`.
    ///
    /// Each invocation starts from the beginning of the script, making the
    /// same fixture safely repeatable across tests.
    pub fn success(
        events: impl IntoIterator<Item = FauxEvent>,
        stop_reason: FauxStopReason,
    ) -> Self {
        Self {
            attempts: vec![FauxAttempt::success(events, stop_reason)],
        }
    }

    /// Build one terminal setup failure delivered through the stream.
    pub fn failure(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            attempts: vec![FauxAttempt::failure(kind, message, Duration::ZERO)],
        }
    }

    /// Build one response that fails in-band after streaming `events`.
    pub fn in_band_failure(
        events: impl IntoIterator<Item = FauxEvent>,
        kind: ErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            attempts: vec![FauxAttempt::in_band_failure(events, kind, message)],
        }
    }

    /// Build a script from setup attempts in their execution order.
    ///
    /// An empty script terminates in-band with [`ErrorKind::Protocol`].
    pub fn attempts(attempts: impl IntoIterator<Item = FauxAttempt>) -> Self {
        Self {
            attempts: attempts.into_iter().collect(),
        }
    }
}

struct FauxAdapter {
    script: FauxScript,
    attempt_count: Arc<AtomicU32>,
}

impl ProtocolAdapter for FauxAdapter {
    fn kind(&self) -> ApiKind {
        ApiKind::OpenAiCompletions
    }

    fn stream(&self, request: PreparedRequest) -> ProtocolEventStream {
        let script = self.script.clone();
        let attempt_count = self.attempt_count.clone();
        let max_retries = request
            .options()
            .max_retries
            .unwrap_or(crate::http::DEFAULT_MAX_RETRIES);
        Box::pin(async_stream::stream! {
            let mut block_id = 0;
            if script.attempts.is_empty() {
                yield ProtocolEvent::Failure {
                    kind: ErrorKind::Protocol,
                    message: "faux script has no attempts".to_string(),
                    diagnostics: Vec::new(),
                };
                return;
            }
            let scripted_attempts = script.attempts.len();
            for (index, attempt) in script.attempts.into_iter().enumerate() {
                attempt_count.fetch_add(1, Ordering::SeqCst);
                match attempt.outcome {
                    AttemptOutcome::Response { events, terminal } => {
                        for event in events {
                            match event {
                                FauxEvent::Text(text) => {
                                    yield ProtocolEvent::TextStart {
                                        block_id,
                                        signature: None,
                                    };
                                    yield ProtocolEvent::TextDelta { block_id, delta: text };
                                    yield ProtocolEvent::TextEnd { block_id };
                                    block_id += 1;
                                }
                                FauxEvent::Usage(usage) => yield ProtocolEvent::Usage(usage),
                                FauxEvent::Delay(duration) => tokio::time::sleep(duration).await,
                                FauxEvent::Thinking { thinking, signature, redacted } => {
                                    yield ProtocolEvent::ThinkingStart {
                                        block_id,
                                        signature,
                                        redacted,
                                    };
                                    yield ProtocolEvent::ThinkingDelta {
                                        block_id,
                                        delta: thinking,
                                    };
                                    yield ProtocolEvent::ThinkingEnd { block_id };
                                    block_id += 1;
                                }
                                FauxEvent::ToolCall { id, name, arguments } => {
                                    yield ProtocolEvent::ToolCallStart { block_id, id, name };
                                    yield ProtocolEvent::ToolCallDelta {
                                        block_id,
                                        delta: arguments,
                                    };
                                    yield ProtocolEvent::ToolCallEnd { block_id };
                                    block_id += 1;
                                }
                            }
                        }
                        match terminal {
                            ResponseTerminal::Success { stop_reason } => {
                                yield ProtocolEvent::Stop(stop_reason.into());
                            }
                            ResponseTerminal::Failure { kind, message } => {
                                yield ProtocolEvent::Failure {
                                    kind,
                                    message,
                                    diagnostics: Vec::new(),
                                };
                            }
                        }
                        return;
                    }
                    AttemptOutcome::Failure { kind, message, retry_delay } => {
                        let retry_number = index as u32 + 1;
                        let has_next_attempt = index + 1 < scripted_attempts;
                        if kind.is_retryable() && has_next_attempt && retry_number <= max_retries {
                            yield ProtocolEvent::Retry {
                                attempt: retry_number,
                                max_attempts: max_retries + 1,
                                delay: retry_delay,
                                kind,
                            };
                            tokio::time::sleep(retry_delay).await;
                        } else {
                            yield ProtocolEvent::Failure {
                                kind,
                                message,
                                diagnostics: Vec::new(),
                            };
                            return;
                        }
                    }
                }
            }
        })
    }
}

/// A deterministic provider/model pair for exercising the public stream API.
///
/// The provider is keyless and its adapter only reads the supplied script, so
/// calling [`stream`](Self::stream) requires neither credentials nor a network
/// listener. A provider can be reused: every stream replays the script from
/// its beginning, while [`attempt_count`](Self::attempt_count) reports the
/// cumulative number of scripted attempts made by this fixture.
pub struct FauxProvider {
    provider: Provider,
    model: Model,
    attempt_count: Arc<AtomicU32>,
}

impl FauxProvider {
    /// Create a keyless faux provider serving one OpenAI-kind `model_id`.
    pub fn new(model_id: impl Into<String>, script: FauxScript) -> Self {
        let model_id = model_id.into();
        let provider_id = "faux";
        let mut model = Model::openai_completions(model_id);
        model.provider = provider_id.to_string();
        let attempt_count = Arc::new(AtomicU32::new(0));
        let adapter = FauxAdapter {
            script,
            attempt_count: attempt_count.clone(),
        };
        let provider = Provider::builder(provider_id, "Faux", "faux://local")
            .adapter(Arc::new(adapter))
            .model(model.clone())
            .build()
            .expect("the fixed faux provider configuration is valid");
        Self {
            provider,
            model,
            attempt_count,
        }
    }

    /// Replay the script through the ordinary public streaming contract.
    pub fn stream(&self, context: &Context, options: &StreamOptions) -> MessageStream {
        self.provider.stream(&self.model, context, options)
    }

    /// Return the model registered on this fixture.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Return the underlying provider for registry-level downstream tests.
    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    /// Number of scripted attempts started across this fixture's streams.
    pub fn attempt_count(&self) -> u32 {
        self.attempt_count.load(Ordering::SeqCst)
    }
}
