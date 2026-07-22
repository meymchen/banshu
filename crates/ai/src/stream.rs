//! The streaming contract: an async-iterable of delta events terminating in a
//! `Done` or `Error` carrying the final [`AssistantMessage`].
//!
//! Errors are **in-band**: the stream never yields a `Result`. A transport
//! failure mid-response terminates with an `Error` event whose message carries
//! whatever partial content had already streamed, so callers keep their tokens.

use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;

use crate::error::ErrorKind;
use crate::types::{
    AssistantContent, AssistantMessage, StopReason, TextContent, ThinkingContent, ToolCall,
};

/// A single incremental event in a streamed assistant response.
///
/// Non-terminal events carry only their own incremental payload — never a full
/// message snapshot. A consumer that wants the message assembled so far reads
/// [`MessageStream::partial`]; the complete message travels exactly once, on
/// the terminal [`Done`](Self::Done)/[`Error`](Self::Error) event.
#[derive(Debug, Clone)]
pub enum AssistantMessageEvent {
    /// Emitted once at the start, before any content.
    Start,
    /// A text content block has begun.
    TextStart {
        /// Index of the content block.
        content_index: usize,
    },
    /// A chunk of text output.
    TextDelta {
        /// Index of the content block this delta belongs to.
        content_index: usize,
        /// The appended text.
        delta: String,
    },
    /// A text content block is complete.
    TextEnd {
        /// Index of the content block.
        content_index: usize,
        /// The completed text content.
        content: TextContent,
    },
    /// A reasoning content block has begun.
    ThinkingStart {
        /// Index of the content block.
        content_index: usize,
    },
    /// A chunk of reasoning output.
    ThinkingDelta {
        /// Index of the content block this delta belongs to.
        content_index: usize,
        /// The appended reasoning text.
        delta: String,
    },
    /// A reasoning content block is complete.
    ThinkingEnd {
        /// Index of the content block.
        content_index: usize,
        /// The completed reasoning content.
        content: ThinkingContent,
    },
    /// A tool-call content block has begun.
    ToolCallStart {
        /// Index of the content block.
        content_index: usize,
    },
    /// A fragment of the tool call's arguments JSON.
    ToolCallDelta {
        /// Index of the content block this delta belongs to.
        content_index: usize,
        /// The appended arguments fragment.
        delta: String,
    },
    /// A completed tool call.
    ToolCallEnd {
        /// Index of the content block.
        content_index: usize,
        /// The assembled tool call.
        tool_call: ToolCall,
    },
    /// The request failed before the response stream started and will be
    /// re-sent after `delay`. Emitted so UIs can show retry progress instead
    /// of a silent pause; consumers that don't care can ignore it.
    Retry {
        /// Which retry this is (1-based).
        attempt: u32,
        /// Total attempts the budget allows (initial request + retries).
        max_attempts: u32,
        /// How long the stream will sleep before re-sending.
        delay: Duration,
        /// Classification of the failure that triggered the retry.
        kind: ErrorKind,
    },
    /// Terminal success — the final assembled message.
    Done {
        /// Why the completion stopped (`Stop`, `Length`, or `ToolUse`).
        reason: StopReason,
        /// The final message.
        message: AssistantMessage,
    },
    /// Terminal failure — the final message with `stop_reason` `Error`/`Aborted`.
    Error {
        /// Why the completion stopped (`Error` or `Aborted`).
        reason: StopReason,
        /// The final message, carrying any partial content and `error_message`.
        error: AssistantMessage,
    },
}

/// A stream of [`AssistantMessageEvent`]s with a terminal [`AssistantMessage`].
///
/// Alongside driving it as a [`Stream`], a caller can inspect progress without
/// consuming events itself: [`partial`](Self::partial) reflects the latest
/// snapshot seen so far, [`result`](Self::result) is `Some` once a terminal
/// `Done`/`Error` has passed through, and [`finish`](Self::finish) drives any
/// remaining events and returns the final message.
pub struct MessageStream {
    inner: Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send>>,
    partial: AssistantMessage,
    terminal: Option<AssistantMessage>,
}

impl MessageStream {
    /// Wrap an event stream.
    pub fn new(stream: impl Stream<Item = AssistantMessageEvent> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
            // Placeholder until the stream's own `Start` event replaces it;
            // every adapter yields `Start` before anything else.
            partial: AssistantMessage::streaming("", "", ""),
            terminal: None,
        }
    }

    /// A stream that yields a single terminal `Error` event. Used when a
    /// request can't even be dispatched (e.g. no provider owns the model).
    pub(crate) fn immediate_error(model: &str, provider: &str, detail: &str) -> Self {
        let mut message = AssistantMessage::streaming(model, provider, "");
        message.stop_reason = StopReason::Error;
        message.error_message = Some(detail.to_string());
        message.error_kind = Some(ErrorKind::Api);
        let event = AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: message,
        };
        Self::new(futures_util::stream::once(async move { event }))
    }

    /// The message as assembled from every event observed so far (via
    /// [`Stream::poll_next`] or [`finish`](Self::finish)). Before the first
    /// event, this is an empty placeholder; its `model`/`provider` are filled
    /// in only once the terminal `Done`/`Error` message arrives.
    ///
    /// A tool call in progress reads back with an empty `id`/`name` and `{}`
    /// arguments until its `ToolCallEnd` — the public `ToolCallStart` event
    /// carries no identity, so the completed call only appears on the terminal
    /// message and at end-of-block.
    pub fn partial(&self) -> &AssistantMessage {
        &self.partial
    }

    /// The final message, once a terminal `Done`/`Error` event has been
    /// observed. `None` until then.
    pub fn result(&self) -> Option<&AssistantMessage> {
        self.terminal.as_ref()
    }

    /// Drive any not-yet-consumed events to completion and return the final
    /// message.
    ///
    /// This never returns a `Result`: failures arrive as an `Error` event whose
    /// message has `stop_reason` `Error`/`Aborted` and an `error_message`.
    pub async fn finish(&mut self) -> AssistantMessage {
        while self.terminal.is_none() {
            if self.next().await.is_none() {
                break;
            }
        }
        self.terminal
            .clone()
            .expect("stream ended without a terminal Done or Error event")
    }

    /// Fold an observed event into `partial`, and capture `terminal` on the
    /// terminal `Done`/`Error`.
    ///
    /// Because non-terminal events no longer carry a message snapshot, the
    /// stream reconstructs `partial` itself by applying each incremental event
    /// in place — an O(1)-per-event projection that mirrors what the internal
    /// assembler built, without the per-delta clone.
    fn record(&mut self, event: &AssistantMessageEvent) {
        match event {
            AssistantMessageEvent::Start | AssistantMessageEvent::Retry { .. } => {}
            AssistantMessageEvent::TextStart { .. } => {
                self.partial
                    .content
                    .push(AssistantContent::Text(TextContent {
                        text: String::new(),
                        signature: None,
                    }));
            }
            AssistantMessageEvent::TextDelta {
                content_index,
                delta,
            } => {
                if let Some(AssistantContent::Text(text)) =
                    self.partial.content.get_mut(*content_index)
                {
                    text.text.push_str(delta);
                }
            }
            AssistantMessageEvent::TextEnd {
                content_index,
                content,
            } => {
                if let Some(slot) = self.partial.content.get_mut(*content_index) {
                    *slot = AssistantContent::Text(content.clone());
                }
            }
            AssistantMessageEvent::ThinkingStart { .. } => {
                self.partial
                    .content
                    .push(AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        signature: None,
                        redacted: false,
                    }));
            }
            AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
            } => {
                if let Some(AssistantContent::Thinking(thinking)) =
                    self.partial.content.get_mut(*content_index)
                {
                    thinking.thinking.push_str(delta);
                }
            }
            AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
            } => {
                if let Some(slot) = self.partial.content.get_mut(*content_index) {
                    *slot = AssistantContent::Thinking(content.clone());
                }
            }
            AssistantMessageEvent::ToolCallStart { .. } => {
                self.partial
                    .content
                    .push(AssistantContent::ToolCall(ToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: serde_json::json!({}),
                        raw_arguments: None,
                    }));
            }
            AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
            } => {
                if let Some(AssistantContent::ToolCall(tool_call)) =
                    self.partial.content.get_mut(*content_index)
                {
                    tool_call
                        .raw_arguments
                        .get_or_insert_default()
                        .push_str(delta);
                }
            }
            AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
            } => {
                if let Some(slot) = self.partial.content.get_mut(*content_index) {
                    *slot = AssistantContent::ToolCall(tool_call.clone());
                }
            }
            AssistantMessageEvent::Done { message, .. } => {
                self.partial = message.clone();
                self.terminal = Some(message.clone());
            }
            AssistantMessageEvent::Error { error, .. } => {
                self.partial = error.clone();
                self.terminal = Some(error.clone());
            }
        }
    }
}

impl Stream for MessageStream {
    type Item = AssistantMessageEvent;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let poll = this.inner.as_mut().poll_next(cx);
        if let Poll::Ready(Some(event)) = &poll {
            this.record(event);
        }
        poll
    }
}
