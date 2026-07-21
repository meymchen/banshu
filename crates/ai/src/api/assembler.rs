//! Turns a stream of adapter-emitted [`ProtocolEvent`]s into the assembled
//! [`AssistantMessage`] plus the existing [`AssistantMessageEvent`]s, assigning
//! a stable `content_index` the first time each `block_id` starts.
//!
//! Ordering violations — a delta/end for an unknown block, a duplicate start,
//! a type mismatch between a block's start and a later event referencing it,
//! or any content-block event after [`ProtocolEvent::Stop`] — are reported as
//! [`ErrorKind::Protocol`] rather than silently accepted or panicking.

use std::collections::HashMap;

use super::parse_arguments;
use super::protocol_event::ProtocolEvent;
use crate::error::ErrorKind;
use crate::stream::AssistantMessageEvent;
use crate::types::{
    AssistantContent, AssistantMessage, Diagnostic, DiagnosticCode, StopReason, TextContent,
    ThinkingContent, ToolCall,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolCall,
}

#[derive(Clone, Copy)]
struct BlockState {
    kind: BlockKind,
    content_index: usize,
    ended: bool,
}

/// Whether a public event is terminal (the stream must stop after it).
pub(crate) fn is_terminal(event: &AssistantMessageEvent) -> bool {
    matches!(event, AssistantMessageEvent::Error { .. })
}

/// Accumulates one streamed assistant response from [`ProtocolEvent`]s.
pub(crate) struct MessageAssembler {
    message: AssistantMessage,
    blocks: HashMap<u64, BlockState>,
    tool_raw: HashMap<u64, String>,
    stopped: bool,
}

impl MessageAssembler {
    /// Start assembling into `message` (already carrying `model`/`provider`/`api`).
    pub(crate) fn new(message: AssistantMessage) -> Self {
        Self {
            message,
            blocks: HashMap::new(),
            tool_raw: HashMap::new(),
            stopped: false,
        }
    }

    /// The message as assembled so far.
    pub(crate) fn partial(&self) -> &AssistantMessage {
        &self.message
    }

    /// Consume the assembler, returning the final assembled message.
    pub(crate) fn into_message(self) -> AssistantMessage {
        self.message
    }

    /// Fail the assembly and return the terminal `Error` event directly —
    /// convenience over `apply(ProtocolEvent::Failure { .. })` for the common
    /// case where the caller always yields it and returns.
    pub(crate) fn fail(
        &mut self,
        kind: ErrorKind,
        message: impl Into<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> AssistantMessageEvent {
        self.apply(ProtocolEvent::Failure {
            kind,
            message: message.into(),
            diagnostics,
        })
        .expect("ProtocolEvent::Failure always produces a terminal event")
    }

    /// Apply one event, returning the public event to emit (if any).
    ///
    /// A `Some(AssistantMessageEvent::Error { .. })` result is always terminal
    /// — see [`is_terminal`]. Callers must not silently discard a `Some`: an
    /// ordering violation delivered this way is the only way it reaches the
    /// caller as the in-band `Error` event the crate's error contract requires.
    #[must_use]
    pub(crate) fn apply(&mut self, event: ProtocolEvent) -> Option<AssistantMessageEvent> {
        if self.stopped && is_content_event(&event) {
            return Some(self.violation("content block event received after Stop"));
        }

        match event {
            ProtocolEvent::TextStart {
                block_id,
                signature,
            } => match self.start_block(block_id, BlockKind::Text) {
                Ok(()) => {
                    self.message
                        .content
                        .push(AssistantContent::Text(TextContent {
                            text: String::new(),
                            signature,
                        }));
                    None
                }
                Err(detail) => Some(self.violation(detail)),
            },
            ProtocolEvent::TextDelta { block_id, delta } => {
                match self.find_block(block_id, BlockKind::Text, "delta") {
                    Ok(content_index) => {
                        if let AssistantContent::Text(text) =
                            &mut self.message.content[content_index]
                        {
                            text.text.push_str(&delta);
                        }
                        Some(AssistantMessageEvent::TextDelta {
                            content_index,
                            delta,
                            partial: self.message.clone(),
                        })
                    }
                    Err(detail) => Some(self.violation(detail)),
                }
            }
            ProtocolEvent::TextEnd { block_id } => {
                match self.find_block(block_id, BlockKind::Text, "end") {
                    Ok(_) => {
                        self.end_block(block_id);
                        None
                    }
                    Err(detail) => Some(self.violation(detail)),
                }
            }
            ProtocolEvent::ThinkingStart {
                block_id,
                signature,
                redacted,
            } => match self.start_block(block_id, BlockKind::Thinking) {
                Ok(()) => {
                    self.message
                        .content
                        .push(AssistantContent::Thinking(ThinkingContent {
                            thinking: String::new(),
                            signature,
                            redacted,
                        }));
                    None
                }
                Err(detail) => Some(self.violation(detail)),
            },
            ProtocolEvent::ThinkingDelta { block_id, delta } => {
                match self.find_block(block_id, BlockKind::Thinking, "delta") {
                    Ok(content_index) => {
                        if let AssistantContent::Thinking(thinking) =
                            &mut self.message.content[content_index]
                        {
                            thinking.thinking.push_str(&delta);
                        }
                        Some(AssistantMessageEvent::ThinkingDelta {
                            content_index,
                            delta,
                            partial: self.message.clone(),
                        })
                    }
                    Err(detail) => Some(self.violation(detail)),
                }
            }
            ProtocolEvent::ThinkingSignature {
                block_id,
                signature,
            } => match self.find_block(block_id, BlockKind::Thinking, "signature") {
                Ok(content_index) => {
                    if let AssistantContent::Thinking(thinking) =
                        &mut self.message.content[content_index]
                    {
                        thinking.signature = Some(signature);
                    }
                    None
                }
                Err(detail) => Some(self.violation(detail)),
            },
            ProtocolEvent::ThinkingEnd { block_id } => {
                match self.find_block(block_id, BlockKind::Thinking, "end") {
                    Ok(_) => {
                        self.end_block(block_id);
                        None
                    }
                    Err(detail) => Some(self.violation(detail)),
                }
            }
            ProtocolEvent::ToolCallStart { block_id, id, name } => {
                match self.start_block(block_id, BlockKind::ToolCall) {
                    Ok(()) => {
                        self.message
                            .content
                            .push(AssistantContent::ToolCall(ToolCall {
                                id,
                                name,
                                arguments: serde_json::json!({}),
                                raw_arguments: None,
                            }));
                        self.tool_raw.insert(block_id, String::new());
                        None
                    }
                    Err(detail) => Some(self.violation(detail)),
                }
            }
            ProtocolEvent::ToolCallDelta { block_id, delta } => {
                match self.find_block(block_id, BlockKind::ToolCall, "delta") {
                    Ok(content_index) => {
                        let raw = self.tool_raw.entry(block_id).or_default();
                        raw.push_str(&delta);
                        if let AssistantContent::ToolCall(tool_call) =
                            &mut self.message.content[content_index]
                        {
                            tool_call.raw_arguments = Some(raw.clone());
                        }
                        None
                    }
                    Err(detail) => Some(self.violation(detail)),
                }
            }
            ProtocolEvent::ToolCallEnd { block_id } => {
                match self.find_block(block_id, BlockKind::ToolCall, "end") {
                    Ok(content_index) => {
                        let raw = self.tool_raw.remove(&block_id).unwrap_or_default();
                        let arguments = parse_arguments(&raw);
                        if let AssistantContent::ToolCall(tool_call) =
                            &mut self.message.content[content_index]
                        {
                            tool_call.arguments = arguments;
                            tool_call.raw_arguments = Some(raw);
                        }
                        self.end_block(block_id);
                        let tool_call = match &self.message.content[content_index] {
                            AssistantContent::ToolCall(tool_call) => tool_call.clone(),
                            _ => unreachable!("content_index was assigned to a ToolCall block"),
                        };
                        Some(AssistantMessageEvent::ToolCallEnd {
                            content_index,
                            tool_call,
                            partial: self.message.clone(),
                        })
                    }
                    Err(detail) => Some(self.violation(detail)),
                }
            }
            ProtocolEvent::Usage(usage) => {
                self.message.usage = usage;
                None
            }
            ProtocolEvent::ResponseMetadata {
                response_id,
                response_model,
            } => {
                if response_id.is_some() {
                    self.message.response_id = response_id;
                }
                if response_model.is_some() {
                    self.message.response_model = response_model;
                }
                None
            }
            ProtocolEvent::Retry {
                attempt,
                max_attempts,
                delay,
                kind,
            } => Some(AssistantMessageEvent::Retry {
                attempt,
                max_attempts,
                delay,
                kind,
                partial: self.message.clone(),
            }),
            ProtocolEvent::Stop(reason) => {
                self.stopped = true;
                self.message.stop_reason = reason;
                None
            }
            ProtocolEvent::Failure {
                kind,
                message,
                diagnostics,
            } => {
                self.message.diagnostics.extend(diagnostics);
                self.message.stop_reason = StopReason::Error;
                self.message.error_kind = Some(kind);
                self.message.error_message = Some(message);
                Some(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: self.message.clone(),
                })
            }
        }
    }

    /// `Err` carries only the violation detail, not a full event — an
    /// `AssistantMessageEvent` embeds a whole `AssistantMessage` and is too
    /// large to return by value from every lookup; callers turn the detail
    /// into an event via [`Self::violation`].
    fn start_block(&mut self, block_id: u64, kind: BlockKind) -> Result<(), String> {
        if self.blocks.contains_key(&block_id) {
            return Err(format!("duplicate start for block {block_id}"));
        }
        let content_index = self.message.content.len();
        self.blocks.insert(
            block_id,
            BlockState {
                kind,
                content_index,
                ended: false,
            },
        );
        Ok(())
    }

    fn find_block(
        &mut self,
        block_id: u64,
        expected: BlockKind,
        action: &str,
    ) -> Result<usize, String> {
        match self.blocks.get(&block_id).copied() {
            None => Err(format!("{action} for unknown block {block_id}")),
            Some(block) if block.ended => {
                Err(format!("{action} for an already-ended block {block_id}"))
            }
            Some(block) if block.kind != expected => {
                Err(format!("{action} type mismatch for block {block_id}"))
            }
            Some(block) => Ok(block.content_index),
        }
    }

    fn end_block(&mut self, block_id: u64) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.ended = true;
        }
    }

    fn violation(&mut self, detail: impl Into<String>) -> AssistantMessageEvent {
        let detail = detail.into();
        self.message.stop_reason = StopReason::Error;
        self.message.error_kind = Some(ErrorKind::Protocol);
        self.message.error_message = Some(detail.clone());
        self.message
            .diagnostics
            .push(Diagnostic::new(DiagnosticCode::ProtocolViolation, detail));
        AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: self.message.clone(),
        }
    }
}

fn is_content_event(event: &ProtocolEvent) -> bool {
    matches!(
        event,
        ProtocolEvent::TextStart { .. }
            | ProtocolEvent::TextDelta { .. }
            | ProtocolEvent::TextEnd { .. }
            | ProtocolEvent::ThinkingStart { .. }
            | ProtocolEvent::ThinkingDelta { .. }
            | ProtocolEvent::ThinkingSignature { .. }
            | ProtocolEvent::ThinkingEnd { .. }
            | ProtocolEvent::ToolCallStart { .. }
            | ProtocolEvent::ToolCallDelta { .. }
            | ProtocolEvent::ToolCallEnd { .. }
    )
}

#[cfg(test)]
mod tests {
    //! `MessageAssembler` and `ProtocolEvent` are `pub(crate)` internals with
    //! no public seam an integration test can reach, so — as with
    //! `crate::sse` — its unit tests live inline.

    use super::*;

    fn assembler() -> MessageAssembler {
        MessageAssembler::new(AssistantMessage::streaming("model", "provider", "test-api"))
    }

    #[test]
    fn assigns_stable_content_index_in_start_order() {
        let mut a = assembler();
        assert!(
            a.apply(ProtocolEvent::ThinkingStart {
                block_id: 1,
                signature: None,
                redacted: false,
            })
            .is_none()
        );
        assert!(
            a.apply(ProtocolEvent::TextStart {
                block_id: 2,
                signature: None,
            })
            .is_none()
        );

        let event = a
            .apply(ProtocolEvent::ThinkingDelta {
                block_id: 1,
                delta: "hmm".into(),
            })
            .expect("delta emits an event");
        assert!(matches!(
            event,
            AssistantMessageEvent::ThinkingDelta {
                content_index: 0,
                ..
            }
        ));

        let event = a
            .apply(ProtocolEvent::TextDelta {
                block_id: 2,
                delta: "hi".into(),
            })
            .expect("delta emits an event");
        assert!(matches!(
            event,
            AssistantMessageEvent::TextDelta {
                content_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn duplicate_start_is_a_protocol_violation() {
        let mut a = assembler();
        let _ = a.apply(ProtocolEvent::TextStart {
            block_id: 1,
            signature: None,
        });
        let event = a
            .apply(ProtocolEvent::TextStart {
                block_id: 1,
                signature: None,
            })
            .expect("duplicate start is terminal");
        assert!(is_terminal(&event));
        match event {
            AssistantMessageEvent::Error { error, .. } => {
                assert_eq!(error.error_kind, Some(ErrorKind::Protocol));
            }
            _ => panic!("expected an Error event"),
        }
    }

    #[test]
    fn delta_for_unknown_block_is_a_protocol_violation() {
        let mut a = assembler();
        let event = a
            .apply(ProtocolEvent::TextDelta {
                block_id: 99,
                delta: "hi".into(),
            })
            .expect("unknown block is terminal");
        assert!(is_terminal(&event));
    }

    #[test]
    fn type_mismatch_is_a_protocol_violation() {
        let mut a = assembler();
        let _ = a.apply(ProtocolEvent::TextStart {
            block_id: 1,
            signature: None,
        });
        let event = a
            .apply(ProtocolEvent::ThinkingDelta {
                block_id: 1,
                delta: "hi".into(),
            })
            .expect("type mismatch is terminal");
        assert!(is_terminal(&event));
    }

    #[test]
    fn content_after_stop_is_a_protocol_violation() {
        let mut a = assembler();
        assert!(a.apply(ProtocolEvent::Stop(StopReason::Stop)).is_none());
        let event = a
            .apply(ProtocolEvent::TextStart {
                block_id: 1,
                signature: None,
            })
            .expect("content after Stop is terminal");
        assert!(is_terminal(&event));
    }

    #[test]
    fn end_for_already_ended_block_is_a_protocol_violation() {
        let mut a = assembler();
        let _ = a.apply(ProtocolEvent::TextStart {
            block_id: 1,
            signature: None,
        });
        assert!(a.apply(ProtocolEvent::TextEnd { block_id: 1 }).is_none());
        let event = a
            .apply(ProtocolEvent::TextEnd { block_id: 1 })
            .expect("double end is terminal");
        assert!(is_terminal(&event));
    }

    #[test]
    fn thinking_signature_updates_an_already_started_block() {
        let mut a = assembler();
        let _ = a.apply(ProtocolEvent::ThinkingStart {
            block_id: 1,
            signature: None,
            redacted: false,
        });
        assert!(
            a.apply(ProtocolEvent::ThinkingSignature {
                block_id: 1,
                signature: "sig-123".into(),
            })
            .is_none()
        );
        match &a.into_message().content[0] {
            AssistantContent::Thinking(t) => assert_eq!(t.signature.as_deref(), Some("sig-123")),
            _ => panic!("expected a thinking block"),
        }
    }

    #[test]
    fn tool_call_end_parses_accumulated_arguments() {
        let mut a = assembler();
        let _ = a.apply(ProtocolEvent::ToolCallStart {
            block_id: 1,
            id: "call_1".into(),
            name: "get_weather".into(),
        });
        let _ = a.apply(ProtocolEvent::ToolCallDelta {
            block_id: 1,
            delta: r#"{"city":"Paris"}"#.into(),
        });
        let event = a
            .apply(ProtocolEvent::ToolCallEnd { block_id: 1 })
            .expect("end emits an event");
        match event {
            AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
                ..
            } => {
                assert_eq!(content_index, 0);
                assert_eq!(tool_call.id, "call_1");
                assert_eq!(tool_call.arguments, serde_json::json!({ "city": "Paris" }));
                assert_eq!(
                    tool_call.raw_arguments.as_deref(),
                    Some(r#"{"city":"Paris"}"#)
                );
            }
            _ => panic!("expected a ToolCallEnd event"),
        }
    }

    #[test]
    fn stop_and_usage_finalize_the_message_without_emitting_an_event() {
        let mut a = assembler();
        assert!(
            a.apply(ProtocolEvent::Usage(crate::types::Usage {
                input: 10,
                output: 5,
                ..Default::default()
            }))
            .is_none()
        );
        assert!(a.apply(ProtocolEvent::Stop(StopReason::Length)).is_none());
        let message = a.into_message();
        assert_eq!(message.usage.input, 10);
        assert_eq!(message.stop_reason, StopReason::Length);
    }
}
