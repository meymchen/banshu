//! The internal protocol-event vocabulary a wire adapter emits, keyed by an
//! adapter-generated opaque `block_id` that ties together every event for the
//! same content block.
//!
//! A [`ProtocolEvent`] stream sits between a protocol adapter (which only
//! understands its own wire JSON) and [`super::assembler::MessageAssembler`]
//! (which assigns the stable public `content_index`, enforces block ordering,
//! and builds the assembled [`AssistantMessage`](crate::AssistantMessage)).
//! See PRD v0.3 §5.4/§6.

use std::time::Duration;

use crate::error::ErrorKind;
use crate::types::{Diagnostic, StopReason, Usage};

/// One incremental event from a protocol adapter.
///
/// Every event after a block's `*Start` reuses that block's `block_id`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum ProtocolEvent {
    /// The first event for a text block.
    TextStart {
        /// Opaque id shared by every event for this block.
        block_id: u64,
        /// Opaque provider signature for this block, if any.
        signature: Option<String>,
    },
    /// An appended chunk of text.
    TextDelta {
        /// Which block this delta belongs to.
        block_id: u64,
        /// The appended text.
        delta: String,
    },
    /// The text block is complete.
    TextEnd {
        /// Which block is ending.
        block_id: u64,
    },
    /// The first event for a thinking (reasoning) block.
    ThinkingStart {
        /// Opaque id shared by every event for this block.
        block_id: u64,
        /// Opaque provider signature for this block, if any.
        signature: Option<String>,
        /// Whether the content was redacted by provider safety filters.
        redacted: bool,
    },
    /// An appended chunk of reasoning text.
    ThinkingDelta {
        /// Which block this delta belongs to.
        block_id: u64,
        /// The appended reasoning text.
        delta: String,
    },
    /// A signature arriving separately from the thinking text itself.
    ///
    /// Not yet constructed: the OpenAI adapter captures its thinking
    /// signature at `ThinkingStart`, and no adapter emits this variant until
    /// the Anthropic migration (`signature_delta`) lands.
    #[allow(dead_code)]
    ThinkingSignature {
        /// Which block this signature belongs to.
        block_id: u64,
        /// The signature value.
        signature: String,
    },
    /// The thinking block is complete.
    ThinkingEnd {
        /// Which block is ending.
        block_id: u64,
    },
    /// The first event for a tool-call block.
    ToolCallStart {
        /// Opaque id shared by every event for this block.
        block_id: u64,
        /// Provider-assigned call id, echoed back on the tool result.
        id: String,
        /// The tool name.
        name: String,
    },
    /// An appended fragment of the tool call's arguments JSON.
    ToolCallDelta {
        /// Which block this delta belongs to.
        block_id: u64,
        /// The appended arguments fragment.
        delta: String,
    },
    /// The tool call is complete; arguments are parsed at this point.
    ToolCallEnd {
        /// Which block is ending.
        block_id: u64,
    },
    /// Token usage for the response so far.
    Usage(Usage),
    /// Provider-supplied response identifiers, when exposed.
    ResponseMetadata {
        /// A provider-supplied request id, if a recognized header carried one.
        response_id: Option<String>,
        /// Concrete routed model id, when it differs from the requested one.
        response_model: Option<String>,
    },
    /// The request failed before the response stream started and will be
    /// retried after `delay`.
    Retry {
        /// Which retry this is (1-based).
        attempt: u32,
        /// Total attempts the budget allows (initial request + retries).
        max_attempts: u32,
        /// How long the adapter will sleep before re-sending.
        delay: Duration,
        /// Classification of the failure that triggered the retry.
        kind: ErrorKind,
    },
    /// The completion stopped for this reason; no further content blocks may
    /// start after this event.
    Stop(StopReason),
    /// A terminal failure.
    Failure {
        /// Classification for `AssistantMessage.error_kind`.
        kind: ErrorKind,
        /// Human-readable, secret-free summary for `error_message`.
        message: String,
        /// Bounded, redacted detail for `AssistantMessage.diagnostics`.
        diagnostics: Vec<Diagnostic>,
    },
}
