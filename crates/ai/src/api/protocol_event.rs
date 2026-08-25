//! The protocol-event vocabulary a [`ProtocolAdapter`](super::ProtocolAdapter)
//! emits, keyed by an adapter-generated opaque `block_id` that ties together
//! every event for the same content block.
//!
//! A [`ProtocolEventStream`] sits between a protocol adapter (which only
//! understands its own wire JSON) and the internal message assembler (which
//! assigns the stable public `content_index`, enforces block ordering, and
//! builds the assembled [`AssistantMessage`](crate::AssistantMessage)). This
//! is the minimal interface a custom adapter must learn: it carries no
//! provider-specific wire JSON and gives no access to the public
//! [`AssistantMessageEvent`](crate::AssistantMessageEvent)s.

use std::pin::Pin;
use std::time::Duration;

use futures_core::Stream;

use crate::error::ErrorKind;
use crate::types::{Diagnostic, StopReason, Usage};

/// The stream a [`ProtocolAdapter`](super::ProtocolAdapter) returns: pinned,
/// boxed, and `'static` so it can own the request's HTTP resources.
pub type ProtocolEventStream = Pin<Box<dyn Stream<Item = ProtocolEvent> + Send + 'static>>;

/// One incremental event from a protocol adapter.
///
/// Every event after a block's `*Start` reuses that block's `block_id`. A
/// well-formed stream ends with exactly one [`Stop`](Self::Stop),
/// [`StopWithRaw`](Self::StopWithRaw), or [`Failure`](Self::Failure); ending
/// the stream without one is a protocol violation the driver reports as
/// [`ErrorKind::Protocol`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ProtocolEvent {
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
    /// A signature arriving separately from the thinking text itself
    /// (Anthropic's `signature_delta`).
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
    /// A non-fatal, safe diagnostic to attach to the assembled message's
    /// `diagnostics` — e.g. the tool-image downgrade (issue #22) on a model without
    /// image input. Produces no public event; it only lands on the final
    /// [`AssistantMessage`](crate::AssistantMessage).
    Diagnostic(Diagnostic),
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
    /// The completion stopped with both its normalized classification and the
    /// exact provider value. Built-in adapters use this when the wire exposes
    /// a stop reason; custom adapters may keep using [`Self::Stop`] when no raw
    /// provider value is available.
    StopWithRaw {
        /// Stable cross-provider classification.
        reason: StopReason,
        /// Exact provider-defined wire value.
        raw_reason: String,
    },
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

impl ProtocolEvent {
    /// Build the appropriate terminal event from normalized and optional raw
    /// stop metadata. Keeping this choice here prevents adapters from each
    /// reimplementing the optional raw-reason branch.
    pub(crate) fn stop(reason: StopReason, raw_reason: Option<String>) -> Self {
        match raw_reason {
            Some(raw_reason) => Self::StopWithRaw { reason, raw_reason },
            None => Self::Stop(reason),
        }
    }

    /// Return the normalized reason carried by either successful terminal
    /// event shape.
    pub(crate) fn normalized_stop_reason(&self) -> Option<StopReason> {
        match self {
            Self::Stop(reason) | Self::StopWithRaw { reason, .. } => Some(*reason),
            _ => None,
        }
    }
}
