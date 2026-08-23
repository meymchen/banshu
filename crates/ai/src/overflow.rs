//! Context-overflow classification: one predicate over the error and usage
//! evidence providers give when a request exceeds the model's context window.
//!
//! Three evidence shapes are recognized, mirroring what the six target
//! providers actually do:
//!
//! 1. **Error evidence** — most providers fail the request with an overflow
//!    message: Anthropic-protocol "prompt is too long" / `request_too_large`
//!    (413), MiniMax's "context window exceeds limit", Kimi/Moonshot's
//!    "exceeded model token limit", DeepSeek/OpenAI-style "maximum context
//!    length is N tokens" / "exceeds the context window". At the HTTP layer
//!    these classify the failure as [`ErrorKind::ContextOverflow`], gated to
//!    400/413 statuses so rate limits (429), quota responses, overload (529),
//!    and server errors can never be misread as overflow.
//! 2. **Silent overflow** (z.ai) — the request succeeds but reported input
//!    usage exceeds the known context window.
//! 3. **Truncation overflow** (Xiaomi MiMo) — the server truncates the input
//!    to fill the window exactly, then stops with `length` and zero output.
//!
//! Classification is deliberately conservative: wording associated with rate
//! limiting, throttling, or overload vetoes an overflow match, and an unknown
//! (zero) context window never invents overflow from usage evidence.

use crate::error::ErrorKind;
use crate::types::{AssistantMessage, StopReason};

/// Overflow evidence patterns: case-insensitive substring chains, each
/// matching when every fragment appears in the text in order. The first
/// fragment doubles as the evidence label surfaced in diagnostics.
///
/// Grounded in the target providers' documented overflow shapes:
///
/// - Anthropic protocol (MiniMax, Kimi): "prompt is too long: N tokens > M
///   maximum", 413 `{"error":{"type":"request_too_large", …}}`
/// - MiniMax: "invalid params, context window exceeds limit"
/// - Kimi For Coding / Moonshot: "Your request exceeded model token limit: N
///   (requested: M)"
/// - DeepSeek: "This model's maximum context length is 65536 tokens. However,
///   you requested N tokens … Please reduce the length of the messages."
/// - OpenAI-compatible: "Your input exceeds the context window of this model",
///   "Input length (N) exceeds model's maximum context length (M)"
/// - z.ai: the non-standard `model_context_window_exceeded` finish reason,
///   when surfaced as error text
const OVERFLOW_PATTERNS: &[&[&str]] = &[
    &["prompt is too long"],
    &["request_too_large"],
    &["model_context_window_exceeded"],
    &["context window exceeds limit"],
    &["exceeded model token limit"],
    &["maximum context length is"],
    &["exceeds the context window"],
    &["exceeds", "maximum context length"],
    &["longer than the model's context length"],
    &["context_length_exceeded"],
    &["context length exceeded"],
    &["reduce the length of the messages"],
];

/// Wording that marks a failure as something other than overflow — rate
/// limiting, throttling, or overload — and vetoes an overflow match even when
/// an overflow pattern is also present (e.g. a 429 body that happens to quote
/// token counts).
const NON_OVERFLOW_PATTERNS: &[&str] = &[
    "rate limit",
    "too many requests",
    "throttl",
    "overloaded",
    "service unavailable",
    "temporarily unavailable",
];

/// The overflow evidence label a non-2xx response carries, if any. Gated to
/// 400/413: every other status keeps its existing classification, so 429 rate
/// limits, quota responses, 529 overload, and 5xx server errors are never
/// reclassified as overflow.
pub(crate) fn http_evidence(status: u16, body: &str) -> Option<&'static str> {
    if !matches!(status, 400 | 413) {
        return None;
    }
    match_overflow_text(body)
}

/// The label of the first overflow pattern matching `text`, or `None` when no
/// pattern matches or a non-overflow veto applies.
fn match_overflow_text(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    if NON_OVERFLOW_PATTERNS
        .iter()
        .any(|veto| lower.contains(veto))
    {
        return None;
    }
    OVERFLOW_PATTERNS
        .iter()
        .find(|chain| matches_chain(&lower, chain))
        .map(|chain| chain[0])
}

/// Every fragment appears in order (case-insensitively — `lower` is already
/// lowercased, and all fragments are lowercase literals).
fn matches_chain(lower: &str, chain: &[&str]) -> bool {
    let mut rest = lower;
    for fragment in chain {
        let Some(at) = rest.find(fragment) else {
            return false;
        };
        rest = &rest[at + fragment.len()..];
    }
    true
}

/// Whether an assembled assistant message shows context-overflow evidence.
///
/// Checks, in order: an error classified as [`ErrorKind::ContextOverflow`] at
/// the HTTP layer, overflow wording in a terminal error message (skipped when
/// the failure already carries a definitive non-overflow kind like
/// `RateLimited` or `Overloaded`), silent overflow (a successful stop whose
/// reported input usage exceeds the known window), and truncation overflow (a
/// `length` stop with zero output and input usage filling at least 99% of the
/// window).
///
/// `context_window` is the model's advertised window; pass `0` when it is
/// unknown — usage-based evidence then never triggers, and only error
/// evidence counts.
pub fn is_context_overflow(message: &AssistantMessage, context_window: u32) -> bool {
    if message.stop_reason == StopReason::Error {
        if message.error_kind == Some(ErrorKind::ContextOverflow) {
            return true;
        }
        // A definitive non-overflow classification vetoes text matching,
        // mirroring the HTTP gate: a rate limit, overload, quota, auth, or
        // transport failure is never overflow, whatever its message says.
        let vetoed = matches!(
            message.error_kind,
            Some(
                ErrorKind::Auth
                    | ErrorKind::QuotaExhausted
                    | ErrorKind::RateLimited
                    | ErrorKind::Overloaded
                    | ErrorKind::ServerError
                    | ErrorKind::Transport
            )
        );
        if !vetoed
            && let Some(error) = &message.error_message
            && match_overflow_text(error).is_some()
        {
            return true;
        }
    }

    // Prompt-side usage: input + cache read. Cache writes are excluded —
    // they are this request's prompt being stored, so counting them too
    // would double-count the same content. (Cost-tier selection uses a
    // different total — it *bills* writes; see `ModelCost::rates_for_input`.)
    let input_tokens = message.usage.input + message.usage.cache_read;
    if context_window > 0 && message.stop_reason == StopReason::Stop {
        // Silent overflow (z.ai): accepted, but usage over the window.
        if input_tokens > u64::from(context_window) {
            return true;
        }
    }
    if context_window > 0 && message.stop_reason == StopReason::Length && message.usage.output == 0
    {
        // Truncation overflow (Xiaomi MiMo): input fills the window, leaving
        // no room to generate.
        if input_tokens >= u64::from(context_window) * 99 / 100 {
            return true;
        }
    }
    false
}
