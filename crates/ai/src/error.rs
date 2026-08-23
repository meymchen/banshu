//! Setup / configuration errors — the `Result` side of the stream contract.
//!
//! Per the design, *in-stream* failures (transport dropping mid-response, an
//! HTTP error status) are encoded in-band as terminal `Error` events on the
//! [`MessageStream`](crate::MessageStream), never as `Result`. This type is for
//! the genuine up-front errors a caller should handle before a stream exists.

/// Errors surfaced by fallible, non-streaming calls in `banshu-ai`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No API key could be resolved for a provider (neither in options nor env).
    #[error("no API key configured for provider `{provider}`")]
    MissingApiKey {
        /// The provider that needed a key.
        provider: String,
    },

    /// A model id was requested that no registered provider owns.
    #[error("unknown model `{model}`")]
    UnknownModel {
        /// The requested model id.
        model: String,
    },

    /// A [`ProviderBuilder`](crate::ProviderBuilder) was handed an invalid
    /// configuration (empty id, no adapters, duplicate protocols, a model the
    /// provider can't serve). Surfaced as a `Result` from
    /// [`build`](crate::ProviderBuilder::build) — never a panic.
    #[error("invalid provider configuration: {0}")]
    Config(String),

    /// An [`AuthResolver`](crate::AuthResolver) failed to produce credentials.
    /// Surfaces in-band as a terminal [`ErrorKind::Auth`] error event, not as a
    /// synchronous `Result` from [`stream`](crate::Provider::stream).
    #[error("authentication failed: {0}")]
    Auth(String),

    /// An interactive login was cancelled through its
    /// [`AuthInteraction`](crate::AuthInteraction) cancellation token.
    #[error("login cancelled")]
    AuthCancelled,

    /// An interactive login exceeded its
    /// [`AuthInteraction`](crate::AuthInteraction) timeout.
    #[error("login timed out after {seconds}s")]
    AuthTimeout {
        /// The timeout that elapsed, in whole seconds.
        seconds: u64,
    },

    /// The stored credential can no longer produce a working access token —
    /// the refresh token was rejected, or none was ever stored — so only a
    /// fresh login helps. The prior credential is left in the store for
    /// diagnosis; it is never silently deleted or overwritten. Through the
    /// stream path this surfaces in-band as a terminal
    /// [`ErrorKind::Auth`] event whose message starts "re-login required".
    #[error("re-login required for provider `{provider}`: {reason}")]
    ReLoginRequired {
        /// The provider whose credential stopped working.
        provider: String,
        /// Why the credential could not be refreshed.
        reason: String,
    },

    /// An underlying HTTP/transport error at request-construction time.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// A JSON (de)serialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// A persisted context snapshot carries a version this crate cannot read.
    #[error("unsupported context snapshot version {found} (expected 1)")]
    UnsupportedSnapshotVersion {
        /// The version the snapshot declared.
        found: u32,
    },
}

/// Convenience alias for results carrying a [`banshu-ai`](crate) [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Structured classification of an in-band stream failure.
///
/// Carried on [`AssistantMessage::error_kind`](crate::AssistantMessage) next to
/// the human-readable `error_message`, and used by the retry loop to decide
/// whether a pre-stream failure is worth re-sending. Downstream callers (e.g.
/// an agent layer deciding whether to re-run a whole turn) should branch on
/// this instead of pattern-matching the message string.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorKind {
    /// 401, or 403 that doesn't look like a quota limit. Not retryable.
    Auth,
    /// Balance/quota/billing exhaustion: 402, or a 403/429 whose body matches
    /// a known quota pattern. Retrying cannot succeed until the account state
    /// changes.
    QuotaExhausted,
    /// Any other 4xx — the request itself is wrong. Not retryable.
    InvalidRequest,
    /// The request exceeded the model's context window: a 400/413 whose body
    /// carries known overflow evidence (see [`crate::is_context_overflow`]).
    /// Not retryable — resending the same request overflows again.
    ContextOverflow,
    /// A genuine rate limit (429). Retryable.
    RateLimited,
    /// The provider is overloaded (529). Retryable.
    Overloaded,
    /// 5xx, 408, or 409 — transient server-side failures. Retryable.
    ServerError,
    /// A connection-level failure (refused, reset, timeout) before any
    /// response arrived. Retryable.
    Transport,
    /// The SSE stream failed after it had started. Never retried by this crate
    /// (the "pre-stream only" contract); callers decide whether to re-run.
    StreamInterrupted,
    /// A wire state-machine, JSON, or termination-protocol error on top of a
    /// valid HTTP/SSE response. Never retryable.
    Protocol,
    /// Everything else: missing API key, an in-stream error event from the
    /// provider, unknown model dispatch. Not retryable.
    Api,
}

impl ErrorKind {
    /// Whether the retry loop (and callers) should consider re-sending.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::RateLimited | Self::Overloaded | Self::ServerError | Self::Transport
        )
    }
}

/// Body substrings that mark a 402/403/429 as quota/billing exhaustion rather
/// than a transient limit. Grounded in provider docs: OpenAI's
/// `insufficient_quota` 429, DeepSeek's 402 wording, and Kimi's "usage limit"
/// phrasing on both quota 429s and the weekly-quota 403.
const QUOTA_PATTERNS: &[&str] = &[
    "insufficient_quota",
    "quota exceeded",
    "billing",
    "out of budget",
    "usage limit",
];

/// Classify a non-2xx response by status code, sniffing the body only to
/// demote 402/403/429 to [`ErrorKind::QuotaExhausted`] and to promote 400/413
/// carrying known overflow evidence to [`ErrorKind::ContextOverflow`]. Every
/// other status keeps its shape, so rate limits, overload, and server errors
/// are never misread as overflow. Returns the matched overflow-evidence label
/// alongside the kind, so callers can name the evidence without re-matching.
pub(crate) fn classify_status(status: u16, body: &str) -> (ErrorKind, Option<&'static str>) {
    let quota_body = || {
        let lower = body.to_lowercase();
        QUOTA_PATTERNS.iter().any(|pattern| lower.contains(pattern))
    };
    let kind = match status {
        402 => ErrorKind::QuotaExhausted,
        403 | 429 if quota_body() => ErrorKind::QuotaExhausted,
        401 | 403 => ErrorKind::Auth,
        429 => ErrorKind::RateLimited,
        529 => ErrorKind::Overloaded,
        400 | 413 => match crate::overflow::http_evidence(status, body) {
            Some(label) => return (ErrorKind::ContextOverflow, Some(label)),
            None => ErrorKind::InvalidRequest,
        },
        408 | 409 | 500..=599 => ErrorKind::ServerError,
        400..=499 => ErrorKind::InvalidRequest,
        _ => ErrorKind::Api,
    };
    (kind, None)
}
