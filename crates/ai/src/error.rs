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

    /// An underlying HTTP/transport error at request-construction time.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// A JSON (de)serialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias for results carrying a [`banshu-ai`](crate) [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
