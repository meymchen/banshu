//! Options controlling a single stream request.

use std::time::Duration;

/// Per-request knobs. All fields are optional; providers ignore what they
/// don't support.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// API key override. Takes precedence over provider env-var resolution.
    pub api_key: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
    /// Maximum client-side retry attempts.
    pub max_retries: Option<u32>,
}
