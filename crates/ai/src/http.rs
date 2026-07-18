//! Shared HTTP client construction, SSE decoding, and retry primitives.
//!
//! Centralizes client creation so every provider shares one connection pool
//! and TLS backend, the `data:` line decoding both wire protocols consume, and
//! the pre-stream retry building blocks ([`send_once`] + [`retry_delay`]) the
//! api modules loop over.

use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;

use crate::error::{ErrorKind, classify_status};

/// Retries attempted when `StreamOptions::max_retries` is unset. Matches the
/// Anthropic/OpenAI official SDK default.
pub(crate) const DEFAULT_MAX_RETRIES: u32 = 2;

/// First-retry backoff ceiling; doubles per retry.
const BACKOFF_BASE: Duration = Duration::from_millis(500);
/// Computed backoff never exceeds this.
const BACKOFF_CAP: Duration = Duration::from_secs(8);
/// A server-provided Retry-After above this is treated as absent.
const RETRY_AFTER_MAX: Duration = Duration::from_secs(60);

/// A classified pre-stream failure from [`send_once`].
pub(crate) struct SendFailure {
    /// What went wrong, for retryability decisions and the terminal message.
    pub kind: ErrorKind,
    /// Human-readable detail (status + body, or the transport error).
    pub detail: String,
    /// Server-provided retry hint, if any.
    pub retry_after: Option<Duration>,
}

/// Send one attempt: a transport error or non-2xx status becomes a classified
/// [`SendFailure`] (consuming the error body for detail and Retry-After).
pub(crate) async fn send_once(
    builder: reqwest::RequestBuilder,
) -> Result<reqwest::Response, SendFailure> {
    match builder.send().await {
        Ok(response) if response.status().is_success() => Ok(response),
        Ok(response) => {
            let status = response.status();
            let retry_after = retry_after(response.headers());
            let body = response.text().await.unwrap_or_default();
            Err(SendFailure {
                kind: classify_status(status.as_u16(), &body),
                detail: format!("HTTP {status}: {body}"),
                retry_after,
            })
        }
        Err(err) => Err(SendFailure {
            kind: ErrorKind::Transport,
            detail: format!("request failed: {err}"),
            retry_after: None,
        }),
    }
}

/// Extract a retry hint: `retry-after-ms` (milliseconds) first, then
/// `retry-after` as decimal seconds. The HTTP-date form is not supported —
/// unparseable values are treated as absent.
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let parse = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<u64>().ok())
    };
    if let Some(ms) = parse("retry-after-ms") {
        return Some(Duration::from_millis(ms));
    }
    parse("retry-after").map(Duration::from_secs)
}

/// Delay before retry `attempt` (1-based): a server Retry-After within
/// `(0, 60s]` is used verbatim (it's a scheduling instruction — no jitter);
/// otherwise exponential backoff with full jitter, `random(0..=500ms·2ⁿ⁻¹)`
/// capped at 8s.
pub(crate) fn retry_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(hint) = retry_after
        && hint > Duration::ZERO
        && hint <= RETRY_AFTER_MAX
    {
        return hint;
    }
    let ceiling = BACKOFF_BASE
        .saturating_mul(1u32 << (attempt.saturating_sub(1)).min(10))
        .min(BACKOFF_CAP);
    Duration::from_millis(fastrand::u64(0..=ceiling.as_millis() as u64))
}

/// Build the default shared HTTP client.
pub(crate) fn build_client() -> reqwest::Client {
    reqwest::Client::builder().build().unwrap_or_default()
}

/// Decode an SSE response body into its `data:` payloads, one per yielded item.
///
/// Events are delimited by a blank line (`\n\n`); `event:` and comment lines are
/// ignored — only `data:` payloads are surfaced. A transport error mid-stream is
/// yielded as `Err` and ends the stream.
pub(crate) fn sse_data_lines(
    response: reqwest::Response,
) -> impl Stream<Item = Result<String, reqwest::Error>> {
    async_stream::try_stream! {
        // Buffer raw bytes and only decode complete event blocks. Decoding each
        // network chunk independently would corrupt a multi-byte UTF-8 character
        // split across a chunk boundary (common with CJK output); event
        // delimiters are ASCII newlines, so a drained block is always whole.
        let mut buffer: Vec<u8> = Vec::new();
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);
            while let Some(pos) = buffer.windows(2).position(|w| w == b"\n\n") {
                let block: Vec<u8> = buffer.drain(..pos + 2).collect();
                for line in String::from_utf8_lossy(&block).lines() {
                    if let Some(data) = line.trim_start().strip_prefix("data:") {
                        yield data.trim().to_string();
                    }
                }
            }
        }
    }
}
