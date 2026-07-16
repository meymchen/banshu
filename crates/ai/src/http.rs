//! Shared HTTP client construction and SSE decoding.
//!
//! Retry/backoff and header-merge helpers will land here as later cycles need
//! them; for now this centralizes client creation so every provider shares one
//! connection pool and TLS backend, plus the `data:` line decoding both wire
//! protocols consume.

use futures_core::Stream;
use futures_util::StreamExt;

/// Build the default shared HTTP client.
pub(crate) fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .build()
        .unwrap_or_default()
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
