//! Read-only request observers: redacted, diagnostic visibility immediately
//! before each send and when response headers arrive.
//!
//! An observer is attached per request via
//! [`StreamOptions::observer`](crate::StreamOptions::observer) and reports on
//! every attempt of the built-in OpenAI-completions and Anthropic-messages
//! dispatch. Everything an observer sees has already been redacted at
//! construction: the URL carries no query, fragment, or userinfo; sensitive
//! header values (API keys, OAuth tokens, `Authorization`, `x-api-key`,
//! cookies, and equivalents — the same classification diagnostics use) are
//! replaced with `[REDACTED]`; and payload snapshots pass through the same
//! secret/base64 redaction pipeline as [`Diagnostic`](crate::Diagnostic)
//! messages. `Debug` output, tracing, and any error surfaced around an
//! observation therefore never contain credential material.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::{ProviderHeaders, is_sensitive_header_name};
use crate::types::redact_sensitive_text;

/// The placeholder substituted for sensitive values everywhere an observer can
/// read them. Matches the marker diagnostics use.
pub(crate) const REDACTED: &str = "[REDACTED]";

/// What the request looks like immediately before one send.
///
/// Constructed only by the crate, already redacted — see the [module
/// documentation](self) for what that covers. An observer receives this by
/// shared reference: it can read but never rewrite the outgoing request.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BeforeSendObservation {
    /// The id of the provider handling the request.
    pub provider: String,
    /// The id of the model being invoked.
    pub model: String,
    /// Which send this is, 1-based: `1` is the initial request, `2` the first
    /// retry, and so on. Every attempt is observed exactly once.
    pub attempt: u32,
    /// The request URL with any query, fragment, and userinfo stripped — a
    /// gateway that rides its key in `?api-key=…` is not exposed.
    pub url: String,
    /// The effective request headers, with sensitive values replaced by
    /// `[REDACTED]`. Header names are visible so an observer can tell *which*
    /// authentication mechanism is in use without seeing its value.
    pub headers: BTreeMap<String, String>,
    /// A snapshot of the JSON payload about to be sent, with secrets and
    /// base64 payloads (e.g. inline images) redacted in place.
    pub payload: serde_json::Value,
}

/// What is known when a response's headers arrive.
///
/// Fired for every response, whatever its status — a retryable `500` is
/// observed just like the `200` that follows it. A transport failure that
/// never receives headers produces no observation.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ResponseObservation {
    /// The id of the provider handling the request.
    pub provider: String,
    /// The id of the model being invoked.
    pub model: String,
    /// Which send this response belongs to, 1-based — the same counter
    /// [`BeforeSendObservation::attempt`] reported.
    pub attempt: u32,
    /// The HTTP status code.
    pub status: u16,
    /// The response headers, with sensitive values (including `Set-Cookie`)
    /// replaced by `[REDACTED]`.
    pub headers: BTreeMap<String, String>,
    /// A provider-supplied request id, when a recognized header
    /// (`x-request-id` / `request-id`) carried one.
    pub request_id: Option<String>,
}

/// A read-only sink for redacted request/response metadata.
///
/// Both methods have empty default implementations, so an observer may
/// implement only the half it cares about. Observations arrive on the task
/// driving the stream: an observer should be cheap and non-blocking, since a
/// slow one delays the request it observes.
///
/// # Failure behaviour
///
/// An observer cannot fail the request it watches. A panic inside either
/// method is caught at the call site: the crate logs one fixed, secret-free
/// `tracing` warning and continues the request unchanged — no credential is
/// exposed (the warning carries no request data), no send is duplicated (the
/// retry loop's state is untouched), and authentication is unaffected (the
/// observer never holds the request).
///
/// # No mutation
///
/// Observations are passed by shared reference precisely so an observer cannot
/// rewrite the outgoing payload or headers. A caller that needs to mutate
/// requests implements a custom [`ProtocolAdapter`](crate::ProtocolAdapter)
/// instead; the observer seam is deliberately diagnostic-only.
pub trait RequestObserver: Send + Sync {
    /// Called once per attempt, immediately before the request is sent.
    fn before_send(&self, observation: &BeforeSendObservation) {
        let _ = observation;
    }

    /// Called once per response when its headers arrive, whatever the status.
    fn on_response(&self, observation: &ResponseObservation) {
        let _ = observation;
    }
}

/// The per-stream state the executor fires observations from: the observer
/// plus everything about the request that does not change between attempts,
/// redacted once up front.
pub(crate) struct ObservationPlan {
    observer: Arc<dyn RequestObserver>,
    provider: String,
    model: String,
    url: String,
    headers: BTreeMap<String, String>,
    payload: serde_json::Value,
}

impl ObservationPlan {
    /// Build the plan for one stream, redacting the URL, headers, and payload
    /// snapshot once — every attempt observes the same request.
    pub(crate) fn new(
        observer: Arc<dyn RequestObserver>,
        provider: String,
        model: String,
        url: &str,
        headers: &ProviderHeaders,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            observer,
            provider,
            model,
            url: redact_url(url),
            headers: redact_request_headers(headers),
            payload: sanitize_payload(payload),
        }
    }

    /// Fire a [`RequestObserver::before_send`] for `attempt`, containing a
    /// panic so it can never disturb the request.
    pub(crate) fn before_send(&self, attempt: u32) {
        let observation = BeforeSendObservation {
            provider: self.provider.clone(),
            model: self.model.clone(),
            attempt,
            url: self.url.clone(),
            headers: self.headers.clone(),
            payload: self.payload.clone(),
        };
        observe(&self.observer, move |observer| {
            observer.before_send(&observation);
        });
    }

    /// Fire a [`RequestObserver::on_response`] for the response to `attempt`.
    pub(crate) fn on_response(
        &self,
        attempt: u32,
        status: u16,
        headers: &reqwest::header::HeaderMap,
        request_id: Option<String>,
    ) {
        let observation = ResponseObservation {
            provider: self.provider.clone(),
            model: self.model.clone(),
            attempt,
            status,
            headers: redact_response_headers(headers),
            request_id,
        };
        observe(&self.observer, move |observer| {
            observer.on_response(&observation);
        });
    }
}

/// Invoke one observer method, catching a panic so it can never fail, divert,
/// or duplicate the request. The warning is fixed text by design: it must not
/// echo request data, since a panicking observer is exactly the one whose
/// handling of secrets cannot be trusted.
fn observe(observer: &Arc<dyn RequestObserver>, call: impl FnOnce(&Arc<dyn RequestObserver>)) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| call(observer))).is_err() {
        tracing::warn!("request observer panicked; the request continues unchanged");
    }
}

/// Strip a URL down to scheme, authority, and path: no query, no fragment, no
/// userinfo. A URL that fails to parse keeps only everything before its first
/// `?` or `#` — still never the part that would carry a key.
fn redact_url(url: &str) -> String {
    match reqwest::Url::parse(url) {
        Ok(mut parsed) => {
            parsed.set_query(None);
            parsed.set_fragment(None);
            let _ = parsed.set_username("");
            let _ = parsed.set_password(None);
            parsed.into()
        }
        Err(_) => url.split(['?', '#']).next().unwrap_or_default().to_string(),
    }
}

/// Redact the effective request headers for observation: sensitive values
/// (the same classification [`RedactedHeaders`](crate) diagnostics use) become
/// `[REDACTED]`; everything else is passed through.
fn redact_request_headers(headers: &ProviderHeaders) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value.as_ref().map(|value| {
                let value = if is_sensitive_header_name(name) {
                    REDACTED.to_string()
                } else {
                    value.clone()
                };
                (name.clone(), value)
            })
        })
        .collect()
}

/// Redact response headers the same way, covering `Set-Cookie` alongside the
/// request-side credential names. Multi-valued headers keep their first value
/// — what `HeaderMap::get`, and therefore most readers, see. Non-UTF-8 values
/// are marked rather than leaked byte-for-byte.
fn redact_response_headers(headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
    let mut redacted = BTreeMap::new();
    for (name, value) in headers {
        let value = if is_sensitive_header_name(name.as_str()) {
            REDACTED.to_string()
        } else {
            value
                .to_str()
                .map(str::to_string)
                .unwrap_or_else(|_| "[non-UTF-8]".to_string())
        };
        redacted.entry(name.as_str().to_string()).or_insert(value);
    }
    redacted
}

/// Redact every string in a payload snapshot with the same pipeline
/// [`Diagnostic`](crate::Diagnostic) messages use — labeled secrets, bearer
/// tokens, data-URL payloads, and long base64 runs — without the diagnostic
/// length cap, so a prompt stays legible to the observer.
fn sanitize_payload(payload: serde_json::Value) -> serde_json::Value {
    match payload {
        serde_json::Value::String(text) => serde_json::Value::String(redact_sensitive_text(&text)),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(sanitize_payload).collect())
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, sanitize_payload(value)))
                .collect(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn headers(pairs: &[(&str, &str)]) -> ProviderHeaders {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), Some(value.to_string())))
            .collect()
    }

    #[test]
    fn url_redaction_strips_query_fragment_and_userinfo() {
        assert_eq!(
            redact_url("https://user:pw@api.example.com/v1/chat?api-key=sekret#frag"),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            redact_url("https://api.example.com/v1/chat"),
            "https://api.example.com/v1/chat"
        );
    }

    #[test]
    fn unparseable_url_redaction_drops_query_and_fragment() {
        assert_eq!(redact_url("not a url?key=sekret#frag"), "not a url");
    }

    #[test]
    fn request_header_redaction_covers_credentials_case_insensitively() {
        let redacted = redact_request_headers(&headers(&[
            ("Authorization", "Bearer sk-live"),
            ("X-Api-Key", "sk-live"),
            ("Cookie", "session=abc"),
            ("Content-Type", "application/json"),
        ]));
        assert_eq!(redacted["Authorization"], REDACTED);
        assert_eq!(redacted["X-Api-Key"], REDACTED);
        assert_eq!(redacted["Cookie"], REDACTED);
        assert_eq!(redacted["Content-Type"], "application/json");
    }

    #[test]
    fn response_header_redaction_covers_set_cookie() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("set-cookie", "session=abc".parse().unwrap());
        headers.insert("content-type", "text/event-stream".parse().unwrap());
        let redacted = redact_response_headers(&headers);
        assert_eq!(redacted["set-cookie"], REDACTED);
        assert_eq!(redacted["content-type"], "text/event-stream");
    }

    #[test]
    fn payload_sanitization_redacts_secrets_but_keeps_text() {
        let payload = serde_json::json!({
            "messages": [
                {"role": "user", "content": "hello, is api_key: sk-live-123 valid?"},
                {"role": "user", "content": format!("data:image/png;base64,{}", "a".repeat(128))},
            ],
            "note": "use Bearer tok-abc to authenticate",
            "max_tokens": 1024,
        });
        let sanitized = sanitize_payload(payload);
        let text = sanitized.to_string();
        assert!(
            !text.contains("sk-live-123"),
            "labeled secret leaked: {text}"
        );
        assert!(!text.contains("tok-abc"), "bearer token leaked: {text}");
        assert!(
            !text.contains(&"a".repeat(128)),
            "base64 payload leaked: {text}"
        );
        assert!(text.contains("hello, is api_key: [REDACTED]"));
        assert_eq!(sanitized["max_tokens"], 1024);
    }

    #[test]
    fn payload_sanitization_keeps_short_base64_like_words() {
        let payload = serde_json::json!({"content": "deadbeef is short"});
        assert_eq!(sanitize_payload(payload)["content"], "deadbeef is short");
    }

    #[test]
    fn a_panicking_observer_is_contained() {
        struct Panicking;
        impl RequestObserver for Panicking {
            fn before_send(&self, _: &BeforeSendObservation) {
                panic!("observer blew up");
            }
            fn on_response(&self, _: &ResponseObservation) {
                panic!("observer blew up");
            }
        }
        let plan = ObservationPlan::new(
            Arc::new(Panicking),
            "provider".into(),
            "model".into(),
            "https://api.example.com/v1/chat",
            &ProviderHeaders::new(),
            serde_json::json!({}),
        );
        plan.before_send(1);
        plan.on_response(1, 200, &reqwest::header::HeaderMap::new(), None);
    }

    #[test]
    fn observation_carries_redacted_request_state() {
        struct Recording(Mutex<Vec<BeforeSendObservation>>);
        impl RequestObserver for Recording {
            fn before_send(&self, observation: &BeforeSendObservation) {
                self.0.lock().unwrap().push(observation.clone());
            }
        }
        let recording = Arc::new(Recording(Mutex::new(Vec::new())));
        let plan = ObservationPlan::new(
            recording.clone(),
            "deepseek".into(),
            "deepseek-chat".into(),
            "https://api.example.com/v1/chat?api-key=sekret",
            &headers(&[("Authorization", "Bearer sk-live")]),
            serde_json::json!({"model": "deepseek-chat"}),
        );
        plan.before_send(2);
        let observed = recording.0.lock().unwrap().pop().unwrap();
        assert_eq!(observed.provider, "deepseek");
        assert_eq!(observed.model, "deepseek-chat");
        assert_eq!(observed.attempt, 2);
        assert_eq!(observed.url, "https://api.example.com/v1/chat");
        assert_eq!(observed.headers["Authorization"], REDACTED);
        let debug = format!("{observed:?}");
        assert!(!debug.contains("sekret"));
        assert!(!debug.contains("sk-live"));
    }
}
