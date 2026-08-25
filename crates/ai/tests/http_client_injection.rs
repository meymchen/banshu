//! Issue #88: application-owned HTTP clients.
//!
//! An application injects one configured `reqwest::Client` — carrying its
//! proxy, certificate, DNS, connection-pool, and default-header policy — into
//! a provider, and every provider-owned request goes through it: built-in
//! inference, custom-adapter dispatch, Catalog Refresh, and Probe. A provider
//! given no client keeps constructing the crate default.

use std::sync::Arc;

use banshu_ai::api::openai_completions::OpenAiCompletions;
use banshu_ai::{
    ApiKind, Context, ErrorKind, Model, PreparedRequest, ProtocolAdapter, ProtocolEvent,
    ProtocolEventStream, Provider, RefreshOutcome, StopReason, StreamOptions,
};
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The identifying default header the injected client stamps on every request.
const MARKER_NAME: &str = "x-banshu-test";
/// The value [`MARKER_NAME`] carries.
const MARKER_VALUE: &str = "injected-client";

/// A client whose only policy is a default header, so the wire can prove
/// which client sent each request.
fn identifying_client() -> reqwest::Client {
    let mut headers = HeaderMap::new();
    headers.insert(MARKER_NAME, HeaderValue::from_static(MARKER_VALUE));
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .expect("valid client")
}

/// A minimal third-party protocol: POSTs a plain-text body and replies with
/// plain text, sending through the `PreparedRequest`'s client — the only
/// client a custom adapter ever sees.
struct PlainTextProtocol;

impl ProtocolAdapter for PlainTextProtocol {
    fn kind(&self) -> ApiKind {
        ApiKind::OpenAiCompletions
    }

    fn stream(&self, request: PreparedRequest) -> ProtocolEventStream {
        let http = request.http_client().clone();
        let url = format!("{}/plain", request.model().base_url.trim_end_matches('/'));
        let events = async move {
            match http.post(&url).body("hi").send().await {
                Ok(response) => match response.text().await {
                    Ok(text) => vec![
                        ProtocolEvent::TextStart {
                            block_id: 0,
                            signature: None,
                        },
                        ProtocolEvent::TextDelta {
                            block_id: 0,
                            delta: text,
                        },
                        ProtocolEvent::TextEnd { block_id: 0 },
                        ProtocolEvent::Stop(StopReason::Stop),
                    ],
                    Err(err) => vec![ProtocolEvent::Failure {
                        kind: ErrorKind::Transport,
                        message: err.to_string(),
                        diagnostics: Vec::new(),
                    }],
                },
                Err(err) => vec![ProtocolEvent::Failure {
                    kind: ErrorKind::Transport,
                    message: err.to_string(),
                    diagnostics: Vec::new(),
                }],
            }
        };
        Box::pin(futures::stream::once(events).flat_map(futures::stream::iter))
    }
}

fn model_for(provider: &str, id: &str, base_url: &str) -> Model {
    let mut model = Model::openai_completions(id).with_base_url(base_url);
    model.provider = provider.to_string();
    model
}

/// A one-line OpenAI chat-completions SSE body.
fn sse_body(text: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{text}\"}},\"finish_reason\":null}}]}}\n\ndata: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n"
    )
}

#[tokio::test]
async fn injected_client_reaches_builtin_inference() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("INFER_INJECT_KEY", "infer-k") };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer infer-k"))
        .and(header(MARKER_NAME, MARKER_VALUE))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body("via injected")),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("acme", "Acme", server.uri(), ["INFER_INJECT_KEY"])
        .with_http_client(identifying_client());

    let model = model_for("acme", "m", &server.uri());
    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "via injected");
    server.verify().await;
}

#[tokio::test]
async fn injected_client_reaches_custom_adapter_dispatch() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/plain"))
        .and(header(MARKER_NAME, MARKER_VALUE))
        .respond_with(ResponseTemplate::new(200).set_body_string("EXT OK"))
        .expect(1)
        .mount(&server)
        .await;

    let model = model_for("ext", "ext-1", &server.uri());
    let provider = Provider::builder("ext", "Ext", server.uri())
        .adapter(Arc::new(PlainTextProtocol))
        .http_client(identifying_client())
        .model(model.clone())
        .build()
        .expect("valid provider");

    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "EXT OK");
    server.verify().await;
}

#[tokio::test]
async fn catalog_refresh_uses_the_injected_client() {
    // SAFETY: a unique env var name keeps this from racing other tests; no key
    // so the probe layer is skipped and only the catalog fetch runs.
    unsafe { std::env::remove_var("CATALOG_INJECT_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .and(header(MARKER_NAME, MARKER_VALUE))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"acme":{"models":{"m":{"name":"M","reasoning":false,"modalities":{"input":["text"]},"limit":{"context":4096,"output":1024},"cost":{"input":1.0,"output":2.0,"cache_read":0.0,"cache_write":0.0}}}}}"#,
        ))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("acme", "Acme", server.uri(), ["CATALOG_INJECT_KEY"])
            .with_models_dev_id("acme")
            .with_http_client(identifying_client());
    let entry = provider
        .refresh_models_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(entry.catalog, RefreshOutcome::Refreshed);
    server.verify().await;
}

#[tokio::test]
async fn probe_uses_the_injected_client() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("PROBE_INJECT_KEY", "probe-k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer probe-k"))
        .and(header(MARKER_NAME, MARKER_VALUE))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"data":[{"id":"probed"}]}"#))
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("acme", "Acme", server.uri(), ["PROBE_INJECT_KEY"])
        .with_http_client(identifying_client());
    let entry = provider
        .refresh_models_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(entry.probe, RefreshOutcome::Refreshed);
    assert!(provider.models().iter().any(|m| m.id == "probed"));
    server.verify().await;
}

#[tokio::test]
async fn unconfigured_provider_still_completes_a_local_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body("via default")),
        )
        .expect(1)
        .mount(&server)
        .await;

    // No client injected: the provider constructs the default one itself.
    let provider = Provider::builder("acme", "Acme", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .build()
        .expect("valid provider");

    let model = model_for("acme", "m", &server.uri());
    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "via default");
    server.verify().await;
}
