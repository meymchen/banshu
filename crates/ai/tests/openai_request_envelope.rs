//! Request-envelope declarations for OpenAI-compatible providers (issue #89).
//!
//! An OpenAI-compatible provider declares the request envelope its endpoint
//! accepts: whether streamed usage may be requested
//! (`OpenAiCompat::streamed_usage`) and which standard output-token field
//! carries the resolved Output Budget (`OpenAiCompat::output_token_field`).
//! The undeclared default stays byte-compatible with the request bodies
//! bundled providers have always sent: `stream_options.include_usage` is
//! requested and `max_tokens` carries the budget.

use std::sync::{Arc, Mutex};

use banshu_ai::{
    BeforeSendObservation, Context, Model, OpenAiCompat, OpenAiOutputTokenField, Provider,
    RequestObserver, ResponseObservation, StreamOptions,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
    "data: [DONE]\n\n",
);

async fn sse_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .mount(&server)
        .await;
    server
}

fn provider(server: &MockServer, compat: OpenAiCompat) -> Provider {
    Provider::openai_compatible("test", "Test", server.uri(), ["TEST_API_KEY"])
        .with_openai_compat(compat)
}

fn model(server: &MockServer) -> Model {
    let mut model = Model::openai_completions("test-model").with_base_url(server.uri());
    model.provider = "test".into();
    model.max_tokens = 8;
    model
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

/// Stream one request and return the exact body the server recorded.
async fn sent_body(
    server: &MockServer,
    provider: &Provider,
    options: &StreamOptions,
) -> serde_json::Value {
    let message = provider
        .stream(&model(server), &Context::new().user("hi"), options)
        .finish()
        .await;
    assert_eq!(message.error_kind, None, "{message:?}");
    let requests = server.received_requests().await.expect("request journal");
    assert_eq!(requests.len(), 1);
    serde_json::from_slice(&requests[0].body).expect("JSON request")
}

#[tokio::test]
async fn the_default_envelope_requests_streamed_usage_and_caps_with_max_tokens() {
    let server = sse_server().await;
    let body = sent_body(
        &server,
        &provider(&server, OpenAiCompat::default()),
        &options(),
    )
    .await;

    assert_eq!(
        body["stream_options"],
        serde_json::json!({"include_usage": true}),
        "{body}"
    );
    assert_eq!(body["max_tokens"], 8, "{body}");
    assert!(body.get("max_completion_tokens").is_none(), "{body}");
}

#[tokio::test]
async fn an_endpoint_without_streamed_usage_is_sent_no_stream_options_at_all() {
    let server = sse_server().await;
    let compat = OpenAiCompat {
        streamed_usage: false,
        ..OpenAiCompat::default()
    };
    let body = sent_body(&server, &provider(&server, compat), &options()).await;

    assert!(body.get("stream_options").is_none(), "{body}");
    assert_eq!(body["max_tokens"], 8, "{body}");
}

#[tokio::test]
async fn max_completion_tokens_carries_the_budget_and_max_tokens_is_absent() {
    let server = sse_server().await;
    let compat = OpenAiCompat {
        output_token_field: OpenAiOutputTokenField::MaxCompletionTokens,
        ..OpenAiCompat::default()
    };
    let body = sent_body(&server, &provider(&server, compat), &options()).await;

    assert_eq!(body["max_completion_tokens"], 8, "{body}");
    assert!(body.get("max_tokens").is_none(), "{body}");
    // An explicit caller budget rides the same selected field, exactly.
    let server = sse_server().await;
    let body = sent_body(
        &server,
        &provider(&server, compat),
        &StreamOptions {
            max_tokens: Some(4),
            ..options()
        },
    )
    .await;
    assert_eq!(body["max_completion_tokens"], 4, "{body}");
    assert!(body.get("max_tokens").is_none(), "{body}");
}

/// Records the payload of every observed attempt.
#[derive(Default)]
struct PayloadObserver {
    payloads: Mutex<Vec<serde_json::Value>>,
}

impl RequestObserver for PayloadObserver {
    fn before_send(&self, observation: &BeforeSendObservation) {
        self.payloads
            .lock()
            .unwrap()
            .push(observation.payload.clone());
    }

    fn on_response(&self, _observation: &ResponseObservation) {}
}

#[tokio::test]
async fn the_observed_payload_is_the_exact_body_the_server_recorded() {
    let server = sse_server().await;
    let compat = OpenAiCompat {
        streamed_usage: false,
        output_token_field: OpenAiOutputTokenField::MaxCompletionTokens,
        ..OpenAiCompat::default()
    };
    let observer = Arc::new(PayloadObserver::default());
    let options = StreamOptions {
        observer: Some(observer.clone()),
        ..options()
    };
    let recorded = sent_body(&server, &provider(&server, compat), &options).await;

    let payloads = observer.payloads.lock().unwrap();
    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0], recorded);
    // The policies are visible in the observation itself.
    assert!(payloads[0].get("stream_options").is_none(), "{recorded}");
    assert_eq!(payloads[0]["max_completion_tokens"], 8, "{recorded}");
}
