//! Read-only request observers (issue #55).
//!
//! Every attempt is observed once before send with a redacted URL, provider,
//! model, attempt number, and payload snapshot; every response is observed
//! when its headers arrive with status, redacted headers, and the provider
//! request id. Credentials never reach an observation — not in the fixtures,
//! not in their `Debug` output — and a panicking observer can neither fail
//! nor duplicate the request it watches.

use std::sync::{Arc, Mutex};

use banshu_ai::{
    AssistantMessageEvent, BeforeSendObservation, Context, Model, Provider, RequestObserver,
    ResponseObservation, StopReason, StreamOptions,
};
use futures::StreamExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const API_KEY: &str = "sk-live-secret-123";

const OPENAI_SSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello, world!\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
    "data: [DONE]\n\n",
);

const ANTHROPIC_SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

fn sse_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

/// One observed call, in the order the observer received them.
#[derive(Debug)]
enum Recorded {
    BeforeSend(BeforeSendObservation),
    Response(ResponseObservation),
}

#[derive(Default)]
struct RecordingObserver {
    log: Mutex<Vec<Recorded>>,
}

impl RecordingObserver {
    fn take(&self) -> Vec<Recorded> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

impl RequestObserver for RecordingObserver {
    fn before_send(&self, observation: &BeforeSendObservation) {
        self.log
            .lock()
            .unwrap()
            .push(Recorded::BeforeSend(observation.clone()));
    }

    fn on_response(&self, observation: &ResponseObservation) {
        self.log
            .lock()
            .unwrap()
            .push(Recorded::Response(observation.clone()));
    }
}

/// A one-line summary of an observed call, for asserting observation order.
fn summary(recorded: &Recorded) -> String {
    match recorded {
        Recorded::BeforeSend(before) => format!("send#{}", before.attempt),
        Recorded::Response(response) => {
            format!("response#{}:{}", response.attempt, response.status)
        }
    }
}

fn options_with_observer(observer: Arc<RecordingObserver>) -> StreamOptions {
    StreamOptions {
        api_key: Some(API_KEY.into()),
        observer: Some(observer),
        ..Default::default()
    }
}

async fn collect_events(
    provider: &Provider,
    model: &Model,
    options: &StreamOptions,
) -> Vec<AssistantMessageEvent> {
    let context = Context::new().user("hi");
    let mut stream = provider.stream(model, &context, options);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

fn openai_fixture(server: &MockServer) -> (Provider, Model) {
    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    (provider, model)
}

#[tokio::test]
async fn before_send_carries_redacted_url_provider_model_attempt_and_payload() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(OPENAI_SSE_BODY))
        .expect(1)
        .mount(&server)
        .await;
    let (provider, model) = openai_fixture(&server);
    let observer = Arc::new(RecordingObserver::default());

    let events = collect_events(&provider, &model, &options_with_observer(observer.clone())).await;
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done { .. })
    ));

    let log = observer.take();
    let [Recorded::BeforeSend(before), Recorded::Response(_)] = &log[..] else {
        panic!("expected one before-send and one response observation, got {log:?}");
    };
    assert_eq!(before.provider, "deepseek");
    assert_eq!(before.model, "deepseek-chat");
    assert_eq!(before.attempt, 1);
    assert_eq!(before.url, format!("{}/chat/completions", server.uri()));
    assert!(
        !before.url.contains('?'),
        "the observed URL must carry no query: {}",
        before.url
    );
    assert_eq!(before.payload["model"], "deepseek-chat");
    assert_eq!(before.payload["messages"][0]["content"], "hi");
    assert_eq!(before.headers["Authorization"], "[REDACTED]");
    assert_eq!(before.headers["Content-Type"], "application/json");
}

#[tokio::test]
async fn response_observation_carries_status_redacted_headers_and_request_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            sse_response(OPENAI_SSE_BODY)
                .insert_header("x-request-id", "req-42")
                .insert_header("x-ratelimit-remaining", "42")
                .insert_header("set-cookie", "session=secretcookie"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (provider, model) = openai_fixture(&server);
    let observer = Arc::new(RecordingObserver::default());

    collect_events(&provider, &model, &options_with_observer(observer.clone())).await;

    let log = observer.take();
    let [_, Recorded::Response(response)] = &log[..] else {
        panic!("expected a response observation, got {log:?}");
    };
    assert_eq!(response.attempt, 1);
    assert_eq!(response.status, 200);
    assert_eq!(response.request_id.as_deref(), Some("req-42"));
    assert_eq!(response.headers["set-cookie"], "[REDACTED]");
    assert_eq!(response.headers["x-ratelimit-remaining"], "42");
    let debug = format!("{response:?}");
    assert!(
        !debug.contains("secretcookie"),
        "Set-Cookie value leaked into Debug: {debug}"
    );
}

#[tokio::test]
async fn credentials_reach_the_server_but_never_the_observer() {
    let server = MockServer::start().await;
    // The server demands the real credential: observation is read-only, so the
    // redaction the observer sees must not have touched the actual request.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", format!("Bearer {API_KEY}")))
        .respond_with(sse_response(OPENAI_SSE_BODY))
        .expect(1)
        .mount(&server)
        .await;
    let (provider, model) = openai_fixture(&server);
    let observer = Arc::new(RecordingObserver::default());
    let options = options_with_observer(observer.clone());

    let events = collect_events(&provider, &model, &options).await;
    assert!(
        matches!(events.last(), Some(AssistantMessageEvent::Done { .. })),
        "the authenticated request must succeed unaltered: {events:?}"
    );
    assert!(
        !format!("{options:?}").contains(API_KEY),
        "StreamOptions Debug leaked the API key"
    );

    let log = observer.take();
    for recorded in &log {
        let debug = format!("{recorded:?}");
        assert!(
            !debug.contains(API_KEY),
            "API key leaked into an observation's Debug: {debug}"
        );
    }
    let Recorded::BeforeSend(before) = &log[0] else {
        panic!("expected a before-send observation first, got {log:?}");
    };
    assert_eq!(before.headers["Authorization"], "[REDACTED]");
    // The header *name* stays visible: an observer may learn which auth
    // mechanism is in use, never its value.
    assert!(
        before.headers.contains_key("Authorization"),
        "the auth header's presence should be observable: {:?}",
        before.headers
    );
}

#[tokio::test]
async fn a_panicking_observer_never_disturbs_the_request() {
    struct Panicking;
    impl RequestObserver for Panicking {
        fn before_send(&self, _: &BeforeSendObservation) {
            panic!("before_send blew up");
        }
        fn on_response(&self, _: &ResponseObservation) {
            panic!("on_response blew up");
        }
    }

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(OPENAI_SSE_BODY))
        // Exactly one send: a panicking observer must not duplicate requests.
        .expect(1)
        .mount(&server)
        .await;
    let (provider, model) = openai_fixture(&server);
    let options = StreamOptions {
        api_key: Some(API_KEY.into()),
        observer: Some(Arc::new(Panicking)),
        ..Default::default()
    };

    let events = collect_events(&provider, &model, &options).await;
    let Some(AssistantMessageEvent::Done { message, .. }) = events.last() else {
        panic!("a panicking observer must not fail the request: {events:?}");
    };
    assert_eq!(message.text(), "Hello, world!");
}

#[tokio::test]
async fn each_retry_attempt_is_observed_once_in_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500)
                .insert_header("retry-after-ms", "5")
                .insert_header("x-request-id", "req-failed")
                .set_body_string("internal error"),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(OPENAI_SSE_BODY).insert_header("x-request-id", "req-ok"))
        .expect(1)
        .mount(&server)
        .await;
    let (provider, model) = openai_fixture(&server);
    let observer = Arc::new(RecordingObserver::default());

    let events = collect_events(&provider, &model, &options_with_observer(observer.clone())).await;
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done { .. })
    ));

    let summary: Vec<String> = observer.take().iter().map(summary).collect();
    assert_eq!(
        summary,
        vec!["send#1", "response#1:500", "send#2", "response#2:200"],
        "each attempt must be observed exactly once, in order"
    );
}

#[tokio::test]
async fn transport_failures_observe_no_response() {
    // Nothing listens here, so no response headers ever arrive.
    let provider = Provider::openai_compatible(
        "deepseek",
        "DeepSeek",
        "http://127.0.0.1:1",
        ["DEEPSEEK_API_KEY"],
    );
    let model = Model::openai_completions("deepseek-chat").with_base_url("http://127.0.0.1:1");
    let observer = Arc::new(RecordingObserver::default());
    let options = StreamOptions {
        max_retries: Some(0),
        ..options_with_observer(observer.clone())
    };

    let events = collect_events(&provider, &model, &options).await;
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error { .. })
    ));

    let summary: Vec<String> = observer.take().iter().map(summary).collect();
    assert_eq!(summary, vec!["send#1"]);
}

#[tokio::test]
async fn anthropic_requests_are_observed_with_redacted_api_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(ANTHROPIC_SSE_BODY))
        .expect(1)
        .mount(&server)
        .await;
    let provider =
        Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("kimi-for-coding").with_base_url(server.uri());
    let observer = Arc::new(RecordingObserver::default());

    let events = collect_events(&provider, &model, &options_with_observer(observer.clone())).await;
    let Some(AssistantMessageEvent::Done { reason, .. }) = events.last() else {
        panic!("expected a terminal Done, got {events:?}");
    };
    assert_eq!(*reason, StopReason::Stop);

    let log = observer.take();
    let Recorded::BeforeSend(before) = &log[0] else {
        panic!("expected a before-send observation first, got {log:?}");
    };
    assert_eq!(before.url, format!("{}/v1/messages", server.uri()));
    assert_eq!(before.headers["x-api-key"], "[REDACTED]");
    assert!(
        !format!("{log:?}").contains(API_KEY),
        "API key leaked into the Anthropic observations"
    );
}
