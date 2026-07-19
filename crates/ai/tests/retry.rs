//! Bounded retry of pre-stream failures.
//!
//! Transient statuses (429/5xx/…) are retried with backoff and surfaced as
//! `Retry` events; Retry-After is honored when reasonable; quota and auth
//! failures are terminal immediately; the retry budget is bounded; and every
//! terminal error carries a structured `ErrorKind`.

use std::time::Duration;

use banshu_ai::{
    AssistantMessageEvent, Context, ErrorKind, Model, Provider, StopReason, StreamOptions,
};
use futures::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn protocol_errors_are_not_retryable() {
    assert!(!ErrorKind::Protocol.is_retryable());
}

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

fn retries(events: &[AssistantMessageEvent]) -> Vec<(u32, u32, Duration, ErrorKind)> {
    events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::Retry {
                attempt,
                max_attempts,
                delay,
                kind,
                ..
            } => Some((*attempt, *max_attempts, *delay, *kind)),
            _ => None,
        })
        .collect()
}

fn options_with_key() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn transient_server_error_is_retried_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500)
                .insert_header("retry-after-ms", "5")
                .set_body_string("internal error"),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(OPENAI_SSE_BODY))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());

    let events = collect_events(&provider, &model, &options_with_key()).await;

    let retries = retries(&events);
    assert_eq!(
        retries,
        vec![(1, 3, Duration::from_millis(5), ErrorKind::ServerError)],
        "expected one Retry event honoring retry-after-ms"
    );
    let Some(AssistantMessageEvent::Done { message, .. }) = events.last() else {
        panic!("expected a terminal Done event, got {:?}", events.last());
    };
    assert_eq!(message.text(), "Hello, world!");
    assert_eq!(message.error_kind, None);
}

#[tokio::test]
async fn retry_after_seconds_is_honored_for_rate_limits() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "1")
                .set_body_string("too many requests"),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(OPENAI_SSE_BODY))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());

    let events = collect_events(&provider, &model, &options_with_key()).await;

    assert_eq!(
        retries(&events),
        vec![(1, 3, Duration::from_secs(1), ErrorKind::RateLimited)]
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done { .. })
    ));
}

#[tokio::test]
async fn quota_exhausted_429_is_terminal_without_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429).set_body_string(
                "{\"error\":{\"code\":\"insufficient_quota\",\"message\":\"You exceeded your current quota\"}}",
            ),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());

    let events = collect_events(&provider, &model, &options_with_key()).await;

    assert!(retries(&events).is_empty(), "quota errors must not retry");
    let Some(AssistantMessageEvent::Error { error, .. }) = events.last() else {
        panic!("expected a terminal Error event");
    };
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.error_kind, Some(ErrorKind::QuotaExhausted));
}

#[tokio::test]
async fn kimi_style_quota_403_is_terminal_without_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(403)
                .set_body_string("You've reached your usage limit for this billing cycle"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::anthropic_compatible("kimi", "Kimi For Coding", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("kimi-for-coding").with_base_url(server.uri());

    let events = collect_events(&provider, &model, &options_with_key()).await;

    assert!(retries(&events).is_empty());
    let Some(AssistantMessageEvent::Error { error, .. }) = events.last() else {
        panic!("expected a terminal Error event");
    };
    assert_eq!(error.error_kind, Some(ErrorKind::QuotaExhausted));
}

#[tokio::test]
async fn retry_budget_is_bounded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("retry-after-ms", "5")
                .set_body_string("service unavailable"),
        )
        .expect(2)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let options = StreamOptions {
        max_retries: Some(1),
        ..options_with_key()
    };

    let events = collect_events(&provider, &model, &options).await;

    assert_eq!(
        retries(&events),
        vec![(1, 2, Duration::from_millis(5), ErrorKind::ServerError)]
    );
    let Some(AssistantMessageEvent::Error { error, .. }) = events.last() else {
        panic!("expected a terminal Error event after the budget is spent");
    };
    assert_eq!(error.error_kind, Some(ErrorKind::ServerError));
    assert!(
        error
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("503"),
        "terminal error should carry the status"
    );
}

#[tokio::test]
async fn max_retries_zero_disables_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let options = StreamOptions {
        max_retries: Some(0),
        ..options_with_key()
    };

    let events = collect_events(&provider, &model, &options).await;

    assert!(retries(&events).is_empty());
    let Some(AssistantMessageEvent::Error { error, .. }) = events.last() else {
        panic!("expected a terminal Error event");
    };
    assert_eq!(error.error_kind, Some(ErrorKind::ServerError));
}

#[tokio::test]
async fn anthropic_path_retries_overloaded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(529)
                .insert_header("retry-after-ms", "5")
                .set_body_string("{\"error\":{\"type\":\"overloaded_error\"}}"),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(ANTHROPIC_SSE_BODY))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::anthropic_compatible("kimi", "Kimi For Coding", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("kimi-for-coding").with_base_url(server.uri());

    let events = collect_events(&provider, &model, &options_with_key()).await;

    assert_eq!(
        retries(&events),
        vec![(1, 3, Duration::from_millis(5), ErrorKind::Overloaded)]
    );
    let Some(AssistantMessageEvent::Done { message, .. }) = events.last() else {
        panic!("expected a terminal Done event, got {:?}", events.last());
    };
    assert_eq!(message.text(), "Hi");
}

#[tokio::test]
async fn auth_failure_is_terminal_with_auth_kind() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());

    let events = collect_events(&provider, &model, &options_with_key()).await;

    assert!(retries(&events).is_empty());
    let Some(AssistantMessageEvent::Error { error, .. }) = events.last() else {
        panic!("expected a terminal Error event");
    };
    assert_eq!(error.error_kind, Some(ErrorKind::Auth));
}
