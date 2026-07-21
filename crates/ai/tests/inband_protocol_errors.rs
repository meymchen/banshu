//! In-stream protocol failures must terminate as `Error`, never `Done`: an
//! OpenAI JSON error payload, an Anthropic `event: error` / `type: error`
//! frame, and malformed `data:` JSON that doesn't match either shape.

use banshu_ai::{Context, ErrorKind, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sse_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

fn options_with_key() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn openai_inband_json_error_terminates_as_error_not_done() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        "data: {\"error\":{\"message\":\"context length exceeded\",\"code\":\"context_length_exceeded\"}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Api));
    let error = message.error_message.expect("expected an error message");
    assert!(error.contains("context length exceeded"));
    assert!(error.contains("context_length_exceeded"));
}

#[tokio::test]
async fn anthropic_event_error_terminates_as_error_not_done() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let provider =
        Provider::anthropic_compatible("kimi", "Kimi For Coding", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("kimi-for-coding").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Api));
    let error = message.error_message.expect("expected an error message");
    assert!(error.contains("Overloaded"));
}

#[tokio::test]
async fn anthropic_type_error_without_named_event_terminates_as_error() {
    let server = MockServer::start().await;
    // No `event: error` line — only the data payload's own `"type":"error"`.
    let body = "data: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"internal error\"}}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let provider =
        Provider::anthropic_compatible("kimi", "Kimi For Coding", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("kimi-for-coding").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Api));
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("internal error")
    );
}

#[tokio::test]
async fn openai_corrupted_json_does_not_produce_done() {
    let server = MockServer::start().await;
    let body = "data: {not valid json at all\n\n";
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Protocol));
}

#[tokio::test]
async fn anthropic_corrupted_json_does_not_produce_done() {
    let server = MockServer::start().await;
    let body = "data: {not valid json at all\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;

    let provider =
        Provider::anthropic_compatible("kimi", "Kimi For Coding", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("kimi-for-coding").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Protocol));
}
