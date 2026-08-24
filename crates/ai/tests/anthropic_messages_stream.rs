//! Seam 1: `Provider::anthropic_compatible().stream()` over the Anthropic
//! Messages SSE protocol.
//!
//! Wire shape borrowed from the Anthropic Messages streaming spec: `x-api-key`
//! plus `anthropic-version` headers, top-level `system`/`max_tokens`, and the
//! message_start, content_block_delta, message_delta, message_stop event
//! sequence.

use banshu_ai::{Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"glm-4.6\",\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello, world!\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn streams_a_minimal_text_completion() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_partial_json(serde_json::json!({
            "model": "glm-4.6",
            "system": [{ "type": "text", "text": "Be terse." }],
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "Say hi" }] }],
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("zai", "Z.AI", server.uri(), ["ZAI_API_KEY"]);
    let model = Model::anthropic_messages("glm-4.6").with_base_url(server.uri());
    let context = Context::new().with_system("Be terse.").user("Say hi");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider.stream(&model, &context, &options).finish().await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(message.text(), "Hello, world!");
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 5);
}

#[tokio::test]
async fn preserves_known_and_unknown_anthropic_stop_reasons() {
    for (raw, normalized) in [
        ("end_turn", StopReason::Stop),
        ("stop_sequence", StopReason::Stop),
        ("max_tokens", StopReason::Length),
        ("tool_use", StopReason::ToolUse),
        ("pause_turn", StopReason::Stop),
        ("refusal", StopReason::Stop),
        ("model_context_window_exceeded", StopReason::Stop),
        ("future_reason", StopReason::Unknown),
    ] {
        let server = MockServer::start().await;
        let body = format!(
            concat!(
                "data: {{\"type\":\"message_start\",\"message\":{{\"usage\":{{\"input_tokens\":1,\"output_tokens\":0}}}}}}\n\n",
                "data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{raw}\"}},\"usage\":{{\"output_tokens\":1}}}}\n\n",
                "data: {{\"type\":\"message_stop\"}}\n\n",
            ),
            raw = raw,
        );
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let provider = Provider::anthropic_compatible("zai", "Z.AI", server.uri(), ["ZAI_API_KEY"]);
        let model = Model::anthropic_messages("glm-4.6").with_base_url(server.uri());
        let message = provider
            .stream(
                &model,
                &Context::new().user("hi"),
                &StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
            )
            .finish()
            .await;

        assert_eq!(message.stop_reason, normalized, "raw reason: {raw}");
        assert_eq!(message.raw_stop_reason.as_deref(), Some(raw));
    }
}
