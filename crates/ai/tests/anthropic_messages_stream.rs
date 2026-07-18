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

    let message = provider
        .stream(&model, &context, &options)
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "Hello, world!");
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 5);
}
