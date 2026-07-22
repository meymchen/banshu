//! Seam 1: Anthropic Messages reasoning ("thinking") blocks.
//!
//! Anthropic streams thinking as a `thinking` content block with `thinking_delta`
//! fragments and a trailing `signature_delta`, ahead of the final text block.

use banshu_ai::{AssistantContent, Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"glm\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me think. \"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"2 + 2 is 4.\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-abc\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"The answer is 4.\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn assembles_thinking_then_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("zai", "Z.AI", server.uri(), ["ZAI_API_KEY"]);
    let model = Model::anthropic_messages("glm-4.6").with_base_url(server.uri());
    let context = Context::new().user("What is 2 + 2?");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider.stream(&model, &context, &options).finish().await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "The answer is 4.");

    let thinking = match &message.content[0] {
        AssistantContent::Thinking(t) => t,
        other => panic!("expected first block to be thinking, got {other:?}"),
    };
    assert_eq!(thinking.thinking, "Let me think. 2 + 2 is 4.");
    assert_eq!(thinking.signature.as_deref(), Some("sig-abc"));
}
