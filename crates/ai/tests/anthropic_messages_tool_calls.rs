//! Seam 1: Anthropic Messages tool_use blocks.
//!
//! A `tool_use` content block carries id/name on `content_block_start`; its
//! JSON arguments stream as `input_json_delta.partial_json` fragments, and the
//! turn ends with `stop_reason: "tool_use"`.

use banshu_ai::{AssistantContent, Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"glm\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Paris\\\"}\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

const MALFORMED_ARGUMENTS_SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"glm\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_bad\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn assembles_a_streamed_tool_call() {
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

    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("k2p5").with_base_url(server.uri());
    let context = Context::new().user("Weather in Paris?");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider.stream(&model, &context, &options).finish().await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let tool_call = message
        .content
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("expected a tool call");
    assert_eq!(tool_call.id, "toolu_1");
    assert_eq!(tool_call.name, "get_weather");
    assert_eq!(tool_call.arguments, serde_json::json!({ "city": "Paris" }));
    assert_eq!(
        tool_call.raw_arguments.as_deref(),
        Some(r#"{"city":"Paris"}"#)
    );
}

#[tokio::test]
async fn preserves_malformed_tool_call_arguments() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(MALFORMED_ARGUMENTS_SSE_BODY),
        )
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("k2p5").with_base_url(server.uri());
    let context = Context::new().user("Weather?");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider.stream(&model, &context, &options).finish().await;

    let tool_call = message
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .expect("expected a tool call");
    assert_eq!(tool_call.arguments, serde_json::json!({}));
    assert_eq!(tool_call.raw_arguments.as_deref(), Some(r#"{"city":"#));
}
