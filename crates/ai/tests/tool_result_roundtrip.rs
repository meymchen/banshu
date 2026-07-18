//! Seam 1: a tool-use round-trip serializes back into the request — the
//! assistant's `tool_use`/`tool_calls` and the following tool result — so a
//! multi-turn agent loop is representable on both protocols.

use banshu_ai::{
    AssistantContent, AssistantMessage, Context, Message, Model, Provider, StreamOptions, ToolCall,
};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn history() -> Context {
    let tool_call = AssistantContent::ToolCall(ToolCall {
        id: "call_1".into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({ "city": "Paris" }),
    });
    Context::new()
        .user("weather in Paris?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![tool_call]),
        )))
        .tool_result("call_1", "get_weather", "72F and sunny")
}

const OPENAI_STOP: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
);

const ANTHROPIC_STOP: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn openai_serializes_assistant_tool_call_and_tool_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "messages": [
                { "role": "user", "content": "weather in Paris?" },
                { "role": "assistant", "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "get_weather" } }] },
                { "role": "tool", "tool_call_id": "call_1", "content": "72F and sunny" },
            ],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(OPENAI_STOP))
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("d", "D", server.uri(), ["X"]);
    let model = Model::openai_completions("m").with_base_url(server.uri());
    let options = StreamOptions {
        api_key: Some("k".into()),
        ..Default::default()
    };
    provider
        .stream(&model, &history(), &options)
        .final_message()
        .await;
}

#[tokio::test]
async fn anthropic_serializes_tool_use_and_tool_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(serde_json::json!({
            "messages": [
                { "role": "user", "content": "weather in Paris?" },
                { "role": "assistant", "content": [{ "type": "tool_use", "id": "call_1", "name": "get_weather", "input": { "city": "Paris" } }] },
                { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "call_1", "content": "72F and sunny" }] },
            ],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(ANTHROPIC_STOP))
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("k", "K", server.uri(), ["X"]);
    let model = Model::anthropic_messages("m").with_base_url(server.uri());
    let options = StreamOptions {
        api_key: Some("k".into()),
        ..Default::default()
    };
    provider
        .stream(&model, &history(), &options)
        .final_message()
        .await;
}
