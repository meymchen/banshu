//! Seam 1: tool definitions from `Context.tools` are serialized into the
//! request body for both wire protocols (OpenAI `tools[].function`, Anthropic
//! `tools[].input_schema`).

use banshu_ai::{Context, Model, Provider, StreamOptions, Tool};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn weather_tool() -> Tool {
    Tool {
        name: "get_weather".into(),
        description: "Get the weather for a city".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    }
}

const OPENAI_STOP: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
);

const ANTHROPIC_STOP: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn openai_completions_sends_tool_definitions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": { "type": "object", "required": ["city"] },
                },
            }],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(OPENAI_STOP))
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("weather?").with_tool(weather_tool());
    let options = StreamOptions {
        api_key: Some("k".into()),
        ..Default::default()
    };

    provider.stream(&model, &context, &options).finish().await;
}

#[tokio::test]
async fn anthropic_messages_sends_tool_definitions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(serde_json::json!({
            "tools": [{
                "name": "get_weather",
                "description": "Get the weather for a city",
                "input_schema": { "type": "object", "required": ["city"] },
            }],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(ANTHROPIC_STOP))
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"]);
    let model = Model::anthropic_messages("k2p5").with_base_url(server.uri());
    let context = Context::new().user("weather?").with_tool(weather_tool());
    let options = StreamOptions {
        api_key: Some("k".into()),
        ..Default::default()
    };

    provider.stream(&model, &context, &options).finish().await;
}
