//! Seam 1: tool-call streaming over openai-completions.
//!
//! Tool-call arguments arrive as fragments across deltas (empty on the opening
//! delta, then string chunks); banshu must accumulate them by index and parse
//! the final JSON, terminating with `StopReason::ToolUse`.

use banshu_ai::{AssistantContent, Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":8}}\n\n",
    "data: [DONE]\n\n",
);

const MALFORMED_ARGUMENTS_SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_bad\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn assembles_a_streamed_tool_call() {
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

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("What's the weather in Paris?");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider
        .stream(&model, &context, &options)
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let tool_call = message
        .content
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("expected a tool call in the assembled message");
    assert_eq!(tool_call.id, "call_abc");
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
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(MALFORMED_ARGUMENTS_SSE_BODY),
        )
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("What's the weather?");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider
        .stream(&model, &context, &options)
        .final_message()
        .await;

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
