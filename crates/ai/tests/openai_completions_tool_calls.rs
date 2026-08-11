//! Seam 1: tool-call streaming over openai-completions.
//!
//! Tool-call identity arrives on the first delta and arguments as fragments
//! afterwards; banshu must stream them through as they arrive — identity on
//! `ToolCallStart`, a best-effort parsed snapshot after every delta — and end
//! with `StopReason::ToolUse`. A terminally unrepairable arguments payload is
//! an in-band protocol error, never a fabricated `{}`.

use banshu_ai::{
    AssistantContent, AssistantMessageEvent, Context, ErrorKind, Model, Provider, StopReason,
    StreamOptions, Tool,
};
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"Paris\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":8}}\n\n",
    "data: [DONE]\n\n",
);

/// The stream ends formally, but the arguments text is cut mid-string.
const TRUNCATED_ARGUMENTS_SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_cut\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// The arguments text is structurally corrupt — not a truncation.
const UNREPAIRABLE_ARGUMENTS_SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_bad\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":Paris}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

async fn server_with(body: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    server
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn stream(server: &MockServer) -> banshu_ai::MessageStream {
    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("What's the weather in Paris?");
    provider.stream(&model, &context, &options())
}

fn tool_call(message: &banshu_ai::AssistantMessage) -> &banshu_ai::ToolCall {
    message
        .content
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("expected a tool call in the message")
}

#[tokio::test]
async fn assembles_a_streamed_tool_call() {
    let server = server_with(SSE_BODY).await;
    let message = stream(&server).finish().await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let tool_call = tool_call(&message);
    assert_eq!(tool_call.id, "call_abc");
    assert_eq!(tool_call.name, "get_weather");
    assert_eq!(tool_call.arguments, serde_json::json!({ "city": "Paris" }));
    assert_eq!(
        tool_call.raw_arguments.as_deref(),
        Some(r#"{"city":"Paris"}"#)
    );

    let tool = Tool {
        name: "get_weather".into(),
        description: "Get weather for a city".into(),
        parameters: serde_json::json!({
            "type": "object",
            "required": ["city"],
            "properties": { "city": { "type": "string" } },
            "additionalProperties": false
        }),
        strict: true,
    };
    assert_eq!(
        tool.validate_arguments(&tool_call.arguments).unwrap(),
        tool_call.arguments,
        "terminal streamed arguments should pass directly into validation"
    );
}

#[tokio::test]
async fn tool_call_start_exposes_identity_before_the_first_delta() {
    let server = server_with(SSE_BODY).await;
    let mut stream = stream(&server);

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));
    match stream.next().await {
        Some(AssistantMessageEvent::ToolCallStart {
            content_index,
            id,
            name,
        }) => {
            assert_eq!(content_index, 0);
            assert_eq!(id, "call_abc");
            assert_eq!(name, "get_weather");
        }
        other => panic!("expected ToolCallStart, got {other:?}"),
    }
    // partial() already carries the identity, before any argument delta.
    let partial = tool_call(stream.partial());
    assert_eq!(partial.id, "call_abc");
    assert_eq!(partial.name, "get_weather");
    assert_eq!(partial.arguments, serde_json::json!({}));
}

#[tokio::test]
async fn every_delta_refreshes_the_partial_arguments_snapshot() {
    let server = server_with(SSE_BODY).await;
    let mut stream = stream(&server);

    // Skip Start and ToolCallStart.
    let _ = stream.next().await;
    let _ = stream.next().await;

    match stream.next().await {
        Some(AssistantMessageEvent::ToolCallDelta {
            content_index,
            delta,
        }) => {
            assert_eq!(content_index, 0);
            assert_eq!(delta, "{\"city\":\"");
        }
        other => panic!("expected the first ToolCallDelta, got {other:?}"),
    }
    let partial = tool_call(stream.partial());
    assert_eq!(partial.arguments, serde_json::json!({ "city": "" }));
    assert_eq!(partial.raw_arguments.as_deref(), Some("{\"city\":\""));

    match stream.next().await {
        Some(AssistantMessageEvent::ToolCallDelta {
            content_index,
            delta,
        }) => {
            assert_eq!(content_index, 0);
            assert_eq!(delta, "Paris\"}");
        }
        other => panic!("expected the second ToolCallDelta, got {other:?}"),
    }
    let partial = tool_call(stream.partial());
    assert_eq!(partial.arguments, serde_json::json!({ "city": "Paris" }));
    assert_eq!(
        partial.raw_arguments.as_deref(),
        Some(r#"{"city":"Paris"}"#)
    );
}

#[tokio::test]
async fn truncated_terminal_arguments_are_repaired_best_effort() {
    let server = server_with(TRUNCATED_ARGUMENTS_SSE_BODY).await;
    let message = stream(&server).finish().await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let tool_call = tool_call(&message);
    // The cut string closes where the input ended; the raw text is untouched.
    assert_eq!(tool_call.arguments, serde_json::json!({ "city": "" }));
    assert_eq!(tool_call.raw_arguments.as_deref(), Some("{\"city\":\""));
}

#[tokio::test]
async fn unrepairable_terminal_arguments_are_an_inband_protocol_error() {
    let server = server_with(UNREPAIRABLE_ARGUMENTS_SSE_BODY).await;
    let mut stream = stream(&server);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    match events.last() {
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error,
        }) => {
            assert_eq!(error.error_kind, Some(ErrorKind::Protocol));
            // The raw text survives on the call; nothing fabricates a
            // successful empty-object parse.
            let tool_call = tool_call(error);
            assert_eq!(tool_call.id, "call_bad");
            assert_eq!(
                tool_call.raw_arguments.as_deref(),
                Some(r#"{"city":Paris}"#)
            );
        }
        other => panic!("expected a terminal protocol Error, got {other:?}"),
    }
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::ToolCallEnd { .. })),
        "an unrepairable call must not end successfully"
    );
}
