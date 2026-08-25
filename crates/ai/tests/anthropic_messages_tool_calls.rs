//! Seam 1: Anthropic Messages tool_use blocks.
//!
//! A `tool_use` content block carries id/name on `content_block_start`; its
//! JSON arguments stream as `input_json_delta.partial_json` fragments, and the
//! turn ends with `stop_reason: "tool_use"`. banshu streams both through:
//! identity on `ToolCallStart`, a best-effort parsed snapshot after every
//! delta, and an in-band protocol error for terminally unrepairable arguments.

use banshu_ai::{
    AssistantContent, AssistantMessageEvent, Context, ErrorKind, Model, Provider, StopReason,
    StreamOptions,
};
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"glm\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"Paris\\\"}\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// The arguments text is cut mid-string when the block stops.
const TRUNCATED_ARGUMENTS_SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"glm\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_cut\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// The arguments text is structurally corrupt — not a truncation.
const UNREPAIRABLE_ARGUMENTS_SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"glm\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_bad\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":Paris}\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

async fn server_with(body: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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
    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("k2p5").with_base_url(server.uri());
    let context = Context::new().user("Weather in Paris?");
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
    assert_eq!(tool_call.id, "toolu_1");
    assert_eq!(tool_call.name, "get_weather");
    assert_eq!(tool_call.arguments, serde_json::json!({ "city": "Paris" }));
    assert_eq!(
        tool_call.raw_arguments.as_deref(),
        Some(r#"{"city":"Paris"}"#)
    );
}

#[tokio::test]
async fn tool_call_start_exposes_identity_before_the_first_delta() {
    let server = server_with(SSE_BODY).await;
    let mut stream = stream(&server);

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start { .. })
    ));
    match stream.next().await {
        Some(AssistantMessageEvent::ToolCallStart {
            content_index,
            id,
            name,
        }) => {
            assert_eq!(content_index, 0);
            assert_eq!(id, "toolu_1");
            assert_eq!(name, "get_weather");
        }
        other => panic!("expected ToolCallStart, got {other:?}"),
    }
    // partial() already carries the identity, before any argument delta.
    let partial = tool_call(stream.partial());
    assert_eq!(partial.id, "toolu_1");
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
            assert_eq!(tool_call.id, "toolu_bad");
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
