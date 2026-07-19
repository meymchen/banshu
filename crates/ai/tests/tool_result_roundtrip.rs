//! Seam 1: a tool-use round-trip serializes back into the request — the
//! assistant's `tool_use`/`tool_calls` and the following tool result — so a
//! multi-turn agent loop is representable on both protocols.

use banshu_ai::{
    AssistantContent, AssistantMessage, Context, Diagnostic, DiagnosticCode, Message, Model,
    Provider, StreamOptions, TextContent, ToolCall, ToolResultMessage, UserContent,
};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn history() -> Context {
    let tool_call = AssistantContent::ToolCall(ToolCall {
        id: "call_1".into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({ "city": "Paris" }),
        raw_arguments: None,
    });
    Context::new()
        .user("weather in Paris?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![tool_call]),
        )))
        .with_message(Message::ToolResult(ToolResultMessage::content(
            "call_1",
            "get_weather",
            vec![
                UserContent::Text(TextContent {
                    text: "72F".into(),
                    signature: None,
                }),
                UserContent::Text(TextContent {
                    text: "and sunny".into(),
                    signature: None,
                }),
            ],
            false,
        )))
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

#[test]
fn tool_result_constructors_preserve_content_order_and_error_state() {
    let content = vec![
        UserContent::Text(TextContent {
            text: "first".into(),
            signature: None,
        }),
        UserContent::Text(TextContent {
            text: "second".into(),
            signature: None,
        }),
    ];
    let result = ToolResultMessage::content("call_1", "read", content.clone(), false);
    assert_eq!(result.content, content);
    assert!(!result.is_error);

    let text = ToolResultMessage::text("call_2", "read", "unchanged");
    assert!(matches!(
        text.content.as_slice(),
        [UserContent::Text(TextContent { text, .. })] if text == "unchanged"
    ));
    assert!(!text.is_error);

    let error = ToolResultMessage::error_text("call_3", "read", "failed");
    assert!(matches!(
        error.content.as_slice(),
        [UserContent::Text(TextContent { text, .. })] if text == "failed"
    ));
    assert!(error.is_error);
}

#[test]
fn assistant_diagnostics_are_bounded_and_serializable() {
    let diagnostic = Diagnostic::new(DiagnosticCode::ProviderError, "界".repeat(1_025));
    assert_eq!(diagnostic.message.chars().count(), 1_024);
    assert_eq!(
        serde_json::to_value(&diagnostic).expect("diagnostic should serialize"),
        serde_json::json!({
            "code": "providerError",
            "message": "界".repeat(1_024),
        })
    );

    let message = AssistantMessage::from_content(Vec::new());
    assert_eq!(message.response_id, None);
    assert!(message.diagnostics.is_empty());
}

#[test]
fn assistant_diagnostics_redact_secrets_and_base64_on_all_serde_paths() {
    let diagnostic = Diagnostic::new(
        DiagnosticCode::ProviderError,
        "Authorization: Bearer top-secret\ndata:image/png;base64,ZmFrZQ==",
    );
    assert!(!diagnostic.message.contains("top-secret"));
    assert!(!diagnostic.message.contains("ZmFrZQ=="));

    let json_diagnostic = Diagnostic::new(
        DiagnosticCode::ProviderError,
        r#"{"Authorization":"json-secret","access_token":"access-secret"}"#,
    );
    assert!(!json_diagnostic.message.contains("json-secret"));
    assert!(!json_diagnostic.message.contains("access-secret"));

    let direct = Diagnostic {
        code: DiagnosticCode::ProviderError,
        message: r#"{"client_secret":"direct-secret","image":"data:image/png;base64,ZmFrZQ=="}"#
            .into(),
    };
    let serialized = serde_json::to_string(&direct).expect("diagnostic should serialize safely");
    assert!(!serialized.contains("direct-secret"));
    assert!(!serialized.contains("ZmFrZQ=="));
    let debug = format!("{direct:?}");
    assert!(!debug.contains("direct-secret"));
    assert!(!debug.contains("ZmFrZQ=="));

    let deserialized: Diagnostic = serde_json::from_value(serde_json::json!({
        "code": "providerError",
        "message": format!("x-api-key: deserialize-secret\n{}", "界".repeat(1_025)),
    }))
    .expect("diagnostic should deserialize safely");
    assert!(!deserialized.message.contains("deserialize-secret"));
    assert!(deserialized.message.chars().count() <= 1_024);
}

#[tokio::test]
async fn openai_serializes_assistant_tool_call_and_tool_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "messages": [
                { "role": "user", "content": "weather in Paris?" },
                { "role": "assistant", "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "get_weather" } }] },
                { "role": "tool", "tool_call_id": "call_1", "content": "72F\nand sunny" },
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
                { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "call_1", "content": "72F\nand sunny" }] },
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
