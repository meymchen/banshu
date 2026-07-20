//! Stable serde for conversation types + `ContextSnapshotV1` (issue #10).
//!
//! The checked-in fixture `tests/fixtures/context_snapshot_v1.json` is a
//! persistence contract: once published, any change that breaks reading it is
//! a breaking format change and needs a new snapshot version.

use banshu_ai::{
    ApiKind, AssistantContent, AssistantMessage, Context, ContextSnapshotV1, Cost, Diagnostic,
    DiagnosticCode, ErrorKind, ImageContent, Message, Modality, StopReason, TextContent,
    ThinkingContent, Tool, ToolCall, ToolResultMessage, Usage, UserContent, UserMessage,
};

const FIXTURE: &str = include_str!("fixtures/context_snapshot_v1.json");

/// The in-memory value the golden fixture encodes, covering every message
/// variant and every content-block variant.
fn fixture_context() -> Context {
    let user = UserMessage {
        content: vec![
            UserContent::Text(TextContent {
                text: "What's the weather in 東京?".to_string(),
                signature: None,
            }),
            UserContent::Image(ImageContent {
                data: "aGVsbG8=".to_string(),
                mime_type: "image/png".to_string(),
            }),
        ],
        timestamp: 1_752_900_000_000,
    };

    let tool_calling_assistant = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "The user wants weather.".to_string(),
                signature: Some("sig-thinking-1".to_string()),
                redacted: false,
            }),
            AssistantContent::Text(TextContent {
                text: "Let me check.".to_string(),
                signature: Some("sig-text-1".to_string()),
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({ "city": "Tokyo" }),
                raw_arguments: Some("{\"city\":\"Tokyo\"}".to_string()),
            }),
        ],
        api: "openai-completions".to_string(),
        provider: "deepseek".to_string(),
        model: "deepseek-chat".to_string(),
        response_model: Some("deepseek-chat-v3".to_string()),
        response_id: Some("resp_123".to_string()),
        diagnostics: vec![Diagnostic::new(
            DiagnosticCode::ProviderError,
            "upstream hiccup, recovered",
        )],
        usage: Usage {
            input: 120,
            output: 45,
            cache_read: 30,
            cache_write: 8,
            cache_write_1h: Some(2),
            reasoning: Some(12),
            total_tokens: 203,
            cost: Cost {
                input: 0.1,
                output: 0.2,
                cache_read: 0.01,
                cache_write: 0.02,
                total: 0.33,
            },
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        error_kind: None,
        timestamp: 1_752_900_001_000,
    };

    let tool_result = ToolResultMessage {
        tool_call_id: "call_1".to_string(),
        tool_name: "get_weather".to_string(),
        content: vec![
            UserContent::Text(TextContent {
                text: "22°C, sunny".to_string(),
                signature: None,
            }),
            UserContent::Image(ImageContent {
                data: "c3Vubnk=".to_string(),
                mime_type: "image/jpeg".to_string(),
            }),
        ],
        is_error: false,
        timestamp: 1_752_900_002_000,
    };

    let failed_assistant = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                signature: Some("opaque-redacted-blob".to_string()),
                redacted: true,
            }),
            AssistantContent::Text(TextContent {
                text: "Partial answer".to_string(),
                signature: None,
            }),
        ],
        api: "anthropic-messages".to_string(),
        provider: "minimax".to_string(),
        model: "MiniMax-M1".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some("provider closed the stream mid-response".to_string()),
        error_kind: Some(ErrorKind::StreamInterrupted),
        timestamp: 1_752_900_003_000,
    };

    Context {
        system_prompt: Some("You are terse.".to_string()),
        messages: vec![
            Message::User(user),
            Message::Assistant(Box::new(tool_calling_assistant)),
            Message::ToolResult(tool_result),
            Message::Assistant(Box::new(failed_assistant)),
        ],
        tools: vec![Tool {
            name: "get_weather".to_string(),
            description: "Look up current weather for a city.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        }],
    }
}

/// The fixture context, serialized through `ContextSnapshotV1`.
fn fixture_snapshot_value() -> serde_json::Value {
    serde_json::to_value(ContextSnapshotV1::new(fixture_context())).expect("serialize")
}

#[test]
fn round_trips_every_message_and_content_variant() {
    let snapshot = ContextSnapshotV1::new(fixture_context());
    let json = serde_json::to_string(&snapshot).expect("serialize");
    let restored: ContextSnapshotV1 = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(snapshot, restored);
}

#[test]
fn round_trips_an_error_tool_result() {
    let context =
        Context::new().with_message(Message::ToolResult(ToolResultMessage::error_text(
            "call_9",
            "get_weather",
            "city not found",
        )));
    let json = serde_json::to_string(&ContextSnapshotV1::new(context.clone())).expect("serialize");
    let restored: ContextSnapshotV1 = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.context, context);
    assert!(json.contains("\"isError\":true"));
}

#[test]
fn golden_fixture_deserializes_to_the_expected_value() {
    let snapshot: ContextSnapshotV1 = serde_json::from_str(FIXTURE).expect("fixture deserializes");
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.context, fixture_context());
}

#[test]
fn golden_fixture_matches_serialized_output() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    assert_eq!(fixture_snapshot_value(), fixture);
}

#[test]
fn new_always_writes_version_1() {
    let snapshot = ContextSnapshotV1::new(Context::new());
    assert_eq!(snapshot.version, 1);
    let value = serde_json::to_value(&snapshot).expect("serialize");
    assert_eq!(value["version"], serde_json::json!(1));
}

#[test]
fn rejects_unknown_snapshot_versions() {
    for version in [0u32, 2, 7] {
        let json = format!("{{\"version\":{version},\"context\":{{\"messages\":[]}}}}");
        let error = serde_json::from_str::<ContextSnapshotV1>(&json)
            .expect_err("unknown version must be rejected");
        let message = error.to_string();
        assert!(
            message.contains(&format!("version {version}")) && message.contains("expected 1"),
            "error should name the offending version, got: {message}"
        );
    }
}

#[test]
fn rejects_unknown_versions_even_with_incompatible_context_shapes() {
    // A future version's context may not parse as today's `Context`; the
    // version check must win over any shape error.
    let json = r#"{"version":3,"context":{"totally":"different"}}"#;
    let error = serde_json::from_str::<ContextSnapshotV1>(json)
        .expect_err("unknown version must be rejected");
    assert!(error.to_string().contains("version 3"));
}

#[test]
fn from_json_reports_unsupported_versions_as_typed_errors() {
    let error = ContextSnapshotV1::from_json(r#"{"version":4,"context":{"whatever":true}}"#)
        .expect_err("unknown version must be rejected");
    assert!(matches!(
        error,
        banshu_ai::Error::UnsupportedSnapshotVersion { found: 4 }
    ));

    // Corrupt JSON and shape errors stay ordinary JSON errors.
    assert!(matches!(
        ContextSnapshotV1::from_json("not json").expect_err("must fail"),
        banshu_ai::Error::Json(_)
    ));

    // And the happy path parses.
    let snapshot =
        ContextSnapshotV1::from_json(FIXTURE).expect("current version parses via from_json");
    assert_eq!(snapshot.context, fixture_context());
}

#[test]
fn accepts_pi_style_string_user_content() {
    let json = r#"{
        "version": 1,
        "context": {
            "messages": [
                { "role": "user", "content": "hello there", "timestamp": 1752900005000 }
            ]
        }
    }"#;
    let snapshot: ContextSnapshotV1 = serde_json::from_str(json).expect("deserialize");
    match &snapshot.context.messages[0] {
        Message::User(user) => {
            assert_eq!(
                user.content,
                vec![UserContent::Text(TextContent {
                    text: "hello there".to_string(),
                    signature: None,
                })]
            );
        }
        other => panic!("expected a user message, got {other:?}"),
    }
}

#[test]
fn message_role_tags_use_the_stable_vocabulary() {
    let value = fixture_snapshot_value();
    let roles: Vec<&str> = value["context"]["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .map(|message| message["role"].as_str().expect("role tag"))
        .collect();
    assert_eq!(roles, ["user", "assistant", "toolResult", "assistant"]);
}

#[test]
fn content_type_tags_use_the_stable_vocabulary() {
    let value = fixture_snapshot_value();
    let types: Vec<&str> = value["context"]["messages"][1]["content"]
        .as_array()
        .expect("content array")
        .iter()
        .map(|block| block["type"].as_str().expect("type tag"))
        .collect();
    assert_eq!(types, ["thinking", "text", "toolCall"]);
    assert_eq!(
        value["context"]["messages"][0]["content"][1]["type"],
        serde_json::json!("image")
    );
}

#[test]
fn enums_serialize_to_stable_string_values() {
    assert_eq!(
        serde_json::to_value(ApiKind::OpenAiCompletions).unwrap(),
        serde_json::json!("openai-completions")
    );
    assert_eq!(
        serde_json::to_value(ApiKind::AnthropicMessages).unwrap(),
        serde_json::json!("anthropic-messages")
    );
    assert_eq!(
        serde_json::to_value(Modality::Image).unwrap(),
        serde_json::json!("image")
    );
    assert_eq!(
        serde_json::to_value(StopReason::ToolUse).unwrap(),
        serde_json::json!("toolUse")
    );
    assert_eq!(
        serde_json::to_value(ErrorKind::QuotaExhausted).unwrap(),
        serde_json::json!("quotaExhausted")
    );
    assert_eq!(
        serde_json::to_value(DiagnosticCode::ProtocolViolation).unwrap(),
        serde_json::json!("protocolViolation")
    );

    // And back: stable strings must deserialize to the same variants.
    assert_eq!(
        serde_json::from_value::<ApiKind>(serde_json::json!("anthropic-messages")).unwrap(),
        ApiKind::AnthropicMessages
    );
    assert_eq!(
        serde_json::from_value::<StopReason>(serde_json::json!("aborted")).unwrap(),
        StopReason::Aborted
    );
    assert_eq!(
        serde_json::from_value::<ErrorKind>(serde_json::json!("streamInterrupted")).unwrap(),
        ErrorKind::StreamInterrupted
    );
}

#[test]
fn external_json_is_camel_case() {
    let value = fixture_snapshot_value();
    let context = &value["context"];
    assert!(context.get("systemPrompt").is_some());

    let assistant = &context["messages"][1];
    for key in [
        "responseModel",
        "responseId",
        "stopReason",
        "diagnostics",
        "usage",
    ] {
        assert!(assistant.get(key).is_some(), "assistant should have {key}");
    }
    let usage = &assistant["usage"];
    for key in ["cacheRead", "cacheWrite", "cacheWrite1h", "totalTokens"] {
        assert!(usage.get(key).is_some(), "usage should have {key}");
    }

    let tool_result = &context["messages"][2];
    for key in ["toolCallId", "toolName", "isError"] {
        assert!(
            tool_result.get(key).is_some(),
            "tool result should have {key}"
        );
    }

    let failed = &context["messages"][3];
    assert!(failed.get("errorMessage").is_some());
    assert!(failed.get("errorKind").is_some());
}

#[test]
fn optional_fields_are_omitted_when_absent() {
    let value = fixture_snapshot_value();

    // User text block without a signature carries no signature key.
    let user_text = &value["context"]["messages"][0]["content"][0];
    assert!(user_text.get("textSignature").is_none());

    // Failed assistant has no response metadata and empty diagnostics.
    let failed = &value["context"]["messages"][3];
    assert!(failed.get("responseModel").is_none());
    assert!(failed.get("responseId").is_none());
    assert!(failed.get("diagnostics").is_none());
    assert!(failed["usage"].get("cacheWrite1h").is_none());
    assert!(failed["usage"].get("reasoning").is_none());

    // Non-redacted thinking omits the `redacted` flag (pi-ai shape).
    let thinking = &value["context"]["messages"][1]["content"][0];
    assert!(thinking.get("redacted").is_none());

    // An empty context serializes to just its messages.
    let empty = serde_json::to_value(ContextSnapshotV1::new(Context::new())).unwrap();
    assert_eq!(
        empty,
        serde_json::json!({ "version": 1, "context": { "messages": [] } })
    );
}

#[test]
fn omitted_optional_fields_deserialize_to_defaults() {
    let json = r#"{
        "version": 1,
        "context": {
            "messages": [
                {
                    "role": "toolResult",
                    "toolCallId": "call_2",
                    "toolName": "noop",
                    "content": [{ "type": "text", "text": "ok" }],
                    "isError": false,
                    "timestamp": 1752900004000
                }
            ]
        }
    }"#;
    let snapshot: ContextSnapshotV1 = serde_json::from_str(json).expect("deserialize");
    assert_eq!(snapshot.context.system_prompt, None);
    assert!(snapshot.context.tools.is_empty());
    match &snapshot.context.messages[0] {
        Message::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "call_2");
            assert_eq!(
                result.content,
                vec![UserContent::Text(TextContent {
                    text: "ok".to_string(),
                    signature: None,
                })]
            );
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
}
