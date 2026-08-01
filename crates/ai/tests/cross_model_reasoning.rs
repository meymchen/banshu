//! Cross-model reasoning and tool-call identity normalization (issue #40).
//!
//! Reasoning state is provider-private: a thinking block's opaque signature
//! (and a text block's `textSignature`) replays verbatim only onto the exact
//! provider, API, and model id that produced it. Replayed anywhere else,
//! non-empty ordinary thinking becomes a plain text block, empty or redacted
//! thinking is omitted, and every signature is dropped. Invalid tool-call ids
//! are rewritten deterministically to `^[a-zA-Z0-9_-]{1,64}$`, with every
//! corresponding tool-result id rewritten to match.

use banshu_ai::{
    AssistantContent, AssistantMessage, Context, DiagnosticCode, Message, Model, Provider,
    StreamOptions, TextContent, ThinkingContent, ToolCall, ToolResultMessage,
};
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const OPENAI_STOP: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

const ANTHROPIC_STOP: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

async fn mount_sse(server: &MockServer, wire_path: &'static str, body: impl Into<String>) {
    Mock::given(method("POST"))
        .and(path(wire_path))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .expect(1)
        .mount(server)
        .await;
}

async fn request_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("request journal");
    serde_json::from_slice(&requests[0].body).expect("JSON request")
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn openai_model(server: &MockServer) -> Model {
    let mut model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    model.provider = "deepseek".into();
    model
}

fn anthropic_model(server: &MockServer, id: &str) -> Model {
    let mut model = Model::anthropic_messages(id).with_base_url(server.uri());
    model.provider = "kimi".into();
    model
}

/// An assistant history message with its producing provider/API/model stamped,
/// as a real stream would have stamped it.
fn assistant(api: &str, provider: &str, model: &str, content: Vec<AssistantContent>) -> Message {
    let mut message = AssistantMessage::from_content(content);
    message.api = api.into();
    message.provider = provider.into();
    message.model = model.into();
    Message::Assistant(Box::new(message))
}

fn replay(assistant_message: Message) -> Context {
    Context::new()
        .user("2+2?")
        .with_message(assistant_message)
        .user("And 3+3?")
}

fn thinking(thinking: &str, signature: Option<&str>, redacted: bool) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: thinking.into(),
        signature: signature.map(Into::into),
        redacted,
    })
}

fn signed_text(text: &str, signature: Option<&str>) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: text.into(),
        signature: signature.map(Into::into),
    })
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

#[tokio::test]
async fn anthropic_reasoning_becomes_unsigned_text_on_an_openai_model() {
    let server = MockServer::start().await;
    mount_sse(&server, "/chat/completions", OPENAI_STOP).await;

    let context = replay(assistant(
        "anthropic-messages",
        "kimi",
        "k2-thinking",
        vec![
            thinking("Anthropic reasoning. ", Some("opaque-sig"), false),
            signed_text("The answer is 4.", Some("text-sig")),
        ],
    ));
    let message = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"])
        .stream(&openai_model(&server), &context, &options())
        .finish()
        .await;

    let assistant_wire = &request_body(&server).await["messages"][1];
    assert_eq!(
        assistant_wire["content"],
        "Anthropic reasoning. The answer is 4."
    );
    let body = assistant_wire.to_string();
    assert!(
        !body.contains("opaque-sig") && !body.contains("text-sig"),
        "provider-private signatures must not leak onto the wire: {body}"
    );
    assert!(
        message
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::ReasoningDowngraded),
        "the downgrade is reported: {:?}",
        message.diagnostics
    );
}

#[tokio::test]
async fn openai_reasoning_becomes_unsigned_text_on_an_anthropic_model() {
    let server = MockServer::start().await;
    mount_sse(&server, "/v1/messages", ANTHROPIC_STOP).await;

    let context = replay(assistant(
        "openai-completions",
        "deepseek",
        "deepseek-chat",
        vec![
            thinking("OpenAI reasoning.", Some("reasoning_content"), false),
            signed_text("The answer is 4.", None),
        ],
    ));
    Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"])
        .stream(
            &anthropic_model(&server, "k2-thinking"),
            &context,
            &options(),
        )
        .finish()
        .await;

    let content = &request_body(&server).await["messages"][1]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "OpenAI reasoning.");
    assert!(content[0].get("signature").is_none());
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "The answer is 4.");
}

#[tokio::test]
async fn redacted_and_empty_thinking_is_omitted_cross_model() {
    let server = MockServer::start().await;
    mount_sse(&server, "/chat/completions", OPENAI_STOP).await;

    let context = replay(assistant(
        "anthropic-messages",
        "kimi",
        "k2-thinking",
        vec![
            thinking("", Some("OPAQUE-DATA"), true),
            thinking("   ", None, false),
            signed_text("The answer is 4.", None),
        ],
    ));
    Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"])
        .stream(&openai_model(&server), &context, &options())
        .finish()
        .await;

    let assistant_wire = &request_body(&server).await["messages"][1];
    assert_eq!(assistant_wire["content"], "The answer is 4.");
    assert!(
        !assistant_wire.to_string().contains("OPAQUE-DATA"),
        "provider-private opaque data is omitted, never leaked"
    );
}

#[tokio::test]
async fn same_model_replay_keeps_opaque_signatures() {
    let server = MockServer::start().await;
    mount_sse(&server, "/v1/messages", ANTHROPIC_STOP).await;

    let context = replay(assistant(
        "anthropic-messages",
        "kimi",
        "k2-thinking",
        vec![thinking("Let me think.", Some("opaque-sig"), false)],
    ));
    Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"])
        .stream(
            &anthropic_model(&server, "k2-thinking"),
            &context,
            &options(),
        )
        .finish()
        .await;

    let content = &request_body(&server).await["messages"][1]["content"];
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Let me think.");
    assert_eq!(content[0]["signature"], "opaque-sig");
}

#[tokio::test]
async fn a_different_model_id_from_the_same_provider_downgrades_reasoning() {
    let server = MockServer::start().await;
    mount_sse(&server, "/v1/messages", ANTHROPIC_STOP).await;

    let context = replay(assistant(
        "anthropic-messages",
        "kimi",
        "k2-thinking",
        vec![thinking("Let me think.", Some("opaque-sig"), false)],
    ));
    Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"])
        .stream(&anthropic_model(&server, "k1"), &context, &options())
        .finish()
        .await;

    let content = &request_body(&server).await["messages"][1]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Let me think.");
    assert!(content[0].get("signature").is_none());
}

#[tokio::test]
async fn invalid_tool_call_ids_are_rewritten_together_with_their_tool_results() {
    let history = || {
        Context::new()
            .user("read the file")
            .with_message(assistant(
                "anthropic-messages",
                "kimi",
                "k2-thinking",
                vec![AssistantContent::ToolCall(ToolCall {
                    id: "call:abc/123".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({}),
                    raw_arguments: None,
                })],
            ))
            .with_message(Message::ToolResult(ToolResultMessage::text(
                "call:abc/123",
                "read",
                "contents",
            )))
            .user("and now?")
    };

    let mut rewritten_ids = Vec::new();
    for _ in 0..2 {
        let server = MockServer::start().await;
        mount_sse(&server, "/v1/messages", ANTHROPIC_STOP).await;
        let message = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"])
            .stream(
                &anthropic_model(&server, "k2-thinking"),
                &history(),
                &options(),
            )
            .finish()
            .await;

        let body = request_body(&server).await;
        let call_id = body["messages"][1]["content"][0]["id"]
            .as_str()
            .expect("tool_use id")
            .to_string();
        let result_id = body["messages"][2]["content"][0]["tool_use_id"]
            .as_str()
            .expect("tool_result id");
        assert!(valid_id(&call_id), "{call_id} must be Anthropic-safe");
        assert_ne!(call_id, "call:abc/123");
        assert_eq!(call_id, result_id, "call and result ids stay paired");
        assert!(
            message
                .diagnostics
                .iter()
                .any(|d| d.code == DiagnosticCode::ToolCallIdRewritten),
            "the rewrite is reported: {:?}",
            message.diagnostics
        );
        rewritten_ids.push(call_id);
    }
    assert_eq!(
        rewritten_ids[0], rewritten_ids[1],
        "the rewrite is deterministic across requests"
    );
}
