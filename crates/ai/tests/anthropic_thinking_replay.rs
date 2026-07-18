//! Thinking replay on the anthropic-messages path.
//!
//! Signed thinking round-trips as `thinking` blocks with their signature;
//! signatureless thinking is downgraded to a text block unless the provider
//! declares `allow_empty_signature`; `redacted_thinking` blocks round-trip
//! verbatim via their opaque payload.

use banshu_ai::{
    AnthropicCompat, AssistantContent, AssistantMessage, Context, Message, Model, Provider,
    StreamOptions, TextContent, ThinkingContent,
};
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const STOP_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

async fn mount_sse(server: &MockServer, body: impl Into<String>) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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

fn provider(server: &MockServer) -> Provider {
    Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"])
}

fn model(server: &MockServer) -> Model {
    Model::anthropic_messages("k2-thinking").with_base_url(server.uri())
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn history(assistant_content: Vec<AssistantContent>) -> Context {
    Context::new()
        .user("2+2?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(assistant_content),
        )))
        .user("And 3+3?")
}

fn thinking(text: &str, signature: Option<&str>) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: text.into(),
        signature: signature.map(Into::into),
        redacted: false,
    })
}

fn text(body: &str) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: body.into(),
        signature: None,
    })
}

#[tokio::test]
async fn replays_signed_thinking_blocks() {
    let server = MockServer::start().await;
    mount_sse(&server, STOP_BODY).await;

    let context = history(vec![
        thinking("Let me think.", Some("sig-abc")),
        text("The answer is 4."),
    ]);
    provider(&server)
        .stream(&model(&server), &context, &options())
        .final_message()
        .await;

    let content = &request_body(&server).await["messages"][1]["content"];
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Let me think.");
    assert_eq!(content[0]["signature"], "sig-abc");
    assert_eq!(content[1]["type"], "text");
}

#[tokio::test]
async fn downgrades_signatureless_thinking_to_text() {
    let server = MockServer::start().await;
    mount_sse(&server, STOP_BODY).await;

    let context = history(vec![
        thinking("Let me think.", None),
        text("The answer is 4."),
    ]);
    provider(&server)
        .stream(&model(&server), &context, &options())
        .final_message()
        .await;

    let content = &request_body(&server).await["messages"][1]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Let me think.");
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "The answer is 4.");
}

#[tokio::test]
async fn preserves_empty_signature_when_allowed() {
    let server = MockServer::start().await;
    mount_sse(&server, STOP_BODY).await;

    let context = history(vec![
        thinking("Let me think.", None),
        text("The answer is 4."),
    ]);
    provider(&server)
        .with_anthropic_compat(AnthropicCompat {
            allow_empty_signature: true,
            ..AnthropicCompat::default()
        })
        .stream(&model(&server), &context, &options())
        .final_message()
        .await;

    let content = &request_body(&server).await["messages"][1]["content"];
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Let me think.");
    assert_eq!(content[0]["signature"], "");
}

#[tokio::test]
async fn round_trips_redacted_thinking() {
    // Capture: a redacted_thinking block arrives with an opaque payload.
    let capture = MockServer::start().await;
    mount_sse(
        &capture,
        concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"OPAQUE\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ),
    )
    .await;
    let message = provider(&capture)
        .stream(&model(&capture), &Context::new().user("hi"), &options())
        .final_message()
        .await;

    let block = message
        .content
        .iter()
        .find_map(|c| match c {
            AssistantContent::Thinking(t) => Some(t),
            _ => None,
        })
        .expect("expected a redacted thinking block");
    assert!(block.redacted);
    assert_eq!(block.signature.as_deref(), Some("OPAQUE"));

    // Replay: the block goes back verbatim as redacted_thinking.
    let replay = MockServer::start().await;
    mount_sse(&replay, STOP_BODY).await;
    let context = Context::new()
        .user("hi")
        .with_message(Message::Assistant(Box::new(message)))
        .user("go on");
    provider(&replay)
        .stream(&model(&replay), &context, &options())
        .final_message()
        .await;

    let content = &request_body(&replay).await["messages"][1]["content"];
    assert_eq!(content[0]["type"], "redacted_thinking");
    assert_eq!(content[0]["data"], "OPAQUE");
}
