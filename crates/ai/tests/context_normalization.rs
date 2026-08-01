//! The single Context normalization pass in front of both protocol adapters
//! (issue #39).
//!
//! One normalized copy is built per request and handed to whichever adapter
//! runs, so cross-model rules live in exactly one place. This file pins the
//! rules the pass owns today: the newest-user modality gate still fails before
//! any HTTP request, a *historical* user image is replaced with the fixed text
//! `(image omitted: model does not support images)` on a text-only model, and
//! the caller's `Context` is deeply equal before and after the stream — the
//! same value can be reused against another model.
//!
//! Tool-result image downgrade rides the same pass; its per-protocol wire
//! shapes and `ImageDowngraded` diagnostics are pinned by
//! `tool_result_images.rs`, its run-collapsing here.
//!
//! Tool-history repair (issue #41) rides the pass too: the last two tests pin
//! the synthetic `No result provided` error result an orphaned historical tool
//! call receives on each protocol's wire, and the absence of assistant turns
//! that ended in `Error` or `Aborted`.

use banshu_ai::{
    AssistantContent, AssistantMessage, Context, DiagnosticCode, ErrorKind, ImageContent, Message,
    Modality, Model, Provider, StopReason, StreamOptions, TextContent, ToolCall, ToolResultMessage,
    UserContent, UserMessage,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const OPENAI_STOP_BODY: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

const ANTHROPIC_STOP_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// A 1x1 transparent PNG.
const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// The fixed text replacing a *user* image the model cannot see.
const USER_PLACEHOLDER: &str = "(image omitted: model does not support images)";

/// Its tool-result counterpart (issue #22).
const TOOL_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

const PROMPT: &str = "what is in this picture?";
const FOLLOW_UP: &str = "describe it in one word";

/// A downgraded user turn on the wire: an image-free message keeps the
/// plain-string shape, so the placeholder joins the text it followed.
fn downgraded_prompt() -> String {
    format!("{PROMPT}{USER_PLACEHOLDER}")
}

async fn mount_openai_sse(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(OPENAI_STOP_BODY),
        )
        .expect(1)
        .mount(server)
        .await;
}

async fn mount_anthropic_sse(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(ANTHROPIC_STOP_BODY),
        )
        .expect(1)
        .mount(server)
        .await;
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn text(text: &str) -> UserContent {
    UserContent::Text(TextContent {
        text: text.into(),
        signature: None,
    })
}

fn image() -> UserContent {
    UserContent::Image(ImageContent {
        data: PNG_B64.into(),
        mime_type: "image/png".into(),
    })
}

fn image_message() -> Message {
    Message::User(UserMessage {
        content: vec![text(PROMPT), image()],
        timestamp: 0,
    })
}

/// An older user turn carrying an image, followed by a newest text-only turn —
/// the shape the modality gate lets through and the normalizer downgrades.
fn history_with_image() -> Context {
    Context::new().with_message(image_message()).user(FOLLOW_UP)
}

fn text_only_openai_model(server: &MockServer) -> Model {
    Model::openai_completions("test-model").with_base_url(server.uri())
}

fn text_only_anthropic_model(server: &MockServer) -> Model {
    Model::anthropic_messages("test-model").with_base_url(server.uri())
}

/// The zai catalog declares glm-4.5v image-capable (issue #21).
fn openai_image_model(server: &MockServer) -> Model {
    let model = Provider::zai()
        .models()
        .iter()
        .find(|m| m.id == "glm-4.5v")
        .expect("glm-4.5v should be in the zai catalog")
        .clone();
    assert!(model.input.contains(&Modality::Image));
    model.with_base_url(server.uri())
}

async fn request_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("request journal");
    serde_json::from_slice(&requests[0].body).expect("JSON request")
}

#[tokio::test]
async fn openai_historical_user_image_is_replaced_with_the_placeholder() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let message = provider
        .stream(
            &text_only_openai_model(&server),
            &history_with_image(),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(body["messages"][0]["content"], json!(downgraded_prompt()));
    assert!(
        !body.to_string().contains(PNG_B64),
        "no image data may reach a text-only model"
    );
    assert_eq!(message.error_kind, None);
    assert_eq!(message.diagnostics.len(), 1);
    assert_eq!(message.diagnostics[0].code, DiagnosticCode::ImageDowngraded);
}

#[tokio::test]
async fn anthropic_historical_user_image_is_replaced_with_the_placeholder() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let provider = Provider::anthropic_compatible("minimax", "MiniMax", server.uri(), ["X"]);
    let message = provider
        .stream(
            &text_only_anthropic_model(&server),
            &history_with_image(),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(body["messages"][0]["content"], json!(downgraded_prompt()));
    assert!(
        !body.to_string().contains(PNG_B64),
        "no image data may reach a text-only model"
    );
    assert_eq!(message.error_kind, None);
    assert_eq!(message.diagnostics.len(), 1);
    assert_eq!(message.diagnostics[0].code, DiagnosticCode::ImageDowngraded);
}

#[tokio::test]
async fn newest_user_image_still_fails_before_http_even_with_history_to_downgrade() {
    let server = MockServer::start().await;

    let context = history_with_image().with_message(image_message());
    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let message = provider
        .stream(&text_only_openai_model(&server), &context, &options())
        .finish()
        .await;

    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    let requests = server.received_requests().await.unwrap_or_default();
    assert!(
        requests.is_empty(),
        "the gate must win before any HTTP request"
    );
}

#[tokio::test]
async fn an_image_capable_model_keeps_historical_user_images() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("zai", "Z.AI", server.uri(), ["X"]);
    let message = provider
        .stream(
            &openai_image_model(&server),
            &history_with_image(),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            { "type": "text", "text": PROMPT },
            {
                "type": "image_url",
                "image_url": { "url": format!("data:image/png;base64,{PNG_B64}") },
            },
        ])
    );
    assert!(message.diagnostics.is_empty(), "nothing was downgraded");
}

#[tokio::test]
async fn consecutive_historical_images_collapse_into_one_placeholder() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let context = Context::new()
        .with_message(Message::User(UserMessage {
            content: vec![text(PROMPT), image(), image()],
            timestamp: 0,
        }))
        .user(FOLLOW_UP);
    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    provider
        .stream(&text_only_openai_model(&server), &context, &options())
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][0]["content"],
        json!(downgraded_prompt()),
        "a run of images reads as one omission, not one per image"
    );
}

/// The same collapse on the tool-result side, where the pre-normalizer code
/// emitted one placeholder line per image.
#[tokio::test]
async fn consecutive_tool_result_images_collapse_into_one_placeholder() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let context = Context::new()
        .user("screenshot the page")
        .with_message(Message::ToolResult(ToolResultMessage::content(
            "call_1",
            "screenshot",
            vec![text("before"), image(), image(), text("after")],
            false,
        )));
    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let message = provider
        .stream(&text_only_openai_model(&server), &context, &options())
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][1]["content"],
        json!(format!("before\n{TOOL_PLACEHOLDER}\nafter")),
        "two adjacent images read as one omission"
    );
    assert_eq!(message.diagnostics.len(), 1);
    assert_eq!(message.diagnostics[0].code, DiagnosticCode::ImageDowngraded);
    assert!(
        message.diagnostics[0].message.starts_with("2 tool-result"),
        "the count still reports both images: {}",
        message.diagnostics[0].message
    );
}

#[tokio::test]
async fn openai_leaves_the_callers_context_untouched() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let context = history_with_image().with_message(Message::ToolResult(
        ToolResultMessage::content("call_1", "screenshot", vec![text("shot"), image()], false),
    ));
    let before = context.clone();

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    provider
        .stream(&text_only_openai_model(&server), &context, &options())
        .finish()
        .await;

    assert_eq!(context, before, "normalization must not mutate the caller");
}

#[tokio::test]
async fn anthropic_leaves_the_callers_context_untouched() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let context = history_with_image().with_message(Message::ToolResult(
        ToolResultMessage::content("call_1", "screenshot", vec![text("shot"), image()], false),
    ));
    let before = context.clone();

    let provider = Provider::anthropic_compatible("minimax", "MiniMax", server.uri(), ["X"]);
    provider
        .stream(&text_only_anthropic_model(&server), &context, &options())
        .finish()
        .await;

    assert_eq!(context, before, "normalization must not mutate the caller");
}

/// The same `Context` value serves an image model and a text-only model in
/// turn, each getting its own normalized copy.
#[tokio::test]
async fn one_context_serves_two_models_in_a_row() {
    let image_server = MockServer::start().await;
    mount_openai_sse(&image_server).await;
    let text_server = MockServer::start().await;
    mount_openai_sse(&text_server).await;

    let context = history_with_image();

    Provider::openai_compatible("zai", "Z.AI", image_server.uri(), ["X"])
        .stream(&openai_image_model(&image_server), &context, &options())
        .finish()
        .await;
    Provider::openai_compatible("deepseek", "DeepSeek", text_server.uri(), ["X"])
        .stream(&text_only_openai_model(&text_server), &context, &options())
        .finish()
        .await;

    assert!(
        request_body(&image_server)
            .await
            .to_string()
            .contains(PNG_B64),
        "the image model still sees the image"
    );
    assert_eq!(
        request_body(&text_server).await["messages"][0]["content"],
        json!(downgraded_prompt()),
        "the text-only model sees the placeholder, not a leftover from the first run"
    );
}

const FAILED_TURN_TEXT: &str = "partial answer from a failed turn";
const ABORTED_TURN_TEXT: &str = "partial answer from an aborted turn";

fn tool_call(id: &str, name: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: json!({ "city": "Paris" }),
        raw_arguments: None,
    })
}

fn assistant_turn(content: Vec<AssistantContent>, stop_reason: StopReason) -> Message {
    let mut message = AssistantMessage::from_content(content);
    message.stop_reason = stop_reason;
    Message::Assistant(Box::new(message))
}

fn assistant_text(text: &str) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: text.into(),
        signature: None,
    })
}

/// Incomplete history (issue #41): `call_2` never got a result, and the two
/// trailing incomplete turns (Error / Aborted) must not be replayed at all.
fn incomplete_tool_history() -> Context {
    Context::new()
        .user("weather in Paris?")
        .with_message(assistant_turn(
            vec![
                tool_call("call_1", "get_weather"),
                tool_call("call_2", "get_time"),
            ],
            StopReason::ToolUse,
        ))
        .with_message(Message::ToolResult(ToolResultMessage::text(
            "call_1",
            "get_weather",
            "72F",
        )))
        .with_message(assistant_turn(
            vec![assistant_text(FAILED_TURN_TEXT)],
            StopReason::Error,
        ))
        .with_message(assistant_turn(
            vec![assistant_text(ABORTED_TURN_TEXT)],
            StopReason::Aborted,
        ))
        .user("and tomorrow?")
}

#[tokio::test]
async fn openai_repairs_orphaned_tool_calls_and_skips_incomplete_turns() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    provider
        .stream(
            &text_only_openai_model(&server),
            &incomplete_tool_history(),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(
        messages.len(),
        5,
        "the incomplete turns are dropped, the orphan repaired: {messages:?}"
    );
    assert_eq!(
        messages[1]["tool_calls"]
            .as_array()
            .expect("the assistant turn keeps both tool calls")
            .len(),
        2
    );
    assert_eq!(
        messages[2],
        json!({ "role": "tool", "tool_call_id": "call_2", "content": "No result provided" }),
        "the orphaned call gets exactly one synthetic error result"
    );
    assert_eq!(
        messages[3],
        json!({ "role": "tool", "tool_call_id": "call_1", "content": "72F" }),
        "the existing result is preserved, not duplicated"
    );
    assert_eq!(
        messages[4],
        json!({ "role": "user", "content": "and tomorrow?" })
    );
    let payload = body.to_string();
    assert!(
        !payload.contains(FAILED_TURN_TEXT) && !payload.contains(ABORTED_TURN_TEXT),
        "an Error/Aborted turn never reaches the wire"
    );
}

#[tokio::test]
async fn anthropic_repairs_orphaned_tool_calls_and_skips_incomplete_turns() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let provider = Provider::anthropic_compatible("minimax", "MiniMax", server.uri(), ["X"]);
    provider
        .stream(
            &text_only_anthropic_model(&server),
            &incomplete_tool_history(),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(
        messages.len(),
        5,
        "the incomplete turns are dropped, the orphan repaired: {messages:?}"
    );
    assert_eq!(
        messages[1]["content"],
        json!([
            { "type": "tool_use", "id": "call_1", "name": "get_weather", "input": { "city": "Paris" } },
            { "type": "tool_use", "id": "call_2", "name": "get_time", "input": { "city": "Paris" } },
        ])
    );
    assert_eq!(
        messages[2],
        json!({ "role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": "call_2",
            "content": "No result provided",
            "is_error": true,
        }] }),
        "the orphaned call gets exactly one synthetic error result"
    );
    assert_eq!(
        messages[3],
        json!({ "role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "72F",
            "is_error": false,
        }] }),
        "the existing result is preserved, not duplicated"
    );
    let payload = body.to_string();
    assert!(
        !payload.contains(FAILED_TURN_TEXT) && !payload.contains(ABORTED_TURN_TEXT),
        "an Error/Aborted turn never reaches the wire"
    );
}
