//! Tool-result images on both protocols, with a loud downgrade on models
//! that cannot see them (issue #22).
//!
//! A text+image tool result reaches the wire in the protocol-compatible
//! shape: OpenAI keeps `tool` messages text-only and trails a run of
//! consecutive tool results with one `user` message carrying every image
//! (tool messages must immediately follow the assistant turn); Anthropic puts
//! `image` blocks inside the `tool_result` content. On a model without
//! `Modality::Image` each image block is replaced with the fixed placeholder
//! text `(tool image omitted: model does not support images)`, the resulting
//! message carries an `ImageDowngraded` diagnostic, and the tool result as a
//! whole is never dropped.

use banshu_ai::{
    AssistantContent, AssistantMessage, Context, DiagnosticCode, ImageContent, Message, Model,
    Provider, StreamOptions, TextContent, ToolCall, ToolResultMessage, UserContent,
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

/// The fixed text replacing an image the model cannot see (issue #22).
const PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

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

fn tool_call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({ "city": "Paris" }),
        raw_arguments: None,
    })
}

/// user → assistant(tool call) → tool result with the given content blocks.
fn history(result_content: Vec<UserContent>) -> Context {
    Context::new()
        .user("weather in Paris?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![tool_call("call_1")]),
        )))
        .with_message(Message::ToolResult(ToolResultMessage::content(
            "call_1",
            "get_weather",
            result_content,
            false,
        )))
}

/// The zai catalog declares glm-4.5v image-capable (issue #21). Re-pointed at the
/// mock, as in `user_images.rs`.
fn openai_image_model(server: &MockServer) -> Model {
    let model = Provider::zai()
        .models()
        .iter()
        .find(|m| m.id == "glm-4.5v")
        .expect("glm-4.5v should be in the zai catalog")
        .clone();
    assert!(
        model.input.contains(&banshu_ai::Modality::Image),
        "zai catalog should declare glm-4.5v image-capable"
    );
    model.with_base_url(server.uri())
}

/// The kimi catalog declares k3 image-capable (issue #21). Re-pointed at the mock.
fn anthropic_image_model(server: &MockServer) -> Model {
    let model = Provider::kimi(std::sync::Arc::new(
        banshu_ai::InMemoryCredentialStore::new(),
    ))
    .models()
    .iter()
    .find(|m| m.id == "k3")
    .expect("k3 should be in the kimi catalog")
    .clone();
    assert!(
        model.input.contains(&banshu_ai::Modality::Image),
        "kimi catalog should declare k3 image-capable"
    );
    model.with_base_url(server.uri())
}

async fn request_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("request journal");
    serde_json::from_slice(&requests[0].body).expect("JSON request")
}

#[tokio::test]
async fn openai_tool_result_image_trails_the_tool_message_in_a_user_message() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("zai", "Z.AI", server.uri(), ["X"]);
    let message = provider
        .stream(
            &openai_image_model(&server),
            &history(vec![text("72F"), image()]),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2],
        json!({ "role": "tool", "tool_call_id": "call_1", "content": "72F" })
    );
    assert_eq!(
        body["messages"][3],
        json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "Attached image(s) from tool result:" },
                { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{PNG_B64}") } },
            ],
        })
    );
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(4));
    assert!(message.diagnostics.is_empty(), "image model: no downgrade");
}

#[tokio::test]
async fn openai_tool_result_image_on_text_only_model_is_replaced_with_placeholder() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let model = Model::openai_completions("test-model").with_base_url(server.uri());
    let message = provider
        .stream(
            &model,
            &history(vec![text("72F"), text("and sunny"), image()]),
            &options(),
        )
        .finish()
        .await;

    // The wire keeps the text blocks and replaces the image with the fixed
    // placeholder; no trailing user message carries the image.
    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2],
        json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": format!("72F\nand sunny\n{PLACEHOLDER}"),
        })
    );
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(3));
    assert!(
        !body.to_string().contains("image_url"),
        "no image data reaches the wire"
    );

    // The downgrade is loud on the resulting message; the stream itself is fine.
    assert_eq!(message.error_kind, None);
    assert_eq!(message.diagnostics.len(), 1);
    assert_eq!(message.diagnostics[0].code, DiagnosticCode::ImageDowngraded);
}

#[tokio::test]
async fn openai_image_only_tool_result_on_text_only_model_is_never_dropped() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let model = Model::openai_completions("test-model").with_base_url(server.uri());
    let message = provider
        .stream(&model, &history(vec![image()]), &options())
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2],
        json!({ "role": "tool", "tool_call_id": "call_1", "content": PLACEHOLDER })
    );
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(3));
    assert_eq!(message.diagnostics.len(), 1);
    assert_eq!(message.diagnostics[0].code, DiagnosticCode::ImageDowngraded);
}

#[tokio::test]
async fn anthropic_tool_result_image_becomes_tool_result_content_blocks() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    // Disabled retention keeps cache breakpoints out of the asserted shape.
    let options = StreamOptions {
        cache_retention: Some(banshu_ai::CacheRetention::Disabled),
        ..options()
    };
    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"]);
    let message = provider
        .stream(
            &anthropic_image_model(&server),
            &history(vec![text("72F"), image()]),
            &options,
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2],
        json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [
                    { "type": "text", "text": "72F" },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": PNG_B64,
                        },
                    },
                ],
                "is_error": false,
            }],
        })
    );
    assert!(message.diagnostics.is_empty(), "image model: no downgrade");
}

#[tokio::test]
async fn anthropic_tool_result_image_on_text_only_model_is_replaced_with_placeholder() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    // Disabled retention keeps cache breakpoints out of the asserted shape.
    let options = StreamOptions {
        cache_retention: Some(banshu_ai::CacheRetention::Disabled),
        ..options()
    };
    let provider = Provider::anthropic_compatible("minimax", "MiniMax", server.uri(), ["X"]);
    let model = Model::anthropic_messages("test-model").with_base_url(server.uri());
    let message = provider
        .stream(
            &model,
            &history(vec![text("72F"), text("and sunny"), image()]),
            &options,
        )
        .finish()
        .await;

    // The downgraded tool result is all text, so it keeps the plain-string
    // shape: text blocks intact, the image replaced with the placeholder.
    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2],
        json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": format!("72F\nand sunny\n{PLACEHOLDER}"),
                "is_error": false,
            }],
        })
    );
    assert!(
        !body.to_string().contains("base64"),
        "no image data reaches the wire"
    );

    assert_eq!(message.error_kind, None);
    assert_eq!(message.diagnostics.len(), 1);
    assert_eq!(message.diagnostics[0].code, DiagnosticCode::ImageDowngraded);
}

#[tokio::test]
async fn anthropic_image_only_tool_result_on_text_only_model_is_never_dropped() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let options = StreamOptions {
        cache_retention: Some(banshu_ai::CacheRetention::Disabled),
        ..options()
    };
    let provider = Provider::anthropic_compatible("minimax", "MiniMax", server.uri(), ["X"]);
    let model = Model::anthropic_messages("test-model").with_base_url(server.uri());
    let message = provider
        .stream(&model, &history(vec![image()]), &options)
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2]["content"][0]["content"],
        json!(PLACEHOLDER)
    );
    assert_eq!(message.diagnostics.len(), 1);
    assert_eq!(message.diagnostics[0].code, DiagnosticCode::ImageDowngraded);
}

#[tokio::test]
async fn openai_consecutive_tool_results_share_one_trailing_user_message() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    // Parallel tool calls: two results in a row, each carrying an image.
    // `tool` messages must immediately follow the assistant turn, so the
    // images of the whole run trail it in a single user message.
    let context = Context::new()
        .user("weather in Paris and Berlin?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![tool_call("call_1"), tool_call("call_2")]),
        )))
        .with_message(Message::ToolResult(ToolResultMessage::content(
            "call_1",
            "get_weather",
            vec![text("72F"), image()],
            false,
        )))
        .with_message(Message::ToolResult(ToolResultMessage::content(
            "call_2",
            "get_weather",
            vec![text("61F"), image()],
            false,
        )));

    let provider = Provider::openai_compatible("zai", "Z.AI", server.uri(), ["X"]);
    provider
        .stream(&openai_image_model(&server), &context, &options())
        .finish()
        .await;

    let body = request_body(&server).await;
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 5);
    assert_eq!(
        messages[2],
        json!({ "role": "tool", "tool_call_id": "call_1", "content": "72F" })
    );
    assert_eq!(
        messages[3],
        json!({ "role": "tool", "tool_call_id": "call_2", "content": "61F" })
    );
    assert_eq!(
        messages[4],
        json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "Attached image(s) from tool result:" },
                { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{PNG_B64}") } },
                { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{PNG_B64}") } },
            ],
        })
    );
}

#[tokio::test]
async fn openai_image_only_tool_result_reads_see_attached_image() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("zai", "Z.AI", server.uri(), ["X"]);
    provider
        .stream(
            &openai_image_model(&server),
            &history(vec![image()]),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2],
        json!({ "role": "tool", "tool_call_id": "call_1", "content": "(see attached image)" })
    );
    assert_eq!(
        body["messages"][3]["content"],
        json!([
            { "type": "text", "text": "Attached image(s) from tool result:" },
            { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{PNG_B64}") } },
        ])
    );
}

#[tokio::test]
async fn anthropic_image_only_tool_result_prepends_placeholder_text() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let options = StreamOptions {
        cache_retention: Some(banshu_ai::CacheRetention::Disabled),
        ..options()
    };
    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"]);
    provider
        .stream(
            &anthropic_image_model(&server),
            &history(vec![image()]),
            &options,
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2]["content"][0]["content"],
        json!([
            { "type": "text", "text": "(see attached image)" },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": PNG_B64,
                },
            },
        ])
    );
}

#[tokio::test]
async fn openai_text_only_tool_result_keeps_string_content_on_image_model() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("zai", "Z.AI", server.uri(), ["X"]);
    provider
        .stream(
            &openai_image_model(&server),
            &history(vec![text("72F"), text("and sunny")]),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2],
        json!({ "role": "tool", "tool_call_id": "call_1", "content": "72F\nand sunny" })
    );
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(3));
}

#[tokio::test]
async fn anthropic_text_only_tool_result_keeps_string_content_on_image_model() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let options = StreamOptions {
        cache_retention: Some(banshu_ai::CacheRetention::Disabled),
        ..options()
    };
    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"]);
    provider
        .stream(
            &anthropic_image_model(&server),
            &history(vec![text("72F"), text("and sunny")]),
            &options,
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][2]["content"][0]["content"],
        json!("72F\nand sunny")
    );
}
