//! User images end-to-end on both protocols, with modality gating.
//!
//! `UserContent::Image` reaches the wire: multi-part `image_url` blocks for
//! OpenAI chat completions, `image` blocks with a base64 `source` for
//! Anthropic messages. A newest-user-message image on a model that does not
//! declare `Modality::Image` terminates in-band with
//! `ErrorKind::InvalidRequest` before any HTTP request is issued.
//!
//! One request fixture per image-capable protocol family: `glm-4.5v`
//! (openai-completions, zai catalog) and `k2p5` (anthropic-messages, kimi
//! catalog) are catalog-declared image models.

use banshu_ai::{
    Context, ErrorKind, ImageContent, Message, Modality, Model, Provider, StreamOptions,
    TextContent, UserContent, UserMessage,
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

const PROMPT: &str = "what is in this picture?";

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

fn image_message() -> Message {
    Message::User(UserMessage {
        content: vec![
            UserContent::Text(TextContent {
                text: PROMPT.into(),
                signature: None,
            }),
            UserContent::Image(ImageContent {
                data: PNG_B64.into(),
                mime_type: "image/png".into(),
            }),
        ],
        timestamp: 0,
    })
}

/// The zai catalog declares glm-4.5v image-capable; the fixture exercises the
/// declared capability, not a hand-set one (§4.3). Re-pointed at the mock.
fn openai_image_model(server: &MockServer) -> Model {
    let model = Provider::zai()
        .models()
        .iter()
        .find(|m| m.id == "glm-4.5v")
        .expect("glm-4.5v should be in the zai catalog")
        .clone();
    assert!(
        model.input.contains(&Modality::Image),
        "zai catalog should declare glm-4.5v image-capable"
    );
    model.with_base_url(server.uri())
}

/// The kimi catalog declares k2p5 image-capable (§4.3). Re-pointed at the mock.
fn anthropic_image_model(server: &MockServer) -> Model {
    let model = Provider::kimi()
        .models()
        .iter()
        .find(|m| m.id == "k2p5")
        .expect("k2p5 should be in the kimi catalog")
        .clone();
    assert!(
        model.input.contains(&Modality::Image),
        "kimi catalog should declare k2p5 image-capable"
    );
    model.with_base_url(server.uri())
}

async fn request_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("request journal");
    serde_json::from_slice(&requests[0].body).expect("JSON request")
}

#[tokio::test]
async fn openai_user_image_becomes_multipart_content() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("zai", "Z.AI", server.uri(), ["X"]);
    provider
        .stream(
            &openai_image_model(&server),
            &Context::new().with_message(image_message()),
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
}

#[tokio::test]
async fn openai_text_only_user_message_keeps_string_content() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("zai", "Z.AI", server.uri(), ["X"]);
    provider
        .stream(
            &openai_image_model(&server),
            &Context::new().user("hi"),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(body["messages"][0]["content"], json!("hi"));
}

#[tokio::test]
async fn anthropic_user_image_becomes_image_block() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    // Disabled retention keeps cache breakpoints out of the asserted shape.
    let options = StreamOptions {
        cache_retention: Some(banshu_ai::CacheRetention::Disabled),
        ..options()
    };
    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"]);
    provider
        .stream(
            &anthropic_image_model(&server),
            &Context::new().with_message(image_message()),
            &options,
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            { "type": "text", "text": PROMPT },
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
async fn anthropic_text_only_user_message_keeps_string_content() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"]);
    provider
        .stream(
            &anthropic_image_model(&server),
            &Context::new().user("hi"),
            &options(),
        )
        .finish()
        .await;

    let body = request_body(&server).await;
    // Current wire shape: the default cache breakpoint turns the last user
    // message into a single text block with `cache_control`.
    assert_eq!(
        body["messages"][0]["content"],
        json!([{ "type": "text", "text": "hi", "cache_control": { "type": "ephemeral" } }])
    );
}

#[tokio::test]
async fn openai_newest_user_image_on_text_only_model_fails_in_band_without_http() {
    let server = MockServer::start().await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let message = provider
        .stream(
            &Model::openai_completions("test-model").with_base_url(server.uri()),
            &Context::new().with_message(image_message()),
            &options(),
        )
        .finish()
        .await;

    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    let requests = server.received_requests().await.unwrap_or_default();
    assert!(requests.is_empty(), "no HTTP request may be issued");
}

#[tokio::test]
async fn anthropic_newest_user_image_on_text_only_model_fails_in_band_without_http() {
    let server = MockServer::start().await;

    let provider = Provider::anthropic_compatible("minimax", "MiniMax", server.uri(), ["X"]);
    let message = provider
        .stream(
            &Model::anthropic_messages("test-model").with_base_url(server.uri()),
            &Context::new().with_message(image_message()),
            &options(),
        )
        .finish()
        .await;

    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    let requests = server.received_requests().await.unwrap_or_default();
    assert!(requests.is_empty(), "no HTTP request may be issued");
}

#[tokio::test]
async fn historical_image_does_not_trip_the_gate() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let context = Context::new()
        .with_message(image_message())
        .user("describe it in one word");
    let message = provider
        .stream(
            &Model::openai_completions("test-model").with_base_url(server.uri()),
            &context,
            &options(),
        )
        .finish()
        .await;

    assert_eq!(message.error_kind, None);
    let requests = server.received_requests().await.unwrap_or_default();
    assert_eq!(
        requests.len(),
        1,
        "only the newest user message gates the request"
    );
}
