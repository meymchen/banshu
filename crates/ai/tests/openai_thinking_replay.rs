//! Thinking round-trips on the openai-completions path.
//!
//! Capture records which wire field the reasoning arrived in
//! (`reasoning_content` / `reasoning` / `reasoning_text`) as the thinking
//! block's signature; replay writes the joined thinking back under that same
//! field. DeepSeek additionally requires `reasoning_content` (`""` when the
//! turn produced no thinking) on every replayed assistant message while a
//! reasoning model is active.

use banshu_ai::{
    AssistantContent, AssistantMessage, Context, Message, Model, OpenAiCompat, Provider,
    StreamOptions, TextContent, ThinkingContent,
};
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn mount_sse(server: &MockServer, body: impl Into<String>) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

fn model(server: &MockServer) -> Model {
    Model::openai_completions("test-model").with_base_url(server.uri())
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
async fn records_the_reasoning_source_field_as_the_signature() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"Let me think. \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"2 + 2 is 4.\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"4\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ),
    )
    .await;

    let provider = Provider::openai_compatible("custom", "Custom", server.uri(), ["X"]);
    let message = provider
        .stream(&model(&server), &Context::new().user("2+2?"), &options())
        .final_message()
        .await;

    let block = message
        .content
        .iter()
        .find_map(|c| match c {
            AssistantContent::Thinking(t) => Some(t),
            _ => None,
        })
        .expect("expected a thinking block");
    assert_eq!(block.thinking, "Let me think. 2 + 2 is 4.");
    assert_eq!(block.signature.as_deref(), Some("reasoning"));
}

#[tokio::test]
async fn replays_thinking_under_its_source_field() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    )
    .await;

    let context = Context::new()
        .user("2+2?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![
                thinking("Let me think.", Some("reasoning_content")),
                text("The answer is 4."),
            ]),
        )))
        .user("And 3+3?");
    let provider = Provider::openai_compatible("custom", "Custom", server.uri(), ["X"]);
    provider
        .stream(&model(&server), &context, &options())
        .final_message()
        .await;

    let assistant = &request_body(&server).await["messages"][1];
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["reasoning_content"], "Let me think.");
    assert_eq!(assistant["content"], "The answer is 4.");
}

#[tokio::test]
async fn drops_signatureless_thinking_on_replay() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    )
    .await;

    let context = Context::new()
        .user("2+2?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![
                thinking("Let me think.", None),
                text("The answer is 4."),
            ]),
        )))
        .user("And 3+3?");
    let provider = Provider::openai_compatible("custom", "Custom", server.uri(), ["X"]);
    provider
        .stream(&model(&server), &context, &options())
        .final_message()
        .await;

    let assistant = &request_body(&server).await["messages"][1];
    assert!(assistant.get("reasoning_content").is_none());
    assert_eq!(assistant["content"], "The answer is 4.");
}

#[tokio::test]
async fn backfills_empty_reasoning_content_for_reasoning_models_when_required() {
    for (model_reasoning, expected) in [(true, Some("")), (false, None)] {
        let server = MockServer::start().await;
        mount_sse(
            &server,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        )
        .await;

        let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"])
            .with_openai_compat(OpenAiCompat {
                requires_reasoning_content_on_assistant_messages: true,
                ..OpenAiCompat::default()
            });
        let mut model = model(&server);
        model.reasoning = model_reasoning;
        let context = Context::new()
            .user("2+2?")
            .with_message(Message::Assistant(Box::new(
                AssistantMessage::from_content(vec![text("The answer is 4.")]),
            )))
            .user("And 3+3?");
        provider
            .stream(&model, &context, &options())
            .final_message()
            .await;

        let assistant = &request_body(&server).await["messages"][1];
        assert_eq!(
            assistant.get("reasoning_content").and_then(Value::as_str),
            expected
        );
    }
}

#[tokio::test]
async fn deepseek_provider_requires_reasoning_content_by_default() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    )
    .await;

    let mut model = model(&server);
    model.reasoning = true;
    let context = Context::new()
        .user("2+2?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![text("The answer is 4.")]),
        )))
        .user("And 3+3?");
    Provider::deepseek()
        .stream(&model, &context, &options())
        .final_message()
        .await;

    let assistant = &request_body(&server).await["messages"][1];
    assert_eq!(assistant["reasoning_content"], "");
}
