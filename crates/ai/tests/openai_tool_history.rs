//! Tool-history wire policies for OpenAI-compatible providers (issue #91).
//!
//! An OpenAI-compatible provider declares the tool-history message shapes its
//! chat template requires: whether each `tool` message carries the tool's
//! `name` (`OpenAiCompat::tool_result_names`) and whether an empty assistant
//! message separates a run of tool results from a following user message
//! (`OpenAiCompat::empty_assistant_separator`). The undeclared default stays
//! byte-compatible with the request bodies bundled providers have always sent:
//! no names, no separator.

use std::sync::{Arc, Mutex};

use banshu_ai::{
    AssistantContent, AssistantMessage, BeforeSendObservation, Context, ImageContent, Message,
    Model, OpenAiCompat, Provider, RequestObserver, ResponseObservation, StreamOptions, ToolCall,
    ToolResultMessage, UserContent,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
    "data: [DONE]\n\n",
);

/// A 1x1 transparent PNG.
const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// A multi-turn tool history: the assistant issued two tool calls, both
/// results came back, and the user followed up.
fn history() -> Context {
    Context::new()
        .user("weather and time in Paris?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![
                AssistantContent::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "city": "Paris" }),
                    raw_arguments: None,
                }),
                AssistantContent::ToolCall(ToolCall {
                    id: "call_2".into(),
                    name: "get_time".into(),
                    arguments: serde_json::json!({ "city": "Paris" }),
                    raw_arguments: None,
                }),
            ]),
        )))
        .with_message(Message::ToolResult(ToolResultMessage::text(
            "call_1",
            "get_weather",
            "72F and sunny",
        )))
        .with_message(Message::ToolResult(ToolResultMessage::text(
            "call_2",
            "get_time",
            "14:00 CET",
        )))
        .user("and in Tokyo?")
}

/// The shared head of every expected message list: the user question and the
/// assistant turn carrying both tool calls.
fn expected_head() -> Vec<serde_json::Value> {
    serde_json::json!([
        { "role": "user", "content": "weather and time in Paris?" },
        { "role": "assistant", "content": null, "tool_calls": [
            { "id": "call_1", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } },
            { "id": "call_2", "type": "function", "function": { "name": "get_time", "arguments": "{\"city\":\"Paris\"}" } },
        ] },
    ])
    .as_array()
    .expect("array")
    .clone()
}

/// Append the messages of `tail` (a JSON array) to `expected`.
fn extend_expected(expected: &mut Vec<serde_json::Value>, tail: serde_json::Value) {
    expected.extend(tail.as_array().expect("array").clone());
}

async fn sse_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .mount(&server)
        .await;
    server
}

fn provider(server: &MockServer, compat: OpenAiCompat) -> Provider {
    Provider::openai_compatible("test", "Test", server.uri(), ["TEST_API_KEY"])
        .with_openai_compat(compat)
}

fn model(server: &MockServer) -> Model {
    let mut model = Model::openai_completions("test-model").with_base_url(server.uri());
    model.provider = "test".into();
    model
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

/// Stream one request over `context` and return the exact `messages` array
/// the server recorded.
async fn sent_messages(
    server: &MockServer,
    provider: &Provider,
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Vec<serde_json::Value> {
    let message = provider.stream(model, context, options).finish().await;
    assert_eq!(message.error_kind, None, "{message:?}");
    let requests = server.received_requests().await.expect("request journal");
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("JSON request");
    body["messages"].as_array().expect("messages array").clone()
}

/// Stream one request over `history()` and return the recorded `messages`.
async fn sent_history(
    server: &MockServer,
    provider: &Provider,
    options: &StreamOptions,
) -> Vec<serde_json::Value> {
    sent_messages(server, provider, &model(server), &history(), options).await
}

#[tokio::test]
async fn the_default_policy_keeps_todays_tool_history_shape() {
    let server = sse_server().await;
    let messages = sent_history(
        &server,
        &provider(&server, OpenAiCompat::default()),
        &options(),
    )
    .await;

    let mut expected = expected_head();
    extend_expected(
        &mut expected,
        serde_json::json!([
            { "role": "tool", "tool_call_id": "call_1", "content": "72F and sunny" },
            { "role": "tool", "tool_call_id": "call_2", "content": "14:00 CET" },
            { "role": "user", "content": "and in Tokyo?" },
        ]),
    );
    assert_eq!(messages, expected);
}

#[tokio::test]
async fn declared_tool_result_names_ride_each_tool_message() {
    let server = sse_server().await;
    let compat = OpenAiCompat {
        tool_result_names: true,
        ..OpenAiCompat::default()
    };
    let messages = sent_history(&server, &provider(&server, compat), &options()).await;

    let mut expected = expected_head();
    extend_expected(
        &mut expected,
        serde_json::json!([
            { "role": "tool", "tool_call_id": "call_1", "content": "72F and sunny", "name": "get_weather" },
            { "role": "tool", "tool_call_id": "call_2", "content": "14:00 CET", "name": "get_time" },
            { "role": "user", "content": "and in Tokyo?" },
        ]),
    );
    assert_eq!(messages, expected);
}

#[tokio::test]
async fn a_declared_separator_closes_the_tool_run_before_the_user_turn() {
    let server = sse_server().await;
    let compat = OpenAiCompat {
        empty_assistant_separator: true,
        ..OpenAiCompat::default()
    };
    let messages = sent_history(&server, &provider(&server, compat), &options()).await;

    let mut expected = expected_head();
    extend_expected(
        &mut expected,
        serde_json::json!([
            { "role": "tool", "tool_call_id": "call_1", "content": "72F and sunny" },
            { "role": "tool", "tool_call_id": "call_2", "content": "14:00 CET" },
            { "role": "assistant", "content": "" },
            { "role": "user", "content": "and in Tokyo?" },
        ]),
    );
    assert_eq!(messages, expected);

    // Exactly one separator, and only at the tool-run → user boundary: the
    // consecutive results stay adjacent and ordered.
    let separators = messages
        .iter()
        .filter(|message| message["role"] == "assistant" && message.get("tool_calls").is_none())
        .count();
    assert_eq!(separators, 1, "{messages:?}");
}

#[tokio::test]
async fn the_separator_precedes_the_image_carrier_and_is_never_duplicated() {
    // A tool result holding an image trails the run with a user message
    // carrying it; that message is a tool-run → user boundary too, so the
    // declared separator closes the run ahead of it — and the real user turn
    // that follows gets no second one.
    let context = Context::new()
        .user("weather in Paris?")
        .with_message(Message::Assistant(Box::new(
            AssistantMessage::from_content(vec![AssistantContent::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({ "city": "Paris" }),
                raw_arguments: None,
            })]),
        )))
        .with_message(Message::ToolResult(ToolResultMessage::content(
            "call_1",
            "get_weather",
            vec![
                UserContent::Text(banshu_ai::TextContent {
                    text: "72F".into(),
                    signature: None,
                }),
                UserContent::Image(ImageContent {
                    data: PNG_B64.into(),
                    mime_type: "image/png".into(),
                }),
            ],
            false,
        )))
        .user("and in Tokyo?");

    let server = sse_server().await;
    let compat = OpenAiCompat {
        empty_assistant_separator: true,
        ..OpenAiCompat::default()
    };
    let mut image_model = model(&server);
    image_model.input.push(banshu_ai::Modality::Image);
    let messages = sent_messages(
        &server,
        &provider(&server, compat),
        &image_model,
        &context,
        &options(),
    )
    .await;

    let mut expected = Vec::new();
    extend_expected(
        &mut expected,
        serde_json::json!([
            { "role": "user", "content": "weather in Paris?" },
            { "role": "assistant", "content": null, "tool_calls": [
                { "id": "call_1", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } },
            ] },
            { "role": "tool", "tool_call_id": "call_1", "content": "72F" },
            { "role": "assistant", "content": "" },
            { "role": "user", "content": [
                { "type": "text", "text": "Attached image(s) from tool result:" },
                { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{PNG_B64}") } },
            ] },
            { "role": "user", "content": "and in Tokyo?" },
        ]),
    );
    assert_eq!(messages, expected);
}

#[tokio::test]
async fn the_two_policies_declare_independently_of_each_other() {
    let server = sse_server().await;
    let compat = OpenAiCompat {
        tool_result_names: true,
        empty_assistant_separator: true,
        ..OpenAiCompat::default()
    };
    let messages = sent_history(&server, &provider(&server, compat), &options()).await;

    let mut expected = expected_head();
    extend_expected(
        &mut expected,
        serde_json::json!([
            { "role": "tool", "tool_call_id": "call_1", "content": "72F and sunny", "name": "get_weather" },
            { "role": "tool", "tool_call_id": "call_2", "content": "14:00 CET", "name": "get_time" },
            { "role": "assistant", "content": "" },
            { "role": "user", "content": "and in Tokyo?" },
        ]),
    );
    assert_eq!(messages, expected);
}

/// Records the payload of every observed attempt.
#[derive(Default)]
struct PayloadObserver {
    payloads: Mutex<Vec<serde_json::Value>>,
}

impl RequestObserver for PayloadObserver {
    fn before_send(&self, observation: &BeforeSendObservation) {
        self.payloads
            .lock()
            .unwrap()
            .push(observation.payload.clone());
    }

    fn on_response(&self, _observation: &ResponseObservation) {}
}

#[tokio::test]
async fn the_observed_payload_is_the_exact_body_the_server_recorded() {
    let server = sse_server().await;
    let compat = OpenAiCompat {
        tool_result_names: true,
        empty_assistant_separator: true,
        ..OpenAiCompat::default()
    };
    let observer = Arc::new(PayloadObserver::default());
    let options = StreamOptions {
        observer: Some(observer.clone()),
        ..options()
    };
    let message = provider(&server, compat)
        .stream(&model(&server), &history(), &options)
        .finish()
        .await;
    assert_eq!(message.error_kind, None, "{message:?}");
    let requests = server.received_requests().await.expect("request journal");
    assert_eq!(requests.len(), 1);
    let recorded: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("JSON request");

    let payloads = observer.payloads.lock().unwrap();
    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0], recorded);
    // Both policies are visible in the observation itself.
    let messages = payloads[0]["messages"].as_array().expect("messages array");
    assert_eq!(messages[2]["name"], "get_weather", "{recorded}");
    assert_eq!(
        messages[4],
        serde_json::json!({ "role": "assistant", "content": "" }),
        "{recorded}"
    );
}
