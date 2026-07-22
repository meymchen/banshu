//! Prompt caching for Anthropic-compatible providers.
//!
//! Request side: `cache_control` breakpoints on the system prompt, the last
//! tool definition, and the last user message (mirroring pi's placement).
//! Response side: cache read/write token usage, including 1h cache writes
//! billed at twice the input rate.

use banshu_ai::{
    AnthropicCompat, CacheRetention, Context, Model, ModelCost, Provider, StreamOptions, Tool,
};
use serde_json::{Value, json};
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
    Model::anthropic_messages("k2p5").with_base_url(server.uri())
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn tool(name: &str) -> Tool {
    Tool {
        name: name.into(),
        description: format!("The {name} tool"),
        parameters: json!({ "type": "object", "properties": {} }),
    }
}

#[tokio::test]
async fn caches_system_prompt_tools_and_last_user_message_by_default() {
    let server = MockServer::start().await;
    mount_sse(&server, STOP_BODY).await;

    let context = Context::new()
        .with_system("Be terse.")
        .with_tool(tool("first"))
        .with_tool(tool("second"))
        .user("hi");
    provider(&server)
        .stream(&model(&server), &context, &options())
        .finish()
        .await;

    let body = request_body(&server).await;
    let ephemeral = json!({ "type": "ephemeral" });
    assert_eq!(
        body["system"],
        json!([{ "type": "text", "text": "Be terse.", "cache_control": ephemeral }]),
    );
    // Only the last tool carries the breakpoint.
    assert!(body["tools"][0].get("cache_control").is_none());
    assert_eq!(body["tools"][1]["cache_control"], ephemeral);
    // The last user message is converted to blocks so the breakpoint can attach.
    assert_eq!(
        body["messages"][0]["content"],
        json!([{ "type": "text", "text": "hi", "cache_control": ephemeral }]),
    );
}

#[tokio::test]
async fn long_retention_requests_the_one_hour_ttl() {
    let server = MockServer::start().await;
    mount_sse(&server, STOP_BODY).await;

    let context = Context::new().with_system("Be terse.").user("hi");
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Long),
        ..options()
    };
    provider(&server)
        .stream(&model(&server), &context, &options)
        .finish()
        .await;

    let body = request_body(&server).await;
    let ephemeral_1h = json!({ "type": "ephemeral", "ttl": "1h" });
    assert_eq!(body["system"][0]["cache_control"], ephemeral_1h);
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        ephemeral_1h
    );
}

#[tokio::test]
async fn disabled_retention_sends_no_cache_control() {
    let server = MockServer::start().await;
    mount_sse(&server, STOP_BODY).await;

    let context = Context::new()
        .with_system("Be terse.")
        .with_tool(tool("first"))
        .user("hi");
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Disabled),
        ..options()
    };
    provider(&server)
        .stream(&model(&server), &context, &options)
        .finish()
        .await;

    let body = request_body(&server).await;
    assert_eq!(
        body["system"],
        json!([{ "type": "text", "text": "Be terse." }])
    );
    assert_eq!(
        body["messages"][0],
        json!({ "role": "user", "content": "hi" })
    );
    assert!(
        !serde_json::to_string(&body)
            .expect("body")
            .contains("cache_control")
    );
}

#[tokio::test]
async fn attaches_the_breakpoint_to_a_trailing_tool_result() {
    let server = MockServer::start().await;
    mount_sse(&server, STOP_BODY).await;

    let context =
        Context::new()
            .user("weather?")
            .tool_result("call_1", "get_weather", "72F and sunny");
    provider(&server)
        .stream(&model(&server), &context, &options())
        .finish()
        .await;

    let body = request_body(&server).await;
    let messages = body["messages"].as_array().expect("messages");
    let last_block = &messages.last().expect("last message")["content"][0];
    assert_eq!(last_block["type"], "tool_result");
    assert_eq!(last_block["cache_control"], json!({ "type": "ephemeral" }));
    // Earlier messages are untouched.
    assert_eq!(
        messages[0],
        json!({ "role": "user", "content": "weather?" })
    );
}

#[tokio::test]
async fn reads_cache_usage_and_bills_one_hour_writes_at_twice_the_input_rate() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{",
            "\"input_tokens\":100,\"output_tokens\":1,",
            "\"cache_read_input_tokens\":1000000,",
            "\"cache_creation_input_tokens\":500000,",
            "\"cache_creation\":{\"ephemeral_1h_input_tokens\":200000}}}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},",
            "\"usage\":{\"output_tokens\":50}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ),
    )
    .await;

    let mut model = model(&server);
    model.cost = ModelCost {
        input: 1.0,
        output: 2.0,
        cache_read: 0.1,
        cache_write: 1.25,
    };

    let message = provider(&server)
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await;

    assert_eq!(message.usage.input, 100);
    assert_eq!(message.usage.output, 50);
    assert_eq!(message.usage.cache_read, 1_000_000);
    assert_eq!(message.usage.cache_write, 500_000);
    assert_eq!(message.usage.cache_write_1h, Some(200_000));
    assert_eq!(message.usage.total_tokens, 1_500_150);
    assert!((message.usage.cost.cache_read - 0.1).abs() < 1e-12);
    // 300k short writes at the cache-write rate + 200k 1h writes at 2x input.
    assert!((message.usage.cost.cache_write - 0.775).abs() < 1e-12);
}

#[tokio::test]
async fn sends_session_affinity_header_only_when_declared_and_caching() {
    for (declared, retention, expected) in [
        (true, CacheRetention::Short, Some("conversation-42")),
        (true, CacheRetention::Disabled, None),
        (false, CacheRetention::Short, None),
    ] {
        let server = MockServer::start().await;
        mount_sse(&server, STOP_BODY).await;

        let provider = provider(&server).with_anthropic_compat(AnthropicCompat {
            send_session_affinity_headers: declared,
            ..AnthropicCompat::default()
        });
        let options = StreamOptions {
            cache_retention: Some(retention),
            session_id: Some("conversation-42".into()),
            ..options()
        };
        provider
            .stream(&model(&server), &Context::new().user("hi"), &options)
            .finish()
            .await;

        let requests = server.received_requests().await.expect("request journal");
        let header = requests[0]
            .headers
            .get("x-session-affinity")
            .map(|v| v.to_str().unwrap().to_owned());
        assert_eq!(header.as_deref(), expected);
    }
}
