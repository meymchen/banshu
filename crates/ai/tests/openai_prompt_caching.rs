//! Prompt caching for OpenAI-compatible providers.
//!
//! Covers request-side controls and the usage variants returned by OpenAI,
//! DeepSeek, OpenRouter-style endpoints, and Moonshot.

use banshu_ai::{
    CacheRetention, Context, Model, ModelCost, OpenAiPromptCaching, Provider, StreamOptions,
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

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn model(server: &MockServer) -> Model {
    Model::openai_completions("test-model").with_base_url(server.uri())
}

#[tokio::test]
async fn normalizes_openai_cache_read_and_write_usage_and_cost() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],",
            "\"usage\":{\"prompt_tokens\":1000000,\"completion_tokens\":100000,",
            "\"total_tokens\":1100000,\"prompt_tokens_details\":{",
            "\"cached_tokens\":600000,\"cache_write_tokens\":100000}}}\n\n",
            "data: [DONE]\n\n",
        ),
    )
    .await;

    let provider = Provider::openai_compatible("openai", "OpenAI", server.uri(), ["X"]);
    let mut model = model(&server);
    model.cost = ModelCost {
        input: 1.0,
        output: 2.0,
        cache_read: 0.1,
        cache_write: 1.25,
    };

    let message = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .final_message()
        .await;

    assert_eq!(message.usage.input, 400_000);
    assert_eq!(message.usage.cache_read, 500_000);
    assert_eq!(message.usage.cache_write, 100_000);
    assert_eq!(message.usage.output, 100_000);
    assert_eq!(message.usage.total_tokens, 1_100_000);
    assert_eq!(message.usage.cost.input, 0.4);
    assert_eq!(message.usage.cost.cache_read, 0.05);
    assert_eq!(message.usage.cost.cache_write, 0.125);
    assert_eq!(message.usage.cost.output, 0.2);
    assert!((message.usage.cost.total - 0.775).abs() < 1e-12);
}

#[tokio::test]
async fn normalizes_deepseek_hit_and_miss_usage_without_double_counting() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],",
            "\"usage\":{\"prompt_tokens\":1000000,\"completion_tokens\":100000,",
            "\"total_tokens\":1100000,\"prompt_cache_hit_tokens\":700000,",
            "\"prompt_cache_miss_tokens\":300000}}\n\n",
            "data: [DONE]\n\n",
        ),
    )
    .await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["X"]);
    let mut model = model(&server);
    model.cost = ModelCost {
        input: 0.27,
        output: 1.10,
        cache_read: 0.07,
        cache_write: 0.0,
    };

    let message = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .final_message()
        .await;

    assert_eq!(message.usage.input, 300_000);
    assert_eq!(message.usage.cache_read, 700_000);
    assert_eq!(message.usage.cache_write, 0);
    assert_eq!(message.usage.output, 100_000);
    assert_eq!(message.usage.total_tokens, 1_100_000);
    assert!((message.usage.cost.input - 0.081).abs() < 1e-12);
    assert!((message.usage.cost.cache_read - 0.049).abs() < 1e-12);
    assert!((message.usage.cost.output - 0.11).abs() < 1e-12);
    assert!((message.usage.cost.total - 0.24).abs() < 1e-12);
}

#[tokio::test]
async fn reads_moonshot_usage_from_the_choice() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\",",
            "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,",
            "\"prompt_tokens_details\":{\"cached_tokens\":6}}}]}\n\n",
            "data: [DONE]\n\n",
        ),
    )
    .await;

    let provider = Provider::openai_compatible("moonshot", "Moonshot", server.uri(), ["X"]);
    let message = provider
        .stream(&model(&server), &Context::new().user("hi"), &options())
        .final_message()
        .await;

    assert_eq!(message.usage.input, 4);
    assert_eq!(message.usage.cache_read, 6);
    assert_eq!(message.usage.output, 2);
    assert_eq!(message.usage.total_tokens, 12);
}

#[tokio::test]
async fn sends_openai_cache_key_and_long_retention_when_declared() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    )
    .await;

    let provider = Provider::openai_compatible("custom", "Custom", server.uri(), ["X"])
        .with_openai_prompt_caching(OpenAiPromptCaching::OpenAi);
    let long_session_id = "x".repeat(80);
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Long),
        session_id: Some(long_session_id),
        ..options()
    };
    provider
        .stream(&model(&server), &Context::new().user("hi"), &options)
        .final_message()
        .await;

    let requests = server.received_requests().await.expect("request journal");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("JSON request");
    assert_eq!(body["prompt_cache_key"], "x".repeat(64));
    assert_eq!(body["prompt_cache_retention"], "24h");
}

#[tokio::test]
async fn generic_or_disabled_providers_send_no_openai_cache_extensions() {
    for (caching, retention) in [
        (OpenAiPromptCaching::Automatic, CacheRetention::Long),
        (OpenAiPromptCaching::OpenAi, CacheRetention::Disabled),
    ] {
        let server = MockServer::start().await;
        mount_sse(
            &server,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        )
        .await;

        let provider = Provider::openai_compatible("custom", "Custom", server.uri(), ["X"])
            .with_openai_prompt_caching(caching);
        let options = StreamOptions {
            cache_retention: Some(retention),
            session_id: Some("conversation-42".into()),
            ..options()
        };
        provider
            .stream(&model(&server), &Context::new().user("hi"), &options)
            .final_message()
            .await;

        let requests = server.received_requests().await.expect("request journal");
        let body: Value = serde_json::from_slice(&requests[0].body).expect("JSON request");
        assert!(body.get("prompt_cache_key").is_none());
        assert!(body.get("prompt_cache_retention").is_none());
    }
}

#[tokio::test]
async fn sends_session_affinity_headers_only_when_enabled() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    )
    .await;

    let provider = Provider::openai_compatible("custom", "Custom", server.uri(), ["X"])
        .with_openai_prompt_caching(OpenAiPromptCaching::SessionAffinityHeaders);
    let options = StreamOptions {
        session_id: Some("conversation-42".into()),
        ..options()
    };
    provider
        .stream(&model(&server), &Context::new().user("hi"), &options)
        .final_message()
        .await;

    let requests = server.received_requests().await.expect("request journal");
    let headers = &requests[0].headers;
    assert_eq!(
        headers.get("session_id").unwrap().to_str().unwrap(),
        "conversation-42"
    );
    assert_eq!(
        headers
            .get("x-client-request-id")
            .unwrap()
            .to_str()
            .unwrap(),
        "conversation-42"
    );
    assert_eq!(
        headers.get("x-session-affinity").unwrap().to_str().unwrap(),
        "conversation-42"
    );
}
