//! Provider request, usage accounting and model catalog regressions.
use banshu_ai::{
    Auth, Context, Model, Provider, ReasoningEffort, ReasoningOptions, StopReason, StreamOptions,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test".into()),
        max_tokens: Some(1024),
        ..Default::default()
    }
}

async fn completion(server: &MockServer, usage: Value) {
    Mock::given(method("POST")).and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/event-stream")
            .set_body_string(format!("data: {}\n\ndata: [DONE]\n\n", json!({
                "choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}], "usage": usage
            }))))
        .mount(server).await;
}

#[tokio::test]
async fn moonshot_top_level_cached_tokens_are_cache_reads() {
    let server = MockServer::start().await;
    completion(
        &server,
        json!({"prompt_tokens":10,"completion_tokens":2,"cached_tokens":6}),
    )
    .await;
    let provider = Provider::moonshot();
    let model = Model::openai_completions("test").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!((message.usage.input, message.usage.cache_read), (4, 6));
}

#[tokio::test]
async fn cache_writes_do_not_reduce_cache_hits() {
    let server = MockServer::start().await;
    completion(
        &server,
        json!({"prompt_tokens":100,"completion_tokens":2,
        "prompt_tokens_details":{"cached_tokens":60,"cache_write_tokens":10}}),
    )
    .await;
    let provider = Provider::openai_compatible("test", "Test", server.uri(), ["UNUSED"]);
    let model = Model::openai_completions("test").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await;
    assert_eq!(
        (
            message.usage.input,
            message.usage.cache_read,
            message.usage.cache_write
        ),
        (30, 60, 10)
    );
}

#[test]
fn bundled_catalog_removes_retired_models_and_includes_deepseek_vision() {
    assert!(
        Provider::xiaomi().models().iter().all(|m| ![
            "mimo-v2-flash",
            "mimo-v2-omni",
            "mimo-v2-pro"
        ]
        .contains(&m.id.as_str()))
    );
    let vision = Provider::deepseek()
        .models()
        .into_iter()
        .find(|m| m.id == "deepseek-v4-flash-vision-exp")
        .expect("vision catalog entry");
    assert!(vision.input.contains(&banshu_ai::Modality::Image));
}

#[test]
fn deepseek_efforts_are_model_specific() {
    let models = Provider::deepseek().models();
    let pro = models.iter().find(|m| m.id == "deepseek-v4-pro").unwrap();
    let flash = models.iter().find(|m| m.id == "deepseek-v4-flash").unwrap();
    assert!(!pro.reasoning.supports(ReasoningEffort::Low));
    assert!(flash.reasoning.supports(ReasoningEffort::Low));
}

#[tokio::test]
async fn zai_refresh_uses_subscription_catalog_and_sends_model_effort() {
    let server = MockServer::start().await;
    Mock::given(method("GET")).and(path("/catalog"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "zai":{"models":{"payg-only":{"reasoning":true}}},
            "zai-coding-plan":{"models":{"glm-test-highspeed":{
                "reasoning":true,"tool_call":true,"reasoning_options":[{"type":"effort","values":["high","max"]}],
                "modalities":{"input":["text"],"output":["text"]}
            }}}
        }))).mount(&server).await;
    completion(&server, json!({})).await;
    let provider = Provider::zai().with_auth(Auth::api_key_env(["BANSHU_UPSTREAM_NO_CREDENTIAL"]));
    provider
        .refresh_models_from(&format!("{}/catalog", server.uri()))
        .await;
    let mut model = provider
        .models()
        .into_iter()
        .find(|m| m.id == "glm-test-highspeed")
        .expect("subscription catalog source");
    assert!(!provider.models().iter().any(|m| m.id == "payg-only"));
    assert!(model.reasoning.supports(ReasoningEffort::Max));
    assert!(!model.reasoning.supports(ReasoningEffort::Low));
    model.base_url = server.uri();
    let mut options = options();
    options.reasoning = Some(ReasoningOptions::new(ReasoningEffort::Max));
    let result = provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;
    assert_eq!(result.stop_reason, StopReason::Stop, "{result:?}");
    let requests = server.received_requests().await.unwrap();
    let body: Value =
        serde_json::from_slice(&requests.iter().find(|r| r.method == "POST").unwrap().body)
            .unwrap();
    assert_eq!(body["reasoning_effort"], "max");
    assert_eq!(body["thinking"]["clear_thinking"], false);
}

#[tokio::test]
async fn moonshot_k3_can_request_effort() {
    let server = MockServer::start().await;
    completion(&server, json!({})).await;
    let provider = Provider::moonshot();
    let mut model = provider
        .models()
        .into_iter()
        .find(|m| m.id == "kimi-k3")
        .unwrap();
    model.base_url = server.uri();
    let mut options = options();
    options.reasoning = Some(ReasoningOptions::new(ReasoningEffort::High));
    let result = provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;
    assert_eq!(result.stop_reason, StopReason::Stop, "{result:?}");
    let body: Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["reasoning_effort"], "high");
}

#[tokio::test]
async fn retired_catalog_models_stay_removed_after_probe_and_offline_restore() {
    use banshu_ai::{InMemoryModelsStore, Models, RefreshOptions};
    use std::sync::Arc;
    let server = MockServer::start().await;
    Mock::given(method("GET")).and(path("/catalog"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"xiaomi":{"models":{
            "mimo-v2.5":{"status":"deprecated","tool_call":true,"modalities":{"input":["text"],"output":["text"]}}
        }}}))).mount(&server).await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"data":[{"id":"mimo-v2.5"}]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    // SAFETY: unique to this test and only used with the local mock endpoint.
    unsafe { std::env::set_var("BANSHU_RETIREMENT_PROBE_KEY", "test") };
    let build = || {
        Provider::openai_compatible(
            "xiaomi",
            "Xiaomi",
            server.uri(),
            ["BANSHU_RETIREMENT_PROBE_KEY"],
        )
        .with_models_dev_id("xiaomi")
    };
    let store = Arc::new(InMemoryModelsStore::new());
    let models = Models::new()
        .with_models_store(store.clone())
        .with_provider(build());
    assert!(models.get("xiaomi", "mimo-v2.5").is_some());
    models
        .refresh_from(&format!("{}/catalog", server.uri()))
        .await;
    assert!(models.get("xiaomi", "mimo-v2.5").is_none());
    let restored = Models::new()
        .with_models_store(store)
        .with_provider(build());
    restored
        .refresh_with(&RefreshOptions {
            allow_network: false,
            ..Default::default()
        })
        .await;
    assert!(restored.get("xiaomi", "mimo-v2.5").is_none());
    unsafe { std::env::remove_var("BANSHU_RETIREMENT_PROBE_KEY") };
}
