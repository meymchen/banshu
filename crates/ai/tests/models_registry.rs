//! Seam 3: the `Models` registry — lookup, auth-gated availability, dispatch.

use banshu_ai::{ApiKind, Context, Model, Models, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn get_looks_up_models_by_provider_and_id() {
    let models = Models::new()
        .with_provider(Provider::deepseek())
        .with_provider(Provider::kimi());

    let found = models
        .get("deepseek", "deepseek-chat")
        .expect("known model");
    assert_eq!(found.provider, "deepseek");
    assert_eq!(found.api, ApiKind::OpenAiCompletions);

    assert!(models.get("deepseek", "no-such-model").is_none());
    assert!(models.get("unregistered", "deepseek-chat").is_none());
}

#[test]
fn available_reflects_env_configured_providers() {
    // SAFETY: single-threaded test setup; we set/clear the provider env vars we
    // gate on so availability is deterministic regardless of the host env.
    unsafe {
        std::env::set_var("DEEPSEEK_API_KEY", "test");
        std::env::remove_var("KIMI_API_KEY");
    }

    let models = Models::new()
        .with_provider(Provider::deepseek())
        .with_provider(Provider::kimi());

    let available = models.available();
    assert!(available.iter().any(|m| m.provider == "deepseek"));
    assert!(available.iter().all(|m| m.provider != "kimi"));
}

const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"routed ok\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn stream_dispatches_to_the_owning_provider() {
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

    let models = Models::new().with_provider(Provider::openai_compatible(
        "acme",
        "Acme",
        server.uri(),
        ["ACME_API_KEY"],
    ));

    // A model owned by the registered `acme` provider, pointed at the mock.
    let mut model = Model::openai_completions("acme-1").with_base_url(server.uri());
    model.provider = "acme".to_string();

    let options = StreamOptions {
        api_key: Some("k".into()),
        ..Default::default()
    };
    let message = models
        .stream(&model, &Context::new().user("hi"), &options)
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "routed ok");
}

#[tokio::test]
async fn stream_for_an_unregistered_provider_is_an_in_band_error() {
    let models = Models::new();
    let model = Model::openai_completions("orphan").with_base_url("http://127.0.0.1:0");

    let message = models
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(message.error_message.is_some());
}
