//! Seam 1: failures are in-band, not `Result`.
//!
//! A non-2xx response and an unresolved API key both terminate the stream with
//! an `Error` event whose message has `stop_reason: Error` and a populated
//! `error_message` — `stream()`/`final_message()` never fail synchronously.

use banshu_ai::{Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn http_error_status_becomes_a_terminal_error_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string("{\"error\":{\"message\":\"invalid api key\"}}"),
        )
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("hi");
    let options = StreamOptions {
        api_key: Some("bad-key".into()),
        ..Default::default()
    };

    let message = provider
        .stream(&model, &context, &options)
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    let error = message.error_message.expect("expected an error message");
    assert!(
        error.contains("401"),
        "error should mention the status: {error}"
    );
    assert!(
        error.contains("invalid api key"),
        "error should include the provider body: {error}"
    );
}

#[tokio::test]
async fn unresolved_api_key_becomes_a_terminal_error_message() {
    // No api_key in options and a deliberately-unset env var.
    let provider = Provider::openai_compatible(
        "deepseek",
        "DeepSeek",
        "http://127.0.0.1:0",
        ["BANSHU_DEFINITELY_UNSET_KEY_ENV"],
    );
    let model = Model::openai_completions("deepseek-chat").with_base_url("http://127.0.0.1:0");
    let context = Context::new().user("hi");

    let message = provider
        .stream(&model, &context, &StreamOptions::default())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or_default()
            .to_lowercase()
            .contains("api key"),
        "error should mention the missing API key"
    );
}
