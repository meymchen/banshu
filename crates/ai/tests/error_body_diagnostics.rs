//! Non-2xx error bodies: a clean `message (code)` summary goes into
//! `error_message`, the raw body only ever reaches `diagnostics` (capped and
//! redacted), and a body large enough to matter is capped before it's even
//! parsed for a message.

use banshu_ai::{Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn options_with_key() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn json_error_message_and_code_are_extracted_into_error_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string(
            "{\"error\":{\"message\":\"bad request\",\"code\":\"invalid_params\"}}",
        ))
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    let error = message.error_message.expect("expected an error message");
    assert!(error.contains("bad request"));
    assert!(error.contains("invalid_params"));
}

#[tokio::test]
async fn oversized_error_body_is_capped_before_message_extraction() {
    let server = MockServer::start().await;
    // The `message` field starts well past the 4096-char cap: once the body
    // is capped before parsing, the JSON is truncated mid-string and
    // extraction fails outright — proving the cap applies before parsing,
    // not just to what's stored afterward.
    let padding = "x".repeat(5_000);
    let body = format!(
        "{{\"pad\":\"{padding}\",\"error\":{{\"message\":\"tail message must be unreachable\"}}}}"
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string(body))
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    let error = message.error_message.unwrap_or_default();
    assert!(
        !error.contains("tail message"),
        "the message field past the cap must never reach error_message: {error}"
    );
    assert_eq!(message.diagnostics.len(), 1);
    assert!(
        message.diagnostics[0].message.chars().count() <= 1024,
        "the diagnostic must stay within its own cap regardless of the raw body size"
    );
}

#[tokio::test]
async fn error_body_diagnostic_redacts_secrets_but_original_body_had_them() {
    let server = MockServer::start().await;
    let secret = "sk-verysecret1234567890";
    let body = format!(
        "{{\"error\":{{\"message\":\"failed\"}},\"debug\":\"Authorization: Bearer {secret}\"}}"
    );
    assert!(body.contains(secret), "test setup sanity check");
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string(body))
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options_with_key())
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.diagnostics.len(), 1);
    assert!(
        !message.diagnostics[0].message.contains(secret),
        "diagnostic must redact the secret: {}",
        message.diagnostics[0].message
    );
    assert!(
        !message
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains(secret),
        "error_message must never carry the raw body at all"
    );
}
