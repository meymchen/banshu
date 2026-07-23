//! Ticket #16: pluggable authentication adapters.
//!
//! Env-var, explicit-key, keyless, and custom-resolver auth all stream
//! end-to-end against wiremock; keyless sends no auth header; a resolver
//! failure terminates in-band as an `Auth` error with no partial content; and
//! an explicit `StreamOptions.api_key` beats whatever a resolver would return.

use std::sync::Arc;

use banshu_ai::{
    Auth, AuthResolver, Context, ErrorKind, Model, Provider, ResolvedAuth, Result, StopReason,
    StreamOptions, async_trait,
};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Minimal one-delta OpenAI completion.
const SSE_BODY: &str = concat!(
    "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
);

fn ok_response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(SSE_BODY)
}

fn model(base_url: &str) -> Model {
    Model::openai_completions("m").with_base_url(base_url)
}

/// A resolver that hands back a fixed key (and optionally a base URL override).
struct StaticAuth {
    api_key: Option<String>,
    base_url: Option<String>,
}

#[async_trait]
impl AuthResolver for StaticAuth {
    async fn check(&self) -> Result<bool> {
        Ok(self.api_key.is_some())
    }
    async fn resolve(&self) -> Result<ResolvedAuth> {
        Ok(ResolvedAuth {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            ..Default::default()
        })
    }
}

/// A resolver that always fails — a stand-in for a missing token file or a
/// broker that is down.
struct FailingAuth;

#[async_trait]
impl AuthResolver for FailingAuth {
    async fn check(&self) -> Result<bool> {
        Ok(false)
    }
    async fn resolve(&self) -> Result<ResolvedAuth> {
        Err(banshu_ai::Error::Auth("token file not found".into()))
    }
}

#[tokio::test]
async fn env_key_streams_and_sends_bearer_header() {
    let var = "BANSHU_AUTH_TEST_ENV_KEY";
    unsafe { std::env::set_var(var, "env-secret") };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer env-secret"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("p", "P", server.uri(), [var]);
    let message = provider
        .stream(
            &model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");
    unsafe { std::env::remove_var(var) };
}

#[tokio::test]
async fn explicit_option_key_streams_and_beats_the_resolver() {
    let server = MockServer::start().await;
    // Only a request bearing the explicit key matches; the resolver's key must
    // never reach the wire.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer explicit"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("p", "P", server.uri(), ["UNUSED"]).with_auth(
        Auth::custom(Arc::new(StaticAuth {
            api_key: Some("resolver-key".into()),
            base_url: None,
        })),
    );
    let options = StreamOptions {
        api_key: Some("explicit".into()),
        ..Default::default()
    };
    let message = provider
        .stream(&model(&server.uri()), &Context::new().user("hi"), &options)
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn explicit_key_wins_even_when_the_resolver_would_fail() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer explicit"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("p", "P", server.uri(), ["UNUSED"])
        .with_auth(Auth::custom(Arc::new(FailingAuth)));
    let options = StreamOptions {
        api_key: Some("explicit".into()),
        ..Default::default()
    };
    let message = provider
        .stream(&model(&server.uri()), &Context::new().user("hi"), &options)
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn custom_resolver_streams_end_to_end() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-custom"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("p", "P", server.uri(), ["UNUSED"]).with_auth(
        Auth::custom(Arc::new(StaticAuth {
            api_key: Some("sk-custom".into()),
            base_url: None,
        })),
    );
    let message = provider
        .stream(
            &model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");
}

#[tokio::test]
async fn keyless_streams_without_an_auth_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("p", "P", server.uri(), ["UNUSED"]).with_auth(Auth::keyless());
    let message = provider
        .stream(
            &model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].headers.contains_key("authorization"),
        "keyless request must carry no auth header"
    );
}

#[tokio::test]
async fn resolver_failure_is_an_inband_auth_error_and_hits_no_endpoint() {
    let server = MockServer::start().await;
    // A resolver failure must terminate before any request is sent.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ok_response())
        .expect(0)
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("p", "P", server.uri(), ["UNUSED"])
        .with_auth(Auth::custom(Arc::new(FailingAuth)));
    let message = provider
        .stream(
            &model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Auth));
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("token file not found"),
        "message should carry the resolver's cause"
    );
    // Partial-free: the failure happens before any content streams.
    assert_eq!(message.text(), "");
    assert!(
        message.content.is_empty(),
        "no partial content on an auth failure"
    );

    let requests = server.received_requests().await.expect("recorded requests");
    assert!(requests.is_empty(), "no request should reach the endpoint");
}

#[tokio::test]
async fn resolver_base_url_override_redirects_the_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-custom"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    // Model points at a black hole; the resolver redirects to the live server.
    let provider = Provider::openai_compatible("p", "P", "http://127.0.0.1:0", ["UNUSED"])
        .with_auth(Auth::custom(Arc::new(StaticAuth {
            api_key: Some("sk-custom".into()),
            base_url: Some(server.uri()),
        })));
    let message = provider
        .stream(
            &model("http://127.0.0.1:0"),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");
}

#[tokio::test]
async fn keyless_anthropic_sends_no_api_key_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
                    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                )),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("p", "P", server.uri(), ["UNUSED"])
        .with_auth(Auth::keyless());
    let anthropic_model = Model::anthropic_messages("m").with_base_url(server.uri());
    let message = provider
        .stream(
            &anthropic_model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].headers.contains_key("x-api-key"),
        "keyless Anthropic request must carry no x-api-key header"
    );
}
