//! Ticket #18: the public extension seams.
//!
//! A third-party `ProtocolAdapter` registers through `ProviderBuilder` and
//! streams end-to-end against wiremock; a mixed-protocol provider routes each
//! model by `Model.api`; invalid builder configurations fail with config
//! errors (never panics); and `Models::available()` consults custom resolvers
//! asynchronously.

use std::sync::Arc;

use banshu_ai::api::anthropic_messages::AnthropicMessages;
use banshu_ai::api::openai_completions::OpenAiCompletions;
use banshu_ai::{
    ApiKind, Auth, AuthResolver, Context, Error, ErrorKind, Model, Models, PreparedRequest,
    ProtocolAdapter, ProtocolEvent, ProtocolEventStream, Provider, ProviderHeaders, ResolvedAuth,
    Result, StopReason, StreamOptions, async_trait,
};
use futures::StreamExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A stand-in third-party protocol: POSTs the context's message count as a
/// plain-text body and replies with plain text — nothing like OpenAI's wire
/// format, proving the adapter seam is protocol-agnostic. It only reads
/// `PreparedRequest` through its accessors.
struct PlainTextProtocol;

impl ProtocolAdapter for PlainTextProtocol {
    fn kind(&self) -> ApiKind {
        ApiKind::OpenAiCompletions
    }

    fn stream(&self, request: PreparedRequest) -> ProtocolEventStream {
        let http = request.http_client().clone();
        let url = format!("{}/plain", request.model().base_url.trim_end_matches('/'));
        let api_key = request.auth().api_key.clone();
        let message_count = request.context().messages.len();
        let events = async move {
            let mut builder = http.post(&url).body(message_count.to_string());
            if let Some(key) = api_key {
                builder = builder.bearer_auth(key);
            }
            match builder.send().await {
                Ok(response) => match response.text().await {
                    Ok(text) => vec![
                        ProtocolEvent::TextStart {
                            block_id: 0,
                            signature: None,
                        },
                        ProtocolEvent::TextDelta {
                            block_id: 0,
                            delta: text,
                        },
                        ProtocolEvent::TextEnd { block_id: 0 },
                        ProtocolEvent::Stop(StopReason::Stop),
                    ],
                    Err(err) => vec![ProtocolEvent::Failure {
                        kind: ErrorKind::Transport,
                        message: err.to_string(),
                        diagnostics: Vec::new(),
                    }],
                },
                Err(err) => vec![ProtocolEvent::Failure {
                    kind: ErrorKind::Transport,
                    message: err.to_string(),
                    diagnostics: Vec::new(),
                }],
            }
        };
        Box::pin(futures::stream::once(events).flat_map(futures::stream::iter))
    }
}

fn model_for(provider: &str, id: &str, base_url: &str) -> Model {
    let mut model = Model::openai_completions(id).with_base_url(base_url);
    model.provider = provider.to_string();
    model
}

#[tokio::test]
async fn external_adapter_streams_end_to_end_via_builder() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/plain"))
        .and(header("authorization", "Bearer sk-ext"))
        .respond_with(ResponseTemplate::new(200).set_body_string("EXT OK"))
        .expect(1)
        .mount(&server)
        .await;

    let model = model_for("ext", "ext-1", &server.uri());
    let provider = Provider::builder("ext", "Ext", server.uri())
        .adapter(Arc::new(PlainTextProtocol))
        .model(model.clone())
        .build()
        .expect("valid provider");

    let models = Models::new().with_provider(provider);
    // The builder-registered model surfaces through the registry.
    assert_eq!(models.get("ext", "ext-1").expect("listed").id, "ext-1");

    let options = StreamOptions {
        api_key: Some("sk-ext".into()),
        ..Default::default()
    };
    let message = models
        .complete(&model, &Context::new().user("hi"), &options)
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "EXT OK");
    assert_eq!(message.provider, "ext");
    assert_eq!(message.api, "openai-completions");
}

#[tokio::test]
async fn mixed_protocol_provider_routes_each_model_by_its_api() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"via openai\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
                    "data: [DONE]\n\n",
                )),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
                    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"via anthropic\"}}\n\n",
                    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                )),
        )
        .expect(1)
        .mount(&server)
        .await;

    let openai_model = model_for("mixed", "m-openai", &server.uri());
    let mut anthropic_model = Model::anthropic_messages("m-anthropic").with_base_url(server.uri());
    anthropic_model.provider = "mixed".to_string();

    // Keyless by default: no credentials needed against the mock.
    let provider = Provider::builder("mixed", "Mixed", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .adapter(Arc::new(AnthropicMessages))
        .model(openai_model.clone())
        .model(anthropic_model.clone())
        .build()
        .expect("both protocols covered");

    let context = Context::new().user("hi");
    let options = StreamOptions::default();
    let openai_message = provider
        .stream(&openai_model, &context, &options)
        .finish()
        .await;
    let anthropic_message = provider
        .stream(&anthropic_model, &context, &options)
        .finish()
        .await;

    assert_eq!(openai_message.stop_reason, StopReason::Stop);
    assert_eq!(openai_message.text(), "via openai");
    assert_eq!(anthropic_message.stop_reason, StopReason::Stop);
    assert_eq!(anthropic_message.text(), "via anthropic");
}

#[test]
fn model_with_an_unregistered_protocol_fails_the_build() {
    let mut model = Model::anthropic_messages("m");
    model.provider = "p".to_string();

    let err = Provider::builder("p", "P", "http://localhost")
        .adapter(Arc::new(OpenAiCompletions))
        .model(model)
        .build()
        .err()
        .expect("build must fail");

    assert!(matches!(err, Error::Config(_)));
    let message = err.to_string();
    assert!(message.contains("anthropic-messages"), "got: {message}");
    assert!(message.contains('p'), "got: {message}");
}

#[test]
fn model_for_another_provider_id_fails_the_build() {
    let model = model_for("someone-else", "m", "http://localhost");

    let err = Provider::builder("p", "P", "http://localhost")
        .adapter(Arc::new(OpenAiCompletions))
        .model(model)
        .build()
        .err()
        .expect("build must fail");

    assert!(matches!(err, Error::Config(_)));
    assert!(err.to_string().contains("someone-else"));
}

#[test]
fn builder_requires_a_non_empty_id() {
    let err = Provider::builder("  ", "P", "http://localhost")
        .adapter(Arc::new(OpenAiCompletions))
        .build()
        .err()
        .expect("build must fail");
    assert!(matches!(err, Error::Config(_)));
    assert!(err.to_string().contains("id"));
}

#[test]
fn builder_requires_at_least_one_adapter() {
    let err = Provider::builder("p", "P", "http://localhost")
        .build()
        .err()
        .expect("build must fail");
    assert!(matches!(err, Error::Config(_)));
    assert!(err.to_string().contains("adapter"));
}

#[test]
fn builder_rejects_two_adapters_for_the_same_protocol() {
    let err = Provider::builder("p", "P", "http://localhost")
        .adapter(Arc::new(OpenAiCompletions))
        .adapter(Arc::new(OpenAiCompletions))
        .build()
        .err()
        .expect("build must fail");
    assert!(matches!(err, Error::Config(_)));
    assert!(err.to_string().contains("openai-completions"));
}

/// A resolver whose availability answer is gated by a flag.
struct Gate(bool);

#[async_trait]
impl AuthResolver for Gate {
    async fn check(&self) -> Result<bool> {
        Ok(self.0)
    }
    async fn resolve(&self) -> Result<ResolvedAuth> {
        Ok(ResolvedAuth::default())
    }
}

#[tokio::test]
async fn available_consults_custom_resolvers_asynchronously() {
    let gated_provider = |id: &str, on: bool| {
        Provider::builder(id, id, "http://localhost")
            .adapter(Arc::new(OpenAiCompletions))
            .auth(Auth::custom(Arc::new(Gate(on))))
            .model(model_for(id, "m", "http://localhost"))
            .build()
            .expect("valid provider")
    };
    // The synchronous best-effort check can't consult a resolver at all —
    // both providers report unavailable there.
    assert!(!gated_provider("probe", true).is_available());

    let models = Models::new()
        .with_provider(gated_provider("gated-on", true))
        .with_provider(gated_provider("gated-off", false));

    let available = models.available().await;
    assert!(available.iter().any(|m| m.provider == "gated-on"));
    assert!(available.iter().all(|m| m.provider != "gated-off"));
}

#[tokio::test]
async fn provider_default_headers_reach_the_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("x-tenant", "acme"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: [DONE]\n\n",
                )),
        )
        .expect(1)
        .mount(&server)
        .await;

    let headers = ProviderHeaders::from([("x-tenant".to_string(), Some("acme".to_string()))]);
    let provider = Provider::builder("p", "P", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .headers(headers)
        .build()
        .expect("valid provider");

    let model = model_for("p", "m", &server.uri());
    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
}
