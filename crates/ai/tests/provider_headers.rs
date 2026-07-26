//! Ticket #19: deterministic custom-header merging across every request level.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use banshu_ai::{
    ApiKind, Auth, AuthResolver, Context, Diagnostic, DiagnosticCode, Model, PreparedRequest,
    ProtocolAdapter, ProtocolEvent, ProtocolEventStream, Provider, ProviderHeaders, ResolvedAuth,
    Result, StreamOptions, async_trait,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Default)]
struct CapturedRequest {
    headers: ProviderHeaders,
    metadata: BTreeMap<String, serde_json::Value>,
}

struct CapturingAdapter {
    captured: Arc<Mutex<CapturedRequest>>,
}

impl ProtocolAdapter for CapturingAdapter {
    fn kind(&self) -> ApiKind {
        ApiKind::OpenAiCompletions
    }

    fn stream(&self, request: PreparedRequest) -> ProtocolEventStream {
        let mut captured = self.captured.lock().expect("capture lock");
        captured.headers = request
            .headers_with_protocol_defaults(&headers(&[("X-Protocol-Deleted", Some("protocol"))]));
        captured.metadata = request.options().metadata.clone();
        Box::pin(futures::stream::iter([ProtocolEvent::Stop(
            banshu_ai::StopReason::Stop,
        )]))
    }
}

struct HeaderAuth {
    api_key: Option<String>,
    headers: ProviderHeaders,
}

#[async_trait]
impl AuthResolver for HeaderAuth {
    async fn check(&self) -> Result<bool> {
        Ok(true)
    }

    async fn resolve(&self) -> Result<ResolvedAuth> {
        Ok(ResolvedAuth {
            api_key: self.api_key.clone(),
            headers: self.headers.clone(),
            ..Default::default()
        })
    }
}

fn headers(entries: &[(&str, Option<&str>)]) -> ProviderHeaders {
    entries
        .iter()
        .map(|(name, value)| {
            (
                (*name).to_string(),
                value.map(std::string::ToString::to_string),
            )
        })
        .collect()
}

#[tokio::test]
async fn prepared_request_merges_custom_headers_case_insensitively_in_priority_order() {
    let captured = Arc::new(Mutex::new(CapturedRequest::default()));
    let provider_headers = headers(&[
        ("x-protocol-deleted", None),
        ("X-Provider-Only", Some("provider")),
        ("X-Provider-Overridden", Some("provider")),
        ("X-Provider-Deleted", Some("provider")),
    ]);
    let auth_headers = headers(&[
        ("x-model-overridden", Some("auth")),
        ("x-model-deleted", None),
        ("X-Auth-Overridden", Some("auth")),
        ("X-Auth-Deleted", Some("auth")),
    ]);
    let provider = Provider::builder("p", "P", "http://localhost")
        .adapter(Arc::new(CapturingAdapter {
            captured: Arc::clone(&captured),
        }))
        .auth(Auth::custom(Arc::new(HeaderAuth {
            api_key: None,
            headers: auth_headers,
        })))
        .headers(provider_headers)
        .build()
        .expect("valid provider");

    let mut model = Model::openai_completions("m");
    model.provider = "p".to_string();
    model.headers = headers(&[
        ("x-provider-overridden", Some("model")),
        ("x-provider-deleted", None),
        ("X-Model-Overridden", Some("model")),
        ("X-Model-Deleted", Some("model")),
    ]);
    let options = StreamOptions {
        headers: headers(&[
            ("x-auth-overridden", Some("request")),
            ("x-auth-deleted", None),
        ]),
        metadata: BTreeMap::from([("trace".to_string(), serde_json::json!({"id": 19}))]),
        ..Default::default()
    };

    provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;

    let captured = captured.lock().expect("capture lock");
    assert_eq!(
        captured.headers,
        headers(&[
            ("X-Provider-Only", Some("provider")),
            ("x-provider-overridden", Some("model")),
            ("x-model-overridden", Some("auth")),
            ("x-auth-overridden", Some("request")),
        ])
    );
    assert_eq!(
        captured.metadata,
        BTreeMap::from([("trace".to_string(), serde_json::json!({"id": 19}))])
    );
}

#[tokio::test]
async fn openai_final_request_applies_protocol_through_request_header_priorities() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: [DONE]\n\n",
                )),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::builder("p", "P", server.uri())
        .adapter(Arc::new(
            banshu_ai::api::openai_completions::OpenAiCompletions,
        ))
        .auth(Auth::custom(Arc::new(HeaderAuth {
            api_key: Some("auth-secret".to_string()),
            headers: headers(&[
                ("X-Auth-Overridden", Some("auth")),
                ("X-Auth-Deleted", Some("auth")),
            ]),
        })))
        .headers(headers(&[
            ("Content-Type", None),
            ("X-Model-Overridden", Some("provider")),
            ("X-Model-Deleted", Some("provider")),
        ]))
        .build()
        .expect("valid provider");

    let mut model = Model::openai_completions("m").with_base_url(server.uri());
    model.provider = "p".to_string();
    model.headers = headers(&[
        ("x-model-overridden", Some("model")),
        ("x-model-deleted", None),
        ("X-Model-Wire", Some("model-wire")),
        ("x-auth-overridden", Some("model")),
    ]);
    let options = StreamOptions {
        headers: headers(&[
            ("authorization", Some("Request choice")),
            ("x-auth-overridden", Some("request")),
            ("x-auth-deleted", None),
        ]),
        ..Default::default()
    };

    provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;

    let requests = server.received_requests().await.expect("request history");
    assert_eq!(requests.len(), 1);
    let request_headers = &requests[0].headers;
    assert!(
        !request_headers.contains_key("content-type"),
        "provider None must delete the protocol default"
    );
    assert_eq!(
        request_headers
            .get("x-model-wire")
            .expect("model header reaches wire")
            .to_str()
            .unwrap(),
        "model-wire"
    );
    assert_eq!(
        request_headers
            .get("authorization")
            .expect("request auth override")
            .to_str()
            .unwrap(),
        "Request choice"
    );
    assert_eq!(
        request_headers.get_all("authorization").iter().count(),
        1,
        "only one value/casing may survive"
    );
    assert_eq!(
        request_headers
            .get("x-auth-overridden")
            .expect("request-level override")
            .to_str()
            .unwrap(),
        "request"
    );
    assert!(!request_headers.contains_key("x-auth-deleted"));
    assert!(!request_headers.contains_key("x-model-deleted"));
}

#[tokio::test]
async fn anthropic_final_request_uses_the_same_header_merge_contract() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::builder("p", "P", server.uri())
        .adapter(Arc::new(
            banshu_ai::api::anthropic_messages::AnthropicMessages,
        ))
        .auth(Auth::custom(Arc::new(HeaderAuth {
            api_key: Some("auth-secret".to_string()),
            headers: ProviderHeaders::new(),
        })))
        .headers(headers(&[("Anthropic-Version", None)]))
        .build()
        .expect("valid provider");

    let mut model = Model::anthropic_messages("m").with_base_url(server.uri());
    model.provider = "p".to_string();
    model.headers = headers(&[("X-Model-Wire", Some("anthropic-model"))]);
    let options = StreamOptions {
        headers: headers(&[("X-API-Key", Some("request-choice"))]),
        ..Default::default()
    };

    provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;

    let requests = server.received_requests().await.expect("request history");
    assert_eq!(requests.len(), 1);
    let request_headers = &requests[0].headers;
    assert!(
        !request_headers.contains_key("anthropic-version"),
        "provider None must delete the protocol version default"
    );
    assert_eq!(
        request_headers
            .get("x-model-wire")
            .expect("model header reaches wire")
            .to_str()
            .unwrap(),
        "anthropic-model"
    );
    assert_eq!(
        request_headers
            .get("x-api-key")
            .expect("request auth override")
            .to_str()
            .unwrap(),
        "request-choice"
    );
    assert_eq!(request_headers.get_all("x-api-key").iter().count(), 1);
}

#[test]
fn diagnostics_and_debug_output_redact_auth_header_values() {
    let sensitive_headers = headers(&[
        ("Authorization", Some("authorization-secret")),
        ("Cookie", Some("cookie-secret")),
        ("X-Gateway-Api-Key", Some("api-key-secret")),
        ("X-Public", Some("visible")),
    ]);
    let mut model = Model::openai_completions("m");
    model.headers = sensitive_headers.clone();
    let options = StreamOptions {
        api_key: Some("option-api-key-secret".to_string()),
        headers: sensitive_headers.clone(),
        metadata: BTreeMap::from([(
            "Authorization".to_string(),
            serde_json::json!("metadata-authorization-secret"),
        )]),
        ..Default::default()
    };
    let auth = ResolvedAuth {
        api_key: Some("resolved-api-key-secret".to_string()),
        headers: sensitive_headers,
        ..Default::default()
    };
    let diagnostic = Diagnostic::new(
        DiagnosticCode::Other,
        concat!(
            "Authorization: authorization-secret\n",
            "Cookie: cookie-secret\n",
            "X-Gateway-Api-Key: api-key-secret",
        ),
    );

    let output = format!("{model:?}\n{options:?}\n{auth:?}\n{diagnostic:?}");
    for secret in [
        "authorization-secret",
        "cookie-secret",
        "api-key-secret",
        "option-api-key-secret",
        "resolved-api-key-secret",
        "metadata-authorization-secret",
    ] {
        assert!(!output.contains(secret), "debug output leaked {secret}");
    }
    assert!(
        output.contains("visible"),
        "non-sensitive header values should remain useful"
    );
}
