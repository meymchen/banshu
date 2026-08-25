//! Prompt caching for OpenAI-compatible providers.
//!
//! Covers request-side cache-routing policies (issue #92) and the usage
//! variants returned by OpenAI, DeepSeek, OpenRouter-style endpoints, and
//! Moonshot.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use banshu_ai::{
    BeforeSendObservation, CacheRetention, Context, ErrorKind, Model, ModelCost,
    OpenAiCacheRetention, OpenAiCompat, OpenAiSessionAffinity, Provider, ProviderHeaders,
    RequestObserver, StreamOptions,
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
        tiers: Vec::new(),
    };

    let message = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
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
        tiers: Vec::new(),
    };

    let message = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
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
        .finish()
        .await;

    assert_eq!(message.usage.input, 4);
    assert_eq!(message.usage.cache_read, 6);
    assert_eq!(message.usage.output, 2);
    assert_eq!(message.usage.total_tokens, 12);
}

const DONE_SSE: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

/// The header trio the `SessionAffinityHeaders` shape sends.
const AFFINITY_HEADERS: [&str; 3] = ["session_id", "x-client-request-id", "x-session-affinity"];

fn provider(server: &MockServer, compat: OpenAiCompat) -> Provider {
    Provider::openai_compatible("custom", "Custom", server.uri(), ["X"]).with_openai_compat(compat)
}

/// Stream one request and return exactly what the server recorded.
async fn sent_request(
    server: &MockServer,
    provider: &Provider,
    options: &StreamOptions,
) -> wiremock::Request {
    provider
        .stream(&model(server), &Context::new().user("hi"), options)
        .finish()
        .await;
    let requests = server.received_requests().await.expect("request journal");
    assert_eq!(requests.len(), 1, "exactly one request should be sent");
    requests.into_iter().next().unwrap()
}

fn sent_body(request: &wiremock::Request) -> Value {
    serde_json::from_slice(&request.body).expect("JSON request")
}

fn header(request: &wiremock::Request, name: &str) -> String {
    request
        .headers
        .get(name)
        .unwrap_or_else(|| panic!("{name} should be sent"))
        .to_str()
        .expect("ASCII header value")
        .to_string()
}

#[tokio::test]
async fn prompt_cache_key_affinity_routes_the_session_id() {
    let server = MockServer::start().await;
    mount_sse(&server, DONE_SSE).await;

    let provider = provider(
        &server,
        OpenAiCompat {
            session_affinity: OpenAiSessionAffinity::PromptCacheKey,
            ..OpenAiCompat::default()
        },
    );
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Short),
        session_id: Some("x".repeat(80)),
        ..options()
    };
    let request = sent_request(&server, &provider, &options).await;

    let body = sent_body(&request);
    assert_eq!(
        body["prompt_cache_key"],
        "x".repeat(64),
        "the key is clamped to the field's limit: {body}"
    );
    assert!(
        body.get("prompt_cache_retention").is_none(),
        "short retention sends no retention field: {body}"
    );
    for name in AFFINITY_HEADERS {
        assert!(
            request.headers.get(name).is_none(),
            "{name} belongs to a different affinity shape"
        );
    }
}

#[tokio::test]
async fn header_affinity_routes_the_session_id_verbatim() {
    let server = MockServer::start().await;
    mount_sse(&server, DONE_SSE).await;

    let provider = provider(
        &server,
        OpenAiCompat {
            session_affinity: OpenAiSessionAffinity::SessionAffinityHeaders,
            ..OpenAiCompat::default()
        },
    );
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Short),
        session_id: Some("conversation-42".into()),
        ..options()
    };
    let request = sent_request(&server, &provider, &options).await;

    for name in AFFINITY_HEADERS {
        assert_eq!(header(&request, name), "conversation-42");
    }
    let body = sent_body(&request);
    assert!(
        body.get("prompt_cache_key").is_none(),
        "header affinity sends no body field: {body}"
    );
}

#[tokio::test]
async fn an_undeclared_affinity_sends_no_routing_field_or_headers() {
    let server = MockServer::start().await;
    mount_sse(&server, DONE_SSE).await;

    let provider = provider(&server, OpenAiCompat::default());
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Short),
        session_id: Some("conversation-42".into()),
        ..options()
    };
    let request = sent_request(&server, &provider, &options).await;

    let body = sent_body(&request);
    assert!(body.get("prompt_cache_key").is_none(), "{body}");
    assert!(body.get("prompt_cache_retention").is_none(), "{body}");
    for name in AFFINITY_HEADERS {
        assert!(
            request.headers.get(name).is_none(),
            "an unconfigured endpoint attests nothing, but {name} was sent"
        );
    }
}

#[tokio::test]
async fn an_attesting_provider_emits_long_retention() {
    let server = MockServer::start().await;
    mount_sse(&server, DONE_SSE).await;

    let provider = provider(
        &server,
        OpenAiCompat {
            session_affinity: OpenAiSessionAffinity::PromptCacheKey,
            cache_retention: OpenAiCacheRetention::Long,
            ..OpenAiCompat::default()
        },
    );
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Long),
        session_id: Some("x".repeat(80)),
        ..options()
    };
    let request = sent_request(&server, &provider, &options).await;

    let body = sent_body(&request);
    assert_eq!(body["prompt_cache_key"], "x".repeat(64), "{body}");
    assert_eq!(body["prompt_cache_retention"], "24h", "{body}");
}

#[tokio::test]
async fn short_retention_sends_no_retention_field_even_when_long_is_attested() {
    let server = MockServer::start().await;
    mount_sse(&server, DONE_SSE).await;

    let provider = provider(
        &server,
        OpenAiCompat {
            session_affinity: OpenAiSessionAffinity::PromptCacheKey,
            cache_retention: OpenAiCacheRetention::Long,
            ..OpenAiCompat::default()
        },
    );
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Short),
        session_id: Some("conversation-42".into()),
        ..options()
    };
    let request = sent_request(&server, &provider, &options).await;

    let body = sent_body(&request);
    assert_eq!(
        body["prompt_cache_key"], "conversation-42",
        "short retention keeps the endpoint's normal cache behavior: {body}"
    );
    assert!(body.get("prompt_cache_retention").is_none(), "{body}");
}

#[tokio::test]
async fn disabled_caching_sends_no_cache_fields_or_affinity_headers() {
    for affinity in [
        OpenAiSessionAffinity::PromptCacheKey,
        OpenAiSessionAffinity::SessionAffinityHeaders,
    ] {
        let server = MockServer::start().await;
        mount_sse(&server, DONE_SSE).await;

        let provider = provider(
            &server,
            OpenAiCompat {
                session_affinity: affinity,
                cache_retention: OpenAiCacheRetention::Long,
                ..OpenAiCompat::default()
            },
        );
        let options = StreamOptions {
            cache_retention: Some(CacheRetention::Disabled),
            session_id: Some("conversation-42".into()),
            ..options()
        };
        let request = sent_request(&server, &provider, &options).await;

        let body = sent_body(&request);
        assert!(
            body.get("prompt_cache_key").is_none(),
            "{affinity:?}: {body}"
        );
        assert!(
            body.get("prompt_cache_retention").is_none(),
            "{affinity:?}: {body}"
        );
        for name in AFFINITY_HEADERS {
            assert!(
                request.headers.get(name).is_none(),
                "{affinity:?}: disabled caching suppresses {name}"
            );
        }
    }
}

#[tokio::test]
async fn an_unsupported_long_retention_is_refused_before_any_request() {
    for compat in [
        // Nothing declared at all.
        OpenAiCompat::default(),
        // Affinity routing declared, but no long-retention attestation.
        OpenAiCompat {
            session_affinity: OpenAiSessionAffinity::PromptCacheKey,
            ..OpenAiCompat::default()
        },
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(DONE_SSE),
            )
            .mount(&server)
            .await;

        let provider = provider(&server, compat);
        let options = StreamOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("conversation-42".into()),
            ..options()
        };
        let message = provider
            .stream(&model(&server), &Context::new().user("hi"), &options)
            .finish()
            .await;

        assert_eq!(
            message.error_kind,
            Some(ErrorKind::InvalidRequest),
            "{compat:?} should refuse an explicit Long in-band"
        );
        assert!(
            server
                .received_requests()
                .await
                .unwrap_or_default()
                .is_empty(),
            "{compat:?} must be refused before any HTTP request"
        );
    }
}

#[tokio::test]
async fn session_affinity_never_touches_credential_headers() {
    let server = MockServer::start().await;
    mount_sse(&server, DONE_SSE).await;

    let provider = provider(
        &server,
        OpenAiCompat {
            session_affinity: OpenAiSessionAffinity::SessionAffinityHeaders,
            ..OpenAiCompat::default()
        },
    );
    let options = StreamOptions {
        session_id: Some("conversation-42".into()),
        headers: ProviderHeaders::from([(
            "x-api-key".to_string(),
            Some("request-layer-secret".to_string()),
        )]),
        ..options()
    };
    let request = sent_request(&server, &provider, &options).await;

    assert_eq!(
        header(&request, "authorization"),
        "Bearer test-key",
        "the generated credential header arrives exactly as auth produced it"
    );
    assert_eq!(
        header(&request, "x-api-key"),
        "request-layer-secret",
        "a request-layer credential header is never rewritten either"
    );
    for name in AFFINITY_HEADERS {
        assert_eq!(header(&request, name), "conversation-42");
    }
}

/// Records the redacted headers and payload of every before-send observation.
#[derive(Default)]
struct WireObserver {
    before_sends: Mutex<Vec<(BTreeMap<String, String>, Value)>>,
}

impl RequestObserver for WireObserver {
    fn before_send(&self, observation: &BeforeSendObservation) {
        self.before_sends
            .lock()
            .unwrap()
            .push((observation.headers.clone(), observation.payload.clone()));
    }
}

#[tokio::test]
async fn the_observed_payload_and_headers_match_what_the_server_received() {
    let server = MockServer::start().await;
    mount_sse(&server, DONE_SSE).await;

    // Long retention is declared independently of the affinity shape, so this
    // request also exercises the retention field without a cache key.
    let provider = provider(
        &server,
        OpenAiCompat {
            session_affinity: OpenAiSessionAffinity::SessionAffinityHeaders,
            cache_retention: OpenAiCacheRetention::Long,
            ..OpenAiCompat::default()
        },
    );
    let observer = Arc::new(WireObserver::default());
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::Long),
        session_id: Some("conversation-42".into()),
        headers: ProviderHeaders::from([("x-trace-id".to_string(), Some("trace-42".to_string()))]),
        observer: Some(observer.clone()),
        ..options()
    };
    let request = sent_request(&server, &provider, &options).await;

    let body = sent_body(&request);
    assert_eq!(body["prompt_cache_retention"], "24h", "{body}");

    let before_sends = observer.before_sends.lock().unwrap();
    let [(headers, payload)] = before_sends.as_slice() else {
        panic!("expected exactly one before-send observation");
    };
    assert_eq!(
        *payload, body,
        "the observer sees the exact body the server recorded"
    );
    for name in AFFINITY_HEADERS {
        assert_eq!(
            headers.get(name).map(String::as_str),
            Some(header(&request, name).as_str()),
            "the observer sees the {name} the server received"
        );
    }
    for name in ["Content-Type", "x-trace-id"] {
        assert_eq!(
            headers.get(name).map(String::as_str),
            Some(header(&request, name).as_str()),
            "the observer's {name} matches the server's"
        );
    }
    assert_eq!(
        headers.get("Authorization").map(String::as_str),
        Some("[REDACTED]"),
        "the credential reaches the server but never the observer"
    );
    assert_eq!(header(&request, "authorization"), "Bearer test-key");
}
