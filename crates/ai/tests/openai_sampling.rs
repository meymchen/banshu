//! Guarded OpenAI-compatible sampling parameters (issue #96).
//!
//! [`StreamOptions::sampling`] is the escape hatch for open-model sampling
//! controls the crate does not model: values merge into the top level of the
//! request body verbatim, and a key that would shadow an adapter-owned field
//! is refused in-band with `ErrorKind::InvalidRequest` — naming the key —
//! before the server ever records a request. The Anthropic adapter ignores
//! the map entirely.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use banshu_ai::api::openai_completions::OpenAiCompletions;
use banshu_ai::{
    BeforeSendObservation, Context, ErrorKind, Model, Provider, RequestObserver, StreamOptions,
};
use proptest::prelude::*;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const DONE_SSE: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

const ANTHROPIC_DONE_SSE: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// The adapter-owned request fields, by family, pinning the public contract
/// the crate's reserved set must keep. The properties below draw from the
/// crate's own constant, and
/// [`the_reserved_set_matches_the_pinned_contract`] fails on any drift
/// between the two — in either direction.
const RESERVED_KEYS: &[&str] = &[
    // Model.
    "model",
    // Messages.
    "messages",
    // Tools.
    "tools",
    // Stream controls.
    "stream",
    "stream_options",
    // Output budget.
    "max_tokens",
    "max_completion_tokens",
    // Reasoning.
    "thinking",
    "reasoning_effort",
    "enable_thinking",
    "chat_template_kwargs",
    // Tool choice.
    "tool_choice",
    // Caching.
    "prompt_cache_key",
    "prompt_cache_retention",
    // Sampling the crate already models.
    "temperature",
    // Caller metadata.
    "metadata",
    // Authentication-related values.
    "api_key",
    "authorization",
    "x-api-key",
];

async fn mock_server() -> MockServer {
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
    server
}

fn provider(server: &MockServer) -> Provider {
    Provider::openai_compatible("custom", "Custom", server.uri(), ["X"])
}

fn model(server: &MockServer) -> Model {
    Model::openai_completions("test-model").with_base_url(server.uri())
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn sampling<K: AsRef<str>>(pairs: impl IntoIterator<Item = (K, Value)>) -> StreamOptions {
    StreamOptions {
        sampling: pairs
            .into_iter()
            .map(|(key, value)| (key.as_ref().to_string(), value))
            .collect(),
        ..options()
    }
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

/// A multi-thread runtime with a mock server and provider already wired up —
/// the shared setup of the wire-level properties.
fn wire_harness() -> (tokio::runtime::Runtime, MockServer, Provider) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let server = runtime.block_on(mock_server());
    let provider = provider(&server);
    (runtime, server, provider)
}

#[test]
fn the_reserved_set_matches_the_pinned_contract() {
    let mut pinned = RESERVED_KEYS.to_vec();
    pinned.sort_unstable();
    let mut reserved = OpenAiCompletions::RESERVED_SAMPLING_KEYS.to_vec();
    reserved.sort_unstable();
    assert_eq!(
        reserved, pinned,
        "the crate's reserved set drifted from the pinned adapter-owned contract"
    );
}

#[tokio::test]
async fn named_open_model_controls_reach_the_wire_verbatim() {
    let server = mock_server().await;
    let request = sent_request(
        &server,
        &provider(&server),
        &sampling([
            ("top_p", json!(0.9)),
            ("top_k", json!(40)),
            ("min_p", json!(0.05)),
            ("repetition_penalty", json!(1.1)),
            ("frequency_penalty", json!(0.5)),
            ("presence_penalty", json!(-0.5)),
            ("seed", json!(42)),
            ("stop", json!(["\n", "END"])),
        ]),
    )
    .await;

    let body = sent_body(&request);
    assert_eq!(body["top_p"], json!(0.9), "{body}");
    assert_eq!(body["top_k"], json!(40), "{body}");
    assert_eq!(body["min_p"], json!(0.05), "{body}");
    assert_eq!(body["repetition_penalty"], json!(1.1), "{body}");
    assert_eq!(body["frequency_penalty"], json!(0.5), "{body}");
    assert_eq!(body["presence_penalty"], json!(-0.5), "{body}");
    assert_eq!(body["seed"], json!(42), "{body}");
    assert_eq!(body["stop"], json!(["\n", "END"]), "{body}");
    // Unknown non-reserved keys remain representable.
    assert!(body.get("adapter_owned_marker").is_none(), "{body}");
}

#[tokio::test]
async fn every_json_value_kind_is_represented_faithfully() {
    let server = mock_server().await;
    let nested = json!({
        "tokens": [106, 271],
        "scale": 1.5,
        "enabled": true,
        "nothing": null,
    });
    let request = sent_request(
        &server,
        &provider(&server),
        &sampling([
            // An integer beyond f64's exact range must survive as an integer.
            ("big_integer", json!(9_007_199_254_740_993i64)),
            ("negative_integer", json!(-7)),
            ("unsigned_integer", json!(u64::MAX)),
            ("float", json!(0.300_000_000_000_000_04)),
            ("boolean", json!(false)),
            ("string", json!("verbatim")),
            ("array", json!([1, "two", 3.0, null])),
            ("object", nested.clone()),
            ("null", Value::Null),
        ]),
    )
    .await;

    let body = sent_body(&request);
    assert_eq!(
        body["big_integer"],
        json!(9_007_199_254_740_993i64),
        "{body}"
    );
    assert_eq!(body["negative_integer"], json!(-7), "{body}");
    assert_eq!(body["unsigned_integer"], json!(u64::MAX), "{body}");
    assert_eq!(body["float"], json!(0.300_000_000_000_000_04), "{body}");
    assert_eq!(body["boolean"], json!(false), "{body}");
    assert_eq!(body["string"], json!("verbatim"), "{body}");
    assert_eq!(body["array"], json!([1, "two", 3.0, null]), "{body}");
    assert_eq!(body["object"], nested, "{body}");
    assert_eq!(body["null"], Value::Null, "{body}");
}

#[tokio::test]
async fn an_empty_sampling_map_adds_nothing_to_the_request() {
    let server = mock_server().await;
    let provider = provider(&server);
    for options in [options(), sampling::<&str>([])] {
        provider
            .stream(&model(&server), &Context::new().user("hi"), &options)
            .finish()
            .await;
    }

    let requests = server.received_requests().await.expect("request journal");
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].body, requests[1].body,
        "an explicit empty map serializes exactly like no map at all"
    );
}

#[tokio::test]
async fn every_reserved_key_fails_before_the_server_records_a_request() {
    let server = mock_server().await;
    let provider = provider(&server);
    for key in RESERVED_KEYS {
        let message = provider
            .stream(
                &model(&server),
                &Context::new().user("hi"),
                &sampling([(key, json!(1))]),
            )
            .finish()
            .await;

        assert_eq!(
            message.error_kind,
            Some(ErrorKind::InvalidRequest),
            "{key} must be refused in-band"
        );
        let detail = message.error_message.unwrap_or_default();
        assert!(detail.contains(key), "{detail} should name `{key}`");
    }
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "a reserved key must be refused before any HTTP request"
    );
}

#[tokio::test]
async fn the_anthropic_adapter_ignores_the_sampling_map() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(ANTHROPIC_DONE_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("custom", "Custom", server.uri(), ["X"]);
    let model = Model::anthropic_messages("test-model").with_base_url(server.uri());
    // Even a key reserved on the OpenAI protocol is inert here: the Anthropic
    // adapter never reads the map.
    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &sampling([("top_p", json!(0.9)), ("model", json!("smuggled-model"))]),
        )
        .finish()
        .await;

    assert!(
        message.error_kind.is_none(),
        "the Anthropic protocol ignores the map, so the request succeeds: {:?}",
        message.error_message
    );
    let requests = server.received_requests().await.expect("request journal");
    assert_eq!(requests.len(), 1);
    let body = sent_body(&requests[0]);
    assert_eq!(
        body["model"],
        json!("test-model"),
        "the adapter's own model field stands: {body}"
    );
    assert!(body.get("top_p").is_none(), "nothing merges: {body}");
}

/// Records the redacted payload of every before-send observation.
#[derive(Default)]
struct WireObserver {
    payloads: Mutex<Vec<Value>>,
}

impl RequestObserver for WireObserver {
    fn before_send(&self, observation: &BeforeSendObservation) {
        self.payloads
            .lock()
            .unwrap()
            .push(observation.payload.clone());
    }
}

#[tokio::test]
async fn the_observer_sees_exactly_the_merged_payload() {
    let server = mock_server().await;
    let observer = Arc::new(WireObserver::default());
    let options = StreamOptions {
        observer: Some(observer.clone()),
        ..sampling([("top_k", json!(40)), ("stop", json!(["END"]))])
    };
    let request = sent_request(&server, &provider(&server), &options).await;

    let body = sent_body(&request);
    assert_eq!(body["top_k"], json!(40), "{body}");
    let payloads = observer.payloads.lock().unwrap();
    let [payload] = payloads.as_slice() else {
        panic!("expected exactly one before-send observation");
    };
    assert_eq!(
        *payload, body,
        "the observer sees the exact merged body the server recorded"
    );
}

/// Any JSON value: integer, float, boolean, string, array, object, or null.
fn arb_json() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(|n| json!(n)),
        any::<u64>().prop_map(|n| json!(n)),
        any::<f64>()
            .prop_filter("JSON cannot carry NaN or infinities", |f| f.is_finite())
            .prop_map(|f| json!(f)),
        "[ -~]{0,24}".prop_map(Value::String),
    ];
    leaf.prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            prop::collection::btree_map("[a-z_]{1,8}", inner, 0..4)
                .prop_map(|map: BTreeMap<String, Value>| map.into_iter().collect()),
        ]
    })
}

/// A top-level sampling key that is not adapter-owned.
fn arb_free_key() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,15}".prop_filter("a reserved key", |key| {
        !OpenAiCompletions::RESERVED_SAMPLING_KEYS.contains(&key.as_str())
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Generated reserved keys, with arbitrary values, are always refused
    /// in-band naming the key — and the server records nothing across the
    /// whole run.
    #[test]
    fn property_reserved_keys_never_reach_the_wire(
        key in prop::sample::select(OpenAiCompletions::RESERVED_SAMPLING_KEYS),
        value in arb_json(),
    ) {
        let (runtime, server, provider) = wire_harness();
        let options = sampling([(key, value)]);
        let message = runtime.block_on(
            provider
                .stream(&model(&server), &Context::new().user("hi"), &options)
                .finish(),
        );

        prop_assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
        let detail = message.error_message.unwrap_or_default();
        prop_assert!(detail.contains(key), "{detail} should name `{key}`");
        let requests = runtime
            .block_on(server.received_requests())
            .unwrap_or_default();
        prop_assert!(
            requests.is_empty(),
            "reserved key `{key}` reached the wire: {requests:?}"
        );
    }

    /// Generated non-reserved keys, with arbitrary JSON values, reach the
    /// wire verbatim alongside the adapter's own fields.
    #[test]
    fn property_free_keys_reach_the_wire_verbatim(
        key in arb_free_key(),
        value in arb_json(),
    ) {
        let (runtime, server, provider) = wire_harness();
        let options = sampling([(key.as_str(), value.clone())]);
        let message = runtime.block_on(
            provider
                .stream(&model(&server), &Context::new().user("hi"), &options)
                .finish(),
        );

        prop_assert!(
            message.error_kind.is_none(),
            "a free key must not be refused: {:?}",
            message.error_message
        );
        let requests = runtime
            .block_on(server.received_requests())
            .unwrap_or_default();
        prop_assert_eq!(requests.len(), 1, "exactly one request should be sent");
        let body = sent_body(&requests[0]);
        prop_assert_eq!(&body[&key], &value, "the value must arrive verbatim");
        // The adapter's own fields are never displaced.
        prop_assert_eq!(&body["model"], &json!("test-model"));
        prop_assert_eq!(&body["stream"], &json!(true));
    }
}
