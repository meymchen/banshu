//! Anthropic-compatible temperature support on the wire (issue #95).
//!
//! An explicit temperature is either sent faithfully or refused before any
//! HTTP request — never silently dropped to make a request succeed. The
//! provider's declared [`AnthropicTemperature`] decides: `Unsupported` — the
//! default — refuses every explicit temperature, `WithoutReasoning` sends it
//! except alongside an enabled reasoning request, and `WithReasoning` sends
//! it alongside every reasoning shape the provider declares. An omitted
//! temperature leaves the request shape untouched.

use std::sync::{Arc, Mutex};

use banshu_ai::{
    AnthropicCompat, AnthropicReasoningFormat, AnthropicTemperature, BeforeSendObservation,
    CapabilitySupport, Context, ErrorKind, InMemoryCredentialStore, MiniMaxRegion, Model, Provider,
    ReasoningCapability, ReasoningEffort, ReasoningOptions, RequestObserver, StreamOptions,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const STOP_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// The temperature every test asks for, pinned as a non-power-of-two so a
/// rewrite cannot hide behind a value that survives rounding.
const TEMPERATURE: f32 = 0.7;

/// The exact JSON number [`TEMPERATURE`] becomes on the wire. JSON has no
/// f32, so the caller's value ships as its exact f64 expansion — the tests
/// pin *that*, proving the adapter neither rounds nor rewrites it.
fn wire_temperature() -> Value {
    json!(f64::from(TEMPERATURE))
}

/// Every reasoning request shape an Anthropic-compatible provider can
/// declare, each paired with the `thinking` value an enabled `high` request
/// puts on the wire. The budget shape's value is per-request, so it is
/// asserted separately.
const REASONING_SHAPES: [(&str, AnthropicReasoningFormat); 3] = [
    ("toggle", AnthropicReasoningFormat::ThinkingToggle),
    ("adaptive", AnthropicReasoningFormat::ThinkingAdaptive),
    ("budget", AnthropicReasoningFormat::ThinkingBudget),
];

/// An output cap with room for any derived budget.
const MAX_TOKENS: u32 = 128_000;

/// The budget the budget shape derives for `high` under [`MAX_TOKENS`].
const DERIVED_HIGH_BUDGET: u32 = 16384;

async fn mock_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(STOP_BODY),
        )
        .mount(&server)
        .await;
    server
}

/// A custom provider declaring `format` as its reasoning shape and
/// `temperature` as its temperature policy — the two declarations the tests
/// combine freely.
fn custom(
    server: &MockServer,
    format: AnthropicReasoningFormat,
    temperature: AnthropicTemperature,
) -> Provider {
    Provider::anthropic_compatible("custom", "Custom", server.uri(), ["TEST_API_KEY"])
        .with_anthropic_compat(AnthropicCompat {
            reasoning_format: format,
            temperature,
            ..AnthropicCompat::default()
        })
}

/// A model to stream against: attests the baseline ladder and, where the
/// provider's shape carries one, a token budget. `max_tokens` stays zero, so
/// the request's own cap decides.
fn model(provider: &Provider, server: &MockServer) -> Model {
    let budget = if provider
        .anthropic_compat()
        .reasoning_format
        .accepts_token_budget()
    {
        CapabilitySupport::Supported
    } else {
        CapabilitySupport::Unsupported
    };
    let mut model = Model::anthropic_messages("custom-thinker").with_base_url(server.uri());
    model.provider = provider.id().to_string();
    model.reasoning = ReasoningCapability::baseline().with_token_budget(budget);
    model
}

fn options(temperature: Option<f32>, reasoning: Option<ReasoningOptions>) -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        max_tokens: Some(MAX_TOKENS),
        temperature,
        reasoning,
        ..Default::default()
    }
}

async fn request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .expect("request journal")
        .iter()
        .map(|request| serde_json::from_slice(&request.body).expect("JSON request"))
        .collect()
}

/// Stream one request against `provider` and return the single request body
/// it put on the wire.
async fn sent_body(provider: &Provider, server: &MockServer, options: StreamOptions) -> Value {
    let message = provider
        .stream(
            &model(provider, server),
            &Context::new().user("hi"),
            &options,
        )
        .finish()
        .await;
    assert_eq!(
        message.error_kind, None,
        "this request should be honoured: {:?}",
        message.error_message
    );
    let mut bodies = request_bodies(server).await;
    assert_eq!(bodies.len(), 1, "exactly one request reached the server");
    bodies.remove(0)
}

/// Stream `options` against `provider`, assert it terminates as
/// `InvalidRequest`, and assert the mock server never saw it. Returns the
/// rejection detail.
async fn rejected(provider: &Provider, server: &MockServer, options: StreamOptions) -> String {
    let message = provider
        .stream(
            &model(provider, server),
            &Context::new().user("hi"),
            &options,
        )
        .finish()
        .await;
    assert_eq!(
        message.error_kind,
        Some(ErrorKind::InvalidRequest),
        "this request should be refused in-band"
    );
    assert!(
        request_bodies(server).await.is_empty(),
        "the refusal happens before any HTTP request"
    );
    message.error_message.unwrap_or_default()
}

fn enabled() -> ReasoningOptions {
    ReasoningOptions::new(ReasoningEffort::High)
}

// ---------------------------------------------------------------------------
// Temperature without reasoning
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_supported_temperature_is_sent_exactly_when_reasoning_is_absent() {
    for policy in [
        AnthropicTemperature::WithoutReasoning,
        AnthropicTemperature::WithReasoning,
    ] {
        for (id, format) in REASONING_SHAPES {
            let server = mock_server().await;
            let provider = custom(&server, format, policy);
            let body = sent_body(&provider, &server, options(Some(TEMPERATURE), None)).await;
            assert_eq!(
                body["temperature"],
                wire_temperature(),
                "{id}/{policy:?}: the requested value ships verbatim"
            );
            assert!(
                body.get("thinking").is_none(),
                "{id}/{policy:?}: no reasoning was requested: {body}"
            );
        }
    }
}

#[tokio::test]
async fn the_wire_bytes_carry_the_requested_value_not_a_rewrite() {
    let server = mock_server().await;
    let provider = custom(
        &server,
        AnthropicReasoningFormat::ThinkingToggle,
        AnthropicTemperature::WithReasoning,
    );
    sent_body(&provider, &server, options(Some(TEMPERATURE), None)).await;

    let requests = server.received_requests().await.expect("request journal");
    let raw = String::from_utf8(requests[0].body.clone()).expect("UTF-8 body");
    let field = format!(
        "\"temperature\":{}",
        serde_json::to_string(&wire_temperature()).expect("a JSON number")
    );
    assert!(
        raw.contains(&field),
        "the wire carries the caller's f32 expanded exactly, not a rewrite: {raw}"
    );
}

// ---------------------------------------------------------------------------
// An endpoint rejecting temperature entirely
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_undeclared_temperature_is_refused_before_any_request() {
    // With and without a reasoning request alongside: the refusal is about
    // the temperature, and the reasoning request alone would have passed.
    for (id, format) in REASONING_SHAPES {
        for reasoning in [None, Some(enabled())] {
            let server = mock_server().await;
            let provider = custom(&server, format, AnthropicTemperature::Unsupported);
            let detail = rejected(&provider, &server, options(Some(TEMPERATURE), reasoning)).await;
            assert!(
                detail.contains("declares no temperature support"),
                "{id}: {detail}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Temperature × each reasoning shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn temperature_and_reasoning_are_emitted_together_only_when_declared() {
    for (id, format) in REASONING_SHAPES {
        // Declared as coexisting: both fields ship, each exactly as asked.
        let server = mock_server().await;
        let provider = custom(&server, format, AnthropicTemperature::WithReasoning);
        let body = sent_body(
            &provider,
            &server,
            options(Some(TEMPERATURE), Some(enabled())),
        )
        .await;
        assert_eq!(body["temperature"], wire_temperature(), "{id}: {body}");
        let thinking = match format {
            AnthropicReasoningFormat::ThinkingBudget => {
                json!({ "type": "enabled", "budget_tokens": DERIVED_HIGH_BUDGET })
            }
            AnthropicReasoningFormat::ThinkingAdaptive => json!({ "type": "adaptive" }),
            _ => json!({ "type": "enabled" }),
        };
        assert_eq!(
            body["thinking"], thinking,
            "{id}: the reasoning shape ships too"
        );

        // Declared without the combination: refused before HTTP, and the
        // refusal names the conflict.
        let server = mock_server().await;
        let provider = custom(&server, format, AnthropicTemperature::WithoutReasoning);
        let detail = rejected(
            &provider,
            &server,
            options(Some(TEMPERATURE), Some(enabled())),
        )
        .await;
        assert!(
            detail.contains("only without an enabled reasoning request"),
            "{id}: {detail}"
        );
    }
}

#[tokio::test]
async fn an_undeclared_reasoning_shape_refuses_reasoning_regardless_of_temperature() {
    // The fourth shape: no reasoning field at all. An enabled request is
    // refused by the reasoning preflight whether or not a temperature rides
    // along — the temperature declaration never masks that refusal — while a
    // temperature alone against a provider declaring it still ships.
    let server = mock_server().await;
    let provider = custom(
        &server,
        AnthropicReasoningFormat::Unsupported,
        AnthropicTemperature::WithReasoning,
    );
    let detail = rejected(
        &provider,
        &server,
        options(Some(TEMPERATURE), Some(enabled())),
    )
    .await;
    assert!(
        detail.contains("no reasoning request format"),
        "the reasoning refusal, not the temperature policy, answers: {detail}"
    );

    let server = mock_server().await;
    let provider = custom(
        &server,
        AnthropicReasoningFormat::Unsupported,
        AnthropicTemperature::WithReasoning,
    );
    let body = sent_body(&provider, &server, options(Some(TEMPERATURE), None)).await;
    assert_eq!(body["temperature"], wire_temperature());
    assert!(body.get("thinking").is_none(), "{body}");
}

#[tokio::test]
async fn a_disabled_reasoning_request_is_no_combination() {
    // `Off` disables reasoning outright, so temperature applies normally even
    // under a policy that forbids the enabled pairing.
    for (id, format) in REASONING_SHAPES {
        let server = mock_server().await;
        let provider = custom(&server, format, AnthropicTemperature::WithoutReasoning);
        let body = sent_body(
            &provider,
            &server,
            options(
                Some(TEMPERATURE),
                Some(ReasoningOptions::new(ReasoningEffort::Off)),
            ),
        )
        .await;
        assert_eq!(body["temperature"], wire_temperature(), "{id}: {body}");
        assert_eq!(
            body["thinking"],
            json!({ "type": "disabled" }),
            "{id}: the disabling toggle still ships"
        );
    }
}

// ---------------------------------------------------------------------------
// Omitting temperature
// ---------------------------------------------------------------------------

#[tokio::test]
async fn omitting_temperature_keeps_the_request_shape_untouched() {
    // The payload carries exactly the keys it carried before the declaration
    // existed, for every policy and reasoning combination.
    const BASE_KEYS: [&str; 4] = ["max_tokens", "messages", "model", "stream"];

    for policy in [
        AnthropicTemperature::Unsupported,
        AnthropicTemperature::WithoutReasoning,
        AnthropicTemperature::WithReasoning,
    ] {
        for (id, format) in REASONING_SHAPES {
            for reasoning in [None, Some(enabled())] {
                let server = mock_server().await;
                let provider = custom(&server, format, policy);
                let body = sent_body(&provider, &server, options(None, reasoning.clone())).await;
                let mut keys: Vec<&str> = body
                    .as_object()
                    .expect("a JSON object")
                    .keys()
                    .map(String::as_str)
                    .collect();
                keys.sort_unstable();
                let mut expected = BASE_KEYS.to_vec();
                if reasoning.is_some() {
                    expected.push("thinking");
                }
                expected.sort_unstable();
                assert_eq!(
                    keys,
                    expected,
                    "{id}/{policy:?}/reasoning={}: the payload shape moved: {body}",
                    reasoning.is_some()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The observer sees what the server received
// ---------------------------------------------------------------------------

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
async fn the_observed_payload_matches_what_the_server_received() {
    // The budget shape matters most here: its `budget_tokens` is derived at
    // body-build time, the place observer/wire drift would appear.
    for format in [
        AnthropicReasoningFormat::ThinkingToggle,
        AnthropicReasoningFormat::ThinkingBudget,
    ] {
        let server = mock_server().await;
        let provider = custom(&server, format, AnthropicTemperature::WithReasoning);
        let observer = Arc::new(WireObserver::default());
        let options = StreamOptions {
            observer: Some(observer.clone()),
            ..options(Some(TEMPERATURE), Some(enabled()))
        };
        sent_body(&provider, &server, options).await;

        let bodies = request_bodies(&server).await;
        let payloads = observer.payloads.lock().unwrap();
        let [payload] = payloads.as_slice() else {
            panic!("expected exactly one before-send observation");
        };
        assert_eq!(
            *payload, bodies[0],
            "{format:?}: the observer sees the exact body the server recorded, \
             temperature included"
        );
        assert_eq!(payload["temperature"], wire_temperature());
    }
}

// ---------------------------------------------------------------------------
// The bundled providers' own declarations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn minimax_sends_temperature_alongside_adaptive_thinking() {
    // MiniMax's Anthropic-compatible reference marks temperature fully
    // supported and names no thinking restriction, so the bundled provider
    // declares the combination.
    let server = mock_server().await;
    let provider = Provider::minimax(
        MiniMaxRegion::Global,
        Arc::new(InMemoryCredentialStore::new()),
    );
    let model = provider
        .models()
        .iter()
        .find(|model| model.reasoning.reasons())
        .expect("a MiniMax model attesting reasoning")
        .clone()
        .with_base_url(server.uri());

    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(TEMPERATURE), Some(enabled())),
        )
        .finish()
        .await;
    assert_eq!(
        message.error_kind, None,
        "MiniMax honours the declared combination: {:?}",
        message.error_message
    );

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies[0]["temperature"], wire_temperature());
    assert_eq!(bodies[0]["thinking"], json!({ "type": "adaptive" }));
}

#[tokio::test]
async fn kimi_refuses_an_explicit_temperature_before_any_request() {
    // Kimi publishes no parameter-level reference for the coding endpoint's
    // Anthropic shape, so the bundled provider declares no temperature
    // support — an explicit one is refused rather than sent on a guess.
    let server = mock_server().await;
    let provider = Provider::kimi(Arc::new(InMemoryCredentialStore::new()));
    let mut model = Model::anthropic_messages("k2p5").with_base_url(server.uri());
    model.provider = "kimi".to_string();

    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(TEMPERATURE), None),
        )
        .finish()
        .await;
    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    assert!(
        message
            .error_message
            .unwrap_or_default()
            .contains("declares no temperature support"),
    );
    assert!(
        request_bodies(&server).await.is_empty(),
        "the refusal happens before any HTTP request"
    );
}
