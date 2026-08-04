//! The unified reasoning options on the Anthropic-compatible wire (issue #44).
//!
//! Issue #42 built the contract and issue #43 put it on the OpenAI-compatible
//! wire. This file pins the other protocol: for every `thinking` request shape
//! banshu's Anthropic-compatible targets document, what an *enabled* request, a
//! *disabled* request, and an *unsupported* request actually do.
//!
//! The rules are the ones #43 established, plus one this protocol adds:
//!
//! - `Off` is a request, not a silence. Every declared shape sends
//!   `thinking: { "type": "disabled" }`; omitting the field would leave a
//!   thinking model thinking.
//! - A supported request carries that shape's fields and nothing else — a
//!   toggle-only endpoint never sees a `budget_tokens`.
//! - An unsupported request never becomes HTTP traffic.
//! - A reasoning budget shares the output cap with the answer, so a budget that
//!   does not fit under the request's final `max_tokens` is refused before
//!   dispatch rather than 400'd by the endpoint.

use banshu_ai::{
    AnthropicCompat, AnthropicReasoningFormat, AssistantContent, AssistantMessage,
    CapabilitySupport, Context, ErrorKind, Message, Model, Provider, ReasoningCapability,
    ReasoningEffort, ReasoningOptions, StreamOptions, ThinkingContent,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const STOP_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// Every reasoning request field any Anthropic-compatible endpoint in scope
/// could carry. A request declares exactly the subset its provider's shape
/// names; each test asserts the rest are absent.
///
/// `thinking` is asserted by whole-value equality, so a stray `budget_tokens`
/// or `effort` *inside* it fails too.
const ALL_REASONING_FIELDS: [&str; 4] =
    ["thinking", "output_config", "reasoning_effort", "reasoning"];

/// The output cap banshu falls back to when neither the request nor the model
/// names one. Mirrors `DEFAULT_MAX_TOKENS` in the adapter.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// The smallest budget Anthropic's budget shape documents, which is also the
/// room banshu keeps for the answer itself when it derives a budget.
const MIN_THINKING_BUDGET: u32 = 1024;

/// The budget the budget shape derives for each level when the caller names
/// none, given an output cap with room to spare. Written down here so a change
/// to the ladder has to be a deliberate edit in two places.
const DERIVED_BUDGETS: [(ReasoningEffort, u32); 4] = [
    (ReasoningEffort::Minimal, 1024),
    (ReasoningEffort::Low, 2048),
    (ReasoningEffort::Medium, 8192),
    (ReasoningEffort::High, 16384),
];

/// The four Anthropic-compatible shapes under test and a provider declaring
/// each: the two vendors banshu bundles, plus a custom provider for the two
/// shapes no vendor declares.
const TARGETS: [(&str, AnthropicReasoningFormat); 4] = [
    ("kimi", AnthropicReasoningFormat::ThinkingToggle),
    ("minimax", AnthropicReasoningFormat::ThinkingAdaptive),
    ("budget", AnthropicReasoningFormat::ThinkingBudget),
    ("silent", AnthropicReasoningFormat::Unsupported),
];

fn provider(id: &str, server: &MockServer) -> Provider {
    match id {
        "kimi" => Provider::kimi(),
        "minimax" => Provider::minimax(),
        // Anthropic's own budget shape, which no bundled vendor declares: a
        // caller pointing `anthropic_compatible` at an endpoint that documents
        // `budget_tokens` declares it themselves.
        "budget" => custom(id, AnthropicReasoningFormat::ThinkingBudget, server),
        "silent" => custom(id, AnthropicReasoningFormat::Unsupported, server),
        other => panic!("`{other}` is not an Anthropic-compatible provider under test"),
    }
}

fn custom(id: &str, format: AnthropicReasoningFormat, server: &MockServer) -> Provider {
    Provider::anthropic_compatible(id, id, server.uri(), ["TEST_API_KEY"]).with_anthropic_compat(
        AnthropicCompat {
            reasoning_format: format,
            ..AnthropicCompat::default()
        },
    )
}

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

fn options(reasoning: Option<ReasoningOptions>) -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        reasoning,
        ..Default::default()
    }
}

/// A model of `provider` to stream against, pointed at `server`: a catalog
/// model that attests a reasoning level, else — for a custom provider banshu
/// bundles no catalog for — a hand-built model attesting the baseline ladder
/// and, where the provider's shape carries one, a token budget.
///
/// Every hand-built model leaves `max_tokens` at zero, so the request's own
/// cap decides; the budget tests set one explicitly.
fn reasoning_model(provider: &Provider, server: &MockServer) -> Model {
    if let Some(model) = provider
        .models()
        .iter()
        .find(|model| model.reasoning.reasons())
    {
        return model.clone().with_base_url(server.uri());
    }
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

async fn request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .expect("request journal")
        .iter()
        .map(|request| serde_json::from_slice(&request.body).expect("JSON request"))
        .collect()
}

/// Stream one request against `provider_id`'s reasoning model and return the
/// single request body it put on the wire.
async fn sent_body(provider_id: &str, options: StreamOptions) -> Value {
    let server = mock_server().await;
    let provider = provider(provider_id, &server);
    let model = reasoning_model(&provider, &server);
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;
    assert_eq!(
        message.error_kind, None,
        "`{provider_id}` should honour this request: {:?}",
        message.error_message
    );

    let mut bodies = request_bodies(&server).await;
    assert_eq!(
        bodies.len(),
        1,
        "exactly one request reached `{provider_id}`"
    );
    bodies.remove(0)
}

/// [`sent_body`] for a plain effort request.
async fn sent_for(provider_id: &str, reasoning: Option<ReasoningOptions>) -> Value {
    sent_body(provider_id, options(reasoning)).await
}

/// Stream `options` against `provider_id`'s reasoning model, assert it
/// terminates as `InvalidRequest`, and assert the mock server never saw it.
/// Returns the rejection detail.
async fn rejected(provider_id: &str, options: StreamOptions) -> String {
    let server = mock_server().await;
    let provider = provider(provider_id, &server);
    let model = reasoning_model(&provider, &server);
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;

    assert_eq!(
        message.error_kind,
        Some(ErrorKind::InvalidRequest),
        "`{provider_id}` should refuse this request in-band"
    );
    assert!(
        request_bodies(&server).await.is_empty(),
        "`{provider_id}` must refuse this request before any HTTP request"
    );
    message.error_message.unwrap_or_default()
}

/// Assert `body` carries exactly `expected` of the reasoning fields, with the
/// given values, and none of the others.
#[track_caller]
fn carries_only(body: &Value, expected: &[(&str, Value)]) {
    for (field, value) in expected {
        assert_eq!(
            body.get(*field),
            Some(value),
            "`{field}` should be {value} in {body}"
        );
    }
    for field in ALL_REASONING_FIELDS {
        if expected.iter().any(|(named, _)| *named == field) {
            continue;
        }
        assert!(
            body.get(field).is_none(),
            "`{field}` belongs to another endpoint's shape, not this one: {body}"
        );
    }
}

fn disabled() -> Value {
    json!({ "type": "disabled" })
}

/// The levels `provider_id`'s reasoning model attests, above `Off`. Enabled
/// requests drive off this rather than a hardcoded ladder: what a provider
/// accepts is its own declaration, and the ladders themselves are pinned in
/// `reasoning_capabilities.rs`.
async fn attested_levels_above_off(provider_id: &str) -> Vec<ReasoningEffort> {
    let server = mock_server().await;
    let provider = provider(provider_id, &server);
    let levels: Vec<ReasoningEffort> = reasoning_model(&provider, &server)
        .reasoning
        .efforts()
        .iter()
        .copied()
        .filter(|effort| *effort > ReasoningEffort::Off)
        .collect();
    assert!(
        !levels.is_empty(),
        "`{provider_id}` should attest a requestable level"
    );
    levels
}

// ---------------------------------------------------------------------------
// The bare `thinking` toggle — Kimi For Coding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_thinking_toggle_shape_enables_with_the_toggle_alone() {
    // Kimi's reference switches thinking with `thinking.type` and documents no
    // budget and no effort field, so every attested level above `Off` reads as
    // "enabled" and nothing rides along.
    for effort in attested_levels_above_off("kimi").await {
        let body = sent_for("kimi", Some(ReasoningOptions::new(effort))).await;
        carries_only(&body, &[("thinking", json!({ "type": "enabled" }))]);
    }
}

#[tokio::test]
async fn the_thinking_toggle_shape_disables_with_the_documented_off_value() {
    let body = sent_for("kimi", Some(ReasoningOptions::new(ReasoningEffort::Off))).await;
    carries_only(&body, &[("thinking", disabled())]);
}

// ---------------------------------------------------------------------------
// The adaptive shape — MiniMax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_adaptive_shape_enables_with_adaptive_and_disables_with_disabled() {
    // MiniMax's Anthropic-compatible reference enables thinking with
    // `adaptive` — the model decides how much to think — and documents
    // `disabled` as the off value. Neither carries a budget or an effort.
    for effort in attested_levels_above_off("minimax").await {
        let body = sent_for("minimax", Some(ReasoningOptions::new(effort))).await;
        carries_only(&body, &[("thinking", json!({ "type": "adaptive" }))]);
    }

    let body = sent_for("minimax", Some(ReasoningOptions::new(ReasoningEffort::Off))).await;
    carries_only(&body, &[("thinking", disabled())]);
}

// ---------------------------------------------------------------------------
// The budget shape — Anthropic's own, for a caller who declares it
// ---------------------------------------------------------------------------

fn capped(reasoning: ReasoningOptions, max_tokens: u32) -> StreamOptions {
    StreamOptions {
        max_tokens: Some(max_tokens),
        ..options(Some(reasoning))
    }
}

#[tokio::test]
async fn the_budget_shape_sends_a_configured_budget_verbatim() {
    let body = sent_body(
        "budget",
        capped(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096),
            32_000,
        ),
    )
    .await;
    carries_only(
        &body,
        &[(
            "thinking",
            json!({ "type": "enabled", "budget_tokens": 4096 }),
        )],
    );
}

#[tokio::test]
async fn the_budget_shape_derives_a_budget_from_the_effort_when_none_is_configured() {
    // This shape has no effort field: the level *is* the budget. The caller
    // asked for a level, not a token count, so banshu spends its own documented
    // ladder rather than refusing a request the shape can express.
    for (effort, expected) in DERIVED_BUDGETS {
        let body = sent_body("budget", capped(ReasoningOptions::new(effort), 128_000)).await;
        carries_only(
            &body,
            &[(
                "thinking",
                json!({ "type": "enabled", "budget_tokens": expected }),
            )],
        );
    }
}

#[tokio::test]
async fn a_derived_budget_is_trimmed_to_leave_room_for_the_answer() {
    // Trimming a budget banshu chose itself is not the clamp this crate
    // refuses: the caller asked for `high`, not for 16384 tokens. A budget the
    // caller *did* name is never trimmed — it is validated instead.
    let body = sent_body(
        "budget",
        capped(
            ReasoningOptions::new(ReasoningEffort::High),
            DEFAULT_MAX_TOKENS,
        ),
    )
    .await;
    carries_only(
        &body,
        &[(
            "thinking",
            json!({
                "type": "enabled",
                "budget_tokens": DEFAULT_MAX_TOKENS - MIN_THINKING_BUDGET,
            }),
        )],
    );
}

#[tokio::test]
async fn the_budget_shape_disables_with_the_toggle_and_no_budget() {
    // A disabled request has nothing to spend, so no budget rides along.
    let body = sent_for("budget", Some(ReasoningOptions::new(ReasoningEffort::Off))).await;
    carries_only(&body, &[("thinking", disabled())]);
}

// ---------------------------------------------------------------------------
// A budget has to fit under the output cap
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_configured_budget_at_or_above_max_tokens_is_refused_before_http() {
    // `max_tokens` caps thinking *and* the answer, so a budget that fills it
    // leaves nothing to answer with — the endpoint would reject it, and banshu
    // says so first, naming both numbers.
    for tokens in [4096, 8000] {
        let detail = rejected(
            "budget",
            capped(
                ReasoningOptions::new(ReasoningEffort::High).with_token_budget(tokens),
                4096,
            ),
        )
        .await;
        assert!(detail.contains(&tokens.to_string()), "{detail}");
        assert!(detail.contains("4096"), "{detail}");
    }

    // One token under the cap is a legal request, so the boundary is exact.
    let body = sent_body(
        "budget",
        capped(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4095),
            4096,
        ),
    )
    .await;
    carries_only(
        &body,
        &[(
            "thinking",
            json!({ "type": "enabled", "budget_tokens": 4095 }),
        )],
    );
}

#[tokio::test]
async fn a_configured_budget_below_the_documented_minimum_is_refused_before_http() {
    let detail = rejected(
        "budget",
        capped(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(512),
            32_000,
        ),
    )
    .await;
    assert!(detail.contains("512"), "{detail}");
    assert!(
        detail.contains(&MIN_THINKING_BUDGET.to_string()),
        "{detail}"
    );
}

#[tokio::test]
async fn an_output_cap_too_small_for_any_budget_is_refused_before_http() {
    // With no room for the minimum budget *and* an answer, there is no budget
    // to derive: the request is refused rather than sent with a budget the
    // shape's own reference calls illegal.
    let detail = rejected(
        "budget",
        capped(ReasoningOptions::new(ReasoningEffort::Low), 1500),
    )
    .await;
    assert!(detail.contains("1500"), "{detail}");

    // The same request under a workable cap goes through, so it is the cap
    // that was refused and not the level.
    let body = sent_body(
        "budget",
        capped(ReasoningOptions::new(ReasoningEffort::Low), 4096),
    )
    .await;
    assert!(body["thinking"]["budget_tokens"].is_number(), "{body}");
}

#[tokio::test]
async fn the_output_cap_a_budget_is_measured_against_is_the_one_that_ships() {
    // The adapter's `max_tokens` ladder is request cap, then model cap, then
    // the crate default — the preflight must measure against the same value,
    // or a budget could pass the check and 400 at the endpoint.
    let server = mock_server().await;
    let provider = provider("budget", &server);
    let mut model = reasoning_model(&provider, &server);
    model.max_tokens = 8192;

    let sent = |reasoning: ReasoningOptions, max_tokens: Option<u32>| {
        let options = StreamOptions {
            max_tokens,
            ..options(Some(reasoning))
        };
        let model = model.clone();
        let provider = &provider;
        async move {
            provider
                .stream(&model, &Context::new().user("hi"), &options)
                .finish()
                .await
        }
    };

    // The model cap decides when the request names none: 8191 fits, 8192 does
    // not — and the crate default of 4096 would have judged both the same way.
    assert_eq!(
        sent(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(8191),
            None
        )
        .await
        .error_kind,
        None,
    );
    assert_eq!(
        sent(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(8192),
            None
        )
        .await
        .error_kind,
        Some(ErrorKind::InvalidRequest),
    );

    // A request cap overrides the model's, in both directions.
    assert_eq!(
        sent(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(8191),
            Some(4096)
        )
        .await
        .error_kind,
        Some(ErrorKind::InvalidRequest),
    );
    assert_eq!(
        sent(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(20_000),
            Some(32_000)
        )
        .await
        .error_kind,
        None,
    );

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies.len(), 2, "only the two legal requests were sent");
    assert_eq!(bodies[0]["max_tokens"], json!(8192));
    assert_eq!(bodies[1]["max_tokens"], json!(32_000));
}

#[tokio::test]
async fn the_budget_shape_refuses_a_model_that_attests_no_budget_at_all() {
    // This shape enables reasoning *by* spending a budget, so a model that
    // attests none cannot make an enabled request — not even one that leaves
    // the number to banshu, which would otherwise send a `budget_tokens` the
    // model said it does not take.
    let server = mock_server().await;
    let provider = provider("budget", &server);
    let mut model = reasoning_model(&provider, &server);
    model.reasoning = ReasoningCapability::baseline();

    let refused = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::High))),
        )
        .finish()
        .await;
    assert_eq!(refused.error_kind, Some(ErrorKind::InvalidRequest));
    assert!(
        refused
            .error_message
            .unwrap_or_default()
            .contains("does not support a reasoning token budget")
    );
    assert!(request_bodies(&server).await.is_empty());

    // `Off` still works: it spends nothing.
    let disabled_request = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::Off))),
        )
        .finish()
        .await;
    assert_eq!(disabled_request.error_kind, None);
    assert_eq!(
        request_bodies(&server).await.remove(0)["thinking"],
        disabled()
    );
}

#[tokio::test]
async fn a_budget_alongside_off_is_refused_rather_than_dropped() {
    // `Off` sends the disabling toggle and nothing else, so a budget requested
    // with it could only be silently discarded.
    let detail = rejected(
        "budget",
        capped(
            ReasoningOptions::new(ReasoningEffort::Off).with_token_budget(2048),
            32_000,
        ),
    )
    .await;
    assert!(detail.contains("off"), "{detail}");
}

#[tokio::test]
async fn only_the_budget_shape_carries_a_budget() {
    // The toggle and adaptive shapes have no budget field, so a budget is
    // refused there even at a level they would otherwise honour.
    for (id, format) in TARGETS {
        assert_eq!(
            format.accepts_token_budget(),
            id == "budget",
            "`{id}` budget support follows its declared shape"
        );
        if format.accepts_token_budget() {
            continue;
        }
        rejected(
            id,
            options(Some(
                ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096),
            )),
        )
        .await;
    }
}

// ---------------------------------------------------------------------------
// No declared shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_provider_with_no_declared_shape_sends_nothing_and_refuses_a_request() {
    let body = sent_for("silent", None).await;
    carries_only(&body, &[]);

    for effort in ReasoningEffort::ALL {
        let detail = rejected("silent", options(Some(ReasoningOptions::new(effort)))).await;
        assert!(detail.contains("no reasoning request format"), "{detail}");
    }
}

#[tokio::test]
async fn every_shape_refuses_exactly_the_levels_its_provider_does_not_attest() {
    for (id, _) in TARGETS {
        if id == "silent" {
            continue;
        }
        let attested = attested_levels_above_off(id).await;
        for effort in ReasoningEffort::ALL {
            if effort == ReasoningEffort::Off || attested.contains(&effort) {
                continue;
            }
            rejected(id, options(Some(ReasoningOptions::new(effort)))).await;
        }
        assert!(
            attested.len() < ReasoningEffort::ALL.len() - 1,
            "`{id}` should refuse at least one level, or the loop above is vacuous"
        );
    }
}

// ---------------------------------------------------------------------------
// What a reasoning request must not disturb
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_reasoning_option_leaves_every_shape_untouched() {
    // Not just "no reasoning field": the payload carries exactly the keys it
    // carried before reasoning options existed, so a shape cannot grow one
    // quietly under some other name.
    const UNCHANGED_KEYS: [&str; 4] = ["model", "max_tokens", "messages", "stream"];

    for (id, _) in TARGETS {
        let body = sent_for(id, None).await;
        carries_only(&body, &[]);
        let mut keys: Vec<&str> = body
            .as_object()
            .expect("a JSON object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        let mut expected = UNCHANGED_KEYS;
        expected.sort_unstable();
        assert_eq!(keys, expected, "`{id}` payload shape moved: {body}");
    }
}

/// A stream that thinks, signs its thinking, and answers.
const THINKING_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me think. \"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"2 + 2 is 4.\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-abc\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"The answer is 4.\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":100}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

async fn thinking_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(THINKING_BODY),
        )
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn an_enabled_request_does_not_disturb_thinking_output_or_its_signature() {
    // The request side changed; the response side must not. Thinking still
    // streams into its own block ahead of the text, carrying the opaque
    // signature the endpoint signed it with.
    let server = thinking_server().await;
    let provider = provider("kimi", &server);
    let model = reasoning_model(&provider, &server);
    let message = provider
        .stream(
            &model,
            &Context::new().user("What is 2 + 2?"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::High))),
        )
        .finish()
        .await;

    let thinking = match message.content.first() {
        Some(AssistantContent::Thinking(block)) => block,
        other => panic!("expected thinking first, got {other:?}"),
    };
    assert_eq!(thinking.thinking, "Let me think. 2 + 2 is 4.");
    assert_eq!(thinking.signature.as_deref(), Some("sig-abc"));
    assert!(!thinking.redacted);
    assert_eq!(message.text(), "The answer is 4.");
}

#[tokio::test]
async fn an_enabled_request_replays_signed_thinking_verbatim() {
    // The request-level `thinking` field and the replayed `thinking` content
    // blocks are different things; asking for reasoning must not disturb the
    // history (issue #40 provenance rules still decide what survives).
    let server = mock_server().await;
    let provider = provider("kimi", &server);
    let model = reasoning_model(&provider, &server);

    let mut history = AssistantMessage::from_content(vec![
        AssistantContent::Thinking(ThinkingContent {
            thinking: "2 + 2 is 4.".into(),
            signature: Some("sig-abc".into()),
            redacted: false,
        }),
        AssistantContent::Text(banshu_ai::TextContent {
            text: "4".into(),
            signature: None,
        }),
    ]);
    history.api = "anthropic-messages".into();
    history.provider = model.provider.clone();
    history.model = model.id.clone();

    let context = Context::new()
        .user("2 + 2?")
        .with_message(Message::Assistant(Box::new(history)))
        .user("And 3 + 3?");

    let message = provider
        .stream(
            &model,
            &context,
            &options(Some(ReasoningOptions::new(ReasoningEffort::High))),
        )
        .finish()
        .await;
    assert_eq!(message.error_kind, None);

    let body = request_bodies(&server).await.remove(0);
    assert_eq!(body["thinking"], json!({ "type": "enabled" }));
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({
            "type": "thinking",
            "thinking": "2 + 2 is 4.",
            "signature": "sig-abc",
        }),
    );
}

#[tokio::test]
async fn thinking_tokens_stay_inside_the_output_count() {
    // Anthropic reports no separate reasoning-token bucket: thinking is billed
    // inside `output_tokens`. banshu attests what the wire says and no more, so
    // `usage.reasoning` stays `None` rather than guessing a split.
    let server = thinking_server().await;
    let provider = provider("kimi", &server);
    let model = reasoning_model(&provider, &server);
    let usage = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::High))),
        )
        .finish()
        .await
        .usage;

    assert_eq!(usage.reasoning, None);
    assert_eq!(usage.output, 100);
    assert_eq!(
        usage.total_tokens,
        usage.input + usage.cache_read + usage.cache_write + usage.output,
        "the total counts output once and thinking never on its own"
    );
}
