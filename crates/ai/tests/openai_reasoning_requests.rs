//! The unified reasoning options on the OpenAI-compatible wire (issue #43).
//!
//! Issue #42 built the contract — what a request may ask for, what a model
//! attests, what a provider declares — and stopped short of the wire. This
//! file pins the wire: for every reasoning request shape the four
//! OpenAI-compatible target providers document, what an *enabled* request, a
//! *disabled* request, and an *unsupported* request actually do.
//!
//! Three rules run through all of it:
//!
//! - `Off` is a request, not a silence. Every declared shape sends the value
//!   its endpoint documents for "do not reason"; omitting the field would
//!   leave a thinking model thinking.
//! - A supported request carries that shape's fields and nothing else. No
//!   compatibility field belonging to some other endpoint rides along.
//! - An unsupported request never becomes HTTP traffic — the preflight refuses
//!   it, and the mock server's journal proves it.

use std::sync::Arc;

use banshu_ai::api::openai_completions::OpenAiCompletions;
use banshu_ai::{
    AssistantContent, CapabilitySupport, Context, ErrorKind, Model, OpenAiChatTemplateKwargs,
    OpenAiCompat, OpenAiReasoningBudgetField, OpenAiReasoningFormat, Provider, ReasoningCapability,
    ReasoningEffort, ReasoningOptions, StreamOptions,
};
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const STOP_BODY: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

/// Every reasoning request field any OpenAI-compatible endpoint in scope could
/// carry. A request declares exactly the subset its provider's shape names;
/// each test asserts the rest are absent.
///
/// Kept in the same order as `REASONING_WIRE_FIELDS` in
/// `reasoning_capabilities.rs`, which pins the same list against the
/// no-reasoning payload: an integration test is its own crate, so the two
/// cannot share a constant, and matching order makes any drift visible.
const ALL_REASONING_FIELDS: [&str; 5] = [
    "reasoning_effort",
    "reasoning",
    "thinking",
    "enable_thinking",
    "chat_template_kwargs",
];

/// The four OpenAI-compatible targets and the request shape each declares.
const TARGETS: [(&str, OpenAiReasoningFormat); 4] = [
    ("deepseek", OpenAiReasoningFormat::ThinkingToggle),
    ("xiaomi", OpenAiReasoningFormat::ThinkingToggleOnly),
    ("zai", OpenAiReasoningFormat::ThinkingToggleOnly),
    ("moonshot", OpenAiReasoningFormat::Unsupported),
];

fn provider(id: &str) -> Provider {
    match id {
        "deepseek" => Provider::deepseek(),
        "xiaomi" => Provider::xiaomi(),
        "zai" => Provider::zai(),
        "moonshot" => Provider::moonshot(),
        "openai" => Provider::openai(),
        other => panic!("`{other}` is not an OpenAI-compatible provider under test"),
    }
}

async fn mock_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

/// A model of `provider` to stream against, pointed at `server`: one that
/// attests a reasoning level, else the provider's first model, else — for a
/// provider banshu bundles no catalog for at all — a hand-built model
/// attesting the baseline ladder, which is what a caller of such a provider
/// supplies themselves.
///
/// The middle case is Moonshot, whose endpoint takes no reasoning field and
/// whose models therefore attest no level; the last is `openai`.
fn reasoning_model(provider: &Provider, server: &MockServer) -> Model {
    let models = provider.models();
    if let Some(model) = models
        .iter()
        .find(|model| model.reasoning.reasons())
        .or_else(|| models.first())
    {
        return model.clone().with_base_url(server.uri());
    }
    let mut model = Model::openai_completions("custom-reasoner").with_base_url(server.uri());
    model.provider = provider.id().to_string();
    model.reasoning = ReasoningCapability::baseline();
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

/// Stream one reasoning request against `provider_id`'s reasoning model and
/// return the single request body it put on the wire.
async fn sent_body(provider_id: &str, reasoning: Option<ReasoningOptions>) -> Value {
    let server = mock_server().await;
    let provider = provider(provider_id);
    let model = reasoning_model(&provider, &server);
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options(reasoning))
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

fn enabled() -> Value {
    serde_json::json!({ "type": "enabled" })
}

fn disabled() -> Value {
    serde_json::json!({ "type": "disabled" })
}

/// The levels `provider_id`'s models actually attest, above `Off`. Every
/// enabled-request test drives off this rather than a hardcoded ladder: what a
/// provider accepts is its own declaration, so a test that hardcoded the list
/// would just re-assert the bug the declaration exists to prevent.
///
/// The exact ladders are pinned independently — against hand-written constants
/// — in `reasoning_capabilities.rs`; this only asks *which* levels to send.
/// Callers that loop over the result must assert it is non-empty, or an
/// emptied vocabulary would turn their test green by skipping its body.
async fn attested_levels_above_off(provider_id: &str) -> Vec<ReasoningEffort> {
    let server = mock_server().await;
    let model = reasoning_model(&provider(provider_id), &server);
    model
        .reasoning
        .efforts()
        .iter()
        .copied()
        .filter(|effort| *effort > ReasoningEffort::Off)
        .collect()
}

/// [`attested_levels_above_off`] for a provider that must have some, failing
/// rather than letting a caller's loop body silently never run.
async fn some_attested_levels_above_off(provider_id: &str) -> Vec<ReasoningEffort> {
    let levels = attested_levels_above_off(provider_id).await;
    assert!(
        !levels.is_empty(),
        "`{provider_id}` should attest a requestable level"
    );
    levels
}

// ---------------------------------------------------------------------------
// `thinking` toggle plus a graded effort — DeepSeek
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_thinking_toggle_shape_enables_with_a_graded_effort() {
    for effort in some_attested_levels_above_off("deepseek").await {
        let body = sent_body("deepseek", Some(ReasoningOptions::new(effort))).await;
        carries_only(
            &body,
            &[
                ("thinking", enabled()),
                ("reasoning_effort", Value::String(effort.to_string())),
            ],
        );
    }
}

#[tokio::test]
async fn the_thinking_toggle_shape_disables_with_the_toggle_alone() {
    // `Off` is an explicit request to stop reasoning, and this shape spells
    // that with the toggle. No effort string rides along: the toggle is what
    // disables, and `reasoning_effort` has no documented off value here.
    let body = sent_body(
        "deepseek",
        Some(ReasoningOptions::new(ReasoningEffort::Off)),
    )
    .await;
    carries_only(&body, &[("thinking", disabled())]);
}

#[tokio::test]
async fn the_graded_shape_speaks_its_providers_vocabulary_not_a_default_ladder() {
    // The point of a declared vocabulary, proven in both directions on the one
    // provider that names one. DeepSeek's reference lists `low`/`high`/`max`
    // and maps `medium`/`xhigh` onto `high`; `minimal` appears nowhere.
    //
    // A hardcoded baseline ladder got both ends wrong: it would have sent
    // `minimal` to an endpoint that has never heard of it, and refused `max`
    // the endpoint documents.
    let attested = some_attested_levels_above_off("deepseek").await;
    assert!(
        !attested.contains(&ReasoningEffort::Minimal),
        "DeepSeek documents no `minimal`: {attested:?}"
    );
    assert!(
        attested.contains(&ReasoningEffort::Max),
        "DeepSeek documents `max`, which the baseline ladder stops short of: {attested:?}"
    );

    rejected("deepseek", ReasoningOptions::new(ReasoningEffort::Minimal)).await;
    let body = sent_body(
        "deepseek",
        Some(ReasoningOptions::new(ReasoningEffort::Max)),
    )
    .await;
    carries_only(
        &body,
        &[
            ("thinking", enabled()),
            ("reasoning_effort", Value::String("max".into())),
        ],
    );
}

// ---------------------------------------------------------------------------
// `thinking` toggle alone — Z.AI and Xiaomi MiMo
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_thinking_toggle_only_shape_has_no_graded_effort() {
    // The shape carries no effort field, so every attested level above `Off`
    // reads as "enabled" — and none of them leaks a `reasoning_effort` string.
    for id in ["zai", "xiaomi"] {
        for effort in some_attested_levels_above_off(id).await {
            let body = sent_body(id, Some(ReasoningOptions::new(effort))).await;
            carries_only(&body, &[("thinking", enabled())]);
        }

        let body = sent_body(id, Some(ReasoningOptions::new(ReasoningEffort::Off))).await;
        carries_only(&body, &[("thinking", disabled())]);
    }
}

// ---------------------------------------------------------------------------
// `reasoning_effort` alone — the plain OpenAI shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_reasoning_effort_shape_sends_the_level_as_a_string() {
    for effort in [
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ] {
        let body = sent_body("openai", Some(ReasoningOptions::new(effort))).await;
        carries_only(
            &body,
            &[("reasoning_effort", Value::String(effort.to_string()))],
        );
    }
}

#[tokio::test]
async fn the_reasoning_effort_shape_disables_with_none_not_silence() {
    // This shape has no toggle, so the disabling value is a level of its own:
    // `none`, which is what the endpoint documents. Our ladder spells the same
    // level `off`, and the two are not interchangeable on the wire.
    let body = sent_body("openai", Some(ReasoningOptions::new(ReasoningEffort::Off))).await;
    carries_only(&body, &[("reasoning_effort", Value::String("none".into()))]);
}

// ---------------------------------------------------------------------------
// Top-level `enable_thinking` — open-model runtimes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_enable_thinking_shape_sends_exact_boolean_states() {
    let server = mock_server().await;
    let provider = Provider::builder("local", "Local", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::EnableThinking,
            ..OpenAiCompat::default()
        })
        .build()
        .expect("valid provider");
    let mut model = Model::openai_completions("reasoner").with_base_url(server.uri());
    model.provider = "local".into();
    model.reasoning = ReasoningCapability::baseline();

    for (effort, expected) in [
        (ReasoningEffort::High, Value::Bool(true)),
        (ReasoningEffort::Off, Value::Bool(false)),
    ] {
        let message = provider
            .stream(
                &model,
                &Context::new().user("hi"),
                &options(Some(ReasoningOptions::new(effort))),
            )
            .finish()
            .await;
        assert_eq!(message.error_kind, None, "{:?}", message.error_message);
        let bodies = request_bodies(&server).await;
        carries_only(
            bodies.last().expect("request body"),
            &[("enable_thinking", expected)],
        );
    }
}

#[tokio::test]
async fn chat_template_kwargs_receive_only_typed_reasoning_values() {
    let server = mock_server().await;
    let provider = Provider::builder("local", "Local", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(OpenAiChatTemplateKwargs {
                enable_thinking: Some("enable_thinking"),
                reasoning_effort: Some("reasoning_effort"),
                token_budget: Some(OpenAiReasoningBudgetField::ThinkingTokenBudget),
            }),
            ..OpenAiCompat::default()
        })
        .build()
        .expect("valid provider");
    let mut model = Model::openai_completions("reasoner").with_base_url(server.uri());
    model.provider = "local".into();
    model.reasoning =
        ReasoningCapability::baseline().with_token_budget(CapabilitySupport::Supported);

    let enabled = StreamOptions {
        max_tokens: Some(8192),
        ..options(Some(
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096),
        ))
    };
    let message = provider
        .stream(&model, &Context::new().user("hi"), &enabled)
        .finish()
        .await;
    assert_eq!(message.error_kind, None, "{:?}", message.error_message);
    let bodies = request_bodies(&server).await;
    carries_only(
        bodies.last().expect("enabled request"),
        &[(
            "chat_template_kwargs",
            serde_json::json!({
                "enable_thinking": true,
                "reasoning_effort": "high",
                "thinking_token_budget": 4096,
            }),
        )],
    );

    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::Off))),
        )
        .finish()
        .await;
    assert_eq!(message.error_kind, None, "{:?}", message.error_message);
    let bodies = request_bodies(&server).await;
    carries_only(
        bodies.last().expect("disabled request"),
        &[(
            "chat_template_kwargs",
            serde_json::json!({ "enable_thinking": false }),
        )],
    );
}

#[tokio::test]
async fn chat_template_budgets_that_conflict_with_disable_or_output_budget_fail_before_http() {
    let server = mock_server().await;
    let provider = Provider::builder("local", "Local", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(OpenAiChatTemplateKwargs {
                enable_thinking: Some("enable_thinking"),
                reasoning_effort: None,
                token_budget: Some(OpenAiReasoningBudgetField::ThinkingBudget),
            }),
            ..OpenAiCompat::default()
        })
        .build()
        .expect("valid provider");
    let mut model = Model::openai_completions("reasoner").with_base_url(server.uri());
    model.provider = "local".into();
    model.reasoning =
        ReasoningCapability::baseline().with_token_budget(CapabilitySupport::Supported);

    for reasoning in [
        ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096),
        ReasoningOptions::new(ReasoningEffort::Off).with_token_budget(1024),
    ] {
        let message = provider
            .stream(
                &model,
                &Context::new().user("hi"),
                &StreamOptions {
                    max_tokens: Some(4096),
                    ..options(Some(reasoning))
                },
            )
            .finish()
            .await;
        assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    }

    assert!(
        request_bodies(&server).await.is_empty(),
        "invalid budgets must fail before HTTP"
    );
}

#[tokio::test]
async fn an_optional_chat_template_budget_does_not_make_every_enabled_request_spend_one() {
    let server = mock_server().await;
    let provider = Provider::builder("local", "Local", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(OpenAiChatTemplateKwargs {
                enable_thinking: Some("enable_thinking"),
                reasoning_effort: None,
                token_budget: Some(OpenAiReasoningBudgetField::ThinkingBudgetTokens),
            }),
            ..OpenAiCompat::default()
        })
        .build()
        .expect("valid provider");
    let mut model = Model::openai_completions("reasoner").with_base_url(server.uri());
    model.provider = "local".into();
    model.reasoning = ReasoningCapability::baseline();

    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::High))),
        )
        .finish()
        .await;
    assert_eq!(message.error_kind, None, "{:?}", message.error_message);
    carries_only(
        &request_bodies(&server).await[0],
        &[(
            "chat_template_kwargs",
            serde_json::json!({ "enable_thinking": true }),
        )],
    );
}

#[tokio::test]
async fn every_declared_chat_template_budget_field_is_sent_verbatim() {
    for (field, name) in [
        (
            OpenAiReasoningBudgetField::ThinkingTokenBudget,
            "thinking_token_budget",
        ),
        (
            OpenAiReasoningBudgetField::ThinkingBudget,
            "thinking_budget",
        ),
        (
            OpenAiReasoningBudgetField::ThinkingBudgetTokens,
            "thinking_budget_tokens",
        ),
    ] {
        let server = mock_server().await;
        let provider = Provider::builder("local", "Local", server.uri())
            .adapter(Arc::new(OpenAiCompletions))
            .openai_compat(OpenAiCompat {
                reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(
                    OpenAiChatTemplateKwargs {
                        enable_thinking: Some("enabled"),
                        reasoning_effort: None,
                        token_budget: Some(field),
                    },
                ),
                ..OpenAiCompat::default()
            })
            .build()
            .expect("valid provider");
        let mut model = Model::openai_completions("reasoner").with_base_url(server.uri());
        model.provider = "local".into();
        model.reasoning =
            ReasoningCapability::baseline().with_token_budget(CapabilitySupport::Supported);

        let message = provider
            .stream(
                &model,
                &Context::new().user("hi"),
                &StreamOptions {
                    max_tokens: Some(4096),
                    ..options(Some(
                        ReasoningOptions::new(ReasoningEffort::High).with_token_budget(2048),
                    ))
                },
            )
            .finish()
            .await;
        assert_eq!(message.error_kind, None, "{:?}", message.error_message);
        let body = &request_bodies(&server).await[0];
        assert_eq!(body["chat_template_kwargs"]["enabled"], true);
        assert_eq!(body["chat_template_kwargs"][name], 2048);
        assert_eq!(
            body["chat_template_kwargs"]
                .as_object()
                .expect("kwargs object")
                .len(),
            2,
            "only the declared enabled state and `{name}` belong in {body}"
        );
    }
}

#[tokio::test]
async fn an_effort_only_chat_template_shape_uses_none_to_disable() {
    let server = mock_server().await;
    let provider = Provider::builder("local", "Local", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(OpenAiChatTemplateKwargs {
                reasoning_effort: Some("effort"),
                ..OpenAiChatTemplateKwargs::default()
            }),
            ..OpenAiCompat::default()
        })
        .build()
        .expect("valid provider");
    let mut model = Model::openai_completions("reasoner").with_base_url(server.uri());
    model.provider = "local".into();
    model.reasoning = ReasoningCapability::baseline();

    for (effort, expected) in [
        (ReasoningEffort::High, "high"),
        (ReasoningEffort::Off, "none"),
    ] {
        let message = provider
            .stream(
                &model,
                &Context::new().user("hi"),
                &options(Some(ReasoningOptions::new(effort))),
            )
            .finish()
            .await;
        assert_eq!(message.error_kind, None, "{:?}", message.error_message);
        let bodies = request_bodies(&server).await;
        carries_only(
            bodies.last().expect("request body"),
            &[(
                "chat_template_kwargs",
                serde_json::json!({ "effort": expected }),
            )],
        );
    }
}

#[tokio::test]
async fn an_unattested_chat_template_budget_fails_before_http() {
    let server = mock_server().await;
    let provider = Provider::builder("local", "Local", server.uri())
        .adapter(Arc::new(OpenAiCompletions))
        .openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(OpenAiChatTemplateKwargs {
                enable_thinking: Some("enabled"),
                reasoning_effort: None,
                token_budget: Some(OpenAiReasoningBudgetField::ThinkingBudget),
            }),
            ..OpenAiCompat::default()
        })
        .build()
        .expect("valid provider");
    let mut model = Model::openai_completions("reasoner").with_base_url(server.uri());
    model.provider = "local".into();
    model.reasoning = ReasoningCapability::baseline();

    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions {
                max_tokens: Some(4096),
                ..options(Some(
                    ReasoningOptions::new(ReasoningEffort::High).with_token_budget(2048),
                ))
            },
        )
        .finish()
        .await;
    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    assert!(request_bodies(&server).await.is_empty());
}

// ---------------------------------------------------------------------------
// No declared shape — Moonshot AI
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_provider_with_no_declared_shape_sends_nothing_and_refuses_a_request() {
    // Moonshot's thinking models decide for themselves and the endpoint takes
    // no reasoning field. Without a request the payload is untouched; with one
    // the preflight refuses rather than inventing a field.
    let body = sent_body("moonshot", None).await;
    carries_only(&body, &[]);

    for effort in ReasoningEffort::ALL {
        rejected("moonshot", ReasoningOptions::new(effort)).await;
    }
}

// ---------------------------------------------------------------------------
// Unsupported requests never become HTTP traffic
// ---------------------------------------------------------------------------

/// Stream `reasoning` against `provider_id`'s reasoning model, assert it
/// terminates as `InvalidRequest`, and assert the mock server never saw it.
async fn rejected(provider_id: &str, reasoning: ReasoningOptions) {
    let server = mock_server().await;
    let provider = provider(provider_id);
    let model = reasoning_model(&provider, &server);
    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(reasoning.clone())),
        )
        .finish()
        .await;

    assert_eq!(
        message.error_kind,
        Some(ErrorKind::InvalidRequest),
        "`{provider_id}` should refuse {reasoning:?} in-band"
    );
    assert!(
        request_bodies(&server).await.is_empty(),
        "`{provider_id}` must refuse {reasoning:?} before any HTTP request"
    );
}

#[tokio::test]
async fn no_declared_shape_carries_a_token_budget() {
    // Effort on this wire is a string, never a token count — so a budget is
    // refused by every OpenAI-compatible shape, including the one that would
    // otherwise honour the level.
    for (id, format) in TARGETS {
        assert!(!format.accepts_token_budget(), "`{id}` carries no budget");
        rejected(
            id,
            ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096),
        )
        .await;
    }
    rejected(
        "openai",
        ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096),
    )
    .await;
}

#[tokio::test]
async fn every_shape_refuses_exactly_the_levels_its_provider_does_not_attest() {
    // The whole ladder against every target: a level the provider's models
    // attest goes through, and every other one is refused before HTTP. Which
    // levels those are differs per provider — that is the point — so the
    // expectation is read from the attestation, not written down here.
    for (id, _) in TARGETS {
        let attested = attested_levels_above_off(id).await;
        for effort in ReasoningEffort::ALL {
            if effort == ReasoningEffort::Off || attested.contains(&effort) {
                continue;
            }
            rejected(id, ReasoningOptions::new(effort)).await;
        }
    }

    // And no target attests the whole ladder, so the loop above is not vacuous.
    for (id, _) in TARGETS {
        assert!(
            attested_levels_above_off(id).await.len() < ReasoningEffort::ALL.len() - 1,
            "`{id}` should refuse at least one level"
        );
    }
}

#[tokio::test]
async fn a_declared_shape_still_refuses_a_model_that_attests_nothing() {
    // The provider's shape is only half the check: a hand-built model attests
    // nothing, so even a level the endpoint understands cannot be requested.
    let server = mock_server().await;
    let mut model = Model::openai_completions("custom").with_base_url(server.uri());
    model.provider = "deepseek".into();
    assert_eq!(model.reasoning, ReasoningCapability::none());

    let message = Provider::deepseek()
        .stream(
            &model,
            &Context::new().user("hi"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::High))),
        )
        .finish()
        .await;

    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    assert!(request_bodies(&server).await.is_empty());
}

// ---------------------------------------------------------------------------
// What a reasoning request must not disturb
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_reasoning_option_leaves_every_shape_untouched() {
    for (id, _) in TARGETS {
        let body = sent_body(id, None).await;
        carries_only(&body, &[]);
    }
    carries_only(&sent_body("openai", None).await, &[]);
}

#[tokio::test]
async fn an_enabled_request_does_not_disturb_thinking_output_or_its_signature() {
    // The request side changed; the response side must not. Thinking still
    // streams into its own block, and still records the wire field it arrived
    // in as the block's signature so replay can put it back there.
    const THINKING_BODY: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me think. \"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"2 + 2 is 4.\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"The answer is 4.\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(THINKING_BODY),
        )
        .mount(&server)
        .await;

    let provider = Provider::deepseek();
    let model = reasoning_model(&provider, &server);
    let message = provider
        .stream(
            &model,
            &Context::new().user("What is 2 + 2?"),
            &options(Some(ReasoningOptions::new(ReasoningEffort::High))),
        )
        .finish()
        .await;

    let thinking = message
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::Thinking(block) => Some(block),
            _ => None,
        })
        .expect("a thinking block");
    assert_eq!(thinking.thinking, "Let me think. 2 + 2 is 4.");
    assert_eq!(thinking.signature.as_deref(), Some("reasoning_content"));
    assert_eq!(message.text(), "The answer is 4.");
    assert!(matches!(
        message.content.first(),
        Some(AssistantContent::Thinking(_))
    ));
}

#[tokio::test]
async fn reasoning_usage_stays_inside_output_and_is_never_added_twice() {
    // `reasoning_tokens` is a *breakdown* of the completion tokens, not a
    // fifth bucket. It must never inflate the output count or the total —
    // whether the endpoint reports a total or leaves banshu to derive one.
    const REPORTED_TOTAL: &str = concat!(
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":100,\"total_tokens\":110,",
        "\"completion_tokens_details\":{\"reasoning_tokens\":40}}}\n\n",
        "data: [DONE]\n\n",
    );
    const DERIVED_TOTAL: &str = concat!(
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":100,",
        "\"completion_tokens_details\":{\"reasoning_tokens\":40}}}\n\n",
        "data: [DONE]\n\n",
    );

    for body in [REPORTED_TOTAL, DERIVED_TOTAL] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let provider = Provider::deepseek();
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

        assert_eq!(usage.reasoning, Some(40));
        assert_eq!(usage.output, 100, "reasoning tokens are part of output");
        assert!(
            usage.reasoning.unwrap_or(0) <= usage.output,
            "reasoning is a subset of output"
        );
        assert_eq!(
            usage.total_tokens,
            usage.input + usage.cache_read + usage.cache_write + usage.output,
            "the total counts output once and reasoning never on its own"
        );
        assert_eq!(usage.total_tokens, 110);
    }
}
