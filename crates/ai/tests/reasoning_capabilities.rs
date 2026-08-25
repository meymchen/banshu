//! The unified reasoning request contract and the metadata it is checked
//! against (issue #42).
//!
//! Three declarations meet here and nothing is inferred from a base URL or a
//! model id:
//!
//! - a request says how much reasoning it wants — [`ReasoningOptions`] on
//!   `StreamOptions`, or `None` for "don't touch reasoning at all";
//! - a model attests which effort levels and whether a token budget its
//!   metadata source knows about — `Model.reasoning`;
//! - a provider declares the request wire shape its endpoint accepts —
//!   `OpenAiCompat::reasoning_format` / `AnthropicCompat::reasoning_format`.
//!
//! A request that none of the three can honour terminates in-band with
//! `ErrorKind::InvalidRequest` before any HTTP request leaves the process.
//! Serializing an honoured request onto each wire shape belongs to
//! `openai_reasoning_requests.rs` (issue #43) and
//! `anthropic_reasoning_requests.rs` (issue #44); this file stays on the
//! contract, and pins that the no-reasoning payload has not moved.

use banshu_ai::{
    AnthropicCompat, AnthropicReasoningFormat, CapabilitySupport, Context, ErrorKind, Model,
    OpenAiReasoningFormat, Provider, ReasoningCapability, ReasoningEffort, ReasoningOptions,
    StreamOptions,
};
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const OPENAI_STOP_BODY: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

const ANTHROPIC_STOP_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// Every reasoning request field either protocol could grow. A request with no
/// reasoning option must carry none of them.
const REASONING_WIRE_FIELDS: [&str; 5] = [
    "reasoning_effort",
    "reasoning",
    "thinking",
    "enable_thinking",
    "chat_template_kwargs",
];

/// The ladder a provider's models attest when the provider declares no effort
/// vocabulary of its own — the honest fallback, not a claim about the endpoint.
const BASELINE: &[ReasoningEffort] = &ReasoningCapability::BASELINE;

/// DeepSeek's own vocabulary: its reference documents `low`/`high`/`max`, plus
/// `off` on the `thinking` toggle. `medium` and `xhigh` are accepted only by
/// being remapped onto `high`, which is a clamp, so they stay out.
const DEEPSEEK_LADDER: &[ReasoningEffort] = &[
    ReasoningEffort::Off,
    ReasoningEffort::Low,
    ReasoningEffort::High,
    ReasoningEffort::Max,
];

/// Moonshot's endpoint has no reasoning request field, so no level is
/// requestable and its models attest none.
const NO_LADDER: &[ReasoningEffort] = &[];

/// The four OpenAI-compatible targets: provider id, the reasoning request
/// shape its endpoint declares, the token-budget support that shape carries,
/// and the effort ladder its models therefore attest.
const OPENAI_TARGETS: [(
    &str,
    OpenAiReasoningFormat,
    CapabilitySupport,
    &[ReasoningEffort],
); 4] = [
    (
        "deepseek",
        OpenAiReasoningFormat::ThinkingToggle,
        CapabilitySupport::Unsupported,
        DEEPSEEK_LADDER,
    ),
    (
        "zai",
        OpenAiReasoningFormat::ThinkingToggleOnly,
        CapabilitySupport::Unsupported,
        BASELINE,
    ),
    (
        "moonshot",
        OpenAiReasoningFormat::Unsupported,
        CapabilitySupport::Unsupported,
        NO_LADDER,
    ),
    (
        "xiaomi",
        OpenAiReasoningFormat::ThinkingToggleOnly,
        CapabilitySupport::Unsupported,
        BASELINE,
    ),
];

/// The two Anthropic-compatible targets, same shape. Neither endpoint's
/// reference documents a `budget_tokens` field — MiniMax enables thinking with
/// `adaptive` and Kimi with the bare toggle — so neither takes a budget, and
/// neither carries an effort field to constrain either.
const ANTHROPIC_TARGETS: [(
    &str,
    AnthropicReasoningFormat,
    CapabilitySupport,
    &[ReasoningEffort],
); 2] = [
    (
        "minimax",
        AnthropicReasoningFormat::ThinkingAdaptive,
        CapabilitySupport::Unsupported,
        BASELINE,
    ),
    (
        "kimi",
        AnthropicReasoningFormat::ThinkingToggle,
        CapabilitySupport::Unsupported,
        BASELINE,
    ),
];

fn provider(id: &str) -> Provider {
    match id {
        "deepseek" => Provider::deepseek(),
        "zai" => Provider::zai(),
        "moonshot" => Provider::moonshot(),
        "xiaomi" => Provider::xiaomi(),
        "minimax" => Provider::minimax(
            banshu_ai::MiniMaxRegion::Global,
            std::sync::Arc::new(banshu_ai::InMemoryCredentialStore::new()),
        ),
        "kimi" => Provider::kimi(std::sync::Arc::new(
            banshu_ai::InMemoryCredentialStore::new(),
        )),
        other => panic!("`{other}` is not one of the six target providers"),
    }
}

/// A custom provider declaring Anthropic's budget shape — the one shape in the
/// crate that carries a token budget, and one no bundled vendor declares. Its
/// wire output is pinned in `anthropic_reasoning_requests.rs`; here it is only
/// the vehicle for the budget checks that are about the *model*.
fn budget_provider(server: &MockServer) -> Provider {
    Provider::anthropic_compatible("custom", "Custom", server.uri(), ["TEST_API_KEY"])
        .with_anthropic_compat(AnthropicCompat {
            reasoning_format: AnthropicReasoningFormat::ThinkingBudget,
            ..AnthropicCompat::default()
        })
}

/// A caller-supplied model of [`budget_provider`], attesting the baseline
/// ladder and whatever `token_budget` support the test needs.
fn budget_model(server: &MockServer, token_budget: CapabilitySupport) -> Model {
    let mut model = Model::anthropic_messages("custom-thinker").with_base_url(server.uri());
    model.provider = "custom".into();
    model.reasoning = ReasoningCapability::baseline().with_token_budget(token_budget);
    model
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

fn reasoning(effort: ReasoningEffort) -> StreamOptions {
    StreamOptions {
        reasoning: Some(ReasoningOptions::new(effort)),
        ..options()
    }
}

fn budget(effort: ReasoningEffort, tokens: u32) -> StreamOptions {
    StreamOptions {
        reasoning: Some(ReasoningOptions::new(effort).with_token_budget(tokens)),
        ..options()
    }
}

async fn mount_openai_sse(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(OPENAI_STOP_BODY),
        )
        .mount(server)
        .await;
}

async fn mount_anthropic_sse(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(ANTHROPIC_STOP_BODY),
        )
        .mount(server)
        .await;
}

/// A model of this provider to stream against, pointed at the mock server:
/// one that attests a reasoning level where the provider offers any, else its
/// first model. Moonshot is the "else": its endpoint takes no reasoning field,
/// so none of its models attests a level — and a request there is refused on
/// the format before the model is ever consulted.
fn reasoning_model(provider: &Provider, server: &MockServer) -> Model {
    let models = provider.models();
    models
        .iter()
        .find(|model| model.reasoning.reasons())
        .or_else(|| models.first())
        .unwrap_or_else(|| panic!("`{}` should serve models", provider.id()))
        .clone()
        .with_base_url(server.uri())
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

// ---------------------------------------------------------------------------
// The request contract
// ---------------------------------------------------------------------------

#[test]
fn a_request_expresses_no_override_or_every_effort_level() {
    // No override is the absence of the option, not a level.
    assert_eq!(StreamOptions::default().reasoning, None);

    // Every level on the ladder is expressible, and each is distinct.
    let ladder = [
        ReasoningEffort::Off,
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ];
    assert_eq!(ReasoningEffort::ALL, ladder);
    for (index, effort) in ladder.into_iter().enumerate() {
        let options = reasoning(effort);
        let requested = options.reasoning.expect("an effort was requested");
        assert_eq!(requested.effort, effort);
        assert_eq!(requested.token_budget, None, "a budget is opt-in");
        // The ladder is ordered, so `Off` is the least and `Max` the most.
        assert_eq!(
            ladder.iter().position(|level| *level == effort),
            Some(index)
        );
    }

    // The optional token budget rides alongside any level.
    let requested = budget(ReasoningEffort::High, 4096)
        .reasoning
        .expect("an effort was requested");
    assert_eq!(requested.effort, ReasoningEffort::High);
    assert_eq!(requested.token_budget, Some(4096));
}

// ---------------------------------------------------------------------------
// The metadata contract
// ---------------------------------------------------------------------------

#[test]
fn model_metadata_lists_effort_levels_and_budget_capability() {
    // Nothing attested is the honest default: no level, budget Unknown.
    let none = ReasoningCapability::none();
    assert!(!none.reasons());
    assert_eq!(none.efforts(), []);
    assert_eq!(none.token_budget(), CapabilitySupport::Unknown);
    for effort in ReasoningEffort::ALL {
        assert!(!none.supports(effort), "{effort} is not attested");
    }

    // A source that says only "this model reasons" attests the baseline
    // ladder; `xhigh` and `max` need an attestation of their own.
    let baseline = ReasoningCapability::baseline();
    assert!(baseline.reasons());
    assert_eq!(
        baseline.efforts(),
        [
            ReasoningEffort::Off,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ]
    );
    assert!(!baseline.supports(ReasoningEffort::XHigh));
    assert!(!baseline.supports(ReasoningEffort::Max));

    // Levels are a set, not a flag: an explicit list is kept as given, and
    // budget capability is declared separately.
    let graded = ReasoningCapability::new([ReasoningEffort::Off, ReasoningEffort::Max])
        .with_token_budget(CapabilitySupport::Supported);
    assert_eq!(
        graded.efforts(),
        [ReasoningEffort::Off, ReasoningEffort::Max]
    );
    assert!(!graded.supports(ReasoningEffort::High));
    assert_eq!(graded.token_budget(), CapabilitySupport::Supported);

    // A model that only knows how to be told "off" does not reason.
    let off_only = ReasoningCapability::new([ReasoningEffort::Off]);
    assert!(!off_only.reasons());
    assert!(off_only.supports(ReasoningEffort::Off));
}

#[test]
fn hand_built_models_attest_nothing_until_told() {
    for model in [
        Model::openai_completions("custom"),
        Model::anthropic_messages("custom"),
    ] {
        assert_eq!(model.reasoning, ReasoningCapability::none());
    }
}

// ---------------------------------------------------------------------------
// The six target providers
// ---------------------------------------------------------------------------

#[test]
fn every_target_provider_declares_its_reasoning_request_format() {
    for (id, format, token_budget, ladder) in OPENAI_TARGETS {
        let provider = provider(id);
        let compat = provider.openai_compat();
        assert_eq!(
            compat.reasoning_format, format,
            "`{id}` declares its reasoning request format"
        );
        assert_eq!(format.accepts_token_budget(), token_budget.is_supported());
        // A provider names its own effort vocabulary or names none, in which
        // case its models fall back to the baseline ladder.
        assert_eq!(compat.reasoning_efforts.unwrap_or(BASELINE), ladder);
    }
    for (id, format, token_budget, ladder) in ANTHROPIC_TARGETS {
        let provider = provider(id);
        let compat = provider.anthropic_compat();
        assert_eq!(
            compat.reasoning_format, format,
            "`{id}` declares its reasoning request format"
        );
        assert_eq!(format.accepts_token_budget(), token_budget.is_supported());
        assert_eq!(compat.reasoning_efforts.unwrap_or(BASELINE), ladder);
    }
}

#[test]
fn every_target_provider_stamps_reasoning_capabilities_onto_its_models() {
    let targets = OPENAI_TARGETS
        .map(|(id, _, budget, ladder)| (id, budget, ladder))
        .into_iter()
        .chain(ANTHROPIC_TARGETS.map(|(id, _, budget, ladder)| (id, budget, ladder)));
    for (id, token_budget, ladder) in targets {
        let provider = provider(id);
        let models = provider.models();
        assert!(!models.is_empty(), "`{id}` should serve models");
        let mut attesting_models = 0;
        for model in &models {
            if model.reasoning.efforts().is_empty() {
                // Either the source says this model does not reason, or its
                // provider's endpoint has no way to ask — both mean no level
                // is requestable, and neither can take a budget.
                assert!(!model.reasoning.reasons());
                assert_eq!(
                    model.reasoning.token_budget(),
                    CapabilitySupport::Unsupported
                );
                continue;
            }
            attesting_models += 1;
            // A model source only says *whether* a model reasons, so the
            // ladder comes from the provider's declared vocabulary — the
            // baseline only where the provider names none.
            assert_eq!(
                model.reasoning.efforts(),
                ladder,
                "`{id}`/`{}` attests its provider's declared ladder",
                model.id
            );
            assert_eq!(
                model.reasoning.token_budget(),
                token_budget,
                "`{id}`/`{}` budget capability comes from the provider's declared format",
                model.id
            );
        }

        if ladder.is_empty() {
            // Moonshot: the endpoint takes no reasoning field, so not one of
            // its models — thinking or not — claims a requestable level.
            assert_eq!(
                attesting_models, 0,
                "`{id}` declares no requestable level, so none of its models may attest one"
            );
        } else {
            assert!(
                attesting_models > 0,
                "`{id}` should serve at least one reasoning model"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Preflight rejection — nothing reaches the endpoint
// ---------------------------------------------------------------------------

/// Stream `options` against a reasoning model of `provider_id` and assert the
/// request never left the process.
async fn rejected(provider_id: &str, options: StreamOptions, expected: &str) {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;
    mount_anthropic_sse(&server).await;

    let provider = provider(provider_id);
    let model = reasoning_model(&provider, &server);
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options)
        .finish()
        .await;

    assert_eq!(
        message.error_kind,
        Some(ErrorKind::InvalidRequest),
        "`{provider_id}` should reject this request in-band"
    );
    let detail = message.error_message.unwrap_or_default();
    assert!(
        detail.contains(expected),
        "`{detail}` should mention `{expected}`"
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "the preflight must win before the mock server is reached"
    );
}

#[tokio::test]
async fn an_effort_the_model_does_not_attest_is_rejected_before_dispatch() {
    // Which level is out of reach depends on the provider's own vocabulary:
    // DeepSeek documents `max` but no `minimal`, while Kimi names none and so
    // stops at the baseline ladder's `high`.
    rejected("deepseek", reasoning(ReasoningEffort::Minimal), "minimal").await;
    rejected("kimi", reasoning(ReasoningEffort::Max), "max").await;
}

#[tokio::test]
async fn a_non_reasoning_model_rejects_every_effort_before_dispatch() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::deepseek();
    let model = provider
        .models()
        .into_iter()
        .find(|model| !model.reasoning.reasons())
        .expect("the deepseek catalog has a non-reasoning model")
        .with_base_url(server.uri());

    for effort in ReasoningEffort::ALL {
        let message = provider
            .stream(&model, &Context::new().user("hi"), &reasoning(effort))
            .finish()
            .await;
        assert_eq!(
            message.error_kind,
            Some(ErrorKind::InvalidRequest),
            "`{}` attests no reasoning, so `{effort}` cannot be requested",
            model.id
        );
    }
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
}

#[tokio::test]
async fn a_provider_declaring_no_reasoning_format_rejects_every_effort() {
    // Moonshot's thinking models reason on their own; the endpoint takes no
    // reasoning request field, so asking for one cannot be honoured.
    rejected(
        "moonshot",
        reasoning(ReasoningEffort::High),
        "no reasoning request format",
    )
    .await;
}

#[tokio::test]
async fn a_token_budget_the_model_does_not_attest_is_rejected_before_dispatch() {
    // DeepSeek's declared shape carries no budget field, so no model of it
    // attests one.
    rejected(
        "deepseek",
        budget(ReasoningEffort::High, 4096),
        "token budget",
    )
    .await;
}

#[tokio::test]
async fn a_model_attesting_a_level_beyond_the_baseline_is_honoured() {
    // The metadata carries a *set of levels*, not a flag: Xiaomi's endpoint
    // takes no effort field, so its models sit on the baseline ladder and no
    // catalog model of it accepts `xhigh` — but a caller who knows their model
    // does says so, and the same request goes through. A boolean could not
    // express the difference, and neither could a single global ladder.
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;

    let provider = Provider::xiaomi();
    let catalog = reasoning_model(&provider, &server);
    assert!(!catalog.reasoning.supports(ReasoningEffort::XHigh));

    let mut declared = catalog.clone();
    declared.reasoning = ReasoningCapability::new(
        ReasoningCapability::BASELINE
            .into_iter()
            .chain([ReasoningEffort::XHigh]),
    );

    let refused = provider
        .stream(
            &catalog,
            &Context::new().user("hi"),
            &reasoning(ReasoningEffort::XHigh),
        )
        .finish()
        .await;
    assert_eq!(refused.error_kind, Some(ErrorKind::InvalidRequest));

    let honoured = provider
        .stream(
            &declared,
            &Context::new().user("hi"),
            &reasoning(ReasoningEffort::XHigh),
        )
        .finish()
        .await;
    assert_eq!(honoured.error_kind, None);
    assert_eq!(request_bodies(&server).await.len(), 1);
}

#[tokio::test]
async fn the_model_budget_check_is_separate_from_the_providers_format() {
    // The custom provider's declared shape *does* carry a budget, so the
    // format check passes — this is the model-side check firing on its own,
    // for a caller-supplied model that attests no budget support.
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let provider = budget_provider(&server);
    let model = budget_model(&server, CapabilitySupport::Unknown);
    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &budget(ReasoningEffort::High, 2048),
        )
        .finish()
        .await;

    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    assert!(
        message
            .error_message
            .unwrap_or_default()
            .contains("does not support a reasoning token budget")
    );
    // Turning reasoning *off* on the same model is fine — a disabled request
    // spends no budget — so it is the budget that was refused, not the model.
    let allowed = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &reasoning(ReasoningEffort::Off),
        )
        .finish()
        .await;
    assert_eq!(allowed.error_kind, None);
    assert_eq!(request_bodies(&server).await.len(), 1);
}

#[tokio::test]
async fn an_attested_effort_and_budget_reach_the_endpoint() {
    let server = MockServer::start().await;
    mount_anthropic_sse(&server).await;

    let provider = budget_provider(&server);
    let model = budget_model(&server, CapabilitySupport::Supported);
    let message = provider
        .stream(
            &model,
            &Context::new().user("hi"),
            &budget(ReasoningEffort::High, 2048),
        )
        .finish()
        .await;

    assert_eq!(message.error_kind, None, "an attested request is honoured");
    assert_eq!(request_bodies(&server).await.len(), 1);
}

// ---------------------------------------------------------------------------
// The unchanged default payload
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_request_without_a_reasoning_option_carries_no_reasoning_field() {
    let server = MockServer::start().await;
    mount_openai_sse(&server).await;
    mount_anthropic_sse(&server).await;

    Provider::deepseek()
        .stream(
            &reasoning_model(&Provider::deepseek(), &server),
            &Context::new().user("hi"),
            &options(),
        )
        .finish()
        .await;
    Provider::kimi(std::sync::Arc::new(
        banshu_ai::InMemoryCredentialStore::new(),
    ))
    .stream(
        &reasoning_model(
            &Provider::kimi(std::sync::Arc::new(
                banshu_ai::InMemoryCredentialStore::new(),
            )),
            &server,
        ),
        &Context::new().user("hi"),
        &options(),
    )
    .finish()
    .await;

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies.len(), 2, "both protocols were exercised");
    for body in bodies {
        for field in REASONING_WIRE_FIELDS {
            assert!(
                body.get(field).is_none(),
                "no reasoning option means no `{field}` in {body}"
            );
        }
    }
}
