//! Context-threshold cost tiers: a `ModelCost` may carry request-wide tiers
//! (models.dev `cost.tiers` with `tier.type == "context"`). The tier with the
//! highest threshold the request's total input usage *strictly exceeds*
//! supplies every rate for that request; at or below a threshold the lower
//! rate set applies. A zero threshold is unknown metadata and never selects a
//! tier, and models without tiers keep flat-rate semantics.

use banshu_ai::{Context, CostTier, Model, ModelCost, Provider, StreamOptions, Usage};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const THRESHOLD: u32 = 200_000;

fn tiered_rates() -> ModelCost {
    ModelCost {
        input: 1.0,
        output: 2.0,
        cache_read: 0.1,
        cache_write: 1.25,
        tiers: vec![CostTier {
            input_tokens_above: THRESHOLD,
            input: 2.0,
            output: 4.0,
            cache_read: 0.2,
            cache_write: 2.5,
        }],
    }
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

/// Stream one OpenAI-completions request whose response reports
/// `prompt_tokens`/`completion_tokens` (plus optional cached tokens) and
/// return the assembled usage.
async fn openai_usage(cost: ModelCost, prompt_tokens: u64, cached_tokens: u64) -> Usage {
    let server = MockServer::start().await;
    let cached = if cached_tokens > 0 {
        format!(",\"prompt_tokens_details\":{{\"cached_tokens\":{cached_tokens}}}")
    } else {
        String::new()
    };
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"ok\"}},\"finish_reason\":null}}]}}\n\n\
                     data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":0{cached}}}}}\n\n\
                     data: [DONE]\n\n",
                )),
        )
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("moonshot", "Moonshot", server.uri(), ["X"]);
    let mut model = Model::openai_completions("kimi-k2").with_base_url(server.uri());
    model.cost = cost;
    provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await
        .usage
}

#[tokio::test]
async fn base_rates_apply_below_and_at_the_threshold() {
    for prompt_tokens in [THRESHOLD as u64 - 1, THRESHOLD as u64] {
        let usage = openai_usage(tiered_rates(), prompt_tokens, 0).await;
        let expected_input = prompt_tokens as f64 / 1_000_000.0 * 1.0;
        assert!(
            (usage.cost.input - expected_input).abs() < 1e-12,
            "{prompt_tokens} tokens should bill at the base rate, got {}",
            usage.cost.input
        );
    }
}

#[tokio::test]
async fn tier_rates_apply_above_the_threshold() {
    let usage = openai_usage(tiered_rates(), THRESHOLD as u64 + 1, 0).await;
    let expected_input = (THRESHOLD as f64 + 1.0) / 1_000_000.0 * 2.0;
    assert!(
        (usage.cost.input - expected_input).abs() < 1e-12,
        "expected the tier input rate, got {}",
        usage.cost.input
    );
}

#[tokio::test]
async fn the_highest_exceeded_tier_wins() {
    let mut cost = tiered_rates();
    cost.tiers.push(CostTier {
        input_tokens_above: 100_000,
        input: 1.5,
        output: 3.0,
        cache_read: 0.15,
        cache_write: 1.5,
    });
    // At the first boundary the base rates still apply; immediately above it
    // the 100k tier applies; above the 200k boundary the 200k tier wins.
    let at_first = openai_usage(cost.clone(), 100_000, 0).await;
    assert!((at_first.cost.input - 0.1).abs() < 1e-12);
    let above_first = openai_usage(cost.clone(), 100_001, 0).await;
    assert!((above_first.cost.input - 0.1500015).abs() < 1e-12);
    let between = openai_usage(cost.clone(), 150_000, 0).await;
    assert!((between.cost.input - 0.225).abs() < 1e-12);
    let above_second = openai_usage(cost, 200_001, 0).await;
    assert!((above_second.cost.input - 0.400002).abs() < 1e-12);
}

#[tokio::test]
async fn cache_reads_count_toward_the_threshold() {
    // 50k uncached input + 150,001 cached tokens = 200,001 total input usage:
    // over the threshold only because cache reads count.
    let usage = openai_usage(tiered_rates(), THRESHOLD as u64 + 1, 150_001).await;
    assert_eq!(usage.input, 50_000);
    assert_eq!(usage.cache_read, 150_001);
    let expected_input = 50_000.0 / 1_000_000.0 * 2.0;
    let expected_read = 150_001.0 / 1_000_000.0 * 0.2;
    assert!((usage.cost.input - expected_input).abs() < 1e-12);
    assert!((usage.cost.cache_read - expected_read).abs() < 1e-12);
}

#[tokio::test]
async fn a_zero_threshold_never_selects_a_tier() {
    let cost = ModelCost {
        tiers: vec![CostTier {
            input_tokens_above: 0,
            input: 99.0,
            output: 99.0,
            cache_read: 99.0,
            cache_write: 99.0,
        }],
        ..tiered_rates()
    };
    let usage = openai_usage(cost, THRESHOLD as u64 + 1, 0).await;
    let expected_input = (THRESHOLD as f64 + 1.0) / 1_000_000.0 * 1.0;
    assert!((usage.cost.input - expected_input).abs() < 1e-12);
}

#[tokio::test]
async fn models_without_tiers_keep_flat_costs() {
    let flat = ModelCost {
        input: 1.0,
        output: 2.0,
        cache_read: 0.1,
        cache_write: 1.25,
        tiers: Vec::new(),
    };
    let usage = openai_usage(flat, 300_000, 100_000).await;
    assert_eq!(usage.input, 200_000);
    // Exactly the pre-tier calculation: every class at the flat rate.
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(usage.cost.input, 0.2), "{}", usage.cost.input);
    assert!(
        close(usage.cost.cache_read, 0.01),
        "{}",
        usage.cost.cache_read
    );
    assert!(
        close(usage.cost.cache_write, 0.0),
        "{}",
        usage.cost.cache_write
    );
    assert!(close(usage.cost.total, 0.21), "{}", usage.cost.total);
}

#[tokio::test]
async fn one_hour_cache_writes_bill_at_twice_the_selected_tier_input_rate() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{",
                    "\"input_tokens\":100001,\"output_tokens\":1,",
                    "\"cache_creation_input_tokens\":100000,",
                    "\"cache_creation\":{\"ephemeral_1h_input_tokens\":100000}}}}\n\n",
                    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},",
                    "\"usage\":{\"output_tokens\":10}}\n\n",
                    "data: {\"type\":\"message_stop\"}\n\n",
                )),
        )
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["X"]);
    let mut model = Model::anthropic_messages("kimi-for-coding").with_base_url(server.uri());
    model.cost = tiered_rates();

    // 200,001 total input usage selects the tier; the 100k 1h write bills at
    // 2x the *tier* input rate (2.0), not the base rate.
    let usage = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await
        .usage;
    assert!((usage.cost.cache_write - 0.4).abs() < 1e-12);
    assert!((usage.cost.input - 0.200002).abs() < 1e-12);
}

#[test]
fn parses_context_tiers_from_models_dev() {
    let api_json = serde_json::json!({
        "google": {
            "models": {
                "gemini-2.5-pro": {
                    "name": "Gemini 2.5 Pro",
                    "tool_call": true,
                    "cost": {
                        "input": 1.25,
                        "output": 10.0,
                        "cache_read": 0.125,
                        "tiers": [
                            {
                                "input": 2.5,
                                "output": 15.0,
                                "cache_read": 0.25,
                                "tier": { "type": "context", "size": 200000 }
                            },
                            {
                                "input": 99.0,
                                "tier": { "type": "other", "size": 100000 }
                            },
                            {
                                "input": 99.0,
                                "tier": { "type": "context", "size": 0 }
                            }
                        ]
                    },
                    "limit": { "context": 1048576, "output": 65536 }
                }
            }
        }
    });

    let models = banshu_ai::models_dev::models_from_api_json(&api_json, "google")
        .expect("provider key parses");
    assert_eq!(models.len(), 1);
    let cost = &models[0].cost;
    assert_eq!(cost.input, 1.25);
    // Only the well-formed context tier survives; non-context types and
    // zero-sized (unknown) boundaries are dropped.
    assert_eq!(
        cost.tiers,
        vec![CostTier {
            input_tokens_above: 200_000,
            input: 2.5,
            output: 15.0,
            cache_read: 0.25,
            cache_write: 0.0,
        }]
    );
}
