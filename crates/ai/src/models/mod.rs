//! Bundled model catalog, generated from [models.dev](https://models.dev) by
//! `cargo run -p xtask -- generate-catalog`. Catalog entries carry only model
//! metadata; the owning [`Provider`](crate::Provider) stamps on `provider`,
//! `base_url`, and the wire protocol when it lists its models.

pub(crate) mod dev;

use serde::Deserialize;

use crate::models_dev::{modality_from_str, model_reasoning_format, reasoning_from_options};
use crate::provider::DeclaredReasoning;
use crate::types::{ApiKind, CapabilitySupport, CostTier, Model, ModelCapabilities, ModelCost};

/// One entry in a bundled `catalog/<provider>.json` file.
#[derive(Deserialize)]
struct CatalogModel {
    id: String,
    name: String,
    reasoning: bool,
    #[serde(default)]
    reasoning_options: Option<serde_json::Value>,
    input: Vec<String>,
    context_window: u32,
    max_tokens: u32,
    cost: CatalogCost,
}

#[derive(Deserialize)]
struct CatalogCost {
    input: f64,
    output: f64,
    cache_read: f64,
    cache_write: f64,
    #[serde(default)]
    tiers: Vec<CostTier>,
}

/// Raw JSON for a provider's bundled catalog, or `None` if none is bundled.
fn raw_catalog(provider_id: &str) -> Option<&'static str> {
    Some(match provider_id {
        "deepseek" => include_str!("catalog/deepseek.json"),
        "zai" => include_str!("catalog/zai.json"),
        "zai-coding-cn" => include_str!("catalog/zai-coding-cn.json"),
        "moonshot-cn" => include_str!("catalog/moonshot-cn.json"),
        "xiaomi-token-plan-cn" => include_str!("catalog/xiaomi-token-plan-cn.json"),
        "xiaomi-token-plan-ams" => include_str!("catalog/xiaomi-token-plan-ams.json"),
        "xiaomi-token-plan-sgp" => include_str!("catalog/xiaomi-token-plan-sgp.json"),
        "qwen-token-plan" => include_str!("catalog/qwen-token-plan.json"),
        "qwen-token-plan-cn" => include_str!("catalog/qwen-token-plan-cn.json"),
        "qwen-token-plan-individual" => include_str!("catalog/qwen-token-plan-individual.json"),
        "minimax" => include_str!("catalog/minimax.json"),
        // The CN region serves the same catalog, stamped with its own
        // provider id and endpoint.
        "minimax-cn" => include_str!("catalog/minimax.json"),
        "moonshot" => include_str!("catalog/moonshot.json"),
        "kimi" => include_str!("catalog/kimi.json"),
        "xiaomi" => include_str!("catalog/xiaomi.json"),
        _ => return None,
    })
}

/// Build the model list for a provider from its bundled catalog, stamping each
/// model with the provider's id, base URL, wire protocol, and what its
/// declared reasoning request shape attests — the effort ladder and the
/// token-budget support.
pub(crate) fn catalog_models(
    provider_id: &str,
    base_url: &str,
    api: ApiKind,
    reasoning: DeclaredReasoning,
) -> Vec<Model> {
    let Some(raw) = raw_catalog(provider_id) else {
        return Vec::new();
    };
    let entries: Vec<CatalogModel> = serde_json::from_str(raw).unwrap_or_default();
    entries
        .into_iter()
        .map(|entry| Model {
            allow_empty_thinking_signature: crate::models_dev::allows_unsigned_thinking(
                provider_id,
                &entry.id,
            ),
            id: entry.id,
            name: entry.name,
            api,
            provider: provider_id.to_string(),
            base_url: base_url.to_string(),
            headers: Default::default(),
            openai_reasoning_format: model_reasoning_format(
                provider_id,
                entry.reasoning_options.as_ref(),
            ),
            anthropic_reasoning_format: crate::models_dev::model_anthropic_reasoning_format(
                provider_id,
                entry.reasoning_options.as_ref(),
            ),
            reasoning: reasoning_from_options(
                entry.reasoning,
                entry.reasoning_options.as_ref(),
                reasoning.token_budget_support(),
                reasoning.efforts,
            ),
            input: entry
                .input
                .iter()
                .filter_map(|modality| modality_from_str(modality))
                .collect(),
            // The generator only emits tool-calling text models, so every
            // bundled entry is attested.
            capabilities: ModelCapabilities {
                tool_calling: CapabilitySupport::Supported,
            },
            cost: ModelCost {
                input: entry.cost.input,
                output: entry.cost.output,
                cache_read: entry.cost.cache_read,
                cache_write: entry.cost.cache_write,
                tiers: entry.cost.tiers,
            },
            context_window: entry.context_window,
            max_tokens: entry.max_tokens,
        })
        .collect()
}
