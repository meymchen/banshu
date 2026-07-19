//! Bundled model catalog, generated from [models.dev](https://models.dev) by
//! `cargo run -p xtask -- generate-catalog`. Catalog entries carry only model
//! metadata; the owning [`Provider`](crate::Provider) stamps on `provider`,
//! `base_url`, and the wire protocol when it lists its models.

pub(crate) mod dev;

use serde::Deserialize;

use crate::types::{ApiKind, Modality, Model, ModelCost};

/// One entry in a bundled `catalog/<provider>.json` file.
#[derive(Deserialize)]
struct CatalogModel {
    id: String,
    name: String,
    reasoning: bool,
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
}

/// Map a models.dev modality string onto the crate's [`Modality`]. Unknown
/// modalities (audio, video, …) are dropped.
pub(crate) fn modality_from_str(modality: &str) -> Option<Modality> {
    match modality {
        "text" => Some(Modality::Text),
        "image" => Some(Modality::Image),
        _ => None,
    }
}

/// Raw JSON for a provider's bundled catalog, or `None` if none is bundled.
fn raw_catalog(provider_id: &str) -> Option<&'static str> {
    Some(match provider_id {
        "deepseek" => include_str!("catalog/deepseek.json"),
        "zai" => include_str!("catalog/zai.json"),
        "minimax" => include_str!("catalog/minimax.json"),
        "moonshot" => include_str!("catalog/moonshot.json"),
        "kimi" => include_str!("catalog/kimi.json"),
        "xiaomi" => include_str!("catalog/xiaomi.json"),
        _ => return None,
    })
}

/// Build the model list for a provider from its bundled catalog, stamping each
/// model with the provider's id, base URL, and wire protocol.
pub(crate) fn catalog_models(provider_id: &str, base_url: &str, api: ApiKind) -> Vec<Model> {
    let Some(raw) = raw_catalog(provider_id) else {
        return Vec::new();
    };
    let entries: Vec<CatalogModel> = serde_json::from_str(raw).unwrap_or_default();
    entries
        .into_iter()
        .map(|entry| Model {
            id: entry.id,
            name: entry.name,
            api,
            provider: provider_id.to_string(),
            base_url: base_url.to_string(),
            reasoning: entry.reasoning,
            input: entry
                .input
                .iter()
                .filter_map(|modality| modality_from_str(modality))
                .collect(),
            cost: ModelCost {
                input: entry.cost.input,
                output: entry.cost.output,
                cache_read: entry.cost.cache_read,
                cache_write: entry.cost.cache_write,
            },
            context_window: entry.context_window,
            max_tokens: entry.max_tokens,
        })
        .collect()
}
