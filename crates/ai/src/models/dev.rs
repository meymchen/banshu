//! Runtime parsing of models.dev `api.json` — the catalog-refresh layer of
//! dynamic discovery. Mirrors the shape the `xtask` generator consumes at
//! build time.

use serde_json::Value;

use crate::types::{ApiKind, Modality, Model, ModelCost};

/// The models.dev entries for `models_dev_id`, stamped with the owning
/// provider's id, base URL, and wire protocol. `None` if the key is missing
/// or malformed.
pub(crate) fn models_from_api_json(
    data: &Value,
    models_dev_id: &str,
    provider_id: &str,
    base_url: &str,
    api: ApiKind,
) -> Option<Vec<Model>> {
    let models = data.get(models_dev_id)?.get("models")?.as_object()?;
    Some(
        models
            .iter()
            .map(|(id, entry)| parse_model(id, entry, provider_id, base_url, api))
            .collect(),
    )
}

fn parse_model(id: &str, entry: &Value, provider_id: &str, base_url: &str, api: ApiKind) -> Model {
    let cost = &entry["cost"];
    Model {
        id: id.to_string(),
        name: entry["name"].as_str().unwrap_or(id).to_string(),
        api,
        provider: provider_id.to_string(),
        base_url: base_url.to_string(),
        reasoning: entry["reasoning"].as_bool().unwrap_or(false),
        input: entry["modalities"]["input"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(super::modality_from_str)
                    .collect()
            })
            .unwrap_or_else(|| vec![Modality::Text]),
        cost: ModelCost {
            input: cost["input"].as_f64().unwrap_or(0.0),
            output: cost["output"].as_f64().unwrap_or(0.0),
            cache_read: cost["cache_read"].as_f64().unwrap_or(0.0),
            cache_write: cost["cache_write"].as_f64().unwrap_or(0.0),
        },
        context_window: entry["limit"]["context"].as_u64().unwrap_or(0) as u32,
        max_tokens: entry["limit"]["output"].as_u64().unwrap_or(0) as u32,
    }
}
