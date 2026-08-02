//! Runtime parsing of models.dev `api.json` — the catalog-refresh layer of
//! dynamic discovery. Field mapping and capability rules live in
//! [`crate::models_dev`], shared with the `xtask` catalog generator; this
//! module only stamps provider identity onto the parsed entries.

use serde_json::Value;

use crate::types::{ApiKind, CapabilitySupport, Model, ModelCapabilities};

/// The models.dev entries for `models_dev_id`, stamped with the owning
/// provider's id, base URL, wire protocol, and the token-budget support its
/// declared reasoning request shape carries. `None` if the key is missing or
/// malformed.
pub(crate) fn models_from_api_json(
    data: &Value,
    models_dev_id: &str,
    provider_id: &str,
    base_url: &str,
    api: ApiKind,
    reasoning_token_budget: CapabilitySupport,
) -> Option<Vec<Model>> {
    let parsed = crate::models_dev::models_from_api_json(data, models_dev_id)?;
    Some(
        parsed
            .into_iter()
            .map(|entry| Model {
                id: entry.id,
                name: entry.name,
                api,
                provider: provider_id.to_string(),
                base_url: base_url.to_string(),
                headers: Default::default(),
                reasoning: crate::models_dev::reasoning_capability(
                    entry.reasoning,
                    reasoning_token_budget,
                ),
                input: entry.input,
                capabilities: ModelCapabilities {
                    tool_calling: entry.tool_calling,
                },
                cost: entry.cost,
                context_window: entry.context_window,
                max_tokens: entry.max_tokens,
            })
            .collect(),
    )
}
