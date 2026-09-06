//! Runtime parsing of models.dev `api.json` — the catalog-refresh layer of
//! dynamic discovery. Field mapping and capability rules live in
//! [`crate::models_dev`], shared with the `xtask` catalog generator; this
//! module only stamps provider identity onto the parsed entries.

use serde_json::Value;

use crate::provider::DeclaredReasoning;
use crate::types::{ApiKind, Model, ModelCapabilities};

/// The models.dev entries for `models_dev_id`, stamped with the owning
/// provider's id, base URL, wire protocol, and what its declared reasoning
/// request shape attests. `None` if the key is missing or malformed.
pub(crate) fn models_from_api_json(
    data: &Value,
    models_dev_id: &str,
    provider_id: &str,
    base_url: &str,
    api: ApiKind,
    reasoning: DeclaredReasoning,
) -> Option<Vec<Model>> {
    let parsed = crate::models_dev::models_from_api_json(data, models_dev_id)?;
    Some(
        parsed
            .into_iter()
            .filter(|entry| !entry.deprecated)
            .map(|entry| Model {
                allow_empty_thinking_signature: crate::models_dev::allows_unsigned_thinking(
                    models_dev_id,
                    &entry.id,
                ),
                id: entry.id,
                name: entry.name,
                api,
                provider: provider_id.to_string(),
                base_url: base_url.to_string(),
                headers: Default::default(),
                openai_reasoning_format: crate::models_dev::model_reasoning_format(
                    models_dev_id,
                    entry.reasoning_options.as_ref(),
                ),
                anthropic_reasoning_format: crate::models_dev::model_anthropic_reasoning_format(
                    models_dev_id,
                    entry.reasoning_options.as_ref(),
                ),
                reasoning: crate::models_dev::reasoning_from_options(
                    entry.reasoning,
                    entry.reasoning_options.as_ref(),
                    reasoning.token_budget_support(),
                    reasoning.efforts,
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
