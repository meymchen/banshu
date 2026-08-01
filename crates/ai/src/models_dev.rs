//! The shared models.dev → banshu mapping: one parse of a models.dev
//! `api.json` entry, one `tool_call` → [`CapabilitySupport`] rule, one
//! agent-catalog filter. Both consumers live here so their rules can never
//! drift apart:
//!
//! - `xtask generate-catalog` parses, keeps only
//!   [`is_tool_calling_text_model`](crate::models_dev::ModelsDevModel::is_tool_calling_text_model)
//!   entries, and writes the bundled catalog.
//! - The runtime Catalog Refresh (the crate's internal `models` module) parses
//!   the same way and stamps each [`Model`](crate::Model) with the mapped
//!   capabilities; gating to tool-calling models happens in
//!   [`Models::agent_models`](crate::Models::agent_models).

use serde_json::Value;

use crate::types::{CapabilitySupport, Modality, ModelCost, ReasoningCapability};

/// A models.dev model entry mapped onto banshu's metadata vocabulary. Carries
/// no provider identity — the caller stamps `provider`/`base_url`/`api`.
#[derive(Debug, Clone)]
pub struct ModelsDevModel {
    /// The models.dev model id.
    pub id: String,
    /// Human-readable display name (falls back to the id).
    pub name: String,
    /// Whether the model supports reasoning / thinking.
    pub reasoning: bool,
    /// Tool-calling support, mapped from models.dev `tool_call`.
    pub tool_calling: CapabilitySupport,
    /// Accepted input modalities; defaults to text when models.dev omits them.
    pub input: Vec<Modality>,
    /// Produced output modalities; empty when models.dev omits them.
    pub output: Vec<Modality>,
    /// Token cost rates.
    pub cost: ModelCost,
    /// Maximum context window in tokens.
    pub context_window: u32,
    /// Maximum output tokens per response.
    pub max_tokens: u32,
}

impl ModelsDevModel {
    /// Whether the model belongs in an agent's model catalog: models.dev
    /// attests tool calling, and the model both accepts and produces text.
    pub fn is_tool_calling_text_model(&self) -> bool {
        self.tool_calling == CapabilitySupport::Supported
            && self.input.contains(&Modality::Text)
            && self.output.contains(&Modality::Text)
    }
}

/// Map a models.dev-style `reasoning` boolean plus the owning provider's
/// declared token-budget support onto a [`ReasoningCapability`].
///
/// `true` attests the [`ReasoningCapability::BASELINE`] ladder and nothing
/// more: models.dev names no effort levels, so `xhigh` and `max` stay
/// unattested rather than assumed, and the budget capability comes from the
/// provider's declared request shape. `false` attests no level at all, and a
/// model that does not reason cannot take a reasoning budget either.
///
/// Both the bundled catalog and the runtime Catalog Refresh go through here,
/// so the two can never disagree about what a `reasoning` flag means.
pub fn reasoning_capability(
    reasoning: bool,
    token_budget: CapabilitySupport,
) -> ReasoningCapability {
    if !reasoning {
        return ReasoningCapability::none().with_token_budget(CapabilitySupport::Unsupported);
    }
    ReasoningCapability::baseline().with_token_budget(token_budget)
}

/// Map models.dev `tool_call` onto [`CapabilitySupport`]: `true` → Supported,
/// `false` → Unsupported, missing → Unknown.
pub fn capability_from_tool_call(tool_call: Option<bool>) -> CapabilitySupport {
    match tool_call {
        Some(true) => CapabilitySupport::Supported,
        Some(false) => CapabilitySupport::Unsupported,
        None => CapabilitySupport::Unknown,
    }
}

/// Map a models.dev modality string onto the crate's [`Modality`]. Unknown
/// modalities (audio, video, …) are dropped.
pub fn modality_from_str(modality: &str) -> Option<Modality> {
    match modality {
        "text" => Some(Modality::Text),
        "image" => Some(Modality::Image),
        _ => None,
    }
}

/// Parse the models.dev entries for `provider_key`. `None` if the key is
/// missing or malformed.
pub fn models_from_api_json(data: &Value, provider_key: &str) -> Option<Vec<ModelsDevModel>> {
    let models = data.get(provider_key)?.get("models")?.as_object()?;
    Some(
        models
            .iter()
            .map(|(id, entry)| parse_model(id, entry))
            .collect(),
    )
}

fn parse_model(id: &str, entry: &Value) -> ModelsDevModel {
    let cost = &entry["cost"];
    ModelsDevModel {
        id: id.to_string(),
        name: entry["name"].as_str().unwrap_or(id).to_string(),
        reasoning: entry["reasoning"].as_bool().unwrap_or(false),
        tool_calling: capability_from_tool_call(entry["tool_call"].as_bool()),
        input: modalities(&entry["modalities"]["input"]).unwrap_or_else(|| vec![Modality::Text]),
        output: modalities(&entry["modalities"]["output"]).unwrap_or_default(),
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

fn modalities(value: &Value) -> Option<Vec<Modality>> {
    value.as_array().map(|items| {
        items
            .iter()
            .filter_map(Value::as_str)
            .filter_map(modality_from_str)
            .collect()
    })
}
