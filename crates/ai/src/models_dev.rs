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

use crate::types::{
    CapabilitySupport, CostTier, Modality, ModelCost, ReasoningCapability, ReasoningEffort,
};

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
    /// Provider-published reasoning controls. Missing metadata uses the
    /// provider's legacy declaration; present metadata is authoritative.
    pub reasoning_options: Option<Value>,
    /// Retired entries are retained as tombstones during refresh.
    pub deprecated: bool,
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
        !self.deprecated
            && self.tool_calling == CapabilitySupport::Supported
            && self.input.contains(&Modality::Text)
            && self.output.contains(&Modality::Text)
    }
}

/// Map a models.dev-style `reasoning` boolean plus what the owning provider
/// declares about its endpoint onto a [`ReasoningCapability`].
///
/// Used only when model-level reasoning controls are absent. `efforts`
/// carries the provider's fallback vocabulary; `None` means it names none and
/// the model falls back to the [`BASELINE`](ReasoningCapability::BASELINE)
/// ladder, which is right for an endpoint whose request shape has no effort
/// field to constrain.
///
/// `false` attests no level at all, and a model that does not reason cannot
/// take a reasoning budget either.
///
/// Both catalog consumers use [`reasoning_from_options`], which calls this
/// fallback only when the source supplies no model-level control metadata.
pub fn reasoning_capability(
    reasoning: bool,
    token_budget: CapabilitySupport,
    efforts: Option<&[ReasoningEffort]>,
) -> ReasoningCapability {
    if !reasoning {
        return ReasoningCapability::none().with_token_budget(CapabilitySupport::Unsupported);
    }
    let ladder = match efforts {
        Some(efforts) => ReasoningCapability::new(efforts.iter().copied()),
        None => ReasoningCapability::baseline(),
    };
    ladder.with_token_budget(token_budget)
}

/// Resolve model-published controls before falling back to a provider-wide
/// vocabulary. Unknown values are never promoted to supported effort levels.
pub fn reasoning_from_options(
    reasoning: bool,
    options: Option<&Value>,
    token_budget: CapabilitySupport,
    fallback: Option<&[ReasoningEffort]>,
) -> ReasoningCapability {
    let Some(options) = options.filter(|_| reasoning) else {
        return reasoning_capability(reasoning, token_budget, fallback);
    };
    let controls: Vec<&Value> = match options {
        Value::Array(items) => items.iter().collect(),
        other => vec![other],
    };
    let mut efforts = Vec::new();
    let mut toggle = false;
    let mut graded = false;
    for control in controls {
        match control["type"].as_str() {
            Some("toggle") => toggle = true,
            Some("effort") => {
                graded = true;
                if let Some(values) = control["values"].as_array() {
                    for value in values {
                        let value = value.as_str().unwrap_or("");
                        if value == "none" {
                            efforts.push(ReasoningEffort::Off);
                        } else if let Some(effort) = ReasoningEffort::ALL
                            .into_iter()
                            .find(|e| e.as_str() == value)
                        {
                            efforts.push(effort);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if toggle {
        efforts.push(ReasoningEffort::Off);
        if !graded {
            efforts.extend(ReasoningCapability::BASELINE);
        }
    }
    ReasoningCapability::new(efforts).with_token_budget(token_budget)
}

/// Wire overrides for documented model controls on the bundled endpoints.
/// Metadata describes capabilities, not an arbitrary HTTP payload; unknown
/// sources keep the caller's provider declaration unchanged.
pub(crate) fn model_reasoning_format(
    source: &str,
    options: Option<&Value>,
) -> Option<crate::OpenAiReasoningFormat> {
    use crate::OpenAiReasoningFormat as Format;
    let options = options?;
    let controls: Vec<&Value> = match options {
        Value::Array(items) => items.iter().collect(),
        other => vec![other],
    };
    let effort = controls.iter().any(|v| v["type"] == "effort");
    let toggle = controls.iter().any(|v| v["type"] == "toggle");
    match source {
        "alibaba-token-plan"
        | "alibaba-token-plan-cn"
        | "qwen-token-plan"
        | "qwen-token-plan-cn"
        | "qwen-token-plan-individual"
            if effort =>
        {
            Some(Format::EnableThinkingWithEffort)
        }
        "zai" | "zai-coding-plan" | "zai-coding-cn" | "zhipuai-coding-plan" if effort => {
            Some(Format::ThinkingToggleWithHistory)
        }
        "moonshot" | "moonshotai" | "moonshotai-cn" | "moonshot-cn" if effort => {
            Some(Format::ReasoningEffort)
        }
        "moonshot" | "moonshotai" | "moonshotai-cn" | "moonshot-cn" if toggle => {
            Some(Format::ThinkingToggleOnly)
        }
        _ => None,
    }
}

pub(crate) fn allows_unsigned_thinking(source: &str, id: &str) -> bool {
    matches!(source, "kimi" | "kimi-for-coding") && matches!(id, "k3" | "kimi-for-coding")
}

pub(crate) fn model_anthropic_reasoning_format(
    source: &str,
    options: Option<&Value>,
) -> Option<crate::AnthropicReasoningFormat> {
    if !matches!(source, "kimi" | "kimi-for-coding") {
        return None;
    }
    let options = options?;
    let has_effort = match options {
        Value::Array(items) => items.iter().any(|item| item["type"] == "effort"),
        other => other["type"] == "effort",
    };
    has_effort.then_some(crate::AnthropicReasoningFormat::ThinkingAdaptiveWithEffort)
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
    let source = if provider_key == "qwen-token-plan-individual" {
        "alibaba-token-plan"
    } else {
        provider_key
    };
    let models = data.get(source)?.get("models")?.as_object()?;
    Some(
        models
            .iter()
            .filter(|(id, _)| model_allowed(provider_key, id))
            .map(|(id, entry)| {
                let mut model = parse_model(id, entry);
                if matches!(source, "zai-coding-plan" | "zhipuai-coding-plan")
                    && let Some(reference) = data
                        .get("zai")
                        .and_then(|p| p.get("models"))
                        .and_then(|m| m.get(id))
                    && reference.get("cost").is_some()
                {
                    model.cost = parse_model(id, reference).cost;
                }
                if matches!(source, "alibaba-token-plan" | "alibaba-token-plan-cn") {
                    model.deprecated |= id == "qwen3.8-max-preview";
                    if matches!(id.as_str(), "glm-5" | "glm-5.1") {
                        let options = model
                            .reasoning_options
                            .get_or_insert_with(|| Value::Array(Vec::new()));
                        if let Some(controls) = options.as_array_mut()
                            && !controls.iter().any(|v| v["type"] == "effort")
                        {
                            controls
                                .push(serde_json::json!({"type":"effort","values":["high","max"]}));
                        }
                    }
                }
                model
            })
            .collect(),
    )
}

// Model allowlist for QwenCloud's Individual plan.
const QWEN_INDIVIDUAL_MODELS: &[&str] = &[
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "glm-5.2",
    "qwen3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-flash",
    "qwen3.8-max",
];

pub(crate) fn model_allowed(source: &str, id: &str) -> bool {
    source != "qwen-token-plan-individual" || QWEN_INDIVIDUAL_MODELS.contains(&id)
}

fn parse_model(id: &str, entry: &Value) -> ModelsDevModel {
    let cost = &entry["cost"];
    ModelsDevModel {
        id: id.to_string(),
        name: entry["name"].as_str().unwrap_or(id).to_string(),
        reasoning: entry["reasoning"].as_bool().unwrap_or(false),
        reasoning_options: entry.get("reasoning_options").cloned(),
        deprecated: entry["status"].as_str() == Some("deprecated"),
        tool_calling: capability_from_tool_call(entry["tool_call"].as_bool()),
        input: modalities(&entry["modalities"]["input"]).unwrap_or_else(|| vec![Modality::Text]),
        output: modalities(&entry["modalities"]["output"]).unwrap_or_default(),
        cost: ModelCost {
            input: cost["input"].as_f64().unwrap_or(0.0),
            output: cost["output"].as_f64().unwrap_or(0.0),
            cache_read: cost["cache_read"].as_f64().unwrap_or(0.0),
            cache_write: cost["cache_write"].as_f64().unwrap_or(0.0),
            tiers: parse_cost_tiers(cost),
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

/// Map models.dev `cost.tiers` entries onto [`CostTier`]s. Only context-type
/// tiers with a known (non-zero) boundary are kept — a zero size is unknown
/// metadata that must never select a tier. Rates a tier omits default to 0,
/// matching how models.dev's own consumers treat partial tier entries.
fn parse_cost_tiers(cost: &Value) -> Vec<CostTier> {
    let Some(tiers) = cost["tiers"].as_array() else {
        return Vec::new();
    };
    tiers
        .iter()
        .filter(|tier| tier["tier"]["type"].as_str() == Some("context"))
        .filter_map(|tier| {
            let size = tier["tier"]["size"].as_u64()?;
            (size > 0).then(|| CostTier {
                input_tokens_above: size as u32,
                input: tier["input"].as_f64().unwrap_or(0.0),
                output: tier["output"].as_f64().unwrap_or(0.0),
                cache_read: tier["cache_read"].as_f64().unwrap_or(0.0),
                cache_write: tier["cache_write"].as_f64().unwrap_or(0.0),
            })
        })
        .collect()
}
