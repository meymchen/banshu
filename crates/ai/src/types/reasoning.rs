//! The unified reasoning contract: what a request may ask for, and what a
//! model's metadata source attests it can do.
//!
//! Three declarations decide whether a reasoning request can be honoured, and
//! none of them is guessed from a base URL or a model id:
//!
//! 1. the request — [`ReasoningOptions`] on
//!    [`StreamOptions::reasoning`](crate::StreamOptions::reasoning), or `None`
//!    for "leave reasoning alone";
//! 2. the model — [`ReasoningCapability`] on [`Model.reasoning`](crate::Model),
//!    the effort levels and token-budget support its metadata source attests;
//! 3. the provider — the reasoning request shape its endpoint accepts,
//!    declared by
//!    [`OpenAiCompat::reasoning_format`](crate::OpenAiCompat::reasoning_format)
//!    or
//!    [`AnthropicCompat::reasoning_format`](crate::AnthropicCompat::reasoning_format).
//!
//! A request none of the three can honour terminates in-band with
//! [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before any
//! HTTP request is sent — never silently clamped onto a level the caller did
//! not ask for.

use crate::types::CapabilitySupport;

/// How much reasoning a request asks a model to do, from least to most.
///
/// [`Off`](Self::Off) is an explicit request *not* to reason, which a provider
/// sends as its own disabling value. The absence of a request is
/// `StreamOptions::reasoning == None`, which leaves the payload untouched.
///
/// [`XHigh`](Self::XHigh) and [`Max`](Self::Max) sit above the ladder every
/// current metadata source attests, so they are only ever accepted by a model
/// that names them explicitly.
#[non_exhaustive]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Reason as little as the provider allows.
    Off,
    /// The provider's smallest non-zero reasoning setting.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Above `High`; only some models offer it.
    XHigh,
    /// The most reasoning a model offers.
    Max,
}

impl ReasoningEffort {
    /// The whole ladder, ascending.
    pub const ALL: [Self; 7] = [
        Self::Off,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::XHigh,
        Self::Max,
    ];

    /// The stable lowercase name, matching the serde representation and the
    /// string every provider that takes a graded effort expects.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A request-level reasoning override.
///
/// Set it on [`StreamOptions::reasoning`](crate::StreamOptions::reasoning).
/// Leaving that field `None` is not the same as requesting
/// [`ReasoningEffort::Off`]: `None` sends no reasoning field at all, while
/// `Off` asks the provider to actively disable reasoning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningOptions {
    /// How much reasoning to ask for.
    pub effort: ReasoningEffort,
    /// An explicit reasoning token budget. Only providers whose declared
    /// request shape carries one, and models that attest it, accept a budget;
    /// anywhere else the request is rejected rather than dropped.
    pub token_budget: Option<u32>,
}

impl ReasoningOptions {
    /// Request `effort` with no explicit token budget.
    pub const fn new(effort: ReasoningEffort) -> Self {
        Self {
            effort,
            token_budget: None,
        }
    }

    /// Also request an explicit reasoning token budget.
    pub const fn with_token_budget(mut self, tokens: u32) -> Self {
        self.token_budget = Some(tokens);
        self
    }
}

impl From<ReasoningEffort> for ReasoningOptions {
    fn from(effort: ReasoningEffort) -> Self {
        Self::new(effort)
    }
}

/// What a model's metadata source attests about reasoning: which effort levels
/// it accepts, and whether an explicit token budget may be requested.
///
/// This replaces a plain "supports reasoning" boolean so that a level a model
/// never offers is rejected instead of quietly becoming a different one. An
/// empty level set means the source attests nothing, and every reasoning
/// request is refused — the same honesty rule
/// [`CapabilitySupport::Unknown`] follows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningCapability {
    /// Ascending, deduplicated; see [`efforts`](Self::efforts).
    efforts: Vec<ReasoningEffort>,
    token_budget: CapabilitySupport,
}

impl ReasoningCapability {
    /// The levels attested for a model whose source says only "this model
    /// reasons": [`Off`](ReasoningEffort::Off) through
    /// [`High`](ReasoningEffort::High). [`XHigh`](ReasoningEffort::XHigh) and
    /// [`Max`](ReasoningEffort::Max) are deliberately absent — no metadata
    /// source in use attests them, and guessing would defeat the point.
    pub const BASELINE: [ReasoningEffort; 5] = [
        ReasoningEffort::Off,
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ];

    /// Nothing attested: no level is accepted and the token budget is
    /// [`CapabilitySupport::Unknown`]. The default.
    pub fn none() -> Self {
        Self::default()
    }

    /// The [`BASELINE`](Self::BASELINE) ladder, token budget still
    /// [`CapabilitySupport::Unknown`].
    pub fn baseline() -> Self {
        Self::new(Self::BASELINE)
    }

    /// Attest exactly `efforts`. Order and duplicates in the input don't
    /// matter — the set is stored ascending and deduplicated.
    pub fn new(efforts: impl IntoIterator<Item = ReasoningEffort>) -> Self {
        let mut efforts: Vec<ReasoningEffort> = efforts.into_iter().collect();
        efforts.sort_unstable();
        efforts.dedup();
        Self {
            efforts,
            token_budget: CapabilitySupport::Unknown,
        }
    }

    /// Declare whether an explicit reasoning token budget may be requested.
    pub fn with_token_budget(mut self, support: CapabilitySupport) -> Self {
        self.token_budget = support;
        self
    }

    /// The attested levels, ascending. Empty means nothing is attested.
    pub fn efforts(&self) -> &[ReasoningEffort] {
        &self.efforts
    }

    /// Whether an explicit reasoning token budget may be requested.
    pub fn token_budget(&self) -> CapabilitySupport {
        self.token_budget
    }

    /// Whether `effort` is one of the attested levels.
    pub fn supports(&self, effort: ReasoningEffort) -> bool {
        self.efforts.contains(&effort)
    }

    /// Whether the model reasons at all — that is, attests any level above
    /// [`ReasoningEffort::Off`]. A model that can only be told "off" does not.
    pub fn is_supported(&self) -> bool {
        self.efforts
            .iter()
            .any(|effort| *effort > ReasoningEffort::Off)
    }
}
