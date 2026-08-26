//! Providers: identity, models, auth, default headers, compat quirks, and the
//! protocol adapters their models speak. Per-vendor constructors (DeepSeek,
//! Z.AI, …) delegate to [`ProviderBuilder`]; custom providers — local servers,
//! mixed-protocol gateways, third-party protocols — are built through
//! [`Provider::builder`] directly.

mod builder;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub use builder::ProviderBuilder;

use crate::api::anthropic_messages::AnthropicMessages;
use crate::api::openai_completions::OpenAiCompletions;
use crate::api::{ProtocolAdapter, api_name};
use crate::auth::{Auth, AuthResolver, ProviderHeaders};
use crate::discovery::{RefreshEntry, RefreshOutcome};
use crate::options::StreamOptions;
use crate::stream::MessageStream;
use crate::types::{ApiKind, CapabilitySupport, Context, Model, ReasoningEffort, ToolChoice};

/// The effort levels DeepSeek's chat-completions reference documents for
/// `reasoning_effort`: `low`, `high`, and `max`, plus
/// [`Off`](ReasoningEffort::Off), which rides the `thinking` toggle rather than
/// the effort string.
///
/// `medium` and `xhigh` are deliberately absent even though the endpoint
/// accepts them: it accepts them by *mapping them onto `high`* "for
/// compatibility". Attesting a level that silently becomes a different one
/// would move the clamp this crate refuses to perform onto the server, where
/// the caller cannot see it — the preflight would pass, the request would
/// succeed, and the answer would come back at an effort nobody asked for.
/// Refusing says so. `minimal` appears nowhere in the reference at all.
const DEEPSEEK_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::Off,
    ReasoningEffort::Low,
    ReasoningEffort::High,
    ReasoningEffort::Max,
];

/// A provider whose endpoint has no reasoning request field documents no
/// effort level either — not even [`Off`](ReasoningEffort::Off), since there is
/// nothing to send it on. Distinct from declaring nothing, which falls back to
/// the baseline ladder.
const NO_REASONING_EFFORTS: &[ReasoningEffort] = &[];

/// The session-affinity routing shape an OpenAI-compatible endpoint accepts.
///
/// A stable session id
/// ([`StreamOptions::session_id`](crate::StreamOptions::session_id)) lets an
/// endpoint route a conversation's traffic onto the same prompt cache. The
/// policy is closed over the supported shapes: exactly the body field and
/// headers the selected variant names receive the session id, and an
/// undeclared endpoint receives none of them. Every provider states its own —
/// nothing is inferred from a base URL or a model id.
///
/// Routing is a cache concern, so a
/// [`Disabled`](crate::CacheRetention::Disabled) retention request suppresses
/// it entirely. It never adds, removes, or rewrites credential headers — the
/// routing fields below share no name with them, and they join the request at
/// the lowest header layer, below every auth layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiSessionAffinity {
    /// Send no session-routing field or header. The default, because an
    /// unconfigured endpoint attests nothing.
    #[default]
    None,
    /// Route by OpenAI's `prompt_cache_key` request-body field, carrying the
    /// session id clamped to the field's 64-character limit.
    PromptCacheKey,
    /// Route by the header trio `session_id`, `x-client-request-id`, and
    /// `x-session-affinity`, each carrying the session id verbatim.
    SessionAffinityHeaders,
}

/// The prompt-cache retention an OpenAI-compatible endpoint accepts.
///
/// Declared independently from session affinity ([`OpenAiSessionAffinity`]):
/// an endpoint may route cache traffic without accepting a retention field,
/// and the two policies compose freely. Every provider states its own —
/// nothing is inferred from a base URL or a model id.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiCacheRetention {
    /// Only the endpoint's normal cache behavior. A
    /// [`Short`](crate::CacheRetention::Short) request — or no retention
    /// preference at all — goes out in the endpoint's normal shape, while an
    /// explicit [`Long`](crate::CacheRetention::Long) request is refused
    /// in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request. The default, because an unconfigured endpoint
    /// attests nothing.
    #[default]
    Short,
    /// The provider attests its endpoint honours OpenAI's
    /// `prompt_cache_retention: "24h"` request field, so an explicit
    /// [`Long`](crate::CacheRetention::Long) request emits it instead of
    /// being refused.
    Long,
}

/// The prompt-cache retention an Anthropic-compatible endpoint accepts.
///
/// Declared independently from tool-definition cache control
/// ([`AnthropicCompat::tool_cache_control`]): an endpoint may cache tool
/// definitions without accepting a longer TTL, and the two policies compose
/// freely. Every provider states its own — nothing is inferred from a base
/// URL or a model id.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnthropicCacheRetention {
    /// Only the endpoint's normal ephemeral breakpoints. A
    /// [`Short`](crate::CacheRetention::Short) request — or no retention
    /// preference at all — goes out as `cache_control: { "type": "ephemeral" }`,
    /// while an explicit [`Long`](crate::CacheRetention::Long) request is
    /// refused in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request. The default, because an unconfigured endpoint
    /// attests nothing.
    #[default]
    Short,
    /// The provider attests its endpoint honours Anthropic's one-hour
    /// cache-control TTL, so an explicit [`Long`](crate::CacheRetention::Long)
    /// request emits `cache_control: { "type": "ephemeral", "ttl": "1h" }`
    /// on every breakpoint instead of being refused.
    ///
    /// Declared by [`Provider::kimi`] and [`Provider::minimax`].
    Long,
}

/// The `temperature` support an Anthropic-compatible endpoint accepts.
///
/// Declared independently from the reasoning request shape
/// ([`AnthropicCompat::reasoning_format`]): an endpoint may accept a
/// `temperature` request field without accepting one alongside an enabled
/// `thinking` request — Anthropic's own extended-thinking reference fixes
/// sampling while thinking runs — and the two policies compose freely. Every
/// provider states its own — nothing is inferred from a base URL or a model
/// id.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnthropicTemperature {
    /// The endpoint takes no `temperature` request field. An explicit
    /// temperature is refused in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request — never silently dropped to make the request
    /// succeed. An omitted temperature leaves the payload untouched either
    /// way. The default, because an unconfigured endpoint attests nothing.
    #[default]
    Unsupported,
    /// The endpoint accepts `temperature`, but not alongside an enabled
    /// reasoning request — the pairing its reference rules out. An explicit
    /// temperature on a request with no reasoning option, or with
    /// [`ReasoningEffort::Off`](crate::ReasoningEffort) (which disables
    /// reasoning outright), goes out exactly as given; one alongside an
    /// enabled reasoning request is refused in-band with
    /// [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) before
    /// any HTTP request.
    WithoutReasoning,
    /// The endpoint accepts `temperature` alongside every reasoning shape it
    /// declares. An explicit temperature always goes out exactly as given.
    ///
    /// Declared by [`Provider::minimax`].
    WithReasoning,
}

/// A documented token-budget keyword accepted by open-model chat templates.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiReasoningBudgetField {
    /// `thinking_token_budget`.
    ThinkingTokenBudget,
    /// `thinking_budget`.
    ThinkingBudget,
    /// `thinking_budget_tokens`.
    ThinkingBudgetTokens,
}

impl OpenAiReasoningBudgetField {
    /// The exact JSON field name put on the wire.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ThinkingTokenBudget => "thinking_token_budget",
            Self::ThinkingBudget => "thinking_budget",
            Self::ThinkingBudgetTokens => "thinking_budget_tokens",
        }
    }
}

/// Typed substitutions inside a `chat_template_kwargs` reasoning declaration.
///
/// Each optional string is the name of a keyword inside that object. The
/// adapter owns the object and supplies only the corresponding typed request
/// value: a boolean enabled state, the requested effort string, or the exact
/// token budget. Callers cannot name an outer request path or supply arbitrary
/// JSON values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenAiChatTemplateKwargs {
    /// Keyword receiving `true` for an enabled request and `false` for an
    /// explicit [`ReasoningEffort::Off`] request.
    pub enable_thinking: Option<&'static str>,
    /// Keyword receiving the requested effort. When no enabled-state keyword
    /// is declared, `Off` is spelled `"none"`; otherwise the boolean keyword
    /// alone carries the disabled state.
    pub reasoning_effort: Option<&'static str>,
    /// Closed keyword name receiving an explicitly requested token budget.
    pub token_budget: Option<OpenAiReasoningBudgetField>,
}

impl OpenAiChatTemplateKwargs {
    fn validate(self) -> Result<(), String> {
        let mut names = Vec::new();
        if let Some(name) = self.enable_thinking {
            names.push(name);
        }
        if let Some(name) = self.reasoning_effort {
            names.push(name);
        }
        if let Some(field) = self.token_budget {
            names.push(field.as_str());
        }
        if names.is_empty() {
            return Err("chat_template_kwargs reasoning declaration carries no values".into());
        }
        if let Some(name) = names.iter().find(|name| name.trim().is_empty()) {
            return Err(format!(
                "chat_template_kwargs reasoning keyword `{name}` must not be empty"
            ));
        }
        for (index, name) in names.iter().enumerate() {
            if names[index + 1..].contains(name) {
                return Err(format!(
                    "chat_template_kwargs reasoning keyword `{name}` has contradictory substitutions"
                ));
            }
        }
        if self.enable_thinking.is_none() && self.reasoning_effort.is_none() {
            return Err(
                "chat_template_kwargs reasoning declaration cannot disable reasoning; declare an enabled-state or effort keyword"
                    .into(),
            );
        }
        Ok(())
    }
}

/// The reasoning request shape an OpenAI-compatible endpoint accepts.
///
/// Every provider states its own — nothing is inferred from a base URL or a
/// model id. A request the declared shape cannot carry is refused before
/// dispatch rather than sent in a shape the endpoint would ignore or reject.
/// Each variant describes wire behavior rather than naming a provider.
/// [`ReasoningEffort::Off`] is always an explicit disabling request, never the
/// absence of a field.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiReasoningFormat {
    /// The endpoint takes no reasoning request field at all. Any reasoning
    /// request against this provider is refused; a model that reasons on its
    /// own still streams its thinking back. The default, because an
    /// unconfigured endpoint attests nothing.
    ///
    /// Declared by [`Provider::moonshot`].
    #[default]
    Unsupported,
    /// A top-level `reasoning_effort: "<effort>"` string and nothing else.
    /// `Off` sends `reasoning_effort: "none"`, the documented disabling value
    /// of this shape.
    ReasoningEffort,
    /// A `thinking: { "type": "enabled" | "disabled" }` toggle carrying a
    /// top-level `reasoning_effort: "<effort>"` when enabled. `Off` sends
    /// `thinking: { "type": "disabled" }` alone — the toggle is what disables
    /// reasoning here, so no effort string rides along.
    ///
    /// Declared by [`Provider::deepseek`].
    ThinkingToggle,
    /// The same toggle and *only* the toggle: no effort string rides along in
    /// either direction, so any level above `Off` reads as
    /// `thinking: { "type": "enabled" }` and `Off` as
    /// `thinking: { "type": "disabled" }`.
    ///
    /// Declared by [`Provider::zai`] and [`Provider::xiaomi`]. Refusing
    /// `Medium` here would make the provider unusable for a caller who simply
    /// wants reasoning on, so the ladder collapses onto the toggle instead —
    /// which is also why these providers declare no
    /// [`reasoning_efforts`](OpenAiCompat::reasoning_efforts) vocabulary:
    /// with no effort field on the wire, there is no vocabulary to name.
    ThinkingToggleOnly,
    /// A top-level `enable_thinking: true | false` boolean. Every enabled
    /// effort maps to `true`; [`ReasoningEffort::Off`] maps to `false`.
    EnableThinking,
    /// Typed reasoning substitutions nested under `chat_template_kwargs`.
    ChatTemplateKwargs(OpenAiChatTemplateKwargs),
}

impl OpenAiReasoningFormat {
    /// Whether this shape carries an explicit reasoning token budget.
    /// Only a `chat_template_kwargs` declaration naming a budget field does.
    pub const fn accepts_token_budget(self) -> bool {
        matches!(
            self,
            Self::ChatTemplateKwargs(OpenAiChatTemplateKwargs {
                token_budget: Some(_),
                ..
            })
        )
    }

    /// Whether the endpoint declares a reasoning request shape at all.
    pub const fn is_declared(self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    pub(crate) fn validate(self) -> Result<(), String> {
        match self {
            Self::ChatTemplateKwargs(kwargs) => kwargs.validate(),
            _ => Ok(()),
        }
    }
}

/// The reasoning request shape an Anthropic-compatible endpoint accepts. Like
/// [`OpenAiReasoningFormat`], always declared and never inferred, and named for
/// the fields it puts on the wire rather than for a vendor.
///
/// Every shape spells "do not reason" the same way —
/// `thinking: { "type": "disabled" }` — because that is the value all three
/// references document; they differ only in how they say "reason".
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnthropicReasoningFormat {
    /// The endpoint takes no `thinking` request field. Any reasoning request
    /// against this provider is refused; a model that reasons on its own still
    /// streams its thinking back. The default, because an unconfigured
    /// endpoint attests nothing.
    #[default]
    Unsupported,
    /// The `thinking: { "type": "enabled" }` toggle and nothing else: no
    /// budget, no effort, so any level above [`ReasoningEffort::Off`] reads as
    /// "enabled".
    ///
    /// Declared by [`Provider::kimi`].
    ThinkingToggle,
    /// Anthropic's `thinking: { "type": "enabled", "budget_tokens": N }`,
    /// where effort is expressed as a token budget. The budget shares the
    /// request's `max_tokens` with the answer, so one that does not fit under
    /// it is refused by the [reasoning
    /// preflight](crate::api) before dispatch.
    ///
    /// No vendor banshu bundles declares this shape; a caller pointing
    /// [`Provider::anthropic_compatible`] at an endpoint that documents
    /// `budget_tokens` declares it themselves.
    ThinkingBudget,
    /// Anthropic's adaptive shape, `thinking: { "type": "adaptive" }`, which
    /// hands the model the decision and takes neither a budget nor an effort.
    ///
    /// Declared by [`Provider::minimax`].
    ThinkingAdaptive,
}

impl AnthropicReasoningFormat {
    /// Whether this shape carries an explicit reasoning token budget.
    pub const fn accepts_token_budget(self) -> bool {
        matches!(self, Self::ThinkingBudget)
    }

    /// Whether the endpoint declares a reasoning request shape at all.
    pub const fn is_declared(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// A provider's declared reasoning request shape, reduced to the facts that
/// don't depend on which protocol declared it.
///
/// This is the only place in the crate that matches on [`ApiKind`] to pick a
/// reasoning format: the reasoning preflight and the model-stamping path both
/// come through here, so they can never disagree about what a provider
/// declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeclaredReasoning {
    /// Whether the endpoint takes any reasoning request field.
    pub(crate) declared: bool,
    /// Whether that field carries an explicit token budget.
    pub(crate) token_budget: bool,
    /// The effort levels the endpoint documents, or `None` when the provider
    /// names no vocabulary of its own.
    pub(crate) efforts: Option<&'static [ReasoningEffort]>,
}

impl DeclaredReasoning {
    /// What the provider declares for the protocol `api` speaks. Routing goes
    /// by the *model's* protocol, so a mixed-protocol provider answers for
    /// whichever side the request is headed to.
    pub(crate) fn of(api: ApiKind, openai: OpenAiCompat, anthropic: AnthropicCompat) -> Self {
        match api {
            ApiKind::OpenAiCompletions => Self {
                declared: openai.reasoning_format.is_declared(),
                token_budget: openai.reasoning_format.accepts_token_budget(),
                efforts: openai.reasoning_efforts,
            },
            ApiKind::AnthropicMessages => Self {
                declared: anthropic.reasoning_format.is_declared(),
                token_budget: anthropic.reasoning_format.accepts_token_budget(),
                efforts: anthropic.reasoning_efforts,
            },
        }
    }

    /// The budget capability stamped onto the models a provider serves. A
    /// property of the endpoint, so it is attested either way — never left
    /// `Unknown`.
    pub(crate) fn token_budget_support(self) -> CapabilitySupport {
        if self.token_budget {
            CapabilitySupport::Supported
        } else {
            CapabilitySupport::Unsupported
        }
    }
}

/// The tool choices an endpoint accepts in a request.
///
/// Every provider states its own — nothing is inferred from a base URL or a
/// model id, and the default declares nothing: an unconfigured endpoint
/// attests nothing, so any explicit [`ToolChoice`] is refused before dispatch
/// rather than sent to an endpoint that might ignore or reject it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolChoiceSupport {
    /// The provider default, made explicit: the model decides whether and
    /// which tool to call.
    pub auto: bool,
    /// Forbid tool calls even though tools are offered.
    pub none: bool,
    /// Require at least one tool call, any tool.
    pub required: bool,
    /// Require one specific tool, named exactly.
    pub named: bool,
}

impl ToolChoiceSupport {
    /// No tool choice is expressible — the default.
    pub const NONE: Self = Self {
        auto: false,
        none: false,
        required: false,
        named: false,
    };

    /// Every tool choice is expressible.
    pub const ALL: Self = Self {
        auto: true,
        none: true,
        required: true,
        named: true,
    };

    /// Whether `choice` can be expressed by the endpoint this was declared for.
    pub fn supports(&self, choice: &ToolChoice) -> bool {
        match choice {
            ToolChoice::Auto => self.auto,
            ToolChoice::None => self.none,
            ToolChoice::Required => self.required,
            ToolChoice::Named(_) => self.named,
        }
    }

    /// The names of the declared choices, for the rejection detail — the
    /// caller learns what would have worked, not just what did not.
    pub(crate) fn supported_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.auto {
            names.push("auto");
        }
        if self.none {
            names.push("none");
        }
        if self.required {
            names.push("required");
        }
        if self.named {
            names.push("named");
        }
        names
    }
}

/// The standard output-token field that carries a request's resolved Output
/// Budget on an OpenAI-compatible endpoint.
///
/// Every provider states its own — nothing is inferred from a base URL or a
/// model id. The policy is closed over the two standard fields: exactly the
/// selected one carries the budget, and the other is absent from the payload.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiOutputTokenField {
    /// `max_tokens` — the long-standing chat-completions field. The default,
    /// matching the request bodies bundled providers have always sent.
    #[default]
    MaxTokens,
    /// `max_completion_tokens` — the field OpenAI's own newer models take in
    /// place of `max_tokens`.
    MaxCompletionTokens,
}

/// What a bare end of stream means on an OpenAI-compatible endpoint.
///
/// The OpenAI wire terminator is `data: [DONE]`, and a `finish_reason`-bearing
/// chunk also terminates formally — some compatible servers close right after
/// it without sending `[DONE]`. This policy answers only for an EOF with
/// neither. Every provider states its own — nothing is inferred from a base
/// URL or a model id.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiStreamTermination {
    /// A bare EOF is a dropped connection, surfaced as an interrupted-stream
    /// failure. The default, because an unconfigured endpoint attests
    /// nothing.
    #[default]
    Strict,
    /// The provider attests its endpoint closes the connection only after the
    /// final chunk, so a clean EOF completes a structurally finished response
    /// without `[DONE]` or `finish_reason`.
    ///
    /// Structural finish is checked before the attestation is trusted: at
    /// least one content block must have started (an empty stream is no
    /// response at all — indistinguishable from a drop before the first
    /// chunk), and every streamed tool call's accumulated arguments must form
    /// complete JSON (an argument-less call counts). Text and thinking carry
    /// no wire terminator of their own, so a chunk cut mid-event stays the
    /// protocol violation it already is, and a mid-stream transport failure
    /// is never an inferred completion — both remain failures, declaration or
    /// not.
    ///
    /// An inferred completion stops as [`StopReason::ToolUse`](crate::StopReason)
    /// when the response contains tool calls and
    /// [`StopReason::Stop`](crate::StopReason) otherwise, with no raw stop
    /// reason — the wire carried none.
    CleanEofCompletion,
}

/// Endpoint quirks declared by an OpenAI-compatible provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiCompat {
    /// The session-affinity routing shape this endpoint accepts. See
    /// [`OpenAiSessionAffinity`]; the default sends no routing field or
    /// header.
    pub session_affinity: OpenAiSessionAffinity,
    /// The prompt-cache retention this endpoint accepts. See
    /// [`OpenAiCacheRetention`]; the default refuses an explicit
    /// long-retention request before dispatch.
    pub cache_retention: OpenAiCacheRetention,
    /// Every replayed assistant message must carry a `reasoning_content`
    /// field (`""` when it produced no thinking) while a reasoning model is
    /// active. DeepSeek requires this.
    pub requires_reasoning_content_on_assistant_messages: bool,
    /// The reasoning request shape this endpoint accepts.
    pub reasoning_format: OpenAiReasoningFormat,
    /// The tool choices this endpoint accepts in a request.
    ///
    /// OpenAI's own reference documents all four (`auto`, `none`, `required`,
    /// and a named tool); many compatible endpoints take only the string
    /// forms, or no `tool_choice` field at all. Declare only what the
    /// endpoint's reference offers — a choice it cannot express is refused
    /// before dispatch, never remapped onto one the caller did not ask for.
    pub tool_choice: ToolChoiceSupport,
    /// The endpoint accepts `strict: true` on function definitions
    /// (schema-constrained tool arguments). Only then does a
    /// [`strict`](crate::Tool::strict) marker reach the wire.
    pub strict_tool_schemas: bool,
    /// The effort levels this endpoint's reference documents, replacing the
    /// [`BASELINE`](crate::ReasoningCapability::BASELINE) ladder on the models
    /// this provider serves.
    ///
    /// A model metadata source says only *whether* a model reasons — never
    /// which levels it takes — so without this the ladder would be the same
    /// invented default everywhere, and a level the endpoint has never heard
    /// of would sail past the reasoning preflight into a `400`. Declaring the
    /// vocabulary narrows *and* widens: a provider documenting `max` gets it,
    /// one documenting no `minimal` refuses it.
    ///
    /// Attest only what the reference actually offers. A level the endpoint
    /// accepts but silently remaps onto another belongs *out* of the list —
    /// including it would relocate the clamp this crate refuses to perform
    /// onto the server, out of the caller's sight.
    ///
    /// Three states, and the difference between the last two matters:
    ///
    /// - `None` — the provider names no vocabulary, so its models keep the
    ///   baseline ladder. Right for an endpoint whose request shape carries no
    ///   effort field, since there a toggle answers for every level.
    /// - `Some(&[…])` — exactly these levels are requestable.
    /// - `Some(&[])` — no level is, so
    ///   [`ReasoningCapability::reasons`](crate::ReasoningCapability::reasons)
    ///   reports `false` for every model this provider serves. Right for an
    ///   endpoint with no reasoning request field at all: those models may
    ///   still stream thinking, but no effort can be *asked* of them.
    pub reasoning_efforts: Option<&'static [ReasoningEffort]>,
    /// Whether the endpoint accepts `stream_options: { "include_usage": true }`,
    /// the request for usage to arrive as a streamed chunk.
    ///
    /// The default is `true`, so usage is requested unless the endpoint
    /// explicitly opts out. Declare `false`
    /// for an endpoint that rejects or ignores `stream_options`: the adapter
    /// then omits the field entirely rather than sending an envelope the
    /// endpoint does not accept. Usage reported anyway — as a final streamed
    /// chunk — is still parsed either way.
    pub streamed_usage: bool,
    /// Which standard output-token field carries the resolved Output Budget.
    /// Exactly the selected field is sent; the other is absent.
    pub output_token_field: OpenAiOutputTokenField,
    /// What a bare end of stream means. See [`OpenAiStreamTermination`]; the
    /// default requires a formal wire terminator.
    pub stream_termination: OpenAiStreamTermination,
    /// Whether each replayed `tool` message carries the tool's `name`
    /// alongside its `tool_call_id`. Some chat templates key tool results by
    /// name; the default omits the field, matching the request bodies bundled
    /// providers have always sent.
    pub tool_result_names: bool,
    /// Whether an empty assistant message separates a run of tool results from
    /// a following user message, sent as `{ "role": "assistant", "content": "" }`.
    /// Some chat templates require strict user/assistant alternation around a
    /// tool run. The separator is inserted only at a tool-run → user boundary
    /// — never between consecutive tool results, and never twice in a row. The
    /// default inserts nothing, matching the request bodies bundled providers
    /// have always sent.
    pub empty_assistant_separator: bool,
}

impl Default for OpenAiCompat {
    /// The undeclared envelope: streamed usage is requested, `max_tokens`
    /// carries the Output Budget, tool history goes out without names or a
    /// separator, and no cache-routing field or header is sent — the request
    /// shape bundled providers have always sent.
    fn default() -> Self {
        Self {
            session_affinity: OpenAiSessionAffinity::default(),
            cache_retention: OpenAiCacheRetention::default(),
            requires_reasoning_content_on_assistant_messages: false,
            reasoning_format: OpenAiReasoningFormat::default(),
            tool_choice: ToolChoiceSupport::default(),
            strict_tool_schemas: false,
            reasoning_efforts: None,
            streamed_usage: true,
            output_token_field: OpenAiOutputTokenField::default(),
            stream_termination: OpenAiStreamTermination::default(),
            tool_result_names: false,
            empty_assistant_separator: false,
        }
    }
}

/// Endpoint quirks declared by an Anthropic-compatible provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnthropicCompat {
    /// Replay signatureless thinking as `signature: ""` instead of
    /// downgrading it to a text block. Some compatible providers emit and
    /// accept empty signatures.
    pub allow_empty_signature: bool,
    /// Send `x-session-affinity` from the session id when caching is enabled,
    /// for providers that route prompt-cache hits by replica.
    pub send_session_affinity_headers: bool,
    /// The prompt-cache retention this endpoint accepts. See
    /// [`AnthropicCacheRetention`]; the default refuses an explicit
    /// long-retention request before dispatch.
    pub cache_retention: AnthropicCacheRetention,
    /// The endpoint accepts `cache_control` on tool definitions, so the last
    /// tool carries a breakpoint that caches the whole definition list. The
    /// default attaches no tool breakpoint — an unconfigured endpoint attests
    /// nothing — while system and message breakpoints are unaffected either
    /// way: suppressing tool cache control never disables caching elsewhere.
    pub tool_cache_control: bool,
    /// The reasoning request shape this endpoint accepts.
    pub reasoning_format: AnthropicReasoningFormat,
    /// The `temperature` support this endpoint accepts. See
    /// [`AnthropicTemperature`]; the default refuses an explicit temperature
    /// before dispatch rather than sending it to an endpoint that might
    /// ignore or reject it.
    pub temperature: AnthropicTemperature,
    /// The tool choices this endpoint accepts in a request. See
    /// [`OpenAiCompat::tool_choice`] — the rule is the same on both
    /// protocols; only the wire spelling differs.
    pub tool_choice: ToolChoiceSupport,
    /// The endpoint accepts `strict: true` on tool definitions
    /// (schema-constrained tool arguments). Only then does a
    /// [`strict`](crate::Tool::strict) marker reach the wire.
    pub strict_tool_schemas: bool,
    /// The effort levels this endpoint's reference documents. See
    /// [`OpenAiCompat::reasoning_efforts`] — the rule is the same on both
    /// protocols. No Anthropic-compatible target names a vocabulary today:
    /// both express effort as a token budget instead.
    pub reasoning_efforts: Option<&'static [ReasoningEffort]>,
}

/// The in-process overlay of dynamically discovered models, layered over the
/// bundled catalog by [`Provider::models`]. Refresh failures leave it
/// untouched; it is lost when the process exits.
#[derive(Default)]
struct Overlay {
    /// models.dev catalog-refresh entries (full metadata; override + append).
    refreshed: Vec<Model>,
    /// Probe-synthesized models (bare ids; append-only, zero-means-unknown).
    probed: Vec<Model>,
}

/// A configured provider: metadata + auth + the protocol adapters its models
/// speak, at most one per [`ApiKind`].
pub struct Provider {
    id: String,
    name: String,
    base_url: String,
    auth: Auth,
    /// The primary protocol: the first adapter registered at build time. Used
    /// for catalog stamping and the list-models probe of a mixed-protocol
    /// provider; request routing always goes by `Model.api`.
    api_kind: ApiKind,
    adapters: HashMap<ApiKind, Arc<dyn ProtocolAdapter>>,
    headers: ProviderHeaders,
    /// Caller-supplied models from the builder, listed ahead of the bundled
    /// catalog and discovery overlay.
    models: Vec<Model>,
    http: reqwest::Client,
    openai_compat: OpenAiCompat,
    anthropic_compat: AnthropicCompat,
    models_dev_id: Option<String>,
    overlay: RwLock<Overlay>,
}

impl Provider {
    /// Start building a custom provider; see [`ProviderBuilder`] for the
    /// invariants [`build`](ProviderBuilder::build) enforces.
    pub fn builder(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> ProviderBuilder {
        ProviderBuilder::new(id, name, base_url)
    }

    /// Build a provider that speaks the OpenAI `chat/completions` protocol.
    ///
    /// `api_key_env` lists environment variables checked, in order, when no
    /// per-request key is supplied.
    ///
    /// # Panics
    ///
    /// Panics when `id` is empty — a caller bug the fallible
    /// [`Provider::builder`] reports as [`Error::Config`](crate::Error)
    /// instead.
    pub fn openai_compatible(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self::builder(id, name, base_url)
            .auth(Auth::api_key_env(api_key_env))
            .adapter(Arc::new(OpenAiCompletions))
            .build()
            .expect("openai_compatible: valid by construction given a non-empty id")
    }

    /// Build a provider that speaks the Anthropic `/v1/messages` protocol.
    ///
    /// `api_key_env` lists environment variables checked, in order, when no
    /// per-request key is supplied.
    ///
    /// # Panics
    ///
    /// Panics when `id` is empty — see [`openai_compatible`](Self::openai_compatible).
    pub fn anthropic_compatible(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self::builder(id, name, base_url)
            .auth(Auth::api_key_env(api_key_env))
            .adapter(Arc::new(AnthropicMessages))
            .build()
            .expect("anthropic_compatible: valid by construction given a non-empty id")
    }

    /// Set the models.dev provider key used by the catalog-refresh layer of
    /// dynamic discovery. Vendor constructors set this; custom providers
    /// without one skip the models.dev layer entirely.
    pub fn with_models_dev_id(mut self, id: impl Into<String>) -> Self {
        self.models_dev_id = Some(id.into());
        self
    }

    /// Configure the endpoint quirks of this OpenAI-compatible provider.
    pub fn with_openai_compat(mut self, compat: OpenAiCompat) -> Self {
        self.openai_compat = compat;
        self
    }

    /// Configure the endpoint quirks of this Anthropic-compatible provider.
    pub fn with_anthropic_compat(mut self, compat: AnthropicCompat) -> Self {
        self.anthropic_compat = compat;
        self
    }

    /// Replace how this provider resolves credentials. Use
    /// [`Auth::keyless`](crate::Auth::keyless) for a local server that needs no
    /// key, or [`Auth::custom`](crate::Auth::custom) for a bespoke resolver.
    /// The generic constructors default to
    /// [`Auth::api_key_env`](crate::Auth::api_key_env).
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// Replace the HTTP client every provider-owned request goes through —
    /// inference, Catalog Refresh, Probe, and the
    /// [`PreparedRequest`](crate::PreparedRequest) handed to adapters — with
    /// an application-owned one carrying the application's proxy,
    /// certificate, DNS, connection-pool, and default-header policy.
    ///
    /// An OAuth session created by a vendor constructor
    /// ([`minimax`](Self::minimax), [`kimi`](Self::kimi)) captured the
    /// previous client at construction and is not retargeted; to give the
    /// credential lifecycle the same client, construct it through
    /// [`OAuthSession::new`](crate::OAuthSession::new) yourself.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = client;
        self
    }

    /// DeepSeek — OpenAI-compatible, `DEEPSEEK_API_KEY`.
    ///
    /// Reasoning: a `thinking` toggle plus `reasoning_effort`. Tool choice:
    /// `auto` and `none` only — the chat-completions reference names no other
    /// values, and the named/required forms are documented only for DeepSeek's
    /// Responses API, a different surface. Strict tool schemas live behind a
    /// beta base URL, so they are not declared here.
    pub fn deepseek() -> Self {
        Self::openai_compatible(
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com",
            ["DEEPSEEK_API_KEY"],
        )
        .with_openai_compat(OpenAiCompat {
            requires_reasoning_content_on_assistant_messages: true,
            reasoning_format: OpenAiReasoningFormat::ThinkingToggle,
            reasoning_efforts: Some(DEEPSEEK_REASONING_EFFORTS),
            tool_choice: ToolChoiceSupport {
                auto: true,
                none: true,
                ..ToolChoiceSupport::NONE
            },
            ..OpenAiCompat::default()
        })
        .with_models_dev_id("deepseek")
    }

    /// Z.AI (GLM coding plan) — OpenAI-compatible, `ZAI_API_KEY`.
    ///
    /// Reasoning: a binary `thinking` toggle, no graded effort. Tool choice:
    /// `auto` only — the only value its reference lists.
    pub fn zai() -> Self {
        Self::openai_compatible(
            "zai",
            "Z.AI",
            "https://api.z.ai/api/coding/paas/v4",
            ["ZAI_API_KEY"],
        )
        .with_openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::ThinkingToggleOnly,
            tool_choice: ToolChoiceSupport {
                auto: true,
                ..ToolChoiceSupport::NONE
            },
            ..OpenAiCompat::default()
        })
        .with_models_dev_id("zai")
    }

    /// MiniMax — Anthropic-compatible, OAuth via the frozen Coding Plan portal
    /// contract for an explicit [`MiniMaxRegion`](crate::MiniMaxRegion), with
    /// `MINIMAX_API_KEY` as an operator override.
    ///
    /// The provider is OAuth-first: `store` is the application-injected
    /// [`CredentialStore`](crate::CredentialStore) the login/refresh/logout
    /// lifecycle persists tokens through
    /// ([`MiniMaxPortalFlow`](crate::MiniMaxPortalFlow) against the region's
    /// portal), and a stored access token authenticates inference as both
    /// `Authorization: Bearer` and `x-api-key` — the two headers the MiniMax
    /// Anthropic-compatible endpoint requires. A set `MINIMAX_API_KEY`
    /// environment variable is an explicit operator choice and wins over the
    /// stored credential. The region names the hosts — CN registers as
    /// `minimax-cn` against `api.minimaxi.com`, Global as `minimax` against
    /// `api.minimax.io`; nothing is inferred from IP.
    ///
    /// Reasoning: the adaptive `thinking` shape. MiniMax's own
    /// Anthropic-compatible reference enables thinking with
    /// `thinking: { "type": "adaptive" }` and documents no `budget_tokens` and
    /// no effort field, so banshu sends neither.
    ///
    /// Its reference also states that M2.x models keep thinking even when sent
    /// `{ "type": "disabled" }`. banshu still sends the disabling value a
    /// request for [`Off`](ReasoningEffort::Off) asks for — the endpoint
    /// accepts it, and what the model then does with it is documented by
    /// MiniMax, not decided here.
    ///
    /// Tool choice: the reference declares `tool_choice` fully supported, so
    /// all four choices are declared; strict tool schemas appear nowhere in
    /// it, so they are not.
    ///
    /// Sampling: the reference marks `temperature` fully supported and names
    /// no restriction against combining it with thinking, so it is declared
    /// alongside every reasoning shape above.
    ///
    /// Caching: the one-hour cache-control TTL and tool-definition
    /// breakpoints are declared, keeping the cache shape this provider has
    /// always sent.
    pub fn minimax(region: crate::MiniMaxRegion, store: Arc<dyn crate::CredentialStore>) -> Self {
        let provider = Self::anthropic_compatible(
            region.provider_id(),
            region.name(),
            region.inference_base_url(),
            ["MINIMAX_API_KEY"],
        )
        .with_anthropic_compat(AnthropicCompat {
            cache_retention: AnthropicCacheRetention::Long,
            tool_cache_control: true,
            reasoning_format: AnthropicReasoningFormat::ThinkingAdaptive,
            temperature: AnthropicTemperature::WithReasoning,
            tool_choice: ToolChoiceSupport::ALL,
            ..AnthropicCompat::default()
        })
        .with_models_dev_id("minimax");
        let session = crate::OAuthSession::new(
            region.provider_id(),
            Arc::new(crate::MiniMaxPortalFlow::new(region)),
            store,
            provider.http.clone(),
        );
        provider.with_auth(Auth::OAuth(
            crate::OAuthAuth::new(session).with_api_key_env(["MINIMAX_API_KEY"]),
        ))
    }

    /// Moonshot AI — OpenAI-compatible, `MOONSHOT_API_KEY`.
    ///
    /// Reasoning: none on the request side — Moonshot's thinking models decide
    /// for themselves, and the endpoint takes no reasoning field, so asking
    /// for a level is refused instead of silently dropped. Its models
    /// therefore attest no level either: a thinking model whose thinking
    /// cannot be steered is not a model you can request an effort from.
    ///
    /// Tool choice: all four choices and strict tool schemas, as its chat
    /// reference documents.
    pub fn moonshot() -> Self {
        Self::openai_compatible(
            "moonshot",
            "Moonshot AI",
            "https://api.moonshot.ai/v1",
            ["MOONSHOT_API_KEY"],
        )
        .with_openai_compat(OpenAiCompat {
            reasoning_efforts: Some(NO_REASONING_EFFORTS),
            tool_choice: ToolChoiceSupport::ALL,
            strict_tool_schemas: true,
            ..OpenAiCompat::default()
        })
        .with_models_dev_id("moonshotai")
    }

    /// Kimi For Coding — Anthropic-compatible, OAuth via the RFC 8628 device
    /// authorization flow, with `KIMI_API_KEY` as an operator override.
    ///
    /// The provider is OAuth-first: `store` is the application-injected
    /// [`CredentialStore`](crate::CredentialStore) the login/refresh/logout
    /// lifecycle persists tokens through ([`KimiDeviceFlow`](crate::KimiDeviceFlow)
    /// against the fixed Kimi auth contract), and a stored access token
    /// authenticates inference as `Authorization: Bearer`. A set
    /// `KIMI_API_KEY` environment variable is an explicit operator choice and
    /// wins over the stored credential.
    ///
    /// Reasoning: the bare `thinking` toggle. Kimi's reference switches
    /// thinking with `thinking: { "type": … }` and states outright that its
    /// models take no `budget_tokens`; the graded `reasoning_effort` its newest
    /// model accepts is a top-level field of Kimi's *OpenAI-compatible*
    /// platform API, not of the coding endpoint's Anthropic shape, so no effort
    /// rides along here.
    ///
    /// Tool choice: none declared — Kimi publishes no parameter-level
    /// reference for the coding endpoint's Anthropic shape, so an explicit
    /// choice is refused rather than sent on a guess. Sampling is the same:
    /// no temperature support is declared either.
    ///
    /// Caching: the one-hour cache-control TTL and tool-definition
    /// breakpoints are declared, keeping the cache shape this provider has
    /// always sent.
    pub fn kimi(store: Arc<dyn crate::CredentialStore>) -> Self {
        let provider = Self::anthropic_compatible(
            "kimi",
            "Kimi For Coding",
            "https://api.kimi.com/coding",
            ["KIMI_API_KEY"],
        )
        .with_anthropic_compat(AnthropicCompat {
            cache_retention: AnthropicCacheRetention::Long,
            tool_cache_control: true,
            reasoning_format: AnthropicReasoningFormat::ThinkingToggle,
            ..AnthropicCompat::default()
        })
        .with_models_dev_id("kimi-for-coding");
        let session = crate::OAuthSession::new(
            "kimi",
            Arc::new(crate::KimiDeviceFlow::new()),
            store,
            provider.http.clone(),
        );
        provider.with_auth(Auth::OAuth(
            crate::OAuthAuth::new(session).with_api_key_env(["KIMI_API_KEY"]),
        ))
    }

    /// Xiaomi MiMo — OpenAI-compatible, `XIAOMI_API_KEY`.
    ///
    /// Reasoning: the `thinking` toggle alone. MiMo's own chat-completions
    /// reference switches reasoning with `thinking: { "type": … }` and
    /// documents no `reasoning_effort` field at all, so banshu sends none —
    /// third-party gateways that re-expose MiMo behind a generic OpenAI schema
    /// are not evidence about `api.xiaomimimo.com`.
    ///
    /// Tool choice: `auto` only — its reference states any other value is
    /// stripped server-side, which is a silent remap banshu refuses to send.
    /// Strict tool schemas are documented, so they are declared.
    pub fn xiaomi() -> Self {
        Self::openai_compatible(
            "xiaomi",
            "Xiaomi MiMo",
            "https://api.xiaomimimo.com/v1",
            ["XIAOMI_API_KEY"],
        )
        .with_openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::ThinkingToggleOnly,
            tool_choice: ToolChoiceSupport {
                auto: true,
                ..ToolChoiceSupport::NONE
            },
            strict_tool_schemas: true,
            ..OpenAiCompat::default()
        })
        .with_models_dev_id("xiaomi")
    }

    /// The provider id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The provider display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The provider base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The provider's primary wire protocol (the first adapter registered at
    /// build time). Request routing goes by `Model.api`, not this.
    pub fn api_kind(&self) -> ApiKind {
        self.api_kind
    }

    /// The endpoint quirks declared for this provider's OpenAI-compatible
    /// protocol, including its reasoning request shape.
    pub fn openai_compat(&self) -> OpenAiCompat {
        self.openai_compat
    }

    /// The endpoint quirks declared for this provider's Anthropic-compatible
    /// protocol, including its reasoning request shape.
    pub fn anthropic_compat(&self) -> AnthropicCompat {
        self.anthropic_compat
    }

    /// What this provider's declared reasoning request shape for `api`
    /// attests — the effort ladder and the token-budget support — as stamped
    /// onto the models it serves.
    fn declared_reasoning(&self, api: ApiKind) -> DeclaredReasoning {
        DeclaredReasoning::of(api, self.openai_compat, self.anthropic_compat)
    }

    /// The provider's models: caller-supplied models first, then the bundled
    /// catalog, then anything dynamic discovery has found — models.dev entries
    /// override same-id entries and append new ones; probe-discovered models
    /// are append-only.
    pub fn models(&self) -> Vec<Model> {
        let mut merged = self.models.clone();
        for model in crate::models::catalog_models(
            &self.id,
            &self.base_url,
            self.api_kind,
            self.declared_reasoning(self.api_kind),
        ) {
            if !merged.iter().any(|known| known.id == model.id) {
                merged.push(model);
            }
        }
        let overlay = self.overlay.read().expect("model overlay lock poisoned");
        for model in &overlay.refreshed {
            match merged.iter_mut().find(|m| m.id == model.id) {
                Some(slot) => *slot = model.clone(),
                None => merged.push(model.clone()),
            }
        }
        for model in &overlay.probed {
            if !merged.iter().any(|m| m.id == model.id) {
                merged.push(model.clone());
            }
        }
        merged
    }

    /// The models.dev provider key for the catalog-refresh layer, if any.
    pub(crate) fn models_dev_id(&self) -> Option<&str> {
        self.models_dev_id.as_deref()
    }

    /// Apply a fetched models.dev `api.json` to this provider's overlay.
    pub(crate) fn apply_models_dev(&self, data: &serde_json::Value) -> RefreshOutcome {
        let Some(key) = &self.models_dev_id else {
            return RefreshOutcome::Skipped;
        };
        match crate::models::dev::models_from_api_json(
            data,
            key,
            &self.id,
            &self.base_url,
            self.api_kind,
            self.declared_reasoning(self.api_kind),
        ) {
            Some(models) => {
                self.overlay
                    .write()
                    .expect("model overlay lock poisoned")
                    .refreshed = models;
                RefreshOutcome::Refreshed
            }
            None => RefreshOutcome::Failed(format!("models.dev has no models for `{key}`")),
        }
    }

    /// Restore the two discovery layers without changing their precedence.
    pub(crate) fn restore_overlay(&self, entry: &crate::ModelsStoreEntry) {
        let mut overlay = self.overlay.write().expect("model overlay lock poisoned");
        if !overlay.refreshed.is_empty() || !overlay.probed.is_empty() {
            return;
        }
        overlay.refreshed = entry
            .models
            .iter()
            .filter(|model| {
                model.provider == self.id && !entry.probed_model_ids.contains(&model.id)
            })
            .cloned()
            .collect();
        overlay.probed = entry
            .models
            .iter()
            .filter(|model| model.provider == self.id && entry.probed_model_ids.contains(&model.id))
            .cloned()
            .collect();
    }

    /// Probe this provider's list-models endpoint, replacing the probed layer
    /// of the overlay with zero-means-unknown models for the returned ids.
    /// Skipped without an API key; 404/405/501 means the endpoint doesn't
    /// exist. Only ids no catalog layer knows ever surface from this layer.
    pub(crate) async fn probe_models(&self) -> RefreshOutcome {
        self.probe_models_with(None).await
    }

    /// Probe this provider while respecting cooperative cancellation.
    pub(crate) async fn probe_models_with(
        &self,
        cancellation: Option<&tokio_util::sync::CancellationToken>,
    ) -> RefreshOutcome {
        let Some(api_key) = self.env_api_key() else {
            return RefreshOutcome::Skipped;
        };
        let base = self.base_url.trim_end_matches('/');
        let request = match self.api_kind {
            ApiKind::OpenAiCompletions => {
                self.http.get(format!("{base}/models")).bearer_auth(api_key)
            }
            ApiKind::AnthropicMessages => self
                .http
                .get(format!("{base}/v1/models"))
                .header("x-api-key", api_key)
                .header(
                    "anthropic-version",
                    crate::api::anthropic_messages::ANTHROPIC_VERSION,
                ),
        };
        let response = match crate::cancel::race(
            cancellation,
            request.timeout(crate::discovery::DISCOVERY_TIMEOUT).send(),
        )
        .await
        {
            Err(_) => return RefreshOutcome::Failed("cancelled".into()),
            Ok(response) => response,
        };
        let response = match response {
            Ok(response) => response,
            Err(err) => return RefreshOutcome::Failed(err.to_string()),
        };
        let status = response.status();
        if matches!(status.as_u16(), 404 | 405 | 501) {
            return RefreshOutcome::EndpointUnsupported;
        }
        if !status.is_success() {
            return RefreshOutcome::Failed(format!("list-models returned HTTP {status}"));
        }
        let listed: crate::discovery::ListModelsResponse =
            match crate::cancel::race(cancellation, response.json()).await {
                Err(_) => return RefreshOutcome::Failed("cancelled".into()),
                Ok(Ok(listed)) => listed,
                Ok(Err(err)) => return RefreshOutcome::Failed(err.to_string()),
            };
        let probed = listed
            .data
            .into_iter()
            .map(|entry| Model {
                name: entry.display_name.unwrap_or_else(|| entry.id.clone()),
                id: entry.id,
                api: self.api_kind,
                provider: self.id.clone(),
                base_url: self.base_url.clone(),
                headers: Default::default(),
                // A bare id attests nothing: no reasoning level, no budget.
                reasoning: crate::types::ReasoningCapability::none(),
                input: vec![crate::types::Modality::Text],
                // A bare id attests nothing: capabilities stay Unknown.
                capabilities: crate::types::ModelCapabilities::default(),
                cost: crate::types::ModelCost::default(),
                context_window: 0,
                max_tokens: 0,
            })
            .collect();
        self.overlay
            .write()
            .expect("model overlay lock poisoned")
            .probed = probed;
        RefreshOutcome::Refreshed
    }

    /// Snapshot the complete effective model set and retain Probe provenance.
    pub(crate) fn overlay_snapshot(&self) -> (Vec<Model>, Vec<String>) {
        let overlay = self.overlay.read().expect("model overlay lock poisoned");
        let catalog = crate::models::catalog_models(
            &self.id,
            &self.base_url,
            self.api_kind,
            self.declared_reasoning(self.api_kind),
        );
        let mut probed_model_ids = Vec::new();
        for model in &overlay.probed {
            if self
                .models
                .iter()
                .chain(&catalog)
                .chain(&overlay.refreshed)
                .any(|known| known.id == model.id)
            {
                continue;
            }
            probed_model_ids.push(model.id.clone());
        }
        drop(overlay);
        (self.models(), probed_model_ids)
    }

    /// Refresh this provider's dynamic models without a registry: fetch
    /// models.dev when a models.dev id is configured, then probe the vendor
    /// list-models endpoint. Best-effort — failures are recorded in the
    /// returned entry and never disturb previously discovered models.
    pub async fn refresh_models(&self) -> RefreshEntry {
        self.refresh_models_from(crate::discovery::MODELS_DEV_URL)
            .await
    }

    /// [`refresh_models`](Self::refresh_models) against a specific models.dev
    /// catalog URL.
    pub async fn refresh_models_from(&self, catalog_url: &str) -> RefreshEntry {
        let catalog = if self.models_dev_id.is_some() {
            match crate::discovery::fetch_models_dev(&self.http, catalog_url).await {
                Ok(data) => self.apply_models_dev(&data),
                Err(err) => RefreshOutcome::Failed(err),
            }
        } else {
            RefreshOutcome::Skipped
        };
        self.refresh_entry(catalog).await
    }

    /// Assemble this provider's report entry from an already-decided catalog
    /// outcome, running the probe layer.
    pub(crate) async fn refresh_entry(&self, catalog: RefreshOutcome) -> RefreshEntry {
        RefreshEntry {
            provider: self.id.clone(),
            catalog,
            probe: self.probe_models().await,
        }
    }

    /// The provider's HTTP client, shared with discovery fetches.
    pub(crate) fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    /// Whether this provider looks usable without further configuration:
    /// keyless providers always are, an api-key-env provider is when one of its
    /// variables is set. A custom-resolver provider reports `false` here
    /// because its resolver can only be consulted asynchronously — use
    /// [`Models::available`](crate::Models::available), which is async and
    /// consults it, for gating.
    pub fn is_available(&self) -> bool {
        self.auth.is_available()
    }

    /// Async availability check behind
    /// [`Models::available`](crate::Models::available): consults the resolver,
    /// so a custom resolver gets its say. A resolver error reads as
    /// unavailable.
    pub(crate) async fn check_available(&self) -> bool {
        self.auth.check().await.unwrap_or(false)
    }

    /// Stream a completion for `model`, routed to the adapter registered for
    /// `model.api`. Never fails synchronously — a model whose protocol this
    /// provider has no adapter for yields an in-band error; see
    /// [`MessageStream`].
    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> MessageStream {
        match self.adapters.get(&model.api) {
            Some(adapter) => crate::api::drive(
                adapter,
                &self.id,
                model,
                context,
                options,
                &self.auth,
                &self.headers,
                &self.http,
                self.openai_compat,
                self.anthropic_compat,
            ),
            None => MessageStream::immediate_error(
                &model.id,
                &self.id,
                &format!(
                    "provider `{}` has no adapter for the `{}` protocol",
                    self.id,
                    api_name(model.api),
                ),
            ),
        }
    }

    /// The OAuth session behind this provider's auth, when it has one — the
    /// handle [`Models::login`](crate::Models::login) and friends delegate to.
    pub fn oauth_session(&self) -> Option<crate::OAuthSession> {
        match &self.auth {
            Auth::OAuth(auth) => Some(auth.session().clone()),
            _ => None,
        }
    }

    /// Best-effort synchronous key lookup from the configured environment
    /// variables. Only [`Auth::api_key_env`] resolves synchronously; keyless
    /// and custom resolvers report `None` here (availability gating and the
    /// list-models probe both treat that as "no key").
    fn env_api_key(&self) -> Option<String> {
        self.auth.env_api_key()
    }
}
