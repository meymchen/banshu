# Provider conformance

This matrix records the bundled provider defaults and model overrides. "Model
dependent" means the catalog or discovery source must attest the capability for
the selected model; `Unknown` is never treated as supported. "Automatic" means
banshu sends no request-side cache extension but still normalizes cache usage
reported by the provider.

Compatibility is defined by each provider's API contract: endpoint,
authentication, accepted request fields, streamed responses and conversation
replay. Provider documentation and observed API behavior determine these
declarations. Mock tests verify request construction and response handling;
they do not establish that a live endpoint accepts a request. Live smoke tests
provide that evidence for the model and credentials used in the run.

| Provider | Protocol | Authentication | Reasoning request | Tools | Images | Prompt caching |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek | OpenAI Chat Completions | `DEEPSEEK_API_KEY` | toggle and model-attested effort; V4 Pro excludes `low` | model dependent; choice `auto`/`none` | model dependent | automatic; DeepSeek hit/miss usage |
| Z.AI | OpenAI Chat Completions | `ZAI_API_KEY` | toggle; graded effort and preserved thinking history where model-attested | model dependent; choice `auto` | model dependent | automatic |
| Moonshot AI | OpenAI Chat Completions | `MOONSHOT_API_KEY` | model dependent; K3 sends graded effort | model dependent; all choices; strict schemas | model dependent | automatic; top-level and nested cache-read usage |
| Xiaomi MiMo | OpenAI Chat Completions | `XIAOMI_API_KEY` | on/off toggle | model dependent; choice `auto`; strict schemas | model dependent | automatic |
| Kimi For Coding | Anthropic Messages | OAuth device flow; `KIMI_API_KEY` override | toggle; K3 uses adaptive thinking and `output_config.effort` | model dependent; no explicit choice attested | model dependent | Anthropic cache breakpoints (system, messages, tools; 1h TTL attested) and usage |
| MiniMax (Global/CN) | Anthropic Messages | OAuth portal flow; `MINIMAX_API_KEY` override | adaptive thinking | model dependent; all choices | model dependent | Anthropic cache breakpoints (system, messages, tools; 1h TTL attested) and usage |

Sampling on the Anthropic-compatible rows: MiniMax declares `temperature`
alongside every reasoning shape it declares (its reference marks it fully
supported and names no thinking restriction); Kimi declares none, so an
explicit temperature is refused before dispatch. OpenAI-compatible sampling
parameters use the caller-owned `StreamOptions::sampling` map and therefore
are not bundled-provider declarations.

## Regional and subscription providers

All rows below use OpenAI Chat Completions. Z.AI CN, Moonshot CN and Xiaomi
Token Plan inherit their vendor's provider defaults above. Qwen Token Plan
declares `EnableThinking`, with model-specific `EnableThinkingWithEffort` when
graded effort is attested; other OpenAI compatibility fields retain their
defaults, including no explicit tool-choice or strict-schema declaration.

| Constructor | Provider id | Endpoint | API-key variable | models.dev source |
| --- | --- | --- | --- | --- |
| `zai_coding_cn()` | `zai-coding-cn` | `https://open.bigmodel.cn/api/coding/paas/v4` | `ZAI_CODING_CN_API_KEY` | `zhipuai-coding-plan` |
| `moonshot_cn()` | `moonshot-cn` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | `moonshotai-cn` |
| `xiaomi_token_plan_cn()` | `xiaomi-token-plan-cn` | `https://token-plan-cn.xiaomimimo.com/v1` | `XIAOMI_TOKEN_PLAN_CN_API_KEY` | `xiaomi-token-plan-cn` |
| `xiaomi_token_plan_ams()` | `xiaomi-token-plan-ams` | `https://token-plan-ams.xiaomimimo.com/v1` | `XIAOMI_TOKEN_PLAN_AMS_API_KEY` | `xiaomi-token-plan-ams` |
| `xiaomi_token_plan_sgp()` | `xiaomi-token-plan-sgp` | `https://token-plan-sgp.xiaomimimo.com/v1` | `XIAOMI_TOKEN_PLAN_SGP_API_KEY` | `xiaomi-token-plan-sgp` |
| `qwen_token_plan()` | `qwen-token-plan` | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `QWEN_TOKEN_PLAN_API_KEY` | `alibaba-token-plan` |
| `qwen_token_plan_cn()` | `qwen-token-plan-cn` | `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` | `QWEN_TOKEN_PLAN_CN_API_KEY` | `alibaba-token-plan-cn` |
| `qwen_token_plan_individual()` | `qwen-token-plan-individual` | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `QWEN_TOKEN_PLAN_API_KEY` | `alibaba-token-plan`, filtered to the Individual allowlist |

Z.AI's international constructor uses `zai-coding-plan`, with `zai` supplying
API-equivalent costs where available. The CN constructor uses its own catalog;
models available in one region are not implicitly offered in the other.

## Model-specific declarations

The generated catalog and Catalog Refresh both preserve `reasoning_options`.
An effort control attests exactly its recognized values; a toggle also attests
`off`. A toggle without graded effort uses the existing `off` through `high`
convenience ladder. An explicit empty control list attests no requestable
effort. Only absent metadata falls back to the provider vocabulary below.

`Model::openai_reasoning_format` and `Model::anthropic_reasoning_format`
override the corresponding provider format for both preflight and dispatch.
Z.AI models with graded controls use `ThinkingToggleWithHistory`; Moonshot
models use `ReasoningEffort` or `ThinkingToggleOnly` according to their
published controls. Kimi models with graded controls use
`ThinkingAdaptiveWithEffort`. Kimi `k3` and `kimi-for-coding` also attest
`allow_empty_thinking_signature`, preserving unsigned thinking during
same-model replay. Other models inherit the provider's signature policy.

Deprecated entries are excluded from generated catalogs. Runtime retirement
notices are persisted in `ModelsStoreEntry::deprecated_model_ids` and suppress
baseline and Probe entries, including after offline restoration. Omission
alone does not retire an existing model.

## Complete compatibility declarations

The following provider-level defaults pin every public compatibility field, including fields
that do not affect the summary above. “Default” means the complete value shown
here, not an unspecified value:

- `OpenAiCompat::default()` is session affinity `None`, cache retention
  `Short`, no required assistant `reasoning_content`, reasoning format
  `Unsupported`, no tool choices, non-strict tool schemas, no provider effort
  vocabulary, streamed usage enabled, output token field `MaxTokens`, stream
  termination `Strict`, no tool-result names, and no empty assistant separator.
- `AnthropicCompat::default()` disallows empty thinking signatures, sends no
  session-affinity header, has cache retention `Short`, no tool-definition
  cache control, reasoning format `Unsupported`, temperature `Unsupported`, no
  tool choices, non-strict tool schemas, and no provider effort vocabulary.

| Provider | Complete OpenAI compatibility | Complete Anthropic compatibility |
| --- | --- | --- |
| DeepSeek | Default except required assistant `reasoning_content`; `ThinkingToggle`; efforts `off`, `low`, `high`, `max`; tool choice `auto`/`none` | Default |
| Z.AI | Default except `ThinkingToggleOnly`; tool choice `auto` | Default |
| Moonshot AI | Default except an explicitly empty effort vocabulary; all tool choices; strict tool schemas | Default |
| Xiaomi MiMo | Default except `ThinkingToggleOnly`; tool choice `auto`; strict tool schemas | Default |
| Kimi For Coding | Default | Default except cache retention `Long`; tool-definition cache control enabled; `ThinkingToggle` |
| MiniMax Global | Default | Default except cache retention `Long`; tool-definition cache control enabled; `ThinkingAdaptive`; temperature `WithReasoning`; all tool choices |
| MiniMax CN | Default | Default except cache retention `Long`; tool-definition cache control enabled; `ThinkingAdaptive`; temperature `WithReasoning`; all tool choices |

`crates/ai/tests/provider_conformance.rs` compares each complete struct value,
so adding a compatibility field or changing a bundled declaration fails the
frozen matrix until this table and its expectation are deliberately updated.
Model overrides are covered separately by `upstream_compat.rs` and
`upstream_reasoning_replay.rs`; regional credentials, catalogs and Qwen request
shapes are covered by `upstream_providers.rs`.

## Automated evidence

Every fixed promise above is exercised without live credentials:

- Protocol, provider identity, endpoints, and API-key environment names:
  `crates/ai/tests/provider_conformance.rs`.
- Reasoning declarations and exact request shapes:
  `crates/ai/tests/provider_conformance.rs`,
  `crates/ai/tests/openai_reasoning_requests.rs`, and
  `crates/ai/tests/anthropic_reasoning_requests.rs`.
- Tool-choice and strict-schema declarations and wire shapes:
  `crates/ai/tests/provider_conformance.rs` and
  `crates/ai/tests/tool_choice.rs`.
- OAuth construction, API-key override, login, refresh, and logout:
  `crates/ai/tests/kimi_oauth.rs`, `crates/ai/tests/minimax_oauth.rs`, and
  `crates/ai/tests/oauth_lifecycle.rs`.
- Image gating and both protocol encodings:
  `crates/ai/tests/user_images.rs` and `crates/ai/tests/tool_result_images.rs`.
- OpenAI/DeepSeek/Moonshot and Anthropic cache request/usage shapes:
  `crates/ai/tests/openai_prompt_caching.rs` and
  `crates/ai/tests/anthropic_prompt_caching.rs`.
- OpenAI-compatible cache-routing policies (session affinity and the
  long-retention attestation — no provider in the matrix declares either):
  `crates/ai/tests/provider_conformance.rs` and
  `crates/ai/tests/openai_prompt_caching.rs`.
- Anthropic-compatible cache policies (the one-hour TTL attestation and
  tool-definition cache control — Kimi and MiniMax declare both, every other
  provider keeps the undeclared defaults):
  `crates/ai/tests/provider_conformance.rs` and
  `crates/ai/tests/anthropic_prompt_caching.rs`.
- Anthropic-compatible temperature declarations (MiniMax attests temperature
  alongside every reasoning shape it declares; every other provider keeps the
  undeclared default, which refuses an explicit temperature before dispatch):
  `crates/ai/tests/provider_conformance.rs` and
  `crates/ai/tests/anthropic_temperature.rs`.
- OpenAI-compatible request envelopes (streamed-usage request and the
  output-token field carrying the Output Budget — every bundled provider
  keeps the default: usage requested, `max_tokens`):
  `crates/ai/tests/provider_conformance.rs` and
  `crates/ai/tests/openai_request_envelope.rs`.
- OpenAI-compatible stream termination (every bundled provider keeps the
  strict default: a bare EOF without `[DONE]` or `finish_reason` is a dropped
  connection; declared clean-EOF completion and its failure modes):
  `crates/ai/tests/provider_conformance.rs` and
  `crates/ai/tests/openai_completions_termination.rs`.
- OpenAI-compatible tool-history declarations (every bundled provider keeps
  names and separators disabled): `crates/ai/tests/provider_conformance.rs`
  and `crates/ai/tests/openai_tool_history.rs`.
- OpenAI-compatible sampling controls and reserved-key protection:
  `crates/ai/tests/openai_sampling.rs`.
- Observer/wire equality after cache, sampling, combined reasoning/temperature,
  and header transforms: `crates/ai/tests/openai_prompt_caching.rs`,
  `crates/ai/tests/openai_sampling.rs`, and
  `crates/ai/tests/anthropic_temperature.rs`. These compare the observer's
  redacted payload and headers with the final values recorded by local HTTP
  servers; credentials are asserted present only on the wire.

Tool calling and image input themselves are explicitly model dependent. Their
catalog attestations are covered by `crates/ai/tests/model_capabilities.rs`;
the matrix deliberately makes no provider-wide promise for either capability.

## Custom OpenAI-compatible reasoning declarations

Bundled-provider request bodies remain frozen by the matrix above. A custom
OpenAI-compatible provider may additionally declare either of these wire
formats:

| Declaration | Enabled request | Disabled request | Optional values |
| --- | --- | --- | --- |
| `OpenAiReasoningFormat::EnableThinking` | top-level `enable_thinking: true` | top-level `enable_thinking: false` | none |
| `OpenAiReasoningFormat::ChatTemplateKwargs(..)` | declared boolean and/or effort keyword inside `chat_template_kwargs` | declared boolean becomes `false`; an effort-only declaration sends `"none"` | an explicit, model-attested token budget |

`OpenAiChatTemplateKwargs` accepts keyword names only for the typed enabled
state and effort values. Budget names are the closed
`OpenAiReasoningBudgetField` enum:

- `thinking_token_budget`
- `thinking_budget`
- `thinking_budget_tokens`

Empty declarations, duplicate keyword destinations, and declarations that
cannot express an explicit disabled state fail `ProviderBuilder::build`.
Unsupported efforts or budgets, a budget paired with `Off`, and a budget that
does not fit below the resolved Output Budget fail in-band before HTTP. The
complete local-server matrix is in
`crates/ai/tests/openai_reasoning_requests.rs`; construction failures are in
`crates/ai/tests/extension_seams.rs`.
