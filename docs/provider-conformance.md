# Provider conformance

This matrix freezes the bundled provider contract planned for 1.0. "Model
dependent" means the catalog or discovery source must attest the capability for
the selected model; `Unknown` is never treated as supported. "Automatic" means
banshu sends no request-side cache extension but still normalizes cache usage
reported by the provider.

| Provider | Protocol | Authentication | Reasoning request | Tools | Images | Prompt caching |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek | OpenAI Chat Completions | `DEEPSEEK_API_KEY` | toggle; `off`, `low`, `high`, `max` | model dependent; choice `auto`/`none` | model dependent | automatic; DeepSeek hit/miss usage |
| Z.AI | OpenAI Chat Completions | `ZAI_API_KEY` | on/off toggle | model dependent; choice `auto` | model dependent | automatic |
| Moonshot AI | OpenAI Chat Completions | `MOONSHOT_API_KEY` | no request control | model dependent; all choices; strict schemas | model dependent | automatic; Moonshot cache-read usage |
| Xiaomi MiMo | OpenAI Chat Completions | `XIAOMI_API_KEY` | on/off toggle | model dependent; choice `auto`; strict schemas | model dependent | automatic |
| Kimi For Coding | Anthropic Messages | OAuth device flow; `KIMI_API_KEY` override | on/off toggle | model dependent; no explicit choice attested | model dependent | Anthropic cache breakpoints (system, messages, tools; 1h TTL attested) and usage |
| MiniMax (Global/CN) | Anthropic Messages | OAuth portal flow; `MINIMAX_API_KEY` override | adaptive thinking | model dependent; all choices | model dependent | Anthropic cache breakpoints (system, messages, tools; 1h TTL attested) and usage |

Sampling on the Anthropic-compatible rows: MiniMax declares `temperature`
alongside every reasoning shape it declares (its reference marks it fully
supported and names no thinking restriction); Kimi declares none, so an
explicit temperature is refused before dispatch. OpenAI-compatible sampling
parameters use the caller-owned `StreamOptions::sampling` map and therefore
are not bundled-provider declarations.

## Complete compatibility declarations

The following values pin every public compatibility field, including fields
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
