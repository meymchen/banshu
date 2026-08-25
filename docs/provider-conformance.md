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
| Kimi For Coding | Anthropic Messages | OAuth device flow; `KIMI_API_KEY` override | on/off toggle | model dependent; no explicit choice attested | model dependent | Anthropic cache breakpoints and usage |
| MiniMax (Global/CN) | Anthropic Messages | OAuth portal flow; `MINIMAX_API_KEY` override | adaptive thinking | model dependent; all choices | model dependent | Anthropic cache breakpoints and usage |

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

Tool calling and image input themselves are explicitly model dependent. Their
catalog attestations are covered by `crates/ai/tests/model_capabilities.rs`;
the matrix deliberately makes no provider-wide promise for either capability.
