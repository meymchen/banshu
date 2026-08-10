# banshu-ai

A Rust library for streaming chat with open-source LLM providers over the
OpenAI-completions and Anthropic-messages wire protocols.

## Language

### Model discovery

**Catalog**:
The bundled, offline model baseline for a provider — JSON generated from
models.dev at build time and compiled into the crate.
_Avoid_: model list, static models

**Catalog Refresh**:
A runtime fetch of models.dev that overrides same-id Catalog entries and adds
new ones, with full metadata.
_Avoid_: dynamic catalog, remote catalog

**Probe**:
A best-effort call to a provider's own list-models endpoint. Yields bare model
ids only; it can add models the Catalog doesn't know, never remove or change
them.
_Avoid_: model listing, discovery call

**Overlay**:
The in-process merged result of Catalog ← Catalog Refresh ← Probe that a
provider serves as its model list. Never persisted; lost when the process
exits.

**Zero-means-unknown**:
The metadata convention for Probe-synthesized models: cost, context window,
and max tokens are `0` to mean "unknown", never guessed.

**Capability Support** (`Supported` / `Unsupported` / `Unknown`):
The honesty convention for model capabilities (`ModelCapabilities`). A
capability is attested only when the metadata source says so — models.dev
`tool_call` maps true/false/missing onto the three variants, and Probe models
are always `Unknown`. `Unknown` is never presented as supported.

**Agent Models** (`Models::agent_models()`):
The subset of the model list attested as tool-calling — the safe pool for an
agent loop. The bundled Catalog is generated to contain exactly these
(tool-calling, text-in/text-out); `models()` keeps serving the full Overlay
including `Unknown` Probe models for explicit selection.
_Avoid_: supported models, tool models

**Reasoning Capability** (`ReasoningCapability`):
What a model's metadata source attests about reasoning: the effort levels it
accepts and whether an explicit token budget may be requested. Replaces a
plain "supports reasoning" boolean so an unattested level is refused rather
than quietly becoming a different one — the same honesty rule as Capability
Support. A source that says only "this model reasons" attests whichever
Reasoning Effort Vocabulary its provider declares, falling back to the baseline
ladder `off`…`high` where the provider declares none; Probe models attest
nothing.
_Avoid_: reasoning flag, thinking support

### Core (established)

**Provider**:
A configured vendor endpoint: id, base URL, env-var auth, wire protocol, and
compat quirks. Mostly data; per-vendor constructors set defaults.

**Registry** (`Models`):
The runtime collection of Providers offering lookup, auth-gated availability,
and stream dispatch by model id.

**In-band error**:
A failure delivered as a stream event carrying partial content, not a
`Result::Err`. Only setup/config errors are `Result`s.

**Context normalization** (issue #39):
The one pass that resolves every cross-model rule, run in stream dispatch
before either protocol adapter builds its wire payload. It takes the caller's
`Context` and the target `Model` and yields a normalized copy plus
diagnostics; the caller's own value is never mutated, so the same `Context` can
be streamed against one model after another. An adapter consumes that copy and
only translates it to its wire shape — it never re-applies a rule itself.
_Avoid_: transform, sanitize

**Reasoning downgrade** (issue #40):
The normalization rule for provider-private reasoning state. A thinking
block's opaque signature — and a text block's `textSignature` — replays
verbatim only onto the exact provider, API, and model id that produced it
(the message's `responseModel` when set, else `model`). Replayed anywhere
else, non-empty ordinary thinking becomes a plain text block, empty or
redacted thinking is omitted, and every signature is dropped; one
`ReasoningDowngraded` diagnostic reports the counts. Provenance that is
missing or differs in any one of the three fields counts as "anywhere else".
_Avoid_: reasoning replay, signature passthrough

**Tool-call id rewrite** (issue #40):
The normalization rule for tool-call identities. Any `ToolCall.id` or
`ToolResultMessage.toolCallId` not matching `^[a-zA-Z0-9_-]{1,64}$` — the
Anthropic tool-use pattern, which the OpenAI side also satisfies — is
rewritten deterministically: invalid characters become `_`, the result is
truncated, and a stable FNV-1a hash of the original id is appended so
distinct invalid ids stay distinct. The rewrite is a pure function of the
original id, so a tool result always tracks its call; one
`ToolCallIdRewritten` diagnostic reports the count.

**Modality gate**:
The one normalization rule that rejects rather than repairs: if the newest user
message carries an image and the model does not declare `Modality::Image`, the
stream terminates in-band with `ErrorKind::InvalidRequest` before any HTTP
request. The caller is asking the model to look at something it cannot see, so
answering the wrong question would be worse than failing.

**Image downgrade**:
The normalization rule for every image the gate does not reject — historical
user turns and tool results alike. On a model without `Modality::Image` each
image block is replaced in place with fixed text — `(image omitted: model does
not support images)` for a user image, `(tool image omitted: model does not
support images)` for a tool result (issue #22) — a consecutive run of images
collapsing into a single placeholder. Text blocks are kept, order is preserved,
no message is dropped, and one `ImageDowngraded` diagnostic per kind lands on
the resulting message — counting images omitted, which a collapsed run makes
larger than the number of placeholders. A downgraded message is all text, so it
takes the plain-string wire shape and the placeholder reads as part of the
turn's text. On an image-capable model nothing is replaced: OpenAI
sends `tool` messages text-only and trails a run of consecutive tool results
with one user message carrying every image; Anthropic puts `image` blocks
inside the `tool_result` content.

**Tool-history repair** (issue #41):
The normalization rule that makes replayed history always form a legal
request. A historical tool call whose result was never recorded receives
exactly one synthetic error result — body `No result provided` — placed right
after the turn that issued it; existing results are preserved and never
duplicated. An assistant turn that ended in `Error` or `Aborted` is dropped
from the normalized copy, and any results answering its calls go with it, so
no result is left pointing at a call that no longer exists. A trailing
assistant turn is left alone — its calls may still be mid-execution.

**Streaming tool call** (issue #45):
A tool call is usable from its first public event: `ToolCallStart` carries the
known `id`/`name`, and every arguments delta refreshes `partial()`'s
best-effort parsed `arguments` snapshot while `raw_arguments` accumulates the
verbatim text. The snapshot comes from the incremental parser, which returns
complete JSON exactly as parsed, closes truncated constructs (open strings,
containers, dangling escapes) on a best-effort basis, and judges structural
corruption unrepairable. A call whose terminal raw text is unrepairable fails
the stream with an in-band `ErrorKind::Protocol` error that preserves the raw
text — it never surfaces as a fabricated `{}`.

**Tool Choice** (`ToolChoice`, issue #46):
The cross-protocol answer to "which tool may or must the model call" — `Auto`,
`None`, `Required`, `Named`, set per request via `StreamOptions::tool_choice`.
Its absence sends no `tool_choice` field at all: the provider's own default
applies, and the payload is byte-for-byte what it was before the option
existed. OpenAI-compatible wires spell the choices `"auto"` / `"none"` /
`"required"` / `{"type":"function","function":{"name":…}}`; Anthropic-compatible
wires spell them `{"type":"auto"}` / `{"type":"none"}` / `{"type":"any"}` /
`{"type":"tool","name":…}`. A `Named` name goes out exactly as given, never
rewritten. What a provider can express is declared per endpoint
(`OpenAiCompat::tool_choice` / `AnthropicCompat::tool_choice`, a
`ToolChoiceSupport`), never inferred: an unconfigured endpoint attests nothing,
so any explicit choice against it is refused. Bundled declarations: OpenAI all
four; Moonshot all four; MiniMax all four (its reference declares `tool_choice`
fully supported); DeepSeek `auto`/`none`; Z.AI and Xiaomi MiMo `auto` only —
MiMo's reference states other values are stripped server-side, a silent remap
banshu refuses to send; Kimi For Coding none, since it publishes no
parameter-level reference for its Anthropic shape.
_Avoid_: forced tool, tool mode

**Tool-choice preflight** (issue #46):
The check in stream dispatch, right after the Reasoning preflight, that the
provider declares support for the requested `ToolChoice` on the model's
protocol. A choice it cannot express terminates in-band with
`ErrorKind::InvalidRequest` before any HTTP request, naming the choices that
would have worked — remapping a `required` onto an `auto` would answer a
question the caller did not ask.

**Strict tool schema** (`Tool::strict`, issue #46):
A caller's marker that a tool's schema is authored to strict-mode rules, asking
the provider for schema-constrained tool arguments. It reaches the wire only
when the provider declares strict tool schemas
(`OpenAiCompat::strict_tool_schemas` /
`AnthropicCompat::strict_tool_schemas`); otherwise the field is omitted
entirely and the tool works unconstrained. Declared by OpenAI, Moonshot, and
Xiaomi MiMo. The marker is skipped in serialized `Tool` JSON when `false`, so
Context Snapshots written before it existed stay byte-identical.
_Avoid_: constrained sampling flag

**Reasoning Effort** (`ReasoningEffort`):
The unified ladder a request asks for — `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, `max`. `off` is an explicit request to disable reasoning,
which a provider sends as its own disabling value; the *absence* of a request
is `StreamOptions::reasoning == None`, which leaves the payload untouched.
_Avoid_: thinking level, reasoning level

**Reasoning Request Format**
(`OpenAiReasoningFormat` / `AnthropicReasoningFormat`):
The wire shape a provider *declares* its endpoint accepts for reasoning —
never inferred from a base URL or a model id. It also decides the
token-budget capability stamped onto the models that provider serves. A
provider that declares none refuses every reasoning request instead of
sending a field the endpoint would ignore.
_Avoid_: thinking format, wire shape

The OpenAI-compatible shapes banshu maps (issue #43), each named for the
fields it puts on the wire rather than for a vendor, and each stating what
`off` sends. Only endpoints banshu targets are claimed; nothing here is a
statement about other services that speak `POST /chat/completions`:

| Shape | Enabled | `off` | Declared by |
| --- | --- | --- | --- |
| `ThinkingToggle` | `thinking:{"type":"enabled"}` + `reasoning_effort:"<level>"` | `thinking:{"type":"disabled"}`, no effort | DeepSeek |
| `ThinkingToggleOnly` | `thinking:{"type":"enabled"}` | `thinking:{"type":"disabled"}` | Z.AI, Xiaomi MiMo |
| `ReasoningEffort` | `reasoning_effort:"<level>"` | `reasoning_effort:"none"` | OpenAI |
| `Unsupported` | — refuses every request | — | Moonshot AI |

`ThinkingToggleOnly` carries no effort field, so the ladder collapses onto its
toggle: any level above `off` reads as "enabled". `off` is spelled `none` in a
`reasoning_effort` field — banshu's own name for the level is `off`, and the
two are not interchangeable on the wire.

The Anthropic-compatible shapes (issue #44), read the same way. Shape names are
scoped to their protocol — Anthropic's `ThinkingToggle` is the analogue of
OpenAI's `ThinkingToggleOnly`, since no Anthropic-compatible shape carries an
effort string for a toggle to sit beside. All three spell `off` as
`thinking:{"type":"disabled"}`, which is the value every one of their references
documents; they differ only in how they say "reason":

| Shape | Enabled | Declared by |
| --- | --- | --- |
| `ThinkingToggle` | `thinking:{"type":"enabled"}` | Kimi For Coding |
| `ThinkingAdaptive` | `thinking:{"type":"adaptive"}` | MiniMax |
| `ThinkingBudget` | `thinking:{"type":"enabled","budget_tokens":N}` | — (a custom provider) |
| `Unsupported` | — refuses every request | — |

No bundled vendor declares `ThinkingBudget`: neither MiniMax's nor Kimi's
reference documents a `budget_tokens` field, and Kimi's says outright that its
models take none. It stays because it is Anthropic's own shape, which a caller
pointing `anthropic_compatible` at such an endpoint declares themselves.

**Reasoning Budget** (`ReasoningOptions::token_budget`, `ThinkingBudget`):
The `ThinkingBudget` shape has no effort field — there the level *is* a token
count — so a request that names no budget spends a documented ladder derived
from the level: 1024 for `minimal`, 2048 `low`, 8192 `medium`, 16384 `high`,
32768 `xhigh`, 65536 `max`. `max_tokens` caps the reasoning and the answer
together, so:

- a budget the *caller* named is sent verbatim, and one that does not fit under
  the request's final `max_tokens` — or falls below the 1024-token minimum the
  shape documents — is refused by the Reasoning preflight before any HTTP;
- an enabled request needs a model whose Reasoning Capability attests a token
  budget, since on this shape a budget is the only way to say "reason";
- a budget *banshu* derived is trimmed to leave 1024 tokens for the answer,
  which is not the clamp this crate refuses to perform: the caller asked for a
  level, not for a token count. An output cap with no room for both the minimum
  budget and an answer is refused rather than sent illegal;
- a budget alongside `off` is refused, since a disabled request sends the
  toggle alone and would have to discard it.

The final `max_tokens` is the request's, else the model's, else 4096 — the
preflight and the wire read the same ladder, so a budget that passes the check
is the one that ships.
_Avoid_: thinking tokens, budget clamp

**Reasoning Effort Vocabulary**
(`OpenAiCompat::reasoning_efforts` / `AnthropicCompat::reasoning_efforts`):
The effort levels a provider's own reference documents, declared alongside its
Reasoning Request Format and stamped onto the models it serves in place of the
baseline ladder. A model metadata source says only *whether* a model reasons,
never which levels it takes, so without this every provider would inherit the
same invented default and a level the endpoint has never heard of would sail
past the Reasoning preflight into a `400`. Declaring narrows *and* widens: a
provider documenting `max` gets it, one documenting no `minimal` refuses it.

A level the endpoint accepts but silently *remaps* onto another stays out of
the vocabulary — attesting it would move the clamp banshu refuses to perform
onto the server, where the caller cannot see it. This is why DeepSeek attests
`off`/`low`/`high`/`max` and not `medium`/`xhigh`, which its reference maps
onto `high`.

Three states, the last two of which differ: declaring nothing keeps the
baseline ladder, right for an endpoint whose shape has no effort field to
constrain; declaring levels makes exactly those requestable; declaring an
*empty* vocabulary makes none requestable and is right for an endpoint with no
reasoning request field at all — those models may still stream thinking, but
no effort can be asked of them, so `ReasoningCapability::reasons()` is `false`.
_Avoid_: effort whitelist, level map

**Reasoning preflight** (issue #42):
The check in stream dispatch, ahead of Context normalization and auth, that a
reasoning request can be honoured by both the model's Reasoning Capability and
the provider's Reasoning Request Format. It reads nothing but the options, the
model, and the provider, so an impossible request terminates in-band with
`ErrorKind::InvalidRequest` before any HTTP request. Like the Modality gate it
rejects rather than repairs — clamping to a level the caller did not ask for
would answer a different question.
_Avoid_: reasoning clamp, effort fallback

**Context Snapshot** (`ContextSnapshotV1`):
The versioned JSON persistence format for a `Context`, pinned by a golden
fixture. The serialized shape (camelCase, `role`/`type` tags) is a published
contract compatible with pi-ai; a snapshot declaring an unknown version is
rejected outright, never parsed best-effort.
