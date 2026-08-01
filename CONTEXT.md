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
Support. A source that says only "this model reasons" attests the baseline
ladder `off`…`high`; `xhigh` and `max` need an attestation of their own, and
Probe models attest nothing.
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
