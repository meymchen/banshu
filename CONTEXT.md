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
The merged result of Catalog ← Catalog Refresh ← Probe that a provider serves
as its model list. Its dynamic layers may be restored from an injected
`ModelsStore` before optional network work. Stored Probe provenance is retained
so a restored zero-means-unknown model still cannot overwrite Catalog or
Catalog Refresh metadata. Failed, cancelled, and 304 refreshes preserve the
last-known-good Overlay; a 304 only advances its checked-at time.

**Refresh Policy** (`RefreshOptions`):
The application-owned decision about whether discovery may use the network,
how long a stored Overlay is fresh, whether to force a network check, and how
to cancel one. `allow_network=false` is the hard offline gate; `force=true`
bypasses freshness only when networking is allowed. HTTP validators belong to
the stored Overlay and are sent by the next Catalog Refresh.

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

**Raw Stop Reason** (`AssistantMessage::raw_stop_reason`, issue #54):
The provider's exact terminal vocabulary for why generation ended, preserved
alongside banshu's stable cross-provider `StopReason`. It is diagnostic rather
than portable: known values retain their existing normalized behavior, while
an unrecognized value normalizes to `Unknown` without losing the original.
_Avoid_: finish reason, provider stop code

**Request Observer** (`RequestObserver`, issue #55):
The read-only diagnostic seam for redacted request visibility, attached per
request via `StreamOptions::observer`. The built-in dispatch reports every
attempt exactly once before send (a 1-based attempt number, redacted URL,
provider, model, and payload snapshot) and every response when its headers
arrive (status, redacted headers, and the provider request id when present) —
a retryable failure's response is observed just like the success that follows
it, while a transport failure that never receives headers observes nothing.
Everything an observer sees is redacted at construction with the same
secret/base64 pipeline diagnostics use: the URL loses its query, fragment,
and userinfo, and sensitive header values (`Authorization`, `x-api-key`,
cookies, OAuth tokens, and equivalents) become `[REDACTED]`. Observations
arrive by shared reference, so an observer cannot rewrite the payload or
headers — mutation stays the job of a custom adapter — and a panicking
observer is caught, logged with one fixed secret-free warning, and the
request continues unchanged: no credential exposure, no duplicate sends, no
authentication changes. A custom protocol adapter owns its dispatch and does
not report.
_Avoid_: request hooks, middleware, interceptor

**Context normalization** (issue #39):
The one pass that resolves every cross-model rule, run in stream dispatch
before either protocol adapter builds its wire payload. It takes the caller's
`Context` and the target `Model` and yields a normalized copy plus
diagnostics; the caller's own value is never mutated, so the same `Context` can
be streamed against one model after another. An adapter consumes that copy and
only translates it to its wire shape — it never re-applies a rule itself.
_Avoid_: transform, sanitize

**Stable Token Estimate** (`Context::estimate_tokens`, issue #52):
A deterministic planning approximation of a Context's input size, not a
provider tokenizer or billable usage. Its published policy counts prompt
content while treating every image as a fixed placeholder.
_Avoid_: token count, tokenizer result

**Output Budget** (`StreamOptions::max_tokens`, issue #52):
The most tokens a response may produce after known context capacity is
considered. An omitted budget is bounded by known model limits; a caller-named
budget is preserved exactly or refused when it cannot fit remaining context,
and zero-valued model limits stay unknown.
_Avoid_: completion allowance, output clamp

**Request Envelope** (`OpenAiCompat::streamed_usage`,
`OpenAiCompat::output_token_field`, `OpenAiOutputTokenField`, issue #89):
The request shape an OpenAI-compatible provider declares its endpoint accepts
for the two cross-cutting envelope fields: whether streamed usage may be
requested (`stream_options: { "include_usage": true }` goes out only when
declared; an endpoint without the declaration gets no `stream_options` field
at all, and usage it reports anyway is still parsed) and which standard
output-token field carries the resolved Output Budget (a closed policy over
`max_tokens` and `max_completion_tokens` — exactly the selected field is sent,
the other is absent). The undeclared default requests streamed usage and caps
with `max_tokens`, byte-compatible with the request bodies bundled providers
have always sent.
_Avoid_: wire envelope, request framing

**Stream Termination** (`OpenAiCompat::stream_termination`,
`OpenAiStreamTermination`, issue #90):
What a bare end of stream means on an OpenAI-compatible endpoint. The formal
wire terminators are `data: [DONE]` and a `finish_reason`-bearing chunk; a
provider may additionally attest that its endpoint closes the connection only
after the final chunk (`CleanEofCompletion`), letting a clean EOF complete a
structurally finished response — at least one content block started, and every
streamed tool call's accumulated arguments complete JSON. The undeclared
default (`Strict`) treats any bare EOF as a dropped connection, and even under
the declaration an unfinished response, a cut chunk, or a transport failure is
never an inferred completion. An inferred completion reports `ToolUse` when
the response contains tool calls and `Stop` otherwise, with no Raw Stop
Reason.
_Avoid_: EOF inference, silent completion

**Tool-history wire policies** (`OpenAiCompat::tool_result_names`,
`OpenAiCompat::empty_assistant_separator`, issue #91):
The tool-history message shapes an OpenAI-compatible provider declares its
chat template requires: whether each replayed `tool` message carries the
tool's `name` alongside its `tool_call_id`, and whether an empty assistant
message (`{ "role": "assistant", "content": "" }`) separates a run of tool
results from a following user message. The separator fires only at a tool-run
→ user boundary — never between consecutive tool results, never twice in a
row, and ahead of the image-carrier user message that trails a run holding
images, which is such a boundary. The undeclared default sends neither,
byte-compatible with the request bodies bundled providers have always sent.
_Avoid_: tool message quirks, chat-template fixes

**Context Overflow** (`ErrorKind::ContextOverflow`, `is_context_overflow`,
issue #53):
One classification over every provider signal that a request exceeded the
model's context window: overflow wording on a 400/413 error (Anthropic's
"prompt is too long" / `request_too_large`, MiniMax's "context window exceeds
limit", Kimi/Moonshot's "exceeded model token limit", DeepSeek/OpenAI-style
"maximum context length" shapes), a successful response whose reported input
usage exceeds a known window (z.ai), and a zero-output `length` stop on a
window the input fills (Xiaomi). Classification is conservative by
construction: the HTTP gate only reclassifies 400/413, rate-limit/throttle
wording vetoes a match, and an unknown zero window never invents overflow from
usage. A classification names its matched evidence in a bounded, redacted
`ContextOverflow` diagnostic.
_Avoid_: context too long, prompt overflow

**Cost Tier** (`CostTier`, issue #53):
An optional request-wide rate set on `ModelCost` (models.dev `cost.tiers`
with `tier.type == "context"`). Total input usage — input + cache read +
cache write — that strictly exceeds a tier's threshold selects that tier's
rates for the whole request; at or below every threshold the base rates
apply. A zero threshold is unknown metadata and never selects, and models
without tiers keep flat-rate costs.
_Avoid_: price band, usage bracket

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

The final `max_tokens` is the resolved Output Budget; Anthropic's required wire
field retains its 4096 fallback only when no model limit is known. The
preflight and wire read the same value, so a reasoning budget that passes the
check is the one that ships.
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

**Credential** (`Credential`, issue #48):
The type-tagged secret material an application persists for a provider —
`apiKey` or `oauth` — with a hand-written redacted `Debug` so secrets never
reach logs. The OAuth half carries the access token, an optional refresh
token, an optional expiry (unix milliseconds; absent means "never expires"),
and an optional `resourceUrl` that overrides the request endpoint at
credential level (HTTPS only; loopback HTTP tolerated for local dev).
_Avoid_: token blob, auth entry

**Credential Store** (`CredentialStore`, issue #48):
The application-injectable seam credentials live behind: read, list, delete,
and `modify` — a serialized read-modify-write that is the only write path
with a read, which is what makes refresh-token rotation atomic. The crate
ships the process-local `InMemoryCredentialStore`; durable or encrypted
storage is the application's implementation of the same trait.
_Avoid_: token cache, keyring

**Auth Interaction** (`AuthInteraction`, issue #48):
Everything a login flow needs from the user side: verification instructions
(URL plus optional user code), an optional browser request the application
may decline, status reports, and the caller's timeout and cancellation token.
Flows run their polling inside `AuthInteraction::wait`, so a stuck login ends
as `Error::AuthTimeout` or `Error::AuthCancelled` rather than hanging.
_Avoid_: prompt callback, device UI

**OAuth Session** (`OAuthSession`, issue #48):
The per-provider OAuth lifecycle `Models::login` / `logout` / `check_auth`
delegate to: login is a plain `Result`-returning call, never a message
stream. Request-time resolution refreshes an expired (or nearly expired —
60s leeway) access token before it is used, and authenticates the request
with the headers the flow declares (`OAuthFlow::token_headers`):
`Authorization: Bearer` by default — an OAuth access token is a bearer token
(RFC 6750) on either wire protocol, never the protocol's API-key header
(issue #49) — unless the endpoint's own contract requires more, as MiniMax's
does (issue #50). A provider may also declare API-key env vars on its OAuth
auth (`OAuthAuth::with_api_key_env`); a set variable is an explicit operator
choice and always wins over the stored credential.
_Avoid_: oauth client, token manager

**Device Authorization** (`KimiDeviceFlow`, issue #49):
The RFC 8628 login flow of Kimi For Coding, against the fixed Kimi auth
contract: the fixed public client id (`KIMI_CLIENT_ID`), `POST
/api/oauth/device_authorization` to start a login, and `POST
/api/oauth/token` for both the device-code poll and the refresh-token grant,
all on the configured auth host (`KIMI_AUTH_HOST`). Verification instructions
reach the user through the Auth Interaction; polling rides out
`authorization_pending`, stretches the interval by five seconds on
`slow_down`, and ends on approval, `access_denied`, `expired_token`, the
device code's own expiry, or the caller's timeout and cancellation.
`Provider::kimi(store)` wires it to the shared credential lifecycle, so the
store the application injects is the one login writes, refresh rotates, and
logout deletes. The auth host is overridable only at construction
(`with_auth_host` — HTTPS, or loopback HTTP for test servers), never through
request metadata; and errors, Debug output, and diagnostics carry only the
fixed OAuth error vocabulary, never token material.
_Avoid_: device code login, kimi auth client

**Portal Authorization** (`MiniMaxPortalFlow`, issue #50):
The login flow of the MiniMax Coding Plan, against the frozen portal
contract: an explicit region (`MiniMaxRegion::Cn` / `Global` — never inferred
from IP) naming the portal (`api.minimaxi.com` / `api.minimax.io`) and
Anthropic-compatible inference (`<portal>/anthropic`) hosts, the fixed public
client id (`MINIMAX_CLIENT_ID`) and scope (`MINIMAX_OAUTH_SCOPE`), `POST
/oauth/code` with PKCE S256 and a random state to start a login, and `POST
/oauth/token` for both the `user_code`-grant poll and the refresh-token
grant. The code response's state must round-trip verbatim or the login is
rejected; its `expired_in` is an absolute-millisecond deadline, while the
token endpoint's reads as relative seconds, absolute seconds, or absolute
milliseconds by magnitude. Every token-endpoint status but `success` and
`error` keeps polling, inside the Auth Interaction's timeout and
cancellation. A MiniMax access token authenticates inference on both
`Authorization: Bearer` and `x-api-key` — the two headers the endpoint
requires — and a token response's `resource_url` (HTTPS only, else
structurally rejected) overrides the inference base URL at credential level.
`Provider::minimax(region, store)` wires it to the shared credential
lifecycle, registering the regions as `minimax` and `minimax-cn`, both
serving the bundled MiniMax catalog with `MINIMAX_API_KEY` as operator
override. The portal host is overridable only at construction (`with_portal`
— HTTPS, or loopback HTTP for test servers); errors carry only the fixed
`status` vocabulary and HTTP statuses, never token material or response
bodies.
_Avoid_: minimax device flow, region detection

**Single-flight refresh** (issue #48):
Concurrent requests against one provider whose tokens expired share a single
refresh HTTP operation: every waiter joins the same shared future and
resolves to the same structured result. A rejected refresh token
(`RefreshError::Invalid`) never deletes or overwrites the stored credential —
it stays for diagnosis and the caller gets `Error::ReLoginRequired`; a
transient failure likewise preserves it and reports `Error::Auth`.
_Avoid_: refresh dedup, token lock

**Faux Provider** (`FauxProvider`, issue #56):
A keyless, network-free provider whose repeatable script exercises the public
streaming contract, including timing, cancellation, setup retries, content,
usage, and in-band failures.
_Avoid_: mock server, fake HTTP provider

**HTTP Client** (`ProviderBuilder::http_client`, `Provider::with_http_client`,
issue #88):
The application-injectable `reqwest::Client` every provider-owned request goes
through — inference, Catalog Refresh, Probe, and the prepared request handed to
a custom protocol adapter — so that traffic shares the application's proxy,
certificate, DNS, connection-pool, and default-header policy. A provider given
no client constructs the crate's feature-selected default. An OAuth Session
captured its client at construction and is not retargeted by a later
replacement.
_Avoid_: transport config, shared client

**Context Snapshot** (`ContextSnapshotV1`):
The versioned JSON persistence format for a `Context`, pinned by a golden
fixture. The serialized shape (camelCase, `role`/`type` tags) is a published
contract compatible with pi-ai; a snapshot declaring an unknown version is
rejected outright, never parsed best-effort.
