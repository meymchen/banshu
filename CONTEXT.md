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
the resulting message. On an image-capable model nothing is replaced: OpenAI
sends `tool` messages text-only and trails a run of consecutive tool results
with one user message carrying every image; Anthropic puts `image` blocks
inside the `tool_result` content.

**Context Snapshot** (`ContextSnapshotV1`):
The versioned JSON persistence format for a `Context`, pinned by a golden
fixture. The serialized shape (camelCase, `role`/`type` tags) is a published
contract compatible with pi-ai; a snapshot declaring an unknown version is
rejected outright, never parsed best-effort.
