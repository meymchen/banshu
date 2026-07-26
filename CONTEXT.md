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

**Context Snapshot** (`ContextSnapshotV1`):
The versioned JSON persistence format for a `Context`, pinned by a golden
fixture. The serialized shape (camelCase, `role`/`type` tags) is a published
contract compatible with pi-ai; a snapshot declaring an unknown version is
rejected outright, never parsed best-effort.
