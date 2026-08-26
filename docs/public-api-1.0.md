# Public API review for 1.0

This review amends the earlier freeze in issue #86. It covers every public
module, root re-export, type alias, constructor, trait, and public field after
the open-model compatibility work in issues #87–#96. The changes below are the
last planned source-breaking changes before 1.0.

## Amendments after the earlier freeze

### Root re-exports and public types

The crate root newly re-exports these compatibility types:

- `OpenAiSessionAffinity` and `OpenAiCacheRetention` for OpenAI cache routing.
- `OpenAiOutputTokenField` and `OpenAiStreamTermination` for request-envelope
  and clean-EOF policy.
- `OpenAiChatTemplateKwargs` and `OpenAiReasoningBudgetField` for typed
  open-model reasoning declarations.
- `AnthropicCacheRetention` and `AnthropicTemperature` for Anthropic cache and
  sampling declarations.

`OpenAiPromptCaching` is no longer exported. Its combined policy could not
represent cache routing and retention independently; the two new OpenAI types
replace it. No other root re-export or public type was added or removed by
issues #87–#96.

### Public fields, variants, methods, and constants

- `OpenAiCompat` adds `session_affinity`, `cache_retention`, `streamed_usage`,
  `output_token_field`, `stream_termination`, `tool_result_names`, and
  `empty_assistant_separator`; its former `prompt_caching` field is removed.
- `AnthropicCompat` adds `cache_retention`, `tool_cache_control`, and
  `temperature`.
- `OpenAiChatTemplateKwargs` is intentionally field-constructible through
  `enable_thinking`, `reasoning_effort`, and `token_budget`.
- `StreamOptions` adds `sampling`, the OpenAI-compatible map for unmodelled
  sampling controls.
- `AssistantMessageEvent::Start` adds its `message` field, and `StopReason`
  adds `Pending`. `OpenAiReasoningFormat` adds `EnableThinking` and
  `ChatTemplateKwargs`.
- `ProviderBuilder::http_client` and `Provider::with_http_client` inject an
  application-owned HTTP client. `OpenAiReasoningBudgetField::as_str` exposes
  the closed budget-field spelling. The doc-hidden
  `OpenAiCompletions::RESERVED_SAMPLING_KEYS` constant exists for downstream
  conformance tests, not as a curated application interface.

### Source-breaking migrations

- Match `AssistantMessageEvent::Start { message }`, not the former unit
  variant. The first event and `MessageStream::partial()` now expose the real
  empty response with `StopReason::Pending`; do not treat an in-progress
  response as terminal `Stop`.
- Replace `OpenAiPromptCaching` and `OpenAiCompat::prompt_caching` with the
  independent `session_affinity` and `cache_retention` declarations.
- Add `..OpenAiCompat::default()`, `..AnthropicCompat::default()`, and
  `..StreamOptions::default()` to struct literals so new declarations and
  request options default explicitly. This is required for literals that
  previously named every public field.
- Treat explicit unsupported requests as errors. Long cache retention and
  Anthropic temperature are checked against provider declarations before
  HTTP; they are no longer guessed, silently dropped, or inferred from a
  provider identity.

The compile-checked [1.0 migration guide](migration-1.0.md) shows the updated
custom-provider setup. The additive client-injection, sampling, reasoning,
request-envelope, termination, and tool-history controls need no migration
until an application opts into them.

## Decisions

- The crate root remains the canonical application-facing import surface.
  `api`, `provider`, and `testing` remain public because custom protocol
  adapters, detailed endpoint configuration, and faux-provider tests need
  those cohesive module surfaces.
- Built-in vendor constructors and the validated `ProviderBuilder` remain the
  provider seams. The two single-protocol constructors remain intentional
  shorthand; unlike the removed setters, they assemble an adapter and auth
  resolver rather than aliasing one field assignment.
- Public domain structs remain constructible for serde interoperability and
  application-owned persistence. `AssistantMessage::from_content`,
  `ContextSnapshotV1`, and `StreamOptions::default` are their standard
  construction paths.
- `ProtocolEvent` and capability/protocol enums remain `non_exhaustive` as part
  of the 1.0 contract. Observer records are also `non_exhaustive` and
  constructed only by the crate.

## Alias audit

The remaining public type aliases each have an explicit interface purpose:

- `Result<T>` fixes the crate's setup-error type throughout public traits.
- `ProviderHeaders` fixes the case-insensitive merge input shape while keeping
  `PreparedRequest` and auth interfaces readable.
- `ModifyCredential<'a>` names the object-safe, atomic credential-store update
  callback used by every `CredentialStore` implementation.
- `ProtocolEventStream` names the owned stream returned by
  `ProtocolAdapter::stream`, hiding pin/box/send boilerplate from adapters.

Three narrow provider setters had no separate contract from mutating one field
of `OpenAiCompat` or `AnthropicCompat` and were removed:

- `with_openai_prompt_caching`
- `with_openai_reasoning_format`
- `with_anthropic_reasoning_format`

Keeping them would create parallel configuration paths and require a new
setter for every future endpoint quirk. The complete compat values are the
sole configuration seams for those quirks.

## Release gates

- `cargo test --workspace --all-features` compiles README examples as crate
  doctests and runs provider/protocol conformance fixtures.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`
  rejects warnings; the crate-level `missing_docs` and
  `rustdoc::broken_intra_doc_links` denies keep public coverage and links hard
  errors.
