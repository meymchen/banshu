# Public API review for 1.0

Review baseline: `banshu-ai` 0.8.0 (`v0.8.0`). The review covers every public
module, root re-export, type alias, constructor, trait, and public field in the
crate after issues #54–#57.

## Decisions

- The crate root remains the canonical application-facing import surface.
  `api`, `provider`, and `testing` remain public because custom protocol
  adapters, detailed compatibility configuration, and downstream faux-provider
  tests need those cohesive module surfaces.
- Built-in vendor constructors and the validated `ProviderBuilder` remain the
  provider seams. The two single-protocol constructors remain intentional
  shorthand; unlike the removed setters, they assemble an adapter and auth
  resolver rather than aliasing one field assignment.
- Public domain structs remain constructible for serde interoperability and
  application-owned persistence. `AssistantMessage::from_content`,
  `ContextSnapshotV1`, and `StreamOptions::default` are the forward-compatible
  construction paths recommended in the migration guide.
- `ProtocolEvent` and capability/protocol enums remain `non_exhaustive` where
  downstream matching must tolerate growth. Observer records are also
  `non_exhaustive` and constructed only by the crate.

## Alias audit

The remaining public type aliases each have a post-1.0 interface purpose:

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
setter for every future compatibility field. The complete compat values are
the sole post-1.0 seams; `docs/migrations-1.0.md` records mechanical replacements.

## Release gates

- `cargo test --workspace --all-features` compiles README examples as crate
  doctests and runs provider/protocol conformance fixtures.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`
  rejects warnings; the crate-level `missing_docs` and
  `rustdoc::broken_intra_doc_links` denies keep public coverage and links hard
  errors.
