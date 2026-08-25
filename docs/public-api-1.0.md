# Public API review for 1.0

The review covers every public module, root re-export, type alias, constructor,
trait, and public field in the crate after issues #54–#57.

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
