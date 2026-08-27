# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0](https://github.com/meymchen/banshu/compare/v1.0.0...v1.1.0) - 2026-08-27

### Added

- add live provider smoke tests ([#111](https://github.com/meymchen/banshu/pull/111))

### Added

- Application-owned HTTP clients can be injected into built-in and custom
  providers, including model discovery and OAuth sessions
  ([#99](https://github.com/meymchen/banshu/pull/99)).
- OpenAI-compatible providers can declare request-envelope fields, clean-EOF
  termination, tool-history shapes, cache routing, typed open-model reasoning,
  and guarded custom sampling parameters; Anthropic-compatible providers can
  declare cache and temperature support
  ([#97](https://github.com/meymchen/banshu/issues/97)).

### Changed

- `AssistantMessageEvent::Start` carries the pending `AssistantMessage`, and
  `MessageStream::partial()` begins with that same response
  ([#98](https://github.com/meymchen/banshu/pull/98)).
- Conversation persistence serializes `Context` directly, without a version
  wrapper or compatibility layer for the unconsumed pre-1.0 shape.
- The built-in provider surface is limited to the six roadmap targets;
  `Provider::openai()` was removed while the custom OpenAI-compatible provider
  seam remains.
- The release toolchain and CI are pinned to Rust 1.98.0, and the published
  package is verified from its own archive before release.

## [0.9.0](https://github.com/meymchen/banshu/compare/v0.8.0...v0.9.0) - 2026-08-25

### Added

- *(ai)* [**breaking**] freeze the 1.0 public API ([#86](https://github.com/meymchen/banshu/pull/86))
- *(ai)* add deterministic faux provider ([#84](https://github.com/meymchen/banshu/pull/84))
- *(ai)* add redacted request observers ([#55](https://github.com/meymchen/banshu/pull/55)) ([#83](https://github.com/meymchen/banshu/pull/83))
- *(ai)* preserve raw stop reasons across protocols ([#54](https://github.com/meymchen/banshu/pull/54))

### Other

- *(ai)* harden protocol boundaries with properties ([#57](https://github.com/meymchen/banshu/pull/57)) ([#85](https://github.com/meymchen/banshu/pull/85))
- *(ai)* stabilize cancellation coverage ([#80](https://github.com/meymchen/banshu/pull/80))

### Changed

- **Breaking:** configure request caching and reasoning through
  `with_openai_compat` / `with_anthropic_compat`; the duplicated
  `with_openai_prompt_caching`, `with_openai_reasoning_format`, and
  `with_anthropic_reasoning_format` convenience setters were removed before
  the 1.0 API freeze.
- Added a compile-checked README, a provider conformance matrix, and a
  warning-free rustdoc CI gate for the planned 1.0 public contract.

## [0.8.0](https://github.com/meymchen/banshu/compare/v0.7.0...v0.8.0) - 2026-08-23

### Added

- *(ai)* [**breaking**] classify context overflow and calculate tiered cost ([#53](https://github.com/meymchen/banshu/pull/53)) ([#79](https://github.com/meymchen/banshu/pull/79))
- *(ai)* guard output budgets with context estimates ([#52](https://github.com/meymchen/banshu/pull/52)) ([#78](https://github.com/meymchen/banshu/pull/78))
- *(ai)* persist and conditionally refresh model overlays ([#76](https://github.com/meymchen/banshu/pull/76))

### Added

- *(ai)* add stable context-token estimates and guarded output budgets ([#52](https://github.com/meymchen/banshu/issues/52))

## [0.7.0](https://github.com/meymchen/banshu/compare/v0.6.0...v0.7.0) - 2026-08-14

### Added

- *(ai)* [**breaking**] add MiniMax Coding Plan OAuth ([#75](https://github.com/meymchen/banshu/pull/75))
- *(ai)* [**breaking**] add Kimi For Coding OAuth device authorization ([#49](https://github.com/meymchen/banshu/pull/49)) ([#74](https://github.com/meymchen/banshu/pull/74))
- *(ai)* add credential storage and OAuth lifecycle abstractions ([#48](https://github.com/meymchen/banshu/pull/48)) ([#72](https://github.com/meymchen/banshu/pull/72))

## [0.6.0](https://github.com/meymchen/banshu/compare/v0.5.0...v0.6.0) - 2026-08-11

### Added

- *(ai)* validate tool arguments against JSON Schema ([#47](https://github.com/meymchen/banshu/pull/47)) ([#70](https://github.com/meymchen/banshu/pull/70))

## [0.5.0](https://github.com/meymchen/banshu/compare/v0.4.0...v0.5.0) - 2026-08-10

### Added

- *(ai)* [**breaking**] add cross-protocol tool choice ([#46](https://github.com/meymchen/banshu/pull/46)) ([#69](https://github.com/meymchen/banshu/pull/69))
- *(ai)* [**breaking**] expose tool identity and partial arguments while streaming ([#67](https://github.com/meymchen/banshu/pull/67))

## [0.4.0](https://github.com/meymchen/banshu/compare/v0.3.0...v0.4.0) - 2026-08-04

### Added

- *(ai)* [**breaking**] map reasoning controls onto Anthropic-compatible providers ([#44](https://github.com/meymchen/banshu/pull/44)) ([#66](https://github.com/meymchen/banshu/pull/66))
- *(ai)* [**breaking**] map reasoning controls onto OpenAI-compatible providers ([#65](https://github.com/meymchen/banshu/pull/65))
- *(ai)* [**breaking**] model reasoning capabilities explicitly ([#64](https://github.com/meymchen/banshu/pull/64))
- *(ai)* repair incomplete tool history before replay ([#63](https://github.com/meymchen/banshu/pull/63))
- *(ai)* normalize cross-model reasoning and tool-call identities ([#40](https://github.com/meymchen/banshu/pull/40)) ([#62](https://github.com/meymchen/banshu/pull/62))
- *(ai)* add immutable context normalization ([#61](https://github.com/meymchen/banshu/pull/61))

### Other

- replace dangling PRD §-references with issue numbers ([#37](https://github.com/meymchen/banshu/pull/37))

## [0.3.0](https://github.com/meymchen/banshu/compare/v0.1.0...v0.3.0) - 2026-07-28

### Added

- tool-result images with graceful downgrade on non-image models ([#35](https://github.com/meymchen/banshu/pull/35))
- user images end-to-end on both protocols with modality gating ([#21](https://github.com/meymchen/banshu/pull/21)) ([#34](https://github.com/meymchen/banshu/pull/34))
- ModelCapabilities.tool_calling + unified catalog/refresh mapping + agent_models() ([#33](https://github.com/meymchen/banshu/pull/33))
- add deterministic provider header merge chain ([#32](https://github.com/meymchen/banshu/pull/32))
- [**breaking**] public extension seams — ProtocolAdapter, PreparedRequest, ProviderBuilder ([#18](https://github.com/meymchen/banshu/pull/18)) ([#31](https://github.com/meymchen/banshu/pull/31))
- explicit cancellation via CancellationToken across all stream phases ([#30](https://github.com/meymchen/banshu/pull/30))
- pluggable auth adapters — api_key_env, keyless, custom AuthResolver ([#29](https://github.com/meymchen/banshu/pull/29))
- [**breaking**] contract the MessageStream API ([#15](https://github.com/meymchen/banshu/pull/15)) ([#28](https://github.com/meymchen/banshu/pull/28))
- Anthropic adapter on ProtocolEvent + MessageAssembler ([#14](https://github.com/meymchen/banshu/pull/14)) ([#27](https://github.com/meymchen/banshu/pull/27))
- OpenAI adapter on ProtocolEvent + MessageAssembler ([#26](https://github.com/meymchen/banshu/pull/26))
- SSE decoder + shared RequestExecutor (#11, #12) ([#25](https://github.com/meymchen/banshu/pull/25))
- [**breaking**] stable serde for conversation types with golden fixtures ([#24](https://github.com/meymchen/banshu/pull/24))
- [**breaking**] add conversation domain groundwork ([#9](https://github.com/meymchen/banshu/pull/9))
- dynamic model discovery via models.dev refresh + vendor /models probes ([#8](https://github.com/meymchen/banshu/pull/8))
- thinking signature replay + per-provider compat flags ([#7](https://github.com/meymchen/banshu/pull/7))

### Other

- rustdoc release gates ([#23](https://github.com/meymchen/banshu/pull/23)) ([#36](https://github.com/meymchen/banshu/pull/36))
- release v0.1.0 ([#5](https://github.com/meymchen/banshu/pull/5))

## [0.1.0](https://github.com/meymchen/banshu/releases/tag/v0.1.0) - 2026-07-18

### Other

- add release-plz automation, CI workflow, and MSRV ([#4](https://github.com/meymchen/banshu/pull/4))
- Add bounded pre-stream retry with structured error classification ([#3](https://github.com/meymchen/banshu/pull/3))
- Add Anthropic prompt caching support ([#2](https://github.com/meymchen/banshu/pull/2))
- Add OpenAI-compatible prompt caching support ([#1](https://github.com/meymchen/banshu/pull/1))
- Initial commit: banshu workspace with ai crate
