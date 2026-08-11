# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- ProtocolEvent + MessageAssembler (expand phase) — OpenAI adapter migrated ([#26](https://github.com/meymchen/banshu/pull/26))
- SSE decoder + shared RequestExecutor (#11, #12) ([#25](https://github.com/meymchen/banshu/pull/25))
- [**breaking**] stable serde for conversation types + ContextSnapshotV1 with golden fixtures ([#24](https://github.com/meymchen/banshu/pull/24))
- [**breaking**] add conversation domain groundwork ([#9](https://github.com/meymchen/banshu/pull/9))
- dynamic model discovery via models.dev refresh + vendor /models probes ([#8](https://github.com/meymchen/banshu/pull/8))
- thinking signature replay + per-provider compat flags ([#7](https://github.com/meymchen/banshu/pull/7))

### Other

- rustdoc release gates + drop unshipped migration doc ([#23](https://github.com/meymchen/banshu/pull/23)) ([#36](https://github.com/meymchen/banshu/pull/36))
- release v0.1.0 ([#5](https://github.com/meymchen/banshu/pull/5))

## [0.1.0](https://github.com/meymchen/banshu/releases/tag/v0.1.0) - 2026-07-18

### Other

- add release-plz automation, CI workflow, and MSRV ([#4](https://github.com/meymchen/banshu/pull/4))
- Add bounded pre-stream retry with structured error classification ([#3](https://github.com/meymchen/banshu/pull/3))
- Add Anthropic prompt caching support ([#2](https://github.com/meymchen/banshu/pull/2))
- Add OpenAI-compatible prompt caching support ([#1](https://github.com/meymchen/banshu/pull/1))
- Initial commit: banshu workspace with ai crate
