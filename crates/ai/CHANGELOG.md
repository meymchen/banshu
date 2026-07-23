# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/meymchen/banshu/compare/v0.1.0...v0.2.0) - 2026-07-23

### Added

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

- release v0.1.0 ([#5](https://github.com/meymchen/banshu/pull/5))

## [0.1.0](https://github.com/meymchen/banshu/releases/tag/v0.1.0) - 2026-07-18

### Other

- add release-plz automation, CI workflow, and MSRV ([#4](https://github.com/meymchen/banshu/pull/4))
- Add bounded pre-stream retry with structured error classification ([#3](https://github.com/meymchen/banshu/pull/3))
- Add Anthropic prompt caching support ([#2](https://github.com/meymchen/banshu/pull/2))
- Add OpenAI-compatible prompt caching support ([#1](https://github.com/meymchen/banshu/pull/1))
- Initial commit: banshu workspace with ai crate
