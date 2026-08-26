# banshu-ai 1.0 release-candidate audit

This document is the reproducible automated audit for issue
[#59](https://github.com/meymchen/banshu/issues/59). It covers the still-applicable
non-human criteria from roadmap milestones 0.4.0 through 0.9.0 and the 1.0
package gates. Real-account smoke tests, maintainer judgment, and publication
remain in the human release issue
[#60](https://github.com/meymchen/banshu/issues/60).

The release is greenfield. Earlier criteria that named `ContextSnapshotV1` are
superseded only in their persistence mechanism: the same serde-shape and
lossless-round-trip promises are now proved directly against `Context`, with no
version wrapper or compatibility shim.

## Reproduce the release candidate

Run these commands from the repository root with Rust 1.98.0, as pinned by
[`rust-toolchain.toml`](../rust-toolchain.toml):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features
cargo check --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
diff -u README.md crates/ai/README.md
cargo test -p xtask --test release_package
cargo deny --all-features check licenses
cargo semver-checks check-release -p banshu-ai --baseline-version 0.9.0 --all-features --release-type major
cargo publish --dry-run -p banshu-ai --all-features
cargo run -p xtask -- verify-release-package
```

The API review also runs the stricter comparison below with
`--release-type minor`:

```sh
cargo semver-checks check-release -p banshu-ai --baseline-version 0.9.0 --all-features --release-type minor
```

That command is expected to exit nonzero with exactly one
`inherent_method_missing` finding, `Provider::openai`. The 2026-08-26 audit
reported 195 passing checks and that one intentional failure. Any other
finding is unexpected and must be fixed or receive an explicit disposition in
this section.

The final command runs the minimum, OAuth, and custom-provider README doctests
and the direct `Context` fixture test from Cargo's unpacked `.crate`, not from
the workspace source tree. The Linux and Windows test matrix, MSRV check,
warning-denied docs, license review, semver review, archive-content test, and
publish dry run are centralized in
[`action.yml`](../.github/actions/release-package/action.yml) and enforced by
[`ci.yml`](../.github/workflows/ci.yml). The same package gates run before the
publication job in [`release-plz.yml`](../.github/workflows/release-plz.yml).

## Roadmap evidence

Each row covers every automated checkbox in the linked issue. The linked tests
exercise the public or protocol boundary named by the criteria; supporting
domain or decision documents are linked when a criterion explicitly requires
them.

### Milestone 0.4.0

| Issue | Direct evidence |
| --- | --- |
| [#39 — immutable Context normalization](https://github.com/meymchen/banshu/issues/39) | Both protocol adapters, historical/newest image behavior, tool-result images, and caller immutability: [`context_normalization.rs`](../crates/ai/tests/context_normalization.rs), [`user_images.rs`](../crates/ai/tests/user_images.rs), and [`tool_result_images.rs`](../crates/ai/tests/tool_result_images.rs). Vocabulary: [`CONTEXT.md`](../CONTEXT.md). |
| [#40 — cross-model reasoning and ids](https://github.com/meymchen/banshu/issues/40) | OpenAI↔Anthropic replay, same-model preservation, redacted/empty thinking, and linked call/result rewrites: [`cross_model_reasoning.rs`](../crates/ai/tests/cross_model_reasoning.rs) and the normalization properties in [`normalize.rs`](../crates/ai/src/api/normalize.rs). |
| [#41 — incomplete tool history](https://github.com/meymchen/banshu/issues/41) | Synthetic results, no duplication, failed-turn removal, and both wire shapes: [`context_normalization.rs`](../crates/ai/tests/context_normalization.rs). The still-applicable persistence promise is direct, lossless `Context` serde: [`context_serde.rs`](../crates/ai/tests/context_serde.rs) and [`context.json`](../crates/ai/tests/fixtures/context.json). |

### Milestone 0.5.0

| Issue | Direct evidence |
| --- | --- |
| [#42 — reasoning capabilities](https://github.com/meymchen/banshu/issues/42) | Every effort and budget capability, six-provider metadata, unsupported preflight failures, and unchanged absent options: [`reasoning_capabilities.rs`](../crates/ai/tests/reasoning_capabilities.rs) and [`provider_conformance.rs`](../crates/ai/tests/provider_conformance.rs). |
| [#43 — OpenAI-compatible reasoning](https://github.com/meymchen/banshu/issues/43) | Enabled, disabled, unsupported, signature, usage, and payload-shape cases for all declared OpenAI formats: [`openai_reasoning_requests.rs`](../crates/ai/tests/openai_reasoning_requests.rs), [`openai_thinking_replay.rs`](../crates/ai/tests/openai_thinking_replay.rs), and the [provider matrix](provider-conformance.md). |
| [#44 — Anthropic-compatible reasoning](https://github.com/meymchen/banshu/issues/44) | Toggle, budget, adaptive, invalid-budget, unchanged-request, thinking, signature, and usage cases: [`anthropic_reasoning_requests.rs`](../crates/ai/tests/anthropic_reasoning_requests.rs) and [`anthropic_thinking_replay.rs`](../crates/ai/tests/anthropic_thinking_replay.rs). |

### Milestone 0.6.0

| Issue | Direct evidence |
| --- | --- |
| [#45 — streaming tool identity and arguments](https://github.com/meymchen/banshu/issues/45) | First-event identity, partial snapshots, raw arguments, repair, and terminal failures for both protocols: [`openai_completions_tool_calls.rs`](../crates/ai/tests/openai_completions_tool_calls.rs), [`anthropic_messages_tool_calls.rs`](../crates/ai/tests/anthropic_messages_tool_calls.rs), and [`partial_json.rs`](../crates/ai/src/partial_json.rs). |
| [#46 — cross-protocol tool choice](https://github.com/meymchen/banshu/issues/46) | All four choices, exact names, strict schemas, provider support, and pre-HTTP rejection for both protocols: [`tool_choice.rs`](../crates/ai/tests/tool_choice.rs). |
| [#47 — tool argument validation](https://github.com/meymchen/banshu/issues/47) | Nested success, all required schema families, precise errors, no coercion/mutation, and streamed arguments: [`tool_argument_validation.rs`](../crates/ai/tests/tool_argument_validation.rs) and [`tool.rs`](../crates/ai/src/types/tool.rs). |

### Milestone 0.7.0

| Issue | Direct evidence |
| --- | --- |
| [#48 — credential and OAuth lifecycle](https://github.com/meymchen/banshu/issues/48) | Tagged/redacted credentials, atomic store modification, interaction lifecycle, registry operations, single-flight refresh, invalid refresh, and API-key priority: [`credentials.rs`](../crates/ai/src/credentials.rs) and [`oauth_lifecycle.rs`](../crates/ai/tests/oauth_lifecycle.rs). |
| [#49 — Kimi OAuth](https://github.com/meymchen/banshu/issues/49) | Frozen device endpoints/client, pending/slow-down/success/denial/expiry/cancellation/timeout, refresh, bearer inference, controlled host override, logout, and secret redaction: [`kimi_oauth.rs`](../crates/ai/tests/kimi_oauth.rs). |
| [#50 — MiniMax OAuth](https://github.com/meymchen/banshu/issues/50) | Explicit CN/Global contracts, PKCE/state/polling, refresh rotation, merged inference headers, HTTPS resource URLs, failure handling, and secret redaction: [`minimax_oauth.rs`](../crates/ai/tests/minimax_oauth.rs). |

### Milestone 0.8.0

| Issue | Direct evidence |
| --- | --- |
| [#51 — persistent model overlays](https://github.com/meymchen/banshu/issues/51) | Stored overlay shape, offline restore, freshness/force, validators/304, cancellation/failure preservation, and Probe precedence: [`dynamic_discovery.rs`](../crates/ai/tests/dynamic_discovery.rs). Decision and vocabulary: [ADR-0001](adr/0001-model-discovery-models-dev-refresh-plus-probe.md) and [`CONTEXT.md`](../CONTEXT.md). |
| [#52 — token estimates and output budgets](https://github.com/meymchen/banshu/issues/52) | Stable estimation policy plus empty, boundary, image/tool, implicit clamp, explicit rejection, unknown-limit, and both-protocol cases: [`context_token_estimate.rs`](../crates/ai/tests/context_token_estimate.rs) and [`context_output_budgets.rs`](../crates/ai/tests/context_output_budgets.rs). |
| [#53 — overflow and tiered cost](https://github.com/meymchen/banshu/issues/53) | Six-provider positive fixtures; 429/quota/timeout/overload negatives; bounded redacted evidence; tier boundaries; flat costs; and unknown-zero handling: [`context_overflow.rs`](../crates/ai/tests/context_overflow.rs) and [`tiered_cost.rs`](../crates/ai/tests/tiered_cost.rs). |

### Milestone 0.9.0

| Issue | Direct evidence |
| --- | --- |
| [#54 — raw stop reasons](https://github.com/meymchen/banshu/issues/54) | Known/unknown raw values, normalized values, streaming, assembly, error/abort behavior, and serde: [`stream_state_machine.rs`](../crates/ai/tests/stream_state_machine.rs), [`anthropic_messages_stream.rs`](../crates/ai/tests/anthropic_messages_stream.rs), and [`context_serde.rs`](../crates/ai/tests/context_serde.rs). The obsolete wrapper clause is superseded by issue #59's direct `Context` requirement. |
| [#55 — request observer](https://github.com/meymchen/banshu/issues/55) | Redacted before-send/response records, request ids, immutable observations, panic containment, credential isolation, and exact retry attempts: [`request_observer.rs`](../crates/ai/tests/request_observer.rs) and [`provider_headers.rs`](../crates/ai/tests/provider_headers.rs). |
| [#56 — faux provider](https://github.com/meymchen/banshu/issues/56) | Deterministic success/usage, delay/cancellation, retries, terminal failure, thinking/signatures, and tool calls without network or credentials: [`faux_provider.rs`](../crates/ai/tests/faux_provider.rs). Public rustdoc is warning-denied by the release commands above. |
| [#57 — protocol-boundary properties](https://github.com/meymchen/banshu/issues/57) | SSE, partial JSON, normalization, and header-merge properties plus checked-in corpora, fixed cases, and replay commands: [`property-testing.md`](property-testing.md), [`sse.rs`](../crates/ai/src/sse.rs), [`partial_json.rs`](../crates/ai/src/partial_json.rs), [`normalize.rs`](../crates/ai/src/api/normalize.rs), and [`auth.rs`](../crates/ai/src/auth.rs). |
| [#58 — frozen 1.0 surface](https://github.com/meymchen/banshu/issues/58) | Six-provider feature matrix and fixed automated evidence: [`provider-conformance.md`](provider-conformance.md) and [`provider_conformance.rs`](../crates/ai/tests/provider_conformance.rs). Quick start, both protocols, custom providers, OAuth, tools, images, reasoning, cancellation, retry, persistence, and errors: [`README.md`](../crates/ai/README.md), compiled as crate doctests. Public docs and API are covered by the warning-denied rustdoc and semver commands above. |

## Package and API disposition

- [`release_package.rs`](../crates/xtask/tests/release_package.rs) checks the
  Cargo archive list. It permits only the direct `Context` release fixture and
  rejects unrelated test fixtures, property corpora, `.scratch`, environment
  files, key material, and credential/token directories.
- The same test proves the archive carries exactly the six promised bundled
  Catalogs and both protocol adapter modules. Custom adapters remain an
  extension seam, not a third built-in protocol or provider.
- [`deny.toml`](../deny.toml) permits only reviewed permissive SPDX licenses;
  `cargo deny --all-features check licenses` evaluates the full dependency
  graph.
- `cargo semver-checks` compares the all-feature public API with published
  `0.9.0` under the 1.0 major-release rules. The API review found one deliberate
  scope removal: `Provider::openai()` was a seventh built-in, outside the six
  roadmap targets. OpenAI-compatible custom providers remain supported through
  the generic extension seam. The direct `Context` persistence change is also
  intentional greenfield serialization, not an invitation to restore the
  unused wrapper.
- [`CHANGELOG.md`](../crates/ai/CHANGELOG.md) describes the pending-message,
  injected-client, compatibility-declaration, sampling, direct-persistence,
  MSRV, and package-verification changes since 0.9.0 without migration guidance.
