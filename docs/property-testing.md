# Property testing

The `banshu-ai` protocol-boundary property suites use `proptest` to exercise:

- SSE framing across arbitrary byte-chunk boundaries, `CR`, `LF`, and `CRLF`
  delimiters, multiline data, comments, and bounded arbitrary input;
- complete and partial JSON semantics over generated values and arbitrary
  Unicode fragments;
- context normalization immutability, determinism, tool-call/result identity,
  and idempotence; and
- case-insensitive header precedence and secret redaction.

These pure properties live in `--lib` unit tests named `property_*`. Run every
one with its checked-in case count:

```sh
cargo test -p banshu-ai --lib property_
```

Wire-level properties — which need a local mock HTTP server, so they live in
integration tests instead — follow the same `property_*` naming. Today those
are the sampling-guard suites, run with:

```sh
cargo test -p banshu-ai --test openai_sampling property_
```

Proptest automatically replays checked-in failure corpora before generating
new cases. The suite's discovered regressions are stored in:

- `crates/ai/proptest-regressions/sse.txt` for bare-`CR` framing; and
- `crates/ai/proptest-regressions/api/normalize.txt` for nondeterministic
  synthetic tool-result timestamps.

Replay the SSE corpus and its property with:

```sh
cargo test -p banshu-ai --lib \
  sse::tests::property_decoding_is_independent_of_chunk_boundaries_and_line_endings
```

The minimized regression is also pinned as a deterministic named test:

```sh
cargo test -p banshu-ai --lib \
  sse::tests::bare_cr_separated_event_is_decoded_without_leaking_the_delimiter
```

Replay the normalization corpus and its named deterministic regression with:

```sh
cargo test -p banshu-ai --lib \
  api::normalize::tests::property_normalization_is_immutable_deterministic_consistent_and_idempotent
cargo test -p banshu-ai --lib \
  api::normalize::tests::synthetic_tool_result_uses_the_issuing_turn_timestamp
```

When a future property fails, keep the generated file under
`proptest-regressions/` and add a named deterministic test before fixing the
production behavior.
