# Migrating from 0.8 to 1.0

This document records every public breaking change since `banshu-ai` 0.8.0.

## Construct public messages and options with defaults

`AssistantMessage` gained `raw_stop_reason`, and `StreamOptions` gained
`observer`. Existing external struct literals must initialize those fields.
Prefer constructors and a default update so future optional additions do not
force another edit:

```rust
use banshu_ai::{AssistantMessage, StreamOptions};

let message = AssistantMessage::from_content(Vec::new());
let options = StreamOptions {
    max_retries: Some(3),
    ..StreamOptions::default()
};

assert!(message.raw_stop_reason.is_none());
assert!(options.observer.is_none());
```

Persisted `AssistantMessage` and `ContextSnapshotV1` JSON remains readable:
both new fields default to absent during deserialization.

## Configure compatibility as one value

The narrow `Provider::with_openai_prompt_caching`,
`Provider::with_openai_reasoning_format`, and
`Provider::with_anthropic_reasoning_format` setters were pre-1.0 conveniences
that duplicated the complete compatibility setters. Move the field update into
`OpenAiCompat` or `AnthropicCompat`:

```rust
use banshu_ai::{OpenAiCompat, OpenAiPromptCaching, Provider};

let provider = Provider::openai_compatible(
    "acme",
    "Acme",
    "https://llm.example/v1",
    ["ACME_API_KEY"],
)
.with_openai_compat(OpenAiCompat {
    prompt_caching: OpenAiPromptCaching::SessionAffinityHeaders,
    ..OpenAiCompat::default()
});

assert_eq!(
    provider.openai_compat().prompt_caching,
    OpenAiPromptCaching::SessionAffinityHeaders,
);
```

This leaves one post-1.0 configuration seam per wire protocol and prevents
separate setters from overwriting one another with unrelated defaults.
