# Migrating to the amended 1.0 contract

Issue #97 amends the earlier planned 1.0 freeze after the open-model
compatibility work. The full interface inventory is in
[`public-api-1.0.md`](public-api-1.0.md); this guide covers source changes an
application must make.

## Construct complete compatibility values

`OpenAiPromptCaching` and `OpenAiCompat::prompt_caching` are replaced by the
independent `session_affinity` and `cache_retention` declarations. Both
compatibility structs gained fields, so use a default update in custom-provider
literals and override only policies the endpoint reference attests. The same
setup can inject the application's HTTP client and declare a typed open-model
reasoning shape:

```rust
use banshu_ai::{
    OpenAiCacheRetention, OpenAiChatTemplateKwargs, OpenAiCompat,
    OpenAiOutputTokenField, OpenAiReasoningFormat, OpenAiSessionAffinity,
    Provider,
};

let http = reqwest::Client::builder().build()?;
let provider = Provider::openai_compatible(
    "local",
    "Local runtime",
    "http://127.0.0.1:8080/v1",
    ["LOCAL_LLM_API_KEY"],
)
.with_http_client(http)
.with_openai_compat(OpenAiCompat {
    session_affinity: OpenAiSessionAffinity::SessionAffinityHeaders,
    cache_retention: OpenAiCacheRetention::Long,
    reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(
        OpenAiChatTemplateKwargs {
            enable_thinking: Some("enable_thinking"),
            reasoning_effort: Some("reasoning_effort"),
            ..OpenAiChatTemplateKwargs::default()
        },
    ),
    streamed_usage: false,
    output_token_field: OpenAiOutputTokenField::MaxCompletionTokens,
    ..OpenAiCompat::default()
});

assert_eq!(provider.id(), "local");
# Ok::<(), Box<dyn std::error::Error>>(())
```

`AnthropicCompat` literals likewise need `..AnthropicCompat::default()` after
setting any attested `cache_retention`, `tool_cache_control`, `temperature`, or
reasoning fields. An explicit long cache lifetime or Anthropic temperature now
fails in-band before HTTP when the provider does not declare support.

## Initialize stream state from `Start`

`AssistantMessageEvent::Start` is now a struct variant carrying the real empty
assistant response. Its stop reason is `Pending`, which distinguishes an
in-progress stream from a terminal response:

```rust
use banshu_ai::{AssistantMessageEvent, StopReason};

fn observe_start(event: &AssistantMessageEvent) {
    if let AssistantMessageEvent::Start { message } = event {
        assert_eq!(message.stop_reason, StopReason::Pending);
        assert!(message.content.is_empty());
    }
}
```

Replace matches on the old unit variant `AssistantMessageEvent::Start` with
`AssistantMessageEvent::Start { message }` or `{ .. }`. `MessageStream::partial`
exposes that same pending message after the start event.

## Add guarded sampling controls

`StreamOptions` gained a `sampling` map. A default update keeps older literals
source-compatible and makes the empty-map behavior explicit. Custom
OpenAI-compatible controls are sent verbatim, while adapter-owned fields are
rejected before authentication or HTTP:

```rust
use std::collections::BTreeMap;
use banshu_ai::{ReasoningEffort, ReasoningOptions, StreamOptions};

let options = StreamOptions {
    reasoning: Some(ReasoningOptions::new(ReasoningEffort::High)),
    sampling: BTreeMap::from([
        ("top_k".into(), serde_json::json!(40)),
        ("repetition_penalty".into(), serde_json::json!(1.05)),
    ]),
    ..StreamOptions::default()
};

assert_eq!(options.sampling["top_k"], 40);
```

The compatibility defaults preserve bundled-provider request fixtures. A wire
shape changes only when a provider deliberately declares a new policy or a
caller opts into a new request control.
