# banshu-ai

`banshu-ai` is a Rust library for streaming chat with open-source LLM
providers over the OpenAI Chat Completions and Anthropic Messages protocols.

## Installation

```toml
[dependencies]
banshu-ai = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Quick start

Set `DEEPSEEK_API_KEY`, then select a catalog model and complete one turn:

```no_run
use banshu_ai::{Context, Provider, StreamOptions};

#[tokio::main]
async fn main() {
    let provider = Provider::deepseek();
    let model = provider
        .models()
        .into_iter()
        .find(|model| model.id == "deepseek-chat")
        .expect("deepseek-chat is in the bundled catalog");

    let message = provider
        .stream(
            &model,
            &Context::new().with_system("Be concise.").user("Hello!"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    if let Some(error) = &message.error_message {
        eprintln!("request failed ({:?}): {error}", message.error_kind);
    } else {
        println!("{}", message.text());
    }
}
```

`Provider::deepseek`, `zai`, `moonshot`, and `xiaomi` use OpenAI Chat
Completions. `Provider::kimi` and `minimax` use Anthropic Messages. The
[provider conformance matrix](docs/provider-conformance.md) records the exact
auth and feature contract for each bundled provider.

## Custom providers

Use the validated builder when a service needs explicit models, authentication,
headers, compatibility settings, or more than one protocol adapter:

```rust
use std::sync::Arc;
use banshu_ai::{Auth, Model, Provider};
use banshu_ai::api::openai_completions::OpenAiCompletions;

let mut model = Model::openai_completions("acme-chat")
    .with_base_url("https://llm.example/v1");
model.provider = "acme".into();

let provider = Provider::builder("acme", "Acme", "https://llm.example/v1")
    .auth(Auth::api_key_env(["ACME_API_KEY"]))
    .adapter(Arc::new(OpenAiCompletions))
    .model(model)
    .build()?;

assert_eq!(provider.id(), "acme");
# Ok::<(), banshu_ai::Error>(())
```

For a simple single-protocol endpoint,
`Provider::openai_compatible(...)` and `Provider::anthropic_compatible(...)`
are concise alternatives. Use `Auth::keyless()` for a local unauthenticated
server or `Auth::custom(...)` for an application-owned resolver.

## OAuth

Kimi For Coding and MiniMax Coding Plan are OAuth-first and accept an
application-owned credential store. An API key in `KIMI_API_KEY` or
`MINIMAX_API_KEY` is an operator override. The following setup path performs no
network access and needs no real credential:

```rust
use std::sync::Arc;
use banshu_ai::{InMemoryCredentialStore, MiniMaxRegion, Models, Provider};

let store = Arc::new(InMemoryCredentialStore::new());
let models = Models::new()
    .with_provider(Provider::kimi(store.clone()))
    .with_provider(Provider::minimax(MiniMaxRegion::Global, store));

assert!(models.provider("kimi").unwrap().oauth_session().is_some());
assert!(models.provider("minimax").unwrap().oauth_session().is_some());
```

To log in, implement `AuthInteractionHandler` to display
`VerificationDetails`, wrap it in `AuthInteraction`, and call
`Models::login(provider_id, &interaction)`. The same registry provides
`check_auth`, `refresh_credential`, and `logout`. Use a durable
`CredentialStore` implementation in production; `InMemoryCredentialStore` is
process-local.

## Tools, images, and reasoning

Add JSON-Schema tools with `Context::with_tool`, return results with
`Context::tool_result`, and select a supported `ToolChoice` through
`StreamOptions::tool_choice`. Streamed `ToolCall` values retain both parsed
`arguments` and the provider's `raw_arguments`; validate the parsed value with
`Tool::validate_arguments` before execution.

Images are `UserContent::Image(ImageContent { data, mime_type })`. The newest
user image is rejected before dispatch when the selected model does not attest
image input; historical and tool-result images are downgraded to explicit text
placeholders. Check `model.input` for `Modality::Image` rather than assuming a
provider-wide capability.

Reasoning is an explicit `ReasoningOptions` request in `StreamOptions`. The
model metadata and provider request format are both checked before dispatch;
unsupported efforts and token budgets fail in-band instead of being silently
clamped. `ReasoningEffort::Off` actively disables reasoning, while a missing
option leaves the provider default untouched.

## Cancellation and retry

Put a `CancellationToken` in `StreamOptions::cancellation` to cover credential
resolution, connection setup, retry sleeps, and response reads. Cancellation
returns a terminal message with `StopReason::Aborted` and keeps content already
streamed. `max_retries` bounds retries before content starts, while
`max_retry_delay` caps a server-supplied `Retry-After`; retry progress is
visible as `AssistantMessageEvent::Retry`.

## Caching and persistence

`cache_retention` and `session_id` control request-side prompt caching where
the protocol/provider supports it. Cache-read and cache-write usage is always
normalized into `Usage` when providers report it.

Persist conversations with `ContextSnapshotV1`; its versioned serde shape is
the compatibility boundary. Model discovery overlays use a separate
application-injected `ModelsStore` and `RefreshOptions`; set
`allow_network = false` for a hard offline restore.

## Error handling

Construction, login, persistence, and other setup operations return
`banshu_ai::Result`. Streaming does not: setup failures and mid-stream failures
arrive in the terminal `AssistantMessage`, classified by `stop_reason` and
`error_kind`, with safe detail in `error_message`. Use the normalized
`stop_reason` for portable behavior and `raw_stop_reason` for provider-specific
diagnostics.

See [1.0 migrations](docs/migrations-1.0.md) before upgrading from the latest
0.x release.
