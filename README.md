# banshu-ai

`banshu-ai` is a Rust library for streaming chat with open-source LLM
providers over the OpenAI Chat Completions and Anthropic Messages protocols.

## Installation

```toml
[dependencies]
banshu-ai = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
# Needed only when constructing an application-owned HTTP client or JSON
# sampling values, as in the custom-provider example below.
reqwest = "0.12"
serde_json = "1"
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

## Live provider smoke tests

From a repository checkout, export one or more provider API keys and run the
live smoke test. It checks every configured provider and skips providers whose
key is absent:

```bash
export DEEPSEEK_API_KEY="..."
export KIMI_API_KEY="..."
export MINIMAX_API_KEY="..."
scripts/smoke-ai.sh
```

The defaults are `deepseek-v4-flash`, `k3-256k`, and `MiniMax-M3` on the
MiniMax CN endpoint. Select one provider, override its bundled-catalog model,
or add reasoning and a two-turn `echo` tool-call check with:

```bash
scripts/smoke-ai.sh --provider kimi
scripts/smoke-ai.sh --provider deepseek --model deepseek-v4-pro
scripts/smoke-ai.sh --provider minimax --extended
```

To exercise Kimi's OAuth device flow with a Kimi Code subscription, make sure
the API-key override is absent, then run:

```bash
unset KIMI_API_KEY
scripts/smoke-ai.sh --provider kimi --oauth
```

The command prints the verification URL and user code, waits for browser
authorization, checks that the in-memory OAuth credential is available, and
uses that credential for the live inference request. After the live checks
pass, it logs out and verifies that the OAuth credential is no longer stored.
The credential exists only within the smoke-test process.

Use `--verbose` to print the request observer's redacted payload, URL, headers,
and response metadata. Each live request has a 30-second timeout, zero retries,
and a small output budget. The script never prints API keys and does not load
`.env` automatically; [the repository `.env.example`](https://github.com/meymchen/banshu/blob/main/.env.example)
lists the supported variables and model overrides.

The crate-owned HTTP client identifies every inference, OAuth, discovery, and
probe request as `banshu-ai/<crate version>`. An application-owned client
injected with `Provider::with_http_client` keeps the application's own default
header policy.

`Provider::deepseek`, `zai`, `moonshot`, and `xiaomi` use OpenAI Chat
Completions. `Provider::kimi` and `minimax` use Anthropic Messages. The
[provider conformance matrix](https://github.com/meymchen/banshu/blob/main/docs/provider-conformance.md) records the exact
auth and feature contract for each bundled provider.

## Custom providers

Use the validated builder when a service needs explicit models, authentication,
an application-owned HTTP client, endpoint compatibility declarations, or more
than one protocol adapter. This setup also shows typed reasoning and unmodelled
OpenAI-compatible sampling controls:

```rust
use std::collections::BTreeMap;
use std::sync::Arc;
use banshu_ai::api::openai_completions::OpenAiCompletions;
use banshu_ai::{
    Auth, Model, OpenAiChatTemplateKwargs, OpenAiCompat,
    OpenAiReasoningBudgetField, OpenAiReasoningFormat, Provider,
    ReasoningEffort, ReasoningOptions, StreamOptions,
};

let mut model = Model::openai_completions("acme-chat")
    .with_base_url("https://llm.example/v1");
model.provider = "acme".into();

let http = reqwest::Client::builder().build()?;
let provider = Provider::builder("acme", "Acme", "https://llm.example/v1")
    .auth(Auth::api_key_env(["ACME_API_KEY"]))
    .adapter(Arc::new(OpenAiCompletions))
    .http_client(http)
    .openai_compat(OpenAiCompat {
        reasoning_format: OpenAiReasoningFormat::ChatTemplateKwargs(
            OpenAiChatTemplateKwargs {
                enable_thinking: Some("enable_thinking"),
                reasoning_effort: Some("reasoning_effort"),
                token_budget: Some(OpenAiReasoningBudgetField::ThinkingBudget),
            },
        ),
        ..OpenAiCompat::default()
    })
    .model(model)
    .build()?;

let options = StreamOptions {
    reasoning: Some(ReasoningOptions::new(ReasoningEffort::High)
        .with_token_budget(2_048)),
    sampling: BTreeMap::from([
        ("top_p".into(), serde_json::json!(0.9)),
        ("min_p".into(), serde_json::json!(0.05)),
    ]),
    ..StreamOptions::default()
};

assert_eq!(provider.id(), "acme");
assert_eq!(options.sampling.len(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
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

Custom OpenAI-compatible runtimes can declare a top-level
`OpenAiReasoningFormat::EnableThinking` boolean or typed
`ChatTemplateKwargs`. Chat-template declarations choose keyword names for the
enabled state and effort, while token budgets use the closed
`OpenAiReasoningBudgetField` vocabulary. The adapter supplies the values and
owns the surrounding request object, so a declaration cannot act as an
arbitrary JSON patch. Explicit budgets must be attested by the model and fit
under the resolved Output Budget.

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

Persist conversations by serializing `Context` directly. Model discovery
overlays use a separate application-injected `ModelsStore` and
`RefreshOptions`; set `allow_network = false` for a hard offline restore.

## Error handling

Construction, login, persistence, and other setup operations return
`banshu_ai::Result`. Streaming does not: setup failures and mid-stream failures
arrive in the terminal `AssistantMessage`, classified by `stop_reason` and
`error_kind`, with safe detail in `error_message`. Use the normalized
`stop_reason` for portable behavior and `raw_stop_reason` for provider-specific
diagnostics.
