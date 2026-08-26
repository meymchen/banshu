use std::env;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use banshu_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, BeforeSendObservation, Context,
    InMemoryCredentialStore, Message, MiniMaxRegion, Model, Provider, ReasoningEffort,
    ReasoningOptions, RequestObserver, ResponseObservation, StopReason, StreamOptions, Tool,
    ToolCall, ToolChoice,
};
use futures_util::StreamExt;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const BASIC_MAX_TOKENS: u32 = 256;
const EXTENDED_MAX_TOKENS: u32 = 1_024;
const TOOL_MAX_TOKENS: u32 = 512;
const TOOL_RESULT_MAX_TOKENS: u32 = 512;
const ECHO_VALUE: &str = "banshu-smoke";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProviderId {
    DeepSeek,
    Kimi,
    MiniMax,
}

impl ProviderId {
    const ALL: [Self; 3] = [Self::DeepSeek, Self::Kimi, Self::MiniMax];

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "deepseek" => Ok(Self::DeepSeek),
            "kimi" => Ok(Self::Kimi),
            "minimax" => Ok(Self::MiniMax),
            _ => Err(format!(
                "unknown provider `{value}`; expected deepseek, kimi, or minimax"
            )),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::DeepSeek => "deepseek",
            Self::Kimi => "kimi",
            Self::MiniMax => "minimax",
        }
    }

    const fn key_env(self) -> &'static str {
        match self {
            Self::DeepSeek => "DEEPSEEK_API_KEY",
            Self::Kimi => "KIMI_API_KEY",
            Self::MiniMax => "MINIMAX_API_KEY",
        }
    }

    const fn model_env(self) -> &'static str {
        match self {
            Self::DeepSeek => "BANSHU_AI_DEEPSEEK_MODEL",
            Self::Kimi => "BANSHU_AI_KIMI_MODEL",
            Self::MiniMax => "BANSHU_AI_MINIMAX_MODEL",
        }
    }

    const fn default_model(self) -> &'static str {
        match self {
            Self::DeepSeek => "deepseek-v4-flash",
            Self::Kimi => "k3-256k",
            Self::MiniMax => "MiniMax-M3",
        }
    }

    fn build(self) -> Provider {
        let store = Arc::new(InMemoryCredentialStore::new());
        match self {
            Self::DeepSeek => Provider::deepseek(),
            Self::Kimi => Provider::kimi(store),
            Self::MiniMax => Provider::minimax(MiniMaxRegion::Cn, store),
        }
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[derive(Debug, Default)]
struct Args {
    provider: Option<ProviderId>,
    model: Option<String>,
    extended: bool,
    verbose: bool,
}

impl Args {
    fn parse() -> Result<Option<Self>, String> {
        let mut parsed = Self::default();
        let mut args = env::args().skip(1);

        while let Some(argument) = args.next() {
            match argument.as_str() {
                "-h" | "--help" => {
                    print_help();
                    return Ok(None);
                }
                "--extended" => parsed.extended = true,
                "--verbose" => parsed.verbose = true,
                "--provider" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--provider requires a value".to_string())?;
                    parsed.provider = Some(ProviderId::parse(&value)?);
                }
                "--model" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--model requires a value".to_string())?;
                    parsed.model = Some(nonempty("--model", value)?);
                }
                _ if argument.starts_with("--provider=") => {
                    let value = argument.trim_start_matches("--provider=");
                    parsed.provider = Some(ProviderId::parse(value)?);
                }
                _ if argument.starts_with("--model=") => {
                    let value = argument.trim_start_matches("--model=").to_string();
                    parsed.model = Some(nonempty("--model", value)?);
                }
                _ => return Err(format!("unknown argument `{argument}`; try --help")),
            }
        }

        if parsed.model.is_some() && parsed.provider.is_none() {
            return Err(
                "--model requires --provider because model ids are provider-specific".into(),
            );
        }

        Ok(Some(parsed))
    }
}

#[derive(Default)]
struct EventStats {
    text_delta: bool,
    thinking_delta: bool,
    tool_call: bool,
}

struct StreamResult {
    message: AssistantMessage,
    events: EventStats,
    elapsed: Duration,
}

struct VerboseObserver;

impl RequestObserver for VerboseObserver {
    fn before_send(&self, observation: &BeforeSendObservation) {
        eprintln!(
            "VERBOSE request provider={} model={} attempt={} url={} headers={:?}",
            observation.provider,
            observation.model,
            observation.attempt,
            observation.url,
            observation.headers
        );
        match serde_json::to_string_pretty(&observation.payload) {
            Ok(payload) => eprintln!("VERBOSE payload={payload}"),
            Err(error) => eprintln!("VERBOSE payload=<serialization failed: {error}>"),
        }
    }

    fn on_response(&self, observation: &ResponseObservation) {
        eprintln!(
            "VERBOSE response provider={} model={} attempt={} status={} request_id={:?} headers={:?}",
            observation.provider,
            observation.model,
            observation.attempt,
            observation.status,
            observation.request_id,
            observation.headers
        );
    }
}

#[tokio::main]
async fn main() {
    let Some(args) = (match Args::parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("ERROR {error}");
            std::process::exit(2);
        }
    }) else {
        return;
    };

    let selected: Vec<_> = args
        .provider
        .map_or_else(|| ProviderId::ALL.to_vec(), |provider| vec![provider]);
    let mut checked = 0_usize;
    let mut failures = 0_usize;

    for provider_id in selected {
        if !env_is_set(provider_id.key_env()) {
            if args.provider.is_some() {
                eprintln!(
                    "FAIL {provider_id}: required environment variable {} is not set",
                    provider_id.key_env()
                );
                failures += 1;
            } else {
                println!(
                    "SKIP {provider_id}: environment variable {} is not set",
                    provider_id.key_env()
                );
            }
            continue;
        }

        checked += 1;
        let provider = provider_id.build();
        let model_id = args
            .model
            .clone()
            .or_else(|| nonempty_env(provider_id.model_env()))
            .unwrap_or_else(|| provider_id.default_model().to_string());
        let Some(model) = provider
            .models()
            .into_iter()
            .find(|candidate| candidate.id == model_id)
        else {
            let available = provider
                .models()
                .into_iter()
                .map(|model| model.id)
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "FAIL {provider_id}: model `{model_id}` is not in the bundled catalog; available: {available}"
            );
            failures += 1;
            continue;
        };

        println!(
            "CHECK {provider_id}: model={model_id} extended={}",
            args.extended
        );
        if let Err(error) = run_provider(provider_id, &provider, &model, &args).await {
            eprintln!("FAIL {provider_id}: {error}");
            failures += 1;
        }
    }

    if checked == 0 && args.provider.is_none() {
        println!("No API keys were configured; no live checks ran.");
    }
    if failures > 0 {
        std::process::exit(1);
    }
}

async fn run_provider(
    provider_id: ProviderId,
    provider: &Provider,
    model: &Model,
    args: &Args,
) -> Result<(), String> {
    let observer = args
        .verbose
        .then(|| Arc::new(VerboseObserver) as Arc<dyn RequestObserver>);

    let basic = run_stream(
        provider,
        model,
        Context::new()
            .with_system("Reply concisely.")
            .user("Reply with exactly: banshu smoke ok"),
        options(BASIC_MAX_TOKENS, None, None, observer.clone()),
    )
    .await?;
    require_text_success("basic", &basic)?;
    print_pass(provider_id, "basic", &basic);

    if !args.extended {
        return Ok(());
    }

    let reasoning = run_stream(
        provider,
        model,
        Context::new()
            .with_system("Solve the tiny problem, then answer concisely.")
            .user("What is 17 + 25?"),
        options(
            EXTENDED_MAX_TOKENS,
            Some(ReasoningOptions::new(ReasoningEffort::Low)),
            None,
            observer.clone(),
        ),
    )
    .await?;
    require_text_success("reasoning", &reasoning)?;
    if !reasoning.events.thinking_delta {
        return Err("reasoning: no non-empty thinking stream event was observed".into());
    }
    print_pass(provider_id, "reasoning", &reasoning);

    run_tool_round_trip(provider_id, provider, model, observer).await
}

async fn run_tool_round_trip(
    provider_id: ProviderId,
    provider: &Provider,
    model: &Model,
    observer: Option<Arc<dyn RequestObserver>>,
) -> Result<(), String> {
    let echo = Tool {
        name: "echo".into(),
        description:
            "Return the supplied value unchanged. Use this when explicitly asked to echo a value."
                .into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "string", "const": ECHO_VALUE }
            },
            "required": ["value"],
            "additionalProperties": false
        }),
        strict: false,
    };
    let context = Context::new()
        .with_system("Follow the tool instruction exactly.")
        .user(format!(
            "Call the echo tool exactly once with value `{ECHO_VALUE}`. Do not answer before calling it."
        ))
        .with_tool(echo.clone());
    let tool_choice = match provider_id {
        ProviderId::DeepSeek => Some(ToolChoice::Auto),
        ProviderId::Kimi => None,
        ProviderId::MiniMax => Some(ToolChoice::Named("echo".into())),
    };
    let first = run_stream(
        provider,
        model,
        context.clone(),
        options(TOOL_MAX_TOKENS, None, tool_choice, observer.clone()),
    )
    .await?;
    require_stop("tool request", &first.message, StopReason::ToolUse)?;
    if !first.events.tool_call {
        return Err("tool request: no tool-call stream event was observed".into());
    }

    let calls: Vec<ToolCall> = first
        .message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect();
    if calls.len() != 1 {
        return Err(format!(
            "tool request: expected exactly one tool call, received {}",
            calls.len()
        ));
    }
    let call = &calls[0];
    if call.name != echo.name {
        return Err(format!(
            "tool request: expected tool `{}`, received `{}`",
            echo.name, call.name
        ));
    }
    echo.validate_arguments(&call.arguments)
        .map_err(|error| format!("tool request: {error}"))?;

    let mut follow_up = context
        .with_message(Message::Assistant(Box::new(first.message)))
        .tool_result(&call.id, &call.name, ECHO_VALUE)
        .user("Now reply with exactly the tool result value.");
    follow_up.tools.clear();
    let second = run_stream(
        provider,
        model,
        follow_up,
        options(TOOL_RESULT_MAX_TOKENS, None, None, observer),
    )
    .await?;
    require_text_success("tool result", &second)?;
    if !second.message.text().contains(ECHO_VALUE) {
        return Err(format!(
            "tool result: final response did not contain `{ECHO_VALUE}`"
        ));
    }
    print_pass(provider_id, "tool-round-trip", &second);
    Ok(())
}

fn options(
    max_tokens: u32,
    reasoning: Option<ReasoningOptions>,
    tool_choice: Option<ToolChoice>,
    observer: Option<Arc<dyn RequestObserver>>,
) -> StreamOptions {
    StreamOptions {
        max_tokens: Some(max_tokens),
        timeout: Some(REQUEST_TIMEOUT),
        max_retries: Some(0),
        reasoning,
        tool_choice,
        observer,
        ..StreamOptions::default()
    }
}

async fn run_stream(
    provider: &Provider,
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> Result<StreamResult, String> {
    let started = Instant::now();
    let mut stream = provider.stream(model, &context, &options);
    let mut events = EventStats::default();

    let message = tokio::time::timeout(REQUEST_TIMEOUT, async {
        while let Some(event) = stream.next().await {
            match event {
                AssistantMessageEvent::TextDelta { ref delta, .. } if !delta.is_empty() => {
                    events.text_delta = true;
                }
                AssistantMessageEvent::ThinkingDelta { ref delta, .. } if !delta.is_empty() => {
                    events.thinking_delta = true;
                }
                AssistantMessageEvent::ToolCallStart { .. }
                | AssistantMessageEvent::ToolCallDelta { .. }
                | AssistantMessageEvent::ToolCallEnd { .. } => {
                    events.tool_call = true;
                }
                _ => {}
            }
        }
        stream.result().cloned()
    })
    .await
    .map_err(|_| {
        format!(
            "request exceeded the {}s timeout",
            REQUEST_TIMEOUT.as_secs()
        )
    })?
    .ok_or_else(|| "stream ended without a terminal event".to_string())?;

    Ok(StreamResult {
        message,
        events,
        elapsed: started.elapsed(),
    })
}

fn require_text_success(label: &str, result: &StreamResult) -> Result<(), String> {
    require_stop(label, &result.message, StopReason::Stop)?;
    if !result.events.text_delta {
        return Err(format!(
            "{label}: no non-empty text stream event was observed"
        ));
    }
    if result.message.text().trim().is_empty() {
        return Err(format!("{label}: final response text was empty"));
    }
    Ok(())
}

fn require_stop(
    label: &str,
    message: &AssistantMessage,
    expected: StopReason,
) -> Result<(), String> {
    if let Some(error) = &message.error_message {
        return Err(format!(
            "{label}: {:?}: {error}",
            message.error_kind.unwrap_or(banshu_ai::ErrorKind::Api)
        ));
    }
    if message.stop_reason != expected {
        return Err(format!(
            "{label}: expected stop reason {expected:?}, received {:?} (raw: {:?})",
            message.stop_reason, message.raw_stop_reason
        ));
    }
    Ok(())
}

fn print_pass(provider: ProviderId, check: &str, result: &StreamResult) {
    let usage = &result.message.usage;
    println!(
        "PASS {provider}/{check}: model={} elapsed_ms={} stop={:?} usage=input:{} output:{} cache_read:{} cache_write:{} text={:?}",
        result.message.model,
        result.elapsed.as_millis(),
        result.message.stop_reason,
        usage.input,
        usage.output,
        usage.cache_read,
        usage.cache_write,
        summarize(&result.message.text())
    );
}

fn summarize(text: &str) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut summary: String = flattened.chars().take(120).collect();
    if flattened.chars().count() > 120 {
        summary.push('…');
    }
    summary
}

fn env_is_set(name: &str) -> bool {
    env::var_os(name).is_some_and(|value| !value.is_empty())
}

fn nonempty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn nonempty(flag: &str, value: String) -> Result<String, String> {
    if value.trim().is_empty() {
        Err(format!("{flag} requires a non-empty value"))
    } else {
        Ok(value)
    }
}

fn print_help() {
    println!(
        "banshu-ai live provider smoke test\n\n\
Usage: live_smoke [OPTIONS]\n\n\
Options:\n  \
  --provider <deepseek|kimi|minimax>  Check one provider (default: all configured)\n  \
  --model <MODEL_ID>                  Override the selected provider's model\n  \
  --extended                          Also check reasoning and a tool round trip\n  \
  --verbose                           Print redacted request and response diagnostics\n  \
  -h, --help                          Print help\n\n\
Environment:\n  \
  DEEPSEEK_API_KEY, KIMI_API_KEY, MINIMAX_API_KEY\n  \
  BANSHU_AI_DEEPSEEK_MODEL, BANSHU_AI_KIMI_MODEL, BANSHU_AI_MINIMAX_MODEL"
    );
}
