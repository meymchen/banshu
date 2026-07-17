//! The OpenAI `chat/completions` streaming protocol.
//!
//! Every provider in banshu that isn't Anthropic-compatible speaks this. The
//! implementation builds the request body synchronously from borrowed context,
//! then streams SSE, mapping deltas into banshu events and assembling the final
//! [`AssistantMessage`].

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use super::{ApiRequest, ChatApi, compute_cost, fail, parse_arguments};
use crate::CacheRetention;
use crate::http;
use crate::provider::OpenAiPromptCaching;
use crate::stream::{AssistantMessageEvent, MessageStream};
use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, Model, StopReason, TextContent,
    ThinkingContent, ToolCall, Usage,
};

/// The OpenAI-completions wire protocol.
pub struct OpenAiCompletions;

const API_NAME: &str = "openai-completions";
const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

impl ChatApi for OpenAiCompletions {
    fn stream(&self, request: ApiRequest<'_>) -> MessageStream {
        // Extract everything the async body needs as owned values up front, so
        // the returned stream is `'static`.
        let body = build_request_body(
            request.model,
            request.context,
            request.options,
            request.openai_prompt_caching,
        );
        let url = format!(
            "{}/chat/completions",
            request.model.base_url.trim_end_matches('/')
        );
        let api_key = request.api_key.clone();
        let http = request.http.clone();
        let model_id = request.model.id.clone();
        let provider = request.model.provider.clone();
        let cost = request.model.cost.clone();
        let timeout = request.options.timeout;
        let cache_retention = request
            .options
            .cache_retention
            .unwrap_or(CacheRetention::Short);
        let session_id = request.options.session_id.clone();
        let prompt_caching = request.openai_prompt_caching;

        let stream = async_stream::stream! {
            let mut message = AssistantMessage::streaming(&model_id, &provider, API_NAME);
            yield AssistantMessageEvent::Start { partial: message.clone() };

            let Some(api_key) = api_key else {
                yield fail(&mut message, "no API key configured");
                return;
            };

            let mut builder = http.post(&url).bearer_auth(api_key).json(&body);
            if prompt_caching == OpenAiPromptCaching::SessionAffinityHeaders
                && cache_retention != CacheRetention::Disabled
                && let Some(session_id) = session_id
            {
                builder = builder
                    .header("session_id", &session_id)
                    .header("x-client-request-id", &session_id)
                    .header("x-session-affinity", session_id);
            }
            if let Some(timeout) = timeout {
                builder = builder.timeout(timeout);
            }

            let response = match builder.send().await {
                Ok(response) => response,
                Err(err) => {
                    yield fail(&mut message, &format!("request failed: {err}"));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let detail = response.text().await.unwrap_or_default();
                yield fail(&mut message, &format!("HTTP {status}: {detail}"));
                return;
            }

            let mut thinking = String::new();
            let mut text = String::new();
            let mut tools: Vec<ToolAccum> = Vec::new();
            let mut usage = Usage::default();
            let mut stop_reason = StopReason::Stop;
            let events = http::sse_data_lines(response);
            let mut events = std::pin::pin!(events);

            'outer: while let Some(data) = events.next().await {
                let data = match data {
                    Ok(data) => data,
                    Err(err) => {
                        yield fail(&mut message, &format!("stream error: {err}"));
                        return;
                    }
                };
                if data == "[DONE]" {
                    break 'outer;
                }
                let Ok(parsed) = serde_json::from_str::<ChatChunk>(&data) else {
                    continue; // ignore keep-alives / malformed lines
                };
                let chunk_usage = parsed
                    .usage
                    .as_ref()
                    .or_else(|| parsed.choices.first().and_then(|choice| choice.usage.as_ref()));
                if let Some(chunk_usage) = chunk_usage {
                    usage = normalize_usage(chunk_usage);
                }
                if let Some(choice) = parsed.choices.into_iter().next() {
                    if let Some(reasoning) = choice.delta.reasoning_content
                        && !reasoning.is_empty()
                    {
                        thinking.push_str(&reasoning);
                        message.content = partial_content(&thinking, &text);
                        yield AssistantMessageEvent::ThinkingDelta {
                            content_index: 0,
                            delta: reasoning,
                            partial: message.clone(),
                        };
                    }
                    if let Some(delta) = choice.delta.content
                        && !delta.is_empty()
                    {
                        text.push_str(&delta);
                        message.content = partial_content(&thinking, &text);
                        yield AssistantMessageEvent::TextDelta {
                            content_index: 0,
                            delta,
                            partial: message.clone(),
                        };
                    }
                    for delta in choice.delta.tool_calls {
                        let slot = delta.index;
                        if tools.len() <= slot {
                            tools.resize_with(slot + 1, ToolAccum::default);
                        }
                        let accum = &mut tools[slot];
                        if let Some(id) = delta.id {
                            accum.id = id;
                        }
                        if let Some(function) = delta.function {
                            if let Some(name) = function.name {
                                accum.name = name;
                            }
                            if let Some(arguments) = function.arguments {
                                accum.arguments.push_str(&arguments);
                            }
                        }
                    }
                    if let Some(reason) = choice.finish_reason {
                        stop_reason = map_stop_reason(&reason);
                    }
                }
            }

            usage.cost = compute_cost(&usage, &cost);
            message.usage = usage;
            message.stop_reason = stop_reason;

            let mut content = partial_content(&thinking, &text);
            for accum in tools {
                if accum.is_empty() {
                    continue;
                }
                let tool_call = ToolCall {
                    id: accum.id,
                    name: accum.name,
                    arguments: parse_arguments(&accum.arguments),
                };
                let content_index = content.len();
                content.push(AssistantContent::ToolCall(tool_call.clone()));
                message.content = content.clone();
                yield AssistantMessageEvent::ToolCallEnd {
                    content_index,
                    tool_call,
                    partial: message.clone(),
                };
            }

            message.content = content;
            yield AssistantMessageEvent::Done { reason: stop_reason, message };
        };

        MessageStream::new(stream)
    }
}

/// Build ordered content from accumulated thinking and text: thinking first
/// (it streams before the answer), then text. Empty sections are omitted.
fn partial_content(thinking: &str, text: &str) -> Vec<AssistantContent> {
    let mut content = Vec::new();
    if !thinking.is_empty() {
        content.push(AssistantContent::Thinking(ThinkingContent {
            thinking: thinking.to_string(),
            signature: None,
            redacted: false,
        }));
    }
    if !text.is_empty() {
        content.push(AssistantContent::Text(TextContent {
            text: text.to_string(),
            signature: None,
        }));
    }
    content
}

/// Accumulates a single tool call's fragments across streamed deltas.
#[derive(Default)]
struct ToolAccum {
    id: String,
    name: String,
    arguments: String,
}

impl ToolAccum {
    fn is_empty(&self) -> bool {
        self.id.is_empty() && self.name.is_empty() && self.arguments.is_empty()
    }
}

/// Map an OpenAI `finish_reason` to a banshu [`StopReason`].
fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "length" => StopReason::Length,
        "tool_calls" => StopReason::ToolUse,
        _ => StopReason::Stop,
    }
}

/// Normalize the cache accounting variants used by OpenAI-compatible APIs.
///
/// `Usage::input` is always the uncached prompt portion. This prevents cached
/// tokens from being billed twice when computing costs.
fn normalize_usage(raw: &ChunkUsage) -> Usage {
    let (input, cache_read, cache_write) =
        if raw.prompt_cache_hit_tokens.is_some() || raw.prompt_cache_miss_tokens.is_some() {
            // DeepSeek reports the hit and miss portions directly.
            let cache_read = raw.prompt_cache_hit_tokens.unwrap_or_else(|| {
                raw.prompt_tokens
                    .saturating_sub(raw.prompt_cache_miss_tokens.unwrap_or(0))
            });
            let input = raw
                .prompt_cache_miss_tokens
                .unwrap_or_else(|| raw.prompt_tokens.saturating_sub(cache_read));
            (input, cache_read, 0)
        } else {
            let reported_cached = raw
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .unwrap_or(0);
            let cache_write = raw
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cache_write_tokens)
                .unwrap_or(0);

            // Some compatible providers include current-request cache writes
            // in `cached_tokens`; pi-ai removes writes from cache reads.
            let cache_read = if cache_write > 0 {
                reported_cached.saturating_sub(cache_write)
            } else {
                reported_cached
            };
            let input = raw
                .prompt_tokens
                .saturating_sub(cache_read)
                .saturating_sub(cache_write);
            (input, cache_read, cache_write)
        };

    let derived_total = input + cache_read + cache_write + raw.completion_tokens;
    Usage {
        input,
        output: raw.completion_tokens,
        cache_read,
        cache_write,
        reasoning: raw
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        total_tokens: raw.total_tokens.unwrap_or(derived_total),
        ..Usage::default()
    }
}

fn clamp_openai_prompt_cache_key(key: &str) -> String {
    key.chars()
        .take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
        .collect()
}

fn build_request_body(
    model: &Model,
    context: &Context,
    options: &crate::StreamOptions,
    prompt_caching: OpenAiPromptCaching,
) -> ChatRequest {
    use serde_json::{Value, json};

    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = &context.system_prompt {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for message in &context.messages {
        match message {
            Message::User(user) => {
                messages.push(json!({ "role": "user", "content": user.text_content() }));
            }
            Message::Assistant(assistant) => {
                let tool_calls: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::ToolCall(call) => Some(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments.to_string(),
                            },
                        })),
                        _ => None,
                    })
                    .collect();
                let text = assistant.text();
                let mut wire = json!({ "role": "assistant" });
                wire["content"] = if text.is_empty() { Value::Null } else { Value::String(text) };
                if !tool_calls.is_empty() {
                    wire["tool_calls"] = Value::Array(tool_calls);
                }
                messages.push(wire);
            }
            Message::ToolResult(result) => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }));
            }
        }
    }

    let tools = context
        .tools
        .iter()
        .map(|tool| WireTool {
            kind: "function",
            function: WireFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            },
        })
        .collect();

    let cache_retention = options.cache_retention.unwrap_or(CacheRetention::Short);
    let openai_cache = prompt_caching == OpenAiPromptCaching::OpenAi
        && cache_retention != CacheRetention::Disabled;

    ChatRequest {
        model: model.id.clone(),
        messages,
        tools,
        stream: true,
        stream_options: StreamOpts { include_usage: true },
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        prompt_cache_key: openai_cache
            .then(|| {
                options
                    .session_id
                    .as_deref()
                    .map(clamp_openai_prompt_cache_key)
            })
            .flatten(),
        prompt_cache_retention: (openai_cache && cache_retention == CacheRetention::Long)
            .then_some("24h"),
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    stream: bool,
    stream_options: StreamOpts,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireFunction,
}

#[derive(Serialize)]
struct WireFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct StreamOpts {
    include_usage: bool,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Deserialize, Default)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u64>,
    #[serde(default)]
    prompt_cache_miss_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
    #[serde(default)]
    cache_write_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}
