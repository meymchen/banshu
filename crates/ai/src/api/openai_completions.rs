//! The OpenAI `chat/completions` streaming protocol.
//!
//! Every provider in banshu that isn't Anthropic-compatible speaks this. The
//! implementation builds the request body synchronously from borrowed context,
//! then streams SSE, mapping deltas into banshu events and assembling the final
//! [`AssistantMessage`].

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use super::assembler::{MessageAssembler, is_terminal};
use super::protocol_event::ProtocolEvent;
use super::{ApiRequest, ChatApi, compute_cost};
use crate::CacheRetention;
use crate::executor::{self, ExecutorEvent};
use crate::http;
use crate::provider::{OpenAiCompat, OpenAiPromptCaching};
use crate::stream::{AssistantMessageEvent, MessageStream};
use crate::types::{
    AssistantContent, AssistantMessage, Context, Diagnostic, DiagnosticCode, Message, Model,
    StopReason, ThinkingContent, Usage,
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
            request.openai_compat,
        );
        let base_url = request.model.base_url.clone();
        let auth = request.auth.clone();
        let explicit_key = request.options.api_key.clone();
        let http = request.http.clone();
        let model_id = request.model.id.clone();
        let provider = request.model.provider.clone();
        let cost = request.model.cost.clone();
        let timeout = request.options.timeout;
        let max_retries = request
            .options
            .max_retries
            .unwrap_or(http::DEFAULT_MAX_RETRIES);
        let max_retry_delay = request.options.max_retry_delay;
        let cache_retention = request
            .options
            .cache_retention
            .unwrap_or(CacheRetention::Short);
        let session_id = request.options.session_id.clone();
        let prompt_caching = request.openai_compat.prompt_caching;

        let stream = async_stream::stream! {
            let mut assembler = MessageAssembler::new(AssistantMessage::streaming(&model_id, &provider, API_NAME));
            yield AssistantMessageEvent::Start;

            let resolved = match crate::auth::resolve_for_request(&auth, explicit_key).await {
                Ok(resolved) => resolved,
                Err(err) => {
                    yield assembler.fail(crate::ErrorKind::Auth, err.to_string(), Vec::new());
                    return;
                }
            };
            let base = resolved.base_url.as_deref().unwrap_or(&base_url);
            let url = format!("{}/chat/completions", base.trim_end_matches('/'));
            let api_key = resolved.api_key;
            let extra_headers = resolved.headers;

            let session_headers = (prompt_caching == OpenAiPromptCaching::SessionAffinityHeaders
                && cache_retention != CacheRetention::Disabled)
                .then_some(session_id)
                .flatten();
            let factory = move || {
                let mut builder = http.post(&url).json(&body);
                if let Some(api_key) = &api_key {
                    builder = builder.bearer_auth(api_key);
                }
                for (name, value) in &extra_headers {
                    if let Some(value) = value {
                        builder = builder.header(name, value);
                    }
                }
                if let Some(session_id) = &session_headers {
                    builder = builder
                        .header("session_id", session_id)
                        .header("x-client-request-id", session_id)
                        .header("x-session-affinity", session_id);
                }
                if let Some(timeout) = timeout {
                    builder = builder.timeout(timeout);
                }
                builder
            };

            let mut next_block_id: u64 = 0;
            let mut thinking_block_id: Option<u64> = None;
            let mut text_block_id: Option<u64> = None;
            let mut tools: Vec<ToolAccum> = Vec::new();
            let mut usage = Usage::default();
            let mut stop_reason = StopReason::Stop;
            // The OpenAI wire terminator is `data: [DONE]`, but some
            // compatible servers close the connection right after the
            // finish_reason-bearing chunk without sending it; either counts
            // as having formally terminated. A bare EOF with neither is a
            // dropped connection, not a completed response.
            let mut terminated_formally = false;

            let mut exec = std::pin::pin!(executor::execute(factory, max_retries, max_retry_delay));
            'outer: while let Some(exec_event) = exec.next().await {
                let data = match exec_event {
                    ExecutorEvent::Retry { attempt, max_attempts, delay, kind } => {
                        if let Some(event) = assembler.apply(ProtocolEvent::Retry { attempt, max_attempts, delay, kind }) {
                            yield event;
                        }
                        continue;
                    }
                    ExecutorEvent::Established { request_id } => {
                        let _ = assembler.apply(ProtocolEvent::ResponseMetadata { response_id: request_id, response_model: None });
                        continue;
                    }
                    ExecutorEvent::Eof => break 'outer,
                    ExecutorEvent::Failed { kind, message: detail, diagnostics } => {
                        yield assembler.fail(kind, detail, diagnostics);
                        return;
                    }
                    ExecutorEvent::Event(sse_event) => sse_event.data,
                };
                if data == "[DONE]" {
                    terminated_formally = true;
                    break 'outer;
                }
                let value = match super::parse_sse_json(data) {
                    Ok(value) => value,
                    Err((detail, diagnostic)) => {
                        yield assembler.fail(crate::ErrorKind::Protocol, detail, vec![diagnostic]);
                        return;
                    }
                };
                if value.get("error").is_some() {
                    let detail = http::json_error_summary(&value)
                        .unwrap_or_else(|| "provider returned an error".to_string());
                    yield assembler.fail(crate::ErrorKind::Api, detail, Vec::new());
                    return;
                }
                let parsed = match serde_json::from_value::<ChatChunk>(value.clone()) {
                    Ok(parsed) => parsed,
                    Err(_) => {
                        yield assembler.fail(
                            crate::ErrorKind::Protocol,
                            "unrecognized SSE chunk shape",
                            vec![Diagnostic::new(DiagnosticCode::ProtocolViolation, value.to_string())],
                        );
                        return;
                    }
                };
                let chunk_usage = parsed
                    .usage
                    .as_ref()
                    .or_else(|| parsed.choices.first().and_then(|choice| choice.usage.as_ref()));
                if let Some(chunk_usage) = chunk_usage {
                    usage = normalize_usage(chunk_usage);
                }
                if let Some(choice) = parsed.choices.into_iter().next() {
                    // Endpoints disagree on the reasoning field name; take the
                    // first non-empty one (some send duplicates) and remember
                    // it as the block's signature so replay can use it.
                    let reasoning_delta = [
                        ("reasoning_content", &choice.delta.reasoning_content),
                        ("reasoning", &choice.delta.reasoning),
                        ("reasoning_text", &choice.delta.reasoning_text),
                    ]
                    .into_iter()
                    .find_map(|(field, value)| {
                        let value = value.as_deref().filter(|v| !v.is_empty())?;
                        Some((field, value.to_string()))
                    });
                    if let Some((field, reasoning)) = reasoning_delta {
                        let block_id = match thinking_block_id {
                            Some(id) => id,
                            None => {
                                let id = next_block_id;
                                next_block_id += 1;
                                thinking_block_id = Some(id);
                                if let Some(event) = assembler.apply(ProtocolEvent::ThinkingStart {
                                    block_id: id,
                                    signature: Some(field.to_string()),
                                    redacted: false,
                                }) {
                                    let terminal = is_terminal(&event);
                                    yield event;
                                    if terminal { return; }
                                }
                                id
                            }
                        };
                        if let Some(event) = assembler.apply(ProtocolEvent::ThinkingDelta { block_id, delta: reasoning }) {
                            let terminal = is_terminal(&event);
                            yield event;
                            if terminal { return; }
                        }
                    }
                    if let Some(delta) = choice.delta.content
                        && !delta.is_empty()
                    {
                        let block_id = match text_block_id {
                            Some(id) => id,
                            None => {
                                let id = next_block_id;
                                next_block_id += 1;
                                text_block_id = Some(id);
                                if let Some(event) = assembler.apply(ProtocolEvent::TextStart { block_id: id, signature: None }) {
                                    let terminal = is_terminal(&event);
                                    yield event;
                                    if terminal { return; }
                                }
                                id
                            }
                        };
                        if let Some(event) = assembler.apply(ProtocolEvent::TextDelta { block_id, delta }) {
                            let terminal = is_terminal(&event);
                            yield event;
                            if terminal { return; }
                        }
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
                        terminated_formally = true;
                    }
                }
            }

            if !terminated_formally {
                yield assembler.fail(
                    crate::ErrorKind::StreamInterrupted,
                    "connection closed before a completion signal ([DONE] or finish_reason)",
                    Vec::new(),
                );
                return;
            }

            // Each `*End` now emits a public `TextEnd`/`ThinkingEnd` event
            // carrying the finished content; a protocol violation (unknown/
            // already-ended/mismatched block) instead comes back as a terminal
            // `Error`, so every site checks `is_terminal` before continuing.
            if let Some(block_id) = thinking_block_id
                && let Some(event) = assembler.apply(ProtocolEvent::ThinkingEnd { block_id })
            {
                let terminal = is_terminal(&event);
                yield event;
                if terminal { return; }
            }
            if let Some(block_id) = text_block_id
                && let Some(event) = assembler.apply(ProtocolEvent::TextEnd { block_id })
            {
                let terminal = is_terminal(&event);
                yield event;
                if terminal { return; }
            }

            // v0.3 collapses a tool call's fragments into one Start+Delta+End
            // each, emitted at completion rather than per wire delta.
            for accum in tools {
                if accum.is_empty() {
                    continue;
                }
                let block_id = next_block_id;
                next_block_id += 1;
                if let Some(event) = assembler.apply(ProtocolEvent::ToolCallStart { block_id, id: accum.id, name: accum.name }) {
                    let terminal = is_terminal(&event);
                    yield event;
                    if terminal { return; }
                }
                if let Some(event) = assembler.apply(ProtocolEvent::ToolCallDelta { block_id, delta: accum.arguments }) {
                    let terminal = is_terminal(&event);
                    yield event;
                    if terminal { return; }
                }
                if let Some(event) = assembler.apply(ProtocolEvent::ToolCallEnd { block_id }) {
                    let terminal = is_terminal(&event);
                    yield event;
                    if terminal { return; }
                }
            }

            usage.cost = compute_cost(&usage, &cost);
            let _ = assembler.apply(ProtocolEvent::Usage(usage));
            let _ = assembler.apply(ProtocolEvent::Stop(stop_reason));

            let message = assembler.into_message();
            yield AssistantMessageEvent::Done { reason: stop_reason, message };
        };

        MessageStream::new(stream)
    }
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
    compat: OpenAiCompat,
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
                wire["content"] = if text.is_empty() {
                    Value::Null
                } else {
                    Value::String(text)
                };
                if !tool_calls.is_empty() {
                    wire["tool_calls"] = Value::Array(tool_calls);
                }
                // Replay thinking under the wire field it arrived in (recorded
                // as the block's signature at capture time); signatureless
                // thinking has nowhere faithful to go and is dropped.
                let thinking: Vec<&ThinkingContent> = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Thinking(block) if !block.thinking.trim().is_empty() => {
                            Some(block)
                        }
                        _ => None,
                    })
                    .collect();
                if let Some(field) = thinking
                    .first()
                    .and_then(|block| block.signature.as_deref())
                    .filter(|field| !field.is_empty())
                {
                    let joined: Vec<&str> = thinking
                        .iter()
                        .map(|block| block.thinking.as_str())
                        .collect();
                    wire[field] = Value::String(joined.join("\n"));
                }
                if compat.requires_reasoning_content_on_assistant_messages
                    && model.reasoning
                    && wire.get("reasoning_content").is_none()
                {
                    wire["reasoning_content"] = Value::String(String::new());
                }
                messages.push(wire);
            }
            Message::ToolResult(result) => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.text_content(),
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
    let openai_cache = compat.prompt_caching == OpenAiPromptCaching::OpenAi
        && cache_retention != CacheRetention::Disabled;

    ChatRequest {
        model: model.id.clone(),
        messages,
        tools,
        stream: true,
        stream_options: StreamOpts {
            include_usage: true,
        },
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
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_text: Option<String>,
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
