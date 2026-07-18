//! The Anthropic `/v1/messages` streaming protocol.
//!
//! Used by banshu's Anthropic-compatible providers (Z.AI, MiniMax, Kimi, …).
//! Speaks the Messages SSE event stream: `message_start` carries input usage,
//! `content_block_delta` carries text/thinking/tool fragments, `message_delta`
//! carries the stop reason and output usage, `message_stop` ends the turn.

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::Value;

use super::{ApiRequest, ChatApi, compute_cost, fail, parse_arguments};
use crate::CacheRetention;
use crate::http;
use crate::stream::{AssistantMessageEvent, MessageStream};
use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, Model, StopReason, TextContent,
    ThinkingContent, ToolCall, Usage,
};

/// The Anthropic Messages wire protocol.
pub struct AnthropicMessages;

const API_NAME: &str = "anthropic-messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

impl ChatApi for AnthropicMessages {
    fn stream(&self, request: ApiRequest<'_>) -> MessageStream {
        let body = build_request_body(request.model, request.context, request.options);
        let url = format!(
            "{}/v1/messages",
            request.model.base_url.trim_end_matches('/')
        );
        let api_key = request.api_key.clone();
        let http_client = request.http.clone();
        let model_id = request.model.id.clone();
        let provider = request.model.provider.clone();
        let cost = request.model.cost.clone();
        let timeout = request.options.timeout;
        let max_retries = request
            .options
            .max_retries
            .unwrap_or(http::DEFAULT_MAX_RETRIES);

        let stream = async_stream::stream! {
            let mut message = AssistantMessage::streaming(&model_id, &provider, API_NAME);
            yield AssistantMessageEvent::Start { partial: message.clone() };

            let Some(api_key) = api_key else {
                yield fail(&mut message, crate::ErrorKind::Api, "no API key configured");
                return;
            };

            let mut builder = http_client
                .post(&url)
                .header("x-api-key", api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .json(&body);
            if let Some(timeout) = timeout {
                builder = builder.timeout(timeout);
            }

            // Bounded pre-stream retry: once SSE decoding starts below, no
            // attempt is ever re-sent.
            let mut attempt: u32 = 0;
            let response = loop {
                let this_attempt = builder
                    .try_clone()
                    .expect("JSON request bodies are cloneable");
                match http::send_once(this_attempt).await {
                    Ok(response) => break response,
                    Err(failure) if failure.kind.is_retryable() && attempt < max_retries => {
                        attempt += 1;
                        let delay = http::retry_delay(attempt, failure.retry_after);
                        yield AssistantMessageEvent::Retry {
                            attempt,
                            max_attempts: max_retries + 1,
                            delay,
                            kind: failure.kind,
                            partial: message.clone(),
                        };
                        tokio::time::sleep(delay).await;
                    }
                    Err(failure) => {
                        yield fail(&mut message, failure.kind, &failure.detail);
                        return;
                    }
                }
            };

            let mut blocks: Vec<BlockAccum> = Vec::new();
            let mut usage = Usage::default();
            let mut stop_reason = StopReason::Stop;
            let events = http::sse_data_lines(response);
            let mut events = std::pin::pin!(events);

            'outer: while let Some(data) = events.next().await {
                let data = match data {
                    Ok(data) => data,
                    Err(err) => {
                        yield fail(
                            &mut message,
                            crate::ErrorKind::StreamInterrupted,
                            &format!("stream error: {err}"),
                        );
                        return;
                    }
                };
                let Ok(value) = serde_json::from_str::<Value>(&data) else {
                    continue;
                };
                match value.get("type").and_then(Value::as_str) {
                    Some("message_start") => {
                        let wire = &value["message"]["usage"];
                        usage.input = wire["input_tokens"].as_u64().unwrap_or(0);
                        usage.output = wire["output_tokens"].as_u64().unwrap_or(0);
                        usage.cache_read = wire["cache_read_input_tokens"].as_u64().unwrap_or(0);
                        usage.cache_write =
                            wire["cache_creation_input_tokens"].as_u64().unwrap_or(0);
                        usage.cache_write_1h =
                            wire["cache_creation"]["ephemeral_1h_input_tokens"].as_u64();
                    }
                    Some("content_block_start") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        let block = &value["content_block"];
                        let accum = match block["type"].as_str() {
                            Some("thinking") => BlockAccum::Thinking {
                                text: String::new(),
                                signature: None,
                            },
                            Some("tool_use") => BlockAccum::ToolCall {
                                id: block["id"].as_str().unwrap_or_default().to_string(),
                                name: block["name"].as_str().unwrap_or_default().to_string(),
                                arguments: String::new(),
                            },
                            _ => BlockAccum::Text(String::new()),
                        };
                        if blocks.len() <= index {
                            blocks.resize_with(index + 1, || BlockAccum::Text(String::new()));
                        }
                        blocks[index] = accum;
                    }
                    Some("content_block_delta") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        let Some(block) = blocks.get_mut(index) else { continue };
                        let delta = &value["delta"];
                        match delta["type"].as_str() {
                            Some("text_delta") => {
                                if let (BlockAccum::Text(text), Some(chunk)) =
                                    (block, delta["text"].as_str())
                                    && !chunk.is_empty()
                                {
                                    text.push_str(chunk);
                                    message.content = assemble(&blocks);
                                    yield AssistantMessageEvent::TextDelta {
                                        content_index: index,
                                        delta: chunk.to_string(),
                                        partial: message.clone(),
                                    };
                                }
                            }
                            Some("thinking_delta") => {
                                if let (BlockAccum::Thinking { text, .. }, Some(chunk)) =
                                    (block, delta["thinking"].as_str())
                                    && !chunk.is_empty()
                                {
                                    text.push_str(chunk);
                                    message.content = assemble(&blocks);
                                    yield AssistantMessageEvent::ThinkingDelta {
                                        content_index: index,
                                        delta: chunk.to_string(),
                                        partial: message.clone(),
                                    };
                                }
                            }
                            Some("signature_delta") => {
                                if let (BlockAccum::Thinking { signature, .. }, Some(sig)) =
                                    (block, delta["signature"].as_str())
                                {
                                    *signature = Some(sig.to_string());
                                }
                            }
                            Some("input_json_delta") => {
                                if let (BlockAccum::ToolCall { arguments, .. }, Some(fragment)) =
                                    (block, delta["partial_json"].as_str())
                                {
                                    arguments.push_str(fragment);
                                }
                            }
                            _ => {}
                        }
                    }
                    Some("message_delta") => {
                        if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                            stop_reason = map_stop_reason(reason);
                        }
                        let wire = &value["usage"];
                        if let Some(output) = wire["output_tokens"].as_u64() {
                            usage.output = output;
                        }
                        if let Some(read) = wire["cache_read_input_tokens"].as_u64() {
                            usage.cache_read = read;
                        }
                        if let Some(write) = wire["cache_creation_input_tokens"].as_u64() {
                            usage.cache_write = write;
                        }
                    }
                    Some("message_stop") => break 'outer,
                    _ => {}
                }
            }

            // Anthropic reports no total; derive it from all token classes.
            usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
            usage.cost = compute_cost(&usage, &cost);
            message.content = assemble(&blocks);
            message.usage = usage;
            message.stop_reason = stop_reason;
            yield AssistantMessageEvent::Done { reason: stop_reason, message };
        };

        MessageStream::new(stream)
    }
}

/// A content block being accumulated across streamed deltas, keyed by index.
enum BlockAccum {
    Text(String),
    Thinking {
        text: String,
        signature: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
}

/// Assemble the ordered content blocks into banshu content, dropping empty
/// text/thinking blocks and parsing tool-call arguments.
fn assemble(blocks: &[BlockAccum]) -> Vec<AssistantContent> {
    blocks
        .iter()
        .filter_map(|block| match block {
            BlockAccum::Text(text) if !text.is_empty() => {
                Some(AssistantContent::Text(TextContent {
                    text: text.clone(),
                    signature: None,
                }))
            }
            BlockAccum::Thinking { text, signature } if !text.is_empty() => {
                Some(AssistantContent::Thinking(ThinkingContent {
                    thinking: text.clone(),
                    signature: signature.clone(),
                    redacted: false,
                }))
            }
            BlockAccum::ToolCall {
                id,
                name,
                arguments,
            } => Some(AssistantContent::ToolCall(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: parse_arguments(arguments),
            })),
            _ => None,
        })
        .collect()
}

/// Map an Anthropic `stop_reason` to a banshu [`StopReason`].
fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Stop,
    }
}

/// The `cache_control` marker to place on cache breakpoints, or `None` when
/// caching is disabled. `Long` retention requests the 1h TTL.
fn cache_control(options: &crate::StreamOptions) -> Option<Value> {
    match options.cache_retention.unwrap_or(CacheRetention::Short) {
        CacheRetention::Disabled => None,
        CacheRetention::Short => Some(serde_json::json!({ "type": "ephemeral" })),
        CacheRetention::Long => Some(serde_json::json!({ "type": "ephemeral", "ttl": "1h" })),
    }
}

fn build_request_body(
    model: &Model,
    context: &Context,
    options: &crate::StreamOptions,
) -> MessagesRequest {
    let cache_control = cache_control(options);
    let mut messages: Vec<Value> = Vec::new();
    for message in &context.messages {
        match message {
            Message::User(user) => {
                messages
                    .push(serde_json::json!({ "role": "user", "content": user.text_content() }));
            }
            Message::Assistant(assistant) => {
                let blocks: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) if !text.text.is_empty() => {
                            Some(serde_json::json!({ "type": "text", "text": text.text }))
                        }
                        AssistantContent::ToolCall(call) => Some(serde_json::json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        })),
                        // Thinking blocks are not replayed for now.
                        _ => None,
                    })
                    .collect();
                messages.push(serde_json::json!({ "role": "assistant", "content": blocks }));
            }
            Message::ToolResult(result) => {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }],
                }));
            }
        }
    }

    // Cache the conversation history: attach the breakpoint to the last
    // block of the last user-role message, converting string content to
    // blocks when needed.
    if let Some(control) = &cache_control
        && let Some(last) = messages.last_mut()
        && last["role"] == "user"
    {
        match &mut last["content"] {
            Value::String(text) => {
                let text = std::mem::take(text);
                last["content"] = serde_json::json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": control,
                }]);
            }
            Value::Array(blocks) => {
                if let Some(block) = blocks.last_mut() {
                    block["cache_control"] = control.clone();
                }
            }
            _ => {}
        }
    }

    let max_tokens = options
        .max_tokens
        .or(Some(model.max_tokens).filter(|&n| n > 0))
        .unwrap_or(DEFAULT_MAX_TOKENS);

    // System prompt goes out as a text block so it can carry a breakpoint.
    let system = context.system_prompt.as_ref().map(|text| {
        let mut block = serde_json::json!({ "type": "text", "text": text });
        if let Some(control) = &cache_control {
            block["cache_control"] = control.clone();
        }
        Value::Array(vec![block])
    });

    // Tools render first in the prompt; one breakpoint on the last tool
    // caches the whole definition list.
    let tool_count = context.tools.len();
    let tools = context
        .tools
        .iter()
        .enumerate()
        .map(|(index, tool)| WireTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.parameters.clone(),
            cache_control: cache_control.clone().filter(|_| index + 1 == tool_count),
        })
        .collect();

    MessagesRequest {
        model: model.id.clone(),
        max_tokens,
        system,
        messages,
        tools,
        stream: true,
        temperature: options.temperature,
    }
}

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Value>,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct WireTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<Value>,
}
