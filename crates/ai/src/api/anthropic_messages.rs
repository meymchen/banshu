//! The Anthropic `/v1/messages` streaming protocol.
//!
//! Used by banshu's Anthropic-compatible providers (Z.AI, MiniMax, Kimi, …).
//! Speaks the Messages SSE event stream: `message_start` carries input usage,
//! `content_block_delta` carries text/thinking/tool fragments, `message_delta`
//! carries the stop reason and output usage, `message_stop` ends the turn.
//! Wire events are translated into [`ProtocolEvent`]s; assembly into public
//! events and the final message is the driver's job (see [`crate::api`]).

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::Value;

use super::protocol_event::ProtocolEvent;
use super::{PreparedRequest, ProtocolAdapter, ProtocolEventStream, compute_cost};
use crate::CacheRetention;
use crate::executor::{self, ExecutorEvent};
use crate::http;
use crate::provider::AnthropicCompat;
use crate::types::{
    ApiKind, AssistantContent, Context, Message, Model, StopReason, ThinkingContent, Usage,
};

/// The Anthropic Messages wire protocol.
pub struct AnthropicMessages;

pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

impl ProtocolAdapter for AnthropicMessages {
    fn kind(&self) -> ApiKind {
        ApiKind::AnthropicMessages
    }

    fn stream(&self, request: PreparedRequest) -> ProtocolEventStream {
        let PreparedRequest {
            model,
            context,
            options,
            auth,
            headers,
            http,
            anthropic_compat,
            ..
        } = request;
        let body = build_request_body(&model, &context, &options, anthropic_compat);
        let base_url = model.base_url.clone();
        let cost = model.cost.clone();
        let timeout = options.timeout;
        let max_retries = options.max_retries.unwrap_or(http::DEFAULT_MAX_RETRIES);
        let max_retry_delay = options.max_retry_delay;
        // Session affinity routes prompt-cache hits to the same replica; it
        // serves nothing when caching is disabled.
        let caching =
            options.cache_retention.unwrap_or(CacheRetention::Short) != CacheRetention::Disabled;
        let session_affinity = (anthropic_compat.send_session_affinity_headers && caching)
            .then_some(options.session_id.clone())
            .flatten();

        let stream = async_stream::stream! {
            let base = auth.base_url.as_deref().unwrap_or(&base_url);
            let url = format!("{}/v1/messages", base.trim_end_matches('/'));
            let api_key = auth.api_key;
            let auth_headers = auth.headers;

            let factory = move || {
                let mut builder = super::apply_headers(
                    http.post(&url)
                        .header("anthropic-version", ANTHROPIC_VERSION)
                        .json(&body),
                    &headers,
                );
                if let Some(api_key) = &api_key {
                    builder = builder.header("x-api-key", api_key);
                }
                builder = super::apply_headers(builder, &auth_headers);
                if let Some(session_id) = &session_affinity {
                    builder = builder.header("x-session-affinity", session_id);
                }
                if let Some(timeout) = timeout {
                    builder = builder.timeout(timeout);
                }
                builder
            };

            // Wire block kinds, keyed by the Anthropic `index` (reused as the
            // assembler `block_id`). Tracks which blocks still need a synthetic
            // end event at finalization — Anthropic omits `content_block_stop`
            // in some streams (e.g. after `redacted_thinking`).
            let mut blocks: Vec<Option<WireBlock>> = Vec::new();
            let mut usage = Usage::default();
            let mut stop_reason = StopReason::Stop;
            // `message_stop` is the only success signal; a bare EOF without it
            // is a dropped connection, not a completed response.
            let mut saw_message_stop = false;

            let mut exec = std::pin::pin!(executor::execute(factory, max_retries, max_retry_delay));
            'outer: while let Some(exec_event) = exec.next().await {
                let data = match exec_event {
                    ExecutorEvent::Retry { attempt, max_attempts, delay, kind } => {
                        yield ProtocolEvent::Retry { attempt, max_attempts, delay, kind };
                        continue;
                    }
                    ExecutorEvent::Established { request_id } => {
                        yield ProtocolEvent::ResponseMetadata { response_id: request_id, response_model: None };
                        continue;
                    }
                    ExecutorEvent::Eof => break 'outer,
                    ExecutorEvent::Failed { kind, message, diagnostics } => {
                        yield ProtocolEvent::Failure { kind, message, diagnostics };
                        return;
                    }
                    ExecutorEvent::Event(sse_event) => sse_event,
                };
                let event_field = data.event;
                let value = match super::parse_sse_json(data.data) {
                    Ok(value) => value,
                    Err((message, diagnostic)) => {
                        yield ProtocolEvent::Failure {
                            kind: crate::ErrorKind::Protocol,
                            message,
                            diagnostics: vec![diagnostic],
                        };
                        return;
                    }
                };
                if event_field.as_deref() == Some("error")
                    || value.get("type").and_then(Value::as_str) == Some("error")
                {
                    let message = http::json_error_summary(&value)
                        .unwrap_or_else(|| "provider returned an error".to_string());
                    yield ProtocolEvent::Failure {
                        kind: crate::ErrorKind::Api,
                        message,
                        diagnostics: Vec::new(),
                    };
                    return;
                }
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
                        let block_id = index as u64;
                        let block = &value["content_block"];
                        let (kind, start) = match block["type"].as_str() {
                            Some("thinking") => (WireBlockKind::Thinking, ProtocolEvent::ThinkingStart {
                                block_id,
                                signature: None,
                                redacted: false,
                            }),
                            // Redacted thinking arrives whole: an opaque
                            // payload carried in the signature slot.
                            Some("redacted_thinking") => (WireBlockKind::Thinking, ProtocolEvent::ThinkingStart {
                                block_id,
                                signature: block["data"].as_str().map(str::to_string),
                                redacted: true,
                            }),
                            Some("tool_use") => (WireBlockKind::ToolCall, ProtocolEvent::ToolCallStart {
                                block_id,
                                id: block["id"].as_str().unwrap_or_default().to_string(),
                                name: block["name"].as_str().unwrap_or_default().to_string(),
                            }),
                            _ => (WireBlockKind::Text, ProtocolEvent::TextStart { block_id, signature: None }),
                        };
                        if blocks.len() <= index {
                            blocks.resize_with(index + 1, || None);
                        }
                        blocks[index] = Some(WireBlock { kind, ended: false });
                        yield start;
                    }
                    Some("content_block_delta") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        let block_id = index as u64;
                        let delta = &value["delta"];
                        let event = match delta["type"].as_str() {
                            Some("text_delta") => delta["text"].as_str()
                                .filter(|chunk| !chunk.is_empty())
                                .map(|chunk| ProtocolEvent::TextDelta { block_id, delta: chunk.to_string() }),
                            Some("thinking_delta") => delta["thinking"].as_str()
                                .filter(|chunk| !chunk.is_empty())
                                .map(|chunk| ProtocolEvent::ThinkingDelta { block_id, delta: chunk.to_string() }),
                            Some("signature_delta") => delta["signature"].as_str()
                                .map(|sig| ProtocolEvent::ThinkingSignature { block_id, signature: sig.to_string() }),
                            Some("input_json_delta") => delta["partial_json"].as_str()
                                .map(|fragment| ProtocolEvent::ToolCallDelta { block_id, delta: fragment.to_string() }),
                            _ => None,
                        };
                        if let Some(event) = event {
                            yield event;
                        }
                    }
                    Some("content_block_stop") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        let block_id = index as u64;
                        if let Some(Some(block)) = blocks.get_mut(index)
                            && !block.ended
                        {
                            block.ended = true;
                            yield block.kind.end_event(block_id);
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
                    Some("message_stop") => {
                        saw_message_stop = true;
                        break 'outer;
                    }
                    _ => {}
                }
            }

            if !saw_message_stop {
                yield ProtocolEvent::Failure {
                    kind: crate::ErrorKind::StreamInterrupted,
                    message: "connection closed before message_stop".to_string(),
                    diagnostics: Vec::new(),
                };
                return;
            }

            // End any blocks the stream left open (Anthropic may omit
            // `content_block_stop`). A tool call's arguments are parsed on end,
            // so this is required for correctness, not just tidiness.
            for (index, slot) in blocks.iter_mut().enumerate() {
                if let Some(block) = slot
                    && !block.ended
                {
                    block.ended = true;
                    yield block.kind.end_event(index as u64);
                }
            }

            // Anthropic reports no total; derive it from all token classes.
            usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
            usage.cost = compute_cost(&usage, &cost);
            yield ProtocolEvent::Usage(usage);
            yield ProtocolEvent::Stop(stop_reason);
        };

        Box::pin(stream)
    }
}

/// The kind of a streamed content block, tracked per wire `index` so a
/// `content_block_stop` (or a synthetic end at finalization) can emit the
/// matching `*End` protocol event.
#[derive(Clone, Copy)]
enum WireBlockKind {
    Text,
    Thinking,
    ToolCall,
}

impl WireBlockKind {
    fn end_event(self, block_id: u64) -> ProtocolEvent {
        match self {
            WireBlockKind::Text => ProtocolEvent::TextEnd { block_id },
            WireBlockKind::Thinking => ProtocolEvent::ThinkingEnd { block_id },
            WireBlockKind::ToolCall => ProtocolEvent::ToolCallEnd { block_id },
        }
    }
}

/// A streamed content block, keyed by its Anthropic wire `index`.
struct WireBlock {
    kind: WireBlockKind,
    ended: bool,
}

/// Serialize a thinking block for history replay. Redacted payloads go back
/// verbatim as `redacted_thinking`; signed thinking keeps its signature;
/// signatureless thinking (e.g. from an aborted stream) is downgraded to a
/// text block unless the provider accepts empty signatures.
fn replay_thinking(block: &ThinkingContent, compat: AnthropicCompat) -> Option<Value> {
    if block.redacted {
        return Some(serde_json::json!({
            "type": "redacted_thinking",
            "data": block.signature.clone().unwrap_or_default(),
        }));
    }
    let signature = block.signature.as_deref().unwrap_or("");
    let has_signature = !signature.trim().is_empty();
    if block.thinking.trim().is_empty() && !has_signature {
        return None;
    }
    if has_signature || compat.allow_empty_signature {
        Some(serde_json::json!({
            "type": "thinking",
            "thinking": block.thinking,
            "signature": if has_signature { signature } else { "" },
        }))
    } else {
        Some(serde_json::json!({ "type": "text", "text": block.thinking }))
    }
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
    compat: AnthropicCompat,
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
                        AssistantContent::Thinking(block) => replay_thinking(block, compat),
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
                        "content": result.text_content(),
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
