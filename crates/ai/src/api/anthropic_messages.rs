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
use crate::observer::ObservationPlan;
use crate::provider::{AnthropicCompat, AnthropicReasoningFormat};
use crate::types::{
    ApiKind, AssistantContent, CapabilitySupport, Context, Message, Model, ReasoningEffort,
    ReasoningOptions, StopReason, ThinkingContent, ToolChoice, ToolResultMessage, Usage,
    UserContent, UserMessage,
};

/// The Anthropic Messages wire protocol.
pub struct AnthropicMessages;

pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// The smallest reasoning budget Anthropic's `budget_tokens` shape documents,
/// and — because `max_tokens` caps the thinking *and* the answer — the room
/// banshu keeps for the answer when it derives a budget of its own.
const MIN_THINKING_BUDGET: u32 = 1024;

impl ProtocolAdapter for AnthropicMessages {
    fn kind(&self) -> ApiKind {
        ApiKind::AnthropicMessages
    }

    fn stream(&self, request: PreparedRequest) -> ProtocolEventStream {
        // Session affinity routes prompt-cache hits to the same replica; it
        // serves nothing when caching is disabled.
        let caching = request
            .options
            .cache_retention
            .unwrap_or(CacheRetention::Short)
            != CacheRetention::Disabled;
        let session_affinity = (request.anthropic_compat.send_session_affinity_headers && caching)
            .then(|| request.options.session_id.clone())
            .flatten();
        let mut protocol_headers = crate::ProviderHeaders::from([
            (
                "Content-Type".to_string(),
                Some("application/json".to_string()),
            ),
            (
                "Anthropic-Version".to_string(),
                Some(ANTHROPIC_VERSION.to_string()),
            ),
        ]);
        if let Some(session_id) = session_affinity {
            protocol_headers.insert("x-session-affinity".to_string(), Some(session_id));
        }
        let final_headers = request.headers_with_protocol_defaults(&protocol_headers);
        let observer = request.options.observer.clone();
        let PreparedRequest {
            provider,
            model,
            context,
            options,
            auth,
            http,
            anthropic_compat,
            ..
        } = request;
        // The payload snapshot the observer sees is the same value the wire
        // body serializes, so the two can never drift apart.
        let payload = serde_json::to_value(build_request_body(
            &model,
            &context,
            &options,
            anthropic_compat,
        ))
        .unwrap_or_default();
        let body = serde_json::to_vec(&payload).unwrap_or_default();
        let base_url = model.base_url.clone();
        let model_id = model.id.clone();
        let cost = model.cost.clone();
        let timeout = options.timeout;
        let max_retries = options.max_retries.unwrap_or(http::DEFAULT_MAX_RETRIES);
        let max_retry_delay = options.max_retry_delay;

        let stream = async_stream::stream! {
            let base = auth.base_url.as_deref().unwrap_or(&base_url);
            let url = format!("{}/v1/messages", base.trim_end_matches('/'));
            let observation = observer.map(|observer| {
                ObservationPlan::new(observer, provider, model_id, &url, &final_headers, payload)
            });
            let factory = move || {
                let mut builder =
                    super::apply_headers(http.post(&url).body(body.clone()), &final_headers);
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
            let mut raw_stop_reason = None;
            // `message_stop` is the only success signal; a bare EOF without it
            // is a dropped connection, not a completed response.
            let mut saw_message_stop = false;

            let mut exec = std::pin::pin!(executor::execute(factory, max_retries, max_retry_delay, observation));
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
                            raw_stop_reason = Some(reason.to_string());
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
            yield ProtocolEvent::stop(stop_reason, raw_stop_reason);
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

/// The output cap this request will ship: the caller's, else the model's, else
/// the crate default.
///
/// The [reasoning preflight](super::reasoning) measures a reasoning budget
/// against the value this returns, so it and [`build_request_body`] must read
/// the same ladder — a budget judged against a different cap could pass the
/// check and still be refused by the endpoint.
pub(crate) fn final_max_tokens(model: &Model, options: &crate::StreamOptions) -> u32 {
    options
        .max_tokens
        .or(Some(model.max_tokens).filter(|&n| n > 0))
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// The budget each effort level spends when the caller names none: 1024 for
/// `minimal`, 2048 for `low`, 8192 for `medium`, 16384 for `high`, and 32768 /
/// 65536 for the two levels above it.
///
/// The budget shape has no effort field — here the level *is* a token count —
/// so a request for `high` has to become a number somewhere. This ladder is
/// banshu's own, climbing from the minimum the shape documents; a caller who
/// wants an exact number sets
/// [`ReasoningOptions::token_budget`](crate::ReasoningOptions::token_budget),
/// which is then sent verbatim.
const fn derived_thinking_budget(effort: ReasoningEffort) -> u32 {
    match effort {
        // Unreachable: `Off` sends the disabling toggle and no budget at all.
        ReasoningEffort::Off => MIN_THINKING_BUDGET,
        ReasoningEffort::Minimal => MIN_THINKING_BUDGET,
        ReasoningEffort::Low => 2 * MIN_THINKING_BUDGET,
        ReasoningEffort::Medium => 8 * MIN_THINKING_BUDGET,
        ReasoningEffort::High => 16 * MIN_THINKING_BUDGET,
        ReasoningEffort::XHigh => 32 * MIN_THINKING_BUDGET,
        ReasoningEffort::Max => 64 * MIN_THINKING_BUDGET,
    }
}

/// The `budget_tokens` an enabled [`ThinkingBudget`](AnthropicReasoningFormat)
/// request puts on the wire, or the reason it cannot be expressed.
///
/// A budget the caller named is never trimmed to fit — that would answer a
/// question they did not ask — so one that does not fit is refused. A budget
/// *banshu* derived is trimmed, because the caller asked for a level, not for a
/// token count.
fn thinking_budget(reasoning: &ReasoningOptions, max_tokens: u32) -> Result<u32, String> {
    if let Some(tokens) = reasoning.token_budget {
        if tokens < MIN_THINKING_BUDGET {
            return Err(format!(
                "a reasoning budget of {tokens} is below the {MIN_THINKING_BUDGET} tokens \
                 this request format documents as its minimum",
            ));
        }
        if tokens >= max_tokens {
            return Err(format!(
                "a reasoning budget of {tokens} does not fit under this request's max_tokens \
                 of {max_tokens}, which caps the reasoning and the answer together",
            ));
        }
        return Ok(tokens);
    }
    let room = max_tokens.saturating_sub(MIN_THINKING_BUDGET);
    if room < MIN_THINKING_BUDGET {
        return Err(format!(
            "this request's max_tokens of {max_tokens} has no room for both the \
             {MIN_THINKING_BUDGET}-token minimum reasoning budget and an answer",
        ));
    }
    Ok(derived_thinking_budget(reasoning.effort).min(room))
}

/// The [reasoning preflight](super::reasoning) check this protocol adds: an
/// enabled budget-shape request has to name — or be able to derive — a budget
/// that fits under the `max_tokens` the request will ship with, on a model that
/// attests a budget may be spent at all.
///
/// Called only where the provider's declared shape carries a budget, so a
/// request reaching [`build_request_body`] always has one to send.
pub(crate) fn validate_thinking_budget(
    model: &Model,
    options: &crate::StreamOptions,
    reasoning: &ReasoningOptions,
) -> Result<(), String> {
    if reasoning.effort == ReasoningEffort::Off {
        // A disabled request sends the toggle alone, so a budget alongside it
        // could only be silently discarded.
        return match reasoning.token_budget {
            Some(tokens) => Err(format!(
                "a reasoning budget of {tokens} cannot be requested alongside effort `off`, \
                 which disables reasoning outright",
            )),
            None => Ok(()),
        };
    }
    // This shape enables reasoning *by* spending a budget, so a model that
    // attests none cannot make an enabled request at all — not even one that
    // leaves the number to banshu. The preflight's own budget check only sees
    // budgets the caller named.
    if model.reasoning.token_budget() != CapabilitySupport::Supported {
        return Err(format!(
            "model `{}` does not support a reasoning token budget, and the reasoning request \
             format declared by provider `{}` can only enable reasoning by spending one",
            model.id, model.provider,
        ));
    }
    thinking_budget(reasoning, final_max_tokens(model, options)).map(drop)
}

/// Map a reasoning request onto the `thinking` field `format` declares, or
/// `None` when the payload carries no reasoning at all.
///
/// The [reasoning preflight](super::reasoning) has already refused anything
/// this cannot express — an undeclared format, an effort the model does not
/// attest, a budget on a shape that carries none or one that does not fit — so
/// every request reaching here is one the endpoint accepts, and nothing is
/// clamped or dropped on the way out.
fn thinking_wire(
    format: AnthropicReasoningFormat,
    reasoning: Option<&ReasoningOptions>,
    max_tokens: u32,
) -> Option<ThinkingRequest> {
    // No request means no override: the payload is byte-identical to one built
    // before reasoning options existed.
    let reasoning = reasoning?;
    match format {
        // Unreachable in practice — the preflight refuses every reasoning
        // request against a provider declaring no shape. Sending nothing is the
        // honest fallback if that ever changes.
        AnthropicReasoningFormat::Unsupported => None,
        // Every declared shape spells "do not reason" the same way, and
        // omitting the field would leave a thinking model thinking.
        _ if reasoning.effort == ReasoningEffort::Off => Some(ThinkingRequest::disabled()),
        AnthropicReasoningFormat::ThinkingToggle => Some(ThinkingRequest::enabled()),
        AnthropicReasoningFormat::ThinkingAdaptive => Some(ThinkingRequest::adaptive()),
        AnthropicReasoningFormat::ThinkingBudget => {
            let budget = thinking_budget(reasoning, max_tokens);
            debug_assert!(
                budget.is_ok(),
                "the reasoning preflight should have refused this request: {budget:?}",
            );
            // A budget that cannot be computed is a preflight bug, not a wire
            // decision. Send the disabling toggle rather than omit the field:
            // an omission reads as the endpoint's own default, which is the one
            // answer nobody asked for.
            Some(budget.map_or_else(
                |_| ThinkingRequest::disabled(),
                ThinkingRequest::with_budget,
            ))
        }
    }
}

/// Map an Anthropic `stop_reason` to a banshu [`StopReason`].
fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn"
        | "stop_sequence"
        | "pause_turn"
        | "refusal"
        | "model_context_window_exceeded" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Unknown,
    }
}

/// The `cache_control` marker to place on cache breakpoints, or `None` when
/// caching is disabled. `Long` retention requests the 1h TTL.
///
/// The [cache-routing preflight](super::cache_routing) has already refused an
/// explicit `Long` against a provider that does not attest the one-hour TTL,
/// so a `Long` reaching here is always one the endpoint accepts.
fn cache_control(options: &crate::StreamOptions) -> Option<Value> {
    match options.cache_retention.unwrap_or(CacheRetention::Short) {
        CacheRetention::Disabled => None,
        CacheRetention::Short => Some(serde_json::json!({ "type": "ephemeral" })),
        CacheRetention::Long => Some(serde_json::json!({ "type": "ephemeral", "ttl": "1h" })),
    }
}

/// Text-only user messages keep the plain-string wire shape; an image turns
/// the message into content blocks, with each image as an `image` block
/// carrying a base64 `source`.
fn user_content_wire(user: &UserMessage) -> Value {
    if !user.has_image() {
        return Value::String(user.text_content());
    }
    Value::Array(user.content.iter().map(content_block_wire).collect())
}

/// One user-or-tool-result content block in Anthropic's shape.
fn content_block_wire(content: &UserContent) -> Value {
    match content {
        UserContent::Text(text) => serde_json::json!({ "type": "text", "text": text.text }),
        UserContent::Image(image) => serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime_type,
                "data": image.data,
            },
        }),
    }
}

/// Tool-result content on the wire. Text-only results keep the plain-string
/// shape; with images the content becomes ordered blocks (each image carrying
/// a base64 `source`), prepending a placeholder text block when the result has
/// no text of its own. A result only still holds images if the model accepts
/// them — the normalizer replaced them with placeholder text otherwise.
fn tool_result_content_wire(result: &ToolResultMessage) -> Value {
    if !result.has_image() {
        return Value::String(result.text_content());
    }
    let mut blocks: Vec<Value> = result.content.iter().map(content_block_wire).collect();
    if result.text_content().is_empty() {
        blocks.insert(
            0,
            serde_json::json!({ "type": "text", "text": "(see attached image)" }),
        );
    }
    Value::Array(blocks)
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
                messages.push(
                    serde_json::json!({ "role": "user", "content": user_content_wire(user) }),
                );
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
                        "content": tool_result_content_wire(result),
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

    let max_tokens = final_max_tokens(model, options);

    // System prompt goes out as a text block so it can carry a breakpoint.
    let system = context.system_prompt.as_ref().map(|text| {
        let mut block = serde_json::json!({ "type": "text", "text": text });
        if let Some(control) = &cache_control {
            block["cache_control"] = control.clone();
        }
        Value::Array(vec![block])
    });

    // Tools render first in the prompt; one breakpoint on the last tool
    // caches the whole definition list. The breakpoint is attached only when
    // the provider declares tool-definition cache control — suppressing it
    // leaves the system and message breakpoints above untouched.
    let tool_count = context.tools.len();
    let tools = context
        .tools
        .iter()
        .enumerate()
        .map(|(index, tool)| WireTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.parameters.clone(),
            strict: (tool.strict && compat.strict_tool_schemas).then_some(true),
            cache_control: cache_control
                .clone()
                .filter(|_| compat.tool_cache_control && index + 1 == tool_count),
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
        thinking: thinking_wire(
            compat.reasoning_format,
            options.reasoning.as_ref(),
            max_tokens,
        ),
        tool_choice: tool_choice_wire(options.tool_choice.as_ref()),
    }
}

/// Map a tool choice onto the `tool_choice` wire field, or `None` when the
/// payload carries no choice at all.
///
/// The [tool-choice preflight](super::tool_choice) has already refused any
/// choice the provider cannot express, so every choice reaching here is one
/// the endpoint accepts — the name of a [`ToolChoice::Named`] goes out exactly
/// as the caller gave it.
fn tool_choice_wire(choice: Option<&ToolChoice>) -> Option<Value> {
    let choice = choice?;
    Some(match choice {
        ToolChoice::Auto => serde_json::json!({ "type": "auto" }),
        ToolChoice::None => serde_json::json!({ "type": "none" }),
        // Anthropic spells "at least one tool, any tool" as `any`.
        ToolChoice::Required => serde_json::json!({ "type": "any" }),
        ToolChoice::Named(name) => serde_json::json!({ "type": "tool", "name": name }),
    })
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
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

/// The `thinking` request object, the whole of banshu's Anthropic-compatible
/// reasoning request surface. `budget_tokens` rides along only on the shape
/// that declares it.
#[derive(Serialize)]
struct ThinkingRequest {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u32>,
}

impl ThinkingRequest {
    /// The disabling value every declared shape documents.
    fn disabled() -> Self {
        Self {
            kind: "disabled",
            budget_tokens: None,
        }
    }

    /// The bare toggle, for a shape that says "reason" with nothing else.
    fn enabled() -> Self {
        Self {
            kind: "enabled",
            budget_tokens: None,
        }
    }

    /// The adaptive shape, which hands the model the decision.
    fn adaptive() -> Self {
        Self {
            kind: "adaptive",
            budget_tokens: None,
        }
    }

    /// The budget shape, which enables reasoning by spending tokens on it.
    fn with_budget(budget_tokens: u32) -> Self {
        Self {
            kind: "enabled",
            budget_tokens: Some(budget_tokens),
        }
    }
}

#[derive(Serialize)]
struct WireTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    /// Sent only when the tool is marked strict *and* the provider declares
    /// strict tool schemas — never `false`, which is the default anyway.
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<Value>,
}
