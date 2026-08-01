//! The one Context normalization pass, run before either protocol adapter
//! builds its wire payload.
//!
//! Cross-model rules live here and nowhere else: an adapter receives an
//! already-normalized [`Context`] copy and only translates it to its own wire
//! shape. The caller's `Context` is never touched, so the same value can be
//! streamed against one model after another.
//!
//! Rules owned by this pass today:
//!
//! - **Modality gate** — an image in the newest user message on a model
//!   without [`Modality::Image`] fails the request outright (the caller is
//!   asking the model to look at something it cannot see, so silently
//!   dropping it would answer the wrong question).
//! - **Image downgrade** — every *other* image, historical user turns and tool
//!   results alike, is replaced in place with fixed placeholder text, with a
//!   consecutive run collapsing into a single placeholder. Order is preserved
//!   and no message is dropped.
//! - **Reasoning downgrade** (issue #40) — provider-private reasoning state
//!   (thinking blocks with their opaque signatures, redacted payloads, text
//!   signatures) replays verbatim only onto the exact provider, API, and
//!   model id that produced it. Replayed anywhere else, non-empty ordinary
//!   thinking becomes a plain text block, empty or redacted thinking is
//!   omitted, and every signature is dropped.
//! - **Tool-call id rewrite** (issue #40) — any tool-call or tool-result id
//!   that does not match `^[a-zA-Z0-9_-]{1,64}$` is rewritten
//!   deterministically; the rewrite is a pure function of the original id, so
//!   a tool result always tracks its call.
//! - **Tool-history repair** (issue #41) — a historical tool call whose result
//!   was never recorded receives exactly one synthetic error result (`No result
//!   provided`), placed right after the turn that issued the call, and an
//!   assistant turn that ended in `Error` or `Aborted` is dropped together with
//!   any results answering its calls. Replay always forms a legal request.

use std::collections::HashSet;

use crate::types::{
    AssistantContent, Context, Diagnostic, DiagnosticCode, Message, Modality, Model, StopReason,
    TextContent, ToolResultMessage, UserContent, UserMessage,
};

/// The fixed text replacing a user image the target model cannot see.
const USER_IMAGE_OMITTED_PLACEHOLDER: &str = "(image omitted: model does not support images)";

/// The fixed text replacing a tool-result image the target model cannot see
/// (issue #22) — distinct from the user placeholder so the model can tell whose
/// image went missing.
const TOOL_IMAGE_OMITTED_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

/// The body of the synthetic error result standing in for a tool call whose
/// real result was never recorded (issue #41).
const NO_RESULT_PROVIDED: &str = "No result provided";

/// A normalized [`Context`] copy and the diagnostics its rules produced.
#[derive(Debug)]
pub(crate) struct Normalized {
    /// The copy adapters build their wire payload from.
    pub(crate) context: Context,
    /// What was changed, for the resulting assistant message.
    pub(crate) diagnostics: Vec<Diagnostic>,
}

/// Normalize `context` for `model`, leaving the caller's value untouched.
///
/// `Err` carries the detail of a modality violation — the request must
/// terminate in-band with [`ErrorKind::InvalidRequest`](crate::ErrorKind) before any HTTP
/// request is issued.
pub(crate) fn normalize(model: &Model, context: &Context) -> Result<Normalized, String> {
    let accepts_images = model.input.contains(&Modality::Image);
    if !accepts_images && newest_user_message(context).is_some_and(|user| user.has_image()) {
        return Err(format!("model `{}` does not accept image input", model.id));
    }

    let mut context = context.clone();
    repair_tool_history(&mut context);

    let mut diagnostics = Vec::new();
    normalize_reasoning(model, &mut context, &mut diagnostics);
    normalize_tool_call_ids(&mut context, &mut diagnostics);

    // Past the gate, the newest user message is known image-free, so every
    // remaining user image is by definition historical.
    if !accepts_images {
        let mut user_images = 0;
        let mut tool_images = 0;
        for message in &mut context.messages {
            match message {
                Message::User(user) => {
                    user_images += omit_images(&mut user.content, USER_IMAGE_OMITTED_PLACEHOLDER);
                }
                Message::ToolResult(result) => {
                    tool_images += omit_images(&mut result.content, TOOL_IMAGE_OMITTED_PLACEHOLDER);
                }
                Message::Assistant(_) => {}
            }
        }
        diagnostics.extend(
            [(user_images, "user"), (tool_images, "tool-result")]
                .into_iter()
                .filter(|(count, _)| *count > 0)
                .map(|(count, kind)| {
                    Diagnostic::new(
                        DiagnosticCode::ImageDowngraded,
                        format!(
                            "{count} {kind} image(s) omitted: model `{}` does not support images",
                            model.id
                        ),
                    )
                }),
        );
    }

    Ok(Normalized {
        context,
        diagnostics,
    })
}

/// Repair incomplete historical tool turns (issue #41): every historical tool
/// call whose result was never recorded receives exactly one synthetic error
/// result, placed right after the turn that issued it so the following run of
/// tool results stays consecutive; an assistant turn that ended in `Error` or
/// `Aborted` is dropped, and any results answering its calls go with it so no
/// result is left pointing at a call that no longer exists. A trailing
/// assistant turn is left alone — its calls may still be mid-execution, so
/// they are not yet historical.
fn repair_tool_history(context: &mut Context) {
    let answered: HashSet<String> = context
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => Some(result.tool_call_id.clone()),
            _ => None,
        })
        .collect();
    let dropped_calls: HashSet<String> = context
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant)
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) =>
            {
                Some(
                    assistant
                        .content
                        .iter()
                        .filter_map(|content| match content {
                            AssistantContent::ToolCall(call) => Some(call.id.clone()),
                            _ => None,
                        }),
                )
            }
            _ => None,
        })
        .flatten()
        .collect();

    // Filter first, then repair: the trailing-turn exemption must be judged
    // against what survives, because dropping a final Error/Aborted turn can
    // make an earlier assistant turn the new trailing one.
    let mut messages = Vec::with_capacity(context.messages.len());
    for message in context.messages.drain(..) {
        match message {
            Message::Assistant(assistant)
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) => {}
            Message::ToolResult(result) if dropped_calls.contains(&result.tool_call_id) => {}
            other => messages.push(other),
        }
    }

    let trailing = messages.len().saturating_sub(1);
    let mut repaired = Vec::with_capacity(messages.len());
    for (index, message) in messages.into_iter().enumerate() {
        match message {
            Message::Assistant(assistant) if index != trailing => {
                let synthetic: Vec<Message> = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::ToolCall(call) if !answered.contains(&call.id) => {
                            Some(Message::ToolResult(ToolResultMessage::error_text(
                                call.id.clone(),
                                call.name.clone(),
                                NO_RESULT_PROVIDED,
                            )))
                        }
                        _ => None,
                    })
                    .collect();
                repaired.push(Message::Assistant(assistant));
                repaired.extend(synthetic);
            }
            other => repaired.push(other),
        }
    }
    context.messages = repaired;
}

/// Reasoning downgrade: reasoning state is private to the exact provider,
/// API, and model id that produced it. A same-provenance assistant message
/// keeps its signatures verbatim; any other has its non-empty thinking
/// converted to plain text, its empty or redacted thinking omitted, and every
/// text signature dropped.
fn normalize_reasoning(model: &Model, context: &mut Context, diagnostics: &mut Vec<Diagnostic>) {
    let target_api = super::api_name(model.api);
    let mut converted = 0;
    let mut omitted = 0;
    let mut stripped = 0;
    for message in &mut context.messages {
        let Message::Assistant(assistant) = message else {
            continue;
        };
        let producing_model = assistant
            .response_model
            .as_deref()
            .unwrap_or(&assistant.model);
        if assistant.provider == model.provider
            && assistant.api == target_api
            && producing_model == model.id
        {
            continue;
        }
        let mut content = Vec::with_capacity(assistant.content.len());
        for block in assistant.content.drain(..) {
            match block {
                AssistantContent::Thinking(thinking)
                    if thinking.redacted || thinking.thinking.trim().is_empty() =>
                {
                    omitted += 1;
                }
                AssistantContent::Thinking(thinking) => {
                    converted += 1;
                    content.push(AssistantContent::Text(TextContent {
                        text: thinking.thinking,
                        signature: None,
                    }));
                }
                AssistantContent::Text(mut text) => {
                    if text.signature.take().is_some() {
                        stripped += 1;
                    }
                    content.push(AssistantContent::Text(text));
                }
                other => content.push(other),
            }
        }
        assistant.content = content;
    }
    if converted + omitted + stripped > 0 {
        let mut parts = Vec::new();
        if converted > 0 {
            parts.push(format!("{converted} thinking block(s) became text"));
        }
        if omitted > 0 {
            parts.push(format!(
                "{omitted} empty/redacted thinking block(s) omitted"
            ));
        }
        if stripped > 0 {
            parts.push(format!("{stripped} text signature(s) dropped"));
        }
        diagnostics.push(Diagnostic::new(
            DiagnosticCode::ReasoningDowngraded,
            format!(
                "{}: reasoning state is private to the producing provider/API/model (target `{}`)",
                parts.join(", "),
                model.id
            ),
        ));
    }
}

/// Tool-call id rewrite: ids must match `^[a-zA-Z0-9_-]{1,64}$` to be safe on
/// any protocol. Anything else is rewritten deterministically; the rewrite is
/// a pure function of the original id, so a tool result always tracks its
/// call without a lookup table.
fn normalize_tool_call_ids(context: &mut Context, diagnostics: &mut Vec<Diagnostic>) {
    let mut rewritten = 0;
    for message in &mut context.messages {
        match message {
            Message::Assistant(assistant) => {
                for block in &mut assistant.content {
                    if let AssistantContent::ToolCall(call) = block
                        && let Some(id) = rewrite_tool_call_id(&call.id)
                    {
                        call.id = id;
                        rewritten += 1;
                    }
                }
            }
            Message::ToolResult(result) => {
                if let Some(id) = rewrite_tool_call_id(&result.tool_call_id) {
                    result.tool_call_id = id;
                    rewritten += 1;
                }
            }
            Message::User(_) => {}
        }
    }
    if rewritten > 0 {
        diagnostics.push(Diagnostic::new(
            DiagnosticCode::ToolCallIdRewritten,
            format!("{rewritten} tool-call/tool-result id(s) rewritten to match `^[a-zA-Z0-9_-]{{1,64}}$`"),
        ));
    }
}

/// The id shape every supported protocol accepts (Anthropic's documented
/// tool-use id pattern, which the OpenAI side also satisfies).
const MAX_TOOL_CALL_ID_LEN: usize = 64;

/// How much of the original id survives into a rewrite — the limit minus the
/// `_` separator and the 16 hex characters of the hash suffix.
const REWRITE_PREFIX_LEN: usize = MAX_TOOL_CALL_ID_LEN - 1 - 16;

fn is_valid_tool_call_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_TOOL_CALL_ID_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Rewrite an invalid id into a deterministic valid one: the original with
/// invalid characters replaced by `_`, truncated, then suffixed with a stable
/// hash of the original so distinct invalid ids stay distinct. Returns `None`
/// when the id is already valid.
fn rewrite_tool_call_id(id: &str) -> Option<String> {
    if is_valid_tool_call_id(id) {
        return None;
    }
    let sanitized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(REWRITE_PREFIX_LEN)
        .collect();
    Some(format!("{sanitized}_{:016x}", fnv1a64(id.as_bytes())))
}

/// FNV-1a 64-bit — a small, stable hash with no dependency, so rewritten ids
/// are identical across processes and crate versions.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(PRIME)
    })
}

/// The last user turn — the one the model is being asked to answer.
fn newest_user_message(context: &Context) -> Option<&UserMessage> {
    context
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::User(user) => Some(user),
            _ => None,
        })
}

/// Replace each image block with `placeholder` text in place, collapsing a run
/// of images (and an image next to an identical placeholder) into one. Returns
/// how many image blocks were replaced.
fn omit_images(content: &mut Vec<UserContent>, placeholder: &str) -> usize {
    if !content
        .iter()
        .any(|block| matches!(block, UserContent::Image(_)))
    {
        return 0;
    }

    let mut omitted = 0;
    let mut previous_was_placeholder = false;
    let mut normalized = Vec::with_capacity(content.len());
    for block in content.drain(..) {
        match block {
            UserContent::Image(_) => {
                omitted += 1;
                if !previous_was_placeholder {
                    normalized.push(UserContent::Text(TextContent {
                        text: placeholder.to_string(),
                        signature: None,
                    }));
                }
                previous_was_placeholder = true;
            }
            UserContent::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                normalized.push(UserContent::Text(text));
            }
        }
    }
    *content = normalized;
    omitted
}

#[cfg(test)]
mod tests {
    //! The normalizer is a `pub(crate)` internal with no public seam an
    //! integration test can reach directly — as with `crate::sse`, its unit
    //! tests live inline. End-to-end coverage of the resulting wire payloads
    //! lives in `tests/context_normalization.rs`.

    use super::*;
    use crate::types::{
        AssistantContent, AssistantMessage, ImageContent, Message, StopReason, ThinkingContent,
        ToolCall, ToolResultMessage, UserMessage,
    };

    fn text(text: &str) -> UserContent {
        UserContent::Text(TextContent {
            text: text.into(),
            signature: None,
        })
    }

    fn image() -> UserContent {
        UserContent::Image(ImageContent {
            data: "AAAA".into(),
            mime_type: "image/png".into(),
        })
    }

    fn user(content: Vec<UserContent>) -> Message {
        Message::User(UserMessage {
            content,
            timestamp: 0,
        })
    }

    fn tool_result(content: Vec<UserContent>) -> Message {
        Message::ToolResult(ToolResultMessage::content("call_1", "shot", content, false))
    }

    fn text_only_model() -> Model {
        Model::openai_completions("text-model")
    }

    fn tool_call(id: &str, name: &str) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::json!({}),
            raw_arguments: None,
        })
    }

    fn assistant(content: Vec<AssistantContent>, stop_reason: StopReason) -> Message {
        let mut message = AssistantMessage::from_content(content);
        message.stop_reason = stop_reason;
        Message::Assistant(Box::new(message))
    }

    fn assistant_text(text: &str) -> AssistantContent {
        AssistantContent::Text(TextContent {
            text: text.into(),
            signature: None,
        })
    }

    fn image_model() -> Model {
        let mut model = Model::openai_completions("image-model");
        model.input = vec![Modality::Text, Modality::Image];
        model
    }

    fn blocks(message: &Message) -> &[UserContent] {
        match message {
            Message::User(user) => &user.content,
            Message::ToolResult(result) => &result.content,
            Message::Assistant(_) => &[],
        }
    }

    #[test]
    fn a_newest_user_image_is_a_modality_violation() {
        let context = Context::new().with_message(user(vec![text("look"), image()]));
        let detail = normalize(&text_only_model(), &context).expect_err("the gate must fire");
        assert!(detail.contains("does not accept image input"), "{detail}");
    }

    #[test]
    fn an_image_model_takes_the_context_as_is() {
        let context = Context::new()
            .with_message(user(vec![text("look"), image()]))
            .with_message(tool_result(vec![image()]));
        let normalized = normalize(&image_model(), &context).expect("no gate on an image model");
        assert_eq!(normalized.context, context);
        assert!(normalized.diagnostics.is_empty());
    }

    #[test]
    fn historical_images_become_placeholders_and_the_caller_keeps_its_context() {
        let context = Context::new()
            .with_message(user(vec![text("look"), image()]))
            .with_message(tool_result(vec![image(), text("done")]))
            .user("and now?");
        let before = context.clone();

        let normalized = normalize(&text_only_model(), &context).expect("the newest turn is text");

        assert_eq!(
            blocks(&normalized.context.messages[0]),
            [text("look"), text(USER_IMAGE_OMITTED_PLACEHOLDER)]
        );
        assert_eq!(
            blocks(&normalized.context.messages[1]),
            [text(TOOL_IMAGE_OMITTED_PLACEHOLDER), text("done")]
        );
        assert_eq!(normalized.diagnostics.len(), 2, "one per image kind");
        assert!(
            normalized
                .diagnostics
                .iter()
                .all(|d| d.code == DiagnosticCode::ImageDowngraded)
        );
        assert_eq!(context, before);
    }

    #[test]
    fn a_run_of_images_collapses_into_one_placeholder_but_is_counted_in_full() {
        let context = Context::new()
            .with_message(user(vec![image(), image(), text("x"), image()]))
            .user("and now?");
        let normalized = normalize(&text_only_model(), &context).expect("the newest turn is text");

        assert_eq!(
            blocks(&normalized.context.messages[0]),
            [
                text(USER_IMAGE_OMITTED_PLACEHOLDER),
                text("x"),
                text(USER_IMAGE_OMITTED_PLACEHOLDER),
            ]
        );
        assert!(
            normalized.diagnostics[0].message.starts_with("3 user"),
            "the count reports images, not placeholders: {}",
            normalized.diagnostics[0].message
        );
    }

    #[test]
    fn normalizing_an_already_normalized_context_changes_nothing() {
        let context = Context::new()
            .with_message(user(vec![image(), image()]))
            .user("and now?");
        let once = normalize(&text_only_model(), &context).expect("the newest turn is text");
        let twice = normalize(&text_only_model(), &once.context).expect("still text");
        assert_eq!(twice.context, once.context);
        assert!(twice.diagnostics.is_empty(), "nothing left to downgrade");
    }

    #[test]
    fn a_text_only_context_is_left_alone() {
        let context = Context::new().user("hi");
        let normalized = normalize(&text_only_model(), &context).expect("no images at all");
        assert_eq!(normalized.context, context);
        assert!(normalized.diagnostics.is_empty());
    }

    #[test]
    fn an_orphaned_tool_call_gets_exactly_one_synthetic_error_result() {
        let context = Context::new()
            .user("weather in Paris?")
            .with_message(assistant(
                vec![
                    tool_call("call_1", "get_weather"),
                    tool_call("call_2", "get_time"),
                ],
                StopReason::ToolUse,
            ))
            .with_message(Message::ToolResult(ToolResultMessage::text(
                "call_1",
                "get_weather",
                "72F",
            )))
            .user("and tomorrow?");
        let before = context.clone();

        let normalized = normalize(&image_model(), &context).expect("repair never fails");

        let messages = &normalized.context.messages;
        assert_eq!(messages.len(), 5, "one synthetic result was inserted");
        let Message::ToolResult(synthetic) = &messages[2] else {
            panic!("the synthetic result follows the assistant turn: {messages:?}")
        };
        assert_eq!(synthetic.tool_call_id, "call_2");
        assert_eq!(synthetic.tool_name, "get_time");
        assert!(synthetic.is_error);
        assert_eq!(synthetic.content, [text("No result provided")]);
        let Message::ToolResult(existing) = &messages[3] else {
            panic!("the existing result keeps its place: {messages:?}")
        };
        assert_eq!(existing.tool_call_id, "call_1");
        assert!(!existing.is_error);
        assert_eq!(existing.content, [text("72F")]);
        assert_eq!(context, before, "the caller's context is untouched");
    }

    #[test]
    fn failed_and_aborted_turns_are_dropped_with_their_results() {
        let context = Context::new()
            .user("hi")
            .with_message(assistant(
                vec![tool_call("call_1", "get_weather")],
                StopReason::ToolUse,
            ))
            .with_message(Message::ToolResult(ToolResultMessage::text(
                "call_1",
                "get_weather",
                "72F",
            )))
            .with_message(assistant(
                vec![assistant_text("partial"), tool_call("call_2", "get_time")],
                StopReason::Error,
            ))
            .with_message(Message::ToolResult(ToolResultMessage::text(
                "call_2", "get_time", "noon",
            )))
            .with_message(assistant(
                vec![assistant_text("cut off")],
                StopReason::Aborted,
            ))
            .user("again");

        let normalized = normalize(&image_model(), &context).expect("repair never fails");

        let messages = &normalized.context.messages;
        assert_eq!(
            messages.len(),
            4,
            "failed/aborted turns and the results answering them are gone: {messages:?}"
        );
        assert!(
            messages.iter().all(|message| match message {
                Message::Assistant(assistant) => assistant.stop_reason == StopReason::ToolUse,
                Message::ToolResult(result) => result.tool_call_id == "call_1",
                Message::User(_) => true,
            }),
            "no trace of the dropped turns remains: {messages:?}"
        );
        assert!(
            matches!(
                messages.as_slice(),
                [
                    Message::User(_),
                    Message::Assistant(_),
                    Message::ToolResult(_),
                    Message::User(_)
                ]
            ),
            "only the healthy turns remain, in order: {messages:?}"
        );
    }

    #[test]
    fn a_trailing_assistant_turn_is_not_yet_history() {
        let context = Context::new()
            .user("weather in Paris?")
            .with_message(assistant(
                vec![tool_call("call_1", "get_weather")],
                StopReason::ToolUse,
            ));

        let normalized = normalize(&image_model(), &context).expect("repair never fails");

        assert_eq!(
            normalized.context, context,
            "a trailing turn may be mid-execution: no synthetic result is invented for it"
        );
    }

    #[test]
    fn a_dropped_trailing_failure_leaves_the_new_trailing_turn_alone() {
        let context = Context::new()
            .user("weather in Paris?")
            .with_message(assistant(
                vec![tool_call("call_1", "get_weather")],
                StopReason::ToolUse,
            ))
            .with_message(assistant(
                vec![assistant_text("something broke")],
                StopReason::Error,
            ));

        let normalized = normalize(&image_model(), &context).expect("repair never fails");

        let messages = &normalized.context.messages;
        assert!(
            matches!(
                messages.as_slice(),
                [Message::User(_), Message::Assistant(_)]
            ),
            "the error turn is dropped and the now-trailing assistant turn keeps its unanswered call: {messages:?}"
        );
    }

    #[test]
    fn repair_is_idempotent() {
        let context = Context::new()
            .with_message(assistant(
                vec![tool_call("call_1", "get_weather")],
                StopReason::ToolUse,
            ))
            .user("next");

        let once = normalize(&image_model(), &context).expect("repair");
        let twice = normalize(&image_model(), &once.context).expect("repair again");

        let results = twice
            .context
            .messages
            .iter()
            .filter(|message| matches!(message, Message::ToolResult(_)))
            .count();
        assert_eq!(results, 1, "the synthetic result is never duplicated");
        assert_eq!(twice.context.messages.len(), once.context.messages.len());
    }

    // --- Reasoning downgrade (issue #40) ---

    fn target_model() -> Model {
        let mut model = Model::openai_completions("model-a");
        model.provider = "provider-a".into();
        model
    }

    fn assistant_from(
        api: &str,
        provider: &str,
        model: &str,
        content: Vec<AssistantContent>,
    ) -> Message {
        let mut message = AssistantMessage::from_content(content);
        message.api = api.into();
        message.provider = provider.into();
        message.model = model.into();
        Message::Assistant(Box::new(message))
    }

    fn thinking_block(thinking: &str, signature: Option<&str>, redacted: bool) -> AssistantContent {
        AssistantContent::Thinking(ThinkingContent {
            thinking: thinking.into(),
            signature: signature.map(Into::into),
            redacted,
        })
    }

    fn signed_text_block(body: &str, signature: Option<&str>) -> AssistantContent {
        AssistantContent::Text(TextContent {
            text: body.into(),
            signature: signature.map(Into::into),
        })
    }

    fn assistant_blocks(message: &Message) -> &[AssistantContent] {
        match message {
            Message::Assistant(assistant) => &assistant.content,
            _ => &[],
        }
    }

    #[test]
    fn signatures_survive_only_when_provider_api_and_model_all_match() {
        let cases = [
            // (api, provider, model, signature survives?)
            ("openai-completions", "provider-a", "model-a", true),
            ("anthropic-messages", "provider-a", "model-a", false),
            ("openai-completions", "provider-b", "model-a", false),
            ("openai-completions", "provider-a", "model-b", false),
            ("", "", "", false),
        ];
        for (api, provider, model, survives) in cases {
            let context = Context::new()
                .with_message(assistant_from(
                    api,
                    provider,
                    model,
                    vec![
                        thinking_block("Let me think.", Some("opaque-sig"), false),
                        signed_text_block("The answer is 4.", Some("text-sig")),
                    ],
                ))
                .user("and now?");
            let normalized = normalize(&target_model(), &context).expect("text-only history");
            let blocks = assistant_blocks(&normalized.context.messages[0]);
            if survives {
                assert_eq!(
                    blocks,
                    &[
                        thinking_block("Let me think.", Some("opaque-sig"), false),
                        signed_text_block("The answer is 4.", Some("text-sig")),
                    ],
                    "{api}/{provider}/{model} must round-trip verbatim"
                );
                assert!(normalized.diagnostics.is_empty());
            } else {
                assert_eq!(
                    blocks,
                    &[
                        signed_text_block("Let me think.", None),
                        signed_text_block("The answer is 4.", None),
                    ],
                    "{api}/{provider}/{model} must downgrade to unsigned text"
                );
                assert!(
                    normalized
                        .diagnostics
                        .iter()
                        .any(|d| d.code == DiagnosticCode::ReasoningDowngraded),
                    "{api}/{provider}/{model} must report the downgrade"
                );
            }
        }
    }

    #[test]
    fn the_response_model_id_is_the_producing_model() {
        for (response_model, survives) in [(Some("model-a"), true), (Some("model-b"), false)] {
            let mut built = AssistantMessage::from_content(vec![thinking_block(
                "Let me think.",
                Some("opaque-sig"),
                false,
            )]);
            built.api = "openai-completions".into();
            built.provider = "provider-a".into();
            built.model = "router-auto".into();
            built.response_model = response_model.map(Into::into);
            let context = Context::new()
                .with_message(Message::Assistant(Box::new(built)))
                .user("and now?");
            let normalized = normalize(&target_model(), &context).expect("text-only history");
            let blocks = assistant_blocks(&normalized.context.messages[0]);
            if survives {
                assert_eq!(
                    blocks,
                    &[thinking_block("Let me think.", Some("opaque-sig"), false)],
                    "response_model `{response_model:?}` matches the target"
                );
            } else {
                assert_eq!(
                    blocks,
                    &[signed_text_block("Let me think.", None)],
                    "response_model `{response_model:?}` does not match the target"
                );
            }
        }
    }

    #[test]
    fn empty_and_redacted_thinking_is_omitted_cross_model() {
        let context = Context::new()
            .with_message(assistant_from(
                "anthropic-messages",
                "kimi",
                "k2-thinking",
                vec![
                    thinking_block("", Some("opaque-sig"), false),
                    thinking_block("   ", None, false),
                    thinking_block("", Some("OPAQUE-DATA"), true),
                    thinking_block("redacted body", Some("OPAQUE-DATA"), true),
                    signed_text_block("The answer is 4.", None),
                ],
            ))
            .user("and now?");
        let normalized = normalize(&target_model(), &context).expect("text-only history");
        assert_eq!(
            assistant_blocks(&normalized.context.messages[0]),
            &[signed_text_block("The answer is 4.", None)]
        );
        let diagnostic = normalized
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::ReasoningDowngraded)
            .expect("the omission is reported");
        assert!(
            diagnostic.message.contains('4'),
            "four blocks omitted: {}",
            diagnostic.message
        );
    }

    #[test]
    fn reasoning_rules_also_run_on_an_image_capable_model() {
        let context = Context::new()
            .with_message(assistant_from(
                "anthropic-messages",
                "kimi",
                "k2-thinking",
                vec![thinking_block("Let me think.", Some("opaque-sig"), false)],
            ))
            .user("and now?");
        let normalized = normalize(&image_model(), &context).expect("text-only history");
        assert_eq!(
            assistant_blocks(&normalized.context.messages[0]),
            &[signed_text_block("Let me think.", None)]
        );
    }

    // --- Tool-call id rewrite (issue #40) ---

    fn tool_call_id_of(message: &Message) -> &str {
        match message {
            Message::Assistant(assistant) => match &assistant.content[0] {
                AssistantContent::ToolCall(call) => &call.id,
                _ => panic!("expected a tool call"),
            },
            Message::ToolResult(result) => &result.tool_call_id,
            _ => panic!("expected a tool call or result"),
        }
    }

    fn id_context(id: &str) -> Context {
        Context::new()
            .with_message(assistant_from(
                "openai-completions",
                "provider-a",
                "model-a",
                vec![tool_call(id, "read")],
            ))
            .with_message(Message::ToolResult(ToolResultMessage::text(
                id, "read", "done",
            )))
            .user("and now?")
    }

    #[test]
    fn valid_tool_call_ids_are_left_alone() {
        for id in ["call_1", "toolu_01AbC-", &"x".repeat(64)] {
            let context = id_context(id);
            let normalized = normalize(&target_model(), &context).expect("text-only history");
            assert_eq!(tool_call_id_of(&normalized.context.messages[0]), id);
            assert_eq!(tool_call_id_of(&normalized.context.messages[1]), id);
            assert!(normalized.diagnostics.is_empty(), "{id} is already valid");
        }
    }

    #[test]
    fn invalid_tool_call_ids_are_rewritten_deterministically_and_consistently() {
        let context = id_context("call:abc/123");
        let once = normalize(&target_model(), &context).expect("text-only history");
        let twice = normalize(&target_model(), &context).expect("deterministic");
        let call_id = tool_call_id_of(&once.context.messages[0]);
        let result_id = tool_call_id_of(&once.context.messages[1]);

        assert_ne!(call_id, "call:abc/123");
        assert_eq!(call_id, result_id, "call and result ids stay paired");
        assert!(call_id.len() <= 64, "{call_id} fits the limit");
        assert!(
            call_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            "{call_id} matches ^[a-zA-Z0-9_-]{{1,64}}$"
        );
        assert_eq!(
            tool_call_id_of(&twice.context.messages[0]),
            call_id,
            "the rewrite is deterministic"
        );
        assert!(
            once.diagnostics
                .iter()
                .any(|d| d.code == DiagnosticCode::ToolCallIdRewritten)
        );
        // Distinct invalid ids stay distinct after the rewrite.
        let other = normalize(&target_model(), &id_context("call:abc/124")).expect("deterministic");
        assert_ne!(tool_call_id_of(&other.context.messages[0]), call_id);
    }

    #[test]
    fn overlong_empty_and_non_ascii_ids_are_rewritten_to_valid_ids() {
        for id in ["x".repeat(100).as_str(), "", "工具-1"] {
            let normalized =
                normalize(&target_model(), &id_context(id)).expect("text-only history");
            let rewritten = tool_call_id_of(&normalized.context.messages[0]);
            assert!(
                !rewritten.is_empty() && rewritten.len() <= 64,
                "{rewritten}"
            );
            assert!(
                rewritten
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
                "{rewritten} is a valid id for input `{id}`"
            );
        }
    }

    #[test]
    fn reasoning_and_id_normalization_are_idempotent() {
        let context = Context::new()
            .with_message(assistant_from(
                "anthropic-messages",
                "kimi",
                "k2-thinking",
                vec![
                    thinking_block("Let me think.", Some("opaque-sig"), false),
                    tool_call("call:abc/123", "read"),
                ],
            ))
            .with_message(Message::ToolResult(ToolResultMessage::text(
                "call:abc/123",
                "read",
                "done",
            )))
            .user("and now?");
        let once = normalize(&target_model(), &context).expect("text-only history");
        let twice = normalize(&target_model(), &once.context).expect("idempotent");
        assert_eq!(twice.context, once.context);
        assert!(twice.diagnostics.is_empty(), "nothing left to rewrite");
    }
}
