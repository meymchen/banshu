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

    if accepts_images {
        return Ok(Normalized {
            context,
            diagnostics: Vec::new(),
        });
    }

    // Past the gate, the newest user message is known image-free, so every
    // remaining user image is by definition historical.
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

    let diagnostics = [(user_images, "user"), (tool_images, "tool-result")]
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
        })
        .collect();

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

    let last = context.messages.len().saturating_sub(1);
    let mut messages = Vec::with_capacity(context.messages.len());
    for (index, message) in context.messages.drain(..).enumerate() {
        match message {
            Message::Assistant(assistant)
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) => {}
            Message::Assistant(assistant) => {
                let synthetic: Vec<Message> = if index == last {
                    Vec::new()
                } else {
                    assistant
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
                        .collect()
                };
                messages.push(Message::Assistant(assistant));
                messages.extend(synthetic);
            }
            Message::ToolResult(result) if dropped_calls.contains(&result.tool_call_id) => {}
            other => messages.push(other),
        }
    }
    context.messages = messages;
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
        AssistantContent, AssistantMessage, ImageContent, Message, StopReason, ToolCall,
        ToolResultMessage, UserMessage,
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
}
