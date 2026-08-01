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

use crate::types::{
    Context, Diagnostic, DiagnosticCode, Message, Modality, Model, TextContent, UserContent,
    UserMessage,
};

/// The fixed text replacing a user image the target model cannot see.
const USER_IMAGE_OMITTED_PLACEHOLDER: &str = "(image omitted: model does not support images)";

/// The fixed text replacing a tool-result image the target model cannot see
/// (issue #22) — distinct from the user placeholder so the model can tell whose
/// image went missing.
const TOOL_IMAGE_OMITTED_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

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
    if accepts_images {
        return Ok(Normalized {
            context: context.clone(),
            diagnostics: Vec::new(),
        });
    }

    // Past the gate, the newest user message is known image-free, so every
    // remaining user image is by definition historical.
    let mut context = context.clone();
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
    use crate::types::{ImageContent, Message, ToolResultMessage, UserMessage};

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
}
