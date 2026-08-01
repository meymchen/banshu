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

use crate::types::{
    AssistantContent, Context, Diagnostic, DiagnosticCode, Message, Modality, Model, TextContent,
    UserContent, UserMessage,
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

    let mut context = context.clone();
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
        AssistantContent, AssistantMessage, ImageContent, Message, ThinkingContent, ToolCall,
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

    // --- Reasoning downgrade (issue #40) ---

    fn target_model() -> Model {
        let mut model = Model::openai_completions("model-a");
        model.provider = "provider-a".into();
        model
    }

    fn assistant(
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

    fn tool_call(id: &str) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall {
            id: id.into(),
            name: "read".into(),
            arguments: serde_json::json!({}),
            raw_arguments: None,
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
                .with_message(assistant(
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
            .with_message(assistant(
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
            .with_message(assistant(
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
            .with_message(assistant(
                "openai-completions",
                "provider-a",
                "model-a",
                vec![tool_call(id)],
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
            .with_message(assistant(
                "anthropic-messages",
                "kimi",
                "k2-thinking",
                vec![
                    thinking_block("Let me think.", Some("opaque-sig"), false),
                    tool_call("call:abc/123"),
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
