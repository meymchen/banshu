use banshu_ai::{
    AssistantContent, AssistantMessage, Context, ImageContent, Message, ThinkingContent, Tool,
    ToolCall, ToolResultMessage, UserContent, UserMessage,
};
use serde_json::json;

#[test]
fn empty_context_estimates_zero_tokens() {
    assert_eq!(Context::new().estimate_tokens(), 0);
}

#[test]
fn estimate_uses_unicode_scalars_and_rounds_the_total_up() {
    let context = Context::new().with_system("1234").user("é😀123");

    assert_eq!(context.estimate_tokens(), 3);
}

#[test]
fn image_and_tool_history_follow_the_stable_public_policy() {
    let assistant = AssistantMessage::from_content(vec![
        AssistantContent::Thinking(ThinkingContent {
            thinking: "1234".into(),
            signature: Some("provider-private-and-not-counted".into()),
            redacted: false,
        }),
        AssistantContent::ToolCall(ToolCall {
            id: "1234".into(),
            name: "1234".into(),
            arguments: json!({}),
            raw_arguments: Some("ignored in favour of parsed arguments".into()),
        }),
    ]);
    let context = Context::new()
        .with_message(Message::User(UserMessage {
            content: vec![UserContent::Image(ImageContent {
                data: "base64-payload-is-not-counted".into(),
                mime_type: "image/png".into(),
            })],
            timestamp: 1,
        }))
        .with_message(Message::Assistant(Box::new(assistant)))
        .with_message(Message::ToolResult(ToolResultMessage::text(
            "1234", "1234", "12",
        )))
        .with_tool(Tool {
            name: "1234".into(),
            description: "1234".into(),
            parameters: json!({}),
            strict: false,
        });

    // One image is 1024 tokens. The remaining stable payload has 34 Unicode
    // scalars: thinking (4), tool call id/name/JSON (4 + 4 + 2), tool result
    // id/name/text (4 + 4 + 2), and tool name/description/schema (4 + 4 + 2).
    assert_eq!(context.estimate_tokens(), 1_033);
}
