//! Seam 1: reasoning ("thinking") streaming over openai-completions.
//!
//! DeepSeek-style reasoners emit `delta.reasoning_content` before the final
//! answer and report `completion_tokens_details.reasoning_tokens`. banshu maps
//! these to a `ThinkingContent` block and `Usage.reasoning`.

use banshu_ai::{AssistantContent, Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Let me think. \"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"2 + 2 is 4.\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer is 4.\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":10,\"completion_tokens_details\":{\"reasoning_tokens\":6}}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn assembles_thinking_then_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .mount(&server)
        .await;

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-reasoner").with_base_url(server.uri());
    let context = Context::new().user("What is 2 + 2?");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider.stream(&model, &context, &options).final_message().await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "The answer is 4.");
    assert_eq!(message.usage.reasoning, Some(6));

    let thinking = message
        .content
        .iter()
        .find_map(|c| match c {
            AssistantContent::Thinking(t) => Some(t.thinking.as_str()),
            _ => None,
        })
        .expect("expected a thinking block");
    assert_eq!(thinking, "Let me think. 2 + 2 is 4.");

    // Thinking must precede the final text block.
    let first = &message.content[0];
    assert!(matches!(first, AssistantContent::Thinking(_)));
}
