//! Seam 1: `Provider::stream()` against a wiremock server.
//!
//! First tracer bullet — a minimal text completion over the OpenAI-completions
//! wire protocol: one `content` SSE delta plus `[DONE]`. Asserts the assembled
//! `final_message()` and that the outgoing request carried the model + messages.

use banshu_ai::{Context, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// OpenAI-style streaming chunks, borrowed from pi's fixture shape.
const SSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello, world!\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn streams_a_minimal_text_completion() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "model": "deepseek-chat",
            "messages": [{ "role": "user", "content": "Say hi" }],
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("Say hi");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider
        .stream(&model, &context, &options)
        .final_message()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "Hello, world!");
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 5);
}
