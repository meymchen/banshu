//! The OpenAI adapter must observe a formal termination signal (`data:
//! [DONE]` or a chunk's `finish_reason`) before reporting success. A bare EOF
//! without either is a dropped connection, not a completed response, and must
//! surface as `ErrorKind::StreamInterrupted` rather than a silently-successful
//! `Done` — content streamed before the drop is still preserved.

use banshu_ai::{Context, ErrorKind, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TRUNCATED_SSE_BODY: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n";

#[tokio::test]
async fn eof_without_a_termination_signal_is_stream_interrupted() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(TRUNCATED_SSE_BODY),
        )
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("hi");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider.stream(&model, &context, &options).finish().await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::StreamInterrupted));
    assert_eq!(
        message.text(),
        "Hello",
        "partial content streamed before the drop must be preserved"
    );
}
