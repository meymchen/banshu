//! The Anthropic adapter treats `message_stop` as the only success signal. A
//! stream that ends (EOF) without it is a dropped connection, not a completed
//! response, and must surface as `ErrorKind::StreamInterrupted` rather than a
//! silently-successful `Done` — content streamed before the drop is preserved.

use banshu_ai::{Context, ErrorKind, Model, Provider, StopReason, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// A well-formed text block and a message_delta, but no closing message_stop.
const TRUNCATED_SSE_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"glm\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
);

#[tokio::test]
async fn eof_without_message_stop_is_stream_interrupted() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(TRUNCATED_SSE_BODY),
        )
        .mount(&server)
        .await;

    let provider = Provider::anthropic_compatible("kimi", "Kimi", server.uri(), ["KIMI_API_KEY"]);
    let model = Model::anthropic_messages("k2p5").with_base_url(server.uri());
    let context = Context::new().user("Say hi");
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
