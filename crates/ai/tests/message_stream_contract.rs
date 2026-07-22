//! `MessageStream::partial()`/`result()`/`finish()` (PRD v0.3 §6.2), added in
//! the ProtocolEvent/MessageAssembler expand phase alongside the pre-existing
//! `final_message()` — both must keep working during the migration.

use banshu_ai::{Context, Model, Provider, StopReason, StreamOptions};
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\", world!\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4}}\n\n",
    "data: [DONE]\n\n",
);

async fn mounted_server() -> MockServer {
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
    server
}

#[tokio::test]
async fn partial_and_result_track_stream_progress() {
    let server = mounted_server().await;
    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("hi");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context, &options);
    assert!(stream.result().is_none());

    // Drain Start + TextStart + both TextDelta events; the terminal Done
    // hasn't arrived yet (TextEnd + Done still pending), so result() must
    // still be None while partial() already reflects both deltas.
    for _ in 0..4 {
        stream.next().await.expect("expected an event");
    }
    assert!(stream.result().is_none());
    assert_eq!(stream.partial().text(), "Hello, world!");

    let message = stream.finish().await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "Hello, world!");
    assert_eq!(
        stream.result().map(|m| m.text()),
        Some("Hello, world!".to_string())
    );
}

#[tokio::test]
async fn finish_drives_remaining_events_without_prior_polling() {
    let server = mounted_server().await;
    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("hi");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context, &options);
    let message = stream.finish().await;

    assert_eq!(message.text(), "Hello, world!");
    assert_eq!(
        stream.result().map(|m| m.stop_reason),
        Some(StopReason::Stop)
    );
}
