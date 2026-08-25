//! The OpenAI adapter observes a formal termination signal (`data: [DONE]` or
//! a chunk's `finish_reason`) before reporting success, unless the provider
//! declares [`OpenAiStreamTermination::CleanEofCompletion`] — an attestation
//! that its endpoint closes the connection only after the final chunk, so a
//! clean EOF completes a structurally finished response.
//!
//! A bare EOF under the strict default is a dropped connection, not a
//! completed response, and so is a declared EOF whose response is *not*
//! structurally finished (a tool call's arguments cut mid-JSON) — both surface
//! as `ErrorKind::StreamInterrupted`. A chunk cut mid-event stays the
//! `ErrorKind::Protocol` violation it already is, and a mid-stream transport
//! failure is never an inferred completion. Content streamed before any
//! failure is still preserved, and every stream emits exactly one terminal
//! `Done`/`Error` event.

use banshu_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context, ErrorKind, MessageStream,
    Model, OpenAiCompat, OpenAiStreamTermination, Provider, StopReason, StreamOptions,
};
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TRUNCATED_SSE_BODY: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n";

/// Two complete text chunks, then a clean EOF — no `[DONE]`, no
/// `finish_reason`. Structurally finished: text carries no wire terminator of
/// its own.
const CLEAN_TEXT_SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
);

/// A tool call whose arguments arrive as two fragments forming complete JSON,
/// then a clean EOF.
const CLEAN_TOOL_CALL_SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
);

/// A tool call whose accumulated arguments are truncated JSON at a clean EOF —
/// structurally unfinished, so inference must not complete it.
const UNFINISHED_TOOL_CALL_SSE_BODY: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Par\"}}]},\"finish_reason\":null}]}\n\n";

/// A complete text chunk followed by a chunk cut mid-JSON (the connection
/// closes before the event, let alone its blank line, is whole).
const CUT_CHUNK_SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel",
);

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

/// A provider that declares clean-EOF completion against the given base URL.
fn declared_provider(base_url: &str) -> Provider {
    Provider::openai_compatible("deepseek", "DeepSeek", base_url, ["DEEPSEEK_API_KEY"])
        .with_openai_compat(OpenAiCompat {
            stream_termination: OpenAiStreamTermination::CleanEofCompletion,
            ..OpenAiCompat::default()
        })
}

/// Stream one "hi" request through a provider declaring clean-EOF completion.
fn stream_declared(base_url: &str) -> MessageStream {
    let provider = declared_provider(base_url);
    let model = Model::openai_completions("deepseek-chat").with_base_url(base_url);
    provider.stream(&model, &Context::new().user("hi"), &options())
}

async fn serve(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(server)
        .await;
}

/// Drive a stream to its end, returning every event and the terminal message.
async fn collect(mut stream: MessageStream) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let message = stream
        .result()
        .expect("a stream that ended observed a terminal event")
        .clone();
    (events, message)
}

/// The contract's single-termination rule: exactly one `Done`/`Error`, and it
/// comes last.
fn assert_single_terminal(events: &[AssistantMessageEvent]) {
    assert!(!events.is_empty(), "every stream emits at least Start");
    let terminals: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| {
            matches!(
                event,
                AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
            )
        })
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        terminals,
        vec![events.len() - 1],
        "exactly one terminal event, and it is last: {events:?}"
    );
}

#[tokio::test]
async fn eof_without_a_termination_signal_is_stream_interrupted() {
    let server = MockServer::start().await;
    serve(&server, TRUNCATED_SSE_BODY).await;

    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("hi");

    let message = provider.stream(&model, &context, &options()).finish().await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason, None);
    assert_eq!(message.error_kind, Some(ErrorKind::StreamInterrupted));
    assert_eq!(
        message.text(),
        "Hello",
        "partial content streamed before the drop must be preserved"
    );
}

#[tokio::test]
async fn declared_clean_eof_completes_a_structurally_finished_text_response() {
    let server = MockServer::start().await;
    serve(&server, CLEAN_TEXT_SSE_BODY).await;

    let (events, message) = collect(stream_declared(&server.uri())).await;

    assert_single_terminal(&events);
    assert!(
        matches!(
            events.last(),
            Some(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                ..
            })
        ),
        "the terminal event is a successful Done: {events:?}"
    );
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        message.raw_stop_reason, None,
        "an inferred completion has no wire stop reason"
    );
    assert_eq!(message.error_kind, None);
    assert_eq!(message.text(), "Hello!");
}

#[tokio::test]
async fn declared_clean_eof_completes_completed_tool_calls() {
    let server = MockServer::start().await;
    serve(&server, CLEAN_TOOL_CALL_SSE_BODY).await;

    let (events, message) = collect(stream_declared(&server.uri())).await;

    assert_single_terminal(&events);
    assert!(
        matches!(
            events.last(),
            Some(AssistantMessageEvent::Done {
                reason: StopReason::ToolUse,
                ..
            })
        ),
        "the terminal event is a successful Done: {events:?}"
    );
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(message.raw_stop_reason, None);
    match &message.content[0] {
        AssistantContent::ToolCall(call) => {
            assert_eq!(call.id, "call_1");
            assert_eq!(call.name, "get_weather");
            assert_eq!(call.arguments, serde_json::json!({ "city": "Paris" }));
        }
        other => panic!("expected a tool call, got {other:?}"),
    }
}

#[tokio::test]
async fn declared_eof_during_an_unfinished_tool_call_is_stream_interrupted() {
    let server = MockServer::start().await;
    serve(&server, UNFINISHED_TOOL_CALL_SSE_BODY).await;

    let (events, message) = collect(stream_declared(&server.uri())).await;

    assert_single_terminal(&events);
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::StreamInterrupted));
    match &message.content[0] {
        AssistantContent::ToolCall(call) => {
            assert_eq!(
                call.raw_arguments.as_deref(),
                Some(r#"{"city":"Par"#),
                "the partial arguments streamed before the drop must be preserved"
            );
        }
        other => panic!("expected a tool call, got {other:?}"),
    }
}

#[tokio::test]
async fn declared_eof_during_an_unfinished_text_chunk_is_a_protocol_failure() {
    let server = MockServer::start().await;
    serve(&server, CUT_CHUNK_SSE_BODY).await;

    let (events, message) = collect(stream_declared(&server.uri())).await;

    assert_single_terminal(&events);
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Protocol));
    assert_eq!(
        message.text(),
        "Hello",
        "content from the whole chunk is preserved; the cut chunk contributed nothing"
    );
}

#[tokio::test]
async fn declared_clean_eof_with_no_content_is_stream_interrupted() {
    let server = MockServer::start().await;
    // A 200 with an empty body: zero events, then a clean EOF. No content
    // block ever started, so there is no structurally finished response to
    // infer — this is indistinguishable from a drop before the first chunk.
    serve(&server, "").await;

    let (events, message) = collect(stream_declared(&server.uri())).await;

    assert_single_terminal(&events);
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::StreamInterrupted));
    assert!(message.content.is_empty());
}

/// A raw TCP responder wiremock can't impersonate: it writes `chunks` as
/// HTTP/1.1 chunked-encoding frames (each one SSE `data: ...\n\n` event), then
/// drops the connection without the terminating `0\r\n\r\n` chunk — a
/// transport failure, not a clean EOF.
async fn dropping_chunked_server(chunks: Vec<String>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buf = [0u8; 4096];
        let _ = socket.read(&mut buf).await;
        let mut response = String::from(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
        );
        for chunk in &chunks {
            response.push_str(&format!("{:x}\r\n{}\r\n", chunk.len(), chunk));
        }
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        socket.flush().await.expect("flush");
        // Dropping the socket here truncates the chunked body: the terminal
        // chunk never arrives.
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn declared_connection_failure_is_never_an_inferred_completion() {
    let base_url = dropping_chunked_server(vec![
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n"
            .to_string(),
    ])
    .await;

    let (events, message) = collect(stream_declared(&base_url)).await;

    assert_single_terminal(&events);
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_kind,
        Some(ErrorKind::StreamInterrupted),
        "a mid-stream transport failure is an interruption, never a completion"
    );
    assert_eq!(
        message.text(),
        "Hello",
        "partial content streamed before the failure must be preserved"
    );
}
