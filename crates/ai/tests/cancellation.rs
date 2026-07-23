//! Ticket #17: explicit cancellation via `StreamOptions.cancellation`.
//!
//! `CancellationToken` must cover the auth-resolver wait, HTTP connect and
//! response-header wait, retry backoff sleeps, and SSE body reads — each
//! terminating the stream with a single `Error { reason: Aborted }` that
//! preserves whatever content had already streamed, with no further retries.
//! Plain-drop (no token at all) must still tear down the underlying HTTP
//! connection instead of leaking it.
//!
//! None of these tests sleep in real time: each cancellation point is
//! exercised by holding that specific await point open (a resolver that
//! never resolves, a TCP peer that never responds, an SSE body that never
//! sends its next chunk) and synchronizing via a `Notify` rather than a
//! timer, or — for the retry-backoff case — paused tokio time.

use std::sync::Arc;
use std::time::Duration;

use banshu_ai::{
    AssistantMessageEvent, Auth, AuthResolver, CancellationToken, Context, Model, Provider,
    ResolvedAuth, Result, StopReason, StreamOptions, async_trait,
};
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn context() -> Context {
    Context::new().user("hi")
}

/// Drive `stream` to completion, collecting every remaining event. Used after
/// cancelling so the test can also assert there is exactly one terminal event
/// (no stray retries or duplicate terminals) rather than just the first one.
async fn collect_remaining(mut stream: banshu_ai::MessageStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

fn assert_single_aborted(events: &[AssistantMessageEvent]) -> &banshu_ai::AssistantMessage {
    assert_eq!(
        events.len(),
        1,
        "expected exactly one event (the Aborted terminal) after cancelling, got {events:?}"
    );
    match &events[0] {
        AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        } => {
            assert_eq!(error.stop_reason, StopReason::Aborted);
            error
        }
        other => panic!("expected an Aborted terminal, got {other:?}"),
    }
}

/// An `AuthResolver` that signals `started` the moment `resolve()` is entered,
/// then hangs forever — simulating an in-flight credential lookup a caller
/// wants to cancel.
struct HangingResolver {
    started: Arc<Notify>,
}

#[async_trait]
impl AuthResolver for HangingResolver {
    async fn check(&self) -> Result<bool> {
        Ok(true)
    }

    async fn resolve(&self) -> Result<ResolvedAuth> {
        self.started.notify_one();
        std::future::pending().await
    }
}

/// A raw TCP listener that accepts one connection, signals `accepted`, and
/// then never writes a response — simulating a connect that succeeds (TCP
/// handshake done) but hangs waiting for response headers.
async fn hanging_connect_server(accepted: Arc<Notify>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        accepted.notify_one();
        // Hold the connection open; never read or write anything further.
        let _socket = socket;
        std::future::pending::<()>().await;
    });
    format!("http://{addr}")
}

/// A raw TCP responder used where wiremock can't hold a connection open
/// mid-response: it writes `chunks` as HTTP/1.1 chunked-encoding frames (each
/// one SSE `data: ...\n\n` event), signals `sent` once they're all written,
/// then holds the connection open — unterminated — forever.
async fn hanging_chunked_server(chunks: Vec<String>, sent: Arc<Notify>) -> String {
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
        sent.notify_one();
        std::future::pending::<()>().await;
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn cancel_during_resolver_wait() {
    let started = Arc::new(Notify::new());
    let token = CancellationToken::new();
    let provider = Provider::openai_compatible("p", "P", "http://127.0.0.1:0", ["UNUSED"])
        .with_auth(Auth::custom(Arc::new(HangingResolver {
            started: started.clone(),
        })));
    let model = Model::openai_completions("m").with_base_url("http://127.0.0.1:0");
    let options = StreamOptions {
        cancellation: Some(token.clone()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));

    let handle = tokio::spawn(collect_remaining(stream));
    started.notified().await;
    token.cancel();

    let events = handle.await.expect("task did not panic");
    let error = assert_single_aborted(&events);
    assert!(error.content.is_empty());
}

#[tokio::test]
async fn cancel_during_http_connect() {
    let accepted = Arc::new(Notify::new());
    let base_url = hanging_connect_server(accepted.clone()).await;
    let token = CancellationToken::new();
    let provider = Provider::openai_compatible("p", "P", base_url.clone(), ["UNUSED"]);
    let model = Model::openai_completions("m").with_base_url(base_url);
    let options = StreamOptions {
        api_key: Some("k".into()),
        cancellation: Some(token.clone()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));

    let handle = tokio::spawn(collect_remaining(stream));
    accepted.notified().await;
    token.cancel();

    let events = handle.await.expect("task did not panic");
    assert_single_aborted(&events);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn cancel_during_retry_backoff_sleep() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("retry-after-ms", "5000")
                .set_body_string("service unavailable"),
        )
        .mount(&server)
        .await;

    let token = CancellationToken::new();
    let provider = Provider::openai_compatible("p", "P", server.uri(), ["UNUSED"]);
    let model = Model::openai_completions("m").with_base_url(server.uri());
    let options = StreamOptions {
        api_key: Some("k".into()),
        cancellation: Some(token.clone()),
        max_retries: Some(3),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));
    // Drive until the Retry event, proving the next poll is about to enter
    // the (paused) backoff sleep.
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Retry { .. })
    ));

    let handle = tokio::spawn(collect_remaining(stream));
    // Single-threaded runtime: this yield lets the spawned task actually run
    // up to (and register its waker inside) the paused sleep before we
    // cancel — no real time elapses either side of it.
    tokio::task::yield_now().await;
    token.cancel();

    let events = handle.await.expect("task did not panic");
    // No further Retry (or any other) event after the abort.
    assert_single_aborted(&events);
}

#[tokio::test]
async fn cancel_before_first_content_terminates_aborted() {
    let sent = Arc::new(Notify::new());
    let base_url = hanging_chunked_server(Vec::new(), sent.clone()).await;
    let token = CancellationToken::new();
    let provider = Provider::openai_compatible("p", "P", base_url.clone(), ["UNUSED"]);
    let model = Model::openai_completions("m").with_base_url(base_url);
    let options = StreamOptions {
        api_key: Some("k".into()),
        cancellation: Some(token.clone()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));

    let handle = tokio::spawn(collect_remaining(stream));
    sent.notified().await;
    token.cancel();

    let events = handle.await.expect("task did not panic");
    let error = assert_single_aborted(&events);
    assert!(error.content.is_empty());
}

#[tokio::test]
async fn cancel_mid_text_preserves_partial_content() {
    let sent = Arc::new(Notify::new());
    let chunk =
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"
            .to_string();
    let base_url = hanging_chunked_server(vec![chunk], sent.clone()).await;
    let token = CancellationToken::new();
    let provider = Provider::openai_compatible("p", "P", base_url.clone(), ["UNUSED"]);
    let model = Model::openai_completions("m").with_base_url(base_url);
    let options = StreamOptions {
        api_key: Some("k".into()),
        cancellation: Some(token.clone()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextStart { content_index: 0 })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextDelta { content_index: 0, ref delta }) if delta == "Hel"
    ));

    let handle = tokio::spawn(collect_remaining(stream));
    sent.notified().await;
    token.cancel();

    let events = handle.await.expect("task did not panic");
    let error = assert_single_aborted(&events);
    assert_eq!(error.text(), "Hel");
}

#[tokio::test]
async fn cancel_mid_thinking_preserves_partial_content() {
    let sent = Arc::new(Notify::new());
    let chunk = "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"pondering\"},\"finish_reason\":null}]}\n\n"
        .to_string();
    let base_url = hanging_chunked_server(vec![chunk], sent.clone()).await;
    let token = CancellationToken::new();
    let provider = Provider::openai_compatible("p", "P", base_url.clone(), ["UNUSED"]);
    let model = Model::openai_completions("m").with_base_url(base_url);
    let options = StreamOptions {
        api_key: Some("k".into()),
        cancellation: Some(token.clone()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ThinkingStart { content_index: 0 })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ThinkingDelta { content_index: 0, ref delta }) if delta == "pondering"
    ));

    let handle = tokio::spawn(collect_remaining(stream));
    sent.notified().await;
    token.cancel();

    let events = handle.await.expect("task did not panic");
    let error = assert_single_aborted(&events);
    match error.content.first() {
        Some(banshu_ai::AssistantContent::Thinking(thinking)) => {
            assert_eq!(thinking.thinking, "pondering");
        }
        other => panic!("expected a preserved thinking block, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_mid_tool_call_preserves_partial_arguments() {
    let sent = Arc::new(Notify::new());
    let chunks = vec![
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n".to_string(),
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"ping\"}}\n\n".to_string(),
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1\"}}\n\n".to_string(),
    ];
    let base_url = hanging_chunked_server(chunks, sent.clone()).await;
    let token = CancellationToken::new();
    let provider = Provider::anthropic_compatible("p", "P", base_url.clone(), ["UNUSED"]);
    let model = Model::anthropic_messages("m").with_base_url(base_url);
    let options = StreamOptions {
        api_key: Some("k".into()),
        cancellation: Some(token.clone()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ToolCallStart { content_index: 0 })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ToolCallDelta { content_index: 0, ref delta }) if delta == "{\"a\":1"
    ));

    let handle = tokio::spawn(collect_remaining(stream));
    sent.notified().await;
    token.cancel();

    let events = handle.await.expect("task did not panic");
    let error = assert_single_aborted(&events);
    match error.content.first() {
        Some(banshu_ai::AssistantContent::ToolCall(call)) => {
            assert_eq!(call.id, "call_1");
            assert_eq!(call.name, "ping");
            assert_eq!(call.raw_arguments.as_deref(), Some("{\"a\":1"));
        }
        other => panic!("expected a preserved tool call block, got {other:?}"),
    }
}

#[tokio::test]
async fn dropping_the_stream_without_cancelling_still_closes_the_connection() {
    let established = Arc::new(Notify::new());
    let closed = Arc::new(Notify::new());
    let established_clone = established.clone();
    let closed_clone = closed.clone();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buf = [0u8; 4096];
        let _ = socket.read(&mut buf).await;
        let response = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
        let _ = socket.write_all(response.as_bytes()).await;
        let _ = socket.flush().await;
        established_clone.notify_one();
        // The client never sends anything further; a read returning Ok(0)
        // (EOF) or an error means it closed its side of the connection.
        let mut scratch = [0u8; 16];
        let _ = socket.read(&mut scratch).await;
        closed_clone.notify_one();
    });

    let base_url = format!("http://{addr}");
    let provider = Provider::openai_compatible("p", "P", base_url.clone(), ["UNUSED"]);
    let model = Model::openai_completions("m").with_base_url(base_url);
    // Deliberately no `cancellation` token: this exercises plain drop.
    let options = StreamOptions {
        api_key: Some("k".into()),
        ..Default::default()
    };

    let mut stream = provider.stream(&model, &context(), &options);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start)
    ));

    let handle = tokio::spawn(async move { stream.next().await });
    established.notified().await;
    // Abort the task holding the stream without it ever observing a
    // terminal event — proving the underlying connection still tears down.
    handle.abort();
    let _ = handle.await;

    tokio::time::timeout(Duration::from_secs(2), closed.notified())
        .await
        .expect("dropping the stream should close the underlying TCP connection, not leak it");
}
