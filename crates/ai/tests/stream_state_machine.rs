//! PRD v0.3 §6.1/§10.1 — the public `MessageStream` state machine.
//!
//! Every stream emits exactly one `Start`, then a full `*Start`/`*Delta`/`*End`
//! sequence per content block (with a stable `content_index` across a block's
//! events), then exactly one terminal `Done`/`Error`. No non-terminal event
//! carries a full message snapshot — the compiler enforces that (there is no
//! `partial` field), and this suite pins the ordering and single-termination.

use banshu_ai::{
    AssistantMessage, AssistantMessageEvent, Context, Model, Provider, StopReason, StreamOptions,
};
use futures_util::StreamExt;
use futures_util::stream;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Interleaves reasoning, text, and a tool call so the ordering and per-block
/// `content_index` can be checked end to end.
const MIXED_SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hmm\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"ping\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4}}\n\n",
    "data: [DONE]\n\n",
);

async fn server_with(body: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    server
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    }
}

async fn collect(server: &MockServer) -> Vec<AssistantMessageEvent> {
    let provider =
        Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    let context = Context::new().user("hi");
    let mut stream = provider.stream(&model, &context, &options());
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// The number of terminal (`Done`/`Error`) events, and their 0-based position.
fn terminals(events: &[AssistantMessageEvent]) -> Vec<usize> {
    events
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            matches!(
                e,
                AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
            )
        })
        .map(|(i, _)| i)
        .collect()
}

#[tokio::test]
async fn full_start_delta_end_sequence_with_stable_content_index() {
    let server = server_with(MIXED_SSE_BODY).await;
    let events = collect(&server).await;

    // Exactly one Start, first.
    assert!(matches!(events.first(), Some(AssistantMessageEvent::Start)));
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, AssistantMessageEvent::Start))
            .count(),
        1
    );

    // Reasoning (block 0), text (block 1), tool call (block 2): every event of
    // a block reuses that block's content_index, and each block emits the full
    // Start→…→End sequence.
    #[derive(Default)]
    struct Seen {
        thinking: (bool, bool, bool),
        text: (bool, bool, bool),
        tool: (bool, bool, bool),
    }
    let mut seen = Seen::default();
    for event in &events {
        match event {
            AssistantMessageEvent::ThinkingStart { content_index } => {
                assert_eq!(*content_index, 0);
                seen.thinking.0 = true;
            }
            AssistantMessageEvent::ThinkingDelta { content_index, .. } => {
                assert_eq!(*content_index, 0);
                seen.thinking.1 = true;
            }
            AssistantMessageEvent::ThinkingEnd { content_index, .. } => {
                assert_eq!(*content_index, 0);
                seen.thinking.2 = true;
            }
            AssistantMessageEvent::TextStart { content_index } => {
                assert_eq!(*content_index, 1);
                seen.text.0 = true;
            }
            AssistantMessageEvent::TextDelta { content_index, .. } => {
                assert_eq!(*content_index, 1);
                seen.text.1 = true;
            }
            AssistantMessageEvent::TextEnd { content_index, .. } => {
                assert_eq!(*content_index, 1);
                seen.text.2 = true;
            }
            AssistantMessageEvent::ToolCallStart { content_index } => {
                assert_eq!(*content_index, 2);
                seen.tool.0 = true;
            }
            AssistantMessageEvent::ToolCallDelta { content_index, .. } => {
                assert_eq!(*content_index, 2);
                seen.tool.1 = true;
            }
            AssistantMessageEvent::ToolCallEnd { content_index, .. } => {
                assert_eq!(*content_index, 2);
                seen.tool.2 = true;
            }
            _ => {}
        }
    }
    assert_eq!(
        seen.thinking,
        (true, true, true),
        "thinking start/delta/end"
    );
    assert_eq!(seen.text, (true, true, true), "text start/delta/end");
    assert_eq!(seen.tool, (true, true, true), "tool start/delta/end");

    // Exactly one terminal, and it is the very last event.
    let terminals = terminals(&events);
    assert_eq!(terminals, vec![events.len() - 1]);
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done {
            reason: StopReason::ToolUse,
            ..
        })
    ));
}

#[tokio::test]
async fn no_delta_event_carries_a_message_snapshot() {
    let server = server_with(MIXED_SSE_BODY).await;
    let events = collect(&server).await;

    // A `partial` field would make these destructures fail to compile — the
    // structural guarantee that deltas cannot smuggle an O(n) clone. Reading
    // progress is `MessageStream::partial()`'s job, tested separately.
    for event in &events {
        match event {
            AssistantMessageEvent::TextDelta {
                content_index: _,
                delta: _,
            }
            | AssistantMessageEvent::ThinkingDelta {
                content_index: _,
                delta: _,
            }
            | AssistantMessageEvent::ToolCallDelta {
                content_index: _,
                delta: _,
            } => {}
            _ => {}
        }
    }
}

/// Each finish_reason maps to its `StopReason` and terminates the stream once.
#[tokio::test]
async fn normal_length_and_tooluse_each_terminate_once() {
    for (finish, expected) in [
        ("stop", StopReason::Stop),
        ("length", StopReason::Length),
        ("tool_calls", StopReason::ToolUse),
    ] {
        let body = format!(
            concat!(
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"x\"}},\"finish_reason\":null}}]}}\n\n",
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\"}}]}}\n\n",
                "data: [DONE]\n\n",
            ),
            finish = finish,
        );
        let server = server_with(&body).await;
        let events = collect(&server).await;
        assert_eq!(
            terminals(&events),
            vec![events.len() - 1],
            "{finish} must terminate exactly once"
        );
        match events.last() {
            Some(AssistantMessageEvent::Done { reason, .. }) => assert_eq!(*reason, expected),
            other => panic!("expected a Done for {finish}, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn http_error_terminates_exactly_once() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(400).set_body_string("{\"error\":{\"message\":\"bad\"}}"),
        )
        .mount(&server)
        .await;
    let events = collect(&server).await;
    assert_eq!(terminals(&events), vec![events.len() - 1]);
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error { .. })
    ));
}

#[tokio::test]
async fn protocol_error_terminates_exactly_once() {
    // A non-`[DONE]`, non-JSON SSE data line is a protocol violation, not a
    // silent success.
    let server = server_with("data: this is not json\n\n").await;
    let events = collect(&server).await;
    assert_eq!(terminals(&events), vec![events.len() - 1]);
    match events.last() {
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error,
        }) => assert_eq!(error.error_kind, Some(banshu_ai::ErrorKind::Protocol)),
        other => panic!("expected a protocol Error, got {other:?}"),
    }
}

/// There is no cancellation entry point yet (issue #17), so the `Aborted`
/// terminal is exercised at the `MessageStream` seam directly: a stream that
/// yields `Start` then an `Aborted` `Error` must surface a single terminal and
/// leave `result()` readable.
#[tokio::test]
async fn aborted_terminal_is_reported_once() {
    let mut aborted = AssistantMessage::from_content(Vec::new());
    aborted.stop_reason = StopReason::Aborted;
    let events = vec![
        AssistantMessageEvent::Start,
        AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error: aborted,
        },
    ];

    let mut message_stream = banshu_ai::MessageStream::new(stream::iter(events));
    let mut collected = Vec::new();
    while let Some(event) = message_stream.next().await {
        collected.push(event);
    }

    assert_eq!(terminals(&collected), vec![1]);
    assert_eq!(
        message_stream.result().map(|m| m.stop_reason),
        Some(StopReason::Aborted)
    );
    // No retry once aborted (§6.3 "不再重试"): no Retry event at any delay.
    assert!(
        !collected
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::Retry { .. }))
    );
}
