use std::time::Duration;

use banshu_ai::testing::{FauxAttempt, FauxEvent, FauxProvider, FauxScript, FauxStopReason};
use banshu_ai::{
    AssistantContent, AssistantMessageEvent, CancellationToken, Context, ErrorKind, StopReason,
    StreamOptions, Usage,
};
use futures::{FutureExt, StreamExt};

#[tokio::test]
async fn start_installs_the_complete_pending_response_identity() {
    let faux = FauxProvider::new("test-model", FauxScript::success([], FauxStopReason::Stop));
    let mut stream = faux.stream(&Context::new().user("hi"), &StreamOptions::default());

    let Some(AssistantMessageEvent::Start { message }) = stream.next().await else {
        panic!("expected Start");
    };

    assert_eq!(message.provider, "faux");
    assert_eq!(message.model, "test-model");
    assert_eq!(message.api, "openai-completions");
    assert!(message.timestamp > 0);
    assert!(message.content.is_empty());
    assert_eq!(message.usage, Usage::default());
    assert_eq!(message.stop_reason, StopReason::Pending);
    assert_eq!(stream.partial(), &message);
    assert!(stream.result().is_none());
}

#[tokio::test]
async fn partial_tracks_every_content_kind_until_success_replaces_it() {
    let faux = FauxProvider::new(
        "test-model",
        FauxScript::success(
            [
                FauxEvent::thinking("consider", None::<String>, false),
                FauxEvent::text("answer"),
                FauxEvent::tool_call("call_1", "lookup", r#"{"q":"rust"}"#),
            ],
            FauxStopReason::ToolUse,
        ),
    );
    let mut stream = faux.stream(&Context::new().user("hi"), &StreamOptions::default());

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start { .. })
    ));

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ThinkingStart { .. })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ThinkingDelta { .. })
    ));
    let AssistantContent::Thinking(thinking) = &stream.partial().content[0] else {
        panic!("expected partial thinking");
    };
    assert_eq!(thinking.thinking, "consider");
    assert_eq!(stream.partial().stop_reason, StopReason::Pending);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ThinkingEnd { .. })
    ));

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextStart { .. })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextDelta { .. })
    ));
    assert_eq!(stream.partial().text(), "answer");
    assert_eq!(stream.partial().stop_reason, StopReason::Pending);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextEnd { .. })
    ));

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ToolCallStart { .. })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ToolCallDelta { .. })
    ));
    let AssistantContent::ToolCall(call) = &stream.partial().content[2] else {
        panic!("expected partial tool call");
    };
    assert_eq!(call.id, "call_1");
    assert_eq!(call.name, "lookup");
    assert_eq!(call.arguments, serde_json::json!({"q": "rust"}));
    assert_eq!(stream.partial().stop_reason, StopReason::Pending);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::ToolCallEnd { .. })
    ));

    let Some(AssistantMessageEvent::Done { reason, message }) = stream.next().await else {
        panic!("expected Done");
    };
    assert_eq!(reason, StopReason::ToolUse);
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_ne!(message.stop_reason, StopReason::Pending);
    assert_eq!(stream.partial(), &message);
    assert_eq!(stream.result(), Some(&message));
}

#[tokio::test]
async fn fixed_success_script_produces_repeatable_content_and_usage() {
    let usage = Usage {
        input: 11,
        output: 4,
        total_tokens: 15,
        ..Usage::default()
    };
    let faux = FauxProvider::new(
        "test-model",
        FauxScript::success(
            [
                FauxEvent::text("Hello, world!"),
                FauxEvent::usage(usage.clone()),
            ],
            FauxStopReason::Stop,
        ),
    );

    let message = faux
        .stream(&Context::new().user("hi"), &StreamOptions::default())
        .finish()
        .await;

    assert_eq!(message.text(), "Hello, world!");
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.usage, usage);
    assert_eq!(faux.attempt_count(), 1);

    let repeated = faux
        .stream(&Context::new().user("again"), &StreamOptions::default())
        .finish()
        .await;
    assert_eq!(repeated.text(), "Hello, world!");
    assert_eq!(repeated.usage, usage);
    assert_eq!(faux.attempt_count(), 2);
}

#[tokio::test]
async fn terminal_failure_is_reported_in_band_without_credentials_or_network() {
    let faux = FauxProvider::new(
        "test-model",
        FauxScript::in_band_failure(
            [FauxEvent::text("partial answer")],
            ErrorKind::Api,
            "scripted provider error",
        ),
    );

    let message = faux
        .stream(&Context::new().user("hi"), &StreamOptions::default())
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_ne!(message.stop_reason, StopReason::Pending);
    assert_eq!(message.text(), "partial answer");
    assert_eq!(message.error_kind, Some(ErrorKind::Api));
    assert_eq!(
        message.error_message.as_deref(),
        Some("scripted provider error")
    );
    assert_eq!(faux.attempt_count(), 1);
}

#[tokio::test]
async fn thinking_signatures_and_tool_calls_are_scriptable_without_protocol_types() {
    let faux = FauxProvider::new(
        "test-model",
        FauxScript::success(
            [
                FauxEvent::thinking("check the weather", Some("opaque-signature"), false),
                FauxEvent::tool_call("call_1", "weather", r#"{"city":"Paris"}"#),
            ],
            FauxStopReason::ToolUse,
        ),
    );

    let message = faux
        .stream(&Context::new().user("weather?"), &StreamOptions::default())
        .finish()
        .await;

    let AssistantContent::Thinking(thinking) = &message.content[0] else {
        panic!("expected thinking content");
    };
    assert_eq!(thinking.thinking, "check the weather");
    assert_eq!(thinking.signature.as_deref(), Some("opaque-signature"));
    let AssistantContent::ToolCall(call) = &message.content[1] else {
        panic!("expected tool call content");
    };
    assert_eq!(call.id, "call_1");
    assert_eq!(call.name, "weather");
    assert_eq!(call.arguments, serde_json::json!({"city": "Paris"}));
    assert_eq!(message.stop_reason, StopReason::ToolUse);
}

#[tokio::test]
async fn retryable_setup_failures_run_an_exact_number_of_attempts_before_success() {
    let faux = FauxProvider::new(
        "test-model",
        FauxScript::attempts([
            FauxAttempt::failure(ErrorKind::ServerError, "first", Duration::ZERO),
            FauxAttempt::failure(ErrorKind::Transport, "second", Duration::ZERO),
            FauxAttempt::success([FauxEvent::text("third time")], FauxStopReason::Stop),
        ]),
    );
    let options = StreamOptions {
        max_retries: Some(2),
        ..StreamOptions::default()
    };
    let mut stream = faux.stream(&Context::new().user("hi"), &options);
    let mut retry_attempts = Vec::new();
    while let Some(event) = stream.next().await {
        if let AssistantMessageEvent::Retry { attempt, kind, .. } = event {
            retry_attempts.push((attempt, kind));
        }
    }

    assert_eq!(
        retry_attempts,
        vec![(1, ErrorKind::ServerError), (2, ErrorKind::Transport)]
    );
    assert_eq!(
        stream.result().expect("terminal message").text(),
        "third time"
    );
    assert_eq!(faux.attempt_count(), 3);
}

#[tokio::test]
async fn retry_budget_exhaustion_stops_at_the_exact_attempt_count() {
    let faux = FauxProvider::new(
        "test-model",
        FauxScript::attempts([
            FauxAttempt::failure(ErrorKind::ServerError, "first", Duration::ZERO),
            FauxAttempt::failure(ErrorKind::Overloaded, "second", Duration::ZERO),
            FauxAttempt::success([FauxEvent::text("unreachable")], FauxStopReason::Stop),
        ]),
    );
    let options = StreamOptions {
        max_retries: Some(1),
        ..StreamOptions::default()
    };

    let message = faux
        .stream(&Context::new().user("hi"), &options)
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Overloaded));
    assert_eq!(message.error_message.as_deref(), Some("second"));
    assert_eq!(faux.attempt_count(), 2);
}

#[tokio::test(start_paused = true)]
async fn cancellation_during_a_scripted_delay_terminates_aborted() {
    let faux = FauxProvider::new(
        "test-model",
        FauxScript::success(
            [
                FauxEvent::text("partial"),
                FauxEvent::delay(Duration::from_secs(60)),
                FauxEvent::text("too late"),
            ],
            FauxStopReason::Stop,
        ),
    );
    let cancellation = CancellationToken::new();
    let options = StreamOptions {
        cancellation: Some(cancellation.clone()),
        ..StreamOptions::default()
    };
    let mut stream = faux.stream(&Context::new().user("hi"), &options);

    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::Start { .. })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextStart { .. })
    ));
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextDelta { .. })
    ));
    assert_eq!(stream.partial().text(), "partial");
    assert_eq!(stream.partial().stop_reason, StopReason::Pending);
    assert!(matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextEnd { .. })
    ));
    let next = stream.next();
    futures::pin_mut!(next);
    assert!(next.as_mut().now_or_never().is_none());
    cancellation.cancel();

    let Some(AssistantMessageEvent::Error { reason, error }) = next.await else {
        panic!("expected an aborted terminal event");
    };
    assert_eq!(reason, StopReason::Aborted);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_ne!(error.stop_reason, StopReason::Pending);
    assert_eq!(error.text(), "partial");
}
