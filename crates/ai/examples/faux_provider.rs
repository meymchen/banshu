use std::time::Duration;

use banshu_ai::testing::{FauxAttempt, FauxEvent, FauxProvider, FauxScript, FauxStopReason};
use banshu_ai::{CancellationToken, Context, ErrorKind, StopReason, StreamOptions};

#[tokio::main]
async fn main() {
    let context = Context::new().user("hello");

    let success = FauxProvider::new(
        "test-model",
        FauxScript::success([FauxEvent::text("hello back")], FauxStopReason::Stop),
    );
    let message = success
        .stream(&context, &StreamOptions::default())
        .finish()
        .await;
    assert_eq!(message.text(), "hello back");

    let failure = FauxProvider::new(
        "test-model",
        FauxScript::in_band_failure(
            [FauxEvent::text("partial")],
            ErrorKind::Api,
            "scripted error",
        ),
    );
    let message = failure
        .stream(&context, &StreamOptions::default())
        .finish()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let delayed = FauxProvider::new(
        "test-model",
        FauxScript::success(
            [FauxEvent::delay(Duration::from_secs(60))],
            FauxStopReason::Stop,
        ),
    );
    let message = delayed
        .stream(
            &context,
            &StreamOptions {
                cancellation: Some(cancellation),
                ..StreamOptions::default()
            },
        )
        .finish()
        .await;
    assert_eq!(message.stop_reason, StopReason::Aborted);

    let retried = FauxProvider::new(
        "test-model",
        FauxScript::attempts([
            FauxAttempt::failure(ErrorKind::ServerError, "try again", Duration::ZERO),
            FauxAttempt::success([FauxEvent::text("recovered")], FauxStopReason::Stop),
        ]),
    );
    let message = retried
        .stream(&context, &StreamOptions::default())
        .finish()
        .await;
    assert_eq!(message.text(), "recovered");
    assert_eq!(retried.attempt_count(), 2);
}
