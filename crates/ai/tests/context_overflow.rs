//! Context-overflow classification.
//!
//! HTTP fixtures cover the known overflow error shapes of all six target
//! providers (DeepSeek, Z.AI, MiniMax, Moonshot, Kimi For Coding, Xiaomi
//! MiMo) across the two wire protocols, plus the silent (usage-based) and
//! truncation (length-stop) overflow shapes. Rate limits, quota responses,
//! timeouts, and overload are never classified as overflow, classifications
//! carry bounded redacted evidence diagnostics, and an unknown (zero) context
//! window never invents overflow from usage.

use banshu_ai::{
    AssistantMessage, Context, DiagnosticCode, ErrorKind, Model, Provider, StopReason,
    StreamOptions, is_context_overflow,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        // Classification must be terminal on its own; the assertion on
        // received-request count proves it.
        max_retries: Some(0),
        ..Default::default()
    }
}

fn openai_provider(server: &MockServer, id: &str) -> Provider {
    Provider::openai_compatible(id, id, server.uri(), ["X"])
}

fn anthropic_provider(server: &MockServer, id: &str) -> Provider {
    Provider::anthropic_compatible(id, id, server.uri(), ["X"])
}

fn openai_model(server: &MockServer, id: &str) -> Model {
    Model::openai_completions(id).with_base_url(server.uri())
}

fn anthropic_model(server: &MockServer, id: &str) -> Model {
    Model::anthropic_messages(id).with_base_url(server.uri())
}

/// Which wire protocol a fixture speaks.
enum Wire {
    OpenAi(&'static str),
    Anthropic(&'static str),
}

impl Wire {
    fn id(&self) -> &'static str {
        match self {
            Wire::OpenAi(id) | Wire::Anthropic(id) => id,
        }
    }

    fn url_path(&self) -> &'static str {
        match self {
            Wire::OpenAi(_) => "/chat/completions",
            Wire::Anthropic(_) => "/v1/messages",
        }
    }
}

/// One failing request/response cycle: mount `status` + `body`, stream, and
/// return the terminal message plus how often the server was hit.
async fn terminal_error(wire: &Wire, status: u16, body: &str) -> (AssistantMessage, usize) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(wire.url_path()))
        .respond_with(ResponseTemplate::new(status).set_body_string(body))
        .mount(&server)
        .await;
    let (provider, model) = match wire {
        Wire::OpenAi(id) => (
            openai_provider(&server, id),
            openai_model(&server, "overflow-fixture"),
        ),
        Wire::Anthropic(id) => (
            anthropic_provider(&server, id),
            anthropic_model(&server, "overflow-fixture"),
        ),
    };
    let message = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await;
    let hits = server.received_requests().await.map_or(0, |r| r.len());
    (message, hits)
}

/// The six target providers' documented context-overflow error shapes.
const OVERFLOW_FIXTURES: &[(Wire, u16, &str, &str)] = &[
    (
        Wire::OpenAi("deepseek"),
        400,
        "{\"error\":{\"message\":\"This model's maximum context length is 65536 tokens. However, you requested 70000 tokens (69900 in the messages, 100 in the completion). Please reduce the length of the messages.\",\"type\":\"invalid_request_error\",\"code\":\"invalid_request_error\"}}",
        "maximum context length is",
    ),
    (
        Wire::OpenAi("zai"),
        400,
        "{\"error\":{\"code\":\"1211\",\"message\":\"model_context_window_exceeded: the prompt exceeds the model's maximum context length\"}}",
        "model_context_window_exceeded",
    ),
    (
        Wire::Anthropic("minimax"),
        400,
        "{\"error\":{\"type\":\"invalid_request_error\",\"message\":\"invalid params, context window exceeds limit\"}}",
        "context window exceeds limit",
    ),
    (
        Wire::OpenAi("moonshot"),
        400,
        "{\"error\":{\"message\":\"Your request exceeded model token limit: 262144 (requested: 300000)\",\"type\":\"invalid_request_error\"}}",
        "exceeded model token limit",
    ),
    (
        Wire::Anthropic("kimi"),
        400,
        "{\"error\":{\"type\":\"invalid_request_error\",\"message\":\"prompt is too long: 300000 tokens > 262144 maximum\"}}",
        "prompt is too long",
    ),
    (
        Wire::OpenAi("xiaomi"),
        400,
        "{\"error\":{\"message\":\"Input length (1050000) exceeds model's maximum context length (1048576).\",\"type\":\"invalid_request_error\"}}",
        "exceeds",
    ),
];

#[tokio::test]
async fn classifies_all_six_providers_overflow_shapes() {
    for (wire, status, body, evidence) in OVERFLOW_FIXTURES {
        let (message, hits) = terminal_error(wire, *status, body).await;
        let provider = wire.id();
        assert_eq!(
            message.error_kind,
            Some(ErrorKind::ContextOverflow),
            "{provider}: expected ContextOverflow, got {:?} ({:?})",
            message.error_kind,
            message.error_message
        );
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            hits, 1,
            "{provider}: overflow must be terminal, not retried"
        );
        assert!(
            is_context_overflow(&message, 262_144),
            "{provider}: the message-level predicate must agree"
        );
        // The classification names the evidence it matched.
        let diagnostic = message
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::ContextOverflow)
            .unwrap_or_else(|| panic!("{provider}: missing evidence diagnostic"));
        assert!(
            diagnostic.message.contains(evidence),
            "{provider}: diagnostic {:?} should name `{evidence}`",
            diagnostic.message
        );
    }
}

#[tokio::test]
async fn anthropic_413_request_too_large_is_overflow() {
    let (message, _) = terminal_error(
        &Wire::Anthropic("kimi"),
        413,
        "{\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}",
    )
    .await;
    assert_eq!(message.error_kind, Some(ErrorKind::ContextOverflow));
    assert!(is_context_overflow(&message, 200_000));
}

/// Representative non-overflow failures for every target provider: 429 rate
/// limits, quota/billing exhaustion, 408 timeouts, and 529 overload. None may
/// classify as context overflow — even the ones whose bodies mention tokens.
#[tokio::test]
async fn never_classifies_rate_quota_timeout_or_overload_as_overflow() {
    let providers: &[Wire] = &[
        Wire::OpenAi("deepseek"),
        Wire::OpenAi("zai"),
        Wire::Anthropic("minimax"),
        Wire::OpenAi("moonshot"),
        Wire::Anthropic("kimi"),
        Wire::OpenAi("xiaomi"),
    ];
    let failures: &[(u16, &str, ErrorKind)] = &[
        (
            429,
            "{\"error\":{\"message\":\"Rate limit reached, please retry later.\",\"type\":\"rate_limit_error\"}}",
            ErrorKind::RateLimited,
        ),
        (
            429,
            "{\"error\":{\"message\":\"Too many tokens per minute, please wait before trying again.\",\"type\":\"rate_limit_error\"}}",
            ErrorKind::RateLimited,
        ),
        (
            429,
            "{\"error\":{\"message\":\"insufficient_quota: quota exceeded for this billing period\"}}",
            ErrorKind::QuotaExhausted,
        ),
        (
            402,
            "{\"error\":{\"message\":\"Insufficient balance\",\"type\":\"billing_error\"}}",
            ErrorKind::QuotaExhausted,
        ),
        (
            408,
            "{\"error\":{\"message\":\"Request timed out\",\"type\":\"timeout_error\"}}",
            ErrorKind::ServerError,
        ),
        (
            529,
            "{\"error\":{\"message\":\"The service is overloaded, please try again later.\",\"type\":\"overloaded_error\"}}",
            ErrorKind::Overloaded,
        ),
    ];
    for wire in providers {
        for (status, body, expected) in failures {
            let (message, _) = terminal_error(wire, *status, body).await;
            assert_eq!(
                message.error_kind,
                Some(*expected),
                "{}: HTTP {status} must stay {expected:?}, got {:?}",
                wire.id(),
                message.error_kind
            );
            assert!(
                !is_context_overflow(&message, 262_144),
                "{}: HTTP {status} must not read as overflow",
                wire.id()
            );
        }
    }
}

#[tokio::test]
async fn overflow_wording_with_rate_limit_veto_stays_invalid_request() {
    // A 400 whose body mixes overflow-ish token wording with rate-limit
    // language: the veto wins, conservatively.
    let (message, _) = terminal_error(
        &Wire::OpenAi("deepseek"),
        400,
        "{\"error\":{\"message\":\"Token rate limit exceeded: maximum context length is 65536 tokens per request window\"}}",
    )
    .await;
    assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
    assert!(!is_context_overflow(&message, 262_144));
}

#[tokio::test]
async fn a_rate_limit_quoting_overflow_wording_is_still_not_overflow() {
    // The HTTP layer keeps the 429 a rate limit (overflow only promotes
    // 400/413); the message-level predicate must defer to that definitive
    // kind even though the body quotes overflow wording with no veto word.
    let (message, _) = terminal_error(
        &Wire::OpenAi("zai"),
        429,
        "{\"error\":{\"message\":\"Your input exceeds the context window of this model\",\"type\":\"rate_limit_error\"}}",
    )
    .await;
    assert_eq!(message.error_kind, Some(ErrorKind::RateLimited));
    assert!(!is_context_overflow(&message, 262_144));
}

#[tokio::test]
async fn evidence_diagnostics_are_bounded_and_redacted() {
    let secret = "sk-overflow-fixture-secret-key";
    let body = format!(
        "{{\"error\":{{\"message\":\"prompt is too long: 300000 tokens > 262144 maximum\",\
         \"type\":\"invalid_request_error\",\
         \"debug\":\"authorization: Bearer {secret}\",\
         \"padding\":\"{}\"}}}}",
        "x".repeat(8 * 1024)
    );
    let (message, _) = terminal_error(&Wire::Anthropic("kimi"), 400, &body).await;
    assert_eq!(message.error_kind, Some(ErrorKind::ContextOverflow));
    assert!(!message.diagnostics.is_empty());
    for diagnostic in &message.diagnostics {
        assert!(
            diagnostic.message.chars().count() <= 1024,
            "diagnostic exceeds the bound: {} chars",
            diagnostic.message.chars().count()
        );
        assert!(
            !diagnostic.message.contains(secret),
            "diagnostic leaks the credential: {:?}",
            diagnostic.message
        );
    }
    // The evidence diagnostic still identifies what matched.
    assert!(
        message
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::ContextOverflow
                && d.message.contains("prompt is too long"))
    );
}

/// A successful OpenAI-protocol stream reporting the given usage.
async fn successful_stream(
    server: &MockServer,
    prompt_tokens: u64,
    completion_tokens: u64,
    finish_reason: &str,
) -> AssistantMessage {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"ok\"}},\"finish_reason\":null}}]}}\n\n\
                     data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish_reason}\"}}],\"usage\":{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":{completion_tokens}}}}}\n\n\
                     data: [DONE]\n\n",
                )),
        )
        .mount(server)
        .await;
    openai_provider(server, "zai")
        .stream(
            &openai_model(server, "glm-5"),
            &Context::new().user("hi"),
            &options(),
        )
        .finish()
        .await
}

#[tokio::test]
async fn silent_overflow_is_detected_when_usage_exceeds_the_window() {
    // z.ai can accept an oversized prompt and return successfully; only the
    // reported usage betrays the overflow.
    let server = MockServer::start().await;
    let message = successful_stream(&server, 300_000, 10, "stop").await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert!(is_context_overflow(&message, 262_144));
}

#[tokio::test]
async fn xiaomi_truncation_overflow_needs_a_full_window_and_zero_output() {
    let server = MockServer::start().await;
    let message = successful_stream(&server, 1_048_576, 0, "length").await;
    assert_eq!(message.stop_reason, StopReason::Length);
    assert!(is_context_overflow(&message, 1_048_576));

    // A length stop that produced output is an ordinary output cap, and a
    // zero-output length stop far below the window is not truncation.
    let server = MockServer::start().await;
    let message = successful_stream(&server, 100_000, 4_096, "length").await;
    assert!(!is_context_overflow(&message, 1_048_576));
    let server = MockServer::start().await;
    let message = successful_stream(&server, 100, 0, "length").await;
    assert!(!is_context_overflow(&message, 1_048_576));
}

#[tokio::test]
async fn an_unknown_zero_window_never_invents_usage_overflow() {
    let server = MockServer::start().await;
    let message = successful_stream(&server, 50_000_000, 0, "length").await;
    // Absurd usage, but with no known window there is nothing to exceed.
    assert!(!is_context_overflow(&message, 0));

    // Error evidence needs no window — it is recognized on its own.
    let (error, _) = terminal_error(
        &Wire::OpenAi("moonshot"),
        400,
        "{\"error\":{\"message\":\"Your request exceeded model token limit: 262144 (requested: 300000)\"}}",
    )
    .await;
    assert!(is_context_overflow(&error, 0));
}

#[tokio::test]
async fn ordinary_completions_are_not_overflow() {
    let server = MockServer::start().await;
    let message = successful_stream(&server, 1_000, 50, "stop").await;
    assert!(!is_context_overflow(&message, 262_144));
}
