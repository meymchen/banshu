//! The cross-protocol tool-choice contract (issue #46).
//!
//! One [`ToolChoice`] vocabulary — `Auto`, `None`, `Required`, `Named` — is
//! serialized onto each wire protocol the way that protocol spells it, and only
//! when the provider's compat declares it can express the choice. A choice the
//! provider cannot express terminates in-band with `ErrorKind::InvalidRequest`
//! before any HTTP request leaves the process; it is never silently remapped
//! onto a choice the caller did not ask for.
//!
//! Two neighbouring rules live here too, because they are pinned by the same
//! request body:
//!
//! - No `tool_choice` option means no `tool_choice` field: the payload is
//!   byte-for-byte what it was before the option existed.
//! - A tool's `strict` marker reaches the wire only when the provider's compat
//!   declares strict tool schemas; otherwise the field is omitted entirely.

use banshu_ai::{
    AnthropicCompat, AssistantMessage, Context, ErrorKind, Model, OpenAiCompat, Provider,
    StreamOptions, Tool, ToolChoice, ToolChoiceSupport,
};
use serde_json::Value;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

const OPENAI_STOP_BODY: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

const ANTHROPIC_STOP_BODY: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// Every choice, so a rejection test cannot forget one.
const ALL_CHOICES: [ToolChoice; 4] = [
    ToolChoice::Auto,
    ToolChoice::None,
    ToolChoice::Required,
    ToolChoice::Named(String::new()),
];

fn weather_tool() -> Tool {
    Tool {
        name: "get_weather".into(),
        description: "Get the weather for a city".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
        strict: false,
    }
}

fn strict_weather_tool() -> Tool {
    Tool {
        strict: true,
        ..weather_tool()
    }
}

fn options(tool_choice: Option<ToolChoice>) -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        tool_choice,
        ..Default::default()
    }
}

/// One of the two wire protocols under test: its endpoint path, stop body,
/// provider constructor, and model shape.
#[derive(Clone, Copy)]
enum Protocol {
    OpenAi,
    Anthropic,
}

impl Protocol {
    const BOTH: [Self; 2] = [Self::OpenAi, Self::Anthropic];

    async fn server(self) -> MockServer {
        let (path, body) = match self {
            Self::OpenAi => ("/chat/completions", OPENAI_STOP_BODY),
            Self::Anthropic => ("/v1/messages", ANTHROPIC_STOP_BODY),
        };
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::path(path))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        server
    }

    /// A provider declaring exactly `tool_choice` support and
    /// `strict_tool_schemas`, nothing else.
    fn provider(
        self,
        server: &MockServer,
        tool_choice: ToolChoiceSupport,
        strict_tool_schemas: bool,
    ) -> Provider {
        match self {
            Self::OpenAi => Provider::openai_compatible("acme", "Acme", server.uri(), ["X"])
                .with_openai_compat(OpenAiCompat {
                    tool_choice,
                    strict_tool_schemas,
                    ..OpenAiCompat::default()
                }),
            Self::Anthropic => Provider::anthropic_compatible("acme", "Acme", server.uri(), ["X"])
                .with_anthropic_compat(AnthropicCompat {
                    tool_choice,
                    strict_tool_schemas,
                    ..AnthropicCompat::default()
                }),
        }
    }

    fn model(self, server: &MockServer) -> Model {
        match self {
            Self::OpenAi => Model::openai_completions("acme-chat").with_base_url(server.uri()),
            Self::Anthropic => Model::anthropic_messages("acme-claude").with_base_url(server.uri()),
        }
    }
}

async fn request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .expect("request journal")
        .iter()
        .map(|request| serde_json::from_slice(&request.body).expect("JSON request"))
        .collect()
}

/// Drive one tools-carrying request to completion, returning the server (for
/// journal assertions) and the terminal message.
async fn run(
    protocol: Protocol,
    support: ToolChoiceSupport,
    strict_tool_schemas: bool,
    tool: Tool,
    choice: Option<ToolChoice>,
) -> (MockServer, AssistantMessage) {
    let server = protocol.server().await;
    let provider = protocol.provider(&server, support, strict_tool_schemas);
    let model = protocol.model(&server);
    let context = Context::new().user("weather?").with_tool(tool);
    let message = provider
        .stream(&model, &context, &options(choice))
        .finish()
        .await;
    (server, message)
}

/// The single request body an honoured `choice` put on the wire.
async fn body(protocol: Protocol, support: ToolChoiceSupport, choice: ToolChoice) -> Value {
    let (server, message) = run(protocol, support, false, weather_tool(), Some(choice)).await;
    assert_eq!(
        message.error_kind, None,
        "this choice should be honoured: {:?}",
        message.error_message
    );
    let mut bodies = request_bodies(&server).await;
    assert_eq!(bodies.len(), 1, "exactly one request should have left");
    bodies.remove(0)
}

/// Assert `choice` fails in-band with an error naming `expected`, and never
/// becomes HTTP traffic.
async fn rejected(
    protocol: Protocol,
    support: ToolChoiceSupport,
    choice: ToolChoice,
    expected: &str,
) {
    let (server, message) = run(protocol, support, false, weather_tool(), Some(choice)).await;
    assert_eq!(
        message.error_kind,
        Some(ErrorKind::InvalidRequest),
        "the choice should be rejected in-band"
    );
    let detail = message.error_message.unwrap_or_default();
    assert!(
        detail.contains(expected),
        "`{detail}` should mention `{expected}`"
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "the preflight must win before the mock server is reached"
    );
}

/// The body of a tools-carrying request (no choice) against a provider with
/// `strict_tool_schemas` set as given, offering a tool marked `strict` or not.
async fn tools_body(protocol: Protocol, strict_tool_schemas: bool, strict: bool) -> Value {
    let tool = if strict {
        strict_weather_tool()
    } else {
        weather_tool()
    };
    let (server, message) = run(
        protocol,
        ToolChoiceSupport::default(),
        strict_tool_schemas,
        tool,
        None,
    )
    .await;
    assert_eq!(message.error_kind, None, "{:?}", message.error_message);
    let mut bodies = request_bodies(&server).await;
    assert_eq!(bodies.len(), 1, "exactly one request should have left");
    bodies.remove(0)
}

// ---------------------------------------------------------------------------
// The wire shape of every supported choice
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_serializes_every_supported_choice() {
    let support = ToolChoiceSupport::ALL;
    let openai = Protocol::OpenAi;
    assert_eq!(
        body(openai, support, ToolChoice::Auto).await["tool_choice"],
        Value::String("auto".into())
    );
    assert_eq!(
        body(openai, support, ToolChoice::None).await["tool_choice"],
        Value::String("none".into())
    );
    assert_eq!(
        body(openai, support, ToolChoice::Required).await["tool_choice"],
        Value::String("required".into())
    );
    assert_eq!(
        body(openai, support, ToolChoice::Named("get_weather".into())).await["tool_choice"],
        serde_json::json!({ "type": "function", "function": { "name": "get_weather" } })
    );
}

#[tokio::test]
async fn anthropic_serializes_every_supported_choice() {
    let support = ToolChoiceSupport::ALL;
    let anthropic = Protocol::Anthropic;
    assert_eq!(
        body(anthropic, support, ToolChoice::Auto).await["tool_choice"],
        serde_json::json!({ "type": "auto" })
    );
    assert_eq!(
        body(anthropic, support, ToolChoice::None).await["tool_choice"],
        serde_json::json!({ "type": "none" })
    );
    assert_eq!(
        body(anthropic, support, ToolChoice::Required).await["tool_choice"],
        serde_json::json!({ "type": "any" })
    );
    assert_eq!(
        body(anthropic, support, ToolChoice::Named("get_weather".into())).await["tool_choice"],
        serde_json::json!({ "type": "tool", "name": "get_weather" })
    );
}

#[tokio::test]
async fn a_named_choice_preserves_the_tool_name_exactly() {
    // A name carrying characters a rewrite might touch goes out verbatim.
    let name = "Get-Weather.v2 (beta)";
    let support = ToolChoiceSupport::ALL;
    assert_eq!(
        body(Protocol::OpenAi, support, ToolChoice::Named(name.into())).await["tool_choice"]["function"]
            ["name"],
        Value::String(name.into())
    );
    assert_eq!(
        body(Protocol::Anthropic, support, ToolChoice::Named(name.into())).await["tool_choice"]["name"],
        Value::String(name.into())
    );
}

#[tokio::test]
async fn no_choice_means_no_tool_choice_field() {
    // Both protocols: the payload is what it was before the option existed.
    for protocol in Protocol::BOTH {
        let (server, message) = run(
            protocol,
            ToolChoiceSupport::default(),
            false,
            weather_tool(),
            None,
        )
        .await;
        assert_eq!(message.error_kind, None, "{:?}", message.error_message);
        let bodies = request_bodies(&server).await;
        assert_eq!(bodies.len(), 1, "exactly one request should have left");
        assert!(
            bodies[0].get("tool_choice").is_none(),
            "no choice means no field: {}",
            bodies[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Preflight rejection — nothing reaches the endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_provider_declaring_no_support_rejects_every_choice() {
    let support = ToolChoiceSupport::default();
    for choice in ALL_CHOICES {
        let choice = match choice {
            ToolChoice::Named(_) => ToolChoice::Named("get_weather".into()),
            other => other,
        };
        for protocol in Protocol::BOTH {
            rejected(protocol, support, choice.clone(), "supported choices: none").await;
        }
    }
}

#[tokio::test]
async fn a_provider_rejects_only_the_choices_it_cannot_express() {
    // The common OpenAI-compatible subset: the string forms only.
    let support = ToolChoiceSupport {
        auto: true,
        none: true,
        required: false,
        named: false,
    };
    for protocol in Protocol::BOTH {
        rejected(protocol, support, ToolChoice::Required, "required").await;
        rejected(
            protocol,
            support,
            ToolChoice::Named("get_weather".into()),
            "get_weather",
        )
        .await;
    }

    // The supported half goes out on the wire.
    assert_eq!(
        body(Protocol::OpenAi, support, ToolChoice::Auto).await["tool_choice"],
        Value::String("auto".into())
    );
    assert_eq!(
        body(Protocol::Anthropic, support, ToolChoice::None).await["tool_choice"],
        serde_json::json!({ "type": "none" })
    );
}

// ---------------------------------------------------------------------------
// Strict tool schemas — sent only when declared
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_sends_strict_only_when_the_compat_declares_it() {
    let openai = Protocol::OpenAi;
    let declared = tools_body(openai, true, true).await;
    assert_eq!(
        declared["tools"][0]["function"]["strict"],
        Value::Bool(true)
    );

    let undeclared = tools_body(openai, false, true).await;
    assert!(
        undeclared["tools"][0]["function"].get("strict").is_none(),
        "an undeclared endpoint gets no strict field: {undeclared}"
    );

    let unmarked = tools_body(openai, true, false).await;
    assert!(
        unmarked["tools"][0]["function"].get("strict").is_none(),
        "an unmarked tool stays unmarked: {unmarked}"
    );
}

#[tokio::test]
async fn anthropic_sends_strict_only_when_the_compat_declares_it() {
    let anthropic = Protocol::Anthropic;
    let declared = tools_body(anthropic, true, true).await;
    assert_eq!(declared["tools"][0]["strict"], Value::Bool(true));

    let undeclared = tools_body(anthropic, false, true).await;
    assert!(
        undeclared["tools"][0].get("strict").is_none(),
        "an undeclared endpoint gets no strict field: {undeclared}"
    );

    let unmarked = tools_body(anthropic, true, false).await;
    assert!(
        unmarked["tools"][0].get("strict").is_none(),
        "an unmarked tool stays unmarked: {unmarked}"
    );
}
