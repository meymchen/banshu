use banshu_ai::{
    AssistantContent, AssistantMessage, Context, ErrorKind, ImageContent, Message, Modality, Model,
    Provider, StreamOptions, Tool, ToolCall, ToolResultMessage, UserContent, UserMessage,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const OPENAI_STOP: &str =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
const ANTHROPIC_STOP: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[derive(Clone, Copy, Debug)]
enum Protocol {
    OpenAi,
    Anthropic,
}

impl Protocol {
    const ALL: [Self; 2] = [Self::OpenAi, Self::Anthropic];

    fn request_path(self) -> &'static str {
        match self {
            Self::OpenAi => "/chat/completions",
            Self::Anthropic => "/v1/messages",
        }
    }

    fn stop_body(self) -> &'static str {
        match self {
            Self::OpenAi => OPENAI_STOP,
            Self::Anthropic => ANTHROPIC_STOP,
        }
    }

    fn provider(self, server: &MockServer) -> Provider {
        match self {
            Self::OpenAi => Provider::openai_compatible(
                "test-openai",
                "Test OpenAI-compatible",
                server.uri(),
                ["TEST_API_KEY"],
            ),
            Self::Anthropic => Provider::anthropic_compatible(
                "test-anthropic",
                "Test Anthropic",
                server.uri(),
                ["TEST_API_KEY"],
            ),
        }
    }

    fn model(self, server: &MockServer, context_window: u32, max_tokens: u32) -> Model {
        let mut model = match self {
            Self::OpenAi => Model::openai_completions("test-model"),
            Self::Anthropic => Model::anthropic_messages("test-model"),
        }
        .with_base_url(server.uri());
        model.provider = match self {
            Self::OpenAi => "test-openai".into(),
            Self::Anthropic => "test-anthropic".into(),
        };
        model.context_window = context_window;
        model.max_tokens = max_tokens;
        model
    }
}

async fn server_for(protocol: Protocol) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(protocol.request_path()))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(protocol.stop_body()),
        )
        .mount(&server)
        .await;
    server
}

fn options(max_tokens: Option<u32>) -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        max_tokens,
        ..Default::default()
    }
}

async fn sent_body(
    protocol: Protocol,
    context: &Context,
    context_window: u32,
    model_max_tokens: u32,
    requested_max_tokens: Option<u32>,
) -> Value {
    let server = server_for(protocol).await;
    let provider = protocol.provider(&server);
    let model = protocol.model(&server, context_window, model_max_tokens);
    let message = provider
        .stream(&model, context, &options(requested_max_tokens))
        .finish()
        .await;
    assert_eq!(message.error_kind, None, "{protocol:?}: {message:?}");
    let requests = server.received_requests().await.expect("request journal");
    assert_eq!(requests.len(), 1, "{protocol:?}");
    serde_json::from_slice(&requests[0].body).expect("JSON request")
}

#[tokio::test]
async fn empty_context_uses_the_known_model_maximum_for_both_protocols() {
    for protocol in Protocol::ALL {
        let body = sent_body(protocol, &Context::new(), 10, 8, None).await;
        assert_eq!(body["max_tokens"], 8, "{protocol:?}: {body}");
    }
}

#[tokio::test]
async fn an_implicit_budget_clamps_to_remaining_context_for_both_protocols() {
    let context = Context::new().user("1234567890123456");
    assert_eq!(context.estimate_tokens(), 4);

    for protocol in Protocol::ALL {
        let body = sent_body(protocol, &context, 10, 8, None).await;
        assert_eq!(body["max_tokens"], 6, "{protocol:?}: {body}");
    }
}

#[tokio::test]
async fn the_exact_remaining_context_boundary_is_accepted_for_both_protocols() {
    let context = Context::new().user("1234567890123456");

    for protocol in Protocol::ALL {
        let body = sent_body(protocol, &context, 10, 8, Some(6)).await;
        assert_eq!(body["max_tokens"], 6, "{protocol:?}: {body}");
    }
}

#[tokio::test]
async fn image_and_tool_history_contribute_to_the_implicit_clamp_for_both_protocols() {
    let context = image_and_tool_context();
    assert_eq!(context.estimate_tokens(), 1_033);

    for protocol in Protocol::ALL {
        let server = server_for(protocol).await;
        let provider = protocol.provider(&server);
        let mut model = protocol.model(&server, 1_040, 20);
        model.input.push(Modality::Image);
        let message = provider
            .stream(&model, &context, &options(None))
            .finish()
            .await;
        assert_eq!(message.error_kind, None, "{protocol:?}: {message:?}");
        let requests = server.received_requests().await.expect("request journal");
        assert_eq!(requests.len(), 1, "{protocol:?}");
        let body: Value = serde_json::from_slice(&requests[0].body).expect("JSON request");
        assert_eq!(body["max_tokens"], 7, "{protocol:?}: {body}");
    }
}

#[tokio::test]
async fn an_explicit_budget_over_remaining_context_is_rejected_before_both_protocols() {
    let context = Context::new().user("1234567890123456");

    for protocol in Protocol::ALL {
        let server = server_for(protocol).await;
        let provider = protocol.provider(&server);
        let model = protocol.model(&server, 10, 8);
        let message = provider
            .stream(&model, &context, &options(Some(7)))
            .finish()
            .await;

        assert_eq!(message.error_kind, Some(ErrorKind::InvalidRequest));
        assert!(
            message
                .error_message
                .as_deref()
                .is_some_and(|detail| detail.contains("remaining context budget of 6 tokens")),
            "{protocol:?}: {:?}",
            message.error_message
        );
        assert!(
            server
                .received_requests()
                .await
                .expect("request journal")
                .is_empty(),
            "{protocol:?} must reject before HTTP"
        );
    }
}

#[tokio::test]
async fn zero_model_limits_stay_unknown_instead_of_becoming_zero_capacity() {
    for protocol in Protocol::ALL {
        let known_output = sent_body(protocol, &Context::new().user("hello"), 0, 8, None).await;
        assert_eq!(
            known_output["max_tokens"], 8,
            "{protocol:?}: {known_output}"
        );

        let known_context = sent_body(
            protocol,
            &Context::new().user("1234567890123456"),
            10,
            0,
            None,
        )
        .await;
        assert_eq!(
            known_context["max_tokens"], 6,
            "{protocol:?}: {known_context}"
        );

        let body = sent_body(protocol, &Context::new().user("hello"), 0, 0, None).await;
        match protocol {
            Protocol::OpenAi => assert!(body.get("max_tokens").is_none(), "{body}"),
            // Anthropic requires the field; the adapter's established protocol
            // fallback remains a wire default, not fabricated model metadata.
            Protocol::Anthropic => assert_eq!(body["max_tokens"], 4_096, "{body}"),
        }
    }
}

fn image_and_tool_context() -> Context {
    let assistant = AssistantMessage::from_content(vec![
        AssistantContent::Thinking(banshu_ai::ThinkingContent {
            thinking: "1234".into(),
            signature: None,
            redacted: false,
        }),
        AssistantContent::ToolCall(ToolCall {
            id: "1234".into(),
            name: "1234".into(),
            arguments: json!({}),
            raw_arguments: None,
        }),
    ]);
    Context::new()
        .with_message(Message::User(UserMessage {
            content: vec![UserContent::Image(ImageContent {
                data: "ignored".into(),
                mime_type: "image/png".into(),
            })],
            timestamp: 1,
        }))
        .with_message(Message::Assistant(Box::new(assistant)))
        .with_message(Message::ToolResult(ToolResultMessage::text(
            "1234", "1234", "12",
        )))
        .with_tool(Tool {
            name: "1234".into(),
            description: "1234".into(),
            parameters: json!({}),
            strict: false,
        })
}
