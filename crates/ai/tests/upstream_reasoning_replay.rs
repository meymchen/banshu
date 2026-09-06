//! Streaming, serialization and same/cross-model replay of upstream reasoning.
use banshu_ai::{
    Context, InMemoryCredentialStore, Message, Model, Provider, StopReason, StreamOptions,
};
use serde_json::{Value, json};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("test".into()),
        max_tokens: Some(1024),
        ..Default::default()
    }
}

#[tokio::test]
async fn reasoning_details_survive_serialization_and_only_replay_to_the_same_model() {
    let server = MockServer::start().await;
    let details = [
        json!({"type":"reasoning.text","text":"first ","index":0}),
        json!({"type":"reasoning.text","text":"second","index":0,"signature":"signed"}),
        json!({"type":"reasoning.summary","summary":"summary ","id":"s"}),
        json!({"type":"reasoning.summary","summary":"end","id":"s"}),
        json!({"type":"reasoning.encrypted","data":"opaque","id":"e"}),
        json!({"type":"reasoning.encrypted","data":"other","id":"e2"}),
    ];
    let mut sse = String::new();
    for detail in &details {
        sse.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"reasoning_details":[detail]}}]})
        ));
    }
    sse.push_str("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;
    let provider = Provider::openai_compatible("gateway", "Gateway", server.uri(), ["UNUSED"]);
    let mut model = Model::openai_completions("model").with_base_url(server.uri());
    model.provider = "gateway".into();
    let first = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await;
    assert_eq!(first.stop_reason, StopReason::Stop);
    let context = Context::new()
        .user("hi")
        .with_message(Message::Assistant(Box::new(first)))
        .user("continue");
    let context: Context = serde_json::from_str(&serde_json::to_string(&context).unwrap()).unwrap();
    provider.stream(&model, &context, &options()).finish().await;
    model.id = "another-model".into();
    provider.stream(&model, &context, &options()).finish().await;
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        body["messages"][1]["reasoning_details"],
        json!([
            {"type":"reasoning.text","text":"first second","index":0,"signature":"signed"},
            {"type":"reasoning.summary","summary":"summary end","id":"s"},
            details[4], details[5]
        ])
    );
    let cross: Value = serde_json::from_slice(&requests[2].body).unwrap();
    assert!(
        cross["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m.get("reasoning_details").is_none())
    );
    assert!(!requests[2].body.windows(6).any(|b| b == b"opaque"));
}

#[tokio::test]
async fn kimi_unsigned_thinking_replays_as_thinking() {
    let server = MockServer::start().await;
    let events = [
        json!({"type":"message_start","message":{"usage":{"input_tokens":1}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"private thought"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
        json!({"type":"message_stop"}),
    ];
    let sse: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;
    let provider = Provider::kimi(Arc::new(InMemoryCredentialStore::new()));
    let mut model = provider
        .models()
        .into_iter()
        .find(|m| m.id == "k3")
        .unwrap();
    model.base_url = server.uri();
    let first = provider
        .stream(&model, &Context::new().user("hi"), &options())
        .finish()
        .await;
    assert_eq!(first.stop_reason, StopReason::Stop);
    let context = Context::new()
        .user("hi")
        .with_message(Message::Assistant(Box::new(first)))
        .user("continue");
    provider.stream(&model, &context, &options()).finish().await;
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(body["messages"][1]["content"][0]["type"], "thinking");
    assert_eq!(body["messages"][1]["content"][0]["signature"], "");
    let mut effort_options = options();
    effort_options.reasoning = Some(banshu_ai::ReasoningEffort::Max.into());
    let result = provider
        .stream(&model, &context, &effort_options)
        .finish()
        .await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[2].body).unwrap();
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "max");
}
