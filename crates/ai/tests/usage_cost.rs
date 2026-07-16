//! Seam 2: cost computation from model rates and reported token usage.

use banshu_ai::{Context, Model, ModelCost, Provider, StreamOptions};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// 1M prompt tokens and 1M completion tokens, so cost equals the per-million rate.
const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1000000,\"completion_tokens\":1000000}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn computes_cost_from_model_rates() {
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

    let provider = Provider::openai_compatible("deepseek", "DeepSeek", server.uri(), ["DEEPSEEK_API_KEY"]);
    let mut model = Model::openai_completions("deepseek-chat").with_base_url(server.uri());
    // DeepSeek-style published rates ($/1M tokens).
    model.cost = ModelCost {
        input: 0.27,
        output: 1.10,
        cache_read: 0.07,
        cache_write: 0.0,
    };
    let context = Context::new().user("hi");
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let message = provider.stream(&model, &context, &options).final_message().await;

    let cost = &message.usage.cost;
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(close(cost.input, 0.27), "input cost was {}", cost.input);
    assert!(close(cost.output, 1.10), "output cost was {}", cost.output);
    assert!(close(cost.total, 1.37), "total cost was {}", cost.total);
    assert_eq!(message.usage.total_tokens, 2_000_000);
}
