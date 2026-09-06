//! Regional and subscription provider identities and model-specific requests.
use banshu_ai::{ApiKind, Context, Provider, ReasoningEffort, StopReason, StreamOptions};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn regional_providers_have_separate_catalogs_endpoints_and_credentials() {
    type ProviderCase = (fn() -> Provider, &'static str, &'static str, &'static str);
    let cases: [ProviderCase; 8] = [
        (
            Provider::zai_coding_cn,
            "zai-coding-cn",
            "https://open.bigmodel.cn/api/coding/paas/v4",
            "ZAI_CODING_CN_API_KEY",
        ),
        (
            Provider::moonshot_cn,
            "moonshot-cn",
            "https://api.moonshot.cn/v1",
            "MOONSHOT_API_KEY",
        ),
        (
            Provider::xiaomi_token_plan_cn,
            "xiaomi-token-plan-cn",
            "https://token-plan-cn.xiaomimimo.com/v1",
            "XIAOMI_TOKEN_PLAN_CN_API_KEY",
        ),
        (
            Provider::xiaomi_token_plan_ams,
            "xiaomi-token-plan-ams",
            "https://token-plan-ams.xiaomimimo.com/v1",
            "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
        ),
        (
            Provider::xiaomi_token_plan_sgp,
            "xiaomi-token-plan-sgp",
            "https://token-plan-sgp.xiaomimimo.com/v1",
            "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
        ),
        (
            Provider::qwen_token_plan,
            "qwen-token-plan",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            "QWEN_TOKEN_PLAN_API_KEY",
        ),
        (
            Provider::qwen_token_plan_cn,
            "qwen-token-plan-cn",
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "QWEN_TOKEN_PLAN_CN_API_KEY",
        ),
        (
            Provider::qwen_token_plan_individual,
            "qwen-token-plan-individual",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            "QWEN_TOKEN_PLAN_API_KEY",
        ),
    ];
    for (build, id, url, key) in cases {
        let provider = build();
        assert_eq!(provider.id(), id);
        assert_eq!(provider.base_url(), url);
        assert_eq!(provider.api_kind(), ApiKind::OpenAiCompletions);
        let models = provider.models();
        assert!(!models.is_empty(), "{id}");
        assert!(models.iter().all(|m| m.provider == id && m.base_url == url));
        assert!(models.iter().all(|m| {
            ![
                "mimo-v2-flash",
                "mimo-v2-pro",
                "mimo-v2-omni",
                "qwen3.8-max-preview",
            ]
            .contains(&m.id.as_str())
        }));
        let previous = std::env::var_os(key);
        // SAFETY: this is the only environment-mutating test in this binary;
        // each value is restored immediately after the synchronous check.
        unsafe { std::env::set_var(key, "test-key") };
        let available = provider.is_available();
        unsafe {
            match previous {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
        assert!(available, "{id} must use {key}");
    }
    assert!(
        Provider::zai_coding_cn()
            .models()
            .iter()
            .any(|m| m.id == "glm-4.6v")
    );
    assert!(Provider::zai().models().iter().all(|m| m.id != "glm-4.6v"));
    let individual = Provider::qwen_token_plan_individual().models();
    assert_eq!(individual.len(), 9);
    assert!(individual.iter().any(|m| m.id == "qwen3.8-flash"));
    assert!(individual.iter().all(|m| m.id != "MiniMax-M2.5"));
}

#[tokio::test]
async fn qwen_token_plan_sends_toggle_and_attested_effort() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/event-stream")
            .set_body_string("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
        .mount(&server).await;
    let provider = Provider::qwen_token_plan();
    let model = provider
        .models()
        .into_iter()
        .find(|m| m.id == "qwen3.8-flash")
        .unwrap()
        .with_base_url(server.uri());
    for effort in [ReasoningEffort::XHigh, ReasoningEffort::Off] {
        let options = StreamOptions {
            api_key: Some("test".into()),
            max_tokens: Some(1024),
            reasoning: Some(effort.into()),
            ..Default::default()
        };
        let result = provider
            .stream(&model, &Context::new().user("hi"), &options)
            .finish()
            .await;
        assert_eq!(result.stop_reason, StopReason::Stop, "{result:?}");
    }
    let requests = server.received_requests().await.unwrap();
    let enabled: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(enabled["enable_thinking"], true);
    assert_eq!(enabled["reasoning_effort"], "xhigh");
    let disabled: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(disabled["enable_thinking"], false);
    assert!(disabled.get("reasoning_effort").is_none());
}

#[test]
fn qwen_individual_refresh_uses_the_same_allowlist_as_its_bundled_catalog() {
    let data = json!({"alibaba-token-plan":{"models":{
        "qwen3.8-flash":{"tool_call":true}, "MiniMax-M2.5":{"tool_call":true}, "qwen3.8-max-preview":{"tool_call":true}
    }}});
    let models =
        banshu_ai::models_dev::models_from_api_json(&data, "qwen-token-plan-individual").unwrap();
    assert_eq!(
        models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["qwen3.8-flash"]
    );
}

#[tokio::test]
async fn individual_probe_cannot_add_models_outside_the_plan() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/catalog"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"alibaba-token-plan":{"models":{}}})),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"data":[{"id":"MiniMax-M2.5"}]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    // SAFETY: this unique variable is only used by this local-server test.
    unsafe { std::env::set_var("BANSHU_INDIVIDUAL_PROBE", "test") };
    let provider = Provider::openai_compatible(
        "qwen-token-plan-individual",
        "Individual",
        server.uri(),
        ["BANSHU_INDIVIDUAL_PROBE"],
    )
    .with_models_dev_id("qwen-token-plan-individual");
    provider
        .refresh_models_from(&format!("{}/catalog", server.uri()))
        .await;
    unsafe { std::env::remove_var("BANSHU_INDIVIDUAL_PROBE") };
    assert!(provider.models().iter().all(|m| m.id != "MiniMax-M2.5"));
}
