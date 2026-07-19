//! Seam: dynamic model discovery — `Models::refresh` / `Provider::refresh_models`
//! against wiremock. Layered merge: bundled catalog ← models.dev refresh
//! (override + append) ← /models probe (append-only, zero-means-unknown).

use banshu_ai::{ApiKind, Modality, Models, Provider, RefreshOutcome};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A models.dev `api.json` excerpt: one known deepseek id with changed
/// metadata, one id the bundled catalog doesn't have.
const MODELS_DEV_JSON: &str = r#"{
  "deepseek": {
    "models": {
      "deepseek-chat": {
        "name": "DeepSeek Chat",
        "reasoning": false,
        "modalities": { "input": ["text"] },
        "limit": { "context": 131072, "output": 8192 },
        "cost": { "input": 9.9, "output": 19.8, "cache_read": 0.5, "cache_write": 0.0 }
      },
      "deepseek-vnext": {
        "name": "DeepSeek VNext",
        "reasoning": true,
        "modalities": { "input": ["text"] },
        "limit": { "context": 262144, "output": 16384 },
        "cost": { "input": 1.0, "output": 2.0, "cache_read": 0.1, "cache_write": 0.0 }
      }
    }
  }
}"#;

#[tokio::test]
async fn refresh_overrides_and_appends_models_dev_entries() {
    // SAFETY: no key → the vendor probe is skipped, so this test never
    // touches the real DeepSeek endpoint.
    unsafe { std::env::remove_var("DEEPSEEK_API_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(MODELS_DEV_JSON))
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::deepseek());
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    let entry = report
        .entries
        .iter()
        .find(|e| e.provider == "deepseek")
        .expect("deepseek report entry");
    assert_eq!(entry.catalog, RefreshOutcome::Refreshed);
    assert_eq!(entry.probe, RefreshOutcome::Skipped);

    // Same-id bundled entry is overridden by the refreshed metadata.
    let chat = models.get("deepseek", "deepseek-chat").expect("known model");
    assert_eq!(chat.cost.input, 9.9);
    assert_eq!(chat.context_window, 131_072);

    // A new id is appended with full metadata, stamped like a catalog model.
    let vnext = models.get("deepseek", "deepseek-vnext").expect("appended");
    assert!(vnext.reasoning);
    assert_eq!(vnext.provider, "deepseek");
    assert_eq!(vnext.base_url, "https://api.deepseek.com");

    // Bundled entries absent from the refresh are kept, not removed.
    assert!(models.get("deepseek", "deepseek-reasoner").is_some());
}

#[tokio::test]
async fn openai_probe_appends_unknown_ids_as_zero_metadata_models() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("ACME_PROBE_KEY", "probe-k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer probe-k"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"object":"list","data":[{"id":"acme-large"},{"id":"acme-mini"}]}"#,
        ))
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::openai_compatible(
        "acme",
        "Acme",
        server.uri(),
        ["ACME_PROBE_KEY"],
    ));
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    let entry = &report.entries[0];
    // No models.dev id → the catalog layer is skipped without a fetch.
    assert_eq!(entry.catalog, RefreshOutcome::Skipped);
    assert_eq!(entry.probe, RefreshOutcome::Refreshed);

    let found = models.get("acme", "acme-large").expect("probed model");
    assert_eq!(found.name, "acme-large");
    assert_eq!(found.api, ApiKind::OpenAiCompletions);
    assert_eq!(found.base_url, server.uri());
    assert_eq!(found.input, vec![Modality::Text]);
    // Zero-means-unknown: nothing is guessed for a bare id.
    assert!(!found.reasoning);
    assert_eq!(found.cost.input, 0.0);
    assert_eq!(found.context_window, 0);
    assert_eq!(found.max_tokens, 0);
    assert!(models.get("acme", "acme-mini").is_some());
}

#[tokio::test]
async fn anthropic_probe_lists_v1_models_with_api_key_header() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("UMBRA_PROBE_KEY", "umbra-k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("x-api-key", "umbra-k"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"data":[{"id":"umbra-opus","display_name":"Umbra Opus","type":"model"}],"has_more":false}"#,
        ))
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::anthropic_compatible(
        "umbra",
        "Umbra",
        server.uri(),
        ["UMBRA_PROBE_KEY"],
    ));
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(report.entries[0].probe, RefreshOutcome::Refreshed);
    let found = models.get("umbra", "umbra-opus").expect("probed model");
    assert_eq!(found.name, "Umbra Opus");
    assert_eq!(found.api, ApiKind::AnthropicMessages);
}

#[tokio::test]
async fn probe_404_reports_endpoint_unsupported() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("NOLIST_PROBE_KEY", "k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::openai_compatible(
        "nolist",
        "NoList",
        server.uri(),
        ["NOLIST_PROBE_KEY"],
    ));
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(report.entries[0].probe, RefreshOutcome::EndpointUnsupported);
}

#[tokio::test]
async fn probe_without_api_key_is_skipped_and_sends_nothing() {
    // SAFETY: a unique env var name keeps this from racing other tests.
    unsafe { std::env::remove_var("KEYLESS_PROBE_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"data":[]}"#))
        .expect(0)
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::openai_compatible(
        "keyless",
        "Keyless",
        server.uri(),
        ["KEYLESS_PROBE_KEY"],
    ));
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(report.entries[0].probe, RefreshOutcome::Skipped);
    server.verify().await;
}

#[tokio::test]
async fn failed_models_dev_fetch_keeps_serving_the_bundled_catalog() {
    // SAFETY: no key → the vendor probe is skipped (no real-endpoint traffic).
    unsafe { std::env::remove_var("DEEPSEEK_API_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::deepseek());
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    assert!(matches!(
        report.entries[0].catalog,
        RefreshOutcome::Failed(_)
    ));
    // The bundled catalog is untouched by the failure.
    assert!(models.get("deepseek", "deepseek-chat").is_some());
}

#[tokio::test]
async fn failed_refresh_keeps_previously_discovered_models() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("FLAKY_PROBE_KEY", "k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(r#"{"data":[{"id":"flaky-one"}]}"#),
        )
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::openai_compatible(
        "flaky",
        "Flaky",
        server.uri(),
        ["FLAKY_PROBE_KEY"],
    ));
    let catalog_url = format!("{}/api.json", server.uri());
    models.refresh_from(&catalog_url).await;
    assert!(models.get("flaky", "flaky-one").is_some());

    // The endpoint starts failing: the overlay from the last good refresh
    // stays in place.
    server.reset().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let report = models.refresh_from(&catalog_url).await;
    assert!(matches!(report.entries[0].probe, RefreshOutcome::Failed(_)));
    assert!(models.get("flaky", "flaky-one").is_some());
}

#[tokio::test]
async fn provider_level_refresh_works_without_a_registry() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("SOLO_PROBE_KEY", "k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"solo":{"models":{"solo-pro":{"name":"Solo Pro","reasoning":true,"modalities":{"input":["text"]},"limit":{"context":32768,"output":4096},"cost":{"input":1.0,"output":2.0,"cache_read":0.1,"cache_write":0.0}}}}}"#,
        ))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"data":[{"id":"solo-x"}]}"#))
        .mount(&server)
        .await;

    let provider =
        Provider::openai_compatible("solo", "Solo", server.uri(), ["SOLO_PROBE_KEY"])
            .with_models_dev_id("solo");
    let entry = provider
        .refresh_models_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(entry.provider, "solo");
    assert_eq!(entry.catalog, RefreshOutcome::Refreshed);
    assert_eq!(entry.probe, RefreshOutcome::Refreshed);

    let models = provider.models();
    assert!(models.iter().any(|m| m.id == "solo-pro" && m.reasoning));
    assert!(models.iter().any(|m| m.id == "solo-x"));
}
