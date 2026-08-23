//! Seam: dynamic model discovery — `Models::refresh` / `Provider::refresh_models`
//! against wiremock. Layered merge: bundled catalog ← models.dev refresh
//! (override + append) ← /models probe (append-only, zero-means-unknown).

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use banshu_ai::{
    ApiKind, InMemoryModelsStore, Modality, Model, Models, ModelsStore, ModelsStoreEntry, Provider,
    RefreshOptions, RefreshOutcome,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
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
async fn offline_refresh_restores_the_stored_overlay_without_network() {
    unsafe { std::env::remove_var("DEEPSEEK_API_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let store = Arc::new(InMemoryModelsStore::new());
    let mut stored = Model::openai_completions("deepseek-stored");
    stored.provider = "deepseek".into();
    stored.name = "DeepSeek Stored".into();
    stored.base_url = "https://api.deepseek.com".into();
    stored.context_window = 64_000;
    store
        .set(ModelsStoreEntry {
            provider_id: "deepseek".into(),
            models: vec![stored],
            probed_model_ids: Vec::new(),
            checked_at: SystemTime::now(),
            etag: Some("\"stored-v1\"".into()),
            last_modified: None,
        })
        .await
        .unwrap();

    let models = Models::new()
        .with_models_store(store)
        .with_provider(Provider::deepseek());
    models
        .refresh_from_with(
            &format!("{}/api.json", server.uri()),
            &RefreshOptions {
                allow_network: false,
                ..RefreshOptions::default()
            },
        )
        .await;

    let restored = models
        .get("deepseek", "deepseek-stored")
        .expect("stored overlay restored");
    assert_eq!(restored.context_window, 64_000);
    server.verify().await;
}

#[tokio::test]
async fn freshness_skips_network_and_force_bypasses_it() {
    unsafe { std::env::remove_var("DEEPSEEK_API_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(MODELS_DEV_JSON))
        .expect(2)
        .mount(&server)
        .await;
    let store = Arc::new(InMemoryModelsStore::new());
    let catalog_url = format!("{}/api.json", server.uri());

    Models::new()
        .with_models_store(store.clone())
        .with_provider(Provider::deepseek())
        .refresh_from(&catalog_url)
        .await;
    let entry = store.get("deepseek").await.unwrap().unwrap();
    assert!(
        entry
            .models
            .iter()
            .any(|model| model.id == "deepseek-vnext")
    );
    assert!(
        entry
            .models
            .iter()
            .any(|model| model.id == "deepseek-reasoner")
    );

    let restored = Models::new()
        .with_models_store(store)
        .with_provider(Provider::deepseek());
    restored
        .refresh_from_with(
            &catalog_url,
            &RefreshOptions {
                max_age: Some(Duration::from_secs(3600)),
                ..RefreshOptions::default()
            },
        )
        .await;
    assert!(restored.get("deepseek", "deepseek-vnext").is_some());

    restored
        .refresh_from_with(
            &catalog_url,
            &RefreshOptions {
                force: true,
                max_age: Some(Duration::from_secs(3600)),
                ..RefreshOptions::default()
            },
        )
        .await;
    server.verify().await;
}

#[tokio::test]
async fn validators_make_304_refresh_fresh_without_clearing_overlay() {
    unsafe { std::env::remove_var("DEEPSEEK_API_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("etag", "\"catalog-v1\"")
                .insert_header("last-modified", "Wed, 21 Oct 2015 07:28:00 GMT")
                .set_body_string(MODELS_DEV_JSON),
        )
        .mount(&server)
        .await;
    let store = Arc::new(InMemoryModelsStore::new());
    let catalog_url = format!("{}/api.json", server.uri());
    let models = Models::new()
        .with_models_store(store.clone())
        .with_provider(Provider::deepseek());
    models.refresh_from(&catalog_url).await;
    let first_checked_at = store.get("deepseek").await.unwrap().unwrap().checked_at;

    tokio::time::sleep(Duration::from_millis(5)).await;
    server.reset().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(304))
        .expect(1)
        .mount(&server)
        .await;

    let report = models
        .refresh_from_with(
            &catalog_url,
            &RefreshOptions {
                force: true,
                ..RefreshOptions::default()
            },
        )
        .await;

    assert_eq!(report.entries[0].catalog, RefreshOutcome::Refreshed);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests[0].headers.get("if-none-match").unwrap(),
        "\"catalog-v1\""
    );
    assert_eq!(
        requests[0].headers.get("if-modified-since").unwrap(),
        "Wed, 21 Oct 2015 07:28:00 GMT"
    );
    assert!(models.get("deepseek", "deepseek-vnext").is_some());
    assert!(store.get("deepseek").await.unwrap().unwrap().checked_at > first_checked_at);
    server.verify().await;
}

#[tokio::test]
async fn available_validators_are_sent_before_filling_a_missing_provider_entry() {
    unsafe {
        std::env::remove_var("CACHED_CATALOG_KEY");
        std::env::remove_var("NEW_CATALOG_KEY");
    };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .and(header("if-none-match", "\"catalog-v1\""))
        .respond_with(ResponseTemplate::new(304))
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(MODELS_DEV_JSON))
        .with_priority(5)
        .expect(1)
        .mount(&server)
        .await;

    let store = Arc::new(InMemoryModelsStore::new());
    let mut cached = Model::openai_completions("cached-model");
    cached.provider = "cached".into();
    cached.base_url = server.uri();
    store
        .set(ModelsStoreEntry {
            provider_id: "cached".into(),
            models: vec![cached],
            probed_model_ids: Vec::new(),
            checked_at: SystemTime::now(),
            etag: Some("\"catalog-v1\"".into()),
            last_modified: None,
        })
        .await
        .unwrap();
    let models = Models::new()
        .with_models_store(store)
        .with_provider(
            Provider::openai_compatible("cached", "Cached", server.uri(), ["CACHED_CATALOG_KEY"])
                .with_models_dev_id("deepseek"),
        )
        .with_provider(
            Provider::openai_compatible("new", "New", server.uri(), ["NEW_CATALOG_KEY"])
                .with_models_dev_id("deepseek"),
        );

    models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    assert!(models.get("new", "deepseek-vnext").is_some());
    server.verify().await;
}

#[tokio::test]
async fn cancelled_refresh_keeps_the_restored_overlay() {
    unsafe { std::env::remove_var("DEEPSEEK_API_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(MODELS_DEV_JSON))
        .mount(&server)
        .await;
    let store = Arc::new(InMemoryModelsStore::new());
    let catalog_url = format!("{}/api.json", server.uri());
    let models = Models::new()
        .with_models_store(store)
        .with_provider(Provider::deepseek());
    models.refresh_from(&catalog_url).await;

    server.reset().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(1))
                .set_body_string("{}"),
        )
        .mount(&server)
        .await;
    let cancellation = banshu_ai::CancellationToken::new();
    let cancel = cancellation.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel.cancel();
    });
    models
        .refresh_from_with(
            &catalog_url,
            &RefreshOptions {
                force: true,
                cancellation: Some(cancellation),
                ..RefreshOptions::default()
            },
        )
        .await;

    assert!(models.get("deepseek", "deepseek-vnext").is_some());
}

#[tokio::test]
async fn cancelled_probe_keeps_the_restored_overlay() {
    unsafe { std::env::set_var("CANCEL_PROBE_KEY", "k") };

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let (headers_sent, wait_for_headers) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 2048];
        let bytes_read = socket.read(&mut request).await.unwrap();
        assert!(bytes_read > 0, "probe request should reach the test server");
        let body = br#"{"data":[{"id":"replacement"}]}"#;
        let headers = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(headers.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
        headers_sent.send(()).ok();
        tokio::time::sleep(Duration::from_secs(1)).await;
        socket.write_all(body).await.ok();
    });
    let store = Arc::new(InMemoryModelsStore::new());
    let mut stored = Model::openai_completions("last-known-good");
    stored.provider = "cancel-probe".into();
    stored.base_url = base_url.clone();
    store
        .set(ModelsStoreEntry {
            provider_id: "cancel-probe".into(),
            models: vec![stored],
            probed_model_ids: vec!["last-known-good".into()],
            checked_at: SystemTime::now(),
            etag: None,
            last_modified: None,
        })
        .await
        .unwrap();
    let models = Models::new()
        .with_models_store(store)
        .with_provider(Provider::openai_compatible(
            "cancel-probe",
            "Cancel Probe",
            base_url,
            ["CANCEL_PROBE_KEY"],
        ));
    let cancellation = banshu_ai::CancellationToken::new();
    let cancel = cancellation.clone();
    tokio::spawn(async move {
        wait_for_headers.await.unwrap();
        cancel.cancel();
    });

    models
        .refresh_from_with(
            "http://127.0.0.1:1/api.json",
            &RefreshOptions {
                force: true,
                cancellation: Some(cancellation),
                ..RefreshOptions::default()
            },
        )
        .await;

    assert!(models.get("cancel-probe", "last-known-good").is_some());
    assert!(models.get("cancel-probe", "replacement").is_none());
}

#[tokio::test]
async fn probe_cannot_overwrite_catalog_refresh_metadata() {
    unsafe { std::env::set_var("LAYERED_PROBE_KEY", "k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"layered":{"models":{"shared":{"name":"Catalog Shared","reasoning":false,"modalities":{"input":["text"]},"limit":{"context":12345,"output":2048},"cost":{"input":1.0,"output":2.0,"cache_read":0.0,"cache_write":0.0}}}}}"#,
        ))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"{"data":[{"id":"shared"},{"id":"probe-only"}]}"#),
        )
        .mount(&server)
        .await;

    let models = Models::new().with_provider(
        Provider::openai_compatible("layered", "Layered", server.uri(), ["LAYERED_PROBE_KEY"])
            .with_models_dev_id("layered"),
    );
    models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(
        models.get("layered", "shared").unwrap().context_window,
        12_345
    );
    assert_eq!(
        models.get("layered", "probe-only").unwrap().context_window,
        0
    );
}

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
    let chat = models
        .get("deepseek", "deepseek-chat")
        .expect("known model");
    assert_eq!(chat.cost.input, 9.9);
    assert_eq!(chat.context_window, 131_072);

    // A new id is appended with full metadata, stamped like a catalog model.
    let vnext = models.get("deepseek", "deepseek-vnext").expect("appended");
    assert!(vnext.reasoning.reasons());
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
    assert!(!found.reasoning.reasons());
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

    let provider = Provider::openai_compatible("solo", "Solo", server.uri(), ["SOLO_PROBE_KEY"])
        .with_models_dev_id("solo");
    let entry = provider
        .refresh_models_from(&format!("{}/api.json", server.uri()))
        .await;

    assert_eq!(entry.provider, "solo");
    assert_eq!(entry.catalog, RefreshOutcome::Refreshed);
    assert_eq!(entry.probe, RefreshOutcome::Refreshed);

    let models = provider.models();
    assert!(
        models
            .iter()
            .any(|m| m.id == "solo-pro" && m.reasoning.reasons())
    );
    assert!(models.iter().any(|m| m.id == "solo-x"));
}
