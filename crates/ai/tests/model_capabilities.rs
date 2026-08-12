//! Seam: honest tool-calling capability signaling — models.dev `tool_call`
//! mapping, the shared catalog filter, probe-model `Unknown`, and the
//! `agent_models()` / `models()` split.

use banshu_ai::models_dev::{capability_from_tool_call, models_from_api_json};
use banshu_ai::{CapabilitySupport, Models, Provider, RefreshOutcome};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn tool_call_maps_true_false_and_missing() {
    assert_eq!(
        capability_from_tool_call(Some(true)),
        CapabilitySupport::Supported
    );
    assert_eq!(
        capability_from_tool_call(Some(false)),
        CapabilitySupport::Unsupported
    );
    assert_eq!(capability_from_tool_call(None), CapabilitySupport::Unknown);
}

/// A models.dev excerpt exercising the mapping: one tool-calling text model,
/// one with `tool_call: false`, one without the field, and one tool-calling
/// model with non-text output.
const CAPABILITIES_JSON: &str = r#"{
  "acme": {
    "models": {
      "acme-agent": {
        "name": "Acme Agent",
        "tool_call": true,
        "modalities": { "input": ["text"], "output": ["text"] },
        "limit": { "context": 32768, "output": 4096 },
        "cost": { "input": 1.0, "output": 2.0, "cache_read": 0.1, "cache_write": 0.0 }
      },
      "acme-plain": {
        "name": "Acme Plain",
        "tool_call": false,
        "modalities": { "input": ["text"], "output": ["text"] }
      },
      "acme-unstated": {
        "name": "Acme Unstated",
        "modalities": { "input": ["text"], "output": ["text"] }
      },
      "acme-artist": {
        "name": "Acme Artist",
        "tool_call": true,
        "modalities": { "input": ["text"], "output": ["image"] }
      }
    }
  }
}"#;

#[test]
fn shared_parse_maps_capabilities_for_every_entry() {
    let data = serde_json::from_str(CAPABILITIES_JSON).unwrap();
    let models = models_from_api_json(&data, "acme").expect("acme models");

    let capability_of = |id: &str| {
        models
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("{id} parsed"))
            .tool_calling
    };
    assert_eq!(capability_of("acme-agent"), CapabilitySupport::Supported);
    assert_eq!(capability_of("acme-plain"), CapabilitySupport::Unsupported);
    assert_eq!(capability_of("acme-unstated"), CapabilitySupport::Unknown);
}

#[test]
fn catalog_filter_keeps_only_tool_calling_text_in_text_out() {
    let data = serde_json::from_str(CAPABILITIES_JSON).unwrap();
    let models = models_from_api_json(&data, "acme").expect("acme models");

    let kept: Vec<&str> = models
        .iter()
        .filter(|m| m.is_tool_calling_text_model())
        .map(|m| m.id.as_str())
        .collect();
    // tool_call false / missing / non-text output are all excluded.
    assert_eq!(kept, ["acme-agent"]);
}

#[tokio::test]
async fn refresh_maps_capabilities_and_agent_models_filters_to_supported() {
    // SAFETY: no key → the vendor probe is skipped, so this test never
    // touches a real endpoint.
    unsafe { std::env::remove_var("ACME_CAP_KEY") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(CAPABILITIES_JSON))
        .mount(&server)
        .await;

    let models = Models::new().with_provider(
        Provider::openai_compatible("acme", "Acme", server.uri(), ["ACME_CAP_KEY"])
            .with_models_dev_id("acme"),
    );
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;
    assert_eq!(report.entries[0].catalog, RefreshOutcome::Refreshed);

    // models() keeps full coverage: every refreshed entry is selectable,
    // whatever its attestation.
    let capability_of = |id: &str| {
        models
            .get("acme", id)
            .unwrap_or_else(|| panic!("{id} served"))
            .capabilities
            .tool_calling
    };
    assert_eq!(capability_of("acme-agent"), CapabilitySupport::Supported);
    assert_eq!(capability_of("acme-plain"), CapabilitySupport::Unsupported);
    assert_eq!(capability_of("acme-unstated"), CapabilitySupport::Unknown);
    assert_eq!(capability_of("acme-artist"), CapabilitySupport::Supported);

    // agent_models() gates strictly on the tool-calling attestation.
    let agent_ids: Vec<String> = models.agent_models().into_iter().map(|m| m.id).collect();
    assert_eq!(agent_ids, ["acme-agent", "acme-artist"]);
}

#[tokio::test]
async fn probe_models_report_unknown_but_stay_selectable() {
    // SAFETY: a unique env var name keeps this key from racing other tests.
    unsafe { std::env::set_var("ACME_CAP_PROBE_KEY", "probe-k") };

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer probe-k"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"{"object":"list","data":[{"id":"acme-mystery"}]}"#),
        )
        .mount(&server)
        .await;

    let models = Models::new().with_provider(Provider::openai_compatible(
        "acme-probe",
        "Acme",
        server.uri(),
        ["ACME_CAP_PROBE_KEY"],
    ));
    let report = models
        .refresh_from(&format!("{}/api.json", server.uri()))
        .await;
    assert_eq!(report.entries[0].probe, RefreshOutcome::Refreshed);

    // Unknown is not presented as supported, but the model stays selectable.
    let probed = models
        .get("acme-probe", "acme-mystery")
        .expect("probed model");
    assert_eq!(probed.capabilities.tool_calling, CapabilitySupport::Unknown);
    assert!(models.models().iter().any(|m| m.id == "acme-mystery"));
    assert!(!models.agent_models().iter().any(|m| m.id == "acme-mystery"));
}

#[test]
fn bundled_catalog_models_are_attested_tool_calling() {
    // Every bundled entry passed the generator's tool-calling text filter, so
    // the catalog is exactly the registry's agent pool before any refresh.
    let models = Models::new()
        .with_provider(Provider::deepseek())
        .with_provider(Provider::kimi(std::sync::Arc::new(
            banshu_ai::InMemoryCredentialStore::new(),
        )));

    let all = models.models();
    assert!(!all.is_empty(), "bundled catalogs should not be empty");
    let agents = models.agent_models();
    assert_eq!(
        all.len(),
        agents.len(),
        "every bundled catalog model should report Supported tool_calling"
    );
    assert!(
        agents
            .iter()
            .all(|m| m.capabilities.tool_calling == CapabilitySupport::Supported)
    );
    // The generator's text-in guarantee is recorded in the catalog itself.
    assert!(
        all.iter()
            .all(|m| m.input.contains(&banshu_ai::Modality::Text)),
        "every bundled catalog model should accept text input"
    );
}
