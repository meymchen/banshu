//! Seam 4: model-catalog lookup from the bundled models.dev snapshot.
//!
//! Asserts structural properties (stable ids present, metadata stamped from the
//! owning provider) rather than exact prices/limits, so a catalog regeneration
//! doesn't churn the test.

use banshu_ai::{ApiKind, Modality, Provider};

#[test]
fn deepseek_catalog_loads_with_provider_metadata() {
    let provider = Provider::deepseek();
    let models = provider.models();

    let chat = models
        .iter()
        .find(|m| m.id == "deepseek-chat")
        .expect("deepseek-chat should be in the bundled catalog");

    // Identity comes from the catalog; provider/base_url/api are stamped on.
    assert_eq!(chat.provider, "deepseek");
    assert_eq!(chat.base_url, provider.base_url());
    assert_eq!(chat.api, ApiKind::OpenAiCompletions);
    assert!(chat.input.contains(&Modality::Text));
    assert!(chat.context_window > 0, "context window should be populated");
    assert!(chat.cost.input > 0.0, "input cost should be populated");
}

#[test]
fn anthropic_vendor_catalog_is_tagged_anthropic() {
    let provider = Provider::kimi();
    let models = provider.models();

    assert!(!models.is_empty(), "kimi catalog should not be empty");
    assert!(models.iter().all(|m| m.api == ApiKind::AnthropicMessages));
    assert!(models.iter().all(|m| m.base_url == provider.base_url()));
    assert!(models.iter().all(|m| m.provider == "kimi"));
}
