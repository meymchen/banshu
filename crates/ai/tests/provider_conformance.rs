//! Fixed, provider-wide promises in `docs/provider-conformance.md`.
//!
//! Image input and tool calling are intentionally absent: those capabilities
//! belong to individual catalog/discovery models and are covered by
//! `model_capabilities.rs`.

use std::sync::{Arc, Mutex};

use banshu_ai::{
    AnthropicReasoningFormat, ApiKind, InMemoryCredentialStore, MiniMaxRegion, OpenAiPromptCaching,
    OpenAiReasoningFormat, Provider, ReasoningEffort, ToolChoice, ToolChoiceSupport,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct Expected {
    build: fn() -> Provider,
    id: &'static str,
    name: &'static str,
    base_url: &'static str,
    api: ApiKind,
    api_key_env: &'static str,
    oauth: bool,
    openai_reasoning: OpenAiReasoningFormat,
    anthropic_reasoning: AnthropicReasoningFormat,
    efforts: Option<&'static [ReasoningEffort]>,
    tool_choice: ToolChoiceSupport,
    strict_tools: bool,
}

fn providers() -> [Expected; 7] {
    [
        Expected {
            build: Provider::deepseek,
            id: "deepseek",
            name: "DeepSeek",
            base_url: "https://api.deepseek.com",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "DEEPSEEK_API_KEY",
            oauth: false,
            openai_reasoning: OpenAiReasoningFormat::ThinkingToggle,
            anthropic_reasoning: AnthropicReasoningFormat::Unsupported,
            efforts: Some(&[
                ReasoningEffort::Off,
                ReasoningEffort::Low,
                ReasoningEffort::High,
                ReasoningEffort::Max,
            ]),
            tool_choice: ToolChoiceSupport {
                auto: true,
                none: true,
                required: false,
                named: false,
            },
            strict_tools: false,
        },
        Expected {
            build: Provider::zai,
            id: "zai",
            name: "Z.AI",
            base_url: "https://api.z.ai/api/coding/paas/v4",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "ZAI_API_KEY",
            oauth: false,
            openai_reasoning: OpenAiReasoningFormat::ThinkingToggleOnly,
            anthropic_reasoning: AnthropicReasoningFormat::Unsupported,
            efforts: None,
            tool_choice: ToolChoiceSupport {
                auto: true,
                ..ToolChoiceSupport::NONE
            },
            strict_tools: false,
        },
        Expected {
            build: Provider::moonshot,
            id: "moonshot",
            name: "Moonshot AI",
            base_url: "https://api.moonshot.ai/v1",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "MOONSHOT_API_KEY",
            oauth: false,
            openai_reasoning: OpenAiReasoningFormat::Unsupported,
            anthropic_reasoning: AnthropicReasoningFormat::Unsupported,
            efforts: Some(&[]),
            tool_choice: ToolChoiceSupport::ALL,
            strict_tools: true,
        },
        Expected {
            build: Provider::xiaomi,
            id: "xiaomi",
            name: "Xiaomi MiMo",
            base_url: "https://api.xiaomimimo.com/v1",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "XIAOMI_API_KEY",
            oauth: false,
            openai_reasoning: OpenAiReasoningFormat::ThinkingToggleOnly,
            anthropic_reasoning: AnthropicReasoningFormat::Unsupported,
            efforts: None,
            tool_choice: ToolChoiceSupport {
                auto: true,
                ..ToolChoiceSupport::NONE
            },
            strict_tools: true,
        },
        Expected {
            build: || Provider::kimi(Arc::new(InMemoryCredentialStore::new())),
            id: "kimi",
            name: "Kimi For Coding",
            base_url: "https://api.kimi.com/coding",
            api: ApiKind::AnthropicMessages,
            api_key_env: "KIMI_API_KEY",
            oauth: true,
            openai_reasoning: OpenAiReasoningFormat::Unsupported,
            anthropic_reasoning: AnthropicReasoningFormat::ThinkingToggle,
            efforts: None,
            tool_choice: ToolChoiceSupport::NONE,
            strict_tools: false,
        },
        Expected {
            build: || {
                Provider::minimax(
                    MiniMaxRegion::Global,
                    Arc::new(InMemoryCredentialStore::new()),
                )
            },
            id: "minimax",
            name: "MiniMax",
            base_url: "https://api.minimax.io/anthropic",
            api: ApiKind::AnthropicMessages,
            api_key_env: "MINIMAX_API_KEY",
            oauth: true,
            openai_reasoning: OpenAiReasoningFormat::Unsupported,
            anthropic_reasoning: AnthropicReasoningFormat::ThinkingAdaptive,
            efforts: None,
            tool_choice: ToolChoiceSupport::ALL,
            strict_tools: false,
        },
        Expected {
            build: || {
                Provider::minimax(MiniMaxRegion::Cn, Arc::new(InMemoryCredentialStore::new()))
            },
            id: "minimax-cn",
            name: "MiniMax CN",
            base_url: "https://api.minimaxi.com/anthropic",
            api: ApiKind::AnthropicMessages,
            api_key_env: "MINIMAX_API_KEY",
            oauth: true,
            openai_reasoning: OpenAiReasoningFormat::Unsupported,
            anthropic_reasoning: AnthropicReasoningFormat::ThinkingAdaptive,
            efforts: None,
            tool_choice: ToolChoiceSupport::ALL,
            strict_tools: false,
        },
    ]
}

#[test]
fn bundled_providers_match_the_frozen_matrix() {
    for expected in providers() {
        let provider = (expected.build)();
        let openai = provider.openai_compat();
        let anthropic = provider.anthropic_compat();

        assert_eq!(provider.id(), expected.id);
        assert_eq!(provider.name(), expected.name);
        assert_eq!(provider.base_url(), expected.base_url);
        assert_eq!(provider.api_kind(), expected.api);
        assert_eq!(openai.prompt_caching, OpenAiPromptCaching::Automatic);
        assert_eq!(openai.reasoning_format, expected.openai_reasoning);
        assert_eq!(anthropic.reasoning_format, expected.anthropic_reasoning);
        assert_eq!(
            match expected.api {
                ApiKind::OpenAiCompletions => openai.reasoning_efforts,
                ApiKind::AnthropicMessages => anthropic.reasoning_efforts,
                _ => unreachable!("all current protocols are covered"),
            },
            expected.efforts,
        );
        let actual_tool_choice = match expected.api {
            ApiKind::OpenAiCompletions => openai.tool_choice,
            ApiKind::AnthropicMessages => anthropic.tool_choice,
            _ => unreachable!("all current protocols are covered"),
        };
        assert_eq!(actual_tool_choice, expected.tool_choice);
        assert_eq!(
            actual_tool_choice.supports(&ToolChoice::Named("tool".into())),
            expected.tool_choice.named,
        );
        let actual_strict = match expected.api {
            ApiKind::OpenAiCompletions => openai.strict_tool_schemas,
            ApiKind::AnthropicMessages => anthropic.strict_tool_schemas,
            _ => unreachable!("all current protocols are covered"),
        };
        assert_eq!(actual_strict, expected.strict_tools);
        assert_eq!(provider.oauth_session().is_some(), expected.oauth);
    }
}

#[test]
fn bundled_auth_accepts_each_documented_api_key_environment_variable() {
    let _guard = ENV_LOCK.lock().expect("environment lock");

    for expected in providers() {
        let previous = std::env::var_os(expected.api_key_env);
        // SAFETY: this test serializes its own environment changes, uses one
        // variable at a time, and restores the process environment below.
        unsafe { std::env::set_var(expected.api_key_env, "conformance-test-key") };
        assert!(
            (expected.build)().is_available(),
            "{} did not read {}",
            expected.id,
            expected.api_key_env,
        );
        match previous {
            Some(value) => {
                // SAFETY: guarded and restored for the same reason as above.
                unsafe { std::env::set_var(expected.api_key_env, value) };
            }
            None => {
                // SAFETY: guarded and restored for the same reason as above.
                unsafe { std::env::remove_var(expected.api_key_env) };
            }
        }
    }
}
