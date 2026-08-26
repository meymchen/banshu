//! Fixed, provider-wide promises in `docs/provider-conformance.md`.
//!
//! Image input and tool calling are intentionally absent: those capabilities
//! belong to individual catalog/discovery models and are covered by
//! `model_capabilities.rs`.

use std::sync::{Arc, Mutex};

use banshu_ai::{
    AnthropicCacheRetention, AnthropicCompat, AnthropicReasoningFormat, AnthropicTemperature,
    ApiKind, InMemoryCredentialStore, MiniMaxRegion, OpenAiCacheRetention, OpenAiCompat,
    OpenAiOutputTokenField, OpenAiReasoningFormat, OpenAiSessionAffinity, OpenAiStreamTermination,
    Provider, ReasoningEffort, ToolChoice, ToolChoiceSupport,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// The complete undeclared OpenAI compatibility value. This is deliberately
/// not `OpenAiCompat::default()`: a new field must break this fixture until the
/// frozen matrix makes a decision about it.
const OPENAI_DEFAULT: OpenAiCompat = OpenAiCompat {
    session_affinity: OpenAiSessionAffinity::None,
    cache_retention: OpenAiCacheRetention::Short,
    requires_reasoning_content_on_assistant_messages: false,
    reasoning_format: OpenAiReasoningFormat::Unsupported,
    tool_choice: ToolChoiceSupport::NONE,
    strict_tool_schemas: false,
    reasoning_efforts: None,
    streamed_usage: true,
    output_token_field: OpenAiOutputTokenField::MaxTokens,
    stream_termination: OpenAiStreamTermination::Strict,
    tool_result_names: false,
    empty_assistant_separator: false,
};

/// The complete undeclared Anthropic compatibility value, explicit for the
/// same compile-time freeze as [`OPENAI_DEFAULT`].
const ANTHROPIC_DEFAULT: AnthropicCompat = AnthropicCompat {
    allow_empty_signature: false,
    send_session_affinity_headers: false,
    cache_retention: AnthropicCacheRetention::Short,
    tool_cache_control: false,
    reasoning_format: AnthropicReasoningFormat::Unsupported,
    temperature: AnthropicTemperature::Unsupported,
    tool_choice: ToolChoiceSupport::NONE,
    strict_tool_schemas: false,
    reasoning_efforts: None,
};

/// Both MiniMax regions expose the same Anthropic wire contract.
const MINIMAX_ANTHROPIC: AnthropicCompat = AnthropicCompat {
    cache_retention: AnthropicCacheRetention::Long,
    tool_cache_control: true,
    reasoning_format: AnthropicReasoningFormat::ThinkingAdaptive,
    temperature: AnthropicTemperature::WithReasoning,
    tool_choice: ToolChoiceSupport::ALL,
    ..ANTHROPIC_DEFAULT
};

struct ProviderExpectation {
    build: fn() -> Provider,
    id: &'static str,
    name: &'static str,
    base_url: &'static str,
    api: ApiKind,
    api_key_env: &'static str,
    oauth: bool,
    openai: OpenAiCompat,
    anthropic: AnthropicCompat,
}

fn providers() -> [ProviderExpectation; 7] {
    [
        ProviderExpectation {
            build: Provider::deepseek,
            id: "deepseek",
            name: "DeepSeek",
            base_url: "https://api.deepseek.com",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "DEEPSEEK_API_KEY",
            oauth: false,
            openai: OpenAiCompat {
                requires_reasoning_content_on_assistant_messages: true,
                reasoning_format: OpenAiReasoningFormat::ThinkingToggle,
                reasoning_efforts: Some(&[
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::High,
                    ReasoningEffort::Max,
                ]),
                tool_choice: ToolChoiceSupport {
                    auto: true,
                    none: true,
                    ..ToolChoiceSupport::NONE
                },
                ..OPENAI_DEFAULT
            },
            anthropic: ANTHROPIC_DEFAULT,
        },
        ProviderExpectation {
            build: Provider::zai,
            id: "zai",
            name: "Z.AI",
            base_url: "https://api.z.ai/api/coding/paas/v4",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "ZAI_API_KEY",
            oauth: false,
            openai: OpenAiCompat {
                reasoning_format: OpenAiReasoningFormat::ThinkingToggleOnly,
                tool_choice: ToolChoiceSupport {
                    auto: true,
                    ..ToolChoiceSupport::NONE
                },
                ..OPENAI_DEFAULT
            },
            anthropic: ANTHROPIC_DEFAULT,
        },
        ProviderExpectation {
            build: Provider::moonshot,
            id: "moonshot",
            name: "Moonshot AI",
            base_url: "https://api.moonshot.ai/v1",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "MOONSHOT_API_KEY",
            oauth: false,
            openai: OpenAiCompat {
                reasoning_efforts: Some(&[]),
                tool_choice: ToolChoiceSupport::ALL,
                strict_tool_schemas: true,
                ..OPENAI_DEFAULT
            },
            anthropic: ANTHROPIC_DEFAULT,
        },
        ProviderExpectation {
            build: Provider::xiaomi,
            id: "xiaomi",
            name: "Xiaomi MiMo",
            base_url: "https://api.xiaomimimo.com/v1",
            api: ApiKind::OpenAiCompletions,
            api_key_env: "XIAOMI_API_KEY",
            oauth: false,
            openai: OpenAiCompat {
                reasoning_format: OpenAiReasoningFormat::ThinkingToggleOnly,
                tool_choice: ToolChoiceSupport {
                    auto: true,
                    ..ToolChoiceSupport::NONE
                },
                strict_tool_schemas: true,
                ..OPENAI_DEFAULT
            },
            anthropic: ANTHROPIC_DEFAULT,
        },
        ProviderExpectation {
            build: || Provider::kimi(Arc::new(InMemoryCredentialStore::new())),
            id: "kimi",
            name: "Kimi For Coding",
            base_url: "https://api.kimi.com/coding",
            api: ApiKind::AnthropicMessages,
            api_key_env: "KIMI_API_KEY",
            oauth: true,
            openai: OPENAI_DEFAULT,
            anthropic: AnthropicCompat {
                cache_retention: AnthropicCacheRetention::Long,
                tool_cache_control: true,
                reasoning_format: AnthropicReasoningFormat::ThinkingToggle,
                ..ANTHROPIC_DEFAULT
            },
        },
        ProviderExpectation {
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
            openai: OPENAI_DEFAULT,
            anthropic: MINIMAX_ANTHROPIC,
        },
        ProviderExpectation {
            build: || {
                Provider::minimax(MiniMaxRegion::Cn, Arc::new(InMemoryCredentialStore::new()))
            },
            id: "minimax-cn",
            name: "MiniMax CN",
            base_url: "https://api.minimaxi.com/anthropic",
            api: ApiKind::AnthropicMessages,
            api_key_env: "MINIMAX_API_KEY",
            oauth: true,
            openai: OPENAI_DEFAULT,
            anthropic: MINIMAX_ANTHROPIC,
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
        assert_eq!(openai, expected.openai, "{}: OpenAI compat", expected.id);
        assert_eq!(
            anthropic, expected.anthropic,
            "{}: Anthropic compat",
            expected.id,
        );
        let (actual_efforts, actual_tool_choice, actual_strict) = match expected.api {
            ApiKind::OpenAiCompletions => (
                openai.reasoning_efforts,
                openai.tool_choice,
                openai.strict_tool_schemas,
            ),
            ApiKind::AnthropicMessages => (
                anthropic.reasoning_efforts,
                anthropic.tool_choice,
                anthropic.strict_tool_schemas,
            ),
            _ => unreachable!("all current protocols are covered"),
        };
        let (expected_efforts, expected_tool_choice, expected_strict) = match expected.api {
            ApiKind::OpenAiCompletions => (
                expected.openai.reasoning_efforts,
                expected.openai.tool_choice,
                expected.openai.strict_tool_schemas,
            ),
            ApiKind::AnthropicMessages => (
                expected.anthropic.reasoning_efforts,
                expected.anthropic.tool_choice,
                expected.anthropic.strict_tool_schemas,
            ),
            _ => unreachable!("all current protocols are covered"),
        };
        assert_eq!(actual_efforts, expected_efforts);
        assert_eq!(actual_tool_choice, expected_tool_choice);
        assert_eq!(
            actual_tool_choice.supports(&ToolChoice::Named("tool".into())),
            expected_tool_choice.named,
        );
        assert_eq!(actual_strict, expected_strict);
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
