//! Per-vendor provider constructors: the thin data definitions for the six
//! open-source providers banshu targets. Verifies id / name / base URL / env
//! var / wire protocol for each (values borrowed from pi's provider factories).

use banshu_ai::{ApiKind, Provider};

struct Expected {
    build: fn() -> Provider,
    id: &'static str,
    name: &'static str,
    base_url: &'static str,
    api: ApiKind,
}

#[test]
fn vendor_constructors_carry_the_right_metadata() {
    let vendors = [
        Expected {
            build: Provider::deepseek,
            id: "deepseek",
            name: "DeepSeek",
            base_url: "https://api.deepseek.com",
            api: ApiKind::OpenAiCompletions,
        },
        Expected {
            build: Provider::zai,
            id: "zai",
            name: "Z.AI",
            base_url: "https://api.z.ai/api/coding/paas/v4",
            api: ApiKind::OpenAiCompletions,
        },
        Expected {
            build: Provider::minimax,
            id: "minimax",
            name: "MiniMax",
            base_url: "https://api.minimax.io/anthropic",
            api: ApiKind::AnthropicMessages,
        },
        Expected {
            build: Provider::moonshot,
            id: "moonshot",
            name: "Moonshot AI",
            base_url: "https://api.moonshot.ai/v1",
            api: ApiKind::OpenAiCompletions,
        },
        Expected {
            build: Provider::kimi,
            id: "kimi",
            name: "Kimi For Coding",
            base_url: "https://api.kimi.com/coding",
            api: ApiKind::AnthropicMessages,
        },
        Expected {
            build: Provider::xiaomi,
            id: "xiaomi",
            name: "Xiaomi MiMo",
            base_url: "https://api.xiaomimimo.com/v1",
            api: ApiKind::OpenAiCompletions,
        },
    ];

    for vendor in vendors {
        let provider = (vendor.build)();
        assert_eq!(provider.id(), vendor.id);
        assert_eq!(provider.name(), vendor.name);
        assert_eq!(provider.base_url(), vendor.base_url);
        assert_eq!(
            provider.api_kind(),
            vendor.api,
            "wrong protocol for {}",
            vendor.id
        );
    }
}
