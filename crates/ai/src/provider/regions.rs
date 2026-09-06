//! Regional and subscription API endpoints and authentication settings.
use super::{OpenAiCompat, OpenAiReasoningFormat, Provider};
use crate::Auth;

impl Provider {
    fn regional(mut self, id: &str, name: &str, url: &str, key: &str, source: &str) -> Self {
        self.id = id.into();
        self.name = name.into();
        self.base_url = url.into();
        self.auth = Auth::api_key_env([key]);
        self.models_dev_id = Some(source.into());
        self
    }

    /// Z.AI's China Coding Plan, authenticated with `ZAI_CODING_CN_API_KEY`.
    pub fn zai_coding_cn() -> Self {
        Self::zai().regional(
            "zai-coding-cn",
            "Z.AI Coding CN",
            "https://open.bigmodel.cn/api/coding/paas/v4",
            "ZAI_CODING_CN_API_KEY",
            "zhipuai-coding-plan",
        )
    }

    /// Moonshot's China API, authenticated with `MOONSHOT_API_KEY`.
    pub fn moonshot_cn() -> Self {
        Self::moonshot().regional(
            "moonshot-cn",
            "Moonshot AI CN",
            "https://api.moonshot.cn/v1",
            "MOONSHOT_API_KEY",
            "moonshotai-cn",
        )
    }

    /// Xiaomi's China Token Plan, authenticated with `XIAOMI_TOKEN_PLAN_CN_API_KEY`.
    pub fn xiaomi_token_plan_cn() -> Self {
        Self::xiaomi().regional(
            "xiaomi-token-plan-cn",
            "Xiaomi Token Plan CN",
            "https://token-plan-cn.xiaomimimo.com/v1",
            "XIAOMI_TOKEN_PLAN_CN_API_KEY",
            "xiaomi-token-plan-cn",
        )
    }

    /// Xiaomi's Amsterdam Token Plan, authenticated with `XIAOMI_TOKEN_PLAN_AMS_API_KEY`.
    pub fn xiaomi_token_plan_ams() -> Self {
        Self::xiaomi().regional(
            "xiaomi-token-plan-ams",
            "Xiaomi Token Plan AMS",
            "https://token-plan-ams.xiaomimimo.com/v1",
            "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
            "xiaomi-token-plan-ams",
        )
    }

    /// Xiaomi's Singapore Token Plan, authenticated with `XIAOMI_TOKEN_PLAN_SGP_API_KEY`.
    pub fn xiaomi_token_plan_sgp() -> Self {
        Self::xiaomi().regional(
            "xiaomi-token-plan-sgp",
            "Xiaomi Token Plan SGP",
            "https://token-plan-sgp.xiaomimimo.com/v1",
            "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
            "xiaomi-token-plan-sgp",
        )
    }

    /// Qwen's international Token Plan, authenticated with `QWEN_TOKEN_PLAN_API_KEY`.
    pub fn qwen_token_plan() -> Self {
        Self::openai_compatible(
            "qwen-token-plan",
            "Qwen Token Plan",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            ["QWEN_TOKEN_PLAN_API_KEY"],
        )
        .with_openai_compat(OpenAiCompat {
            reasoning_format: OpenAiReasoningFormat::EnableThinking,
            ..OpenAiCompat::default()
        })
        .with_models_dev_id("alibaba-token-plan")
    }

    /// Qwen's China Token Plan, authenticated with `QWEN_TOKEN_PLAN_CN_API_KEY`.
    pub fn qwen_token_plan_cn() -> Self {
        Self::qwen_token_plan().regional(
            "qwen-token-plan-cn",
            "Qwen Token Plan CN",
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "QWEN_TOKEN_PLAN_CN_API_KEY",
            "alibaba-token-plan-cn",
        )
    }

    /// Qwen's Individual plan, sharing the international key and endpoint but
    /// restricted to its published model allowlist.
    pub fn qwen_token_plan_individual() -> Self {
        Self::qwen_token_plan().regional(
            "qwen-token-plan-individual",
            "Qwen Token Plan Individual",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            "QWEN_TOKEN_PLAN_API_KEY",
            "qwen-token-plan-individual",
        )
    }
}
