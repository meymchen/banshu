//! The reasoning preflight: the one place a reasoning request is checked
//! against what the target model and provider actually declare.
//!
//! It runs in [`api::drive`](super::drive) before context normalization, auth
//! resolution, and any HTTP request, because it reads nothing but the request
//! options, the model's metadata, and the provider's declared request shape —
//! so a request that cannot be honoured fails at the earliest honest moment.
//!
//! Rejecting is the whole point. Silently clamping `xhigh` down to `high`, or
//! dropping a token budget the endpoint has no field for, would answer a
//! question the caller did not ask; the crate reports
//! [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) instead and
//! names the levels that would have worked.

use crate::provider::{AnthropicCompat, OpenAiCompat};
use crate::types::{ApiKind, CapabilitySupport, Model, ReasoningOptions};

/// What the provider's declared request shape can carry, reduced to the two
/// facts the preflight needs from it.
struct DeclaredFormat {
    /// Whether the endpoint accepts any reasoning request field at all.
    supported: bool,
    /// Whether it carries an explicit reasoning token budget.
    token_budget: bool,
}

impl DeclaredFormat {
    fn of(api: ApiKind, openai: OpenAiCompat, anthropic: AnthropicCompat) -> Self {
        match api {
            ApiKind::OpenAiCompletions => Self {
                supported: openai.reasoning_format.is_supported(),
                token_budget: openai.reasoning_format.accepts_token_budget(),
            },
            ApiKind::AnthropicMessages => Self {
                supported: anthropic.reasoning_format.is_supported(),
                token_budget: anthropic.reasoning_format.accepts_token_budget(),
            },
        }
    }
}

/// Check `reasoning` against the target model and provider. `Ok(())` means the
/// request can be honoured — including the common case of no reasoning option
/// at all, which leaves the payload untouched. `Err` carries the detail for an
/// in-band `InvalidRequest`.
pub(crate) fn validate(
    model: &Model,
    reasoning: Option<&ReasoningOptions>,
    openai: OpenAiCompat,
    anthropic: AnthropicCompat,
) -> Result<(), String> {
    let Some(reasoning) = reasoning else {
        return Ok(());
    };
    let effort = reasoning.effort;
    let format = DeclaredFormat::of(model.api, openai, anthropic);

    if !format.supported {
        return Err(format!(
            "provider `{}` declares no reasoning request format for the `{}` protocol, \
             so effort `{effort}` cannot be requested",
            model.provider,
            super::api_name(model.api),
        ));
    }

    if !model.reasoning.supports(effort) {
        let attested = model.reasoning.efforts();
        let levels = if attested.is_empty() {
            "none".to_string()
        } else {
            attested
                .iter()
                .map(|level| level.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(format!(
            "model `{}` does not support reasoning effort `{effort}`; attested levels: {levels}",
            model.id,
        ));
    }

    if let Some(tokens) = reasoning.token_budget {
        if !format.token_budget {
            return Err(format!(
                "the reasoning request format declared by provider `{}` carries no token budget, \
                 so a budget of {tokens} cannot be requested",
                model.provider,
            ));
        }
        if model.reasoning.token_budget() != CapabilitySupport::Supported {
            return Err(format!(
                "model `{}` does not support a reasoning token budget, \
                 so a budget of {tokens} cannot be requested",
                model.id,
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{AnthropicReasoningFormat, OpenAiReasoningFormat};
    use crate::types::{ReasoningCapability, ReasoningEffort};

    fn openai_format(format: OpenAiReasoningFormat) -> OpenAiCompat {
        OpenAiCompat {
            reasoning_format: format,
            ..OpenAiCompat::default()
        }
    }

    fn anthropic_format(format: AnthropicReasoningFormat) -> AnthropicCompat {
        AnthropicCompat {
            reasoning_format: format,
            ..AnthropicCompat::default()
        }
    }

    fn model(reasoning: ReasoningCapability) -> Model {
        let mut model = Model::openai_completions("test-model");
        model.provider = "test".to_string();
        model.reasoning = reasoning;
        model
    }

    fn check(model: &Model, reasoning: Option<&ReasoningOptions>) -> Result<(), String> {
        validate(
            model,
            reasoning,
            openai_format(OpenAiReasoningFormat::ReasoningEffort),
            anthropic_format(AnthropicReasoningFormat::ThinkingBudget),
        )
    }

    #[test]
    fn no_reasoning_option_is_always_honoured() {
        // Even where nothing is attested anywhere — the payload is untouched.
        let model = model(ReasoningCapability::none());
        assert_eq!(
            validate(
                &model,
                None,
                OpenAiCompat::default(),
                AnthropicCompat::default()
            ),
            Ok(())
        );
        assert_eq!(check(&model, None), Ok(()));
    }

    #[test]
    fn an_attested_level_is_honoured() {
        let model = model(ReasoningCapability::baseline());
        for effort in ReasoningCapability::BASELINE {
            assert_eq!(check(&model, Some(&ReasoningOptions::new(effort))), Ok(()));
        }
    }

    #[test]
    fn a_provider_without_a_declared_format_rejects_every_level() {
        let model = model(ReasoningCapability::baseline());
        for effort in ReasoningEffort::ALL {
            let error = validate(
                &model,
                Some(&ReasoningOptions::new(effort)),
                OpenAiCompat::default(),
                AnthropicCompat::default(),
            )
            .expect_err("an undeclared format cannot carry a reasoning request");
            assert!(error.contains("no reasoning request format"), "{error}");
        }
    }

    #[test]
    fn an_unattested_level_is_rejected_and_names_the_attested_ones() {
        let model = model(ReasoningCapability::baseline());
        for effort in [ReasoningEffort::XHigh, ReasoningEffort::Max] {
            let error = check(&model, Some(&ReasoningOptions::new(effort)))
                .expect_err("the baseline ladder stops at `high`");
            assert!(error.contains(effort.as_str()), "{error}");
            assert!(
                error.contains("off, minimal, low, medium, high"),
                "the caller should learn what would have worked: {error}"
            );
        }
    }

    #[test]
    fn a_model_attesting_nothing_rejects_every_level() {
        let model = model(ReasoningCapability::none());
        for effort in ReasoningEffort::ALL {
            let error = check(&model, Some(&ReasoningOptions::new(effort)))
                .expect_err("nothing attested means nothing accepted");
            assert!(error.contains("attested levels: none"), "{error}");
        }
    }

    #[test]
    fn a_budget_needs_both_a_format_that_carries_it_and_a_model_that_attests_it() {
        let requested = ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096);

        // The OpenAI-compatible shapes carry no budget field.
        let attested =
            model(ReasoningCapability::baseline().with_token_budget(CapabilitySupport::Supported));
        let error = check(&attested, Some(&requested)).expect_err("no budget field on this format");
        assert!(error.contains("carries no token budget"), "{error}");

        // A format that carries one, but a model that does not attest it.
        let mut unattested = model(ReasoningCapability::baseline());
        unattested.api = ApiKind::AnthropicMessages;
        let error =
            check(&unattested, Some(&requested)).expect_err("the model attests no budget support");
        assert!(
            error.contains("does not support a reasoning token budget"),
            "{error}"
        );

        // Both declared: honoured.
        let mut both = attested;
        both.api = ApiKind::AnthropicMessages;
        assert_eq!(check(&both, Some(&requested)), Ok(()));
    }

    #[test]
    fn the_format_is_chosen_by_the_models_protocol_not_the_providers_primary() {
        // A mixed-protocol provider declares both shapes; a model routes to
        // its own. Here only the Anthropic side is declared.
        let mut model = model(ReasoningCapability::baseline());
        model.api = ApiKind::AnthropicMessages;
        let requested = ReasoningOptions::new(ReasoningEffort::High);
        assert_eq!(
            validate(
                &model,
                Some(&requested),
                OpenAiCompat::default(),
                anthropic_format(AnthropicReasoningFormat::ThinkingEffort),
            ),
            Ok(())
        );

        model.api = ApiKind::OpenAiCompletions;
        assert!(
            validate(
                &model,
                Some(&requested),
                OpenAiCompat::default(),
                anthropic_format(AnthropicReasoningFormat::ThinkingEffort),
            )
            .is_err()
        );
    }
}
