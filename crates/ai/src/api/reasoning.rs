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

use crate::options::StreamOptions;
use crate::provider::{AnthropicCompat, DeclaredReasoning, OpenAiCompat};
use crate::types::{ApiKind, CapabilitySupport, Model, ReasoningEffort};

/// Check `options.reasoning` against the target model and provider. `Ok(())`
/// means the request can be honoured — including the common case of no
/// reasoning option at all, which leaves the payload untouched. `Err` carries
/// the detail for an in-band `InvalidRequest`.
pub(crate) fn validate(
    model: &Model,
    options: &StreamOptions,
    openai: OpenAiCompat,
    anthropic: AnthropicCompat,
) -> Result<(), String> {
    let Some(reasoning) = options.reasoning.as_ref() else {
        return Ok(());
    };
    let effort = reasoning.effort;
    // Routing goes by the model's protocol, so a mixed-protocol provider is
    // judged on the side this request is headed to.
    let format = DeclaredReasoning::of(model.api, openai, anthropic);

    if !format.declared {
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
        if effort == ReasoningEffort::Off {
            return Err(format!(
                "a reasoning budget of {tokens} cannot be requested alongside effort `off`, \
                 which disables reasoning outright",
            ));
        }
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
        if model.api == ApiKind::OpenAiCompletions
            && options.max_tokens.is_some_and(|output| tokens >= output)
        {
            return Err(format!(
                "a reasoning budget of {tokens} does not fit under the resolved Output Budget \
                 of {} for model `{}`",
                options.max_tokens.expect("checked as present"),
                model.id,
            ));
        }
    }

    if model.api == ApiKind::AnthropicMessages && format.token_budget {
        // Anthropic's budget-only shape must derive a number when the caller
        // names none and enforces its own documented minimum. OpenAI-compatible
        // chat-template budgets are optional and were fully checked above.
        super::anthropic_messages::validate_thinking_budget(model, options, reasoning)?;
    }

    Ok(())
}

/// Unit coverage for what `tests/reasoning_capabilities.rs` cannot reach
/// end-to-end: the exact rejection details, and the protocol-routing choice a
/// mixed-protocol provider forces. Every rejection *path* is also pinned
/// against a mock server there; these pin the words and the wiring.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{AnthropicReasoningFormat, OpenAiReasoningFormat};
    use crate::types::{ApiKind, ReasoningCapability, ReasoningEffort, ReasoningOptions};

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

    /// A request asking for `reasoning`, under an output cap with room for any
    /// budget: what fits under a *small* cap is the budget-shape rule, pinned
    /// end-to-end in `tests/anthropic_reasoning_requests.rs`.
    fn asked(reasoning: Option<ReasoningOptions>) -> StreamOptions {
        StreamOptions {
            max_tokens: Some(32_000),
            reasoning,
            ..StreamOptions::default()
        }
    }

    fn check(model: &Model, reasoning: Option<ReasoningOptions>) -> Result<(), String> {
        validate(
            model,
            &asked(reasoning),
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
                &asked(None),
                OpenAiCompat::default(),
                AnthropicCompat::default()
            ),
            Ok(())
        );
        assert_eq!(check(&model, None), Ok(()));
    }

    #[test]
    fn a_rejection_names_the_levels_that_would_have_worked() {
        let attested = model(ReasoningCapability::baseline());
        for effort in [ReasoningEffort::XHigh, ReasoningEffort::Max] {
            let error = check(&attested, Some(ReasoningOptions::new(effort)))
                .expect_err("the baseline ladder stops at `high`");
            assert!(error.contains(effort.as_str()), "{error}");
            assert!(
                error.contains("off, minimal, low, medium, high"),
                "the caller should learn what would have worked: {error}"
            );
        }

        // A model attesting nothing says so rather than printing an empty list.
        let nothing = model(ReasoningCapability::none());
        let error = check(&nothing, Some(ReasoningOptions::new(ReasoningEffort::Low)))
            .expect_err("nothing attested means nothing accepted");
        assert!(error.contains("attested levels: none"), "{error}");
    }

    #[test]
    fn a_budget_needs_both_a_format_that_carries_it_and_a_model_that_attests_it() {
        let requested = ReasoningOptions::new(ReasoningEffort::High).with_token_budget(4096);

        // The OpenAI-compatible shapes carry no budget field.
        let attested =
            model(ReasoningCapability::baseline().with_token_budget(CapabilitySupport::Supported));
        let error =
            check(&attested, Some(requested.clone())).expect_err("no budget field on this format");
        assert!(error.contains("carries no token budget"), "{error}");

        // A format that carries one, but a model that does not attest it.
        let mut unattested = model(ReasoningCapability::baseline());
        unattested.api = ApiKind::AnthropicMessages;
        let error = check(&unattested, Some(requested.clone()))
            .expect_err("the model attests no budget support");
        assert!(
            error.contains("does not support a reasoning token budget"),
            "{error}"
        );

        // Both declared: honoured.
        let mut both = attested;
        both.api = ApiKind::AnthropicMessages;
        assert_eq!(check(&both, Some(requested)), Ok(()));
    }

    #[test]
    fn the_format_is_chosen_by_the_models_protocol_not_the_providers_primary() {
        // A mixed-protocol provider declares both shapes; a model routes to
        // its own. Here only the Anthropic side is declared.
        let mut model = model(ReasoningCapability::baseline());
        model.api = ApiKind::AnthropicMessages;
        let requested = asked(Some(ReasoningOptions::new(ReasoningEffort::High)));
        assert_eq!(
            validate(
                &model,
                &requested,
                OpenAiCompat::default(),
                anthropic_format(AnthropicReasoningFormat::ThinkingAdaptive),
            ),
            Ok(())
        );

        model.api = ApiKind::OpenAiCompletions;
        assert!(
            validate(
                &model,
                &requested,
                OpenAiCompat::default(),
                anthropic_format(AnthropicReasoningFormat::ThinkingAdaptive),
            )
            .is_err()
        );
    }
}
