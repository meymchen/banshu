//! The temperature preflight: the one place an explicitly requested
//! [`StreamOptions::temperature`] is checked against what the provider
//! declares its endpoint can express.
//!
//! It runs in [`api::drive`](super::drive) before context normalization, auth
//! resolution, and any HTTP request, because it reads nothing but the request
//! options, the model's protocol, and the provider's declared compat — so a
//! request that cannot be honoured fails at the earliest honest moment. It
//! runs after the [reasoning preflight](super::reasoning), so an "enabled
//! reasoning request" here is always one that will actually be sent.
//!
//! Rejecting is the whole point. Silently dropping a temperature the endpoint
//! has no field for — or one its reference forbids alongside enabled
//! reasoning — would answer a question the caller did not ask; the crate
//! reports [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest)
//! instead and names what would have worked.
//!
//! Only the Anthropic Messages protocol declares temperature support today
//! ([`AnthropicCompat::temperature`](crate::AnthropicCompat::temperature));
//! an OpenAI-compatible request passes through untouched.

use crate::options::StreamOptions;
use crate::provider::{AnthropicCompat, AnthropicTemperature, OpenAiCompat};
use crate::types::{ApiKind, Model, ReasoningEffort, ReasoningOptions};

/// Whether the declared policy permits an explicit temperature on a request
/// carrying `reasoning` as its reasoning option. The preflight refuses what
/// this forbids; the adapter debug-asserts the preflight ran.
pub(crate) fn permits(policy: AnthropicTemperature, reasoning: Option<&ReasoningOptions>) -> bool {
    match policy {
        AnthropicTemperature::Unsupported => false,
        // `Off` disables reasoning outright — a disabled toggle is no
        // reasoning combination, so only an *enabled* request conflicts.
        AnthropicTemperature::WithoutReasoning => {
            reasoning.is_none_or(|request| request.effort == ReasoningEffort::Off)
        }
        AnthropicTemperature::WithReasoning => true,
    }
}

/// Check `options.temperature` against what the provider declares for the
/// model's protocol. `Ok(())` means the request can be honoured — including
/// the common case of no temperature at all, which leaves the payload
/// untouched. `Err` carries the detail for an in-band `InvalidRequest`.
pub(crate) fn validate(
    model: &Model,
    options: &StreamOptions,
    _openai: OpenAiCompat,
    anthropic: AnthropicCompat,
) -> Result<(), String> {
    let Some(temperature) = options.temperature else {
        return Ok(());
    };
    // Routing goes by the model's protocol. Only the Anthropic side declares
    // a temperature policy; the OpenAI side keeps its existing envelope.
    if model.api != ApiKind::AnthropicMessages {
        return Ok(());
    }
    match anthropic.temperature {
        AnthropicTemperature::WithReasoning => Ok(()),
        // The only conditional policy: permitted, this request is honoured.
        policy if permits(policy, options.reasoning.as_ref()) => Ok(()),
        AnthropicTemperature::Unsupported => Err(format!(
            "provider `{}` cannot express a temperature of {temperature} for the `{}` \
             protocol; the provider declares no temperature support, so omit it to use \
             the endpoint's own sampling",
            model.provider,
            super::api_name(model.api),
        )),
        AnthropicTemperature::WithoutReasoning => Err(format!(
            "provider `{}` accepts a temperature of {temperature} only without an enabled \
             reasoning request; omit the temperature or the reasoning request",
            model.provider,
        )),
    }
}

/// Unit coverage for what `tests/anthropic_temperature.rs` cannot reach
/// end-to-end: the exact rejection details, and the protocol routing a
/// mixed-protocol provider forces. Every rejection *path* is also pinned
/// against a mock server there; these pin the words and the wiring.
#[cfg(test)]
mod tests {
    use super::*;

    fn model(api: ApiKind) -> Model {
        let mut model = match api {
            ApiKind::OpenAiCompletions => Model::openai_completions("test-model"),
            ApiKind::AnthropicMessages => Model::anthropic_messages("test-model"),
        };
        model.provider = "test".to_string();
        model
    }

    fn asked(temperature: Option<f32>, reasoning: Option<ReasoningOptions>) -> StreamOptions {
        StreamOptions {
            temperature,
            reasoning,
            ..StreamOptions::default()
        }
    }

    fn anthropic(temperature: AnthropicTemperature) -> AnthropicCompat {
        AnthropicCompat {
            temperature,
            ..AnthropicCompat::default()
        }
    }

    #[test]
    fn no_temperature_is_always_honoured() {
        // Even where nothing is attested anywhere — the payload is untouched.
        for api in [ApiKind::OpenAiCompletions, ApiKind::AnthropicMessages] {
            assert_eq!(
                validate(
                    &model(api),
                    &asked(None, Some(ReasoningOptions::new(ReasoningEffort::High))),
                    OpenAiCompat::default(),
                    AnthropicCompat::default(),
                ),
                Ok(()),
            );
        }
    }

    #[test]
    fn the_openai_protocol_is_not_gated_by_the_anthropic_declaration() {
        // A mixed-protocol provider declares nothing on the Anthropic side;
        // a model routed to OpenAI keeps its existing envelope regardless.
        assert_eq!(
            validate(
                &model(ApiKind::OpenAiCompletions),
                &asked(Some(0.7), None),
                OpenAiCompat::default(),
                AnthropicCompat::default(),
            ),
            Ok(()),
        );
    }

    #[test]
    fn an_undeclared_temperature_is_refused_naming_the_value() {
        for reasoning in [
            None,
            Some(ReasoningOptions::new(ReasoningEffort::Off)),
            Some(ReasoningOptions::new(ReasoningEffort::High)),
        ] {
            let error = validate(
                &model(ApiKind::AnthropicMessages),
                &asked(Some(0.7), reasoning),
                OpenAiCompat::default(),
                anthropic(AnthropicTemperature::Unsupported),
            )
            .expect_err("an unconfigured endpoint attests nothing");
            assert!(error.contains("provider `test`"), "{error}");
            assert!(error.contains("temperature of 0.7"), "{error}");
            assert!(error.contains("anthropic-messages"), "{error}");
            assert!(
                error.contains("omit it"),
                "the caller should learn what would have worked: {error}"
            );
        }
    }

    #[test]
    fn without_reasoning_refuses_exactly_the_enabled_combination() {
        let compat = anthropic(AnthropicTemperature::WithoutReasoning);
        let anthropic_model = model(ApiKind::AnthropicMessages);

        // No reasoning option, and an explicit `Off`, both go through.
        for reasoning in [None, Some(ReasoningOptions::new(ReasoningEffort::Off))] {
            let allowed = validate(
                &anthropic_model,
                &asked(Some(0.7), reasoning.clone()),
                OpenAiCompat::default(),
                compat,
            );
            assert_eq!(
                allowed,
                Ok(()),
                "{reasoning:?} is no enabled reasoning request"
            );
        }

        // Every enabled level conflicts.
        for effort in ReasoningEffort::ALL {
            if effort == ReasoningEffort::Off {
                continue;
            }
            let error = validate(
                &anthropic_model,
                &asked(Some(0.7), Some(ReasoningOptions::new(effort))),
                OpenAiCompat::default(),
                compat,
            )
            .expect_err("temperature alongside enabled reasoning is refused");
            assert!(error.contains("provider `test`"), "{error}");
            assert!(
                error.contains("only without an enabled reasoning request"),
                "{error}"
            );
        }
    }

    #[test]
    fn with_reasoning_permits_every_combination() {
        let compat = anthropic(AnthropicTemperature::WithReasoning);
        for reasoning in [
            None,
            Some(ReasoningOptions::new(ReasoningEffort::Off)),
            Some(ReasoningOptions::new(ReasoningEffort::Max)),
        ] {
            assert_eq!(
                validate(
                    &model(ApiKind::AnthropicMessages),
                    &asked(Some(1.0), reasoning),
                    OpenAiCompat::default(),
                    compat,
                ),
                Ok(()),
            );
        }
    }
}
