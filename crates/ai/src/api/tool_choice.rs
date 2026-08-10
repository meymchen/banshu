//! The tool-choice preflight: the one place a requested [`ToolChoice`] is
//! checked against what the provider declares its endpoint can express.
//!
//! It runs in [`api::drive`](super::drive) before context normalization, auth
//! resolution, and any HTTP request, because it reads nothing but the request
//! options, the model's protocol, and the provider's declared compat — so a
//! request that cannot be honoured fails at the earliest honest moment.
//!
//! Rejecting is the whole point. Silently remapping a `required` onto an
//! `auto`, or dropping a named choice the endpoint has no field for, would
//! answer a question the caller did not ask; the crate reports
//! [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) instead and
//! names the choices that would have worked.

use crate::options::StreamOptions;
use crate::provider::{AnthropicCompat, OpenAiCompat};
use crate::types::{ApiKind, Model, ToolChoice};

/// How a choice is named in a rejection detail.
fn choice_label(choice: &ToolChoice) -> String {
    match choice {
        ToolChoice::Auto => "`auto`".to_string(),
        ToolChoice::None => "`none`".to_string(),
        ToolChoice::Required => "`required`".to_string(),
        ToolChoice::Named(name) => format!("named tool `{name}`"),
    }
}

/// Check `options.tool_choice` against what the provider declares for the
/// model's protocol. `Ok(())` means the request can be honoured — including
/// the common case of no tool choice at all, which leaves the payload
/// untouched. `Err` carries the detail for an in-band `InvalidRequest`.
pub(crate) fn validate(
    model: &Model,
    options: &StreamOptions,
    openai: OpenAiCompat,
    anthropic: AnthropicCompat,
) -> Result<(), String> {
    let Some(choice) = options.tool_choice.as_ref() else {
        return Ok(());
    };
    // Routing goes by the model's protocol, so a mixed-protocol provider is
    // judged on the side this request is headed to.
    let support = match model.api {
        ApiKind::OpenAiCompletions => openai.tool_choice,
        ApiKind::AnthropicMessages => anthropic.tool_choice,
    };
    if support.supports(choice) {
        return Ok(());
    }
    let supported = support.supported_names();
    let list = if supported.is_empty() {
        "none".to_string()
    } else {
        supported.join(", ")
    };
    Err(format!(
        "provider `{}` cannot express tool choice {} for the `{}` protocol; \
         supported choices: {list}",
        model.provider,
        choice_label(choice),
        super::api_name(model.api),
    ))
}

/// Unit coverage for what `tests/tool_choice.rs` cannot reach end-to-end: the
/// exact rejection details, and the protocol-routing choice a mixed-protocol
/// provider forces. Every rejection *path* is also pinned against a mock
/// server there; these pin the words and the wiring.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolChoiceSupport;

    fn model(api: ApiKind) -> Model {
        let mut model = match api {
            ApiKind::OpenAiCompletions => Model::openai_completions("test-model"),
            ApiKind::AnthropicMessages => Model::anthropic_messages("test-model"),
        };
        model.provider = "test".to_string();
        model
    }

    fn asked(choice: Option<ToolChoice>) -> StreamOptions {
        StreamOptions {
            tool_choice: choice,
            ..StreamOptions::default()
        }
    }

    #[test]
    fn no_tool_choice_is_always_honoured() {
        // Even where nothing is declared anywhere — the payload is untouched.
        let model = model(ApiKind::OpenAiCompletions);
        assert_eq!(
            validate(
                &model,
                &asked(None),
                OpenAiCompat::default(),
                AnthropicCompat::default()
            ),
            Ok(())
        );
    }

    #[test]
    fn a_rejection_names_the_choices_that_would_have_worked() {
        let model = model(ApiKind::OpenAiCompletions);
        let openai = OpenAiCompat {
            tool_choice: ToolChoiceSupport {
                auto: true,
                none: true,
                ..ToolChoiceSupport::NONE
            },
            ..OpenAiCompat::default()
        };
        let error = validate(
            &model,
            &asked(Some(ToolChoice::Required)),
            openai,
            AnthropicCompat::default(),
        )
        .expect_err("required is not declared");
        assert!(error.contains("`required`"), "{error}");
        assert!(
            error.contains("supported choices: auto, none"),
            "the caller should learn what would have worked: {error}"
        );

        // A named rejection names the requested tool.
        let error = validate(
            &model,
            &asked(Some(ToolChoice::Named("get_weather".to_string()))),
            openai,
            AnthropicCompat::default(),
        )
        .expect_err("named is not declared");
        assert!(error.contains("named tool `get_weather`"), "{error}");

        // Nothing declared says so rather than printing an empty list.
        let error = validate(
            &model,
            &asked(Some(ToolChoice::Auto)),
            OpenAiCompat::default(),
            AnthropicCompat::default(),
        )
        .expect_err("an unconfigured endpoint attests nothing");
        assert!(error.contains("supported choices: none"), "{error}");
    }

    #[test]
    fn the_support_is_chosen_by_the_models_protocol_not_the_providers_primary() {
        // A mixed-protocol provider declares both sides; a model routes to its
        // own. Here only the Anthropic side declares support.
        let anthropic = AnthropicCompat {
            tool_choice: ToolChoiceSupport::ALL,
            ..AnthropicCompat::default()
        };
        let asked = asked(Some(ToolChoice::Auto));
        assert_eq!(
            validate(
                &model(ApiKind::AnthropicMessages),
                &asked,
                OpenAiCompat::default(),
                anthropic,
            ),
            Ok(())
        );
        assert!(
            validate(
                &model(ApiKind::OpenAiCompletions),
                &asked,
                OpenAiCompat::default(),
                anthropic,
            )
            .is_err()
        );
    }
}
