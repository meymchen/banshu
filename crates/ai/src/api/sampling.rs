//! The sampling preflight: the one place [`StreamOptions::sampling`] is
//! checked against the request fields the adapter owns.
//!
//! It runs in [`api::drive`](super::drive) before context normalization, auth
//! resolution, and any HTTP request, because it reads nothing but the request
//! options and the model's protocol — so a request that cannot be honoured
//! fails at the earliest honest moment, before the server ever records one.
//!
//! Rejecting is the whole point. The sampling map is an escape hatch for
//! open-model controls the crate does not model, not a back door around the
//! fields the adapter produces: letting a caller-supplied `messages` or
//! `max_tokens` silently win over the adapter's own would answer a question
//! the caller did not ask. The crate reports
//! [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) instead and
//! names the offending key.
//!
//! Only the OpenAI-completions protocol consumes the map; an Anthropic
//! Messages request ignores it wholesale, reserved keys included, so nothing
//! here gates it.

use crate::options::StreamOptions;
use crate::types::{ApiKind, Model};

/// Every top-level request field the OpenAI-completions adapter owns, by
/// family:
///
/// - model: `model`
/// - messages: `messages`
/// - tools: `tools`
/// - stream controls: `stream`, `stream_options`
/// - output budget: `max_tokens`, `max_completion_tokens`
/// - reasoning: `thinking`, `reasoning_effort`, `enable_thinking`,
///   `chat_template_kwargs`
/// - tool choice: `tool_choice`
/// - caching: `prompt_cache_key`, `prompt_cache_retention`
/// - sampling already modelled by the crate: `temperature`
/// - caller metadata and diagnostics: `metadata`
/// - authentication-related values: `api_key`, `authorization`, `x-api-key`
///
/// Wire keys compare case-sensitively — JSON field names do.
pub(crate) const RESERVED_SAMPLING_KEYS: &[&str] = &[
    "model",
    "messages",
    "tools",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "thinking",
    "reasoning_effort",
    "enable_thinking",
    "chat_template_kwargs",
    "tool_choice",
    "prompt_cache_key",
    "prompt_cache_retention",
    "temperature",
    "metadata",
    "api_key",
    "authorization",
    "x-api-key",
];

/// Check `options.sampling` against the adapter-owned fields for the model's
/// protocol. `Ok(())` means every key may merge into the request — including
/// the common case of an empty map, which leaves the payload untouched, and
/// any map at all on the Anthropic protocol, which ignores it. `Err` carries
/// the detail for an in-band `InvalidRequest` naming the offending key.
pub(crate) fn validate(model: &Model, options: &StreamOptions) -> Result<(), String> {
    if options.sampling.is_empty() || model.api != ApiKind::OpenAiCompletions {
        return Ok(());
    }
    // BTreeMap order makes the reported key deterministic when several
    // collide: the lexicographically first offender is named.
    if let Some(key) = options
        .sampling
        .keys()
        .find(|key| RESERVED_SAMPLING_KEYS.contains(&key.as_str()))
    {
        return Err(format!(
            "sampling parameter `{key}` overrides a request field the adapter owns for the \
             `{}` protocol; provider `{}` reserves fields covering the model, messages, \
             tools, stream controls, output budget, reasoning, tool choice, caching, \
             metadata, and authentication, so remove `{key}` from the sampling map",
            super::api_name(model.api),
            model.provider,
        ));
    }
    Ok(())
}

/// Unit coverage for what `tests/openai_sampling.rs` cannot reach end-to-end:
/// the exact rejection detail, every reserved key, and the protocol routing a
/// mixed-protocol provider forces. The rejection *path* — in-band
/// `InvalidRequest` before the server records a request — is pinned against a
/// mock server there; these pin the words and the wiring.
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

    fn asked(keys: &[&str]) -> StreamOptions {
        StreamOptions {
            sampling: keys
                .iter()
                .map(|key| (key.to_string(), serde_json::Value::Null))
                .collect(),
            ..StreamOptions::default()
        }
    }

    #[test]
    fn an_empty_map_is_always_honoured() {
        for api in [ApiKind::OpenAiCompletions, ApiKind::AnthropicMessages] {
            assert_eq!(validate(&model(api), &StreamOptions::default()), Ok(()));
        }
    }

    #[test]
    fn open_model_sampling_controls_pass() {
        let options = asked(&[
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "logit_bias",
            "a_key_nobody_has_heard_of",
        ]);
        assert_eq!(
            validate(&model(ApiKind::OpenAiCompletions), &options),
            Ok(())
        );
    }

    #[test]
    fn every_reserved_key_is_refused_naming_the_key() {
        for key in RESERVED_SAMPLING_KEYS {
            let error = validate(&model(ApiKind::OpenAiCompletions), &asked(&[key]))
                .expect_err("a reserved key can never merge");
            assert!(error.contains(&format!("`{key}`")), "{error}");
            assert!(error.contains("provider `test`"), "{error}");
            assert!(error.contains("openai-completions"), "{error}");
            assert!(
                error.contains("remove"),
                "the caller should learn what would have worked: {error}"
            );
        }
    }

    #[test]
    fn one_reserved_key_among_many_is_the_one_named() {
        let error = validate(
            &model(ApiKind::OpenAiCompletions),
            &asked(&["top_p", "messages", "stop"]),
        )
        .expect_err("one reserved key poisons the whole map");
        assert!(error.contains("`messages`"), "{error}");
    }

    #[test]
    fn reserved_keys_compare_case_sensitively() {
        // JSON field names are case-sensitive; `Model` is not the wire field.
        let options = asked(&["Model", "STREAM", "Max_Tokens"]);
        assert_eq!(
            validate(&model(ApiKind::OpenAiCompletions), &options),
            Ok(())
        );
    }

    #[test]
    fn the_anthropic_protocol_ignores_the_map_reserved_keys_included() {
        let options = asked(&["model", "messages", "top_p"]);
        assert_eq!(
            validate(&model(ApiKind::AnthropicMessages), &options),
            Ok(()),
            "the Anthropic adapter never reads the sampling map"
        );
    }
}
