//! The cache-routing preflight: the one place an explicitly requested
//! [`CacheRetention`] is checked against what the provider declares its
//! endpoint can express.
//!
//! It runs in [`api::drive`](super::drive) before context normalization, auth
//! resolution, and any HTTP request, because it reads nothing but the request
//! options, the model's protocol, and the provider's declared compat — so a
//! request that cannot be honoured fails at the earliest honest moment.
//!
//! Rejecting is the whole point. Silently dropping a long-retention request
//! onto the endpoint's normal cache behavior would answer a question the
//! caller did not ask; the crate reports
//! [`ErrorKind::InvalidRequest`](crate::ErrorKind::InvalidRequest) instead and
//! names the retention that would have worked.

use crate::options::{CacheRetention, StreamOptions};
use crate::provider::{OpenAiCacheRetention, OpenAiCompat};
use crate::types::{ApiKind, Model};

/// Check `options.cache_retention` against what the provider declares for the
/// model's protocol. `Ok(())` means the request can be honoured — including
/// the common cases of no preference, `Short`, and `Disabled`, none of which
/// need an attestation. `Err` carries the detail for an in-band
/// `InvalidRequest`.
pub(crate) fn validate(
    model: &Model,
    options: &StreamOptions,
    openai: OpenAiCompat,
) -> Result<(), String> {
    if options.cache_retention != Some(CacheRetention::Long) {
        return Ok(());
    }
    match model.api {
        // The Anthropic adapter expresses long retention through its
        // one-hour cache-control TTL, which every Messages endpoint takes, so
        // only the OpenAI-compatible side can fail this check.
        ApiKind::AnthropicMessages => Ok(()),
        ApiKind::OpenAiCompletions => {
            if openai.cache_retention == OpenAiCacheRetention::Long {
                return Ok(());
            }
            Err(format!(
                "provider `{}` cannot express long prompt-cache retention for the `{}` \
                 protocol; supported retention: short, disabled",
                model.provider,
                super::api_name(model.api),
            ))
        }
    }
}

/// Unit coverage for what `tests/openai_prompt_caching.rs` cannot reach
/// end-to-end: the exact rejection detail, and the protocol routing a
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

    fn asked(retention: Option<CacheRetention>) -> StreamOptions {
        StreamOptions {
            cache_retention: retention,
            ..StreamOptions::default()
        }
    }

    #[test]
    fn anything_short_of_an_explicit_long_needs_no_attestation() {
        let model = model(ApiKind::OpenAiCompletions);
        for retention in [
            None,
            Some(CacheRetention::Short),
            Some(CacheRetention::Disabled),
        ] {
            assert_eq!(
                validate(&model, &asked(retention), OpenAiCompat::default()),
                Ok(()),
                "{retention:?} leaves the endpoint's normal cache behavior alone"
            );
        }
    }

    #[test]
    fn an_attesting_provider_takes_long_retention() {
        let model = model(ApiKind::OpenAiCompletions);
        let openai = OpenAiCompat {
            cache_retention: OpenAiCacheRetention::Long,
            ..OpenAiCompat::default()
        };
        assert_eq!(
            validate(&model, &asked(Some(CacheRetention::Long)), openai),
            Ok(())
        );
    }

    #[test]
    fn an_unattested_long_is_refused_with_the_working_retention_named() {
        let model = model(ApiKind::OpenAiCompletions);
        let error = validate(
            &model,
            &asked(Some(CacheRetention::Long)),
            OpenAiCompat::default(),
        )
        .expect_err("an unconfigured endpoint attests nothing");
        assert!(error.contains("provider `test`"), "{error}");
        assert!(
            error.contains("supported retention: short, disabled"),
            "the caller should learn what would have worked: {error}"
        );
    }

    #[test]
    fn the_anthropic_protocol_always_expresses_long_retention() {
        let model = model(ApiKind::AnthropicMessages);
        assert_eq!(
            validate(
                &model,
                &asked(Some(CacheRetention::Long)),
                OpenAiCompat::default(),
            ),
            Ok(()),
            "the one-hour cache-control TTL needs no attestation"
        );
    }
}
