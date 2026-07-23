//! Pluggable authentication.
//!
//! A provider resolves credentials through an [`AuthResolver`] rather than a
//! fixed environment variable. Three built-in adapters cover the common cases:
//! [`Auth::api_key_env`] (the historical behaviour — the first set variable in
//! a fallback list), [`Auth::keyless`] (local servers that need no credentials,
//! e.g. llama.cpp / vLLM), and [`Auth::custom`] for anything else.
//!
//! Resolution runs in-band inside the stream, so a resolver failure surfaces as
//! a terminal [`ErrorKind::Auth`](crate::ErrorKind::Auth) error event rather
//! than a synchronous `Result` — [`Provider::stream`](crate::Provider::stream)
//! never fails up front.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{Error, Result};

/// Custom request headers contributed by an auth resolver.
///
/// A `None` value deletes a same-named header supplied by a lower-priority
/// layer. The full priority chain (provider/model/request levels) and its
/// case-insensitive merge land with the headers work; today the only source is
/// [`ResolvedAuth::headers`], so a `Some` value is attached and a `None` value
/// is a no-op.
pub type ProviderHeaders = BTreeMap<String, Option<String>>;

/// Credentials and endpoint overrides produced by an [`AuthResolver`].
#[derive(Debug, Clone, Default)]
pub struct ResolvedAuth {
    /// The API key to authenticate with, if the endpoint needs one. `None`
    /// sends no auth header (a keyless endpoint).
    pub api_key: Option<String>,
    /// Extra headers to attach to each request.
    pub headers: ProviderHeaders,
    /// Overrides the model's base URL when set — for a resolver that also
    /// discovers the endpoint. `None` keeps the model's configured base URL.
    pub base_url: Option<String>,
}

/// Resolves the credentials a provider authenticates with.
///
/// Object-safe and async so a custom resolver can do real work (read a file,
/// refresh a token, call a broker). Built-in adapters are provided via
/// [`Auth`]; implement this trait directly only for [`Auth::custom`].
#[async_trait]
pub trait AuthResolver: Send + Sync {
    /// Whether credentials are currently obtainable, without necessarily
    /// producing them. For availability gating.
    async fn check(&self) -> Result<bool>;

    /// Produce the credentials for a request. An error terminates the stream
    /// with an [`ErrorKind::Auth`](crate::ErrorKind::Auth) event.
    async fn resolve(&self) -> Result<ResolvedAuth>;
}

/// The built-in authentication adapters.
#[derive(Clone)]
pub enum Auth {
    /// Read the key from the first set variable in this fallback list. When no
    /// listed variable is set, resolution fails with
    /// [`ErrorKind::Auth`](crate::ErrorKind::Auth).
    ApiKeyEnv(Vec<String>),
    /// No credentials — the endpoint accepts unauthenticated requests, so no
    /// auth header is sent.
    Keyless,
    /// A caller-supplied resolver.
    Custom(Arc<dyn AuthResolver>),
}

impl Auth {
    /// Read the key from the first set of these environment variables, in
    /// order.
    pub fn api_key_env(vars: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::ApiKeyEnv(vars.into_iter().map(Into::into).collect())
    }

    /// No authentication — for local servers that accept unauthenticated
    /// requests.
    pub fn keyless() -> Self {
        Self::Keyless
    }

    /// A caller-supplied [`AuthResolver`].
    pub fn custom(resolver: Arc<dyn AuthResolver>) -> Self {
        Self::Custom(resolver)
    }

    /// Synchronous best-effort key lookup, used by the list-models probe (which
    /// needs the key string, not just its presence). Only [`Auth::ApiKeyEnv`]
    /// can answer synchronously; keyless has no key, and custom resolvers
    /// require async resolution.
    pub(crate) fn env_api_key(&self) -> Option<String> {
        match self {
            Self::ApiKeyEnv(vars) => vars.iter().find_map(|name| std::env::var(name).ok()),
            _ => None,
        }
    }

    /// Best-effort *synchronous* availability, for `Models::available` gating.
    /// Keyless is always available; api-key-env is available when a listed
    /// variable is set. A custom resolver can only answer via the async
    /// [`AuthResolver::check`], so it reports `false` here — async availability
    /// gating lands with the Models rework.
    pub(crate) fn is_available(&self) -> bool {
        match self {
            Self::ApiKeyEnv(vars) => vars.iter().any(|name| std::env::var(name).is_ok()),
            Self::Keyless => true,
            Self::Custom(_) => false,
        }
    }
}

#[async_trait]
impl AuthResolver for Auth {
    async fn check(&self) -> Result<bool> {
        match self {
            Self::ApiKeyEnv(vars) => Ok(vars.iter().any(|name| std::env::var(name).is_ok())),
            Self::Keyless => Ok(true),
            Self::Custom(resolver) => resolver.check().await,
        }
    }

    async fn resolve(&self) -> Result<ResolvedAuth> {
        match self {
            Self::ApiKeyEnv(vars) => match vars.iter().find_map(|name| std::env::var(name).ok()) {
                Some(key) => Ok(ResolvedAuth {
                    api_key: Some(key),
                    ..Default::default()
                }),
                None => Err(Error::Auth(format!(
                    "no API key found in environment variable(s): {}",
                    vars.join(", ")
                ))),
            },
            Self::Keyless => Ok(ResolvedAuth::default()),
            Self::Custom(resolver) => resolver.resolve().await,
        }
    }
}

/// Resolve the effective auth for a request.
///
/// An explicit [`StreamOptions::api_key`](crate::StreamOptions::api_key)
/// short-circuits the resolver entirely and stands alone as the key, so an
/// explicitly-keyed request succeeds even when the resolver would fail (e.g. an
/// unset env var). Otherwise the resolver runs and any failure propagates to
/// the caller as an [`ErrorKind::Auth`](crate::ErrorKind::Auth) terminal event.
pub(crate) async fn resolve_for_request(
    auth: &Auth,
    explicit_key: Option<String>,
) -> Result<ResolvedAuth> {
    match explicit_key {
        Some(key) => Ok(ResolvedAuth {
            api_key: Some(key),
            ..Default::default()
        }),
        None => auth.resolve().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticResolver(&'static str);

    #[async_trait]
    impl AuthResolver for StaticResolver {
        async fn check(&self) -> Result<bool> {
            Ok(true)
        }
        async fn resolve(&self) -> Result<ResolvedAuth> {
            Ok(ResolvedAuth {
                api_key: Some(self.0.to_string()),
                ..Default::default()
            })
        }
    }

    struct FailingResolver;

    #[async_trait]
    impl AuthResolver for FailingResolver {
        async fn check(&self) -> Result<bool> {
            Ok(false)
        }
        async fn resolve(&self) -> Result<ResolvedAuth> {
            Err(Error::Auth("no token on disk".into()))
        }
    }

    #[tokio::test]
    async fn keyless_resolves_to_no_key_and_is_always_available() {
        let auth = Auth::keyless();
        assert!(auth.check().await.unwrap());
        assert!(auth.resolve().await.unwrap().api_key.is_none());
    }

    #[tokio::test]
    async fn api_key_env_missing_variable_is_an_auth_error() {
        let auth = Auth::api_key_env(["BANSHU_AUTH_UNIT_DEFINITELY_UNSET"]);
        assert!(!auth.check().await.unwrap());
        let err = auth.resolve().await.unwrap_err();
        assert!(matches!(err, Error::Auth(_)));
        assert!(err.to_string().to_lowercase().contains("api key"));
    }

    #[tokio::test]
    async fn custom_resolver_delegates() {
        let auth = Auth::custom(Arc::new(StaticResolver("sk-abc")));
        assert_eq!(
            auth.resolve().await.unwrap().api_key.as_deref(),
            Some("sk-abc")
        );

        let failing = Auth::custom(Arc::new(FailingResolver));
        assert!(!failing.check().await.unwrap());
        assert!(failing.resolve().await.is_err());
    }

    #[tokio::test]
    async fn explicit_key_short_circuits_a_failing_resolver() {
        let auth = Auth::custom(Arc::new(FailingResolver));
        let resolved = resolve_for_request(&auth, Some("explicit".into()))
            .await
            .expect("explicit key should bypass the resolver");
        assert_eq!(resolved.api_key.as_deref(), Some("explicit"));
    }
}
