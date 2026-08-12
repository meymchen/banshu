//! Pluggable authentication.
//!
//! A provider resolves credentials through an [`AuthResolver`] rather than a
//! fixed environment variable. The built-in adapters cover the common cases:
//! [`Auth::api_key_env`] (the historical behaviour — the first set variable in
//! a fallback list), [`Auth::keyless`] (local servers that need no credentials,
//! e.g. llama.cpp / vLLM), [`Auth::oauth`] (a stored OAuth credential with
//! request-time refresh, optionally behind priority env vars), and
//! [`Auth::custom`] for anything else.
//!
//! Resolution runs in-band inside the stream, so a resolver failure surfaces as
//! a terminal [`ErrorKind::Auth`](crate::ErrorKind::Auth) error event rather
//! than a synchronous `Result` — [`Provider::stream`](crate::Provider::stream)
//! never fails up front.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{Error, Result};

/// Custom request headers contributed at provider, model, resolved-auth, and
/// request levels.
///
/// Names merge case-insensitively in the fixed priority chain documented on
/// [`StreamOptions::headers`](crate::StreamOptions::headers). `None` deletes a
/// same-named lower-priority header.
pub type ProviderHeaders = BTreeMap<String, Option<String>>;

pub(crate) struct RedactedHeaders<'a>(pub(crate) &'a ProviderHeaders);

impl std::fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = formatter.debug_map();
        for (name, value) in self.0 {
            if is_sensitive_header_name(name) {
                map.entry(name, &value.as_ref().map(|_| "[REDACTED]"));
            } else {
                map.entry(name, value);
            }
        }
        map.finish()
    }
}

pub(crate) fn is_sensitive_header_name(name: &str) -> bool {
    let lowercase = name.to_ascii_lowercase();
    let compact = lowercase
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>();
    matches!(
        lowercase.as_str(),
        "authorization"
            | "proxy-authorization"
            | "cookie"
            | "set-cookie"
            | "x-auth-token"
            | "access-token"
            | "refresh-token"
            | "id-token"
            | "client-secret"
    ) || compact.contains("apikey")
        || compact.contains("accesstoken")
        || compact.contains("refreshtoken")
        || compact.contains("clientsecret")
}

/// Merge header layers from lowest to highest priority.
///
/// Header names compare case-insensitively. A higher-priority value replaces
/// both the value and casing of a lower-priority entry; `None` removes it.
pub(crate) fn merge_header_layers<'a>(
    layers: impl IntoIterator<Item = &'a ProviderHeaders>,
) -> ProviderHeaders {
    let mut merged = BTreeMap::<String, (String, String)>::new();
    for layer in layers {
        for (name, value) in layer {
            let normalized = name.to_ascii_lowercase();
            match value {
                Some(value) => {
                    merged.insert(normalized, (name.clone(), value.clone()));
                }
                None => {
                    merged.remove(&normalized);
                }
            }
        }
    }
    merged
        .into_values()
        .map(|(name, value)| (name, Some(value)))
        .collect()
}

/// Credentials and endpoint overrides produced by an [`AuthResolver`].
#[derive(Clone, Default)]
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

impl std::fmt::Debug for ResolvedAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedAuth")
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("headers", &RedactedHeaders(&self.headers))
            .field("base_url", &self.base_url)
            .finish()
    }
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
    /// An OAuth credential lifecycle; see [`OAuthAuth`].
    OAuth(OAuthAuth),
}

/// OAuth-backed authentication: an [`OAuthSession`](crate::OAuthSession) plus
/// an optional API-key env-var list that takes priority when set.
///
/// A set environment variable is an explicit operator choice and always wins
/// over the stored OAuth credential; only with none set does resolution fall
/// back to the credential (refreshing it at request time when expired).
#[derive(Clone)]
pub struct OAuthAuth {
    session: crate::oauth::OAuthSession,
    api_key_env: Vec<String>,
}

impl OAuthAuth {
    /// OAuth-only authentication over `session`.
    pub fn new(session: crate::oauth::OAuthSession) -> Self {
        Self {
            session,
            api_key_env: Vec::new(),
        }
    }

    /// Environment variables whose API key, when set, is used instead of the
    /// OAuth credential.
    pub fn with_api_key_env(mut self, vars: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.api_key_env = vars.into_iter().map(Into::into).collect();
        self
    }

    /// The session behind this adapter, for
    /// [`Models::login`](crate::Models::login) and friends.
    pub(crate) fn session(&self) -> &crate::oauth::OAuthSession {
        &self.session
    }

    fn env_api_key(&self) -> Option<String> {
        self.api_key_env
            .iter()
            .find_map(|name| std::env::var(name).ok())
    }
}

#[async_trait]
impl AuthResolver for OAuthAuth {
    async fn check(&self) -> Result<bool> {
        if self.env_api_key().is_some() {
            return Ok(true);
        }
        self.session.check_auth().await
    }

    async fn resolve(&self) -> Result<ResolvedAuth> {
        if let Some(key) = self.env_api_key() {
            return Ok(ResolvedAuth {
                api_key: Some(key),
                ..Default::default()
            });
        }
        self.session.resolve().await
    }
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

    /// OAuth-backed authentication over `session`. Add API-key env vars that
    /// take priority with [`OAuthAuth::with_api_key_env`].
    pub fn oauth(session: crate::oauth::OAuthSession) -> Self {
        Self::OAuth(OAuthAuth::new(session))
    }

    /// Synchronous best-effort key lookup, used by the list-models probe (which
    /// needs the key string, not just its presence). Only [`Auth::ApiKeyEnv`]
    /// can answer synchronously; keyless has no key, and custom resolvers
    /// require async resolution.
    pub(crate) fn env_api_key(&self) -> Option<String> {
        match self {
            Self::ApiKeyEnv(vars) => vars.iter().find_map(|name| std::env::var(name).ok()),
            Self::OAuth(auth) => auth.env_api_key(),
            _ => None,
        }
    }

    /// Best-effort *synchronous* availability, for
    /// [`Provider::is_available`](crate::Provider::is_available). Keyless is
    /// always available; api-key-env is available when a listed variable is
    /// set. A custom resolver can only answer via the async
    /// [`AuthResolver::check`], so it reports `false` here — the async
    /// [`Models::available`](crate::Models::available) consults it instead.
    pub(crate) fn is_available(&self) -> bool {
        match self {
            Self::ApiKeyEnv(vars) => vars.iter().any(|name| std::env::var(name).is_ok()),
            Self::Keyless => true,
            Self::Custom(_) => false,
            Self::OAuth(auth) => auth.env_api_key().is_some(),
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
            Self::OAuth(auth) => auth.check().await,
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
            Self::OAuth(auth) => auth.resolve().await,
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
