//! Credential storage: the type-tagged [`Credential`] an application persists,
//! and the [`CredentialStore`] seam it persists it through.
//!
//! The crate owns the in-memory implementation ([`InMemoryCredentialStore`])
//! and the coordination rules — a [`modify`](CredentialStore::modify) call is
//! a serialized read-modify-write, so a refresh-token rotation can never be
//! torn by a concurrent write. Applications that want durable or encrypted
//! storage implement [`CredentialStore`] themselves and inject it; the OAuth
//! lifecycle ([`OAuthSession`](crate::OAuthSession)) works against the trait,
//! never against a concrete store.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;

use crate::error::{Error, Result};

/// A stored credential for one provider, tagged by kind.
///
/// The serialized shape (`type: "apiKey"` / `type: "oauth"`, camelCase fields)
/// is the contract an application-side durable store round-trips. `Debug` is
/// hand-written and redacted: secrets never appear in logs.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum Credential {
    /// A static API key.
    #[serde(rename = "apiKey")]
    ApiKey(ApiKeyCredential),
    /// OAuth access/refresh tokens.
    #[serde(rename = "oauth")]
    OAuth(OAuthCredential),
}

impl Credential {
    /// The OAuth tokens, when this credential is an OAuth one.
    pub fn as_oauth(&self) -> Option<&OAuthCredential> {
        match self {
            Self::OAuth(credential) => Some(credential),
            Self::ApiKey(_) => None,
        }
    }
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(credential) => credential.fmt(formatter),
            Self::OAuth(credential) => credential.fmt(formatter),
        }
    }
}

/// A static API key.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ApiKeyCredential {
    /// The key material.
    pub key: String,
}

impl std::fmt::Debug for ApiKeyCredential {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ApiKeyCredential")
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// OAuth tokens as stored between runs.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthCredential {
    /// The token that authenticates requests.
    pub access_token: String,
    /// The token that mints new access tokens, when the flow issued one. A
    /// refresh response that omits it carries the prior one over — servers
    /// commonly rotate access tokens without re-issuing refresh tokens.
    pub refresh_token: Option<String>,
    /// When the access token stops working, as unix milliseconds. `None`
    /// means the flow attested no expiry, and the token is used as-is.
    pub expires_at: Option<i64>,
    /// A credential-level base URL the provider endpoint is overridden with
    /// (HTTPS only, enforced by [`with_resource_url`](Self::with_resource_url)).
    #[serde(default)]
    pub resource_url: Option<String>,
}

impl OAuthCredential {
    /// Tokens with no credential-level base URL.
    pub fn new(
        access_token: impl Into<String>,
        refresh_token: Option<String>,
        expires_at: Option<i64>,
    ) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token,
            expires_at,
            resource_url: None,
        }
    }

    /// Attach a credential-level base URL. Only `https://` URLs are accepted —
    /// a credential must never redirect requests onto a plaintext or
    /// non-HTTP(S) endpoint — with one exception: `http://` to a loopback
    /// host (`localhost`, `127.0.0.1`, `::1`) for local development and test
    /// servers. Anything else is an [`Error::Config`](crate::Error::Config),
    /// so an invalid scheme fails at construction rather than mid-request.
    pub fn with_resource_url(mut self, url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        if !is_valid_resource_url(&url) {
            return Err(Error::Config(format!(
                "credential resource URL must be https:// (or http:// to a loopback host), got `{url}`"
            )));
        }
        self.resource_url = Some(url);
        Ok(self)
    }

    /// Whether the access token is expired at `now`, or will be within
    /// `leeway`. A credential with no attested expiry never expires; one with
    /// a malformed (negative) timestamp is treated as already expired.
    pub fn expires_within(&self, now: SystemTime, leeway: Duration) -> bool {
        let Some(expires_at) = self.expires_at else {
            return false;
        };
        if expires_at < 0 {
            return true;
        }
        let expiry = UNIX_EPOCH + Duration::from_millis(expires_at as u64);
        now.checked_add(leeway)
            .is_none_or(|deadline| deadline >= expiry)
    }

    /// The credential a refresh leaves behind: the refreshed tokens, with the
    /// refresh token and resource URL carried over from `prior` when the
    /// refresh response omitted them.
    pub(crate) fn merged_from_refresh(self, prior: &Self) -> Self {
        Self {
            refresh_token: self.refresh_token.or_else(|| prior.refresh_token.clone()),
            resource_url: self.resource_url.or_else(|| prior.resource_url.clone()),
            ..self
        }
    }
}

impl std::fmt::Debug for OAuthCredential {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OAuthCredential")
            .field("access_token", &"[REDACTED]")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("expires_at", &self.expires_at)
            .field("resource_url", &self.resource_url)
            .finish()
    }
}

/// `https://` always; `http://` only to a loopback host.
pub(crate) fn is_valid_resource_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    if lower.starts_with("https://") {
        return true;
    }
    let Some(authority) = lower.strip_prefix("http://") else {
        return false;
    };
    let authority = authority.split('/').next().unwrap_or("");
    let host = match authority.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => authority.split(':').next().unwrap_or(""),
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// The update half of a [`CredentialStore::modify`] call: handed the current
/// credential (`None` when absent), it returns the credential to store
/// (`None` deletes).
pub type ModifyCredential<'a> =
    Box<dyn FnOnce(Option<Credential>) -> Result<Option<Credential>> + Send + 'a>;

/// Application-injectable credential storage.
///
/// `modify` is the only write path with a read: implementations must
/// serialize it per provider id so the read-modify-write is atomic — that is
/// what makes a refresh-token rotation safe under concurrent requests.
/// `get`/`list`/`delete` may be eventually consistent with one another, but a
/// completed `modify` must be visible to the next `modify`.
#[async_trait]
pub trait CredentialStore: Send + Sync {
    /// The credential stored for `provider_id`, if any.
    async fn get(&self, provider_id: &str) -> Result<Option<Credential>>;

    /// The provider ids that currently have a credential.
    async fn list(&self) -> Result<Vec<String>>;

    /// Atomically read the current credential for `provider_id`, run `update`,
    /// and store its result (`None` deletes). Returns the stored result.
    async fn modify(
        &self,
        provider_id: &str,
        update: ModifyCredential<'_>,
    ) -> Result<Option<Credential>>;

    /// Remove the credential for `provider_id`. Deleting an absent credential
    /// is not an error.
    async fn delete(&self, provider_id: &str) -> Result<()>;
}

/// The crate-owned [`CredentialStore`]: process-local, lost on exit.
///
/// Modifications hold the lock across the whole read-modify-write, so
/// concurrent rotations serialize instead of tearing.
#[derive(Default)]
pub struct InMemoryCredentialStore {
    credentials: Mutex<HashMap<String, Credential>>,
}

impl InMemoryCredentialStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for InMemoryCredentialStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = formatter.debug_list();
        if let Ok(credentials) = self.credentials.lock() {
            list.entries(credentials.keys());
        }
        list.finish()
    }
}

#[async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn get(&self, provider_id: &str) -> Result<Option<Credential>> {
        Ok(self
            .credentials
            .lock()
            .expect("credential store lock poisoned")
            .get(provider_id)
            .cloned())
    }

    async fn list(&self) -> Result<Vec<String>> {
        let mut ids: Vec<String> = self
            .credentials
            .lock()
            .expect("credential store lock poisoned")
            .keys()
            .cloned()
            .collect();
        ids.sort();
        Ok(ids)
    }

    async fn modify(
        &self,
        provider_id: &str,
        update: ModifyCredential<'_>,
    ) -> Result<Option<Credential>> {
        let mut credentials = self
            .credentials
            .lock()
            .expect("credential store lock poisoned");
        let next = update(credentials.get(provider_id).cloned())?;
        match next.clone() {
            Some(credential) => credentials.insert(provider_id.to_string(), credential),
            None => credentials.remove(provider_id),
        };
        Ok(next)
    }

    async fn delete(&self, provider_id: &str) -> Result<()> {
        self.credentials
            .lock()
            .expect("credential store lock poisoned")
            .remove(provider_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oauth() -> Credential {
        Credential::OAuth(
            OAuthCredential::new(
                "access-secret",
                Some("refresh-secret".into()),
                Some(1_900_000),
            )
            .with_resource_url("https://inference.example.com")
            .unwrap(),
        )
    }

    #[test]
    fn credential_serde_round_trip_keeps_type_tag() {
        let api_key = Credential::ApiKey(ApiKeyCredential {
            key: "sk-secret".into(),
        });
        let json = serde_json::to_value(&api_key).unwrap();
        assert_eq!(json["type"], "apiKey");
        assert_eq!(json["key"], "sk-secret");
        assert_eq!(serde_json::from_value::<Credential>(json).unwrap(), api_key);

        let json = serde_json::to_value(oauth()).unwrap();
        assert_eq!(json["type"], "oauth");
        assert_eq!(json["accessToken"], "access-secret");
        assert_eq!(json["refreshToken"], "refresh-secret");
        assert_eq!(json["expiresAt"], 1_900_000);
        assert_eq!(json["resourceUrl"], "https://inference.example.com");
        assert_eq!(serde_json::from_value::<Credential>(json).unwrap(), oauth());
    }

    #[test]
    fn resource_url_defaults_when_absent_from_json() {
        let credential: Credential = serde_json::from_value(serde_json::json!({
            "type": "oauth",
            "accessToken": "a",
            "refreshToken": null,
            "expiresAt": null,
        }))
        .unwrap();
        assert_eq!(credential.as_oauth().unwrap().resource_url, None);
    }

    #[test]
    fn debug_redacts_every_secret() {
        let api_key = Credential::ApiKey(ApiKeyCredential {
            key: "sk-secret".into(),
        });
        let rendered = format!("{api_key:?}");
        assert!(rendered.contains("[REDACTED]"));
        assert!(!rendered.contains("sk-secret"));

        let rendered = format!("{:?}", oauth());
        assert!(rendered.contains("OAuthCredential"));
        assert!(rendered.contains("1900000"));
        assert!(rendered.contains("https://inference.example.com"));
        assert!(!rendered.contains("access-secret"));
        assert!(!rendered.contains("refresh-secret"));
    }

    #[test]
    fn resource_url_must_be_https() {
        assert!(
            OAuthCredential::new("a", None, None)
                .with_resource_url("https://ok.example.com")
                .is_ok()
        );
        // Plain HTTP is loopback-only: local dev and test servers.
        for loopback in [
            "http://127.0.0.1:57166",
            "http://localhost:3000/base",
            "http://[::1]:8080",
        ] {
            assert!(
                OAuthCredential::new("a", None, None)
                    .with_resource_url(loopback)
                    .is_ok(),
                "{loopback}"
            );
        }
        for bad in [
            "http://plain.example.com",
            "http://192.168.0.10",
            "ftp://x",
            "not a url",
        ] {
            let err = OAuthCredential::new("a", None, None)
                .with_resource_url(bad)
                .unwrap_err();
            assert!(matches!(err, Error::Config(_)), "{bad}");
        }
    }

    #[test]
    fn expires_within_honours_leeway_and_missing_expiry() {
        let now = UNIX_EPOCH + Duration::from_millis(1_000_000);
        let soon = OAuthCredential::new("a", None, Some(1_000_500));
        assert!(!soon.expires_within(now, Duration::ZERO));
        assert!(soon.expires_within(now, Duration::from_secs(1)));

        let never = OAuthCredential::new("a", None, None);
        assert!(!never.expires_within(now, Duration::from_secs(3600)));

        let malformed = OAuthCredential::new("a", None, Some(-1));
        assert!(malformed.expires_within(now, Duration::ZERO));
    }

    #[tokio::test]
    async fn store_get_modify_list_delete() {
        let store = InMemoryCredentialStore::new();
        assert_eq!(store.get("kimi").await.unwrap(), None);
        assert_eq!(store.list().await.unwrap(), Vec::<String>::new());

        let stored = store
            .modify(
                "kimi",
                Box::new(|current| {
                    assert_eq!(current, None);
                    Ok(Some(oauth()))
                }),
            )
            .await
            .unwrap();
        assert_eq!(stored, Some(oauth()));
        assert_eq!(store.get("kimi").await.unwrap(), Some(oauth()));
        assert_eq!(store.list().await.unwrap(), vec!["kimi".to_string()]);

        // A modify returning None deletes.
        let stored = store.modify("kimi", Box::new(|_| Ok(None))).await.unwrap();
        assert_eq!(stored, None);
        assert_eq!(store.get("kimi").await.unwrap(), None);

        store
            .modify("kimi", Box::new(|_| Ok(Some(oauth()))))
            .await
            .unwrap();
        store.delete("kimi").await.unwrap();
        assert_eq!(store.get("kimi").await.unwrap(), None);
        // Deleting an absent credential is not an error.
        store.delete("kimi").await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_modify_serializes_refresh_token_rotation() {
        use std::sync::Arc;

        let store = Arc::new(InMemoryCredentialStore::new());
        store
            .modify(
                "kimi",
                Box::new(|_| {
                    Ok(Some(Credential::OAuth(OAuthCredential::new(
                        "access-0",
                        Some("refresh-0".into()),
                        None,
                    ))))
                }),
            )
            .await
            .unwrap();

        // Every task rotates the refresh token it actually read. Without a
        // serialized modify, rotations would be lost and the final counter
        // would fall short.
        const TASKS: usize = 8;
        const ROTATIONS: usize = 25;
        let mut handles = Vec::new();
        for _ in 0..TASKS {
            let store = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                for _ in 0..ROTATIONS {
                    store
                        .modify(
                            "kimi",
                            Box::new(|current| {
                                let Some(Credential::OAuth(credential)) = current else {
                                    panic!("credential vanished mid-rotation");
                                };
                                let prior = credential
                                    .refresh_token
                                    .unwrap()
                                    .trim_start_matches("refresh-")
                                    .parse::<u64>()
                                    .unwrap();
                                Ok(Some(Credential::OAuth(OAuthCredential::new(
                                    credential.access_token,
                                    Some(format!("refresh-{}", prior + 1)),
                                    None,
                                ))))
                            }),
                        )
                        .await
                        .unwrap();
                }
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }

        let Some(Credential::OAuth(credential)) = store.get("kimi").await.unwrap() else {
            panic!("credential vanished");
        };
        assert_eq!(
            credential.refresh_token.as_deref(),
            Some(format!("refresh-{}", TASKS * ROTATIONS).as_str())
        );
    }
}
