//! MiniMax Coding Plan OAuth: the frozen portal contract for the explicit CN
//! and Global regions.
//!
//! The contract is pinned by the PRD against MiniMax's official OpenClaw
//! integration: a fixed public client id and scope, `POST /oauth/code` with
//! PKCE S256 and a random state to start a login, and `POST /oauth/token` for
//! both the `user_code` poll and the refresh-token grant, all against the
//! region's portal host. Tokens obtained here authenticate Anthropic-compatible
//! inference at the region's endpoint with every header that endpoint requires
//! — `Authorization: Bearer` and `x-api-key` alike (see
//! [`MiniMaxPortalFlow::token_headers`]).
//!
//! The region is an explicit caller choice ([`MiniMaxRegion`]), never inferred
//! from IP or any other signal. The portal host is overridable only through
//! [`with_portal`](MiniMaxPortalFlow::with_portal) — an explicit, validated,
//! construction-time choice for tests and controlled environments. Nothing at
//! request time can redirect it.

use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::Digest;

use crate::auth::ProviderHeaders;
use crate::credentials::OAuthCredential;
use crate::error::{Error, Result};
use crate::oauth::{AuthInteraction, OAuthFlow, RefreshError, VerificationDetails};

/// The fixed public OAuth client id of the MiniMax Coding Plan contract — the
/// one MiniMax's official OpenClaw integration uses.
pub const MINIMAX_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";

/// The fixed OAuth scope of the MiniMax Coding Plan contract.
pub const MINIMAX_OAUTH_SCOPE: &str = "group_id profile model.completion";

/// The frozen polling grant type of the token endpoint.
const USER_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:user_code";

/// The poll interval when the authorization response names none — and the
/// floor a server-named interval is raised to after the first poll, so a
/// misbehaving portal cannot turn a long login into a hammer.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(2000);

/// `expired_in` below this reads as relative seconds.
const RELATIVE_EXPIRY_SECONDS_THRESHOLD: i64 = 1_000_000_000;

/// `expired_in` at or above this reads as absolute milliseconds; between the
/// two thresholds it reads as absolute seconds.
const ABSOLUTE_EXPIRY_MS_THRESHOLD: i64 = 1_000_000_000_000;

/// The expiry failure: the authorization's own absolute-millisecond deadline
/// elapsed while polling.
const USER_CODE_EXPIRED: &str =
    "the MiniMax user code expired before authorization completed; start a new login";

/// An explicit MiniMax Coding Plan region. There is no default and no
/// inference from IP: the caller names the region, and the region names the
/// portal and inference hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxRegion {
    /// Mainland China: portal `https://api.minimaxi.com`, inference
    /// `https://api.minimaxi.com/anthropic`.
    Cn,
    /// Everywhere else: portal `https://api.minimax.io`, inference
    /// `https://api.minimax.io/anthropic`.
    Global,
}

impl MiniMaxRegion {
    /// The region's frozen OAuth portal host.
    pub fn portal(self) -> &'static str {
        match self {
            Self::Cn => "https://api.minimaxi.com",
            Self::Global => "https://api.minimax.io",
        }
    }

    /// The region's frozen Anthropic-compatible inference endpoint.
    pub fn inference_base_url(self) -> &'static str {
        match self {
            Self::Cn => "https://api.minimaxi.com/anthropic",
            Self::Global => "https://api.minimax.io/anthropic",
        }
    }

    /// The provider id [`Provider::minimax`](crate::Provider::minimax) registers
    /// the region under.
    pub fn provider_id(self) -> &'static str {
        match self {
            Self::Cn => "minimax-cn",
            Self::Global => "minimax",
        }
    }

    /// The region's display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cn => "MiniMax CN",
            Self::Global => "MiniMax",
        }
    }
}

/// The frozen MiniMax Coding Plan login flow for one region.
///
/// Stateless and cheap to construct; the credential lifecycle (storage,
/// single-flight refresh, expiry checks) lives in
/// [`OAuthSession`](crate::OAuthSession), which this flow is handed to.
#[derive(Debug, Clone)]
pub struct MiniMaxPortalFlow {
    portal: String,
}

impl MiniMaxPortalFlow {
    /// The flow against the region's frozen portal host.
    pub fn new(region: MiniMaxRegion) -> Self {
        Self {
            portal: region.portal().to_string(),
        }
    }

    /// Point the flow at a different portal host — an explicit, controlled/test
    /// configuration. The same rule as a credential-level resource URL applies:
    /// HTTPS only, with loopback HTTP tolerated for local test servers.
    /// Anything else is an [`Error::Config`] at construction, so a
    /// misconfiguration fails before any login traffic flows.
    pub fn with_portal(mut self, host: impl Into<String>) -> Result<Self> {
        let host = host.into();
        if !crate::credentials::is_valid_resource_url(&host) {
            return Err(Error::Config(format!(
                "MiniMax portal host must be https:// (or http:// to a loopback host), got `{host}`"
            )));
        }
        self.portal = host.trim_end_matches('/').to_string();
        Ok(self)
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.portal)
    }
}

/// A PKCE S256 verifier/challenge pair plus the random anti-CSRF state, all
/// base64url without padding.
struct Pkce {
    verifier: String,
    challenge: String,
    state: String,
}

fn random_base64url(bytes: usize) -> String {
    let mut buffer = vec![0u8; bytes];
    getrandom::fill(&mut buffer).expect("the OS random source is available");
    URL_SAFE_NO_PAD.encode(buffer)
}

fn generate_pkce() -> Pkce {
    let verifier = random_base64url(32);
    let challenge = URL_SAFE_NO_PAD.encode(sha2::Sha256::digest(verifier.as_bytes()));
    let state = random_base64url(16);
    Pkce {
        verifier,
        challenge,
        state,
    }
}

/// The `/oauth/code` response. `expired_in` is the contract's
/// absolute-millisecond deadline; `interval` a poll interval in milliseconds.
#[derive(serde::Deserialize)]
struct Authorization {
    user_code: Option<String>,
    verification_uri: Option<String>,
    expired_in: Option<i64>,
    interval: Option<i64>,
    state: Option<String>,
}

/// The token endpoint's payload: a fixed `status` vocabulary, with tokens on
/// `success`. Only `status` is ever surfaced — the rest of an error body
/// (`base_resp`, free-form text) could carry material that must never reach a
/// log.
#[derive(serde::Deserialize)]
struct TokenPayload {
    status: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    expired_in: Option<i64>,
    resource_url: Option<String>,
}

/// Normalize the token endpoint's `expired_in` to absolute unix milliseconds:
/// relative seconds below a billion, absolute seconds below a trillion,
/// absolute milliseconds from there up. Anything else attests nothing.
fn normalize_expires(expired_in: i64) -> Option<i64> {
    if expired_in <= 0 {
        return None;
    }
    if expired_in < RELATIVE_EXPIRY_SECONDS_THRESHOLD {
        Some(crate::types::now_ms() + expired_in * 1000)
    } else if expired_in < ABSOLUTE_EXPIRY_MS_THRESHOLD {
        Some(expired_in * 1000)
    } else {
        Some(expired_in)
    }
}

impl TokenPayload {
    /// The credential a `success` payload attests, or why it doesn't. A
    /// success without its tokens is malformed; an `https`-violating resource
    /// URL fails structurally.
    fn into_credential(self) -> Result<OAuthCredential> {
        let (Some(access_token), Some(refresh_token)) = (self.access_token, self.refresh_token)
        else {
            return Err(Error::Auth(
                "the MiniMax token endpoint returned an incomplete success payload".into(),
            ));
        };
        let Some(expires_at) = self.expired_in.and_then(normalize_expires) else {
            return Err(Error::Auth(
                "the MiniMax token endpoint returned an invalid expiry".into(),
            ));
        };
        let credential = OAuthCredential::new(access_token, Some(refresh_token), Some(expires_at));
        match self.resource_url {
            Some(url) => credential.with_resource_url(url),
            None => Ok(credential),
        }
    }
}

#[async_trait]
impl OAuthFlow for MiniMaxPortalFlow {
    async fn login(
        &self,
        http: &reqwest::Client,
        interaction: &AuthInteraction,
    ) -> Result<OAuthCredential> {
        let pkce = generate_pkce();
        let response = http
            .post(self.endpoint("/oauth/code"))
            .form(&[
                ("response_type", "code"),
                ("client_id", MINIMAX_CLIENT_ID),
                ("scope", MINIMAX_OAUTH_SCOPE),
                ("code_challenge", pkce.challenge.as_str()),
                ("code_challenge_method", "S256"),
                ("state", pkce.state.as_str()),
            ])
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Auth(format!(
                "MiniMax authorization failed: HTTP {status}"
            )));
        }
        let authorization: Authorization = response.json().await.map_err(|_| {
            Error::Auth("MiniMax authorization returned a malformed response".into())
        })?;
        // The state round-trip is the CSRF check: a portal that answers with
        // any other state is not answering this login.
        if authorization.state.as_deref() != Some(pkce.state.as_str()) {
            return Err(Error::Auth(
                "MiniMax OAuth state mismatch: the portal answered a different login".into(),
            ));
        }
        let (Some(user_code), Some(url)) =
            (authorization.user_code, authorization.verification_uri)
        else {
            return Err(Error::Auth(
                "MiniMax authorization returned an incomplete payload (missing user_code or verification_uri)"
                    .into(),
            ));
        };
        let Some(deadline_ms) = authorization.expired_in.filter(|deadline| *deadline > 0) else {
            return Err(Error::Auth(
                "MiniMax authorization returned an invalid expired_in".into(),
            ));
        };
        interaction
            .show_verification(&VerificationDetails {
                url: url.clone(),
                user_code: Some(user_code.clone()),
                instructions: None,
            })
            .await?;
        // Best-effort: a declined or failed browser open never fails the
        // login — the printed instructions suffice.
        let _ = interaction.open_browser(&url).await;

        let mut interval = authorization
            .interval
            .filter(|ms| *ms > 0)
            .map(|ms| Duration::from_millis(ms as u64))
            .unwrap_or(DEFAULT_POLL_INTERVAL);
        let token_url = self.endpoint("/oauth/token");

        interaction
            .wait(async {
                loop {
                    let remaining_ms = deadline_ms - crate::types::now_ms();
                    if remaining_ms <= 0 {
                        return Err(Error::Auth(USER_CODE_EXPIRED.into()));
                    }
                    let response = http
                        .post(&token_url)
                        .form(&[
                            ("grant_type", USER_CODE_GRANT),
                            ("client_id", MINIMAX_CLIENT_ID),
                            ("user_code", user_code.as_str()),
                            ("code_verifier", pkce.verifier.as_str()),
                        ])
                        .send()
                        .await?;
                    let status = response.status();
                    if !status.is_success() {
                        return Err(Error::Auth(format!(
                            "the MiniMax token endpoint returned HTTP {status}"
                        )));
                    }
                    let payload: TokenPayload = response.json().await.map_err(|_| {
                        Error::Auth(
                            "the MiniMax token endpoint returned a malformed response".into(),
                        )
                    })?;
                    match payload.status.as_deref() {
                        Some("success") => return payload.into_credential(),
                        Some("error") => {
                            return Err(Error::Auth(
                                "the MiniMax portal reported an authorization error".into(),
                            ));
                        }
                        // Every other status — the contract's `pending`
                        // included — keeps polling.
                        _ => {}
                    }
                    interaction
                        .report_status("waiting for MiniMax authorization")
                        .await;
                    tokio::time::sleep(interval.min(Duration::from_millis(remaining_ms as u64)))
                        .await;
                    interval = interval.max(DEFAULT_POLL_INTERVAL);
                }
            })
            .await
    }

    async fn refresh(
        &self,
        http: &reqwest::Client,
        credential: &OAuthCredential,
    ) -> std::result::Result<OAuthCredential, RefreshError> {
        let Some(refresh_token) = credential.refresh_token.clone() else {
            return Err(RefreshError::Invalid(
                "the stored credential has no refresh token".into(),
            ));
        };
        let response = http
            .post(self.endpoint("/oauth/token"))
            .form(&[
                ("grant_type", "refresh_token"),
                ("client_id", MINIMAX_CLIENT_ID),
                ("refresh_token", refresh_token.as_str()),
            ])
            .send()
            .await
            .map_err(RefreshError::from)?;
        let status = response.status();
        if status.is_client_error() {
            // A 4xx is the portal rejecting the refresh token itself; only a
            // fresh login produces a working credential.
            return Err(RefreshError::Invalid(format!(
                "the MiniMax token endpoint rejected the refresh: HTTP {status}"
            )));
        }
        if !status.is_success() {
            return Err(RefreshError::Transient(format!(
                "the MiniMax token endpoint returned HTTP {status}"
            )));
        }
        let payload: TokenPayload = response.json().await.map_err(|_| {
            RefreshError::Transient(
                "the MiniMax token endpoint returned a malformed response".into(),
            )
        })?;
        match payload.status.as_deref() {
            Some("success") => {}
            // The portal answered and rejected the grant.
            _ => {
                return Err(RefreshError::Invalid(
                    "the MiniMax portal rejected the refresh token".into(),
                ));
            }
        }
        let Some(access_token) = payload.access_token else {
            return Err(RefreshError::Transient(
                "the MiniMax token endpoint returned an incomplete success payload".into(),
            ));
        };
        let expires_at = payload.expired_in.and_then(normalize_expires);
        // A refresh response that omits the refresh token or resource URL
        // inherits the prior one (the session's merge rule).
        let credential = OAuthCredential::new(access_token, payload.refresh_token, expires_at);
        match payload.resource_url {
            // An HTTPS-violating resource URL fails structurally, as on the
            // login path — but as Transient, not Invalid: the stored
            // credential is unharmed, and a fresh login would only hit the
            // same structural rejection, so there is nothing to re-login for.
            Some(url) => credential
                .with_resource_url(url)
                .map_err(|err| RefreshError::Transient(err.to_string())),
            None => Ok(credential),
        }
    }

    fn token_headers(&self, access_token: &str) -> ProviderHeaders {
        // The MiniMax Anthropic-compatible endpoint requires the access token
        // on both its bearer and its API-key header; dropping either fails the
        // request, so the frozen contract maps the token onto both.
        ProviderHeaders::from([
            (
                "authorization".to_string(),
                Some(format!("Bearer {access_token}")),
            ),
            ("x-api-key".to_string(), Some(access_token.to_string())),
        ])
    }
}
