//! Kimi For Coding's OAuth flow: RFC 8628 device authorization against the
//! fixed Kimi auth contract.
//!
//! The contract is pinned by the provider — the same one the official `kimi`
//! CLI speaks: a fixed public client id, `POST /api/oauth/device_authorization`
//! to start a login, and `POST /api/oauth/token` for both the device-code poll
//! and the refresh-token grant, all against the configured auth host
//! ([`KIMI_AUTH_HOST`] by default). Tokens obtained here authenticate inference
//! at the Kimi coding endpoint as `Authorization: Bearer` (see
//! [`OAuthSession::resolve`](crate::OAuthSession)).
//!
//! The auth host is overridable only through
//! [`with_auth_host`](KimiDeviceFlow::with_auth_host) — an explicit,
//! validated, construction-time choice for tests and controlled environments.
//! Nothing at request time (options, headers, model metadata) can redirect it.

use std::time::Duration;

use async_trait::async_trait;

use crate::credentials::OAuthCredential;
use crate::error::{Error, Result};
use crate::oauth::{AuthInteraction, OAuthFlow, RefreshError, VerificationDetails};

/// The fixed public OAuth client id of the Kimi auth contract — the one the
/// official `kimi` CLI uses, confirmed against its binary.
pub const KIMI_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";

/// The fixed Kimi auth host.
pub const KIMI_AUTH_HOST: &str = "https://auth.kimi.com";

/// The device authorization endpoint path (RFC 8628 section 3.1).
const DEVICE_AUTHORIZATION_PATH: &str = "/api/oauth/device_authorization";

/// The token endpoint path — both the device-code poll and the refresh grant.
const TOKEN_PATH: &str = "/api/oauth/token";

/// The device-code grant type (RFC 8628 section 3.4).
const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// The poll interval when the device authorization response names none.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// What a `slow_down` answer adds to the poll interval (RFC 8628 section 3.5).
const SLOW_DOWN_PENALTY: Duration = Duration::from_secs(5);

/// The expiry failure — the same whether the server answered `expired_token`
/// or the device code's own `expires_in` elapsed while polling.
const DEVICE_CODE_EXPIRED: &str =
    "the Kimi device code expired before authorization completed; start a new login";

/// The RFC 8628 device authorization flow for Kimi For Coding.
///
/// Stateless and cheap to construct; the credential lifecycle (storage,
/// single-flight refresh, expiry checks) lives in
/// [`OAuthSession`](crate::OAuthSession), which this flow is handed to.
#[derive(Debug, Clone)]
pub struct KimiDeviceFlow {
    auth_host: String,
}

impl KimiDeviceFlow {
    /// The flow against the fixed Kimi auth host.
    pub fn new() -> Self {
        Self {
            auth_host: KIMI_AUTH_HOST.to_string(),
        }
    }

    /// Point the flow at a different auth host — an explicit, controlled/test
    /// configuration. The same rule as a credential-level resource URL
    /// applies: HTTPS only, with loopback HTTP tolerated for local test
    /// servers. Anything else is an [`Error::Config`] at construction, so a
    /// misconfiguration fails before any login traffic flows.
    pub fn with_auth_host(mut self, host: impl Into<String>) -> Result<Self> {
        let host = host.into();
        if !crate::credentials::is_valid_resource_url(&host) {
            return Err(Error::Config(format!(
                "Kimi auth host must be https:// (or http:// to a loopback host), got `{host}`"
            )));
        }
        self.auth_host = host.trim_end_matches('/').to_string();
        Ok(self)
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.auth_host)
    }
}

impl Default for KimiDeviceFlow {
    fn default() -> Self {
        Self::new()
    }
}

/// The device authorization response (RFC 8628 section 3.2).
#[derive(serde::Deserialize)]
struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: Option<String>,
    verification_uri_complete: Option<String>,
    expires_in: Option<i64>,
    interval: Option<i64>,
}

/// A token endpoint success body: `access_token` is required, everything else
/// optional. A refresh response that omits the refresh token inherits the
/// prior one (the session's merge rule).
#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
}

/// An OAuth error body (`{"error": "…"}`). Only the error code is ever read:
/// it comes from a fixed vocabulary, so it is safe to surface. The rest of
/// the body — descriptions, URIs — never reaches an error message, keeping
/// token material out of logs and diagnostics.
#[derive(serde::Deserialize)]
struct OAuthErrorBody {
    error: String,
}

/// Read the error code out of a token-endpoint error body. Only the code is
/// ever surfaced: it comes from a fixed vocabulary, while the rest of the
/// body — descriptions, URIs — could carry material that must never reach a
/// log.
async fn read_error_code(response: reqwest::Response) -> Option<String> {
    response
        .json::<OAuthErrorBody>()
        .await
        .ok()
        .map(|body| body.error)
}

impl TokenResponse {
    fn into_credential(self) -> OAuthCredential {
        let expires_at = self
            .expires_in
            .map(|secs| crate::types::now_ms() + secs.max(0) * 1000);
        OAuthCredential::new(self.access_token, self.refresh_token, expires_at)
    }
}

#[async_trait]
impl OAuthFlow for KimiDeviceFlow {
    async fn login(
        &self,
        http: &reqwest::Client,
        interaction: &AuthInteraction,
    ) -> Result<OAuthCredential> {
        let response = http
            .post(self.endpoint(DEVICE_AUTHORIZATION_PATH))
            .form(&[("client_id", KIMI_CLIENT_ID)])
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Auth(format!(
                "Kimi device authorization failed: HTTP {status}"
            )));
        }
        let device: DeviceAuthorization = response.json().await.map_err(|_| {
            Error::Auth("Kimi device authorization returned a malformed response".into())
        })?;

        // The complete URI already carries the user code, so it is the better
        // thing to open or display when the server sends one.
        let url = device
            .verification_uri_complete
            .or(device.verification_uri)
            .ok_or_else(|| {
                Error::Auth("Kimi device authorization returned no verification URI".into())
            })?;
        interaction
            .show_verification(&VerificationDetails {
                url: url.clone(),
                user_code: Some(device.user_code),
                instructions: None,
            })
            .await?;
        // Best-effort: a declined or failed browser open never fails the
        // login — the printed instructions suffice.
        let _ = interaction.open_browser(&url).await;

        let deadline = device
            .expires_in
            .map(|secs| tokio::time::Instant::now() + Duration::from_secs(secs.max(0) as u64));
        let mut interval = device
            .interval
            .map(|secs| Duration::from_secs(secs.max(1) as u64))
            .unwrap_or(DEFAULT_POLL_INTERVAL);
        let device_code = device.device_code;
        let token_url = self.endpoint(TOKEN_PATH);

        interaction
            .wait(async {
                loop {
                    if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
                        return Err(Error::Auth(DEVICE_CODE_EXPIRED.into()));
                    }
                    let response = http
                        .post(&token_url)
                        .form(&[
                            ("grant_type", DEVICE_GRANT_TYPE),
                            ("client_id", KIMI_CLIENT_ID),
                            ("device_code", device_code.as_str()),
                        ])
                        .send()
                        .await?;
                    let status = response.status();
                    if status.is_success() {
                        let token: TokenResponse = response.json().await.map_err(|_| {
                            Error::Auth(
                                "the Kimi token endpoint returned a malformed response".into(),
                            )
                        })?;
                        return Ok(token.into_credential());
                    }
                    let error = read_error_code(response).await;
                    match error.as_deref() {
                        Some("authorization_pending") => {}
                        Some("slow_down") => interval += SLOW_DOWN_PENALTY,
                        Some("expired_token") => {
                            return Err(Error::Auth(DEVICE_CODE_EXPIRED.into()));
                        }
                        Some("access_denied") => {
                            return Err(Error::Auth(
                                "the user denied the Kimi authorization request".into(),
                            ));
                        }
                        Some(code) => {
                            return Err(Error::Auth(format!(
                                "Kimi device authorization failed: {code}"
                            )));
                        }
                        None => {
                            return Err(Error::Auth(format!(
                                "the Kimi token endpoint returned HTTP {status}"
                            )));
                        }
                    }
                    interaction
                        .report_status("waiting for Kimi authorization")
                        .await;
                    tokio::time::sleep(interval).await;
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
            .post(self.endpoint(TOKEN_PATH))
            .form(&[
                ("grant_type", "refresh_token"),
                ("client_id", KIMI_CLIENT_ID),
                ("refresh_token", refresh_token.as_str()),
            ])
            .send()
            .await
            .map_err(RefreshError::from)?;
        let status = response.status();
        if status.is_success() {
            let token: TokenResponse = response.json().await.map_err(|_| {
                RefreshError::Transient(
                    "the Kimi token endpoint returned a malformed response".into(),
                )
            })?;
            return Ok(token.into_credential());
        }
        let error = read_error_code(response).await;
        match error.as_deref() {
            Some("invalid_grant") => Err(RefreshError::Invalid("invalid_grant".into())),
            Some(code) => Err(RefreshError::Transient(format!(
                "the Kimi token endpoint rejected the refresh: {code}"
            ))),
            None => Err(RefreshError::Transient(format!(
                "the Kimi token endpoint returned HTTP {status}"
            ))),
        }
    }
}
