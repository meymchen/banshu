//! The OAuth lifecycle: interactive login driven through an
//! [`AuthInteraction`], request-time refresh coordinated by an
//! [`OAuthSession`], and logout.
//!
//! The crate owns the coordination — a login is a plain `Result`-returning
//! call (never a message stream), refreshes for one provider are single-flight
//! so concurrent requests share one HTTP operation, and a rejected refresh
//! token preserves the stored credential for diagnosis instead of deleting it.
//! The wire-level flow (device authorization, PKCE, …) is supplied per
//! provider as an [`OAuthFlow`]; durable storage is supplied by the
//! application as a [`CredentialStore`](crate::CredentialStore).

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures_util::future::{self, BoxFuture, Either, FutureExt, Shared};
use std::future::Future;
use tokio_util::sync::CancellationToken;

use crate::auth::ResolvedAuth;
use crate::credentials::{Credential, CredentialStore, OAuthCredential};
use crate::error::{Error, Result};

/// How long before its stated expiry an access token is refreshed. A request
/// dispatched with a token that dies mid-flight would fail for a reason the
/// caller cannot see, so the leeway is spent rather than the request.
pub(crate) const EXPIRY_LEEWAY: Duration = Duration::from_secs(60);

/// The default overall login timeout — five minutes for a human to complete a
/// device-code or browser round trip.
pub const DEFAULT_LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

/// What the user must do to authorize a login, shown by the application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationDetails {
    /// The URL the user visits.
    pub url: String,
    /// The code the user enters there, for flows that use one.
    pub user_code: Option<String>,
    /// Free-form instructions from the flow, when it has any.
    pub instructions: Option<String>,
}

/// The application's half of an interactive login.
///
/// Only [`show_verification`](Self::show_verification) is required — without
/// it the user can never learn what to do. Opening a browser is optional
/// (return `Ok(false)` and the flow proceeds without it); status reports may
/// be ignored.
#[async_trait]
pub trait AuthInteractionHandler: Send + Sync {
    /// Present the verification instructions to the user.
    async fn show_verification(&self, details: &VerificationDetails) -> Result<()>;

    /// Try to open `url` in a browser. Return `true` when handled; the
    /// default declines, and the flow proceeds on the printed instructions.
    async fn open_browser(&self, _url: &str) -> Result<bool> {
        Ok(false)
    }

    /// Report a progress line ("waiting for authorization…"). The default
    /// ignores it.
    async fn report_status(&self, _message: &str) {}
}

/// Everything a login flow needs to talk to the user, with the caller's
/// timeout and cancellation built in.
///
/// Flows run their polling inside [`wait`](Self::wait), which races the step
/// against the timeout and the cancellation token and reports which one ended
/// it — [`Error::AuthTimeout`] or [`Error::AuthCancelled`].
pub struct AuthInteraction {
    handler: Arc<dyn AuthInteractionHandler>,
    timeout: Duration,
    cancellation: CancellationToken,
}

impl AuthInteraction {
    /// An interaction over `handler` with the
    /// [default timeout](DEFAULT_LOGIN_TIMEOUT) and a fresh cancellation
    /// token.
    pub fn new(handler: Arc<dyn AuthInteractionHandler>) -> Self {
        Self {
            handler,
            timeout: DEFAULT_LOGIN_TIMEOUT,
            cancellation: CancellationToken::new(),
        }
    }

    /// Bound the whole login. When it elapses,
    /// [`wait`](Self::wait) fails with [`Error::AuthTimeout`].
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Cancel the login when `token` fires —
    /// [`wait`](Self::wait) fails with [`Error::AuthCancelled`].
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// Forward verification instructions to the application.
    pub async fn show_verification(&self, details: &VerificationDetails) -> Result<()> {
        self.handler.show_verification(details).await
    }

    /// Ask the application to open a browser; `false` means it did not.
    pub async fn open_browser(&self, url: &str) -> Result<bool> {
        self.handler.open_browser(url).await
    }

    /// Forward a progress line to the application.
    pub async fn report_status(&self, message: &str) {
        self.handler.report_status(message).await;
    }

    /// The caller's cancellation token, for flows that select on it directly.
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// The configured login timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Run one login step to completion, cancellation, or timeout —
    /// whichever comes first. Cancellation and timeout drop `step`.
    pub async fn wait<T>(&self, step: impl Future<Output = Result<T>> + Send) -> Result<T> {
        let cancellation = self.cancellation.clone();
        let raced = future::select(step.boxed(), cancellation.cancelled().boxed());
        match tokio::time::timeout(self.timeout, raced).await {
            Ok(Either::Left((result, _))) => result,
            Ok(Either::Right(((), _))) => Err(Error::AuthCancelled),
            Err(_) => Err(Error::AuthTimeout {
                seconds: self.timeout.as_secs(),
            }),
        }
    }
}

/// Why a token refresh failed. The distinction decides what happens to the
/// stored credential — and it is never deleted either way.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RefreshError {
    /// The server rejected the refresh token itself (OAuth `invalid_grant`).
    /// Only a fresh login can produce a working credential; the stored one is
    /// preserved for diagnosis and surfaces as
    /// [`Error::ReLoginRequired`].
    #[error("refresh token rejected: {0}")]
    Invalid(String),
    /// A transient failure — transport, a 5xx, a malformed response. The
    /// stored credential is preserved and a later request retries the
    /// refresh.
    #[error("refresh failed: {0}")]
    Transient(String),
}

impl From<reqwest::Error> for RefreshError {
    fn from(err: reqwest::Error) -> Self {
        Self::Transient(err.to_string())
    }
}

/// The provider-specific half of OAuth: how to log in and how to refresh.
///
/// Implemented once per vendor flow (device authorization, PKCE, …). The
/// [`OAuthSession`] owns everything around these two calls — storage,
/// single-flight coordination, expiry checks — so an implementation only ever
/// talks HTTP.
#[async_trait]
pub trait OAuthFlow: Send + Sync {
    /// Drive an interactive login, presenting instructions through
    /// `interaction`. Long polling must run inside
    /// [`AuthInteraction::wait`] so caller timeout and cancellation apply.
    async fn login(
        &self,
        http: &reqwest::Client,
        interaction: &AuthInteraction,
    ) -> Result<OAuthCredential>;

    /// Exchange `credential`'s refresh token for fresh tokens. A server
    /// rejection of the refresh token must be [`RefreshError::Invalid`];
    /// anything transient is [`RefreshError::Transient`]. A response that
    /// omits the refresh token or resource URL inherits the prior one.
    async fn refresh(
        &self,
        http: &reqwest::Client,
        credential: &OAuthCredential,
    ) -> std::result::Result<OAuthCredential, RefreshError>;

    /// The request headers a resolved access token authenticates with. The
    /// default is `Authorization: Bearer` alone — an OAuth access token is a
    /// bearer token (RFC 6750) on either wire protocol. A flow whose endpoint
    /// contract requires the token on further headers (MiniMax's
    /// Anthropic-compatible endpoint also requires `x-api-key`) overrides
    /// this; the headers merge into the request's fixed header chain like any
    /// resolved-auth layer.
    fn token_headers(&self, access_token: &str) -> crate::auth::ProviderHeaders {
        crate::auth::ProviderHeaders::from([(
            "authorization".to_string(),
            Some(format!("Bearer {access_token}")),
        )])
    }
}

/// A shared refresh in flight: every waiter clones the same future and
/// resolves to the same structured result.
type SharedRefresh =
    Shared<BoxFuture<'static, std::result::Result<OAuthCredential, Arc<RefreshError>>>>;

struct InFlightRefresh {
    generation: u64,
    future: SharedRefresh,
}

/// The OAuth lifecycle for one provider: login, logout, auth checks, and
/// request-time token resolution against an application-injected
/// [`CredentialStore`].
///
/// Cloning is cheap and shares the store handle and the single-flight slot —
/// clones of one session coordinate with each other.
#[derive(Clone)]
pub struct OAuthSession {
    inner: Arc<Inner>,
}

struct Inner {
    provider_id: String,
    flow: Arc<dyn OAuthFlow>,
    store: Arc<dyn CredentialStore>,
    http: reqwest::Client,
    /// The single-flight slot plus a monotonically increasing generation, so a
    /// waiter can clear only the refresh it actually waited on — never a
    /// newer one that has since taken the slot.
    in_flight: Mutex<(u64, Option<InFlightRefresh>)>,
}

impl OAuthSession {
    /// A session coordinating `flow` against `store`, using `http` for both
    /// login and refresh traffic.
    pub fn new(
        provider_id: impl Into<String>,
        flow: Arc<dyn OAuthFlow>,
        store: Arc<dyn CredentialStore>,
        http: reqwest::Client,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                provider_id: provider_id.into(),
                flow,
                store,
                http,
                in_flight: Mutex::new((0, None)),
            }),
        }
    }

    /// The provider this session logs in to.
    pub fn provider_id(&self) -> &str {
        &self.inner.provider_id
    }

    /// The credential store this session coordinates.
    pub fn store(&self) -> &Arc<dyn CredentialStore> {
        &self.inner.store
    }

    /// Run an interactive login and store the resulting credential,
    /// replacing any previous one. Cancellation and timeout come from
    /// `interaction`; the login is a plain call, never a message stream.
    pub async fn login(&self, interaction: &AuthInteraction) -> Result<OAuthCredential> {
        let credential = interaction
            .wait(self.inner.flow.login(&self.inner.http, interaction))
            .await?;
        let stored = credential.clone();
        self.inner
            .store
            .modify(
                &self.inner.provider_id,
                Box::new(move |_| Ok(Some(Credential::OAuth(stored)))),
            )
            .await?;
        Ok(credential)
    }

    /// Delete the stored credential. Logging out when never logged in is not
    /// an error.
    pub async fn logout(&self) -> Result<()> {
        self.inner.store.delete(&self.inner.provider_id).await
    }

    /// Whether a usable credential is stored — the check behind
    /// [`Models::check_auth`](crate::Models::check_auth). An expired OAuth
    /// credential still counts: request-time refresh will either renew it or
    /// fail with [`Error::ReLoginRequired`], both of which say the user did
    /// log in.
    pub async fn check_auth(&self) -> Result<bool> {
        Ok(matches!(
            self.inner.store.get(&self.inner.provider_id).await?,
            Some(Credential::OAuth(_))
        ))
    }

    /// The credential to authenticate a request with, refreshing first when
    /// the stored access token is expired (or nearly). A credential that
    /// cannot be refreshed fails with [`Error::ReLoginRequired`] and stays in
    /// the store.
    pub async fn resolve(&self) -> Result<ResolvedAuth> {
        let stored = self.inner.store.get(&self.inner.provider_id).await?;
        let Some(credential) = stored.as_ref().and_then(Credential::as_oauth).cloned() else {
            return Err(self.re_login_required("no OAuth credential is stored"));
        };
        let credential = if credential.expires_within(SystemTime::now(), EXPIRY_LEEWAY) {
            self.refresh().await?
        } else {
            credential
        };
        // The flow declares which headers its endpoint authenticates with —
        // `Authorization: Bearer` by default (RFC 6750), more where the
        // contract requires it — attached directly, never through `api_key`,
        // whose protocol-native placement (e.g. Anthropic's `x-api-key`) is
        // for API keys, not tokens.
        Ok(ResolvedAuth {
            api_key: None,
            headers: self.inner.flow.token_headers(&credential.access_token),
            base_url: credential.resource_url,
        })
    }

    /// Refresh the stored credential now, sharing one in-flight refresh
    /// across all concurrent callers: they wait on the same operation and
    /// resolve to the same structured result.
    pub async fn refresh(&self) -> Result<OAuthCredential> {
        let (generation, shared) = {
            let mut in_flight = self.inner.in_flight.lock().expect("refresh lock poisoned");
            match &in_flight.1 {
                Some(existing) => (existing.generation, existing.future.clone()),
                None => {
                    in_flight.0 += 1;
                    let generation = in_flight.0;
                    let future = self.shared_refresh();
                    in_flight.1 = Some(InFlightRefresh {
                        generation,
                        future: future.clone(),
                    });
                    (generation, future)
                }
            }
        };
        let result = shared.await;
        {
            let mut in_flight = self.inner.in_flight.lock().expect("refresh lock poisoned");
            // Clear only the refresh this call actually waited on — a later
            // one may already have taken the slot.
            if matches!(&in_flight.1, Some(current) if current.generation == generation) {
                in_flight.1 = None;
            }
        }
        result.map_err(|err| match &*err {
            RefreshError::Invalid(reason) => self.re_login_required(reason),
            RefreshError::Transient(message) => Error::Auth(format!(
                "token refresh failed for provider `{}`: {message}",
                self.inner.provider_id
            )),
        })
    }

    /// The single HTTP refresh operation behind [`refresh`](Self::refresh),
    /// boxed and `'static` so it can be shared.
    fn shared_refresh(&self) -> SharedRefresh {
        let this = self.clone();
        async move { this.refresh_once().await }.boxed().shared()
    }

    async fn refresh_once(&self) -> std::result::Result<OAuthCredential, Arc<RefreshError>> {
        let stored = self
            .inner
            .store
            .get(&self.inner.provider_id)
            .await
            .map_err(|err| Arc::new(RefreshError::Transient(err.to_string())))?;
        let Some(credential) = stored.as_ref().and_then(Credential::as_oauth).cloned() else {
            return Err(Arc::new(RefreshError::Invalid(
                "no OAuth credential is stored".into(),
            )));
        };
        if credential.refresh_token.is_none() {
            return Err(Arc::new(RefreshError::Invalid(
                "the stored credential has no refresh token".into(),
            )));
        }
        let refreshed = self
            .inner
            .flow
            .refresh(&self.inner.http, &credential)
            .await
            .map_err(Arc::new)?
            .merged_from_refresh(&credential);
        // Compare-and-swap: the rotation lands only if the credential is
        // still the one this refresh read. A login or logout that raced us
        // wins — its credential is fresher than anything this refresh could
        // produce — and every waiter resolves to the surviving credential.
        let expected = credential;
        let rotated = refreshed;
        let outcome = self
            .inner
            .store
            .modify(
                &self.inner.provider_id,
                Box::new(move |current| match current {
                    Some(Credential::OAuth(current)) if current == expected => {
                        Ok(Some(Credential::OAuth(rotated)))
                    }
                    surviving => Ok(surviving),
                }),
            )
            .await
            .map_err(|err| Arc::new(RefreshError::Transient(err.to_string())))?;
        match outcome {
            Some(Credential::OAuth(credential)) => Ok(credential),
            // A logout won the race: nothing to authenticate with any more.
            _ => Err(Arc::new(RefreshError::Invalid(
                "the credential changed while the refresh was in flight".into(),
            ))),
        }
    }

    fn re_login_required(&self, reason: impl Into<String>) -> Error {
        Error::ReLoginRequired {
            provider: self.inner.provider_id.clone(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::InMemoryCredentialStore;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct RecordingHandler {
        events: Mutex<Vec<String>>,
        browser: bool,
    }

    #[async_trait]
    impl AuthInteractionHandler for RecordingHandler {
        async fn show_verification(&self, details: &VerificationDetails) -> Result<()> {
            self.events.lock().unwrap().push(format!(
                "verify:{}:{}",
                details.url,
                details.user_code.as_deref().unwrap_or("-")
            ));
            Ok(())
        }
        async fn open_browser(&self, url: &str) -> Result<bool> {
            self.events.lock().unwrap().push(format!("browser:{url}"));
            Ok(self.browser)
        }
        async fn report_status(&self, message: &str) {
            self.events
                .lock()
                .unwrap()
                .push(format!("status:{message}"));
        }
    }

    #[derive(Clone)]
    enum LoginBehavior {
        Succeed(OAuthCredential),
        Pend,
    }

    struct FakeFlow {
        login: Mutex<LoginBehavior>,
        refresh_calls: AtomicUsize,
        refresh_delay: Duration,
        refresh_results: Mutex<VecDeque<std::result::Result<OAuthCredential, RefreshError>>>,
    }

    impl FakeFlow {
        fn logged_in(credential: OAuthCredential) -> Self {
            Self {
                login: Mutex::new(LoginBehavior::Succeed(credential)),
                refresh_calls: AtomicUsize::new(0),
                refresh_delay: Duration::ZERO,
                refresh_results: Mutex::new(VecDeque::new()),
            }
        }
    }

    #[async_trait]
    impl OAuthFlow for FakeFlow {
        async fn login(
            &self,
            _http: &reqwest::Client,
            interaction: &AuthInteraction,
        ) -> Result<OAuthCredential> {
            let behavior = self.login.lock().unwrap().clone();
            match behavior {
                LoginBehavior::Succeed(credential) => {
                    interaction
                        .show_verification(&VerificationDetails {
                            url: "https://example.com/device".into(),
                            user_code: Some("ABCD-EFGH".into()),
                            instructions: None,
                        })
                        .await?;
                    interaction
                        .open_browser("https://example.com/device")
                        .await?;
                    interaction.report_status("waiting for authorization").await;
                    Ok(credential)
                }
                LoginBehavior::Pend => future::pending().await,
            }
        }

        async fn refresh(
            &self,
            _http: &reqwest::Client,
            _credential: &OAuthCredential,
        ) -> std::result::Result<OAuthCredential, RefreshError> {
            self.refresh_calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.refresh_delay).await;
            self.refresh_results
                .lock()
                .unwrap()
                .pop_front()
                .expect("no scripted refresh result")
        }
    }

    fn session(flow: FakeFlow) -> (OAuthSession, Arc<InMemoryCredentialStore>, Arc<FakeFlow>) {
        let flow = Arc::new(flow);
        let store = Arc::new(InMemoryCredentialStore::new());
        let session =
            OAuthSession::new("test", flow.clone(), store.clone(), reqwest::Client::new());
        (session, store, flow)
    }

    fn fresh_credential() -> OAuthCredential {
        OAuthCredential::new(
            "access-fresh",
            Some("refresh-1".into()),
            Some(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64
                    + 3_600_000,
            ),
        )
    }

    fn expired_credential() -> OAuthCredential {
        OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000))
    }

    #[tokio::test]
    async fn login_drives_interaction_and_stores_credential() {
        let handler = Arc::new(RecordingHandler {
            browser: true,
            ..RecordingHandler::default()
        });
        let expected = fresh_credential();
        let (session, store, _) = session(FakeFlow::logged_in(expected.clone()));
        let interaction = AuthInteraction::new(handler.clone());

        let credential = session.login(&interaction).await.unwrap();
        assert_eq!(credential, expected);
        assert!(session.check_auth().await.unwrap());
        assert_eq!(
            store.get("test").await.unwrap(),
            Some(Credential::OAuth(expected))
        );
        assert_eq!(
            handler.events.lock().unwrap().as_slice(),
            [
                "verify:https://example.com/device:ABCD-EFGH",
                "browser:https://example.com/device",
                "status:waiting for authorization",
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn login_times_out() {
        let mut flow = FakeFlow::logged_in(fresh_credential());
        *flow.login.get_mut().unwrap() = LoginBehavior::Pend;
        let (session, _, _) = session(flow);
        let interaction = AuthInteraction::new(Arc::new(RecordingHandler::default()))
            .with_timeout(Duration::from_secs(30));

        let err = session.login(&interaction).await.unwrap_err();
        assert!(matches!(err, Error::AuthTimeout { seconds: 30 }));
        assert!(!session.check_auth().await.unwrap());
    }

    #[tokio::test]
    async fn login_cancellation_aborts_the_wait() {
        let mut flow = FakeFlow::logged_in(fresh_credential());
        *flow.login.get_mut().unwrap() = LoginBehavior::Pend;
        let (session, _, _) = session(flow);
        let token = CancellationToken::new();
        let interaction = AuthInteraction::new(Arc::new(RecordingHandler::default()))
            .with_cancellation(token.clone());

        let login = tokio::spawn(async move { session.login(&interaction).await });
        token.cancel();
        assert!(matches!(
            login.await.unwrap().unwrap_err(),
            Error::AuthCancelled
        ));
    }

    #[tokio::test]
    async fn logout_deletes_and_flips_check_auth() {
        let (session, store, _) = session(FakeFlow::logged_in(fresh_credential()));
        store
            .modify(
                "test",
                Box::new(|_| Ok(Some(Credential::OAuth(fresh_credential())))),
            )
            .await
            .unwrap();
        assert!(session.check_auth().await.unwrap());

        session.logout().await.unwrap();
        assert!(!session.check_auth().await.unwrap());
        assert_eq!(store.get("test").await.unwrap(), None);
        // Logging out twice is fine.
        session.logout().await.unwrap();
    }

    #[tokio::test]
    async fn resolve_uses_fresh_token_without_refreshing() {
        let mut flow = FakeFlow::logged_in(fresh_credential());
        flow.refresh_results
            .get_mut()
            .unwrap()
            .push_back(Ok(fresh_credential()));
        let (session, store, _) = session(flow);
        store
            .modify(
                "test",
                Box::new(|_| {
                    Ok(Some(Credential::OAuth(
                        fresh_credential()
                            .with_resource_url("https://inference.example.com")
                            .unwrap(),
                    )))
                }),
            )
            .await
            .unwrap();

        let resolved = session.resolve().await.unwrap();
        assert_eq!(resolved.api_key, None);
        assert_eq!(
            resolved
                .headers
                .get("authorization")
                .and_then(Option::as_deref),
            Some("Bearer access-fresh"),
            "an OAuth access token authenticates as a bearer token"
        );
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://inference.example.com")
        );
    }

    #[tokio::test]
    async fn concurrent_resolves_share_one_refresh() {
        let flow = FakeFlow {
            login: Mutex::new(LoginBehavior::Pend),
            refresh_calls: AtomicUsize::new(0),
            refresh_delay: Duration::from_millis(50),
            refresh_results: Mutex::new(VecDeque::from([Ok(OAuthCredential::new(
                "access-renewed",
                None, // server rotates access only; prior refresh token carries over
                None,
            ))])),
        };
        let (session, store, flow) = session(flow);
        store
            .modify(
                "test",
                Box::new(|_| Ok(Some(Credential::OAuth(expired_credential())))),
            )
            .await
            .unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let session = session.clone();
            handles.push(tokio::spawn(async move { session.resolve().await }));
        }
        for handle in handles {
            let resolved = handle.await.unwrap().unwrap();
            assert_eq!(
                resolved
                    .headers
                    .get("authorization")
                    .and_then(Option::as_deref),
                Some("Bearer access-renewed")
            );
        }
        assert_eq!(flow.refresh_calls.load(Ordering::SeqCst), 1);
        // The rotation landed in the store, refresh token carried over.
        let Some(Credential::OAuth(credential)) = store.get("test").await.unwrap() else {
            panic!("credential vanished");
        };
        assert_eq!(credential.access_token, "access-renewed");
        assert_eq!(credential.refresh_token.as_deref(), Some("refresh-1"));

        // The next resolve uses the renewed token without a second refresh:
        // it attested no expiry.
        session.resolve().await.unwrap();
        assert_eq!(flow.refresh_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn refresh_yields_to_a_concurrent_login() {
        let flow = FakeFlow {
            login: Mutex::new(LoginBehavior::Pend),
            refresh_calls: AtomicUsize::new(0),
            refresh_delay: Duration::from_millis(200),
            refresh_results: Mutex::new(VecDeque::from([Ok(OAuthCredential::new(
                "access-renewed",
                None,
                None,
            ))])),
        };
        let (session, store, _) = session(flow);
        store
            .modify(
                "test",
                Box::new(|_| Ok(Some(Credential::OAuth(expired_credential())))),
            )
            .await
            .unwrap();

        let refreshing = tokio::spawn({
            let session = session.clone();
            async move { session.refresh().await }
        });
        // While the refresh HTTP call is in flight, a login lands a newer
        // credential. The refresh must not overwrite it with stale tokens.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let logged_in = fresh_credential();
        store
            .modify(
                "test",
                Box::new({
                    let logged_in = logged_in.clone();
                    move |_| Ok(Some(Credential::OAuth(logged_in)))
                }),
            )
            .await
            .unwrap();

        let outcome = refreshing.await.unwrap().unwrap();
        assert_eq!(outcome, logged_in);
        assert_eq!(
            store.get("test").await.unwrap(),
            Some(Credential::OAuth(logged_in))
        );
    }

    #[tokio::test]
    async fn invalid_refresh_preserves_credential_and_demands_re_login() {
        let mut flow = FakeFlow::logged_in(fresh_credential());
        flow.refresh_results
            .get_mut()
            .unwrap()
            .push_back(Err(RefreshError::Invalid("invalid_grant".into())));
        let (session, store, _) = session(flow);
        store
            .modify(
                "test",
                Box::new(|_| Ok(Some(Credential::OAuth(expired_credential())))),
            )
            .await
            .unwrap();

        let err = session.resolve().await.unwrap_err();
        assert!(
            matches!(&err, Error::ReLoginRequired { provider, reason } if provider == "test" && reason == "invalid_grant"),
            "unexpected error: {err}"
        );
        // The prior credential is still there for diagnosis.
        assert_eq!(
            store.get("test").await.unwrap(),
            Some(Credential::OAuth(expired_credential()))
        );
    }

    #[tokio::test]
    async fn transient_refresh_failure_preserves_credential() {
        let mut flow = FakeFlow::logged_in(fresh_credential());
        flow.refresh_results
            .get_mut()
            .unwrap()
            .push_back(Err(RefreshError::Transient("connection reset".into())));
        let (session, store, _) = session(flow);
        store
            .modify(
                "test",
                Box::new(|_| Ok(Some(Credential::OAuth(expired_credential())))),
            )
            .await
            .unwrap();

        let err = session.resolve().await.unwrap_err();
        assert!(matches!(err, Error::Auth(message) if message.contains("connection reset")));
        assert_eq!(
            store.get("test").await.unwrap(),
            Some(Credential::OAuth(expired_credential()))
        );
    }

    #[tokio::test]
    async fn expired_credential_without_refresh_token_demands_re_login() {
        let (session, store, _) = session(FakeFlow::logged_in(fresh_credential()));
        let mut stale = expired_credential();
        stale.refresh_token = None;
        store
            .modify(
                "test",
                Box::new(move |_| Ok(Some(Credential::OAuth(stale)))),
            )
            .await
            .unwrap();

        let err = session.resolve().await.unwrap_err();
        assert!(matches!(err, Error::ReLoginRequired { .. }));
    }

    #[tokio::test]
    async fn resolve_without_any_credential_demands_re_login() {
        let (session, _, _) = session(FakeFlow::logged_in(fresh_credential()));
        let err = session.resolve().await.unwrap_err();
        assert!(matches!(err, Error::ReLoginRequired { .. }));
    }
}
