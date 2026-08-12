//! Ticket #48: credential storage and the OAuth lifecycle, end to end.
//!
//! A device-code-style test flow talks real HTTP to wiremock: login drives
//! the `AuthInteraction` and stores a credential; an expired credential is
//! refreshed at request time; concurrent streams share one refresh HTTP call;
//! an invalid refresh keeps the prior credential and fails in-band; a set
//! API-key env var beats the OAuth credential; and a credential-level
//! `resource_url` redirects the request.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use banshu_ai::{
    Auth, AuthInteraction, AuthInteractionHandler, Context, Credential, CredentialStore, ErrorKind,
    InMemoryCredentialStore, Models, OAuthCredential, OAuthFlow, OAuthSession, Provider,
    RefreshError, Result, StopReason, StreamOptions, VerificationDetails, async_trait,
};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Minimal one-delta OpenAI completion.
const SSE_BODY: &str = concat!(
    "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
);

fn ok_response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(SSE_BODY)
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn expired_credential() -> OAuthCredential {
    OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000))
}

/// A device-code-style flow against the mock server: `POST /oauth/device`
/// starts a login, `POST /oauth/token` both completes it and refreshes.
struct DeviceFlow {
    base: String,
}

#[async_trait]
impl OAuthFlow for DeviceFlow {
    async fn login(
        &self,
        http: &reqwest::Client,
        interaction: &AuthInteraction,
    ) -> Result<OAuthCredential> {
        let device: serde_json::Value = http
            .post(format!("{}/oauth/device", self.base))
            .send()
            .await?
            .json()
            .await?;
        let verification_uri = device["verification_uri"].as_str().unwrap().to_string();
        interaction
            .show_verification(&VerificationDetails {
                url: verification_uri.clone(),
                user_code: device["user_code"].as_str().map(str::to_string),
                instructions: None,
            })
            .await?;
        interaction.open_browser(&verification_uri).await?;
        let device_code = device["device_code"].as_str().unwrap().to_string();
        let base = self.base.clone();
        interaction
            .wait(async move {
                loop {
                    let response = http
                        .post(format!("{base}/oauth/token"))
                        .json(&serde_json::json!({
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                            "device_code": device_code,
                        }))
                        .send()
                        .await?;
                    if response.status().is_success() {
                        let body: serde_json::Value = response.json().await?;
                        return Ok(OAuthCredential::new(
                            body["access_token"].as_str().unwrap(),
                            body["refresh_token"].as_str().map(str::to_string),
                            body["expires_in"]
                                .as_i64()
                                .map(|secs| now_ms() + secs * 1000),
                        ));
                    }
                    interaction.report_status("authorization pending").await;
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
    }

    async fn refresh(
        &self,
        http: &reqwest::Client,
        credential: &OAuthCredential,
    ) -> std::result::Result<OAuthCredential, RefreshError> {
        let response = http
            .post(format!("{}/oauth/token", self.base))
            .json(&serde_json::json!({
                "grant_type": "refresh_token",
                "refresh_token": credential.refresh_token,
            }))
            .send()
            .await?;
        let status = response.status();
        let body: serde_json::Value = response.json().await?;
        if status.is_success() {
            return Ok(OAuthCredential::new(
                body["access_token"].as_str().unwrap(),
                body["refresh_token"].as_str().map(str::to_string),
                body["expires_in"]
                    .as_i64()
                    .map(|secs| now_ms() + secs * 1000),
            ));
        }
        if body["error"] == "invalid_grant" {
            Err(RefreshError::Invalid("invalid_grant".into()))
        } else {
            Err(RefreshError::Transient(format!(
                "token endpoint HTTP {status}"
            )))
        }
    }
}

#[derive(Default)]
struct RecordingHandler {
    events: Mutex<Vec<String>>,
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
        Ok(true)
    }
    async fn report_status(&self, message: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("status:{message}"));
    }
}

struct Rig {
    models: Models,
    store: Arc<InMemoryCredentialStore>,
}

/// A `Models` registry with one OAuth provider (`p`) backed by a `DeviceFlow`
/// against `base`.
fn rig(base: &str) -> Rig {
    let store = Arc::new(InMemoryCredentialStore::new());
    let session = OAuthSession::new(
        "p",
        Arc::new(DeviceFlow { base: base.into() }),
        store.clone(),
        reqwest::Client::new(),
    );
    let provider =
        Provider::openai_compatible("p", "P", base, ["BANSHU_OAUTH_TEST_DEFINITELY_UNSET"])
            .with_auth(Auth::oauth(session));
    Rig {
        models: Models::new().with_provider(provider),
        store,
    }
}

async fn seed(store: &Arc<InMemoryCredentialStore>, credential: OAuthCredential) {
    store
        .modify(
            "p",
            Box::new(move |_| Ok(Some(Credential::OAuth(credential)))),
        )
        .await
        .unwrap();
}

async fn stream_once(models: &Models, base_url: &str) -> banshu_ai::AssistantMessage {
    let mut model = banshu_ai::Model::openai_completions("m").with_base_url(base_url);
    model.provider = "p".into();
    models
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await
}

#[tokio::test]
async fn login_check_auth_and_logout_round_trip() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/device"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "device_code": "dc-1",
            "verification_uri": "https://example.com/activate",
            "user_code": "WDJB-MJHT",
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "access-live",
            "refresh_token": "refresh-live",
            "expires_in": 3600,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let handler = Arc::new(RecordingHandler::default());
    let credential = rig
        .models
        .login("p", &AuthInteraction::new(handler.clone()))
        .await
        .unwrap();
    assert_eq!(credential.access_token, "access-live");
    assert_eq!(credential.refresh_token.as_deref(), Some("refresh-live"));
    assert!(rig.models.check_auth("p").await.unwrap());
    assert_eq!(
        handler.events.lock().unwrap().as_slice(),
        [
            "verify:https://example.com/activate:WDJB-MJHT",
            "browser:https://example.com/activate",
        ]
    );

    rig.models.logout("p").await.unwrap();
    assert!(!rig.models.check_auth("p").await.unwrap());
    assert_eq!(rig.store.get("p").await.unwrap(), None);
}

#[tokio::test]
async fn login_rejects_providers_without_oauth() {
    let server = MockServer::start().await;
    let rig = rig(&server.uri());
    let handler = Arc::new(RecordingHandler::default());
    let interaction = AuthInteraction::new(handler);

    assert!(rig.models.login("nope", &interaction).await.is_err());
    assert!(rig.models.logout("nope").await.is_err());
    assert!(rig.models.check_auth("nope").await.is_err());
    assert!(rig.models.refresh_credential("nope").await.is_err());

    let api_key_only = Models::new().with_provider(Provider::openai_compatible(
        "q",
        "Q",
        server.uri(),
        ["BANSHU_OAUTH_TEST_DEFINITELY_UNSET"],
    ));
    assert!(api_key_only.login("q", &interaction).await.is_err());
    assert!(api_key_only.logout("q").await.is_err());
    assert!(api_key_only.refresh_credential("q").await.is_err());
    // check-auth works for API-key providers too: no key set, not authenticated.
    assert!(!api_key_only.check_auth("q").await.unwrap());
}

#[tokio::test]
async fn expired_credential_refreshes_before_the_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "access-renewed",
            "refresh_token": "refresh-2",
            "expires_in": 3600,
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer access-renewed"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    seed(&rig.store, expired_credential()).await;

    let message = stream_once(&rig.models, &server.uri()).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");

    // The rotation landed in the store.
    let Some(Credential::OAuth(credential)) = rig.store.get("p").await.unwrap() else {
        panic!("credential vanished");
    };
    assert_eq!(credential.access_token, "access-renewed");
    assert_eq!(credential.refresh_token.as_deref(), Some("refresh-2"));
}

#[tokio::test]
async fn concurrent_requests_share_one_refresh_http_call() {
    let server = MockServer::start().await;
    // Slow the refresh down so the concurrent requests really overlap.
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "access_token": "access-renewed",
                    "refresh_token": "refresh-2",
                    "expires_in": 3600,
                }))
                .set_delay(Duration::from_millis(100)),
        )
        .expect(1)
        .mount(&server)
        .await;
    const REQUESTS: u64 = 8;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer access-renewed"))
        .respond_with(ok_response())
        .expect(REQUESTS)
        .mount(&server)
        .await;

    let rig = Arc::new(rig(&server.uri()));
    seed(&rig.store, expired_credential()).await;

    let mut handles = Vec::new();
    for _ in 0..REQUESTS {
        let rig = Arc::clone(&rig);
        let base = server.uri();
        handles.push(tokio::spawn(async move {
            stream_once(&rig.models, &base).await
        }));
    }
    for handle in handles {
        assert_eq!(handle.await.unwrap().stop_reason, StopReason::Stop);
    }
    // The `expect(1)`/`expect(REQUESTS)` mocks verify on drop: one refresh
    // HTTP call total, every request bearing the same renewed token.
}

#[tokio::test]
async fn invalid_refresh_keeps_credential_and_fails_in_band() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": "invalid_grant",
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ok_response())
        .expect(0)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    seed(&rig.store, expired_credential()).await;

    let message = stream_once(&rig.models, &server.uri()).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Auth));
    let detail = message.error_message.as_deref().unwrap_or_default();
    assert!(
        detail.contains("re-login required") && detail.contains('p'),
        "message should demand a re-login, got: {detail}"
    );

    // The prior credential is preserved for diagnosis.
    assert_eq!(
        rig.store.get("p").await.unwrap(),
        Some(Credential::OAuth(expired_credential()))
    );
}

#[tokio::test]
async fn api_key_env_takes_priority_over_the_oauth_credential() {
    let var = "BANSHU_OAUTH_TEST_PRIORITY_KEY";
    unsafe { std::env::set_var(var, "env-key") };

    let server = MockServer::start().await;
    // No refresh may happen while the env key is set.
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(0)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer env-key"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let store = Arc::new(InMemoryCredentialStore::new());
    let session = OAuthSession::new(
        "p",
        Arc::new(DeviceFlow { base: server.uri() }),
        store.clone(),
        reqwest::Client::new(),
    );
    let provider = Provider::openai_compatible("p", "P", server.uri(), ["UNUSED"]).with_auth(
        Auth::OAuth(banshu_ai::OAuthAuth::new(session).with_api_key_env([var])),
    );
    // Even an expired stored credential stays unused while the key is set.
    seed(&store, expired_credential()).await;
    let models = Models::new().with_provider(provider);

    assert!(models.check_auth("p").await.unwrap());
    let message = stream_once(&models, &server.uri()).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    // The stored credential is untouched — no refresh, no deletion.
    assert_eq!(
        store.get("p").await.unwrap(),
        Some(Credential::OAuth(expired_credential()))
    );

    unsafe { std::env::remove_var(var) };
}

#[tokio::test]
async fn credential_resource_url_overrides_the_request_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer access-live"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;

    // Provider and model point at a black hole; the credential redirects.
    let store = Arc::new(InMemoryCredentialStore::new());
    let session = OAuthSession::new(
        "p",
        Arc::new(DeviceFlow { base: server.uri() }),
        store.clone(),
        reqwest::Client::new(),
    );
    let provider = Provider::openai_compatible("p", "P", "http://127.0.0.1:0", ["UNUSED"])
        .with_auth(Auth::oauth(session));
    let models = Models::new().with_provider(provider);
    seed(
        &store,
        OAuthCredential::new(
            "access-live",
            Some("refresh-1".into()),
            Some(now_ms() + 3_600_000),
        )
        .with_resource_url(server.uri())
        .unwrap(),
    )
    .await;

    let message = stream_once(&models, "http://127.0.0.1:0").await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");
}
