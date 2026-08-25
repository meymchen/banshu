//! Ticket #49: Kimi For Coding OAuth — RFC 8628 device authorization against
//! the fixed Kimi auth contract (`client_id`, `/api/oauth/device_authorization`,
//! `/api/oauth/token`), refresh through the shared credential lifecycle, bearer
//! authentication at the coding endpoint, logout availability, and the rule
//! that the auth host is overridable only through explicit test configuration.
//!
//! Every test talks to wiremock; no real account or environment credential is
//! ever used, and token material must never appear in errors or diagnostics.

mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use banshu_ai::{
    Auth, AuthInteraction, AuthInteractionHandler, Context, Credential, CredentialStore, Error,
    ErrorKind, InMemoryCredentialStore, KimiDeviceFlow, Model, Models, OAuthAuth, OAuthCredential,
    OAuthSession, Provider, Result, StopReason, StreamOptions, VerificationDetails, async_trait,
};
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// The fixed client id of the Kimi auth contract, confirmed against the
/// official `kimi` CLI binary.
const KIMI_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";

/// Minimal Anthropic messages stream: start, one text delta, stop.
const ANTHROPIC_SSE_BODY: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn anthropic_ok_response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(ANTHROPIC_SSE_BODY)
}

/// A device authorization response from the mock auth host. `interval` of one
/// second keeps real-clock tests quick; `expires_in` long enough to never fire
/// unless a test says otherwise.
fn device_authorization_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "device_code": "dc-1",
        "user_code": "WDJB-MJHT",
        "verification_uri": "https://www.kimi.com/coding/device",
        "verification_uri_complete": "https://www.kimi.com/coding/device?user_code=WDJB-MJHT",
        "expires_in": 300,
        "interval": 1,
    }))
}

fn token_success_response(access: &str, refresh: Option<&str>) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "access_token": access,
        "refresh_token": refresh,
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "kimi-code",
    }))
}

fn token_error_response(error: &str) -> ResponseTemplate {
    ResponseTemplate::new(400).set_body_json(serde_json::json!({ "error": error }))
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

/// A `Models` registry with the Kimi provider pointed at the mock server for
/// both auth and inference, driven by the real [`KimiDeviceFlow`].
fn rig(server_uri: &str) -> Rig {
    rig_with_api_key_env(server_uri, "BANSHU_KIMI_TEST_DEFINITELY_UNSET")
}

fn rig_with_api_key_env(server_uri: &str, api_key_env: &'static str) -> Rig {
    let store = Arc::new(InMemoryCredentialStore::new());
    let flow = KimiDeviceFlow::new()
        .with_auth_host(server_uri)
        .expect("loopback auth host is valid");
    let session = OAuthSession::new(
        "kimi",
        Arc::new(flow),
        store.clone(),
        reqwest::Client::new(),
    );
    let provider =
        Provider::anthropic_compatible("kimi", "Kimi For Coding", server_uri, [api_key_env])
            .with_auth(Auth::OAuth(
                OAuthAuth::new(session).with_api_key_env([api_key_env]),
            ));
    Rig {
        models: Models::new().with_provider(provider),
        store,
    }
}

async fn seed(store: &Arc<InMemoryCredentialStore>, credential: OAuthCredential) {
    store
        .modify(
            "kimi",
            Box::new(move |_| Ok(Some(Credential::OAuth(credential)))),
        )
        .await
        .unwrap();
}

fn kimi_model(base_url: &str) -> Model {
    let mut model = Model::anthropic_messages("kimi-for-coding").with_base_url(base_url);
    model.provider = "kimi".into();
    model
}

/// A token-endpoint responder that serves one scripted response per poll.
fn scripted_token_responder(
    responses: Vec<ResponseTemplate>,
) -> impl Fn(&Request) -> ResponseTemplate {
    let next = Arc::new(Mutex::new(responses.into_iter()));
    move |_| next.lock().unwrap().next().unwrap_or_else(token_never)
}

fn token_never() -> ResponseTemplate {
    panic!("the token endpoint received more polls than scripted")
}

#[tokio::test]
async fn login_uses_the_fixed_device_authorization_contract() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .and(body_string_contains(format!("client_id={KIMI_CLIENT_ID}")))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .and(body_string_contains(
            "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code",
        ))
        .and(body_string_contains(format!("client_id={KIMI_CLIENT_ID}")))
        .and(body_string_contains("device_code=dc-1"))
        .respond_with(token_success_response("access-live", Some("refresh-live")))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let handler = Arc::new(RecordingHandler::default());
    let before = now_ms();
    let credential = rig
        .models
        .login("kimi", &AuthInteraction::new(handler.clone()))
        .await
        .unwrap();

    assert_eq!(credential.access_token, "access-live");
    assert_eq!(credential.refresh_token.as_deref(), Some("refresh-live"));
    let expires_at = credential.expires_at.expect("expires_in attests an expiry");
    assert!(expires_at >= before + 3_600_000 && expires_at <= now_ms() + 3_600_000);
    assert!(rig.models.check_auth("kimi").await.unwrap());
    assert_eq!(
        rig.store.get("kimi").await.unwrap(),
        Some(Credential::OAuth(credential))
    );
    // The complete verification URI wins when the server sends one; the user
    // code is always presented alongside.
    assert_eq!(
        handler.events.lock().unwrap().as_slice(),
        [
            "verify:https://www.kimi.com/coding/device?user_code=WDJB-MJHT:WDJB-MJHT",
            "browser:https://www.kimi.com/coding/device?user_code=WDJB-MJHT",
        ]
    );
}

#[tokio::test]
async fn login_falls_back_to_the_plain_verification_uri() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "device_code": "dc-1",
            "user_code": "WDJB-MJHT",
            "verification_uri": "https://www.kimi.com/coding/device",
            "interval": 1,
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(token_success_response("access-live", Some("refresh-live")))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let handler = Arc::new(RecordingHandler::default());
    rig.models
        .login("kimi", &AuthInteraction::new(handler.clone()))
        .await
        .unwrap();
    assert_eq!(
        handler.events.lock().unwrap().as_slice(),
        [
            "verify:https://www.kimi.com/coding/device:WDJB-MJHT",
            "browser:https://www.kimi.com/coding/device",
        ]
    );
}

#[tokio::test]
async fn authorization_pending_is_polled_through() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(scripted_token_responder(vec![
            token_error_response("authorization_pending"),
            token_error_response("authorization_pending"),
            token_success_response("access-live", Some("refresh-live")),
        ]))
        .expect(3)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let handler = Arc::new(RecordingHandler::default());
    let start = std::time::Instant::now();
    let credential = rig
        .models
        .login("kimi", &AuthInteraction::new(handler.clone()))
        .await
        .unwrap();

    assert_eq!(credential.access_token, "access-live");
    // Two pending polls, one-second interval each.
    assert!(start.elapsed() >= Duration::from_secs(2));
    let events = handler.events.lock().unwrap();
    assert!(
        events
            .iter()
            .filter(|event| event.starts_with("status:"))
            .count()
            >= 2,
        "pending polls report progress, got: {events:?}"
    );
}

#[tokio::test]
async fn slow_down_lengthens_the_poll_interval() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(scripted_token_responder(vec![
            token_error_response("slow_down"),
            token_success_response("access-live", Some("refresh-live")),
        ]))
        .expect(2)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let start = std::time::Instant::now();
    let credential = rig
        .models
        .login(
            "kimi",
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap();

    assert_eq!(credential.access_token, "access-live");
    // RFC 8628: slow_down adds five seconds to the one-second interval.
    assert!(start.elapsed() >= Duration::from_secs(6));
}

#[tokio::test]
async fn access_denied_fails_the_login_without_leaking_the_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": "access_denied",
            "error_description": "canary-secret-value",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let err = rig
        .models
        .login(
            "kimi",
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(err.to_string().contains("denied"), "{err}");
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
    assert!(!rig.models.check_auth("kimi").await.unwrap());
}

#[tokio::test]
async fn expired_token_fails_the_login() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(token_error_response("expired_token"))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let err = rig
        .models
        .login(
            "kimi",
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(err.to_string().contains("expired"), "{err}");
}

#[tokio::test]
async fn device_code_expiry_deadline_ends_polling() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "device_code": "dc-1",
            "user_code": "WDJB-MJHT",
            "verification_uri": "https://www.kimi.com/coding/device",
            "expires_in": 3,
            "interval": 1,
        })))
        .expect(1)
        .mount(&server)
        .await;
    // The user never approves; every poll says pending.
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(token_error_response("authorization_pending"))
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let start = std::time::Instant::now();
    let err = rig
        .models
        .login(
            "kimi",
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(err.to_string().contains("expired"), "{err}");
    assert!(start.elapsed() >= Duration::from_secs(3));
}

#[tokio::test]
async fn cancellation_aborts_the_login() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(token_error_response("authorization_pending"))
        .expect(0)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let token = banshu_ai::CancellationToken::new();
    let interaction = common::cancelling_interaction(token);

    // The browser callback cancels after authorization succeeds but before
    // token polling starts, so the real flow exercises `interaction.wait`
    // without racing HTTP keep-alive or wall-clock sleeps.
    let err = rig.models.login("kimi", &interaction).await.unwrap_err();
    assert!(
        matches!(err, Error::AuthCancelled),
        "unexpected error: {err:?}"
    );
}

#[tokio::test]
async fn login_times_out_when_the_user_never_approves() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(token_error_response("authorization_pending"))
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let interaction = AuthInteraction::new(Arc::new(RecordingHandler::default()))
        .with_timeout(Duration::from_secs(2));

    let err = rig.models.login("kimi", &interaction).await.unwrap_err();
    assert!(matches!(err, Error::AuthTimeout { seconds: 2 }));
}

#[tokio::test]
async fn malformed_token_response_leaks_nothing() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "not_an_access_token": "canary-secret-value",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let err = rig
        .models
        .login(
            "kimi",
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
}

#[tokio::test]
async fn refresh_uses_the_fixed_token_contract_and_rotates_tokens() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .and(body_string_contains(format!("client_id={KIMI_CLIENT_ID}")))
        .and(body_string_contains("refresh_token=refresh-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "access-renewed",
            // The server rotates the access token only; the prior refresh
            // token carries over.
            "expires_in": 3600,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    seed(
        &rig.store,
        OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000)),
    )
    .await;

    let credential = rig.models.refresh_credential("kimi").await.unwrap();
    assert_eq!(credential.access_token, "access-renewed");
    assert_eq!(credential.refresh_token.as_deref(), Some("refresh-1"));
    assert_eq!(
        rig.store.get("kimi").await.unwrap(),
        Some(Credential::OAuth(credential))
    );
}

#[tokio::test]
async fn rejected_refresh_token_preserves_the_credential() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": "invalid_grant",
            "error_description": "canary-secret-value",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let stale = OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000));
    seed(&rig.store, stale.clone()).await;

    let err = rig.models.refresh_credential("kimi").await.unwrap_err();
    assert!(
        matches!(&err, Error::ReLoginRequired { provider, .. } if provider == "kimi"),
        "unexpected error: {err}"
    );
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
    // Preserved for diagnosis, never silently deleted.
    assert_eq!(
        rig.store.get("kimi").await.unwrap(),
        Some(Credential::OAuth(stale))
    );
}

#[tokio::test]
async fn transient_refresh_failure_preserves_the_credential() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(ResponseTemplate::new(500).set_body_string("canary-secret-value"))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let stale = OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000));
    seed(&rig.store, stale.clone()).await;

    let err = rig.models.refresh_credential("kimi").await.unwrap_err();
    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
    assert_eq!(
        rig.store.get("kimi").await.unwrap(),
        Some(Credential::OAuth(stale))
    );
}

#[test]
fn auth_host_override_is_explicit_and_validated() {
    // The default is the fixed Kimi auth host.
    let flow = KimiDeviceFlow::new();
    assert!(format!("{flow:?}").contains("auth.kimi.com"));

    // Explicit overrides are accepted for HTTPS and loopback HTTP only.
    assert!(
        KimiDeviceFlow::new()
            .with_auth_host("https://auth.example.com")
            .is_ok()
    );
    assert!(
        KimiDeviceFlow::new()
            .with_auth_host("http://127.0.0.1:57166")
            .is_ok()
    );
    for bad in [
        "http://plain.example.com",
        "http://192.168.0.10",
        "ftp://x",
        "not a url",
    ] {
        let err = KimiDeviceFlow::new().with_auth_host(bad).unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{bad}");
    }
}

#[tokio::test]
async fn access_token_authenticates_inference_via_bearer() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("authorization", "Bearer access-live"))
        .respond_with(anthropic_ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    seed(
        &rig.store,
        OAuthCredential::new(
            "access-live",
            Some("refresh-1".into()),
            Some(now_ms() + 3_600_000),
        ),
    )
    .await;

    let message = rig
        .models
        .stream(
            &kimi_model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].headers.contains_key("x-api-key"),
        "an OAuth access token is a bearer token, never an x-api-key"
    );
}

#[tokio::test]
async fn env_api_key_wins_over_the_stored_oauth_credential() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-env-key"))
        .respond_with(anthropic_ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let var = "BANSHU_KIMI_TEST_ENV_PRIORITY";
    let rig = rig_with_api_key_env(&server.uri(), var);
    seed(
        &rig.store,
        OAuthCredential::new(
            "access-live",
            Some("refresh-1".into()),
            Some(now_ms() + 3_600_000),
        ),
    )
    .await;

    let saved = std::env::var_os(var);
    // SAFETY: this integration test runs in its own process, uses a unique
    // variable, and restores the prior value before making assertions.
    unsafe { std::env::set_var(var, "sk-env-key") };
    let message = rig
        .models
        .stream(
            &kimi_model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;
    match saved {
        Some(value) => {
            // SAFETY: restore the variable changed above.
            unsafe { std::env::set_var(var, value) };
        }
        None => {
            // SAFETY: restore the variable changed above.
            unsafe { std::env::remove_var(var) };
        }
    }

    assert_eq!(message.stop_reason, StopReason::Stop);
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].headers.contains_key("authorization"),
        "the API-key override must replace the stored OAuth bearer token"
    );
}

#[tokio::test]
async fn expired_credential_refreshes_then_authenticates_inference() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .respond_with(token_success_response("access-renewed", Some("refresh-2")))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("authorization", "Bearer access-renewed"))
        .respond_with(anthropic_ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    seed(
        &rig.store,
        OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000)),
    )
    .await;

    let message = rig
        .models
        .stream(
            &kimi_model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    let Some(Credential::OAuth(credential)) = rig.store.get("kimi").await.unwrap() else {
        panic!("credential vanished");
    };
    assert_eq!(credential.access_token, "access-renewed");
    assert_eq!(credential.refresh_token.as_deref(), Some("refresh-2"));
}

#[tokio::test]
async fn failed_auth_surfaces_in_band_without_partial_content() {
    let server = MockServer::start().await;
    // No credential stored: the request must fail before any HTTP call.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(anthropic_ok_response())
        .expect(0)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    let message = rig
        .models
        .stream(
            &kimi_model(&server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_kind, Some(ErrorKind::Auth));
    assert!(message.content.is_empty());
}

#[tokio::test]
async fn logout_deletes_the_credential_and_reports_unavailable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/device_authorization"))
        .respond_with(device_authorization_response())
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/oauth/token"))
        .respond_with(token_success_response("access-live", Some("refresh-live")))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(&server.uri());
    rig.models
        .login(
            "kimi",
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap();
    assert!(rig.models.check_auth("kimi").await.unwrap());
    assert!(
        !rig.models.available().await.is_empty(),
        "a logged-in Kimi provider serves its models"
    );

    rig.models.logout("kimi").await.unwrap();
    assert!(!rig.models.check_auth("kimi").await.unwrap());
    assert_eq!(rig.store.get("kimi").await.unwrap(), None);
    assert!(
        rig.models.available().await.is_empty(),
        "after logout Kimi OAuth reports unavailable"
    );
}

#[tokio::test]
async fn kimi_constructor_wires_the_oauth_lifecycle() {
    let provider = Provider::kimi(Arc::new(InMemoryCredentialStore::new()));
    assert_eq!(provider.id(), "kimi");
    assert_eq!(provider.base_url(), "https://api.kimi.com/coding");
    assert!(
        provider.oauth_session().is_some(),
        "the bundled Kimi provider participates in the OAuth lifecycle"
    );

    // With no credential stored and no KIMI_API_KEY in the environment, the
    // provider reports unavailable.
    let saved = std::env::var("KIMI_API_KEY").ok();
    unsafe { std::env::remove_var("KIMI_API_KEY") };
    let models = Models::new().with_provider(provider);
    assert!(!models.check_auth("kimi").await.unwrap());
    if let Some(saved) = saved {
        unsafe { std::env::set_var("KIMI_API_KEY", saved) };
    }
}
