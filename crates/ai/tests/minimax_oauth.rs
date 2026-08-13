//! Ticket #50: MiniMax Coding Plan OAuth — the frozen portal contract for the
//! explicit CN and Global regions: `POST /oauth/code` with PKCE S256 and a
//! random state, `POST /oauth/token` polling with the `user_code` grant,
//! refresh through the same regional token endpoint, a credential-level
//! `resource_url` (HTTPS only), and Anthropic-compatible inference
//! authenticated with every required bearer/API-key header.
//!
//! Every test talks to wiremock; no real account or environment credential is
//! ever used, region is never inferred from IP, and token material must never
//! appear in errors or diagnostics.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use banshu_ai::{
    Auth, AuthInteraction, AuthInteractionHandler, Context, Credential, CredentialStore, Error,
    ErrorKind, InMemoryCredentialStore, MINIMAX_CLIENT_ID, MiniMaxPortalFlow, MiniMaxRegion, Model,
    Models, OAuthCredential, OAuthSession, Provider, ProviderHeaders, Result, StopReason,
    StreamOptions, VerificationDetails, async_trait,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::Digest;
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// The frozen OAuth scope of the MiniMax Coding Plan contract.
const MINIMAX_OAUTH_SCOPE: &str = "group_id profile model.completion";

/// The frozen polling grant type.
const USER_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:user_code";

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

/// Read a form field out of a urlencoded request body. The frozen contract's
/// values are all base64url or plain tokens, which form-encoding leaves
/// untouched, so no percent-decoding is needed here.
fn form_field(request: &Request, field: &str) -> Option<String> {
    let body = String::from_utf8_lossy(&request.body);
    body.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (key == field).then(|| value.to_string())
    })
}

/// A `/oauth/code` responder that echoes the request's `state` — or
/// `state_override` when a test scripts a mismatch — inside the frozen
/// authorization payload. `interval` of one millisecond keeps real-clock
/// polling quick; `expired_in` is the contract's absolute-millisecond
/// deadline, generous unless a test says otherwise.
fn code_responder(state_override: Option<&'static str>) -> impl Fn(&Request) -> ResponseTemplate {
    move |request: &Request| {
        let state = state_override
            .map(str::to_string)
            .or_else(|| form_field(request, "state"))
            .expect("the code request carries a state");
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "user_code": "UC-42",
            "verification_uri": "https://www.minimax.io/oauth/verify",
            "expired_in": now_ms() + 300_000,
            "interval": 1,
            "state": state,
        }))
    }
}

fn token_success_response(access: &str, refresh: Option<&str>) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "status": "success",
        "access_token": access,
        "refresh_token": refresh,
        // Relative seconds: the small end of the frozen `expired_in` ladder.
        "expired_in": 3600,
        "resource_url": "https://inference.minimax.example/anthropic",
    }))
}

fn token_pending_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({ "status": "pending" }))
}

/// A token-endpoint responder that serves one scripted response per poll.
fn scripted_token_responder(
    responses: Vec<ResponseTemplate>,
) -> impl Fn(&Request) -> ResponseTemplate {
    let next = Arc::new(Mutex::new(responses.into_iter()));
    move |_| {
        next.lock()
            .unwrap()
            .next()
            .unwrap_or_else(|| panic!("the token endpoint received more polls than scripted"))
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
    provider_id: &'static str,
}

/// A `Models` registry with the MiniMax provider for `region` pointed at the
/// mock server for both portal and inference, driven by the real
/// [`MiniMaxPortalFlow`].
fn rig(region: MiniMaxRegion, server_uri: &str) -> Rig {
    rig_with_api_key_env(region, server_uri, "BANSHU_MINIMAX_TEST_DEFINITELY_UNSET")
}

/// [`rig`] with a chosen API-key env var. Tests that *set* a variable must
/// pick one no other test's provider reads: the environment is process-wide
/// and the test binary runs multithreaded.
fn rig_with_api_key_env(region: MiniMaxRegion, server_uri: &str, api_key_env: &str) -> Rig {
    let store = Arc::new(InMemoryCredentialStore::new());
    let flow = MiniMaxPortalFlow::new(region)
        .with_portal(server_uri)
        .expect("loopback portal is valid");
    let provider_id = region.provider_id();
    let session = OAuthSession::new(
        provider_id,
        Arc::new(flow),
        store.clone(),
        reqwest::Client::new(),
    );
    let provider =
        Provider::anthropic_compatible(provider_id, "MiniMax", server_uri, [api_key_env])
            // Wire the env override the way `Provider::minimax` does: a set
            // variable is an explicit operator choice and wins over the stored
            // credential.
            .with_auth(Auth::OAuth(
                banshu_ai::OAuthAuth::new(session).with_api_key_env([api_key_env]),
            ));
    Rig {
        models: Models::new().with_provider(provider),
        store,
        provider_id,
    }
}

async fn seed(
    store: &Arc<InMemoryCredentialStore>,
    provider_id: &str,
    credential: OAuthCredential,
) {
    let provider_id = provider_id.to_string();
    store
        .modify(
            &provider_id,
            Box::new(move |_| Ok(Some(Credential::OAuth(credential)))),
        )
        .await
        .unwrap();
}

fn minimax_model(provider_id: &str, base_url: &str) -> Model {
    let mut model = Model::anthropic_messages("MiniMax-M2").with_base_url(base_url);
    model.provider = provider_id.into();
    model
}

/// Mount the frozen `/oauth/code` fixture: method, path, and every form field
/// the contract names.
async fn mount_code_endpoint(
    server: &MockServer,
    state_override: Option<&'static str>,
    times: u64,
) {
    Mock::given(method("POST"))
        .and(path("/oauth/code"))
        .and(body_string_contains("response_type=code"))
        .and(body_string_contains(format!(
            "client_id={MINIMAX_CLIENT_ID}"
        )))
        .and(body_string_contains(
            "scope=group_id+profile+model.completion",
        ))
        .and(body_string_contains("code_challenge_method=S256"))
        .and(body_string_contains("code_challenge="))
        .and(body_string_contains("state="))
        .respond_with(code_responder(state_override))
        .expect(times)
        .mount(server)
        .await;
}

#[tokio::test]
async fn login_uses_the_frozen_pkce_contract_in_both_regions() {
    let mut states = Vec::new();
    let mut verifiers = Vec::new();
    for region in [MiniMaxRegion::Cn, MiniMaxRegion::Global] {
        let server = MockServer::start().await;
        mount_code_endpoint(&server, None, 1).await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .and(body_string_contains(format!(
                "grant_type={}",
                USER_CODE_GRANT.replace(':', "%3A")
            )))
            .and(body_string_contains(format!(
                "client_id={MINIMAX_CLIENT_ID}"
            )))
            .and(body_string_contains("user_code=UC-42"))
            .and(body_string_contains("code_verifier="))
            .respond_with(token_success_response("access-live", Some("refresh-live")))
            .expect(1)
            .mount(&server)
            .await;

        let rig = rig(region, &server.uri());
        let handler = Arc::new(RecordingHandler::default());
        let before = now_ms();
        let credential = rig
            .models
            .login(rig.provider_id, &AuthInteraction::new(handler.clone()))
            .await
            .unwrap();

        assert_eq!(credential.access_token, "access-live");
        assert_eq!(credential.refresh_token.as_deref(), Some("refresh-live"));
        let expires_at = credential.expires_at.expect("expired_in attests an expiry");
        assert!(expires_at >= before + 3_600_000 && expires_at <= now_ms() + 3_600_000);
        assert_eq!(
            credential.resource_url.as_deref(),
            Some("https://inference.minimax.example/anthropic")
        );
        assert_eq!(
            rig.store.get(rig.provider_id).await.unwrap(),
            Some(Credential::OAuth(credential))
        );
        assert_eq!(
            handler.events.lock().unwrap().as_slice(),
            [
                "verify:https://www.minimax.io/oauth/verify:UC-42",
                "browser:https://www.minimax.io/oauth/verify",
            ]
        );

        // PKCE S256: the challenge sent to /oauth/code is the base64url
        // SHA-256 of the verifier sent to /oauth/token — never the verifier
        // itself.
        let requests = server.received_requests().await.expect("recorded requests");
        let code = requests
            .iter()
            .find(|request| request.url.path() == "/oauth/code")
            .expect("a code request");
        let token = requests
            .iter()
            .find(|request| request.url.path() == "/oauth/token")
            .expect("a token request");
        let challenge = form_field(code, "code_challenge").expect("a code challenge");
        let verifier = form_field(token, "code_verifier").expect("a code verifier");
        let expected = URL_SAFE_NO_PAD.encode(sha2::Sha256::digest(verifier.as_bytes()));
        assert_eq!(challenge, expected, "the challenge is S256 of the verifier");
        assert_ne!(challenge, verifier, "S256, never the plain verifier");

        // The state is random per login and round-trips verbatim.
        let state = form_field(code, "state").expect("a state");
        assert!(!state.is_empty());
        states.push(state);
        verifiers.push(verifier);
    }
    assert_ne!(
        states[0], states[1],
        "the anti-CSRF state is random per login"
    );
    assert_ne!(
        verifiers[0], verifiers[1],
        "the PKCE verifier is random per login"
    );
}

#[tokio::test]
async fn state_mismatch_rejects_the_login_before_any_polling() {
    let server = MockServer::start().await;
    mount_code_endpoint(&server, Some("tampered-state"), 1).await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(token_success_response("access-live", Some("refresh-live")))
        .expect(0)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let err = rig
        .models
        .login(
            rig.provider_id,
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(err.to_string().contains("state"), "{err}");
    assert!(!rig.models.check_auth(rig.provider_id).await.unwrap());
}

#[tokio::test]
async fn malformed_code_response_fails_the_login() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/code"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "user_code": "UC-42",
            // No verification_uri, no state.
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Cn, &server.uri());
    let err = rig
        .models
        .login(
            rig.provider_id,
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
}

#[tokio::test]
async fn code_endpoint_http_failure_leaks_no_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/code"))
        .respond_with(ResponseTemplate::new(500).set_body_string("canary-secret-value"))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let err = rig
        .models
        .login(
            rig.provider_id,
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(err.to_string().contains("500"), "{err}");
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
}

#[tokio::test]
async fn pending_polls_run_until_success() {
    let server = MockServer::start().await;
    mount_code_endpoint(&server, None, 1).await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(scripted_token_responder(vec![
            token_pending_response(),
            token_pending_response(),
            token_success_response("access-live", Some("refresh-live")),
        ]))
        .expect(3)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Cn, &server.uri());
    let handler = Arc::new(RecordingHandler::default());
    let credential = rig
        .models
        .login(rig.provider_id, &AuthInteraction::new(handler.clone()))
        .await
        .unwrap();

    assert_eq!(credential.access_token, "access-live");
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
async fn error_status_from_the_portal_fails_the_login() {
    let server = MockServer::start().await;
    mount_code_endpoint(&server, None, 1).await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "error",
            "base_resp": { "status_code": 1, "status_msg": "canary-secret-value" },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let err = rig
        .models
        .login(
            rig.provider_id,
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
}

#[tokio::test]
async fn malformed_token_response_leaks_nothing() {
    let server = MockServer::start().await;
    mount_code_endpoint(&server, None, 1).await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "success",
            // No access token, no refresh token, no expiry.
            "notification_message": "canary-secret-value",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let err = rig
        .models
        .login(
            rig.provider_id,
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
}

#[tokio::test]
async fn cancellation_aborts_the_login() {
    let server = MockServer::start().await;
    mount_code_endpoint(&server, None, 1).await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(token_pending_response())
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let token = banshu_ai::CancellationToken::new();
    let interaction = AuthInteraction::new(Arc::new(RecordingHandler::default()))
        .with_cancellation(token.clone());

    let provider_id = rig.provider_id;
    let login = tokio::spawn(async move { rig.models.login(provider_id, &interaction).await });
    // Let the first poll land, then cancel while the flow waits out the
    // interval.
    tokio::time::sleep(Duration::from_millis(100)).await;
    token.cancel();

    assert!(matches!(
        login.await.unwrap().unwrap_err(),
        Error::AuthCancelled
    ));
}

#[tokio::test]
async fn login_times_out_when_the_user_never_approves() {
    let server = MockServer::start().await;
    mount_code_endpoint(&server, None, 1).await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(token_pending_response())
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let interaction = AuthInteraction::new(Arc::new(RecordingHandler::default()))
        .with_timeout(Duration::from_secs(2));

    let err = rig
        .models
        .login(rig.provider_id, &interaction)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::AuthTimeout { seconds: 2 }));
}

#[tokio::test]
async fn user_code_expiry_deadline_ends_polling() {
    let server = MockServer::start().await;
    // The authorization's absolute-millisecond deadline lands two seconds out.
    let deadline = now_ms() + 2_000;
    Mock::given(method("POST"))
        .and(path("/oauth/code"))
        .respond_with(move |request: &Request| {
            let state = form_field(request, "state").expect("a state");
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "user_code": "UC-42",
                "verification_uri": "https://www.minimax.io/oauth/verify",
                "expired_in": deadline,
                "interval": 1,
                "state": state,
            }))
        })
        .expect(1)
        .mount(&server)
        .await;
    // The user never approves; every poll says pending.
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(token_pending_response())
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let start = std::time::Instant::now();
    let err = rig
        .models
        .login(
            rig.provider_id,
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(err.to_string().contains("expired"), "{err}");
    assert!(start.elapsed() >= Duration::from_secs(2));
}

#[test]
fn regions_name_their_frozen_hosts_explicitly() {
    assert_eq!(MiniMaxRegion::Cn.portal(), "https://api.minimaxi.com");
    assert_eq!(
        MiniMaxRegion::Cn.inference_base_url(),
        "https://api.minimaxi.com/anthropic"
    );
    assert_eq!(MiniMaxRegion::Cn.provider_id(), "minimax-cn");
    assert_eq!(MiniMaxRegion::Global.portal(), "https://api.minimax.io");
    assert_eq!(
        MiniMaxRegion::Global.inference_base_url(),
        "https://api.minimax.io/anthropic"
    );
    assert_eq!(MiniMaxRegion::Global.provider_id(), "minimax");
    assert_eq!(MINIMAX_CLIENT_ID, "78257093-7e40-4613-99e0-527b14b39113");
    assert_eq!(MINIMAX_OAUTH_SCOPE, "group_id profile model.completion");
}

#[test]
fn portal_override_is_explicit_and_validated() {
    // The default is the region's frozen portal host.
    let flow = MiniMaxPortalFlow::new(MiniMaxRegion::Cn);
    assert!(format!("{flow:?}").contains("api.minimaxi.com"));
    let flow = MiniMaxPortalFlow::new(MiniMaxRegion::Global);
    assert!(format!("{flow:?}").contains("api.minimax.io"));

    // Explicit overrides are accepted for HTTPS and loopback HTTP only.
    assert!(
        MiniMaxPortalFlow::new(MiniMaxRegion::Global)
            .with_portal("https://portal.example.com")
            .is_ok()
    );
    assert!(
        MiniMaxPortalFlow::new(MiniMaxRegion::Global)
            .with_portal("http://127.0.0.1:57166")
            .is_ok()
    );
    for bad in [
        "http://plain.example.com",
        "http://192.168.0.10",
        "ftp://x",
        "not a url",
    ] {
        let err = MiniMaxPortalFlow::new(MiniMaxRegion::Global)
            .with_portal(bad)
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{bad}");
    }
}

#[tokio::test]
async fn refresh_uses_the_same_regional_token_endpoint_and_rotates() {
    let server = MockServer::start().await;
    // The mock bakes its absolute-millisecond expiry in at mount time, so the
    // assertion window opens here.
    let before = now_ms();
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .and(body_string_contains(format!(
            "client_id={MINIMAX_CLIENT_ID}"
        )))
        .and(body_string_contains("refresh_token=refresh-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "success",
            "access_token": "access-renewed",
            // The server rotates the access token only; the prior refresh
            // token carries over. `expired_in` as absolute milliseconds — the
            // top end of the frozen ladder.
            "expired_in": now_ms() + 7_200_000,
            "resource_url": "https://inference.minimax.example/anthropic",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Cn, &server.uri());
    seed(
        &rig.store,
        rig.provider_id,
        OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000)),
    )
    .await;

    let credential = rig
        .models
        .refresh_credential(rig.provider_id)
        .await
        .unwrap();
    assert_eq!(credential.access_token, "access-renewed");
    assert_eq!(credential.refresh_token.as_deref(), Some("refresh-1"));
    let expires_at = credential.expires_at.expect("expired_in attests an expiry");
    assert!(expires_at >= before + 7_200_000 && expires_at <= now_ms() + 7_200_000);
    assert_eq!(
        credential.resource_url.as_deref(),
        Some("https://inference.minimax.example/anthropic")
    );
    // The rotation landed atomically in the store.
    assert_eq!(
        rig.store.get(rig.provider_id).await.unwrap(),
        Some(Credential::OAuth(credential))
    );
}

#[tokio::test]
async fn refresh_reads_epoch_seconds_expiries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "success",
            "access_token": "access-renewed",
            // Between a billion and a trillion, `expired_in` reads as absolute
            // seconds.
            "expired_in": 1_900_000_000i64,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    seed(
        &rig.store,
        rig.provider_id,
        OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000)),
    )
    .await;

    let credential = rig
        .models
        .refresh_credential(rig.provider_id)
        .await
        .unwrap();
    assert_eq!(credential.expires_at, Some(1_900_000_000_000));
}

#[tokio::test]
async fn rejected_refresh_preserves_the_prior_credential() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "status": "error",
            "base_resp": { "status_code": 400, "status_msg": "canary-secret-value" },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let stale = OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000));
    seed(&rig.store, rig.provider_id, stale.clone()).await;

    let err = rig
        .models
        .refresh_credential(rig.provider_id)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::ReLoginRequired { provider, .. } if provider == rig.provider_id),
        "unexpected error: {err}"
    );
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
    // Preserved for diagnosis, never silently deleted or overwritten.
    assert_eq!(
        rig.store.get(rig.provider_id).await.unwrap(),
        Some(Credential::OAuth(stale))
    );
}

#[tokio::test]
async fn transient_refresh_failure_preserves_the_credential() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(500).set_body_string("canary-secret-value"))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let stale = OAuthCredential::new("access-stale", Some("refresh-1".into()), Some(1_000));
    seed(&rig.store, rig.provider_id, stale.clone()).await;

    let err = rig
        .models
        .refresh_credential(rig.provider_id)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Auth(_)), "unexpected error: {err}");
    assert!(!err.to_string().contains("canary-secret-value"), "{err}");
    assert_eq!(
        rig.store.get(rig.provider_id).await.unwrap(),
        Some(Credential::OAuth(stale))
    );
}

#[tokio::test]
async fn https_violating_resource_url_fails_the_login_structurally() {
    let server = MockServer::start().await;
    mount_code_endpoint(&server, None, 1).await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "success",
            "access_token": "access-live",
            "refresh_token": "refresh-live",
            "expired_in": 3600,
            // Plain HTTP off loopback: never a credential-level base URL.
            "resource_url": "http://plain.example.com/anthropic",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let err = rig
        .models
        .login(
            rig.provider_id,
            &AuthInteraction::new(Arc::new(RecordingHandler::default())),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Config(_)), "unexpected error: {err}");
    assert!(!rig.models.check_auth(rig.provider_id).await.unwrap());
}

#[tokio::test]
async fn inference_sends_every_required_auth_header_after_merging() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        // The frozen contract maps the access token onto both headers the
        // Anthropic-compatible endpoint requires.
        .and(header("authorization", "Bearer access-live"))
        .and(header("x-api-key", "access-live"))
        // …and deterministic header merging keeps the layers around them.
        .and(header("x-trace", "t-1"))
        .respond_with(anthropic_ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    seed(
        &rig.store,
        rig.provider_id,
        OAuthCredential::new(
            "access-live",
            Some("refresh-1".into()),
            Some(now_ms() + 3_600_000),
        ),
    )
    .await;

    let mut model = minimax_model(rig.provider_id, &server.uri());
    model.headers = ProviderHeaders::from([("x-trace".to_string(), Some("t-1".to_string()))]);
    let message = rig
        .models
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");
}

#[tokio::test]
async fn credential_resource_url_overrides_the_inference_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("authorization", "Bearer access-live"))
        .and(header("x-api-key", "access-live"))
        .respond_with(anthropic_ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig(MiniMaxRegion::Cn, &server.uri());
    seed(
        &rig.store,
        rig.provider_id,
        OAuthCredential::new(
            "access-live",
            Some("refresh-1".into()),
            Some(now_ms() + 3_600_000),
        )
        .with_resource_url(server.uri())
        .expect("loopback resource URL is valid"),
    )
    .await;

    // The model's own base URL is unreachable: the request only succeeds when
    // the credential-level resource URL replaces it.
    let model = minimax_model(rig.provider_id, "http://127.0.0.1:9/anthropic");
    let message = rig
        .models
        .stream(
            &model,
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hi");
}

#[tokio::test]
async fn env_api_key_wins_over_the_stored_credential() {
    let server = MockServer::start().await;
    // An explicit API key resolves through the protocol-native header alone —
    // no OAuth bearer rides along.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-env-key"))
        .respond_with(anthropic_ok_response())
        .expect(1)
        .mount(&server)
        .await;

    let rig = rig_with_api_key_env(
        MiniMaxRegion::Global,
        &server.uri(),
        "BANSHU_MINIMAX_TEST_ENV_PRIORITY",
    );
    seed(
        &rig.store,
        rig.provider_id,
        OAuthCredential::new(
            "access-live",
            Some("refresh-1".into()),
            Some(now_ms() + 3_600_000),
        ),
    )
    .await;

    // The rig's provider reads this variable; setting it is the explicit
    // operator choice that wins over the stored credential.
    let var = "BANSHU_MINIMAX_TEST_ENV_PRIORITY";
    let saved = std::env::var(var).ok();
    unsafe { std::env::set_var(var, "sk-env-key") };
    let message = rig
        .models
        .stream(
            &minimax_model(rig.provider_id, &server.uri()),
            &Context::new().user("hi"),
            &StreamOptions::default(),
        )
        .finish()
        .await;
    match saved {
        Some(value) => unsafe { std::env::set_var(var, value) },
        None => unsafe { std::env::remove_var(var) },
    }

    assert_eq!(message.stop_reason, StopReason::Stop);
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].headers.contains_key("authorization"),
        "an explicit API key stands alone; no OAuth bearer rides along"
    );
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

    let rig = rig(MiniMaxRegion::Global, &server.uri());
    let message = rig
        .models
        .stream(
            &minimax_model(rig.provider_id, &server.uri()),
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
async fn minimax_constructors_wire_the_regional_oauth_lifecycle() {
    let global = Provider::minimax(
        MiniMaxRegion::Global,
        Arc::new(InMemoryCredentialStore::new()),
    );
    assert_eq!(global.id(), "minimax");
    assert_eq!(global.name(), "MiniMax");
    assert_eq!(global.base_url(), "https://api.minimax.io/anthropic");
    assert!(
        global.oauth_session().is_some(),
        "the bundled MiniMax provider participates in the OAuth lifecycle"
    );

    let cn = Provider::minimax(MiniMaxRegion::Cn, Arc::new(InMemoryCredentialStore::new()));
    assert_eq!(cn.id(), "minimax-cn");
    assert_eq!(cn.name(), "MiniMax CN");
    assert_eq!(cn.base_url(), "https://api.minimaxi.com/anthropic");
    assert!(cn.oauth_session().is_some());
    // The CN region serves the same bundled catalog, stamped with its own
    // provider id and inference endpoint.
    let models = cn.models();
    assert!(
        !models.is_empty(),
        "the CN provider serves the MiniMax catalog"
    );
    assert!(
        models.iter().all(|model| model.provider == "minimax-cn"
            && model.base_url == "https://api.minimaxi.com/anthropic"),
        "catalog models are stamped with the CN provider and endpoint"
    );

    // With no credential stored and no MINIMAX_API_KEY in the environment,
    // neither region reports authenticated.
    let saved = std::env::var("MINIMAX_API_KEY").ok();
    unsafe { std::env::remove_var("MINIMAX_API_KEY") };
    let registry = Models::new().with_provider(global).with_provider(cn);
    assert!(!registry.check_auth("minimax").await.unwrap());
    assert!(!registry.check_auth("minimax-cn").await.unwrap());
    if let Some(saved) = saved {
        unsafe { std::env::set_var("MINIMAX_API_KEY", saved) };
    }
}
