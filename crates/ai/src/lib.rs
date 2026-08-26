#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![doc = include_str!("../../../README.md")]
//! # banshu-ai
//!
//! Unified LLM API with automatic model discovery and provider configuration.
//!
//! This is the core crate of the `banshu` workspace. It provides a single,
//! provider-agnostic surface for talking to language models, focused on
//! open-source models and providers:
//!
//! - DeepSeek
//! - Z.AI
//! - MiniMax
//! - Moonshot AI
//! - Kimi For Coding
//! - Xiaomi MiMo
//! - Any OpenAI-compatible API
//! - Any Anthropic-compatible API

pub mod api;
/// Shared models.dev mapping — a published seam for `xtask generate-catalog`,
/// not curated end-user API.
#[doc(hidden)]
pub mod models_dev;
pub mod provider;
pub mod testing;

mod auth;
mod cancel;
mod credentials;
mod discovery;
mod error;
mod executor;
mod http;
mod kimi;
mod minimax;
mod models;
mod models_store;
mod oauth;
mod observer;
mod options;
mod overflow;
mod partial_json;
mod registry;
mod sse;
mod stream;
mod types;

pub use api::{PreparedRequest, ProtocolAdapter, ProtocolEvent, ProtocolEventStream};
pub use async_trait::async_trait;
pub use auth::{Auth, AuthResolver, OAuthAuth, ProviderHeaders, ResolvedAuth};
pub use credentials::{
    ApiKeyCredential, Credential, CredentialStore, InMemoryCredentialStore, ModifyCredential,
    OAuthCredential,
};
pub use discovery::{RefreshEntry, RefreshOutcome, RefreshReport};
pub use error::{Error, ErrorKind, Result};
pub use kimi::{KIMI_AUTH_HOST, KIMI_CLIENT_ID, KimiDeviceFlow};
pub use minimax::{MINIMAX_CLIENT_ID, MINIMAX_OAUTH_SCOPE, MiniMaxPortalFlow, MiniMaxRegion};
pub use models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry, RefreshOptions};
pub use oauth::{
    AuthInteraction, AuthInteractionHandler, DEFAULT_LOGIN_TIMEOUT, OAuthFlow, OAuthSession,
    RefreshError, VerificationDetails,
};
pub use observer::{BeforeSendObservation, RequestObserver, ResponseObservation};
pub use options::{CacheRetention, StreamOptions};
pub use overflow::is_context_overflow;
pub use provider::{
    AnthropicCacheRetention, AnthropicCompat, AnthropicReasoningFormat, OpenAiCacheRetention,
    OpenAiChatTemplateKwargs, OpenAiCompat, OpenAiOutputTokenField, OpenAiReasoningBudgetField,
    OpenAiReasoningFormat, OpenAiSessionAffinity, OpenAiStreamTermination, Provider,
    ProviderBuilder, ToolChoiceSupport,
};
pub use registry::Models;
pub use stream::{AssistantMessageEvent, MessageStream};
/// Re-exported so callers can construct a [`StreamOptions::cancellation`]
/// token without adding their own `tokio-util` dependency.
pub use tokio_util::sync::CancellationToken;
pub use types::*;
