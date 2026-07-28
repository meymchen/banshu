#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
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

mod auth;
mod cancel;
mod discovery;
mod error;
mod executor;
mod http;
mod models;
mod options;
mod registry;
mod sse;
mod stream;
mod types;

pub use api::{PreparedRequest, ProtocolAdapter, ProtocolEvent, ProtocolEventStream};
pub use async_trait::async_trait;
pub use auth::{Auth, AuthResolver, ProviderHeaders, ResolvedAuth};
pub use discovery::{RefreshEntry, RefreshOutcome, RefreshReport};
pub use error::{Error, ErrorKind, Result};
pub use options::{CacheRetention, StreamOptions};
pub use provider::{AnthropicCompat, OpenAiCompat, OpenAiPromptCaching, Provider, ProviderBuilder};
pub use registry::Models;
pub use stream::{AssistantMessageEvent, MessageStream};
/// Re-exported so callers can construct a [`StreamOptions::cancellation`]
/// token without adding their own `tokio-util` dependency.
pub use tokio_util::sync::CancellationToken;
pub use types::*;
