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
pub mod provider;

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

pub use discovery::{RefreshEntry, RefreshOutcome, RefreshReport};
pub use error::{Error, ErrorKind, Result};
pub use options::{CacheRetention, StreamOptions};
pub use provider::{AnthropicCompat, OpenAiCompat, OpenAiPromptCaching, Provider};
pub use registry::Models;
pub use stream::{AssistantMessageEvent, MessageStream};
pub use types::*;
