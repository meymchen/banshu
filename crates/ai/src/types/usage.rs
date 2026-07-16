//! Token usage and computed cost for a completion.

/// Computed cost in USD, broken out by token class.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cost {
    /// Cost of input (prompt) tokens.
    pub input: f64,
    /// Cost of output (completion) tokens.
    pub output: f64,
    /// Cost of cache-read tokens.
    pub cache_read: f64,
    /// Cost of cache-write tokens.
    pub cache_write: f64,
    /// Sum of all cost components.
    pub total: f64,
}

/// Token counts reported by the provider plus derived [`Cost`].
///
/// The cache fields are kept even though prompt caching is deferred, so that
/// cost stays correct once caching lands without a type change.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    /// Input (prompt) tokens.
    pub input: u64,
    /// Output (completion) tokens.
    pub output: u64,
    /// Tokens served from cache.
    pub cache_read: u64,
    /// Tokens written to cache.
    pub cache_write: u64,
    /// Reasoning tokens, when the provider reports them (a subset of `output`).
    pub reasoning: Option<u64>,
    /// Total tokens across input and output.
    pub total_tokens: u64,
    /// Computed monetary cost.
    pub cost: Cost,
}
