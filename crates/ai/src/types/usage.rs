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
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    /// Uncached input (prompt) tokens, billed at the normal input rate.
    pub input: u64,
    /// Output (completion) tokens.
    pub output: u64,
    /// Tokens served from cache.
    pub cache_read: u64,
    /// Tokens written to cache.
    pub cache_write: u64,
    /// Tokens written with the 1h cache TTL (a subset of `cache_write`), when
    /// the provider reports them. Anthropic bills these at twice the input
    /// rate instead of the short-TTL cache-write rate.
    pub cache_write_1h: Option<u64>,
    /// Reasoning tokens, when the provider reports them (a subset of `output`).
    pub reasoning: Option<u64>,
    /// Total tokens across input and output.
    pub total_tokens: u64,
    /// Computed monetary cost.
    pub cost: Cost,
}
