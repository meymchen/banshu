//! Resolve the output cap from explicit options and known model limits.

use crate::{Context, Model, StreamOptions};

/// Return the output cap the request should ship, or why the caller's explicit
/// cap cannot fit known model capacity.
pub(super) fn resolve(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<Option<u32>, String> {
    let remaining = (model.context_window > 0).then(|| {
        u64::from(model.context_window)
            .saturating_sub(context.estimate_tokens())
            .try_into()
            .expect("remaining context cannot exceed its u32 model limit")
    });
    let model_max = (model.max_tokens > 0).then_some(model.max_tokens);

    if let Some(requested) = options.max_tokens {
        if remaining.is_some_and(|known| requested > known) {
            return Err(format!(
                "requested max_tokens of {requested} exceeds the remaining context budget of {} \
                 tokens for model `{}`",
                remaining.expect("checked as known"),
                model.id,
            ));
        }
        return Ok(Some(requested));
    }

    Ok(match (model_max, remaining) {
        (Some(maximum), Some(remaining)) => Some(maximum.min(remaining)),
        (Some(maximum), None) => Some(maximum),
        (None, Some(remaining)) => Some(remaining),
        (None, None) => None,
    })
}
