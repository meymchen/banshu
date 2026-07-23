//! Cooperative cancellation, shared by [`crate::executor`] and both protocol
//! adapters, wired to
//! [`StreamOptions::cancellation`](crate::StreamOptions::cancellation).
//!
//! Built on `futures_util::future::select` rather than `tokio::select!` so
//! this crate doesn't need tokio's `macros` feature just for this.

use std::future::Future;

use futures_util::future::{self, Either};
use futures_util::pin_mut;
use tokio_util::sync::CancellationToken;

/// The awaited operation was cancelled via a [`CancellationToken`].
pub(crate) struct Aborted;

/// Race `fut` against `token`'s cancellation, if a token is set.
///
/// `token.cancelled()` is polled first, so an already-cancelled token wins
/// even when `fut` would also be ready on the same poll.
pub(crate) async fn race<F: Future>(
    token: Option<&CancellationToken>,
    fut: F,
) -> Result<F::Output, Aborted> {
    let Some(token) = token else {
        return Ok(fut.await);
    };
    let cancelled = token.cancelled();
    pin_mut!(cancelled);
    pin_mut!(fut);
    match future::select(cancelled, fut).await {
        Either::Left(((), _)) => Err(Aborted),
        Either::Right((value, _)) => Ok(value),
    }
}
