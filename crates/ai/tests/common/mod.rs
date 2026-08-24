use std::sync::Arc;

use banshu_ai::{
    AuthInteraction, AuthInteractionHandler, CancellationToken, Result, VerificationDetails,
    async_trait,
};

struct CancellingHandler {
    token: CancellationToken,
}

#[async_trait]
impl AuthInteractionHandler for CancellingHandler {
    async fn show_verification(&self, _details: &VerificationDetails) -> Result<()> {
        Ok(())
    }

    async fn open_browser(&self, _url: &str) -> Result<bool> {
        self.token.cancel();
        Ok(true)
    }
}

/// Builds an interaction that cancels immediately after opening the browser.
pub fn cancelling_interaction(token: CancellationToken) -> AuthInteraction {
    AuthInteraction::new(Arc::new(CancellingHandler {
        token: token.clone(),
    }))
    .with_cancellation(token)
}
