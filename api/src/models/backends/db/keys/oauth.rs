//! The keys to OAuth data in Redis

use openidconnect::CsrfToken;

use crate::utils::Shared;

/// The key to a oauth context from a oauth state/CSRF token
///
/// # Arguments
///
/// * `provider` - The provider this oauth context is for
/// * `csrf_token` - The state or CSRF token related to this nonce
/// * `shared` - Shared Thorium objects
pub fn context(provider: &str, csrf_token: &CsrfToken, shared: &Shared) -> String {
    format!(
        "{ns}:oauth:context:{provider}:{csrf_token}",
        ns = shared.config.thorium.namespace,
        provider = provider,
        csrf_token = csrf_token.secret()
    )
}

/// The key to map provider aliases to usernames
///
/// # Arguments
///
/// * `provider` - The name of the provider to map an alias to a username for
/// * `shared` - Shared Thorium objects
pub fn alias_to_username(provider: &str, shared: &Shared) -> String {
    format!(
        "{ns}:oauth:alias_to_user:{provider}",
        ns = shared.config.thorium.namespace,
        provider = provider
    )
}

/// The key to OAuth registration session info
///
/// This does not use a CSRF token, state, or nonce from the OAuth flow.
///
/// # Arguments
///
/// * `provider` - The provider this session is for
/// * `token` - The oauth registration token for this registration session
/// * `shared` - Shared Thorium objects
pub fn registration_session(provider: &str, token: &str, shared: &Shared) -> String {
    format!(
        "{ns}:oauth:registration_sessions:{provider}:{token}",
        ns = shared.config.thorium.namespace,
        provider = provider,
        token = token
    )
}

/// The key to an OAuth link verification token
///
/// # Arguments
///
/// * `provider` - The provider this session is for
/// * `username` - The user who wants to link a new OAuth provider to their account
/// * `token` - The oauth link verification token for this new auth link
/// * `shared` - Shared Thorium objects
pub fn link_token(provider: &str, username: &str, token: &str, shared: &Shared) -> String {
    format!(
        "{ns}:oauth:link_token:{provider}:{username}:{token}",
        ns = shared.config.thorium.namespace,
        provider = provider,
        token = token
    )
}
