// TypeScript mirror of `api/src/models/oauth.rs` (repo: thorium-oauth, branch OAuthSupport).
// The OAuth/OIDC flow reuses the same opaque Thorium token as password login, so the
// `Authed` arm carries the existing `UserAuthResponse` shape.

// project imports
import { ThoriumRole, UserSettings, UserAuthResponse } from './users';

/// The successful-login arm of the callback response (an existing, linked user).
export type OAuthAuthed = {
  /// The Thorium token + expiry, identical to a password login response
  Authed: UserAuthResponse;
};

/// The new-user arm of the callback response: a short-lived registration session token
/// for an OIDC identity that is not yet linked to any Thorium account.
export type OAuthNewUser = {
  /// The registration session token to pass back to `POST /oauth/{name}/register`
  NewUser: string;
};

/// The verify-email arm: the OAuth identity is linked to an existing account whose email is not yet
/// verified, so no token is issued until the user verifies via the emailed link and signs in again.
export type OAuthVerifyEmail = {
  /// The email address that still needs to be verified
  VerifyEmail: string;
};

/// Response body of `GET /api/oauth/{name}/callback`.
///
/// Mirrors the externally-tagged Rust enum `OAuthMaybeAuthed`, which serializes as one of
/// `{ "Authed": { token, expires } }`, `{ "VerifyEmail": "<email>" }`, or `{ "NewUser": "<session_token>" }`.
export type OAuthMaybeAuthed = OAuthAuthed | OAuthVerifyEmail | OAuthNewUser;

/// Type guard: the callback authenticated an existing linked user.
export function isAuthed(result: OAuthMaybeAuthed): result is OAuthAuthed {
  return 'Authed' in result;
}

/// Type guard: the linked account must verify its email before a token is issued.
export function isVerifyEmail(result: OAuthMaybeAuthed): result is OAuthVerifyEmail {
  return 'VerifyEmail' in result;
}

/// Type guard: the callback identity has no Thorium account yet and must register.
export function isNewUser(result: OAuthMaybeAuthed): result is OAuthNewUser {
  return 'NewUser' in result;
}

/// Request body for `POST /api/oauth/{name}/register`.
///
/// NOTE: the backend currently forces `role` to `User` and only honors `skip_verification`
/// with a secret key, so the registration UI does not surface those fields — they are mirrored
/// here only to match the API model.
export type OAuthUserCreate = {
  /// The registration session token returned by the callback (`OAuthNewUser.NewUser`)
  session_token: string;
  /// The desired username for the new account
  username: string;
  /// The email for the new account (must be unique across Thorium)
  email: string;
  /// The requested role (ignored by the backend, forced to User)
  role?: ThoriumRole;
  /// Initial user settings (defaulted server-side if omitted)
  settings?: UserSettings;
  /// Skip email verification (requires a secret key; not surfaced in the UI)
  skip_verification?: boolean;
};

/// Request body for `POST /api/oauth/{name}/username/available`.
export type OAuthUsernameCheck = {
  /// The username to check for availability
  username: string;
  /// A valid registration session token (required to mitigate username enumeration)
  session_token: string;
};

/// Query params for `GET`/`DELETE /api/oauth/{name}/link` (delivered via an account-link email).
export type OAuthLinkParams = {
  /// The existing Thorium username the new provider would be linked to
  username: string;
  /// The single-use, time-limited link token from the email
  token: string;
};

/// Query params the IdP appends when redirecting back to `/oauth/{name}/callback`.
export type OAuthCallbackParams = {
  /// The OAuth authorization code
  code: string;
  /// The CSRF state value (owned/validated by the backend; the SPA forwards it verbatim)
  state: string;
};

/// The outcome of submitting an OAuth registration, distinguishing a freshly created account from the
/// account-link flow (the email already belongs to an existing account, so the server emailed a link
/// instead of creating an account) and hard errors.
export enum OAuthRegisterStatus {
  /// A brand-new, auto-verified account was created; an auth token was issued
  Created = 'created',
  /// The account was created but its email must be verified before a token is issued
  VerifyEmail = 'verify_email',
  /// The email matches an existing account — the server emailed an account-link link to finish linking
  LinkEmailSent = 'link_email_sent',
  /// Registration failed (e.g. the email belongs to a different user)
  Error = 'error',
}

/// Discriminated result of `registerOAuthUser`.
export type OAuthRegisterResult =
  | { status: OAuthRegisterStatus.Created; auth: UserAuthResponse }
  | { status: OAuthRegisterStatus.VerifyEmail; email: string }
  | { status: OAuthRegisterStatus.LinkEmailSent }
  | { status: OAuthRegisterStatus.Error; message: string };

/// The outcome of confirming an emailed account-link (`confirmOAuthLink`). The backend collapses every
/// failure mode (bad/used/expired token, OAuth not configured) into a single 401 so the unauthenticated
/// endpoint never reveals account or configuration state, hence `Expired` covers all of them. `Error` is
/// reserved for unexpected failures (e.g. the network is down) so we don't falsely claim the single-use
/// token was consumed.
export enum OAuthLinkConfirmStatus {
  /// The alias was linked (204)
  Linked = 'linked',
  /// The link token is invalid, expired, or already used (uniform 401)
  Expired = 'expired',
  /// An unexpected failure occurred (network error, unexpected status)
  Error = 'error',
}
