/// The roles a user can have (mirrors the backend `UserRole` enum)
export enum RoleKey {
  /// An admin can see all data in Thorium and perform any action
  Admin = 'Admin',
  /// An Analyst is given access to all samples in Thorium
  Analyst = 'Analyst',
  /// A developer can create tools
  Developer = 'Developer',
  /// A user can upload files and run jobs
  User = 'User',
}

export enum ScopedTokenRoleKey {
  Developer = 'Developer',
  User = 'User',
}

/// The developer role's sub-permissions
export type ThoriumDeveloperRoleValue = { k8s: boolean; bare_metal: boolean; windows: boolean; external: boolean; kvm: boolean };

export type Role = {
  Admin: RoleKey.Admin;
  Analyst: RoleKey.Analyst;
  Developer: {
    Developer: ThoriumDeveloperRoleValue;
  };
  User: RoleKey.User;
};

export type ThoriumRole = {
  Admin?: RoleKey.Admin;
  Analyst?: RoleKey.Analyst;
  Developer?: ThoriumDeveloperRoleValue;
  User?: RoleKey.User;
};

/// The info to inject about this user on a Unix/Linux system
export type UnixInfo = {
  /// The unix user id of this user
  user: number;
  /// The unix group id of this user
  group: number;
};

/// An AI endpoint configuration
export type AiEndpoint = {
  /// The URL endpoint for the AI service
  url: string;
  /// The API key for the AI service
  api_key: string;
  /// The model to use from this AI service
  model: string;
};

/// AI settings for a user
export type AiSettings = {
  /// Multiple named AI endpoints
  endpoints: { [name: string]: AiEndpoint };
  /// The default endpoint name
  default_endpoint: string;
};

/// Any user specific settings
export type UserSettings = {
  /// The theme this user uses in the webUI
  theme: string;
  /// The AI settings for this user
  ai?: AiSettings;
};

/// A user within Thorium that does not have its password
export type UserInfo = {
  /// The username of this user
  username: string;
  /// The role for this user
  role: ThoriumRole;
  /// This users email
  email: string;
  /// The groups this user is in (or all groups for admin)
  groups: string[];
  /// Admin-only list of groups admin user is 'actually' in
  actual_groups?: string[];
  /// The token for this user
  token: string;
  /// When this users token expires
  token_expiration: string;
  /// The info to inject about this user on unix/linux systems
  unix?: UnixInfo;
  /// The settings this user has set
  settings: UserSettings;
  /// Whether this user is a local user or not
  local: boolean;
  /// Whether this user has been verified already or not
  verified: boolean;
};

/// Response to a successful auth
export type UserAuthResponse = {
  /// The token to use to talk to Thorium
  token: string;
  /// The date/time this token expires
  expires: string;
};

/// The raw wire shape of the API `AuthResponse` enum (serde externally tagged).
/// Either a successful auth carrying a token, or a prompt to verify the user's email.
export type RawAuthResponse = { Authed: { token: string; expires: string } } | { VerifyEmail: string };

/// Normalized outcome of a registration attempt.
///
/// `authed` is returned when the deployment auto-verifies the account (no email backend
/// configured, or verification skipped) and the user is immediately logged in. `verify_email`
/// is returned when the user must verify their email before they can log in.
export type CreateUserResult = { status: 'authed'; token: string; expires: string } | { status: 'verify_email'; email: string };

/// Normalized outcome of a password authentication attempt. `authed` carries a token; `verify_email`
/// means the account exists but its email is not yet verified (mirrors the API `AuthResponse` enum).
export type PasswordAuthResult = { status: 'authed'; token: string; expires: string } | { status: 'verify_email'; email: string };

/// Normalized outcome of a resend-verification-email request.
export enum ResendVerificationStatus {
  /// A new verification email was sent (`200`).
  Sent = 'sent',
  /// Rate-limited: a verification email was sent too recently (`429`).
  Cooldown = 'cooldown',
  /// The account's email is already verified, so resending is not possible (`409`).
  AlreadyVerified = 'already_verified',
  /// The request failed for another reason (error already surfaced to the caller).
  Failed = 'failed',
}

/// Result of `resendVerificationEmail`. `Sent`/`Cooldown` carry the seconds to wait before retrying.
export type ResendVerificationResult =
  | { status: ResendVerificationStatus.Sent; retryAfterSecs: number }
  | { status: ResendVerificationStatus.Cooldown; retryAfterSecs: number }
  | { status: ResendVerificationStatus.AlreadyVerified }
  | { status: ResendVerificationStatus.Failed };

/// The outcome of confirming an emailed verification link (`verifyEmail`). The backend collapses every
/// failure mode (wrong/used/expired token, unknown account) into a single 401 so the unauthenticated
/// endpoint never reveals whether an account exists, hence `Expired` covers all of them. `Error` is
/// reserved for unexpected failures (e.g. the network is down) so we don't falsely claim the single-use
/// token was consumed.
export enum EmailVerifyStatus {
  /// The email was verified (204)
  Verified = 'verified',
  /// The verification token is invalid, expired, or already used (uniform 401)
  Expired = 'expired',
  /// An unexpected failure occurred (network error, unexpected status)
  Error = 'error',
}

/// A scoped token belonging to the current user (mirrors the backend `ScopedToken` struct)
export type ScopedToken = {
  /// The token name (unique per user, 1-50 lowercase characters)
  name: string;
  /// Username of the owner
  owner: string;
  /// Role for scoped token
  role: ScopedTokenRole;
  /// Groups this token is limited to
  groups: string[];
  /// Current token value (hex-encoded); rotates on expiration
  token: string;
  /// When the current token value expires (ISO datetime); triggers auto-rotation
  token_expiration: string;
  /// Optional permanent expiration after which the token is deleted (ISO datetime or null)
  expires: string | null;
};

/// Request body for creating a scoped token (mirrors `ScopedTokenRequest`)
export type ScopedTokenRequest = {
  /// Name of the scoped token
  name: string;
  /// Role for scoped token
  role: ScopedTokenRole;
  /// List of groups for scoped token
  groups: string[];
  /// Optional expiry datetime
  expires?: string;
};

/// Request body for updating a scoped token (mirrors `ScopedTokenUpdate`)
export type ScopedTokenUpdate = {
  /// Role for scoped token
  role: ScopedTokenRole;
  add_groups: string[];
  remove_groups: string[];
  expires?: string;
  clear_expires: boolean;
};

export type ScopedTokenRole =
  | {
      Developer: ThoriumDeveloperRoleValue;
    }
  | ScopedTokenRoleKey.User;
