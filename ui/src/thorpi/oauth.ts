import axios from 'axios';

// project imports
import client, { parseRequestError } from './client';
import { getApiUrl } from '@utilities/url';
import {
  OAuthLinkConfirmStatus,
  OAuthMaybeAuthed,
  OAuthRegisterResult,
  OAuthRegisterStatus,
  OAuthUserCreate,
  OAuthUsernameCheck,
} from '@models/oauth';
import { RawAuthResponse } from '@models/users';

/// Extract a human-readable message from an axios API error (string body or `{ error }` object).
function extractApiMessage(error: unknown): string {
  if (axios.isAxiosError(error) && error.response) {
    const data: unknown = error.response.data;
    if (typeof data === 'string') {
      return data;
    }
    if (data && typeof data === 'object' && 'error' in data) {
      const inner: unknown = data.error;
      return typeof inner === 'string' ? inner : JSON.stringify(inner);
    }
  }
  return '';
}

/**
 * List the OAuth/OIDC providers configured for this Thorium instance.
 *
 * A `401` response means OAuth is simply not configured; this is treated as "no providers"
 * and is swallowed silently (the caller renders only the password form) rather than surfacing
 * a misleading "Permission Denied" on the login page.
 *
 * @param errorHandler - Called with a formatted message on a non-401 failure.
 * @returns The provider names, or `null` if OAuth is disabled or the request failed.
 */
export async function listOAuthProviders(errorHandler: (error: string) => void): Promise<string[] | null> {
  return client
    .get<string[]>('/oauth/')
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      // 401 = OAuth not configured for this instance — not an error worth showing.
      if (axios.isAxiosError(error) && error.response?.status == 401) {
        return null;
      }
      parseRequestError(error, errorHandler, 'List OAuth Providers');
      return null;
    });
}

/**
 * Exchange the authorization `code`/`state` the IdP appended to the callback URL for either an
 * authenticated session (existing linked user) or a new-user registration token.
 *
 * @param provider - The provider name from the callback route.
 * @param code - The OAuth authorization code from the IdP redirect.
 * @param state - The CSRF state from the IdP redirect (forwarded verbatim; validated server-side).
 * @param errorHandler - Called with a formatted message if the exchange fails.
 * @returns An `OAuthMaybeAuthed` (Authed or NewUser), or `null` on failure.
 */
export async function exchangeOAuthCallback(
  provider: string,
  code: string,
  state: string,
  errorHandler: (error: string) => void,
): Promise<OAuthMaybeAuthed | null> {
  const url = `/oauth/${encodeURIComponent(provider)}/callback`;
  return client
    .get<OAuthMaybeAuthed>(url, { params: { code, state } })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'OAuth Sign-in');
      return null;
    });
}

/**
 * Register a new Thorium account for a freshly-authenticated OIDC identity, or initiate account
 * linking when the email already belongs to an existing account.
 *
 * The backend returns `409` for two distinct cases: when the email belongs to the *same* username it
 * emails an account-link link (the linking happy path), and when it belongs to a *different* user it
 * is a hard error. We disambiguate on the message so the UI can treat the link case as success.
 *
 * @param provider - The provider name.
 * @param body - The registration request (carries the in-memory session token).
 * @returns A discriminated {@link OAuthRegisterResult}: created, link-email-sent, or error.
 */
export async function registerOAuthUser(provider: string, body: OAuthUserCreate): Promise<OAuthRegisterResult> {
  const url = `/oauth/${encodeURIComponent(provider)}/register`;
  return client
    .post<RawAuthResponse>(url, body)
    .then((res): OAuthRegisterResult => {
      if (res?.status == 200 && res.data) {
        const data = res.data;
        // a brand-new, auto-verified account is authed immediately and carries a token
        if ('Authed' in data) {
          return { status: OAuthRegisterStatus.Created, auth: { token: data.Authed.token, expires: data.Authed.expires } };
        }
        // otherwise the account was created but its email must be verified first
        return { status: OAuthRegisterStatus.VerifyEmail, email: data.VerifyEmail };
      }
      return { status: OAuthRegisterStatus.Error, message: 'Unexpected response while creating the account.' };
    })
    .catch((error: unknown): OAuthRegisterResult => {
      const message = extractApiMessage(error);
      // A 409 pointing at the account-link email is the linking happy path, not a failure.
      if (axios.isAxiosError(error) && error.response?.status == 409 && /check your email|account link/i.test(message)) {
        return { status: OAuthRegisterStatus.LinkEmailSent };
      }
      return { status: OAuthRegisterStatus.Error, message: message || 'Failed to create the account.' };
    });
}

/**
 * Check whether a username is available during OAuth registration.
 *
 * A `409` response means the username is taken — an expected answer, not an error, so it does
 * not call `errorHandler`.
 *
 * @param provider - The provider name.
 * @param body - The username to check plus the registration session token.
 * @param errorHandler - Called with a formatted message on an unexpected (non-409) failure.
 * @returns `true` if available (`204`), `false` if taken (`409`) or on failure.
 */
export async function checkOAuthUsername(
  provider: string,
  body: OAuthUsernameCheck,
  errorHandler: (error: string) => void,
): Promise<boolean> {
  const url = `/oauth/${encodeURIComponent(provider)}/username/available`;
  return client
    .post(url, body)
    .then((res) => res?.status == 204)
    .catch((error: unknown) => {
      // 409 = taken; a normal answer rather than an error.
      if (axios.isAxiosError(error) && error.response?.status == 409) {
        return false;
      }
      parseRequestError(error, errorHandler, 'Check Username');
      return false;
    });
}

/**
 * Confirm an emailed account-link (the "yes, link this provider" action on the link page).
 *
 * The backend returns `204` on success and a uniform `401` for every failure (bad/used/expired token,
 * OAuth not configured) so it never reveals account or configuration state. The `401` is therefore an
 * expected answer, not an error: it maps to {@link OAuthLinkConfirmStatus.Expired} and does NOT call
 * `errorHandler` (same approach as {@link listOAuthProviders}'s 401 handling). Only unexpected failures
 * (e.g. the network is down) map to {@link OAuthLinkConfirmStatus.Error}, so a transient outage is never
 * misreported as a consumed single-use token.
 *
 * @param provider - The provider name.
 * @param username - The existing Thorium username (from the email link).
 * @param token - The single-use link token (from the email link).
 * @param errorHandler - Called with a formatted message only on an unexpected (non-401) failure.
 * @returns An {@link OAuthLinkConfirmStatus}: `Linked` (204), `Expired` (401), or `Error`.
 */
export async function confirmOAuthLink(
  provider: string,
  username: string,
  token: string,
  errorHandler: (error: string) => void,
): Promise<OAuthLinkConfirmStatus> {
  const url = `/oauth/${encodeURIComponent(provider)}/link`;
  return client
    .get(url, { params: { username, token } })
    .then((res) => (res?.status == 204 ? OAuthLinkConfirmStatus.Linked : OAuthLinkConfirmStatus.Error))
    .catch((error: unknown) => {
      // 401 = invalid/expired/used token (uniform anti-enumeration answer); an expected outcome.
      if (axios.isAxiosError(error) && error.response?.status == 401) {
        return OAuthLinkConfirmStatus.Expired;
      }
      parseRequestError(error, errorHandler, 'Confirm Account Link');
      return OAuthLinkConfirmStatus.Error;
    });
}

/**
 * Revoke a pending account-link request (the "this wasn't me / cancel" action on the link page).
 *
 * @param provider - The provider name.
 * @param username - The existing Thorium username (from the email link).
 * @param token - The single-use link token (from the email link).
 * @param errorHandler - Called with a formatted message on failure.
 * @returns `true` on success (`204`), `false` otherwise.
 */
export async function revokeOAuthLink(
  provider: string,
  username: string,
  token: string,
  errorHandler: (error: string) => void,
): Promise<boolean> {
  const url = `/oauth/${encodeURIComponent(provider)}/link`;
  return client
    .delete(url, { params: { username, token } })
    .then((res) => res?.status == 204)
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Cancel Account Link');
      return false;
    });
}

/**
 * Build the absolute URL that starts the OAuth flow for a provider.
 *
 * This is NOT an axios call: the endpoint responds with a `303` redirect to the external IdP, so
 * the browser must perform a real top-level navigation (`window.location.assign`) — an XHR would
 * try to follow the cross-origin redirect and fail. Uses {@link getApiUrl} so it targets the API
 * origin (which may differ from the SPA origin in local dev).
 *
 * @param provider - The provider name to start authentication with.
 * @returns The fully-qualified `/api/oauth/{provider}/auth` URL.
 */
export function buildOAuthAuthUrl(provider: string): string {
  return `${getApiUrl()}/oauth/${encodeURIComponent(provider)}/auth`;
}
