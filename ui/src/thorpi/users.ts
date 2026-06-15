import axios, { AxiosResponse } from 'axios';
import client, { parseRequestError, thoriumCookieAuthConfig } from './client';

// project imports
import { profileGraphicFilename } from '@utilities/profileGraphic';
import {
  CreateUserResult,
  EmailVerifyStatus,
  PasswordAuthResult,
  RawAuthResponse,
  ResendVerificationResult,
  ResendVerificationStatus,
  ScopedToken,
  ScopedTokenRequest,
  ScopedTokenUpdate,
  UserInfo,
} from '@models/users';

/**
 * Authenticate with a username and password (`POST /users/auth`).
 *
 * Credentials are sent as an HTTP Basic `Authorization` header.
 *
 * @param username - The user's username.
 * @param password - The user's password.
 * @param errorHandler - Called with a formatted message if authentication fails.
 * @returns A {@link PasswordAuthResult} (`authed` with a token, or `verify_email`), or `null` if
 *   authentication failed.
 */
export async function authUserPass(
  username: string,
  password: string,
  errorHandler: (error: string) => void,
): Promise<PasswordAuthResult | null> {
  const url = '/users/auth';
  const header = { Authorization: 'basic ' + btoa(username + ':' + password) };
  return client
    .post<RawAuthResponse>(url, {}, { headers: header })
    .then((res): PasswordAuthResult | null => {
      if (res?.status == 200) {
        const data = res.data;
        // a verified user is authed and carries a token
        if ('Authed' in data) {
          return { status: 'authed', token: data.Authed.token, expires: data.Authed.expires };
        }
        // the account exists but its email must be verified before login can complete
        return { status: 'verify_email', email: data.VerifyEmail };
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Password Auth');
      return null;
    });
}

/**
 * Validate an API token and exchange it for a session token (`POST /users/auth`).
 *
 * The token is sent as a `token`-scheme `Authorization` header. Failures are logged to the
 * console rather than surfaced through an error handler.
 *
 * @param token - The API token to authenticate with.
 * @returns The resulting session token, or `null` if the token was invalid or the request failed.
 */
export async function authUserToken(token: string): Promise<string | null> {
  const header = { Authorization: 'token ' + btoa(token) };
  return client
    .post<RawAuthResponse>('/users/auth', {}, { headers: header })
    .then((res) => {
      if (res?.status == 200 && 'Authed' in res.data) {
        return res.data.Authed.token;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, console.log, 'Token Auth');
      return null;
    });
}

/**
 * Create a new user (`POST /users/`).
 *
 * @param name - The username for the new user.
 * @param email - The new user's email address.
 * @param password - The new user's initial password.
 * @param role - The Thorium role to assign (e.g. `Admin`, `User`).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The {@link CreateUserResult} — either `authed` (auto-verified, includes a token) or
 *   `verify_email` (the user must verify their email before logging in) — or `null` if the
 *   request failed.
 */
export async function createUser(
  name: string,
  email: string,
  password: string,
  role: string,
  errorHandler: (error: string) => void,
): Promise<CreateUserResult | null> {
  const url = '/users/';
  const data = { username: name, email: email, password: password, role: role };
  return client
    .post<RawAuthResponse>(url, data)
    .then((res) => {
      if (res?.status == 200) {
        const body = res.data;
        // auto-verified deployments return a token the caller can log in with
        if ('Authed' in body) {
          return { status: 'authed' as const, token: body.Authed.token, expires: body.Authed.expires };
        }
        // otherwise the account was created but the user must verify their email first
        return { status: 'verify_email' as const, email: body.VerifyEmail };
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create User');
      return null;
    });
}

/**
 * Read the `Retry-After` header (in seconds) from an axios headers object.
 *
 * Exported for unit testing. Returns `0` when the header is missing or unparseable.
 *
 * @param headers - The axios response `headers` object (keys are lower-cased by axios).
 * @returns The number of seconds to wait, or `0` if not present/invalid.
 */
export function parseRetryAfter(headers: unknown): number {
  if (headers && typeof headers === 'object') {
    const raw = (headers as Record<string, unknown>)['retry-after'];
    const secs = typeof raw === 'string' ? parseInt(raw, 10) : typeof raw === 'number' ? raw : NaN;
    if (!Number.isNaN(secs) && secs > 0) {
      return secs;
    }
  }
  return 0;
}

/**
 * Resend the email-verification message for an unverified user
 * (`GET /users/resend/verify/email/{username}`).
 *
 * The endpoint is rate-limited; both the success (`200`) and rate-limited (`429`) responses carry a
 * `Retry-After` header (seconds) used to drive a cooldown countdown.
 *
 * @param username - The username to resend the verification email for.
 * @param errorHandler - Called with a formatted message when the email is already verified (`409`) or
 *   on an unexpected failure (not a `429` cooldown).
 * @returns A {@link ResendVerificationResult}: `Sent`/`Cooldown` carry `retryAfterSecs`;
 *   `AlreadyVerified` (the email is already verified); or `Failed`.
 */
export async function resendVerificationEmail(username: string, errorHandler: (error: string) => void): Promise<ResendVerificationResult> {
  const url = `/users/resend/verify/email/${encodeURIComponent(username)}`;
  return client
    .get(url)
    .then((res): ResendVerificationResult => ({ status: ResendVerificationStatus.Sent, retryAfterSecs: parseRetryAfter(res?.headers) }))
    .catch((error: unknown): ResendVerificationResult => {
      if (axios.isAxiosError(error)) {
        // 429 = a verification email was sent too recently; a normal cooldown answer, not an error
        if (error.response?.status == 429) {
          return { status: ResendVerificationStatus.Cooldown, retryAfterSecs: parseRetryAfter(error.response.headers) };
        }
        // 409 = the email is already verified; surface the message so the caller can stop offering resend
        if (error.response?.status == 409) {
          parseRequestError(error, errorHandler, 'Resend Verification Email');
          return { status: ResendVerificationStatus.AlreadyVerified };
        }
      }
      parseRequestError(error, errorHandler, 'Resend Verification Email');
      return { status: ResendVerificationStatus.Failed };
    });
}

/**
 * Confirm an emailed account-verification link (`GET /users/verify/{username}/email/{token}`).
 *
 * The backend returns `204` on success and a uniform `401` for every failure (wrong/used/expired token,
 * unknown account) so it never reveals whether an account exists. The `401` is therefore an expected
 * answer, not an error: it maps to {@link EmailVerifyStatus.Expired} and does NOT call `errorHandler`.
 * Only unexpected failures (e.g. the network is down) map to {@link EmailVerifyStatus.Error}, so a
 * transient outage is never misreported as a consumed single-use token.
 *
 * @param username - The username from the verification link.
 * @param token - The single-use verification token from the link.
 * @param errorHandler - Called with a formatted message only on an unexpected (non-401) failure.
 * @returns An {@link EmailVerifyStatus}: `Verified` (204), `Expired` (401), or `Error`.
 */
export async function verifyEmail(username: string, token: string, errorHandler: (error: string) => void): Promise<EmailVerifyStatus> {
  const url = `/users/verify/${encodeURIComponent(username)}/email/${encodeURIComponent(token)}`;
  return client
    .get(url)
    .then((res) => (res?.status == 204 ? EmailVerifyStatus.Verified : EmailVerifyStatus.Error))
    .catch((error: unknown) => {
      // 401 = invalid/expired/used token (uniform anti-enumeration answer); an expected outcome.
      if (axios.isAxiosError(error) && error.response?.status == 401) {
        return EmailVerifyStatus.Expired;
      }
      parseRequestError(error, errorHandler, 'Verify Email');
      return EmailVerifyStatus.Error;
    });
}

/**
 * Fetch a single user's info by username (`GET /users/user/{username}`).
 *
 * Failures are logged to the console rather than surfaced through an error handler.
 *
 * @param username - The username of the user to fetch.
 * @returns The {@link UserInfo}, or `null` if not found or the request failed.
 */
export async function getUser(username: string): Promise<UserInfo | null> {
  const url = '/users/user/' + username;
  return client
    .get<UserInfo>(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, console.log, 'Get User');
      return null;
    });
}

/**
 * List users, either as names or full details (`GET /users/` or `/users/details/`).
 *
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param details - When `true`, return full {@link UserInfo} objects; when `false`, return usernames.
 * @param cursor - Pagination cursor from a previous call, or `null` for the first page.
 * @param limit - Maximum number of users to return per page (defaults to 1000).
 * @returns An array of {@link UserInfo} details (when `details`) or usernames, or `null` if the request failed.
 */
export async function listUsers(
  errorHandler: (error: string) => void,
  details = false,
  cursor: string | null = null,
  limit = 1000,
): Promise<UserInfo[] | string[] | null> {
  let url = '/users/';
  if (details) {
    url += 'details/';
  }
  const params: { limit: number; cursor?: string } = { limit };
  if (cursor) {
    params['cursor'] = cursor;
  }
  return client
    .get<UserInfo[] | string[]>(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Users');
      return null;
    });
}

/**
 * Log the current user out, invalidating their session (`POST /users/logout`).
 *
 * Unlike the other client functions, this returns the raw axios response and does not swallow
 * errors, so callers should handle rejection themselves.
 *
 * @returns The raw axios response for the logout request.
 */
export async function logout(): Promise<AxiosResponse> {
  return client.post('/users/logout');
}

/**
 * Update the currently authenticated user's own profile (`PATCH /users/`).
 *
 * @param data - The profile fields to change (e.g. email, password, settings).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the update succeeded (HTTP 204), otherwise `false`.
 */
export async function updateUser(data: Record<string, unknown>, errorHandler: (error: string) => void): Promise<boolean> {
  return client
    .patch(`/users/`, data, thoriumCookieAuthConfig())
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Update User');
      return false;
    });
}

/**
 * Update another user's profile by username (admin operation) (`PATCH /users/user/{username}`).
 *
 * @param data - The profile fields to change for the target user.
 * @param username - The username of the user to update.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the update succeeded (HTTP 204), otherwise `false`.
 */
export async function updateSingleUser(
  data: Record<string, unknown>,
  username: string,
  errorHandler: (error: string) => void,
): Promise<boolean> {
  const url = '/users/user/' + username;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Update User');
      return false;
    });
}

/**
 * Fetch the currently authenticated user's own info (`GET /users/whoami`).
 *
 * Failures are logged to the console rather than surfaced through an error handler.
 *
 * @returns The current user's {@link UserInfo}, or `null` if not authenticated or the request failed.
 */
export async function whoami(): Promise<UserInfo | null> {
  return client
    .get<UserInfo>('/users/whoami')
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, console.log, 'Who Am I');
      return null;
    });
}

/**
 * Upload (or replace) the current user's profile icon (`POST /users/image`).
 *
 * The icon is sent as multipart form data in an `image` field. The backend streams it to S3
 * and stores only the path on the user, so subsequent `whoami` calls stay lightweight. The part
 * filename is derived from the blob's MIME type (the API keys the stored extension off the part's
 * Content-Type), so animated GIFs and video clips keep the correct extension on the server.
 *
 * @param image - The image/video blob to upload (already resized for static images).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the upload succeeded (HTTP 204), otherwise `false`.
 */
export async function uploadUserImage(image: Blob, errorHandler: (error: string) => void): Promise<boolean> {
  const form = new FormData();
  form.set('image', image, profileGraphicFilename(image.type));
  return client
    .post('/users/image', form)
    .then((res) => res?.status == 204)
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Upload User Image');
      return false;
    });
}

/**
 * Remove the current user's profile icon (`DELETE /users/image`).
 *
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the icon was removed (HTTP 204), otherwise `false`.
 */
export async function deleteUserImage(errorHandler: (error: string) => void): Promise<boolean> {
  return client
    .delete('/users/image')
    .then((res) => res?.status == 204)
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete User Image');
      return false;
    });
}

/**
 * A user's profile icon fetched as a browser-displayable object URL plus its MIME type.
 */
export interface UserImage {
  /** Object URL for the icon; the caller must revoke it with `URL.revokeObjectURL` when done. */
  url: string;
  /** The icon's MIME type (e.g. `image/png`, `video/mp4`), used to pick an `<img>` vs `<video>` renderer. */
  type: string;
}

/**
 * Fetch a user's profile icon as a browser-displayable object URL and its MIME type
 * (`GET /users/user/{username}/image`).
 *
 * The blob is wrapped in an object URL via `URL.createObjectURL`; callers are responsible for
 * revoking it with `URL.revokeObjectURL` when done to avoid leaking memory. The blob's MIME type is
 * returned alongside so callers can render animated GIFs/images with `<img>` and video with `<video>`.
 * A missing icon (`404`) is an expected outcome and is never reported; any other failure is passed to
 * `errorHandler` when one is supplied, so unexpected errors stay reportable. Either way the function
 * returns `null`.
 *
 * @param username - The username whose icon to fetch.
 * @param errorHandler - Optional callback for a formatted message on an unexpected (non-404) failure.
 * @returns The icon's object URL and MIME type, or `null` if the user has no icon or the request failed.
 */
export async function fetchUserImage(username: string, errorHandler?: (error: string) => void): Promise<UserImage | null> {
  return client
    .get<Blob>(`/users/user/${encodeURIComponent(username)}/image`, { responseType: 'blob' })
    .then((res) => (res?.status == 200 && res.data ? { url: URL.createObjectURL(res.data), type: res.data.type } : null))
    .catch((error: unknown) => {
      // a missing icon (404) is expected and stays silent; surface anything else when a handler is given
      if (errorHandler && !(axios.isAxiosError(error) && error.response?.status == 404)) {
        parseRequestError(error, errorHandler, 'Fetch User Image');
      }
      return null;
    });
}

/**
 * Delete a user by username (`DELETE /users/delete/{user}`).
 *
 * @param user - The username of the user to delete.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the user was deleted (HTTP 204), otherwise `false`.
 */
export async function deleteUser(user: string, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/users/delete/' + user;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete User');
      return false;
    });
}

/**
 * List all scoped tokens for the current user (`GET /users/tokens/`).
 *
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns An array of {@link ScopedToken}, or `null` if the request failed.
 */
export async function listScopedTokens(errorHandler: (error: string) => void): Promise<ScopedToken[] | null> {
  return client
    .get<ScopedToken[]>('/users/tokens/', thoriumCookieAuthConfig())
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Scoped Tokens');
      return null;
    });
}

/**
 * Create a new scoped token (`POST /users/tokens/`).
 *
 * @param data - The token name, groups scope, and optional permanent expiration.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The created {@link ScopedToken}, or `null` if the request failed.
 */
export async function createScopedToken(data: ScopedTokenRequest, errorHandler: (error: string) => void): Promise<ScopedToken | null> {
  return client
    .post<ScopedToken>('/users/tokens/', data, thoriumCookieAuthConfig())
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create Scoped Token');
      return null;
    });
}

/**
 * Get a specific scoped token by name (`GET /users/tokens/{name}`).
 *
 * The backend auto-rotates the token value if it has expired, so the returned token
 * may carry a new value.
 *
 * @param name - The token name.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The {@link ScopedToken}, or `null` if not found or the request failed.
 */
export async function getScopedToken(name: string, errorHandler: (error: string) => void): Promise<ScopedToken | null> {
  return client
    .get<ScopedToken>(`/users/tokens/${encodeURIComponent(name)}`, thoriumCookieAuthConfig())
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Scoped Token');
      return null;
    });
}

/**
 * Update a scoped token's groups or expiration (`PATCH /users/tokens/{name}`).
 *
 * @param name - The token name.
 * @param data - Groups to add/remove and expiration changes.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The updated {@link ScopedToken}, or `null` if the request failed.
 */
export async function updateScopedToken(
  name: string,
  data: ScopedTokenUpdate,
  errorHandler: (error: string) => void,
): Promise<ScopedToken | null> {
  return client
    .patch<ScopedToken>(`/users/tokens/${encodeURIComponent(name)}`, data, thoriumCookieAuthConfig())
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Update Scoped Token');
      return null;
    });
}

/**
 * Delete a scoped token by name (`DELETE /users/tokens/{name}`).
 *
 * @param name - The token name.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the token was deleted (HTTP 204), otherwise `false`.
 */
export async function deleteScopedToken(name: string, errorHandler: (error: string) => void): Promise<boolean> {
  return client
    .delete(`/users/tokens/${encodeURIComponent(name)}`, thoriumCookieAuthConfig())
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete Scoped Token');
      return false;
    });
}
