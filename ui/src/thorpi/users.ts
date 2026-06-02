import { AxiosResponse } from 'axios';
import client, { parseRequestError } from './client';

// project imports
import { UserAuthResponse, UserInfo } from '@models/users';

/**
 * Authenticate with a username and password (`POST /users/auth`).
 *
 * Credentials are sent as an HTTP Basic `Authorization` header.
 *
 * @param username - The user's username.
 * @param password - The user's password.
 * @param errorHandler - Called with a formatted message if authentication fails.
 * @returns The {@link UserAuthResponse} (including the session token), or `null` if authentication failed.
 */
export async function authUserPass(
  username: string,
  password: string,
  errorHandler: (error: string) => void,
): Promise<UserAuthResponse | null> {
  const url = '/users/auth';
  const header = { Authorization: 'basic ' + btoa(username + ':' + password) };
  return client
    .post<UserAuthResponse>(url, {}, { headers: header })
    .then((res) => {
      if (res?.status == 200) {
        return res.data;
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
    .post<UserAuthResponse>('/users/auth', {}, { headers: header })
    .then((res) => {
      if (res?.status == 200 && res.data?.token) {
        return res.data.token;
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
 * @returns The created user's {@link UserAuthResponse}, or `null` if the request failed.
 */
export async function createUser(
  name: string,
  email: string,
  password: string,
  role: string,
  errorHandler: (error: string) => void,
): Promise<UserAuthResponse | null> {
  const url = '/users/';
  const data = { username: name, email: email, password: password, role: role };
  return client
    .post<UserAuthResponse>(url, data)
    .then((res) => {
      if (res?.status == 200) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create User');
      return null;
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
    .patch(`/users/`, data)
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
