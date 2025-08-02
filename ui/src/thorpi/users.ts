// import the base client function that loads from the config
// and injects the token via axios intercepts
import { AxiosResponse } from 'axios';
import client, { parseRequestError } from './client';
import { UserAuthResponse, UserInfo } from '@models';

/**
 * Auth user by password to get a token
 * @async
 * @function
 * @param {string} username - name of the user to auth
 * @param {string} password - the users password
 * @param {(error: string) => void)} errorHandler - error handler function
 * @returns {Promise<UserAuthResponse | null>} - the users token and token expiration date
 */
export async function authUserPass(
  username: string,
  password: string,
  errorHandler: (error: string) => void,
): Promise<UserAuthResponse | null> {
  const url = '/users/auth';
  // build basic auth header and assign Authorization filed to value
  const header = { Authorization: 'basic ' + btoa(username + ':' + password) };
  return client
    .post(url, {}, { headers: header })
    .then((res) => {
      if (res?.status == 200) {
        return res.data as UserAuthResponse;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Password Auth');
      return null;
    });
}

/**
 * Auth user by token.
 * @async
 * @function
 * @param {string} token - the users Thorium token
 * @returns {Promise<string | null>} - the users token and token expiration date
 */
export async function authUserToken(token: string): Promise<string | null> {
  // build basic auth header and assign Authorization filed to value
  const header = { Authorization: 'token ' + btoa(token) };
  return client
    .post('/users/auth', {}, { headers: header })
    .then((res) => {
      if (res?.status == 200 && res.data?.token) {
        return res.data.token;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, console.log, 'Token Auth');
      return null;
    });
}

/**
 * Create a new user
 * @async
 * @function
 * @param {string} name - name of user to create
 * @param {string} email - email for user
 * @param {string} password - password for new user
 * @param {string} role - Thorium role for user
 * @param {object} errorHandler - error handler function
 * @returns {Promise<UserAuthResponse | null>} - request response
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
    .post(url, data)
    .then((res) => {
      if (res?.status == 200) {
        return res.data as UserAuthResponse;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Token Auth');
      return null;
    });
}

/**
 * Get a User's info by username
 * @async
 * @function
 * @param {string} username - name of user to get
 * @returns {Promise<any | null>} - the user details
 */
export async function getUser(username: string): Promise<any | null> {
  const url = '/users/user/' + username;
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, console.log, 'Get User');
      return null;
    });
}

/**
 * Get a list of users and their details
 * @async
 * @function
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {boolean} details - whether to get user details or just a list of user names
 * @param {string} cursor - the cursor value to continue listing from
 * @param {number} limit - the number of users to return
 * @returns {Promise<any | null>} - list of users
 */
export async function listUsers(errorHandler: (error: string) => void, details = false, cursor = null, limit = 1000): Promise<any | null> {
  let url = '/users/';
  if (details) {
    url += 'details/';
  }
  const params: any = {};
  if (cursor) {
    params['cursor'] = cursor;
  }
  params['limit'] = limit;
  return client
    .get(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'List Users');
      return null;
    });
}

/**
 * Log user out to destroy token.
 * @async
 * @function
 * @returns {Promise<AxiosResponse<any, any>>} - request response
 */
export async function logout(): Promise<AxiosResponse<any, any>> {
  return client.post('/users/logout');
}

/**
 * Update your own user info
 * @async
 * @function
 * @param {object} data - json to patch user with
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - request response
 */
export async function updateUser(data: any, errorHandler: (error: string) => void): Promise<boolean> {
  return client
    .patch(`/users/`, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update User');
      return false;
    });
}

/**
 * Update a single user's information with username.
 * @async
 * @function
 * @param {object} data - json to patch user with
 * @param {string} username - username of user to patch
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - request response
 */
export async function updateSingleUser(data: any, username: string, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/users/user/' + username;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update User');
      return false;
    });
}

/**
 * Get a users name and info from the token.
 * @async
 * @function
 * @returns {Promise<any | null>} - user's account info.
 */
export async function whoami(): Promise<UserInfo | null> {
  return client
    .get('/users/whoami')
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data as UserInfo;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, console.log, 'Who Am I');
      return null;
    });
}

/**
 * Delete a user by name.
 * @async
 * @function
 * @param {string} user - name of user to delete
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - request response
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
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete User');
      return false;
    });
}
