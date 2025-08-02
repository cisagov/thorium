// import the base client function that loads from the config
// and injects the token via axios intercepts
import { Group } from 'models';
import client, { parseRequestError } from './client';

/**
 * Create a new Thorium group.
 * @async
 * @function
 * @param {any} data - Group name and description object
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - groups details
 */
export async function createGroup(data: any, errorHandler: (error: string) => void): Promise<boolean> {
  return client
    .post('/groups/', data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Create Group');
      return false;
    });
}

/**
 * Delete a group by name.
 * @async
 * @function
 * @param {string} group - Group to get details about
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - groups details
 */
export async function deleteGroup(group: string, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/groups/' + group;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete Group');
      return false;
    });
}

/**
 * Get details for a group.
 * @async
 * @function
 * @param {string} group - Group to get details about
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<any | null>} - groups details
 */
export async function getGroup(group: string, errorHandler: (error: string) => void): Promise<any | null> {
  const url = '/groups/' + group + '/details/';
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
        // server responded with unauthorized
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get Group');
      return null;
    });
}

/**
 * Get a list of groups with optional details.
 * @async
 * @function
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {boolean} details - Whether to return details for listed groups
 * @param {string} cursor - the cursor value to continue listing from
 * @param {number} limit - number of groups to return
 * @returns {Promise<Group[] | string[] | null>} - groups list
 */
export async function listGroups(
  errorHandler: (error: string) => void,
  details = false,
  cursor = null,
  limit = 1000,
): Promise<Group[] | string[] | null> {
  let url = '/groups/';
  if (details) {
    url += 'details/';
  }
  // pass in limit and cursor value
  const params: any = { limit: limit };
  if (cursor) {
    params['cursor'] = cursor;
  }
  return client
    .get(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        if (details && res.data.details) {
          return res.data.details as Group[];
        } else if (!details && res.data.names) {
          return res.data.names as string[];
        } else {
          return [];
        }
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'List Groups');
      return null;
    });
}

/**
 * Update details for a group.
 * @async
 * @function
 * @param {string} group - Name of group to update
 * @param {any} data - Json body to patch group with
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - Request response
 */
export async function updateGroup(group: string, data: any, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/groups/' + group;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update Group');
      return false;
    });
}
