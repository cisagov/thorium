// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';

// Debugging errors randomly inserted.
// Valid values 0-100 (percentage chance of error)
const RANDOM_DEBUG_ERRORS = 0;

/**
 * Submit reactions for a file.
 * @async
 * @function
 * @param {any} reaction - Reactions to submit
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {Map} tags - An array of tags to add to each reaction in list
 * @returns {object} - promise object representing reaction submission response
 */
export async function createReaction(reaction: any, errorHandler: (error: string) => void, tags = null): Promise<any | null> {
  const url = '/reactions/';
  if (tags != null) {
    reaction['tags'] = tags;
  }
  if (RANDOM_DEBUG_ERRORS) {
    if (Math.floor(Math.random() * 100) < RANDOM_DEBUG_ERRORS) {
      errorHandler(`Failed to create reaction: Permission Denied`);
      return null;
    }
  }
  return client
    .post(url, reaction)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Create Reaction');
      return null;
    });
}

/**
 * Get details for a reaction by UUID.
 * @async
 * @function
 * @param {string} group - Group to get details about
 * @param {string} uuid - ID of reaction to get
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {object} - groups details
 */
export async function getReaction(group: string, uuid: string, errorHandler: (error: string) => void): Promise<any | null> {
  const url = '/reactions/' + group + '/' + uuid;
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get Reaction');
      return null;
    });
}

/**
 * Get logs for a reaction
 * @async
 * @function
 * @param {string} group - Group to get details about
 * @param {string} uuid - ID of reaction to get
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {number} cursor - The number of log lines to skip
 * @param {number} limit - The number of log lines to retrieve
 * @returns {Promise<any | null>} - reaction logs
 */
export async function getReactionLogs(
  group: string,
  uuid: string,
  errorHandler: (error: string) => void,
  cursor = null,
  limit = 100,
): Promise<any | null> {
  const url = '/reactions/logs/' + group + '/' + uuid;
  // pass in limit and cursor value
  const params: { cursor?: string; limit?: number } = {};
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
      parseRequestError(error, errorHandler, 'Get System Stats');
      return null;
    });
}

/**
 * Get logs for a reaction stage
 * @async
 * @function
 * @param {string} group - Group to get details about
 * @param {string} uuid - ID of reaction to get
 * @param {string} stage - Name of stage (image) within reaction
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {number} cursor - The number of log lines to skip
 * @param {number} limit - The number of log lines to retrieve
 * @returns {Promise<string[] | null>} - reaction logs
 */
export const getReactionStageLogs = async (
  group: string,
  uuid: string,
  stage: string,
  errorHandler: (error: string) => void,
  cursor = null,
  limit = 100,
): Promise<string[] | null> => {
  const url = '/reactions/logs/' + group + '/' + uuid + '/' + stage;
  // pass in limit and cursor value
  const params: { cursor?: string; limit?: number } = {};
  if (cursor) {
    params['cursor'] = cursor;
  }
  params['limit'] = limit;
  return client
    .get(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data?.logs) {
        return res.data.logs;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get Reaction Stage Logs');
      return null;
    });
};

/**
 * Get a list of reactions with optional details.
 * @async
 * @function
 * @param {string} group - group that pipeline is a member of
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {string} [pipeline] - name of pipeline to get reactions for
 * @param {string} [tag] - tag to get reactions for
 * @param {boolean} [details] - whether to return details for listed pipelines
 * @param {string} cursor - the cursor value to continue listing from
 * @param {number} limit - number of reactions to return
 * @returns {Promise<any | null>} - reactions list
 */
export async function listReactions(
  group: string,
  errorHandler: (error: string) => void,
  pipeline = '',
  tag = '',
  details = false,
  cursor = null,
  limit = 1000,
): Promise<any | null> {
  let url = '/reactions/';
  if (tag == '') {
    url += 'list/' + group + '/' + pipeline + '/';
  } else {
    url += 'tag/' + group + '/' + tag + '/';
  }
  if (details) {
    url += 'details/';
  }
  // pass in limit and cursor value
  const params: { cursor?: string; limit?: number } = {};
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
      parseRequestError(error, errorHandler, 'List Reactions');
      return null;
    });
}

/**
 * Delete reaction by UUID.
 * @async
 * @function
 * @param {string} group - Group to get details about
 * @param {string} uuid - ID of reaction to get
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - promise object representing reaction post response.
 */
export async function deleteReaction(group: string, uuid: string, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/reactions/' + group + '/' + uuid;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete Reaction');
      return false;
    });
}
