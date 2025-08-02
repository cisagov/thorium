// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';
import { Filters } from '@models';

/**
 * Get a list of repos by date range.
 * @async
 * @function
 * @param {object} [data] - optional request parameters which includes:
 *   - groups: to which the repos are viewable
 *   - start: start date for search range
 *   - end: end date for search range
 *   - limit:  the max number of submissions to return
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {boolean} [details] - whether to return details for listed submissions
 * @param {string} cursor - the cursor value to continue listing from
 * @returns {Promise<{entityList: any: entityCursor} | null>} - Promise object representing a list of file details.
 */
export async function listRepos(
  data: Filters,
  errorHandler: (error: string) => void,
  details?: boolean | null,
  cursor?: string | null,
): Promise<{ entityList: any[]; entityCursor: string | null }> {
  // build url parameters including optional args if specified
  let url = '/repos';
  if (details) {
    url += '/details/';
  }
  // pass in cursor value
  if (cursor) {
    data['cursor'] = cursor;
  }
  return client
    .get(url, { params: data })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        const cursor = res.data.cursor ? (res.data.cursor as string) : null;
        return { entityList: res.data.data as any[], entityCursor: cursor };
      }
      return { entityList: [], entityCursor: null };
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'List Repos');
      return { entityList: [], entityCursor: null };
    });
}
