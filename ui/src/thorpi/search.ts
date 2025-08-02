// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';
import { ElasticIndex, SearchFilters } from '@models';

/**
 * Search for query string
 * @async
 * @function
 * @param {string} query - The search string to query results
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {ElasticIndex[]} indexes - The indexes to search
 * @param {string[] | undefined | null} groups - A list of groups to filter results
 * @param {string | undefined | null} start - The start date range of filtered results
 * @param {string | undefined | null,} end - The end date range of filtered results
 * @param {string | undefined | null} cursor - The UUID cursor for an existing search
 * @param {number} limit - The max number of results to return
 * @returns {Promise<{ entityList: any; entityCursor: string | null }>} - results object
 */
export async function search(
  query: string,
  errorHandler: (error: string) => void,
  indexes?: ElasticIndex[],
  groups?: string[] | undefined | null,
  start?: string | undefined | null,
  end?: string | undefined | null,
  cursor?: string | undefined | null,
  limit = 100,
): Promise<{ entityList: any; entityCursor: string | null }> {
  const url = '/search/';
  // pass in params to search request
  const params: SearchFilters = { query: query };
  if (indexes) {
    params['indexes'] = indexes;
  }
  if (groups) {
    params['groups'] = groups;
  }
  if (start) {
    params['start'] = start;
  }
  if (end) {
    params['end'] = end;
  }
  if (cursor) {
    params['cursor'] = cursor;
  }
  params['limit'] = limit;

  return client
    .get(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        const cursor = res.data.cursor ? (res.data.cursor as string) : null;
        return { entityList: res.data.data as any[], entityCursor: cursor };
      }
      return { entityList: [], entityCursor: null };
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Search Elastic');
      return { entityList: [], entityCursor: null };
    });
}
