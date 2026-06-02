import client, { parseRequestError } from './client';

// project imports
import { ElasticDoc, ElasticIndex, SearchFilters } from '@models/search';

interface ApiCursor<T> {
  cursor?: string;
  data: T[];
}

/**
 * Run a full-text search against Elasticsearch-indexed data (`GET /search/`).
 *
 * Results are paginated via an opaque cursor: pass the returned `entityCursor` back in on the
 * next call to fetch the following page (`null` cursor means no more pages).
 *
 * @param query - The Lucene/full-text query string.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param indexes - Optional list of {@link ElasticIndex}es to restrict the search to.
 * @param groups - Optional list of groups to restrict results to.
 * @param start - Optional ISO start timestamp bounding the result time range.
 * @param end - Optional ISO end timestamp bounding the result time range.
 * @param cursor - Pagination cursor from a previous call, or `null`/omitted for the first page.
 * @param limit - Maximum number of documents to return per page (defaults to 100).
 * @returns The page of {@link ElasticDoc}s and the next-page cursor (`entityCursor` is `null` when exhausted).
 *          On failure, returns an empty list and a `null` cursor.
 */
export async function search(
  query: string,
  errorHandler: (error: string) => void,
  indexes?: ElasticIndex[],
  groups?: string[] | null,
  start?: string | null,
  end?: string | null,
  cursor?: string | null,
  limit = 100,
): Promise<{ entityList: ElasticDoc[]; entityCursor: string | null }> {
  const url = '/search/';
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
    .get<ApiCursor<ElasticDoc>>(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        const cursor = res.data.cursor ?? null;
        return { entityList: res.data.data, entityCursor: cursor };
      }
      return { entityList: [], entityCursor: null };
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Search Elastic');
      return { entityList: [], entityCursor: null };
    });
}
