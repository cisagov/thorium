import client, { parseRequestError } from './client';

// project imports
import { Repo } from '@models/repos';
import { Filters } from '@models/search';

interface ApiCursor<T> {
  cursor?: string;
  data: T[];
}

/**
 * List code repositories matching the given filters (`GET /repos` or `/repos/details/`).
 *
 * Results are paginated via an opaque cursor: pass the returned `entityCursor` back in on the
 * next call to fetch the following page (`null` cursor means no more pages).
 *
 * @param data - Search/filter parameters (groups, tags, time range, limit, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param details - When `true`, request full {@link Repo} objects (`/details/`) instead of summaries.
 * @param cursor - Pagination cursor from a previous call, or `null`/omitted for the first page.
 * @returns The page of repos and the next-page cursor (`entityCursor` is `null` when exhausted).
 *          On failure, returns an empty list and a `null` cursor.
 */
export async function listRepos(
  data: Filters,
  errorHandler: (error: string) => void,
  details?: boolean | null,
  cursor?: string | null,
): Promise<{ entityList: Repo[]; entityCursor: string | null }> {
  let url = '/repos';
  if (details) {
    url += '/details/';
  }
  if (cursor) {
    data['cursor'] = cursor;
  }
  return client
    .get<ApiCursor<Repo>>(url, { params: data })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        const cursor = res.data.cursor ?? null;
        return { entityList: res.data.data, entityCursor: cursor };
      }
      return { entityList: [], entityCursor: null };
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Repos');
      return { entityList: [], entityCursor: null };
    });
}
