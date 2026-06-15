// project imports
import { getCachedResultFile, setCachedResultFile } from './resultFileCache';
import { getResultsFile } from '@thorpi/results';

/**
 * In-flight fetches keyed by `${resultId}::${name}`. Two overlapping requests for the same result
 * file (e.g. opening a preview while the "download all" zip build fetches the same file) both miss
 * the cache — which is only populated after the response resolves — so they would otherwise both hit
 * the network. Sharing the pending promise collapses them into a single request.
 */
const inFlight = new Map<string, Promise<ArrayBuffer | null>>();

/**
 * Fetch a result file's bytes, preferring the shared LRU cache and populating it on a miss.
 *
 * Used by both the Files tab (preview/download) and the "download all" zip builder so a file that
 * was already previewed or downloaded is not re-fetched. Cache reads/writes are keyed by the
 * result's unique id, so reruns of the same tool never share entries. Concurrent requests for the
 * same result file share a single in-flight fetch rather than each hitting the network.
 *
 * @param sha256 - The SHA256 of the file the result belongs to.
 * @param tool - The tool that produced the result.
 * @param resultId - The unique id of the specific result/run (used as the cache key + API path).
 * @param name - The name of the result file to fetch.
 * @param errorHandler - Called with a formatted message if the download fails.
 * @returns The file bytes, or `null` if the request failed.
 */
export async function fetchResultFileCached(
  sha256: string,
  tool: string,
  resultId: string,
  name: string,
  errorHandler: (error: string) => void,
): Promise<ArrayBuffer | null> {
  const cached = getCachedResultFile(resultId, name);
  if (cached) {
    return cached;
  }
  // collapse concurrent fetches of the same file onto one shared request
  const key = `${resultId}::${name}`;
  const pending = inFlight.get(key);
  if (pending) {
    return pending;
  }
  const request = getResultsFile(sha256, tool, resultId, name, errorHandler)
    .then((res) => {
      const bytes = res?.data ?? null;
      if (bytes) {
        setCachedResultFile(resultId, name, bytes);
      }
      return bytes;
    })
    .finally(() => {
      // drop the entry once settled so a later fetch (e.g. after eviction) can run again
      inFlight.delete(key);
    });
  inFlight.set(key, request);
  return request;
}
