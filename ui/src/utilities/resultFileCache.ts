/**
 * A shared, app-wide LRU cache for downloaded tool result-file bytes.
 *
 * Previewing or downloading a result file stores its bytes here so that re-opening a preview,
 * re-downloading, or building the "download all" zip can reuse them instead of re-fetching. The
 * cache is a module-level singleton so it survives `FilesTab` mount/unmount (switching tabs) and is
 * reachable from the zip download path, while an overall byte budget with LRU eviction keeps memory
 * bounded.
 *
 * Entries are keyed by the *result's unique id* (a uuid, distinct for every tool rerun) plus the file
 * name — never by name alone — so a file name reused by a later rerun of the same tool cannot collide
 * with a stale entry.
 */

// ---------------------------------------------------------------------------
// Cache tuning constants — adjust these to change memory behavior.
// ---------------------------------------------------------------------------
/** Maximum total bytes retained across all cached result files (512 MiB). */
export const MAX_CACHE_BYTES = 512 * 1024 * 1024;
/** Individual files larger than this are never cached (25 MiB). */
export const MAX_ITEM_BYTES = 25 * 1024 * 1024;
// ---------------------------------------------------------------------------

/** A least-recently-used cache of result-file bytes with a total byte budget. */
export interface ResultFileCache {
  /** Get cached bytes, marking the entry most-recently-used. Returns `undefined` on a miss. */
  get(resultId: string, name: string): ArrayBuffer | undefined;
  /** Whether the given result file is currently cached (does not affect LRU ordering). */
  has(resultId: string, name: string): boolean;
  /** Cache bytes for a result file. No-op if the item exceeds {@link MAX_ITEM_BYTES}. */
  set(resultId: string, name: string, bytes: ArrayBuffer): void;
  /** Drop everything from the cache. */
  clear(): void;
  /** Current total bytes retained — exposed for tests and diagnostics. */
  bytes(): number;
}

/** Build the unique cache key for a result file from the result's id and the file name. */
function cacheKey(resultId: string, name: string): string {
  return `${resultId}::${name}`;
}

/**
 * Create an LRU result-file cache.
 *
 * A `Map`'s insertion order doubles as the LRU list: the first key is the least-recently-used and a
 * `get`/`set` moves its entry to the end (most-recently-used).
 *
 * @param maxBytes - Total byte budget before eviction kicks in. Defaults to {@link MAX_CACHE_BYTES}.
 * @param maxItemBytes - Largest single item that may be cached. Defaults to {@link MAX_ITEM_BYTES}.
 * @returns A {@link ResultFileCache} instance.
 */
export function createResultFileCache(maxBytes: number = MAX_CACHE_BYTES, maxItemBytes: number = MAX_ITEM_BYTES): ResultFileCache {
  // an item larger than the whole budget could never be cached without first evicting everything;
  // clamp so a single un-cacheable item can't wipe the cache and the post-eviction guard stays unreachable
  maxItemBytes = Math.min(maxItemBytes, maxBytes);
  const store = new Map<string, ArrayBuffer>();
  let totalBytes = 0;

  return {
    get(resultId, name) {
      const key = cacheKey(resultId, name);
      const found = store.get(key);
      if (found === undefined) {
        return undefined;
      }
      // reinsert to mark most-recently-used
      store.delete(key);
      store.set(key, found);
      return found;
    },

    has(resultId, name) {
      return store.has(cacheKey(resultId, name));
    },

    set(resultId, name, bytes) {
      // never cache oversized files — return them to the caller without retaining them
      if (bytes.byteLength > maxItemBytes) {
        return;
      }
      const key = cacheKey(resultId, name);
      // replacing an existing entry: drop the old size from the running total first
      const existing = store.get(key);
      if (existing !== undefined) {
        totalBytes -= existing.byteLength;
        store.delete(key);
      }
      // evict least-recently-used entries until the new item fits within the budget
      while (totalBytes + bytes.byteLength > maxBytes) {
        const oldestKey = store.keys().next().value;
        if (oldestKey === undefined) {
          break; // nothing left to evict
        }
        const oldest = store.get(oldestKey);
        if (oldest !== undefined) {
          totalBytes -= oldest.byteLength;
        }
        store.delete(oldestKey);
      }
      // guard against an item that still doesn't fit (shouldn't happen: maxItemBytes <= maxBytes)
      if (totalBytes + bytes.byteLength > maxBytes) {
        return;
      }
      store.set(key, bytes);
      totalBytes += bytes.byteLength;
    },

    clear() {
      store.clear();
      totalBytes = 0;
    },

    bytes() {
      return totalBytes;
    },
  };
}

/** The app-wide shared result-file cache instance. */
const sharedCache = createResultFileCache();

/** Get cached bytes for a result file, or `undefined` on a miss. See {@link ResultFileCache.get}. */
export function getCachedResultFile(resultId: string, name: string): ArrayBuffer | undefined {
  return sharedCache.get(resultId, name);
}

/** Cache bytes for a result file (no-op if too large). See {@link ResultFileCache.set}. */
export function setCachedResultFile(resultId: string, name: string, bytes: ArrayBuffer): void {
  sharedCache.set(resultId, name, bytes);
}

/** Clear the shared result-file cache. */
export function clearResultFileCache(): void {
  sharedCache.clear();
}
