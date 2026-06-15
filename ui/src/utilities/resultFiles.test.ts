import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// project imports
import { fetchResultFileCached } from './resultFiles';
import { clearResultFileCache, getCachedResultFile } from './resultFileCache';
import { getResultsFile } from '@thorpi/results';

vi.mock('@thorpi/results', () => ({
  getResultsFile: vi.fn(),
}));

const mockedGetResultsFile = vi.mocked(getResultsFile);

/** Build a minimal axios-like response carrying the given bytes. */
function bytesResponse(bytes: ArrayBuffer) {
  return { data: bytes } as Awaited<ReturnType<typeof getResultsFile>>;
}

describe('fetchResultFileCached', () => {
  const errorHandler = vi.fn();

  beforeEach(() => {
    clearResultFileCache();
    mockedGetResultsFile.mockReset();
    errorHandler.mockReset();
  });

  afterEach(() => {
    clearResultFileCache();
  });

  it('returns cached bytes without hitting the network on a cache hit', async () => {
    const bytes = new ArrayBuffer(8);
    mockedGetResultsFile.mockResolvedValueOnce(bytesResponse(bytes));
    // first call populates the cache
    await fetchResultFileCached('sha', 'tool', 'rid', 'file.txt', errorHandler);
    expect(mockedGetResultsFile).toHaveBeenCalledTimes(1);
    // second call is served from the cache — no additional fetch
    const again = await fetchResultFileCached('sha', 'tool', 'rid', 'file.txt', errorHandler);
    expect(again).toBe(bytes);
    expect(mockedGetResultsFile).toHaveBeenCalledTimes(1);
  });

  it('fetches and populates the cache on a miss', async () => {
    const bytes = new ArrayBuffer(16);
    mockedGetResultsFile.mockResolvedValueOnce(bytesResponse(bytes));
    const result = await fetchResultFileCached('sha', 'tool', 'rid', 'file.txt', errorHandler);
    expect(result).toBe(bytes);
    expect(getCachedResultFile('rid', 'file.txt')).toBe(bytes);
  });

  it('returns null and caches nothing when the fetch fails', async () => {
    mockedGetResultsFile.mockResolvedValueOnce(null);
    const result = await fetchResultFileCached('sha', 'tool', 'rid', 'missing.txt', errorHandler);
    expect(result).toBeNull();
    expect(getCachedResultFile('rid', 'missing.txt')).toBeUndefined();
  });

  it('shares a single in-flight request for concurrent misses of the same file', async () => {
    const bytes = new ArrayBuffer(32);
    mockedGetResultsFile.mockResolvedValueOnce(bytesResponse(bytes));
    // two overlapping calls before the fetch resolves should collapse into one request
    const [a, b] = await Promise.all([
      fetchResultFileCached('sha', 'tool', 'rid', 'file.txt', errorHandler),
      fetchResultFileCached('sha', 'tool', 'rid', 'file.txt', errorHandler),
    ]);
    expect(a).toBe(bytes);
    expect(b).toBe(bytes);
    expect(mockedGetResultsFile).toHaveBeenCalledTimes(1);
  });
});
