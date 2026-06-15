import { describe, expect, it, vi } from 'vitest';

// project imports
import { AUTO_FETCH_THRESHOLD, chunk, fetchChildrenBatches, isManualChildFetch } from './useChildrenMetadata';
import { Sample } from '@models/files';

// Minimal Sample stub — only the fields exercised by the batch loop matter here.
function sampleFor(sha256: string): Sample {
  return { sha256, sha1: '', md5: '', tags: {}, submissions: [], comments: [] };
}

describe('chunk', () => {
  it('splits a list into fixed-size chunks with a shorter final chunk', () => {
    expect(chunk([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it('returns a single chunk when the list fits in one batch', () => {
    expect(chunk([1, 2, 3], 10)).toEqual([[1, 2, 3]]);
  });

  it('returns no chunks for an empty list', () => {
    expect(chunk([], 4)).toEqual([]);
  });

  it('yields one chunk for a non-positive size rather than looping forever', () => {
    expect(chunk([1, 2], 0)).toEqual([[1, 2]]);
    expect(chunk([], 0)).toEqual([]);
  });
});

describe('isManualChildFetch', () => {
  it('is false below the threshold', () => {
    expect(isManualChildFetch(0)).toBe(false);
    expect(isManualChildFetch(AUTO_FETCH_THRESHOLD - 1)).toBe(false);
  });

  it('is true at or above the threshold', () => {
    expect(isManualChildFetch(AUTO_FETCH_THRESHOLD)).toBe(true);
    expect(isManualChildFetch(AUTO_FETCH_THRESHOLD + 50)).toBe(true);
  });
});

describe('fetchChildrenBatches', () => {
  it('resolves all shas across multiple batches and reports each batch', async () => {
    const shas = ['a', 'b', 'c', 'd', 'e'];
    const fetchOne = vi.fn((sha: string): Promise<Sample | null> => Promise.resolve(sampleFor(sha)));
    const batches: { resolved: Record<string, Sample>; attempted: number }[] = [];
    const merged: Record<string, Sample> = {};

    await fetchChildrenBatches(
      shas,
      fetchOne,
      (resolved, attempted) => {
        batches.push({ resolved, attempted });
        Object.assign(merged, resolved);
      },
      2,
    );

    expect(fetchOne).toHaveBeenCalledTimes(5);
    // 5 items, batch size 2 -> batches of 2, 2, 1
    expect(batches.map((b) => b.attempted)).toEqual([2, 2, 1]);
    expect(Object.keys(merged).sort()).toEqual(shas);
  });

  it('omits null (failed) fetches from resolved but still counts them as attempted', async () => {
    const shas = ['ok1', 'bad', 'ok2'];
    const fetchOne = vi.fn((sha: string): Promise<Sample | null> => Promise.resolve(sha === 'bad' ? null : sampleFor(sha)));
    let totalAttempted = 0;
    const merged: Record<string, Sample> = {};

    await fetchChildrenBatches(
      shas,
      fetchOne,
      (resolved, attempted) => {
        totalAttempted += attempted;
        Object.assign(merged, resolved);
      },
      10,
    );

    // attempted count includes the failure so a progress bar can reach the total
    expect(totalAttempted).toBe(3);
    // but the failed sha is not in the resolved map (its row keeps the sha256 fallback)
    expect(Object.keys(merged).sort()).toEqual(['ok1', 'ok2']);
    expect(merged.bad).toBeUndefined();
  });

  it('does nothing for an empty sha list', async () => {
    const fetchOne = vi.fn((sha: string): Promise<Sample | null> => Promise.resolve(sampleFor(sha)));
    const onBatch = vi.fn();

    await fetchChildrenBatches([], fetchOne, onBatch, 4);

    expect(fetchOne).not.toHaveBeenCalled();
    expect(onBatch).not.toHaveBeenCalled();
  });
});
