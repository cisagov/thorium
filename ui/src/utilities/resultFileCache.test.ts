import { describe, it, expect } from 'vitest';

// project imports
import { createResultFileCache, MAX_CACHE_BYTES, MAX_ITEM_BYTES } from './resultFileCache';

/** Build an ArrayBuffer of the given byte length. */
function buf(size: number): ArrayBuffer {
  return new ArrayBuffer(size);
}

describe('createResultFileCache', () => {
  it('stores and retrieves bytes and tracks total size', () => {
    const cache = createResultFileCache(1000, 500);
    const bytes = buf(40);
    cache.set('r1', 'a.txt', bytes);

    expect(cache.get('r1', 'a.txt')).toBe(bytes);
    expect(cache.has('r1', 'a.txt')).toBe(true);
    expect(cache.bytes()).toBe(40);
    expect(cache.get('r1', 'missing')).toBeUndefined();
  });

  it('keys entries by result id + name so tool reruns do not collide', () => {
    const cache = createResultFileCache(1000, 500);
    const first = buf(10);
    const second = buf(20);
    // same file name, different result ids (e.g. a rerun of the same tool)
    cache.set('run-1', 'out.bin', first);
    cache.set('run-2', 'out.bin', second);

    expect(cache.get('run-1', 'out.bin')).toBe(first);
    expect(cache.get('run-2', 'out.bin')).toBe(second);
    expect(cache.bytes()).toBe(30);
  });

  it('never caches an item larger than the per-item cap', () => {
    const cache = createResultFileCache(1000, 40);
    cache.set('r1', 'big.bin', buf(41));

    expect(cache.has('r1', 'big.bin')).toBe(false);
    expect(cache.bytes()).toBe(0);
  });

  it('replacing an entry updates the total size rather than accumulating', () => {
    const cache = createResultFileCache(1000, 500);
    cache.set('r1', 'a', buf(40));
    cache.set('r1', 'a', buf(30));

    expect(cache.bytes()).toBe(30);
    expect(cache.get('r1', 'a')?.byteLength).toBe(30);
  });

  it('evicts least-recently-used entries when the byte budget is exceeded', () => {
    const cache = createResultFileCache(100, 500);
    cache.set('r', 'a', buf(40));
    cache.set('r', 'b', buf(40)); // total 80

    // touch 'a' so 'b' becomes least-recently-used
    expect(cache.get('r', 'a')).toBeDefined();

    cache.set('r', 'c', buf(40)); // 80 + 40 = 120 > 100 -> evict LRU ('b')

    expect(cache.has('r', 'b')).toBe(false);
    expect(cache.has('r', 'a')).toBe(true);
    expect(cache.has('r', 'c')).toBe(true);
    expect(cache.bytes()).toBe(80);
  });

  it('clear empties the cache and resets the total', () => {
    const cache = createResultFileCache(1000, 500);
    cache.set('r1', 'a', buf(40));
    cache.clear();

    expect(cache.has('r1', 'a')).toBe(false);
    expect(cache.bytes()).toBe(0);
  });

  it('exposes the documented default tuning constants', () => {
    expect(MAX_CACHE_BYTES).toBe(512 * 1024 * 1024);
    expect(MAX_ITEM_BYTES).toBe(25 * 1024 * 1024);
    expect(MAX_ITEM_BYTES).toBeLessThan(MAX_CACHE_BYTES);
  });
});
