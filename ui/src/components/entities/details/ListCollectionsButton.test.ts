import { describe, it, expect } from 'vitest';

// project imports
import { buildCollectionsBrowsingUrl } from './ListCollectionsButton';
import { BlankCollection, Collection, CollectionKind, CollectionMetaFields } from '@models/entities/collections';

/**
 * Build a collection with the given group list and collection-metadata overrides on top of the blank
 * defaults, so each test only states the fields it cares about.
 */
function collectionWith(groups: string[], meta: Partial<CollectionMetaFields>): Collection {
  const base = structuredClone(BlankCollection);
  base.groups = groups;
  base.metadata.Collection = { ...base.metadata.Collection, ...meta };
  return base;
}

/** Parse the query string of a built browsing URL into a URLSearchParams for assertions. */
function paramsOf(url: string): URLSearchParams {
  return new URLSearchParams(url.slice(url.indexOf('?') + 1));
}

describe('buildCollectionsBrowsingUrl', () => {
  it('routes to the lowercased collection kind path', () => {
    const url = buildCollectionsBrowsingUrl(collectionWith([], { collection_kind: CollectionKind.Files }));
    expect(url.startsWith('/files?')).toBe(true);
  });

  it('includes groups when ignore_groups is false', () => {
    const url = buildCollectionsBrowsingUrl(collectionWith(['a', 'b'], { ignore_groups: false }));
    expect(paramsOf(url).getAll('groups')).toEqual(['a', 'b']);
  });

  it('omits groups when ignore_groups is true', () => {
    const url = buildCollectionsBrowsingUrl(collectionWith(['a', 'b'], { ignore_groups: true }));
    expect(paramsOf(url).getAll('groups')).toEqual([]);
  });

  it('bracket-encodes each tag key and repeats per value', () => {
    const url = buildCollectionsBrowsingUrl(collectionWith([], { collection_tags: { family: ['emotet', 'trickbot'], os: ['windows'] } }));
    const params = paramsOf(url);
    expect(params.getAll('tags[family]')).toEqual(['emotet', 'trickbot']);
    expect(params.getAll('tags[os]')).toEqual(['windows']);
  });

  it('includes start and end only when present', () => {
    const withRange = paramsOf(buildCollectionsBrowsingUrl(collectionWith([], { start: '2020-01-01', end: '2020-02-01' })));
    expect(withRange.get('start')).toBe('2020-01-01');
    expect(withRange.get('end')).toBe('2020-02-01');
    const withoutRange = paramsOf(buildCollectionsBrowsingUrl(collectionWith([], { start: null, end: null })));
    expect(withoutRange.has('start')).toBe(false);
    expect(withoutRange.has('end')).toBe(false);
  });

  it('always sets tags_case_insensitive as a stringified boolean', () => {
    const insensitive = paramsOf(buildCollectionsBrowsingUrl(collectionWith([], { tags_case_insensitive: true })));
    expect(insensitive.get('tags_case_insensitive')).toBe('true');
    const sensitive = paramsOf(buildCollectionsBrowsingUrl(collectionWith([], { tags_case_insensitive: false })));
    expect(sensitive.get('tags_case_insensitive')).toBe('false');
  });
});
