import { describe, expect, it } from 'vitest';

// project imports
import { entriesToDict, valueToEntries } from './EnvironmentVariables';

describe('valueToEntries', () => {
  it('seeds a single empty row for an empty dictionary', () => {
    expect(valueToEntries({})).toEqual([{ key: '', value: '' }]);
  });

  it('maps each dictionary entry to a key/value row', () => {
    expect(valueToEntries({ FOO: 'bar', BAZ: 'qux' })).toEqual([
      { key: 'FOO', value: 'bar' },
      { key: 'BAZ', value: 'qux' },
    ]);
  });

  it('renders null values as empty strings', () => {
    expect(valueToEntries({ FOO: null })).toEqual([{ key: 'FOO', value: '' }]);
  });
});

describe('entriesToDict', () => {
  it('drops rows with an empty key (in-progress trailing rows)', () => {
    expect(
      entriesToDict([
        { key: 'FOO', value: 'bar' },
        { key: '', value: '' },
      ]),
    ).toEqual({ FOO: 'bar' });
  });

  it('stores empty values as null', () => {
    expect(entriesToDict([{ key: 'FOO', value: '' }])).toEqual({ FOO: null });
  });

  it('returns an empty dictionary when every row has an empty key', () => {
    expect(entriesToDict([{ key: '', value: '' }])).toEqual({});
  });
});

describe('env row round-trip', () => {
  it('preserves keyed entries through dict -> entries -> dict', () => {
    const dict = { FOO: 'bar', BAZ: null };
    expect(entriesToDict(valueToEntries(dict))).toEqual(dict);
  });
});
