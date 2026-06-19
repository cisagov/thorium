import { describe, it, expect } from 'vitest';

// project imports
import { Clause, ClauseCondition, DefaultClausesEntities } from './ClauseTypes';
import { AbsoluteSelection, RelativeSelection, TimeSelection } from './timepicker/utils';
import { clausesAndTimeToParams, clausesCodec, paramsToClauses, paramsToClausesAndTime, paramsToTime } from './urlState';

const single = (category: string, field: string, value: string, condition = ClauseCondition.Is): Clause =>
  ({ category, field, condition, value: { value } }) as Clause;
const multi = (category: string, field: string, values: string[], condition = ClauseCondition.IsOneOf): Clause =>
  ({ category, field, condition, value: { values } }) as Clause;

const NO_DEFAULTS = { clauses: [] as Clause[], time: { mode: 'all' } as TimeSelection };

// encode -> decode and compare the resulting clause set, order-insensitively
function roundTripClauses(clauses: Clause[], defaultClauses: Clause[] = []): Clause[] {
  const params = clausesAndTimeToParams(clauses, { mode: 'all' });
  return paramsToClauses(params, defaultClauses);
}

describe('clause url round-trips', () => {
  it('round-trips text clauses as query params', () => {
    const out = roundTripClauses([single('text', 'text', 'evil.exe')]);
    expect(out).toEqual([single('text', 'text', 'evil.exe')]);
  });

  it('round-trips a single group as `Is` and multiple groups as `IsOneOf`', () => {
    expect(roundTripClauses([single('group', 'group', 'corp')])).toEqual([single('group', 'group', 'corp')]);
    expect(roundTripClauses([multi('group', 'group', ['corp', 'team-a'])])).toEqual([multi('group', 'group', ['corp', 'team-a'])]);
  });

  it('round-trips indexes', () => {
    expect(roundTripClauses([multi('index', 'index', ['SampleResults', 'SampleTags'])])).toEqual([
      multi('index', 'index', ['SampleResults', 'SampleTags']),
    ]);
  });

  it('round-trips tag clauses as tags[KEY]=value', () => {
    const params = clausesAndTimeToParams([single('tag', 'family', 'emotet')], { mode: 'all' });
    expect(params.getAll('tags[family]')).toEqual(['emotet']);
    expect(paramsToClauses(params)).toEqual([single('tag', 'family', 'emotet')]);
  });

  it('round-trips limit', () => {
    expect(roundTripClauses([single('limit', 'limit', '50')])).toEqual([single('limit', 'limit', '50')]);
  });

  it('round-trips hidden tags as a single `Are` clause', () => {
    const out = roundTripClauses([multi('hidden tags', 'hidden tags', ['Results', 'Parent'], ClauseCondition.Are)]);
    expect(out).toEqual([multi('hidden tags', 'hidden tags', ['Results', 'Parent'], ClauseCondition.Are)]);
  });

  it('round-trips arbitrary fields through the generic fallback', () => {
    const clauses = [single('Users', 'username', 'alice', ClauseCondition.Includes), multi('role', 'role', ['Admin', 'Analyst'])];
    const params = clausesAndTimeToParams(clauses, { mode: 'all' });
    expect(params.getAll('c').length).toBe(2);
    expect(paramsToClauses(params)).toEqual(clauses);
  });

  it('preserves values containing separators through the generic fallback', () => {
    const clauses = [single('Owners', 'owner', 'a|b,c')];
    expect(roundTripClauses(clauses)).toEqual(clauses);
  });
});

describe('default merging', () => {
  it('injects default hidden tags when the URL has no hide param', () => {
    const defaults = DefaultClausesEntities();
    const out = paramsToClauses(new URLSearchParams('groups=corp'), defaults);
    expect(out).toContainEqual(single('group', 'group', 'corp'));
    expect(out.some((c) => c.field === 'hidden tags')).toBe(true);
  });

  it('does not override an explicit hide param with defaults', () => {
    const out = paramsToClauses(new URLSearchParams('hide=OnlyThis'), DefaultClausesEntities());
    const hidden = out.filter((c) => c.field === 'hidden tags');
    expect(hidden).toHaveLength(1);
    expect(hidden[0].value).toEqual({ values: ['OnlyThis'] });
  });

  it('an empty URL with no defaults yields no clauses', () => {
    expect(paramsToClauses(new URLSearchParams(''), [])).toEqual([]);
  });
});

describe('cleared hidden tags (nohide sentinel)', () => {
  it('emits nohide=1 when defaults have hidden tags but the cleared state has none', () => {
    const codec = clausesCodec(DefaultClausesEntities());
    const params = new URLSearchParams();
    codec.encode([], params); // user cleared all clauses including the default hidden tags
    expect(params.get('nohide')).toBe('1');
  });

  it('does not re-inject defaults when nohide is present', () => {
    const out = paramsToClauses(new URLSearchParams('nohide=1'), DefaultClausesEntities());
    expect(out.some((c) => c.field === 'hidden tags')).toBe(false);
  });

  it('round-trips a cleared state back to no hidden tags', () => {
    const codec = clausesCodec(DefaultClausesEntities());
    const params = new URLSearchParams();
    codec.encode([], params);
    expect((codec.decode(params) ?? []).some((c) => c.field === 'hidden tags')).toBe(false);
  });

  it('still injects defaults for a fresh (empty) URL', () => {
    const codec = clausesCodec(DefaultClausesEntities());
    expect((codec.decode(new URLSearchParams()) ?? []).some((c) => c.field === 'hidden tags')).toBe(true);
  });

  it('never emits nohide for pages whose defaults have no hidden tags', () => {
    const codec = clausesCodec([]);
    const params = new URLSearchParams();
    codec.encode([], params);
    expect(params.get('nohide')).toBeNull();
  });
});

describe('time url round-trips', () => {
  it('encodes `all` to nothing and decodes back to `all`', () => {
    const params = clausesAndTimeToParams([], { mode: 'all' });
    expect(params.toString()).toBe('');
    expect(paramsToTime(params)).toEqual({ mode: 'all' });
  });

  it('round-trips a relative selection (with rounding)', () => {
    const sel: RelativeSelection = { mode: 'relative', amount: 7, unit: 'day', round: true };
    const params = clausesAndTimeToParams([], sel);
    expect(params.get('last')).toBe('7d');
    expect(params.get('round')).toBe('1');
    expect(paramsToTime(params)).toEqual(sel);
  });

  it('round-trips a relative selection without rounding', () => {
    const sel: RelativeSelection = { mode: 'relative', amount: 3, unit: 'month', round: false };
    const params = clausesAndTimeToParams([], sel);
    expect(params.get('last')).toBe('3mo');
    expect(paramsToTime(params)).toEqual(sel);
  });

  it('round-trips an absolute selection', () => {
    const start = new Date('2026-01-01T00:00:00.000Z');
    const end = new Date('2026-02-01T00:00:00.000Z');
    const sel: AbsoluteSelection = { mode: 'absolute', start, end };
    const params = clausesAndTimeToParams([], sel);
    const decoded = paramsToTime(params);
    expect(decoded).toEqual(sel);
  });

  // open-ended collections set only one bound; the missing side is filled rather than dropped
  it('decodes a `start`-only range, filling `end` with now', () => {
    const start = new Date('2026-01-01T00:00:00.000Z');
    const decoded = paramsToTime(new URLSearchParams({ start: start.toISOString() }));
    expect(decoded.mode).toBe('absolute');
    if (decoded.mode === 'absolute') {
      expect(decoded.start).toEqual(start);
      expect(decoded.end.getTime()).toBeGreaterThan(start.getTime());
    }
  });

  it('decodes an `end`-only range, filling `start` with the epoch', () => {
    const end = new Date('2026-02-01T00:00:00.000Z');
    const decoded = paramsToTime(new URLSearchParams({ end: end.toISOString() }));
    expect(decoded).toEqual({ mode: 'absolute', start: new Date(0), end });
  });

  it('falls back to the default when an only-bound is unparseable', () => {
    expect(paramsToTime(new URLSearchParams({ start: 'not-a-date' }))).toEqual({ mode: 'all' });
  });
});

describe('legacy link compatibility', () => {
  it('decodes a TagBadge link (?tags[KEY]=v&limit=10)', () => {
    const out = paramsToClauses(new URLSearchParams('tags[av]=malicious&limit=10'));
    expect(out).toContainEqual(single('tag', 'av', 'malicious'));
    expect(out).toContainEqual(single('limit', 'limit', '10'));
  });

  it('decodes a buildCollectionsBrowsingUrl link (groups + tags + start/end)', () => {
    const params = new URLSearchParams();
    params.append('groups', 'corp');
    params.append('tags[family]', 'emotet');
    params.set('start', '2026-01-01T00:00:00.000Z');
    params.set('end', '2026-02-01T00:00:00.000Z');
    const { clauses, time } = paramsToClausesAndTime(params, NO_DEFAULTS);
    expect(clauses).toContainEqual(single('group', 'group', 'corp'));
    expect(clauses).toContainEqual(single('tag', 'family', 'emotet'));
    expect(time.mode).toBe('absolute');
  });
});
