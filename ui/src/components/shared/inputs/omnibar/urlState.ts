/**
 * Serialize omnibar state (clauses + time selection) to/from the URL query string.
 *
 * Well-known fields use readable, legacy-compatible param keys (`query`, `groups`, `indexes`,
 * `tags[KEY]`, `limit`, `hide`, `start`/`end`/`last`) so existing link producers (TagBadge,
 * buildCollectionsBrowsingUrl) and old bookmarks keep working. Any other clause field rides a
 * generic, lossless fallback (`c=`), so the codec works on every browse page regardless of its
 * field vocabulary (creator, name, role, username, …).
 *
 * Clauses and time are exposed as two independent {@link ParamCodec}s so they can be bound to the
 * URL separately (they own disjoint keys and never clobber one another). Built on the generic
 * codec seam (`@utilities/url`); knows nothing about React.
 */

// project imports
import { Clause, ClauseCondition, ClauseIsMulti, CondIsMulti, NewTextClause, parseClauseCondition } from './ClauseTypes';
import { RelativeUnit, TimeSelection } from './timepicker/utils';
import type { ParamCodec } from '@utilities/url/codecs';

export type OmniState = { clauses: Clause[]; time: TimeSelection };

// Param keys owned by the clause codec (excluding the dynamic `tags[KEY]` keys). `nohide` is the
// sentinel marking "hidden tags explicitly cleared" so it can be distinguished from "unset".
const CLAUSE_STATIC_KEYS = ['query', 'groups', 'indexes', 'limit', 'hide', 'nohide', 'c'];
// Param keys owned by the time codec.
const TIME_KEYS = ['last', 'round', 'start', 'end'];

const UNIT_TO_ABBR: Record<RelativeUnit, string> = {
  minute: 'min',
  hour: 'h',
  day: 'd',
  week: 'w',
  month: 'mo',
  year: 'y',
};
const ABBR_TO_UNIT: Record<string, RelativeUnit> = Object.fromEntries(
  Object.entries(UNIT_TO_ABBR).map(([unit, abbr]) => [abbr, unit as RelativeUnit]),
);

// pull every value off a clause regardless of single/multi shape
function clauseValues(clause: Clause): string[] {
  return ClauseIsMulti(clause) ? clause.value.values : [clause.value.value];
}

// build a correctly-typed clause from raw parts, choosing single vs multi by condition
function makeClause(category: string, field: string, condition: ClauseCondition, values: string[]): Clause {
  if (CondIsMulti(condition)) {
    return { category, field, condition, value: { values } };
  }
  return { category, field, condition, value: { value: values[0] ?? '' } };
}

// a list-valued field is `Is` when it has one value, `IsOneOf` when it has several
function multiCondition(values: string[]): ClauseCondition {
  return values.length > 1 ? ClauseCondition.IsOneOf : ClauseCondition.Is;
}

// ---- generic fallback (any field without a readable key) ----------------------------------------

function encodeGenericClause(clause: Clause): string {
  const valuesPart = clauseValues(clause).map(encodeURIComponent).join(',');
  return [clause.category, clause.field, clause.condition, valuesPart].map(encodeURIComponent).join('|');
}

function decodeGenericClause(raw: string): Clause | undefined {
  const parts = raw.split('|').map(decodeURIComponent);
  if (parts.length !== 4) return undefined;
  const [category, field, condStr, valuesStr] = parts;
  const condition = parseClauseCondition(condStr);
  if (!condition) return undefined;
  const values = valuesStr ? valuesStr.split(',').map(decodeURIComponent) : [];
  return makeClause(category, field, condition, values);
}

// ---- clauses <-> params ------------------------------------------------------------------------

/** Encode omnibar clauses into `params` (mutates in place). */
export function clausesToParams(clauses: Clause[], params: URLSearchParams): void {
  clauses.forEach((clause) => {
    const values = clauseValues(clause);
    if (clause.field === 'hidden tags') {
      values.forEach((value) => params.append('hide', value));
    } else if (clause.field === 'text') {
      values.forEach((value) => params.append('query', value));
    } else if (clause.field === 'group') {
      values.forEach((value) => params.append('groups', value));
    } else if (clause.field === 'index') {
      values.forEach((value) => params.append('indexes', value));
    } else if (clause.field === 'limit') {
      params.set('limit', values[0] ?? '');
    } else if (clause.category === 'tag') {
      values.forEach((value) => params.append(`tags[${clause.field}]`, value));
    } else {
      params.append('c', encodeGenericClause(clause));
    }
  });
}

/**
 * Decode omnibar clauses from `params`. When the URL is silent about hidden tags, the default
 * hidden-tags clause(s) from `defaultClauses` are preserved so browse results stay filtered — unless
 * the `nohide` sentinel is present, which marks the hidden-tags filter as explicitly cleared (so the
 * defaults are not re-injected).
 */
export function paramsToClauses(params: URLSearchParams, defaultClauses: Clause[] = []): Clause[] {
  const clauses: Clause[] = [];

  params.getAll('query').forEach((value) => clauses.push(NewTextClause(value)));

  const groups = params.getAll('groups');
  if (groups.length > 0) {
    clauses.push(makeClause('group', 'group', multiCondition(groups), groups));
  }

  const indexes = params.getAll('indexes');
  if (indexes.length > 0) {
    clauses.push(makeClause('index', 'index', multiCondition(indexes), indexes));
  }

  // tags[KEY]=value -> one `Is` clause per value
  for (const key of params.keys()) {
    const match = /^tags\[(.+)\]$/.exec(key);
    if (match) {
      params.getAll(key).forEach((value) => clauses.push(makeClause('tag', match[1], ClauseCondition.Is, [value])));
    }
  }

  const limit = params.get('limit');
  if (limit) {
    clauses.push(makeClause('limit', 'limit', ClauseCondition.Is, [limit]));
  }

  const hide = params.getAll('hide');
  if (hide.length > 0) {
    clauses.push(makeClause('hidden tags', 'hidden tags', ClauseCondition.Are, hide));
  }

  params.getAll('c').forEach((raw) => {
    const clause = decodeGenericClause(raw);
    if (clause) {
      clauses.push(clause);
    }
  });

  // preserve default hidden-tags filtering when the URL doesn't specify it — but not when the user
  // has explicitly cleared it (`nohide` sentinel)
  const hiddenCleared = params.get('nohide') === '1';
  if (!hiddenCleared && !clauses.some((clause) => clause.field === 'hidden tags')) {
    defaultClauses.filter((clause) => clause.field === 'hidden tags').forEach((clause) => clauses.push(clause));
  }

  return clauses;
}

// ---- time <-> params ---------------------------------------------------------------------------

/** Encode a time selection into `params` (mutates in place). `all` encodes to nothing. */
export function timeToParams(time: TimeSelection, params: URLSearchParams): void {
  if (time.mode === 'relative') {
    params.set('last', `${time.amount}${UNIT_TO_ABBR[time.unit]}`);
    if (time.round) {
      params.set('round', '1');
    }
  } else if (time.mode === 'absolute') {
    params.set('start', time.start.toISOString());
    params.set('end', time.end.toISOString());
  }
}

/** Decode a time selection from `params`, falling back to `defaultTime` when no time params exist. */
export function paramsToTime(params: URLSearchParams, defaultTime: TimeSelection = { mode: 'all' }): TimeSelection {
  const last = params.get('last');
  if (last) {
    const match = /^(\d+)(min|h|d|w|mo|y)$/.exec(last);
    if (match) {
      const unit = ABBR_TO_UNIT[match[2]];
      if (unit) {
        return { mode: 'relative', amount: Number.parseInt(match[1], 10), unit, round: params.get('round') === '1' };
      }
    }
  }
  // Accept a single-sided range: collections (and legacy links) may specify only `start` or only
  // `end` because each bound is independently optional in the data model. The omnibar's absolute
  // selection requires both, so fill the missing side — open start -> epoch, open end -> now — to
  // preserve the date filter instead of silently dropping it back to "all".
  const start = params.get('start');
  const end = params.get('end');
  if (start || end) {
    const startDate = start ? new Date(start) : new Date(0);
    const endDate = end ? new Date(end) : new Date();
    if (!Number.isNaN(startDate.getTime()) && !Number.isNaN(endDate.getTime())) {
      return { mode: 'absolute', start: startDate, end: endDate };
    }
  }
  return defaultTime;
}

// ---- combined helpers (for tests and legacy link producers) ------------------------------------

/** Encode both clauses and time into a fresh-or-existing `params`. */
export function clausesAndTimeToParams(
  clauses: Clause[],
  time: TimeSelection,
  params: URLSearchParams = new URLSearchParams(),
): URLSearchParams {
  clausesToParams(clauses, params);
  timeToParams(time, params);
  return params;
}

/** Decode both clauses and time, merging in `defaults` where the URL is silent. */
export function paramsToClausesAndTime(params: URLSearchParams, defaults: OmniState): OmniState {
  return { clauses: paramsToClauses(params, defaults.clauses), time: paramsToTime(params, defaults.time) };
}

// ---- codecs ------------------------------------------------------------------------------------

/** A {@link ParamCodec} binding omnibar clauses to the URL, seeded from `defaultClauses`. */
export function clausesCodec(defaultClauses: Clause[]): ParamCodec<Clause[]> {
  const defaultsHaveHidden = defaultClauses.some((clause) => clause.field === 'hidden tags');
  return {
    keys: (params) => [...CLAUSE_STATIC_KEYS, ...Array.from(params.keys()).filter((key) => key.startsWith('tags['))],
    encode: (clauses, params) => {
      clausesToParams(clauses, params);
      // only relevant for pages whose defaults include hidden tags: if the user cleared them, record
      // the `nohide` sentinel so decode doesn't silently restore the defaults
      if (defaultsHaveHidden && !clauses.some((clause) => clause.field === 'hidden tags')) {
        params.set('nohide', '1');
      }
    },
    decode: (params) => paramsToClauses(params, defaultClauses),
  };
}

/** A {@link ParamCodec} binding the time selection to the URL, seeded from `defaultTime`. */
export function timeCodec(defaultTime: TimeSelection): ParamCodec<TimeSelection> {
  return {
    keys: () => TIME_KEYS,
    encode: (time, params) => timeToParams(time, params),
    decode: (params) => paramsToTime(params, defaultTime),
  };
}
