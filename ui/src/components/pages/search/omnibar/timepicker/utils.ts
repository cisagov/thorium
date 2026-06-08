import { safeDateToStringConversion } from '@utilities/inputs';

export type RelativeUnit = 'minute' | 'hour' | 'day' | 'week' | 'month' | 'year';

export type RelativeSelection = { mode: 'relative'; amount: number; unit: RelativeUnit; round?: boolean };

export type AbsoluteSelection = { mode: 'absolute'; start: Date; end: Date };
export type AllTime = { mode: 'all' };
export type TimeSelection = RelativeSelection | AbsoluteSelection | AllTime;
export type PresetOptions = Record<string, RelativeSelection>;
export type DateRange = { start: Date | null; end: Date | null };

export function defaultTimeSelection(): TimeSelection {
  return { mode: 'all' };
}

function startOfUTCDay(d: Date): Date {
  return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 0, 0, 0, 0));
}

export function TimeSelectionToStrings(ts: TimeSelection): [string | null, string | null] {
  const [start, end] = TimeSelectionToDateRange(ts);
  return [safeDateToStringConversion(start), safeDateToStringConversion(end)];
}

export function TimeSelectionToDateRange(ts: TimeSelection): [Date | null, Date | null] {
  if (ts.mode == 'all') return [null, null];

  if (ts.mode === 'absolute') {
    return [ts.start, ts.end];
  }
  const end = new Date();
  let start = startOfLast(ts.amount, ts.unit, end);
  if (ts.round) {
    start = startOfUTCDay(start);
  }
  return [start, end];
}

const MS_BY_FIXED_UNIT = {
  minute: 60_000,
  hour: 3_600_000,
  day: 86_400_000,
  week: 604_800_000,
} as const;

function daysInMonthUTC(year: number, month0: number) {
  // month0: 0-11
  // Day 0 of next month is last day of current month
  return new Date(Date.UTC(year, month0 + 1, 0)).getUTCDate();
}

/**
 * Computes the start date for "last X unit(s)" ending at `end` (default: now).
 * - minute/hour/day/week: fixed duration subtraction
 * - month/year: calendar subtraction with day-of-month clamping
 *
 * Uses UTC fields to avoid DST/local-time surprises.
 */
export function startOfLast(amount: number, unit: RelativeUnit, end: Date = new Date()): Date {
  if (!Number.isFinite(amount) || !Number.isInteger(amount) || amount < 0) {
    throw new Error(`amount must be a non-negative integer number`);
  }
  if (!(end instanceof Date) || Number.isNaN(end.getTime())) {
    throw new Error(`end must be a valid Date`);
  }

  // Fixed-duration units. Anything shorter than a month.
  if (unit in MS_BY_FIXED_UNIT) {
    const ms = MS_BY_FIXED_UNIT[unit] * amount;
    return new Date(end.getTime() - ms);
  }

  // Calendar units (month/year)
  const y = end.getUTCFullYear();
  const m = end.getUTCMonth(); // 0-11
  const d = end.getUTCDate(); // 1-31

  // Keep time-of-day the same (UTC) by carrying over h/m/s/ms
  const hh = end.getUTCHours();
  const mm = end.getUTCMinutes();
  const ss = end.getUTCSeconds();
  const ms = end.getUTCMilliseconds();

  let targetYear = y;
  let targetMonth = m;

  if (unit === 'month') {
    //convert current year/month to single month index,
    //subtract requested number of months, then convert back.
    //helps with year boundaries
    const totalMonths = y * 12 + m - amount;
    targetYear = Math.floor(totalMonths / 12);
    //recover month in 0-11 range. extra modulo is to prevent negative issues
    targetMonth = ((totalMonths % 12) + 12) % 12;
  } else {
    // unit === 'year'
    targetYear = y - amount;
    targetMonth = m;
  }

  const dim = daysInMonthUTC(targetYear, targetMonth);
  const clampedDay = Math.min(d, dim);

  return new Date(Date.UTC(targetYear, targetMonth, clampedDay, hh, mm, ss, ms));
}
