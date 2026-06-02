import { parseISO } from 'date-fns';

/**
 * Parse an ISO date string into a `Date` without throwing on malformed input.
 *
 * @param date - The ISO date string to parse (may be `null`).
 * @returns The parsed `Date`, or `null` if the value is not a valid date.
 * @throws Re-throws any error that is not a `RangeError` (i.e. unexpected failures).
 */
export function safeStringToDateConversion(date: string | null) {
  try {
    return parseISO(date as unknown as string);
  } catch (e) {
    if (e instanceof RangeError) {
      // this hits if the passed in value is not a valid date
      return null;
    } else {
      throw e;
    }
  }
}

/**
 * Convert a `Date` to an ISO string without throwing on an invalid date.
 *
 * @param date - The date to serialize (may be `null`).
 * @returns The ISO string, or `null` if `date` is `null` or an invalid date.
 * @throws Re-throws any error that is not a `RangeError` (i.e. unexpected failures).
 */
export function safeDateToStringConversion(date: Date | null) {
  if (date == null) {
    return null;
  }
  try {
    return date.toISOString();
  } catch (e) {
    if (e instanceof RangeError) {
      // this hits if the passed in value is not a valid date
      return null;
    } else {
      throw e;
    }
  }
}

/**
 * Parse a JSON string without throwing on malformed input.
 *
 * Useful for values from untrusted sources such as session storage or input fields.
 *
 * @param unsafeJSON - The JSON string to parse.
 * @returns The parsed value, or `null` if the string is not valid JSON.
 * @throws Re-throws any error that is not a `SyntaxError` (i.e. unexpected failures).
 */
export function safeParseJSON(unsafeJSON: string): unknown {
  try {
    return JSON.parse(unsafeJSON) as unknown;
  } catch (e) {
    if (e instanceof SyntaxError) {
      // this hits if the passed in value is not valid JSON
      return null;
    } else {
      throw e;
    }
  }
}
