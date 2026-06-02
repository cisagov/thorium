import { getData } from 'country-list';

/**
 * Shared country option/lookup helpers for vendor entities.
 *
 * The `country-list` library appends " (the)" to some country names (e.g.
 * "United States of America (the)", "Russian Federation (the)"), but the API's
 * country source (the Rust `isocountry` crate) returns names without it (e.g.
 * "United States of America"). We strip the suffix so the options we show and
 * the values returned by the API line up.
 *
 * Because `country-list`'s own `getCode`/`getName` only match the original
 * "(the)" names, we build our own name<->code lookup off the stripped names so
 * codes can still be resolved from what the user actually selects.
 */
const THE_SUFFIX = ' (the)';

const stripThe = (name: string): string => (name.endsWith(THE_SUFFIX) ? name.slice(0, -THE_SUFFIX.length) : name);

const countryData = getData()
  .map(({ code, name }) => ({ code, name: stripThe(name) }))
  .sort((a, b) => a.name.localeCompare(b.name));

/** Sorted list of country display names (matching the API's name source) used as selector options. */
export const CountryNames: string[] = countryData.map((country) => country.name);

const nameToCode: Record<string, string> = Object.fromEntries(countryData.map((country) => [country.name, country.code]));

/**
 * Resolve an ISO 3166-1 alpha-2 country code from a (stripped) display name.
 *
 * Returns undefined if the name is not a known country.
 */
export const getCountryCode = (name: string): string | undefined => nameToCode[name];
