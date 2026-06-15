/**
 * Normalize a resource description for display.
 *
 * Some legacy records store the literal string `"null"` (rather than an absent/empty description) as
 * the description — a serialization artifact from older data. Treating that sentinel as empty keeps it
 * from rendering as the word "null" in the UI. A genuinely empty/absent description also yields `''`.
 *
 * @param description - The raw description from the API (may be `null`/`undefined` or the `"null"` string).
 * @returns The description to display, or an empty string when it is absent or the `"null"` sentinel.
 */
export function cleanDescription(description?: string | null): string {
  return description && description !== 'null' ? description : '';
}
