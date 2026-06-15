/**
 * Flatten an entity request's `key -> values[]` tags into `key: value` chip labels.
 *
 * @param tags - The tag map to flatten (each key maps to zero or more values).
 * @returns One `key: value` string per value, in key/value order.
 */
export function flattenEntityTags(tags: Record<string, string[]>): string[] {
  return Object.entries(tags).flatMap(([key, values]) => values.map((value) => `${key}: ${value}`));
}
