// project imports
import { humanize } from '@utilities/humanize';

/**
 * Look up a human-readable display label for a key, falling back to {@link humanize} when the key has
 * no explicit mapping. Shared by the entity/association label tables so the humanize-fallback behavior
 * lives in one place.
 *
 * @template K - The string-literal key type the label table is keyed by (e.g. an enum's value union).
 * @param labels - A table mapping each known key to its display label.
 * @param key - A known key, or a raw string carried by call sites that only have a `string`.
 * @returns The mapped label, or the humanized key when it isn't present in `labels`.
 */
export function labelWithFallback<K extends string>(labels: Record<K, string>, key: K | string): string {
  return labels[key as K] ?? humanize(String(key));
}
