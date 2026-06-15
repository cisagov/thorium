/**
 * Turn a machine key (snake_case, camelCase, or PascalCase) into a human-readable, title-cased label.
 *
 * e.g. `image_path` -> `Image Path`, `windowsProcess` -> `Windows Process`, `WindowsProcessTree` ->
 * `Windows Process Tree`. This is the shared fallback used by the entity/association label maps for
 * unmapped values, and directly for generic metadata field keys that have no explicit label.
 *
 * It splits on underscores and lower→upper boundaries only, so acronym runs (e.g. `CVE`) are left
 * intact rather than exploded — for values where acronyms matter, prefer an explicit label map.
 *
 * @param key - The raw key to humanize.
 * @returns The humanized, title-cased label (empty string in, empty string out).
 */
export function humanize(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
