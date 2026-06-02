/**
 * Generate a unique name for a copied resource using a `-copy` suffix.
 *
 * Any existing copy suffix on `baseName` (the new `-copy[-N]` form or the legacy ` (copy[ N])`
 * form) is stripped first so suffixes don't stack. If `<base>-copy` is taken, an incrementing
 * `-copy-N` (starting at 2) is appended until an unused name is found. The suffix deliberately
 * avoids spaces and parentheses so the name is safe in URLs and identifiers.
 *
 * @param baseName - The original resource name to derive a copy name from.
 * @param existingNames - Names already in use that the result must not collide with.
 * @returns A unique copy name not present in `existingNames`.
 */
export function generateCopyName(baseName: string, existingNames: string[]): string {
  const nameSet = new Set(existingNames);
  // Strip an existing copy suffix (new `-copy[-N]` or legacy ` (copy[ N])`) so suffixes don't stack.
  const stripped = baseName.replace(/(?:-copy(?:-\d+)?|\s*\(copy(?:\s+\d+)?\))$/, '');
  const candidate = `${stripped}-copy`;
  if (!nameSet.has(candidate)) return candidate;
  for (let i = 2; ; i++) {
    const numbered = `${stripped}-copy-${i}`;
    if (!nameSet.has(numbered)) return numbered;
  }
}
