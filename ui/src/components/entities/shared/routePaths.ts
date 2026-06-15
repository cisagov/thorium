// project imports
import { Entities } from '@models/entities';

/**
 * Strip the dynamic parts of a route pattern so it can be used as a base path.
 * Removes a trailing `/:param` segment and a trailing `/*` wildcard.
 *
 * @param path - The route pattern to normalize (e.g. `/device/:entityID` or `/repos/*`).
 * @returns The path with any trailing `/:param` or `/*` removed.
 */
export const normalizeRoutePath = (path: string): string => path.replace(/\/:[^/]+$/, '').replace(/\/\*$/, '');

/**
 * Build a map from entity type to its normalized base path from a route map.
 *
 * Each route map entry associates a route pattern with an entity type (looked up via
 * `getType`). When multiple patterns map to the same type, a non-wildcard route is
 * preferred over a wildcard (`/*`) one so links point at the canonical path.
 *
 * @template V - The value type stored in the route map.
 * @param routes - The route map keyed by route pattern.
 * @param getType - Extracts the entity type from a route map value.
 * @returns A partial map from entity type to its normalized base path.
 */
export const buildPathByType = <V>(routes: Record<string, V>, getType: (value: V) => Entities): Partial<Record<Entities, string>> =>
  Object.entries(routes).reduce(
    (acc, [path, value]) => {
      const normalizedPath = normalizeRoutePath(path);
      const type = getType(value);
      const existing = acc[type];
      // prefer a non-wildcard route over a wildcard one so links use the canonical path
      if (!existing || existing.endsWith('*')) {
        acc[type] = normalizedPath;
      }
      return acc;
    },
    {} as Partial<Record<Entities, string>>,
  );
