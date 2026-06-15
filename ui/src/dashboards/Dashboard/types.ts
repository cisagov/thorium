// spec: ./SPEC.md

// project imports
import type { Seed } from '@models/trees';

/**
 * The decoded result of the dashboard's URL seed params: the {@link Seed} to build the graph from
 * plus the validated crawl depth.
 *
 * Returned by `decodeSeedParams` and accepted by `encodeSeedParams` so the URL ⇄ graph boundary is
 * a single round-trippable value.
 */
export interface DashboardSeedParams {
  /// The seed handed to `getInitialTree` / `GraphDataProvider`.
  seed: Seed;
  /// The validated, clamped crawl depth (0..=10, default 2).
  depth: number;
}
