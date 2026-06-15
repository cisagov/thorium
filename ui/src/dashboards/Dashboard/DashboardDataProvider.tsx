import React, { createContext, useContext, useMemo } from 'react';

// spec: ./SPEC.md

// project imports
import { collectSampleSha256s } from './deriveStats';
import { useGraphData } from '@components/associations/data/GraphDataContext';

/**
 * The value exposed by {@link DashboardDataProvider} to the dashboard tiles.
 *
 * The derived, graph-backed model plus the graph's `loading`/`error` so tiles render loading and error
 * states without touching the graph context directly.
 */
export interface DashboardData {
  /// The deduped sha256s of every file in the graph, recomputed whenever the shared graph changes.
  sampleSha256s: string[];
  /// Whether the initial graph fetch is in flight (grows do not set this — see SPEC).
  loading: boolean;
  /// The graph fetch error message, or `null` when there is none.
  error: string | null;
}

/// Context carrying the derived dashboard data; `undefined` until a provider supplies it.
const DashboardDataContext = createContext<DashboardData | undefined>(undefined);

/**
 * Access the derived dashboard data.
 *
 * @returns The current {@link DashboardData}.
 * @throws If called outside a {@link DashboardDataProvider}.
 */
export const useDashboardData = (): DashboardData => {
  const context = useContext(DashboardDataContext);
  if (context === undefined) {
    throw new Error('useDashboardData must be used within a DashboardDataProvider');
  }
  return context;
};

/// Props for {@link DashboardDataProvider}.
interface DashboardDataProviderProps {
  /// The subtree that consumes the derived dashboard data.
  children: React.ReactNode;
}

/**
 * Derive the dashboard data from the shared association graph and expose it via context.
 *
 * Reads the graph context (`graph.data_map`, `graphVersion`, `loading`, `error`) and computes
 * {@link collectSampleSha256s} memoized on `graphVersion` — the single signal the graph context bumps
 * whenever `data_map` changes — so the potentially-expensive walk runs only when the graph actually changes.
 *
 * @param children - The subtree that consumes the derived dashboard data.
 * @returns The provider wrapping `children`.
 */
export const DashboardDataProvider: React.FC<DashboardDataProviderProps> = ({ children }) => {
  const { graph, graphVersion, loading, error } = useGraphData();
  const sampleSha256s = useMemo(
    () => collectSampleSha256s(graph.data_map),
    // graphVersion is bumped whenever graph.data_map changes; graph itself is a stable ref
    [graphVersion],
  );
  const value = useMemo<DashboardData>(() => ({ sampleSha256s, loading, error }), [sampleSha256s, loading, error]);
  return <DashboardDataContext.Provider value={value}>{children}</DashboardDataContext.Provider>;
};

export default DashboardDataProvider;
