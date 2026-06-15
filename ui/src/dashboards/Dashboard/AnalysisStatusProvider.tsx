import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

// spec: ./SPEC.md

// project imports
import { useDashboardData } from './DashboardDataProvider';
import { useAuth } from '@utilities/auth';
import { listReactions } from '@thorpi/reactions';
import { Reaction } from '@models/reactions';

/**
 * The number of files fetched per batch. The Analysis Status lookup is a client-side fan-out of
 * N files × M groups requests (the API has no bulk reactions-by-tags query — see SPEC), so the batch
 * size bounds the request burst; "Load more" fetches the next batch on demand.
 */
const FILE_BATCH_SIZE = 25;

/// The per-request page size for `listReactions` (matches `ReactionStatus`'s bounded page size).
const REACTIONS_PAGE_LIMIT = 10000;

/**
 * The number of (group, file) reaction lookups run concurrently. The fan-out is deduped by reaction id, so
 * request order is irrelevant; a small pool overlaps the network latency of independent lookups without
 * bursting the API. The per-batch file cap ({@link FILE_BATCH_SIZE}) still bounds the total work.
 */
const REACTIONS_CONCURRENCY = 5;

/**
 * Page through every reaction for one (group, file) pair, merging results into `byId`.
 *
 * @param group - The group to search within.
 * @param sha256 - The file sha256 to look up reactions for.
 * @param byId - The shared reaction-by-id accumulator to merge into (deduped across pairs).
 * @param errorHandler - Called with a formatted message if an underlying request fails.
 */
async function fetchPairReactions(
  group: string,
  sha256: string,
  byId: Map<string, Reaction>,
  errorHandler: (error: string) => void,
): Promise<void> {
  let more = true;
  let cursor: string | null = null;
  while (more) {
    const res = await listReactions(group, errorHandler, '', sha256, true, cursor, REACTIONS_PAGE_LIMIT);
    if (res && 'details' in res) {
      for (const reaction of res.details) byId.set(reaction.id, reaction);
      if (res.cursor == undefined) {
        more = false;
      } else {
        cursor = String(res.cursor);
      }
    } else {
      more = false;
    }
  }
}

/**
 * Fetch and dedupe every reaction across `sha256s` × `groups`, following pagination per (group, file).
 *
 * Mirrors `ReactionStatus`'s per-(group, file) lookup — paging through
 * `listReactions(group, errCb, '', sha256, true, cursor, limit)` until the cursor is exhausted — but runs the
 * (group, file) pairs through a bounded-concurrency pool ({@link REACTIONS_CONCURRENCY}) rather than strictly
 * serially, overlapping request latency. All `Reaction`s merge into one by-id map so cross-pair duplicates
 * collapse regardless of completion order. The API exposes no bulk/OR reactions-by-tags query, so this
 * bounded fan-out is intentional (see SPEC); callers cap `sha256s` to a batch before invoking.
 *
 * @param sha256s - The file sha256s to look up reactions for (already capped to a batch).
 * @param groups - The groups to search within (the user's groups).
 * @param errorHandler - Called with a formatted message if any underlying request fails.
 * @returns The deduped reactions across every (group, file) pair.
 */
export async function fetchDashboardReactions(
  sha256s: string[],
  groups: string[],
  errorHandler: (error: string) => void,
): Promise<Reaction[]> {
  const byId = new Map<string, Reaction>();
  // materialize every (group, file) pair, then drain them through a fixed set of workers so at most
  // REACTIONS_CONCURRENCY lookups are in flight at once
  const pairs: { group: string; sha256: string }[] = [];
  for (const group of groups) {
    for (const sha256 of sha256s) {
      pairs.push({ group, sha256 });
    }
  }
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < pairs.length) {
      // claim the next pair index atomically (single-threaded JS: the increment can't interleave)
      const { group, sha256 } = pairs[next++];
      await fetchPairReactions(group, sha256, byId, errorHandler);
    }
  };
  await Promise.all(Array.from({ length: Math.min(REACTIONS_CONCURRENCY, pairs.length) }, () => worker()));
  return Array.from(byId.values());
}

/// The state managed by {@link useDashboardReactions}.
interface DashboardReactionsState {
  /// The deduped reactions fetched so far (across every loaded batch).
  reactions: Reaction[];
  /// Whether a batch fetch is currently in flight.
  loading: boolean;
  /// The most recent fetch error, or `null`.
  error: string | null;
  /// The number of files loaded so far (drives the "Load more" affordance).
  loadedCount: number;
}

/// The value exposed by {@link useAnalysisStatus}: the reactions state plus batch/refresh/lazy controls.
export interface AnalysisStatus extends DashboardReactionsState {
  /// The total number of files the analysis status is computed over.
  totalFiles: number;
  /// Whether the panel has scrolled into view yet (the first batch fetches only once it has).
  inView: boolean;
  /// Whether unfetched files remain (drives the "Load more" affordance).
  hasMore: boolean;
  /// Fetch the next batch of files' reactions.
  loadMore: () => void;
  /// Re-run the lookup from the first batch, discarding previously loaded reactions.
  refresh: () => void;
  /// Signal that the panel has scrolled into view, arming the first lazy batch.
  notifyInView: () => void;
}

/**
 * Lazily fan out reaction lookups over the dashboard's files, in bounded batches, with a manual refresh.
 *
 * The first batch is fetched only when `enabled` becomes true (the panel scrolls into view). Each call to
 * the returned `loadMore` fetches the next {@link FILE_BATCH_SIZE} files and merges the results (deduped by
 * id). `refresh` keeps the one-shot guard armed, resets the accumulated state, and re-runs the first batch,
 * so the dashboard's Refresh control can force a fresh lookup. `hasMore` is true while unfetched files
 * remain. An internal fetch epoch discards any in-flight batch whose results land after a `refresh` (or after
 * the `sha256s` identity changes under it), so a stale batch can never merge into freshly-reset state.
 *
 * @param sha256s - All file sha256s in the dashboard graph.
 * @param groups - The user's groups to search within.
 * @param enabled - Gate that triggers the first batch (e.g. in-view).
 * @returns The reactions state plus `loadMore`/`hasMore`/`refresh` controls.
 */
export function useDashboardReactions(
  sha256s: string[],
  groups: string[],
  enabled: boolean,
): DashboardReactionsState & { loadMore: () => void; hasMore: boolean; refresh: () => void } {
  const [state, setState] = useState<DashboardReactionsState>({ reactions: [], loading: false, error: null, loadedCount: 0 });
  // guard so the in-view trigger fires the first batch exactly once (kept armed across a refresh)
  const startedRef = useRef(false);
  // monotonic fetch epoch: bumped on refresh and whenever the file set changes; a batch that started under
  // an older epoch discards its result so overlapping batches never merge stale reactions or clobber loadedCount
  const epochRef = useRef(0);

  const fetchBatch = useCallback(
    async (from: number) => {
      const epoch = epochRef.current;
      const batch = sha256s.slice(from, from + FILE_BATCH_SIZE);
      if (batch.length === 0) {
        setState((prev) => ({ ...prev, loadedCount: sha256s.length }));
        return;
      }
      setState((prev) => ({ ...prev, loading: true, error: null }));
      let batchError: string | null = null;
      const fetched = await fetchDashboardReactions(batch, groups, (error) => {
        batchError = error;
      });
      // a refresh (or a file-set change) moved the epoch on while this batch was in flight: drop its result
      if (epoch !== epochRef.current) {
        return;
      }
      setState((prev) => {
        // merge into a by-id map so re-runs and cross-group duplicates collapse
        const byId = new Map(prev.reactions.map((r) => [r.id, r]));
        for (const reaction of fetched) byId.set(reaction.id, reaction);
        return {
          reactions: Array.from(byId.values()),
          loading: false,
          error: batchError,
          loadedCount: Math.min(from + FILE_BATCH_SIZE, sha256s.length),
        };
      });
    },
    [sha256s, groups],
  );

  // when the file set changes identity, invalidate any in-flight batch and re-arm the one-shot trigger so
  // the effect below refetches the first batch against the new files rather than keeping stale rows
  const sha256sKey = sha256s.join(',');
  useEffect(() => {
    epochRef.current += 1;
    startedRef.current = false;
  }, [sha256sKey]);

  useEffect(() => {
    if (enabled && !startedRef.current && sha256s.length > 0 && groups.length > 0) {
      startedRef.current = true;
      void fetchBatch(0);
    }
  }, [enabled, sha256s.length, groups.length, fetchBatch]);

  const loadMore = useCallback(() => {
    void fetchBatch(state.loadedCount);
  }, [fetchBatch, state.loadedCount]);

  // keep the one-shot guard armed, bump the epoch so any in-flight batch is discarded, reset the accumulated
  // reactions, then re-run the first batch from scratch
  const refresh = useCallback(() => {
    startedRef.current = true;
    epochRef.current += 1;
    setState({ reactions: [], loading: false, error: null, loadedCount: 0 });
    void fetchBatch(0);
  }, [fetchBatch]);

  const hasMore = state.loadedCount < sha256s.length;
  return { ...state, loadMore, hasMore, refresh };
}

/// Context carrying the dashboard's Analysis Status state; `undefined` until a provider supplies it.
const AnalysisStatusContext = createContext<AnalysisStatus | undefined>(undefined);

/**
 * Access the dashboard's Analysis Status state (reactions + batch/refresh controls).
 *
 * @returns The current {@link AnalysisStatus}.
 * @throws If called outside an {@link AnalysisStatusProvider}.
 */
export const useAnalysisStatus = (): AnalysisStatus => {
  const context = useContext(AnalysisStatusContext);
  if (context === undefined) {
    throw new Error('useAnalysisStatus must be used within an AnalysisStatusProvider');
  }
  return context;
};

/// Props for {@link AnalysisStatusProvider}.
interface AnalysisStatusProviderProps {
  /// The subtree that consumes the Analysis Status state.
  children: React.ReactNode;
}

/**
 * Own the dashboard's Analysis Status reactions state and expose it (plus a `refresh`) via context.
 *
 * Hosts the `useDashboardReactions` hook at the dashboard level (above the panel) so the controls bar's
 * Refresh action can trigger `refresh()` while the panel renders the same state. Reads the dashboard's file
 * sha256s from {@link useDashboardData} and the user's groups from {@link useAuth}. The lazy first batch is
 * still gated on the panel scrolling into view: the panel calls `notifyInView()` (via its own
 * `useInView`), which flips an internal `inView` flag here that arms `useDashboardReactions`. Keeping the
 * gate as a provider-owned flag (rather than a prop) lets the state live entirely above the panel while
 * preserving the lazy fetch.
 *
 * @param children - The subtree that consumes the Analysis Status state.
 * @returns The provider wrapping `children`.
 */
export const AnalysisStatusProvider: React.FC<AnalysisStatusProviderProps> = ({ children }) => {
  const { sampleSha256s } = useDashboardData();
  const { userInfo } = useAuth();
  // memoize the groups array on userInfo so its identity is stable across renders — otherwise a fresh `[]`
  // each render would recreate the hook's fetch callbacks and churn the context value
  const groups = useMemo(() => userInfo?.groups ?? [], [userInfo]);
  // armed by the panel's in-view signal; once true it stays true (triggerOnce semantics)
  const [inView, setInView] = useState(false);
  const notifyInView = useCallback(() => setInView(true), []);
  const { reactions, loading, error, loadedCount, loadMore, hasMore, refresh } = useDashboardReactions(sampleSha256s, groups, inView);
  // memoize the context value so consumers re-render only when a constituent field actually changes,
  // matching the sibling DashboardDataProvider's memoized-value convention
  const value = useMemo<AnalysisStatus>(
    () => ({
      reactions,
      loading,
      error,
      loadedCount,
      totalFiles: sampleSha256s.length,
      inView,
      hasMore,
      loadMore,
      refresh,
      notifyInView,
    }),
    [reactions, loading, error, loadedCount, sampleSha256s.length, inView, hasMore, loadMore, refresh, notifyInView],
  );
  return <AnalysisStatusContext.Provider value={value}>{children}</AnalysisStatusContext.Provider>;
};

export default AnalysisStatusProvider;
