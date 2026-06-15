// spec: ../ToolResult.spec.md
import { useCallback, useEffect, useRef, useState } from 'react';

// project imports
import { getFileDetails } from '@thorpi/files';
import { useAuth } from '@utilities/auth';
import { Sample } from '@models/files';
import { Output } from '@models/results';

/** Child sets at or above this size are fetched on demand (manual button) rather than auto-fetched. */
export const AUTO_FETCH_THRESHOLD = 100;
/** How many child SHA256s to request per parallel batch. */
export const CHILD_BATCH_SIZE = 100;

/** True when a child set is large enough to require a manual (on-demand) fetch. */
export const isManualChildFetch = (total: number): boolean => total >= AUTO_FETCH_THRESHOLD;

/** Split a list into fixed-size chunks (last chunk may be shorter). A non-positive size yields one chunk. */
export function chunk<T>(list: T[], size: number): T[][] {
  if (size <= 0) return list.length ? [list] : [];
  const out: T[][] = [];
  for (let i = 0; i < list.length; i += size) {
    out.push(list.slice(i, i + size));
  }
  return out;
}

/**
 * Fetch child file details in sequential parallel batches.
 *
 * Each batch resolves its SHA256s concurrently, then `onBatch` is invoked with the samples that
 * resolved (failed/`null` fetches are omitted) and the number *attempted* in the batch — so callers
 * can render progressively and keep a progress count that reaches the total even when some fail.
 *
 * @param shas - Child SHA256s to resolve.
 * @param fetchOne - Fetches a single sample by SHA256 (returns `null` on failure; must not throw).
 * @param onBatch - Called after each batch with the newly resolved samples and the batch's size.
 * @param batchSize - Max SHA256s fetched in parallel per batch (defaults to {@link CHILD_BATCH_SIZE}).
 */
export async function fetchChildrenBatches(
  shas: string[],
  fetchOne: (sha: string) => Promise<Sample | null>,
  onBatch: (resolved: Record<string, Sample>, attempted: number) => void,
  batchSize: number = CHILD_BATCH_SIZE,
): Promise<void> {
  for (const batch of chunk(shas, batchSize)) {
    const details = await Promise.all(batch.map((sha) => fetchOne(sha)));
    const resolved: Record<string, Sample> = {};
    batch.forEach((sha, i) => {
      const sample = details[i];
      if (sample) resolved[sha] = sample;
    });
    onBatch(resolved, batch.length);
  }
}

/** Progress of resolving a result's child SHA256s to their file details. */
export type ChildrenFetchStatus = 'idle' | 'loading' | 'done';

/** Shared child-metadata state returned by {@link useChildrenMetadata}. */
export interface ChildrenMetadata {
  /** Resolved file details keyed by child SHA256 (missing entries are unfetched or failed). */
  samples: Record<string, Sample>;
  status: ChildrenFetchStatus;
  /** Number of children whose fetch has been attempted (counts failures too). */
  loaded: number;
  /** Total child count for the active result. */
  total: number;
  /** Whether the child set requires a manual fetch (>= {@link AUTO_FETCH_THRESHOLD}). */
  isManual: boolean;
  /** Trigger the (batched) fetch; used for large child sets. Small sets auto-fetch when active. */
  fetch: () => void;
}

/**
 * Own the child-metadata fetch state for a tool result so the header fetch button (in `ToolResult`)
 * and the `ChildrenTab` body share one source of truth.
 *
 * Small child sets (`< AUTO_FETCH_THRESHOLD`) auto-fetch the first time the Children tab is opened;
 * large sets fetch on demand via {@link ChildrenMetadata.fetch}. State is keyed on `result.id` and
 * resets when it changes, so switching result versions (or a poll refresh producing a new `Output`
 * object with the same id) doesn't trigger a refetch storm.
 *
 * @param result - The active tool result whose `children` map is resolved (may be undefined before
 *   results load; the hook then reports an empty, idle state).
 * @param active - Whether the Children tab is currently selected (gates the lazy auto-fetch).
 * @returns The shared {@link ChildrenMetadata} state.
 */
export function useChildrenMetadata(result: Output | undefined, active: boolean): ChildrenMetadata {
  const { checkCookie } = useAuth();
  const [samples, setSamples] = useState<Record<string, Sample>>({});
  const [status, setStatus] = useState<ChildrenFetchStatus>('idle');
  const [loaded, setLoaded] = useState(0);

  const resultId = result?.id ?? '';
  const total = result ? Object.keys(result.children).length : 0;
  const manual = isManualChildFetch(total);

  // fetch at most once per result id (covers strict-mode double effects + rapid clicks/races)
  const startedRef = useRef<string | null>(null);

  // reset when the active result changes; keyed on the stable id, not the object, so polling that
  // returns a fresh Output with the same id doesn't wipe already-fetched samples
  useEffect(() => {
    startedRef.current = null;
    setSamples({});
    setStatus('idle');
    setLoaded(0);
  }, [resultId]);

  const runFetch = useCallback(async () => {
    if (!result || startedRef.current === resultId) return;
    startedRef.current = resultId;
    const shas = Object.keys(result.children);
    if (shas.length === 0) {
      setStatus('done');
      return;
    }
    setStatus('loading');
    const errorHandler = () => void checkCookie();
    await fetchChildrenBatches(
      shas,
      (sha) => getFileDetails(sha, errorHandler),
      (resolved, attempted) => {
        // a version switch nulls startedRef and resets state; a batch from the previous version's
        // in-flight fetch must not merge stale samples or inflate the new version's progress count
        if (startedRef.current !== resultId) return;
        setSamples((prev) => ({ ...prev, ...resolved }));
        setLoaded((prev) => prev + attempted);
      },
    );
    // the trailing 'done' must not mark the new version done (which would hide the manual fetch
    // button and suppress auto-fetch, stranding the new version's children)
    if (startedRef.current !== resultId) return;
    setStatus('done');
  }, [resultId, result?.children, checkCookie]);

  // lazy auto-fetch for small child sets, only once the tab is actually opened
  useEffect(() => {
    if (active && !manual && total > 0 && status === 'idle') {
      void runFetch();
    }
  }, [active, manual, total, status, runFetch]);

  const fetch = useCallback(() => void runFetch(), [runFetch]);

  return { samples, status, loaded, total, isManual: manual, fetch };
}
