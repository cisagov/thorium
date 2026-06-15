import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { FaArrowsRotate } from 'react-icons/fa6';
import { useInView } from 'react-intersection-observer';

// spec: ./SPEC.md

// project imports
import { useAnalysisStatus } from './AnalysisStatusProvider';
import { AnalysisSortColumn, SortDirection, sortReactions } from './analysisTable';
import {
  AnalysisBody,
  AnalysisFilterChip,
  AnalysisFooter,
  AnalysisRow,
  AnalysisSummary,
  AnalysisTable,
  AnalysisTableScroll,
  AnalysisTile,
  FileLinks,
  SortCaret,
  SortHeaderButton,
  SpinningIcon,
  StatusPill,
  TileHeader,
  TileHeaderRow,
} from './styles';
import { Button, ButtonSize, ButtonVariant, IconButton } from '@components/shared/buttons';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { OverlayTipBottom } from '@components/shared/overlay/tips';
import { ReactionStatus } from '@models/reactions';

/// The number of leading sha256 characters shown in the File column before truncating with an ellipsis.
const HASH_DISPLAY_LEN = 12;

/// The visual tokens for each reaction status pill (theme-aware CSS variables).
const STATUS_TOKENS: Record<ReactionStatus, { bg: string; fg: string }> = {
  [ReactionStatus.Completed]: { bg: 'var(--thorium-ok-bg)', fg: 'var(--thorium-button-text)' },
  [ReactionStatus.Failed]: { bg: 'var(--thorium-danger-bg)', fg: 'var(--thorium-button-text)' },
  [ReactionStatus.Started]: { bg: 'var(--thorium-info-bg)', fg: 'var(--thorium-button-text)' },
  [ReactionStatus.Created]: { bg: 'var(--thorium-secondary-panel-bg)', fg: 'var(--thorium-text)' },
};

/// The table columns, in display order, each mapped to its sortable key and header label.
const COLUMNS: { key: AnalysisSortColumn; label: string }[] = [
  { key: AnalysisSortColumn.Pipeline, label: 'Pipeline' },
  { key: AnalysisSortColumn.File, label: 'File' },
  { key: AnalysisSortColumn.Group, label: 'Group' },
  { key: AnalysisSortColumn.Status, label: 'Status' },
];

/**
 * A colored status pill for a reaction status, falling back to a neutral style for unknown values.
 *
 * @param status - The reaction status string.
 * @returns The status pill.
 */
const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const token = STATUS_TOKENS[status as ReactionStatus] ?? { bg: 'var(--thorium-secondary-panel-bg)', fg: 'var(--thorium-text)' };
  return (
    <StatusPill $bg={token.bg} $fg={token.fg}>
      {status}
    </StatusPill>
  );
};

/**
 * The dashboard's "Analysis Status" section: a lazy, batched view of the reactions run against the
 * dashboard's files.
 *
 * Presentation only — the reactions state (and its lazy first-batch / Load-more / refresh controls) is
 * owned by {@link AnalysisStatusProvider} and read via {@link useAnalysisStatus}. The panel adds three
 * view-only affordances on top of that shared state: a header **Refresh** button (re-runs the lookup via
 * the provider's `refresh`), **status filtering** by clicking a summary count chip, and **column
 * sorting** — rows default to file → pipeline → status ascending, and clicking a column header makes it
 * the primary key (toggling direction on repeat clicks) with the default order tiebreaking beneath it.
 * The panel keeps its own `useInView` and calls `notifyInView` once it scrolls into view (mirroring the
 * graph tile's lazy mount), arming the first batch. The fan-out is a bounded client-side workaround for
 * the absence of a bulk reactions-by-tags endpoint (recorded in SPEC).
 *
 * @returns The Analysis Status section.
 */
const AnalysisStatusPanel: React.FC = () => {
  const { reactions, loading, error, loadedCount, totalFiles, inView, hasMore, loadMore, refresh, notifyInView } = useAnalysisStatus();
  // fetch only once the section scrolls into view, mirroring the graph tile's lazy-mount approach
  const { ref, inView: sectionInView } = useInView({ triggerOnce: true });
  useEffect(() => {
    if (sectionInView) notifyInView();
  }, [sectionInView, notifyInView]);

  // the currently-selected status filter (null = show all), toggled by clicking a summary count chip
  const [statusFilter, setStatusFilter] = useState<ReactionStatus | null>(null);
  const toggleStatusFilter = useCallback((status: ReactionStatus) => setStatusFilter((prev) => (prev === status ? null : status)), []);
  // the user-selected primary sort column (null = default file → pipeline → status order) and direction
  const [sort, setSort] = useState<{ column: AnalysisSortColumn | null; direction: SortDirection }>({
    column: null,
    direction: SortDirection.Asc,
  });
  // clicking a header selects it as the primary key; clicking the active header flips its direction
  const handleSort = useCallback((column: AnalysisSortColumn) => {
    setSort((prev) =>
      prev.column === column
        ? { column, direction: prev.direction === SortDirection.Asc ? SortDirection.Desc : SortDirection.Asc }
        : { column, direction: SortDirection.Asc },
    );
  }, []);

  // count reactions by status for the compact summary row (always over the full set, so the filter
  // chips show every available status regardless of the active filter); memoized on the reactions
  // identity so a large set isn't re-tallied on every unrelated render (sort/filter clicks)
  const statusCounts = useMemo(() => {
    const counts = new Map<ReactionStatus, number>();
    for (const reaction of reactions) counts.set(reaction.status, (counts.get(reaction.status) ?? 0) + 1);
    return counts;
  }, [reactions]);

  // apply the status filter, then the multi-key sort — both are pure view transforms over shared state
  const visibleReactions = useMemo(() => {
    const filtered = statusFilter ? reactions.filter((reaction) => reaction.status === statusFilter) : reactions;
    return sortReactions(filtered, sort.column, sort.direction);
  }, [reactions, statusFilter, sort]);

  const noFiles = totalFiles === 0;
  const showEmpty = inView && !loading && !error && reactions.length === 0 && loadedCount > 0;
  // a filter that matches nothing (distinct from "no reactions at all")
  const filteredEmpty = reactions.length > 0 && visibleReactions.length === 0;

  return (
    <AnalysisRow ref={ref}>
      <AnalysisTile>
        <TileHeader>
          <TileHeaderRow>
            <span>Analysis Status</span>
            <OverlayTipBottom tip="Refresh analysis statuses">
              {/* wrap in a span so the tip still shows while the button is disabled during a refresh */}
              <span>
                <IconButton size={ButtonSize.Small} onClick={refresh} disabled={loading || noFiles} aria-label="Refresh analysis statuses">
                  <SpinningIcon $spinning={loading}>
                    <FaArrowsRotate size={14} />
                  </SpinningIcon>
                </IconButton>
              </span>
            </OverlayTipBottom>
          </TileHeaderRow>
        </TileHeader>
        <AnalysisBody>
          {noFiles ? (
            <AlertBanner severity={Severity.Info}>This dashboard has no files to check for analysis.</AlertBanner>
          ) : (
            <>
              {reactions.length > 0 && (
                <AnalysisSummary>
                  <span>
                    {reactions.length} reaction{reactions.length === 1 ? '' : 's'} across {loadedCount} of {totalFiles} file
                    {totalFiles === 1 ? '' : 's'}
                    {statusFilter && ` · showing ${visibleReactions.length} ${statusFilter}`}
                  </span>
                  {Array.from(statusCounts.entries()).map(([status, count]) => (
                    <AnalysisFilterChip
                      key={status}
                      $active={statusFilter === status}
                      aria-pressed={statusFilter === status}
                      aria-label={`Filter by ${status} status`}
                      onClick={() => toggleStatusFilter(status)}
                    >
                      <StatusBadge status={status} /> {count}
                    </AnalysisFilterChip>
                  ))}
                </AnalysisSummary>
              )}
              {error && <AlertBanner severity={Severity.Error}>{error}</AlertBanner>}
              <LoadingSpinner loading={loading} />
              {showEmpty && <AlertBanner severity={Severity.Info}>No reactions found for these files.</AlertBanner>}
              {filteredEmpty && <AlertBanner severity={Severity.Info}>No {statusFilter} reactions.</AlertBanner>}
              {visibleReactions.length > 0 && (
                <AnalysisTableScroll>
                  <AnalysisTable>
                    <thead>
                      <tr>
                        {COLUMNS.map(({ key, label }) => (
                          <th key={key}>
                            <SortHeaderButton onClick={() => handleSort(key)} aria-label={`Sort by ${label}`}>
                              {label}
                              {sort.column === key && <SortCaret>{sort.direction === SortDirection.Asc ? '▲' : '▼'}</SortCaret>}
                            </SortHeaderButton>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {visibleReactions.map((reaction) => (
                        <tr key={reaction.id}>
                          <td>
                            <Link to={`/reaction/${reaction.group}/${reaction.id}`}>{reaction.pipeline}</Link>
                          </td>
                          <td>
                            <FileLinks>
                              {reaction.samples.map((sha256) => (
                                <Link key={sha256} to={`/file/${sha256}`} title={sha256}>
                                  {sha256.length > HASH_DISPLAY_LEN ? `${sha256.slice(0, HASH_DISPLAY_LEN)}…` : sha256}
                                </Link>
                              ))}
                            </FileLinks>
                          </td>
                          <td>{reaction.group}</td>
                          <td>
                            <StatusBadge status={reaction.status} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </AnalysisTable>
                </AnalysisTableScroll>
              )}
              {hasMore && loadedCount > 0 && !loading && (
                <AnalysisFooter>
                  <span>
                    Showing {loadedCount} of {totalFiles} files
                  </span>
                  <Button variant={ButtonVariant.Secondary} size={ButtonSize.Small} onClick={loadMore}>
                    Load more
                  </Button>
                  <span aria-hidden />
                </AnalysisFooter>
              )}
            </>
          )}
        </AnalysisBody>
      </AnalysisTile>
    </AnalysisRow>
  );
};

export default AnalysisStatusPanel;
