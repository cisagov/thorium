import React, { useMemo, useState } from 'react';

// spec: ./SPEC.md

// project imports
import { collectTypeCounts, countTagKey } from './chartData';
import { ChartBlock, ChartHeader, ChartsRow, ChartTitle, KeyPicker } from './charts/charts.styled';
import VisxBarChart, { type VisxBar } from './charts/VisxBarChart';
import { useDashboardData } from './DashboardDataProvider';
import { makeIncludeClause } from './deriveStats';
import { useVisibleNodes } from './nodes';
import { collectSeedSummary } from './seedSummary';
import {
  EmptyStats,
  SeedSummaryChip,
  SeedSummaryHeader,
  SeedSummaryPrefix,
  StatsBody,
  StatsTile,
  TileHeader,
  UpdatingIndicator,
} from './styles';
import { toggleTagValue } from './tagFilter';
import { getNodeColor } from '@components/associations/graph/styles';
import { VisualState } from '@components/associations/graph/types';
import { useGraphData } from '@components/associations/data/GraphDataContext';
import { type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { NodeType, type Seed } from '@models/trees';

/**
 * The tag keys the Tag-values chart can chart, in picker order. `FileType` (a file's classified type) is
 * the default; `FileTypeExtension` (the filename extension) and `MIMEType` (the file's MIME type) are the
 * alternates.
 */
const TAG_KEY_OPTIONS = ['FileType', 'FileTypeExtension', 'MIMEType'] as const;

/// Props for {@link StatsPanel}.
export interface StatsPanelProps {
  /**
   * The shared clause list (URL-backed) — the same value/setter given to the surrounding
   * `EntityBrowserProvider`. Kept as props so the charts, the omnibar strip, and the tags tile all mutate
   * one authoritative clause list.
   */
  clauses: Clause[];
  /** Setter for the shared {@link clauses}. */
  setClauses: (next: Clause[]) => void;
  /**
   * Called with the omnibar clause a clicked **Types** bar maps to (an `Include` kind whitelist). Wired to
   * the entity browser's clause state by the composing page so type clicks route through the same
   * dedupe/merge path as other bar clicks.
   */
  onBarClick: (clause: Clause) => void;
  /**
   * The decoded dashboard seed. Its resources are resolved (against the graph's `data_map`) into a
   * human-readable "seeded by" summary rendered as the tile header.
   */
  seed: Seed;
}

/**
 * The dashboard's top stats tile: two responsive visx bar charts over the current (visible) node set.
 *
 * Both charts count over the same node set as the tags tile — the entity browser's `visibleSet` when a
 * filter is active (so the charts *downselect* with the filtered view) and the whole `data_map` otherwise —
 * via the shared {@link useVisibleNodes} hook.
 *
 * - **Types chart** blends Files, Repos, and one bar per entity kind ({@link collectTypeCounts}); colors
 *   come from the graph's `getNodeColor` so bars match node colors. Clicking a bar injects an `Include`
 *   (whitelist) clause for that node type via `onBarClick`.
 * - **Tag-values chart** shows value counts for a configurable tag key ({@link countTagKey}), defaulting to
 *   `FileType` with `FileTypeExtension` and `MIMEType` also offered via a small key picker (component
 *   state). Clicking a bar toggles the value into the key's single is-one-of tag clause via
 *   {@link toggleTagValue}.
 *
 * @param clauses - The shared clause list.
 * @param setClauses - Setter for the shared clause list (used by tag-value clicks).
 * @param onBarClick - Called with the `Include` clause a clicked Types bar maps to.
 * @param seed - The decoded dashboard seed, summarized in the tile header.
 * @returns The stats tile.
 */
const StatsPanel: React.FC<StatsPanelProps> = ({ clauses, setClauses, onBarClick, seed }) => {
  const { loading } = useDashboardData();
  const { graph, graphVersion } = useGraphData();
  const [tagKey, setTagKey] = useState<string>(TAG_KEY_OPTIONS[0]);

  // the "seeded by" summary resolves the seed's resources to display names against the graph's data_map;
  // recompute when the seed changes or the graph reloads (graphVersion) so names fill in as nodes arrive
  const seedSummary = useMemo(() => collectSeedSummary(seed, graph.data_map ?? {}), [seed, graph, graphVersion]);

  // node set = only the visible ids when a filter is active, otherwise every node in the graph (the same
  // downselect the tags tile uses, via the shared useVisibleNodes hook)
  const nodes = useVisibleNodes();

  // Types chart bars: one per node type/kind, colored to match the graph's node colors
  const typeBars = useMemo<VisxBar[]>(
    () =>
      collectTypeCounts(nodes).map((type) => ({
        id: type.kind,
        label: type.label,
        value: type.value,
        // getNodeColor keys on NodeType (Entities | GraphTag); every type kind is a valid NodeType
        color: getNodeColor(type.kind as unknown as NodeType, VisualState.Basic),
      })),
    [nodes],
  );

  // Tag-values chart bars for the selected key
  const tagBars = useMemo<VisxBar[]>(
    () => countTagKey(nodes, tagKey).map((bar) => ({ id: bar.value, label: bar.value, value: bar.count })),
    [nodes, tagKey],
  );

  // a Types bar click whitelists its kind: the bar id is the raw kind. Built through the shared
  // makeIncludeClause so the Include-clause shape is defined in one place
  const onTypeClick = (id: string): void => {
    onBarClick(makeIncludeClause(id));
  };

  // a Tag-values bar click toggles the value into the key's single is-one-of clause (merging same-key clicks)
  const onTagClick = (value: string): void => {
    setClauses(toggleTagValue(clauses, tagKey, value));
  };

  const empty = typeBars.length === 0 && tagBars.length === 0;

  return (
    <StatsTile>
      <TileHeader>
        <SeedSummaryHeader>
          <SeedSummaryPrefix>Seeded by</SeedSummaryPrefix>
          {seedSummary.length > 0 ? (
            seedSummary.map((item, index) => (
              <SeedSummaryChip key={`${item.kind}-${index}-${item.label}`} title={item.label}>
                {item.label}
              </SeedSummaryChip>
            ))
          ) : (
            <UpdatingIndicator>nothing yet</UpdatingIndicator>
          )}
          {loading ? <UpdatingIndicator>updating…</UpdatingIndicator> : null}
        </SeedSummaryHeader>
      </TileHeader>
      <StatsBody>
        {empty ? (
          <EmptyStats>{loading ? 'Loading stats…' : 'No stats to display'}</EmptyStats>
        ) : (
          <ChartsRow>
            <ChartBlock>
              <ChartHeader>
                <ChartTitle>Types</ChartTitle>
              </ChartHeader>
              <VisxBarChart bars={typeBars} onBarClick={onTypeClick} ariaLabel="Node type breakdown" />
            </ChartBlock>
            <ChartBlock>
              <ChartHeader>
                <ChartTitle>Tags</ChartTitle>
                <KeyPicker aria-label="Tag key to chart" value={tagKey} onChange={(event) => setTagKey(event.target.value)}>
                  {TAG_KEY_OPTIONS.map((key) => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
                </KeyPicker>
              </ChartHeader>
              <VisxBarChart bars={tagBars} onBarClick={onTagClick} ariaLabel={`${tagKey} value breakdown`} />
            </ChartBlock>
          </ChartsRow>
        )}
      </StatsBody>
    </StatsTile>
  );
};

export default StatsPanel;
