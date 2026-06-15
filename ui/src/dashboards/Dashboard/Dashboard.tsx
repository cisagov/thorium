import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom';

// spec: ./SPEC.md

// project imports
import AnalysisStatusPanel from './AnalysisStatusPanel';
import { AnalysisStatusProvider, useAnalysisStatus } from './AnalysisStatusProvider';
import BrowserTile from './BrowserTile';
import DashboardControlsBar from './DashboardControlsBar';
import DashboardDataProvider from './DashboardDataProvider';
import DashboardGraphTile from './DashboardGraphTile';
import DashboardOmnibar from './DashboardOmnibar';
import { makeDepthClause, withDepthClause } from './depthClause';
import PaneExpandToggle from './PaneExpandToggle';
import { decodeSeedParams } from './seedParams';
import StatsPanel from './StatsPanel';
import {
  ColWrap,
  ContentGrid,
  ContentTile,
  ControlsRow,
  DashboardLayout,
  OmnibarRow,
  PaneScroll,
  StatsRow,
  TileHeader,
  TileHeaderActions,
  TileHeaderRow,
  ULTRA_WIDE_BREAKPOINT,
} from './styles';
import { resetFilterClauses, toggleTagValue } from './tagFilter';
import { getDepthFromClauses } from '@components/associations/browsing/EntityBrowser/browserHelpers';
import { EntityBrowserProvider } from '@components/associations/browsing/EntityBrowser/EntityBrowserContext';
import EntitySortControls from '@components/associations/browsing/EntityBrowser/EntitySortControls';
import { MAX_DEPTH } from '@components/associations/browsing/EntityBrowser/omnibarOptions';
import { SortMode } from '@components/associations/browsing/EntityBrowser/types';
import { GraphDataProvider, useGraphData } from '@components/associations/data/GraphDataContext';
import Page from '@components/pages/Page';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { Clause, ClauseIsMulti } from '@components/shared/inputs/omnibar/ClauseTypes';
import { useOmnibarUrlState } from '@components/shared/inputs/omnibar/useOmnibarUrlState';
import { Tabs } from '@components/shared/tabs';
import { boolCodec, setCodec, stringCodec } from '@utilities/url/codecs';
import { useUrlState } from '@utilities/url/useUrlState';
import { useMediaQuery } from '@utilities/useMediaQuery';
import type { Seed } from '@models/trees';

/// The seed keys the dashboard owns; the decoded seed is memoized on just these so unrelated URL changes
/// (omnibar clause params, tab hash) never trigger a graph refetch.
const SEED_KEYS = ['sample', 'entity', 'repo', 'tag', 'depth'] as const;

/// The codec binding the hidden-node set to repeated `hidden=<id>` params. Module-level so its identity is
/// stable across renders (the {@link useUrlState} memo contract keys only on the URL, expecting a stable codec).
const HIDDEN_CODEC = setCodec('hidden');
/// The codec binding the flagged-only toggle to `flagged=1`. Module-level for the same stable-identity reason.
const FLAGGED_CODEC = boolCodec('flagged');
/// The fallback hidden-node set (empty) shared by every render, so an absent `hidden` param decodes to one
/// stable reference rather than a fresh Set each render.
const EMPTY_HIDDEN: Set<string> = new Set();

/// The codec binding the focused (expanded-to-fill) pane to `panel=browser|graph`; absent means the split
/// view. Module-level so its identity is stable across renders (the {@link useUrlState} memo contract).
const PANEL_CODEC = stringCodec('panel');
/// The codec binding the re-rooted (focused) subtree to `focus=<nodeId>`; absent means the natural roots.
/// Disjoint from the seed keys, so changing focus never refetches the graph or remounts the content.
const FOCUS_CODEC = stringCodec('focus');
/// The codec binding the re-rooted (gear) view to `root=<nodeId>`; absent means the natural roots. Distinct
/// from `focus` (which prunes to a subtree) — re-root reorders the whole graph under the node. Also disjoint
/// from the seed keys, so re-rooting never refetches or remounts.
const ROOT_CODEC = stringCodec('root');
/// The codec binding the sort mode to `sort=<mode>`; the default (Flags) is stored as the empty string so it
/// writes no param, keeping the URL clean. Module-level for stable identity (the {@link useUrlState} contract).
const SORT_CODEC = stringCodec('sort');
/// The codec binding the group-by-type toggle to `ungrouped=1`. Grouping is ON by default, so the param is
/// **inverted** — it only appears when the user turns grouping OFF — matching the `flagged=1`-when-on convention.
const UNGROUPED_CODEC = boolCodec('ungrouped');

/**
 * The content tabs shown below the ultra-wide breakpoint. Persisted in the URL hash so a shared link
 * restores the view; the raw enum value is the hash (e.g. `#graph`).
 */
enum ContentTab {
  /// The entity browser tile.
  Entities = 'entities',
  /// The association graph tile.
  Graph = 'graph',
}

/// Which content pane is focused (expanded to fill the region) via the ⤢ toggle; drives the `panel` URL key.
enum PaneFocus {
  /// The entity browser pane fills the region (the graph pane is hidden but kept mounted).
  Browser = 'browser',
  /// The association graph pane fills the region (the browser pane is hidden but kept mounted).
  Graph = 'graph',
}

/**
 * The arrangement of the content region, derived from the viewport and the focused pane. All three are one
 * stable element tree differing only by CSS/props, so switching never re-parents (remounts) the heavy tiles.
 */
enum ContentMode {
  /// Ultra-wide, no pane focused: browser and graph side by side (two columns).
  Split = 'split',
  /// Ultra-wide, a pane focused: that pane fills the single column, the other is hidden (still mounted).
  Expanded = 'expanded',
  /// Narrow: a tab bar toggles which single-column pane is shown (the other hidden, still mounted).
  Tabs = 'tabs',
}

/**
 * True when a seed carries at least one resource (sample/entity/repo/tag). An empty seed is the
 * empty-dashboard state (warning banner + builder link).
 *
 * @param seed - The decoded seed.
 * @returns Whether the seed has any resource to build a graph from.
 */
export function seedHasResources(seed: Seed): boolean {
  return Boolean(seed.samples?.length || seed.entities?.length || seed.repos?.length || (seed.tags && Object.keys(seed.tags).length > 0));
}

/**
 * Whether two clauses are identical (same category/field/condition and same value(s)), used to avoid
 * appending a duplicate clause when a stats bar is clicked twice.
 *
 * @param a - The first clause.
 * @param b - The second clause.
 * @returns Whether the clauses are equal.
 */
export function clausesEqual(a: Clause, b: Clause): boolean {
  if (a.category !== b.category || a.field !== b.field || a.condition !== b.condition) {
    return false;
  }
  const aValues = ClauseIsMulti(a) ? a.value.values : [a.value.value];
  const bValues = ClauseIsMulti(b) ? b.value.values : [b.value.value];
  return aValues.length === bValues.length && aValues.every((v, i) => v === bValues[i]);
}

/**
 * Append `clause` to `clauses` unless an identical clause is already present.
 *
 * @param clauses - The current clause list.
 * @param clause - The clause to append.
 * @returns The list with `clause` appended, or the original list when it is already present.
 */
export function dedupeAppend(clauses: Clause[], clause: Clause): Clause[] {
  if (clauses.some((existing) => clausesEqual(existing, clause))) {
    return clauses;
  }
  return [...clauses, clause];
}

/**
 * The composed dashboard content (rendered once a non-empty seed is present).
 *
 * Owns the URL-backed clause state (via {@link useOmnibarUrlState}) and the URL-backed hidden/flagged state
 * (via {@link useUrlState}), hands them to the {@link EntityBrowserProvider} as controlled state, and lays out the stats panel, the
 * always-shown omnibar strip, and the browser/graph tiles. On ultra-wide viewports the tiles sit side by
 * side (both active); otherwise they live under {@link Tabs} with both panels mounted and toggled via
 * `display: none`, the active tab persisted in the URL hash so shared links restore the view.
 */
export interface DashboardContentProps {
  /// The decoded seed (already known to have resources).
  seed: Seed;
  /// The crawl depth captured once at mount; stable so the graph provider never refetches on depth change.
  depthAtMount: number;
}

export const DashboardContent: React.FC<DashboardContentProps> = ({ seed, depthAtMount }) => {
  const location = useLocation();
  const navigate = useNavigate();
  // clauses live in the URL (disjoint keys from the seed's sample/entity/repo/tag/depth) so filters are
  // shareable; the same clauses/setClauses drive the provider, the omnibar strip, and the stats-bar clicks
  const { clauses, setClauses } = useOmnibarUrlState({ clauses: [], time: { mode: 'all' } });
  // hidden/flagged are URL-backed (disjoint keys from the seed and omnibar clauses) and fed to the provider
  // as controlled state, so hiding a node / flagging survives refresh and is shareable via the dashboard link
  const [hiddenNodes, setHiddenNodes] = useUrlState(HIDDEN_CODEC, EMPTY_HIDDEN);
  const [flaggedOnly, setFlaggedOnly] = useUrlState(FLAGGED_CODEC, false);
  // the focused subtree and the re-rooted view are each URL-backed so they're shareable and survive refresh
  // ('' ⇒ natural roots). They are **mutually exclusive**, so both keys are written in ONE atomic update via
  // `setExclusiveRoot` — two separate useUrlState writes in a single click tick would race (each computes from
  // the pre-click params, so the second clobbers the first), which would silently drop the change.
  const [focusRoot] = useUrlState(FOCUS_CODEC, '');
  const [reRoot] = useUrlState(ROOT_CODEC, '');
  const [, setRootParams] = useSearchParams();
  const setExclusiveRoot = useCallback(
    (focus: string, root: string) => {
      setRootParams(
        (prev) => {
          const out = new URLSearchParams(prev);
          // clear both keys, then write only the active one (empty ⇒ absent), so focus and root never coexist
          FOCUS_CODEC.keys(out).forEach((key) => out.delete(key));
          ROOT_CODEC.keys(out).forEach((key) => out.delete(key));
          FOCUS_CODEC.encode(focus, out);
          ROOT_CODEC.encode(root, out);
          return out;
        },
        { replace: true },
      );
    },
    [setRootParams],
  );
  // sort/group are URL-backed too, so the chosen ordering/grouping survives refresh and is shareable. The
  // default sort (Flags) writes no `sort` param; grouping is on by default, tracked via the inverted
  // `ungrouped` flag so the default writes no param either.
  const [sortParam, setSortParam] = useUrlState(SORT_CODEC, '');
  const [ungrouped, setUngrouped] = useUrlState(UNGROUPED_CODEC, false);

  // one-shot: if the URL carried no depth clause, seed one for the mount depth so the omnibar always
  // shows the current crawl depth. Guarded by a ref so it runs once and never fights later user edits.
  const seededDepthRef = useRef(false);
  useEffect(() => {
    if (!seededDepthRef.current && getDepthFromClauses(clauses, 0) === 0) {
      seededDepthRef.current = true;
      setClauses([...clauses, makeDepthClause(depthAtMount)]);
    }
  }, [clauses, setClauses, depthAtMount]);

  // the current crawl depth reflected in the omnibar (falls back to the mount depth before the one-shot
  // seed lands); Grow Level raises it by one, bounded by MAX_DEPTH, replacing the single depth clause
  const currentDepth = getDepthFromClauses(clauses, depthAtMount);
  const onGrowLevel = useCallback(() => {
    setClauses(withDepthClause(clauses, Math.min(currentDepth + 1, MAX_DEPTH)));
  }, [clauses, setClauses, currentDepth]);

  const isUltraWide = useMediaQuery(`(min-width: ${ULTRA_WIDE_BREAKPOINT})`);

  // the active content tab is read from / written to the URL hash so shared links restore the view; an
  // unknown/absent hash falls back to the entities tab
  const activeTab = location.hash === `#${ContentTab.Graph}` ? ContentTab.Graph : ContentTab.Entities;
  const setActiveTab = useCallback(
    (tab: ContentTab) => {
      // replace so tab switches don't stack history entries (back/forward stays page-level)
      void navigate({ hash: `#${tab}`, search: location.search }, { replace: true });
    },
    [navigate, location.search],
  );

  // a clicked stats bar updates the shared clause list: tag-category clauses merge into the key's
  // is-one-of set (so clicking several values on one key becomes one "any of" clause and clicking the
  // same value twice toggles it off); non-tag (Include kind) clauses append once, deduped
  const onBarClick = useCallback(
    (clause: Clause) => {
      if (clause.category === 'tag') {
        const value = ClauseIsMulti(clause) ? clause.value.values[0] : clause.value.value;
        setClauses(toggleTagValue(clauses, clause.field, value));
      } else {
        setClauses(dedupeAppend(clauses, clause));
      }
    },
    [clauses, setClauses],
  );

  return (
    <GraphDataProvider initial={seed} depth={depthAtMount}>
      <DashboardDataProvider>
        <AnalysisStatusProvider>
          <EntityBrowserProvider
            roots={{ kind: 'initial' }}
            defaultDepth={depthAtMount}
            clauses={clauses}
            setClauses={setClauses}
            hiddenNodes={hiddenNodes}
            onHiddenNodesChange={setHiddenNodes}
            flaggedOnly={flaggedOnly}
            setFlaggedOnly={setFlaggedOnly}
            sortMode={(sortParam as SortMode) || SortMode.Flags}
            setSortMode={(mode) => setSortParam(mode === SortMode.Flags ? '' : mode)}
            groupByResource={!ungrouped}
            setGroupByResource={(on) => setUngrouped(!on)}
            focusRoot={focusRoot || null}
            onFocusRootChange={(id) => setExclusiveRoot(id ?? '', '')}
            reRoot={reRoot || null}
            onReRootChange={(id) => setExclusiveRoot('', id ?? '')}
          >
            <DashboardBody
              seed={seed}
              clauses={clauses}
              setClauses={setClauses}
              currentDepth={currentDepth}
              onGrowLevel={onGrowLevel}
              onBarClick={onBarClick}
              isUltraWide={isUltraWide}
              activeTab={activeTab}
              setActiveTab={setActiveTab}
            />
          </EntityBrowserProvider>
        </AnalysisStatusProvider>
      </DashboardDataProvider>
    </GraphDataProvider>
  );
};

/// Props for {@link DashboardBody}.
interface DashboardBodyProps {
  /// The decoded dashboard seed, summarized in the stats-tile header.
  seed: Seed;
  /// The shared clause list (URL-backed).
  clauses: Clause[];
  /// Setter for the shared clause list.
  setClauses: (next: Clause[]) => void;
  /// The current crawl depth (drives the Grow control).
  currentDepth: number;
  /// Called to increase the crawl depth by one level.
  onGrowLevel: () => void;
  /// Called with the omnibar clause a clicked stats bar maps to.
  onBarClick: (clause: Clause) => void;
  /// Whether the ultra-wide two-column layout is active.
  isUltraWide: boolean;
  /// The active content tab (narrow layout only).
  activeTab: ContentTab;
  /// Setter for the active content tab.
  setActiveTab: (tab: ContentTab) => void;
}

/**
 * The dashboard's rendered body, living inside the full provider stack so it can read the graph and
 * Analysis Status contexts.
 *
 * Owns the Reset-filters and Refresh-data wiring the controls bar needs: Reset replaces the clause list
 * with {@link resetFilterClauses}; Refresh reloads the shared graph (`useGraphData().reload()`, which bumps
 * `graphVersion` so the stats/file-list derivers recompute) and re-runs the Analysis Status reactions
 * (`useAnalysisStatus().refresh()`), spinning the Refresh icon until the graph reload settles. Lays out the
 * stats panel, controls bar, omnibar strip, and the browser/graph/analysis tiles per the responsive
 * arrangement.
 *
 * @returns The dashboard body.
 */
const DashboardBody: React.FC<DashboardBodyProps> = ({
  seed,
  clauses,
  setClauses,
  currentDepth,
  onGrowLevel,
  onBarClick,
  isUltraWide,
  activeTab,
  setActiveTab,
}) => {
  const { reload, growing } = useGraphData();
  const { refresh: refreshReactions } = useAnalysisStatus();
  const [refreshing, setRefreshing] = useState(false);

  const onResetFilters = useCallback(() => {
    setClauses(resetFilterClauses(clauses));
  }, [clauses, setClauses]);

  // refresh both data sources: reload() re-fetches the graph (bumping graphVersion so the stats/file-list
  // derivers recompute), and refreshReactions() re-runs the Analysis Status fan-out. The spinner tracks the
  // graph reload (the longer async op); the reactions refresh manages its own loading state in the panel.
  const onRefresh = useCallback(() => {
    refreshReactions();
    setRefreshing(true);
    void reload().finally(() => setRefreshing(false));
  }, [reload, refreshReactions]);

  // the focused (expanded-to-fill) pane, URL-backed so a focused view is shareable/survives refresh. Only
  // meaningful in the ultra-wide two-column layout; the narrow (tabs) layout ignores it. Persisted as a plain
  // string; narrowed to the `PaneFocus` enum (or '') so comparisons stay enum-typed.
  const [rawPanel, setPanel] = useUrlState(PANEL_CODEC, '');
  const expandedPane = rawPanel as PaneFocus | '';
  // the content arrangement: narrow → tabs; ultra-wide with a focused pane → expanded; else the split view
  const mode = !isUltraWide ? ContentMode.Tabs : expandedPane ? ContentMode.Expanded : ContentMode.Split;
  // pane visibility: BOTH are shown in split AND expanded (expanded just stacks them into one column with the
  // focused pane on top); in the narrow tabs layout only the active tab shows. The graph's `active` (WebGL gate)
  // tracks its visibility, so it stays alive when stacked below rather than being torn down.
  const browserVisible = mode !== ContentMode.Tabs || activeTab === ContentTab.Entities;
  const graphVisible = mode !== ContentMode.Tabs || activeTab === ContentTab.Graph;
  // in the single-column expanded stack, the focused pane sits on top via grid `order` (no re-parenting)
  const graphFocused = mode === ContentMode.Expanded && expandedPane === PaneFocus.Graph;
  const browserOrder = graphFocused ? 1 : 0;
  const graphOrder = graphFocused ? 0 : 1;
  // the expand/collapse toggles are a shared state: while EITHER pane is expanded, BOTH toggles read as
  // "collapse" and clicking either restores the split view; only in the split view do they read as "expand"
  // (each focusing its own pane). So the two buttons never disagree.
  const anyExpanded = mode === ContentMode.Expanded;
  const toggleBrowserExpand = useCallback(() => setPanel(anyExpanded ? '' : PaneFocus.Browser), [anyExpanded, setPanel]);
  const toggleGraphExpand = useCallback(() => setPanel(anyExpanded ? '' : PaneFocus.Graph), [anyExpanded, setPanel]);

  // measure the two split columns so the reactions tile can tuck under the SHORTER one (greedy masonry)
  // rather than always sitting full-width below; only the anchors are measured, so the reactions tile's own
  // height never feeds back into the choice (no oscillation) and the anchors are never moved (no remount).
  const browserRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<HTMLDivElement | null>(null);
  const [shortCol, setShortCol] = useState<'1' | '2'>('1');
  useEffect(() => {
    const browserEl = browserRef.current;
    const graphEl = graphRef.current;
    if (!browserEl || !graphEl || typeof ResizeObserver === 'undefined') return;
    let raf = 0;
    const measure = () => {
      raf = 0;
      // ties (and the taller browser) keep the reactions tile under column 1; a taller browser pushes it to 2
      setShortCol(browserEl.offsetHeight <= graphEl.offsetHeight ? '1' : '2');
    };
    // coalesce a burst of resizes into one measure on the next frame
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(browserEl);
    observer.observe(graphEl);
    measure();
    return () => {
      observer.disconnect();
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);
  // In split, the reactions tile nests INSIDE whichever column is shorter (its column's own flex stack →
  // flush under that pane, true masonry); in expanded/tabs it renders full-width below both columns. Each
  // column is a flex stack (ColWrap), so reactions never leaves a gap the way a shared grid row would.
  const split = mode === ContentMode.Split;
  const reactionsInBrowser = split && shortCol === '1';
  const reactionsInGraph = split && shortCol === '2';

  return (
    <DashboardLayout>
      <StatsRow>
        <StatsPanel seed={seed} clauses={clauses} setClauses={setClauses} onBarClick={onBarClick} />
      </StatsRow>
      <ControlsRow>
        <DashboardControlsBar
          currentDepth={currentDepth}
          maxDepth={MAX_DEPTH}
          onGrowLevel={onGrowLevel}
          onResetFilters={onResetFilters}
          onRefresh={onRefresh}
          refreshing={refreshing}
          growing={growing}
        />
      </ControlsRow>
      <OmnibarRow>
        <DashboardOmnibar clauses={clauses} setClauses={setClauses} />
      </OmnibarRow>
      {/* the browser and graph panes always live in the same two ColWrap columns, so switching
          split ⇄ expanded ⇄ tabs never re-parents (remounts) the browser's state or the graph's WebGL
          canvas. Each column is a flex stack, so in split the reactions tile nests flush under the shorter
          column's pane; in expanded/tabs it renders full-width below. The tab bar shows only in the narrow
          layout; the ⤢ focus toggles show only ultra-wide. */}
      {mode === ContentMode.Tabs && (
        <Tabs<ContentTab>
          aria-label="Dashboard content"
          active={activeTab}
          onChange={setActiveTab}
          tabs={[
            { key: ContentTab.Entities, label: 'Entities' },
            { key: ContentTab.Graph, label: 'Graph' },
          ]}
        />
      )}
      <ContentGrid $columns={split ? 2 : 1}>
        <ColWrap $order={browserOrder} $hidden={!browserVisible}>
          <ContentTile ref={browserRef}>
            {/* the pane header (with the ⤢ focus toggle) shows only in the two-column ultra-wide layout; in
                the narrow tabs layout the tab bar already labels the pane, so no header is rendered */}
            {isUltraWide && (
              <TileHeader>
                <TileHeaderRow>
                  <span>Entities</span>
                  {/* ultra-wide hosts the sort/group controls here (right before the expand toggle); the
                      browser body suppresses its own copy via showSortControls={false} below */}
                  <TileHeaderActions>
                    <EntitySortControls />
                    <PaneExpandToggle
                      expanded={anyExpanded}
                      onToggle={toggleBrowserExpand}
                      expandTip="Expand entities (stack the graph below)"
                      expandAriaLabel="Expand entities and stack the graph below"
                    />
                  </TileHeaderActions>
                </TileHeaderRow>
              </TileHeader>
            )}
            <PaneScroll>
              <BrowserTile showSortControls={!isUltraWide} />
            </PaneScroll>
          </ContentTile>
          {reactionsInBrowser && <AnalysisStatusPanel />}
        </ColWrap>
        <ColWrap $order={graphOrder} $hidden={!graphVisible}>
          {/* wrapper measured for the masonry choice — its height is the graph tile alone (reactions is a
              sibling below), so the shorter-column decision never feeds back on itself */}
          <div ref={graphRef} style={{ minWidth: 0 }}>
            <DashboardGraphTile
              active={graphVisible}
              canExpand={isUltraWide}
              expanded={anyExpanded}
              onToggleExpand={toggleGraphExpand}
              fill={mode === ContentMode.Expanded}
            />
          </div>
          {reactionsInGraph && <AnalysisStatusPanel />}
        </ColWrap>
      </ContentGrid>
      {/* single-column layouts: reactions spans full width below both columns */}
      {!split && <AnalysisStatusPanel />}
    </DashboardLayout>
  );
};

/**
 * The custom dashboard page (`/dashboard/view`).
 *
 * Decodes the URL seed params into a {@link Seed} plus crawl depth, memoizing the seed on just the seed
 * keys so unrelated URL edits (omnibar clauses, tab hash) never refetch the graph, and capturing the
 * depth once at mount so the graph provider's `depth` prop stays stable (deepening flows through the
 * omnibar `depth` clause → `growToDepth`, not a refetch). An empty seed renders a warning banner linking
 * to the dashboard builder; otherwise the composed provider stack + layout is rendered.
 *
 * @returns The dashboard page.
 */
const Dashboard: React.FC = () => {
  const [searchParams] = useSearchParams();
  // memoize the decoded seed on the *seed-key subset* of the query string; changing an omnibar clause or
  // the tab hash leaves this string unchanged so GraphDataProvider does not refetch
  const seedParamString = useMemo(() => {
    const subset = new URLSearchParams();
    for (const key of SEED_KEYS) {
      for (const value of searchParams.getAll(key)) {
        subset.append(key, value);
      }
    }
    return subset.toString();
  }, [searchParams]);
  const { seed, depth } = useMemo(() => decodeSeedParams(new URLSearchParams(seedParamString)), [seedParamString]);

  return (
    <Page title="Dashboard" className="full-min-width">
      {seedHasResources(seed) ? (
        // key on the seed identity so navigating between two /dashboard/view URLs (back/forward, in-app
        // links) remounts the content: each dashboard gets a fresh depth capture and depth-clause seeding
        // while the depth prop stays stable within a single dashboard's lifetime
        <DashboardContent key={seedParamString} seed={seed} depthAtMount={depth} />
      ) : (
        <AlertBanner severity={Severity.Warning}>
          This dashboard has no resources. <Link to="/dashboard/build">Build a dashboard</Link> to choose files, repos, entities, or tags to
          visualize.
        </AlertBanner>
      )}
    </Page>
  );
};

export default Dashboard;
