// spec: ./EntityBrowser.spec.md
import React, { Fragment, ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { FaHouse } from 'react-icons/fa6';

// project imports
import BrowserToolbar from './BrowserToolbar';
import { DUPLICATE_HIGHLIGHT_CLASS } from './duplicateHighlight';
import EntityRow from './EntityRow';
import EntitySortControls from './EntitySortControls';
import EntityTreeLevel, { PAGE_SIZE } from './EntityTreeLevel';
import { EntityBrowserProvider, useEntityBrowser } from './EntityBrowserContext';
import { BrowserHeader, BrowserRoot, Crumb, CrumbSep, CurrentCrumb, FocusBar, ShowMoreButton, ShowMoreRow } from './EntityBrowser.styled';
import { EntityBrowserProps } from './types';
import { useGraphData } from '../../data/GraphDataContext';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';

// a shared empty path identity for root rows, so nested levels keyed on `path` don't recompute their children
// when a fresh `new Set()` would otherwise be passed on every parent render. Consumers only read it (EntityRow
// copies it before adding its own id), so a single shared instance is safe.
const EMPTY_PATH: Set<string> = new Set();

/**
 * The focus breadcrumb shown above the tree while it is re-rooted (focused) at a node: a clickable trail from
 * **All** (clears the focus, restoring the natural roots) down through the ancestors to the current focus root
 * (the bold, non-interactive last crumb). Clicking an ancestor pops the focus up to it. Renders nothing when
 * the tree is not focused.
 *
 * @returns The focus breadcrumb bar, or `null` when not focused.
 */
const FocusBreadcrumb: React.FC = () => {
  const { focusRoot, focusAncestors, setFocusRoot } = useEntityBrowser();
  if (!focusRoot || focusAncestors.length === 0) return null;
  const lastIndex = focusAncestors.length - 1;
  return (
    <FocusBar aria-label="Focused subtree">
      <Crumb type="button" onClick={() => setFocusRoot(null)} aria-label="Show all — clear the focus">
        <FaHouse size={11} aria-hidden /> All
      </Crumb>
      {focusAncestors.map((crumb, i) => (
        <Fragment key={crumb.id}>
          <CrumbSep aria-hidden>›</CrumbSep>
          {i === lastIndex ? (
            <CurrentCrumb title={crumb.label}>{crumb.label}</CurrentCrumb>
          ) : (
            <Crumb type="button" onClick={() => setFocusRoot(crumb.id)} title={crumb.label}>
              {crumb.label}
            </Crumb>
          )}
        </Fragment>
      ))}
    </FocusBar>
  );
};

/** Props for {@link EntityBrowserBody}. */
interface EntityBrowserBodyProps {
  /**
   * When true, each root is rendered as its own expandable row; when false the roots' children render directly
   * as the top level (file-details tab, where the file itself is implicit).
   */
  showRootNodes: boolean;
  /**
   * The toolbar node to render above the tree. Defaults to the built-in {@link BrowserToolbar}; pass a custom
   * strip to replace it, or `null` to render no toolbar (a dashboard composes its own controls outside the
   * body). This composition slot avoids a `showToolbar` boolean so callers can substitute a different toolbar
   * rather than only turning it off/on.
   */
  toolbar?: ReactNode;
  /**
   * Whether to render the browser's own header row with the {@link EntitySortControls}. Defaults to `true`
   * (file-details tab and the narrow/tabs dashboard layout). The ultra-wide dashboard sets this `false` and
   * renders the same controls in its "Entities" tile header instead, so they aren't shown twice.
   */
  showSortControls?: boolean;
}

/**
 * The rendered body of the browser (inside the provider): handles loading/empty/error states, then renders
 * the toolbar plus either the root rows or — when `showRootNodes` is false — the roots' children directly
 * (used by the file-details tab, where the file itself is implicit). Exported so a dashboard can compose its
 * own {@link EntityBrowserProvider} (with controlled state) directly around this body instead of nesting a
 * second provider via {@link EntityBrowser}.
 */
export const EntityBrowserBody: React.FC<EntityBrowserBodyProps> = ({
  showRootNodes,
  toolbar = <BrowserToolbar />,
  showSortControls = true,
}) => {
  const { loading, error, graphVersion } = useGraphData();
  const { roots, visibleSet, hiddenNodes, pinnedDuplicate } = useEntityBrowser();
  // paginate the root level the same way nested levels do, so a graph with thousands of seeds doesn't mount
  // thousands of root rows at once
  const [rootLimit, setRootLimit] = useState(PAGE_SIZE);
  // the browser root, so Escape can clear the pinned duplicate highlight scoped to this instance
  const rootRef = useRef<HTMLDivElement>(null);
  // Escape unpins the current duplicate highlight: drop the pin and strip the class from every occurrence in
  // this browser (a later hover re-applies transient highlighting as normal)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || !pinnedDuplicate.current) return;
      pinnedDuplicate.current = null;
      rootRef.current?.querySelectorAll(`.${DUPLICATE_HIGHLIGHT_CLASS}`).forEach((el) => el.classList.remove(DUPLICATE_HIGHLIGHT_CLASS));
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [pinnedDuplicate]);

  // stable path set per root so nested levels don't recompute their children every render
  const rootPaths = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const root of roots) map.set(root.id, new Set([root.id]));
    return map;
  }, [roots]);

  if (error) {
    return <AlertBanner severity={Severity.Error}>{error}</AlertBanner>;
  }
  // graphVersion is 0 until the shared provider's initial fetch resolves
  if (loading || graphVersion === 0) {
    return <LoadingSpinner loading={true} />;
  }
  if (roots.length === 0) {
    return <AlertBanner severity={Severity.Info}>No Associated Entities</AlertBanner>;
  }

  // drop hidden roots here (the effectiveChildren short-circuit only prunes hidden *children* within a subtree)
  const rootsShown = roots.filter((r) => !hiddenNodes.has(r.id));
  const shownRoots = visibleSet ? rootsShown.filter((r) => visibleSet.has(r.id)) : rootsShown;
  const visibleRoots = shownRoots.slice(0, rootLimit);
  const remainingRoots = shownRoots.length - visibleRoots.length;

  return (
    <BrowserRoot ref={rootRef} data-testid="entity-browser">
      {toolbar}
      {showSortControls && (
        <BrowserHeader>
          <EntitySortControls />
        </BrowserHeader>
      )}
      <FocusBreadcrumb />
      {shownRoots.length === 0 ? (
        <AlertBanner severity={Severity.Info}>No matching items.</AlertBanner>
      ) : showRootNodes ? (
        visibleRoots.map((root) => <EntityRow key={root.id} nodeId={root.id} rowKey={`root:${root.id}`} depth={0} path={EMPTY_PATH} />)
      ) : (
        // file-tab mode: the file is implicit — render its associations directly as the top level
        visibleRoots.map((root) => (
          <EntityTreeLevel
            key={root.id}
            parentId={root.id}
            path={rootPaths.get(root.id) ?? new Set([root.id])}
            depth={0}
            rowKeyPrefix={`root:${root.id}`}
          />
        ))
      )}
      {remainingRoots > 0 && (
        <ShowMoreRow>
          <ShowMoreButton onClick={() => setRootLimit((l) => l + PAGE_SIZE)}>Show more ({remainingRoots} remaining)</ShowMoreButton>
        </ShowMoreRow>
      )}
    </BrowserRoot>
  );
};

/**
 * A generic, graph-driven browser of associated entities/files/repos/tags for a starting context. Reads the
 * shared {@link useGraphData} graph (so no extra fetch — growth converges with the association graph), and
 * lets the user filter, control per-layer visibility (show / pass-through / skip), expand into associations,
 * and inspect full metadata + tags inline.
 */
const EntityBrowser: React.FC<EntityBrowserProps> = ({
  roots,
  inView = true,
  defaultPolicies,
  fallbackPolicy,
  showRootNodes = true,
  defaultDepth,
}) => {
  if (inView === false) {
    return null;
  }
  return (
    <EntityBrowserProvider roots={roots} defaultPolicies={defaultPolicies} fallbackPolicy={fallbackPolicy} defaultDepth={defaultDepth}>
      <EntityBrowserBody showRootNodes={showRootNodes} />
    </EntityBrowserProvider>
  );
};

export default EntityBrowser;
