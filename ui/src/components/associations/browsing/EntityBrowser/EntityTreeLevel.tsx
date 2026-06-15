// spec: ./EntityBrowser.spec.md
import React, { Fragment, useMemo, useState } from 'react';

// project imports
import { compareByFlagStats, effectiveChildren, groupByKind } from './browserHelpers';
import EntityRow from './EntityRow';
import LayerHeader from './LayerHeader';
import { useEntityBrowser } from './EntityBrowserContext';
import { Level, ShowMoreButton, ShowMoreRow } from './EntityBrowser.styled';
import { useGraphData } from '../../data/GraphDataContext';
import { NodeType } from '@models/trees';

// how many rows to render per level before a "show more" control (keeps noisy graphs responsive)
export const PAGE_SIZE = 25;

interface EntityTreeLevelProps {
  parentId: string;
  /** Ancestor node ids on the path to (and including) `parentId` — for the cycle guard. */
  path: Set<string>;
  depth: number;
  /** The parent row's key; child row keys are derived from it to stay path-unique. */
  rowKeyPrefix: string;
  /** Arrival context of `parentId`: how many reversed (against-direction) hops preceded it. */
  reverseDepth?: number;
  /** Arrival context of `parentId`: whether it was itself reached via a reversed edge. */
  viaReversed?: boolean;
}

/**
 * Renders one parent's effective children: policy-resolved (Skip/PassThrough/Show), optionally filtered,
 * grouped by kind under {@link LayerHeader}s (only when a header adds value), and paginated per level.
 */
const EntityTreeLevel: React.FC<EntityTreeLevelProps> = ({
  parentId,
  path,
  depth,
  rowKeyPrefix,
  reverseDepth = 0,
  viaReversed = false,
}) => {
  const { graph } = useGraphData();
  const browser = useEntityBrowser();
  const [limit, setLimit] = useState(PAGE_SIZE);

  const children = useMemo(
    () => effectiveChildren(parentId, browser.index, graph, browser.traversalConfig, path, reverseDepth, viaReversed),
    [browser.index, browser.traversalConfig, parentId, path, graph, reverseDepth, viaReversed],
  );

  const filtered = browser.visibleSet ? children.filter((c) => browser.visibleSet!.has(c.edge.id)) : children;

  // a level with no children to show renders nothing: rows with no associations aren't expandable (the row's
  // chevron is suppressed), so an expanded-into-nothing empty note would only add noise
  if (filtered.length === 0) {
    return null;
  }

  // sort this level's rows by the selected flag-stat mode (descending), tiebroken through the priority order,
  // reading O(1) from the precomputed flag stats. Grouping preserves this order, so groups end up ordered by
  // their top member and rows within a group are ordered too.
  const sorted = [...filtered].sort((a, b) => compareByFlagStats(a.edge.id, b.edge.id, browser.flagStats, browser.sortMode));
  // group by kind (with layer headers) unless the user turned grouping off — then render one flat sorted list
  const groups = browser.groupByResource ? groupByKind(sorted, graph) : [{ nodeType: 'all' as NodeType, children: sorted }];
  // a header earns its row only when grouping is on AND it disambiguates (multiple kinds) or summarizes (>1)
  const showHeaders = browser.groupByResource && (groups.length > 1 || sorted.length > 1);

  // paginate across the flattened child list while keeping group boundaries
  let rendered = 0;
  const groupEls: React.ReactNode[] = [];
  for (const group of groups) {
    if (rendered >= limit) break;
    const slice = group.children.slice(0, limit - rendered);
    rendered += slice.length;
    groupEls.push(
      <Fragment key={group.nodeType}>
        {showHeaders && <LayerHeader nodeType={group.nodeType} groupChildren={group.children} rowKeyPrefix={rowKeyPrefix} />}
        {slice.map((child) => (
          <EntityRow
            key={`${rowKeyPrefix}/${child.edge.id}`}
            nodeId={child.edge.id}
            edge={child.edge}
            breadcrumb={child.breadcrumb}
            rowKey={`${rowKeyPrefix}/${child.edge.id}`}
            path={path}
            depth={depth}
            reverseDepth={child.reverseDepth ?? 0}
            viaReversed={child.viaReversed ?? false}
          />
        ))}
      </Fragment>,
    );
  }

  const remaining = filtered.length - rendered;

  return (
    <Level $depth={depth}>
      {groupEls}
      {remaining > 0 && (
        <ShowMoreRow>
          <ShowMoreButton onClick={() => setLimit((l) => l + PAGE_SIZE)}>Show more ({remaining} remaining)</ShowMoreButton>
        </ShowMoreRow>
      )}
    </Level>
  );
};

// memoized so an expand/collapse elsewhere in the tree doesn't re-render every level; the level re-renders on
// its own prop changes (parentId/path/depth/context) and its local "show more" state
export default React.memo(EntityTreeLevel);
