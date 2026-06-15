// spec: ./EntityBrowser.spec.md
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FaBullseye, FaChevronRight, FaEyeSlash, FaFlag, FaGear, FaSeedling, FaTags } from 'react-icons/fa6';

// project imports
import { effectiveChildren, getDisplayTags, nodeTypeOf } from './browserHelpers';
import {
  applyHighlight,
  clearHighlight,
  collectOccurrences,
  DUPLICATE_HIGHLIGHT_CLASS,
  scrollToNextOccurrence,
} from './duplicateHighlight';
import EntityTreeLevel from './EntityTreeLevel';
import MetadataBox from './MetadataBox';
import { useEntityBrowser } from './EntityBrowserContext';
import {
  BadgeDivider,
  BadgeGroup,
  BadgeMetric,
  Chevron,
  DepthPill,
  DuplicateBadge,
  DuplicateGroupNumber,
  FlagBadge,
  FocusButton,
  GrowBadge,
  HeaderChips,
  HeaderIdentity,
  HeaderLead,
  HeaderTrail,
  HideButton,
  HideSlot,
  INDENT_CAP,
  IdentifierLink,
  IdentifierText,
  InfoBox,
  KindBadge,
  ReRootButton,
  RelationshipBadge,
  RowContainer,
  RowHeader,
  RowSpinner,
  TagBadge,
  TagBadgeGroup,
  TagBadgeKey,
  TagBadgeValue,
  TagOverflowBadge,
  ViaBadge,
} from './EntityBrowser.styled';
import { hasContextualDisplayChildren, toDisplayCfg, TreeEdge } from '../treeHelpers';
import { FocusSource, useGraphData } from '../../data/GraphDataContext';
import { getNodeName } from '../../utilities';
import EntityTypeIcon from '@components/entities/shared/EntityTypeIcon';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import EntitySummaryHover from '@components/shared/info/EntitySummaryHover';
import { treeNodeToInfo } from '@components/shared/info/info';
import { Entities, entityLabel } from '@models/entities';
import { TreeNodeKey } from '@models/trees';

interface EntityRowProps {
  nodeId: string;
  /** Unique per occurrence (path-derived) so DAG duplicates expand independently. */
  rowKey: string;
  /** Ancestor node ids on the path to (and including) this row's parent — for the children's cycle guard. */
  path: Set<string>;
  depth: number;
  /** The incoming edge (for the relationship badge); omitted for roots. */
  edge?: TreeEdge;
  /** Names of pass-through layers elided above this row. */
  breadcrumb?: string[];
  /** Arrival context: how many reversed (against-direction) hops preceded this row. */
  reverseDepth?: number;
  /** Arrival context: whether this row was itself reached via a reversed edge. */
  viaReversed?: boolean;
}

/**
 * A single node as an "info box": a header (clicking it expands the node's associated CHILDREN — and lazily
 * grows the shared graph once if growable) plus, directly under it, a condensed "details" caret that reveals
 * the node's metadata on demand. Clicking the name navigates to that resource's details page. Nested
 * associations render below the box when expanded.
 */
const EntityRow: React.FC<EntityRowProps> = ({ nodeId, rowKey, path, depth, edge, breadcrumb, reverseDepth = 0, viaReversed = false }) => {
  const { graph, growable, grow, setFocusedNode } = useGraphData();
  const browser = useEntityBrowser();
  const [growing, setGrowing] = useState(false);
  // the "details" body's expanded state is owned here (not inside MetadataBox) so the header's hover preview
  // can be suppressed while the pinned details are open — the two would otherwise show the same summary twice
  const [detailsExpanded, setDetailsExpanded] = useState(false);

  const node = graph.data_map[nodeId];
  const info = useMemo(() => (node ? treeNodeToInfo(node) : null), [node]);
  // descriptive tag chips for the header (capped, noise keys dropped); recomputed only when the node changes
  const displayTags = useMemo(() => (node ? getDisplayTags(node) : { shown: [], overflow: 0, overflowLabels: [] }), [node]);
  const nodeType = nodeTypeOf(nodeId, graph);
  const childrenExpanded = browser.isChildrenExpanded(rowKey, nodeId, viaReversed, reverseDepth);
  const isDuplicate = browser.multiParent.has(nodeId);
  // ephemeral correlation number shown on the badge so occurrences of the same node visibly match
  const duplicateGroupNumber = isDuplicate ? browser.duplicateGroupIds.get(nodeId) : undefined;
  // the header element, so the mount effect can re-apply a pinned highlight after a remount (e.g. re-root)
  const rowHeaderRef = useRef<HTMLDivElement>(null);
  const isGrowable = growable.has(nodeId);
  // flag entities and danger-tag pairs within this node's subtree, read O(1) from the precomputed stats (no crawl)
  const flagCount = browser.flagStats.get(nodeId)?.flags ?? 0;
  const dangerTagCount = browser.flagStats.get(nodeId)?.dangerTags ?? 0;
  // combined tooltip naming whichever counts are present (the badge only renders when at least one is > 0)
  const significanceTitle = `${[
    flagCount > 0 ? `${flagCount} flag${flagCount === 1 ? '' : 's'}` : '',
    dangerTagCount > 0 ? `${dangerTagCount} danger tag${dangerTagCount === 1 ? '' : 's'}` : '',
  ]
    .filter(Boolean)
    .join(', ')} within this branch`;
  // expandable when growable OR it has any display child in THIS arrival context (forward children, plus
  // reverse relationship edges bounded by the reverse-depth rules). When some children are hidden we fall back
  // to effectiveChildren (which drops hidden subtrees) so a parent whose only children are hidden shows no
  // chevron; the cheap contextual check suffices when nothing is hidden.
  const hasHidden = browser.hiddenNodes.size > 0;
  const canExpand = useMemo(() => {
    if (isGrowable) return true;
    if (!hasHidden) {
      return hasContextualDisplayChildren(browser.index, nodeId, toDisplayCfg(browser.traversalConfig), viaReversed, reverseDepth);
    }
    return (
      effectiveChildren(nodeId, browser.index, graph, browser.traversalConfig, new Set([nodeId]), reverseDepth, viaReversed).length > 0
    );
  }, [isGrowable, hasHidden, browser.index, browser.traversalConfig, nodeId, graph, viaReversed, reverseDepth]);

  const pathWithSelf = useMemo(() => {
    const next = new Set(path);
    next.add(nodeId);
    return next;
  }, [path, nodeId]);

  const onToggleChildren = () => {
    const willExpand = !childrenExpanded;
    browser.setChildrenExpanded(rowKey, willExpand);
    // selecting a row drives the side-by-side association graph to focus this node (see spec: row → graph focus)
    setFocusedNode(nodeId, FocusSource.Tree);
    // grow the shared graph once when first descending into a growable node
    if (willExpand && isGrowable && !browser.grownNodes.has(nodeId)) {
      browser.grownNodes.add(nodeId);
      setGrowing(true);
      void grow(nodeId).finally(() => setGrowing(false));
    }
  };
  // hide this node and its whole subtree from the tree (entities view only); stop propagation so the header's
  // toggle/focus doesn't also fire
  const onHide = (e: React.MouseEvent) => {
    e.stopPropagation();
    browser.hideNode(nodeId);
  };
  // focus (bullseye) the tree at this node — prune to its subtree; stop propagation so the header's toggle
  // doesn't also fire
  const onFocus = (e: React.MouseEvent) => {
    e.stopPropagation();
    browser.setFocusRoot(nodeId);
  };
  // re-root (gear) the whole view at this node — reorder around it, keeping every node visible; stop
  // propagation so the header's toggle doesn't also fire
  const onReRoot = (e: React.MouseEvent) => {
    e.stopPropagation();
    browser.setReRoot(nodeId);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onToggleChildren();
    }
  };
  // hovering the badge lights up every visible occurrence of this node, and writes the "occurrence N of M"
  // count (derived from the same DOM query) onto the badge so a screen reader / tooltip reports position
  const onDuplicateEnter = (e: React.MouseEvent<HTMLElement>) => {
    const badge = e.currentTarget;
    const list = collectOccurrences(badge, nodeId);
    applyHighlight(list);
    const own = badge.closest('[data-node-id]');
    const position = own ? list.indexOf(own as HTMLElement) + 1 : 0;
    const label =
      list.length > 1
        ? `Duplicate group ${duplicateGroupNumber} — visible occurrence ${position} of ${list.length}. Click to pin the highlight and jump to the next occurrence.`
        : `Duplicate group ${duplicateGroupNumber} — appears under multiple parents in this tree.`;
    badge.title = label;
    badge.setAttribute('aria-label', label);
  };
  // clear the transient highlight on leave — unless THIS node is the pinned one (its highlight must persist)
  const onDuplicateLeave = (e: React.MouseEvent<HTMLElement>) => {
    if (browser.pinnedDuplicate.current === nodeId) return;
    clearHighlight(collectOccurrences(e.currentTarget, nodeId));
  };
  // pin this group's highlight (moving the pin off any other group) and jump to the next occurrence, wrapping
  // to the first after the last. stopPropagation keeps the row's expand toggle from also firing.
  const activateDuplicate = (badge: HTMLElement) => {
    const prev = browser.pinnedDuplicate.current;
    if (prev && prev !== nodeId) {
      clearHighlight(collectOccurrences(badge, prev));
    }
    browser.pinnedDuplicate.current = nodeId;
    const list = collectOccurrences(badge, nodeId);
    applyHighlight(list);
    const own = badge.closest('[data-node-id]');
    if (own) scrollToNextOccurrence(list, own as HTMLElement);
  };
  const onDuplicateClick = (e: React.MouseEvent<HTMLElement>) => {
    e.stopPropagation();
    activateDuplicate(e.currentTarget);
  };
  // the row header's onKeyDown swallows Enter/Space from descendants (and preventDefaults the native button
  // click), so the badge must handle — and stop — its own key activation here
  const onDuplicateKeyDown = (e: React.KeyboardEvent<HTMLElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      e.stopPropagation();
      activateDuplicate(e.currentTarget);
    }
  };
  // a row mounted (or re-mounted, e.g. after a re-root rebuilds row keys) while its node is the pinned duplicate
  // re-applies the highlight class the imperative helpers can't reach a not-yet-mounted element with
  useEffect(() => {
    if (isDuplicate && browser.pinnedDuplicate.current === nodeId) {
      rowHeaderRef.current?.classList.add(DUPLICATE_HIGHLIGHT_CLASS);
    }
  }, [isDuplicate, nodeId, browser.pinnedDuplicate]);

  // getNodeName returns '' for unnamed nodes, which `??` would keep; fall through with `||` to the node id
  const title = info?.title || (node ? getNodeName(node, 80) : '') || nodeId;

  // the name block (icon + identifier). The hover summary preview wraps ONLY this — not the whole header — so
  // hovering the trailing focus/hide affordances (or the badges) doesn't pop the overlay in the way.
  const identity = (
    <HeaderIdentity>
      <EntityTypeIcon kind={nodeType as Entities} size={14} />
      {info?.titleHref ? (
        <IdentifierLink href={info.titleHref} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
          {title}
        </IdentifierLink>
      ) : (
        <IdentifierText>{title}</IdentifierText>
      )}
    </HeaderIdentity>
  );

  // the collapsed header row
  const header = (
    <RowHeader
      ref={rowHeaderRef}
      role="button"
      tabIndex={0}
      data-testid="entity-row"
      data-node-id={nodeId}
      aria-expanded={childrenExpanded}
      onClick={onToggleChildren}
      onKeyDown={onKeyDown}
    >
      <Chevron $expanded={childrenExpanded} $hidden={!canExpand}>
        <FaChevronRight size={10} />
      </Chevron>
      <HeaderLead>
        {/* hovering the name shows the summary preview to its right; suppressed while the details body is open
            (it already shows the same summary) or when the node has no describable info */}
        {info && !detailsExpanded ? (
          <EntitySummaryHover model={info} duplicate={isDuplicate} placement="right">
            {identity}
          </EntitySummaryHover>
        ) : (
          identity
        )}
        <HeaderChips>
          <BadgeGroup>
            <KindBadge>{entityLabel(nodeType)}</KindBadge>
            {edge && (
              <RelationshipBadge title={edge.containerLabel ? `${edge.label} ${edge.containerLabel}` : edge.label}>
                {edge.label}
                {edge.containerLabel ? ` ${edge.containerLabel}` : ''}
              </RelationshipBadge>
            )}
            {breadcrumb && breadcrumb.length > 0 && (
              <ViaBadge title={`Reached through: ${breadcrumb.join(' › ')}`}>via {breadcrumb.join(' › ')}</ViaBadge>
            )}
            {(flagCount > 0 || dangerTagCount > 0) && (
              <FlagBadge title={significanceTitle}>
                {flagCount > 0 && (
                  <BadgeMetric>
                    <FaFlag size={9} aria-hidden /> {flagCount}
                  </BadgeMetric>
                )}
                {flagCount > 0 && dangerTagCount > 0 && <BadgeDivider aria-hidden />}
                {dangerTagCount > 0 && (
                  <BadgeMetric>
                    <FaTags size={9} aria-hidden /> {dangerTagCount}
                  </BadgeMetric>
                )}
              </FlagBadge>
            )}
            {isDuplicate && (
              <DuplicateBadge
                as="button"
                type="button"
                title={`Duplicate group ${duplicateGroupNumber} — appears under multiple parents in this tree. Click to highlight all occurrences and jump to the next.`}
                onClick={onDuplicateClick}
                onKeyDown={onDuplicateKeyDown}
                onMouseEnter={onDuplicateEnter}
                onMouseLeave={onDuplicateLeave}
              >
                Duplicate
                {duplicateGroupNumber !== undefined && <DuplicateGroupNumber>·{duplicateGroupNumber}</DuplicateGroupNumber>}
              </DuplicateBadge>
            )}
            {isGrowable && (
              <GrowBadge title="More associations can be loaded — expand to grow">
                <FaSeedling size={10} /> grow
              </GrowBadge>
            )}
          </BadgeGroup>
          {displayTags.shown.length > 0 && (
            <TagBadgeGroup>
              {displayTags.shown.map((tag) => (
                <TagBadge key={`${tag.key}:${tag.value}`} title={tag.value ? `${tag.label}: ${tag.value}` : tag.label}>
                  <TagBadgeKey>{tag.label}</TagBadgeKey>
                  {/* skip the value span (and its gap) for a value-less tag so it can't show a trailing gap */}
                  {tag.value && <TagBadgeValue>{tag.value}</TagBadgeValue>}
                </TagBadge>
              ))}
              {displayTags.overflow > 0 && (
                <TagOverflowBadge title={displayTags.overflowLabels.join('\n')}>+{displayTags.overflow}</TagOverflowBadge>
              )}
            </TagBadgeGroup>
          )}
        </HeaderChips>
      </HeaderLead>
      <HeaderTrail>
        {growing && <RowSpinner aria-label="loading" />}
        {/* focus (re-root) affordance: hover-revealed on shallow rows; once nesting passes the indent cap
                it promotes to an always-visible depth pill (the pill doubles as the focus trigger). Only shown
                when the row actually has a subtree to focus into. */}
        {canExpand &&
          (depth > INDENT_CAP ? (
            <OverlayTipTop tip="Focus on this subtree (reset the nesting here)">
              <DepthPill type="button" aria-label={`Focus on ${title} — nested ${depth} levels deep`} onClick={onFocus}>
                <FaBullseye size={10} aria-hidden /> {depth}
              </DepthPill>
            </OverlayTipTop>
          ) : (
            <OverlayTipTop tip="Focus on this subtree">
              <FocusButton type="button" aria-label={`Focus on ${title} — show only this subtree`} onClick={onFocus}>
                <FaBullseye size={13} />
              </FocusButton>
            </OverlayTipTop>
          ))}
        {/* re-root (gear) affordance: reorder the whole view around this node, keeping every node visible
                (distinct from focus, which prunes to a subtree). Hidden on the node that is already the active
                re-root, where it would be a no-op. Available on every other row, including leaves. */}
        {browser.reRoot !== nodeId && (
          <OverlayTipTop tip="Re-root the view here (reorder around this node)">
            <ReRootButton type="button" aria-label={`Re-root the view at ${title}`} onClick={onReRoot}>
              <FaGear size={13} />
            </ReRootButton>
          </OverlayTipTop>
        )}
        <HideSlot>
          <OverlayTipTop tip="Exclude this item and everything under it">
            <HideButton type="button" aria-label={`Exclude ${title} and everything under it`} onClick={onHide}>
              <FaEyeSlash size={13} />
            </HideButton>
          </OverlayTipTop>
        </HideSlot>
      </HeaderTrail>
    </RowHeader>
  );

  return (
    <RowContainer>
      <InfoBox data-testid="entity-infobox">
        {header}
        {/* pass the entity id for entity nodes so the box lazily fetches the full record (rich content the
            graph node omits); File/Repo/Tag nodes carry everything already, so no id → no fetch */}
        {info && (
          <MetadataBox
            model={info}
            entityId={node?.[TreeNodeKey.Entity]?.id}
            expanded={detailsExpanded}
            onExpandedChange={setDetailsExpanded}
          />
        )}
      </InfoBox>
      {childrenExpanded && (
        <EntityTreeLevel
          parentId={nodeId}
          path={pathWithSelf}
          depth={depth + 1}
          rowKeyPrefix={rowKey}
          reverseDepth={reverseDepth}
          viaReversed={viaReversed}
        />
      )}
    </RowContainer>
  );
};

// memoized so a parent re-render (e.g. an unrelated expand elsewhere in the tree) doesn't re-render every row;
// the row still re-renders when its own props change or the context values it reads update
export default React.memo(EntityRow);
