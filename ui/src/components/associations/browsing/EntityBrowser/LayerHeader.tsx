// spec: ./EntityBrowser.spec.md
import React from 'react';
import { BsArrowsCollapse, BsArrowsExpand } from 'react-icons/bs';
import { FaEyeSlash } from 'react-icons/fa6';

// project imports
import { getNodeTags, hasDangerTags } from './browserHelpers';
import { AggBadge, GroupCount, GroupHeaderActions, GroupHeaderButton, GroupHeaderRow } from './EntityBrowser.styled';
import { useEntityBrowser } from './EntityBrowserContext';
import { EffectiveChild } from './types';
import { hasContextualDisplayChildren, toDisplayCfg } from '../treeHelpers';
import EntityTypeIcon from '@components/entities/shared/EntityTypeIcon';
import { useGraphData } from '../../data/GraphDataContext';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { ClauseCondition, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { getStringFieldListFromClauses } from '@components/shared/inputs/omnibar/utils';
import { bucketTags, countTagValues } from '@components/tags/utilities';
import { Entities, entityLabel } from '@models/entities';
import { NodeType } from '@models/trees';

interface LayerHeaderProps {
  nodeType: NodeType;
  /** All children in this kind group (full group, even when the level paginates the rows below). */
  groupChildren: EffectiveChild[];
  /** The parent level's row-key prefix, so this header can address its group's rows (`<prefix>/<id>`). */
  rowKeyPrefix: string;
}

/**
 * Group header for a layer (node kind) at one tree level: the kind icon + label + member count, plus
 * aggregate significance badges (how many members carry danger tags, and total ATT&CK / MBC techniques). A
 * trailing `+` on the count flags that some members are growable-but-ungrown, so the count is a floor. Its
 * trailing action cluster carries a **collapse/expand-this-subsection** control (toggles every expandable
 * row in the group at once) and an **exclude-this-kind** control (appends an `Exclude` clause; labelled
 * "Exclude" rather than "Hide" since it maps to the omnibar `Exclude` verb, distinct from the per-node `Hide`).
 */
const LayerHeader: React.FC<LayerHeaderProps> = ({ nodeType, groupChildren, rowKeyPrefix }) => {
  const { graph, growable } = useGraphData();
  const { clauses, setClauses, index, traversalConfig, isChildrenExpanded, setManyChildrenExpanded } = useEntityBrowser();
  const label = entityLabel(nodeType);
  // exclude this whole kind by appending an `Exclude is <kind>` clause (kind-level Skip: prune the category and
  // its subtrees). A single-value clause so it renders as its own removable omnibar tile; deduped so clicking
  // twice doesn't stack, and a no-op when the kind is already excluded.
  const alreadyExcluded = getStringFieldListFromClauses(clauses, 'Exclude').includes(nodeType);
  const onExcludeKind = () => {
    if (alreadyExcluded) return;
    const clause: Clause = { category: 'Exclude', field: 'Exclude', condition: ClauseCondition.Is, value: { value: nodeType } };
    setClauses([...clauses, clause]);
  };
  // the rows in this group that actually have a subtree to collapse/expand (a leaf with no children/grow has
  // nothing to toggle, so excluding it keeps the control's state honest and avoids expanding empty rows)
  const displayCfg = toDisplayCfg(traversalConfig);
  const expandable = groupChildren.filter(
    (child) =>
      growable.has(child.edge.id) ||
      hasContextualDisplayChildren(index, child.edge.id, displayCfg, child.viaReversed ?? false, child.reverseDepth ?? 0),
  );
  const rowKeys = expandable.map((child) => `${rowKeyPrefix}/${child.edge.id}`);
  // the section reads as "expanded" when any of its expandable rows is currently expanded; the button then
  // collapses them all (and vice-versa)
  const anyExpanded = expandable.some((child) =>
    isChildrenExpanded(`${rowKeyPrefix}/${child.edge.id}`, child.edge.id, child.viaReversed ?? false, child.reverseDepth ?? 0),
  );

  let dangerNodes = 0;
  let attack = 0;
  let mbc = 0;
  let anyGrowable = false;
  for (const child of groupChildren) {
    if (growable.has(child.edge.id)) anyGrowable = true;
    const node = graph.data_map[child.edge.id];
    if (!node) continue;
    const buckets = bucketTags(getNodeTags(node));
    if (hasDangerTags(getNodeTags(node))) dangerNodes += 1;
    attack += countTagValues(buckets.attack);
    mbc += countTagValues(buckets.mbc);
  }

  return (
    <GroupHeaderRow>
      <EntityTypeIcon kind={nodeType as Entities} size={13} />
      <span>{label}</span>
      <GroupCount>
        {groupChildren.length}
        {anyGrowable ? '+' : ''}
      </GroupCount>
      {dangerNodes > 0 && (
        <AggBadge $danger title={`${dangerNodes} with danger-classified tags`}>
          ⚠ {dangerNodes}
        </AggBadge>
      )}
      {attack > 0 && <AggBadge title="ATT&CK techniques">ATT&CK {attack}</AggBadge>}
      {mbc > 0 && <AggBadge title="MBC behaviors">MBC {mbc}</AggBadge>}
      <GroupHeaderActions>
        {rowKeys.length > 0 && (
          <OverlayTipTop tip={anyExpanded ? 'Collapse this section' : 'Expand this section'}>
            <GroupHeaderButton
              type="button"
              aria-label={anyExpanded ? `Collapse all ${label} items` : `Expand all ${label} items`}
              onClick={() => setManyChildrenExpanded(rowKeys, !anyExpanded)}
            >
              {anyExpanded ? <BsArrowsCollapse size={13} /> : <BsArrowsExpand size={13} />}
            </GroupHeaderButton>
          </OverlayTipTop>
        )}
        {!alreadyExcluded && (
          <OverlayTipTop tip={`Exclude all ${label} items`}>
            <GroupHeaderButton type="button" aria-label={`Exclude all ${label} items`} onClick={onExcludeKind}>
              <FaEyeSlash size={12} />
            </GroupHeaderButton>
          </OverlayTipTop>
        )}
      </GroupHeaderActions>
    </GroupHeaderRow>
  );
};

export default LayerHeader;
