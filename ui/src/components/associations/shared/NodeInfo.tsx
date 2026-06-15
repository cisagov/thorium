import React from 'react';

// project imports
import EntitySummary, { SummaryVariant } from '@components/shared/info/EntitySummary';
import { treeNodeToInfo } from '@components/shared/info/info';
import { TreeNode } from '@models/trees';

type NodeInfoProps = {
  // arbitrary Thorium node data
  node: TreeNode;
};

/**
 * Graph side-panel info for a tree node. Thin wrapper over the shared {@link EntitySummary} renderer
 * (fed by `treeNodeToInfo`) in its expanded variant, so entities/files/repos render consistently in
 * the panel and the hover overlays. Falls back to raw JSON for unrecognized node shapes.
 */
const NodeInfo: React.FC<NodeInfoProps> = ({ node }) => {
  if (Object.keys(node).length === 0) {
    return <></>;
  }
  const model = treeNodeToInfo(node);
  if (!model) {
    return <div className="m-2">{JSON.stringify(node, null, 2)}</div>;
  }
  return <EntitySummary model={model} variant={SummaryVariant.Expanded} />;
};

export default NodeInfo;
