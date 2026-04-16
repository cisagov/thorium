import React from 'react';

import { TreeNode } from '@models/trees';
import { getNodePreviewData, renderTagPreview } from './treeHelpers';

interface NodePreviewContentProps {
  nodeData: TreeNode;
  isDuplicate?: boolean;
}

const NodePreviewContent: React.FC<NodePreviewContentProps> = ({ nodeData, isDuplicate }) => {
  const preview = getNodePreviewData(nodeData);
  return (
    <>
      <div className="preview-type">{preview.type}</div>
      {preview.fields.map(
        (f, i) =>
          f.value && (
            <div key={i} className="preview-field">
              {f.label && f.label !== '' && <strong>{f.label}: </strong>}
              {f.value}
              {f.label == undefined || (f.label === '' && <>{f.value}</>)}
            </div>
          ),
      )}
      {renderTagPreview(preview.tags)}
      {isDuplicate && <div className="preview-duplicate-warn">Duplicate: appears under multiple parents in this tree</div>}
    </>
  );
};

export default NodePreviewContent;
