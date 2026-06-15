import React from 'react';
import { GoSidebarExpand, GoSidebarCollapse } from 'react-icons/go';

// project imports
import NodeInfo from '../shared/NodeInfo';
import EdgeInfo from '../shared/EdgeInfo';
import type { SelectedElement } from './controls/types';
import { OverlayTipLeft } from '@components/shared/overlay/tips';
import type { TreeNode } from '@models/trees';
import { PreviewContainer, PreviewScroll, PreviewToggleButton, PreviewCollapseButton } from './Shared';

// spec: ./AssociationGraph.spec.md

interface DataPreviewPanelProps {
  selectedElement: SelectedElement | null;
  nodeData: TreeNode | undefined;
  minimized: boolean;
  onToggleMinimize: () => void;
}

const DataPreviewPanel: React.FC<DataPreviewPanelProps> = ({ selectedElement, nodeData, minimized, onToggleMinimize }) => {
  if (!selectedElement) return null;

  if (minimized) {
    return (
      <OverlayTipLeft tip="Show Node Info">
        <PreviewToggleButton onClick={onToggleMinimize}>
          <GoSidebarExpand size={14} />
        </PreviewToggleButton>
      </OverlayTipLeft>
    );
  }

  return (
    <PreviewContainer>
      {/* collapse button floats over the top-right corner, above the content, so it no longer takes a header row */}
      <OverlayTipLeft tip="Minimize">
        <PreviewCollapseButton onClick={onToggleMinimize}>
          <GoSidebarCollapse size={14} />
        </PreviewCollapseButton>
      </OverlayTipLeft>
      <PreviewScroll>
        {selectedElement.kind === 'node' && nodeData && <NodeInfo node={nodeData} />}
        {selectedElement.kind === 'link' && (
          <EdgeInfo
            edge={{
              data: {
                source: selectedElement.source,
                target: selectedElement.target,
                label: selectedElement.label,
              },
            }}
          />
        )}
      </PreviewScroll>
    </PreviewContainer>
  );
};

export default DataPreviewPanel;
