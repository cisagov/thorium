import React from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';
import { GoSidebarExpand, GoSidebarCollapse } from 'react-icons/go';

// project imports
import NodeInfo from '../shared/NodeInfo';
import EdgeInfo from '../shared/EdgeInfo';
import type { SelectedElement } from './controls/types';
import type { TreeNode } from '@models/trees';
import { PreviewContainer, PreviewHeader, PreviewToggleButton, MinimizeButton } from './Shared';

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
      <OverlayTrigger placement="left" overlay={<Tooltip>Show Node Info</Tooltip>}>
        <PreviewToggleButton onClick={onToggleMinimize}>
          <GoSidebarExpand size={14} />
        </PreviewToggleButton>
      </OverlayTrigger>
    );
  }

  return (
    <PreviewContainer>
      <PreviewHeader>
        <OverlayTrigger placement="left" overlay={<Tooltip>Minimize</Tooltip>}>
          <MinimizeButton onClick={onToggleMinimize}>
            <GoSidebarCollapse size={14} />
          </MinimizeButton>
        </OverlayTrigger>
      </PreviewHeader>
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
    </PreviewContainer>
  );
};

export default DataPreviewPanel;
