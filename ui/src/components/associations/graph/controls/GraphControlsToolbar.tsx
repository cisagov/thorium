import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FaCog, FaProjectDiagram, FaBolt, FaCamera, FaArrowRight } from 'react-icons/fa';
import { FaHexagonNodes } from 'react-icons/fa6';
import type { ForceGraph3DInstance } from '3d-force-graph';

import type { GraphControls, DisplayAction, SectionKey } from './types';
import { ToolbarContainer, ToolbarIconButton, NodeCount, ToolbarSpinner } from './Toolbar.styled';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import ScrollableSelect from '@components/shared/inputs/ScrollableSelect';
import ToolbarButton from './ToolbarButton';
import GraphSection from './GraphSection';
import ForcesSection from './ForcesSection';
import NodesSection from './NodesSection';
import EdgesSection from './EdgesSection';
import ExportSection from './ExportSection';

interface GraphControlsToolbarProps {
  graphId: string;
  controls: GraphControls;
  updateControls: React.ActionDispatch<[action: DisplayAction]>;
  graphInstance: ForceGraph3DInstance | null;
  nodeCount: number;
  loading: boolean;
}

const GraphControlsToolbar: React.FC<GraphControlsToolbarProps> = ({
  graphId,
  controls,
  updateControls,
  graphInstance,
  nodeCount,
  loading,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<SectionKey | null>(null);
  const toolbarRef = useRef<HTMLDivElement>(null);

  const handleToggleSection = (key: SectionKey) => {
    setActiveSection((prev) => (prev === key ? null : key));
  };

  const handleGearToggle = () => {
    if (isOpen) {
      setActiveSection(null);
    }
    setIsOpen((prev) => !prev);
  };

  const handleClickOutside = useCallback(
    (e: MouseEvent) => {
      if (activeSection && toolbarRef.current && !toolbarRef.current.contains(e.target as Node)) {
        const popover = (e.target as Element).closest('.popover');
        if (!popover) {
          setActiveSection(null);
        }
      }
    },
    [activeSection],
  );

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [handleClickOutside]);

  return (
    <ToolbarContainer ref={toolbarRef}>
      {isOpen ? (
        <ToolbarIconButton $active onClick={handleGearToggle}>
          <FaCog size={16} />
        </ToolbarIconButton>
      ) : (
        <OverlayTipTop tip="Controls">
          <ToolbarIconButton onClick={handleGearToggle}>
            <FaCog size={16} />
          </ToolbarIconButton>
        </OverlayTipTop>
      )}

      {isOpen && (
        <>
          <ToolbarButton
            sectionKey="graph"
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaProjectDiagram size={14} />}
            title="View"
          >
            <GraphSection graphId={graphId} controls={controls} updateControls={updateControls} graphInstance={graphInstance} />
          </ToolbarButton>

          <ToolbarButton
            sectionKey="forces"
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaBolt size={14} />}
            title="Forces"
          >
            <ForcesSection graphId={graphId} controls={controls} updateControls={updateControls} graphInstance={graphInstance} />
          </ToolbarButton>

          <ToolbarButton
            sectionKey="nodes"
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaHexagonNodes size={14} />}
            title="Nodes"
          >
            <NodesSection controls={controls} updateControls={updateControls} />
          </ToolbarButton>

          <ToolbarButton
            sectionKey="edges"
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaArrowRight size={14} style={{ transform: 'rotate(-45deg)' }} />}
            title="Edges"
          >
            <EdgesSection controls={controls} updateControls={updateControls} />
          </ToolbarButton>

          <ToolbarButton
            sectionKey="export"
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaCamera size={14} />}
            title="Export"
          >
            <ExportSection graphId={graphId} controls={controls} updateControls={updateControls} graphInstance={graphInstance} />
          </ToolbarButton>

          <ScrollableSelect value={controls.depth} onChange={(v) => updateControls({ type: 'depth', state: v })} min={1} windowSize={5} />
        </>
      )}

      <NodeCount>
        {loading && <ToolbarSpinner animation="border" size="sm" variant="secondary" />}
        Nodes: {nodeCount}
      </NodeCount>
    </ToolbarContainer>
  );
};

export default GraphControlsToolbar;
