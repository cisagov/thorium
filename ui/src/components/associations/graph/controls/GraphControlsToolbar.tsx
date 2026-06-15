import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FaCog, FaProjectDiagram, FaBolt, FaCamera, FaArrowRight } from 'react-icons/fa';
import { FaHexagonNodes } from 'react-icons/fa6';

// project imports
import EdgesSection from './EdgesSection';
import ExportSection from './ExportSection';
import ForcesSection from './ForcesSection';
import GraphSection from './GraphSection';
import NodesSection from './NodesSection';
import { ToolbarContainer, ToolbarIconButton, NodeCount, ToolbarSpinner } from './Toolbar.styled';
import ToolbarButton from './ToolbarButton';
import { SectionKey } from './types';
import type { GraphControls, DisplayAction } from './types';
import type { GraphInstance } from '../types';
import ScrollableSelect from '@components/shared/inputs/ScrollableSelect';
import { OverlayTipTop } from '@components/shared/overlay/tips';

// spec: ./GraphControlsToolbar.spec.md

interface GraphControlsToolbarProps {
  graphId: string;
  controls: GraphControls;
  updateControls: React.ActionDispatch<[action: DisplayAction]>;
  graphInstance: GraphInstance | null;
  nodeCount: number;
  loading: boolean;
  /** True while a grow/growToDepth is in flight — the depth control disables so rapid changes don't queue work. */
  growing: boolean;
}

const GraphControlsToolbar: React.FC<GraphControlsToolbarProps> = ({
  graphId,
  controls,
  updateControls,
  graphInstance,
  nodeCount,
  loading,
  growing,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<SectionKey | null>(null);
  const [depthMenuOpen, setDepthMenuOpen] = useState(false);
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
        <ToolbarIconButton $active aria-label="Toggle controls" onClick={handleGearToggle}>
          <FaCog size={16} />
        </ToolbarIconButton>
      ) : (
        <OverlayTipTop tip="Controls">
          <ToolbarIconButton aria-label="Toggle controls" onClick={handleGearToggle}>
            <FaCog size={16} />
          </ToolbarIconButton>
        </OverlayTipTop>
      )}

      {isOpen && (
        <>
          <ToolbarButton
            sectionKey={SectionKey.Graph}
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaProjectDiagram size={14} />}
            title="View"
          >
            <GraphSection
              graphId={graphId}
              controls={controls}
              updateControls={updateControls}
              graphInstance={graphInstance}
              nodeCount={nodeCount}
            />
          </ToolbarButton>

          <ToolbarButton
            sectionKey={SectionKey.Forces}
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaBolt size={14} />}
            title="Forces"
          >
            <ForcesSection
              graphId={graphId}
              controls={controls}
              updateControls={updateControls}
              graphInstance={graphInstance}
              nodeCount={nodeCount}
            />
          </ToolbarButton>

          <ToolbarButton
            sectionKey={SectionKey.Nodes}
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaHexagonNodes size={14} />}
            title="Nodes"
          >
            <NodesSection controls={controls} updateControls={updateControls} />
          </ToolbarButton>

          <ToolbarButton
            sectionKey={SectionKey.Edges}
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaArrowRight size={14} style={{ transform: 'rotate(-45deg)' }} />}
            title="Edges"
          >
            <EdgesSection controls={controls} updateControls={updateControls} />
          </ToolbarButton>

          <ToolbarButton
            sectionKey={SectionKey.Export}
            activeSection={activeSection}
            onToggle={handleToggleSection}
            icon={<FaCamera size={14} />}
            title="Export"
          >
            <ExportSection
              graphId={graphId}
              controls={controls}
              updateControls={updateControls}
              graphInstance={graphInstance}
              nodeCount={nodeCount}
            />
          </ToolbarButton>

          <OverlayTipTop tip={growing ? 'Growing…' : 'Depth'} disabled={depthMenuOpen}>
            <ScrollableSelect
              value={controls.depth}
              onChange={(v) => updateControls({ type: 'depth', state: v })}
              min={1}
              windowSize={5}
              onOpenChange={setDepthMenuOpen}
              disabled={growing}
            />
          </OverlayTipTop>
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
