import React from 'react';
import { Dropdown, Form } from 'react-bootstrap';

// project imports
import { NodeRenderMode } from './types';
import type { SectionProps } from './types';
import { PopoverBody, Divider, MenuList, MenuDropdown } from './Toolbar.styled';
import LabeledRange from './LabeledRange';
import { LABEL_MAX_PX, LABEL_MIN_PX, LABEL_TARGET_PX } from '../labelScale';

// spec: ./GraphControlsToolbar.spec.md

const NODE_STYLE_LABELS: Record<string, string> = {
  spheres: 'Spheres',
  icons: 'Icons',
};

const NodesSection: React.FC<SectionProps> = ({ controls, updateControls }) => (
  <PopoverBody>
    <MenuList $inset>
      <MenuDropdown>
        <Dropdown.Toggle variant="" size="sm" id="node-style-dropdown">
          {NODE_STYLE_LABELS[controls.nodeRenderMode]}
        </Dropdown.Toggle>
        <Dropdown.Menu>
          <Dropdown.Item onClick={() => updateControls({ type: 'nodeRenderMode', state: NodeRenderMode.Spheres })}>Spheres</Dropdown.Item>
          <Dropdown.Item onClick={() => updateControls({ type: 'nodeRenderMode', state: NodeRenderMode.Icons })}>Icons</Dropdown.Item>
        </Dropdown.Menu>
      </MenuDropdown>
    </MenuList>

    <LabeledRange
      id="form-node-size"
      label="Size"
      value={controls.nodeRelSize}
      min={1}
      max={20}
      step={1}
      onChange={(v) => updateControls({ type: 'nodeRelSize', state: v })}
    />
    <LabeledRange
      id="form-node-opacity"
      label="Opacity"
      value={controls.nodeOpacity}
      min={0.1}
      max={1.0}
      step={0.05}
      onChange={(v) => updateControls({ type: 'nodeOpacity', state: v })}
    />

    <Divider />

    <Form.Check
      type="switch"
      id="form-show-node-labels"
      label="Labels"
      checked={controls.showNodeLabels}
      onChange={() => updateControls({ type: 'showNodeLabels', state: !controls.showNodeLabels })}
    />
    {controls.showNodeLabels && (
      <>
        <LabeledRange
          id="form-node-label-scale"
          label="Label Size"
          value={controls.nodeLabelScale}
          min={0.5}
          max={2.0}
          step={0.1}
          formatValue={(v) => `${Math.round(Math.min(LABEL_MAX_PX, Math.max(LABEL_MIN_PX, v * LABEL_TARGET_PX)))}px`}
          onChange={(v) => updateControls({ type: 'nodeLabelScale', state: v })}
        />
        <LabeledRange
          id="form-node-label-density"
          label="Label Density"
          tooltip="How many labels are shown before zooming in reveals more"
          value={controls.nodeLabelDensity}
          min={0.1}
          max={1.0}
          step={0.1}
          formatValue={(v) => `${(v * 100).toFixed(0)}%`}
          onChange={(v) => updateControls({ type: 'nodeLabelDensity', state: v })}
        />
      </>
    )}
    <Form.Check
      type="switch"
      id="form-node-drag"
      label="Draggable"
      checked={controls.enableNodeDrag}
      onChange={() => updateControls({ type: 'enableNodeDrag', state: !controls.enableNodeDrag })}
    />
  </PopoverBody>
);

export default NodesSection;
