import React from 'react';
import { Dropdown } from 'react-bootstrap';

import type { DagMode, GraphSectionProps } from './types';
import { PopoverBody, Divider, MenuList, MenuItem, MenuDropdown } from './Toolbar.styled';
import LabeledRange from './LabeledRange';

const DAG_MODE_LABELS: Record<string, string> = {
  none: 'None',
  td: 'Top-Down',
  bu: 'Bottom-Up',
  lr: 'Left-Right',
  rl: 'Right-Left',
  zout: 'Z-Out',
  zin: 'Z-In',
  radialout: 'Radial Out',
  radialin: 'Radial In',
};

const ForcesSection: React.FC<GraphSectionProps> = ({ controls, updateControls, graphInstance }) => (
  <PopoverBody>
    <LabeledRange
      id="form-charge"
      label="Charge"
      value={controls.chargeStrength}
      min={-500}
      max={-10}
      step={10}
      onChange={(v) => updateControls({ type: 'chargeStrength', state: v })}
    />
    <LabeledRange
      id="form-velocity-decay"
      label="Friction"
      value={controls.velocityDecay}
      min={0.1}
      max={1.0}
      step={0.05}
      onChange={(v) => updateControls({ type: 'velocityDecay', state: v })}
    />
    <LabeledRange
      id="form-warmup"
      label="Warmup"
      value={controls.warmupTicks}
      min={0}
      max={300}
      step={10}
      onChange={(v) => updateControls({ type: 'warmupTicks', state: v })}
    />
    <LabeledRange
      id="form-cooldown"
      label="Cooldown"
      value={controls.cooldownTime}
      min={1000}
      max={30000}
      step={1000}
      formatValue={(v) => `${(v / 1000).toFixed(0)}s`}
      onChange={(v) => updateControls({ type: 'cooldownTime', state: v })}
    />

    <Divider />

    <MenuList $inset>
      <MenuDropdown>
        <Dropdown.Toggle variant="" size="sm" id="dag-mode-dropdown">
          DAG: {DAG_MODE_LABELS[controls.dagMode ?? 'none']}
        </Dropdown.Toggle>
        <Dropdown.Menu>
          {Object.entries(DAG_MODE_LABELS).map(([key, label]) => (
            <Dropdown.Item key={key} onClick={() => updateControls({ type: 'dagMode', state: key === 'none' ? null : (key as DagMode) })}>
              {label}
            </Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </MenuDropdown>
      <MenuItem onClick={() => graphInstance?.d3ReheatSimulation()}>Reheat</MenuItem>
    </MenuList>

    {controls.dagMode && (
      <LabeledRange
        id="form-dag-distance"
        label="DAG Spacing"
        value={controls.dagLevelDistance ?? 50}
        min={10}
        max={300}
        step={10}
        formatValue={(v) => (controls.dagLevelDistance === null ? 'auto' : String(v))}
        onChange={(v) => updateControls({ type: 'dagLevelDistance', state: v })}
      />
    )}
  </PopoverBody>
);

export default ForcesSection;
