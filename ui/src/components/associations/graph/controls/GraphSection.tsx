import React from 'react';
import { Dropdown, Form } from 'react-bootstrap';

// project imports
import type { GraphSectionProps } from './types';
import { computeSizeDefaults } from './sizeDefaults';
import { MenuList, MenuItem, MenuDropdown, PopoverBody, Divider } from './Toolbar.styled';

// spec: ./GraphControlsToolbar.spec.md

const GraphSection: React.FC<GraphSectionProps> = ({ controls, updateControls, graphInstance, nodeCount }) => (
  <PopoverBody>
    <MenuList $inset>
      <MenuDropdown>
        <Dropdown.Toggle variant="" size="sm" id="dimensions-dropdown">
          {controls.numDimensions}D
        </Dropdown.Toggle>
        <Dropdown.Menu>
          <Dropdown.Item onClick={() => updateControls({ type: 'numDimensions', state: 2 })}>2D</Dropdown.Item>
          <Dropdown.Item onClick={() => updateControls({ type: 'numDimensions', state: 3 })}>3D</Dropdown.Item>
        </Dropdown.Menu>
      </MenuDropdown>
      <MenuItem onClick={() => graphInstance?.zoomToFit(1000, 50)}>Fit All</MenuItem>
      <MenuItem onClick={() => updateControls({ type: 'applySizeDefaults', state: computeSizeDefaults(nodeCount) })}>Optimize</MenuItem>
      <MenuItem
        onClick={() => {
          updateControls({ type: 'resetSizeOverrides' });
          updateControls({ type: 'applySizeDefaults', state: computeSizeDefaults(nodeCount) });
        }}
      >
        Reset
      </MenuItem>
    </MenuList>

    <Divider />

    <Form.Check
      type="switch"
      id="form-focus-on-click"
      label="Focus on Click"
      checked={controls.focusOnClick}
      onChange={() => {
        const next = !controls.focusOnClick;
        updateControls({ type: 'focusOnClick', state: next });
        if (next) updateControls({ type: 'refitOnGrow', state: false });
      }}
    />
    {controls.focusOnClick && (
      <Form.Check
        type="switch"
        id="form-fit-neighborhood"
        label="Fit Neighborhood"
        checked={controls.fitNeighborhoodOnFocus}
        onChange={() => updateControls({ type: 'fitNeighborhoodOnFocus', state: !controls.fitNeighborhoodOnFocus })}
      />
    )}

    <Divider />

    <Form.Check
      type="switch"
      id="form-refit-on-grow"
      label="Refit on Grow"
      checked={controls.refitOnGrow}
      onChange={() => {
        const next = !controls.refitOnGrow;
        updateControls({ type: 'refitOnGrow', state: next });
        if (next) updateControls({ type: 'focusOnClick', state: false });
      }}
    />

    <Divider />

    <Form.Check
      type="switch"
      id="form-show-grid"
      label="Show Grid"
      checked={controls.showGrid}
      onChange={() => updateControls({ type: 'showGrid', state: !controls.showGrid })}
    />
  </PopoverBody>
);

export default GraphSection;
