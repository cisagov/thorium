import React from 'react';
import { Dropdown, Form } from 'react-bootstrap';

import type { GraphSectionProps } from './types';
import { MenuList, MenuItem, MenuDropdown, PopoverBody, Divider } from './Toolbar.styled';
import LabeledRange from './LabeledRange';

const GraphSection: React.FC<GraphSectionProps> = ({ controls, updateControls, graphInstance }) => (
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
      <>
        <Form.Check
          type="switch"
          id="form-adjust-distance"
          label="Adjust Distance"
          checked={controls.adjustDistanceOnFocus}
          onChange={() => updateControls({ type: 'adjustDistanceOnFocus', state: !controls.adjustDistanceOnFocus })}
        />
        {controls.adjustDistanceOnFocus && (
          <LabeledRange
            id="form-focus-distance-ratio"
            label="Distance"
            value={controls.focusDistanceRatio}
            min={0.1}
            max={2}
            step={0.1}
            formatValue={(v) => (v >= 2 ? 'Fit All' : `${v.toFixed(1)}x`)}
            onChange={(v) => updateControls({ type: 'focusDistanceRatio', state: v })}
          />
        )}
      </>
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
        if (next) {
          updateControls({ type: 'focusOnClick', state: false });
          updateControls({ type: 'adjustDistanceOnFocus', state: false });
          updateControls({ type: 'focusDistanceRatio', state: 1 });
        }
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
