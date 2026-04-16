import React from 'react';
import { Form } from 'react-bootstrap';

import type { SectionProps } from './types';
import { PopoverBody, Divider } from './Toolbar.styled';
import LabeledRange from './LabeledRange';

const EdgesSection: React.FC<SectionProps> = ({ controls, updateControls }) => (
  <PopoverBody>
    <LabeledRange
      id="form-edge-width"
      label="Width"
      value={controls.edgeWidth}
      min={0.1}
      max={2}
      step={0.1}
      onChange={(v) => updateControls({ type: 'edgeWidth', state: v })}
    />
    <LabeledRange
      id="form-edge-length"
      label="Length"
      value={controls.edgeLength}
      min={10}
      max={200}
      step={10}
      onChange={(v) => updateControls({ type: 'edgeLength', state: v })}
    />
    <LabeledRange
      id="form-link-strength"
      label="Strength"
      value={controls.edgeLinkStrength}
      min={0.1}
      max={2.0}
      step={0.1}
      onChange={(v) => updateControls({ type: 'edgeLinkStrength', state: v })}
    />
    <LabeledRange
      id="form-edge-opacity"
      label="Opacity"
      value={controls.edgeOpacity}
      min={0.1}
      max={1.0}
      step={0.05}
      onChange={(v) => updateControls({ type: 'edgeOpacity', state: v })}
    />
    <LabeledRange
      id="form-arrow-length"
      label="Arrows"
      value={controls.arrowLength}
      min={0}
      max={15}
      step={0.5}
      onChange={(v) => updateControls({ type: 'arrowLength', state: v })}
    />

    <Divider />

    <LabeledRange
      id="form-particles"
      label="Particles"
      value={controls.directionalParticles}
      min={0}
      max={10}
      step={1}
      onChange={(v) => updateControls({ type: 'directionalParticles', state: v })}
    />
    {controls.directionalParticles > 0 && (
      <LabeledRange
        id="form-particle-speed"
        label="Speed"
        value={Math.round(controls.particleSpeed / 0.002)}
        min={1}
        max={10}
        step={1}
        onChange={(v) => updateControls({ type: 'particleSpeed', state: v * 0.002 })}
      />
    )}

    <Divider />

    <Form.Check
      type="switch"
      id="form-show-edge-labels"
      label="Labels"
      checked={controls.showEdgeLabels}
      onChange={() => updateControls({ type: 'showEdgeLabels', state: !controls.showEdgeLabels })}
    />
    {controls.showEdgeLabels && (
      <>
        <LabeledRange
          id="form-edge-label-scale"
          label="Label Size"
          value={controls.edgeLabelScale}
          min={0.5}
          max={3}
          step={0.1}
          formatValue={(v) => `${v.toFixed(1)}x`}
          onChange={(v) => updateControls({ type: 'edgeLabelScale', state: v })}
        />
        <LabeledRange
          id="form-edge-label-density"
          label="Label Density"
          value={controls.edgeLabelDensity}
          min={0.1}
          max={1.0}
          step={0.1}
          formatValue={(v) => `${(v * 100).toFixed(0)}%`}
          onChange={(v) => updateControls({ type: 'edgeLabelDensity', state: v })}
        />
        <LabeledRange
          id="form-edge-label-min-size"
          label="Min Font Size"
          value={controls.edgeLabelMinSize}
          min={0.5}
          max={5}
          step={0.5}
          formatValue={(v) => `${v.toFixed(1)}x`}
          onChange={(v) => updateControls({ type: 'edgeLabelMinSize', state: v })}
        />
      </>
    )}
  </PopoverBody>
);

export default EdgesSection;
