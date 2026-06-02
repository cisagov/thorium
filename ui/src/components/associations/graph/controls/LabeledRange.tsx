import React from 'react';

import { ControlRow, ControlLabel, RangeInput } from './Toolbar.styled';
import { OverlayTipTop } from '@components/shared/overlay/tips';

interface LabeledRangeProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  tooltip?: string;
  onChange: (value: number) => void;
}

const LabeledRange: React.FC<LabeledRangeProps> = ({ id, label, value, min, max, step, formatValue, tooltip, onChange }) => {
  const labelContent = (
    <ControlLabel htmlFor={id}>
      {label} ({formatValue ? formatValue(value) : value})
    </ControlLabel>
  );

  return (
    <ControlRow>
      {tooltip ? <OverlayTipTop tip={tooltip}>{labelContent}</OverlayTipTop> : labelContent}
      <RangeInput id={id} min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
    </ControlRow>
  );
};

export default LabeledRange;
