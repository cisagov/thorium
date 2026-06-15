// spec: ./SPEC.md

import React from 'react';

// project imports
import { DepthGroup } from './styles';
import { MAX_DEPTH } from '../Dashboard/seedParams';
import ScrollableSelect from '@components/shared/inputs/ScrollableSelect';
import { OverlayTipTop } from '@components/shared/overlay/tips';

/// The minimum crawl depth the builder allows.
const MIN_DEPTH = 1;

/**
 * Props for {@link DepthControl}.
 */
interface DepthControlProps {
  /// The current crawl depth.
  depth: number;
  /// Called with the new depth when the user picks one.
  onChange: (depth: number) => void;
}

/**
 * The crawl-depth selector for the builder.
 *
 * Wraps the shared {@link ScrollableSelect} (default 2, range 1..=10) with a label and an overlay
 * tip. The chosen depth is encoded into the dashboard URL alongside the seed resources.
 *
 * @param props - See {@link DepthControlProps}.
 * @returns The depth selector control.
 */
const DepthControl: React.FC<DepthControlProps> = ({ depth, onChange }) => {
  return (
    <DepthGroup>
      <span>Depth</span>
      <OverlayTipTop tip="How many association hops to crawl from each seed resource (default 2, max 10). You can deepen further from the dashboard.">
        <span>
          <ScrollableSelect value={depth} onChange={onChange} min={MIN_DEPTH} max={MAX_DEPTH} />
        </span>
      </OverlayTipTop>
    </DepthGroup>
  );
};

export default DepthControl;
