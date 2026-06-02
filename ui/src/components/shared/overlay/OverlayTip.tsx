import React, { useState } from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';

export enum Placement {
  Left = 'left',
  Right = 'right',
  Bottom = 'bottom',
  Top = 'top',
}

export interface OverlayTipProps {
  children: React.ReactNode;
  tip: string;
  wide?: boolean;
  className?: string;
  disabled?: boolean;
}

type OverlayTipBaseProps = OverlayTipProps & { placement: Placement };

const OverlayTip: React.FC<OverlayTipBaseProps> = ({ children, tip, wide = false, className = '', placement, disabled = false }) => {
  const [showOverlay, setShowOverlay] = useState(false);
  const [hoveringOverlay, setHoveringOverlay] = useState(false);

  const updateHoveringOverlay = (hovering: boolean) => {
    if (hovering) {
      setHoveringOverlay(true);
      setShowOverlay(true);
    } else {
      setHoveringOverlay(false);
      setShowOverlay(false);
    }
  };

  const updateShowOverlay = (show: boolean) => {
    if (show) {
      setShowOverlay(true);
    } else if (!hoveringOverlay && !show) {
      setShowOverlay(false);
    }
  };
  return (
    <OverlayTrigger
      show={showOverlay && !disabled}
      onToggle={(show) => updateShowOverlay(show)}
      placement={placement}
      overlay={
        <Tooltip
          className={`${wide ? 'tooltip-wide' : ''} ${className}`.trim()}
          onMouseLeave={() => updateHoveringOverlay(false)}
          onMouseEnter={() => updateHoveringOverlay(true)}
        >
          {tip}
        </Tooltip>
      }
    >
      <span>{children}</span>
    </OverlayTrigger>
  );
};

export default OverlayTip;
