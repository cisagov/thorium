import React, { useState } from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';

enum Placement {
  Left = 'left',
  Right = 'right',
  Bottom = 'bottom',
  Top = 'top',
}

interface OverlayTipProps {
  children: React.ReactNode;
  tip: string;
  wide?: boolean;
  className?: string;
}
//  placement: Placement;

type OverlayTipBaseProps = OverlayTipProps & { placement: Placement };

const OverlayTip: React.FC<OverlayTipBaseProps> = ({ children, tip, wide = false, placement }) => {
  // use trigger="click" for OverlayTrigger for troubleshooting css/style issues
  const [showOverlay, setShowOverlay] = useState(false);
  const [hoveringOverlay, setHoveringOverlay] = useState(false);

  // update whether overlay tip is hovered over by mouse
  const updateHoveringOverlay = (hovering: boolean) => {
    if (hovering) {
      setHoveringOverlay(true);
      setShowOverlay(true);
    } else {
      setHoveringOverlay(false);
      setShowOverlay(false);
    }
  };

  // update whether overlay should be showing
  const updateShowOverlay = (show: boolean) => {
    if (show) {
      setShowOverlay(true);
      // don't allow closing of tip by Overlay trigger when hovering the overlay tip itself
    } else if (!hoveringOverlay && !show) {
      setShowOverlay(false);
    }
  };
  return (
    <OverlayTrigger
      show={showOverlay}
      onToggle={(show) => updateShowOverlay(show)}
      placement={placement}
      overlay={
        <Tooltip
          className={wide ? 'tooltip-wide' : ''}
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

export const OverlayTipLeft: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '' }) => {
  return (
    <OverlayTip tip={tip} wide={wide} placement={Placement.Left}>
      {children}
    </OverlayTip>
  );
};

export const OverlayTipRight: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '' }) => {
  return (
    <OverlayTip tip={tip} wide={wide} placement={Placement.Right}>
      {children}
    </OverlayTip>
  );
};

export const OverlayTipBottom: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '' }) => {
  return (
    <OverlayTip tip={tip} wide={wide} placement={Placement.Bottom}>
      {children}
    </OverlayTip>
  );
};

export const OverlayTipTop: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '' }) => {
  return (
    <OverlayTip tip={tip} wide={wide} placement={Placement.Top}>
      {children}
    </OverlayTip>
  );
};
