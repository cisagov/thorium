import React from 'react';

// project imports
import OverlayTip, { OverlayTipProps, Placement } from './OverlayTip';

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
