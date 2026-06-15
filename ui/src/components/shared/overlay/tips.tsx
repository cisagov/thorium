import React from 'react';

// project imports
import OverlayTip, { OverlayTipProps, Placement } from './OverlayTip';

// spec: ./OverlayTip.spec.md

export const OverlayTipLeft: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '', disabled, block }) => {
  return (
    <OverlayTip tip={tip} wide={wide} className={className} placement={Placement.Left} disabled={disabled} block={block}>
      {children}
    </OverlayTip>
  );
};

export const OverlayTipRight: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '', disabled, block }) => {
  return (
    <OverlayTip tip={tip} wide={wide} className={className} placement={Placement.Right} disabled={disabled} block={block}>
      {children}
    </OverlayTip>
  );
};

export const OverlayTipBottom: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '', disabled, block }) => {
  return (
    <OverlayTip tip={tip} wide={wide} className={className} placement={Placement.Bottom} disabled={disabled} block={block}>
      {children}
    </OverlayTip>
  );
};

export const OverlayTipTop: React.FC<OverlayTipProps> = ({ children, tip, wide = false, className = '', disabled, block }) => {
  return (
    <OverlayTip tip={tip} wide={wide} className={className} placement={Placement.Top} disabled={disabled} block={block}>
      {children}
    </OverlayTip>
  );
};
