import React, { useEffect, useRef, useState } from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';

// spec: ./OverlayTip.spec.md

// Small delay before showing a tooltip so a quick pass-through hover (moving directly across several
// tipped sections without pausing) doesn't flash/stack tooltips and churn the DOM.
const SHOW_DELAY_MS = 150;

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
  /// Render the trigger wrapper as a block so its box matches a block child (e.g. a full-height
  /// textarea). Without this, the default inline wrapper anchors the tooltip to the inline baseline,
  /// pushing side-placed tooltips to the bottom corner of a tall field instead of its vertical center.
  block?: boolean;
}

type OverlayTipBaseProps = OverlayTipProps & { placement: Placement };

const OverlayTip: React.FC<OverlayTipBaseProps> = ({
  children,
  tip,
  wide = false,
  className = '',
  placement,
  disabled = false,
  block = false,
}) => {
  const [showOverlay, setShowOverlay] = useState(false);
  const [hoveringOverlay, setHoveringOverlay] = useState(false);
  const showTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const clearShowTimer = () => clearTimeout(showTimer.current);
  useEffect(() => clearShowTimer, []);

  const updateHoveringOverlay = (hovering: boolean) => {
    clearShowTimer();
    if (hovering) {
      setHoveringOverlay(true);
      setShowOverlay(true);
    } else {
      setHoveringOverlay(false);
      setShowOverlay(false);
    }
  };

  const updateShowOverlay = (show: boolean) => {
    clearShowTimer();
    if (show) {
      // Delay the show so rapid pass-through hovers never trigger the tooltip (avoids the flash/stack
      // that momentarily grows the page and jiggles the scrollbar).
      showTimer.current = setTimeout(() => setShowOverlay(true), SHOW_DELAY_MS);
    } else if (!hoveringOverlay) {
      setShowOverlay(false);
    }
  };
  return (
    <OverlayTrigger
      show={showOverlay && !disabled}
      onToggle={(show) => updateShowOverlay(show)}
      placement={placement}
      flip
      // Render the tooltip in a portal with fixed positioning so it never expands the page's scroll
      // area (an inline, absolutely-positioned tooltip placed past the viewport edge grows the
      // document and toggles a scrollbar on hover — a "jiggle"). Fixed + viewport overflow handling
      // keeps it on-screen instead.
      container={typeof document !== 'undefined' ? document.body : undefined}
      popperConfig={{ strategy: 'fixed' }}
      overlay={
        <Tooltip
          className={`tip-fixed ${wide ? 'tooltip-wide' : ''} ${className}`.trim()}
          onMouseLeave={() => updateHoveringOverlay(false)}
          onMouseEnter={() => updateHoveringOverlay(true)}
        >
          {tip}
        </Tooltip>
      }
    >
      <span style={block ? { display: 'block' } : undefined}>{children}</span>
    </OverlayTrigger>
  );
};

export default OverlayTip;
