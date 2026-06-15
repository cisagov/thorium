import React, { useRef } from 'react';
import { Overlay, Popover } from 'react-bootstrap';

// project imports
import EntitySummary, { SummaryVariant } from './EntitySummary';
import { InfoModel, InfoNote } from './info';
import { SummaryPopover } from './SummaryPopover';
import { useHoverKeepOpen } from './useHoverKeepOpen';

// spec: ./SPEC.md

type Placement = React.ComponentProps<typeof Overlay>['placement'];

export interface EntitySummaryHoverProps {
  model: InfoModel;
  /** The trigger element (must forward a ref and accept mouse handlers — a DOM/styled element). */
  children: React.ReactElement;
  /** Extra notes passed through to {@link EntitySummary} (e.g. a duplicate-node warning). */
  notes?: InfoNote[];
  placement?: Placement;
  /**
   * Optional element to anchor the popover to instead of the trigger. Use when the trigger is wide (e.g. a
   * full-width row) but the popover should sit next to a specific inner element (e.g. the name) — hover
   * detection stays on the whole trigger.
   */
  anchorRef?: React.RefObject<HTMLElement | null>;
  /** Labels / part tokens to hide in the preview (see {@link EntitySummary} `exclude`). */
  exclude?: string[];
  /** Multi-parent flag passed through to {@link EntitySummary} (renders a compact Duplicate badge). */
  duplicate?: boolean;
}

/**
 * Wrap a trigger element so hovering it shows an {@link EntitySummary} preview popover.
 *
 * Uses a controlled `Overlay` (rather than `OverlayTrigger`) with show/hide timers so the pointer can
 * move from the trigger onto the popover and scroll long summaries without it closing — `OverlayTrigger`'s
 * hover trigger hides before the pointer reaches the popover.
 */
const EntitySummaryHover: React.FC<EntitySummaryHoverProps> = ({
  model,
  children,
  notes,
  placement = 'auto',
  anchorRef,
  exclude,
  duplicate,
}) => {
  const targetRef = useRef<HTMLElement>(null);
  const { show, scheduleShow, scheduleHide, cancelHide } = useHoverKeepOpen();

  // clone the trigger to attach the ref + hover handlers (React 19: ref is a normal prop)
  const trigger = React.cloneElement(children as React.ReactElement<Record<string, unknown>>, {
    ref: targetRef,
    onMouseEnter: scheduleShow,
    onMouseLeave: scheduleHide,
  });

  return (
    <>
      {trigger}
      <Overlay
        target={anchorRef?.current ?? targetRef.current}
        show={show}
        placement={placement}
        container={document.body}
        popperConfig={{ modifiers: [{ name: 'offset', options: { offset: [0, 8] } }] }}
      >
        {(props) => (
          <SummaryPopover {...props} onMouseEnter={cancelHide} onMouseLeave={scheduleHide}>
            <Popover.Body>
              <EntitySummary model={model} variant={SummaryVariant.Compact} notes={notes} exclude={exclude} duplicate={duplicate} />
            </Popover.Body>
          </SummaryPopover>
        )}
      </Overlay>
    </>
  );
};

export default EntitySummaryHover;
