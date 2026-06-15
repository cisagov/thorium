import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Default delay before showing on hover, so quick pointer passes over a trigger don't flash the popover.
 */
export const HOVER_SHOW_DELAY = 300;
/**
 * Default delay before hiding after the pointer leaves, so it can travel from the trigger onto the popover
 * (e.g. to scroll long content) without the popover closing out from under it.
 */
export const HOVER_HIDE_DELAY = 200;

/** The controls returned by {@link useHoverKeepOpen} for wiring a keep-open hover overlay. */
export interface HoverKeepOpen {
  /** Whether the overlay should currently be shown. */
  show: boolean;
  /** Schedule the overlay to show after the show delay (cancels any pending hide). */
  scheduleShow: () => void;
  /** Schedule the overlay to hide after the hide delay (cancels any pending show). */
  scheduleHide: () => void;
  /** Cancel a pending hide — call when the pointer enters the overlay so it stays open. */
  cancelHide: () => void;
}

/**
 * Keep-open hover state for an overlay that the pointer must be able to travel onto (e.g. a scrollable
 * summary popover). A short show delay avoids flashing on quick passes; a hide delay lets the pointer
 * cross the gap from trigger to overlay without it closing. Timers are cleared on unmount.
 *
 * @param showDelay - Delay in ms before showing on hover. Defaults to {@link HOVER_SHOW_DELAY}.
 * @param hideDelay - Delay in ms before hiding after leaving. Defaults to {@link HOVER_HIDE_DELAY}.
 * @returns The current `show` flag plus schedule/cancel handlers to bind to trigger and overlay events.
 */
export function useHoverKeepOpen(showDelay: number = HOVER_SHOW_DELAY, hideDelay: number = HOVER_HIDE_DELAY): HoverKeepOpen {
  const [show, setShow] = useState(false);
  const showTimer = useRef<number | undefined>(undefined);
  const hideTimer = useRef<number | undefined>(undefined);
  const scheduleShow = useCallback(() => {
    window.clearTimeout(hideTimer.current);
    showTimer.current = window.setTimeout(() => setShow(true), showDelay);
  }, [showDelay]);
  const scheduleHide = useCallback(() => {
    window.clearTimeout(showTimer.current);
    hideTimer.current = window.setTimeout(() => setShow(false), hideDelay);
  }, [hideDelay]);
  const cancelHide = useCallback(() => window.clearTimeout(hideTimer.current), []);
  useEffect(
    () => () => {
      window.clearTimeout(showTimer.current);
      window.clearTimeout(hideTimer.current);
    },
    [],
  );
  return { show, scheduleShow, scheduleHide, cancelHide };
}
