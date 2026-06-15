import React, { useLayoutEffect, useRef, useState } from 'react';
import styled from 'styled-components';

// spec: ./SPEC.md

// project imports
import { spacers } from '@styles';

/**
 * The clipped content wrapper. While collapsed it caps its height at `$maxPx`. When `$scroll` is set the
 * collapsed cap becomes a scroll area (so the content can be browsed without expanding). The clipped
 * bottom edge fades (`mask-image`) while collapsed AND overflowing — the mask is anchored to the box's
 * bottom, so it stays a **static** bottom fade even as the scroll area scrolls.
 */
const Clip = styled.div<{ $collapsed: boolean; $maxPx: number; $fade: boolean; $scroll: boolean }>`
  position: relative;
  overflow: ${({ $collapsed, $scroll }) => ($collapsed && $scroll ? 'hidden auto' : 'hidden')};
  max-height: ${({ $collapsed, $maxPx }) => ($collapsed ? `${$maxPx}px` : 'none')};
  ${({ $fade }) =>
    $fade &&
    `
    -webkit-mask-image: linear-gradient(to bottom, black 55%, transparent 100%);
    mask-image: linear-gradient(to bottom, black 55%, transparent 100%);
  `}
`;

// the show-more/less (or caller-labeled) toggle button, styled as an inline link
const ToggleButton = styled.button`
  margin-top: 2px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--thorium-link-text);
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;

  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

// centers the toggle horizontally when it is placed at the top of the collapsible. `$reserve` keeps a
// constant-height slot even when the toggle isn't shown, so the content below never shifts as the toggle
// appears/disappears (the dashboard filters reserve this so the tag tiles keep a fixed start position).
const TopToggleRow = styled.div<{ $reserve?: boolean }>`
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: ${spacers.two};
  ${({ $reserve }) => $reserve && 'min-height: 1.4rem;'}
`;

// centers the toggle horizontally when it is placed below the content (adaptive/collapsed case)
const BottomToggleRow = styled.div`
  display: flex;
  justify-content: center;
  margin-top: ${spacers.two};
`;

/** Where the collapse toggle sits relative to the clipped content. */
export enum TogglePosition {
  /** Below the content (the default markdown/show-more pattern). */
  Bottom = 'bottom',
  /** Above the content, horizontally centered. */
  Top = 'top',
  /**
   * Below while collapsed (so it reads as "expand", under the fade), above once expanded (so the
   * "collapse" control sits at the top of the now-tall content).
   */
  Adaptive = 'adaptive',
}

/** Props for {@link Collapsible}. */
export interface CollapsibleProps {
  /**
   * The content to clip. Either a node (default measure-and-clip mode) or a function of the collapsed
   * state — pair the function form with {@link CollapsibleProps.hasMore} to render a truncated set while
   * collapsed and the full set when expanded (the caller owns the truncation, so nothing is clipped).
   */
  children: React.ReactNode | ((collapsed: boolean) => React.ReactNode);
  /** Collapsed height cap in px; taller content is clipped behind the expand toggle (measure mode only). */
  maxPx: number;
  /**
   * When provided, the toggle is driven by this "there is more to show" signal instead of a measured
   * overflow, and the content renders at its natural height (no max-height clip or fade). Use with a
   * function `children` that returns a truncated set while collapsed and the full set when expanded — the
   * caller owns the truncation. Leave undefined for the default measure-and-clip behavior other callers use.
   */
  hasMore?: boolean;
  /**
   * Render the toggle label for the given collapsed state. Defaults to `Show more`/`Show less`. Callers
   * override to match their context (e.g. `⌄ filters` / `⌃ filters`).
   */
  renderToggleLabel?: (collapsed: boolean) => React.ReactNode;
  /** Whether to start collapsed. Defaults to `true`. Ignored when {@link CollapsibleProps.collapsed} is set. */
  defaultCollapsed?: boolean;
  /**
   * Controlled collapsed state. When provided, the component uses this instead of its own internal
   * state and the toggle calls {@link CollapsibleProps.onToggleCollapsed} with the requested next
   * value — letting a parent own the expanded/collapsed state (e.g. to preserve it across remounts,
   * as the dashboard tags tile does when the balanced-column layout re-parents a tile). When omitted,
   * the component manages its own state seeded by {@link CollapsibleProps.defaultCollapsed}.
   */
  collapsed?: boolean;
  /** Called with the requested next collapsed state when the toggle is pressed in controlled mode. */
  onToggleCollapsed?: (next: boolean) => void;
  /**
   * Where the toggle sits relative to the content. `Bottom` (default) keeps the show-more/less pattern;
   * `Top` renders a horizontally-centered toggle above the content; `Adaptive` puts it at the bottom while
   * collapsed (under the fade, reading as "expand") and at the top once expanded (the "collapse" control).
   */
  togglePosition?: TogglePosition;
  /**
   * When true, the collapsed cap becomes a vertical scroll area so the content can be browsed without
   * expanding; the static bottom fade stays intact. Defaults to false (clip only, still faded).
   */
  scrollWhenCollapsed?: boolean;
  /**
   * When true (with `Top` toggle position), a constant-height slot is reserved for the toggle above the
   * content even when the toggle isn't currently shown — so the content's start position never shifts as the
   * toggle appears/disappears. Used by the dashboard filters so the tag tiles keep a fixed top edge.
   */
  reserveToggle?: boolean;
  /** Optional class name applied to the clip wrapper (lets callers scope content styles). */
  className?: string;
}

/** The default toggle label when a caller supplies none. */
function defaultToggleLabel(collapsed: boolean): React.ReactNode {
  return collapsed ? 'Show more' : 'Show less';
}

/**
 * Clip arbitrary content to `maxPx` when collapsed, fading the clipped edge and offering a toggle only
 * when the content actually overflows.
 *
 * Generalizes the collapse/fade/measure pattern: it measures the inner content's `scrollHeight` against
 * `maxPx` via a `ResizeObserver` (which fires on mount AND whenever the content resizes — async loads,
 * grid reflow) so the toggle reliably appears once the content overflows, and short content shows no
 * chrome. The measure never keys off the toggle, so the toggle persists once expanded. When collapsed and
 * overflowing the bottom edge fades (static, anchored to the box); `scrollWhenCollapsed` additionally makes
 * the collapsed cap a scroll area (fade kept). Callers control the toggle label via `renderToggleLabel`.
 *
 * Controlled ("has more") mode: when `hasMore` is provided the component skips the measure/clip/fade and
 * drives the toggle from that flag, rendering the content at its natural height. Pair it with a function
 * `children` that returns a truncated set while collapsed and the full set when expanded — this lets a
 * caller mount only a preview of a large list (e.g. the tags tile's top-N values) and mount the rest on
 * expand, instead of mounting everything and merely CSS-clipping it.
 *
 * @param children - The content to clip, or a `(collapsed) => node` function (controlled truncation).
 * @param maxPx - Collapsed height cap in px (measure mode only).
 * @param hasMore - When set, drives the toggle from this flag instead of a measured overflow (controlled mode).
 * @param renderToggleLabel - Optional toggle-label renderer (defaults to `Show more`/`Show less`).
 * @param defaultCollapsed - Whether to start collapsed (defaults to `true`; ignored when `collapsed` is set).
 * @param collapsed - Controlled collapsed state; when set the parent owns it via `onToggleCollapsed`.
 * @param onToggleCollapsed - Called with the requested next collapsed state in controlled mode.
 * @param togglePosition - Where the toggle sits relative to the content (defaults to `Bottom`).
 * @param scrollWhenCollapsed - When true, the collapsed cap scrolls while keeping the static bottom fade.
 * @param className - Optional class name applied to the clip wrapper.
 * @returns The collapsible content plus its toggle (when overflowing).
 */
const Collapsible: React.FC<CollapsibleProps> = ({
  children,
  maxPx,
  hasMore,
  renderToggleLabel = defaultToggleLabel,
  defaultCollapsed = true,
  collapsed: controlledCollapsed,
  onToggleCollapsed,
  togglePosition = TogglePosition.Bottom,
  scrollWhenCollapsed = false,
  reserveToggle = false,
  className,
}) => {
  // controlled "has more" mode: the caller truncates the content itself (via function children) and tells
  // us whether more exists, so we skip the measure/clip/fade and drive the toggle from that flag instead
  const controlled = hasMore !== undefined;
  // controlled-collapse mode: the parent owns the collapsed state (so it survives remounts). When the
  // `collapsed` prop is absent we fall back to internal state seeded by `defaultCollapsed`
  const collapseControlled = controlledCollapsed !== undefined;
  const [internalCollapsed, setInternalCollapsed] = useState(defaultCollapsed);
  const collapsed = collapseControlled ? controlledCollapsed : internalCollapsed;
  const [overflowing, setOverflowing] = useState(false);
  // the inner, uncapped content wrapper: its full height drives the overflow measure (the Clip itself is
  // capped, so we measure the content, not the clip)
  const contentRef = useRef<HTMLDivElement>(null);

  // re-measure whenever the content's size changes (not just on mount) so async-loaded content — e.g. tags
  // that populate after the graph loads, or a grid that reflows on resize — reliably reveals the toggle.
  // Skipped in controlled mode, where `hasMore` drives the toggle and content shows at its natural height.
  useLayoutEffect(() => {
    if (controlled) return;
    const el = contentRef.current;
    if (!el) return;
    const measure = () => setOverflowing(el.scrollHeight > maxPx + 1);
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [maxPx, controlled]);

  const showToggle = controlled ? hasMore : overflowing;
  // route the toggle to the parent's setter when collapse is controlled, else to internal state
  const toggleCollapsed = () => {
    const next = !collapsed;
    if (collapseControlled) {
      onToggleCollapsed?.(next);
    } else {
      setInternalCollapsed(next);
    }
  };
  const toggle = showToggle && (
    <ToggleButton type="button" onClick={toggleCollapsed}>
      {renderToggleLabel(collapsed)}
    </ToggleButton>
  );

  // resolve function children against the current collapsed state (controlled truncation)
  const content = typeof children === 'function' ? children(collapsed) : children;

  // resolve Adaptive to a concrete side per collapsed state: bottom while collapsed, top once expanded
  const effectivePosition =
    togglePosition === TogglePosition.Adaptive ? (collapsed ? TogglePosition.Bottom : TogglePosition.Top) : togglePosition;
  // center the bottom toggle for Adaptive (e.g. the dashboard filters); plain Bottom (CollapsibleMarkdown's
  // show-more) keeps its inline, left-aligned placement
  const bottomToggle =
    effectivePosition === TogglePosition.Bottom && toggle ? (
      togglePosition === TogglePosition.Adaptive ? (
        <BottomToggleRow>{toggle}</BottomToggleRow>
      ) : (
        toggle
      )
    ) : null;

  // render the top toggle row when the toggle is shown, OR (with `reserveToggle`) always — keeping a
  // constant-height slot so the content below doesn't shift as the toggle appears/disappears
  const topToggle =
    effectivePosition === TogglePosition.Top && (toggle || reserveToggle) ? (
      <TopToggleRow $reserve={reserveToggle}>{toggle}</TopToggleRow>
    ) : null;

  return (
    <>
      {topToggle}
      <Clip
        className={className}
        $collapsed={controlled ? false : collapsed}
        $maxPx={maxPx}
        $fade={!controlled && collapsed && overflowing}
        $scroll={scrollWhenCollapsed}
      >
        <div ref={contentRef}>{content}</div>
      </Clip>
      {bottomToggle}
    </>
  );
};

export default Collapsible;
