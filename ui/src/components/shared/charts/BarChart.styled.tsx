import styled from 'styled-components';

// spec: ./SPEC.md

/** Vertical stack that holds all bar rows. */
export const BarChartContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.25rem 0;
`;

/**
 * A single bar row laid out as [icon?] [label] [track] [value]. Rendered as either a static `<div>` or
 * an interactive `<button>` via the `as` prop.
 */
export const BarRow = styled.div<{ $interactive: boolean }>`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  width: 100%;
  /* reset button chrome when this row is rendered as a <button> */
  margin: 0;
  padding: 0;
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: ${(p) => (p.$interactive ? 'pointer' : 'default')};
  border-radius: 4px;

  ${(p) =>
    p.$interactive
      ? `
    &:hover {
      background: var(--thorium-highlight-panel-bg, #363940);
    }
    &:focus-visible {
      outline: 2px solid var(--thorium-link-text, #5b9bd5);
      outline-offset: 1px;
    }
  `
      : ''}
`;

/** Optional leading icon slot. */
export const BarIcon = styled.span`
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
  color: var(--thorium-secondary-text, #aab);
`;

/** Fixed-width label column so tracks align across rows. */
export const BarLabel = styled.span`
  min-width: 90px;
  font-size: 0.8rem;
  color: var(--thorium-text);
  text-align: right;
  text-transform: capitalize;
`;

/** The full-width background track the fill sits inside. */
export const BarTrack = styled.div`
  flex: 1;
  height: 20px;
  background: var(--thorium-secondary-panel-bg, #2a2d35);
  border-radius: 3px;
  overflow: hidden;
  position: relative;
`;

/** The proportionally-sized colored fill. */
export const BarFill = styled.div<{ $width: number; $color: string }>`
  height: 100%;
  width: ${(p) => p.$width}%;
  background: ${(p) => p.$color};
  border-radius: 3px;
  transition: width 0.4s ease;
  min-width: ${(p) => (p.$width > 0 ? '2px' : '0')};
`;

/** The numeric value shown at the end of each row. */
export const BarCount = styled.span`
  min-width: 32px;
  font-size: 0.8rem;
  color: var(--thorium-secondary-text, #aab);
  font-variant-numeric: tabular-nums;
`;
