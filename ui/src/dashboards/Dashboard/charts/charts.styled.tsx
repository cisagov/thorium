import styled from 'styled-components';

// spec: ./SPEC.md

// project imports
import { scaling, spacers } from '@styles';

/**
 * The width at which the two-chart stats row splits from one stacked column into two side-by-side columns.
 * `scaling.lg` (992px) is the shared breakpoint nearest the intended ~900px split, so the value stays
 * centrally auditable rather than a magic literal.
 */
const CHARTS_TWO_COLUMN_BREAKPOINT = scaling.lg;

/// The measured wrapper around a {@link VisxBarChart}'s SVG; full-width so the ResizeObserver tracks the column.
/// `position: relative` anchors the absolutely-positioned {@link ChartTooltip} to this box.
export const ChartContainer = styled.div`
  position: relative;
  width: 100%;
  min-width: 0;
`;

/**
 * The floating hover/focus tooltip for a single bar, anchored to the bar's top-center and lifted above it
 * via `translate(-50%, -100%)`. `pointer-events: none` keeps it from intercepting bar clicks. Two lines:
 * the full label and, muted below it, the count in parentheses.
 */
export const ChartTooltip = styled.div`
  position: absolute;
  z-index: 2;
  pointer-events: none;
  transform: translate(-50%, -100%);
  padding: ${spacers.one} ${spacers.two};
  border-radius: 6px;
  white-space: nowrap;
  text-align: center;
  font-size: 0.75rem;
  line-height: 1.25;
  background: var(--thorium-panel-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-panel-border);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
`;

/// The muted count line (in parentheses) shown beneath the label inside {@link ChartTooltip}.
export const ChartTooltipCount = styled.div`
  color: var(--thorium-secondary-text);
`;

/// Muted placeholder shown when a chart has no bars to render.
export const EmptyChart = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 80px;
  color: var(--thorium-secondary-text);
  font-size: 0.85rem;
`;

/// A single chart's block: a small title above the chart body.
export const ChartBlock = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${spacers.one};
  min-width: 0;
`;

/// The header row of a chart block: the title and, when present, a key picker beside it.
export const ChartHeader = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: ${spacers.three};
`;

/// A chart's title (e.g. "Types", "Tag values").
export const ChartTitle = styled.h3`
  margin: 0;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--thorium-secondary-text);
`;

/// The native key-picker select used by the tag-values chart to choose which tag key to chart.
export const KeyPicker = styled.select`
  font-size: 0.78rem;
  padding: ${spacers.one} ${spacers.two};
  border-radius: 6px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-panel-border);
  cursor: pointer;
`;

/**
 * The two-chart row inside the stats tile: the Types chart and the Tag-values chart side by side on wide
 * viewports, stacking to one column when the tile is narrow. `minmax(0, 1fr)` lets each chart's SVG shrink
 * with the column so the grid never overflows.
 */
export const ChartsRow = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: ${spacers.four};

  @media (min-width: ${CHARTS_TWO_COLUMN_BREAKPOINT}) {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }
`;
