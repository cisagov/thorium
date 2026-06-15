import React, { ReactNode, useMemo } from 'react';

// project imports
import { BarChartContainer, BarCount, BarFill, BarIcon, BarLabel, BarRow, BarTrack } from './BarChart.styled';

// spec: ./SPEC.md

/** Default fill color used when a bar does not specify its own `color`. */
const DEFAULT_BAR_COLOR = 'var(--thorium-link-text, #5b9bd5)';

/** A single datum rendered as one horizontal bar. */
export interface BarDatum {
  /** Stable identity used as the React key and for uniqueness. */
  id: string;
  /** Human-readable label; always displayed (never color-only). */
  label: string;
  /** Numeric value; always displayed and used to size the fill. */
  value: number;
  /** Optional CSS color for the fill; defaults to a theme variable. */
  color?: string;
  /** Optional leading icon rendered before the label. */
  icon?: ReactNode;
  /** When provided, the bar becomes an interactive `<button>` that calls this on activation. */
  onClick?: () => void;
}

/** Props for the shared, accessible horizontal `BarChart`. */
export interface BarChartProps {
  /** The bars to render, in display order. */
  bars: BarDatum[];
  /** Denominator for fill width; defaults to the largest bar value. Values `<= 0` fall back to `1`. */
  max?: number;
  /** Accessible name for the chart group. */
  ariaLabel?: string;
  /** Optional class applied to the outer container. */
  className?: string;
}

/**
 * A generic, accessible horizontal bar chart.
 *
 * Each bar always shows its text label and numeric value (never color-only encoding), with a
 * proportionally-sized fill. When a bar supplies an `onClick`, it renders as a real `<button>`
 * with a descriptive `aria-label`, visible focus, and native keyboard activation; otherwise it
 * renders as a non-interactive element.
 *
 * @param bars - The bars to render, in display order.
 * @param max - Denominator for fill width; defaults to the largest bar value. Clamped to at least 1.
 * @param ariaLabel - Accessible name for the chart group. Defaults to `'Bar chart'`.
 * @param className - Optional class applied to the outer container.
 * @returns The rendered bar chart group.
 */
const BarChart: React.FC<BarChartProps> = ({ bars, max, ariaLabel = 'Bar chart', className }) => {
  // clamp the denominator to at least 1 so a zero/negative max never yields NaN or divide-by-zero
  const effectiveMax = useMemo(() => {
    const derived = max ?? Math.max(0, ...bars.map((b) => b.value));
    return derived > 0 ? derived : 1;
  }, [max, bars]);
  return (
    <BarChartContainer role="group" aria-label={ariaLabel} className={className}>
      {bars.map((bar) => {
        // clamp fill to [0, 100] so out-of-range values render sanely
        const width = Math.max(0, Math.min(100, (bar.value / effectiveMax) * 100));
        const interactive = typeof bar.onClick === 'function';
        return (
          <BarRow
            key={bar.id}
            $interactive={interactive}
            as={interactive ? 'button' : 'div'}
            type={interactive ? 'button' : undefined}
            onClick={bar.onClick}
            aria-label={interactive ? `${bar.label}: ${bar.value}` : undefined}
          >
            {bar.icon ? <BarIcon>{bar.icon}</BarIcon> : null}
            <BarLabel>{bar.label}</BarLabel>
            <BarTrack>
              <BarFill $width={width} $color={bar.color ?? DEFAULT_BAR_COLOR} />
            </BarTrack>
            <BarCount>{bar.value}</BarCount>
          </BarRow>
        );
      })}
    </BarChartContainer>
  );
};

export default BarChart;
