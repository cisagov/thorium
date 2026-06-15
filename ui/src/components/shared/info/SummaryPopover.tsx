import { Popover } from 'react-bootstrap';
import styled from 'styled-components';

// spec: ./SPEC.md

/**
 * The single themed popover shell for entity/file/repo summary previews (hover overlays + the graph's
 * cursor-anchored portal). Replaces the former per-surface `InfoPopover` and `PreviewPopover`.
 *
 * Only `.popover-body` scrolls (the sole scroll container), so long summaries scroll without a nested
 * double scrollbar. The rendered {@link EntitySummary} wrapper must not set its own `overflow`.
 */
export const SummaryPopover = styled(Popover)`
  --bs-popover-max-width: 420px;
  --bs-popover-bg: var(--thorium-secondary-panel-bg);
  --bs-popover-border-color: var(--thorium-panel-border);
  --bs-popover-body-color: var(--thorium-text);
  --bs-popover-arrow-border: var(--thorium-panel-border);

  .popover-body {
    padding: 10px 14px;
    max-height: 60vh;
    overflow: auto;
  }
`;
