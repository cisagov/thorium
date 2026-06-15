import styled from 'styled-components';

// project imports
import { spacers } from '@styles';

// spec: ./Button.spec.md

/**
 * A horizontal, centered row of action buttons for details-page toolbars.
 *
 * Mirrors the centered action row on the generic entity details page so the file, repo, and entity
 * details pages present their toolbars consistently. Wraps onto multiple lines on narrow viewports.
 */
const ButtonToolbar = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${spacers.three};
  flex-wrap: wrap;
`;

export default ButtonToolbar;
