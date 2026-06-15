import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

/**
 * A dependency-free, theme-aware circular spinner. `$size` sets both the diameter and (implicitly) the
 * border thickness, so the same component covers inline row spinners and larger loading indicators.
 */
export const StyledSpinner = styled.span<{ $size?: number }>`
  display: inline-block;
  flex: 0 0 auto;
  width: ${({ $size = 12 }) => $size}px;
  height: ${({ $size = 12 }) => $size}px;
  border: ${({ $size = 12 }) => Math.max(2, Math.round($size / 7))}px solid var(--thorium-panel-border);
  border-top-color: var(--thorium-highlight-text);
  border-radius: 50%;
  animation: ${spin} 0.7s linear infinite;
`;

export default StyledSpinner;
