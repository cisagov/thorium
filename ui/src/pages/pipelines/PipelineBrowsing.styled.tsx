import styled from 'styled-components';

/// The top header bar: count on the left, title centered, create button on the right
export const HeaderBar = styled.div`
  display: flex;
  justify-content: space-between;
`;

/// Centered row holding the pipelines omnibar
export const OmnibarRow = styled.div`
  display: flex;
  justify-content: center;
`;

/// The pipeline count badge (replaces the react-bootstrap Badge + .count-badge utility)
export const CountBadge = styled.span`
  display: inline-block;
  padding: 0.35em 0.65em;
  font-size: 0.75em;
  font-weight: 700;
  line-height: 1;
  border-radius: 0.375rem;
  background-color: var(--thorium-nav-panel-bg);
  color: var(--thorium-nav-text);
`;
