import styled from 'styled-components';

// Shared chrome for the standalone OAuth pages (callback + link landing).

export const Centered = styled.div`
  display: flex;
  justify-content: center;
  padding-top: 2rem;
`;

export const CardBox = styled.div`
  width: 100%;
  max-width: 32rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 2rem;
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

export const Message = styled.p`
  color: var(--thorium-secondary-text);
  text-align: center;
`;
