import styled from 'styled-components';

const CloseButton = styled.button`
  width: 20px;
  height: 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: auto;
  background: transparent;
  border: none;
  font-size: 15px;
  color: var(--thorium-text);
  margin-left: 4px;
  margin-right: 4px;
  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

export default CloseButton;
