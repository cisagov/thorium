import styled from 'styled-components';

const ToggleSwitch = styled.input.attrs({ type: 'checkbox', role: 'switch' })`
  appearance: none;
  width: 36px;
  height: 20px;
  background: var(--thorium-panel-border);
  border-radius: 10px;
  position: relative;
  cursor: pointer;
  transition: background 0.2s;
  flex-shrink: 0;

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: var(--thorium-text);
    border-radius: 50%;
    transition: transform 0.2s;
  }

  &:checked {
    background: var(--thorium-highlight-panel-border);
  }

  &:checked::after {
    transform: translateX(16px);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

export default ToggleSwitch;
