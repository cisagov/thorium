import React from 'react';
import styled from 'styled-components';

// project imports
import { BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';

export enum ViewMode {
  Form = 'form',
  Editor = 'editor',
}

const Wrapper = styled.div`
  display: flex;
  gap: ${BUTTON_BAR_GAP};
  align-items: center;
`;

const Btn = styled.button<{ $active: boolean }>`
  padding: 4px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 3px;
  background: ${(props) => (props.$active ? 'var(--thorium-highlight-panel-bg)' : 'transparent')};
  color: var(--thorium-text);
  cursor: pointer;
  font-size: 12px;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

export interface ViewModeToggleProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}

const ViewModeToggle: React.FC<ViewModeToggleProps> = ({ viewMode, onViewModeChange }) => (
  <Wrapper>
    <Btn type="button" $active={viewMode === ViewMode.Form} onClick={() => onViewModeChange(ViewMode.Form)}>
      Form
    </Btn>
    <Btn type="button" $active={viewMode === ViewMode.Editor} onClick={() => onViewModeChange(ViewMode.Editor)}>
      Editor
    </Btn>
  </Wrapper>
);

export default ViewModeToggle;
