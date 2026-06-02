import React from 'react';
import styled from 'styled-components';

// project imports
import { BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';
import { FormatType } from '@utilities/rules/types';

const Wrapper = styled.div`
  display: flex;
  gap: ${BUTTON_BAR_GAP};
  align-items: center;
`;

const Label = styled.span`
  font-size: 12px;
  color: var(--thorium-secondary-text);
  font-weight: 600;
`;

const Btn = styled.button<{ $active: boolean }>`
  padding: 4px 12px;
  border: 1px solid ${(props) => (props.$active ? 'var(--thorium-highlight-panel-border)' : 'var(--thorium-panel-border)')};
  border-radius: 3px;
  background: ${(props) => (props.$active ? 'var(--thorium-highlight-panel-bg)' : 'var(--thorium-secondary-panel-bg)')};
  color: ${(props) => (props.$active ? 'var(--thorium-highlight-text)' : 'var(--thorium-secondary-text)')};
  cursor: pointer;
  font-size: 12px;
  font-family: monospace;
  font-weight: ${(props) => (props.$active ? '600' : '400')};
  transition:
    background 0.15s,
    border-color 0.15s,
    color 0.15s;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-highlight-text);
    border-color: var(--thorium-highlight-panel-border);
  }
`;

export interface FormatToggleProps {
  format: FormatType;
  onFormatChange: (format: FormatType) => void;
}

const FormatToggle: React.FC<FormatToggleProps> = ({ format, onFormatChange }) => (
  <Wrapper>
    <Label>Format:</Label>
    <Btn type="button" $active={format === FormatType.YAML} onClick={() => onFormatChange(FormatType.YAML)}>
      YAML
    </Btn>
    <Btn type="button" $active={format === FormatType.JSON} onClick={() => onFormatChange(FormatType.JSON)}>
      JSON
    </Btn>
  </Wrapper>
);

export default FormatToggle;
