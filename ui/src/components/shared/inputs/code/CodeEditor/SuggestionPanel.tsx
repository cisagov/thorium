import React, { useState } from 'react';
import styled from 'styled-components';
import type { Suggestion } from '@utilities/rules/types';

const Panel = styled.div`
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 8px 12px;
  font-size: 12px;
  max-height: 200px;
  overflow-y: auto;
`;

const SuggestionRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
  padding: 6px 0;
  color: var(--thorium-text);

  & + & {
    border-top: 1px solid var(--thorium-highlight-panel-border);
  }
`;

const FieldLabel = styled.span`
  font-weight: 600;
  color: var(--thorium-link-text-alt);
  white-space: nowrap;
  flex-shrink: 0;
`;

const Message = styled.span`
  color: var(--thorium-secondary-text);
  flex-basis: 100%;
  font-size: 11px;
  margin-bottom: 2px;
`;

const ValueChip = styled.span`
  display: inline-block;
  background-color: var(--thorium-highlight-panel-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 3px;
  padding: 1px 6px;
  margin: 1px 3px;
  font-family: monospace;
  font-size: 11px;
  cursor: pointer;

  &:hover {
    background-color: var(--thorium-info-secondary-bg);
    color: var(--thorium-text);
  }
`;

const ValuesContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  flex-basis: 100%;
`;

const AddButton = styled.span`
  display: inline-block;
  background-color: var(--thorium-highlight-panel-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 3px;
  padding: 1px 6px;
  margin: 1px 3px;
  font-family: monospace;
  font-size: 11px;
  cursor: pointer;

  &:hover {
    background-color: var(--thorium-info-secondary-bg);
    color: var(--thorium-text);
  }
`;

const ToggleButton = styled.button`
  background: none;
  border: none;
  color: var(--thorium-link-text);
  cursor: pointer;
  font-size: 12px;
  padding: 4px 0;
  text-decoration: underline;

  &:hover {
    color: var(--thorium-link-text-alt);
  }
`;

const headerStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
};

const headerLabelStyle: React.CSSProperties = {
  color: 'var(--thorium-secondary-text)',
  fontWeight: 600,
};

interface SuggestionPanelProps {
  suggestions: Suggestion[];
  onValueClick?: (field: string, value: string, isList?: boolean) => void;
}

const SuggestionPanel: React.FC<SuggestionPanelProps> = ({ suggestions, onValueClick }) => {
  const [collapsed, setCollapsed] = useState(false);

  if (suggestions.length === 0) return null;

  return (
    <Panel>
      <div style={{ ...headerStyle, marginBottom: collapsed ? 0 : 4 }}>
        <span style={headerLabelStyle}>Suggestions ({suggestions.length})</span>
        <ToggleButton onClick={() => setCollapsed((prev) => !prev)}>{collapsed ? 'Show' : 'Hide'}</ToggleButton>
      </div>
      {!collapsed &&
        suggestions.map((suggestion, idx) => (
          <SuggestionRow key={`${suggestion.field}-${idx}`}>
            <FieldLabel>{suggestion.field}</FieldLabel>
            <Message>{suggestion.message}</Message>
            {suggestion.values && suggestion.values.length > 0 ? (
              <ValuesContainer>
                {suggestion.values.map((val) => (
                  <ValueChip
                    key={val}
                    onClick={() => onValueClick?.(suggestion.field, val, suggestion.isList)}
                    title={`Click to use '${val}'`}
                  >
                    {val}
                  </ValueChip>
                ))}
              </ValuesContainer>
            ) : (
              <AddButton onClick={() => onValueClick?.(suggestion.field, '', suggestion.isList)} title={`Add '${suggestion.field}' field`}>
                Add
              </AddButton>
            )}
          </SuggestionRow>
        ))}
    </Panel>
  );
};

export default SuggestionPanel;
