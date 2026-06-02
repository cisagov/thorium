import React, { useState } from 'react';
import styled from 'styled-components';
import type { Suggestion, FieldSchema } from '@utilities/rules/types';
import { FieldValueType } from '@utilities/rules/types';

const Panel = styled.div`
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 8px 12px;
  font-size: 12px;
  height: 200px;
  min-height: 60px;
  overflow-y: auto;
  resize: vertical;
`;

const SectionTitle = styled.div`
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--thorium-secondary-text);
  padding: 6px 0 2px;
  border-bottom: 1px solid var(--thorium-panel-border);
  margin-top: 4px;

  &:first-child {
    margin-top: 0;
  }
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

const TypeBadge = styled.span`
  display: inline-block;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.5px;
  padding: 1px 5px;
  border-radius: 3px;
  background-color: var(--thorium-highlight-panel-bg);
  color: var(--thorium-secondary-text);
  border: 1px solid var(--thorium-panel-border);
  flex-shrink: 0;
`;

function typeLabel(schema?: FieldSchema): string | null {
  if (!schema) return null;
  if (schema.typeName) return schema.typeName;
  switch (schema.type) {
    case FieldValueType.String:
      return 'string';
    case FieldValueType.Number:
      return 'number';
    case FieldValueType.Boolean:
      return 'bool';
    case FieldValueType.Enum:
      return 'enum';
    case FieldValueType.Object:
      return 'object';
    case FieldValueType.StringArray:
      return 'list';
    default:
      return null;
  }
}

function displayFieldName(field: string): string {
  return field;
}

function groupByCategory(suggestions: Suggestion[]): { category: string; items: Suggestion[] }[] {
  const groups: { category: string; items: Suggestion[] }[] = [];
  let current: { category: string; items: Suggestion[] } | null = null;

  for (const s of suggestions) {
    const cat = s.category ?? '';
    if (!current || current.category !== cat) {
      current = { category: cat, items: [] };
      groups.push(current);
    }
    current.items.push(s);
  }

  return groups;
}

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

const RemoveButton = styled.span`
  display: inline-block;
  background-color: var(--thorium-danger-bg, #e74c3c);
  color: var(--thorium-button-text);
  border: 1px solid var(--thorium-danger-bg, #e74c3c);
  border-radius: 3px;
  padding: 1px 6px;
  margin: 1px 3px;
  font-family: monospace;
  font-size: 11px;
  cursor: pointer;

  &:hover {
    filter: brightness(1.15);
  }
`;

const PopulateButton = styled.span`
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

interface SuggestionPanelProps {
  suggestions: Suggestion[];
  onValueClick?: (
    field: string,
    value: string,
    isList?: boolean,
    isMapEntry?: boolean,
    isRemoval?: boolean,
    schema?: FieldSchema,
    isReplace?: boolean,
  ) => void;
}

const SuggestionPanel: React.FC<SuggestionPanelProps> = ({ suggestions, onValueClick }) => {
  const [collapsed, setCollapsed] = useState(false);

  if (suggestions.length === 0) return null;

  const hasCategories = suggestions.some((s) => s.category);
  const groups = hasCategories ? groupByCategory(suggestions) : [{ category: '', items: suggestions }];

  return (
    <Panel>
      <div style={{ ...headerStyle, marginBottom: collapsed ? 0 : 4 }}>
        <span style={headerLabelStyle}>Suggestions ({suggestions.length})</span>
        <ToggleButton onClick={() => setCollapsed((prev) => !prev)}>{collapsed ? 'Show' : 'Hide'}</ToggleButton>
      </div>
      {!collapsed &&
        groups.map((group) => (
          <React.Fragment key={group.category}>
            {hasCategories && group.category && <SectionTitle>{group.category}</SectionTitle>}
            {group.items.map((suggestion, idx) => {
              const badge = typeLabel(suggestion.schema);
              const label = displayFieldName(suggestion.field);
              return (
                <SuggestionRow key={`${suggestion.field}-${idx}`}>
                  <FieldLabel title={suggestion.field}>{label}</FieldLabel>
                  {badge && <TypeBadge>{badge}</TypeBadge>}
                  <Message>{suggestion.message}</Message>
                  {suggestion.isRemoval ? (
                    <RemoveButton
                      onClick={() => onValueClick?.(suggestion.field, '', undefined, undefined, true, undefined)}
                      title={`Remove '${suggestion.field}' field and all subkeys`}
                    >
                      Remove
                    </RemoveButton>
                  ) : suggestion.isReplace ? (
                    <PopulateButton
                      onClick={() =>
                        onValueClick?.(suggestion.field, '', suggestion.isList, suggestion.isMapEntry, undefined, suggestion.schema, true)
                      }
                      title={`Populate '${suggestion.field}' with default structure`}
                    >
                      Populate
                    </PopulateButton>
                  ) : suggestion.values && suggestion.values.length > 0 && !suggestion.schema?.variants ? (
                    <ValuesContainer>
                      {suggestion.values.map((val) => (
                        <ValueChip
                          key={val}
                          onClick={() =>
                            onValueClick?.(suggestion.field, val, suggestion.isList, suggestion.isMapEntry, undefined, suggestion.schema)
                          }
                          title={`Click to use '${val}'`}
                        >
                          {val}
                        </ValueChip>
                      ))}
                    </ValuesContainer>
                  ) : (
                    <AddButton
                      onClick={() =>
                        onValueClick?.(suggestion.field, '', suggestion.isList, suggestion.isMapEntry, undefined, suggestion.schema)
                      }
                      title={`Add '${suggestion.field}' field`}
                    >
                      Add
                    </AddButton>
                  )}
                </SuggestionRow>
              );
            })}
          </React.Fragment>
        ))}
    </Panel>
  );
};

export default SuggestionPanel;
