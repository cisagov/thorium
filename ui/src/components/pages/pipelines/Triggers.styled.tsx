import styled from 'styled-components';

/// The outer container holding the list of trigger cards and the add button
export const Panel = styled.div`
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  padding: 8px 12px;
`;

/// A single trigger's editable card
export const TriggerCard = styled.div`
  border: 1px solid var(--thorium-highlight-panel-border);
  border-radius: 4px;
  padding: 10px 12px;
  margin-bottom: 10px;
`;

/// The card header row holding the name field, type field, and delete column.
/// Top-aligned so the delete column lines up with the inputs (not the labels).
export const CardHeader = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
`;

/// A flexible field column (name / type) within the card header
export const HeaderField = styled.div`
  flex: 1;
`;

/// The fixed column holding the delete button, kept narrow so the fields take the rest
export const DeleteCol = styled.div`
  display: flex;
  flex-direction: column;
  flex: 0 0 auto;
`;

/// A small danger-styled icon button (delete / remove)
export const IconButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 38px;
  height: 34px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-danger-bg);
  color: var(--thorium-button-text);
  cursor: pointer;
`;

/// An invisible placeholder matching IconButton's footprint so key/value rows keep a
/// constant input width whether or not a delete button is present on the row.
export const ButtonSpacer = styled.div`
  flex: 0 0 auto;
  width: 38px;
  height: 34px;
  visibility: hidden;
`;

/// A labeled section (Tag Types / Required / Not) within a Tag trigger card
export const FilterSection = styled.div`
  margin: 6px 0;
`;

/// A single key/value row; its two inputs share the available width and a fixed
/// delete column keeps that width stable across rows.
export const FilterRow = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 4px;

  > input {
    flex: 1 1 0;
    min-width: 0;
  }
`;

/// The compact add button (matches the image create form's section add buttons)
export const AddButton = styled.button`
  padding: 4px 16px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-ok-bg);
  color: var(--thorium-button-text);
  font-weight: 700;
  cursor: pointer;
`;

/// The note shown when no triggers are configured
export const EmptyNote = styled.div`
  color: var(--thorium-secondary-text);
  font-size: 13px;
  margin-bottom: 8px;
`;
