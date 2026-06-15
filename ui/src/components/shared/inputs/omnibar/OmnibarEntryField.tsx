import React, { KeyboardEvent, ClipboardEvent } from 'react';
import { EditSession, NewEditSessionClean, OmnibarEditMode } from './EditingTypes';
import { OmnibarClause } from './Clause';
import { styled } from 'styled-components';
import { OmnibarNewClause } from './NewClause';
import { Clause, ClauseDraft, ClauseIsMulti, CondIsMulti, GetMostSpecificCondition, GetValidFields } from './ClauseTypes';
import { OmnibarOptionMap } from './options';

const OmnibarEntryFieldDiv = styled.div<{ $dropdownVisible: boolean; $timepickerVisible: boolean }>`
  border: 1px solid var(--thorium-omnibar-border);
  color: var(--thorium-text);
  background-color: var(--thorium-omnibar-bg);
  display: flex;
  flex-wrap: nowrap;
  min-height: 40px;
  cursor: text;
  overflow: hidden;
  padding: 2px 5px;
  border-radius: 5px;

  ${({ $dropdownVisible }) =>
    $dropdownVisible &&
    `
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
    `}
  ${({ $timepickerVisible }) =>
    $timepickerVisible &&
    `
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    `}
`;

const ClauseContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  flex: 1 1 auto;
  align-items: flex-start;
  align-content: center;
  overflow: auto;
`;

type OmnibarEntryFieldProps = {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  editingState: EditSession;
  setEditingState: (next: EditSession) => void;
  keyHandler: (e: KeyboardEvent<HTMLInputElement>) => void;
  defaultOptions: OmnibarOptionMap;
  blankPlaceholder: string;
  dropdownVisible: boolean;
  timepickerVisible: boolean;
  /** Extra removable tiles rendered after the clause chips, before the new-clause input. */
  extraChips?: React.ReactNode;
};

const OmnibarEntryField: React.FC<OmnibarEntryFieldProps> = ({
  clauses,
  setClauses,
  editingState,
  setEditingState,
  keyHandler,
  defaultOptions,
  blankPlaceholder,
  dropdownVisible,
  timepickerVisible,
  extraChips,
}) => {
  const beginEdit = (idx: number) => {
    const clause = clauses[idx];
    const draft = ClauseIsMulti(clause) ? '' : clause.value.value;
    setEditingState({
      mode: OmnibarEditMode.EditExisting,
      textDraft: draft,
      //hardcoded -- only value can be edited.
      part: 'value',
      clauseIdx: idx,
      clauseDraft: structuredClone(clause),
    });
  };

  const onDelete = (idx: number) => {
    setClauses(clauses.filter((_clause, index) => index !== idx));
  };

  const resetFocus = () => {
    setEditingState({ mode: OmnibarEditMode.Idle });
  };

  const onContainerClick = () => {
    if (editingState.mode == OmnibarEditMode.Idle) {
      setEditingState(NewEditSessionClean());
    }
  };

  const handlePaste = (e: ClipboardEvent<HTMLInputElement>) => {
    if (editingState.mode !== OmnibarEditMode.EditNew) return;
    //only work on field or category
    if (!['field', 'category'].includes(editingState.part)) return;

    const newEditState = { ...editingState };

    const clipboardText = e.clipboardData.getData('text');
    const enteredText = (e.target as HTMLInputElement).value;
    const cursorPosition = (e.target as HTMLInputElement).selectionStart;
    //if text is already entered and cursor position is not at end just paste as normal
    if (enteredText.length > 0 && cursorPosition !== enteredText.length) return;
    //combine text
    const fullText = enteredText + clipboardText;
    const cond = GetMostSpecificCondition(fullText);
    if (cond === undefined) return;

    const split = fullText.split(cond);
    const clipboardKey = split[0].trim();
    const value = split.slice(1).join(cond); //if condition is present in value

    let validFieldList = GetValidFields(defaultOptions);

    if (editingState.part == 'field') {
      //filter fields that already have category
      //should only work with tags (a category that has multiple fields)
      validFieldList = validFieldList.filter((v) => v.category == editingState.clauseDraft.category);
      if (validFieldList.length == 0) return;
    }

    const fieldMatches = validFieldList.filter((v) => v.field == clipboardKey);
    //no match, paste normal
    if (fieldMatches.length == 0) return;
    if (fieldMatches.length > 1) {
      //multiple fields have same name (undefined behavior)
      return;
    }

    const fieldMatch = fieldMatches[0];
    const category = fieldMatch.category;
    const field = fieldMatch.field;
    const opt = defaultOptions[category].fields[field];

    if (!opt.conditions.includes(cond)) return; //invalid condition

    const newClause: ClauseDraft = {
      category: category,
      field: clipboardKey,
      condition: cond,
    };
    if (CondIsMulti(cond)) {
      const valueSplit = value.split(',');
      newClause.values = valueSplit.map((v) => v.trim());
      newEditState.textDraft = '';
    } else {
      newClause.value = value.trim();
      newEditState.textDraft = value.trim();
    }
    newEditState.clauseDraft = newClause;
    newEditState.part = 'value';
    setEditingState(newEditState);
    e.preventDefault();
  };

  const clauseItems = clauses.map((clause, idx) => {
    const isEditing = editingState.mode === OmnibarEditMode.EditExisting && editingState.clauseIdx == idx;
    return (
      <OmnibarClause
        key={idx}
        clause={clause}
        draft={isEditing ? editingState.textDraft : ''}
        editingMode={isEditing ? editingState.part : null}
        setDraft={(next) => {
          if (isEditing) {
            setEditingState({ ...editingState, textDraft: next });
          }
        }}
        onKeyHandler={keyHandler}
        onBeginEdit={() => beginEdit(idx)}
        onDelete={() => onDelete(idx)}
        onFocusButton={resetFocus}
      />
    );
  });

  return (
    <OmnibarEntryFieldDiv $dropdownVisible={dropdownVisible} $timepickerVisible={timepickerVisible}>
      <ClauseContainer onClick={onContainerClick}>
        {clauseItems}
        {extraChips}
        <OmnibarNewClause
          editingState={editingState}
          setEditingState={setEditingState}
          onKeyHandler={keyHandler}
          onPaste={handlePaste}
          placeholder={blankPlaceholder}
        />
      </ClauseContainer>
    </OmnibarEntryFieldDiv>
  );
};

export default OmnibarEntryField;
