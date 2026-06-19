import { ChangeEvent, KeyboardEvent, ClipboardEvent, useEffect, useRef } from 'react';
import { EditSession, NewEditSessionClean, OmnibarEditMode } from './EditingTypes';
import { styled, css } from 'styled-components';
import { ClauseComponentInput } from './ClauseComponent';
import { CategoryLogo } from './CategoryLogo';

const NewClauseContainer = styled.div<{ $isActive: boolean }>`
  color: var(--thorium-text);
  background-color: inherit;
  display: inline-flex;
  cursor: text;
  overflow: hidden;
  padding: 2px 5px;
  align-items: center;
  justify-content: center;
  align-self: center;
  gap: 3px;
  ${({ $isActive }) =>
    $isActive &&
    css`
      border: 1px solid var(--thorium-text);
      border-radius: 5px;
    `}
`;

type OmnibarNewClauseProps = {
  editingState: EditSession;
  setEditingState: (next: EditSession) => void;
  onKeyHandler: (e: KeyboardEvent<HTMLInputElement>) => void;
  onPaste: (e: ClipboardEvent<HTMLInputElement>) => void;
  placeholder: string;
};
export const OmnibarNewClause: React.FC<OmnibarNewClauseProps> = ({
  editingState,
  setEditingState,
  onKeyHandler,
  onPaste,
  placeholder,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const activePart = editingState.mode === OmnibarEditMode.EditNew ? editingState.part : null;
  const draft = editingState.mode == OmnibarEditMode.EditNew ? editingState.textDraft : '';
  const isEditing = editingState.mode == OmnibarEditMode.EditNew;
  const category = isEditing ? editingState.clauseDraft.category : undefined;
  const field = isEditing ? editingState.clauseDraft.field : undefined;
  const condition = isEditing ? editingState.clauseDraft.condition! : undefined;

  let currPlaceholder = placeholder;
  if (isEditing) {
    switch (editingState.part) {
      case 'field':
        currPlaceholder = 'Enter a search field...';
        break;
      case 'condition':
        currPlaceholder = 'Enter a condition...';
        break;
      case 'value':
        currPlaceholder = 'Enter a value...';
        break;
    }
  }

  useEffect(() => {
    if (editingState.mode === OmnibarEditMode.EditNew) {
      inputRef.current?.focus();
    }
  }, [activePart]);

  const onFocus = () => {
    if (editingState.mode === OmnibarEditMode.EditNew) return;
    setEditingState(NewEditSessionClean());
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const next = e.target.value;
    if (editingState.mode !== OmnibarEditMode.EditNew) {
      return;
    }
    setEditingState({ ...editingState, textDraft: next });
  };

  return (
    <NewClauseContainer $isActive={isEditing}>
      {category && <CategoryLogo category={category} />}
      {field && field !== 'text' && (
        //only show field if set and we're not doing just text
        <>
          <span>{field}</span>
          {condition && <span>{condition}</span>}
        </>
      )}
      <ClauseComponentInput
        ref={inputRef}
        placeholder={currPlaceholder}
        value={draft}
        onFocus={onFocus}
        onChange={onChange}
        onKeyDown={onKeyHandler}
        onPaste={onPaste}
      />
    </NewClauseContainer>
  );
};
