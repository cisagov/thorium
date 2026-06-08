import { KeyboardEvent } from 'react';
import { styled } from 'styled-components';
import { FaX } from 'react-icons/fa6';
import { ClauseComponent, ClauseComponentSpan, ClauseComponentStatic } from './ClauseComponent';
import { getClauseColorClass } from './utils';
import { Clause, ClausePart, GetValueString } from './ClauseTypes';
import { CategoryLogo } from './CategoryLogo';

const DeleteButton = styled.button`
  padding: 0 4px;
  border: 0px;
  height: 100%;
  background-color: inherit;
  color: inherit;
`;

const ClauseContainer = styled.div`
  border-radius: 5px;
  border: 1px solid var(--thorium-text);
  padding: 2px 5px;
  width: auto;
  margin: 2px;
  box-sizing: content-box;
  letter-spacing: normal;

  display: inline-flex;
  align-items: center;
  align-self: center;
  justify-content: center;
  gap: 2px;
`;

type OmnibarClauseProps = {
  clause: Clause;
  draft: string;
  editingMode: ClausePart | null;
  setDraft: (next: string) => void;
  onKeyHandler: (e: KeyboardEvent<HTMLInputElement>) => void;
  onBeginEdit: () => void;
  onDelete: () => void;
  onFocusButton: () => void;
};

export const OmnibarClause: React.FC<OmnibarClauseProps> = ({
  clause,
  draft,
  editingMode,
  setDraft,
  onKeyHandler,
  onBeginEdit,
  onDelete,
  onFocusButton,
}) => {
  let tagClass = '';
  tagClass = getClauseColorClass(clause);
  return (
    <ClauseContainer className={tagClass} onClick={(e) => e.stopPropagation()}>
      <CategoryLogo category={clause.category} />
      {clause.field !== 'text' && (
        //if a 'text' field, we hide field and comparison and just show the value
        <>
          <ClauseComponentStatic value={clause.field} />
          <ClauseComponentSpan style={{ textDecoration: 'underline' }}>{clause.condition}</ClauseComponentSpan>
        </>
      )}
      <ClauseComponent
        value={GetValueString(clause)}
        draft={draft}
        setDraft={setDraft}
        placeholder="value"
        isEditing={editingMode === 'value'}
        onBeginEdit={onBeginEdit}
        onKeyDown={onKeyHandler}
      />
      <DeleteButton title="delete clause" onClick={onDelete} onFocus={onFocusButton}>
        <FaX size={12} />
      </DeleteButton>
    </ClauseContainer>
  );
};
