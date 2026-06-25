import OmnibarDropdown from './Dropdown';
import { DropdownState, EditSession, KeyName, NewEditSessionClean, OmnibarEditMode } from './EditingTypes';
import {
  AddConditionToClauseDraft,
  AddFieldToClauseDraft,
  ToggleValueInClause,
  Clause,
  ClauseCondition,
  ClauseDraft,
  ClauseIsMulti,
  CondIsMulti,
  ConvertClauseToDraft,
  ConvertDraftToClause,
  DraftIsComplete,
  NewTextClause,
  ToggleValueInClauseDraft,
  AddCategoryToClauseDraft,
  parseClauseCondition,
} from './ClauseTypes';
import OmnibarEntryField from './OmnibarEntryField';
import { useRef, useState, KeyboardEvent } from 'react';
import styled from 'styled-components';
import { getDropdownOptions } from './utils';
import { OmnibarOptionMap } from './options';

const OmnibarContainer = styled.div`
  width: 100%;
  position: relative;
`;

type OmnibarProps = {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  dropdownOptions: OmnibarOptionMap;
  placeholder?: string;
  timepickerVisible?: boolean;
};

const Omnibar: React.FC<OmnibarProps> = ({
  clauses,
  dropdownOptions,
  setClauses,
  placeholder = 'Enter a query...',
  timepickerVisible = false,
}) => {
  const [editingState, setEditingState] = useState<EditSession>({ mode: OmnibarEditMode.Idle });
  const [dropdownState, setDropdownState] = useState<DropdownState>({ index: 0, isSelecting: false });
  const containerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const filteredDropdownOptions = getDropdownOptions(clauses, editingState, dropdownOptions, getCurrPartial());

  function setNewEditSession() {
    setEditingState(NewEditSessionClean());
  }

  function onNewClauseEnter(next: string) {
    //Called on an 'Enter' keypress or a dropdown select (enter or click)
    if (editingState.mode !== OmnibarEditMode.EditNew) return;
    if (editingState.part === 'category') {
      newClauseEnterCategory(next);
    } else if (editingState.part === 'field') {
      newClauseEnterField(next);
    } else if (editingState.part === 'condition') {
      newClauseEnterCondition(next);
    } else if (editingState.part === 'value') {
      newClauseEnterValue(next);
    }
  }

  function newClauseEnterCategory(next: string) {
    if (editingState.mode !== OmnibarEditMode.EditNew) return;
    let newEditState: EditSession = { ...editingState, textDraft: '' };
    let newDraft = { ...newEditState.clauseDraft };
    if (!Object.hasOwn(dropdownOptions, next)) {
      //if category does not exist, attempt to add as
      //text clause and return
      addTextClauseAndReset(next);
      return;
    }

    //category exists
    const categoryObj = dropdownOptions[next];
    const fieldKeys = Object.keys(categoryObj.fields);
    newDraft = AddCategoryToClauseDraft(newDraft, next);
    if (fieldKeys.length == 1) {
      //only one field. auto set and move to conditions
      const field = fieldKeys[0];
      newDraft = AddFieldToClauseDraft(newDraft, field);
      const fieldObj = categoryObj.fields[field];
      const conditions = fieldObj.conditions;
      if (conditions.length == 1) {
        //one condition, auto set it and move to value
        newEditState.part = 'value';
        newDraft = AddConditionToClauseDraft(newDraft, conditions[0]);
      } else {
        //multiple conditions. need to select
        newEditState.part = 'condition';
      }
    } else {
      //multiple field keys. need to select
      newEditState.part = 'field';
    }
    newEditState = { ...newEditState, clauseDraft: newDraft };
    setDropdownState({ index: 0, isSelecting: false });
    setEditingState(newEditState);
  }

  function newClauseEnterField(next: string) {
    //NOTE: right now just used for tags (key that share category)
    if (editingState.mode !== OmnibarEditMode.EditNew) return;
    const newEditState: EditSession = { ...editingState, textDraft: '' };
    const category = editingState.clauseDraft.category!;
    const categoryObj = dropdownOptions[category];
    const fieldOpts = categoryObj.fields;
    // resolve the field's conditions: a known field uses its defined conditions; an unknown field is
    // allowed only when the category permits creatable keys (e.g. user-entered tag keys), defaulting
    // to a single `Is` condition
    let conditions: ClauseCondition[];
    if (Object.hasOwn(fieldOpts, next)) {
      conditions = fieldOpts[next].conditions;
    } else if (categoryObj.fieldsCreatable) {
      conditions = [ClauseCondition.Is];
    } else {
      console.error('field does not exist, returning');
      return;
    }
    newEditState.clauseDraft = AddFieldToClauseDraft(newEditState.clauseDraft, next);

    if (conditions.length == 1) {
      //one condition. auto set and move on to value
      newEditState.part = 'value';
      newEditState.clauseDraft = AddConditionToClauseDraft(newEditState.clauseDraft, conditions[0]);
    } else {
      newEditState.part = 'condition';
    }
    setDropdownState({ index: 0, isSelecting: false });
    setEditingState(newEditState);
  }

  function newClauseEnterCondition(next: string) {
    if (editingState.mode !== OmnibarEditMode.EditNew) return;
    const newEditState: EditSession = { ...editingState, textDraft: '' };
    const cond = parseClauseCondition(next);

    if (cond === undefined) return;

    newEditState.part = 'value';
    newEditState.clauseDraft = AddConditionToClauseDraft(newEditState.clauseDraft, cond);

    setDropdownState({ index: 0, isSelecting: false });
    setEditingState(newEditState);
  }

  function newClauseEnterValue(next: string) {
    if (editingState.mode !== OmnibarEditMode.EditNew) return;
    const newEditState: EditSession = { ...editingState, textDraft: '' };
    const condition = editingState.clauseDraft.condition!;
    //toggle value. this will add to list if multi or set value if single
    newEditState.clauseDraft = ToggleValueInClauseDraft(newEditState.clauseDraft, next);
    if (!CondIsMulti(condition) && DraftIsComplete(newEditState.clauseDraft)) {
      //only finish on enter if single value. will need to tab out on multi condition
      setClauses([...clauses, ConvertDraftToClause(newEditState.clauseDraft)]);
      //reset new edit state
      setDropdownState({ index: 0, isSelecting: false });
      setNewEditSession();
      return;
    }
    setEditingState(newEditState);
  }

  function addTextClauseAndReset(text: string) {
    if (text === '') return;
    //NOTE: only enter text if text is a valid option
    if (Object.hasOwn(dropdownOptions, 'text')) {
      setClauses([...clauses, NewTextClause(text)]);
      setNewEditSession();
    }
  }

  function onEditClauseEnter(next: string) {
    //when 'Enter' keypress or option is selected for an already existing clause
    if (editingState.mode !== OmnibarEditMode.EditExisting) return;
    //only value is editable
    if (editingState.part !== 'value') return;

    const oldClause = editingState.clauseDraft;
    const newClause = ToggleValueInClause(oldClause, next);

    if (ClauseIsMulti(oldClause)) {
      setEditingState({ ...editingState, clauseDraft: newClause });
    } else {
      updateClauses(newClause, editingState.clauseIdx);
      setEditingState({ mode: OmnibarEditMode.Idle });
    }
  }

  function updateClauses(newClause: Clause, clauseIdx: number) {
    const newClauses = clauses.map((clause, idx) => (idx === clauseIdx ? newClause : clause));
    //remove any empty clauses
    const filterClauses = newClauses.filter((clause) => {
      if (ClauseIsMulti(clause)) {
        return clause.value.values.length > 0;
      } else {
        return clause.value.value.length > 0;
      }
    });
    setClauses(filterClauses);
  }

  function handleDropdownSelect(idx: number) {
    //function called when 'Enter' pressed on dropdown selection or
    //an option is clicked
    const selectedOption = filteredDropdownOptions[idx];
    if (editingState.mode == OmnibarEditMode.EditNew) {
      if (selectedOption.value == 'text' && editingState.textDraft.length > 0) {
        addTextClauseAndReset(editingState.textDraft);
        return;
      }
      onNewClauseEnter(selectedOption.value);
      return;
    } else if (editingState.mode == OmnibarEditMode.EditExisting) {
      onEditClauseEnter(selectedOption.value);
      return;
    }
    setEditingState({ mode: OmnibarEditMode.Idle });
  }

  function getCurrPartial(): ClauseDraft {
    //helper function to get the current partial clause draft
    if (editingState.mode === OmnibarEditMode.Idle) return {};
    if (editingState.mode === OmnibarEditMode.EditExisting) return ConvertClauseToDraft(editingState.clauseDraft);
    return editingState.clauseDraft;
  }

  function handleKeypress(e: KeyboardEvent<HTMLInputElement>) {
    if (editingState.mode == OmnibarEditMode.Idle) return;

    switch (e.key as KeyName) {
      case KeyName.Enter:
        if (dropdownState.isSelecting) {
          handleDropdownSelect(dropdownState.index);
        } else if (editingState.mode === OmnibarEditMode.EditNew) {
          onNewClauseEnter(editingState.textDraft);
        } else {
          onEditClauseEnter(editingState.textDraft);
        }
        e.preventDefault();
        break;
      case KeyName.ArrowDown:
        e.preventDefault();
        if (dropdownState.isSelecting && dropdownState.index < filteredDropdownOptions.length - 1) {
          setDropdownState({ index: dropdownState.index + 1, isSelecting: true });
        } else if (!dropdownState.isSelecting) {
          //always start at top
          setDropdownState({ index: 0, isSelecting: true });
        }
        break;
      case KeyName.ArrowUp:
        e.preventDefault();
        if (dropdownState.index > 0) {
          setDropdownState({ index: dropdownState.index - 1, isSelecting: true });
        } else {
          setDropdownState({ index: 0, isSelecting: false });
        }
        break;
      case KeyName.ArrowRight:
      case KeyName.Tab:
        HandleTab(e);
        break;
      default:
        setDropdownState({ index: 0, isSelecting: false });
    }
  }

  function HandleTab(e: KeyboardEvent<HTMLInputElement>) {
    if (editingState.mode == OmnibarEditMode.Idle) return;
    if (editingState.part !== 'value') return;

    if (editingState.mode == OmnibarEditMode.EditNew) {
      const draft = { ...editingState.clauseDraft };
      if (editingState.textDraft.length > 0) {
        //WARN: might have unintended consequences. test
        draft.value = editingState.textDraft;
      }
      if (DraftIsComplete(draft)) {
        setClauses([...clauses, ConvertDraftToClause(draft)]);
      }
      setNewEditSession();
      e.preventDefault();
    } else if (editingState.mode == OmnibarEditMode.EditExisting) {
      setClauses(clauses.map((clause, idx) => (idx === editingState.clauseIdx ? editingState.clauseDraft : clause)));
      setEditingState({ mode: OmnibarEditMode.Idle });
      e.preventDefault();
    }
  }

  function OnBlurNew() {
    //commit if partial is full
    if (editingState.mode !== OmnibarEditMode.EditNew) return;
    if (DraftIsComplete(editingState.clauseDraft)) {
      setClauses([...clauses, ConvertDraftToClause(editingState.clauseDraft)]);
    } else {
      //if partial would be full with typed text, commit
      if (editingState.part == 'value' && editingState.textDraft != '') {
        const newClauseDraft = ToggleValueInClauseDraft(editingState.clauseDraft, editingState.textDraft);
        if (DraftIsComplete(newClauseDraft)) {
          setClauses([...clauses, ConvertDraftToClause(newClauseDraft)]);
        }
      }
      //if no category selected and 'text' is valid category, create text clause
      if (editingState.part == 'category' && editingState.textDraft != '') {
        addTextClauseAndReset(editingState.textDraft);
      }
    }
  }

  function OnBlurExisting() {
    if (editingState.mode !== OmnibarEditMode.EditExisting) return;
    let newClause = editingState.clauseDraft;
    if (!ClauseIsMulti(editingState.clauseDraft)) {
      if (editingState.textDraft !== editingState.clauseDraft.value.value) {
        newClause = { ...editingState.clauseDraft, value: { value: editingState.textDraft } };
      }
    }
    updateClauses(newClause, editingState.clauseIdx);
  }

  const dropdownVisible = editingState.mode != OmnibarEditMode.Idle && filteredDropdownOptions.length > 0;

  return (
    <OmnibarContainer
      ref={containerRef}
      onBlur={(e) => {
        const next = e.relatedTarget;
        const focusStaysInOmnibar = !!(next && containerRef.current?.contains(next));
        const focusOnDropdown = !!(next && dropdownRef.current?.contains(next));
        if (!focusOnDropdown) {
          //want dropdown code to handle if we click on dropdown.
          //if not focusing on dropdown, handle blur as normal
          OnBlurNew();
          OnBlurExisting();
        }
        if (!focusStaysInOmnibar) {
          setEditingState({ mode: OmnibarEditMode.Idle });
          setDropdownState({ index: 0, isSelecting: false });
        }
      }}
    >
      <OmnibarEntryField
        clauses={clauses}
        setClauses={setClauses}
        editingState={editingState}
        setEditingState={setEditingState}
        keyHandler={handleKeypress}
        defaultOptions={dropdownOptions}
        blankPlaceholder={placeholder}
        dropdownVisible={dropdownVisible}
        timepickerVisible={timepickerVisible}
      />
      {dropdownVisible && (
        <OmnibarDropdown
          ref={dropdownRef}
          options={filteredDropdownOptions}
          dropdownState={dropdownState}
          setFocusIdx={(newIdx) => setDropdownState({ index: newIdx, isSelecting: true })}
          onSelect={handleDropdownSelect}
          onMouseLeave={() => setDropdownState({ index: 0, isSelecting: false })}
        />
      )}
    </OmnibarContainer>
  );
};
export default Omnibar;
