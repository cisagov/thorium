import { KeyboardEvent, useRef, FocusEvent } from 'react';
import { Tag } from './tag';
import { createReactSelectStyles, tagEntryIsEmpty } from '@utilities';
import { TagEntry } from 'models';
import { styled } from 'styled-components';
import { EditingMode, FocusState } from './editing_types';
import { updateTags } from './utilities';
import TagEntryButtons from './tag_entry_buttons';

const TagEntryFieldDiv = styled.div`
  border: 1px solid var(--thorium-panel-border);
  color: var(--thorium-text);
  background-color: var(--thorium-secondary-panel-bg);
  border-radius: var(--bs-border-radius);
  display: flex;
  flex-wrap: nowrap;
  min-height: 40px;
  cursor: text;
  overflow: hidden;
  padding: 2px 5px;
`;

const TagContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  flex: 1 1 auto;
  align-items: flex-start;
  align-content: center;
  overflow: auto;
`;

const PlaceholderSpan = styled.span`
  color: var(--thorium-secondary-text);
  display: flex;
  align-items: center;
  padding-bottom: 2px;
  padding-left: 4px;
  margin: 2px;
  margin-left: 8px;
`;

type TagEntryProps = {
  tags: TagEntry[];
  placeholderText: string;
  editState: FocusState;
  setEditState: (newState: FocusState) => void;
  handleKeypress: (e: KeyboardEvent, idx: number, s: EditingMode) => void;
  setTags: (newTags: TagEntry[]) => void;
  handleBlur: (e: FocusEvent) => void;
  focusNewTag: () => void;
  handleDeleteAll: () => void;
};

const TagEntryField: React.FC<TagEntryProps> = ({
  tags,
  editState,
  placeholderText,
  setEditState,
  setTags,
  handleBlur,
  handleKeypress,
  focusNewTag,
  handleDeleteAll,
}) => {
  const TagEntryRef = useRef<HTMLDivElement>(null);

  function handleChange(tagIdx: number, newTag: Partial<TagEntry>) {
    const newTags = updateTags(tags, tagIdx, newTag);
    setTags(newTags);
  }

  function handleFocus(idx: number, editMode: EditingMode) {
    setEditState({ idx: idx, editMode: editMode });
  }

  function deleteKey(idx: number) {
    setTags(tags.filter((_tag, i) => i !== idx));
    if (editState.editMode !== EditingMode.Disabled) {
      setEditState({ idx: -1, editMode: EditingMode.Disabled });
    }
  }

  function handleFocusTrash() {
    setEditState({ idx: -1, editMode: EditingMode.Disabled });
  }

  //List of all the current tags
  const tagItems = tags.map((tag, idx) => {
    return (
      <Tag
        key={idx}
        data={tag}
        editMode={editState.idx === idx ? editState.editMode : EditingMode.Disabled}
        onChange={(newTag) => handleChange(idx, newTag)}
        handleKeyDown={(e, state) => handleKeypress(e, idx, state)}
        handleFocus={(state) => handleFocus(idx, state)}
        handleDelete={() => deleteKey(idx)}
        handleBlur={handleBlur}
        handleFocusTrash={handleFocusTrash}
      />
    );
  });

  //placeholder text. no tags, not editing, no text in current tag
  const noTagItem = tags.length === 0 && editState.editMode == EditingMode.Disabled && (
    <PlaceholderSpan
      tabIndex={0} //so tab can focus on it
      onFocus={() => focusNewTag()}
    >
      {placeholderText}
    </PlaceholderSpan>
  );

  function handleClick() {
    if (tags.length > 0 && editState.idx == tags.length - 1 && tagEntryIsEmpty(tags[tags.length - 1])) {
      //already focused on last tag and currently editing.
      //Instead of making new tag just stop editing.
      setEditState({ editMode: EditingMode.Disabled, idx: -1 });
      return;
    }
    focusNewTag();
  }

  return (
    <TagEntryFieldDiv
      ref={TagEntryRef}
      aria-label="Tag Entry Field"
      role="listbox"
      style={createReactSelectStyles('White', 'rgb(160, 162, 163)')}
    >
      <TagContainer tabIndex={-1} onClick={() => handleClick()}>
        {tagItems}
        {noTagItem}
      </TagContainer>
      <TagEntryButtons tagLength={tags.length} handleBlur={handleBlur} handleDeleteAll={handleDeleteAll} handleDownArrow={handleClick} />
    </TagEntryFieldDiv>
  );
};
export default TagEntryField;
