import React, { useState, useEffect, KeyboardEvent, useRef, FocusEvent } from 'react';
import TagDropdown from './dropdown';
import TagEntryField from './tag_entry_field';
import { EditingMode, FocusState, KeyName } from './editing_types';
import { TagEntry, TagOptions } from 'models';
import { loadTagOptionsFromLocalStorage, tagEntryIsEmpty } from '@utilities';
import { Modal } from 'react-bootstrap';
import { Button } from '@mui/material';
import { getCurrTypedText, getDropdownOptions, updateTags } from './utils';
import styled from 'styled-components';

const TagSelectContainer = styled.div`
  width: 100%;
  position: relative;
`;

export type TagSelectProps = {
  tags: TagEntry[];
  placeholderText: string;
  setTags: (v: TagEntry[]) => void;
};

const TagSelect: React.FC<TagSelectProps> = ({ tags, setTags, placeholderText = 'Enter Tags' }) => {
  //State for keeping track of focus. Which tag is being edited (idx) and editing mode (disabled, first (key), second (value))
  const [focusState, setFocusState] = useState<FocusState>({ editMode: EditingMode.Disabled, idx: -1 });
  //dropdown state. Data for creating options, highlighted option idx
  const [dropdownData, setDropdownData] = useState<TagOptions>();
  //in special cases can be -1 (No text typed in current focus input. Used for tabbing out)
  const [selectedDropdownOptionIdx, setSelectedDropdownOptionIdx] = useState(0);
  //Delete modal for deleting all tags.
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const tagSelectRef = useRef<HTMLDivElement>(null);

  //computed properties
  const currTypedText = getCurrTypedText(tags, focusState);
  const dropdownOptions = getDropdownOptions(tags, dropdownData, focusState, currTypedText);

  //get dropdown data on component load
  useEffect(() => {
    let timeoutFn: ReturnType<typeof setTimeout>;
    const loadDropdownData = () => {
      const data = loadTagOptionsFromLocalStorage();
      if (data) {
        setDropdownData(data);
      } else {
        timeoutFn = setTimeout(() => {
          loadDropdownData();
        }, 1000);
      }
    };
    loadDropdownData();
    //needed to clean up on component unmount
    return () => {
      if (timeoutFn) {
        clearTimeout(timeoutFn);
      }
    };
  }, []);

  //always reset the dropdown focus on any typing
  useEffect(() => {
    resetDropdownOptionIdx();
  }, [focusState.editMode, currTypedText]);

  //helper function to reset dropdown option
  function resetDropdownOptionIdx() {
    if (currTypedText.length == 0) {
      //want -1 so if we can tab out without selecting an option.
      setSelectedDropdownOptionIdx(-1);
    } else {
      setSelectedDropdownOptionIdx(0);
    }
  }

  function createNewTagAndFocus(currTags: TagEntry[] | null = null) {
    const t = currTags ? currTags : tags;
    const tagLength = t.length;
    //if last tag is empty, just focus there
    if (t.length > 0 && tagEntryIsEmpty(t[t.length - 1])) {
      setTags(t);
      setFocusState({ idx: tagLength - 1, editMode: EditingMode.First });
      return;
    }
    setTags([...t, { key: '', value: '' }]);
    setFocusState({ idx: tagLength, editMode: EditingMode.First });
  }

  function handleKeypress(e: KeyboardEvent, idx: number, editMode: EditingMode) {
    switch (e.key as KeyName) {
      case KeyName.Enter:
        handleEnter(e, idx, editMode);
        e.preventDefault();
        e.stopPropagation();
        break;
      case KeyName.Tab:
        handleTab(e, idx, editMode);
        break;
      case KeyName.ArrowDown:
        e.preventDefault();
        if (selectedDropdownOptionIdx < dropdownOptions.length - 1) {
          setSelectedDropdownOptionIdx(selectedDropdownOptionIdx + 1);
        }
        break;
      case KeyName.ArrowUp:
        e.preventDefault();
        if (selectedDropdownOptionIdx > 0) {
          setSelectedDropdownOptionIdx(selectedDropdownOptionIdx - 1);
        }
        break;
    }
  }

  function handleTab(e: KeyboardEvent, idx: number, editMode: EditingMode) {
    if (focusState.idx == tags.length - 1 && selectedDropdownOptionIdx < 0 && focusState.editMode == EditingMode.Second) {
      //want to tab as normal if tag empty and didn't select any options
      return;
    }
    handleTagUpdateAndFocusChange(e, idx, editMode);
  }

  function handleEnter(e: KeyboardEvent, idx: number, editMode: EditingMode) {
    if (e.shiftKey) return;
    handleTagUpdateAndFocusChange(e, idx, editMode);
  }

  /**
   * Handles tag update (and, if relevant) focus switching. Used for Tab and Enter
   * keypresses. If editing first part of tag, update and set focus to second part.
   * If editing second part, update. If it's the last tag, set focus to a new tag.
   * */
  function handleTagUpdateAndFocusChange(e: KeyboardEvent, idx: number, editMode: EditingMode) {
    const selectedOption = selectedDropdownOptionIdx >= 0 ? dropdownOptions[selectedDropdownOptionIdx] : '';
    if (editMode === EditingMode.First) {
      setTags(updateTags(tags, idx, { key: selectedOption }));
      //on enter need to change the focus state. Tab handles automatically
      if ((e.key as KeyName) == KeyName.Enter) {
        setFocusState({ ...focusState, editMode: EditingMode.Second });
      }
    } else if (editMode === EditingMode.Second) {
      const newTags = updateTags(tags, idx, { value: selectedOption });
      if (idx == tags.length - 1) {
        createNewTagAndFocus(newTags);
        e.preventDefault();
      } else {
        //not the last tag, just edit the tag list but don't change focus
        setTags(newTags);
        if ((e.key as KeyName) == KeyName.Enter) {
          setFocusState({ ...focusState, editMode: EditingMode.Disabled });
        }
      }
    }
  }

  function handleDropdownSelect(idx: number) {
    const selectedOption = dropdownOptions[idx];
    const editIdx = focusState.idx;
    if (focusState.editMode == EditingMode.First) {
      //edit the currently focused tag's key
      setTags(updateTags(tags, editIdx, { key: selectedOption }));
      setFocusState({ ...focusState, editMode: EditingMode.Second });
    } else if (focusState.editMode === EditingMode.Second) {
      //edit the currently focused tag's value
      //don't immediately call setTags because we need to pass this in to create New tag
      const currTags = updateTags(tags, editIdx, { value: selectedOption });
      if (focusState.idx == tags.length - 1) {
        //if this is the last tag we want to make a new tag and focus on it
        //(so user doesn't keep clicking for each new tag)
        createNewTagAndFocus(currTags);
      } else {
        //if it's not the new tag we'll update the tags and stop editing.
        setTags(currTags);
        setFocusState({ idx: -1, editMode: EditingMode.Disabled });
      }
    }
  }

  function handleSelectBlur(e: FocusEvent) {
    const next = e.relatedTarget;
    const focusStaysInTagSelect = !!(next && tagSelectRef.current?.contains(next));
    //leaving the tag select component
    if (!focusStaysInTagSelect) {
      //remove empty tags when leaving component
      setTags(tags.filter((tag) => !tagEntryIsEmpty(tag)));
      setFocusState({ idx: -1, editMode: EditingMode.Disabled });
      return;
    }
  }

  function handleModalButtonDeleteClick() {
    setFocusState({ idx: -1, editMode: EditingMode.Disabled });
    setTags([]);
    setShowDeleteModal(false);
  }

  return (
    <>
      <Modal show={showDeleteModal} onHide={() => setShowDeleteModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Confirm deletion</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Do you really want to delete all tags?</p>
        </Modal.Body>
        <Modal.Footer style={{ gap: '10px' }} className="d-flex justify-content-center">
          <Button className="danger-btn" onClick={() => handleModalButtonDeleteClick()}>
            Confirm
          </Button>
          <Button className="primary-btn" onClick={() => setShowDeleteModal(false)}>
            Cancel
          </Button>
        </Modal.Footer>
      </Modal>
      <TagSelectContainer ref={tagSelectRef}>
        <TagEntryField
          tags={tags}
          editState={focusState}
          placeholderText={placeholderText}
          setEditState={(newState) => setFocusState(newState)}
          setTags={(tagList) => setTags(tagList)}
          handleKeypress={handleKeypress}
          handleBlur={handleSelectBlur}
          focusNewTag={() => createNewTagAndFocus()}
          handleDeleteAll={() => setShowDeleteModal(true)}
        />
        {focusState.editMode !== EditingMode.Disabled && dropdownOptions.length > 0 && (
          <TagDropdown
            focusIdx={selectedDropdownOptionIdx}
            setFocusIdx={(idx) => setSelectedDropdownOptionIdx(idx)}
            options={dropdownOptions}
            onSelect={handleDropdownSelect}
          />
        )}
      </TagSelectContainer>
    </>
  );
};
export default TagSelect;
