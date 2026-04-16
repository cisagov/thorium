import { useEffect, useRef, KeyboardEvent, FocusEvent, ClipboardEvent, useLayoutEffect } from 'react';
import { styled } from 'styled-components';
import { FaX } from 'react-icons/fa6';

// project imports
import { EditingMode } from './models';
import { TagEntry } from '@models/tags';
import { tagIsInvalid } from '@utilities/tags';
import { getTagColorClass } from '@components/tags/utilities';

type TagFieldProps = {
  value: string;
  isEditing: boolean;
  className: string;
  placeholder: string;
  onChangeValue: (newValue: string) => void;
  onFocus: () => void;
  onKeyDown: (e: KeyboardEvent) => void;
  onBlur: (e: FocusEvent) => void;
  onPaste: (e: ClipboardEvent) => void;
};

const TagFieldInput = styled.input`
  background: transparent;
  border: none;
  outline: none;
  color: inherit;
  width: auto;
  margin: 0 2px;
  box-sizing: content-box;
  letter-spacing: normal;

  &::placeholder {
    color: rgba(255, 255, 255, 0.6);
  }
`;

const HiddenSpan = styled.span`
  margin: 0 2px;
  visibility: hidden;
  position: absolute;
  white-space: pre;
  letter-spacing: normal;
`;

const TagField: React.FC<TagFieldProps> = ({
  value,
  isEditing,
  className,
  placeholder,
  onChangeValue,
  onFocus,
  onKeyDown,
  onBlur,
  onPaste,
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const spanRef = useRef<HTMLSpanElement>(null);

  //set focus to current tag if we start editing it.
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  //keep input ref the same width as the hidden span
  useLayoutEffect(() => {
    if (inputRef.current && spanRef.current) {
      inputRef.current.style.width = `${spanRef.current.offsetWidth}px`;
    }
  }, [value]);

  //HiddenSpan is used to calculate the width of the input field. Can remove when firefox supports field-sizing field
  return (
    <>
      <HiddenSpan ref={spanRef}>{value || placeholder}</HiddenSpan>
      <TagFieldInput
        className={className}
        ref={inputRef}
        value={value}
        onChange={(newValue) => onChangeValue(newValue.target.value)}
        onKeyDown={onKeyDown}
        onBlur={onBlur}
        onFocus={onFocus}
        placeholder={placeholder}
        onPaste={onPaste}
      />
    </>
  );
};

const TagContainer = styled.div<{ $focused?: boolean; $error?: boolean }>`
  border-radius: 5px;
  border: 1px solid black;
  color: white;
  white-space: nowrap;
  display: inline-flex;
  justify-content: center;
  align-items: center;
  margin: 1px;
  padding: 0 3px;
  ${(props) => props.$focused && ` box-shadow: 2px 2px 5px 2px rgba(0, 0, 0, .3);`};
  ${(props) => props.$error && `box-shadow: 0px 0px 3px 2px rgb(255, 0, 0);`};
`;

const TagDeleteButton = styled.button`
  padding: 0 4px;
  border: 0px;
  height: 100%;
`;

type TagProps = {
  data: TagEntry;
  editMode: EditingMode;
  isNewTag?: boolean;
  onChange: (updatedTag: Partial<TagEntry>) => void;
  handleFocus: (state: EditingMode) => void;
  handleKeyDown: (e: KeyboardEvent, state: EditingMode) => void;
  handleDelete: () => void;
  handleBlur: (e: FocusEvent) => void;
  handleFocusTrash: () => void;
  onPaste: (e: ClipboardEvent, state: EditingMode) => void;
};

export const Tag: React.FC<TagProps> = ({
  data,
  editMode,
  onChange,
  handleFocus,
  handleKeyDown,
  handleDelete,
  handleBlur,
  handleFocusTrash,
  onPaste,
}) => {
  const tagClass = getTagColorClass(data.key, data.value);
  return (
    <TagContainer
      className={tagClass}
      onClick={(e) => e.stopPropagation()}
      $focused={editMode !== EditingMode.Disabled}
      $error={editMode === EditingMode.Disabled && tagIsInvalid(data)}
    >
      <TagField
        className={tagClass}
        value={data.key}
        placeholder="Enter a key..."
        isEditing={editMode === EditingMode.First}
        onFocus={() => handleFocus(EditingMode.First)}
        onKeyDown={(e) => handleKeyDown(e, EditingMode.First)}
        onBlur={handleBlur}
        onChangeValue={(v) => onChange({ key: v })}
        onPaste={(e) => onPaste(e, EditingMode.First)}
      />
      <span className={tagClass} onClick={() => handleFocus(EditingMode.Disabled)}>
        <b> {'='} </b>
      </span>
      <TagField
        className={tagClass}
        value={data.value}
        placeholder="Enter a value..."
        isEditing={editMode === EditingMode.Second}
        onFocus={() => handleFocus(EditingMode.Second)}
        onKeyDown={(e) => handleKeyDown(e, EditingMode.Second)}
        onBlur={handleBlur}
        onChangeValue={(v) => onChange({ value: v })}
        onPaste={(e) => onPaste(e, EditingMode.Second)}
      />
      <TagDeleteButton
        tabIndex={0}
        title="delete tag"
        className={tagClass}
        onClick={() => handleDelete()}
        onFocus={() => handleFocusTrash()}
        onBlur={handleBlur}
      >
        <FaX size={12} />
      </TagDeleteButton>
    </TagContainer>
  );
};

export default Tag;
