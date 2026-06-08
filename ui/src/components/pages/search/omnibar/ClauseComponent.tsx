import { useRef, useLayoutEffect, KeyboardEvent } from 'react';
import { styled } from 'styled-components';

export const ClauseComponentSpan = styled.span`
  margin: 0 2px;
  white-space: pre;
  cursor: text;
  color: inherit;
`;

export const ClauseComponentInput = styled.input`
  background: transparent;
  border: none;
  outline: none;
  margin: 0 2px;
  letter-spacing: normal;
  font: inherit;
  box-sizing: content-box;
  color: inherit;

  &::placeholder {
    color: var(--thorium-omnibar-placeholder);
    opacity: 1; /* Firefox */
  }
`;

const HiddenMesaureSpan = styled.span`
  visibility: hidden;
  position: absolute;
  margin: 0 2px;
  white-space: pre;
  letter-spacing: normal;
  font: inherit;
`;

type ClauseComponentProps = {
  value: string; //true value
  draft: string; //draft value
  isEditing: boolean;
  placeholder: string;
  setDraft: (next: string) => void;
  onBeginEdit: () => void;
  // onBlur: () => void;
  onKeyDown: (e: KeyboardEvent<HTMLInputElement>) => void;
};

export const ClauseComponent: React.FC<ClauseComponentProps> = ({
  value,
  isEditing,
  placeholder,
  draft,
  setDraft,
  onBeginEdit,
  // onBlur,
  onKeyDown,
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const measureRef = useRef<HTMLSpanElement | null>(null);
  const measureText = draft || value || placeholder || ' ';

  //keep input ref the same width as the hidden span
  useLayoutEffect(() => {
    if (isEditing && inputRef.current && measureRef.current) {
      inputRef.current.style.width = `${measureRef.current.offsetWidth + 1}px`;
    }
  }, [isEditing, draft, value, placeholder]);

  return (
    <>
      <HiddenMesaureSpan ref={measureRef}>{measureText}</HiddenMesaureSpan>
      {isEditing ? (
        <ClauseComponentInput
          autoFocus
          ref={inputRef}
          value={draft}
          placeholder={placeholder}
          onChange={(e) => setDraft(e.target.value)}
          // onBlur={onBlur}
          onKeyDown={onKeyDown}
        />
      ) : (
        <ClauseComponentSpan onClick={onBeginEdit}>{value || placeholder}</ClauseComponentSpan>
      )}
    </>
  );
};

export const ClauseComponentStatic: React.FC<{ value: string }> = ({ value }) => {
  return <ClauseComponentSpan>{value}</ClauseComponentSpan>;
};
