import { useEffect, useRef } from 'react';
import { styled } from 'styled-components';

const TagDropdownDiv = styled.div`
  position: absolute;
  background-color: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-top-left-radius: var(--bs-border-radius);
  border-bottom-left-radius: var(--bs-border-radius);
  z-index: 1000;
  margin-top: 5px;
  width: 100%;
  max-height: 400px;
  overflow-y: scroll;
`;

const DropdownOptionDiv = styled.div<{ $focused: boolean }>`
  padding: 5px;
  cursor: pointer;
  background-color: ${(props) => (props.$focused ? '#999999' : 'transparent')};
`;

export type DropdownProps = {
  options: string[];
  focusIdx: number;
  setFocusIdx: (idx: number) => void;
  onSelect: (idx: number) => void;
};

const TagDropdown: React.FC<DropdownProps> = ({ options, focusIdx, setFocusIdx, onSelect }) => {
  const dropdownRef = useRef<HTMLDivElement>(null);
  //Scroll to position of highlighted element

  useEffect(() => {
    if (!dropdownRef.current) return;
    //fixes bug where after selecting tag key value will be pre-scrolled down.
    if (focusIdx == -1) dropdownRef.current.scrollTop = 0;

    const focusedElement = dropdownRef.current.children[focusIdx];
    if (!focusedElement) return;

    const dropdownHeight = dropdownRef.current.clientHeight;
    const dropdownTopEdge = dropdownRef.current.getBoundingClientRect().top;
    const focusedElementTopEdge = focusedElement.getBoundingClientRect().top;
    //get number of pixels between the focused element in dropdown and top of dropdown
    const focusedElementOffset = focusedElementTopEdge - dropdownTopEdge;
    //add element height to the offset to get the offset of bottom
    const focusedElementBottomOffset = focusedElementOffset + focusedElement.clientHeight;
    // Remember -- pixels start at 0 at top of screen and increase going down.
    // Negative focusedElementOffset == higher on screen. Positive == lower on screen
    if (focusedElementOffset < 0) {
      //less than 0, that means we need to 'scroll' up. The focused element is above the top of the dropdown
      //increase the scroll window by the offset (top of highlighted element will be top of scroll window)
      dropdownRef.current.scrollTop += focusedElementOffset;
    } else if (focusedElementBottomOffset > dropdownHeight) {
      //if offset plus height of entry is more than dropdown (i.e. the entire entry is not being shown)
      dropdownRef.current.scrollTop += focusedElementBottomOffset - dropdownHeight;
    }
  }, [focusIdx]);

  return (
    <TagDropdownDiv ref={dropdownRef} tabIndex={-1}>
      {options.map((option, idx) => (
        <DropdownOptionDiv key={idx} onClick={() => onSelect(idx)} onMouseMove={() => setFocusIdx(idx)} $focused={idx === focusIdx}>
          {option}
        </DropdownOptionDiv>
      ))}
    </TagDropdownDiv>
  );
};

export default TagDropdown;
