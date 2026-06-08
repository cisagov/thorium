import { useEffect } from 'react';
import { styled } from 'styled-components';
import { DropdownOption, DropdownState } from './EditingTypes';
import { FaCheck } from 'react-icons/fa';
import { CategoryLogo } from './CategoryLogo';

const OmnibarDropdownDiv = styled.div`
  position: absolute;
  border: 1px solid var(--thorium-omnibar-border);
  background-color: var(--thorium-omnibar-bg);
  // border-bottom-left-radius: 5px;
  border-top: none;
  z-index: 1000;
  width: 100%;
  box-shadow: 0 12px 14px var(--thorium-omnibar-dropdown-shadow);
  max-height: 400px;
  overflow-y: auto;
`;

const DropdownOptionDiv = styled.div<{ $focused: boolean }>`
  padding: 5px;
  cursor: pointer;
  background-color: ${(props) => (props.$focused ? 'var(--thorium-omnibar-dropdown-highlight)' : 'inherit')};
  display: flex;
  justify-content: space-between;
  gap: 10px;
`;

const DropdownOptionHelpText = styled.span`
  color: var(--thorium-secondary-text);
`;

const DropdownOptionValueDiv = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
`;

export type DropdownProps = {
  options: DropdownOption[];
  dropdownState: DropdownState;
  setFocusIdx: (idx: number) => void;
  onSelect: (idx: number) => void;
  onMouseLeave: () => void;
  ref: React.RefObject<HTMLDivElement | null>;
};

const OmnibarDropdown: React.FC<DropdownProps> = ({ options, dropdownState, setFocusIdx, onSelect, onMouseLeave, ref }) => {
  //Scroll to position of highlighted element
  const focusIdx = dropdownState.index;

  useEffect(() => {
    if (!ref.current) return;
    //fixes bug where after selecting tag key value will be pre-scrolled down.
    if (focusIdx == -1) ref.current.scrollTop = 0;

    const focusedElement = ref.current.children[focusIdx];
    if (!focusedElement) return;

    const dropdownHeight = ref.current.clientHeight;
    const dropdownTopEdge = ref.current.getBoundingClientRect().top;
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
      ref.current.scrollTop += focusedElementOffset;
    } else if (focusedElementBottomOffset > dropdownHeight) {
      //if offset plus height of entry is more than dropdown (i.e. the entire entry is not being shown)
      ref.current.scrollTop += focusedElementBottomOffset - dropdownHeight;
    }
  }, [focusIdx]);

  return (
    <OmnibarDropdownDiv ref={ref} tabIndex={-1} onMouseLeave={onMouseLeave} role="listbox">
      {options.map((option, idx) => (
        <DropdownOptionDiv
          key={idx}
          onClick={() => onSelect(idx)}
          onMouseMove={() => setFocusIdx(idx)}
          $focused={idx === focusIdx && dropdownState.isSelecting}
          role="option"
        >
          <DropdownOptionValueDiv>
            <CategoryLogo category={option.category} />
            <span> {option.value} </span>
          </DropdownOptionValueDiv>
          <div>
            {option.helpText && <DropdownOptionHelpText>{option.helpText}</DropdownOptionHelpText>}
            {option.hasCheckmark && <FaCheck />}
          </div>
        </DropdownOptionDiv>
      ))}
    </OmnibarDropdownDiv>
  );
};

export default OmnibarDropdown;
