import React, { FocusEvent } from 'react';
import { styled } from 'styled-components';
import { FaX, FaAngleDown } from 'react-icons/fa6';

const ButtonContainer = styled.div`
  display: flex;
  align-items: center;
`;

const ControlButton = styled.button`
  border: 0px;
  height: 100%;
  background-color: inherit;
  padding: 8px;
  width: 36px;
  height: 36px;
  color: rgb(204, 204, 204);
  &:hover {
    color: rgb(120, 120, 120);
  }
`;

const Divider = styled.span`
  width: 1px;
  background: rgb(204, 204, 204);
  align-self: center;
  height: 80%;
`;

type TagEntryButtonsProps = {
  tagLength: number;
  handleDeleteAll: () => void;
  handleBlur: (e: FocusEvent) => void;
  handleDownArrow: () => void;
};

const TagEntryButtons: React.FC<TagEntryButtonsProps> = ({ tagLength, handleDeleteAll, handleBlur, handleDownArrow: focusNewTag }) => {
  return (
    <ButtonContainer onBlur={handleBlur}>
      {tagLength > 0 && (
        <ControlButton title="delete all tags" onClick={() => handleDeleteAll()}>
          <FaX size="14px" />
        </ControlButton>
      )}
      <Divider />
      <ControlButton tabIndex={-1} onClick={focusNewTag}>
        <FaAngleDown size="18px" />
      </ControlButton>
    </ButtonContainer>
  );
};
export default TagEntryButtons;
