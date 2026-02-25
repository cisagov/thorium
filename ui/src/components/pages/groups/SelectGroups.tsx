import React from 'react';
import { Button } from 'react-bootstrap';

type SelectGroupsProps = {
  groups: string[];
  setGroups: (groups: string[]) => void;
  disabled: boolean;
  clearState: () => void;
};

const SelectGroups: React.FC<SelectGroupsProps> = ({ groups, setGroups, disabled, clearState = null }) => {
  if (groups) {
    return (
      <>
        {Object.keys(groups)
          .sort()
          .map((group) => (
            <Button
              key={group}
              className={`m-1 primary-btn ${groups[group] ? 'selected' : 'unselected'}`}
              variant=""
              disabled={disabled}
              onClick={() => {
                setGroups({ ...groups, [group]: !groups[group] });
                if (clearState != null) {
                  clearState();
                }
              }}
            >
              <b style={{ fontSize: '1rem' }}>{group}</b>
            </Button>
          ))}
      </>
    );
  }
  return null;
};

export default SelectGroups;
