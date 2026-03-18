import { Button } from 'react-bootstrap';

const SelectGroups = ({ groups, setGroups, disabled, clearState = null }) => {
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
                if (clearState) {
                  clearState();
                }
              }}
            >
              <font size="3">
                <b>{group}</b>
              </font>
            </Button>
          ))}
      </>
    );
  }
  return null;
};

export default SelectGroups;
