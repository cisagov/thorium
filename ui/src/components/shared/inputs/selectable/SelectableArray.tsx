import { useState, useEffect } from 'react';
import { Button, Col, Form, Row } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';

interface SelectableArrayProps {
  initialEntries?: string[];
  setEntries: (entries: string[]) => void;
  placeholder?: string | string[];
  disabled?: boolean;
  trim?: boolean;
}

const SelectableArray = ({ initialEntries, setEntries, placeholder, disabled, trim }: SelectableArrayProps) => {
  const [arrayEntries, setArrayEntries] = useState<string[]>(
    initialEntries && Array.isArray(initialEntries) && initialEntries.length > 0 ? initialEntries : [''],
  );
  const [selectableKeys, setSelectableKeys] = useState<Record<string, boolean>>({});

  // initializer for selected and unselected keys
  const setInitialSelectable = (): Record<string, boolean> => {
    const availableKeys: Record<string, boolean> = {};
    if (Array.isArray(placeholder) && placeholder.length > 0) {
      placeholder.forEach((singleKey) => {
        availableKeys[singleKey] = !arrayEntries.includes(singleKey);
      });
    }
    return availableKeys;
  };

  // needed for create/copy due to placeholder prop being initalized to empty
  useEffect(() => {
    if (placeholder && Array.isArray(placeholder) && placeholder.length > 0) {
      setSelectableKeys(setInitialSelectable());
    }
  }, [placeholder]);

  // pass back only non-empty entries to caller
  const updateEntries = (updatedArray: string[]) => {
    const filteredArray = updatedArray.filter((item) => item != '');
    setEntries(filteredArray);
  };

  // update the list of array values
  const handleInputChange = (value: string, index: number, previousValue?: string) => {
    const newArray = [...arrayEntries];
    newArray[index] = trim ? value.trim() : value;
    if (index == newArray.length - 1) {
      newArray.push('');
    }
    if (placeholder) {
      let newSelects: Record<string, boolean> = { ...selectableKeys, [value]: false };
      if (previousValue) {
        newSelects = { ...newSelects, [previousValue]: true };
      }
      setSelectableKeys({ ...newSelects });
    }
    setArrayEntries(newArray);
    updateEntries(newArray);
  };

  // handle removal of entries using trash button
  const handleRemoveClick = (index: number, previousValue?: string) => {
    const newArray = [...arrayEntries];
    newArray.splice(index, 1);
    if (newArray.length == 0) {
      setArrayEntries(['']);
      updateEntries([]);
    } else {
      setArrayEntries(newArray);
      updateEntries(newArray);
    }
    if (previousValue) {
      setSelectableKeys({ ...selectableKeys, [previousValue]: true });
    }
  };

  return (
    <div>
      {arrayEntries.map((entry, index) => {
        const currentValue = entry;
        return (
          <div key={index} className="mt-2">
            <Row className="mb-2 image-fields">
              <Col className="pe-2">
                {!Array.isArray(placeholder) ? (
                  <Form.Control
                    type="text"
                    placeholder={placeholder}
                    value={entry}
                    disabled={disabled}
                    onChange={(e) => handleInputChange(e.target.value, index)}
                    onClick={(e) => handleInputChange((e.target as HTMLInputElement).value, index)}
                  />
                ) : (
                  <Form.Select
                    value={currentValue}
                    disabled={disabled}
                    onChange={(e) => handleInputChange(e.target.value, index, currentValue)}
                  >
                    {entry == '' && <option value="">Select an Image</option>}
                    {arrayEntries[index] != '' && <option>{arrayEntries[index]}</option>}
                    {Object.keys(selectableKeys)
                      .filter((item) => selectableKeys[item])
                      .map((entry, index) => (
                        <option key={index} value={entry}>
                          {entry}
                        </option>
                      ))}
                  </Form.Select>
                )}
              </Col>
              <Col xs="auto" className="ps-2">
                <Button
                  size="sm"
                  className="danger-btn mt-2"
                  variant=""
                  disabled={arrayEntries.length == 1 && arrayEntries[0] == ''}
                  onClick={() => handleRemoveClick(index, currentValue)}
                >
                  <FaTrash />
                </Button>
              </Col>
            </Row>
          </div>
        );
      })}
    </div>
  );
};

export default SelectableArray;
