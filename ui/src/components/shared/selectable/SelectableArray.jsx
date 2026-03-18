import React, { useState, useEffect } from 'react';
import { Button, Col, Form, Row } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';

const SelectableArray = ({ initialEntries, setEntries, placeholder, disabled, trim }) => {
  const [arrayEntries, setArrayEntries] = useState(
    initialEntries && Array.isArray(initialEntries) && initialEntries.length > 0 ? initialEntries : [''],
  );
  const [selectableKeys, setSelectableKeys] = useState({});
  // initializer for selected and unselected keys
  const setInitialSelectable = () => {
    const availableKeys = {};
    if (Array.isArray(placeholder) && placeholder.length > 0) {
      // set all selected options to false
      placeholder.forEach((singleKey) => {
        if (arrayEntries.includes(singleKey)) {
          availableKeys[singleKey] = false;
        } else {
          availableKeys[singleKey] = true;
        }
      });
    }

    return availableKeys;
  };

  // needed for create/copy due to placeholder prop being initalized to empty
  // during the scope of the setInitialSelectable function passed to useState
  useEffect(() => {
    if (placeholder && Array.isArray(placeholder) && placeholder.length > 0) {
      setSelectableKeys(setInitialSelectable());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [placeholder]);

  // call this any time updates are made to arrayEntries
  const updateEntries = (updatedArray) => {
    // only pass back to caller array items that have been entered
    const filteredArray = updatedArray.filter((item) => item != '');
    setEntries(filteredArray);
  };

  // update the list of key/value pairs
  const handleInputChange = (value, index, previousValue) => {
    // create a temp list of all array values before any updated input
    const newArray = [...arrayEntries];
    // add updated/new value to array at supplied location
    newArray[index] = trim ? value.trim() : value;
    // check if last index is empty and add blank entry if so
    if (index == newArray.length - 1) {
      newArray.push('');
    }
    // if there are selectable keys available
    if (placeholder) {
      let newSelects = { ...selectableKeys, [value]: false };
      if (previousValue) {
        newSelects = { ...newSelects, [previousValue]: true };
      }
      setSelectableKeys({ ...newSelects });
    }
    setArrayEntries(newArray);
    // pass back updated list with input changes
    updateEntries(newArray);
  };

  // handle removal of tags using trash button
  const handleRemoveClick = (index, previousValue) => {
    const newArray = [...arrayEntries];
    newArray.splice(index, 1);
    // list must have one blank entry if last entry is deleted
    if (newArray.length == 0) {
      setArrayEntries(['']);
      // pass back a blank array
      updateEntries([]);
    } else {
      setArrayEntries(newArray);
      // pass back updated list without removed item
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
              <Col md className="pe-2">
                {!Array.isArray(placeholder) ? (
                  <Form.Control
                    type="text"
                    placeholder={placeholder}
                    value={entry}
                    disabled={disabled}
                    onChange={(e) => handleInputChange(e.target.value, index)}
                    onClick={(e) => handleInputChange(e.target.value, index)}
                  />
                ) : (
                  <Form.Select
                    value={currentValue}
                    disabled={disabled}
                    onChange={(e) => handleInputChange(e.target.value, index, currentValue)}
                  >
                    {entry == '' && <option value="">Select an Image</option>}
                    {/* At minimum display the option of the default value */}
                    {arrayEntries[index] != '' && <option>{arrayEntries[index]}</option>}
                    {/* Show all images that have not been selected yet */}
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
              <Col xs={1} className="ps-2">
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
