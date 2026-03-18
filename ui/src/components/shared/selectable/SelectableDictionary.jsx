import React, { useState, useEffect } from 'react';
import { Button, Col, Form, Row } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';

const SelectableDictionary = ({ entries, disabled, keys, setEntries, keyPlaceholder, valuePlaceholder, deleted, setDeleted, trim }) => {
  const [selectableKeys, setSelectableKeys] = useState({});

  // add some default empty entries if none exist
  if (entries.length == 0) {
    entries = [{ key: '', value: '' }];
  }

  // initializer for selected and unselected keys
  const setInitialSelectable = () => {
    const availableKeys = {};
    if (entries && keys && keys.length > 0) {
      // get pool of all selected keys
      const allSelectedKeys = entries.map((item) => {
        if (item['key'].trim() != '') {
          return item['key'];
        }
      });

      // set all selected options to false
      keys.forEach((singleKey) => {
        if (allSelectedKeys.includes(singleKey)) {
          availableKeys[singleKey] = false;
        } else {
          availableKeys[singleKey] = true;
        }
      });
    }
    return availableKeys;
  };

  // needed for create/copy due to keys prop being initialized to empty
  // during the scope of the setInitialSelectable function passed to useState
  useEffect(() => {
    if (keys && Array.isArray(keys) && keys.length > 0) {
      setSelectableKeys(setInitialSelectable());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keys]);

  // update the list of key/value pairs
  const handleInputChange = (e, index, previousValue) => {
    const { name, value } = e.target;
    const list = [...entries];
    if (name == 'key') {
      list[index][name] = trim ? value.trim() : value;
    } else {
      list[index][name] = value;
    }

    // if there are selectable keys available
    if (keys) {
      let newSelects = { ...selectableKeys, [value]: false };
      if (previousValue) {
        newSelects = { ...newSelects, [previousValue]: true };
      }
      setSelectableKeys({ ...newSelects });
    }
    setEntries(list);
    handleAddInput(e, index);
  };

  // handle adding new input fields
  const handleAddInput = (e, index) => {
    // don't create a new input if we aren't editing the last input group and that last input group isn't empty
    if (index == entries.length - 1 && !(entries[index].key == '' && entries[index].value == '')) {
      setEntries([...entries, { key: '', value: '' }]);
    }
  };

  // handle removal of items using trash button
  const handleRemoveClick = (index, listLength, previousValue) => {
    // if the single key:value is deleted
    // just clear the key and value
    if (index == 0 && listLength == 1) {
      setEntries([{ key: '', value: '' }]);
    } else {
      const list = [...entries];
      list.splice(index, 1);
      setEntries(list);
    }
    if (previousValue) {
      setSelectableKeys({ ...selectableKeys, [previousValue]: true });
    }
    // track deleted items
    if (deleted && entries[index]['key'].trim() != '') {
      setDeleted([...deleted, entries[index]['key']]);
    }
  };

  return (
    <div>
      {entries.length > 0 && (
        <>
          {entries.map((x, i) => {
            const currentValue = x.key == null ? '' : x.key;
            return (
              <div key={i} className="mt-2">
                <Row className="g-3">
                  {keys && (
                    <Col md>
                      <Form.Select
                        name="key"
                        value={currentValue}
                        disabled={disabled}
                        onChange={(e) => handleInputChange(e, i, currentValue)}
                        onClick={(e) => handleAddInput(e, i)}
                      >
                        {/* Select a key option for new empty selects */}
                        {x.key == '' && <option value="">Select an Image</option>}
                        {/* At minimum display the key option of the default value */}
                        {x.key.length > 0 && <option>{x.key == null ? '' : x.key}</option>}
                        {/* Show all key options that have not been selected yet */}
                        {Object.keys(selectableKeys)
                          .filter((option) => selectableKeys[option])
                          .map((singleKey, index) => (
                            <option key={index} value={singleKey}>
                              {singleKey}
                            </option>
                          ))}
                      </Form.Select>
                    </Col>
                  )}
                  {!keys && (
                    <Col md>
                      <Form.Control
                        name="key"
                        type="textarea"
                        placeholder={keyPlaceholder}
                        value={x.key == null ? '' : x.key}
                        disabled={disabled}
                        onChange={(e) => handleInputChange(e, i)}
                        onClick={(e) => handleAddInput(e, i)}
                      />
                    </Col>
                  )}
                  <Col md className="pe-2">
                    <Form.Control
                      name="value"
                      type="textarea"
                      disabled={disabled}
                      placeholder={valuePlaceholder}
                      value={x.value == null ? '' : x.value}
                      onChange={(e) => handleInputChange(e, i)}
                      onClick={(e) => handleAddInput(e, i)}
                    />
                  </Col>
                  <Col xs={1} className="d-flex justify-content-end">
                    {entries.length > 0 && (
                      <Button
                        size="sm"
                        className="danger-btn"
                        variant=""
                        disabled={disabled}
                        onClick={() => handleRemoveClick(i, entries.length, currentValue)}
                      >
                        <FaTrash />
                      </Button>
                    )}
                  </Col>
                </Row>
              </div>
            );
          })}
        </>
      )}
    </div>
  );
};

export default SelectableDictionary;
