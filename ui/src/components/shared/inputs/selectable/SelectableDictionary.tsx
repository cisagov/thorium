import { useState, useEffect, ChangeEvent } from 'react';
import { Button, Col, Form, Row } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';

export interface DictionaryEntry {
  key: string;
  value: string;
}

interface SelectableDictionaryProps {
  entries: DictionaryEntry[];
  disabled?: boolean;
  keys?: string[];
  setEntries: (entries: DictionaryEntry[]) => void;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
  deleted?: string[];
  setDeleted?: (deleted: string[]) => void;
  trim?: boolean;
}

const SelectableDictionary = ({
  entries,
  disabled,
  keys,
  setEntries,
  keyPlaceholder,
  valuePlaceholder,
  deleted,
  setDeleted,
  trim,
}: SelectableDictionaryProps) => {
  const [selectableKeys, setSelectableKeys] = useState<Record<string, boolean>>({});

  // add default empty entries if none exist
  let currentEntries = entries;
  if (currentEntries.length == 0) {
    currentEntries = [{ key: '', value: '' }];
  }

  // initializer for selected and unselected keys
  const setInitialSelectable = (): Record<string, boolean> => {
    const availableKeys: Record<string, boolean> = {};
    if (currentEntries && keys && keys.length > 0) {
      const allSelectedKeys = currentEntries.map((item) => {
        if (item.key.trim() != '') {
          return item.key;
        }
        return undefined;
      });

      keys.forEach((singleKey) => {
        availableKeys[singleKey] = !allSelectedKeys.includes(singleKey);
      });
    }
    return availableKeys;
  };

  // needed for create/copy due to keys prop being initialized to empty
  useEffect(() => {
    if (keys && Array.isArray(keys) && keys.length > 0) {
      setSelectableKeys(setInitialSelectable());
    }
  }, [keys]);

  // update the list of key/value pairs
  const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>, index: number, previousValue?: string) => {
    const { name, value } = e.target;
    const list = [...currentEntries];
    if (name == 'key') {
      list[index] = { ...list[index], key: trim ? value.trim() : value };
    } else {
      list[index] = { ...list[index], value };
    }

    if (keys) {
      let newSelects: Record<string, boolean> = { ...selectableKeys, [value]: false };
      if (previousValue) {
        newSelects = { ...newSelects, [previousValue]: true };
      }
      setSelectableKeys({ ...newSelects });
    }
    setEntries(list);
    handleAddInput(index, list);
  };

  // handle adding new input fields
  const handleAddInput = (index: number, list: DictionaryEntry[]) => {
    if (index == list.length - 1 && !(list[index].key == '' && list[index].value == '')) {
      setEntries([...list, { key: '', value: '' }]);
    }
  };

  // handle removal of items using trash button
  const handleRemoveClick = (index: number, listLength: number, previousValue?: string) => {
    if (index == 0 && listLength == 1) {
      setEntries([{ key: '', value: '' }]);
    } else {
      const list = [...currentEntries];
      list.splice(index, 1);
      setEntries(list);
    }
    if (previousValue) {
      setSelectableKeys({ ...selectableKeys, [previousValue]: true });
    }
    // track deleted items
    if (deleted && setDeleted && currentEntries[index].key.trim() != '') {
      setDeleted([...deleted, currentEntries[index].key]);
    }
  };

  return (
    <div>
      {currentEntries.length > 0 && (
        <>
          {currentEntries.map((x, i) => {
            const currentValue = x.key == null ? '' : x.key;
            return (
              <div key={i} className="mt-2">
                <Row className="g-3">
                  {keys ? (
                    <Col md>
                      <Form.Select
                        name="key"
                        value={currentValue}
                        disabled={disabled}
                        onChange={(e) => handleInputChange(e, i, currentValue)}
                        onClick={() => handleAddInput(i, currentEntries)}
                      >
                        {x.key == '' && <option value="">Select an Image</option>}
                        {x.key.length > 0 && <option>{x.key == null ? '' : x.key}</option>}
                        {Object.keys(selectableKeys)
                          .filter((option) => selectableKeys[option])
                          .map((singleKey, index) => (
                            <option key={index} value={singleKey}>
                              {singleKey}
                            </option>
                          ))}
                      </Form.Select>
                    </Col>
                  ) : (
                    <Col md>
                      <Form.Control
                        name="key"
                        type="textarea"
                        placeholder={keyPlaceholder}
                        value={x.key == null ? '' : x.key}
                        disabled={disabled}
                        onChange={(e) => handleInputChange(e as unknown as ChangeEvent<HTMLInputElement>, i)}
                        onClick={() => handleAddInput(i, currentEntries)}
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
                      onChange={(e) => handleInputChange(e as unknown as ChangeEvent<HTMLInputElement>, i)}
                      onClick={() => handleAddInput(i, currentEntries)}
                    />
                  </Col>
                  <Col xs="auto" className="d-flex align-items-center">
                    {currentEntries.length > 0 && (
                      <Button
                        size="sm"
                        className="danger-btn"
                        variant=""
                        disabled={disabled}
                        onClick={() => handleRemoveClick(i, currentEntries.length, currentValue)}
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
