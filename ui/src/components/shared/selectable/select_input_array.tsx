import React, { KeyboardEventHandler } from 'react';

import CreatableSelect from 'react-select/creatable';
import Select from 'react-select';
import { createReactSelectStyles } from '@utilities';

const components = {
  //DropdownIndicator: null,
};

// Object structure needed for select object values
interface SelectOption {
  readonly label: string;
  readonly value: string;
}

const createSelectOption = (label: string, prefix: string = '', valuesMap?: { [key: string]: string }): SelectOption => {
  if (valuesMap === undefined || !(label in valuesMap)) {
    return {
      label: `${label}`,
      value: `${label}`, //`${prefix}-${label}` this may be important
    };
  } else {
    return {
      label: `${valuesMap[label]}`,
      value: `${label}`,
    };
  }
};

// Reformat an array of initial string values as an Array<SelectOption>
function formatInitialValues(initialValues: Array<string>, valuesMap?: { [key: string]: string }): Array<SelectOption> {
  const formattedValues: Array<SelectOption> = [];
  // iterate over list and convert to a selectOption value
  initialValues.map((value, idx) => {
    formattedValues.push(createSelectOption(value, `${idx}`, valuesMap));
  });
  return formattedValues;
}

interface SelectInputProps {
  values: Array<string>; // starting list of string/badge values
  disabled?: boolean; // whether field is disabled
  options?: Array<string>; // options to select from initially
  valuesMap?: { [key: string]: string }; // mapping of unique keys to label values
  onChange: (input: Array<string>) => void; // call back updating caller with values
  onCreate?: (input: any) => void; // call back when a new value is created
  isCreatable?: boolean;
  defaultMessage?: string; // default field message when no initial value is provided
}

const DefaultMessage = 'Type each item and press enter...';

const SelectInputArray: React.FC<SelectInputProps> = ({
  values,
  disabled,
  valuesMap,
  options,
  onChange,
  onCreate,
  isCreatable = true,
  defaultMessage = DefaultMessage,
}) => {
  const [inputValue, setInputValue] = React.useState('');
  const [value, setValue] = React.useState<SelectOption[]>(formatInitialValues(values, valuesMap));
  const [valueOptions, setValueOptions] = React.useState<SelectOption[]>(formatInitialValues(options ? options : [], valuesMap));
  const selectStyle = createReactSelectStyles('White', 'rgb(160, 162, 163)');

  // control optional props to prevent menu from opening
  const selectProps: any = {};
  if (valueOptions.length == 0) {
    selectProps['menuIsOpen'] = false;
  }

  // control updates to the select component through key presses
  const handleKeyDown: KeyboardEventHandler = (event) => {
    if (!inputValue) return;
    switch (event.key) {
      case 'Enter':
      case 'Tab':
        const newValue = createSelectOption(inputValue, `${value.length}`, valuesMap);
        // need to check if newValue is in value or valueOptions and not duplicate
        // if not creatable need to check if value is in options and if not don't add to value
        if (
          isCreatable ||
          (!value.map((some) => some.value).includes(newValue.value) && valueOptions.map((some) => some.value).includes(newValue.value))
        ) {
          setValue((prev) => [...prev, newValue]);
          onChange([...value.map((option) => option.value), inputValue]);
          setInputValue('');
          handleCreateOption(newValue);
        }
        event.preventDefault();
    }
  };

  const handleCreateOption = (value: SelectOption) => {
    if (valueOptions.some((option) => option.value == value.value)) {
      return;
    }
    setValueOptions((prev) => [...prev, value]);
  };

  const handleCreateCallback = (value: any) => {
    if (onCreate) {
      onCreate(value);
    }
  };

  if (isCreatable) {
    return (
      <CreatableSelect
        {...selectProps}
        isDisabled={disabled}
        isMulti
        isClearable
        styles={selectStyle}
        components={components}
        inputValue={inputValue}
        onCreate={handleCreateCallback}
        onChange={(newValue: SelectOption[]) => {
          setValue(newValue);
          // pass current selected options to parent callback
          const updatedValues = inputValue
            ? [...newValue.map((option) => option.value), inputValue]
            : [...newValue.map((option) => option.value)];
          onChange(updatedValues);
        }}
        onInputChange={(newValue) => setInputValue(newValue)}
        onKeyDown={handleKeyDown}
        placeholder={defaultMessage}
        value={value}
        options={valueOptions}
      />
    );
  } else {
    return (
      <Select
        {...selectProps}
        isDisabled={disabled}
        isMulti
        isClearable
        styles={selectStyle}
        components={components}
        inputValue={inputValue}
        onChange={(newValue: SelectOption[]) => {
          setValue(newValue);
          // pass current selected options to parent callback
          const updatedValues = inputValue
            ? [...newValue.map((option) => option.value), inputValue]
            : [...newValue.map((option) => option.value)];
          onChange(updatedValues);
        }}
        onInputChange={(newValue) => setInputValue(newValue)}
        onKeyDown={handleKeyDown}
        placeholder={defaultMessage}
        value={value}
        options={valueOptions}
      />
    );
  }
};

export default SelectInputArray;
