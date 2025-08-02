import React, { KeyboardEventHandler, useEffect, useState } from 'react';

import CreatableSelect from 'react-select/creatable';
import { createReactSelectStyles } from '@utilities';
import { ValueMap } from '@models';

// Default field message
const DefaultMessage = 'Type a value...';

// Object structure needed for select object values
interface SelectOption {
  readonly label: string;
  readonly value: string;
}

// Create a SelectOption object form a simple string
const createSelectOption = (label: string, valueMap?: ValueMap): SelectOption => {
  // when we provide a valuesMap we reverse the label and the value because the label is not unique
  if (valueMap !== undefined && label in valueMap) {
    return {
      label: `${valueMap[label]}`,
      value: `${label}`,
    };
  }
  return {
    label: `${label}`,
    value: `${label}`,
  };
};

// Reformat an array of initial string values as an Array<SelectOption>
function formatOptions(initialValues: Array<string>, valueMap?: ValueMap): Array<SelectOption> {
  const formattedValues: Array<SelectOption> = [];

  // iterate over list and convert to a selectOption value
  initialValues.map((value) => {
    formattedValues.push(createSelectOption(value, valueMap));
  });

  if (valueMap != undefined) {
    Object.keys(valueMap).map((key) => {
      formattedValues.push(createSelectOption(key, valueMap));
    });
  }

  return formattedValues;
}

interface SelectInputProps {
  value?: string; // starting list of string/badge values
  options?: string[];
  className?: string;
  onCreate?: (input: string | undefined) => void;
  onChange: (input: string) => void; // call back updating caller with values
  valueMap?: ValueMap; // mapping of ids to label values
  defaultMessage?: string; // default field message when no initial value is provided
  disabled: boolean; // whether this component is disabled
}

const SelectInput: React.FC<SelectInputProps> = ({
  value = '',
  options = [],
  valueMap,
  className,
  onChange,
  onCreate,
  disabled = false,
  defaultMessage = DefaultMessage,
}) => {
  const [selectOptions, setSelectOptions] = useState(formatOptions(options, valueMap));
  const [selectValue, setSelectValue] = useState<SelectOption | null>(createSelectOption(value, valueMap));
  const selectStyle = createReactSelectStyles('White', 'rgb(160, 162, 163)');

  // create a new option in the list, also any call backs to API would trigger here
  const handleCreate = (inputValue: string) => {
    const newOption = createSelectOption(inputValue, valueMap);
    setSelectOptions((prev) => [...prev, newOption]);
    setSelectValue(newOption);
    // call create option call back
    if (onCreate != undefined) {
      onCreate(inputValue);
    }
    onChange(inputValue);
  };

  useEffect(() => {
    setSelectOptions(formatOptions(options, valueMap));
  }, [valueMap]);

  return (
    <CreatableSelect
      isDisabled={disabled}
      className={className}
      isClearable
      onChange={(newValue) => {
        setSelectValue(newValue);
        console.log(newValue?.value);
        onChange(newValue?.value ? newValue.value : '');
      }}
      onCreateOption={(newValue) => handleCreate(newValue)}
      styles={selectStyle}
      placeholder={defaultMessage}
      value={selectValue}
      options={selectOptions}
    />
  );
};

export default SelectInput;
