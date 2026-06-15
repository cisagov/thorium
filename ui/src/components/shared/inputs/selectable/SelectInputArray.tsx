import React, { KeyboardEventHandler, useEffect } from 'react';
import CreatableSelect from 'react-select/creatable';
import Select, { SelectComponentsConfig, GroupBase } from 'react-select';

// project imports
import { createReactSelectStyles } from '@utilities/select';

// spec: ./selectable.spec.md

// Object structure needed for select object values
interface SelectOption {
  readonly label: string;
  readonly value: string;
}

// The react-select `components` override shape for this control's option/value objects. Exported so
// callers (e.g. hiding the dropdown indicator/menu) can supply their own overrides via the
// `componentsOverride` prop while keeping the default (no override) for existing callers.
export type SelectInputArrayComponents = SelectComponentsConfig<SelectOption, true, GroupBase<SelectOption>>;

// The default `components` override: an empty object keeps react-select's built-in components,
// preserving the historical behavior for every caller that does not pass `componentsOverride`.
const DEFAULT_COMPONENTS: SelectInputArrayComponents = {};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const createSelectOption = (label: string, _prefix: string = '', valuesMap?: { [key: string]: string }): SelectOption => {
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
  error?: boolean; // whether to render the control with a danger outline
  /**
   * react-select components override; defaults to no override (existing behavior). A caller can pass
   * `{ Menu: () => null }` (or similar) to hide the menu when it drives the options list itself.
   */
  componentsOverride?: SelectInputArrayComponents;
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
  error = false,
  componentsOverride = DEFAULT_COMPONENTS,
}) => {
  const [inputValue, setInputValue] = React.useState('');
  const [value, setValue] = React.useState<SelectOption[]>(formatInitialValues(values, valuesMap));
  // prop-driven options (rebuilt whenever `options`/`valuesMap` change so shrinking the prop drops
  // stale entries) kept separate from values the user creates in the creatable variant, which must
  // survive prop-identity churn (callers pass fresh array identities every render)
  const [propOptions, setPropOptions] = React.useState<SelectOption[]>(formatInitialValues(options ? options : [], valuesMap));
  const [created, setCreated] = React.useState<SelectOption[]>([]);
  const selectStyle = createReactSelectStyles('White', 'rgb(160, 162, 163)', error);

  // Rebuild the prop-driven options directly from the `options` prop so it is effectively controlled:
  // shrinking `options` removes entries (the old accumulate-only merge could never drop a stale
  // option). User-created values live in `created` and are merged below so they are not wiped here.
  useEffect(() => {
    setPropOptions(formatInitialValues(options ? options : [], valuesMap));
  }, [options, valuesMap]);

  // the rendered options: prop-driven entries plus any user-created values, deduped by value so a
  // created value that later appears in `options` doesn't render twice
  const valueOptions = React.useMemo<SelectOption[]>(() => {
    const seen = new Set(propOptions.map((option) => option.value));
    return [...propOptions, ...created.filter((option) => !seen.has(option.value))];
  }, [propOptions, created]);

  // control optional props to prevent menu from opening
  const selectProps: { menuIsOpen?: boolean } = {};
  if (valueOptions.length == 0) {
    selectProps.menuIsOpen = false;
  }

  // in case values (or the id->label map that renders them) change externally, resync the chips.
  // valuesMap is a dep so labels refresh when the map resolves after the values (the builder case,
  // where selected ids arrive before their human-readable labels).
  useEffect(() => {
    setValue(formatInitialValues(values, valuesMap));
  }, [values, valuesMap]);

  // control updates to the select component through key presses
  const handleKeyDown: KeyboardEventHandler = (event) => {
    if (!inputValue) return;
    switch (event.key) {
      case 'Enter':
      case 'Tab': {
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
          // Only the creatable variant grows the options list with newly typed values. For the
          // non-creatable variant the typed value already exists in `options` (it must, to be
          // accepted above), so pushing it back would duplicate a controlled option.
          if (isCreatable) {
            handleCreateOption(newValue);
          }
        }
        event.preventDefault();
        break;
      }
    }
  };

  const handleCreateOption = (value: SelectOption) => {
    if (valueOptions.some((option) => option.value == value.value)) {
      return;
    }
    // persist created values separately so a parent re-render (which rebuilds propOptions from a
    // fresh array identity) can't drop them from the menu
    setCreated((prev) => [...prev, value]);
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
        components={componentsOverride}
        inputValue={inputValue}
        onCreateOption={handleCreateCallback}
        onChange={(newValue: readonly SelectOption[]) => {
          setValue([...newValue]);
          // pass current selected options to parent callback
          const updatedValues = newValue.map((option) => option.value);
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
        components={componentsOverride}
        inputValue={inputValue}
        onChange={(newValue: readonly SelectOption[]) => {
          setValue([...newValue]);
          // pass current selected options to parent callback
          const updatedValues = newValue.map((option) => option.value);
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
