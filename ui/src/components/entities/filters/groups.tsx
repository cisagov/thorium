import { useState } from 'react';

// project imports
import { SelectInputArray } from '@components';
import { FilterDiv } from './shared';

interface FilterGroupsProps {
  selected: string[]; // array of selected group names
  options: string[]; // array of group name options
  onChange: (groups: string[]) => void; // set groups callback
  disabled: boolean; // disable changes to groups
}

export const FilterGroups: React.FC<FilterGroupsProps> = ({ selected, options, onChange, disabled }) => {
  return (
    <FilterDiv>
      <SelectInputArray
        defaultMessage="Select a group"
        disabled={disabled}
        isCreatable={false}
        options={options}
        values={selected.sort()}
        onChange={onChange}
      />
    </FilterDiv>
  );
};
