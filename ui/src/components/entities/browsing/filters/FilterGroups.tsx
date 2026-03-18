// project imports
import FilterDiv from './FilterDiv';
import SelectInputArray from '@components/shared/selectable/SelectInputArray';

interface FilterGroupsProps {
  selected: string[]; // array of selected group names
  options: string[]; // array of group name options
  onChange: (groups: string[]) => void; // set groups callback
  disabled: boolean; // disable changes to groups
}

const FilterGroups: React.FC<FilterGroupsProps> = ({ selected, options, onChange, disabled }) => {
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

export default FilterGroups;
