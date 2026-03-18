// project imports
import { DEFAULT_HIDE_TAG_KEYS } from './params';
import FilterDiv from './FilterDiv';
import SelectInputArray from '@components/shared/selectable/SelectInputArray';

interface FilterTagDisplayKeysProps {
  selected: string[];
  disabled: boolean;
  options: string[];
  onChange: (excludeKeys: string[]) => void;
}

const FilterTagDisplayKeys: React.FC<FilterTagDisplayKeysProps> = ({ selected, onChange, options, disabled }) => {
  return (
    <FilterDiv>
      <SelectInputArray
        defaultMessage="Hide tags"
        disabled={disabled}
        isCreatable={true}
        options={options}
        values={selected.length > 0 ? selected.toSorted() : DEFAULT_HIDE_TAG_KEYS}
        onChange={onChange}
      />
    </FilterDiv>
  );
};

export default FilterTagDisplayKeys;
