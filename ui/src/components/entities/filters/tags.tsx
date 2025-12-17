// project imports
import { DEFAULT_HIDE_TAG_KEYS, SelectInputArray, TagSelect } from '@components';
import { FilterDiv } from './shared';
import { FilterTags } from '@models';
import { filterTagsToTagEntryList, tagEntryListToFilterTags } from '@utilities';

interface FilterTagDisplayKeysProps {
  selected: string[];
  disabled: boolean;
  options: string[];
  onChange: (excludeKeys: string[]) => void;
}

export const FilterTagDisplayKeys: React.FC<FilterTagDisplayKeysProps> = ({ selected, onChange, options, disabled }) => {
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

interface FilterTagsProps {
  selected: FilterTags | null | undefined;
  disabled: boolean;
  onChange: (tags: FilterTags) => void;
}

export const FilterTagsField: React.FC<FilterTagsProps> = ({ selected, onChange, disabled }) => {
  return (
    <FilterDiv>
      <TagSelect
        tags={filterTagsToTagEntryList(selected ? selected : {})}
        setTags={(tagEntries) => {
          onChange(tagEntryListToFilterTags(tagEntries));
        }}
        placeholderText="Select Tags"
      />
    </FilterDiv>
  );
};
