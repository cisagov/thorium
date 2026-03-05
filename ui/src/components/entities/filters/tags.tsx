// project imports
import { SelectInputArray, TagSelect } from '@components';
import { DEFAULT_HIDE_TAG_KEYS } from './utilities';
import { FilterDiv } from './shared';
import { RequestTags } from '@models';
import { requestTagsToTagEntryList, tagEntriesToRequestTags } from '@utilities';

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
  selected: RequestTags | null | undefined;
  disabled: boolean;
  onChange: (tags: RequestTags) => void;
}

export const FilterTagsField: React.FC<FilterTagsProps> = ({ selected, onChange, disabled }) => {
  return (
    <FilterDiv>
      <TagSelect
        tags={requestTagsToTagEntryList(selected ? selected : {})}
        setTags={(tagEntries) => {
          onChange(tagEntriesToRequestTags(tagEntries));
        }}
        placeholderText="Select Tags"
      />
    </FilterDiv>
  );
};
