// project imports
import FilterDiv from './FilterDiv';
import TagSelect from '@components/tags/TagSelect/TagSelect';
import { RequestTags } from '@models/tags';
import { requestTagsToTagEntryList, tagEntriesToRequestTags } from '@utilities/tags';

interface FilterTagsProps {
  selected: RequestTags | null | undefined;
  disabled: boolean;
  onChange: (tags: RequestTags) => void;
}

const FilterTagsField: React.FC<FilterTagsProps> = ({ selected, onChange, disabled }) => {
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

export default FilterTagsField;
