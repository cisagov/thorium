// project imports
import { tagIsInvalid } from '@utilities';
import { Filters, FilterTags } from 'models';

// default tag keys to hide for each entity item being listed
export const DEFAULT_HIDE_TAG_KEYS = ['Results', 'Parent', 'submitter'];

// default number of results to render when listing files
export const DEFAULT_LIST_LIMIT = 25;

// get all possible limit options including the current value
export function getLimitOptions(currentLimit: number): Array<number> {
  // add limit to limit options if it is not one of the defaults
  const limitOptions = [25, 50, 100, 500];
  if (currentLimit != 0 && !limitOptions.includes(currentLimit)) {
    limitOptions.push(currentLimit);
    return limitOptions.sort(function (a, b) {
      return a - b;
    });
  }
  return limitOptions;
}

export const clearInvalidTags = (filters: Filters): Filters => {
  const newTags: FilterTags = {};
  if (filters.tags == null) {
    return filters;
  }
  const tagKeys = Object.keys(filters.tags);
  tagKeys.forEach((key) => {
    const values = filters.tags ? filters.tags[key] : [];
    const filteredTags = values.filter((value) => !tagIsInvalid({ key: key, value: value }));
    if (filteredTags.length > 0) {
      newTags[key] = filteredTags;
    }
  });
  return { ...filters, tags: newTags };
};
