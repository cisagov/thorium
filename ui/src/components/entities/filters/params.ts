// project imports
import { Filters, FilterTags } from 'models';
import { DEFAULT_HIDE_TAG_KEYS } from './filters';

// verify defaults and excluded arrays are equal sets of keys
const excludeKeysAreDefault = (excludeKeys: string[]) => {
  return (
    DEFAULT_HIDE_TAG_KEYS.every((value: string) => excludeKeys.map((item) => item.toUpperCase()).includes(value.toUpperCase())) &&
    DEFAULT_HIDE_TAG_KEYS.length == excludeKeys.length
  );
};

// encode filters to search params
export function encodeFiltersToParams(filters: Filters) {
  const encodedFilters = [];
  // encode limit
  if (filters.limit) {
    encodedFilters.push(`limit=${encodeURIComponent(filters.limit)}`);
  }
  // encode groups
  if (filters.groups) {
    if (Array.isArray(filters.groups)) {
      filters.groups.map((group) => {
        encodedFilters.push(`groups=${encodeURIComponent(group)}`);
      });
    } else {
      encodedFilters.push(`groups=${encodeURIComponent(filters.groups)}`);
    }
  }
  // encode nested tags
  for (const key in filters.tags) {
    filters.tags[key].map((value) => {
      encodedFilters.push(`tags[${encodeURIComponent(key)}]=${encodeURIComponent(value)}`);
    });
  }
  // start (earliest) filter date range
  if (filters.hasOwnProperty('start')) {
    encodedFilters.push(`start=${filters.start}`);
  }
  // end (latest) filter date range
  if (filters.hasOwnProperty('end')) {
    encodedFilters.push(`end=${filters.end}`);
  }
  // add tag filters case sensitivity flag only if false (default is true)
  if (filters.hasOwnProperty('tags_case_insensitive') && filters.tags_case_insensitive == false) {
    encodedFilters.push(`tags_case_insensitive=false`);
  }
  // add exclude display tags only if those tags are not defaults
  if (filters.hasOwnProperty('hideTags') && filters.hideTags != undefined && !excludeKeysAreDefault(filters.hideTags)) {
    filters.hideTags.map((key: string) => {
      encodedFilters.push(`hide=${encodeURIComponent(key)}`);
    });
  }
  // Join all parameters with '&' to form the query string
  return encodedFilters.join('&');
}

// decode search params to filters
export function decodeParamsToFilters(searchParams: URLSearchParams) {
  const params: Filters = {};
  const tags: FilterTags = {};
  // Iterate over each search parameter
  for (const [key, value] of searchParams.entries()) {
    // skip empty values
    if (value == '') {
      continue;
    }
    // parse tags list
    if (key.startsWith('tags[')) {
      // break up tags keys from tags prefix
      const keyTokens = key.split(/\[|\]/).filter(Boolean);
      if (keyTokens.length == 2) {
        if (tags.hasOwnProperty(keyTokens[1])) {
          // don't save duplicate tags
          if (!tags[keyTokens[1]].includes(value)) {
            tags[keyTokens[1]].push(value);
          }
        } else {
          tags[keyTokens[1]] = [value];
        }
      }
      // parse groups list for submission group membership
    } else if (key == 'groups') {
      if ('groups' in params && params.groups) {
        params.groups.push(value);
      } else {
        params['groups'] = [value];
      }
      // parse exclude display tags
    } else if (key == 'hide') {
      if ('hideTags' in params && params.hideTags instanceof Array) {
        if (!(value in params.hideTags)) {
          params.hideTags.push(value);
        }
      } else {
        params['hideTags'] = [value];
      }
      // parse if case insensitive tag search
    } else if (key == 'tags_case_insensitive') {
      params['tags_case_insensitive'] = value == 'true';
      // put all else keys as single key/value pairs in params
    } else {
      function updateFilterField<T extends keyof Filters>(field: T, value: Filters[T]): void {
        params[field] = value;
      }
      updateFilterField(key as keyof Filters, value);
    }
  }
  if (Object.keys(tags).length > 0) {
    params['tags'] = tags;
  }
  // default for tags_case_sensitive is default when not specified in params
  if (params['tags_case_insensitive'] === undefined) {
    params['tags_case_insensitive'] = true;
  }
  // add in default hide
  if (params['hideTags'] === undefined) {
    params['hideTags'] = DEFAULT_HIDE_TAG_KEYS;
  }
  return params;
}
