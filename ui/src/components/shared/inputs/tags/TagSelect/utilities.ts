// project imports
import { EditingMode, FocusState } from './models';
import { TagEntry, TagOptions } from '@models/tags';

/**
 * Helper function for getting the current typed text based on state
 * @param {TagEntry[]} tags
 * @param {FocusState} focusState
 * @returns {string}
 * */
export function getCurrTypedText(tags: TagEntry[], focusState: FocusState): string {
  if (focusState.idx === -1) {
    return '';
  }
  const currTag = tags[focusState.idx];
  if (focusState.editMode === EditingMode.First) {
    return currTag.key;
  } else if (focusState.editMode === EditingMode.Second) {
    return currTag.value;
  }
  return '';
}

/**
 * Gets all dropdown options based on current state / typed text
 * @param {TagEntry[]} tags
 * @param {FocusState} focusState
 * @param {TagOptions | undefined} dropdownData
 * @param {string} currTypedText
 * @returns {string[]} String list of options
 *
 * Selects dropdown options based on current tag state and main list of dropdown data.
 * Filters values using {@link filterOptions}
 * */
export function getDropdownOptions(
  tags: TagEntry[],
  dropdownData: TagOptions | undefined,
  focusState: FocusState,
  currTypedText: string,
): string[] {
  const options: string[] = [];
  //add to start of list so an enter uses typed text
  if (currTypedText.length > 0) {
    options.push(currTypedText);
  }
  //return just a list with current typed text if options doesn't exist
  if (!dropdownData) return options;
  if (focusState.editMode == EditingMode.First) {
    options.push(...Object.keys(dropdownData).sort());
  } else if (focusState.editMode == EditingMode.Second) {
    const tag = tags[focusState.idx];
    if (tag.key in dropdownData) {
      options.push(...dropdownData[tag.key].sort());
    }
  }
  return filterOptions(options, currTypedText);
}

/**
 * Filters options array based on current typed text.
 * @param {string[]} options
 * @param {string} currTypedText
 * @returns {string[]}
 *
 * Filter out text that has no matching substring. Sort on if text matches
 * */
function filterOptions(options: string[], currTypedText: string): string[] {
  if (!currTypedText) return options;
  const normalizedTypedText = currTypedText.toLowerCase();
  const partialMatches = options
    .slice(1)
    .filter((option) => {
      const normalizedOpt = option.toLowerCase().trim();
      return normalizedOpt.includes(normalizedTypedText);
    })
    .sort((a, b) => {
      const normalizedA = a.toLowerCase().trim();
      const normalizedB = b.toLowerCase().trim();
      const isExactMatchA = normalizedA === normalizedTypedText;
      const isExactMatchB = normalizedB === normalizedTypedText;
      //exact matches prioritized
      if (isExactMatchA && !isExactMatchB) return -1;
      if (!isExactMatchA && isExactMatchB) return 1;
      const startsWithA = normalizedA.startsWith(normalizedTypedText);
      const startsWithB = normalizedB.startsWith(normalizedTypedText);
      //starts with matches prioritized
      if (startsWithA && !startsWithB) return -1;
      if (!startsWithA && startsWithB) return 1;
      return 0;
    });
  return [options[0], ...partialMatches];
}

/**
 * Helper function for quickly updating tag list
 * @param {TagEntry[]} tags Array of tags
 * @param {number} idx Index of tag to update
 * @param {Partial<TagEntry>} updatedTag Tag to update
 * @returns {TagEntry[]} modified tag list
 *
 * Cleaner version of the map() function that is more descriptive
 * */
export function updateTags(tags: TagEntry[], idx: number, updatedTag: Partial<TagEntry>): TagEntry[] {
  return tags.map((tag, i) => (i === idx ? { ...tag, ...updatedTag } : tag));
}
