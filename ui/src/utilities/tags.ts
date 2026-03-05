import { countFileTags } from '@thorpi';
import { RequestTags, TagEntry, TagOptions } from 'models';

import rawAttackTagDefaults from '../../mitre_tags/attackTagsList.tags?raw';
import rawMbcTagDefaults from '../../mitre_tags/MBCTagsList.tags?raw';

//Type for storing TagOptions to local storage with an expiry time.
type StorageType = {
  tagCounts: TagOptions;
  expireTime: number;
};

const LOCAL_STORAGE_KEY = 'tagCounts';
const DEFAULT_EXPIRE_TIMEOUT = 1000 * 60 * 60 * 24 * 7; //1 week
const LIMIT = 10_000;
const attackTagOptions = String(rawAttackTagDefaults).split('\n');
const mbcTagOptions = String(rawMbcTagDefaults).split('\n');

/**
 * Save tag count to local storage.
 * @param {TagOptions} newTagCounts - Tag Options to save
 *
 * Saves TagOption data to local storage with a timeout
 * Strips out ATT&CK and MBC data (which we can load from the static files)
 * */
function saveTagCountToLocalStorage(newTagCounts: TagOptions) {
  //strip ATT&CK, MBC. We will add full versions in on load
  const copy = { ...newTagCounts };
  delete copy['ATT&CK'];
  delete copy['MBC'];
  const storageData: StorageType = {
    tagCounts: newTagCounts,
    expireTime: Date.now(),
  };
  localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(storageData));
}

/**
 * Load tags from API. Helper function called from {@link fetchTags}.
 * */
async function loadTagsFromApi() {
  const tagCounts = await countFileTags({ limit: LIMIT }, console.error, null);
  if (tagCounts) {
    const tagOptions: TagOptions = {};
    Object.keys(tagCounts.tags).forEach((key) => {
      const values = Object.keys(tagCounts.tags[key].values);
      tagOptions[key] = values;
    });
    saveTagCountToLocalStorage(tagOptions);
  }
  return null;
}

/**
 * Fetch tag options from local storage
 *
 * Checks if tags are present in local storage, is not corrupted, and not expired.
 * If any checks fail, send an api request to save tags to local storage.
 * If tags are valid don't do anything. Load using {@link load}
 * */
export async function fetchLocalStorageTags() {
  let storageData: StorageType | null = null;
  const storageDataString = localStorage.getItem('tagCounts');

  //not present in local storage
  if (!storageDataString) {
    return loadTagsFromApi();
  }
  //check if local storage data is corrupted
  try {
    storageData = JSON.parse(storageDataString) as StorageType;
  } catch {
    //local storage corrupted
    return loadTagsFromApi();
  }
  //older than expire time, refresh
  if (Date.now() - storageData.expireTime > DEFAULT_EXPIRE_TIMEOUT) {
    return loadTagsFromApi();
  }
  return null;
}

/**
 * Remove tag options from local storage
 * */
export function clearTagDataFromLocalStorage() {
  localStorage.removeItem(LOCAL_STORAGE_KEY);
}

/**
 * Retrieve TagOptions from local storage if exists.
 * @returns {TagOptions | null} Tag Options if exists in local storage
 *
 * Retrieves TagOptions from local storage. Adds in the ATT&CK and MBC
 * tags to the result object.
 * */
export function loadTagOptionsFromLocalStorage(): TagOptions | null {
  let storageData: StorageType | null = null;
  const storageDataString = localStorage.getItem('tagCounts');

  if (!storageDataString) return null;
  //Try to load data string -- if corrupted return null
  try {
    storageData = JSON.parse(storageDataString) as StorageType;
  } catch {
    return null;
  }
  const tags = storageData.tagCounts;
  tags['ATT&CK'] = attackTagOptions.filter((item) => item); //remove empty strings
  tags['MBC'] = mbcTagOptions.filter((item) => item);
  return tags;
}

/**
 * Check if a tag is invalid.
 * @param {TagEntry} tag - Tag to check. Has a key and a value
 * @param {boolean} ignore_empty - boolean flag to ignore empty tags
 * @returns {boolean} true if the tag is valid, false otherwise
 *
 * A tag is invalid if either the key or the value are empty.
 * If the 'ignore_empty' parameter is set to true, tags where both key and value are empty are treated as valid.
 * */
export function tagIsInvalid(tag: TagEntry, ignore_empty: boolean = false): boolean {
  const keyLength = tag.key.trim().length;
  const valueLength = tag.value.trim().length;
  if (!ignore_empty && keyLength + valueLength == 0) {
    return true;
  } else if (keyLength == 0 && valueLength > 0) {
    return true;
  } else if (valueLength == 0 && keyLength > 0) {
    return true;
  }
  return false;
}

/**
 * Check if any tag is invalid.
 *
 * @param {TagEntry[]} tags - Tags to check. Each has a key and value
 * @param {boolean} ignore_empty - Boolean flag to ignore empty tags
 * @returns {boolean} true if all tags are valid, false otherwise
 *
 * This function uses {@link tagIsInvalid} to determine if any tags are invalid.
 * A tag is invalid if the key or value is empty.
 * If the 'ignore_empty' parameter is set to true, tags where both key and value are empty are treated as valid.
 * */
export function hasInvalidTags(tags: TagEntry[], ignore_empty: boolean = true): boolean {
  tags.forEach((tag) => {
    if (tagIsInvalid(tag, ignore_empty)) {
      return true;
    }
  });
  return false;
}

/**
 * Conversion function between RequestTags and TagEntry[]
 * @param {RequestTags} tags
 * @returns {TagEntry[]}
 *
 * Conversion function to flatten RequestTags object. Useful for TagSelect
 * component
 **/
export function requestTagsToTagEntryList(tags: RequestTags): TagEntry[] {
  const tagEntries: TagEntry[] = [];
  Object.keys(tags).forEach((key) => {
    const values = tags[key];
    values.forEach((value) => {
      tagEntries.push({ key: key, value: value });
    });
  });
  return tagEntries;
}

/**
 * Conversion function between TagEntry[] and RequestTags
 * @param {TagEntry[]} tagEntry
 * @returns {RequestTags}
 *
 * Conversion function from TagEntry[] to RequestTags object. Useful for TagSelect
 * component
 **/
export function tagEntriesToRequestTags(tagEntry: TagEntry[]): RequestTags {
  const newTags: RequestTags = {};
  tagEntry.forEach((tag) => {
    if (tag.key in newTags) {
      if (!newTags[tag.key].includes(tag.value)) {
        newTags[tag.key].push(tag.value);
      }
    } else {
      newTags[tag.key] = [tag.value];
    }
  });
  return newTags;
}

/**
 * Computes which collection tag values were added and removed between two tag maps.
 *
 * A tag map is shaped as: `{ [key: string]: string[] }`, where each key can have multiple values.
 *
 * @param {RequestTags} currentTags - The existing collection tags in the Thorium backend.
 * @param {RequestTags} pendingTags - The edited (pending) collection tags.
 * @returns An object containing:
 *  - `toAdd`: keys/values present in `pendingTags` but not in `currentTags`
 *  - `toDelete`: keys/values present in `currentTags` but not in `pendingTags`
 */
export function diffTagUpdate(
  currentTags: RequestTags | undefined,
  pendingTags: RequestTags | undefined,
): { toAdd: RequestTags; toDelete: RequestTags } {
  // set current/pending tags to empty map if undefined
  const current = currentTags ?? {};
  const pending = pendingTags ?? {};
  // get a set of all keys both in current and pending
  const keys = new Set<string>([...Object.keys(current), ...Object.keys(pending)]);
  // declare maps of keys/values to add and delete
  const toAdd: RequestTags = {};
  const toDelete: RequestTags = {};
  for (const k of keys) {
    // get the set of values from current/pending, or default to empty set
    const currentValues = new Set(current[k] ?? []);
    const pendingValues = new Set(pending[k] ?? []);
    // values to add are everything in pending that's not in current
    const valuesToAdd = [...pendingValues].filter((v) => !currentValues.has(v));
    // values to delete are everything in current that's not in pending
    const valuesToDelete = [...currentValues].filter((v) => !pendingValues.has(v));
    // add the values to add and delete to our main maps
    if (valuesToAdd.length) {
      toAdd[k] = valuesToAdd;
    }
    if (valuesToDelete.length) {
      toDelete[k] = valuesToDelete;
    }
  }
  return { toAdd, toDelete };
}

/**
 * Check if a specific tag is empty
 * @param {TagEntry} tag - tag to check
 * @returns {boolean} whether tag contains key and/or value string
 *
 * If key and value, stripped of spaces / newlines, is empty, return true
 * */
export function tagEntryIsEmpty(tag: TagEntry): boolean {
  return tag.key.trim().length + tag.value.trim().length == 0;
}
