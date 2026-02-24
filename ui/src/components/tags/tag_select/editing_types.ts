/* These are only relevant for the tag select component.
 * Used across a few different files; a little cleaner when all pulled out.
 * */
export enum EditingMode {
  Disabled, // Not Editing -- tag select is inactive
  First, // editing first element of tag (tag key)
  Second, // editing second element of tag (tag value)
}

export type FocusState = {
  // enum -- either NOT, FIRST, SECOND.
  editMode: EditingMode;
  // index of edited tag in tag array
  idx: number;
};

export enum KeyName {
  Enter = 'Enter',
  Tab = 'Tab',
  ArrowDown = 'ArrowDown',
  ArrowUp = 'ArrowUp',
}
