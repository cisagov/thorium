/**
 * Comparator for sorting strings that may hold integers.
 *
 * When both values parse as numbers they are compared numerically (so `"10"` sorts after `"9"`);
 * otherwise they fall back to lexicographic comparison.
 *
 * @param first - The first value to compare.
 * @param second - The second value to compare.
 * @returns A negative number if `first` sorts before `second`, `0` if equal, positive otherwise.
 */
export function sortIntegerStrings(first: string, second: string) {
  if (!isNaN(Number(first)) && !isNaN(Number(second))) {
    return Number(first) - Number(second);
  }
  // first or second is NaN
  if (first > second) {
    return 1;
  } else if (first === second) {
    return 0;
  }
  //else if (first < second) ...
  return -1;
}
