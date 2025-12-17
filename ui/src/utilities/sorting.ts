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
