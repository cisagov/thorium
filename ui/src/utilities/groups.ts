// return a list of groups from file submissions with no duplicates
export function getUniqueSubmissionGroups(submissions: any): string[] {
  const uniqueGroupsList: string[] = [];
  for (const submission of submissions) {
    uniqueGroupsList.push(...submission.groups.filter((group: string) => !uniqueGroupsList.includes(group)));
  }
  return uniqueGroupsList;
}
