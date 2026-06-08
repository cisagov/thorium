import { Group, GroupUsers } from '@models/groups';

// return a list of groups from file submissions with no duplicates
export function getUniqueSubmissionGroups(submissions: any): string[] {
  const uniqueGroupsList: string[] = [];
  for (const submission of submissions) {
    uniqueGroupsList.push(...submission.groups.filter((group: string) => !uniqueGroupsList.includes(group)));
  }
  return uniqueGroupsList;
}

export function getAllGroupUsers(userObj: GroupUsers): string[] {
  const allUsers = [...userObj.combined, ...userObj.direct, ...userObj.metagroups];
  return [...new Set(allUsers)].sort();
}

export function hasOverlap(a: string[], b: string[]): boolean {
  const setB = new Set(b);
  return a.some((x) => setB.has(x));
}
