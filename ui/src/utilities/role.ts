import { Group, RoleKey, ThoriumRole, UserInfo } from '@models';

// get the Thorium role for a given user: Admin, Developer, or User
export function getThoriumRole(role: ThoriumRole) {
  if (typeof role == 'string') {
    return role;
  } else if (typeof role === 'object' && typeof role !== 'function' && role !== null) {
    if ('Developer' in role) {
      return 'Developer';
    } else {
      return '';
    }
  } else {
    // role might be null or function, either way thats invalid, use an empty string.
    return '';
  }
}

// get the Group role for a given user as a string: Owner, Manager, Monitor, or User
export function getGroupRole(group: Group, user: string) {
  if (group.owners.combined.includes(user)) {
    return 'Owner';
  } else if (group.managers.combined.includes(user)) {
    return 'Manager';
  } else if (group.monitors.combined.includes(user)) {
    return 'Monitor';
  } else if (group.users.combined.includes(user)) {
    return 'User';
  } else {
    return '';
  }
}

// check if user is a group admin
export function isGroupAdmin(group: Group, userInfo: UserInfo) {
  // user is a group admin if they are a manager, owner or Thorium admin
  return group.owners.combined.includes(userInfo.username) ||
    group.managers.combined.includes(userInfo.username) ||
    userInfo.role == RoleKey
    ? true
    : false;
}
