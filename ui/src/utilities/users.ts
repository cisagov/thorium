import { RoleKey } from '@models';

// get the Thorium role for a given user: Admin, Developer, or User
export const getUserRole = (role: any): RoleKey | undefined => {
  if (typeof role == 'string') {
    if (Object.values(RoleKey).includes(role as RoleKey)) {
      return role as RoleKey;
    }
  } else if (typeof role === 'object' && typeof role !== 'function' && role !== null) {
    if ('Developer' in role) {
      return RoleKey.Developer;
    }
  }
  // catch all no role found
  return undefined;
};
