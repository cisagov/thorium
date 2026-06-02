// project imports
import { RoleKey } from '@models/users';

/**
 * Resolve a user's Thorium role from its raw serialized form.
 *
 * Most roles serialize as a plain string (e.g. `"Admin"`); the Developer role serializes as an
 * object (`{ Developer: {...} }`), which is detected via the `Developer` key.
 *
 * @param role - The raw role value from a user record (string or object form).
 * @returns The matching {@link RoleKey}, or `undefined` if the value is not a recognized role.
 */
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
