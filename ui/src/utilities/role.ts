// project imports
import { RoleKey, ThoriumRole, UserInfo } from '@models/users';
import { Group, GroupRoleKey } from '@models/groups';

// NOTE: action-authorization predicates (who may modify/delete/develop) live in
// `permissions.ts` and mirror the backend. This module only resolves a user's
// role identity and its display metadata (badges).

/**
 * Resolve a user's Thorium role (Admin, Analyst, Developer, or User) from its serialized form.
 *
 * The Developer role is serialized as an object (`{ Developer: {...} }`) while every other role is
 * its plain string name, so the Developer case is detected by key presence.
 *
 * @param role - The serialized {@link ThoriumRole} from a user record.
 * @returns The corresponding {@link RoleKey}.
 */
export function getThoriumRole(role: ThoriumRole): RoleKey {
  // the Developer role is serialized as an object ({ Developer: {...} }),
  // every other role is serialized as its plain string name
  if (Object.keys(role).includes(RoleKey.Developer)) {
    return RoleKey.Developer;
  }
  return role as any as RoleKey;
}

/**
 * Resolve a user's role within a specific group.
 *
 * Checks membership lists in precedence order, mirroring the backend `Group::role`
 * (owner > manager > analyst > user > monitor), returning the highest role held.
 *
 * @param group - The group to resolve the role in.
 * @param user - The username to look up.
 * @returns The user's {@link GroupRoleKey}, or `''` if they hold no role in the group.
 */
export function getGroupRole(group: Group, user: string): GroupRoleKey | '' {
  if (group.owners.combined.includes(user)) {
    return GroupRoleKey.Owner;
  } else if (group.managers.combined.includes(user)) {
    return GroupRoleKey.Manager;
  } else if (group.analysts.includes(user)) {
    return GroupRoleKey.Analyst;
  } else if (group.users.combined.includes(user)) {
    return GroupRoleKey.User;
  } else if (group.monitors.combined.includes(user)) {
    return GroupRoleKey.Monitor;
  } else {
    return '';
  }
}

/// Display metadata for a role badge
export interface RoleBadgeMeta {
  /// The text shown on the badge
  label: string;
  /// The theme color class for the badge background
  className: string;
}

// Thorium-wide role badge styling, keyed by role
const THORIUM_ROLE_BADGES: Record<RoleKey, RoleBadgeMeta> = {
  [RoleKey.Admin]: { label: 'Admin', className: 'bg-maroon' },
  [RoleKey.Analyst]: { label: 'Analyst', className: 'bg-goldenrod' },
  [RoleKey.Developer]: { label: 'Developer', className: 'bg-corn-flower' },
  [RoleKey.User]: { label: 'User', className: 'bg-cadet' },
};

/**
 * Get the badge display metadata (label + color class) for a user's Thorium role.
 *
 * @param role - The serialized {@link ThoriumRole} to render a badge for.
 * @returns The {@link RoleBadgeMeta} for that role.
 */
export function getThoriumRoleBadge(role: ThoriumRole): RoleBadgeMeta {
  return THORIUM_ROLE_BADGES[getThoriumRole(role)];
}

/// Display metadata for a group role badge, including its tooltip
export interface GroupRoleBadgeMeta extends RoleBadgeMeta {
  /// The tooltip describing the role's abilities
  tooltip: string;
}

// Group role badge styling and tooltips, keyed by group role
const GROUP_ROLE_BADGES: Record<GroupRoleKey, GroupRoleBadgeMeta> = {
  [GroupRoleKey.Owner]: {
    label: 'Owner',
    className: 'bg-dark-slate',
    tooltip:
      'You are an Owner of this group. Owners can add/remove any member within the group. An owner can also access and edit all group resources.',
  },
  [GroupRoleKey.Manager]: {
    label: 'Manager',
    className: 'bg-corn-flower',
    tooltip: 'You are a Manager of this group. Managers can edit non-Owner membership within the group as well as all group resources.',
  },
  [GroupRoleKey.Analyst]: {
    label: 'Analyst',
    className: 'bg-goldenrod',
    tooltip: 'You are an Analyst. Analysts have global access to all samples in Thorium and can analyze resources across every group.',
  },
  [GroupRoleKey.User]: {
    label: 'User',
    className: 'bg-cadet',
    tooltip:
      'You are a User in this group. A user can view group membership and resources. Users can also upload samples/repos and conduct analysis on them.',
  },
  [GroupRoleKey.Monitor]: {
    label: 'Monitor',
    className: 'bg-grey',
    tooltip:
      'You are a Monitor of this group. Monitors can view group membership, track running jobs and analyze tool results. Monitors cannot run jobs or modify any group resources.',
  },
};

/**
 * Get the badge display metadata (label, color class, tooltip) for a user's role in a group.
 *
 * Falls back to a Thorium Admin badge for admins who hold no explicit group role, since admins
 * always have full access.
 *
 * @param group - The group to resolve the role badge for.
 * @param userInfo - The user to render a badge for.
 * @returns The {@link GroupRoleBadgeMeta}, or `null` if the user has no group role and is not an admin.
 */
export function getGroupRoleBadge(group: Group, userInfo: UserInfo): GroupRoleBadgeMeta | null {
  const role = getGroupRole(group, userInfo.username);
  if (role) {
    return GROUP_ROLE_BADGES[role];
  }
  // Thorium admins always have full access even without an explicit group role
  if (getThoriumRole(userInfo.role) == RoleKey.Admin) {
    return {
      label: 'Admin',
      className: 'bg-maroon',
      tooltip: 'You are a Thorium admin. You have all the permissions.',
    };
  }
  return null;
}
