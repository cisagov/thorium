import { describe, it, expect } from 'vitest';

// project imports
import { getThoriumRole, getThoriumRoleBadge, getGroupRole, getGroupRoleBadge } from './role';
import { Group, GroupRoleKey, GroupUsers } from '@models/groups';
import { RoleKey, ThoriumRole, UserInfo } from '@models/users';

// build a GroupUsers role bucket from a list of combined members
function roleBucket(combined: string[] = []): GroupUsers {
  return { combined, direct: combined, metagroups: [] };
}

// build a minimal Group with the given per-role membership
function makeGroup(overrides: Partial<Group> = {}): Group {
  return {
    name: 'test-group',
    owners: roleBucket(),
    managers: roleBucket(),
    analysts: [],
    users: roleBucket(),
    monitors: roleBucket(),
    allowed: {
      files: true,
      repos: true,
      tags: true,
      images: true,
      pipelines: true,
      reactions: true,
      results: true,
      comments: true,
      entities: true,
    },
    ...overrides,
  };
}

// build a minimal UserInfo with the given username and Thorium role.
// Non-developer roles are plain strings at runtime; Developer is an object.
function makeUser(username: string, role: ThoriumRole): UserInfo {
  return {
    username,
    role,
    email: `${username}@example.com`,
    groups: [],
    token: 'token',
    token_expiration: '2099-01-01T00:00:00Z',
    settings: { theme: 'Dark' },
    local: true,
    verified: true,
  };
}

// runtime role values as the API actually serializes them
const ADMIN = 'Admin' as unknown as ThoriumRole;
const ANALYST = 'Analyst' as unknown as ThoriumRole;
const USER = 'User' as unknown as ThoriumRole;
const DEVELOPER = { Developer: { k8s: true, bare_metal: false, windows: false, external: false, kvm: false } } as unknown as ThoriumRole;

describe('getThoriumRole', () => {
  it('resolves each Thorium-wide role, including Analyst', () => {
    expect(getThoriumRole(ADMIN)).toBe(RoleKey.Admin);
    expect(getThoriumRole(ANALYST)).toBe(RoleKey.Analyst);
    expect(getThoriumRole(USER)).toBe(RoleKey.User);
    expect(getThoriumRole(DEVELOPER)).toBe(RoleKey.Developer);
  });
});

describe('getThoriumRoleBadge', () => {
  it('returns a label and color for every role (Analyst included)', () => {
    expect(getThoriumRoleBadge(ADMIN)).toEqual({ label: 'Admin', className: 'bg-maroon' });
    expect(getThoriumRoleBadge(ANALYST)).toEqual({ label: 'Analyst', className: 'bg-goldenrod' });
    expect(getThoriumRoleBadge(DEVELOPER)).toEqual({ label: 'Developer', className: 'bg-corn-flower' });
    expect(getThoriumRoleBadge(USER)).toEqual({ label: 'User', className: 'bg-cadet' });
  });
});

describe('getGroupRole', () => {
  it('returns the matching group role for each membership type', () => {
    expect(getGroupRole(makeGroup({ owners: roleBucket(['alice']) }), 'alice')).toBe(GroupRoleKey.Owner);
    expect(getGroupRole(makeGroup({ managers: roleBucket(['bob']) }), 'bob')).toBe(GroupRoleKey.Manager);
    expect(getGroupRole(makeGroup({ analysts: ['carol'] }), 'carol')).toBe(GroupRoleKey.Analyst);
    expect(getGroupRole(makeGroup({ users: roleBucket(['dave']) }), 'dave')).toBe(GroupRoleKey.User);
    expect(getGroupRole(makeGroup({ monitors: roleBucket(['erin']) }), 'erin')).toBe(GroupRoleKey.Monitor);
  });

  it('returns the analyst role for a global analyst with no other membership', () => {
    expect(getGroupRole(makeGroup({ analysts: ['carol'] }), 'carol')).toBe(GroupRoleKey.Analyst);
  });

  it('returns empty string for a non-member', () => {
    expect(getGroupRole(makeGroup(), 'nobody')).toBe('');
  });

  it('follows backend precedence (owner > manager > analyst > user > monitor)', () => {
    const group = makeGroup({
      owners: roleBucket(['frank']),
      managers: roleBucket(['frank']),
      analysts: ['frank'],
    });
    expect(getGroupRole(group, 'frank')).toBe(GroupRoleKey.Owner);
    const group2 = makeGroup({ analysts: ['grace'], users: roleBucket(['grace']) });
    expect(getGroupRole(group2, 'grace')).toBe(GroupRoleKey.Analyst);
  });
});

describe('getGroupRoleBadge', () => {
  it('returns an Analyst badge (with tooltip) for a global analyst', () => {
    const group = makeGroup({ analysts: ['carol'] });
    const badge = getGroupRoleBadge(group, makeUser('carol', ANALYST));
    expect(badge?.label).toBe('Analyst');
    expect(badge?.className).toBe('bg-goldenrod');
    expect(badge?.tooltip).toContain('Analyst');
  });

  it('returns the explicit group role badge over the analyst fallback', () => {
    const group = makeGroup({ owners: roleBucket(['alice']) });
    const badge = getGroupRoleBadge(group, makeUser('alice', USER));
    expect(badge?.label).toBe('Owner');
    expect(badge?.tooltip).toContain('Owner');
  });

  it('falls back to an Admin badge for Thorium admins with no group role', () => {
    const badge = getGroupRoleBadge(makeGroup(), makeUser('admin', ADMIN));
    expect(badge?.label).toBe('Admin');
    expect(badge?.className).toBe('bg-maroon');
  });

  it('returns null for a user with no role who is not an admin', () => {
    expect(getGroupRoleBadge(makeGroup(), makeUser('nobody', USER))).toBeNull();
  });
});
