import { describe, it, expect } from 'vitest';

// project imports
import { filterUsers } from './UserBrowsing';
import { Clause, ClauseCondition } from '@components/shared/inputs/omnibar/ClauseTypes';
import { RoleKey, ThoriumRole, UserInfo } from '@models/users';

type UserExtras = { email?: string; verified?: boolean; local?: boolean; has_image?: boolean };

// build a minimal UserInfo with the fields filterUsers depends on
function makeUser(username: string, role: ThoriumRole, groups: string[], extra: UserExtras = {}): UserInfo {
  return {
    username,
    role,
    email: extra.email ?? `${username}@thorium.dev`,
    groups,
    token: 'tok',
    token_expiration: '2099-01-01T00:00:00Z',
    settings: { theme: 'Dark' },
    local: extra.local ?? true,
    verified: extra.verified ?? true,
    has_image: extra.has_image ?? false,
  };
}

// Non-Developer roles serialize as plain strings at runtime; only Developer is an object.
// getThoriumRole resolves both shapes, so the fixtures mirror the real wire format here.
const alice = makeUser('alice', RoleKey.Admin as unknown as ThoriumRole, ['team-a'], { verified: true, local: true });
const bob = makeUser('bob', RoleKey.User as unknown as ThoriumRole, ['team-a', 'team-b'], { verified: false, local: true });
const carol = makeUser('carol', { Developer: { k8s: true, bare_metal: false, windows: false, external: false, kvm: false } }, ['team-c'], {
  verified: true,
  local: false,
});
const USERS = [alice, bob, carol];

function single(category: string, field: string, value: string): Clause {
  return { category, field, condition: ClauseCondition.Is, value: { value } };
}
function multi(category: string, field: string, values: string[]): Clause {
  return { category, field, condition: ClauseCondition.IsOneOf, value: { values } };
}
function includes(category: string, field: string, value: string): Clause {
  return { category, field, condition: ClauseCondition.Includes, value: { value } };
}

describe('filterUsers', () => {
  it('returns all users when there are no clauses', () => {
    expect(filterUsers(USERS, [])).toEqual(USERS);
  });

  it('filters by exact username (is)', () => {
    expect(filterUsers(USERS, [single('username', 'username', 'bob')])).toEqual([bob]);
  });

  it('filters by username substring (includes)', () => {
    expect(filterUsers(USERS, [includes('username', 'username', 'ali')])).toEqual([alice]);
  });

  it('filters by email substring (includes)', () => {
    expect(filterUsers(USERS, [includes('email', 'email', 'carol')])).toEqual([carol]);
  });

  it('filters by exact email (is)', () => {
    expect(filterUsers(USERS, [single('email', 'email', 'carol@thorium.dev')])).toEqual([carol]);
    // a substring is NOT an exact match
    expect(filterUsers(USERS, [single('email', 'email', 'carol')])).toEqual([]);
  });

  it('filters by the verified flag', () => {
    expect(filterUsers(USERS, [single('verified', 'verified', 'true')])).toEqual([alice, carol]);
    expect(filterUsers(USERS, [single('verified', 'verified', 'false')])).toEqual([bob]);
  });

  it('filters by the local flag', () => {
    expect(filterUsers(USERS, [single('local', 'local', 'true')])).toEqual([alice, bob]);
    expect(filterUsers(USERS, [single('local', 'local', 'false')])).toEqual([carol]);
  });

  it('filters by group membership overlap', () => {
    expect(filterUsers(USERS, [single('group', 'group', 'team-a')])).toEqual([alice, bob]);
    expect(filterUsers(USERS, [multi('group', 'group', ['team-b', 'team-c'])])).toEqual([bob, carol]);
  });

  it('filters by a single string role', () => {
    expect(filterUsers(USERS, [single('role', 'role', RoleKey.Admin)])).toEqual([alice]);
  });

  it('resolves the Developer object role to its string name', () => {
    expect(filterUsers(USERS, [single('role', 'role', RoleKey.Developer)])).toEqual([carol]);
  });

  it('filters by one of several roles', () => {
    expect(filterUsers(USERS, [multi('role', 'role', [RoleKey.Admin, RoleKey.User])])).toEqual([alice, bob]);
  });

  it('combines group and role filters (AND)', () => {
    const clauses = [single('group', 'group', 'team-a'), single('role', 'role', RoleKey.User)];
    expect(filterUsers(USERS, clauses)).toEqual([bob]);
  });

  it('returns no users when no one matches', () => {
    expect(filterUsers(USERS, [single('role', 'role', RoleKey.Analyst)])).toEqual([]);
  });
});
