import { describe, it, expect } from 'vitest';

// project imports
import { getAllGroupUsers, getGroupMemberCount, getUniqueSubmissionGroups, hasOverlap } from './groups';
import { Group, GroupUsers } from '@models/groups';

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

describe('getGroupMemberCount', () => {
  it('returns 0 for an empty group', () => {
    expect(getGroupMemberCount(makeGroup())).toBe(0);
  });

  it('includes analysts in the total (matches backend member_count)', () => {
    const group = makeGroup({
      owners: roleBucket(['alice']),
      managers: roleBucket(['bob']),
      analysts: ['carol', 'dave'],
      users: roleBucket(['erin']),
      monitors: roleBucket(['frank']),
    });
    // 1 owner + 1 manager + 2 analysts + 1 user + 1 monitor = 6
    expect(getGroupMemberCount(group)).toBe(6);
  });

  it('counts a group that only has analysts', () => {
    expect(getGroupMemberCount(makeGroup({ analysts: ['carol', 'dave'] }))).toBe(2);
  });
});

describe('getUniqueSubmissionGroups', () => {
  it('de-duplicates groups across submissions', () => {
    const submissions = [{ groups: ['a', 'b'] }, { groups: ['b', 'c'] }];
    expect(getUniqueSubmissionGroups(submissions)).toEqual(['a', 'b', 'c']);
  });
});

describe('getAllGroupUsers', () => {
  it('merges combined, direct, and metagroups, de-duplicated and sorted', () => {
    const users: GroupUsers = { combined: ['bob', 'alice'], direct: ['alice'], metagroups: ['carol', 'bob'] };
    expect(getAllGroupUsers(users)).toEqual(['alice', 'bob', 'carol']);
  });

  it('returns an empty list when no members exist', () => {
    expect(getAllGroupUsers({ combined: [], direct: [], metagroups: [] })).toEqual([]);
  });
});

describe('hasOverlap', () => {
  it('returns true when the lists share at least one element', () => {
    expect(hasOverlap(['a', 'b'], ['x', 'b'])).toBe(true);
  });

  it('returns false when the lists are disjoint', () => {
    expect(hasOverlap(['a', 'b'], ['x', 'y'])).toBe(false);
  });

  it('returns false when either list is empty', () => {
    expect(hasOverlap([], ['a'])).toBe(false);
    expect(hasOverlap(['a'], [])).toBe(false);
  });
});
