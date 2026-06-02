import { describe, it, expect } from 'vitest';

// project imports
import { RoleKey } from './users';

// The RoleKey enum is the source of truth for both the UserProfile role badge
// and the assignable roles in the UserBrowsing "Edit Role" dropdown
// (which iterates Object.keys(RoleKey)). It must match the backend UserRole enum.
describe('RoleKey', () => {
  it('matches the backend UserRole variants exactly', () => {
    expect(Object.values(RoleKey).sort()).toEqual(['Admin', 'Analyst', 'Developer', 'User']);
  });

  it('includes the Analyst role', () => {
    expect(RoleKey.Analyst).toBe('Analyst');
  });

  it('does not include the removed phantom Reporter role', () => {
    expect(Object.values(RoleKey)).not.toContain('Reporter');
    expect((RoleKey as Record<string, string>).Reporter).toBeUndefined();
  });
});
