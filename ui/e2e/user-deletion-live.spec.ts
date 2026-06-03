import { test, expect } from '@playwright/test';
import { authenticate, buildClient, TEST_USER, TEST_PASS } from './helpers';

// LIVE test: validates the full user-deletion flow against a real API (an admin creates a throwaway
// user, then deletes it and confirms it's gone). Creating a local user requires the instance secret
// key, so this is gated on THORIUM_SECRET_KEY. The Redis-key cleanup that deletion performs
// (user_data, users set, users_token_map, users_email_map, oauth:alias_to_user) cannot be asserted
// from Playwright — it is validated separately at the Redis level (see the OAuth harness notes).
const SECRET_KEY = process.env.THORIUM_SECRET_KEY || '';

test.describe('User deletion (live)', () => {
  test.skip(!SECRET_KEY, 'requires THORIUM_SECRET_KEY (admin creds + secret key) to create a throwaway user');

  test('an admin creates then deletes a user and it is removed', async () => {
    const adminToken = await authenticate(TEST_USER, TEST_PASS);
    const admin = buildClient(adminToken);
    const username = `deltest${Date.now()}`;
    const email = `${username}@example.test`;

    // create a throwaway local user (secret key required for local/admin accounts)
    const created = await admin.post(
      '/users/',
      { username, email, password: 'DelTestP@ss1', role: 'User' },
      { headers: { 'secret-key': SECRET_KEY }, validateStatus: () => true },
    );
    expect([200, 204]).toContain(created.status);

    // it exists
    const before = await admin.get(`/users/user/${username}`, { validateStatus: () => true });
    expect(before.status).toBe(200);

    // delete it
    const del = await admin.delete(`/users/delete/${username}`, { validateStatus: () => true });
    expect(del.status).toBe(204);

    // it's gone
    const after = await admin.get(`/users/user/${username}`, { validateStatus: () => true });
    expect(after.status).toBe(404);
  });
});
