import { test, expect, Page } from '@playwright/test';
import { MOCK_USER } from './helpers';

// These specs exercise the OAuth UI against mocked /api/oauth responses (no live IdP needed).
// The full IdP round-trip is validated manually via the mock-oauth2-server harness (see SPEC.md).

const AUTH_RESPONSE = { token: 'oauth-token', expires: '2099-01-01T00:00:00Z' };

interface ApiMocks {
  providers?: string[] | number; // GET /oauth/ — array (200) or a status code (e.g. 401 = disabled)
  callback?: object; // GET /oauth/{p}/callback body
  usernameStatus?: number; // POST /oauth/{p}/username/available status (204 available / 409 taken)
  register?: object | string; // POST /oauth/{p}/register body
  registerStatus?: number; // POST /oauth/{p}/register status (200 created / 409 link-or-conflict)
  linkStatus?: number; // GET /oauth/{p}/link status (204 linked / 401 expired)
  whoami?: object | null; // GET /users/whoami (object = logged in, null = 401)
}

// Single dispatcher for all /api traffic so route precedence is unambiguous (mirrors helpers.ts).
async function mockApi(page: Page, m: ApiMocks): Promise<void> {
  await page.route('**/api/**', async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    if (/\/api\/oauth\/(\?.*)?$/.test(url)) {
      if (typeof m.providers === 'number') {
        return route.fulfill({ status: m.providers, contentType: 'application/json', body: '"OAuth is not configured!"' });
      }
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(m.providers ?? []) });
    }
    if (url.includes('/oauth/') && url.includes('/callback')) {
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(m.callback ?? {}) });
    }
    if (url.includes('/oauth/') && url.includes('/username/available')) {
      return route.fulfill({ status: m.usernameStatus ?? 204 });
    }
    if (url.includes('/oauth/') && url.includes('/register')) {
      return route.fulfill({
        status: m.registerStatus ?? 200,
        contentType: 'application/json',
        body: JSON.stringify(m.register ?? AUTH_RESPONSE),
      });
    }
    if (url.includes('/oauth/') && url.includes('/link')) {
      // DELETE = revoke (XHR, 204); GET = confirm (XHR, 204 linked / 401 expired). The page maps the
      // result to a client-side navigate to the /linked landing page.
      if (method === 'DELETE') {
        return route.fulfill({ status: 204 });
      }
      return route.fulfill({ status: m.linkStatus ?? 204, contentType: 'application/json', body: '"This account-link has expired or was already used"' });
    }
    if (url.includes('/users/whoami')) {
      if (m.whoami) {
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(m.whoami) });
      }
      return route.fulfill({ status: 401, contentType: 'application/json', body: '"unauthorized"' });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
}

test.describe('OAuth login buttons on /auth', () => {
  test('renders a button per configured provider alongside the password form', async ({ page }) => {
    await mockApi(page, { providers: ['google', 'github'] });

    await page.goto('/auth');

    await expect(page.getByRole('button', { name: /Sign in with Google/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Sign in with GitHub/i })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Login' })).toBeVisible();
    await expect(page.getByPlaceholder('username')).toBeVisible();
  });

  test('shows no provider buttons when OAuth is disabled (401)', async ({ page }) => {
    await mockApi(page, { providers: 401 });

    await page.goto('/auth');

    await expect(page.getByRole('button', { name: 'Login' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Sign in with/i })).toHaveCount(0);
  });
});

test.describe('OAuth callback', () => {
  test('Authed result logs the user in and redirects away from the callback', async ({ page }) => {
    await mockApi(page, { callback: { Authed: AUTH_RESPONSE }, whoami: MOCK_USER });

    await page.goto('/oauth/google/callback?code=abc&state=xyz');

    await page.waitForURL((url) => !url.pathname.includes('/oauth/'), { timeout: 15000 });
    expect(new URL(page.url()).pathname).toBe('/');
  });

  test('NewUser with an available username creates the account', async ({ page }) => {
    await mockApi(page, {
      callback: { NewUser: 'session-123' },
      usernameStatus: 204,
      register: { Authed: AUTH_RESPONSE },
      whoami: MOCK_USER,
    });

    await page.goto('/oauth/google/callback?code=abc&state=xyz');

    await page.getByPlaceholder('Choose a username').fill('alice');
    await page.getByPlaceholder('Enter your email').fill('alice@example.com');
    await expect(page.getByText('Username is available')).toBeVisible({ timeout: 5000 });

    await page.getByRole('button', { name: 'Create account' }).click();
    await page.waitForURL((url) => !url.pathname.includes('/oauth/'), { timeout: 15000 });
  });

  test('NewUser with a taken username switches to Link Account and sends the link email', async ({ page }) => {
    await mockApi(page, {
      callback: { NewUser: 'session-123' },
      usernameStatus: 409, // username taken -> link mode
      registerStatus: 409,
      register: 'A user with this email already exists. Please check your email for a account link email!',
    });

    await page.goto('/oauth/google/callback?code=abc&state=xyz');

    await page.getByPlaceholder('Choose a username').fill('existinguser');
    await expect(page.getByText(/already exists/i)).toBeVisible({ timeout: 5000 });
    const linkBtn = page.getByRole('button', { name: 'Link Account' });
    await expect(linkBtn).toBeVisible();
    // The create-mode button must not be present in link mode.
    await expect(page.getByRole('button', { name: 'Create account' })).toHaveCount(0);

    await page.getByPlaceholder('Enter your email').fill('existinguser@example.com');
    await linkBtn.click();

    await expect(page.getByText(/check your email to finish linking/i)).toBeVisible({ timeout: 5000 });
  });

  test('provider cancel/error shows a friendly message with retry', async ({ page }) => {
    await mockApi(page, {});

    await page.goto('/oauth/google/callback?error=access_denied');

    await expect(page.getByText(/cancelled or denied/i)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Try again' })).toBeVisible();
  });
});

test.describe('OAuth account-link pages', () => {
  test('catch page confirm navigates to the linked landing page on success', async ({ page }) => {
    await mockApi(page, { linkStatus: 204 });

    await page.goto('/oauth/google/link?username=alice&token=link-token');
    await page.getByRole('button', { name: 'Confirm' }).click();

    await page.waitForURL((url) => url.pathname.endsWith('/linked') && url.search.includes('status=ok'), { timeout: 15000 });
    await expect(page.getByText(/now linked to your account/i)).toBeVisible();
  });

  test('catch page confirm navigates to the expired landing page on a 401', async ({ page }) => {
    await mockApi(page, { linkStatus: 401 });

    await page.goto('/oauth/google/link?username=alice&token=link-token');
    await page.getByRole('button', { name: 'Confirm' }).click();

    await page.waitForURL((url) => url.pathname.endsWith('/linked') && url.search.includes('status=expired'), { timeout: 15000 });
    await expect(page.getByText(/expired or was already used/i)).toBeVisible();
  });

  test('linked landing page shows success and expired states', async ({ page }) => {
    await mockApi(page, {});

    await page.goto('/oauth/google/linked?status=ok');
    await expect(page.getByText(/now linked to your account/i)).toBeVisible();

    await page.goto('/oauth/google/linked?status=expired');
    await expect(page.getByText(/expired or was already used/i)).toBeVisible();
  });
});
