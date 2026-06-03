import { test, expect } from '@playwright/test';

// LIVE end-to-end OAuth test: drives the real Thorium OAuth API + a local mock-oauth2-server IdP
// (no /api mocking). Requires the API on :8085 with provider "mock" configured against
// http://localhost:9100/default and redirect_base http://localhost:8210. Run via the dedicated
// temp Playwright config (see the run command in the task notes), not the default suite.

const API_URL = process.env.THORIUM_API_URL || 'http://localhost:8080';

test.describe('OAuth live end-to-end (mock-oauth2-server)', () => {
  // Only run when a mock-configured OAuth API is actually reachable; otherwise skip so this
  // harness-specific test doesn't fail in environments without the local IdP setup.
  test.beforeEach(async ({ request }) => {
    let providers: string[] = [];
    try {
      const res = await request.get(`${API_URL}/api/oauth/`);
      if (res.ok()) providers = (await res.json()) as string[];
    } catch {
      // API not reachable
    }
    test.skip(!providers.includes('mock'), 'live OAuth harness (mock IdP + API) not running');
  });

  test('new user registers via OAuth, then signs in again as a returning user', async ({ page, context }) => {
    const stamp = Date.now();
    const sub = `e2e-sub-${stamp}`;
    const username = `e2euser${stamp}`;
    const email = `e2e${stamp}@example.com`;

    const signInAtIdp = async (subject: string) => {
      // The provider button triggers a full-page nav to the API /auth, which 303s to the mock IdP.
      await page.getByRole('button', { name: /Sign in with Mock/i }).click();
      await page.waitForURL(/localhost:9100/, { timeout: 20000 });
      // mock-oauth2-server interactive login: the "username" becomes the OIDC subject.
      await page.locator('input[name="username"]').fill(subject);
      await page.locator('input[type="submit"]').click();
    };

    const isInApp = (url: URL) => !url.pathname.startsWith('/oauth/') && url.pathname !== '/auth';

    // --- First time: unknown OIDC identity -> registration -> logged in ---
    await page.goto('/auth');
    await expect(page.getByRole('button', { name: /Sign in with Mock/i })).toBeVisible({ timeout: 15000 });
    await signInAtIdp(sub);

    await page.waitForURL(/\/oauth\/mock\/callback/, { timeout: 20000 });
    await page.getByPlaceholder('Choose a username').fill(username);
    await page.getByPlaceholder('Enter your email').fill(email);
    await expect(page.getByText('Username is available')).toBeVisible({ timeout: 8000 });
    await page.getByRole('button', { name: 'Create account' }).click();

    await page.waitForURL(isInApp, { timeout: 25000 });
    const cookies = await context.cookies();
    expect(cookies.find((c) => c.name === 'THORIUM_TOKEN')?.value).toBeTruthy();

    // --- Returning user: same OIDC subject is now linked -> straight in, no registration form ---
    await context.clearCookies();
    await page.goto('/auth');
    await expect(page.getByRole('button', { name: /Sign in with Mock/i })).toBeVisible({ timeout: 15000 });
    await signInAtIdp(sub);

    await page.waitForURL(isInApp, { timeout: 25000 });
    // The registration form must NOT appear for an already-linked identity.
    await expect(page.getByPlaceholder('Choose a username')).toHaveCount(0);
    const cookies2 = await context.cookies();
    expect(cookies2.find((c) => c.name === 'THORIUM_TOKEN')?.value).toBeTruthy();
  });
});
