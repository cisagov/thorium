import { test, expect, Page } from '@playwright/test';

// Catch + landing pages for the account-creation verification email link, mirroring the OAuth
// account-link pages. The catch page calls the verify endpoint via XHR (204 ok / uniform 401 expired)
// then navigates to the landing page. The real backend token consume is covered by the live harness.

async function mockApi(page: Page, verifyStatus: number = 204): Promise<void> {
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (url.includes('/users/verify/') && url.includes('/email/')) {
      return route.fulfill({ status: verifyStatus, contentType: 'application/json', body: '"This verification link has expired or was already used"' });
    }
    if (url.includes('/users/whoami')) {
      return route.fulfill({ status: 401, contentType: 'application/json', body: '"unauthorized"' });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
}

test.describe('Email verification pages', () => {
  test('catch page confirm navigates to the verified landing page on success', async ({ page }) => {
    await mockApi(page, 204);

    await page.goto('/users/verify/alice/email/tok');
    await page.getByRole('button', { name: 'Verify email' }).click();

    await page.waitForURL((u) => u.pathname.endsWith('/verified') && u.search.includes('status=ok'), { timeout: 15000 });
    await expect(page.getByText(/email is verified/i)).toBeVisible();
  });

  test('catch page confirm navigates to the expired landing page on a 401', async ({ page }) => {
    await mockApi(page, 401);

    await page.goto('/users/verify/alice/email/tok');
    await page.getByRole('button', { name: 'Verify email' }).click();

    await page.waitForURL((u) => u.pathname.endsWith('/verified') && u.search.includes('status=expired'), { timeout: 15000 });
    await expect(page.getByText(/expired or was already used/i)).toBeVisible();
  });

  test('verified landing page shows success and expired states', async ({ page }) => {
    await mockApi(page);

    await page.goto('/users/verify/alice/verified?status=ok');
    await expect(page.getByText(/email is verified/i)).toBeVisible();

    await page.goto('/users/verify/alice/verified?status=expired');
    await expect(page.getByText(/expired or was already used/i)).toBeVisible();
  });
});
