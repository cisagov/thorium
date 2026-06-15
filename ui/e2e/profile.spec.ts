import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { MOCK_USER } from './helpers';

// Mocked test: validates the general /profile page behavior against a stubbed API (no live backend
// needed). It covers the placeholder-vs-avatar rendering and Upload/Change/Remove action wiring, the
// account sections (groups, role, sign-in badges), the token show/hide toggle, and the theme select
// persisting through PATCH /users. The dedicated icon round-trip (multipart upload + nav avatar) lives
// in profile-icon.spec.ts / profile-icon-live.spec.ts; this spec focuses on the surrounding page.

// A tiny valid 1x1 PNG returned by the mocked image route and used as the uploaded file.
const ONE_BY_ONE_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

// Stand up a stateful whoami + image + update mock. `has_image` and the theme are reflected through
// whoami so the page's optimistic refresh (refreshUserInfo) sees the new values.
async function setupProfileMock(page: Page) {
  const state = { hasImage: false, theme: 'Dark', patchedThemes: [] as string[] };

  // catch-all first so the specific routes registered below take priority
  await page.route('**/api/**', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));

  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ...MOCK_USER, has_image: state.hasImage, settings: { theme: state.theme } }),
    }),
  );

  // lazy icon fetch: 200 with PNG bytes when set, 404 otherwise
  await page.route('**/api/users/user/*/image', (route) =>
    state.hasImage
      ? route.fulfill({ status: 200, contentType: 'image/png', body: ONE_BY_ONE_PNG })
      : route.fulfill({ status: 404, body: '' }),
  );

  // upload (POST) / remove (DELETE) toggle the stored icon state
  await page.route('**/api/users/image', (route) => {
    const method = route.request().method();
    if (method === 'POST') {
      state.hasImage = true;
      return route.fulfill({ status: 204, body: '' });
    }
    if (method === 'DELETE') {
      state.hasImage = false;
      return route.fulfill({ status: 204, body: '' });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });

  // theme change: PATCH /users/ persists the new theme so the follow-up whoami reflects it
  await page.route('**/api/users/', (route) => {
    if (route.request().method() === 'PATCH') {
      const body = route.request().postDataJSON() as { settings?: { theme?: string } };
      const theme = body?.settings?.theme;
      if (theme) {
        state.theme = theme;
        state.patchedThemes.push(theme);
      }
      return route.fulfill({ status: 204, body: '' });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });

  await page.context().addCookies([{ name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' }]);
  return state;
}

test.describe('Profile page (mocked)', () => {
  test('shows the placeholder icon and no delete control when no icon is set', async ({ page }) => {
    await setupProfileMock(page);
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    // no avatar image anywhere until an icon is uploaded — the placeholder icon (svg) stands in
    await expect(page.locator('img[src^="blob:"]')).toHaveCount(0);
    // the avatar itself is the click-to-upload control; the delete control only appears once set
    await expect(page.getByRole('button', { name: 'Click to upload profile picture' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Delete this profile picture' })).toHaveCount(0);
  });

  test('uploading shows the profile avatar and the delete control, which clears it', async ({ page }) => {
    await setupProfileMock(page);
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    await page.locator('input[type="file"]').setInputFiles({ name: 'icon.png', mimeType: 'image/png', buffer: ONE_BY_ONE_PNG });

    // the large avatar on the profile page renders the uploaded icon as an object URL
    await expect(page.locator('img[src^="blob:"]').first()).toBeVisible({ timeout: 15000 });
    // the corner delete (trash) control now appears
    await expect(page.getByRole('button', { name: 'Delete this profile picture' })).toBeVisible();

    // deleting it restores the placeholder and hides the delete control
    await page.getByRole('button', { name: 'Delete this profile picture' }).click();
    await expect(page.locator('img[src^="blob:"]')).toHaveCount(0, { timeout: 15000 });
    await expect(page.getByRole('button', { name: 'Delete this profile picture' })).toHaveCount(0);
  });

  test('renders the account sections: groups, role, and sign-in badges', async ({ page }) => {
    await setupProfileMock(page);
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    // group membership and Thorium role render as color pills (scoped by color class so the sidebar's
    // "Admin" nav item doesn't collide with the role badge)
    await expect(page.locator('span.bg-blue', { hasText: 'system' })).toBeVisible();
    await expect(page.locator('span.bg-maroon', { hasText: 'Admin' })).toBeVisible();
    // read-only sign-in method badges
    await expect(page.getByText('Local Login', { exact: true })).toBeVisible();
    await expect(page.getByText('Email verified', { exact: true })).toBeVisible();
  });

  test('token is masked until the eye toggle is clicked, then can be hidden again', async ({ page }) => {
    await setupProfileMock(page);
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    // masked by default — the raw token is not shown
    await expect(page.getByText(MOCK_USER.token)).toHaveCount(0);

    // reveal it via the eye toggle at the end of the token
    await page.getByRole('button', { name: 'Show token' }).click();
    await expect(page.getByText(MOCK_USER.token)).toBeVisible();

    // and mask it again
    await page.getByRole('button', { name: 'Hide token' }).click();
    await expect(page.getByText(MOCK_USER.token)).toHaveCount(0);
  });

  test('changing the theme PATCHes the user and reflects the new value', async ({ page }) => {
    const state = await setupProfileMock(page);
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    const themeSelect = page.locator('select');
    await expect(themeSelect).toHaveValue('Dark');

    // pick a new theme; the page PATCHes /users/ then refreshes whoami to apply it
    await themeSelect.selectOption('Ocean');
    await expect.poll(() => state.patchedThemes).toContain('Ocean');
    await expect(themeSelect).toHaveValue('Ocean', { timeout: 15000 });
  });
});
