import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { MOCK_USER } from './helpers';

// Mocked test: validates the profile-icon UI wiring against a stubbed API (no live backend needed).
// It proves the nav avatar reflects whoami's `has_image`, that the icon is fetched lazily from the
// dedicated image route (never embedded in whoami), that uploading POSTs multipart to /users/image
// and then shows the avatar, and that removing it DELETEs and hides the avatar again.

// A tiny valid 1x1 PNG returned by the mocked image route.
const ONE_BY_ONE_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

// Stand up a stateful whoami + image-route mock. `has_image` is reflected through whoami and the
// icon is served from GET /users/user/{username}/image, mirroring the real lazy-load path.
async function setupProfileMock(page: Page) {
  const state = { hasImage: false };

  // catch-all first so the specific routes registered below take priority
  await page.route('**/api/**', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));

  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ...MOCK_USER, has_image: state.hasImage }),
    }),
  );

  // lazy icon fetch: 200 with PNG bytes when set, 404 otherwise
  await page.route('**/api/users/user/*/image', (route) =>
    state.hasImage
      ? route.fulfill({ status: 200, contentType: 'image/png', body: ONE_BY_ONE_PNG })
      : route.fulfill({ status: 404, body: '' }),
  );

  // upload (POST) / remove (DELETE) toggle the stored state
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

  await page.context().addCookies([{ name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' }]);
  return state;
}

test.describe('Profile icon (mocked)', () => {
  test('whoami never embeds the icon bytes', async ({ page }) => {
    await setupProfileMock(page);
    const whoamiBodies: string[] = [];
    page.on('response', (res) => {
      if (res.url().includes('/api/users/whoami')) {
        void res.text().then((b) => whoamiBodies.push(b));
      }
    });

    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    expect(whoamiBodies.length).toBeGreaterThan(0);
    // the lazy-load contract: whoami carries only the has_image flag, never image bytes
    for (const body of whoamiBodies) {
      expect(body).not.toContain('data:image');
      expect(body).toContain('has_image');
    }
  });

  test('renders an existing icon in the nav avatar', async ({ page }) => {
    const state = await setupProfileMock(page);
    state.hasImage = true;

    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    // the icon is fetched lazily and shown as an object URL
    await expect(page.locator('nav img[src^="blob:"]')).toHaveCount(1, { timeout: 15000 });
  });

  test('upload sets the avatar and remove clears it', async ({ page }) => {
    await setupProfileMock(page);

    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    const navAvatar = page.locator('nav img[src^="blob:"]');
    // no avatar until an icon is uploaded
    await expect(navAvatar).toHaveCount(0);

    // upload via the hidden file input (resized client-side, POSTed as multipart)
    await page.locator('input[type="file"]').setInputFiles({
      name: 'icon.png',
      mimeType: 'image/png',
      buffer: ONE_BY_ONE_PNG,
    });

    // the icon round-trips through (mocked) whoami + the image route and renders in the nav
    await expect(navAvatar).toHaveCount(1, { timeout: 15000 });

    // remove it and confirm the avatar disappears
    await page.getByRole('button', { name: 'Delete this profile picture' }).click();
    await expect(navAvatar).toHaveCount(0, { timeout: 15000 });
  });
});
