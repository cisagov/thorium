import { test, expect } from '@playwright/test';
import { authenticate, buildClient, healthCheck, loginViaUI, TEST_USER, TEST_PASS } from './helpers';

// LIVE test: validates the profile-icon feature end-to-end against a real API. A user uploads an
// icon on /profile, which is resized client-side and POSTed as multipart to /users/image (stored in
// S3). The small round avatar — fetched lazily from GET /users/user/{username}/image — must then
// render to the left of the username in the top nav (and survive a reload, proving has_image
// round-trips through whoami). Removing it (DELETE /users/image) must clear the avatar everywhere.
//
// A tiny valid 1x1 PNG used as the uploaded file. The UI resizes whatever is uploaded, so the
// source dimensions don't matter — only that it decodes as an image.
const ONE_BY_ONE_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

test.describe('Profile icon (live)', () => {
  test.beforeAll(async () => {
    const healthy = await healthCheck().catch(() => false);
    test.skip(!healthy, 'requires a live Thorium API at THORIUM_API_URL');
  });

  // Clear any previously-set icon so reruns start from a known state.
  test.afterEach(async () => {
    const token = await authenticate(TEST_USER, TEST_PASS);
    await buildClient(token).delete('/users/image', { validateStatus: () => true });
  });

  test('upload sets the nav avatar, persists across reload, and removal clears it', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');

    // avatars are loaded lazily from the image route and rendered as object (blob) URLs
    const navAvatar = page.locator('nav img[src^="blob:"]');
    // no avatar before an icon is set
    await expect(navAvatar).toHaveCount(0);

    // upload an icon via the hidden file input (resized client-side, POSTed as multipart)
    await page.locator('input[type="file"]').setInputFiles({
      name: 'icon.png',
      mimeType: 'image/png',
      buffer: ONE_BY_ONE_PNG,
    });

    // the small round avatar appears in the nav, left of the username
    await expect(navAvatar).toHaveCount(1, { timeout: 15000 });

    // it persisted server-side: a fresh load (whoami -> has_image -> image route) still shows it
    await page.reload();
    await page.waitForLoadState('networkidle');
    await expect(page.locator('nav img[src^="blob:"]')).toHaveCount(1, { timeout: 15000 });

    // remove it and confirm the avatar disappears
    await page.getByRole('button', { name: 'Delete this profile picture' }).click();
    await expect(page.locator('nav img[src^="blob:"]')).toHaveCount(0, { timeout: 15000 });
  });
});
