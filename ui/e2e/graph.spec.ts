import { test, expect } from '@playwright/test';
import path from 'path';
import {
  authenticate,
  createEntity,
  createAssociation,
  deleteEntity,
  uploadFile,
  snapshot,
} from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');
const USER = process.env.THORIUM_USER || 'test';
const PASS = process.env.THORIUM_PASS || 'INSECURE_DEV_PASSWORD';

async function loginViaUI(page: import('@playwright/test').Page) {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.locator('input[placeholder="username"]').fill(USER);
  await page.locator('input[placeholder="password"]').fill(PASS);
  await page.locator('button:has-text("Login")').click();
  await page.waitForURL((url) => !url.pathname.includes('/auth'), { timeout: 15000 });
}

test.describe('3D Graph Visual Validation', () => {
  let token: string;
  let vendorId: string;
  let deviceId: string;
  let fileSha256: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);

    vendorId = await createEntity(token, 'PlaywrightVendor', 'Vendor', ['system']);
    deviceId = await createEntity(token, 'PlaywrightDevice', 'Device', ['system']);

    const file = await uploadFile(
      token,
      Buffer.from('playwright test file content'),
      'playwright-test.bin',
      ['system'],
    );
    fileSha256 = file.sha256;

    await createAssociation(token, {
      kind: 'DevelopedBy',
      source: { Entity: { id: deviceId, name: 'PlaywrightDevice' } },
      targets: [{ Entity: { id: vendorId, name: 'PlaywrightVendor' } }],
      groups: ['system'],
      is_bidirectional: false,
    });

    await createAssociation(token, {
      kind: 'FirmwareFor',
      source: { File: fileSha256 },
      targets: [{ Entity: { id: deviceId, name: 'PlaywrightDevice' } }],
      groups: ['system'],
      is_bidirectional: false,
    });
  });

  test.afterAll(async () => {
    await deleteEntity(token, vendorId).catch(() => {});
    await deleteEntity(token, deviceId).catch(() => {});
  });

  test('graph page renders with seeded entity data', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ entities: [deviceId] });
    const input = page.locator('input.form-control').first();
    await input.fill(seedJson);

    await snapshot(page, SCREENSHOT_DIR, 'graph-before-canvas');

    await page.waitForSelector('canvas', { timeout: 30000 });
    await page.waitForTimeout(5000);

    await snapshot(page, SCREENSHOT_DIR, 'graph-full-page');

    const graph3dCard = page.locator('.card').filter({ hasText: 'Association Graph 3D' }).first();
    if (await graph3dCard.isVisible()) {
      await graph3dCard.screenshot({
        path: path.join(SCREENSHOT_DIR, 'graph-3d-section.png'),
      });
    }

    const canvases = page.locator('canvas');
    await expect(canvases.first()).toBeVisible();
  });

  test('graph renders with file seed data', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ samples: [fileSha256] });
    const input = page.locator('input.form-control').first();
    await input.fill(seedJson);

    await page.waitForSelector('canvas', { timeout: 30000 });
    await page.waitForTimeout(5000);

    await snapshot(page, SCREENSHOT_DIR, 'graph-file-seed');

    const canvases = page.locator('canvas');
    await expect(canvases.first()).toBeVisible();
  });

  test('graph controls are visible and functional', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ entities: [deviceId] });
    await page.locator('input.form-control').first().fill(seedJson);
    await page.waitForSelector('canvas', { timeout: 30000 });
    await page.waitForTimeout(3000);

    await expect(page.locator('text=Nodes:')).toBeVisible();

    const gearBtn = page.locator('button[title="Toggle controls"]');
    await gearBtn.click();
    await page.waitForTimeout(300);

    const graphBtn = page.locator('button[title="Graph"]');
    await expect(graphBtn).toBeVisible();
    await graphBtn.click();
    await page.waitForTimeout(300);

    await expect(page.locator('text=Reheat')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Fit All')).toBeVisible();

    const exportBtn = page.locator('button[title="Export"]');
    await expect(exportBtn).toBeVisible();
    await exportBtn.click();
    await page.waitForTimeout(300);

    await expect(page.locator('text=PNG')).toBeVisible();
    await expect(page.locator('text=JPEG')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'graph-with-controls');
  });
});
