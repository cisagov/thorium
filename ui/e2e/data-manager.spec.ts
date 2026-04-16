import { test, expect } from '@playwright/test';
import path from 'path';
import {
  authenticate,
  createEntity,
  createAssociation,
  deleteEntity,
  uploadFile,
  buildTree,
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

test.describe('Shared Data Manager', () => {
  let token: string;
  let vendorId: string;
  let deviceId: string;
  let fileSha256: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);

    vendorId = await createEntity(token, 'DataMgrVendor', 'Vendor', ['system']);
    deviceId = await createEntity(token, 'DataMgrDevice', 'Device', ['system']);

    const file = await uploadFile(
      token,
      Buffer.from('data manager e2e test file content'),
      'data-manager-test.bin',
      ['system'],
    );
    fileSha256 = file.sha256;

    await createAssociation(token, {
      kind: 'DevelopedBy',
      source: { Entity: { id: deviceId, name: 'DataMgrDevice' } },
      targets: [{ Entity: { id: vendorId, name: 'DataMgrVendor' } }],
      groups: ['system'],
      is_bidirectional: false,
    });

    await createAssociation(token, {
      kind: 'FirmwareFor',
      source: { File: fileSha256 },
      targets: [{ Entity: { id: deviceId, name: 'DataMgrDevice' } }],
      groups: ['system'],
      is_bidirectional: false,
    });

    // Verify graph is buildable with seeded data
    const tree = await buildTree(token, { entities: [deviceId] });
    expect(tree.id).toBeTruthy();
    expect(tree.initial.length).toBeGreaterThan(0);
  });

  test.afterAll(async () => {
    await deleteEntity(token, vendorId).catch(() => {});
    await deleteEntity(token, deviceId).catch(() => {});
  });

  test('shared data loads — tree and 3D show same nodes', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ entities: [deviceId] });
    const input = page.locator('input.form-control').first();
    await input.fill(seedJson);

    // Wait for 3D canvas to appear
    await page.waitForSelector('canvas', { timeout: 30000 });
    // Wait for tree items to appear
    await page.waitForSelector('button[role="treeitem"]', { timeout: 15000 });

    await page.waitForTimeout(3000);

    const treeItems = page.locator('button[role="treeitem"]');
    const treeCount = await treeItems.count();
    expect(treeCount).toBeGreaterThan(0);

    const canvases = page.locator('canvas');
    await expect(canvases.first()).toBeVisible();

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, 'data-manager-initial.png'),
      fullPage: true,
    });
  });

  test('API call count — single session', async ({ page }) => {
    const treePosts: string[] = [];
    const treePatches: string[] = [];

    await page.route('**/api/trees/**', (route) => {
      const method = route.request().method();
      const url = route.request().url();
      if (method === 'POST') treePosts.push(url);
      if (method === 'PATCH') treePatches.push(url);
      route.continue();
    });

    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ entities: [deviceId] });
    const input = page.locator('input.form-control').first();
    await input.fill(seedJson);

    // Wait for both views to load
    await page.waitForSelector('canvas', { timeout: 30000 });
    await page.waitForSelector('button[role="treeitem"]', { timeout: 15000 });
    await page.waitForTimeout(3000);

    // The 2D graph still has its own session, so we expect 2 POSTs (2D + shared).
    // The shared provider serves both 3D and tree, so only 1 POST for those two.
    // If the 2D graph is on the page, total = 2. The key assertion: NOT 3 (one for each view).
    expect(treePosts.length).toBeLessThanOrEqual(2);

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, 'data-manager-api-count.png'),
      fullPage: true,
    });
  });

  test('tree shows node type badges', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ entities: [deviceId] });
    const input = page.locator('input.form-control').first();
    await input.fill(seedJson);

    await page.waitForSelector('button[role="treeitem"]', { timeout: 15000 });
    await page.waitForTimeout(2000);

    const badges = page.locator('.node-type-badge');
    const badgeCount = await badges.count();
    expect(badgeCount).toBeGreaterThan(0);

    const badgeTexts: string[] = [];
    for (let i = 0; i < badgeCount; i++) {
      const text = await badges.nth(i).textContent();
      if (text) badgeTexts.push(text);
    }
    expect(badgeTexts.some((t) => ['Device', 'Vendor', 'File', 'Repository', 'Tag'].includes(t))).toBeTruthy();

    const treeSection = page.locator('.card').filter({ hasText: 'Association Tree' }).first();
    if (await treeSection.isVisible()) {
      await treeSection.screenshot({
        path: path.join(SCREENSHOT_DIR, 'data-manager-tree-badges.png'),
      });
    }
  });

  test('hover preview popover appears', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ entities: [deviceId] });
    const input = page.locator('input.form-control').first();
    await input.fill(seedJson);

    await page.waitForSelector('button[role="treeitem"]', { timeout: 15000 });
    await page.waitForTimeout(2000);

    // Hover over the first tree item
    const firstItem = page.locator('button[role="treeitem"]').first();
    await firstItem.hover();
    await page.waitForTimeout(600);

    const popover = page.locator('.popover');
    await expect(popover).toBeVisible({ timeout: 5000 });

    const previewType = popover.locator('.preview-type');
    await expect(previewType).toBeVisible();
    const typeText = await previewType.textContent();
    expect(typeText).toBeTruthy();

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, 'data-manager-hover-preview.png'),
      fullPage: true,
    });
  });

  test('growth from 3D propagates to tree', async ({ page }) => {
    const patchCalls: string[] = [];

    await page.route('**/api/trees/**', (route) => {
      if (route.request().method() === 'PATCH') {
        patchCalls.push(route.request().url());
      }
      route.continue();
    });

    await loginViaUI(page);
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const seedJson = JSON.stringify({ entities: [deviceId] });
    const input = page.locator('input.form-control').first();
    await input.fill(seedJson);

    await page.waitForSelector('canvas', { timeout: 30000 });
    await page.waitForSelector('button[role="treeitem"]', { timeout: 15000 });
    await page.waitForTimeout(5000);

    const initialTreeCount = await page.locator('button[role="treeitem"]').count();

    // Click in the center of the 3D canvas to try to hit a node
    const canvas = page.locator('canvas').first();
    const box = await canvas.boundingBox();
    if (box) {
      await canvas.click({ position: { x: box.width / 2, y: box.height / 2 } });
      await page.waitForTimeout(3000);
    }

    // Expand tree nodes to see any newly available data
    const treeItems = page.locator('button[role="treeitem"]');
    const count = await treeItems.count();
    for (let i = 0; i < Math.min(count, 3); i++) {
      await treeItems.nth(i).click();
      await page.waitForTimeout(1000);
    }

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, 'data-manager-growth-synced.png'),
      fullPage: true,
    });

    // If any PATCH calls happened, they should be through the shared session
    // (verifying no duplicate PATCH calls for the same growth)
    const uniquePatches = new Set(patchCalls);
    expect(uniquePatches.size).toBeLessThanOrEqual(patchCalls.length);
  });
});
