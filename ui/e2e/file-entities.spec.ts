import { test, expect } from '@playwright/test';
import { authenticate, createAssociation, createEntity, deleteEntity, loginViaUI, uploadFile, TEST_USER, TEST_PASS } from './helpers';

// Live spec: seeds a file with a directly-associated device, and a vendor one hop past the device, then
// drives the file-details Entities tab (the generic EntityBrowser) — nested expand, lazy grow, the "View"
// metadata axis, filtering, and layer policies. Requires a live API (THORIUM_API_URL), like graph.spec.ts.
test.describe('File Details — Entities tab (EntityBrowser)', () => {
  let token: string;
  let vendorId: string;
  let deviceId: string;
  let fileSha256: string;

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    vendorId = await createEntity(token, 'EntityBrowserVendor', 'Vendor', ['system']);
    deviceId = await createEntity(token, 'EntityBrowserDevice', 'Device', ['system']);

    const file = await uploadFile(token, Buffer.from('entity browser test file'), 'entity-browser-test.bin', ['system']);
    fileSha256 = file.sha256;

    // file --FirmwareFor--> device  (direct association, shown at the top level)
    await createAssociation(token, {
      kind: 'FirmwareFor',
      source: { File: fileSha256 },
      targets: [{ Entity: { id: deviceId, name: 'EntityBrowserDevice' } }],
      groups: ['system'],
      is_bidirectional: false,
    });
    // device --DevelopedBy--> vendor  (one hop past the device; surfaced only after expanding/growing)
    await createAssociation(token, {
      kind: 'DevelopedBy',
      source: { Entity: { id: deviceId, name: 'EntityBrowserDevice' } },
      targets: [{ Entity: { id: vendorId, name: 'EntityBrowserVendor' } }],
      groups: ['system'],
      is_bidirectional: false,
    });
  });

  test.afterAll(async () => {
    await deleteEntity(token, vendorId).catch(() => {});
    await deleteEntity(token, deviceId).catch(() => {});
  });

  const gotoEntitiesTab = async (page: import('@playwright/test').Page) => {
    await loginViaUI(page);
    await page.goto(`/file/${fileSha256}#entities`);
    await page.waitForLoadState('networkidle');
    await expect(page.getByTestId('entity-browser')).toBeVisible({ timeout: 30000 });
  };

  const deviceRow = (page: import('@playwright/test').Page) =>
    page.locator('[data-testid="entity-row"]').filter({ hasText: 'EntityBrowserDevice' });
  const vendorRow = (page: import('@playwright/test').Page) =>
    page.locator('[data-testid="entity-row"]').filter({ hasText: 'EntityBrowserVendor' });

  test('nests the file associations automatically to the default depth', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(page.getByPlaceholder('Filter entities…')).toBeVisible();
    // the device is a direct association; the vendor is one hop past it — both auto-nest at the default depth
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    await expect(vendorRow(page).first()).toBeVisible({ timeout: 30000 });
  });

  test('clicking an auto-expanded row collapses its children', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(vendorRow(page).first()).toBeVisible({ timeout: 30000 });
    // the device auto-expanded; clicking it collapses, hiding the nested vendor
    await deviceRow(page).first().click();
    await expect(vendorRow(page)).toHaveCount(0);
  });

  test('the "details" caret reveals a row\'s metadata on demand', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    // metadata is collapsed by default (condensed) — the device's id appears only after opening its details caret
    const deviceBox = page.getByTestId('entity-infobox').filter({ hasText: 'EntityBrowserDevice' }).first();
    await expect(page.getByText(deviceId, { exact: false })).toHaveCount(0);
    await deviceBox.getByTestId('entity-details-toggle').first().click();
    await expect(deviceBox.getByText(deviceId, { exact: false })).toBeVisible();
  });

  test('omnibar text clause narrows the list', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    const omni = page.getByPlaceholder('Filter entities…');
    await omni.click();
    await omni.fill('no-such-entity-xyz');
    await omni.press('Enter'); // commits a free-text clause
    await expect(page.getByText('No matching items')).toBeVisible();
  });

  test('Flagged Only hides entities with no danger tags / Flag association', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    // the seeded device has no danger tags and no associated Flag entity, so Flagged Only should hide it
    await page.getByTestId('entity-browser-flagged').click();
    await expect(deviceRow(page)).toHaveCount(0);
  });
});
