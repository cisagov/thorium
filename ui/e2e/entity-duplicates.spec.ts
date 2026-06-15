import { test, expect, Page } from '@playwright/test';
import { authenticate, createAssociation, createEntity, deleteEntity, loginViaUI, uploadFile, TEST_USER, TEST_PASS } from './helpers';

// Live spec: seeds a file whose tree contains a single vendor reachable under TWO devices, making the vendor a
// DAG "duplicate" (multi-parent). Drives the file-details Entities tab (EntityBrowser) to verify the duplicate
// grouping affordances: the correlation number on the badge, hover cross-highlight of all occurrences, click to
// pin, and Escape to unpin. Requires a live API (THORIUM_API_URL), like file-entities.spec.ts.
test.describe('File Details — Entities tab (duplicate grouping)', () => {
  let token: string;
  let vendorId: string;
  let deviceAId: string;
  let deviceBId: string;
  let fileSha256: string;

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    vendorId = await createEntity(token, 'DupBrowserVendor', 'Vendor', ['system']);
    deviceAId = await createEntity(token, 'DupBrowserDeviceA', 'Device', ['system']);
    deviceBId = await createEntity(token, 'DupBrowserDeviceB', 'Device', ['system']);

    const file = await uploadFile(token, Buffer.from('duplicate entity browser test file'), 'dup-entity-test.bin', ['system']);
    fileSha256 = file.sha256;

    // file --FirmwareFor--> deviceA and deviceB (two direct associations, both shown at the top level)
    for (const [id, name] of [
      [deviceAId, 'DupBrowserDeviceA'],
      [deviceBId, 'DupBrowserDeviceB'],
    ]) {
      await createAssociation(token, {
        kind: 'FirmwareFor',
        source: { File: fileSha256 },
        targets: [{ Entity: { id, name } }],
        groups: ['system'],
        is_bidirectional: false,
      });
    }
    // both devices --DevelopedBy--> the SAME vendor, so the vendor has two distinct parents in the tree (a
    // multi-parent "duplicate" that renders once under each device)
    for (const [id, name] of [
      [deviceAId, 'DupBrowserDeviceA'],
      [deviceBId, 'DupBrowserDeviceB'],
    ]) {
      await createAssociation(token, {
        kind: 'DevelopedBy',
        source: { Entity: { id, name } },
        targets: [{ Entity: { id: vendorId, name: 'DupBrowserVendor' } }],
        groups: ['system'],
        is_bidirectional: false,
      });
    }
  });

  test.afterAll(async () => {
    await deleteEntity(token, vendorId).catch(() => {});
    await deleteEntity(token, deviceAId).catch(() => {});
    await deleteEntity(token, deviceBId).catch(() => {});
  });

  const gotoEntitiesTab = async (page: Page) => {
    await loginViaUI(page);
    await page.goto(`/file/${fileSha256}#entities`);
    await page.waitForLoadState('networkidle');
    await expect(page.getByTestId('entity-browser')).toBeVisible({ timeout: 30000 });
  };

  const vendorRows = (page: Page) => page.locator('[data-testid="entity-row"]').filter({ hasText: 'DupBrowserVendor' });
  const vendorBadges = (page: Page) => vendorRows(page).getByRole('button', { name: /Duplicate/ });

  test('renders both occurrences with the same correlation number', async ({ page }) => {
    await gotoEntitiesTab(page);
    // the vendor is reachable under both devices, so it auto-nests twice at the default depth
    await expect(vendorRows(page)).toHaveCount(2, { timeout: 30000 });
    await expect(vendorBadges(page)).toHaveCount(2);
    // both badges show "Duplicate ·N" with the SAME N (same node id → same correlation number)
    const first = (await vendorBadges(page).nth(0).innerText()).trim();
    const second = (await vendorBadges(page).nth(1).innerText()).trim();
    expect(first).toMatch(/Duplicate\s*·\s*\d+/);
    expect(second).toBe(first);
  });

  test('hovering a duplicate highlights every occurrence; leaving clears it', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(vendorRows(page)).toHaveCount(2, { timeout: 30000 });
    await vendorBadges(page).nth(0).hover();
    // both occurrences light up together (keyed by node id)
    await expect(vendorRows(page).nth(0)).toHaveClass(/duplicate-highlight/);
    await expect(vendorRows(page).nth(1)).toHaveClass(/duplicate-highlight/);
    // moving away from an UNPINNED duplicate clears the transient highlight
    await page.getByTestId('entity-browser').hover({ position: { x: 1, y: 1 } });
    await expect(vendorRows(page).nth(0)).not.toHaveClass(/duplicate-highlight/);
  });

  test('clicking pins the highlight (survives mouse-leave) and Escape unpins', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(vendorRows(page)).toHaveCount(2, { timeout: 30000 });
    await vendorBadges(page).nth(0).click();
    // pinned: still highlighted after the cursor moves off the badge
    await page.getByTestId('entity-browser').hover({ position: { x: 1, y: 1 } });
    await expect(vendorRows(page).nth(0)).toHaveClass(/duplicate-highlight/);
    await expect(vendorRows(page).nth(1)).toHaveClass(/duplicate-highlight/);
    // Escape unpins and clears the highlight from all occurrences
    await page.keyboard.press('Escape');
    await expect(vendorRows(page).nth(0)).not.toHaveClass(/duplicate-highlight/);
    await expect(vendorRows(page).nth(1)).not.toHaveClass(/duplicate-highlight/);
  });

  test('clicking the badge does not expand or collapse the row', async ({ page }) => {
    await gotoEntitiesTab(page);
    await expect(vendorRows(page)).toHaveCount(2, { timeout: 30000 });
    // clicking the badge pins/jumps but must NOT toggle the row (stopPropagation) — the vendor rows stay put
    await vendorBadges(page).nth(0).click();
    await expect(vendorRows(page)).toHaveCount(2);
  });
});
