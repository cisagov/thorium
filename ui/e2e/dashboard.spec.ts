import { test, expect, Page } from '@playwright/test';
import path from 'path';
import {
  authenticate,
  createAssociation,
  createEntity,
  deleteEntity,
  loginViaUI,
  snapshot,
  uploadFile,
  TEST_USER,
  TEST_PASS,
} from './helpers';

// Live spec for the custom dashboard page (/dashboard/view). Seeds a file + a device (+ association),
// opens the dashboard deep-linked to those seeds, and exercises the stats panel, omnibar, the hide/undo
// path in the dashboard's own strip, the empty/malformed states, deep-link filter restore, and the
// responsive side-by-side vs tabbed layouts. Requires a live API (THORIUM_API_URL), like graph.spec.ts.
const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

// unique suffix so repeated runs / parallel workers don't collide on entity names
const RUN = Date.now();
const DEVICE_NAME = `DashboardDevice-${RUN}`;
const VENDOR_NAME = `DashboardVendor-${RUN}`;

test.describe('Custom Dashboard (/dashboard/view)', () => {
  let token: string;
  let deviceId: string;
  let vendorId: string;
  let fileSha256: string;

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    deviceId = await createEntity(token, DEVICE_NAME, 'Device', ['system']);
    vendorId = await createEntity(token, VENDOR_NAME, 'Vendor', ['system']);

    const file = await uploadFile(token, Buffer.from(`dashboard test file ${RUN}`), `dashboard-test-${RUN}.bin`, ['system']);
    fileSha256 = file.sha256;

    // file --FirmwareFor--> device  (direct association surfaced at the top level)
    await createAssociation(token, {
      kind: 'FirmwareFor',
      source: { File: fileSha256 },
      targets: [{ Entity: { id: deviceId, name: DEVICE_NAME } }],
      groups: ['system'],
      is_bidirectional: false,
    });
    // device --DevelopedBy--> vendor  (one hop past the device; nests under it)
    await createAssociation(token, {
      kind: 'DevelopedBy',
      source: { Entity: { id: deviceId, name: DEVICE_NAME } },
      targets: [{ Entity: { id: vendorId, name: VENDOR_NAME } }],
      groups: ['system'],
      is_bidirectional: false,
    });
  });

  test.afterAll(async () => {
    await deleteEntity(token, deviceId).catch(() => {});
    await deleteEntity(token, vendorId).catch(() => {});
  });

  // Deep-link to the dashboard seeded with the file + device and wait for the browser body to render.
  const openDashboard = async (page: Page, extraParams = '') => {
    await loginViaUI(page);
    const params = `sample=${fileSha256}&entity=${deviceId}${extraParams ? `&${extraParams}` : ''}`;
    await page.goto(`/dashboard/view?${params}`);
    await page.waitForLoadState('networkidle');
    // the entity-browser body only mounts once the shared graph's initial fetch resolves
    await expect(page.getByTestId('entity-browser')).toBeVisible({ timeout: 30000 });
  };

  const deviceRow = (page: Page) => page.locator('[data-testid="entity-row"]').filter({ hasText: DEVICE_NAME });
  const vendorRow = (page: Page) => page.locator('[data-testid="entity-row"]').filter({ hasText: VENDOR_NAME });

  test('renders the stats panel with clickable bars for the seeded resources', async ({ page }) => {
    await openDashboard(page);
    // the stats tile is always shown; the Entities series has a per-kind bar
    await expect(page.getByText('Stats')).toBeVisible();
    await expect(page.locator('[role="group"][aria-label="Entities breakdown"]')).toBeVisible({ timeout: 30000 });
    await snapshot(page, SCREENSHOT_DIR, 'dashboard-stats');
    // a kind bar renders as an accessible <button> (e.g. "Device: 1")
    const deviceBar = page.getByRole('button', { name: /^Device: \d+$/ });
    await expect(deviceBar.first()).toBeVisible();
  });

  test('clicking a kind bar narrows the browser to that kind', async ({ page }) => {
    await openDashboard(page);
    // both the device (Device kind) and vendor (Vendor kind) are present before filtering
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    await expect(vendorRow(page).first()).toBeVisible({ timeout: 30000 });
    // clicking the Device kind bar injects an Include(Device) clause, whitelisting Device and dropping Vendor
    const deviceBar = page.getByRole('button', { name: /^Device: \d+$/ }).first();
    await deviceBar.click();
    await expect(deviceRow(page).first()).toBeVisible();
    await expect(vendorRow(page)).toHaveCount(0);
  });

  test('omnibar text clause narrows the list', async ({ page }) => {
    await openDashboard(page);
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    // the dashboard omnibar strip uses the same "Filter entities…" placeholder as the browser toolbar
    const omni = page.getByPlaceholder('Filter entities…');
    await omni.click();
    await omni.fill('no-such-entity-xyz');
    await omni.press('Enter');
    await expect(page.getByText('No matching items')).toBeVisible();
  });

  test('hiding a row hides it + its subtree; "Hidden (n)" appears in the strip and clears', async ({ page }) => {
    await openDashboard(page);
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    await expect(vendorRow(page).first()).toBeVisible({ timeout: 30000 });
    // hovering the device row reveals its exclude control (eye-slash), labelled per the row title
    const device = deviceRow(page).first();
    await device.hover();
    const hideBtn = page.getByRole('button', { name: new RegExp(`Exclude ${DEVICE_NAME} and everything under it`) });
    await hideBtn.first().click();
    // the device and its nested vendor both vanish from the entities view
    await expect(deviceRow(page)).toHaveCount(0);
    await expect(vendorRow(page)).toHaveCount(0);
    // the undo path lives in the dashboard's own strip: "Hidden (1)" chip + clear-all
    const hiddenChip = page.getByTestId('entity-browser-hidden');
    await expect(hiddenChip).toBeVisible();
    await expect(hiddenChip).toContainText('Hidden (1)');
    await hiddenChip.click();
    await page.getByTestId('entity-browser-hidden-clear').click();
    // clearing restores the hidden node (and its subtree)
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
  });

  test('re-rooting a row (gear) reorders the view under it, adds ?root=, and clears from the omnibar', async ({ page }) => {
    await openDashboard(page);
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
    await expect(vendorRow(page).first()).toBeVisible({ timeout: 30000 });
    // hovering the device row reveals its gear (re-root) control, labelled per the row title
    const device = deviceRow(page).first();
    await device.hover();
    await page
      .getByRole('button', { name: new RegExp(`Re-root the view at ${DEVICE_NAME}`) })
      .first()
      .click();
    // the re-root is URL-backed (shareable) via ?root=<id>
    await expect(page).toHaveURL(/[?&]root=/);
    // re-root reorders rather than prunes: the associated vendor is still visible (re-nested, not hidden)
    await expect(vendorRow(page).first()).toBeVisible();
    // the omnibar surfaces a "Rooted: …" tile whose × clears the re-root back to the default view
    await expect(page.getByTestId('entity-browser-reroot')).toBeVisible();
    await page.getByTestId('entity-browser-reroot-clear').click();
    await expect(page).not.toHaveURL(/[?&]root=/);
    await expect(deviceRow(page).first()).toBeVisible({ timeout: 30000 });
  });

  test('deep-link with an omnibar clause param restores the filter', async ({ page }) => {
    // useOmnibarUrlState serializes a free-text clause as ?query=<text>; a deep link should restore it
    await openDashboard(page, 'query=no-such-entity-xyz');
    // the restored text clause immediately narrows the list to nothing
    await expect(page.getByText('No matching items')).toBeVisible({ timeout: 30000 });
    // the omnibar shows the restored clause text
    await expect(page.getByText('no-such-entity-xyz').first()).toBeVisible();
  });

  test('empty dashboard (no seed params) shows a warning banner + builder link', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/dashboard/view');
    await page.waitForLoadState('networkidle');
    // no resources => warning AlertBanner with a link to the builder
    await expect(page.getByText('This dashboard has no resources.')).toBeVisible();
    const builderLink = page.getByRole('link', { name: /Build a dashboard/i });
    await expect(builderLink).toBeVisible();
    await expect(builderLink).toHaveAttribute('href', '/dashboard/build');
    await snapshot(page, SCREENSHOT_DIR, 'dashboard-empty');
  });

  test('malformed params render without crashing', async ({ page }) => {
    await loginViaUI(page);
    // a junk sha256 is still a resource param, so the page composes the dashboard (and the codec clamps
    // junk depth to the default) rather than erroring
    await page.goto('/dashboard/view?sample=not-a-real-sha256&depth=notanumber');
    await page.waitForLoadState('networkidle');
    // the page still renders its shell (stats tile header) and does not throw / white-screen
    await expect(page.getByText('Stats')).toBeVisible({ timeout: 30000 });
    await snapshot(page, SCREENSHOT_DIR, 'dashboard-malformed');
  });

  test('ultra-wide viewport shows the browser and graph tiles side by side', async ({ page }) => {
    await page.setViewportSize({ width: 2100, height: 1200 });
    await openDashboard(page);
    // both tiles are mounted at once above the ultra-wide breakpoint (no tab bar)
    await expect(page.getByText('Association Graph')).toBeVisible({ timeout: 30000 });
    await expect(page.getByTestId('entity-browser')).toBeVisible();
    // there is no Dashboard-content tablist in the side-by-side layout
    await expect(page.getByRole('tablist', { name: 'Dashboard content' })).toHaveCount(0);
    await snapshot(page, SCREENSHOT_DIR, 'dashboard-side-by-side');
  });

  test('narrow viewport uses tabs; switching to Graph mounts the canvas', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await openDashboard(page);
    // below the ultra-wide breakpoint the content collapses into Entities / Graph tabs
    const graphTab = page.getByRole('tab', { name: 'Graph' });
    await expect(graphTab).toBeVisible({ timeout: 30000 });
    await graphTab.click();
    // the lazy 3D graph mounts its WebGL canvas once its tab is active
    await page.waitForSelector('canvas', { timeout: 30000 });
    await expect(page.locator('canvas').first()).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'dashboard-tabs-graph');
  });

  test('keyboard: a stats bar can be focused and activated', async ({ page }) => {
    await openDashboard(page);
    await expect(vendorRow(page).first()).toBeVisible({ timeout: 30000 });
    // focus the Device kind bar directly (it is a real <button>) and activate it via the keyboard
    const deviceBar = page.getByRole('button', { name: /^Device: \d+$/ }).first();
    await deviceBar.focus();
    await expect(deviceBar).toBeFocused();
    await deviceBar.press('Enter');
    // keyboard activation applies the same Include(Device) filter as a click
    await expect(deviceRow(page).first()).toBeVisible();
    await expect(vendorRow(page)).toHaveCount(0);
  });
});
