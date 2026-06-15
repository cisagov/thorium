import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { authenticate, createEntity, deleteEntity, loginViaUI, snapshot, uploadFile, TEST_USER, TEST_PASS } from './helpers';

// Live spec for the Dashboard Builder page (/dashboard/build). Seeds a file + a device via the API, then
// drives the builder UI: the resource-type picker (incl. Tag mode), the uploads-first File browse
// sections, per-row Add, the selection-panel chips (remove / re-add / no-free-text), the URL-backed
// selection state (back/refresh restore), Create → /dashboard/view, and the details-page entry button.
// Requires a live API (THORIUM_API_URL), like graph.spec.ts.
const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const RUN = Date.now();
const DEVICE_NAME = `BuilderDevice-${RUN}`;
const FILE_NAME = `builder-test-${RUN}.bin`;

test.describe('Dashboard Builder (/dashboard/build)', () => {
  let token: string;
  let deviceId: string;
  let fileSha256: string;

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    deviceId = await createEntity(token, DEVICE_NAME, 'Device', ['system']);
    const file = await uploadFile(token, Buffer.from(`builder test file ${RUN}`), FILE_NAME, ['system']);
    fileSha256 = file.sha256;
  });

  test.afterAll(async () => {
    await deleteEntity(token, deviceId).catch(() => {});
  });

  const openBuilder = async (page: Page) => {
    await loginViaUI(page);
    await page.goto('/dashboard/build');
    await page.waitForLoadState('networkidle');
    await expect(page.getByLabel('Resource type to browse')).toBeVisible({ timeout: 30000 });
  };

  // the resource-type picker is a native <select> labelled "Resource type to browse"
  const typePicker = (page: Page) => page.getByLabel('Resource type to browse');

  test('navigates Dashboards → Build from the sidebar', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    // the Dashboards nav group holds a "Build" child linking to /dashboard/build
    await page.getByText('Dashboards', { exact: true }).first().click();
    await page.waitForTimeout(500);
    const buildLink = page.getByRole('link', { name: 'Build', exact: true });
    await expect(buildLink).toBeVisible();
    await buildLink.click();
    await page.waitForURL((url) => url.pathname === '/dashboard/build', { timeout: 15000 });
    await expect(typePicker(page)).toBeVisible({ timeout: 30000 });
  });

  test('File type shows "Your uploads" and "All recent" sections', async ({ page }) => {
    await openBuilder(page);
    // File is the default browse mode; the uploads-first list renders both labeled sections
    await expect(page.getByText('Your uploads', { exact: true })).toBeVisible({ timeout: 30000 });
    await expect(page.getByText('All recent', { exact: true })).toBeVisible({ timeout: 30000 });
    await snapshot(page, SCREENSHOT_DIR, 'builder-file-sections');
  });

  test('adding a file then a device yields two selection chips', async ({ page }) => {
    await openBuilder(page);
    // add the seeded file from a browse row (aria-label "Add <label> to dashboard")
    const addFile = page.getByRole('button', { name: /^Add .* to dashboard$/ });
    await expect(addFile.first()).toBeVisible({ timeout: 30000 });
    await addFile.first().click();
    // that row flips to a disabled "Added" state
    await expect(page.getByRole('button', { name: /already added$/ }).first()).toBeVisible();

    // switch to Device browse and add the seeded device
    await typePicker(page).selectOption('Device');
    const deviceAdd = page.getByRole('button', { name: new RegExp(`^Add ${DEVICE_NAME} to dashboard$`) });
    await expect(deviceAdd.first()).toBeVisible({ timeout: 30000 });
    await deviceAdd.first().click();

    // the selection panel now has the device chip (react-select renders chips as multi-value labels)
    await expect(page.getByText('Selected resources')).toBeVisible();
    await expect(page.getByText(DEVICE_NAME, { exact: false }).last()).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'builder-two-selections');
  });

  test('removing a chip and re-adding it from the dropdown restores it', async ({ page }) => {
    await openBuilder(page);
    // add the device so we have a chip to operate on
    await typePicker(page).selectOption('Device');
    const deviceAdd = page.getByRole('button', { name: new RegExp(`^Add ${DEVICE_NAME} to dashboard$`) });
    await expect(deviceAdd.first()).toBeVisible({ timeout: 30000 });
    await deviceAdd.first().click();
    await expect(page.getByText(DEVICE_NAME, { exact: false }).last()).toBeVisible();

    // remove the chip via its react-select "x" (the multi-value remove control)
    const removeChip = page.locator('.react-select__multi-value__remove, [class*="multi-value__remove"]').first();
    await removeChip.click();
    // once removed, the device is no longer in the browse row's "Added" state
    await expect(deviceAdd.first()).toBeVisible();

    // the removed item is the sole re-add option in the selection dropdown; open it and pick it
    const select = page.locator('.react-select__control, [class*="-control"]').last();
    await select.click();
    await page.getByText(DEVICE_NAME, { exact: false }).last().click();
    // re-added: the row goes back to "Added"
    await expect(page.getByRole('button', { name: new RegExp(`${DEVICE_NAME} already added$`) }).first()).toBeVisible();
  });

  test('typing free text in the selection dropdown creates nothing', async ({ page }) => {
    await openBuilder(page);
    // the selection panel select is non-creatable (isCreatable={false}); typing must not add a chip
    const select = page.locator('.react-select__control, [class*="-control"]').last();
    await select.click();
    await page.keyboard.type('totally-made-up-value');
    await page.keyboard.press('Enter');
    // no chip is created from free text; the empty-state message remains
    await expect(page.getByText('Browse and add resources below…')).toBeVisible();
  });

  test('Tag mode adds a key/value tag as a chip', async ({ page }) => {
    await openBuilder(page);
    await typePicker(page).selectOption('Tag');
    // Tag mode swaps the browse list for a TagSelect key/value entry
    await expect(page.getByText('Enter a tag key and value…')).toBeVisible({ timeout: 30000 });
    // focusing the placeholder span creates a new tag and focuses its "key" input; TagSelect uses
    // a two-step key -> value flow, each committed with Enter
    await page.getByText('Enter a tag key and value…').click();
    await page.getByPlaceholder('Enter a key...').fill('family');
    await page.keyboard.press('Enter');
    await page.getByPlaceholder('Enter a value...').fill('emotet');
    await page.keyboard.press('Enter');
    // only a complete key+value pair enables Add; click it to push a Tag selection
    const addTag = page.getByRole('button', { name: 'Add tag to dashboard' });
    await expect(addTag).toBeEnabled({ timeout: 10000 });
    await addTag.click();
    // the tag appears as a "family: emotet" chip in the selection panel
    await expect(page.getByText('family: emotet', { exact: false }).last()).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'builder-tag-mode');
  });

  test('Create navigates to /dashboard/view with the expected params and renders', async ({ page }) => {
    await openBuilder(page);
    // add the device as the seed
    await typePicker(page).selectOption('Device');
    const deviceAdd = page.getByRole('button', { name: new RegExp(`^Add ${DEVICE_NAME} to dashboard$`) });
    await expect(deviceAdd.first()).toBeVisible({ timeout: 30000 });
    await deviceAdd.first().click();
    await expect(page.getByText(DEVICE_NAME, { exact: false }).last()).toBeVisible();

    // Create opens the dashboard for the selected resources
    await page.getByRole('button', { name: 'Create dashboard' }).click();
    await page.waitForURL((url) => url.pathname === '/dashboard/view', { timeout: 15000 });
    // the entity uuid is encoded as ?entity=<uuid> and the depth default is carried along
    expect(page.url()).toContain(`entity=${deviceId}`);
    // the dashboard actually renders for the seeded device
    await expect(page.getByTestId('entity-browser')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="entity-row"]').filter({ hasText: DEVICE_NAME }).first()).toBeVisible({ timeout: 30000 });
  });

  test('back / refresh retains builder selections (URL-backed)', async ({ page }) => {
    await openBuilder(page);
    await typePicker(page).selectOption('Device');
    const deviceAdd = page.getByRole('button', { name: new RegExp(`^Add ${DEVICE_NAME} to dashboard$`) });
    await expect(deviceAdd.first()).toBeVisible({ timeout: 30000 });
    await deviceAdd.first().click();
    // the selection is mirrored into the builder's own URL (?entity=<uuid>)
    await page.waitForFunction((id) => window.location.search.includes(`entity=${id}`), deviceId, { timeout: 10000 });

    // a full reload re-hydrates the selection from the URL
    await page.reload();
    await page.waitForLoadState('networkidle');
    await expect(page.getByText('Selected resources')).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(DEVICE_NAME, { exact: false }).last()).toBeVisible({ timeout: 30000 });
  });

  test('file details "Build Dashboard" pre-populates the builder', async ({ page }) => {
    await loginViaUI(page);
    await page.goto(`/file/${fileSha256}`);
    await page.waitForLoadState('networkidle');
    // the details header carries a shared BuildDashboardButton for the file
    const buildBtn = page.getByRole('button', { name: 'Build a dashboard from this file' });
    await expect(buildBtn).toBeVisible({ timeout: 30000 });
    await buildBtn.click();
    // it navigates to the builder with the file pre-seeded (?sample=<sha256>)
    await page.waitForURL((url) => url.pathname === '/dashboard/build', { timeout: 15000 });
    expect(page.url()).toContain(`sample=${fileSha256}`);
    // the builder hydrates the pre-seeded file as a selection chip
    await expect(page.getByText('Selected resources')).toBeVisible({ timeout: 30000 });
    await snapshot(page, SCREENSHOT_DIR, 'builder-prepopulated-from-file');
  });
});
