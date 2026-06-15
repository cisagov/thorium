import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { authenticate, createEntity, deleteEntity, loginViaUI, snapshot, TEST_USER, TEST_PASS } from './helpers';

// Live spec for the incident dashboard page (/dashboard/incident). The page has two states driven by the
// ?incident=<uuid> URL param: with no param it renders an incident picker; with a param it seeds the shared
// DashboardContent with the incident entity. Requires a live API (THORIUM_API_URL), like dashboard.spec.ts.
const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

// unique suffix so repeated runs / parallel workers don't collide on entity names
const RUN = Date.now();
const INCIDENT_NAME = `DashboardIncident-${RUN}`;

test.describe('Incident Dashboard (/dashboard/incident)', () => {
  let token: string;
  let incidentId: string;

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    incidentId = await createEntity(token, INCIDENT_NAME, 'Incident', ['system']);
  });

  test.afterAll(async () => {
    await deleteEntity(token, incidentId).catch(() => {});
  });

  // Deep-link straight to the seeded dashboard and wait for the browser body to render.
  const openSeeded = async (page: Page) => {
    await loginViaUI(page);
    await page.goto(`/dashboard/incident?incident=${incidentId}`);
    await page.waitForLoadState('networkidle');
    // the entity-browser body only mounts once the shared graph's initial fetch resolves
    await expect(page.getByTestId('entity-browser')).toBeVisible({ timeout: 30000 });
  };

  test('no ?incident param shows the picker', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/dashboard/incident');
    await page.waitForLoadState('networkidle');
    // the picker heading + intro are shown; no seeded dashboard tiles yet
    await expect(page.getByText('Incident Dashboard')).toBeVisible();
    await expect(page.getByText('Select an incident to view its dashboard.')).toBeVisible();
    await expect(page.getByTestId('entity-browser')).toHaveCount(0);
    await snapshot(page, SCREENSHOT_DIR, 'incident-dashboard-picker');
  });

  test('selecting an incident sets ?incident and renders the seeded dashboard', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/dashboard/incident');
    await page.waitForLoadState('networkidle');
    // open the react-select menu and pick the incident created for this run by its name
    const select = page.locator('.react-select__control, [class*="-control"]').first();
    await select.click();
    await page.getByText(INCIDENT_NAME, { exact: false }).first().click();
    // the URL now carries the selected incident id and the dashboard body mounts
    await expect(page).toHaveURL(new RegExp(`incident=${incidentId}`));
    await expect(page.getByTestId('entity-browser')).toBeVisible({ timeout: 30000 });
    await expect(page.getByText('Stats')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'incident-dashboard-seeded');
  });

  test('direct deep-link with ?incident renders the seeded dashboard', async ({ page }) => {
    await openSeeded(page);
    // the header shows the selected incident and the shared dashboard tiles render
    await expect(page.getByText(new RegExp(`Incident: (${INCIDENT_NAME}|${incidentId})`))).toBeVisible();
    await expect(page.getByText('Stats')).toBeVisible();
    // the omnibar strip is the same one the general dashboard uses
    await expect(page.getByPlaceholder('Filter entities…')).toBeVisible();
  });

  test('"Change incident" returns to the picker', async ({ page }) => {
    await openSeeded(page);
    // clearing the incident param drops back to the picker state
    await page.getByRole('button', { name: 'Change incident' }).click();
    await expect(page).not.toHaveURL(/incident=/);
    await expect(page.getByText('Select an incident to view its dashboard.')).toBeVisible();
    await expect(page.getByTestId('entity-browser')).toHaveCount(0);
  });
});
