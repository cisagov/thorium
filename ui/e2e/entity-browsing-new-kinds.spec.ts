import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { loginViaUI, snapshot } from './helpers';

// Live spec for the config-driven entity BROWSING pages of the kinds that landed with the file-entities
// feature: Flag, Incident, CompiledFunction, DecompiledFunction, PeSection, PeImport. Each browse route is
// generated from an EntityBrowseConfig (title, per-kind column headers, a creatable "+" button, and a
// fetchEntities backed by search). The list rows themselves come from the search index, which lags entity
// creation, so this spec asserts the deterministic page shell — the title, the column headers, and the
// create affordance — rather than a freshly-seeded row (row-level rendering is covered by the details spec
// and the tool-results/file-entities specs). Requires a live API (THORIUM_API_URL), like graph.spec.ts.
const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

// Each new browse route with the strings the shared factory renders: the page title, the create-button
// tooltip label, and a distinctive column header from that kind's `entityHeaders`.
interface BrowseCase {
  route: string;
  title: string;
  createLabel: string;
  header: string;
  screenshot: string;
}

const CASES: BrowseCase[] = [
  { route: '/flags', title: 'Flags', createLabel: 'Create a new Flag.', header: 'Confidence', screenshot: 'browse-flags' },
  { route: '/incidents', title: 'Incidents', createLabel: 'Create a new Incident.', header: 'Cover Term', screenshot: 'browse-incidents' },
  {
    route: '/functions/compiled',
    title: 'Compiled Functions',
    createLabel: 'Create a new Compiled Function.',
    header: 'Address · Instructions',
    screenshot: 'browse-compiled-functions',
  },
  {
    route: '/functions/decompiled',
    title: 'Decompiled Functions',
    createLabel: 'Create a new Decompiled Function.',
    header: 'Address · Tools',
    screenshot: 'browse-decompiled-functions',
  },
  {
    route: '/pe/sections',
    title: 'PE Sections',
    createLabel: 'Create a new PE Section.',
    header: 'Raw · Virtual · Entropy',
    screenshot: 'browse-pe-sections',
  },
  { route: '/pe/imports', title: 'PE Imports', createLabel: 'Create a new PE Import.', header: 'Library', screenshot: 'browse-pe-imports' },
];

test.describe('Entity Browsing — new kinds', () => {
  // open a browse route and wait for the page's title (config.title) to render — the deterministic shell
  // marker that is present regardless of whether the search-backed list has any results yet
  const openBrowse = async (page: Page, route: string, title: string) => {
    await loginViaUI(page);
    await page.goto(route);
    await page.waitForLoadState('networkidle');
    await expect(page.getByText(title, { exact: true }).first()).toBeVisible({ timeout: 30000 });
  };

  for (const c of CASES) {
    test(`${c.title} browse page renders its title, create button, and column headers`, async ({ page }) => {
      await openBrowse(page, c.route, c.title);
      // the creatable "+" button carries a per-kind tooltip; hovering surfaces its label
      const createBtn = page.getByRole('button', { name: '+' }).first();
      await expect(createBtn).toBeVisible();
      await createBtn.hover();
      await expect(page.getByText(c.createLabel)).toBeVisible({ timeout: 10000 });
      // once the initial fetch resolves the list shows either rows or a no-results banner; either way the
      // column headers (from entityHeaders) are rendered
      await expect(page.getByText(c.header, { exact: true }).first()).toBeVisible({ timeout: 30000 });
      await snapshot(page, SCREENSHOT_DIR, c.screenshot);
    });
  }

  test('the create "+" button navigates to the kind\'s create page', async ({ page }) => {
    // representative kind: Flags. The "+" routes to /create/flag via getCreatePathByEntity, confirming the
    // browse → create wiring for the new kinds.
    await openBrowse(page, '/flags', 'Flags');
    await page.getByRole('button', { name: '+' }).first().click();
    await page.waitForURL((url) => url.pathname.startsWith('/create/flag'), { timeout: 15000 });
    // the create page mounts the shared entity-create factory (its own Create button tooltip appears)
    await expect(page).toHaveURL(/\/create\/flag/);
  });
});
