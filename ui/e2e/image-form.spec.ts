import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { snapshot, MOCK_USER } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const MOCK_GROUP = {
  name: 'system',
  owners: { combined: ['test'], direct: ['test'], metagroups: [] },
  managers: { combined: [], direct: [], metagroups: [] },
  analysts: [],
  users: { combined: [], direct: [], metagroups: [] },
  monitors: { combined: [], direct: [], metagroups: [] },
  description: 'System group',
  allowed: {
    files: true,
    repos: true,
    tags: true,
    images: true,
    pipelines: true,
    reactions: true,
    results: true,
    comments: true,
    entities: true,
  },
};

const MOCK_IMAGE = {
  group: 'system',
  name: 'e2e-form-test',
  creator: 'test',
  version: { Custom: '1.0' },
  scaler: 'K8s',
  image: 'registry.example.com/tool:latest',
  timeout: 600,
  description: 'Image for form layout testing',
  display_type: 'JSON',
  args: { entrypoint: ['/bin/sh'], command: ['-c', 'echo hello'], reaction: '--reaction', output: 'Append' },
  resources: { cpu: 1000, memory: 512, ephemeral_storage: 1024 },
  env: { DEBUG: '1', LOG_LEVEL: 'info' },
  dependencies: {},
  volumes: [{ name: 'data', mount_path: '/data', size: '1Gi' }],
  output_collection: { group: 'system' },
  security_context: { user: 1000, group: 1000, allow_privilege_escalation: false },
  child_filters: { mime: ['application/pdf'], file_name: [], file_extension: ['exe'], submit_non_matches: false },
  network_policies: ['allow-dns'],
  modifiers: '/opt/modifiers',
  bans: [],
  collect_logs: true,
  generator: false,
  used_by: [],
  kvm: undefined,
};

async function setupAuth(page: Page) {
  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }),
  );
  await page.context().addCookies([
    { name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' },
  ]);
}

async function setupImageMocks(page: Page) {
  await page.route('**/api/groups/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }),
  );
  await page.route('**/api/images/*/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_IMAGE] }) }),
  );
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (url.includes('/users/whoami')) return route.fallback();
    if (url.includes('/groups/')) return route.fallback();
    if (url.includes('/images/') && url.includes('/details/')) return route.fallback();
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
}

async function enterEditFormMode(page: Page) {
  await page.goto('/images');
  await page.waitForSelector('.accordion', { timeout: 10000 });

  // click Edit in the accordion header
  await page.locator('[data-testid="header-btn-edit"]').first().click();
  await page.waitForTimeout(1000);

  // verify Form toggle is visible (exact match to avoid matching Bootstrap Form)
  await expect(page.getByRole('button', { name: 'Form', exact: true })).toBeVisible({ timeout: 5000 });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
test.describe('Image Form Layout', () => {
  test.beforeEach(async ({ page }) => {
    await setupAuth(page);
    await setupImageMocks(page);
  });

  test('edit mode defaults to Form view with toggle visible', async ({ page }) => {
    await enterEditFormMode(page);

    await expect(page.getByRole('button', { name: 'Form', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Editor', exact: true })).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'image-form-edit-toggle');
  });

  test('form sections have consistent right edges', async ({ page }) => {
    await enterEditFormMode(page);

    // collect right edges of section header labels (bold text in EditMiddle)
    // these are the labels like "Image Fields", "Resources", etc.
    const sectionLabels = ['Image Fields', 'Resources', 'Arguments', 'Output Collection',
      'Dependencies', 'Environment', 'Child Filters', 'Modifiers', 'Security Context',
      'Network Policies', 'Volumes'];

    const rightEdges: number[] = [];
    for (const label of sectionLabels) {
      const el = page.locator(`b:has-text("${label}")`).first();
      const isVisible = await el.isVisible().catch(() => false);
      if (!isVisible) continue;

      // get the closest parent flex row — the SectionRow
      // measure its full bounding box (right edge = x + width)
      const row = el.locator('xpath=ancestor::div[1]/ancestor::div[1]');
      const box = await row.boundingBox().catch(() => null);
      if (box) {
        rightEdges.push(Math.round(box.x + box.width));
      }
    }

    expect(rightEdges.length).toBeGreaterThanOrEqual(5);

    // all sections should terminate at the same right edge (within 5px)
    const maxRight = Math.max(...rightEdges);
    const minRight = Math.min(...rightEdges);
    expect(maxRight - minRight).toBeLessThanOrEqual(5);

    await snapshot(page, SCREENSHOT_DIR, 'image-form-right-edges');
  });

  test('form view full page screenshot', async ({ page }) => {
    await enterEditFormMode(page);
    await snapshot(page, SCREENSHOT_DIR, 'image-form-full');
  });

  test('form toggle has equal spacing above and below', async ({ page }) => {
    await enterEditFormMode(page);

    const toggleBtn = page.getByRole('button', { name: 'Form', exact: true });
    // the toggle wrapper is the parent of the parent (button > Wrapper > CenterRow)
    const centerRow = toggleBtn.locator('xpath=ancestor::div[2]');
    const margins = await centerRow.evaluate((el) => {
      const cs = getComputedStyle(el);
      return { marginTop: cs.marginTop, marginBottom: cs.marginBottom };
    });

    expect(margins.marginTop).toBe(margins.marginBottom);
    expect(margins.marginTop).not.toBe('0px');
  });

  test('accordion header buttons use consistent gap', async ({ page }) => {
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    // navigate up from a header button to find the HeaderActions container
    const copyBtn = page.locator('[data-testid="header-btn-copy"]').first();
    // HeaderActions (div) > OverlayTipBottom (span) > HeaderBtn (span)
    const headerActions = copyBtn.locator('xpath=ancestor::div[1]');
    const gap = await headerActions.evaluate((el) => getComputedStyle(el).gap);
    expect(gap).toBe('4px');
  });
});
