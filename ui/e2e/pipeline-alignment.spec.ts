import { test, Page, Locator } from '@playwright/test';
import path from 'path';
import { MOCK_USER } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const MOCK_GROUP = {
  name: 'system',
  owners: { combined: ['test'], direct: ['test'], metagroups: [] },
  managers: { combined: [], direct: [], metagroups: [] },
  analysts: [],
  users: { combined: [], direct: [], metagroups: [] },
  monitors: { combined: [], direct: [], metagroups: [] },
  description: 'System group',
  allowed: { files: true, repos: true, tags: true, images: true, pipelines: true, reactions: true, results: true, comments: true, entities: true },
};

// Each case exercises a different stage shape so we can eyeball horizontal/vertical line alignment.
const CASES: { name: string; order: (string | string[])[] }[] = [
  { name: 'align-single', order: ['solo'] },
  { name: 'align-sequential', order: ['seq-one', 'seq-two', 'seq-three'] },
  { name: 'align-parallel', order: [['par-a', 'par-b', 'par-c']] },
  { name: 'align-mixed', order: ['pre', ['mid-a', 'mid-b'], 'post'] },
  { name: 'align-par2par', order: [['a1', 'a2'], ['b1', 'b2', 'b3']] },
  // Parallel images with very different name lengths — the reported alignment failure case.
  { name: 'align-uneven', order: ['x', ['short', 'a-much-longer-image-name', 'mid-len-name'], 'done'] },
  // A near-max-width (240px) node followed by another stage — checks node/stage overlap.
  { name: 'align-wide', order: ['extremely-long-image-name-here', 'next-stage'] },
];

const ALL_IMAGES = [...new Set(CASES.flatMap((c) => c.order.flatMap((s) => (Array.isArray(s) ? s : [s]))))];

function mockPipeline(name: string, order: (string | string[])[]) {
  return { group: 'system', name, creator: 'test', order, sla: 604800, description: '', triggers: {}, bans: [] };
}

async function setupMocks(page: Page) {
  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }),
  );
  await page.route('**/api/groups/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }),
  );
  await page.route('**/api/pipelines/list/**/details/**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ details: CASES.map((c) => mockPipeline(c.name, c.order)) }),
    }),
  );
  await page.route('**/api/images/system/', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ names: ALL_IMAGES }) }),
  );
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (url.includes('/users/whoami') || url.includes('/groups/') || url.includes('/pipelines/list') || url.includes('/images/system'))
      return route.fallback();
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.context().addCookies([{ name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' }]);
}

function itemByName(page: Page, name: string): Locator {
  return page.locator('[data-testid="accordion-item"]', { has: page.locator('.accordion-item-name .text', { hasText: name }) });
}

test('pipeline order diagram alignment — single / sequential / parallel / mixed', async ({ page }) => {
  await page.setViewportSize({ width: 1400, height: 1000 });
  await setupMocks(page);
  await page.goto('/pipelines');
  await page.waitForSelector('[data-testid="accordion-item"]', { timeout: 15000 });

  for (const c of CASES) {
    const item = itemByName(page, c.name);
    await item.locator('[data-testid="accordion-header"]').click();
    const flow = item.locator('.react-flow');
    await flow.waitFor({ state: 'visible', timeout: 10000 });
    // let fitView + edge layout settle
    await page.waitForTimeout(900);
    await flow.screenshot({ path: path.join(SCREENSHOT_DIR, `${c.name}.png`) });
    // collapse again so stacked diagrams don't overlap the next click target
    await item.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(200);
  }
});
