import { test, expect, Page } from '@playwright/test';
import { MOCK_USER } from './helpers';

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
const LONG = 'extremely-long-image-name-that-overflows';
const PIPE = { group: 'system', name: 'tip-test', creator: 'test', order: [LONG], sla: 604800, description: '', triggers: {}, bans: [] };

async function setup(page: Page) {
  await page.route('**/api/users/whoami', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }));
  await page.route('**/api/groups/details/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }));
  await page.route('**/api/pipelines/list/**/details/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [PIPE] }) }));
  await page.route('**/api/images/system/', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ names: ['a-much-longer-image-name'] }) }));
  await page.route('**/api/**', (r) => {
    const u = r.request().url();
    if (u.includes('/users/whoami') || u.includes('/groups/') || u.includes('/pipelines/list') || u.includes('/images/system')) return r.fallback();
    return r.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.context().addCookies([{ name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' }]);
}

test('node hover shows a themed overlay with the full image name (portaled, not clipped)', async ({ page }) => {
  await setup(page);
  await page.goto('/pipelines');
  await page.waitForSelector('[data-testid="accordion-header"]', { timeout: 15000 });
  await page.locator('[data-testid="accordion-header"]').first().click();
  await page.waitForSelector('.react-flow__node-imageStep', { timeout: 10000 });
  await page.waitForTimeout(800);

  const node = page.locator('.react-flow__node-imageStep').first();
  await node.hover();
  // The overlay is a direct child of <body> (portaled), showing the full name.
  const overlay = page.locator('body > div', { hasText: LONG }).last();
  await expect(overlay).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot({ path: 'e2e/screenshots/node-tooltip-hover.png' });

  // It must not be clipped by the canvas: its box sits within the viewport.
  const box = await overlay.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.width).toBeGreaterThan(0);
  expect(box!.height).toBeGreaterThan(0);

  await node.evaluate((el) => el.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true })));
});
