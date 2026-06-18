import { test, expect, Page } from '@playwright/test';
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
// A tall parallel stage so the canvas is at its capped height (resizing is meaningful).
const PIPE = { group: 'system', name: 'resize-test', creator: 'test', order: [['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']], sla: 604800, description: '', triggers: {}, bans: [] };

async function setup(page: Page) {
  await page.route('**/api/users/whoami', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }));
  await page.route('**/api/groups/details/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }));
  await page.route('**/api/pipelines/list/**/details/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [PIPE] }) }));
  await page.route('**/api/images/system/', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ names: PIPE.order[0] }) }));
  await page.route('**/api/**', (r) => {
    const u = r.request().url();
    if (u.includes('/users/whoami') || u.includes('/groups/') || u.includes('/pipelines/list') || u.includes('/images/system')) return r.fallback();
    return r.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.context().addCookies([{ name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' }]);
}

async function flowHeight(page: Page): Promise<number> {
  return page.locator('.react-flow').first().evaluate((el) => (el.parentElement as HTMLElement).getBoundingClientRect().height);
}

async function openDiagram(page: Page) {
  await setup(page);
  await page.goto('/pipelines');
  await page.waitForSelector('[data-testid="accordion-header"]', { timeout: 15000 });
  await page.locator('[data-testid="accordion-header"]').first().click();
  await page.waitForSelector('.react-flow', { timeout: 10000 });
  await page.waitForTimeout(600);
}

async function dragHandle(page: Page, dy: number) {
  const hb = await page.locator('div[title="Drag to resize the diagram height"]').boundingBox();
  expect(hb).not.toBeNull();
  await page.mouse.move(hb!.x + hb!.width / 2, hb!.y + hb!.height / 2);
  await page.mouse.down();
  await page.mouse.move(hb!.x + hb!.width / 2, hb!.y + hb!.height / 2 + dy, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(300);
}

test('resize handle expands the canvas height', async ({ page }) => {
  await openDiagram(page);
  const startH = await flowHeight(page);
  await dragHandle(page, 160); // drag down → taller
  const expandedH = await flowHeight(page);
  expect(expandedH).toBeGreaterThan(startH + 100);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'resize-expanded.png') });
});

test('resize handle contracts the canvas but clamps at the minimum height', async ({ page }) => {
  await openDiagram(page);
  const startH = await flowHeight(page);
  await dragHandle(page, -400); // drag up well past the minimum
  const contractedH = await flowHeight(page);
  expect(contractedH).toBeLessThan(startH);
  expect(contractedH).toBeGreaterThanOrEqual(135); // MIN_CANVAS_HEIGHT (140) minus border tolerance
  expect(contractedH).toBeLessThan(190);
});
