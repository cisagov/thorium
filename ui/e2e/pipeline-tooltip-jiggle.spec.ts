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

async function setup(page: Page) {
  await page.route('**/api/users/whoami', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }));
  await page.route('**/api/groups/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }));
  await page.route('**/api/images/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ names: ['img-a', 'img-b'] }) }));
  await page.route('**/api/**', (r) => {
    const u = r.request().url();
    if (u.includes('/users/whoami') || u.includes('/groups/') || u.includes('/images/')) return r.fallback();
    return r.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.context().addCookies([{ name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' }]);
}

async function metrics(page: Page) {
  return page.evaluate(() => {
    const el = document.scrollingElement || document.documentElement;
    return {
      scrollW: el.scrollWidth,
      clientW: el.clientWidth,
      scrollH: el.scrollHeight,
      clientH: el.clientHeight,
      overflowX: el.scrollWidth - el.clientWidth,
    };
  });
}

// At a viewport narrow enough that a right-placed tooltip on the full-width content would reach past
// the edge, hovering must NOT grow the document's scroll width (which would toggle a scrollbar = jiggle).
test('create-page tooltips do not grow the page scroll width on hover (no scrollbar jiggle)', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await setup(page);
  await page.goto('/create/pipeline');
  await expect(page.getByText('Create A Pipeline')).toBeVisible({ timeout: 30000 });
  await page.waitForTimeout(300);

  const base = await metrics(page);

  await page.getByText('No event triggers configured.').hover();
  await expect(page.locator('[role="tooltip"]', { hasText: 'Automatic triggers' })).toBeVisible();
  const onTriggers = await metrics(page);
  expect(onTriggers.scrollW).toBeLessThanOrEqual(base.scrollW);
  // The tooltip itself stays within the viewport (flips in instead of overflowing the right edge).
  const tipBox = await page.locator('[role="tooltip"]', { hasText: 'Automatic triggers' }).boundingBox();
  expect(tipBox).not.toBeNull();
  expect(tipBox!.x + tipBox!.width).toBeLessThanOrEqual((page.viewportSize()?.width ?? 1280) + 1);

  await page.mouse.move(5, 5);
  await page.waitForTimeout(200);
  await page.locator('textarea[placeholder="describe this pipeline"]').hover();
  await expect(page.locator('[role="tooltip"]').first()).toBeVisible();
  const onDesc = await metrics(page);
  expect(onDesc.scrollW).toBeLessThanOrEqual(base.scrollW);
});
