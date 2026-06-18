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

test('create page: SLA starts blank with default as placeholder', async ({ page }) => {
  await setup(page);
  await page.goto('/create/pipeline');
  // First navigation can be slow (vite lazily compiles the route); wait for the page to render.
  await expect(page.getByText('Create A Pipeline')).toBeVisible({ timeout: 30000 });
  const sla = page.locator('input[placeholder="604800"]');
  await expect(sla).toBeVisible({ timeout: 15000 });
  await expect(sla).toHaveValue('');
  await page.screenshot({ path: 'e2e/screenshots/create-sla-placeholder.png' });

  // Typing replaces the placeholder with a real value.
  await sla.fill('123');
  await expect(sla).toHaveValue('123');
});

test('create page: description tooltip is centered on the field, not bottom-right', async ({ page }) => {
  await setup(page);
  await page.goto('/create/pipeline');
  const textarea = page.locator('textarea[placeholder="describe this pipeline"]');
  await expect(textarea).toBeVisible();
  await textarea.hover();

  const tooltip = page.locator('[role="tooltip"]').first();
  await expect(tooltip).toBeVisible();

  const fieldBox = await textarea.boundingBox();
  const tipBox = await tooltip.boundingBox();
  expect(fieldBox).not.toBeNull();
  expect(tipBox).not.toBeNull();

  // Vertically centered on the field (not anchored to the bottom corner). It sits beside the field —
  // right when there's room, flipping left on a narrow viewport — but always within the viewport.
  const fieldCenterY = fieldBox!.y + fieldBox!.height / 2;
  const tipCenterY = tipBox!.y + tipBox!.height / 2;
  expect(Math.abs(tipCenterY - fieldCenterY)).toBeLessThan(fieldBox!.height / 3);
  expect(tipBox!.x).toBeGreaterThanOrEqual(0);
  expect(tipBox!.x + tipBox!.width).toBeLessThanOrEqual((page.viewportSize()?.width ?? 1280) + 1);

  await page.screenshot({ path: 'e2e/screenshots/create-description-tooltip.png' });
});

test('create page: triggers tip shows on hovering the content; order has no tip', async ({ page }) => {
  await setup(page);
  await page.goto('/create/pipeline');

  // Hover the triggers editor content (empty panel) → the section tip appears.
  const triggersContent = page.getByText('No event triggers configured.');
  await expect(triggersContent).toBeVisible();
  await triggersContent.hover();
  const tip = page.locator('[role="tooltip"]', { hasText: 'Automatic triggers' });
  await expect(tip).toBeVisible();
  await page.screenshot({ path: 'e2e/screenshots/create-triggers-tooltip.png' });

  // The Order title carries no tooltip (its instructions are a note in the content instead).
  await page.getByRole('heading', { name: 'Order', exact: true }).hover();
  await page.waitForTimeout(400);
  await expect(page.locator('[role="tooltip"]', { hasText: 'order of images' })).toHaveCount(0);
});
