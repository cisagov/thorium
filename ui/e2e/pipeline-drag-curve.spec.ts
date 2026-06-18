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
const PIPE = { group: 'system', name: 'drag-test', creator: 'test', order: ['start-img', ['par-a', 'par-b'], 'end-img'], sla: 604800, description: '', triggers: {}, bans: [] };

async function setup(page: Page) {
  await page.route('**/api/users/whoami', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }));
  await page.route('**/api/groups/details/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }));
  await page.route('**/api/pipelines/list/**/details/**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [PIPE] }) }));
  await page.route('**/api/images/system/', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ names: ['start-img', 'par-a', 'par-b', 'end-img'] }) }));
  await page.route('**/api/**', (r) => {
    const u = r.request().url();
    if (u.includes('/users/whoami') || u.includes('/groups/') || u.includes('/pipelines/list') || u.includes('/images/system')) return r.fallback();
    return r.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.context().addCookies([{ name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' }]);
}

async function open(page: Page) {
  await setup(page);
  await page.goto('/pipelines');
  await page.waitForSelector('[data-testid="accordion-header"]', { timeout: 15000 });
  await page.locator('[data-testid="accordion-header"]').first().click();
  await page.waitForSelector('.react-flow__node-imageStep', { timeout: 10000 });
  await page.waitForTimeout(600);
}

function nodeTransformY(page: Page, id: string) {
  return page.locator(`.react-flow__node[data-id="${id}"]`).evaluate((el) => {
    const m = new DOMMatrixReadOnly(getComputedStyle(el).transform);
    return m.f; // translateY
  });
}

// Parse an SVG path's straight `L` segments (tracking the current point through M/L/Q). Q commands are
// the rounded corners between axis-aligned runs and are intentionally ignored here.
function lineSegments(d: string): Array<{ dx: number; dy: number }> {
  const tokens = d.trim().split(/[\s,]+/);
  const segs: Array<{ dx: number; dy: number }> = [];
  let cx = 0;
  let cy = 0;
  let i = 0;
  while (i < tokens.length) {
    const cmd = tokens[i++];
    if (cmd === 'M') {
      cx = parseFloat(tokens[i++]);
      cy = parseFloat(tokens[i++]);
    } else if (cmd === 'L') {
      const x = parseFloat(tokens[i++]);
      const y = parseFloat(tokens[i++]);
      segs.push({ dx: x - cx, dy: y - cy });
      cx = x;
      cy = y;
    } else if (cmd === 'Q') {
      i += 2; // control point
      cx = parseFloat(tokens[i++]);
      cy = parseFloat(tokens[i++]);
    } else {
      i++;
    }
  }
  return segs;
}

async function assertAllEdgesOrthogonal(page: Page) {
  const paths = await page.locator('.react-flow__edge-path').evaluateAll((els) => els.map((e) => e.getAttribute('d') || ''));
  expect(paths.length).toBeGreaterThan(0);
  for (const d of paths) {
    for (const seg of lineSegments(d)) {
      // Every straight run must be axis-aligned (purely horizontal or vertical) — no diagonals.
      expect(Math.min(Math.abs(seg.dx), Math.abs(seg.dy))).toBeLessThan(1.5);
    }
  }
}

const DRAGS: { name: string; dx: number; dy: number }[] = [
  { name: 'down', dx: 40, dy: 150 },
  { name: 'up', dx: 40, dy: -150 },
  { name: 'left-backward', dx: -260, dy: 50 }, // backward, below center → detour lane goes up
  { name: 'left-backward-up', dx: -260, dy: -150 }, // backward, above center → detour lane flips down
];

for (const drag of DRAGS) {
  test(`free drag (${drag.name}) keeps every edge orthogonal (right angles, no curves)`, async ({ page }) => {
    await open(page);
    const node = page.locator('.react-flow__node[data-id="step-1-0"]'); // a parallel node (offset row)
    const box = await node.boundingBox();
    expect(box).not.toBeNull();

    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await page.mouse.down();
    await page.mouse.move(box!.x + box!.width / 2 + drag.dx, box!.y + box!.height / 2 + drag.dy, { steps: 14 });

    await assertAllEdgesOrthogonal(page);
    await page.screenshot({ path: `e2e/screenshots/drag-${drag.name}.png` });
    await page.mouse.up();
  });
}

test('free drag moves the node vertically (no center-row snap)', async ({ page }) => {
  await open(page);
  const node = page.locator('.react-flow__node[data-id="step-1-0"]');
  const box = await node.boundingBox();
  const beforeY = await nodeTransformY(page, 'step-1-0');
  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await page.mouse.down();
  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2 + 120, { steps: 12 });
  const duringY = await nodeTransformY(page, 'step-1-0');
  await page.mouse.up();
  // The node follows the cursor downward (it is not snapped back to the center row).
  expect(duringY - beforeY).toBeGreaterThan(80);
});
