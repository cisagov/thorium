import { test, expect } from '@playwright/test';
import path from 'path';
import { snapshot, setupMockAuth } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

test.describe('OverlayWindow', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/overlay-window');
    await page.waitForLoadState('domcontentloaded');
  });

  test('test page renders with all toggle buttons', async ({ page }) => {
    await expect(page.locator('[data-testid="toggle-fixed-center"]')).toBeVisible();
    await expect(page.locator('[data-testid="toggle-fixed-bottom-right"]')).toBeVisible();
    await expect(page.locator('[data-testid="toggle-fixed-custom"]')).toBeVisible();
    await expect(page.locator('[data-testid="toggle-absolute-ref"]')).toBeVisible();
    await expect(page.locator('[data-testid="toggle-absolute-custom"]')).toBeVisible();
    await expect(page.locator('[data-testid="toggle-fixed-no-drag"]')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-initial');
  });

  test('Fixed + Center window appears centered in viewport', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    const window = page.locator('.test-window-fixed-center');
    await expect(window).toBeVisible();

    // verify fixed positioning
    const position = await window.evaluate((el) => getComputedStyle(el).position);
    expect(position).toBe('fixed');

    // verify roughly centered (within 100px tolerance for bounds clamping)
    const box = await window.boundingBox();
    const viewport = page.viewportSize()!;
    expect(box).toBeTruthy();
    expect(Math.abs(box!.x + box!.width / 2 - viewport.width / 2)).toBeLessThan(100);
    expect(Math.abs(box!.y + box!.height / 2 - viewport.height / 2)).toBeLessThan(100);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-fixed-center');
  });

  test('Fixed + BottomRight window appears near bottom-right', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-bottom-right"]').click();
    const window = page.locator('.test-window-fixed-bottom-right');
    await expect(window).toBeVisible();

    const box = await window.boundingBox();
    const viewport = page.viewportSize()!;
    expect(box).toBeTruthy();
    // should be in the bottom-right quadrant
    expect(box!.x + box!.width).toBeGreaterThan(viewport.width / 2);
    expect(box!.y + box!.height).toBeGreaterThan(viewport.height / 2);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-fixed-bottom-right');
  });

  test('Fixed + Custom position places window at explicit coordinates', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-custom"]').click();
    const window = page.locator('.test-window-fixed-custom');
    await expect(window).toBeVisible();

    const box = await window.boundingBox();
    expect(box).toBeTruthy();
    // customPosition is { top: 100, left: 150 }
    expect(Math.abs(box!.y - 100)).toBeLessThan(5);
    expect(Math.abs(box!.x - 150)).toBeLessThan(5);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-fixed-custom');
  });

  test('Absolute + parentRef positions window relative to button', async ({ page }) => {
    const button = page.locator('[data-testid="toggle-absolute-ref"]');
    await button.click();
    const window = page.locator('.test-window-absolute-ref');
    await expect(window).toBeVisible();

    // verify absolute positioning
    const position = await window.evaluate((el) => getComputedStyle(el).position);
    expect(position).toBe('absolute');

    // window should appear near the button (below it with Placement.Bottom)
    const buttonBox = await button.boundingBox();
    const windowBox = await window.boundingBox();
    expect(buttonBox).toBeTruthy();
    expect(windowBox).toBeTruthy();
    // window top should be at or below button bottom
    expect(windowBox!.y).toBeGreaterThanOrEqual(buttonBox!.y);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-absolute-ref');
  });

  test('Absolute + Custom position places window at explicit document coordinates', async ({ page }) => {
    await page.locator('[data-testid="toggle-absolute-custom"]').click();
    const window = page.locator('.test-window-absolute-custom');
    await expect(window).toBeVisible();

    const position = await window.evaluate((el) => getComputedStyle(el).position);
    expect(position).toBe('absolute');

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-absolute-custom');
  });

  test('Fixed non-draggable window cannot be dragged', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-no-drag"]').click();
    const window = page.locator('.test-window-fixed-no-drag');
    await expect(window).toBeVisible();

    const boxBefore = await window.boundingBox();
    expect(boxBefore).toBeTruthy();

    // attempt to drag the header
    const header = window.locator('div').first();
    await header.hover();
    await page.mouse.down();
    await page.mouse.move(boxBefore!.x + 200, boxBefore!.y + 200);
    await page.mouse.up();

    const boxAfter = await window.boundingBox();
    expect(boxAfter).toBeTruthy();
    // position should not have changed
    expect(Math.abs(boxAfter!.x - boxBefore!.x)).toBeLessThan(2);
    expect(Math.abs(boxAfter!.y - boxBefore!.y)).toBeLessThan(2);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-fixed-no-drag');
  });

  test('window can be dragged to a new position', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    const window = page.locator('.test-window-fixed-center');
    await expect(window).toBeVisible();

    const boxBefore = await window.boundingBox();
    expect(boxBefore).toBeTruthy();

    // drag the header 100px right and 50px down
    const headerY = boxBefore!.y + 15;
    const headerX = boxBefore!.x + boxBefore!.width / 2;
    await page.mouse.move(headerX, headerY);
    await page.mouse.down();
    await page.mouse.move(headerX + 100, headerY + 50, { steps: 5 });
    await page.mouse.up();

    const boxAfter = await window.boundingBox();
    expect(boxAfter).toBeTruthy();
    // position should have shifted approximately +100x, +50y
    expect(boxAfter!.x - boxBefore!.x).toBeGreaterThan(50);
    expect(boxAfter!.y - boxBefore!.y).toBeGreaterThan(20);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-dragged');
  });

  test('window can be resized via corner handle', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    const window = page.locator('.test-window-fixed-center');
    await expect(window).toBeVisible();

    const boxBefore = await window.boundingBox();
    expect(boxBefore).toBeTruthy();

    // grab the bottom-right corner and drag outward
    const cornerX = boxBefore!.x + boxBefore!.width - 3;
    const cornerY = boxBefore!.y + boxBefore!.height - 3;
    await page.mouse.move(cornerX, cornerY);
    await page.mouse.down();
    await page.mouse.move(cornerX + 80, cornerY + 60, { steps: 5 });
    await page.mouse.up();

    const boxAfter = await window.boundingBox();
    expect(boxAfter).toBeTruthy();
    // size should have increased
    expect(boxAfter!.width).toBeGreaterThan(boxBefore!.width + 30);
    expect(boxAfter!.height).toBeGreaterThan(boxBefore!.height + 20);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-resized');
  });

  test('close button hides window and fires onHide', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    const window = page.locator('.test-window-fixed-center');
    await expect(window).toBeVisible();

    // click the X close button
    const closeBtn = window.locator('button').filter({ hasText: 'X' });
    await closeBtn.click();

    // window should be hidden
    await expect(window).not.toBeVisible();

    // verify onHide was called by checking the status bar
    const statusBar = page.locator('[data-testid="closed-windows"]');
    await expect(statusBar).toContainText('fixedCenter');

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-closed');
  });

  test('clicking a background window brings it to front', async ({ page }) => {
    // open two windows
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await page.locator('[data-testid="toggle-fixed-bottom-right"]').click();

    const window1 = page.locator('.test-window-fixed-center');
    const window2 = page.locator('.test-window-fixed-bottom-right');
    await expect(window1).toBeVisible();
    await expect(window2).toBeVisible();

    // window2 was opened second so it should be on top initially
    const z1Before = await window1.evaluate((el) => parseInt(el.style.zIndex || '0'));
    const z2Before = await window2.evaluate((el) => parseInt(el.style.zIndex || '0'));
    expect(z2Before).toBeGreaterThan(z1Before);

    // click window1 to bring it to front
    await window1.click();

    const z1After = await window1.evaluate((el) => parseInt(el.style.zIndex || '0'));
    const z2After = await window2.evaluate((el) => parseInt(el.style.zIndex || '0'));
    expect(z1After).toBeGreaterThan(z2After);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-z-index');
  });

  test('window can be dragged then resized in sequence', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    const win = page.locator('.test-window-fixed-center');
    await expect(win).toBeVisible();

    const boxInitial = await win.boundingBox();
    expect(boxInitial).toBeTruthy();

    // step 1: drag to upper-left area
    const headerY = boxInitial!.y + 15;
    const headerX = boxInitial!.x + boxInitial!.width / 2;
    await page.mouse.move(headerX, headerY);
    await page.mouse.down();
    await page.mouse.move(80, 80, { steps: 8 });
    await page.mouse.up();

    const boxAfterDrag = await win.boundingBox();
    expect(boxAfterDrag).toBeTruthy();
    expect(boxAfterDrag!.width).toBe(boxInitial!.width);
    expect(boxAfterDrag!.height).toBe(boxInitial!.height);

    // step 2: resize by dragging the right edge outward
    const rightEdgeX = boxAfterDrag!.x + boxAfterDrag!.width - 1;
    const rightEdgeY = boxAfterDrag!.y + boxAfterDrag!.height / 2;
    await page.mouse.move(rightEdgeX, rightEdgeY);
    await page.mouse.down();
    await page.mouse.move(rightEdgeX + 120, rightEdgeY, { steps: 5 });
    await page.mouse.up();

    const boxAfterResize = await win.boundingBox();
    expect(boxAfterResize).toBeTruthy();
    expect(boxAfterResize!.width).toBeGreaterThan(boxAfterDrag!.width + 50);
    // height should be unchanged from edge-only resize
    expect(Math.abs(boxAfterResize!.height - boxAfterDrag!.height)).toBeLessThan(5);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-drag-then-resize');
  });

  test('window resized via bottom edge only changes height', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    const win = page.locator('.test-window-fixed-center');
    await expect(win).toBeVisible();

    const boxBefore = await win.boundingBox();
    expect(boxBefore).toBeTruthy();

    // grab bottom edge (center of bottom edge, not corner)
    const bottomX = boxBefore!.x + boxBefore!.width / 2;
    const bottomY = boxBefore!.y + boxBefore!.height - 1;
    await page.mouse.move(bottomX, bottomY);
    await page.mouse.down();
    await page.mouse.move(bottomX, bottomY + 100, { steps: 5 });
    await page.mouse.up();

    const boxAfter = await win.boundingBox();
    expect(boxAfter).toBeTruthy();
    expect(boxAfter!.height).toBeGreaterThan(boxBefore!.height + 40);
    // width should be unchanged
    expect(Math.abs(boxAfter!.width - boxBefore!.width)).toBeLessThan(5);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-resize-bottom-edge');
  });

  test('window resized via top-left corner adjusts position and size', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    const win = page.locator('.test-window-fixed-center');
    await expect(win).toBeVisible();

    const boxBefore = await win.boundingBox();
    expect(boxBefore).toBeTruthy();

    // grab top-left corner
    const cornerX = boxBefore!.x + 3;
    const cornerY = boxBefore!.y + 3;
    await page.mouse.move(cornerX, cornerY);
    await page.mouse.down();
    await page.mouse.move(cornerX - 60, cornerY - 40, { steps: 5 });
    await page.mouse.up();

    const boxAfter = await win.boundingBox();
    expect(boxAfter).toBeTruthy();
    // size should have grown in both dimensions
    expect(boxAfter!.width).toBeGreaterThan(boxBefore!.width + 20);
    expect(boxAfter!.height).toBeGreaterThan(boxBefore!.height + 10);
    // top-left corner should have moved up and left
    expect(boxAfter!.x).toBeLessThan(boxBefore!.x);
    expect(boxAfter!.y).toBeLessThan(boxBefore!.y);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-resize-top-left');
  });

  test('custom-positioned window can be dragged from its initial position', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-custom"]').click();
    const win = page.locator('.test-window-fixed-custom');
    await expect(win).toBeVisible();

    const boxBefore = await win.boundingBox();
    expect(boxBefore).toBeTruthy();
    // verify it starts at custom position
    expect(Math.abs(boxBefore!.y - 100)).toBeLessThan(5);
    expect(Math.abs(boxBefore!.x - 150)).toBeLessThan(5);

    // drag it 150px right and 80px down
    const headerX = boxBefore!.x + boxBefore!.width / 2;
    const headerY = boxBefore!.y + 15;
    await page.mouse.move(headerX, headerY);
    await page.mouse.down();
    await page.mouse.move(headerX + 150, headerY + 80, { steps: 5 });
    await page.mouse.up();

    const boxAfter = await win.boundingBox();
    expect(boxAfter).toBeTruthy();
    expect(boxAfter!.x - boxBefore!.x).toBeGreaterThan(100);
    expect(boxAfter!.y - boxBefore!.y).toBeGreaterThan(50);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-custom-dragged');
  });

  test('multiple window variants screenshot', async ({ page }) => {
    // open several windows at once for a composite screenshot
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await page.locator('[data-testid="toggle-fixed-custom"]').click();
    await page.locator('[data-testid="toggle-fixed-no-drag"]').click();

    await expect(page.locator('.test-window-fixed-center')).toBeVisible();
    await expect(page.locator('.test-window-fixed-custom')).toBeVisible();
    await expect(page.locator('.test-window-fixed-no-drag')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-multiple');
  });

  test('headerless window with onHide shows floating close button', async ({ page }) => {
    await page.locator('[data-testid="toggle-headerless-menu"]').click();
    const win = page.locator('.test-window-headerless-menu');
    await expect(win).toBeVisible();

    // should have no header element
    const headers = win.locator('div').filter({ has: page.locator('button:has-text("X")') });
    // the floating close button should exist as a direct child, not inside a header
    const closeBtn = win.locator(':scope > button').filter({ hasText: 'X' });
    await expect(closeBtn).toBeVisible();

    // content should fill from top (no 40px header margin)
    const content = page.locator('[data-testid="window-headerless-menu"]');
    await expect(content).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-headerless-menu');
  });

  test('headerless window close button hides window and fires onHide', async ({ page }) => {
    await page.locator('[data-testid="toggle-headerless-menu"]').click();
    const win = page.locator('.test-window-headerless-menu');
    await expect(win).toBeVisible();

    // click the floating close button
    const closeBtn = win.locator(':scope > button').filter({ hasText: 'X' });
    await closeBtn.click();

    await expect(win).not.toBeVisible();
    const statusBar = page.locator('[data-testid="closed-windows"]');
    await expect(statusBar).toContainText('headerlessMenu');
  });

  test('headerless window without onHide has no close button', async ({ page }) => {
    await page.locator('[data-testid="toggle-headerless-no-close"]').click();
    const win = page.locator('.test-window-headerless-no-close');
    await expect(win).toBeVisible();

    // should have no close button at all
    const closeBtn = win.locator('button').filter({ hasText: 'X' });
    await expect(closeBtn).toHaveCount(0);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-headerless-no-close');
  });

  test('headerless window cannot be dragged', async ({ page }) => {
    await page.locator('[data-testid="toggle-headerless-menu"]').click();
    const win = page.locator('.test-window-headerless-menu');
    await expect(win).toBeVisible();

    const boxBefore = await win.boundingBox();
    expect(boxBefore).toBeTruthy();

    // attempt to drag from content area
    const centerX = boxBefore!.x + boxBefore!.width / 2;
    const centerY = boxBefore!.y + boxBefore!.height / 2;
    await page.mouse.move(centerX, centerY);
    await page.mouse.down();
    await page.mouse.move(centerX + 100, centerY + 100, { steps: 5 });
    await page.mouse.up();

    const boxAfter = await win.boundingBox();
    expect(boxAfter).toBeTruthy();
    // position should not have changed (no drag handle)
    expect(Math.abs(boxAfter!.x - boxBefore!.x)).toBeLessThan(2);
    expect(Math.abs(boxAfter!.y - boxBefore!.y)).toBeLessThan(2);
  });

  test('headerless window can still be resized', async ({ page }) => {
    await page.locator('[data-testid="toggle-headerless-menu"]').click();
    const win = page.locator('.test-window-headerless-menu');
    await expect(win).toBeVisible();

    const boxBefore = await win.boundingBox();
    expect(boxBefore).toBeTruthy();

    // resize from bottom-right corner
    const cornerX = boxBefore!.x + boxBefore!.width - 3;
    const cornerY = boxBefore!.y + boxBefore!.height - 3;
    await page.mouse.move(cornerX, cornerY);
    await page.mouse.down();
    await page.mouse.move(cornerX + 50, cornerY + 40, { steps: 5 });
    await page.mouse.up();

    const boxAfter = await win.boundingBox();
    expect(boxAfter).toBeTruthy();
    expect(boxAfter!.width).toBeGreaterThan(boxBefore!.width + 20);
    expect(boxAfter!.height).toBeGreaterThan(boxBefore!.height + 15);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-headerless-resized');
  });
});

test.describe('WindowManager Multi-Window', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/overlay-window');
    await page.waitForLoadState('domcontentloaded');
  });

  test('all windows can be opened simultaneously', async ({ page }) => {
    const toggleIds = [
      'toggle-fixed-center',
      'toggle-fixed-bottom-right',
      'toggle-fixed-custom',
      'toggle-absolute-ref',
      'toggle-absolute-custom',
      'toggle-fixed-no-drag',
      'toggle-headerless-menu',
      'toggle-headerless-no-close',
    ];
    for (const id of toggleIds) {
      await page.locator(`[data-testid="${id}"]`).click();
    }

    const windowClasses = [
      '.test-window-fixed-center',
      '.test-window-fixed-bottom-right',
      '.test-window-fixed-custom',
      '.test-window-absolute-ref',
      '.test-window-absolute-custom',
      '.test-window-fixed-no-drag',
      '.test-window-headerless-menu',
      '.test-window-headerless-no-close',
    ];
    for (const cls of windowClasses) {
      await expect(page.locator(cls)).toBeVisible();
    }

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-all-windows');
  });

  test('each window gets a unique z-index', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await page.locator('[data-testid="toggle-fixed-bottom-right"]').click();
    await page.locator('[data-testid="toggle-fixed-custom"]').click();

    const w1 = page.locator('.test-window-fixed-center');
    const w2 = page.locator('.test-window-fixed-bottom-right');
    const w3 = page.locator('.test-window-fixed-custom');

    await expect(w1).toBeVisible();
    await expect(w2).toBeVisible();
    await expect(w3).toBeVisible();

    const z1 = await w1.evaluate((el) => parseInt(el.style.zIndex || '0'));
    const z2 = await w2.evaluate((el) => parseInt(el.style.zIndex || '0'));
    const z3 = await w3.evaluate((el) => parseInt(el.style.zIndex || '0'));

    // all z-indices should be distinct
    const zSet = new Set([z1, z2, z3]);
    expect(zSet.size).toBe(3);
    // all should be within the manager's range (1000-4000)
    expect(z1).toBeGreaterThanOrEqual(1000);
    expect(z2).toBeGreaterThanOrEqual(1000);
    expect(z3).toBeGreaterThanOrEqual(1000);
  });

  test('clicking windows in sequence rotates z-order correctly', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await page.locator('[data-testid="toggle-fixed-bottom-right"]').click();
    await page.locator('[data-testid="toggle-fixed-custom"]').click();

    const w1 = page.locator('.test-window-fixed-center');
    const w2 = page.locator('.test-window-fixed-bottom-right');
    const w3 = page.locator('.test-window-fixed-custom');

    // w3 should be on top (opened last)
    let z1 = await w1.evaluate((el) => parseInt(el.style.zIndex || '0'));
    let z3 = await w3.evaluate((el) => parseInt(el.style.zIndex || '0'));
    expect(z3).toBeGreaterThan(z1);

    // click w1 to bring it to front
    await w1.click();
    z1 = await w1.evaluate((el) => parseInt(el.style.zIndex || '0'));
    let z2 = await w2.evaluate((el) => parseInt(el.style.zIndex || '0'));
    z3 = await w3.evaluate((el) => parseInt(el.style.zIndex || '0'));
    expect(z1).toBeGreaterThan(z2);
    expect(z1).toBeGreaterThan(z3);

    // click w2 to bring it to front
    await w2.click();
    z1 = await w1.evaluate((el) => parseInt(el.style.zIndex || '0'));
    z2 = await w2.evaluate((el) => parseInt(el.style.zIndex || '0'));
    z3 = await w3.evaluate((el) => parseInt(el.style.zIndex || '0'));
    expect(z2).toBeGreaterThan(z1);
    expect(z2).toBeGreaterThan(z3);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-z-rotation');
  });

  test('closing one window does not affect others', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await page.locator('[data-testid="toggle-fixed-bottom-right"]').click();
    await page.locator('[data-testid="toggle-fixed-custom"]').click();

    const w1 = page.locator('.test-window-fixed-center');
    const w2 = page.locator('.test-window-fixed-bottom-right');
    const w3 = page.locator('.test-window-fixed-custom');

    await expect(w1).toBeVisible();
    await expect(w2).toBeVisible();
    await expect(w3).toBeVisible();

    // close w2
    const closeBtn = w2.locator('button').filter({ hasText: 'X' });
    await closeBtn.click();

    await expect(w2).not.toBeVisible();
    await expect(w1).toBeVisible();
    await expect(w3).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-close-one');
  });

  test('closing and reopening a window preserves z-index management', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await page.locator('[data-testid="toggle-fixed-bottom-right"]').click();

    const w1 = page.locator('.test-window-fixed-center');
    const w2 = page.locator('.test-window-fixed-bottom-right');

    await expect(w1).toBeVisible();
    await expect(w2).toBeVisible();

    // close w1 via its X button
    const closeBtn = w1.locator('button').filter({ hasText: 'X' });
    await closeBtn.click();
    await expect(w1).not.toBeVisible();

    // reopen w1 via the toggle button
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await expect(w1).toBeVisible();

    // click w1 to bring it to front — z-index management should still work after close/reopen
    await w1.click();
    const z1 = await w1.evaluate((el) => parseInt(el.style.zIndex || '0'));
    const z2 = await w2.evaluate((el) => parseInt(el.style.zIndex || '0'));
    expect(z1).toBeGreaterThan(z2);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-reopen');
  });

  test('dragging one window does not move others', async ({ page }) => {
    await page.locator('[data-testid="toggle-fixed-center"]').click();
    await page.locator('[data-testid="toggle-fixed-bottom-right"]').click();

    const w1 = page.locator('.test-window-fixed-center');
    const w2 = page.locator('.test-window-fixed-bottom-right');

    await expect(w1).toBeVisible();
    await expect(w2).toBeVisible();

    const w2BoxBefore = await w2.boundingBox();
    expect(w2BoxBefore).toBeTruthy();

    // drag w1
    const w1Box = await w1.boundingBox();
    expect(w1Box).toBeTruthy();
    const headerX = w1Box!.x + w1Box!.width / 2;
    const headerY = w1Box!.y + 15;
    await page.mouse.move(headerX, headerY);
    await page.mouse.down();
    await page.mouse.move(headerX + 100, headerY + 80, { steps: 5 });
    await page.mouse.up();

    // w2 should not have moved
    const w2BoxAfter = await w2.boundingBox();
    expect(w2BoxAfter).toBeTruthy();
    expect(Math.abs(w2BoxAfter!.x - w2BoxBefore!.x)).toBeLessThan(2);
    expect(Math.abs(w2BoxAfter!.y - w2BoxBefore!.y)).toBeLessThan(2);

    await snapshot(page, SCREENSHOT_DIR, 'overlay-window-drag-independence');
  });
});
