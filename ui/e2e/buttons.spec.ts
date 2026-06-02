import { test, expect } from '@playwright/test';
import path from 'path';
import { snapshot, setupMockAuth } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

test.describe('Button Components', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/buttons');
    await page.waitForLoadState('domcontentloaded');
  });

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  test('all filled variant buttons render', async ({ page }) => {
    const variants = ['primary', 'secondary', 'ok', 'danger', 'warning', 'info'];
    for (const v of variants) {
      await expect(page.locator(`[data-testid="btn-${v}"]`)).toBeVisible();
    }
    await snapshot(page, SCREENSHOT_DIR, 'buttons-filled-variants');
  });

  test('ghost and icon variant buttons render', async ({ page }) => {
    await expect(page.locator('[data-testid="btn-ghost"]')).toBeVisible();
    await expect(page.locator('[data-testid="iconbtn-pen"]')).toBeVisible();
    await expect(page.locator('[data-testid="iconbtn-plus-round"]')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'buttons-ghost-icon');
  });

  test('all size groups render', async ({ page }) => {
    for (const s of ['xs', 'sm', 'md', 'lg']) {
      await expect(page.locator(`[data-testid="size-group-${s}"]`)).toBeVisible();
      await expect(page.locator(`[data-testid="btn-primary-${s}"]`)).toBeVisible();
    }
    await snapshot(page, SCREENSHOT_DIR, 'buttons-sizes');
  });

  test('buttons with icons render both icon and text', async ({ page }) => {
    const createBtn = page.locator('[data-testid="btn-icon-create"]');
    await expect(createBtn).toBeVisible();
    await expect(createBtn).toContainText('Create');
    // should have an SVG icon inside
    await expect(createBtn.locator('svg')).toBeVisible();
  });

  test('iconbutton filled variants render', async ({ page }) => {
    await expect(page.locator('[data-testid="iconbtn-filled-primary"]')).toBeVisible();
    await expect(page.locator('[data-testid="iconbtn-filled-ok"]')).toBeVisible();
    await expect(page.locator('[data-testid="iconbtn-filled-danger"]')).toBeVisible();
    await expect(page.locator('[data-testid="iconbtn-filled-warning"]')).toBeVisible();
  });

  test('round icon buttons have circular border-radius', async ({ page }) => {
    const roundBtn = page.locator('[data-testid="iconbtn-round-default"]');
    await expect(roundBtn).toBeVisible();
    const borderRadius = await roundBtn.evaluate((el) => getComputedStyle(el).borderRadius);
    expect(borderRadius).toBe('50%');
  });

  test('disabled variant buttons render with disabled attribute', async ({ page }) => {
    const variants = ['primary', 'secondary', 'ok', 'danger', 'warning', 'info'];
    for (const v of variants) {
      const btn = page.locator(`[data-testid="btn-disabled-${v}"]`);
      await expect(btn).toBeVisible();
      await expect(btn).toBeDisabled();
    }

    await expect(page.locator('[data-testid="btn-disabled-ghost"]')).toBeDisabled();
    await expect(page.locator('[data-testid="iconbtn-disabled-pen"]')).toBeDisabled();
    await expect(page.locator('[data-testid="iconbtn-disabled-trash"]')).toBeDisabled();

    await snapshot(page, SCREENSHOT_DIR, 'buttons-disabled');
  });

  // ---------------------------------------------------------------------------
  // Sizing
  // ---------------------------------------------------------------------------

  test('button font size increases with size prop', async ({ page }) => {
    const sizes = ['xs', 'sm', 'md', 'lg'];
    const fontSizes: number[] = [];
    for (const s of sizes) {
      const btn = page.locator(`[data-testid="btn-primary-${s}"]`);
      const fontSize = await btn.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
      fontSizes.push(fontSize);
    }
    for (let i = 1; i < fontSizes.length; i++) {
      expect(fontSizes[i]).toBeGreaterThan(fontSizes[i - 1]);
    }
  });

  test('iconbutton dimensions increase with size prop', async ({ page }) => {
    const sizes = ['xs', 'sm', 'md', 'lg'];
    const dims: number[] = [];
    for (const s of sizes) {
      const btn = page.locator(`[data-testid="iconbtn-${s}"]`);
      const box = await btn.boundingBox();
      expect(box).toBeTruthy();
      dims.push(box!.width);
    }
    for (let i = 1; i < dims.length; i++) {
      expect(dims[i]).toBeGreaterThan(dims[i - 1]);
    }
  });

  test('iconbutton is square', async ({ page }) => {
    const btn = page.locator('[data-testid="iconbtn-pen"]');
    const box = await btn.boundingBox();
    expect(box).toBeTruthy();
    expect(Math.abs(box!.width - box!.height)).toBeLessThan(2);
  });

  // ---------------------------------------------------------------------------
  // Clicks
  // ---------------------------------------------------------------------------

  test('click on enabled button increments counter', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-click-ok"]');
    const counter = page.locator('[data-testid="click-count-ok"]');
    await expect(counter).toHaveText('0');

    await btn.click();
    await expect(counter).toHaveText('1');

    await btn.click();
    await btn.click();
    await expect(counter).toHaveText('3');
  });

  test('click on iconbutton increments counter', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-click-icon"]');
    const counter = page.locator('[data-testid="click-count-icon"]');
    await expect(counter).toHaveText('0');

    await btn.click();
    await expect(counter).toHaveText('1');
  });

  test('multiple buttons track independent click counts', async ({ page }) => {
    await page.locator('[data-testid="btn-click-ok"]').click();
    await page.locator('[data-testid="btn-click-ok"]').click();
    await page.locator('[data-testid="btn-click-primary"]').click();

    await expect(page.locator('[data-testid="click-count-ok"]')).toHaveText('2');
    await expect(page.locator('[data-testid="click-count-primary"]')).toHaveText('1');
    await expect(page.locator('[data-testid="click-count-icon"]')).toHaveText('0');
  });

  test('disabled button does not fire onClick', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-click-disabled"]');
    const result = page.locator('[data-testid="disabled-click-result"]');
    await expect(result).toHaveText('blocked');

    // attempt click via playwright — disabled buttons won't receive the event
    await btn.click({ force: true });
    await expect(result).toHaveText('blocked');
  });

  // ---------------------------------------------------------------------------
  // Tab Navigation & Focus
  // ---------------------------------------------------------------------------

  test('tab moves focus through buttons in order', async ({ page }) => {
    const firstBtn = page.locator('[data-testid="tab-btn-1"]');
    await firstBtn.focus();
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-1');

    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-2');

    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-3');
  });

  test('tab skips disabled button', async ({ page }) => {
    const thirdBtn = page.locator('[data-testid="tab-btn-3"]');
    await thirdBtn.focus();
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-3');

    // next tab should skip disabled and land on tab-btn-4
    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-4');
  });

  test('tab reaches iconbutton', async ({ page }) => {
    const fourthBtn = page.locator('[data-testid="tab-btn-4"]');
    await fourthBtn.focus();

    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-5');
  });

  test('shift+tab navigates backwards', async ({ page }) => {
    const fourthBtn = page.locator('[data-testid="tab-btn-4"]');
    await fourthBtn.focus();
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-4');

    await page.keyboard.press('Shift+Tab');
    // should skip disabled and land on tab-btn-3
    await expect(page.locator('[data-testid="last-focused"]')).toContainText('tab-btn-3');
  });

  test('enter key triggers click on focused button', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-click-ok"]');
    const counter = page.locator('[data-testid="click-count-ok"]');
    await expect(counter).toHaveText('0');

    await btn.focus();
    await page.keyboard.press('Enter');
    await expect(counter).toHaveText('1');
  });

  test('space key triggers click on focused button', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-click-primary"]');
    const counter = page.locator('[data-testid="click-count-primary"]');
    await expect(counter).toHaveText('0');

    await btn.focus();
    await page.keyboard.press('Space');
    await expect(counter).toHaveText('1');
  });

  // ---------------------------------------------------------------------------
  // Focus-visible styling
  // ---------------------------------------------------------------------------

  test('focus-visible shows box-shadow ring', async ({ page }) => {
    // use keyboard to trigger focus-visible (tab into the button)
    const firstBtn = page.locator('[data-testid="tab-btn-1"]');
    await firstBtn.focus();
    await page.keyboard.press('Tab');

    const secondBtn = page.locator('[data-testid="tab-btn-2"]');
    const boxShadow = await secondBtn.evaluate((el) => getComputedStyle(el).boxShadow);
    // focus-visible should produce a non-none box-shadow
    expect(boxShadow).not.toBe('none');

    await snapshot(page, SCREENSHOT_DIR, 'buttons-focus-visible');
  });

  // ---------------------------------------------------------------------------
  // Button element attributes
  // ---------------------------------------------------------------------------

  test('buttons default to type="button"', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-primary"]');
    const type = await btn.getAttribute('type');
    expect(type).toBe('button');
  });

  test('disabled buttons have cursor not-allowed', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-disabled-primary"]');
    const cursor = await btn.evaluate((el) => getComputedStyle(el).cursor);
    expect(cursor).toBe('not-allowed');
  });

  test('enabled buttons have cursor pointer', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-primary"]');
    const cursor = await btn.evaluate((el) => getComputedStyle(el).cursor);
    expect(cursor).toBe('pointer');
  });

  test('disabled buttons have reduced opacity', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-disabled-primary"]');
    const opacity = await btn.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
    expect(opacity).toBeLessThan(1);
    expect(opacity).toBeGreaterThan(0);
  });

  // ---------------------------------------------------------------------------
  // Common patterns
  // ---------------------------------------------------------------------------

  test('common pattern sections render', async ({ page }) => {
    await expect(page.locator('[data-testid="pattern-delete"]')).toBeVisible();
    await expect(page.locator('[data-testid="pattern-cancel"]')).toBeVisible();
    await expect(page.locator('[data-testid="pattern-save"]')).toBeVisible();
    await expect(page.locator('[data-testid="pattern-discard"]')).toBeVisible();
    await expect(page.locator('[data-testid="pattern-toolbar-pen"]')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'buttons-patterns');
  });

  // ---------------------------------------------------------------------------
  // Full page screenshot
  // ---------------------------------------------------------------------------

  test('full page screenshot', async ({ page }) => {
    await snapshot(page, SCREENSHOT_DIR, 'buttons-full-page');
  });
});
