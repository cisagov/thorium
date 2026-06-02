import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { snapshot, setupMockAuth, MOCK_USER } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

test.describe('Sidebar Navigation Visual Validation', () => {
  test('sidebar renders all categories at full width (1400px)', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-full-width');

    await expect(page.locator('text=Analyze')).toBeVisible();
    await expect(page.locator('text=Search')).toBeVisible();
    await expect(page.locator('text=Browse')).toBeVisible();
    await expect(page.locator('text=Tools')).toBeVisible();
    await expect(page.locator('text=Groups')).toBeVisible();
    await expect(page.locator('text=Admin')).toBeVisible();
  });

  test('sidebar renders icon-only mode at narrow width (1000px)', async ({ page }) => {
    await page.setViewportSize({ width: 1000, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-icon-only');

    await expect(page.locator('text=Analyze')).not.toBeVisible();
    await expect(page.locator('text=Search')).not.toBeVisible();
  });

  test('sidebar hidden at mobile width (500px)', async ({ page }) => {
    await page.setViewportSize({ width: 500, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-mobile-hidden');
  });

  test('category expand/collapse shows subcategories', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await expect(page.locator('text=Files')).not.toBeVisible();

    const browseCategory = page.locator('text=Browse');
    await browseCategory.click();
    await page.waitForTimeout(500);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-browse-expanded');

    await expect(page.locator('text=Files')).toBeVisible();
    await expect(page.locator('text=Repos')).toBeVisible();
    await expect(page.locator('text=Collections')).toBeVisible();
    await expect(page.locator('text=Devices')).toBeVisible();
    await expect(page.locator('text=Vendors')).toBeVisible();
    await expect(page.locator('text=File Systems')).toBeVisible();
    await expect(page.locator('text=Sigma Rules')).toBeVisible();
  });

  test('tools category expand shows subcategories', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    const toolsCategory = page.locator('text=Tools');
    await toolsCategory.click();
    await page.waitForTimeout(500);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-tools-expanded');

    await expect(page.locator('text=Pipelines')).toBeVisible();
    await expect(page.locator('text=Images')).toBeVisible();
    await expect(page.locator('text=Stats')).toBeVisible();
  });

  test('admin category expand shows subcategories', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    const adminCategory = page.locator('text=Admin');
    await adminCategory.click();
    await page.waitForTimeout(500);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-admin-expanded');

    await expect(page.locator('text=Users')).toBeVisible();
    await expect(page.locator('text=Settings')).toBeVisible();
  });

  test('multiple categories can be expanded simultaneously', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await page.locator('text=Browse').click();
    await page.waitForTimeout(300);
    await page.locator('text=Tools').click();
    await page.waitForTimeout(300);
    await page.locator('text=Admin').click();
    await page.waitForTimeout(500);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-all-expanded');

    await expect(page.locator('text=Files')).toBeVisible();
    await expect(page.locator('text=Pipelines')).toBeVisible();
    await expect(page.locator('text=Settings')).toBeVisible();
  });

  test('flyout appears on hover for collapsed category', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await page.evaluate(() => {
      const el = document.querySelector('[data-testid="category-Browse"]');
      if (el) {
        el.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, cancelable: true }));
        el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, cancelable: true }));
        el.dispatchEvent(new MouseEvent('mouseenter', { bubbles: false, cancelable: false }));
      }
    });
    await page.waitForTimeout(500);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-browse-flyout');

    await expect(page.locator('text=Files').last()).toBeVisible();
    await expect(page.locator('text=Repos').last()).toBeVisible();
    await expect(page.locator('text=Sigma Rules').last()).toBeVisible();
  });

  test('flyout in icon-only mode on hover', async ({ page }) => {
    await page.setViewportSize({ width: 1000, height: 900 });
    await setupMockAuth(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await page.evaluate(() => {
      const el = document.querySelector('[data-testid="category-Browse"]');
      if (el) {
        el.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, cancelable: true }));
        el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, cancelable: true }));
        el.dispatchEvent(new MouseEvent('mouseenter', { bubbles: false, cancelable: false }));
      }
    });
    await page.waitForTimeout(500);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-icon-flyout');

    await expect(page.locator('text=Files').last()).toBeVisible();
  });

  test('non-admin user does not see Admin category', async ({ page }) => {
    const nonAdminUser = { ...MOCK_USER, role: 'User' };
    await page.route('**/api/users/whoami', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(nonAdminUser) }),
    );
    await page.route('**/api/**', (route) => {
      if (route.request().url().includes('/users/whoami')) return route.fallback();
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.context().addCookies([{
      name: 'THORIUM_TOKEN',
      value: MOCK_USER.token,
      domain: 'localhost',
      path: '/',
    }]);

    await page.setViewportSize({ width: 1400, height: 900 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    await snapshot(page, SCREENSHOT_DIR, 'sidebar-non-admin');

    await expect(page.locator('text=Browse')).toBeVisible();
    await expect(page.locator('text=Admin')).not.toBeVisible();
  });
});
