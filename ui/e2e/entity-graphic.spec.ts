import { test, expect } from '@playwright/test';
import path from 'path';
import {
  authenticate,
  createEntityWithImage,
  deleteEntity,
  snapshot,
  loginViaUI,
  TEST_USER,
  TEST_PASS,
} from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const TEST_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#e5913d">
  <circle cx="12" cy="12" r="10"/>
  <path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="#fff" stroke-width="1.5" stroke-linecap="round" fill="none"/>
  <circle cx="9" cy="9" r="1.5" fill="#fff"/>
  <circle cx="15" cy="9" r="1.5" fill="#fff"/>
</svg>`;

interface MaskedElementInfo {
  maskImage: string;
  bgColor: string;
  width: number;
  height: number;
}

async function findMaskedElement(container: import('@playwright/test').Locator): Promise<MaskedElementInfo | null> {
  return container.evaluate((el) => {
    const allEls = el.querySelectorAll('*');
    for (const child of allEls) {
      const style = getComputedStyle(child);
      const mask = style.maskImage || style.webkitMaskImage;
      if (mask && mask !== 'none') {
        return {
          maskImage: mask,
          bgColor: style.backgroundColor,
          width: child.getBoundingClientRect().width,
          height: child.getBoundingClientRect().height,
        };
      }
    }
    return null;
  });
}

test.describe('Entity Graphic — SVG Icon Color', () => {
  let token: string;
  let deviceId: string;
  let vendorId: string;

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);

    const svgImage = {
      data: Buffer.from(TEST_SVG),
      filename: 'icon.svg',
      contentType: 'image/svg+xml',
    };

    deviceId = await createEntityWithImage(token, 'E2E-SVG-Device', 'Device', ['system'], svgImage);
    vendorId = await createEntityWithImage(token, 'E2E-SVG-Vendor', 'Vendor', ['system'], svgImage);
  });

  test.afterAll(async () => {
    await deleteEntity(token, deviceId).catch(() => {});
    await deleteEntity(token, vendorId).catch(() => {});
  });

  test('device browsing page shows SVG icon matching text color', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/devices');
    await page.waitForLoadState('networkidle');

    const entityRow = page.locator('a.no-decoration').filter({ hasText: 'E2E-SVG-Device' }).first();
    await expect(entityRow).toBeVisible({ timeout: 15000 });

    await page.waitForTimeout(2000);

    const masked = await findMaskedElement(entityRow);
    expect(masked).toBeTruthy();
    expect(masked!.maskImage).not.toBe('none');

    expect(masked!.bgColor).not.toBe('rgba(0, 0, 0, 0)');
    expect(masked!.bgColor).not.toBe('transparent');

    const textColor = await entityRow.evaluate((el) => getComputedStyle(el).color);
    expect(masked!.bgColor).toBe(textColor);

    await snapshot(page, SCREENSHOT_DIR, 'device-browsing-svg-icon');
  });

  test('vendor browsing page shows SVG icon matching text color', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/vendors');
    await page.waitForLoadState('networkidle');

    const entityRow = page.locator('a.no-decoration').filter({ hasText: 'E2E-SVG-Vendor' }).first();
    await expect(entityRow).toBeVisible({ timeout: 15000 });

    await page.waitForTimeout(2000);

    const masked = await findMaskedElement(entityRow);
    expect(masked).toBeTruthy();
    expect(masked!.maskImage).not.toBe('none');

    const textColor = await entityRow.evaluate((el) => getComputedStyle(el).color);
    expect(masked!.bgColor).toBe(textColor);

    await snapshot(page, SCREENSHOT_DIR, 'vendor-browsing-svg-icon');
  });

  test('SVG icon does not increase row height', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/devices');
    await page.waitForLoadState('networkidle');

    const entityRow = page.locator('a.no-decoration').filter({ hasText: 'E2E-SVG-Device' }).first();
    await expect(entityRow).toBeVisible({ timeout: 15000 });

    await page.waitForTimeout(2000);

    const masked = await findMaskedElement(entityRow);
    expect(masked).toBeTruthy();

    // Icon is 1.2em (~19px), must stay compact
    expect(masked!.height).toBeLessThanOrEqual(24);
    expect(masked!.width).toBeLessThanOrEqual(24);
  });

  test('device details page shows SVG graphic with text color in header', async ({ page }) => {
    await loginViaUI(page);
    await page.goto(`/device/${deviceId}`);
    await page.waitForLoadState('networkidle');

    await expect(page.getByText('E2E-SVG-Device').first()).toBeVisible({ timeout: 15000 });

    // Poll for the masked element — the image blob fetch is async
    let masked: MaskedElementInfo | null = null;
    for (let i = 0; i < 10; i++) {
      await page.waitForTimeout(1000);
      masked = await page.evaluate(() => {
        const cards = document.querySelectorAll('.card');
        for (const card of cards) {
          const divs = card.querySelectorAll('*');
          for (const div of divs) {
            const style = getComputedStyle(div);
            const mask = style.maskImage || style.webkitMaskImage;
            if (mask && mask !== 'none') {
              return {
                maskImage: mask,
                bgColor: style.backgroundColor,
                width: div.getBoundingClientRect().width,
                height: div.getBoundingClientRect().height,
              };
            }
          }
        }
        return null;
      });
      if (masked) break;
    }

    expect(masked).toBeTruthy();
    expect(masked!.maskImage).not.toBe('none');
    expect(masked!.bgColor).not.toBe('rgba(0, 0, 0, 0)');
    expect(masked!.width).toBe(80);
    expect(masked!.height).toBe(80);

    await snapshot(page, SCREENSHOT_DIR, 'device-details-svg-graphic');
  });
});
