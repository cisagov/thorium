import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { snapshot } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const MOCK_USER = {
  username: 'test',
  role: 'Admin',
  email: 'test@thorium.dev',
  groups: ['system'],
  token: 'mock-token-for-visual-test',
  token_expiration: '2099-01-01T00:00:00Z',
  settings: { theme: 'Dark' },
  local: true,
  verified: true,
};

async function setupMockAuth(page: Page) {
  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }),
  );
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (url.includes('/users/whoami')) return route.fallback();
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.context().addCookies([{
    name: 'THORIUM_TOKEN',
    value: MOCK_USER.token,
    domain: 'localhost',
    path: '/',
  }]);
}

async function waitForEditor(page: Page) {
  await page.waitForSelector('.cm-editor', { timeout: 10000 });
  await page.waitForTimeout(500);
}

async function waitForLinter(page: Page) {
  await page.waitForTimeout(600);
}

async function setEditorContent(page: Page, text: string) {
  await page.evaluate((content) => {
    const container = document.querySelector('.cm-editor')?.parentElement as HTMLElement & { _cmView?: { state: { doc: { length: number } }; dispatch: (spec: unknown) => void } };
    const view = container?._cmView;
    if (view) {
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: content },
      });
    }
  }, text);
}

const VALID_RULE = `title: Okta User Account Locked Out
id: 14701da0-4b0f-4ee6-9c95-2ffb4e73bb9a
status: test
description: Detects when a user account is locked out.
references:
    - https://developer.okta.com/docs/reference/api/system-log/
    - https://developer.okta.com/docs/reference/api/event-types/
author: Austin Songer @austinsonger
date: 2021-09-12
modified: 2022-10-09
tags:
    - attack.impact
logsource:
    product: okta
    service: okta
detection:
    selection:
        displaymessage: Max sign in attempts exceeded
    condition: selection
falsepositives:
    - Unknown
level: medium`;

test.describe('Sigma Rule CodeEditor', () => {
  test.beforeEach(async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    await setupMockAuth(page);
    await page.goto('/test/sigma');

    try {
      await waitForEditor(page);
    } catch {
      if (consoleErrors.length > 0) {
        throw new Error(`Editor failed to load. Console errors:\n${consoleErrors.join('\n')}`);
      }
      throw new Error('Editor failed to load (no console errors captured)');
    }
  });

  test('valid rule renders without error diagnostics', async ({ page }) => {
    await waitForLinter(page);
    const errorRanges = page.locator('.cm-lintRange-error');
    await expect(errorRanges).toHaveCount(0);

    await snapshot(page, SCREENSHOT_DIR, 'sigma-valid-rule');
  });

  test('invalid status shows error with red tint', async ({ page }) => {
    await setEditorContent(page, `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel
status: invalid_status`);

    await waitForLinter(page);

    const errorRanges = page.locator('.cm-lintRange-error');
    await expect(errorRanges.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'sigma-error-status');
  });

  test('unknown field shows warning with amber tint', async ({ page }) => {
    await setEditorContent(page, VALID_RULE + '\nfoobar: baz');

    await waitForLinter(page);

    const warningRanges = page.locator('.cm-lintRange-warning');
    await expect(warningRanges.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'sigma-warning-unknown-field');
  });

  test('overlapping error and warning shows purple tint', async ({ page }) => {
    await setEditorContent(page, `title: Test Rule
detection:
    sel:
        Image: test
    condition: sel
foobar: baz`);

    await waitForLinter(page);

    const overlap = page.locator('.cm-lintRange-error.cm-lint-has-warning');
    await expect(overlap.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'sigma-overlap-error-warning');
  });

  test('tooltip appears on hover with rounded corners', async ({ page }) => {
    await setEditorContent(page, `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel
status: badvalue`);

    await waitForLinter(page);

    const errorRange = page.locator('.cm-lintRange-error').first();
    await errorRange.hover();
    await page.waitForTimeout(300);

    const tooltip = page.locator('.cm-tooltip-lint');
    await expect(tooltip).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'sigma-tooltip-hover');
  });

  test('suggestion panel shows for valid rule', async ({ page }) => {
    await waitForLinter(page);

    const panel = page.locator('text=Suggestions');
    await expect(panel).toBeVisible();
  });

  test('suggestion chip click shows preview decoration', async ({ page }) => {
    await waitForLinter(page);

    const chipVisible = await page.locator('span').filter({ hasText: 'process_creation' }).first().isVisible();
    if (chipVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__sigmaTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'logsource.category', value: 'process_creation', format: 'yaml' }) });
        }
      });
      await page.waitForTimeout(500);

      const preview = page.locator('.cm-suggestion-preview');
      await expect(preview).toBeVisible({ timeout: 5000 });

      await snapshot(page, SCREENSHOT_DIR, 'sigma-suggestion-preview');
    }
  });

  test('accept button inserts suggested text', async ({ page }) => {
    await waitForLinter(page);

    const chipVisible = await page.locator('span').filter({ hasText: 'process_creation' }).first().isVisible();
    if (chipVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__sigmaTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'logsource.category', value: 'process_creation', format: 'yaml' }) });
        }
      });
      await page.waitForTimeout(500);

      const acceptBtn = page.locator('.cm-suggestion-preview button', { hasText: 'Accept' });
      await expect(acceptBtn).toBeVisible({ timeout: 5000 });
      await acceptBtn.click();
      await page.waitForTimeout(200);

      const editorText = await page.locator('.cm-content').textContent();
      expect(editorText).toContain('process_creation');

      await snapshot(page, SCREENSHOT_DIR, 'sigma-suggestion-accepted');
    }
  });

  test('dismiss button clears preview without changes', async ({ page }) => {
    await waitForLinter(page);

    const editorTextBefore = await page.locator('.cm-content').textContent();

    const chipVisible = await page.locator('span').filter({ hasText: 'process_creation' }).first().isVisible();
    if (chipVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__sigmaTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'logsource.category', value: 'process_creation', format: 'yaml' }) });
        }
      });
      await page.waitForTimeout(500);

      const dismissBtn = page.locator('.cm-suggestion-preview button', { hasText: 'Dismiss' });
      await expect(dismissBtn).toBeVisible({ timeout: 5000 });
      await dismissBtn.click();
      await page.waitForTimeout(200);

      const preview = page.locator('.cm-suggestion-preview');
      await expect(preview).toHaveCount(0);

      const editorTextAfter = await page.locator('.cm-content').textContent();
      expect(editorTextAfter).toBe(editorTextBefore);
    }
  });

  test('parsed output updates with valid YAML', async ({ page }) => {
    await waitForLinter(page);

    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('End');
    await page.keyboard.type(' ', { delay: 5 });
    await page.waitForTimeout(200);

    const parsedOutput = page.locator('[data-testid="parsed-output"]');
    await expect(parsedOutput).toContainText('"title"', { timeout: 5000 });
    const text = await parsedOutput.textContent();
    expect(text).toContain('Okta User Account Locked Out');
  });

  test('captures screenshot of full editor with suggestions', async ({ page }) => {
    await waitForLinter(page);

    await snapshot(page, SCREENSHOT_DIR, 'sigma-editor-full');
  });
});
