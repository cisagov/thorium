import { test, expect, Page } from '@playwright/test';
import path from 'path';
import {
  snapshot,
  setupMockAuth,
  waitForEditor,
  waitForLinter,
  setEditorContent,
} from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

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

  test('unknown field shows info diagnostic', async ({ page }) => {
    await setEditorContent(page, VALID_RULE + '\nfoobar: baz');

    await waitForLinter(page);

    const infoRanges = page.locator('.cm-lintRange-info');
    await expect(infoRanges.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'sigma-info-unknown-field');
  });

  test('overlapping error and info shows combined tint', async ({ page }) => {
    await setEditorContent(page, `title: Test Rule
detection:
    sel:
        Image: test
    condition: sel
foobar: baz`);

    await waitForLinter(page);

    const overlap = page.locator('.cm-lintRange-error.cm-lint-has-info');
    await expect(overlap.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'sigma-overlap-error-info');
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

// ---------------------------------------------------------------------------
// Sigma — Suggestion Click Actions
// ---------------------------------------------------------------------------
test.describe('Sigma — Suggestion Click Actions', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/sigma');
    await waitForEditor(page);
  });

  test('click logsource.category chip inserts valid YAML', async ({ page }) => {
    const rule = `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel`;

    await setEditorContent(page, rule);
    await waitForLinter(page);

    const categoryLabel = page.locator('span[title="logsource.category"]');
    const categoryRow = categoryLabel.locator('xpath=ancestor::div[1]');
    await categoryRow.locator('span[title*="process_creation"]').click();
    await page.waitForTimeout(300);

    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });
    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('process_creation');
    expect(text).toContain('product: windows');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'sigma-category-chip-click');
  });

  test('click status value chip inserts valid YAML', async ({ page }) => {
    const rule = `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel`;

    await setEditorContent(page, rule);
    await waitForLinter(page);

    const statusLabel = page.locator('span[title="status"]');
    const statusRow = statusLabel.locator('xpath=ancestor::div[1]');
    await statusRow.locator('span[title*="test"]').click();
    await page.waitForTimeout(300);

    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });
    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('status');
    expect(text).toContain('title: Test Rule');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('add missing field via Add button inserts valid YAML', async ({ page }) => {
    const rule = `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel`;

    await setEditorContent(page, rule);
    await waitForLinter(page);

    await page.locator(`span[title="Add 'id' field"]`).click();
    await page.waitForTimeout(300);

    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const input = preview.locator('input').first();
    await input.fill('929a690e-bef0-4204-a928-ef5e620d6fcc');
    await input.press('Enter');
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('id:');
    expect(text).toContain('929a690e');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('add list field via Add button inserts valid YAML', async ({ page }) => {
    const rule = `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel`;

    await setEditorContent(page, rule);
    await waitForLinter(page);

    await page.locator(`span[title="Add 'tags' field"]`).click();
    await page.waitForTimeout(300);

    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const input = preview.locator('input').first();
    await input.fill('attack.t1059');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('tags');
    expect(text).toContain('attack.t1059');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('click logsource.product chip inserts valid YAML', async ({ page }) => {
    const rule = `title: Test Rule
logsource:
    category: process_creation
detection:
    sel:
        Image: test
    condition: sel`;

    await setEditorContent(page, rule);
    await waitForLinter(page);

    const productLabel = page.locator('span[title="logsource.product"]');
    const productRow = productLabel.locator('xpath=ancestor::div[1]');
    await productRow.locator('span[title*="windows"]').click();
    await page.waitForTimeout(300);

    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });
    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('windows');
    expect(text).toContain('category: process_creation');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('sequential: add status then level, both present and valid YAML', async ({ page }) => {
    const rule = `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel`;

    await setEditorContent(page, rule);
    await waitForLinter(page);

    // Add status
    const statusLabel = page.locator('span[title="status"]');
    const statusRow = statusLabel.locator('xpath=ancestor::div[1]');
    await statusRow.locator('span[title*="test"]').click();
    await page.waitForTimeout(300);

    let preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });
    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    let text = await getEditorText(page);
    expect(text).toContain('status');

    await waitForLinter(page);

    // Add level
    const levelLabel = page.locator('span[title="level"]');
    const levelRow = levelLabel.locator('xpath=ancestor::div[1]');
    await levelRow.locator('span[title*="medium"]').click();
    await page.waitForTimeout(300);

    preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });
    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    text = await getEditorText(page);
    expect(text).toContain('status');
    expect(text).toContain('level');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'sigma-sequential-suggestions');
  });
});

// ---------------------------------------------------------------------------
// Sigma — Detection Diagnostics
// ---------------------------------------------------------------------------
test.describe('Sigma — Detection Diagnostics', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/sigma');
    await waitForEditor(page);
  });

  test('condition referencing undefined identifier shows warning', async ({ page }) => {
    const rule = `title: Test Rule
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: undefined_identifier`;

    await setEditorContent(page, rule);
    await waitForLinter(page);

    const warnings = page.locator('.cm-lintRange-warning');
    await expect(warnings.first()).toBeVisible({ timeout: 5000 });
  });
});

async function getEditorText(page: Page): Promise<string> {
  return page.evaluate(() => {
    const container = document.querySelector('.cm-editor')?.parentElement as HTMLElement & {
      _cmView?: { state: { doc: { toString: () => string } } };
    };
    return container?._cmView?.state.doc.toString() ?? '';
  });
}
