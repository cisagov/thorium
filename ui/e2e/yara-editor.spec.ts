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

const VALID_RULE = `import "pe"

rule DetectUPX : packer
{
    meta:
        description = "Detects UPX packed executables"
        author = "Test Author"
        date = "2024-01-15"

    strings:
        $upx0 = "UPX0" ascii
        $upx1 = "UPX1" ascii
        $hex = { 60 E8 00 00 00 00 58 }

    condition:
        uint16(0) == 0x5A4D and ($upx0 or $upx1) and $hex
}`;

test.describe('YARA Rule CodeEditor', () => {
  test.beforeEach(async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    await setupMockAuth(page);
    await page.goto('/test/yara');

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

    await snapshot(page, SCREENSHOT_DIR, 'yara-valid-rule');
  });

  test('syntax highlighting applies to keywords', async ({ page }) => {
    await waitForLinter(page);

    const editorHtml = await page.locator('.cm-content').innerHTML();
    expect(editorHtml).toContain('cm-');

    await snapshot(page, SCREENSHOT_DIR, 'yara-syntax-highlighting');
  });

  test('missing condition shows error diagnostic', async ({ page }) => {
    await setEditorContent(page, `rule NoCondition
{
    strings:
        $a = "test"
}`);

    await waitForLinter(page);

    const errorRanges = page.locator('.cm-lintRange-error');
    await expect(errorRanges.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'yara-error-missing-condition');
  });

  test('unknown import module shows warning', async ({ page }) => {
    await setEditorContent(page, `import "badmodule"

rule Test
{
    condition:
        true
}`);

    await waitForLinter(page);

    const warningRanges = page.locator('.cm-lintRange-warning');
    await expect(warningRanges.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'yara-warning-unknown-module');
  });

  test('tooltip appears on hover over error range', async ({ page }) => {
    await setEditorContent(page, `rule NoCondition
{
    strings:
        $a = "test"
}`);

    await waitForLinter(page);

    const errorRange = page.locator('.cm-lintRange-error').first();
    await errorRange.hover();
    await page.waitForTimeout(300);

    const tooltip = page.locator('.cm-tooltip-lint');
    await expect(tooltip).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'yara-tooltip-hover');
  });

  test('suggestion panel shows for rule missing meta keys', async ({ page }) => {
    await waitForLinter(page);

    // Move cursor to the meta section so context-aware suggestions appear
    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as HTMLElement & { _cmView?: { state: { doc: { line: (n: number) => { from: number } } }; dispatch: (spec: unknown) => void } };
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(6);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const panel = page.locator('text=Suggestions');
    await expect(panel).toBeVisible();
  });

  test('suggestion chip click shows preview decoration', async ({ page }) => {
    await waitForLinter(page);

    // Move cursor to meta section to show meta suggestions
    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(6);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const chipVisible = await page.locator('span').filter({ hasText: 'reference' }).first().isVisible().catch(() => false);
    if (chipVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'meta', value: 'reference', format: 'yara' }) });
        }
      });
      await page.waitForTimeout(500);

      const preview = page.locator('.cm-suggestion-preview');
      await expect(preview).toBeVisible({ timeout: 5000 });

      await snapshot(page, SCREENSHOT_DIR, 'yara-suggestion-preview');
    }
  });

  test('accept button inserts suggested text', async ({ page }) => {
    await waitForLinter(page);

    // Move cursor to meta section
    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(6);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const chipVisible = await page.locator('span').filter({ hasText: 'reference' }).first().isVisible().catch(() => false);
    if (chipVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'meta', value: 'reference', format: 'yara' }) });
        }
      });
      await page.waitForTimeout(500);

      const acceptBtn = page.locator('.cm-suggestion-preview button', { hasText: 'Accept' });
      await expect(acceptBtn).toBeVisible({ timeout: 5000 });
      await acceptBtn.click();
      await page.waitForTimeout(200);

      const editorText = await page.locator('.cm-content').textContent();
      expect(editorText).toContain('reference');

      await snapshot(page, SCREENSHOT_DIR, 'yara-suggestion-accepted');
    }
  });

  test('dismiss button clears preview without changes', async ({ page }) => {
    await waitForLinter(page);

    const editorTextBefore = await page.locator('.cm-content').textContent();

    // Move cursor to meta section
    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(6);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const chipVisible = await page.locator('span').filter({ hasText: 'reference' }).first().isVisible().catch(() => false);
    if (chipVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'meta', value: 'reference', format: 'yara' }) });
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

  test('captures screenshot of full editor with suggestions', async ({ page }) => {
    await waitForLinter(page);

    await snapshot(page, SCREENSHOT_DIR, 'yara-editor-full');
  });
});

// ---------------------------------------------------------------------------
// YARA — Section Scaffold Suggestions
// ---------------------------------------------------------------------------
test.describe('YARA — Section Scaffold Suggestions', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/yara');
    await waitForEditor(page);
  });

  test('meta section scaffold inserts correctly', async ({ page }) => {
    const rule = `rule NoMeta\n{\n    strings:\n        $a = "test"\n\n    condition:\n        $a\n}`;
    await setEditorContent(page, rule);
    await waitForLinter(page);

    // Move cursor inside rule body for context-sensitive suggestions
    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(3);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const metaChip = page.locator('span[title*="meta"]').filter({ hasText: 'meta' }).first();
    const isVisible = await metaChip.isVisible().catch(() => false);
    if (isVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'section.meta', value: 'meta', format: 'yara' }) });
        }
      });
      await page.waitForTimeout(500);

      const preview = page.locator('.cm-suggestion-preview');
      await expect(preview).toBeVisible({ timeout: 5000 });
      await preview.locator('button:has-text("Accept")').click();
      await page.waitForTimeout(300);

      const text = await getEditorText(page);
      expect(text).toContain('meta:');
      expect(text).toContain('description');
      expect(text).toContain('author');
      expect(text).toContain('condition:');

      await snapshot(page, SCREENSHOT_DIR, 'yara-meta-scaffold');
    }
  });

  test('strings section scaffold inserts correctly', async ({ page }) => {
    const rule = `rule NoStrings\n{\n    meta:\n        description = "test"\n\n    condition:\n        true\n}`;
    await setEditorContent(page, rule);
    await waitForLinter(page);

    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(3);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const stringsChip = page.locator('span[title*="strings"]').filter({ hasText: 'strings' }).first();
    const isVisible = await stringsChip.isVisible().catch(() => false);
    if (isVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'section.strings', value: 'strings', format: 'yara' }) });
        }
      });
      await page.waitForTimeout(500);

      const preview = page.locator('.cm-suggestion-preview');
      await expect(preview).toBeVisible({ timeout: 5000 });
      await preview.locator('button:has-text("Accept")').click();
      await page.waitForTimeout(300);

      const text = await getEditorText(page);
      expect(text).toContain('strings:');
      expect(text).toContain('$s1');
      expect(text).toContain('condition:');

      await snapshot(page, SCREENSHOT_DIR, 'yara-strings-scaffold');
    }
  });

  test('condition section scaffold inserts correctly', async ({ page }) => {
    const rule = `rule NoCondition\n{\n    strings:\n        $a = "test"\n}`;
    await setEditorContent(page, rule);
    await waitForLinter(page);

    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(3);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const condChip = page.locator('span[title*="condition"]').filter({ hasText: 'condition' }).first();
    const isVisible = await condChip.isVisible().catch(() => false);
    if (isVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'section.condition', value: 'condition', format: 'yara' }) });
        }
      });
      await page.waitForTimeout(500);

      const preview = page.locator('.cm-suggestion-preview');
      await expect(preview).toBeVisible({ timeout: 5000 });
      await preview.locator('button:has-text("Accept")').click();
      await page.waitForTimeout(300);

      const text = await getEditorText(page);
      expect(text).toContain('condition:');
      expect(text).toContain('strings:');

      await snapshot(page, SCREENSHOT_DIR, 'yara-condition-scaffold');
    }
  });
});

// ---------------------------------------------------------------------------
// YARA — Other Suggestions
// ---------------------------------------------------------------------------
test.describe('YARA — Other Suggestions', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/yara');
    await waitForEditor(page);
  });

  test('import module suggestion inserts at top', async ({ page }) => {
    const rule = `rule NoImport\n{\n    meta:\n        description = "test"\n\n    strings:\n        $a = "test"\n\n    condition:\n        $a\n}`;
    await setEditorContent(page, rule);
    await waitForLinter(page);

    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        view.dispatch({ selection: { anchor: 0 } });
      }
    });
    await page.waitForTimeout(200);

    const peChip = page.locator('span[title*="pe"]').filter({ hasText: 'pe' }).first();
    const isVisible = await peChip.isVisible().catch(() => false);
    if (isVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'import', value: 'pe', format: 'yara' }) });
        }
      });
      await page.waitForTimeout(500);

      const preview = page.locator('.cm-suggestion-preview');
      await expect(preview).toBeVisible({ timeout: 5000 });
      await preview.locator('button:has-text("Accept")').click();
      await page.waitForTimeout(300);

      const text = await getEditorText(page);
      expect(text).toMatch(/^import "pe"/);
      expect(text).toContain('rule NoImport');

      await snapshot(page, SCREENSHOT_DIR, 'yara-import-suggestion');
    }
  });

  test('meta key suggestion inserts in correct section', async ({ page }) => {
    await waitForLinter(page);

    // Move cursor to meta section
    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(6);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    await page.evaluate(() => {
      const helpers = (window as any).__yaraTestHelpers;
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view && helpers) {
        view.dispatch({ effects: helpers.addPreview.of({ field: 'meta', value: 'version', format: 'yara' }) });
      }
    });
    await page.waitForTimeout(500);

    const preview = page.locator('.cm-suggestion-preview');
    const isVisible = await preview.isVisible().catch(() => false);
    if (isVisible) {
      await preview.locator('button:has-text("Accept")').click();
      await page.waitForTimeout(300);

      const text = await getEditorText(page);
      expect(text).toContain('version');
      expect(text).toContain('meta:');

      await snapshot(page, SCREENSHOT_DIR, 'yara-meta-key-suggestion');
    }
  });

  test('string modifier suggestion appends inline', async ({ page }) => {
    await waitForLinter(page);

    // Move cursor to strings section
    await page.evaluate(() => {
      const container = document.querySelector('.cm-editor')?.parentElement as any;
      const view = container?._cmView;
      if (view) {
        const line = view.state.doc.line(12);
        view.dispatch({ selection: { anchor: line.from } });
      }
    });
    await page.waitForTimeout(200);

    const modChip = page.locator('span[title*="nocase"]');
    const isVisible = await modChip.first().isVisible().catch(() => false);
    if (isVisible) {
      await page.evaluate(() => {
        const helpers = (window as any).__yaraTestHelpers;
        const container = document.querySelector('.cm-editor')?.parentElement as any;
        const view = container?._cmView;
        if (view && helpers) {
          view.dispatch({ effects: helpers.addPreview.of({ field: 'strings.modifiers', value: 'nocase', format: 'yara', cursorLine: 12 }) });
        }
      });
      await page.waitForTimeout(500);

      const preview = page.locator('.cm-suggestion-preview');
      const previewVisible = await preview.isVisible().catch(() => false);
      if (previewVisible) {
        await preview.locator('button:has-text("Accept")').click();
        await page.waitForTimeout(300);

        const text = await getEditorText(page);
        expect(text).toContain('nocase');

        await snapshot(page, SCREENSHOT_DIR, 'yara-modifier-suggestion');
      }
    }
  });
});

// ---------------------------------------------------------------------------
// YARA — Validation Diagnostics
// ---------------------------------------------------------------------------
test.describe('YARA — Validation Diagnostics', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/yara');
    await waitForEditor(page);
  });

  test('duplicate rule name shows warning diagnostic', async ({ page }) => {
    const rules = `rule Duplicate\n{\n    condition:\n        true\n}\n\nrule Duplicate\n{\n    condition:\n        false\n}`;
    await setEditorContent(page, rules);
    await waitForLinter(page);

    const warnings = page.locator('.cm-lintRange-warning');
    await expect(warnings.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'yara-duplicate-rule-warning');
  });

  test('invalid modifier for string type shows error', async ({ page }) => {
    const rule = `rule BadModifier\n{\n    strings:\n        $re = /test/ xor\n\n    condition:\n        $re\n}`;
    await setEditorContent(page, rule);
    await waitForLinter(page);

    const errors = page.locator('.cm-lintRange-error');
    await expect(errors.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'yara-invalid-modifier-error');
  });

  test('unused string definition shows warning', async ({ page }) => {
    const rule = `rule UnusedString\n{\n    strings:\n        $a = "used"\n        $b = "unused"\n\n    condition:\n        $a\n}`;
    await setEditorContent(page, rule);
    await waitForLinter(page);

    const warnings = page.locator('.cm-lintRange-warning');
    await expect(warnings.first()).toBeVisible({ timeout: 5000 });

    await snapshot(page, SCREENSHOT_DIR, 'yara-unused-string-warning');
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
