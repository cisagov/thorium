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

async function clickSuggestionAdd(page: Page, field: string) {
  await page.locator(`span[title="Add '${field}' field"]`).click();
  await page.waitForTimeout(300);
}

async function clickSuggestionPopulate(page: Page, field: string) {
  await page.locator(`span[title="Populate '${field}' with default structure"]`).click();
  await page.waitForTimeout(300);
}

async function acceptPreview(page: Page) {
  const preview = page.locator('.cm-suggestion-preview');
  await expect(preview).toBeVisible({ timeout: 3000 });
  const acceptBtn = preview.locator('button:has-text("Accept")');
  await acceptBtn.click({ timeout: 5000 });
  await page.waitForTimeout(300);
}


async function getEditorText(page: Page): Promise<string> {
  return page.evaluate(() => {
    const container = document.querySelector('.cm-editor')?.parentElement as HTMLElement & {
      _cmView?: { state: { doc: { toString: () => string } } };
    };
    return container?._cmView?.state.doc.toString() ?? '';
  });
}

async function switchToJson(page: Page) {
  await page.locator('button:has-text("JSON")').click();
  await page.waitForTimeout(500);
  await waitForEditor(page);
}

async function switchToYaml(page: Page) {
  await page.locator('button:has-text("YAML")').click();
  await page.waitForTimeout(500);
  await waitForEditor(page);
}

test.describe('Image/Pipeline Editor — Suggestion Insertion', () => {
  test.beforeEach(async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    await setupMockAuth(page);
    await page.goto('/test/image-pipeline');

    try {
      await waitForEditor(page);
    } catch {
      if (consoleErrors.length > 0) {
        throw new Error(`Editor failed to load. Console errors:\n${consoleErrors.join('\n')}`);
      }
      throw new Error('Editor failed to load (no console errors captured)');
    }
  });

  test('YAML: add field under parent with inline empty {}', async ({ page }) => {
    const yaml = `group: analysis
name: test-tool
scaler: K8s
resources: {}
timeout: 300`;

    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'resources.cpu');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    await acceptPreview(page);

    const text = await getEditorText(page);
    expect(text).not.toContain('{}');
    expect(text).toContain('resources:\n');
    expect(text).toMatch(/resources:\n\s+cpu:/);
    expect(text).toContain('timeout: 300');

    await snapshot(page, SCREENSHOT_DIR, 'image-yaml-inline-obj-fix');
  });

  test('YAML: add nested field under parent with inline empty {}', async ({ page }) => {
    const yaml = `group: analysis
name: test-tool
scaler: K8s
dependencies: {}
timeout: 300`;

    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'dependencies.samples');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    await acceptPreview(page);

    const text = await getEditorText(page);
    expect(text).not.toContain('dependencies: {}');
    expect(text).toContain('dependencies:\n');
    expect(text).toMatch(/dependencies:\n\s+samples:/);

    await snapshot(page, SCREENSHOT_DIR, 'image-yaml-nested-inline-obj-fix');
  });

  test('YAML: populate null object field creates valid YAML', async ({ page }) => {
    const yaml = `group: analysis
name: test-tool
scaler: K8s
lifetime: null
timeout: 300`;

    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionPopulate(page, 'lifetime');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    // Fill required fields: counter (select) and amount (input)
    const counterSelect = preview.locator('select').first();
    await counterSelect.selectOption('jobs');
    const amountInput = preview.locator('input[inputmode="numeric"]').first();
    await amountInput.fill('32');
    await amountInput.press('Enter');
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    expect(text).not.toContain('lifetime: null');
    expect(text).toContain('lifetime:\n');
    expect(text).toMatch(/lifetime:\n\s+counter:/);

    await snapshot(page, SCREENSHOT_DIR, 'image-yaml-populate-null');
  });

  test('JSON: add field under parent with inline empty {}', async ({ page }) => {
    await switchToJson(page);

    const json = `{
  "group": "analysis",
  "name": "test-tool",
  "scaler": "K8s",
  "resources": {},
  "timeout": 300
}`;

    await setEditorContent(page, json);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'resources.cpu');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    await acceptPreview(page);

    const text = await getEditorText(page);
    expect(text).toContain('"cpu"');
    expect(text).toContain('"timeout"');

    const parsed = JSON.parse(text);
    expect(parsed.resources).toBeDefined();
    expect(typeof parsed.resources).toBe('object');
    expect(parsed.timeout).toBe(300);

    await snapshot(page, SCREENSHOT_DIR, 'image-json-inline-obj-fix');
  });

  test('JSON: populate null object field creates valid JSON', async ({ page }) => {
    await switchToJson(page);

    const json = `{
  "group": "analysis",
  "name": "test-tool",
  "scaler": "K8s",
  "lifetime": null,
  "timeout": 300
}`;

    await setEditorContent(page, json);
    await waitForLinter(page);

    await clickSuggestionPopulate(page, 'lifetime');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    // Fill required fields: counter (select) and amount (input)
    const counterSelect = preview.locator('select').first();
    await counterSelect.selectOption('jobs');
    const amountInput = preview.locator('input[inputmode="numeric"]').first();
    await amountInput.fill('32');
    await amountInput.press('Enter');
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    const parsed = JSON.parse(text);
    expect(parsed.lifetime).toBeDefined();
    expect(typeof parsed.lifetime).toBe('object');
    expect(parsed.lifetime).not.toBeNull();
    expect(parsed.timeout).toBe(300);

    await snapshot(page, SCREENSHOT_DIR, 'image-json-populate-null');
  });

  test('YAML: enum suggestion value under existing parent works', async ({ page }) => {
    const yaml = `group: analysis
name: test-tool
scaler: K8s
dependencies:
  samples:
    location: /tmp/thorium/samples`;

    await setEditorContent(page, yaml);
    await waitForLinter(page);

    // Find the strategy row by its field label title, then click the Paths chip
    const strategyLabel = page.locator('span[title="dependencies.samples.strategy"]');
    const strategyRow = strategyLabel.locator('xpath=ancestor::div[1]');
    await strategyRow.locator('span:has-text("Paths")').first().click();
    await page.waitForTimeout(300);

    await acceptPreview(page);

    const text = await getEditorText(page);
    expect(text).toContain('strategy: Paths');
    expect(text).toContain('location: /tmp/thorium/samples');

    await snapshot(page, SCREENSHOT_DIR, 'image-yaml-enum-suggestion');
  });

  test('JSON: enum suggestion value under existing parent works', async ({ page }) => {
    await switchToJson(page);

    const json = `{
  "group": "analysis",
  "name": "test-tool",
  "scaler": "K8s",
  "dependencies": {
    "samples": {
      "location": "/tmp/thorium/samples"
    }
  }
}`;

    await setEditorContent(page, json);
    await waitForLinter(page);

    const strategyLabel = page.locator('span[title="dependencies.samples.strategy"]');
    const strategyRow = strategyLabel.locator('xpath=ancestor::div[1]');
    await strategyRow.locator('span:has-text("Paths")').first().click();
    await page.waitForTimeout(300);

    await acceptPreview(page);

    const text = await getEditorText(page);
    const parsed = JSON.parse(text);
    expect(parsed.dependencies.samples.strategy).toBe('Paths');
    expect(parsed.dependencies.samples.location).toBe('/tmp/thorium/samples');

    await snapshot(page, SCREENSHOT_DIR, 'image-json-enum-suggestion');
  });

  test('format switching preserves data', async ({ page }) => {
    await waitForLinter(page);

    const yamlText = await getEditorText(page);
    expect(yamlText).toContain('group:');
    expect(yamlText).toContain('name:');

    await switchToJson(page);
    const jsonText = await getEditorText(page);
    expect(jsonText).toContain('"group"');

    await switchToYaml(page);
    const roundTripped = await getEditorText(page);
    expect(roundTripped).toContain('group:');
    expect(roundTripped).toContain('name:');

    await snapshot(page, SCREENSHOT_DIR, 'image-format-roundtrip');
  });
});

// ---------------------------------------------------------------------------
// Image Editor — Validation Diagnostics
// ---------------------------------------------------------------------------
test.describe('Image Editor — Validation Diagnostics', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/image-pipeline');
    await waitForEditor(page);
  });

  test('valid image config shows no error diagnostics', async ({ page }) => {
    await waitForLinter(page);
    const errors = page.locator('.cm-lintRange-error');
    await expect(errors).toHaveCount(0);
  });

  test('missing required field shows error diagnostic', async ({ page }) => {
    await setEditorContent(page, `name: test-tool\nscaler: K8s`);
    await waitForLinter(page);

    const errors = page.locator('.cm-lintRange-error');
    await expect(errors.first()).toBeVisible({ timeout: 5000 });
  });

  test('invalid enum value shows error diagnostic', async ({ page }) => {
    await setEditorContent(page, `group: test\nname: tool\nscaler: Docker`);
    await waitForLinter(page);

    const errors = page.locator('.cm-lintRange-error');
    await expect(errors.first()).toBeVisible({ timeout: 5000 });
  });

  test('unknown field shows warning diagnostic', async ({ page }) => {
    await setEditorContent(page, `group: test\nname: tool\nscaler: K8s\nfoobar: baz`);
    await waitForLinter(page);

    const warnings = page.locator('.cm-lintRange-warning');
    await expect(warnings.first()).toBeVisible({ timeout: 5000 });
  });

  test('type error shows error diagnostic', async ({ page }) => {
    await setEditorContent(page, `group: test\nname: tool\ntimeout: not-a-number`);
    await waitForLinter(page);

    const errors = page.locator('.cm-lintRange-error');
    await expect(errors.first()).toBeVisible({ timeout: 5000 });
  });
});

// ---------------------------------------------------------------------------
// Image Editor — Suggestion Actions
// ---------------------------------------------------------------------------
test.describe('Image Editor — Suggestion Actions', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/image-pipeline');
    await waitForEditor(page);
  });

  test('YAML: add simple number field produces valid output', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'timeout');
    await acceptPreview(page);

    const text = await getEditorText(page);
    expect(text).toContain('timeout:');
    expect(text).toContain('group: analysis');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('YAML: remove unknown field produces valid output', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s\nfoobar: baz\ntimeout: 300`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await page.locator(`span[title="Remove 'foobar' field and all subkeys"]`).click();
    await page.waitForTimeout(300);

    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });
    await preview.locator('button:has-text("Remove")').click();
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).not.toContain('foobar');
    expect(text).toContain('group: analysis');
    expect(text).toContain('timeout: 300');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('YAML: add list field via list widget produces valid output', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s\nargs:\n  output: Append`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'args.entrypoint');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const input = preview.locator('input').first();
    await input.fill('/bin/sh');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('entrypoint');
    expect(text).toContain('/bin/sh');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('YAML: add map entry produces valid output', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s\nenv: {}`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'env.VAR_NAME');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('DEBUG');
    await nameInput.press('Enter');
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('DEBUG');
    expect(text).not.toContain('env: {}');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('YAML: scaler Kvm shows kvm suggestion and inserts valid object', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: Kvm`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await page.locator(`span[title="Add 'kvm' field"]`).first().click();
    await page.waitForTimeout(300);
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const xmlInput = preview.locator('input').first();
    await xmlInput.fill('/path/to/vm.xml');
    const qcow2Input = preview.locator('input').nth(1);
    await qcow2Input.fill('/path/to/disk.qcow2');
    await qcow2Input.press('Enter');
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('kvm:');
    expect(text).toContain('/path/to/vm.xml');
    expect(text).toContain('/path/to/disk.qcow2');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('YAML: sequential add two fields produces valid output', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    // Add timeout
    await clickSuggestionAdd(page, 'timeout');
    await acceptPreview(page);

    let text = await getEditorText(page);
    expect(text).toContain('timeout:');

    await waitForLinter(page);

    // Add description
    await clickSuggestionAdd(page, 'description');
    await acceptPreview(page);

    text = await getEditorText(page);
    expect(text).toContain('timeout:');
    expect(text).toContain('description:');
    expect(text).toContain('group: analysis');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'image-yaml-sequential-add');
  });

  test('YAML: variant dropdown selects Kwarg form for args.output', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s\nargs:\n  reaction: my-react`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'args.output');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const variantSelect = preview.locator('select').first();
    await variantSelect.selectOption('Kwarg');
    await page.waitForTimeout(200);

    const input = preview.locator('input').first();
    await input.fill('--output');
    await input.press('Enter');
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('output');
    expect(text).toContain('Kwarg');
    expect(text).toContain('--output');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'image-yaml-variant-kwarg');
  });

  test('YAML: variant dropdown selects unit variant for args.output', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s\nargs:\n  reaction: my-react`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'args.output');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    // Select Append (unit variant — no inner value)
    const variantSelect = preview.locator('select').first();
    await variantSelect.selectOption('Append');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('output: Append');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('YAML: variant dropdown for spawn_limit with Basic form', async ({ page }) => {
    const yaml = `group: analysis\nname: test-tool\nscaler: K8s`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'spawn_limit');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    // Select Basic variant
    const variantSelect = preview.locator('select').first();
    await variantSelect.selectOption('Basic');
    await page.waitForTimeout(200);

    const input = preview.locator('input').first();
    await input.fill('10');
    await input.press('Enter');
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('spawn_limit');
    expect(text).toContain('Basic');
    expect(text).toContain('10');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'image-yaml-variant-spawn-limit');
  });

  test('JSON: variant dropdown selects Kwarg form for args.output', async ({ page }) => {
    await switchToJson(page);

    const json = `{\n  "group": "analysis",\n  "name": "test-tool",\n  "scaler": "K8s",\n  "args": {\n    "reaction": "my-react"\n  }\n}`;
    await setEditorContent(page, json);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'args.output');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const variantSelect = preview.locator('select').first();
    await variantSelect.selectOption('Kwarg');
    await page.waitForTimeout(200);

    const input = preview.locator('input').first();
    await input.fill('--output');
    await input.press('Enter');
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).toContain('"Kwarg"');
    expect(text).toContain('--output');

    await snapshot(page, SCREENSHOT_DIR, 'image-json-variant-kwarg');
  });
});

// ---------------------------------------------------------------------------
// Image Editor — JSON Validation and Suggestions
// ---------------------------------------------------------------------------
test.describe('Image Editor — JSON Mode', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/image-pipeline');
    await waitForEditor(page);
    await switchToJson(page);
  });

  test('valid JSON shows no error diagnostics', async ({ page }) => {
    const json = `{\n  "group": "analysis",\n  "name": "test-tool",\n  "scaler": "K8s",\n  "timeout": 300\n}`;
    await setEditorContent(page, json);
    await waitForLinter(page);

    const errors = page.locator('.cm-lintRange-error');
    await expect(errors).toHaveCount(0);
  });

  test('JSON: remove unknown field produces valid JSON', async ({ page }) => {
    const json = `{\n  "group": "analysis",\n  "name": "test-tool",\n  "scaler": "K8s",\n  "foobar": "baz",\n  "timeout": 300\n}`;
    await setEditorContent(page, json);
    await waitForLinter(page);

    await page.locator(`span[title="Remove 'foobar' field and all subkeys"]`).click();
    await page.waitForTimeout(300);

    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });
    await preview.locator('button:has-text("Remove")').click();
    await page.waitForTimeout(300);

    const text = await getEditorText(page);
    expect(text).not.toContain('foobar');
    const parsed = JSON.parse(text);
    expect(parsed.group).toBe('analysis');
    expect(parsed.timeout).toBe(300);
  });

  test('JSON: add simple field produces valid JSON', async ({ page }) => {
    const json = `{\n  "group": "analysis",\n  "name": "test-tool",\n  "scaler": "K8s"\n}`;
    await setEditorContent(page, json);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'timeout');
    await acceptPreview(page);

    const text = await getEditorText(page);
    const parsed = JSON.parse(text);
    expect(parsed.group).toBe('analysis');
    expect(parsed).toHaveProperty('timeout');
  });

  test('JSON: sequential add two fields produces valid JSON', async ({ page }) => {
    const json = `{\n  "group": "analysis",\n  "name": "test-tool",\n  "scaler": "K8s"\n}`;
    await setEditorContent(page, json);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'timeout');
    await acceptPreview(page);

    let text = await getEditorText(page);
    let parsed = JSON.parse(text);
    expect(parsed).toHaveProperty('timeout');

    await waitForLinter(page);

    await clickSuggestionAdd(page, 'description');
    await acceptPreview(page);

    text = await getEditorText(page);
    parsed = JSON.parse(text);
    expect(parsed).toHaveProperty('timeout');
    expect(parsed).toHaveProperty('description');
    expect(parsed.group).toBe('analysis');
  });
});

// ---------------------------------------------------------------------------
// Pipeline Editor
// ---------------------------------------------------------------------------
test.describe('Pipeline Editor', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/image-pipeline');
    await waitForEditor(page);
    await page.locator('button:has-text("Pipeline")').click();
    await page.waitForTimeout(500);
    await waitForEditor(page);
  });

  test('valid pipeline shows no error diagnostics', async ({ page }) => {
    await waitForLinter(page);
    const errors = page.locator('.cm-lintRange-error');
    await expect(errors).toHaveCount(0);
  });

  test('missing required order field shows error', async ({ page }) => {
    await setEditorContent(page, `group: analysis\nname: my-pipeline`);
    await waitForLinter(page);

    const errors = page.locator('.cm-lintRange-error');
    await expect(errors.first()).toBeVisible({ timeout: 5000 });
  });

  test('unknown field shows warning', async ({ page }) => {
    await setEditorContent(page, `group: analysis\nname: test\norder:\n  - img1\nfoobar: baz`);
    await waitForLinter(page);

    const warnings = page.locator('.cm-lintRange-warning');
    await expect(warnings.first()).toBeVisible({ timeout: 5000 });
  });

  test('add missing sla field produces valid YAML', async ({ page }) => {
    await setEditorContent(page, `group: analysis\nname: test\norder:\n  - img1`);
    await waitForLinter(page);

    await clickSuggestionAdd(page, 'sla');
    await acceptPreview(page);

    const text = await getEditorText(page);
    expect(text).toContain('sla:');
    expect(text).toContain('group: analysis');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);
  });

  test('parsed output updates on valid edit', async ({ page }) => {
    // Trigger a change to force parsed output update
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('End');
    await page.keyboard.type(' ', { delay: 5 });
    await page.waitForTimeout(300);

    const parsedOutput = page.locator('[data-testid="parsed-output"]');
    await expect(parsedOutput).toContainText('"group"', { timeout: 5000 });
    await expect(parsedOutput).toContainText('triage');
  });

  test('add Tag trigger with select and key-value rows produces valid YAML', async ({ page }) => {
    await setEditorContent(page, `group: analysis\nname: test\norder:\n  - img1\ntriggers: {}`);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionPopulate(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    // Fill trigger name
    const nameInput = preview.locator('input').first();
    await nameInput.fill('my-trigger');
    await page.waitForTimeout(200);

    // Select Tag variant (first select is variant type)
    const typeSelect = preview.locator('select').first();
    await typeSelect.selectOption('Tag');
    await page.waitForTimeout(300);

    // Select Files for tag_types (second select)
    const tagTypeSelect = preview.locator('select').nth(1);
    await tagTypeSelect.selectOption('Files');
    await page.waitForTimeout(200);

    // Fill a key-value pair in the required section
    const keyInputs = preview.locator('input[type="text"][placeholder="tag-key"]');
    await keyInputs.first().fill('malware');
    await page.waitForTimeout(200);
    const valInputs = preview.locator('input[type="text"][placeholder="value"]');
    await valInputs.first().fill('trojan');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    expect(text).toContain('my-trigger');
    expect(text).toContain('Tag');
    expect(text).toContain('tag_types');
    expect(text).toContain('Files');
    expect(text).toContain('required');
    expect(text).toContain('malware');
    expect(text).toContain('trojan');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-tag-trigger-full');
  });

  test('JSON: populate triggers inserts inside existing braces', async ({ page }) => {
    await switchToJson(page);

    const json = `{
  "group": "analysis",
  "name": "test",
  "order": ["img1"],
  "triggers": {}
}`;

    await setEditorContent(page, json);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionPopulate(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('my-trigger');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    expect(text).toContain('"triggers"');
    expect(text).toContain('"my-trigger"');

    const parsed = JSON.parse(text);
    expect(parsed.triggers).toBeDefined();
    expect(parsed.triggers['my-trigger']).toBeDefined();
    expect(parsed.order).toEqual(['img1']);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-json-trigger-populate');
  });

  test('Tag trigger shows subtitles and not section has initial row', async ({ page }) => {
    await setEditorContent(page, `group: analysis\nname: test\norder:\n  - img1\ntriggers: {}`);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionPopulate(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('my-trigger');
    await page.waitForTimeout(200);

    const typeSelect = preview.locator('select').first();
    await typeSelect.selectOption('Tag');
    await page.waitForTimeout(300);

    await expect(preview.locator('text=required (must have):')).toBeVisible();
    await expect(preview.locator('text=not (must not have):')).toBeVisible();

    const notKeyInputs = preview.locator('input[type="text"][placeholder="tag-key"]');
    const count = await notKeyInputs.count();
    expect(count).toBeGreaterThanOrEqual(2);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-tag-trigger-subtitles');
  });
});

// ---------------------------------------------------------------------------
// Pipeline Editor — Trigger Workflows
// ---------------------------------------------------------------------------
test.describe('Pipeline Editor — Trigger Workflows', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await page.goto('/test/image-pipeline');
    await waitForEditor(page);
    await page.locator('button:has-text("Pipeline")').click();
    await page.waitForTimeout(500);
    await waitForEditor(page);
  });

  // --- Single-action tests ---

  test('YAML: populate empty triggers with NewSample', async ({ page }) => {
    const yaml = `group: analysis\nname: test\norder:\n  - img1\ntriggers: {}`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionPopulate(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('ns-trigger');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    expect(text).toContain('ns-trigger');
    expect(text).toContain('NewSample');
    expect(text).not.toMatch(/triggers:\s*\{\}/);

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-trigger-yaml-populate-newsample');
  });

  test('JSON: populate empty triggers with NewSample', async ({ page }) => {
    await switchToJson(page);

    const json = `{
  "group": "analysis",
  "name": "test",
  "order": ["img1"],
  "triggers": {}
}`;

    await setEditorContent(page, json);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionPopulate(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('ns-trigger');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    const parsed = JSON.parse(text);
    expect(parsed.triggers['ns-trigger']).toBe('NewSample');
    expect(parsed.triggers).toBeDefined();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-trigger-json-populate-newsample');
  });

  test('YAML: add trigger to existing NewSample', async ({ page }) => {
    const yaml = `group: analysis\nname: test\norder:\n  - img1\ntriggers:\n  t1: NewSample`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionAdd(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('t2');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    expect(text).toContain('t1');
    expect(text).toContain('t2');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-trigger-yaml-add-to-existing');
  });

  test('JSON: add trigger to existing NewSample', async ({ page }) => {
    await switchToJson(page);

    const json = `{
  "group": "analysis",
  "name": "test",
  "order": ["img1"],
  "triggers": {
    "t1": "NewSample"
  }
}`;

    await setEditorContent(page, json);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionAdd(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('t2');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    const parsed = JSON.parse(text);
    expect(parsed.triggers.t1).toBeDefined();
    expect(parsed.triggers.t2).toBeDefined();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-trigger-json-add-to-existing');
  });

  // --- Multi-step workflow tests ---

  test('YAML: W1 — populate NewSample then add Tag', async ({ page }) => {
    const yaml = `group: analysis\nname: test\norder:\n  - img1\ntriggers: {}`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    // Step 1: Populate with NewSample
    await clickSuggestionPopulate(page, 'triggers.trigger-name');
    let preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput1 = preview.locator('input').first();
    await nameInput1.fill('first');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    await waitForLinter(page);

    // Step 2: Add another trigger with Tag variant
    await clickSuggestionAdd(page, 'triggers.trigger-name');
    preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput2 = preview.locator('input').first();
    await nameInput2.fill('second');
    await page.waitForTimeout(200);

    const typeSelect = preview.locator('select').first();
    await typeSelect.selectOption('Tag');
    await page.waitForTimeout(300);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    expect(text).toContain('first');
    expect(text).toContain('second');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-trigger-yaml-w1-populate-then-add');
  });

  test('JSON: W1 — populate NewSample then add Tag', async ({ page }) => {
    await switchToJson(page);

    const json = `{
  "group": "analysis",
  "name": "test",
  "order": ["img1"],
  "triggers": {}
}`;

    await setEditorContent(page, json);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    // Step 1: Populate with NewSample
    await clickSuggestionPopulate(page, 'triggers.trigger-name');
    let preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput1 = preview.locator('input').first();
    await nameInput1.fill('first');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    await waitForLinter(page);

    // Step 2: Add another trigger with Tag variant
    await clickSuggestionAdd(page, 'triggers.trigger-name');
    preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput2 = preview.locator('input').first();
    await nameInput2.fill('second');
    await page.waitForTimeout(200);

    const typeSelect = preview.locator('select').first();
    await typeSelect.selectOption('Tag');
    await page.waitForTimeout(300);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    const parsed = JSON.parse(text);
    expect(parsed.triggers.first).toBeDefined();
    expect(parsed.triggers.second).toBeDefined();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-trigger-json-w1-populate-then-add');
  });

  test('YAML: add trigger when triggers absent', async ({ page }) => {
    const yaml = `group: analysis\nname: test\norder:\n  - img1`;
    await setEditorContent(page, yaml);
    await waitForLinter(page);
    await page.waitForTimeout(300);

    await clickSuggestionAdd(page, 'triggers.trigger-name');
    const preview = page.locator('.cm-suggestion-preview');
    await expect(preview).toBeVisible({ timeout: 3000 });

    const nameInput = preview.locator('input').first();
    await nameInput.fill('new-trig');
    await page.waitForTimeout(200);

    await preview.locator('button:has-text("Accept")').click({ timeout: 5000 });
    await page.waitForTimeout(500);

    const text = await getEditorText(page);
    expect(text).toContain('triggers');
    expect(text).toContain('new-trig');

    const { parseDocument } = await import('yaml');
    const doc = parseDocument(text);
    expect(doc.errors).toHaveLength(0);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-trigger-yaml-add-when-absent');
  });
});
