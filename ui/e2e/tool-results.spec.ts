import { test, expect } from '@playwright/test';
import path from 'path';
import {
  authenticate,
  createEntity,
  createAssociation,
  deleteEntity,
  uploadFile,
  uploadResults,
  snapshot,
  loginViaUI,
  TEST_USER,
  TEST_PASS,
} from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

/**
 * Covers the file-details tabs that can be seeded without an analysis agent:
 *  - the Results tab renders (empty state when no tool results exist), and
 *  - the new top-level Entities tab lists entities associated with the file.
 *
 * The per-tool-result tile (Result/Files/Children/Entities tabs, download, render, diff) requires
 * agent-produced results and is exercised manually / by future agent-backed fixtures.
 */
test.describe('File Details — Results & Entities tabs', () => {
  let token: string;
  let deviceId: string;
  let fileSha256: string;
  const deviceName = 'PlaywrightResultsDevice';

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);

    deviceId = await createEntity(token, deviceName, 'Device', ['system']);

    const file = await uploadFile(token, Buffer.from('tool-results e2e file content'), 'tool-results-e2e.bin', ['system']);
    fileSha256 = file.sha256;

    // associate the file with the device so the Entities tab has something to list
    await createAssociation(token, {
      kind: 'FirmwareFor',
      source: { File: fileSha256 },
      targets: [{ Entity: { id: deviceId, name: deviceName } }],
      groups: ['system'],
      is_bidirectional: false,
    });
  });

  test.afterAll(async () => {
    await deleteEntity(token, deviceId).catch(() => {});
  });

  test('file details page shows the Results and Entities tabs', async ({ page }) => {
    await loginViaUI(page);
    await page.goto(`/file/${fileSha256}`);
    await page.waitForLoadState('networkidle');

    // both top-level tabs are present (react-bootstrap Tab.Container renders Nav.Links as role=tab)
    await expect(page.getByRole('tab', { name: 'Results', exact: true })).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Entities', exact: true })).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'file-details-results-tab');
  });

  test('Entities tab lists the associated entity with a link to its details', async ({ page }) => {
    await loginViaUI(page);
    await page.goto(`/file/${fileSha256}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('tab', { name: 'Entities', exact: true }).click();

    // the associated device should appear, linking to its details page, with the association kind
    const entityLink = page.getByRole('link', { name: new RegExp(deviceName) });
    await expect(entityLink).toBeVisible({ timeout: 15000 });
    await expect(entityLink).toHaveAttribute('href', new RegExp(`/device/${deviceId}`));
    await expect(page.getByText('FirmwareFor')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'file-details-entities-tab');
  });
});

/**
 * Exercises the redesigned per-tool-result tile by seeding real results (two versions, each with a
 * JSON and a binary result file) via the results API, then driving the tabs, deferred file render,
 * version selector, and diff window in the browser.
 */
test.describe('Tool Result Tile', () => {
  let token: string;
  let fileSha256: string;
  const tool = 'pwresulttool';
  const jsonFileContent = Buffer.from(JSON.stringify({ seeded: 'render-value', nested: { ok: true } }));
  // a binary buffer with a NUL byte so the renderer classifies it as hex
  const binFileContent = Buffer.from([0xde, 0xad, 0x00, 0xbe, 0xef, 0x01, 0x02, 0x03]);
  const mdFileContent = Buffer.from('# Seeded Heading\n\nSome **bold** text.\n');

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    // unique content so each run gets a fresh sha256 (results accumulate per-file otherwise)
    const file = await uploadFile(token, Buffer.from(`tool-result tile e2e content ${Date.now()}`), 'tile-e2e.bin', ['system']);
    fileSha256 = file.sha256;

    const files = [
      { name: 'report.json', content: jsonFileContent, contentType: 'application/json' },
      { name: 'dump.bin', content: binFileContent, contentType: 'application/octet-stream' },
      { name: 'report.md', content: mdFileContent, contentType: 'text/markdown' },
    ];
    // two versions so the version selector + diff button appear; both carry the result files
    await uploadResults(token, fileSha256, {
      tool,
      result: JSON.stringify({ summary: 'first-version', score: 1 }),
      displayType: 'Json',
      groups: ['system'],
      version: '1.0.0',
      files,
    });
    await uploadResults(token, fileSha256, {
      tool,
      result: JSON.stringify({ summary: 'second-version', score: 2 }),
      displayType: 'Json',
      groups: ['system'],
      version: '2.0.0',
      files,
    });
  });

  // open the file page on the Results tab with the seeded tile present
  async function openResults(page: import('@playwright/test').Page) {
    await loginViaUI(page);
    await page.goto(`/file/${fileSha256}`);
    await page.waitForLoadState('networkidle');
    await expect(page.getByText(tool, { exact: true }).first()).toBeVisible({ timeout: 20000 });
  }

  test('renders the tile header, Result tab content, and tab bar', async ({ page }) => {
    await openResults(page);
    // Result tab is active by default and shows the structured result (latest version = second)
    await expect(page.getByText('second-version')).toBeVisible();
    // tile tab bar includes Result and Files (with a count badge)
    await expect(page.getByRole('tab', { name: 'Result', exact: true })).toBeVisible();
    await expect(page.getByRole('tab', { name: /Files/ })).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'tool-result-tile');
  });

  test('Files tab renders a JSON file in an overlay window and offers download', async ({ page }) => {
    await openResults(page);
    await page.getByRole('tab', { name: /Files/ }).click();

    // the seeded files are listed with render + download actions
    await expect(page.getByText('report.json')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Download report.json' })).toBeVisible();

    // clicking the eye downloads + renders the JSON in a floating window
    await page.getByRole('button', { name: 'View report.json' }).click();
    await expect(page.getByText('render-value')).toBeVisible({ timeout: 15000 });
    await snapshot(page, SCREENSHOT_DIR, 'tool-result-render-json');
  });

  test('Files tab renders a binary file as a hex dump', async ({ page }) => {
    await openResults(page);
    await page.getByRole('tab', { name: /Files/ }).click();

    await page.getByRole('button', { name: 'View dump.bin' }).click();
    // the hex view shows an offset gutter and the value-inspector prompt
    await expect(page.getByText('00000000')).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Select bytes to inspect their value.')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'tool-result-render-hex');
  });

  test('preview toolbar combines Copy/Download and toggles them for the raw editor', async ({ page }) => {
    await openResults(page);
    await page.getByRole('tab', { name: /Files/ }).click();

    await page.getByRole('button', { name: 'View report.json' }).click();
    // default JSON tree view: toolbar exposes the renderer picker + original Download + file Copy
    await expect(page.getByText('render-value')).toBeVisible({ timeout: 15000 });
    const rendererSelect = page.getByLabel('Select renderer');
    await expect(rendererSelect).toBeVisible();
    await expect(page.getByRole('button', { name: 'Download original file' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Copy file content' })).toBeVisible();

    // switching to the raw editor toggles the SAME buttons to the edited-buffer variants and drops
    // the original-download button; the compact JSON is prettified onto multiple lines
    await rendererSelect.selectOption('Editor');
    await expect(page.locator('.cm-editor')).toBeVisible({ timeout: 15000 });
    await expect(page.getByRole('button', { name: 'Download edited content' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Copy edited content' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Download original file' })).toHaveCount(0);
    // prettify-on-open: CodeMirror renders one .cm-line per line, so a prettified object spans many
    expect(await page.locator('.cm-editor .cm-line').count()).toBeGreaterThan(1);
    await snapshot(page, SCREENSHOT_DIR, 'tool-result-render-editor');
  });

  test('preview toolbar hides Copy for binary (hex) files', async ({ page }) => {
    await openResults(page);
    await page.getByRole('tab', { name: /Files/ }).click();

    await page.getByRole('button', { name: 'View dump.bin' }).click();
    // hex view is binary — Download stays, Copy is hidden
    await expect(page.getByText('00000000')).toBeVisible({ timeout: 15000 });
    await expect(page.getByRole('button', { name: 'Download original file' })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Copy/ })).toHaveCount(0);
  });

  test('Files tab renders a markdown file and can switch to the raw editor', async ({ page }) => {
    await openResults(page);
    await page.getByRole('tab', { name: /Files/ }).click();

    await page.getByRole('button', { name: 'View report.md' }).click();
    // markdown is the default renderer: the '#' heading renders as an actual heading element
    await expect(page.getByRole('heading', { name: 'Seeded Heading' })).toBeVisible({ timeout: 15000 });

    // switching to the raw editor shows the literal markdown source instead
    await page.getByLabel('Select renderer').selectOption('Editor');
    await expect(page.locator('.cm-editor')).toBeVisible({ timeout: 15000 });
    await snapshot(page, SCREENSHOT_DIR, 'tool-result-render-markdown');
  });

  test('version selector and diff window work with multiple versions', async ({ page }) => {
    await openResults(page);
    // version selector lists both seeded versions
    const versionSelect = page.getByLabel('Select result version');
    await expect(versionSelect).toBeVisible();
    await expect(versionSelect.locator('option')).toHaveCount(2);

    // selecting the older version swaps the rendered result
    await versionSelect.selectOption({ index: 1 });
    await expect(page.getByText('first-version')).toBeVisible();

    // the diff button opens the diff window comparing the two result versions
    await page.getByRole('button', { name: 'Diff results' }).click();
    await expect(page.getByText(`Diff: ${tool}`)).toBeVisible({ timeout: 15000 });
    await expect(page.getByLabel('What to compare')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'tool-result-diff');
  });
});

/**
 * Exercises the collapsible tile body refactor: tall tab content is capped to a default height that
 * scrolls internally (preview without expanding), and the Show more / Show less toggle expands it.
 * Children name-resolution + the fetch button require agent-produced children (not seedable via the
 * results API), so those are covered by unit tests (`useChildrenMetadata.test.ts`) + manual checks.
 */
test.describe('Tool Result Tile — collapsible scroll', () => {
  let token: string;
  let fileSha256: string;
  const tool = 'pwscrolltool';
  // a String result far taller than the 420px default body height so the tile collapses + scrolls
  const tallResult = Array.from({ length: 200 }, (_, i) => `line ${i} of a very long tool result body`).join('\n');

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    const file = await uploadFile(token, Buffer.from(`tile-scroll e2e content ${Date.now()}`), 'tile-scroll-e2e.bin', ['system']);
    fileSha256 = file.sha256;
    await uploadResults(token, fileSha256, {
      tool,
      result: tallResult,
      displayType: 'String',
      groups: ['system'],
      version: '1.0.0',
    });
  });

  test('collapsed body scrolls internally and Show more expands it', async ({ page }) => {
    await loginViaUI(page);
    await page.goto(`/file/${fileSha256}`);
    await page.waitForLoadState('networkidle');
    await expect(page.getByText(tool, { exact: true }).first()).toBeVisible({ timeout: 20000 });

    // the toggle only appears when content overflows the default height
    const showMore = page.getByRole('button', { name: /Show more/ });
    await expect(showMore).toBeVisible();

    // while collapsed, the body is capped and scrollable (content taller than the visible box)
    const scroll = page.getByTestId('result-body-scroll').first();
    const collapsed = await scroll.evaluate((el) => ({
      clientHeight: el.clientHeight,
      scrollHeight: el.scrollHeight,
      overflowY: getComputedStyle(el).overflowY,
    }));
    expect(collapsed.overflowY).toBe('auto');
    expect(collapsed.scrollHeight).toBeGreaterThan(collapsed.clientHeight);

    // expanding removes the cap so the whole body is shown and the toggle flips to Show less
    await showMore.click();
    await expect(page.getByRole('button', { name: /Show less/ })).toBeVisible();
    const expanded = await scroll.evaluate((el) => el.clientHeight);
    expect(expanded).toBeGreaterThan(collapsed.clientHeight);
    await snapshot(page, SCREENSHOT_DIR, 'tool-result-collapsible-scroll');
  });
});
