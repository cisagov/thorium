import { test, expect, Page, Locator } from '@playwright/test';
import path from 'path';
import { snapshot, MOCK_USER } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const MOCK_GROUP = {
  name: 'system',
  owners: { combined: ['test'], direct: ['test'], metagroups: [] },
  managers: { combined: [], direct: [], metagroups: [] },
  analysts: [],
  users: { combined: [], direct: [], metagroups: [] },
  monitors: { combined: [], direct: [], metagroups: [] },
  description: 'System group',
  allowed: {
    files: true,
    repos: true,
    tags: true,
    images: true,
    pipelines: true,
    reactions: true,
    results: true,
    comments: true,
    entities: true,
  },
};

const MOCK_IMAGE = {
  group: 'system',
  name: 'e2e-btn-test',
  creator: 'test',
  version: { Custom: '1.0' },
  scaler: 'K8s',
  image: 'registry.example.com/tool:latest',
  timeout: 600,
  description: 'Image for button styling test',
  display_type: 'JSON',
  args: {},
  resources: { cpu: 1000, memory: 512 },
  env: {},
  dependencies: {},
  volumes: [],
  output_collection: {},
  security_context: {},
  child_filters: {},
  network_policies: [],
  bans: [],
  collect_logs: false,
  generator: false,
  used_by: [],
};

const MOCK_PIPELINE = {
  group: 'system',
  name: 'e2e-btn-pipeline',
  creator: 'test',
  order: [['tool-a', 'tool-b'], 'tool-c'],
  sla: 604800,
  description: 'Pipeline for button styling test',
  triggers: {},
  bans: [],
};

const BOOTSTRAP_PRIMARY_BLUE = 'rgb(13, 110, 253)';
const BOOTSTRAP_PRIMARY_HOVER = 'rgb(11, 94, 215)';
const BOOTSTRAP_FOCUS_RGB = '49, 132, 253';

async function setupAuth(page: Page) {
  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }),
  );
  await page.context().addCookies([
    { name: 'THORIUM_TOKEN', value: MOCK_USER.token, domain: 'localhost', path: '/' },
  ]);
}

async function setupCatchAll(page: Page) {
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (url.includes('/users/whoami')) return route.fallback();
    if (url.includes('/groups/')) return route.fallback();
    if (url.includes('/images/') && url.includes('/details/')) return route.fallback();
    if (url.includes('/pipelines/') && url.includes('/details/')) return route.fallback();
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
}

async function setupImageMocks(page: Page) {
  await page.route('**/api/groups/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }),
  );
  await page.route('**/api/images/*/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_IMAGE] }) }),
  );
  await setupCatchAll(page);
}

async function setupPipelineMocks(page: Page) {
  await page.route('**/api/groups/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }),
  );
  await page.route('**/api/pipelines/list/**/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_PIPELINE] }) }),
  );
  await page.route('**/api/images/system/', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ names: ['tool-a', 'tool-b', 'tool-c'] }) }),
  );
  await setupCatchAll(page);
}

async function getComputedStyleProps(locator: Locator) {
  return locator.evaluate((el) => {
    const cs = getComputedStyle(el);
    return {
      backgroundColor: cs.backgroundColor,
      borderColor: cs.borderColor,
      color: cs.color,
      padding: cs.padding,
      fontSize: cs.fontSize,
      lineHeight: cs.lineHeight,
      display: cs.display,
      borderRadius: cs.borderRadius,
      boxShadow: cs.boxShadow,
    };
  });
}

// Finds a header action button by data-testid
function headerBtn(page: Page, action: string): Locator {
  return page.locator(`[data-testid="header-btn-${action}"]`).first();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
test.describe('Accordion Header Button Styling', () => {
  test('image accordion buttons render with correct variant colors', async ({ page }) => {
    await setupAuth(page);
    await setupImageMocks(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    const copyBtn = await getComputedStyleProps(headerBtn(page, 'copy'));
    const editBtn = await getComputedStyleProps(headerBtn(page, 'edit'));
    const deleteBtn = await getComputedStyleProps(headerBtn(page, 'delete'));

    // none should use Bootstrap primary blue
    expect(copyBtn.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_BLUE);
    expect(editBtn.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_BLUE);
    expect(deleteBtn.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_BLUE);

    // all three should have distinct background colors
    expect(copyBtn.backgroundColor).not.toBe(editBtn.backgroundColor);
    expect(copyBtn.backgroundColor).not.toBe(deleteBtn.backgroundColor);
    expect(editBtn.backgroundColor).not.toBe(deleteBtn.backgroundColor);

    // buttons should have visible border-radius (not 0)
    expect(copyBtn.borderRadius).not.toBe('0px');
    expect(editBtn.borderRadius).not.toBe('0px');
    expect(deleteBtn.borderRadius).not.toBe('0px');

    await snapshot(page, SCREENSHOT_DIR, 'accordion-image-buttons');
  });

  test('pipeline accordion buttons render with correct variant colors', async ({ page }) => {
    await setupAuth(page);
    await setupPipelineMocks(page);
    await page.goto('/pipelines');
    // The pipelines page uses the custom styled Accordion (no bootstrap `.accordion` class), so wait
    // for a header action button instead.
    await page.waitForSelector('[data-testid="header-btn-copy"]', { timeout: 10000 });

    const copyBtn = await getComputedStyleProps(headerBtn(page, 'copy'));
    const editBtn = await getComputedStyleProps(headerBtn(page, 'edit'));
    const deleteBtn = await getComputedStyleProps(headerBtn(page, 'delete'));

    expect(copyBtn.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_BLUE);
    expect(editBtn.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_BLUE);
    expect(deleteBtn.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_BLUE);

    expect(copyBtn.backgroundColor).not.toBe(editBtn.backgroundColor);
    expect(copyBtn.backgroundColor).not.toBe(deleteBtn.backgroundColor);

    await snapshot(page, SCREENSHOT_DIR, 'accordion-pipeline-buttons');
  });

  test('accordion header buttons match custom button component colors', async ({ page }) => {
    await setupAuth(page);
    await setupImageMocks(page);

    // collect reference colors from the custom Button component
    await page.goto('/test/buttons');
    await page.waitForLoadState('domcontentloaded');

    const refOk = await getComputedStyleProps(page.locator('[data-testid="btn-ok"]'));
    const refSecondary = await getComputedStyleProps(page.locator('[data-testid="btn-secondary"]'));
    const refWarning = await getComputedStyleProps(page.locator('[data-testid="btn-warning"]'));

    // navigate to images page and compare
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    const accordionCopy = await getComputedStyleProps(headerBtn(page, 'copy'));
    const accordionEdit = await getComputedStyleProps(headerBtn(page, 'edit'));
    const accordionDelete = await getComputedStyleProps(headerBtn(page, 'delete'));

    // background colors should match — both use the same --thorium-* design tokens
    expect(accordionCopy.backgroundColor).toBe(refOk.backgroundColor);
    expect(accordionEdit.backgroundColor).toBe(refSecondary.backgroundColor);
    expect(accordionDelete.backgroundColor).toBe(refWarning.backgroundColor);

    // border-radius should match the custom button
    expect(accordionCopy.borderRadius).toBe(refOk.borderRadius);
  });

  test('accordion header buttons do not show Bootstrap primary blue on hover', async ({ page }) => {
    await setupAuth(page);
    await setupImageMocks(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    const copyBtn = headerBtn(page, 'copy');

    // hover over the Copy button in the accordion header
    await copyBtn.hover();
    await page.waitForTimeout(200);
    const hoverStyles = await getComputedStyleProps(copyBtn);

    // background should NOT change to Bootstrap blue on hover
    expect(hoverStyles.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_BLUE);
    expect(hoverStyles.backgroundColor).not.toBe(BOOTSTRAP_PRIMARY_HOVER);

    await snapshot(page, SCREENSHOT_DIR, 'accordion-button-hover');
  });

  test('accordion header buttons do not show blue focus ring', async ({ page }) => {
    await setupAuth(page);
    await setupImageMocks(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    const editBtn = headerBtn(page, 'edit');

    // focus the button to trigger focus-visible
    await editBtn.focus();
    await page.waitForTimeout(200);
    const focusStyles = await editBtn.evaluate((el) => {
      const cs = getComputedStyle(el);
      return { boxShadow: cs.boxShadow };
    });

    // should NOT have Bootstrap primary blue focus shadow
    if (focusStyles.boxShadow && focusStyles.boxShadow !== 'none') {
      expect(focusStyles.boxShadow).not.toContain(BOOTSTRAP_FOCUS_RGB);
    }

    await snapshot(page, SCREENSHOT_DIR, 'accordion-button-focus');
  });

  test('accordion buttons render as span, not nested button', async ({ page }) => {
    await setupAuth(page);
    await setupImageMocks(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    // HeaderBtn inside accordion-header should be <span>, not <button>
    const headerBtnTag = await headerBtn(page, 'copy').evaluate((el) => el.tagName.toLowerCase());
    expect(headerBtnTag).toBe('span');

    // the accordion-button itself is a <button>
    const accordionBtnTag = await page.locator('.accordion-button').first().evaluate((el) => el.tagName.toLowerCase());
    expect(accordionBtnTag).toBe('button');

    // no nested <button> elements inside the accordion <button>
    const nestedButtons = await page.locator('.accordion-button button').count();
    expect(nestedButtons).toBe(0);
  });

  test('side-by-side screenshot comparison', async ({ page }) => {
    await setupAuth(page);
    await setupImageMocks(page);

    // screenshot the custom button test page for reference
    await page.goto('/test/buttons');
    await page.waitForLoadState('domcontentloaded');
    await snapshot(page, SCREENSHOT_DIR, 'accordion-ref-custom-buttons');

    // screenshot images page accordion with header buttons
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 10000 });
    await page.locator('.accordion-header').first().click();
    await page.waitForTimeout(500);
    await snapshot(page, SCREENSHOT_DIR, 'accordion-ref-image-buttons');
  });
});
