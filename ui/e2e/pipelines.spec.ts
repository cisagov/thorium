import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { snapshot, MOCK_USER, waitForEditor } from './helpers';

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

const MOCK_PIPELINE = {
  group: 'system',
  name: 'test-pipeline',
  creator: 'test',
  order: [['tool-a', 'tool-b'], 'tool-c'],
  sla: 604800,
  description: 'A test pipeline for E2E testing',
  triggers: {},
};

const MOCK_PIPELINE_WITH_TRIGGER = {
  group: 'system',
  name: 'triggered-pipeline',
  creator: 'test',
  order: ['scanner'],
  sla: 86400,
  description: 'Pipeline with event trigger',
  triggers: {
    'new-sample-trigger': 'NewSample',
  },
};

async function setupMockAuth(page: Page) {
  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }),
  );
  await page.context().addCookies([
    {
      name: 'THORIUM_TOKEN',
      value: MOCK_USER.token,
      domain: 'localhost',
      path: '/',
    },
  ]);
}

const MOCK_GROUP_IMAGES = ['tool-a', 'tool-b', 'tool-c', 'scanner', 'extra-tool', 'analyzer'];

async function setupPipelineMocks(page: Page, pipelines = [MOCK_PIPELINE, MOCK_PIPELINE_WITH_TRIGGER]) {
  await page.route('**/api/groups/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: [MOCK_GROUP] }) }),
  );
  await page.route('**/api/pipelines/list/**/details/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ details: pipelines }) }),
  );
  await page.route('**/api/pipelines/', (route) => {
    if (route.request().method() === 'POST') {
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.route('**/api/pipelines/*/*', (route) => {
    if (route.request().method() === 'DELETE') {
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    }
    if (route.request().method() === 'PATCH') {
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.route('**/api/images/system/', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ names: MOCK_GROUP_IMAGES }) }),
  );
}

test.describe('Pipelines Page', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockAuth(page);
    await setupPipelineMocks(page);
  });

  test('pipeline list renders with correct count', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    const badge = page.locator('.count-badge');
    await expect(badge).toContainText('2');

    const items = page.locator('.accordion-item');
    await expect(items).toHaveCount(2);

    await snapshot(page, SCREENSHOT_DIR, 'pipelines-list');
  });

  test('pipeline names and groups display correctly', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await expect(page.locator('.accordion-item-name .text').first()).toContainText('test-pipeline');
    await expect(page.locator('.accordion-item-ownership i').first()).toContainText('system');
  });

  test('expand pipeline shows details', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForTimeout(500);

    await expect(page.locator('.bg-blue').first()).toContainText('test');
    await expect(page.locator('text=A test pipeline for E2E testing')).toBeVisible();
    await expect(page.locator('text=604800')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-details-expanded');
  });

  test('pipeline with NewSample trigger displays trigger type', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').nth(1).click();
    await page.waitForTimeout(500);

    await expect(page.locator('text=new-sample-trigger')).toBeVisible();
    await expect(page.locator('text=NewSample')).toBeVisible();
  });

  test('create pipeline button opens modal', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.ok-btn:has-text("+")').click();
    await page.waitForTimeout(500);

    const modal = page.locator('.modal');
    await expect(modal).toBeVisible();
    await expect(modal.locator('.modal-title')).toContainText('Create New Pipeline');

    await waitForEditor(page);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-create-modal');
  });

  test('create pipeline modal has format toggle', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.ok-btn:has-text("+")').click();
    await page.waitForTimeout(500);

    const yamlBtn = page.locator('button:has-text("YAML")');
    const jsonBtn = page.locator('button:has-text("JSON")');
    await expect(yamlBtn).toBeVisible();
    await expect(jsonBtn).toBeVisible();
  });

  test('edit button enters editor mode', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForTimeout(500);

    await page.locator('button:has-text("Edit")').first().click();
    await page.waitForTimeout(500);

    await waitForEditor(page);
    await expect(page.locator('button:has-text("Discard")').first()).toBeVisible();
    await expect(page.locator('button:has-text("Accept")').first()).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-edit-mode');
  });

  test('cancel edit returns to view mode', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForTimeout(500);

    await page.locator('button:has-text("Edit")').first().click();
    await page.waitForTimeout(500);
    await waitForEditor(page);

    await page.locator('button:has-text("Discard")').first().click();
    await page.waitForTimeout(300);

    await expect(page.locator('.cm-editor')).toHaveCount(0);
    await expect(page.locator('text=A test pipeline for E2E testing')).toBeVisible();
  });

  test('delete pipeline shows confirmation modal', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForTimeout(500);

    await page.locator('button:has-text("Delete")').first().click();
    await page.waitForTimeout(300);

    const modal = page.locator('.modal');
    await expect(modal).toBeVisible();
    await expect(modal).toContainText('Confirm deletion');
    await expect(modal).toContainText('test-pipeline');

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-delete-confirm');
  });

  test('confirm delete calls API', async ({ page }) => {
    let deleteRequested = false;
    await page.route('**/api/pipelines/system/test-pipeline', (route) => {
      if (route.request().method() === 'DELETE') {
        deleteRequested = true;
        return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
      }
      return route.fallback();
    });

    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForTimeout(500);

    await page.locator('button:has-text("Delete")').first().click();
    await page.waitForTimeout(300);

    await page.locator('.danger-btn:has-text("Confirm")').click();
    await page.waitForTimeout(500);

    expect(deleteRequested).toBe(true);
  });

  test('format toggle switches between YAML and JSON in create modal', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.ok-btn:has-text("+")').click();
    await page.waitForTimeout(500);
    await waitForEditor(page);

    const editorContent = await page.locator('.cm-content').textContent();
    expect(editorContent).toContain('group');

    await page.locator('button:has-text("JSON")').click();
    await page.waitForTimeout(500);
    await waitForEditor(page);

    const jsonContent = await page.locator('.cm-content').textContent();
    expect(jsonContent).toContain('{');
  });

  test('pipeline order flow renders with correct nodes', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow', { timeout: 5000 });
    await page.waitForTimeout(500);

    const firstBody = page.locator('.accordion-item').first().locator('.accordion-body');
    const imageNodes = firstBody.locator('.react-flow__node-imageStep');
    await expect(imageNodes).toHaveCount(3);

    const terminalNodes = firstBody.locator('.react-flow__node-terminal');
    await expect(terminalNodes).toHaveCount(2);
  });

  test('pipeline order flow displays correct image names', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node', { timeout: 5000 });
    await page.waitForTimeout(500);

    const nodeTexts = await page.locator('.react-flow__node-imageStep').allTextContents();
    expect(nodeTexts).toContain('tool-a');
    expect(nodeTexts).toContain('tool-b');
    expect(nodeTexts).toContain('tool-c');
  });

  test('pipeline order flow renders edges between steps', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node', { timeout: 5000 });
    await page.waitForTimeout(500);

    const edges = page.locator('.react-flow__edge');
    await expect(edges).toHaveCount(6);
  });

  test('pipeline order flow is properly sized within container', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node', { timeout: 5000 });
    await page.waitForTimeout(800);

    const flowContainer = page.locator('.react-flow').first();
    const containerBox = await flowContainer.boundingBox();
    expect(containerBox).not.toBeNull();
    expect(containerBox!.width).toBeGreaterThan(100);
    expect(containerBox!.height).toBeGreaterThan(50);

    const firstNode = page.locator('.react-flow__node-imageStep').first();
    const nodeBox = await firstNode.boundingBox();
    expect(nodeBox).not.toBeNull();

    expect(nodeBox!.x).toBeGreaterThanOrEqual(containerBox!.x);
    expect(nodeBox!.y).toBeGreaterThanOrEqual(containerBox!.y);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-order-flow');
  });

  test('pipeline order flow survives edit round-trip', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    const firstBody = page.locator('.accordion-item').first().locator('.accordion-body');
    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node', { timeout: 5000 });

    await page.locator('button:has-text("Edit")').first().click();
    await page.waitForTimeout(500);
    await waitForEditor(page);

    await expect(firstBody.locator('.react-flow')).toHaveCount(0);

    await page.locator('button:has-text("Discard")').first().click();
    await page.waitForSelector('.react-flow__node', { timeout: 5000 });
    await page.waitForTimeout(500);

    const imageNodes = firstBody.locator('.react-flow__node-imageStep');
    await expect(imageNodes).toHaveCount(3);
  });

  test('single-step pipeline order flow renders one node', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').nth(1).click();
    await page.waitForTimeout(1000);

    const secondBody = page.locator('.accordion-item').nth(1).locator('.accordion-body');
    await expect(secondBody).toBeVisible({ timeout: 5000 });

    const imageNodes = secondBody.locator('.react-flow__node-imageStep');
    await expect(imageNodes).toHaveCount(1);

    const nodeText = await imageNodes.first().textContent();
    expect(nodeText).toContain('scanner');

    const terminalNodes = secondBody.locator('.react-flow__node-terminal');
    await expect(terminalNodes).toHaveCount(2);

    const edges = secondBody.locator('.react-flow__edge');
    await expect(edges).toHaveCount(2);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-single-step-flow');
  });

  test('right-click on empty pane shows context menu with Insert Image', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow', { timeout: 5000 });
    await page.waitForTimeout(500);

    const flow = page.locator('.react-flow').first();
    await flow.click({ button: 'right', position: { x: 50, y: 50 } });
    await page.waitForTimeout(300);

    const contextMenu = page.locator('div:has-text("Insert Image")').filter({ hasText: 'Insert Image' }).first();
    await expect(contextMenu).toBeVisible();
  });

  test('right-click on image node shows context menu with Insert and Remove options', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node-imageStep', { timeout: 5000 });
    await page.waitForTimeout(500);

    const node = page.locator('.react-flow__node-imageStep').first();
    await node.click({ button: 'right' });
    await page.waitForTimeout(300);

    await expect(page.locator('text=Insert Image')).toBeVisible();
    await expect(page.locator('text=Remove Image')).toBeVisible();
  });

  test('context menu dismissed on Escape', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow', { timeout: 5000 });
    await page.waitForTimeout(500);

    const flow = page.locator('.react-flow').first();
    await flow.click({ button: 'right', position: { x: 50, y: 50 } });
    await page.waitForTimeout(300);
    await expect(page.locator('text=Insert Image')).toBeVisible();

    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    await expect(page.locator('text=Insert Image')).not.toBeVisible();
  });

  test('Insert Image opens select dropdown with available images', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow', { timeout: 5000 });
    await page.waitForTimeout(500);

    const flow = page.locator('.react-flow').first();
    await flow.click({ button: 'right', position: { x: 50, y: 50 } });
    await page.waitForTimeout(300);

    await page.locator('text=Insert Image').click();
    await page.waitForTimeout(300);

    // Select dropdown should be visible with available images (not already in pipeline)
    const selectMenu = page.locator('[class*="menu"]');
    await expect(selectMenu.first()).toBeVisible();

    // Images already in the pipeline (tool-a, tool-b, tool-c) should be excluded
    // Images not in the pipeline (extra-tool, analyzer) should be available
    const options = page.locator('[class*="option"]');
    const optionTexts = await options.allTextContents();
    expect(optionTexts.some((t) => t.includes('extra-tool'))).toBe(true);
    expect(optionTexts.some((t) => t.includes('analyzer'))).toBe(true);
    expect(optionTexts.every((t) => !t.includes('tool-a'))).toBe(true);
    expect(optionTexts.every((t) => !t.includes('tool-b'))).toBe(true);
    expect(optionTexts.every((t) => !t.includes('tool-c'))).toBe(true);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-insert-image-select');
  });

  test('selecting an image from dropdown adds it to the diagram and shows Apply bar', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node-imageStep', { timeout: 5000 });
    await page.waitForTimeout(500);

    const firstBody = page.locator('.accordion-item').first().locator('.accordion-body');

    // Start with 3 image nodes
    await expect(firstBody.locator('.react-flow__node-imageStep')).toHaveCount(3);

    const flow = page.locator('.react-flow').first();
    await flow.click({ button: 'right', position: { x: 50, y: 50 } });
    await page.waitForTimeout(300);

    await page.locator('text=Insert Image').click();
    await page.waitForTimeout(300);

    // Click an available option
    await page.locator('[class*="option"]', { hasText: 'extra-tool' }).click();
    await page.waitForTimeout(500);

    // Should now have 4 image nodes
    await expect(firstBody.locator('.react-flow__node-imageStep')).toHaveCount(4);

    // Apply/Discard bar should appear
    await expect(page.locator('button:has-text("Apply")')).toBeVisible();
    await expect(page.locator('button:has-text("Discard")')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-image-inserted');
  });

  test('Remove Image removes a node and shows Apply bar', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node-imageStep', { timeout: 5000 });
    await page.waitForTimeout(500);

    const firstBody = page.locator('.accordion-item').first().locator('.accordion-body');
    await expect(firstBody.locator('.react-flow__node-imageStep')).toHaveCount(3);

    // Right-click on a node to get Remove Image option
    const node = firstBody.locator('.react-flow__node-imageStep').first();
    await node.click({ button: 'right' });
    await page.waitForTimeout(300);

    await page.locator('text=Remove Image').click();
    await page.waitForTimeout(500);

    // Should now have 2 image nodes
    await expect(firstBody.locator('.react-flow__node-imageStep')).toHaveCount(2);

    // Apply/Discard bar should appear
    await expect(page.locator('button:has-text("Apply")')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-image-removed');
  });

  test('Discard reverts pending diagram changes', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow__node-imageStep', { timeout: 5000 });
    await page.waitForTimeout(500);

    const firstBody = page.locator('.accordion-item').first().locator('.accordion-body');
    await expect(firstBody.locator('.react-flow__node-imageStep')).toHaveCount(3);

    // Remove an image
    const node = firstBody.locator('.react-flow__node-imageStep').first();
    await node.click({ button: 'right' });
    await page.waitForTimeout(300);
    await page.locator('text=Remove Image').click();
    await page.waitForTimeout(500);
    await expect(firstBody.locator('.react-flow__node-imageStep')).toHaveCount(2);

    // Click Discard
    await page.locator('button:has-text("Discard")').click();
    await page.waitForTimeout(500);

    // Should be back to 3 nodes
    await expect(firstBody.locator('.react-flow__node-imageStep')).toHaveCount(3);
    await expect(page.locator('button:has-text("Apply")')).not.toBeVisible();
  });

  test('double-click on empty pane opens image select dropdown', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow', { timeout: 5000 });
    await page.waitForTimeout(500);

    const flow = page.locator('.react-flow').first();
    await flow.dblclick({ position: { x: 50, y: 50 } });
    await page.waitForTimeout(300);

    // Select dropdown should be visible
    const selectMenu = page.locator('[class*="menu"]');
    await expect(selectMenu.first()).toBeVisible();
  });

  test('select dropdown dismissed on Escape', async ({ page }) => {
    await page.goto('/pipelines');
    await page.waitForSelector('.accordion', { timeout: 10000 });

    await page.locator('.accordion-header').first().click();
    await page.waitForSelector('.react-flow', { timeout: 5000 });
    await page.waitForTimeout(500);

    const flow = page.locator('.react-flow').first();
    await flow.dblclick({ position: { x: 50, y: 50 } });
    await page.waitForTimeout(300);

    const selectMenu = page.locator('[class*="menu"]');
    await expect(selectMenu.first()).toBeVisible();

    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);

    await expect(selectMenu).not.toBeVisible();
  });
});
