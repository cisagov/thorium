import { test, expect } from '@playwright/test';
import path from 'path';
import {
  snapshot,
  loginViaUI,
  authenticate,
  createImageViaAPI,
  deleteImageViaAPI,
  createPipelineViaAPI,
  getPipelineViaAPI,
  updatePipelineViaAPI,
  deletePipelineViaAPI,
  waitForEditor,
  setEditorContent,
  TEST_USER,
  TEST_PASS,
} from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const PIPELINE_FULL = {
  group: 'system',
  name: 'e2e-pipeline-full',
  order: [['step-a', 'step-b'], 'step-c'],
  sla: 86400,
  triggers: {
    'new-upload': 'NewSample',
    'tag-match': {
      Tag: {
        tag_types: ['Files'],
        required: { malware: ['true'] },
        not: { benign: ['true'] },
      },
    },
  },
  description: 'E2E test pipeline with **markdown** description',
};

const PIPELINE_MINIMAL = {
  group: 'system',
  name: 'e2e-pipeline-minimal',
  order: ['single-step'],
};

let token: string;

const STUB_IMAGE = { scaler: 'External', display_type: 'JSON' };
const PIPELINE_IMAGES = ['step-a', 'step-b', 'step-c', 'single-step'];

async function createStubImages(tok: string) {
  for (const name of PIPELINE_IMAGES) {
    await deleteImageViaAPI(tok, 'system', name);
    await createImageViaAPI(tok, { group: 'system', name, ...STUB_IMAGE });
  }
}

async function deleteStubImages(tok: string) {
  for (const name of PIPELINE_IMAGES) {
    await deleteImageViaAPI(tok, 'system', name);
  }
}

test.describe('Pipeline Live API', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await createStubImages(token);
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-full');
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-minimal');
    await createPipelineViaAPI(token, PIPELINE_FULL);
    await createPipelineViaAPI(token, PIPELINE_MINIMAL);
  });

  test.afterAll(async () => {
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-full');
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-minimal');
    await deleteStubImages(token);
  });

  test('full pipeline displays all configured fields', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const fullItem = page.locator('[data-testid="accordion-item"]', { has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-full' }) });
    await fullItem.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await expect(fullItem.locator('.bg-blue').first()).toContainText('test');
    await expect(fullItem.locator('text=86400')).toBeVisible();
    await expect(fullItem.locator('text=markdown')).toBeVisible();
    await expect(fullItem.locator('text=new-upload')).toBeVisible();
    await expect(fullItem.locator('text=NewSample')).toBeVisible();
    await expect(fullItem.locator('text=tag-match')).toBeVisible();

    await fullItem.locator('.react-flow').waitFor({ timeout: 5000 });
    const imageNodes = fullItem.locator('.react-flow__node-imageStep');
    await expect(imageNodes).toHaveCount(3);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-full-display');
  });

  test('minimal pipeline uses defaults correctly', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const minItem = page.locator('[data-testid="accordion-item"]', { has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-minimal' }) });
    await minItem.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await expect(minItem.locator('b:has-text("SLA")')).toBeVisible();
    await expect(minItem.locator('b:has-text("Event Triggers")')).toBeVisible();

    await minItem.locator('.react-flow').waitFor({ timeout: 5000 });
    const imageNodes = minItem.locator('.react-flow__node-imageStep');
    await expect(imageNodes).toHaveCount(1);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-minimal-display');
  });

  test('patch pipeline SLA and description, verify display updates', async ({ page }) => {
    await updatePipelineViaAPI(token, 'system', 'e2e-pipeline-minimal', {
      sla: 3600,
      description: 'Updated via API',
    });

    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const minItem = page.locator('[data-testid="accordion-item"]', { has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-minimal' }) });
    await minItem.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await expect(minItem.locator('text=3600')).toBeVisible();
    await expect(minItem.locator('text=Updated via API')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-patched-sla');
  });

  test('patch pipeline triggers: add and remove, verify display', async ({ page }) => {
    await updatePipelineViaAPI(token, 'system', 'e2e-pipeline-full', {
      triggers: { 'new-trigger': 'NewSample' },
      remove_triggers: ['new-upload'],
    });

    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const fullItem = page.locator('[data-testid="accordion-item"]', { has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-full' }) });
    await fullItem.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await expect(fullItem.locator('text=new-trigger')).toBeVisible();
    await expect(fullItem.locator('text=new-upload')).not.toBeVisible();
    await expect(fullItem.locator('text=tag-match')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-patched-triggers');
  });

  test('patch pipeline order, verify flow visualization updates', async ({ page }) => {
    await updatePipelineViaAPI(token, 'system', 'e2e-pipeline-minimal', {
      order: [['step-a', 'step-b'], 'step-c'],
    });

    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const minItem = page.locator('[data-testid="accordion-item"]', { has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-minimal' }) });
    await minItem.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await minItem.locator('.react-flow').waitFor({ timeout: 5000 });
    const imageNodes = minItem.locator('.react-flow__node-imageStep');
    await expect(imageNodes).toHaveCount(3);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-patched-order');
  });

  test('editor round-trip: edit pipeline via UI editor, verify API state', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const minItem = page.locator('[data-testid="accordion-item"]', { has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-minimal' }) });
    await minItem.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await minItem.locator('.secondary-btn:has-text("Edit")').click();
    await page.waitForTimeout(300);
    // edit opens in the form view; switch to the editor view to drive it via YAML
    await minItem.locator('button:has-text("Editor")').click();
    await waitForEditor(page);

    const yaml = 'order:\n  - single-step\nsla: 7200\ndescription: Edited via UI';
    await setEditorContent(page, yaml);
    await page.waitForTimeout(300);

    await minItem.locator('[data-testid="header-btn-accept"]').click();
    await page.waitForTimeout(1000);

    const apiData = await getPipelineViaAPI(token, 'system', 'e2e-pipeline-minimal');
    expect(apiData).not.toBeNull();
    expect(apiData!.sla).toBe(7200);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-editor-roundtrip');
  });
});

// ---------------------------------------------------------------------------
// UI-driven create
// ---------------------------------------------------------------------------
test.describe('Pipeline Create via UI', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await createStubImages(token);
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-ui-created');
  });

  test.afterAll(async () => {
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-ui-created');
    await deleteStubImages(token);
  });

  test('create pipeline via UI page', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    await page.locator('[data-testid="create-pipeline-btn"]').click();
    await page.waitForURL('**/create/pipeline');

    // drive creation through the editor view so the whole pipeline can be set via YAML
    await page.locator('button:has-text("Editor")').click();
    await waitForEditor(page);

    const yaml = [
      'group: system',
      'name: e2e-pipeline-ui-created',
      'order:',
      '  - step-a',
      'sla: 3600',
      'description: Created via UI',
    ].join('\n');
    await setEditorContent(page, yaml);
    await page.waitForTimeout(300);

    await page.locator('[data-testid="create-submit"]').click();
    await page.waitForTimeout(2000);

    const apiData = await getPipelineViaAPI(token, 'system', 'e2e-pipeline-ui-created');
    expect(apiData).not.toBeNull();
    expect(apiData!.sla).toBe(3600);
    expect(apiData!.description).toBe('Created via UI');

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-create-ui');
  });
});

// ---------------------------------------------------------------------------
// UI-driven copy
// ---------------------------------------------------------------------------
test.describe('Pipeline Copy via UI', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await createStubImages(token);
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-copy-src');
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-copy-result');
    await createPipelineViaAPI(token, {
      group: 'system',
      name: 'e2e-pipeline-copy-src',
      order: ['step-a', 'step-b'],
      sla: 7200,
      description: 'Pipeline to copy',
    });
  });

  test.afterAll(async () => {
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-copy-src');
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-copy-result');
    await deleteStubImages(token);
  });

  test('copy pipeline populates template and creates', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const item = page.locator('[data-testid="accordion-item"]', {
      has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-copy-src' }),
    });
    await item.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(500);

    await item.locator('[data-testid="header-btn-copy"]').click();
    await page.waitForURL('**/create/pipeline');

    // copy seeds the create page with the source pipeline; verify via the editor view
    await page.locator('button:has-text("Editor")').click();
    await waitForEditor(page);

    const content = await page.locator('.cm-content').textContent();
    expect(content).toContain('copy-src');
    expect(content).toContain('7200');

    // Replace name with valid copy name
    const yaml = [
      'group: system',
      'name: e2e-pipeline-copy-result',
      'order:',
      '  - step-a',
      '  - step-b',
      'sla: 7200',
      'description: Pipeline to copy',
    ].join('\n');
    await setEditorContent(page, yaml);
    await page.waitForTimeout(300);

    await page.locator('[data-testid="create-submit"]').click();
    await page.waitForTimeout(2000);

    const apiData = await getPipelineViaAPI(token, 'system', 'e2e-pipeline-copy-result');
    expect(apiData).not.toBeNull();
    expect(apiData!.sla).toBe(7200);

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-copy-ui');
  });
});

// ---------------------------------------------------------------------------
// UI-driven delete
// ---------------------------------------------------------------------------
test.describe('Pipeline Delete via UI', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await createStubImages(token);
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-del-target');
    await createPipelineViaAPI(token, {
      group: 'system',
      name: 'e2e-pipeline-del-target',
      order: ['step-a'],
      description: 'Will be deleted',
    });
  });

  test.afterAll(async () => {
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-del-target');
    await deleteStubImages(token);
  });

  test('delete pipeline via UI confirmation modal', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const item = page.locator('[data-testid="accordion-item"]', {
      has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-del-target' }),
    });
    await expect(item).toBeVisible();
    await item.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(500);

    await item.locator('.warning-btn:has-text("Delete")').click();
    await page.waitForTimeout(500);

    const modal = page.locator('.modal');
    await expect(modal).toBeVisible();
    await expect(modal).toContainText('e2e-pipeline-del-target');

    await page.locator('.danger-btn:has-text("Confirm")').click();
    await page.waitForTimeout(2000);

    await expect(
      page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-del-target' }),
    ).not.toBeVisible({ timeout: 5000 });

    const apiData = await getPipelineViaAPI(token, 'system', 'e2e-pipeline-del-target');
    expect(apiData).toBeNull();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-delete-ui');
  });
});

// ---------------------------------------------------------------------------
// Discard edit
// ---------------------------------------------------------------------------
test.describe('Pipeline Discard Edit', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await createStubImages(token);
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-discard');
    await createPipelineViaAPI(token, {
      group: 'system',
      name: 'e2e-pipeline-discard',
      order: ['step-a'],
      sla: 1000,
      description: 'Discard test pipeline',
    });
  });

  test.afterAll(async () => {
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-discard');
    await deleteStubImages(token);
  });

  test('discard edit returns to view mode without saving', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const item = page.locator('[data-testid="accordion-item"]', {
      has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-discard' }),
    });
    await item.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await item.locator('.secondary-btn:has-text("Edit")').click();
    await page.waitForTimeout(500);
    await expect(item.locator('.secondary-btn:has-text("Discard")')).toBeVisible();

    await item.locator('.secondary-btn:has-text("Discard")').click();
    await page.waitForTimeout(500);

    // Edit form gone, view mode restored
    await expect(page.locator('.cm-editor')).not.toBeVisible();
    await expect(item.locator('text=Discard test pipeline')).toBeVisible();
    await expect(item.locator('.secondary-btn:has-text("Edit")')).toBeVisible();

    // API unchanged
    const apiData = await getPipelineViaAPI(token, 'system', 'e2e-pipeline-discard');
    expect(apiData!.sla).toBe(1000);
  });
});

// ---------------------------------------------------------------------------
// Ban display
// ---------------------------------------------------------------------------
test.describe('Pipeline Ban Display', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await createStubImages(token);
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-banned');
    await createPipelineViaAPI(token, {
      group: 'system',
      name: 'e2e-pipeline-banned',
      order: ['step-a'],
      description: 'Pipeline with ban',
    });
    await updatePipelineViaAPI(token, 'system', 'e2e-pipeline-banned', {
      bans: {
        bans_added: [
          {
            id: '00000000-0000-0000-0000-000000000002',
            time_banned: new Date().toISOString(),
            ban_kind: { Generic: { msg: 'Pipeline banned for testing' } },
          },
        ],
        bans_removed: [],
      },
    });
  });

  test.afterAll(async () => {
    await deletePipelineViaAPI(token, 'system', 'e2e-pipeline-banned');
    await deleteStubImages(token);
  });

  test('banned pipeline shows warning icon and ban details', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/pipelines');
    await page.waitForSelector('[data-testid="accordion"]', { timeout: 15000 });

    const item = page.locator('[data-testid="accordion-item"]', {
      has: page.locator('.accordion-item-name .text', { hasText: 'e2e-pipeline-banned' }),
    });

    // Ban warning icon in header
    await expect(item.locator('.accordion-item-status svg')).toBeVisible();

    await item.locator('[data-testid="accordion-header"]').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=Pipeline banned for testing')).toBeVisible();
    await expect(item.locator('text=Banned on')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'pipeline-live-ban-display');
  });
});
