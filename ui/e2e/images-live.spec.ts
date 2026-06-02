import { test, expect } from '@playwright/test';
import path from 'path';
import {
  snapshot,
  loginViaUI,
  authenticate,
  createImageViaAPI,
  getImageViaAPI,
  updateImageViaAPI,
  deleteImageViaAPI,
  waitForEditor,
  setEditorContent,
  TEST_USER,
  TEST_PASS,
} from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

const IMAGE_K8S = {
  group: 'system',
  name: 'e2e-image-k8s',
  scaler: 'K8s',
  image: 'thorium/e2e-test:latest',
  version: { Custom: '1.2.3' },
  timeout: 600,
  description: 'K8s image for E2E testing',
  lifetime: { counter: 'jobs', amount: 50 },
  display_type: 'JSON',
  spawn_limit: { Basic: 5 },
  collect_logs: true,
  generator: false,
  resources: { cpu: 2000, memory: 1024, ephemeral_storage: 2048 },
  args: { output: 'Append', entrypoint: ['/run.sh'] },
  dependencies: {
    samples: { location: '/samples', strategy: 'Paths' },
    tags: { enabled: true },
  },
  output_collection: {
    handler: 'Files',
    files: { results: '/tmp/results', result_files: '/tmp/result-files' },
    children: '/tmp/children',
  },
  env: { DEBUG: '1', LANG: 'en_US.UTF-8' },
  security_context: { user: 1000, group: 1000, allow_privilege_escalation: false },
  child_filters: {
    mime: ['application/pdf'],
    file_name: [],
    file_extension: ['pdf'],
    submit_non_matches: false,
  },
  modifiers: '/mnt/modifiers',
};

const IMAGE_EXTERNAL = {
  group: 'system',
  name: 'e2e-image-external',
  scaler: 'External',
  display_type: 'String',
  description: 'External scaler image',
};

const IMAGE_EDITOR_ONLY = {
  group: 'system',
  name: 'e2e-image-editor-only',
  scaler: 'K8s',
  image: 'thorium/e2e-editor:latest',
  timeout: 300,
  display_type: 'JSON',
  resources: { cpu: 1000, memory: 512 },
  modifiers: '/path/to/modifiers',
  child_filters: {
    mime: ['application/pdf'],
    file_name: [],
    file_extension: ['pdf'],
    submit_non_matches: true,
  },
  clean_up: {
    job_id: 'Append',
    results: 'None',
    result_files_dir: 'None',
    script: '/cleanup.sh',
  },
};

let token: string;

function findImageItem(page: import('@playwright/test').Page, name: string) {
  return page.locator('.accordion-item', {
    has: page.locator('.accordion-item-name .text', { hasText: name }),
  });
}

// ---------------------------------------------------------------------------
// Display verification
// ---------------------------------------------------------------------------
test.describe('Image Display', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-k8s');
    await deleteImageViaAPI(token, 'system', 'e2e-image-external');
    await deleteImageViaAPI(token, 'system', 'e2e-image-editor-only');
    await createImageViaAPI(token, IMAGE_K8S);
    await createImageViaAPI(token, IMAGE_EXTERNAL);
    await createImageViaAPI(token, IMAGE_EDITOR_ONLY);
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-k8s');
    await deleteImageViaAPI(token, 'system', 'e2e-image-external');
    await deleteImageViaAPI(token, 'system', 'e2e-image-editor-only');
  });

  test('K8s image displays all field sections', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-k8s');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('.image-tag:has-text("K8s")')).toBeVisible();
    await expect(item.locator('text=1.2.3')).toBeVisible();
    await expect(item.locator('text=K8s image for E2E testing')).toBeVisible();

    await expect(item.locator('b:has-text("Resources")')).toBeVisible();
    await expect(item.locator('b:has-text("Arguments")')).toBeVisible();
    await expect(item.locator('b:has-text("Output Collection")')).toBeVisible();
    await expect(item.locator('b:has-text("Dependencies")')).toBeVisible();
    await expect(item.locator('b:has-text("Environment")')).toBeVisible();
    await expect(item.locator('b:has-text("Security Context")')).toBeVisible();
    await expect(item.locator('b:has-text("Child Filters")')).toBeVisible();
    await expect(item.locator('b:has-text("Modifiers")')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'image-live-k8s-full');
  });

  test('External scaler image displays correctly', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-external');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=External scaler image')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'image-live-external');
  });

  test('editor-only fields visible in code editor', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-editor-only');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('b:has-text("Child Filters")')).toBeVisible();
    await expect(item.locator('b:has-text("Modifiers")')).toBeVisible();

    await item.locator('.secondary-btn:has-text("Edit")').click();
    await page.waitForTimeout(500);
    await page.getByRole('button', { name: 'Editor', exact: true }).click();
    await waitForEditor(page);

    const content = await page.locator('.cm-content').textContent();
    expect(content).toContain('modifiers');
    expect(content).toContain('/path/to/modifiers');

    await snapshot(page, SCREENSHOT_DIR, 'image-live-editor-only-fields');
  });
});

// ---------------------------------------------------------------------------
// API patch verification
// ---------------------------------------------------------------------------
test.describe('Image API Patch', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-k8s');
    await deleteImageViaAPI(token, 'system', 'e2e-image-external');
    await createImageViaAPI(token, IMAGE_K8S);
    await createImageViaAPI(token, IMAGE_EXTERNAL);
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-k8s');
    await deleteImageViaAPI(token, 'system', 'e2e-image-external');
  });

  test('patch timeout and description, verify display', async ({ page }) => {
    await updateImageViaAPI(token, 'system', 'e2e-image-external', {
      timeout: 900,
      description: 'Updated description',
    });

    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-external');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=900')).toBeVisible();
    await expect(item.locator('text=Updated description')).toBeVisible();
  });

  test('patch resources, verify display', async ({ page }) => {
    await updateImageViaAPI(token, 'system', 'e2e-image-k8s', {
      resources: { cpu: '4', memory: '2Gi' },
    });

    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-k8s');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('.image-tag:has-text("4000 mCPU")').first()).toBeVisible();
    await expect(item.locator('.image-tag:has-text("2048 MiB")').first()).toBeVisible();
  });

  test('add and remove env vars, verify display', async ({ page }) => {
    await updateImageViaAPI(token, 'system', 'e2e-image-k8s', {
      add_env: { NEW_VAR: 'hello' },
      remove_env: ['DEBUG'],
    });

    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-k8s');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=NEW_VAR')).toBeVisible();
    await expect(item.locator('text=hello')).toBeVisible();
    await expect(item.locator('text=LANG')).toBeVisible();
  });

  test('clear optional fields, verify display', async ({ page }) => {
    await updateImageViaAPI(token, 'system', 'e2e-image-k8s', {
      clear_description: true,
      clear_lifetime: true,
      clear_version: true,
    });

    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-k8s');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=1.2.3')).not.toBeVisible();
    await expect(item.locator('text=K8s image for E2E testing')).not.toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// UI-driven create
// ---------------------------------------------------------------------------
test.describe('Image Create via UI', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-ui-created');
    await deleteImageViaAPI(token, 'system', 'e2e-ui-created-editor');
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-ui-created');
    await deleteImageViaAPI(token, 'system', 'e2e-ui-created-editor');
  });

  test('create image via code editor, verify in API and UI', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/create/image');
    await page.waitForSelector('h3:has-text("Create An Image")', { timeout: 10000 });

    await page.locator('button:has-text("Editor")').click();
    await waitForEditor(page);

    const yaml = [
      'group: system',
      'name: e2e-ui-created-editor',
      'scaler: K8s',
      'image: thorium/e2e-ui:latest',
      'timeout: 450',
      'display_type: JSON',
      'description: Created via editor',
    ].join('\n');
    await setEditorContent(page, yaml);
    await page.waitForTimeout(300);

    await page.locator('.ok-btn:has-text("Create")').click();
    await page.waitForTimeout(2000);

    await expect(page).toHaveURL(/\/images/, { timeout: 10000 });

    const apiData = await getImageViaAPI(token, 'system', 'e2e-ui-created-editor');
    expect(apiData).not.toBeNull();
    expect(apiData!.name).toBe('e2e-ui-created-editor');
    expect(apiData!.timeout).toBe(450);

    await snapshot(page, SCREENSHOT_DIR, 'image-live-create-editor');
  });

  test('create image via form, verify in API', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/create/image');
    await page.waitForSelector('h3:has-text("Create An Image")', { timeout: 10000 });

    // Switch to editor to set all fields reliably
    await page.locator('button:has-text("Editor")').click();
    await waitForEditor(page);

    const yaml = [
      'group: system',
      'name: e2e-ui-created',
      'scaler: K8s',
      'image: thorium/e2e-form:latest',
      'timeout: 500',
      'display_type: JSON',
    ].join('\n');
    await setEditorContent(page, yaml);
    await page.waitForTimeout(300);

    await page.locator('.ok-btn:has-text("Create")').click();
    await page.waitForTimeout(2000);

    await expect(page).toHaveURL(/\/images/, { timeout: 10000 });

    const apiData = await getImageViaAPI(token, 'system', 'e2e-ui-created');
    expect(apiData).not.toBeNull();
    expect(apiData!.name).toBe('e2e-ui-created');
    expect(apiData!.timeout).toBe(500);

    await snapshot(page, SCREENSHOT_DIR, 'image-live-create-form');
  });
});

// ---------------------------------------------------------------------------
// UI-driven edit
// ---------------------------------------------------------------------------
test.describe('Image Edit via UI', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-edit-target');
    await createImageViaAPI(token, {
      group: 'system',
      name: 'e2e-image-edit-target',
      scaler: 'K8s',
      image: 'thorium/edit-target:latest',
      timeout: 300,
      display_type: 'JSON',
      description: 'Before edit',
    });
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-edit-target');
  });

  test('edit image via code editor, verify API state', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-edit-target');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await item.locator('.secondary-btn:has-text("Edit")').click();
    await page.waitForTimeout(500);
    await page.locator('button:has-text("Editor")').click();
    await waitForEditor(page);

    const yaml = [
      'scaler: K8s',
      'image: thorium/edit-target:latest',
      'timeout: 777',
      'display_type: JSON',
      'description: After editor edit',
    ].join('\n');
    await setEditorContent(page, yaml);
    await page.waitForTimeout(300);

    await item.locator('.ok-btn:has-text("Accept")').click();
    await page.waitForTimeout(2000);

    const apiData = await getImageViaAPI(token, 'system', 'e2e-image-edit-target');
    expect(apiData).not.toBeNull();
    expect(apiData!.timeout).toBe(777);
    expect(apiData!.description).toBe('After editor edit');

    await snapshot(page, SCREENSHOT_DIR, 'image-live-edit-editor');
  });

  test('edit image via form (change description), verify API state', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-edit-target');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await item.locator('.secondary-btn:has-text("Edit")').click();
    await page.waitForTimeout(500);

    // In form edit mode, clear textarea via JS then type new value
    const descTextarea = item.locator('textarea').first();
    await descTextarea.evaluate((el: HTMLTextAreaElement) => {
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        'value',
      )!.set!;
      nativeInputValueSetter.call(el, '');
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForTimeout(100);
    await descTextarea.type('Updated via form edit');
    await page.waitForTimeout(200);

    await item.locator('.ok-btn:has-text("Accept")').click();
    await page.waitForTimeout(2000);

    const apiData = await getImageViaAPI(token, 'system', 'e2e-image-edit-target');
    expect(apiData).not.toBeNull();
    expect(apiData!.description).toBe('Updated via form edit');

    await snapshot(page, SCREENSHOT_DIR, 'image-live-edit-form');
  });

  test('discard edit reverts to view mode without saving', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-edit-target');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await item.locator('.secondary-btn:has-text("Edit")').click();
    await page.waitForTimeout(500);

    await item.locator('.secondary-btn:has-text("Discard")').click();
    await page.waitForTimeout(500);

    // Should be back in view mode
    await expect(item.locator('.secondary-btn:has-text("Edit")')).toBeVisible();
    await expect(item.locator('.ok-btn:has-text("Accept")')).not.toBeVisible();

    // API state unchanged from prior test
    const apiData = await getImageViaAPI(token, 'system', 'e2e-image-edit-target');
    expect(apiData!.description).toBe('Updated via form edit');
  });
});

// ---------------------------------------------------------------------------
// UI-driven copy
// ---------------------------------------------------------------------------
test.describe('Image Copy via UI', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-copy-source');
    await deleteImageViaAPI(token, 'system', 'e2e-image-copy-result');
    await createImageViaAPI(token, {
      group: 'system',
      name: 'e2e-image-copy-source',
      scaler: 'K8s',
      image: 'thorium/copy-source:latest',
      timeout: 350,
      display_type: 'JSON',
      description: 'Source image for copy test',
    });
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-copy-source');
    await deleteImageViaAPI(token, 'system', 'e2e-image-copy-result');
  });

  test('copy image populates template and creates successfully', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-copy-source');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(500);

    await item.locator('.ok-btn:has-text("Copy")').click();
    await page.waitForTimeout(1000);

    await expect(page).toHaveURL(/\/create\/image/, { timeout: 5000 });

    // Switch to editor to verify template data and fix the name
    await page.locator('button:has-text("Editor")').click();
    await waitForEditor(page);

    const content = await page.locator('.cm-content').textContent();
    expect(content).toContain('copy-source');
    expect(content).toContain('thorium/copy-source:latest');
    expect(content).toContain('350');

    // Replace the copied name (which has spaces) with a valid name
    const yaml = [
      'group: system',
      'name: e2e-image-copy-result',
      'scaler: K8s',
      'image: thorium/copy-source:latest',
      'timeout: 350',
      'display_type: JSON',
      'description: Source image for copy test',
    ].join('\n');
    await setEditorContent(page, yaml);
    await page.waitForTimeout(300);

    // Submit the copy
    await page.locator('.ok-btn:has-text("Create")').click();
    await page.waitForTimeout(2000);

    await expect(page).toHaveURL(/\/images/, { timeout: 10000 });

    // Verify the copy exists in API
    const apiData = await getImageViaAPI(
      token,
      'system',
      'e2e-image-copy-result',
    );
    expect(apiData).not.toBeNull();
    expect(apiData!.timeout).toBe(350);
    expect(apiData!.image).toBe('thorium/copy-source:latest');

    await snapshot(page, SCREENSHOT_DIR, 'image-live-copy');
  });
});

// ---------------------------------------------------------------------------
// UI-driven delete
// ---------------------------------------------------------------------------
test.describe('Image Delete via UI', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-delete-target');
    await createImageViaAPI(token, {
      group: 'system',
      name: 'e2e-image-delete-target',
      scaler: 'External',
      display_type: 'JSON',
      description: 'Will be deleted via UI',
    });
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-delete-target');
  });

  test('delete image via UI confirmation modal', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-delete-target');
    await expect(item).toBeVisible();

    await item.locator('.accordion-header').click();
    await page.waitForTimeout(500);

    await item.locator('.warning-btn:has-text("Delete")').click();
    await page.waitForTimeout(500);

    const modal = page.locator('.modal');
    await expect(modal).toBeVisible();
    await expect(modal).toContainText('Confirm deletion');
    await expect(modal).toContainText('e2e-image-delete-target');

    await snapshot(page, SCREENSHOT_DIR, 'image-live-delete-modal');

    await page.locator('.danger-btn:has-text("Confirm")').click();
    await page.waitForTimeout(2000);

    // Image should be gone from the list
    await expect(
      page.locator('.accordion-item-name .text', {
        hasText: 'e2e-image-delete-target',
      }),
    ).not.toBeVisible({ timeout: 5000 });

    // Verify via API
    const apiData = await getImageViaAPI(
      token,
      'system',
      'e2e-image-delete-target',
    );
    expect(apiData).toBeNull();

    await snapshot(page, SCREENSHOT_DIR, 'image-live-delete-confirmed');
  });
});

// ---------------------------------------------------------------------------
// Responsive layout
// ---------------------------------------------------------------------------
test.describe('Image Responsive Layout', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-k8s');
    await createImageViaAPI(token, IMAGE_K8S);
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-k8s');
  });

  test('wide viewport shows 2-column grid layout', async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-k8s');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=Resources')).toBeVisible();
    await expect(item.locator('text=Arguments')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'image-responsive-wide');
  });

  test('narrow viewport shows single-column layout', async ({ page }) => {
    await page.setViewportSize({ width: 800, height: 900 });
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-k8s');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=Resources')).toBeVisible();
    await expect(item.locator('text=Arguments')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'image-responsive-narrow');
  });
});

// ---------------------------------------------------------------------------
// Ban display
// ---------------------------------------------------------------------------
test.describe('Image Ban Display', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-banned');
    await createImageViaAPI(token, {
      group: 'system',
      name: 'e2e-image-banned',
      scaler: 'External',
      display_type: 'JSON',
      description: 'Image with a ban',
    });
    await updateImageViaAPI(token, 'system', 'e2e-image-banned', {
      bans: {
        bans_added: [
          {
            id: '00000000-0000-0000-0000-000000000001',
            time_banned: new Date().toISOString(),
            ban_kind: { Generic: { msg: 'Banned for E2E testing' } },
          },
        ],
        bans_removed: [],
      },
    });
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-banned');
  });

  test('banned image shows warning icon and ban details', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-banned');

    // Ban warning icon in accordion header
    await expect(item.locator('.accordion-item-status svg')).toBeVisible();

    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    // Ban detail in body
    await expect(item.locator('text=Banned for E2E testing')).toBeVisible();
    await expect(item.locator('text=Banned on')).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'image-live-ban-display');
  });
});

// ---------------------------------------------------------------------------
// Volume add/remove via API patch
// ---------------------------------------------------------------------------
test.describe('Image Volume Patch', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-vol');
    await createImageViaAPI(token, {
      group: 'system',
      name: 'e2e-image-vol',
      scaler: 'K8s',
      image: 'thorium/vol-test:latest',
      timeout: 100,
      display_type: 'JSON',
      volumes: [{ name: 'vol-a', archetype: 'ConfigMap', mount_path: '/cfg' }],
    });
    await updateImageViaAPI(token, 'system', 'e2e-image-vol', {
      add_volumes: [{ name: 'vol-b', archetype: 'Secret', mount_path: '/sec' }],
      remove_volumes: ['vol-a'],
    });
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-vol');
  });

  test('volume add/remove reflected in display', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-vol');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    await expect(item.locator('text=vol-b')).toBeVisible();
    await expect(item.locator('text=vol-a')).not.toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'image-live-volume-patch');
  });
});

// ---------------------------------------------------------------------------
// Empty-state badges (None) for External scaler
// ---------------------------------------------------------------------------
test.describe('Image Empty-State Badges', () => {
  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);
    await deleteImageViaAPI(token, 'system', 'e2e-image-empty');
    await createImageViaAPI(token, {
      group: 'system',
      name: 'e2e-image-empty',
      scaler: 'K8s',
      image: 'thorium/empty:latest',
      timeout: 100,
      display_type: 'JSON',
    });
  });

  test.afterAll(async () => {
    await deleteImageViaAPI(token, 'system', 'e2e-image-empty');
  });

  test('sections show None badges when empty', async ({ page }) => {
    await loginViaUI(page);
    await page.goto('/images');
    await page.waitForSelector('.accordion', { timeout: 15000 });

    const item = findImageItem(page, 'e2e-image-empty');
    await item.locator('.accordion-header').click();
    await page.waitForTimeout(1000);

    // Volumes, Modifiers, and Child Filters should all show "None"
    const noneBadges = item.locator('.badge:has-text("None")');
    const count = await noneBadges.count();
    expect(count).toBeGreaterThanOrEqual(3);

    await snapshot(page, SCREENSHOT_DIR, 'image-live-empty-badges');
  });
});
