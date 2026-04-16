import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';
import { authenticate, snapshot } from './helpers';

const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');
const USER = process.env.THORIUM_USER || 'test';
const PASS = process.env.THORIUM_PASS || 'INSECURE_DEV_PASSWORD';

async function loginViaUI(page: import('@playwright/test').Page) {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.locator('input[placeholder="username"]').fill(USER);
  await page.locator('input[placeholder="password"]').fill(PASS);
  await page.locator('button:has-text("Login")').click();
  await page.waitForURL((url) => !url.pathname.includes('/auth'), { timeout: 15000 });
}

function createTempFile(suffix: string, content?: string): string {
  const filePath = path.join(import.meta.dirname, `upload-test-${suffix}-${Date.now()}.bin`);
  fs.writeFileSync(filePath, content || `e2e test content ${suffix} ${Date.now()}`);
  return filePath;
}

function cleanupFiles(...files: string[]) {
  for (const f of files) {
    try { fs.unlinkSync(f); } catch { /* ignore */ }
  }
}

/**
 * Get the tab pane controlled by a specific tab element.
 * Uses aria-controls to find the exact pane, avoiding ambiguity with nested
 * tab panes (e.g., the Carved tab's PCAP sub-pane stays .active.show even
 * when a different origin tab is selected).
 */
async function getTabPane(page: import('@playwright/test').Page, tabLocator: import('@playwright/test').Locator) {
  const panelId = await tabLocator.getAttribute('aria-controls');
  if (!panelId) throw new Error('Tab has no aria-controls attribute');
  return page.locator(`[id="${panelId}"]`);
}


async function selectGroup(page: import('@playwright/test').Page, group = 'static') {
  const groupSelect = page.locator('input[id*="react-select"]').first();
  await groupSelect.click();
  await groupSelect.fill(group);
  await page.locator('[id*="react-select"][id*="option"]', { hasText: group }).first().click();
  await page.waitForTimeout(500);
}

async function attachFile(page: import('@playwright/test').Page, filePath: string) {
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(filePath);
  await expect(page.locator('text=Accepted Files')).toBeVisible({ timeout: 5000 });
}

test.describe('Upload Page — Form Rendering', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('renders upload form with all required sections', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    await expect(page.locator('.title', { hasText: 'Upload' })).toBeVisible();
    await expect(page.locator('input[type="file"]')).toBeAttached();
    await expect(page.locator('textarea[placeholder="Add Description"]')).toBeVisible();
    await expect(page.locator('button.ok-btn', { hasText: 'Upload' })).toBeVisible();

    const tlpButtons = page.locator('.tlp-btn');
    await expect(tlpButtons).toHaveCount(5);
    await expect(page.locator('.tlp-btn', { hasText: 'CLEAR' })).toBeVisible();
    await expect(page.locator('.tlp-btn', { hasText: 'GREEN' })).toBeVisible();
    await expect(page.locator('.tlp-btn', { hasText: 'AMBER' }).first()).toBeVisible();
    await expect(page.locator('.tlp-btn', { hasText: 'RED' })).toBeVisible();

    const originTabs = page.locator('[role="tab"]');
    const tabTexts = await originTabs.allTextContents();
    expect(tabTexts).toEqual(expect.arrayContaining([
      'Downloaded', 'Transformed', 'Unpacked', 'Carved', 'Wire', 'Incident', 'Memory Dump',
    ]));

    await snapshot(page, SCREENSHOT_DIR, 'upload-form-initial');
  });
});

test.describe('Upload Page — TLP Selection', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('toggles TLP buttons exclusively', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const clearBtn = page.locator('.tlp-btn', { hasText: 'CLEAR' });
    const greenBtn = page.locator('.tlp-btn', { hasText: 'GREEN' });
    const redBtn = page.locator('.tlp-btn', { hasText: 'RED' });

    await greenBtn.click();
    await expect(greenBtn).toHaveClass(/selected/);
    await expect(clearBtn).not.toHaveClass(/selected/);
    await expect(redBtn).not.toHaveClass(/selected/);

    await snapshot(page, SCREENSHOT_DIR, 'upload-tlp-green');

    await redBtn.click();
    await expect(redBtn).toHaveClass(/selected/);
    await expect(greenBtn).not.toHaveClass(/selected/);

    await snapshot(page, SCREENSHOT_DIR, 'upload-tlp-red');

    await redBtn.click();
    await expect(redBtn).not.toHaveClass(/selected/);

    await snapshot(page, SCREENSHOT_DIR, 'upload-tlp-none');
  });
});

test.describe('Upload Page — Origin Tabs', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('Downloaded tab — renders URL and Site Name fields', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const downloadedTab = page.locator('[role="tab"]', { hasText: 'Downloaded' });
    await expect(downloadedTab).toHaveAttribute('aria-selected', 'true');

    const pane = await getTabPane(page, downloadedTab);
    const urlInput = pane.locator('input[placeholder="badsite.xyz"]');
    await expect(urlInput).toBeVisible();
    await urlInput.fill('https://malware.example.com/payload');

    const siteNameInput = pane.locator('input[placeholder="optional"]');
    await expect(siteNameInput).toBeVisible();
    await siteNameInput.fill('malware.example.com');

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-downloaded');
  });

  test('Transformed tab — renders Parent, Tool, Flags fields', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const transformedTab = page.locator('[role="tab"]', { hasText: 'Transformed' });
    await transformedTab.click();
    await expect(transformedTab).toHaveAttribute('aria-selected', 'true');

    const pane = await getTabPane(page, transformedTab);
    const parentInput = pane.locator('input[placeholder="SHA256"]');
    await expect(parentInput).toBeVisible();
    await parentInput.fill('a'.repeat(64));

    const optionalInputs = pane.locator('input[placeholder="optional"]');
    await expect(optionalInputs).toHaveCount(2);
    await optionalInputs.nth(0).fill('upx');
    await optionalInputs.nth(1).fill('-d --best');

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-transformed');
  });

  test('Unpacked tab — renders same Parent, Tool, Flags fields as Transformed', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const unpackedTab = page.locator('[role="tab"]', { hasText: 'Unpacked' });
    await unpackedTab.click();
    await expect(unpackedTab).toHaveAttribute('aria-selected', 'true');

    const pane = await getTabPane(page, unpackedTab);
    const parentInput = pane.locator('input[placeholder="SHA256"]');
    await expect(parentInput).toBeVisible();
    await parentInput.fill('b'.repeat(64));

    const optionalInputs = pane.locator('input[placeholder="optional"]');
    await expect(optionalInputs).toHaveCount(2);

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-unpacked');
  });

  test('Carved tab — renders Parent, Tool, and PCAP sub-fields', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const carvedTab = page.locator('[role="tab"]', { hasText: 'Carved' });
    await carvedTab.click();
    await expect(carvedTab).toHaveAttribute('aria-selected', 'true');

    const pcapTab = page.locator('[role="tab"]', { hasText: 'PCAP' });
    await expect(pcapTab).toBeVisible();
    const unknownTab = page.locator('[role="tab"]', { hasText: 'Unknown' });
    await expect(unknownTab).toBeVisible();

    await expect(pcapTab).toHaveAttribute('aria-selected', 'true');

    const carvedPane = await getTabPane(page, carvedTab);
    const parentInput = carvedPane.locator('input[placeholder="SHA256"]');
    await expect(parentInput).toBeVisible();
    await parentInput.fill('c'.repeat(64));

    const pcapPane = await getTabPane(page, pcapTab);
    const sourceIpLabel = pcapPane.locator('text=Source IP');
    await expect(sourceIpLabel).toBeVisible();
    const destIpLabel = pcapPane.locator('text=Destination IP');
    await expect(destIpLabel).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-carved-pcap');

    await unknownTab.click();
    await expect(unknownTab).toHaveAttribute('aria-selected', 'true');

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-carved-unknown');
  });

  test('Wire tab — renders Sniffer, Source, Destination fields', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const wireTab = page.locator('[role="tab"]', { hasText: 'Wire' });
    await wireTab.click();
    await expect(wireTab).toHaveAttribute('aria-selected', 'true');

    const pane = await getTabPane(page, wireTab);
    await expect(pane.locator('.subtitle', { hasText: 'Sniffer' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Source' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Destination' })).toBeVisible();

    const nameInput = pane.locator('input[placeholder="name"]');
    await expect(nameInput).toBeVisible();
    await nameInput.fill('wireshark');

    const optionalInputs = pane.locator('input[placeholder="optional"]');
    await expect(optionalInputs).toHaveCount(2);
    await optionalInputs.nth(0).fill('192.168.1.1');
    await optionalInputs.nth(1).fill('10.0.0.1');

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-wire');
  });

  test('Incident tab — renders all 6 incident fields', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const incidentTab = page.locator('[role="tab"]', { hasText: 'Incident' });
    await incidentTab.click();
    await expect(incidentTab).toHaveAttribute('aria-selected', 'true');

    const pane = await getTabPane(page, incidentTab);
    await expect(pane.locator('.subtitle', { hasText: 'Incident ID' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Cover Term' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Mission Team' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Network' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Machine' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Location' })).toBeVisible();

    const nameInput = pane.locator('input[placeholder="name"]');
    await expect(nameInput).toBeVisible();
    await nameInput.fill('INC-2024-001');

    const optionalInputs = pane.locator('input[placeholder="optional"]');
    await expect(optionalInputs).toHaveCount(5);
    await optionalInputs.nth(0).fill('ALPHA');
    await optionalInputs.nth(1).fill('TEAM-A');
    await optionalInputs.nth(2).fill('DMZ');
    await optionalInputs.nth(3).fill('SERVER-01');
    await optionalInputs.nth(4).fill('DC-EAST');

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-incident');
  });

  test('Memory Dump tab — renders Memory Type, Parent, Reconstructed, Base Address', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const memoryDumpTab = page.locator('[role="tab"]', { hasText: 'Memory Dump' });
    await memoryDumpTab.click();
    await expect(memoryDumpTab).toHaveAttribute('aria-selected', 'true');

    const pane = await getTabPane(page, memoryDumpTab);
    await expect(pane.locator('.subtitle', { hasText: 'Memory Type' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Parent' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Reconstructed' })).toBeVisible();
    await expect(pane.locator('.subtitle', { hasText: 'Base Address' })).toBeVisible();

    const typeInput = pane.locator('input[placeholder="type"]');
    await expect(typeInput).toBeVisible();
    await typeInput.fill('LSASS');

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-memorydump');
  });

  test('switching between origin tabs preserves field values', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const urlInput = page.locator('input[placeholder="badsite.xyz"]');
    await urlInput.fill('https://test.example.com');

    const wireTab = page.locator('[role="tab"]', { hasText: 'Wire' });
    await wireTab.click();
    await expect(wireTab).toHaveAttribute('aria-selected', 'true');

    const pane = await getTabPane(page, wireTab);
    const snifferInput = pane.locator('input[placeholder="name"]');
    await snifferInput.fill('tcpdump');

    const downloadedTab = page.locator('[role="tab"]', { hasText: 'Downloaded' });
    await downloadedTab.click();
    await expect(downloadedTab).toHaveAttribute('aria-selected', 'true');

    const urlInputAgain = page.locator('input[placeholder="badsite.xyz"]');
    await expect(urlInputAgain).toHaveValue('https://test.example.com');

    await snapshot(page, SCREENSHOT_DIR, 'upload-origin-tab-persistence');
  });
});

test.describe('Upload Page — Validation', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('shows error when uploading without selecting a file', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
    await uploadButton.click();

    const errorAlert = page.locator('.alert-danger');
    await expect(errorAlert).toBeVisible({ timeout: 5000 });
    await expect(errorAlert).toContainText('file');

    await snapshot(page, SCREENSHOT_DIR, 'upload-validation-no-file');
  });

  test('shows error when uploading without selecting a group', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFile = createTempFile('no-group');
    try {
      await attachFile(page, tmpFile);

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      const errorAlert = page.locator('.alert-danger');
      await expect(errorAlert).toBeVisible({ timeout: 5000 });
      await expect(errorAlert).toContainText(/group/i);

      await snapshot(page, SCREENSHOT_DIR, 'upload-validation-no-group');
    } finally {
      cleanupFiles(tmpFile);
    }
  });
});

test.describe('Upload Page — Downloaded Origin Full Upload', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads a file with Downloaded origin and verifies success', async ({ page }) => {
    test.setTimeout(120_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFile = createTempFile('downloaded');
    try {
      await attachFile(page, tmpFile);
      await selectGroup(page);

      await page.locator('textarea[placeholder="Add Description"]').fill('E2E Downloaded origin test');
      await page.locator('input[placeholder="badsite.xyz"]').fill('https://example.com/test');
      await page.locator('.tlp-btn', { hasText: 'GREEN' }).click();

      await snapshot(page, SCREENSHOT_DIR, 'upload-downloaded-filled');

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      const successAlert = page.locator('.alert-success', { hasText: 'File uploaded successfully' });
      await expect(successAlert).toBeVisible({ timeout: 60000 });

      const shaLink = successAlert.locator('a.link-text');
      await expect(shaLink).toBeVisible({ timeout: 5000 });
      const sha256 = await shaLink.textContent();
      expect(sha256).toBeTruthy();
      expect(sha256!.length).toBe(64);

      await snapshot(page, SCREENSHOT_DIR, 'upload-downloaded-success');
    } finally {
      cleanupFiles(tmpFile);
    }
  });
});

test.describe('Upload Page — Transformed Origin Upload', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads a file with Transformed origin', async ({ page }) => {
    test.setTimeout(120_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFile = createTempFile('transformed');
    try {
      await attachFile(page, tmpFile);
      await selectGroup(page);

      const transformedTab = page.locator('[role="tab"]', { hasText: 'Transformed' });
      await transformedTab.click();
      await expect(transformedTab).toHaveAttribute('aria-selected', 'true');

      const pane = await getTabPane(page, transformedTab);
      await pane.locator('input[placeholder="SHA256"]').fill('a'.repeat(64));
      const optionalInputs = pane.locator('input[placeholder="optional"]');
      await optionalInputs.nth(0).fill('upx');
      await optionalInputs.nth(1).fill('-d');

      await snapshot(page, SCREENSHOT_DIR, 'upload-transformed-filled');

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      const successAlert = page.locator('.alert-success', { hasText: 'File uploaded successfully' });
      await expect(successAlert).toBeVisible({ timeout: 60000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-transformed-success');
    } finally {
      cleanupFiles(tmpFile);
    }
  });
});

test.describe('Upload Page — Wire Origin Upload', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads a file with Wire origin', async ({ page }) => {
    test.setTimeout(120_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFile = createTempFile('wire');
    try {
      await attachFile(page, tmpFile);
      await selectGroup(page);

      const wireTab = page.locator('[role="tab"]', { hasText: 'Wire' });
      await wireTab.click();

      const pane = await getTabPane(page, wireTab);
      await pane.locator('input[placeholder="name"]').fill('wireshark');
      const optionalInputs = pane.locator('input[placeholder="optional"]');
      await optionalInputs.nth(0).fill('192.168.1.100');
      await optionalInputs.nth(1).fill('10.0.0.5');

      await snapshot(page, SCREENSHOT_DIR, 'upload-wire-filled');

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      const successAlert = page.locator('.alert-success', { hasText: 'File uploaded successfully' });
      await expect(successAlert).toBeVisible({ timeout: 60000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-wire-success');
    } finally {
      cleanupFiles(tmpFile);
    }
  });
});

test.describe('Upload Page — Incident Origin Upload', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads a file with Incident origin', async ({ page }) => {
    test.setTimeout(120_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFile = createTempFile('incident');
    try {
      await attachFile(page, tmpFile);
      await selectGroup(page);

      const incidentTab = page.locator('[role="tab"]', { hasText: 'Incident' });
      await incidentTab.click();

      const pane = await getTabPane(page, incidentTab);
      await pane.locator('input[placeholder="name"]').fill('INC-2024-001');
      const optionalInputs = pane.locator('input[placeholder="optional"]');
      await optionalInputs.nth(0).fill('ALPHA');
      await optionalInputs.nth(1).fill('TEAM-A');
      await optionalInputs.nth(2).fill('DMZ');
      await optionalInputs.nth(3).fill('SERVER-01');
      await optionalInputs.nth(4).fill('DC-EAST');

      await snapshot(page, SCREENSHOT_DIR, 'upload-incident-filled');

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      const successAlert = page.locator('.alert-success', { hasText: 'File uploaded successfully' });
      await expect(successAlert).toBeVisible({ timeout: 60000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-incident-success');
    } finally {
      cleanupFiles(tmpFile);
    }
  });
});

test.describe('Upload Page — Multi-file Upload with Status Dashboard', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads multiple files and shows the status dashboard', async ({ page }) => {
    test.setTimeout(180_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFiles = [
      createTempFile('multi-1'),
      createTempFile('multi-2'),
      createTempFile('multi-3'),
    ];

    try {
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles(tmpFiles);
      await expect(page.locator('text=Accepted Files')).toBeVisible({ timeout: 5000 });

      await selectGroup(page);

      await snapshot(page, SCREENSHOT_DIR, 'upload-multi-filled');

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      const totalLabel = page.locator('text=Total');
      await expect(totalLabel).toBeVisible({ timeout: 15000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-multi-dashboard-progress');

      await expect(page.locator('.status-file', { hasText: 'Filename' })).toBeVisible({ timeout: 30000 });
      await expect(page.locator('.status-msg', { hasText: 'Status' })).toBeVisible();
      await expect(page.locator('.status-percent', { hasText: 'Progress' })).toBeVisible();

      const filesUploadedText = page.locator('text=/\\d+ Files Uploaded Successfully/');
      await expect(filesUploadedText).toBeVisible({ timeout: 120000 });

      const backButton = page.locator('button.ok-btn', { hasText: 'Back' });
      await expect(backButton).toBeVisible({ timeout: 5000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-multi-dashboard-complete');

      await backButton.click();
      await expect(page.locator('button.ok-btn', { hasText: 'Upload' })).toBeVisible({ timeout: 5000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-multi-back-to-form');
    } finally {
      cleanupFiles(...tmpFiles);
    }
  });
});

test.describe('Upload Page — Carved Origin with PCAP Fields', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads a file with Carved/PCAP origin', async ({ page }) => {
    test.setTimeout(120_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFile = createTempFile('carved');
    try {
      await attachFile(page, tmpFile);
      await selectGroup(page);

      const carvedTab = page.locator('[role="tab"]', { hasText: 'Carved' });
      await carvedTab.click();
      await expect(carvedTab).toHaveAttribute('aria-selected', 'true');

      const pane = await getTabPane(page, carvedTab);
      await pane.locator('input[placeholder="SHA256"]').fill('d'.repeat(64));

      await snapshot(page, SCREENSHOT_DIR, 'upload-carved-filled');

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      const successAlert = page.locator('.alert-success', { hasText: 'File uploaded successfully' });
      await expect(successAlert).toBeVisible({ timeout: 60000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-carved-success');
    } finally {
      cleanupFiles(tmpFile);
    }
  });
});

test.describe('Upload Page — Carved Origin Validation', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('shows validation feedback for invalid PCAP IP fields', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const carvedTab = page.locator('[role="tab"]', { hasText: 'Carved' });
    await carvedTab.click();

    const pcapTab = page.locator('[role="tab"]', { hasText: 'PCAP' });
    const pcapPane = await getTabPane(page, pcapTab);

    const sourceIpInput = pcapPane.locator('input').first();
    await sourceIpInput.fill('not-an-ip');

    const invalidFeedback = pcapPane.locator('.invalid-feedback').first();
    await expect(invalidFeedback).toBeVisible({ timeout: 3000 });
    await expect(invalidFeedback).toContainText('IPv4/IPv6');

    await snapshot(page, SCREENSHOT_DIR, 'upload-carved-validation');
  });
});

test.describe('Upload Page — Description and Tags', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('can add description and tags to upload form', async ({ page }) => {
    test.setTimeout(60_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const descField = page.locator('textarea[placeholder="Add Description"]');
    await descField.fill('This is a detailed test description for E2E validation');
    await expect(descField).toHaveValue('This is a detailed test description for E2E validation');

    const tagPlaceholder = page.locator('span', { hasText: 'Add Tags' });
    await tagPlaceholder.click();

    const tagKeyInput = page.locator('input[placeholder="Enter a key..."]');
    await expect(tagKeyInput).toBeVisible({ timeout: 3000 });
    await tagKeyInput.fill('test-key');
    await tagKeyInput.press('Tab');
    await page.waitForTimeout(300);

    const tagValueInput = page.locator('input[placeholder="Enter a value..."]');
    await tagValueInput.fill('test-value');
    await tagValueInput.press('Escape');
    await page.waitForTimeout(300);

    await descField.click();
    await page.waitForTimeout(300);

    await snapshot(page, SCREENSHOT_DIR, 'upload-description-tags');
  });
});

test.describe('Upload Page — Memory Dump Origin Upload', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads a file with Memory Dump origin', async ({ page }) => {
    test.setTimeout(120_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    const tmpFile = createTempFile('memdump');
    try {
      await attachFile(page, tmpFile);
      await selectGroup(page);

      const memDumpTab = page.locator('[role="tab"]', { hasText: 'Memory Dump' });
      await memDumpTab.click();

      const pane = await getTabPane(page, memDumpTab);
      await pane.locator('input[placeholder="type"]').fill('LSASS');
      await pane.locator('input[placeholder="optional"]').first().fill('abc123def456abc123def456abc123def456abc123def456abc123def456abcd');

      await snapshot(page, SCREENSHOT_DIR, 'upload-memdump-filled');

      const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
      await uploadButton.click();

      // Wait for either success or error alert
      const resultAlert = page.locator('.alert-success, .alert-danger').first();
      await expect(resultAlert).toBeVisible({ timeout: 60000 });
      await snapshot(page, SCREENSHOT_DIR, 'upload-memdump-result');

      const successAlert = page.locator('.alert-success', { hasText: 'File uploaded successfully' });
      await expect(successAlert).toBeVisible({ timeout: 5000 });

      await snapshot(page, SCREENSHOT_DIR, 'upload-memdump-success');
    } finally {
      cleanupFiles(tmpFile);
    }
  });
});
