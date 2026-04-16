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

test.describe('File Upload', () => {
  let token: string;

  test.beforeAll(async () => {
    token = await authenticate(USER, PASS);
  });

  test('uploads a file with origin, tags, and pipeline reaction', async ({ page }) => {
    test.setTimeout(120_000);
    await loginViaUI(page);
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    await snapshot(page, SCREENSHOT_DIR, 'upload-initial');

    // --- 1. Drop a file via the dropzone input ---
    const uniqueContent = `e2e upload test ${Date.now()}`;
    const tmpFile = path.join(import.meta.dirname, `upload-test-${Date.now()}.bin`);
    fs.writeFileSync(tmpFile, uniqueContent);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(tmpFile);
    await expect(page.locator('text=Accepted Files')).toBeVisible({ timeout: 5000 });

    // --- 2. Select group "static" (pipelines live there) ---
    // SelectInputArray uses react-select — click the control, type, pick from dropdown
    const groupSelect = page.locator('input[id*="react-select"]').first();
    await groupSelect.click();
    await groupSelect.fill('static');
    await page.locator('[id*="react-select"][id*="option"]', { hasText: 'static' }).first().click();
    await page.waitForTimeout(500);

    // --- 3. Add a description ---
    await page.locator('textarea[placeholder="Add Description"]').fill('E2E test upload');

    // --- 4. Add tags ---
    // TagSelect has a placeholder span "Add Tags" that creates a tag entry on click
    const tagPlaceholder = page.locator('span', { hasText: 'Add Tags' });
    await tagPlaceholder.click();
    // Now a tag entry appears with key input (placeholder "Enter a key...")
    const tagKeyInput = page.locator('input[placeholder="Enter a key..."]');
    await expect(tagKeyInput).toBeVisible({ timeout: 3000 });
    await tagKeyInput.fill('e2e-test');
    await tagKeyInput.press('Tab');
    await page.waitForTimeout(300);
    // Value input should now be focused (placeholder "Enter a value...")
    const tagValueInput = page.locator('input[placeholder="Enter a value..."]');
    await tagValueInput.fill('true');
    await tagValueInput.press('Escape');
    await page.waitForTimeout(300);
    // Click outside the tag select to dismiss any dropdown overlay
    await page.locator('textarea[placeholder="Add Description"]').click();
    await page.waitForTimeout(300);

    // --- 5. Select TLP GREEN ---
    await page.locator('button', { hasText: 'GREEN' }).click({ force: true });

    // --- 6. Set origin info — Downloaded tab (default) with URL ---
    await page.locator('input[placeholder="badsite.xyz"]').fill('https://example.com/test-file.bin');
    await page.locator('input[placeholder="optional"]').first().fill('example.com');

    await snapshot(page, SCREENSHOT_DIR, 'upload-form-filled');

    // --- 7. Select a pipeline reaction ---
    // Wait for pipelines to load in the SelectPipelines panel
    const pipelineButton = page.locator('button', { hasText: 'strings' }).first();
    await expect(pipelineButton).toBeVisible({ timeout: 10000 });
    await pipelineButton.click();
    await expect(pipelineButton).toHaveClass(/selected/, { timeout: 3000 });

    await snapshot(page, SCREENSHOT_DIR, 'upload-pipeline-selected');

    // --- 8. Click Upload ---
    const uploadButton = page.locator('button.ok-btn', { hasText: 'Upload' });
    await expect(uploadButton).toBeVisible();
    await uploadButton.click();

    // --- 9. Wait for upload to complete ---
    // Single-file upload stays on the form view with inline alerts.
    // The success alert shows "File uploaded successfully:" with a SHA256 link.
    const successAlert = page.locator('.alert-success', { hasText: 'File uploaded successfully' });
    await expect(successAlert).toBeVisible({ timeout: 60000 });

    await snapshot(page, SCREENSHOT_DIR, 'upload-complete');

    // --- 10. Verify results ---
    // SHA256 link inside the success alert
    const shaLink = successAlert.locator('a.link-text');
    await expect(shaLink).toBeVisible({ timeout: 5000 });
    const sha256 = await shaLink.textContent();
    expect(sha256).toBeTruthy();
    expect(sha256!.length).toBe(64);

    // Reaction submission alert — "Successfully submitted reaction ... for pipeline strings"
    const reactionAlert = page.locator('.alert-info', { hasText: 'Successfully submitted reaction' });
    await expect(reactionAlert).toBeVisible({ timeout: 30000 });
    await expect(reactionAlert).toContainText('strings');
    await expect(reactionAlert).toContainText('static');

    // Reaction link should be present
    const reactionLink = reactionAlert.locator('a.link-text');
    await expect(reactionLink).toBeVisible();

    await snapshot(page, SCREENSHOT_DIR, 'upload-reactions-complete');

    // Clean up temp file
    fs.unlinkSync(tmpFile);
  });
});
