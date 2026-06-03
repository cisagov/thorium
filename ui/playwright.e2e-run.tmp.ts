import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  retries: 0,
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  use: {
    baseURL: 'http://localhost:8210',
    headless: true,
    screenshot: 'off',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev -- --port 8210 --strictPort',
    port: 8210,
    reuseExistingServer: false,
    env: {
      THORIUM_API_URL: process.env.THORIUM_API_URL || 'http://localhost:8080',
    },
  },
});
