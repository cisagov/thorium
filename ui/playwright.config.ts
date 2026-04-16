import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  retries: 0,
  use: {
    baseURL: 'http://localhost:8000',
    headless: true,
    screenshot: 'off',
    trace: 'on-first-retry',
  },
  webServer: {
    command: 'npm run dev',
    port: 8000,
    reuseExistingServer: true,
    env: {
      REACT_APP_API_URL: process.env.THORIUM_API_URL || 'http://localhost:8080/api',
    },
  },
});
